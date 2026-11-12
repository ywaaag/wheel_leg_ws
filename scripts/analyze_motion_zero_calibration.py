"""只读已有CSV和USD，检查关节零位能否由双轮平地约束确定。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import time

import numpy as np
from scipy.optimize import brentq, least_squares

from native_stand.artifacts.io import file_sha256, new_dir, write_json
from native_stand.tasks.motion.calibration_geometry import (
    PlanarLegGeometry, corrected_roll_gravity, plane_jacobian, support_heights, wheel_centers,
)


def read_observations(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        names = next(reader)
        if "Time" not in names:
            return None, "无时间轴，不能把统计表当作同帧记录"
        indices = {}
        for index, name in enumerate(names):
            match = re.search(r"(?:^|\.)observation\[(\d+)\]$", name)
            if match:
                key = int(match.group(1))
                if key in indices:
                    raise ValueError(f"重复观测字段：{path.name}, {key}")
                indices[key] = index
        if set(range(25)) - indices.keys():
            return None, "无完整Actor前25项，缺双腿或重力信息"
        data = np.asarray(list(reader), dtype=float)
    if not np.isfinite(data).all() or data.shape[1] != len(names):
        raise ValueError(f"日志不是有限矩形数表：{path}")
    t = data[:, names.index("Time")]
    if len(t) < 2 or (np.diff(t) <= 0).any():
        raise ValueError(f"时间不是严格递增：{path}")
    return (t, data[:, [indices[j] for j in range(25)]]), None


def calm_windows(t, obs):
    """按原时间轴选择不重叠1秒窗口，不用可能被清零的Actor dq判静止。"""
    selected = []
    bins = np.floor(t - t[0]).astype(int)
    for key in np.unique(bins):
        ix = np.flatnonzero(bins == key)
        if len(ix) < 70 or t[ix[-1]] - t[ix[0]] < .8:
            continue
        tt, x = t[ix], obs[ix]
        if np.diff(tt).max() > .03:
            continue
        q, g = x[:, 6:10] * 1.5, x[:, 3:6]
        norm = np.linalg.norm(g, axis=1)
        if ((norm < .98) | (norm > 1.02)).any():
            continue
        g = g / norm[:, None]
        tilt = np.rad2deg(np.arccos(np.clip(-g[:, 1], -1, 1)))
        if tilt.max() > 5 or (np.abs(q) > .8).any():
            continue
        if q.std(0).max() > .01 or abs(np.gradient(q, tt, axis=0)).mean(0).max() > .15:
            continue
        if np.linalg.norm(x[:, :3] * 4, axis=1).mean() > .15:
            continue
        if np.ptp(q, axis=0).max() < 2e-4:
            continue  # 全部固定的调试值不能可靠证明采样还在更新。
        if not np.all(np.abs(x[:, 24] * .5 - .34) < .002):
            continue
        selected.append(dict(start_row=int(ix[0]), end_row=int(ix[-1]), count=len(ix),
                             first_s=float(tt[0]), last_s=float(tt[-1]),
                             q_rad=np.median(q, axis=0).tolist(),
                             gravity=np.median(g, axis=0).tolist(),
                             max_q_std_rad=float(q.std(0).max()),
                             actor_leg_dq_all_zero=bool((x[:, [10, 11, 13, 14]] == 0).all())))
    return selected


def fit_explanations(q, g, geometry):
    # 这个指定方向只给出一种等效解，不能确定四个独立电机零偏。
    direction = np.array([1., 1., 1., -1.])
    def discrepancy(angle):
        h = support_heights(q + angle * direction, g, geometry)
        return h[0] - h[1]
    bound = np.deg2rad(12)
    correction = brentq(discrepancy, -bound, bound, xtol=1e-13)
    def roll_discrepancy(angle):
        h = support_heights(q, corrected_roll_gravity(g, angle), geometry)
        return h[0] - h[1]
    roll = brentq(roll_discrepancy, -bound, bound, xtol=1e-13)
    return dict(equal_opposite_leg_opening_correction_deg=float(np.rad2deg(correction)),
                joint_correction_direction=[1, 1, 1, -1],
                same_gap_explained_by_roll_correction_deg=float(np.rad2deg(roll)),
                plane_residual_after_joint_correction_m=float(discrepancy(correction)),
                plane_residual_after_roll_correction_m=float(roll_discrepancy(roll)))


def height_examples(q, g, geometry):
    rows = []
    for height in (.33, .34, .35, .36):
        dr = brentq(lambda a: support_heights(q + [a, a, 0, 0], g, geometry)[0] - height,
                    -.2, .2, xtol=1e-13)
        dl = brentq(lambda a: support_heights(q + [0, 0, -a, a], g, geometry)[1] - height,
                    -.2, .2, xtol=1e-13)
        correction = np.array([dr, dr, -dl, dl])
        rows.append(dict(assumed_actual_root_height_m=height,
                         assumed_joint_correction_deg=np.rad2deg(correction).tolist(),
                         fitted_support_heights_m=support_heights(q + correction, g, geometry).tolist()))
    return rows


def audit_asset(asset_path, q, geometry, source_pose):
    # 只解USD被动关节和坐标变换，不启动引擎、不推进物理。
    from native_stand.recovery.asset_poses import AssetPoses
    asset = AssetPoses(asset_path)
    names = ("left_hip_pitch", "left_active_link_pitch", "right_hip_pitch", "right_active_link_pitch")
    checks = []
    for angles in q:
        solved, closure = asset.solve(dict(zip(names, angles)), source_pose)
        bodies = asset.forward(solved)
        actual = np.array([bodies[side + "_wheel"][:3, 3] for side in ("left", "right")])
        predicted = wheel_centers(angles, geometry)
        checks.append(dict(center_max_error_m=float(abs(actual - predicted).max()),
                           closure_max_m=float(closure.max())))
    if max(r["center_max_error_m"] for r in checks) > .00025:
        raise ValueError("简化几何与USD轮心差超过0.25mm，不输出零位分析")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--geometry-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    spec = json.loads(args.geometry_spec.read_text())
    asset = Path(spec["asset_path"])
    geometry = PlanarLegGeometry(.21, .25, 47.1033888, 47.1080837,
                                 (.2204889953136444, -.22051099501550198),
                                 .000024, -.000014, .06, .0165)
    paths = sorted(p for p in args.logs.glob("*.csv") if "recomputed_action" not in p.name)
    from native_stand.tasks.motion import calibration_geometry
    protected = {str(p.resolve()): file_sha256(p) for p in paths + [
        args.geometry_spec, asset, Path(__file__), Path(calibration_geometry.__file__)]}
    records, excluded, windows = [], [], []
    for path in paths:
        data, reason = read_observations(path)
        if data is None:
            excluded.append(dict(file=path.name, reason=reason))
            continue
        t, obs = data
        selected = calm_windows(t, obs)
        if not selected:
            excluded.append(dict(file=path.name, reason="没有通过本次规则的1秒平稳窗口"))
            continue
        q = np.array([w["q_rad"] for w in selected])
        g = np.array([w["gravity"] for w in selected])
        heights = support_heights(q, g, geometry)
        if (heights < .28).any() or (heights > .42).any():
            raise ValueError(f"{path.name}平稳窗口不在声明站姿几何范围")
        qq, gg = np.median(q, 0), np.median(g, 0)
        median_heights = support_heights(qq, gg, geometry)
        records.append(dict(file=path.name, windows=len(selected), samples=sum(w["count"] for w in selected),
                            first_s=selected[0]["first_s"], last_s=selected[-1]["last_s"],
                            q_median_deg=np.rad2deg(qq).tolist(), gravity_median=gg.tolist(),
                            required_root_heights_m=median_heights.tolist(),
                            right_minus_left_height_mm=float(np.diff(median_heights)[0] * -1000),
                            gap_window_P05_P95_mm=np.percentile((heights[:, 0] - heights[:, 1]) * 1000, [5, 95]).tolist(),
                            **fit_explanations(qq, gg, geometry)))
        for w in selected:
            windows.append(dict(file=path.name, **w))
    q = np.array([w["q_rad"] for w in windows])
    g = np.array([w["gravity"] for w in windows])
    jac = plane_jacobian(q, g, geometry)
    _, singular, vh = np.linalg.svd(jac, full_matrices=False)
    # 仅作多解诊断：每份日志等权，避免最长记录独占拟合。
    weights = np.array([1 / np.sqrt(next(r["windows"] for r in records if r["file"] == w["file"])) for w in windows])
    def residual(delta):
        heights = support_heights(q + delta, g, geometry)
        return (heights[:, 0] - heights[:, 1]) * weights
    unconstrained = least_squares(residual, np.zeros(4), bounds=(-np.deg2rad(15), np.deg2rad(15)),
                                  max_nfev=300, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    representative = next(r for r in records if r["file"] == "Ozone_DataSampling_261104魔改前正常.csv")
    pose = min(spec["motion_height_dr"]["height_poses"], key=lambda r: abs(r["height_m"] - .34))["joint_pos"]
    checks = audit_asset(asset, np.array([r["q_median_deg"] for r in records]) * np.pi / 180, geometry, pose)
    result = dict(status="COMPLETED_OFFLINE_GEOMETRY", calibration_status="NOT_IDENTIFIED",
                  motor_zero_changes_allowed=False,
                  definitions=dict(correction="q_true=q_reported+delta, model joint-side rad; not raw motor offset", channels=[0, 1, 3, 4],
                                   inferred_height="root height required if that wheel alone is tangent to a rigid flat plane",
                                   obs24="height command only; never treated as measured height"),
                  assumptions=["原USD名义刚性几何", "原日志IMU方向和31D单位映射", "比较双轮同一平面支撑的几何一致性，接触没有独立实测"],
                  selection=dict(window_s=1, minimum_span_s=.8, minimum_rows=70, max_gap_s=.03,
                                 max_tilt_deg=5, max_q_std_rad=.01, max_mean_q_derivative_rad_s=.15,
                                 max_mean_gyro_rad_s=.15, minimum_q_peak_to_peak_rad=.0002,
                                 height_command_m=.34, owner="UNKNOWN; no inference based on nonzero debug torque",
                                 q_derivative="derived from recorded position; zeroed Actor dq not used"),
                  source_sha256=protected, files_examined=len(paths), records=records, excluded=excluded,
                  windows=len(windows), selected_samples=sum(w["count"] for w in windows),
                  local_plane_identifiability=dict(singular_values_m_per_rad=singular.tolist(),
                                                  condition_number=float(singular[0] / singular[-1]),
                                                  weakest_direction=vh[-1].tolist(),
                                                  note="同零位假设下的局部数值敏感性；不同日志未记录重新标零，不能混作独立标定证据"),
                  overfit_warning=dict(unregularized_delta_deg=np.rad2deg(unconstrained.x).tolist(),
                                       equal_file_weight_plane_RMS_mm=float(np.sqrt(np.sum(residual(unconstrained.x)**2)
                                                                                   / np.sum(weights**2)) * 1000),
                                       at_search_bound=bool(np.any(abs(unconstrained.x) > np.deg2rad(14.9))),
                                       status="DIAGNOSTIC_ONLY; not selected hardware calibration"),
                  height_assumption_demonstration=dict(file=representative["file"],
                    preserved_leg_swing=True,
                    cases=height_examples(np.deg2rad(representative["q_median_deg"]),
                                          representative["gravity_median"], geometry)),
                  usd_geometry_checks=checks, external_inputs_unchanged=all(file_sha256(Path(p)) == s for p, s in protected.items()),
                  physical_steps=0, policy_inferences=0, optimizer_steps=0, wall_s=time.monotonic()-started)
    if not result["external_inputs_unchanged"]:
        raise RuntimeError("只读输入在分析期间变化")
    output = new_dir(args.output_dir)
    write_json(output / "analysis.json", result)
    write_json(output / "selected_windows.json", windows)
    write_report(output, result)
    print(json.dumps(dict(output=str(output), files=len(records), windows=len(windows),
                          samples=result["selected_samples"], calibration_status=result["calibration_status"])))


def write_report(output, result):
    rows = ["|日志日期／工况|窗口数|右／左所需高度mm|相切高度差mm|仅归于左右腿角度的等效修正°|仅归于横滚的等效修正°|",
            "|---|---:|---|---:|---:|---:|"]
    for r in result["records"]:
        label = r["file"].replace("Ozone_DataSampling_", "").replace(".csv", "")
        rows.append(f'|{label}|{r["windows"]}|{r["required_root_heights_m"][0]*1000:.1f} / {r["required_root_heights_m"][1]*1000:.1f}|'
                    f'{r["right_minus_left_height_mm"]:.2f}|{r["equal_opposite_leg_opening_correction_deg"]:.2f}|'
                    f'{r["same_gap_explained_by_roll_correction_deg"]:.2f}|')
    examples = ["|假设独立实测root高|右hip修正°|右active修正°|左hip修正°|左active修正°|",
                "|---|---:|---:|---:|---:|"]
    for r in result["height_assumption_demonstration"]["cases"]:
        examples.append(f'|{r["assumed_actual_root_height_m"]:.2f}m|' +
                        "|".join(f"{x:+.2f}" for x in r["assumed_joint_correction_deg"]) + "|")
    gap = [r["right_minus_left_height_mm"] for r in result["records"]]
    text = f'''# 已有实机日志能否反算关节零位

**可以发现重复出现的左右几何不一致，但现有日志不足以确定四个电机应各修正几度。本次不推荐直接修改电控零位。**

## 分析范围与边界

只读{result['files_examined']}份原始CSV，使用其中{len(result['records'])}份的{result['windows']}个不重叠平稳窗口、{result['selected_samples']}条记录。仅计算几何与离线拟合；未启动物理、网络推理、训练或硬件。双轮接触、独立实测高度、IMU标定和逐次固件／重新标零记录缺失。输入SHA、筛选行号、剔除原因及数值结果分别保存在analysis.json和selected_windows.json。

## 已经重复出现的差异

四个关节输入按1.5rad解码；重力使用同帧模型坐标，考虑轮胎圆柱半径和侧倾时的轮宽。右／左指几何侧，四腿顺序ch0/1/3/4。根据位置变化筛平稳窗口，不拿被清零的Actor dq证明静止，也不把非零RL调试值当作控制归属。

将每个轮子分别放在同一水平地面，反算它所需的root高度。两轮同地支撑应给出相同高度。所选各日志的中位构型均为右侧要求更高，差{min(gap):.2f}～{max(gap):.2f}mm。

{chr(10).join(rows)}

“左右腿角度等效修正”只选一个方便比较的方向：模型角修正为[t,t,t,−t]。它使右侧两杆角增加t、左侧两杆角减少t。这是一种消除相切高度差的解，不是四个零位的辨识结果。横滚列是旋转日志重力向量得到的另一种解；两列不能叠加，更不能当作电机offset。

圆柱轮刚性模型与USD各日志中位姿态的轮心最大差为{max(r['center_max_error_m'] for r in result['usd_geometry_checks'])*1000:.3f}mm。上述毫米差并非简化公式本身产生的厘米级误差。

## 为什么现在不能直接调四个零位

1. 高度命令0.34m不等于实测0.34m。真实高度变化会同时改变两腿的关节角。
2. 一条腿的hip和active共同决定腿长和摆角。仅用平地接触，不能把所有偏差唯一分配给这两个关节。
3. 约一度的IMU横滚偏差、毫米级轮胎受压或结构变形，都可能解释一部分左右差异。当前日志没有独立信息把它们分开。
4. 不能假定跨多次上机始终用了完全相同的零位。日志数量多，不等于有更多独立几何参照。

仅按左右接触误差拟合四个自由零偏，解为{np.round(result['overfit_warning']['unregularized_delta_deg'], 2).tolist()}°，是否顶住±15°搜索边界：{result['overfit_warning']['at_search_bound']}。这是不可靠的过拟合示例，不是电控修正建议。局部导数条件数约{result['local_plane_identifiability']['condition_number']:.1f}；不同角度组合在本批窄范围姿态下难以区分。

## 同一日志，假设高度不同就会得到不同零偏

选11月4日正常输入记录的平稳中位构型。仅假设实际高度，保留每侧腿摆角，用两侧共同开合角分别拟合，两轮都可以与平面相切：

{chr(10).join(examples)}

这些量定义为q_true=q_reported+delta，单位是模型关节侧角度，不是原始电机编码器offset。表格说明缺少独立几何测量时，不能从其中任一行挑一套交给电控。

## 最小补充测量与操作顺序

1. 固定一组可重复姿态：电机关闭或由电控采用原有方式稳住，禁止用RL追求某个输入数值。记录同帧四腿角和重力；标明是有负载还是架空。
2. 机体基准面用水平仪测pitch／roll；测轮轴与髋轴中心的相对位置。若只想校零，侧视照片直接测hip轴→膝轴、active轴→其短杆末端这两条连线，相对于机体向下的角度更直接。
3. 四关节分别按部署协议的β／α公式得到几何q_true，与日志q_reported相减；再在第二个不同姿态验证差值近似不变。这样才能同时检查零位、正方向和比例。
4. 确认后统一修改反馈与目标的坐标转换，并复核previous和PD使用同一坐标。不要只给Actor加偏置，也不要依据模型输出或LQR／RL力矩差来标零。

当前可交给电控的是“左右几何一致性需要复核”的证据和上述最小测量清单，不是带正负号的电机零位修改表。
'''
    (output / "REPORT.md").write_text(text)


if __name__ == "__main__":
    main()
