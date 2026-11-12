"""离线复算Motion站立日志、部署模型和已声明动作解码；不启动物理。"""
from __future__ import annotations

import argparse
import csv
import ctypes
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy import signal

from native_stand.artifacts.io import file_sha256, new_dir, write_json

LEGS = [0, 1, 3, 4]


def read_log(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        fields = next(reader)
        data = np.asarray(list(reader), dtype=float)
    if data.shape[1] != len(fields) or not np.isfinite(data).all():
        raise ValueError("CSV必须是有限、完整的数值记录")
    def columns(stem, count):
        found = {}
        for i, name in enumerate(fields):
            match = re.search(r"(?:^|\.)" + re.escape(stem) + r"\[(\d+)\]$", name)
            if match:
                if int(match[1]) in found:
                    raise ValueError("CSV字段重复：" + stem)
                found[int(match[1])] = i
        if set(found) != set(range(count)):
            raise ValueError("CSV字段不完整：" + stem)
        return data[:, [found[i] for i in range(count)]]
    time = data[:, fields.index("Time")]
    if (np.diff(time) <= 0).any():
        raise ValueError("时间必须严格递增")
    obs = columns("observation", 25)
    g = obs[:, 3:6]
    angles = np.rad2deg(np.c_[np.arctan2(g[:, 2], -g[:, 1]),
                              np.arctan2(g[:, 0], -g[:, 1])])
    leg, wheel = columns("rl_leg_torque_nm", 4), columns("rl_wheel_torque_nm", 2)
    return dict(t=time, obs=obs, mean=columns("mean_action", 6),
                leg_command=leg, wheel_command=wheel,
                active=(abs(np.c_[leg, wheel]) > 1e-7).any(1),
                q=obs[:, 6:10] * 1.5, dq=obs[:, [10, 11, 13, 14]] * 20,
                angles=angles, fields=fields)


def spans(mask, time):
    edge = np.diff(np.r_[False, mask, False].astype(int))
    return [dict(first_row=int(b), last_row=int(e - 1), count=int(e - b),
                 first_s=float(time[b]), last_s=float(time[e - 1]),
                 span_s=float(time[e - 1] - time[b]))
            for b, e in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))]


def stats(values):
    return dict(mean=values.mean(0).tolist(), std=values.std(0).tolist(),
                min=values.min(0).tolist(), max=values.max(0).tolist(),
                abs_P95=np.percentile(abs(values), 95, axis=0).tolist(),
                abs_peak=abs(values).max(0).tolist())


def spectrum(t, values):
    if len(t) < 100 or t[-1] - t[0] < .99:
        return None
    if np.diff(t).max() > .03:
        return dict(status="时间断档，未拼接频谱")
    grid = np.arange(t[0], t[-1] + 1e-9, .01)
    v = np.stack([np.interp(grid, t, values[:, j]) for j in range(values.shape[1])], -1)
    f, power = signal.welch(v, fs=100, nperseg=min(512, len(grid)), axis=0)
    band, near = (f >= 10) & (f <= 40), (f >= 20) & (f <= 26)
    return dict(status="100Hz记录的描述频谱", resolution_hz=float(f[1] - f[0]),
                peak10_40_hz=f[band][power[band].argmax(0)].tolist(),
                RMS10_40=np.sqrt(np.trapezoid(power[band], f[band], axis=0)).tolist(),
                RMS20_26=np.sqrt(np.trapezoid(power[near], f[near], axis=0)).tolist())


def metrics(log, lo, hi, target):
    mask = (log["t"] >= lo) & (log["t"] < hi) & log["active"]
    if not mask.any():
        return None
    t, q, dq = (log[k][mask] for k in ("t", "q", "dq"))
    return dict(first_s=float(t[0]), last_s=float(t[-1]), samples=len(t),
                q_rad=stats(q), dq_rad_s=stats(dq), pitch_roll_deg=stats(log["angles"][mask]),
                commands=stats(log["obs"][mask, 22:25] * [4, 4, .5]),
                leg_target_step_RMS_rad=np.sqrt(np.mean(np.diff(target[mask][:, LEGS], axis=0)**2, axis=0)).tolist(),
                wheel_target_rad_s=stats(target[mask][:, [2, 5]]),
                leg_command_nm=stats(log["leg_command"][mask]), wheel_command_nm=stats(log["wheel_command"][mask]),
                leg_command_cap_fraction=(abs(log["leg_command"][mask]) >= 39.999).mean(0).tolist(),
                wheel_command_cap_fraction=(abs(log["wheel_command"][mask]) >= 2.446947864 - 1e-4).mean(0).tolist(),
                q_spectrum=spectrum(t, q), dq_spectrum=spectrum(t, dq))


def load_adapter(package):
    spec = importlib.util.spec_from_file_location("selected_deployment_adapter", package / "reference_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--candidate-package", type=Path, action="append", default=[])
    parser.add_argument("--old-log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    log = read_log(args.csv)
    packages = [args.package] + args.candidate_package
    inputs = [args.csv, Path(__file__), args.package / "reference_adapter.py", args.package / "output_contract.json"]
    for p in packages:
        inputs.extend([p / "actor.onnx", p / "manifest.json"])
    if args.old_log:
        inputs.append(args.old_log)
    protected = {str(p.resolve()): file_sha256(p) for p in inputs}
    output = new_dir(args.output_dir)
    actor_input = np.c_[log["obs"], np.tile([1, 0, 0, 1, 0, 0], (len(log["t"]), 1))].astype(np.float32)
    active_ids = np.flatnonzero(log["active"])
    if not len(active_ids):
        raise ValueError("没有非零RL调试力矩段")
    comparison, prediction = [], None
    import onnx
    from onnx.reference import ReferenceEvaluator
    for p in packages:
        graph = onnx.load(p / "actor.onnx")
        onnx.checker.check_model(graph)
        manifest = json.loads((p / "manifest.json").read_text())
        if "actor_onnx_sha256" in manifest and file_sha256(p / "actor.onnx") != manifest["actor_onnx_sha256"]:
            raise ValueError("部署ONNX与manifest身份不同")
        pred = ReferenceEvaluator(graph).run(None, {graph.graph.input[0].name: actor_input})[0]
        error = abs(pred[active_ids] - log["mean"][active_ids])
        comparison.append(dict(package=str(p.resolve()), MAE=float(error.mean()),
                               P99=float(np.percentile(error, 99)), max_abs=float(error.max()),
                               rows_matched_1e5=int((error.max(1) < 1e-5).sum()), rows=len(error)))
        if p == args.package:
            prediction = pred
    adapter = load_adapter(args.package)
    declaration = adapter.contract()
    leg_scale = declaration['leg_halfwidth_rad']
    wheel_scale = declaration['wheel_halfwidth_rad_s']
    if leg_scale != .15 or wheel_scale != 80:
        raise ValueError("本次只支持明确的支撑中心0.15rad／80rad/s输出，不能套旧解码")
    if (abs(log['mean'][active_ids]) > 1.000001).any():
        raise ValueError("记录的确定性mean超过Tanh范围")
    target = np.empty_like(log["mean"])
    # 合同内固定命令高度，输入q/previous尺度独立保持1.5/100。
    for i, height in enumerate(log["obs"][:, 24] * .5):
        if .30 <= height <= .40:
            target[i] = log["mean"][i] * [leg_scale, leg_scale, wheel_scale, leg_scale, leg_scale, wheel_scale]
            target[i, LEGS] += adapter.centers(float(height))
        else:
            target[i] = 0
    error = abs(prediction - log["mean"]).max(1)
    bad = log["active"] & (error > 1e-5)
    stable = log["active"] & (log["t"] >= 30)
    previous = log["obs"][:,16:22] * np.array([1.5,1.5,1.5,1.5,100,100])
    pe = abs(previous[1:] - target[:-1, [0, 1, 3, 4, 2, 5]])
    # 使用冻结实际C控制函数，直接传物理目标，避免旧／新动作解码混用。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import audit_motion31_electrical_cpu as cm
    library = Path("artifacts/2026_11_01__motion600_real_log_audit/firmware_cpu/motion_firmware.so")
    protected[str(library.resolve())] = file_sha256(library)
    lib = ctypes.CDLL(str(library))
    lib.RLMotion31ComputePdTorque.argtypes = [ctypes.POINTER(cm.Snapshot)] + [cm.FP]*3
    lib.RLMotion31MapPolicyLegTorqueToMotorOrder.argtypes = [cm.FP]*2
    lib.RLMotion31MapPolicyWheelTorqueToForwardOrder.argtypes = [cm.FP]*2
    lib.motion_fw_geometry.argtypes = [cm.FP]*3
    lib.RLControlCompensateSpring.argtypes = [cm.FP]*4
    check_ids = np.unique(np.r_[active_ids[::10], np.flatnonzero((log['t']>=30)&(log['t']<40))])
    c_errors = []
    for i in check_ids:
        x = log["obs"][i]
        sample = cm.Snapshot(q=cm.F(4)(*(x[6:10]*1.5)), dq=cm.F(6)(*(x[10:16]*20)), height=.34)
        request, limited = cm.F(6)(), cm.F(6)()
        motor, final = cm.F(4)(), cm.F(4)()
        wheel, length, spread = cm.F(2)(), cm.F(2)(), cm.F(2)()
        assert lib.RLMotion31ComputePdTorque(ctypes.byref(sample), cm.F(6)(*target[i]), request, limited) == 0
        assert lib.RLMotion31MapPolicyLegTorqueToMotorOrder(limited, motor) == 0
        assert lib.RLMotion31MapPolicyWheelTorqueToForwardOrder(limited, wheel) == 0
        assert lib.motion_fw_geometry(cm.F(4)(*(x[6:10]*1.5)), length, spread) == 0
        assert lib.RLControlCompensateSpring(motor, length, spread, final) == 0
        predicted = np.r_[np.array(final), np.clip(np.array(wheel), -2.446947864, 2.446947864)]
        c_errors.append(abs(predicted - np.r_[log["leg_command"][i],log["wheel_command"][i]]))
    c_errors = np.array(c_errors)
    full_c = c_errors[log["t"][check_ids] >= 30]
    zero = log["active"] & (abs(log["obs"][:,22]*4)<.01) & (abs(log["obs"][:,23]*4)<.01)
    result = dict(source_csv=str(args.csv.resolve()), log_sha256=file_sha256(args.csv), rows=len(log["t"]), columns=len(log["fields"]),
                  time_range_s=[float(log["t"][0]),float(log["t"][-1])],
                  sample_dt_s=stats(np.diff(log["t"])), debug_command_segments=spans(log["active"],log["t"]),
                  missing_fields=["controller_owner/fault", "build及模型身份", "目标采用序号", "PD反馈时刻", "实测root高度", "双轮接触力", "实际机体前进速度", "实际电流／轴矩"],
                  assumed_unrecorded_task_inputs=[1,0,0,1,0,0], model_comparison=comparison,
                  selected_model=dict(package=str(args.package.resolve()), SHA256=file_sha256(args.package/'actor.onnx'),
                    full_active_match_fraction=float((error[log['active']]<=1e-5).mean()),
                    matched_after30_max_abs=float(error[stable].max()),
                    mismatched_spans=spans(bad,log['t']),mismatched_rows=int(bad.sum()),
                    caution="起段输入／mean／previous未完全同步，保留原行不删除；不能凭此判断固件推理错误"),
                  action_contract=declaration['action_version'],
                  recorded_actor_dq=dict(nonzero_fraction_per_leg=(log['dq'][log['active']]!=0).mean(0).tolist()),
                  previous_after30=dict(max_abs=pe[stable[1:]].max(0).tolist(),
                                        within1e5_fraction=float((pe[stable[1:]].max(1)<1e-5).mean()),
                                        scope="上一记录请求目标的数值一致，采用时刻未独立记录"),
                  C_control_check=dict(samples=len(check_ids),after30_samples=len(full_c), after30_max_error_Nm=full_c.max(0).tolist(),
                    all_sample_max_error_Nm=c_errors.max(0).tolist(),
                    control_formula="legs Kp40/Kd2, wheels Kv0.2; frozen C mapping and spring compensation; input current logged q/dq and decoded physical targets",
                    scope="软件命令数值核验，非烧录身份或实际电机出力"),
                  windows={name:metrics(log,lo,hi,target) for name,lo,hi in [
                      ('all',log['t'][active_ids[0]],log['t'][active_ids[-1]]+.001),('initial',17.95,20),('early_quiet',20,30),
                      ('oscillation',30,40),('oscillation_core',32,39),('later_quiet',45,155)]},
                  exact_zero_command=dict(rows=int(zero.sum()),longest=sorted(spans(zero,log['t']),key=lambda x:x['span_s'],reverse=True)[:3]),
                  ten_second_bins=[metrics(log,lo,lo+10,target) for lo in range(20,150,10)],
                  physical_steps=0, PPO_updates=0, supervised_updates=0, source_sha256=protected)
    if args.old_log:
        old = read_log(args.old_log)
        ix = np.flatnonzero(old['active'])
        old_target = old['mean'] * [1.5,1.5,100,1.5,1.5,100]
        result['historical_comparison'] = dict(source=str(args.old_log.resolve()),
           metrics=metrics(old,float(old['t'][ix[0]]),float(old['t'][ix[-1]])+.001,old_target),
           scope="不同模型／命令／初态的历史量级比较，不归因于单一训练或接口改动")
    assert all(file_sha256(Path(p))==sha for p,sha in protected.items())
    result['all_sources_unchanged'] = True
    write_json(output/'analysis.json',result)
    draw(output,log,target)
    report(output,result)
    print(json.dumps({'output':str(output),'rows':len(log['t']), 'active':result['debug_command_segments'],
                      'model_after30_max_error':result['selected_model']['matched_after30_max_abs']}))


def draw(output,log,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#2369bd','#e38124','#28a174','#a551ab']
    names=['Right hip','Right active','Left hip','Left active']
    selected = log['active']
    t = log['t'][selected]
    fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
    axes[0].plot(t,log['angles'][selected,0],label='Pitch (deg)')
    axes[0].plot(t,log['angles'][selected,1],label='Roll (deg)')
    for j in range(4):
        axes[1].plot(t,log['dq'][selected,j],color=colors[j],lw=.7,label=names[j])
        axes[2].plot(t,target[selected,LEGS[j]],color=colors[j],lw=.7)
    axes[3].plot(t,log['obs'][selected,22]*4,label='vx command (m/s)')
    axes[3].plot(t,log['obs'][selected,23]*4,label='yaw command (rad/s)')
    for ax,label in zip(axes,['Body angle (deg)','Leg dq (rad/s)','Leg target (rad)','Commands']):
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.axvspan(30,40,color='#e38124',alpha=.12)
    for ax in (axes[0],axes[1],axes[3]):ax.legend(loc='upper right',ncol=4,fontsize=8)
    axes[-1].set_xlim(t[0],t[-1]);axes[-1].set_xlabel('Recorded time (s)')
    fig.suptitle('Motion2320 hardware log: normal leg dq; transient oscillation then lower-amplitude motion')
    fig.tight_layout();fig.savefig(output/'overview.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,6),sharex='col')
    for col,(lo,hi) in enumerate([(34,36),(70,72)]):
        mask=(log['t']>=lo)&(log['t']<hi)
        q=log['q'][mask]-np.median(log['q'][mask],axis=0)
        for j in range(4):
            axes[0,col].plot(log['t'][mask],np.rad2deg(q[:,j]),color=colors[j],lw=.8,label=names[j])
            axes[1,col].plot(log['t'][mask],log['dq'][mask,j],color=colors[j],lw=.8)
        axes[0,col].set_title(f'{lo}-{hi}s: '+('oscillation' if col==0 else 'later low-amplitude motion'))
        axes[0,col].set_ylim(-2.5,2.5);axes[1,col].set_ylim(-7,7)
        axes[1,col].set_xlabel('Recorded time (s)')
        for ax in axes[:,col]:ax.grid(alpha=.25)
    axes[0,0].set_ylabel('q - window median (deg)');axes[1,0].set_ylabel('Leg dq (rad/s)')
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(output/'oscillation_comparison.png',dpi=150);plt.close(fig)


def report(output,r):
    rows=['|时间段|腿dq标准差范围rad/s|关节10～40Hz幅度RMS范围°|俯仰绝对峰°|横滚绝对峰°|',
          '|---|---:|---:|---:|---:|']
    for name,label in [('early_quiet','20～30s'),('oscillation','30～40s'),('later_quiet','45～155s')]:
        w=r['windows'][name];v=w['dq_rad_s']['std'];b=np.rad2deg(w['q_spectrum']['RMS10_40']);ang=w['pitch_roll_deg']['abs_peak']
        rows.append(f'|{label}|{min(v):.3f}～{max(v):.3f}|{min(b):.4f}～{max(b):.4f}|{ang[0]:.2f}|{ang[1]:.2f}|')
    segment=r['debug_command_segments'][0]
    z=r['exact_zero_command']['longest'][0]
    text=f'''# 11月6日新部署小模型：站立和振荡复盘

**本次记录支持正常dq条件下维持直立明显改善，但仍出现一段约25Hz振荡。不能将本次称为全程无抖或零命令站立认证。**

## 范围与缺失信息

原始日志已保存为logs/Ozone_DataSampling_261106normal3.csv，SHA256 `{r['log_sha256']}`。共{r['rows']}行、{r['columns']}列，约161.67秒，主要采样间隔约10ms。仅CPU读取、ONNX推理和冻结C公式核验；未启动仿真、PPO、监督更新或硬件。日志没有owner/fault、采用序号、build、实际高度、接触力、电流或机体实际前进速度。分析以用户本次实机观察结合已有记录解释，RL非零调试力矩仅用于定位分析段。

## 模型和控制公式

记录最接近部署包`{Path(r['selected_model']['package']).name}`，ONNX SHA `{r['selected_model']['SHA256']}`。约30秒以后所有非零RL调试力矩行，同输入推理的最大动作误差为{r['selected_model']['matched_after30_max_abs']:.3g}。Actor未记录的6个任务常量按Motion/NONE补齐。

起段有{r['selected_model']['mismatched_rows']}条输入与mean不能同帧复现，均在最初数秒；原行保留，不拼接成已同步证据。具体区间在analysis.json。30秒以后，previous与上一记录的0.15/80物理目标数值匹配；它仍不能独立证明每次实际采用时刻。

四腿Actor dq均有非零变化。直接调用冻结电控C函数，将新协议解码后的物理目标送入PD及补偿，30秒后抽查的{r['C_control_check']['after30_samples']}帧四腿／两轮命令最大误差分别为{max(r['C_control_check']['after30_max_error_Nm'][:4]):.3g}／{max(r['C_control_check']['after30_max_error_Nm'][4:]):.3g}Nm。数值与腿40/2、轮0.2及原补偿一致，不支持“这次靠清Actor dq或去掉PD-D才变稳”的解释。

## 实际记录表现

非零RL调试力矩段：{segment['first_s']:.3f}～{segment['last_s']:.3f}秒，跨度{segment['span_s']:.2f}秒。该段俯仰未超过约7.84°、横滚未超过约1°；45～155秒保持低幅波动，没有在本段出现持续增长到倾倒姿态的记录。

{chr(10).join(rows)}

30～40秒四个关节的位置和速度共同出现约25Hz主峰。这是位置也在往复变化，不只是dq数值噪声。该时段同时有较大的前进命令变化，命令范围与逐窗指标见analysis.json；不能凭日志确定是主动加速、外推扰动或接触变化触发。

45～155秒仍有微小高频波动，但10～40Hz位置幅度和腿速波动均明显低于上述抖动段，约25Hz没有继续持续放大。后半段改善是真实记录支持的结论。

## 不能把命令当成实际速度

整段vx命令约−2.07～+2.20m/s，yaw命令约−0.206～+0.309rad/s。高度命令固定0.34m，不等于实测高度。两项速度命令都接近0（绝对值小于0.01）的最长连续记录只有{z['span_s']:.2f}秒。因此本次能说明较长时间维持直立，不能认证原地站立、精确速度跟随、无漂移或双轮持续承重。

非零RL命令段末尾约156秒时六路命令归零，随后关节构型变化；owner/fault未记录，不把这个退出点自动分类为策略跌倒或保护触发。

## 下一次最有价值的验证

1. 固定本次模型和实际固件，不根据单次改善立即换最新权重。
2. 零vx/yaw命令连续记录30～60秒，四腿dq和PD-D照常；再重复几次LQR→RL接管，分别记录成功和失败。
3. 低速前后与转向分开，小幅增加。标记命令改变、人工推搡和退出时刻，重点查30～40秒这类振荡是否可重复触发以及是否自行衰减。
4. 最少增加owner/fault、采用目标及序号、PD dq和时间、实际电流；同步保留本次完整obs/mean。实测高度和接触信息可按现有条件补充。

本次新输出接口、训练、蒸馏、执行时序与以往均有变化，没有单因素对照，不能把全部改善只归因于某个scale或奖励。训练下一步应保留本次低幅稳定行为，针对有命令变化的短暂振荡和受扰恢复补样本，而不是继续让所有dq或力矩趋零。

详细结果：analysis.json。图片：overview.png和oscillation_comparison.png。原始CSV、模型和活动训练均未改动。
'''
    (output/'REPORT.md').write_text(text)


if __name__ == '__main__':
    main()
