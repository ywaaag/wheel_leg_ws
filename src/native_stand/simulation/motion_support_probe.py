"""Bounded fixed-Actor MuJoCo support/scale interventions; never a trainer."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np

from native_stand.artifacts.io import file_sha256, read_json, write_json, write_npz

VERSION = "motion_support_probe_v1"
LEGS = [0, 1, 3, 4]


def validate(config):
    if config.get("version") != VERSION:
        raise ValueError("未知承重诊断版本")
    if not 1 <= len(config["cases"]) <= 64:
        raise ValueError("诊断必须有1～64个有限case")
    if not 1 <= config["wall_s"] <= 1800 or not 1 <= config["max_output_bytes"] <= 512 * 2**20:
        raise ValueError("诊断必须有明确墙钟和输出空间边界")
    names = set()
    for case in config["cases"]:
        values = [
            case[k] for k in ("height_m", "vx_mps", "yaw_rad_s", "leg_scale_rad", "duration_s")
        ]
        if any(
            isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
            for v in values
        ):
            raise ValueError("case参数必须为有限数值")
        if (
            not 0.2 <= case["height_m"] <= 0.4
            or abs(case["vx_mps"]) > 3.6
            or abs(case["yaw_rad_s"]) > 2
        ):
            raise ValueError("诊断命令超出Motion范围")
        if case["leg_scale_rad"] not in (0.15, 0.3, 0.5) or not 15 <= case["duration_s"] <= 40:
            raise ValueError("只允许显式.15/.30/.50rad及15～40秒case")
        if case.get("mapping") != "rescale_source_raw":
            raise ValueError("必须明确旧Actor raw重解码干预，不是兼容模型迁移")
        offsets = case.get("center_delta_rad", [0.0] * 4)
        if len(offsets) != 4 or not all(
            isinstance(x, int | float)
            and not isinstance(x, bool)
            and math.isfinite(x)
            and abs(x) <= 0.35
            for x in offsets
        ):
            raise ValueError("中心偏移必须是四腿有限显式候选")
        if case["name"] in names or Path(case["name"]).name != case["name"]:
            raise ValueError("case名称冲突或非法")
        start_height = case.get("center_delta_start_height_m")
        if start_height is not None and (
            isinstance(start_height, bool)
            or not isinstance(start_height, int | float)
            or not math.isfinite(start_height)
            or not case["height_m"] < start_height <= 0.4
        ):
            raise ValueError("低高度中心干预起点必须高于case目标且不超过0.40m")
        names.add(case["name"])
        if case.get("spawn", "flat") not in (
            "flat",
            "steep_waves_H1",
            "cobblestones_H1",
            "broken_rubble_H1",
            "pitted_road_H1",
        ):
            raise ValueError("未知诊断地形")
        if case.get("spawn", "flat") != "flat" and not config.get("terrain_map", False):
            raise ValueError("崎岖地形需要显式启用SHA绑定地图")
    # Reserve trace rows, NPZ headers, the small C library and JSON reports before
    # opening an engine. Do not start a case that cannot fit its declared budget.
    reserve = sum(math.ceil(c["duration_s"] * 100) * 2048 + 262144 for c in config["cases"])
    if config["max_output_bytes"] < reserve + 1048576:
        raise ValueError("输出预算不足以容纳声明的全部case与诊断收据")
    return config


def transform_for(case, source_scale=0.15):
    """Explicit intervention after original C decoding; original Actor remains unchanged."""
    scale = case["leg_scale_rad"]
    offsets = np.array(case.get("center_delta_rad", [0.0] * 4), dtype=np.float32)

    def transform(raw, height, source_target):
        target = source_target.copy()
        start = case.get("center_delta_start_height_m")
        fraction = (
            1.0 if start is None else np.clip((start - height) / (start - case["height_m"]), 0, 1)
        )
        target[LEGS] += (scale - source_scale) * np.clip(raw[LEGS], -1.0, 1.0) + fraction * offsets
        return target

    return transform


def summarize(samples, case, *, termination):
    if not samples:
        return {"status": "UNKNOWN", "termination": termination, "center_candidate_eligible": False}
    times = np.array([x["time"] for x in samples])
    import torch

    from native_stand.tasks.motion.maneuver_program import extended_scored_start

    start = float(
        extended_scored_start(
            torch.tensor([[case["vx_mps"], case["yaw_rad_s"]]]),
            torch.tensor([case["height_m"]]),
        )[0]
    )
    selected = [x for x in samples if x["time"] >= max(start, case["duration_s"] - 10)]
    complete = bool(termination is None and times[-1] >= case["duration_s"] - 0.02)
    if not selected:
        return {
            "status": "UNKNOWN",
            "termination": termination,
            "center_candidate_eligible": False,
            "completed": complete,
            "physical_policy_samples": len(samples),
        }

    def array(k):
        return np.array([x[k] for x in selected], dtype=np.float64)

    height_mae = float(np.abs(array("height") - case["height_m"]).mean())
    support = float(array("dual_support").mean())
    drift = float(np.abs(array("vx")).mean())
    roll = float(np.abs(array("roll")).max())
    pitch = float(np.abs(array("pitch")).max())
    dq_rms = float(np.sqrt((array("dq")[:, LEGS] ** 2).mean()))
    q = array("q")[:, LEGS]
    adopted = array("adopted")[:, LEGS]
    target = np.median(adopted, axis=0)
    symmetric = np.array(
        [
            (target[0] - target[2]) / 2,
            (target[1] + target[3]) / 2,
            -(target[0] - target[2]) / 2,
            (target[1] + target[3]) / 2,
        ]
    )
    projection_error = float(np.abs(adopted - symmetric).max())
    gates = dict(
        complete=complete,
        stationary_command=case["vx_mps"] == case["yaw_rad_s"] == 0,
        height=height_mae <= 0.005,
        dual_support=support >= 0.98,
        drift=drift <= 0.05,
        attitude=max(roll, pitch) <= 0.05,
        low_motion=dq_rms <= 0.3,
        symmetric_target=projection_error <= 0.02,
    )
    return dict(
        status="OBSERVED",
        termination=termination,
        completed=complete,
        steady_samples=len(selected),
        window_s=[float(selected[0]["time"]), float(selected[-1]["time"])],
        height_mean_m=float(array("height").mean()),
        height_MAE_m=height_mae,
        vx_mean_mps=float(array("vx").mean()),
        yaw_mean_rad_s=float(array("yaw").mean()),
        roll_abs_max_rad=roll,
        pitch_abs_max_rad=pitch,
        dual_support_fraction=support,
        q_mean_rad=q.mean(0).tolist(),
        adopted_leg_mean_rad=adopted.mean(0).tolist(),
        leg_dq_RMS_rad_s=dq_rms,
        raw_mean=array("raw").mean(0).tolist(),
        source_target_mean=array("source_requested").mean(0).tolist(),
        motor_mean_Nm=array("motor").mean(0).tolist(),
        NET_mean_Nm=array("NET").mean(0).tolist(),
        root_position_span_m=float(
            np.linalg.norm(array("position_xy")[-1] - array("position_xy")[0])
        ),
        center_candidate_gates=gates,
        center_candidate_eligible=all(gates.values()),
        symmetric_target_candidate_rad=symmetric.tolist() if all(gates.values()) else None,
        target_projection_error_rad=projection_error,
        capability_accepted=False,
    )


def from_file(path, *, dry_run=False):
    from .playback import plan

    cfg = validate(read_json(path))
    prepared = plan(
        cfg["run"],
        cfg["checkpoint"],
        path_mapping=cfg["path_mapping"],
        checkpoint_sha256=cfg["checkpoint_sha256"],
        solver_profile=cfg["solver_profile"],
        terrain_map=cfg.get("terrain_map", False),
    )
    if prepared.job.task != "motion" or prepared.job.action.leg_scale_rad not in (0.15, 0.5):
        raise ValueError("诊断必须使用明确版本的Motion模型")
    result = dict(
        version=VERSION,
        plan=prepared.record(),
        cases=copy.deepcopy(cfg["cases"]),
        old_actor_under_intervention=any(
            c["leg_scale_rad"] != prepared.job.action.leg_scale_rad
            or any(c.get("center_delta_rad", ()))
            for c in cfg["cases"]
        ),
        physics_equivalence_claimed=False,
        capability_accepted=False,
        updates_charged=0,
        wall_s=cfg["wall_s"],
    )
    if dry_run:
        return {**result, "starts_engine": False, "writes": False}
    return run(cfg, prepared, result)


def run(cfg, prepared, result):
    from native_stand.tasks.motion.maneuver_program import extended_numpy_program

    from .playback import Controls, create_player

    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", cfg)
    started = time.monotonic()
    rows = []
    physical_steps = 0
    for case in cfg["cases"]:
        if time.monotonic() - started > cfg["wall_s"]:
            break
        plant = None
        samples = []
        termination = None
        case_steps = 0
        directory = output / case["name"]
        directory.mkdir()
        try:
            _, plant, _, _ = create_player(
                prepared,
                directory=directory,
                diagnostic_target_transform=transform_for(case, prepared.job.action.leg_scale_rad),
            )
            plant.recording = False
            if case.get("spawn", "flat") != "flat":
                plant.spawn = case["spawn"]
                plant.reset()
            dt = plant.physics_dt
            for _step in range(round(case["duration_s"] / dt)):
                if time.monotonic() - started > cfg["wall_s"]:
                    termination = "wall_limit"
                    break
                command = extended_numpy_program(
                    float(plant.data.time), [case["vx_mps"], case["yaw_rad_s"], case["height_m"]]
                )
                plant.step(
                    Controls(vx=float(command[0]), yaw=float(command[1]), height=float(command[2]))
                )
                case_steps += 1
                physical_steps += 1
                if case_steps % 10:
                    continue
                state = plant.telemetry()
                rec = plant.last_record
                samples.append(
                    {
                        **{
                            k: state[k]
                            for k in [
                                "height",
                                "vx",
                                "yaw",
                                "roll",
                                "pitch",
                                "q",
                                "dq",
                                "adopted",
                                "motor",
                                "NET",
                                "position_xy",
                            ]
                        },
                        "time": float(plant.data.time),
                        "command": command.copy(),
                        "dual_support": state["contact_geometric_left"]
                        and state["contact_geometric_right"],
                        "raw": rec["raw"].copy(),
                        "observation": rec["observation"].copy(),
                        "source_requested": rec["source_requested"].copy(),
                    }
                )
                if (
                    abs(state["roll"]) > 0.5
                    or abs(state["pitch"]) > 0.5
                    or not 0.12 <= state["height"] <= 0.46
                ):
                    termination = "attitude_or_height_boundary"
                    break
        except Exception as error:
            termination = f"{type(error).__name__}: {error}"
        finally:
            if plant is not None:
                plant.close()
        if samples:
            write_npz(
                directory / "trace.npz",
                {k: np.asarray([x[k] for x in samples]) for k in samples[0]},
            )
        row = {
            "case": case,
            "physics_steps": case_steps,
            **summarize(samples, case, termination=termination),
        }
        rows.append(row)
        write_json(directory / "summary.json", row)
        if (
            sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            > cfg["max_output_bytes"]
        ):
            break
    result.update(
        results=rows,
        physics_steps=physical_steps,
        elapsed_s=time.monotonic() - started,
        all_cases_executed=len(rows) == len(cfg["cases"]),
        checkpoint_sha256=file_sha256(prepared.checkpoint),
    )
    write_json(output / "result.json", result)
    return result
