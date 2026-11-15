"""Bounded Recovery-to-Motion switch diagnosis with one continuous physical world."""

import ctypes
import dataclasses
import math
import time
from pathlib import Path

import numpy as np
import torch

from native_stand.artifacts.io import file_sha256, read_json, write_json, write_npz
from native_stand.experiments.config import job_from_record
from native_stand.learning.inference import InferenceSpec, load_model
from native_stand.simulation.recovery_mujoco import RecoveryMuJoCoCollection


class GateSample(ctypes.Structure):
    _fields_ = [
        ("tick_ms", ctypes.c_uint32),
        ("valid", ctypes.c_uint8),
        ("height_m", ctypes.c_float),
        ("pitch_rad", ctypes.c_float),
        ("roll_rad", ctypes.c_float),
        ("angular_speed_rad_s", ctypes.c_float),
        ("normal_force_N", ctypes.c_float * 2),
        ("maximum_leg_speed_rad_s", ctypes.c_float),
    ]


class GateConfig(ctypes.Structure):
    _fields_ = [("dwell_ms", ctypes.c_uint32)] + [
        (k, ctypes.c_float)
        for k in [
            "height_min_m",
            "height_max_m",
            "command_height_m",
            "height_error_m",
            "tilt_rad",
            "angular_speed_rad_s",
            "leg_speed_rad_s",
            "support_min_N",
            "maximum_leg_target_jump_rad",
            "maximum_wheel_target_jump_rad_s",
        ]
    ]


class GateState(ctypes.Structure):
    _fields_ = [(k, ctypes.c_uint32) for k in ["since_ms", "last_ms", "request_tick_ms"]] + [
        (k, ctypes.c_uint8) for k in ["dwelling", "observed", "requested", "committed"]
    ]


def plan(c):
    if c["version"] != "recovery_motion_handoff_v1":
        raise ValueError("Unknown handoff diagnostic contract")
    for ref in [c["recovery"], c["motion"], c["source_run"], c["gate_source"], c["gate_header"]]:
        if file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Handoff dependency SHA changed: " + ref["path"])
    seconds = c["seconds"]
    if (
        not math.isfinite(seconds)
        or not 10 <= seconds <= 30
        or abs(seconds * 100 - round(seconds * 100)) > 1e-8
    ):
        raise ValueError("seconds must be 10..30 on the 10ms policy grid")
    for key in ("integration_steps", "storage_bytes"):
        if type(c["limits"][key]) is not int or c["limits"][key] < 1:
            raise ValueError("Positive integer resource limit required: " + key)
    if not math.isfinite(c["limits"]["wall_s"]) or c["limits"]["wall_s"] <= 0:
        raise ValueError("Positive finite wall_s required")
    for group, key in ((c["variants"], "name"), (c["cases"], "case_id")):
        names = [v[key] for v in group]
        if (
            not names
            or len(set(names)) != len(names)
            or any(not n or Path(n).name != n or n in (".", "..") for n in names)
        ):
            raise ValueError("Unique safe nonempty variant/case names required")
    for variant in c["variants"]:
        if variant["kind"] not in ("motion_standing", "recovery_only", "fixed", "gated", "posture"):
            raise ValueError("Unknown handoff variant")
        if variant["kind"] in ("gated", "posture") and (
            type(variant.get("dwell_ms")) is not int or not 1 <= variant["dwell_ms"] <= 1000
        ):
            raise ValueError("Positive bounded dwell_ms required")
        if variant["kind"] == "fixed" and not 0 <= variant["switch_s"] < seconds:
            raise ValueError("Fixed switch must be inside the episode")
    maximum_steps = round(seconds / 0.0005) * len(c["variants"]) * len(c["cases"])
    if maximum_steps > c["limits"]["integration_steps"]:
        raise ValueError("Full handoff plan exceeds integration_steps")
    # Raw float trace + ZIP overhead and metadata/build reserve; compression can only reduce it.
    storage_bound = maximum_steps * 512 + 16 * 1024**2
    if storage_bound > c["limits"]["storage_bytes"]:
        raise ValueError("Full handoff plan exceeds conservative storage_bytes bound")
    return dict(
        operation="sim2sim",
        mode="recovery_motion_handoff",
        writes=False,
        starts_engine=False,
        capability_accepted=False,
        estimated_storage_upper_bytes=storage_bound,
        variants=c["variants"],
        cases=c["cases"],
        maximum_integration_steps=maximum_steps,
        limits=c["limits"],
        training_updates=0,
        hardware="OFF",
        same_world_switch=True,
    )


def run(c, output):
    import subprocess

    p = plan(c)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    total = 0
    records = []
    started = time.monotonic()
    terminal = dict(status="RUNNING", actual_integration_steps=0)
    try:
        write_json(output / "plan.json", p)
        write_json(output / "protocol.json", c)
        library = output / "handoff_gate.so"
        subprocess.run(
            [
                "cc",
                "-O2",
                "-std=c99",
                "-shared",
                "-fPIC",
                "-Wall",
                "-Wextra",
                "-Werror",
                c["gate_source"]["path"],
                "-I" + str(Path(c["gate_header"]["path"]).parent),
                "-lm",
                "-o",
                str(library),
            ],
            check=True,
            timeout=max(0.001, c["limits"]["wall_s"] - (time.monotonic() - started)),
        )
        gate = ctypes.CDLL(str(library))
        gate.RLHandoffObserve.argtypes = [
            ctypes.POINTER(GateState),
            ctypes.POINTER(GateConfig),
            ctypes.POINTER(GateSample),
        ]
        gate.RLHandoffObserve.restype = ctypes.c_int
        ar = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
        gate.RLHandoffCanCommit.argtypes = [
            ctypes.POINTER(GateState),
            ctypes.POINTER(GateConfig),
            ctypes.c_uint32,
            ar,
            ar,
        ]
        gate.RLHandoffCanCommit.restype = ctypes.c_int
        posture = any(v["kind"] == "posture" for v in c["variants"])
        if posture:
            gate.RLHandoffObservePosture.argtypes = [
                ctypes.POINTER(GateState),
                ctypes.POINTER(GateConfig),
                ctypes.c_uint32,
                ctypes.c_int,
                ar,
                ar,
                ar,
                ar,
                ar,
                ctypes.c_float,
            ]
            gate.RLHandoffObservePosture.restype = ctypes.c_int
        recovery = load_model(
            c["recovery"]["path"],
            InferenceSpec.from_record(c["recovery_spec"]),
            sha256=c["recovery"]["sha256"],
        )
        motion = load_model(
            c["motion"]["path"],
            InferenceSpec.from_record(c["motion_spec"]),
            sha256=c["motion"]["sha256"],
        )
        base = job_from_record(read_json(c["source_run"]["path"])["config"])
        torch.set_num_threads(1)
        for variant in c["variants"]:
            for case in c["cases"]:
                if time.monotonic() - started > c["limits"]["wall_s"]:
                    raise TimeoutError("Handoff diagnostic wall boundary")
                assets = dict(base.assets)
                cfg = dict(assets["recovery_mujoco"])
                selected = dict(case)
                if variant["kind"] == "motion_standing":
                    selected["pose"] = "standing_h0.34"
                cfg["episode_seconds"] = c["seconds"]
                cfg["cases"] = {k: [selected] for k in ["train", "development", "holdout"]}
                assets["recovery_mujoco"] = cfg
                job = dataclasses.replace(
                    base,
                    assets=assets,
                    runtime=dataclasses.replace(base.runtime, envs=1, device="cpu"),
                )
                env = RecoveryMuJoCoCollection(job)
                state = GateState()
                g = GateConfig(variant.get("dwell_ms", 200), **c["gate"])
                owner = "motion" if variant["kind"] == "motion_standing" else "recovery"
                switch = 0.0 if owner == "motion" else None
                maxjump = [0.0, 0.0]
                rows = {
                    k: []
                    for k in [
                        "observation",
                        "requested",
                        "network_mean",
                        "owner",
                        "state_1kHz",
                        "q_1kHz",
                        "dq_1kHz",
                        "motor_1kHz",
                        "NET_1kHz",
                        "adopted_1kHz",
                        "dq_HP10_1kHz",
                        "gate_requested",
                    ]
                }
                fault = False
                truncated = False
                episode_steps = 0
                delays = None
                try:
                    for t in range(round(c["seconds"] / 0.01)):
                        check_step_budget(c["limits"], total, started)
                        x = env.observations()["student_actor"].clone()
                        mx = x.clone()
                        mx[:, 25:28] = torch.tensor([1.0, 0.0, 0.0])
                        with torch.inference_mode():
                            motion_mean = motion.mean(mx)
                            if (
                                c["motion_spec"]["output_contract"]["action_version"]
                                == "motion_multiheight_support_center_poly5_015_80_v1"
                            ):
                                from native_stand.simulation.control_core import ControlCore, build

                                if not hasattr(motion, "physical_decoder"):
                                    motion.physical_decoder = ControlCore(
                                        build(output / "motion_decoder.so"),
                                        c["motion_spec"]["output_contract"],
                                    )
                                motion_target = torch.from_numpy(
                                    np.stack(
                                        [
                                            motion.physical_decoder.decode(row, 0.34)
                                            for row in motion_mean.numpy()
                                        ]
                                    )
                                )
                            else:
                                motion_target = motion_mean * torch.tensor(
                                    [1.5, 1.5, 100.0, 1.5, 1.5, 100.0]
                                )
                            if owner == "recovery":
                                ready = variant["kind"] in ("gated", "posture") and state.requested
                                if variant["kind"] == "fixed":
                                    ready = t * 0.01 >= variant["switch_s"]
                                if ready:
                                    previous = env.execution.adopted[0].numpy().copy()
                                    target = motion_target[0].numpy().copy()
                                    if variant["kind"] == "fixed" or gate.RLHandoffCanCommit(
                                        ctypes.byref(state),
                                        ctypes.byref(g),
                                        t * 10,
                                        previous,
                                        target,
                                    ):
                                        owner = "motion"
                                        state.committed = 1
                                        switch = t * 0.01
                                        maxjump = [
                                            float(abs((target - previous)[[0, 1, 3, 4]]).max()),
                                            float(abs((target - previous)[[2, 5]]).max()),
                                        ]
                            mean = motion_mean if owner == "motion" else recovery.actor.mean(x)
                            if owner == "motion":
                                action = motion_target
                            elif (
                                c["recovery_spec"]["output_contract"]["action_version"]
                                == "firmware_absolute_linear150_100_v1"
                            ):
                                action = mean * torch.tensor([1.5, 1.5, 100.0, 1.5, 1.5, 100.0])
                            else:
                                action = recovery.actor.decode_mean(mean)
                        received = mx if owner == "motion" else x
                        try:
                            result = env.step(action)
                        except BaseException as exc:
                            total += int(
                                np.asarray(getattr(exc, "actual_physics_steps", [0])).sum()
                            )
                            raise
                        steps = int(result["extras"]["actual_physics_steps"].sum())
                        episode_steps += steps
                        total += steps
                        if total > c["limits"]["integration_steps"]:
                            raise RuntimeError("Handoff integration budget exceeded")
                        conditions = result["extras"]["collection_conditions"]
                        delays = {
                            key: conditions[key].tolist()
                            for key in (
                                "actor_age_ms",
                                "target_delay_ms",
                                "wheel_left_ms",
                                "wheel_right_ms",
                            )
                        }
                        trace = result["extras"]["collection_trace"]
                        s = trace["state_1kHz"][0].numpy()
                        dq = trace["dq_1kHz"][0].numpy()
                        q = trace["q_1kHz"][0].numpy()
                        for i in range(10):
                            # Source gyro is CAD-body angular velocity; norm is frame-invariant.
                            angular = float(np.linalg.norm(received[0, :3].numpy() * 4))
                            sample = GateSample(
                                t * 10 + i + 1,
                                1,
                                float(s[i, 3]),
                                float(s[i, 4]),
                                float(s[i, 5]),
                                angular,
                                (ctypes.c_float * 2)(float(s[i, 9]), float(s[i, 10])),
                                float(abs(dq[i, [0, 1, 3, 4]]).max()),
                            )
                            if variant["kind"] == "posture":
                                # Same instantaneous encoder/IMU readiness as MCU; no
                                # ground height/contact force is used by this gate.
                                from native_stand.simulation.control_core import ControlCore, build

                                if not hasattr(motion, "physical_decoder"):
                                    motion.physical_decoder = ControlCore(
                                        build(output / "motion_decoder.so"),
                                        c["motion_spec"]["output_contract"],
                                    )
                                center = motion.physical_decoder.decode(
                                    np.zeros(6, np.float32), 0.34
                                )[[0, 1, 3, 4]].copy()
                                roll, pitch = float(s[i, 5]), float(s[i, 4])
                                gravity = np.asarray(
                                    [
                                        np.sin(roll) * np.cos(pitch),
                                        -np.cos(roll) * np.cos(pitch),
                                        np.sin(pitch),
                                    ],
                                    dtype=np.float32,
                                )
                                # Gate only needs gyro norm; record this approximation.
                                gyro = np.asarray([angular, 0.0, 0.0], dtype=np.float32)
                                gate.RLHandoffObservePosture(
                                    ctypes.byref(state),
                                    ctypes.byref(g),
                                    t * 10 + i + 1,
                                    1,
                                    q[i, [0, 1, 3, 4]].copy(),
                                    dq[i].copy(),
                                    gyro,
                                    gravity,
                                    center,
                                    variant.get("maximum_center_error_rad", 0.25),
                                )
                            else:
                                gate.RLHandoffObserve(
                                    ctypes.byref(state), ctypes.byref(g), ctypes.byref(sample)
                                )
                        rows["observation"].append(received[0].numpy())
                        rows["requested"].append(action[0].numpy())
                        rows["network_mean"].append(mean[0].numpy())
                        rows["owner"].append(owner == "motion")
                        rows["gate_requested"].append(state.requested)
                        for key in [
                            "state_1kHz",
                            "q_1kHz",
                            "dq_1kHz",
                            "motor_1kHz",
                            "NET_1kHz",
                            "adopted_1kHz",
                            "dq_HP10_1kHz",
                        ]:
                            rows[key].append(trace[key][0].numpy())
                        if result["terminated"][0] or result["truncated"][0]:
                            fault = bool(result["terminated"][0])
                            truncated = bool(result["truncated"][0])
                            break
                    arrays = {k: np.array(v) for k, v in rows.items()}
                    folder = output / (variant["name"] + "__" + case["case_id"])
                    folder.mkdir()
                    used = sum(f.stat().st_size for f in output.rglob("*") if f.is_file())
                    raw_bytes = sum(a.nbytes for a in arrays.values()) + 1024**2
                    if used + raw_bytes + 1024**2 > c["limits"]["storage_bytes"]:
                        raise RuntimeError("Handoff storage budget boundary")
                    write_npz(folder / "trace.npz", arrays)
                    s = arrays["state_1kHz"].reshape(-1, 13)
                    hp = arrays["dq_HP10_1kHz"].reshape(-1, 4)
                    times = np.arange(1, len(s) + 1) * 0.001
                    dual = (s[:, 9:11] > 2).all(1)
                    quality = (
                        (abs(s[:, 3] - 0.34) <= 0.02)
                        & (abs(s[:, 4:6]) <= 0.18).all(1)
                        & dual
                        & (s[:, 11] <= 0.005)
                    )
                    baseline = variant["kind"] == "recovery_only"
                    # The sample at switch_s belongs to the preceding Recovery call.
                    # Use the actual controller of each 10ms call, not rounded timestamps.
                    after = (
                        np.ones(len(times), bool)
                        if baseline
                        else np.repeat(arrays["owner"].astype(bool), 10)
                    )
                    edges = np.diff(np.r_[False, quality & after, False].astype(int))
                    a = np.flatnonzero(edges == 1)
                    b = np.flatnonzero(edges == -1)
                    longest = float((b - a).max() * 0.001) if len(a) else 0.0
                    tail = (times >= max(0, times[-1] - 5)) & after
                    row = dict(
                        variant=variant,
                        case=case,
                        switch_s=switch,
                        own_control=True,
                        world_reset_at_switch=False,
                        previous_cleared_at_switch=False,
                        execution_queues_cleared_at_switch=False,
                        post_switch_raw_hold_s=longest,
                        post_switch_hold10=bool(
                            not baseline and switch is not None and longest >= 10 - 1e-6
                        ),
                        recovery_only_hold10=bool(baseline and longest >= 10 - 1e-6),
                        hold_window_controller="recovery" if baseline else "motion",
                        truncated=truncated,
                        effective_delays_ms=delays,
                        fault=fault,
                        seconds=len(s) * 0.001,
                        first_target_jump_leg_rad=maxjump[0],
                        first_target_jump_wheel_rad_s=maxjump[1],
                        after_switch_dual_support_fraction=float(dual[after].mean())
                        if after.any()
                        else None,
                        last5s_abs_vx_mean=float(abs(s[tail, 0]).mean()) if tail.any() else None,
                        last5s_HP10_RMS=np.sqrt((hp[tail] ** 2).mean(0)).tolist()
                        if tail.any()
                        else None,
                        actual_integration_steps=episode_steps,
                        trace=dict(
                            path=str(folder / "trace.npz"), sha256=file_sha256(folder / "trace.npz")
                        ),
                    )
                    records.append(row)
                    write_json(output / "episodes.json", records)
                    print(
                        f"HANDOFF {variant['name']} {case['case_id']} switch={switch} hold={longest:.3f} fault={fault}",
                        flush=True,
                    )
                finally:
                    env.close()
        terminal["status"] = "COMPLETED"
    except BaseException as exc:
        terminal.update(status="FAILED", failure=repr(exc))
        raise
    finally:
        terminal.update(
            actual_integration_steps=total,
            wall_s=time.monotonic() - started,
            training_updates=0,
            hardware="OFF",
        )
        write_json(output / "terminal.json", terminal)
    return dict(**terminal, path=str(output), episodes=records)


def check_step_budget(limits, total, started):
    """Collection integrates one indivisible 10ms call (20 x .5ms substeps)."""
    if total + 20 > limits["integration_steps"]:
        raise RuntimeError("Handoff integration budget boundary before step")
    if time.monotonic() - started >= limits["wall_s"]:
        raise TimeoutError("Handoff diagnostic wall boundary before step")
