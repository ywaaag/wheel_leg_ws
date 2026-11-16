"""Bounded command/event diagnostics through the existing MotionPlayer physics loop."""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path

import numpy as np

from native_stand.artifacts.io import file_sha256, json_digest, read_json, write_json, write_npz
from native_stand.tasks.motion.replication_metrics import summarize

VERSION = "motion_replication_probe_v1"
# Frozen USD reference gradients, transposed from modal_basis.json. Unit vector
# norm, not per-motor peak; these are reference-pose modes, not live decoupling.
MODES = np.array(
    [
        [-0.7071067811726777, -0.7071067812004174, 0, 0],
        [-0.707229012938308, 0.706984528301933, 0, 0],
        [0, 0, 0.7071067811865476, -0.7071067811865476],
        [0, 0, 0.7071603185038644, 0.7070532398154421],
    ]
)


def validate(cfg):
    if cfg["version"] != VERSION or not 1 <= len(cfg["cases"]) <= 64:
        raise ValueError("Expected bounded replication cases")
    if not 1 <= cfg["wall_s"] <= 1800 or not 1 <= cfg["max_physics_steps"] <= 2000000:
        raise ValueError("Explicit finite wall/physical limits required")
    if not 1 <= cfg["max_output_bytes"] <= 512 * 2**20:
        raise ValueError("Explicit bounded output required")
    if "actor_packet_age_ms" in cfg and (
        type(cfg["actor_packet_age_ms"]) is not int or cfg["actor_packet_age_ms"] not in (0, 20, 30)
    ):
        raise ValueError("Explicit Actor packet stress must be 0, 20 or 30ms")
    names = set()
    steps = 0
    for c in cfg["cases"]:
        if c["name"] in names or Path(c["name"]).name != c["name"]:
            raise ValueError("Unique safe case names required")
        names.add(c["name"])
        if c["role"] not in ("train", "development", "holdout") or type(c["seed"]) is not int:
            raise ValueError("Explicit episode role and seed required")
        if not 5 <= c["duration_s"] <= 60:
            raise ValueError("Bounded 5-60s cases required")
        if not 0 <= c["score_window_s"][0] < c["score_window_s"][1] <= c["duration_s"]:
            raise ValueError("Explicit score window required")
        times = []
        for row in c["commands"]:
            if len(row) != 4 or not all(
                isinstance(x, int | float) and math.isfinite(x) for x in row
            ):
                raise ValueError("Finite time/vx/yaw/height knots required")
            if (
                not 0 <= row[0] < c["duration_s"]
                or abs(row[1]) > 3.6
                or abs(row[2]) > 2
                or not 0.2 <= row[3] <= 0.4
            ):
                raise ValueError("Command outside declared Motion scope")
            times.append(row[0])
        if (
            not times
            or times[0] != 0
            or any(b <= a for a, b in zip(times, times[1:], strict=False))
        ):
            raise ValueError("Ordered command knots must start at zero")
        for e in c.get("events", []):
            if not 0 < e["at_s"] < c["duration_s"]:
                raise ValueError("Event outside episode")
            if e["kind"] == "push_velocity":
                if e["axis"] not in ("forward", "lateral") or abs(e["delta_mps"]) not in (0.5, 1.0):
                    raise ValueError("Explicit .5/1m/s diagnostic state push required")
            elif e["kind"] == "modal":
                if (
                    e["mode"] not in range(4)
                    or e["sign"] not in (-1, 1)
                    or e["peak_Nm"] not in (2, 6)
                    or e["duration_s"] != 0.1
                ):
                    raise ValueError("Explicit unit modal vector peak and 100ms required")
            else:
                raise ValueError("Unsupported physical event")
        steps += round(c["duration_s"] * 1000)
    if steps > cfg["max_physics_steps"] or steps * 650 + 2**20 > cfg["max_output_bytes"]:
        raise ValueError("Insufficient declared physical/output budget")
    return cfg


def command_at(case, seconds):
    knots = case["commands"]
    chosen = next((row for row in reversed(knots) if seconds + 1e-9 >= row[0]), knots[0])
    return np.asarray(chosen[1:], np.float32)


def event_torque(case, tick):
    tau = np.zeros(6)
    for e in case.get("events", []):
        if e["kind"] == "modal" and round(e["at_s"] * 1000) <= tick < round(
            (e["at_s"] + 0.1) * 1000
        ):
            phase = (tick - round(e["at_s"] * 1000)) / 100
            tau[[0, 1, 3, 4]] += (
                MODES[e["mode"]] * e["sign"] * e["peak_Nm"] * math.sin(math.pi * phase)
            )
    return tau


def prepare(cfg):
    from native_stand.learning.inference import InferenceSpec, load_model
    from native_stand.simulation.playback import plan

    base = cfg["plant"]
    prepared = plan(
        base["run"],
        base["checkpoint"],
        path_mapping=base["path_mapping"],
        checkpoint_sha256=base["sha256"],
        solver_profile=base["solver_profile"],
    )
    ref = cfg["actor"]
    spec = InferenceSpec.from_record(ref["inference"])
    model = load_model(ref["path"], spec, sha256=ref["sha256"])
    for field, digest in (
        ("prefix_onnx", "prefix_sha256"),
        ("prefix_output_contract", "prefix_output_contract_sha256"),
    ):
        if (
            prepared.job.assets.get(field)
            and file_sha256(prepared.job.assets[field]) != prepared.job.assets[digest]
        ):
            raise ValueError("Common player dependency SHA differs: " + field)
    if (
        spec.input_dim != 31
        or spec.kind != "tanh"
        or spec.input_contract["version"] != "multiskill_mcu_v2"
    ):
        raise ValueError("Only explicit deterministic Motion31 Actor supported")
    low, high = spec.output_contract["height_command_range_m"]
    if any(not low <= row[3] <= high for c in cfg["cases"] for row in c["commands"]):
        raise ValueError("Teacher requested outside its own height contract")
    job = dataclasses.replace(
        prepared.job,
        model=dataclasses.replace(
            prepared.job.model,
            actor_hidden_dims=spec.hidden_dims,
            action_version=spec.output_contract["action_version"],
        ),
        action=dataclasses.replace(
            prepared.job.action,
            version=spec.output_contract["action_version"],
            leg_scale_rad=spec.output_contract["leg_halfwidth_rad"],
        ),
    )
    timing = dict(prepared.timing)
    if "actor_packet_age_ms" in cfg:
        for key in ("actor_packet_nominal_ms", "actor_leg_packet_nominal_ms"):
            timing[key] = cfg["actor_packet_age_ms"]
    prepared = dataclasses.replace(
        prepared, job=job, output_contract=spec.output_contract, timing=timing
    )
    return prepared, model


def from_file(path, *, dry_run=False):
    from native_stand.simulation.control_core import ControlCore, build
    from native_stand.simulation.mujoco_playback import MotionPlayer

    cfg = validate(read_json(path))
    prepared, model = prepare(cfg)
    receipt = dict(
        version=VERSION,
        actor=cfg["actor"],
        common_plan=prepared.record(),
        protocol_sha256=file_sha256(path),
        PPO_updates=0,
        BC_rounds=0,
        contact_force_readback="mj_contactForce after step/forward; includes positive-distance margin constraints",
        planned_physics_steps=sum(round(c["duration_s"] * 1000) for c in cfg["cases"]),
        starts_engine=not dry_run,
        writes=not dry_run,
    )
    if dry_run:
        return receipt
    root = Path(cfg["output"])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "protocol.json", cfg)
    binding = build(root / "control.so")
    started = time.monotonic()
    results = []
    total_steps = 0
    for case in cfg["cases"]:
        if time.monotonic() - started >= cfg["wall_s"] or total_steps >= cfg["max_physics_steps"]:
            break
        output = root / case["name"]
        output.mkdir()
        rows = []
        plant = None
        termination = None
        physics = 0
        try:
            core = ControlCore(binding, prepared.output_contract)
            plant = MotionPlayer(prepared, model.source.payload, core, actor=model.actor)
            plant.recording = False
            # Same legal position reset; seed changes only a declared tiny root velocity.
            rng = np.random.default_rng(case["seed"])
            jitter = rng.uniform(-0.002, 0.002, 6)
            plant.data.qvel[:6] = jitter
            plant.mj.mj_forward(plant.model, plant.data)
            initial = dict(
                qpos=plant.data.qpos.tolist(),
                qvel=plant.data.qvel.tolist(),
                jitter_contract="root free joint velocity uniform[-.002,.002], linear m/s; angular rad/s",
                actual_physics_dt_s=float(plant.model.opt.timestep),
                actual_solver=plant.solver_readback,
                actual_timing=dict(plant.execution.timing),
                actual_control=dataclasses.asdict(plant.execution.control),
                actual_actuator_ctrlrange=plant.model.actuator_ctrlrange.tolist(),
                modal_basis_model_leg_order=MODES.tolist(),
            )
            if abs(float(plant.model.opt.timestep) - 0.001) > 1e-12:
                raise ValueError("Actual physical clock is not 1ms")
            for tick in range(round(case["duration_s"] * 1000)):
                if time.monotonic() - started >= cfg["wall_s"]:
                    termination = "wall_limit"
                    break
                pushes = np.zeros(3)
                for e in case.get("events", []):
                    if e["kind"] == "push_velocity" and tick == round(e["at_s"] * 1000):
                        rotation = plant.data.xmat[plant.robot_body].reshape(3, 3)
                        flu = rotation[:, [2, 0, 1]] * [-1.0, -1.0, 1.0]
                        delta = flu[:, 0 if e["axis"] == "forward" else 1] * e["delta_mps"]
                        plant.data.qvel[:3] += delta
                        pushes += delta
                        plant.mj.mj_forward(plant.model, plant.data)
                command = command_at(case, tick * 0.001)
                external = event_torque(case, tick)
                from native_stand.simulation.playback import Controls

                before = plant.actual_physics_steps
                try:
                    plant.step(
                        Controls(
                            vx=float(command[0]), yaw=float(command[1]), height=float(command[2])
                        ),
                        external_torque_nm=external,
                    )
                finally:
                    physics += plant.actual_physics_steps - before
                    total_steps += plant.actual_physics_steps - before
                s = plant.telemetry()
                r = plant.last_record
                e = plant.execution
                rows.append(
                    dict(
                        time=plant.data.time,
                        command=command.copy(),
                        **{
                            k: s[k]
                            for k in (
                                "vx",
                                "vy",
                                "yaw",
                                "height",
                                "roll",
                                "pitch",
                                "q",
                                "dq",
                                "position_xy",
                            )
                        },
                        normal_force_N=[
                            s["normal_force_geometric_right_N"],
                            s["normal_force_geometric_left_N"],
                        ],
                        positive_gap_normal_force_N=[
                            s["positive_gap_normal_force_geometric_right_N"],
                            s["positive_gap_normal_force_geometric_left_N"],
                        ],
                        raw=r["raw"].copy(),
                        observation=r["observation"].copy(),
                        previous=r["previous"].copy(),
                        requested=e.requested.copy(),
                        adopted=e.core.adopted.copy(),
                        raw_PD=e.raw_PD.copy(),
                        driver=e.driver.copy(),
                        motor=e.motor.copy(),
                        NET=e.NET.copy(),
                        external=external.copy(),
                        root_velocity_delta_world=pushes,
                        control_owner=2,
                        policy_sample=(tick % 10 == 0),
                        actor_input_tick_ms=tick // 10 * 10,
                        physical_observation_age_ms=prepared.timing["actor_packet_nominal_ms"],
                    )
                )
                if abs(s["roll"]) > 0.5 or abs(s["pitch"]) > 0.5 or not 0.12 <= s["height"] <= 0.46:
                    termination = "attitude_or_height_boundary"
                    break
        except Exception as exc:
            termination = f"{type(exc).__name__}: {exc}"
        finally:
            if plant is not None:
                plant.close()
        if rows:
            trace = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
            trace = {
                k: v.astype(np.float32) if v.dtype.kind == "f" else v for k, v in trace.items()
            }
            write_npz(output / "trace.npz", trace)
            report = summarize(trace, case, termination, cfg["gates"])
            # Public dataset identity; one physical episode can never be split by frame.
            mask = trace["policy_sample"].astype(bool)
            key = json_digest(
                dict(protocol=receipt["protocol_sha256"], case=case, actor=cfg["actor"]["sha256"])
            )
            write_npz(
                output / "observations.npz",
                dict(
                    states=trace["observation"][mask],
                    episodes=np.zeros(mask.sum(), int),
                    physical_episode_keys=np.full(mask.sum(), key),
                    roles=np.full(mask.sum(), case["role"]),
                    sample_ids=np.array([f"{key}:{i}" for i in np.flatnonzero(mask)]),
                    control_owner=np.full(mask.sum(), 2),
                    transition_valid=np.full(mask.sum(), termination is None),
                    condition_scenario=np.full(mask.sum(), case["scenario"]),
                ),
            )
            report.update(initial_state=initial, trace_sha256=file_sha256(output / "trace.npz"))
        else:
            report = dict(status="UNKNOWN", termination=termination, teacher_eligible=False)
        row = dict(case=case, physics_steps=physics, **report)
        results.append(row)
        write_json(output / "summary.json", row)
        write_json(
            root / "result.json",
            dict(
                **receipt,
                results=results,
                physics_steps=total_steps,
                elapsed_s=time.monotonic() - started,
                all_cases_executed=len(results) == len(cfg["cases"]),
            ),
        )
        if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) >= cfg["max_output_bytes"]:
            break
    final = dict(
        **receipt,
        results=results,
        physics_steps=total_steps,
        elapsed_s=time.monotonic() - started,
        all_cases_executed=len(results) == len(cfg["cases"]),
        C_binding=binding,
        actual_output_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
    )
    write_json(root / "result.json", final)
    return final
