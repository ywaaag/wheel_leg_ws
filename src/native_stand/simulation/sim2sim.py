"""Explicit recorded or sequential live cross-engine diagnostics; never a trainer."""

from pathlib import Path

from native_stand.artifacts.io import file_sha256, json_digest, read_json, write_json, write_npz

REQUIRED = (
    "model",
    "input_contract",
    "output_contract",
    "scene",
    "command",
    "initial_state",
    "timing",
    "V4",
    "motor_caps",
    "scoring_window_s",
)


def plan(config):
    if config.get("version") == "sim2sim_live_protocol_v1":
        return live_plan(config)
    if config.get("version") != "sim2sim_recorded_protocol_v1":
        raise ValueError("Unknown sim2sim protocol")
    missing = [key for key in REQUIRED if key not in config]
    if missing or len(config.get("traces", ())) != 2:
        raise ValueError("Incomplete explicit sim2sim protocol: " + str(missing))
    for trace in config["traces"]:
        if not {"path", "sha256", "engine", "integration", "contact_sampling"} <= trace.keys():
            raise ValueError("Trace requires engine/integration/contact semantics")
    return dict(
        operation="sim2sim",
        mode="recorded",
        protocol=config,
        writes=False,
        starts_engine=False,
        extra_simulation_steps=0,
        capability_accepted=False,
    )


def diagnose(config, output):
    import numpy as np

    plan(config)
    model = None
    if "path" in config["model"]:
        from native_stand.learning.inference import InferenceSpec, load_model

        model = load_model(
            config["model"]["path"],
            InferenceSpec.from_record(config["model"]["inference_spec"]),
            sha256=config["model"]["sha256"],
        )
    rows = []
    low, high = config["scoring_window_s"]
    for ref in config["traces"]:
        if file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Diagnostic trace identity differs")
        with np.load(ref["path"], allow_pickle=False) as data:
            time = data["time"]
            if time.ndim != 1 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
                raise ValueError("Trace time is not finite/strictly increasing")
            mask = (time >= low) & (time < high)
            complete = bool(len(time) and time[0] <= low and time[-1] >= high)
            metrics = {}
            for name in ("q", "dq", "motor", "NET", "state"):
                if name not in data:
                    raise ValueError("Missing raw diagnostic signal: " + name)
                values = data[name][mask].astype(np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite diagnostic signal")
                metrics[name] = (
                    dict(
                        mean=values.mean(0).tolist(),
                        rms=np.sqrt(np.mean(values * values, axis=0)).tolist(),
                    )
                    if len(values)
                    else None
                )
            if model is not None:
                import torch

                if "observation" not in data or "policy_mean" not in data:
                    raise ValueError(
                        "Model-bound sim2sim traces require observation and policy_mean"
                    )
                predicted = model.mean(
                    torch.from_numpy(data["observation"][mask].astype("float32"))
                ).numpy()
                actual = data["policy_mean"][mask]
                metrics["policy_mean_replay_max_abs_error"] = (
                    float(np.max(np.abs(predicted - actual))) if len(predicted) else None
                )
            rows.append(
                dict(
                    engine=ref["engine"],
                    integration=ref["integration"],
                    contact_sampling=ref["contact_sampling"],
                    fixed_window_complete=complete,
                    samples=int(mask.sum()),
                    metrics=metrics,
                    termination_reason=ref.get("termination_reason", "UNKNOWN"),
                )
            )
    result = dict(
        version="sim2sim_recorded_diagnostics_v1",
        protocol=config,
        engines=rows,
        status="COMPLETE" if all(r["fixed_window_complete"] for r in rows) else "UNKNOWN",
        physical_equivalence_claimed=False,
        capability_accepted=False,
        extra_simulation_steps=0,
    )
    write_json(output, result)
    return result


def from_file(path, output=None, *, dry_run=False):
    config = read_json(Path(path))
    if dry_run:
        return plan(config)
    return (
        run_live(config, output)
        if config.get("version") == "sim2sim_live_protocol_v1"
        else diagnose(config, output)
    )


def live_plan(config):
    if any(key not in config for key in REQUIRED) or len(config.get("backends", ())) != 2:
        raise ValueError("Live sim2sim requires complete explicit protocol and two backends")
    if type(config.get("steps")) is not int or config["steps"] < 1:
        raise ValueError("Live sim2sim requires a positive bounded policy-step budget")
    if not {"path", "sha256", "inference_spec"} <= config["model"].keys():
        raise ValueError("Live diagnostic requires a content-identified inference model")
    from native_stand.learning.inference import InferenceSpec

    spec = InferenceSpec.from_record(config["model"]["inference_spec"])
    if json_digest(spec.input_contract) != json_digest(config["input_contract"]) or json_digest(
        spec.output_contract
    ) != json_digest(config["output_contract"]):
        raise ValueError("Model inference contracts differ from diagnostic protocol")
    timing = config["timing"]
    if timing.get("physics_ms", 0) <= 0 or timing.get("policy_ms", 0) <= 0:
        raise ValueError("Live protocol requires explicit positive physical/policy periods")
    if config.get("max_output_bytes", 1) <= 0 or config.get("wall_s", 1) <= 0:
        raise ValueError("Live resource bounds must be positive")
    for endpoint in config["backends"]:
        if endpoint.get("adapter") not in ("collection", "mujoco_playback"):
            raise ValueError("Unsupported diagnostic execution adapter")
        required = ("engine", "integration", "contact_sampling")
        if any(key not in endpoint for key in required):
            raise ValueError("Backend must declare integration/contact sampling")
        if endpoint["adapter"] == "collection":
            from native_stand.experiments.config import load_job_config

            from .dispatch import execution_service

            job = load_job_config(endpoint["config"])
            if job.assets.get("application_lifecycle", {}).get("owner") == "worker_atexit":
                raise ValueError(
                    "Sequential sim2sim requires synchronous engine close; worker_atexit is unsupported"
                )
            if job.runtime.envs != 1:
                raise ValueError(
                    "Explicit cross-engine protocol currently requires one environment per sequential backend"
                )
            service = execution_service(job)
            if service["service"] == "unsupported":
                raise ValueError(service["reason"])
        elif not {"run", "checkpoint"} <= endpoint.keys():
            raise ValueError("Playback endpoint requires explicit run/checkpoint")
    return dict(
        operation="sim2sim",
        mode="live",
        protocol=config,
        sequential_engines=True,
        requested_policy_steps_per_backend=config["steps"],
        writes=False,
        starts_engine=False,
        capability_accepted=False,
    )


class CollectionDiagnostic:
    def __init__(self, endpoint, protocol, directory):
        from native_stand.experiments.config import load_job_config

        from .collection import make_environment

        self.env = None
        try:
            job = load_job_config(endpoint["config"])
            self.env = make_environment(job)
            if not hasattr(self.env, "configure_diagnostic"):
                raise ValueError(
                    "Collection backend lacks explicit diagnostic condition/readback interface"
                )
            self.actual = self.env.configure_diagnostic(protocol)
            initial = getattr(self.env, "_snapshot", None)
            if initial is not None:
                self.actual["actual_initial_state"] = {
                    key: value.detach().cpu().tolist() if hasattr(value, "detach") else value
                    for key, value in initial.items()
                }
            self.physics_dt = job.control.physics_dt_s
        except BaseException:
            self.close()
            raise

    def observations(self):
        return self.env.observations()["student_actor"]

    def step(self, mean):
        return self.env.step(mean)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


class PlaybackDiagnostic:
    def __init__(self, endpoint, protocol, directory):
        import numpy as np

        from native_stand.learning.inference import spec_for_job

        from .playback import Controls, create_player
        from .playback import plan as playback_plan

        self.plant = None
        self.prepared = playback_plan(endpoint["run"], endpoint["checkpoint"])
        prepared = self.prepared
        source, self.plant, _, _ = create_player(prepared, directory=directory)
        try:
            if prepared.job.assets.get("terrain_scene"):
                raise ValueError("MuJoCo playback has no declared rough-terrain diagnostic adapter")
            if prepared.job.task == "jump":
                raise ValueError(
                    "Jump playback trace protocol needs its own state schema; use collection adapter"
                )
            if source.sha256 != protocol["model"]["sha256"]:
                raise ValueError("Playback Actor identity differs")
            self.controls = Controls(vx=protocol["command"][0], yaw=protocol["command"][1])
            self.physics_dt = prepared.job.control.physics_dt_s
            self.policy_substeps = round(prepared.job.control.policy_dt_s / self.physics_dt)
            self.actual = dict(
                input_contract=spec_for_job(prepared.job, prepared.output_contract).input_contract,
                output_contract=prepared.output_contract,
                command=[
                    self.controls.vx,
                    self.controls.yaw,
                    prepared.job.control.height_command_m,
                ],
                scene=_scene_readback(self.plant, protocol["scene"]),
                initial_state={"kind": "model_bound_reset", "seed": prepared.job.runtime.seed},
                actual_initial_state=dict(
                    root_pose_xyzw=np.concatenate(
                        (self.plant.data.qpos[:3], self.plant.data.qpos[[4, 5, 6, 3]])
                    ).tolist(),
                    root_velocity=self.plant.data.qvel[:6].tolist(),
                    q=self.plant.data.qpos[self.plant.qids].tolist(),
                    dq=self.plant.data.qvel[self.plant.vids].tolist(),
                    time_s=float(self.plant.data.time),
                ),
                timing=dict(
                    physics_ms=self.physics_dt * 1000,
                    PD_ms=self.plant.execution.control.pd_dt_s * 1000,
                    policy_ms=self.plant.execution.control.policy_dt_s * 1000,
                    actor_packet_ms=self.plant.execution.timing["actor_packet_nominal_ms"],
                    actor_leg_packet_ms=self.plant.execution.timing["actor_leg_packet_nominal_ms"],
                    target_ms=self.plant.execution.timing["target_nominal_ms"],
                    wheel_transport_ms=self.plant.execution.timing["wheel_transport_nominal_ms"],
                    wheel_tau_ms=self.plant.execution.timing["wheel_tau_ms"],
                ),
                V4=dict(protocol["V4"]),
                motor_caps=list(self.plant.execution.control.effort_limits_nm),
            )
            self.actual.update(
                engine="MuJoCo",
                integration=dict(
                    dt_s=float(self.plant.model.opt.timestep),
                    integrator=int(self.plant.model.opt.integrator),
                    iterations=int(self.plant.model.opt.iterations),
                    tolerance=float(self.plant.model.opt.tolerance),
                ),
                contact_sampling={
                    "kind": "policy_end_state_last_applied_motor_NET; solver contact semantics retained"
                },
            )
            if protocol["V4"].get("sha256") != prepared.assets["bindings"]["V4"]["sha256"]:
                raise ValueError("Playback V4 identity differs")
            self.np = np
        except BaseException:
            self.close()
            raise

    def observations(self):
        # Player owns its exact delayed Actor encoding and deterministic factory.
        return None

    def step(self, mean):
        p = self.plant
        before = float(p.data.time)
        failure = None
        try:
            for _ in range(self.policy_substeps):
                p.step(self.controls)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            failure = repr(exc)
        executed = round((float(p.data.time) - before) / self.physics_dt)
        record = p.records[-1] if p.records else p.last_record
        if record is None:
            raise RuntimeError("Playback failed before any diagnostic observation")
        state = self.np.concatenate((p.data.qpos[:3], p.data.qpos[[4, 5, 6, 3]], p.data.qvel[:6]))
        return dict(
            terminated=[failure is not None],
            truncated=[False],
            extras=dict(
                actual_physics_steps=[executed],
                failure=failure,
                diagnostic=dict(
                    q=p.data.qpos[p.qids][None],
                    dq=p.data.qvel[p.vids][None],
                    motor=p.execution.motor[None].copy(),
                    NET=p.execution.NET[None].copy(),
                    state=state[None],
                ),
                observation=record["observation"][None],
                policy_mean=record["raw"][None],
            ),
        )

    def close(self):
        if self.plant is not None:
            self.plant.close()
            self.plant = None


def _scene_readback(plant, declared):
    """Translate explicit flat-scene units only; never copy an unverified surface."""
    from dataclasses import asdict

    actual = asdict(plant.scene_config)
    if declared.get("version") == "flat_scene_v1":
        return actual
    frozen_plane = {
        "collider",
        "height_m",
        "normal_axis",
        "width_m",
        "length_m",
        "static_friction",
        "dynamic_friction",
        "restitution",
        "material_path",
    }
    if set(declared) == frozen_plane and (
        declared["collider"] == "plane"
        and declared["normal_axis"] == "Z"
        and declared["height_m"] == actual["ground_z_m"]
        and declared["static_friction"] == declared["dynamic_friction"] == actual["friction"]
        and declared["restitution"] == 0
    ):
        return dict(declared)
    # Frozen Isaac templates describe a plane by height and friction rather
    # than SceneConfig. Only that exact subset can share the same declaration.
    if set(declared) == {"height_m", "friction"} and declared == {
        "height_m": actual["ground_z_m"],
        "friction": actual["friction"],
    }:
        return dict(declared)
    raise ValueError("Playback actual flat scene differs from diagnostic protocol")


def make_diagnostic_session(endpoint, protocol, directory):
    cls = CollectionDiagnostic if endpoint["adapter"] == "collection" else PlaybackDiagnostic
    return cls(endpoint, protocol, directory)


def run_live(config, output):
    import numpy as np
    import torch

    from native_stand.learning.inference import InferenceSpec, load_model

    live_plan(config)
    import time

    output = Path(output)
    directory = output.parent / (output.stem + "_traces")
    directory.mkdir(parents=True, exist_ok=False)
    model = load_model(
        config["model"]["path"],
        InferenceSpec.from_record(config["model"]["inference_spec"]),
        sha256=config["model"]["sha256"],
    )
    traces = []
    for index, endpoint in enumerate(config["backends"]):
        session = None
        rows = []
        physical_steps = 0
        reason = "bounded_steps"
        started = time.monotonic()
        bytes_retained = 0
        inference_device = None
        try:
            session = make_diagnostic_session(endpoint, config, directory / str(index))
            for key in ("engine", "integration", "contact_sampling"):
                if key not in session.actual or json_digest(session.actual[key]) != json_digest(
                    endpoint[key]
                ):
                    raise ValueError("Actual backend implementation differs: " + key)
            for key in REQUIRED[1:-1]:
                if key not in session.actual or json_digest(session.actual[key]) != json_digest(
                    config[key]
                ):
                    raise ValueError("Actual backend diagnostic condition differs: " + key)
            for _ in range(config["steps"]):
                obs = session.observations()
                if obs is not None:
                    if inference_device != obs.device:
                        model.actor.to(obs.device)
                        inference_device = obs.device
                    obs = obs.detach().clone()
                    mean = model.mean(obs)
                else:
                    mean = None
                result = session.step(mean)
                extra = result["extras"]
                durations = np.asarray(extra["actual_physics_steps"], dtype=np.int64)
                if durations.shape != (1,) or durations[0] < 0:
                    raise ValueError("Diagnostic requires one real environment and actual substeps")
                physical_steps += int(durations[0])
                diagnostic = extra["diagnostic"]
                row = {"time": physical_steps * session.physics_dt}
                for key in ("q", "dq", "motor", "NET", "state"):
                    value = diagnostic[key]
                    if torch.is_tensor(value):
                        value = value.detach().cpu().numpy()
                    value = np.asarray(value)
                    if len(value) != 1 or not np.isfinite(value).all():
                        raise ValueError("Invalid actual diagnostic signal: " + key)
                    row[key] = value[0].copy()
                actual_obs = extra.get("observation", obs)
                actual_mean = extra.get("policy_mean", mean)
                for key, value in (("observation", actual_obs), ("policy_mean", actual_mean)):
                    if torch.is_tensor(value):
                        value = value.detach().cpu().numpy()
                    row[key] = np.asarray(value)[0].copy()
                rows.append(row)
                bytes_retained += sum(
                    value.nbytes for value in row.values() if hasattr(value, "nbytes")
                )
                if bytes_retained >= config.get("max_output_bytes", 256 * 1024 * 1024):
                    reason = "output_limit"
                    break
                if time.monotonic() - started >= config.get("wall_s", 3600):
                    reason = "wall_limit"
                    break
                ended = bool(torch.as_tensor(result["terminated"]).any())
                timeout = bool(torch.as_tensor(result["truncated"]).any())
                expected_substeps = round(
                    config["timing"]["policy_ms"] / config["timing"]["physics_ms"]
                )
                short = durations[0] != expected_substeps
                if extra.get("failure") or ended or timeout or short:
                    reason = extra.get("failure") or (
                        "terminated" if ended else "truncated" if timeout else "short_step_fault"
                    )
                    break
        except BaseException as failure:
            receipt = getattr(failure, "actual_physics_steps", None)
            if receipt is not None:
                receipt = np.asarray(receipt)
                if receipt.shape == (1,) and np.isfinite(receipt).all() and receipt[0] >= 0:
                    physical_steps += int(receipt[0])
                else:
                    failure.add_note("Invalid physical-step fault receipt; consumption UNKNOWN")
            if rows:
                write_npz(
                    directory / f"backend_{index}_partial.npz",
                    {key: np.stack([row[key] for row in rows]) for key in rows[0]},
                )
            write_json(
                directory / f"backend_{index}_failure.json",
                dict(
                    error=repr(failure),
                    actual_physics_steps=physical_steps,
                    fault_substeps=receipt.tolist() if receipt is not None else "UNKNOWN",
                    fault_state="UNKNOWN",
                    samples=len(rows),
                ),
            )
            if session is not None:
                try:
                    session.close()
                except BaseException as closing:
                    failure.add_note("Diagnostic cleanup failed: " + repr(closing))
                session = None
            raise
        finally:
            if session is not None:
                session.close()
        if not rows:
            raise ValueError("Diagnostic backend produced no actual samples")
        path = directory / f"backend_{index}.npz"
        write_npz(path, {key: np.stack([row[key] for row in rows]) for key in rows[0]})
        trace = dict(
            path=str(path.resolve()),
            sha256=file_sha256(path),
            engine=endpoint["engine"],
            integration=endpoint["integration"],
            contact_sampling=endpoint["contact_sampling"],
            termination_reason=reason,
            actual_physics_steps=physical_steps,
            actual_conditions=session.actual,
        )
        traces.append(trace)
        write_json(
            directory / "collection.json", dict(traces=traces, completed_backends=len(traces))
        )
    recorded = {**config, "version": "sim2sim_recorded_protocol_v1", "traces": traces}
    result = diagnose(recorded, output)
    result.update(
        mode="live",
        extra_simulation_steps=sum(t["actual_physics_steps"] for t in traces),
        sequential_engines=True,
    )
    write_json(output, result)
    return result
