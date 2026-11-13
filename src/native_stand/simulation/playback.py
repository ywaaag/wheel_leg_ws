"""唯一MuJoCo播放生命周期；GUI只提供输入/显示，dry-run只做计划。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from native_stand.artifacts.io import file_sha256, read_json
from native_stand.experiments.config import job_from_record

ASSET_BINDINGS = Path(__file__).with_name("playback_assets.json")


@dataclass(frozen=True)
class PlaybackPlan:
    run: Path
    checkpoint: Path
    job: object
    assets: dict
    output_contract: dict
    timing: dict

    def record(self):
        return dict(
            run=str(self.run),
            checkpoint=str(self.checkpoint),
            task=self.job.task,
            action_version=self.job.action.version,
            control=self.job.as_run_record()["control"],
            timing=self.timing,
            asset_sha256=self.assets["asset_sha256"],
            C_core="jump_ABI4" if self.job.task == "jump" else "rl_policy_core_v1",
            driver_caps_nm=list(self.job.control.effort_limits_nm),
            intermediate_wheel_PD_cap_nm=max(
                self.job.control.effort_limits_nm[2], self.job.control.effort_limits_nm[5]
            ),
            physics_steps=0,
            DR="nominal anchor",
            execution="one shared GUI/headless loop",
            terrain_map=bool(self.assets.get("terrain_map")),
            solver_profile=self.assets.get("solver_profile", "legacy_xml_v1"),
            capability_accepted=False,
        )


def plan(
    run,
    checkpoint,
    *,
    path_mapping=None,
    checkpoint_sha256=None,
    terrain_map=False,
    solver_profile="legacy_xml_v1",
):
    run = Path(run).resolve()
    record = read_json(run / "run.json")
    if record is None:
        raise ValueError("run.json不存在")
    from native_stand.artifacts.runtime_paths import execution_job

    config = dict(record["config"])
    if path_mapping is not None:
        config["assets"] = dict(
            config["assets"],
            path_mapping={
                "path": str(Path(path_mapping).resolve()),
                "sha256": file_sha256(path_mapping),
            },
        )
    job = execution_job(job_from_record(config))
    path = Path(checkpoint)
    if not path.is_absolute():
        path = run / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if checkpoint_sha256 is not None and file_sha256(path) != checkpoint_sha256:
        raise ValueError("播放checkpoint SHA不同")
    if terrain_map:
        if job.task != "motion":
            raise ValueError("H1交互地图只支持Motion")
        from .playback_terrain import load_atlas

        load_atlas(job.assets["terrain_scene"])
    if job.task == "jump":
        from .jump_backend import validate_job

        validate_job(job)
        return PlaybackPlan(
            run,
            path,
            job,
            dict(
                bindings=job.assets["jump_bindings"],
                asset_sha256=job.assets["jump_bindings"]["asset"]["sha256"],
            ),
            job.assets["jump_contract_data"],
            dict(
                physics_ms=1,
                PD_ms=5,
                Actor_ms=10,
                target_ms=5,
                wheel_ms=[7, 11],
                wheel_tau_ms=[0, 0],
            ),
        )
    if any(job.control.effort_limits_nm[i] != 4.8 for i in (2, 5)):
        raise ValueError("播放器只接受轮限矩4.8 N·m合同；旧限矩用法已退役")
    if job.task == "motion":
        from native_stand.artifacts.runtime_paths import execution_json

        template = execution_json(job.assets["runtime_template"], job.assets)
        if file_sha256(job.assets["runtime_template"]) != job.assets["runtime_template_sha256"]:
            raise ValueError("播放器环境模板SHA不同")
        contract_ref = {
            "path": job.assets["output_contract"],
            "sha256": job.assets["output_contract_sha256"],
        }
        runtime = None
    elif job.task == "recovery":
        ref = job.assets["recovery_isaac_runtime"]
        if file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Recovery执行来源SHA不同")
        runtime = read_json(ref["path"])
        if runtime["version"] != "recovery_standalone_Isaac_runtime_v1":
            raise ValueError("只接入Recovery全程控制的高度版本")
        ref = runtime["bindings"]["scene_spec"]
        if file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Recovery场景来源SHA不同")
        template = read_json(ref["path"])
        contract_ref = job.assets["recovery_training_contract"]
    else:
        raise ValueError("该任务需要显式播放执行适配")
    from native_stand.artifacts.runtime_paths import RuntimePaths

    assets = RuntimePaths.from_assets(job.assets).view(read_json(ASSET_BINDINGS))
    if assets["asset_sha256"] != template["asset_sha256"]:
        raise ValueError("该USD尚无绑定的MuJoCo机械描述")
    for name, item in assets["bindings"].items():
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError("播放器资产身份不同: " + name)
    if file_sha256(template["asset_path"]) != assets["asset_sha256"]:
        raise ValueError("原始USD身份不同")
    output = read_json(contract_ref["path"])
    if file_sha256(contract_ref["path"]) != contract_ref["sha256"]:
        raise ValueError("播放器输出合同SHA不同")
    if output["action_version"] != job.action.version:
        raise ValueError("模型与动作版本不同")
    if job.task == "motion":
        from native_stand.tasks.motion.action_contract import ActionDecoder
        from native_stand.tasks.motion.spec import build_task

        decoder = ActionDecoder.from_contract(output)
        if decoder.center != build_task(job).center:
            raise ValueError("播放器冻结中心不同")
        from native_stand.tasks.motion.delayed_execution import validate_configuration

        timing = dict(validate_configuration(job.assets["delayed_execution"]))
    else:
        from native_stand.tasks.recovery.height_context import VERSION

        if job.model.critic_privileged_version != VERSION:
            raise ValueError("独立Recovery高度版本必须绑定909D因果合同")
        if runtime["bindings"]["V4_contract"]["sha256"] != assets["bindings"]["V4"]["sha256"]:
            raise ValueError("Recovery V4与绑定机械模型不同")
        assets["bindings"]["initial_states"] = runtime["bindings"]["initial_states"]
        from native_stand.simulation.recovery_isaac.execution import nominal_timing

        timing = nominal_timing()
    if (job.control.physics_dt_s, job.control.pd_dt_s, job.control.policy_dt_s) != (
        0.001,
        0.005,
        0.01,
    ):
        raise ValueError("该执行版本要求显式1/5/10ms时钟")
    from native_stand.students.mujoco_solver import PROFILES

    if solver_profile not in PROFILES:
        raise ValueError("未知MuJoCo求解器profile")
    assets["solver_profile"] = solver_profile
    assets["terrain_map"] = terrain_map
    assets["checkpoint_sha256"] = checkpoint_sha256
    return PlaybackPlan(run, path, job, assets, output, timing)


@dataclass
class Controls:
    paused: bool = False
    reset_requested: bool = False
    teacher: bool = False
    jump_requested: bool = False
    vx: float = 0.0
    yaw: float = 0.0
    height: float | None = None
    quit_requested: bool = False

    def key(self, keycode):
        if keycode == 32:
            self.paused = not self.paused
        elif keycode in (ord("R"), ord("r")):
            self.reset_requested = True
        elif keycode in (ord("J"), ord("j")):
            self.jump_requested = True
        elif keycode in (ord("T"), ord("t")):
            self.teacher = not self.teacher
        elif keycode == 265:
            self.vx += 0.25
        elif keycode == 264:
            self.vx -= 0.25
        elif keycode == 263:
            self.yaw += 0.25
        elif keycode == 262:
            self.yaw -= 0.25


def run_loop(plant, controls, *, steps, viewer=None):
    """暂停不推进任何时钟；切模保持现场；只有显式reset重建物理初态。"""
    import time

    completed = 0
    render_at = 0.0
    while steps is None or completed < steps:
        if viewer is not None and not viewer.is_running():
            break
        if hasattr(controls, "sync"):
            controls.sync(plant)
        if controls.quit_requested:
            break
        if controls.reset_requested:
            plant.reset()
            controls.reset_requested = False
        if not controls.paused:
            try:
                plant.step(controls)
                completed += 1
            except (ValueError, FloatingPointError) as error:
                if not hasattr(controls, "fault"):
                    raise
                controls.fault(str(error))
        if viewer is not None:
            now = time.monotonic()
            if controls.paused or now >= render_at:
                if getattr(plant, "web_owner", False):
                    viewer.cam.lookat[:] = plant.data.qpos[:3]
                viewer.sync()
                render_at = now + 1 / 60
            time.sleep(1 / 60 if controls.paused else max(0.0, plant.physics_dt))
        elif controls.paused:
            raise ValueError("headless暂停需要外部输入源，拒绝无界空转")
    return dict(
        physics_steps=completed, control_tick=plant.execution.tick, capability_accepted=False
    )


def create_player(prepared, *, directory=None):
    import tempfile

    from native_stand.artifacts.checkpoint import load_actor_source

    from .control_core import ControlCore, build
    from .mujoco_playback import MotionPlayer

    source = load_actor_source(prepared.checkpoint)
    expected_sha = prepared.assets.get("checkpoint_sha256")
    if expected_sha is not None and source.sha256 != expected_sha:
        raise ValueError("打开的实际模型SHA不同")
    output = source.payload.get("output_contract", source.payload.get("contract"))
    if (
        source.payload.get(
            "action_version", (output or {}).get("action_version", (output or {}).get("version"))
        )
        != prepared.job.action.version
    ):
        raise ValueError("实际模型缺少匹配的明确动作版本")
    if output is None:
        raise ValueError("实际模型缺少输出合同")
    if prepared.job.task == "jump":
        from .jump_player import JumpPlayer

        folder = Path(directory or tempfile.mkdtemp(prefix="wheel-leg-play-"))
        plant = JumpPlayer(prepared, source.payload, folder)
        binding = dict(ABI=4, path=plant.execution.c.library, sha256=plant.execution.c.library_sha)
        return source, plant, binding, folder
    if prepared.job.task == "motion":
        from native_stand.tasks.motion.encoding import INPUT_VERSION

        if source.payload.get("interface_version") != INPUT_VERSION:
            raise ValueError("Motion checkpoint缺少明确的31D语义版本")
        if source.payload.get("execution_timing") != prepared.timing:
            raise ValueError("Motion checkpoint时序与执行合同不同")
        for field, value in (
            ("input_leg_q_and_previous_scale_rad", 1.5),
            ("input_wheel_previous_scale_rad_s", 100.0),
        ):
            if output.get(field) != value:
                raise ValueError("Motion checkpoint输入尺度不同")
    from .control_core import decoder_configuration

    if bytes(decoder_configuration(output)) != bytes(
        decoder_configuration(prepared.output_contract)
    ):
        raise ValueError("实际模型与播放执行合同不同")
    folder = Path(directory or tempfile.mkdtemp(prefix="wheel-leg-play-"))
    binding = build(folder / "control.so")
    core = ControlCore(binding, output)
    from native_stand.learning.inference import from_source, spec_for_job

    inference = from_source(source, spec_for_job(prepared.job, prepared.output_contract))
    plant = MotionPlayer(prepared, source.payload, core, actor=inference.actor)
    return source, plant, binding, folder


def run(
    prepared, *, headless=False, steps=None, panel_port=None, panel_host="127.0.0.1", no_save=False
):
    import shutil
    import tempfile

    # Web sessions are always ephemeral, regardless of the recording default for batch playback.
    no_save = no_save or panel_port is not None
    # Build products are ephemeral; no-save never writes trajectory/result/log files.
    folder = Path(tempfile.mkdtemp(prefix="wheel-leg-play-", dir="/dev/shm" if no_save else None))
    try:
        source, plant, binding, _ = create_player(prepared, directory=folder)
        plant.recording = not no_save
        return _run_player(
            prepared,
            source,
            plant,
            binding,
            folder,
            headless=headless,
            steps=steps,
            panel_port=panel_port,
            panel_host=panel_host,
            no_save=no_save,
        )
    finally:
        if no_save:
            shutil.rmtree(folder)


def _run_player(
    prepared,
    source,
    plant,
    binding,
    folder,
    *,
    headless,
    steps,
    panel_port=None,
    panel_host="127.0.0.1",
    no_save=False,
):
    server = None
    try:
        controls = Controls()
        if panel_port is not None:
            if prepared.job.task != "motion":
                raise ValueError("网页交互目前只支持Motion")
            from .playback_panel import WebControls, start

            controls = WebControls(plant)
            server, _ = start(controls.plan, panel_port, host=panel_host)
            print(
                f"Motion panel http://{panel_host}:{server.server_port} (no-save={no_save})",
                flush=True,
            )
        if headless:
            if steps is None or steps < 1:
                raise ValueError("headless必须给出正整数--steps")
            result = run_loop(plant, controls, steps=steps)
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(
                plant.model, plant.data, key_callback=controls.key
            ) as viewer:
                viewer.cam.distance = 2.5
                viewer.cam.elevation = -20
                viewer.cam.lookat[:] = plant.data.qpos[:3]
                result = run_loop(plant, controls, steps=steps, viewer=viewer)
        result.update(
            checkpoint=str(source.path), sha256=source.sha256, C_core=binding, saved=not no_save
        )
        if not no_save:
            from native_stand.artifacts.io import write_json, write_npz

            write_npz(folder / "trace.npz", plant.trace())
            result["trace_output"] = str(folder / "trace.npz")
            write_json(folder / "result.json", result)
        return result
    finally:
        if server is not None:
            from native_stand.students.motion_panel import stop

            stop(server)
        plant.close()
