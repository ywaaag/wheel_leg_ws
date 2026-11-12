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
            capability_accepted=False,
        )


def plan(run, checkpoint):
    run = Path(run).resolve()
    record = read_json(run / "run.json")
    if record is None:
        raise ValueError("run.json不存在")
    from native_stand.artifacts.runtime_paths import execution_job

    job = execution_job(job_from_record(record["config"]))
    path = Path(checkpoint)
    if not path.is_absolute():
        path = run / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
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
    if job.task == "motion":
        template = read_json(job.assets["runtime_template"])
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
    return PlaybackPlan(run, path, job, assets, output, timing)


@dataclass
class Controls:
    paused: bool = False
    reset_requested: bool = False
    teacher: bool = False
    jump_requested: bool = False
    vx: float = 0.0
    yaw: float = 0.0

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
        if controls.reset_requested:
            plant.reset()
            controls.reset_requested = False
        if not controls.paused:
            plant.step(controls)
            completed += 1
        if viewer is not None:
            now = time.monotonic()
            if controls.paused or now >= render_at:
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
    from .control_core import decoder_configuration

    if bytes(decoder_configuration(output)) != bytes(
        decoder_configuration(prepared.output_contract)
    ):
        raise ValueError("实际模型与播放执行合同不同")
    folder = Path(directory or tempfile.mkdtemp(prefix="wheel-leg-play-"))
    binding = build(folder / "control.so")
    core = ControlCore(binding, output)
    plant = MotionPlayer(prepared, source.payload, core)
    return source, plant, binding, folder


def run(prepared, *, headless=False, steps=None):
    source, plant, binding, folder = create_player(prepared)
    return _run_player(prepared, source, plant, binding, folder, headless=headless, steps=steps)


def _run_player(prepared, source, plant, binding, folder, *, headless, steps):
    try:
        controls = Controls()
        if headless:
            if steps is None or steps < 1:
                raise ValueError("headless必须给出正整数--steps")
            result = run_loop(plant, controls, steps=steps)
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(
                plant.model, plant.data, key_callback=controls.key
            ) as viewer:
                result = run_loop(plant, controls, steps=steps, viewer=viewer)
        from native_stand.artifacts.io import write_json, write_npz

        write_npz(folder / "trace.npz", plant.trace())
        result.update(
            checkpoint=str(source.path),
            sha256=source.sha256,
            C_core=binding,
            trace_output=str(folder / "trace.npz"),
        )
        write_json(folder / "result.json", result)
        return result
    finally:
        plant.close()
