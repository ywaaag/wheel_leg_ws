"""ABI4 MuJoCo environment factory; public lifecycle uses the common PPO adapter."""

from pathlib import Path

from native_stand.artifacts.io import file_sha256, read_json
from native_stand.tasks.jump.spec import build_task
from native_stand.tasks.jump.unified import VERSION


def validate_job(job):
    build_task(job)
    if job.action.version != VERSION or job.backend != "mujoco" or job.runtime.device != "cpu":
        raise ValueError("ABI4 Jump仅开放当前MuJoCo CPU执行链；Isaac未实现")
    from native_stand.tasks.jump.adaptive_course import VERSION as COURSE
    from native_stand.tasks.jump.unified_reward import VERSION as REWARD

    if job.reward.version != REWARD or job.curriculum.version != COURSE:
        raise ValueError("Jump奖励/课程版本不同")
    if (job.control.physics_dt_s, job.control.pd_dt_s, job.control.policy_dt_s) != (
        0.001,
        0.005,
        0.01,
    ):
        raise ValueError("Jump物理/PD/Actor时钟不同")
    if (job.action.leg_scale_rad, job.action.wheel_scale_rad_s) != (0.5, 10.0):
        raise ValueError("Jump raw增益不同")
    for ref in job.assets["jump_bindings"].values():
        if file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Jump来源身份不同: " + ref["path"])


def preflight(job):
    """Shared owner submission boundary; no engine, compilation or integration."""
    validate_job(job)
    return dict(
        status="CPU_PREFLIGHT_PASS",
        backend="MuJoCo CPU",
        Isaac_GPU=False,
        physical_steps=0,
        PPO_updates=0,
    )


def make_plant(job, *, output, charge_step):
    from .jump_build import build
    from .jump_response import Response
    from .jump_unified import UnifiedPlant

    refs = job.assets["jump_bindings"]
    binding = build(Path(output) / "jump_abi4.so")
    plant = UnifiedPlant(
        read_json(refs["description"]["path"]),
        read_json(refs["spring_geometry"]["path"])["springs"],
        read_json(refs["mechanics"]["path"])["mechanics"],
        read_json(refs["V4"]["path"]),
        Response,
        unified_library=binding["path"],
        unified_library_sha=binding["sha256"],
        charge_step=charge_step,
        solver="closure_contact_tight_v1",
    )
    plant.binding = binding
    return plant


def make_env(job, *, output=None):
    import tempfile

    from native_stand.tasks.jump.adaptive_course import AdaptiveCourse

    from .jump_batch import JumpBatch
    from .jump_unified_learning import UnifiedLearningEnv

    validate_job(job)
    output = Path(output or tempfile.mkdtemp(prefix="jump-execution-"))
    plants = []
    try:
        for index in range(job.runtime.envs):
            course = AdaptiveCourse(seed=job.runtime.seed + index, **job.assets["jump_course"])
            plant = make_plant(job, output=output / str(index), charge_step=lambda n: None)
            plants.append(UnifiedLearningEnv(plant, course=course))
        return JumpBatch(plants, job)
    except BaseException:
        for env in plants:
            env.close()
        raise


def available(job):
    if job.task != "jump" or job.action.version != VERSION or job.backend != "mujoco":
        return False, "只开放ABI4 Jump MuJoCo组合"
    return True, ""


def make_collection_env(job, *, resources):
    if resources != job.runtime:
        raise ValueError("Collection resources differ from runtime request")
    return make_env(job)
