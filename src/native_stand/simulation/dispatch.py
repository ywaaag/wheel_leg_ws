"""操作×任务×算法×后端的显式分派表。

不建立插件框架：一张表加少量函数即可，未支持的组合在创建run之前拒绝，
避免把world_model塞进PPO计数循环后返回COMPLETED。
"""

from __future__ import annotations

from typing import Any

from native_stand.experiments.config import JobConfig, load_job_config

from .engine import Backend

OPERATIONS = ("train", "evaluate", "distill", "diagnose")
BACKENDS = {
    "cpu": "native_stand.simulation.cpu",
    "isaac_recovery": "native_stand.simulation.recovery_isaac.backend",
    "isaac": "native_stand.simulation.isaac.engine",
    "mujoco": "native_stand.simulation.mujoco",
}

# 训练组合以 tasks/registry.py 为唯一来源；本表只保留后端模块选择。
TRAINING_BACKEND = {
    "cpu": "native_stand.simulation.cpu",
}


def _training_combinations() -> dict[tuple[str, str], str]:
    """task/algorithm → 后端模块名；来自registry，不在dispatch重复名单。"""
    from native_stand.tasks.registry import TASKS

    combos: dict[tuple[str, str], str] = {}
    for task_id, definition in TASKS.items():
        for algorithm in definition["supported_algorithms"]:
            backends = definition["supported_backends"]
            combos[(task_id, algorithm)] = backends
    return combos


# 评估：当前由统一评估服务执行，后端决定环境来源。
EVALUATION_BACKENDS = ("cpu", "isaac", "mujoco")


class DispatchError(ValueError):
    """组合不支持或后端不可用。"""


def backend_module(job: JobConfig):
    name = (
        execution_service(job)["module"]
        if execution_service(job)["service"] != "unsupported"
        else None
    )
    if name is None:
        raise DispatchError(f"未知后端: {job.backend}；允许: {sorted(BACKENDS)}")
    return __import__(name, fromlist=["make_env"])


def backend_available(module, job: JobConfig) -> tuple[bool, str]:
    """优先问后端自己的available(job)；否则退回模块常量声明。"""
    if hasattr(module, "available"):
        return module.available(job)
    if getattr(module, "ENGINE_AVAILABLE", False):
        return True, ""
    reason = getattr(module, "unavailable_reason", lambda _job: "后端不可用")(job)
    return False, reason


def backend_for(job: JobConfig) -> Backend:
    """训练阶段的后端；组合不支持时在创建run前就失败。"""
    combos = _training_combinations()
    key = (job.task, job.algorithm)
    if key not in combos:
        raise DispatchError(
            f"训练不支持 task={job.task} algorithm={job.algorithm}；允许: {sorted(combos)}"
        )
    service = execution_service(job)
    if service["service"] == "unsupported":
        raise DispatchError(service["reason"])
    module = backend_module(job)
    ok, reason = backend_available(module, job)
    if not ok:
        raise DispatchError(reason)
    if hasattr(module, "validate_job"):
        module.validate_job(job)
    return Backend(
        name=job.backend,
        make_env=module.make_env,
        semantics=getattr(module, "SEMANTICS", {}),
        make_measured_env=getattr(module, "make_measured_env", None),
    )


PRODUCTION_MOTION = "native_stand.simulation.isaac.motion_backend"


def execution_service(job: JobConfig) -> dict[str, Any]:
    """同一有效配置只对应一个执行服务；dry-run与worker都读这里。

    返回 {"service": "production_motion"|"generic", "module": ..., "reason": ...}。
    """
    from native_stand.tasks.registry import get_task_definition

    declared = get_task_definition(job.task).get("execution_contracts", {}).get(job.action.version)
    if declared is not None:
        return {
            "service": "generic" if job.backend in declared["backends"] else "unsupported",
            "module": declared["module"],
            "reason": "显式任务执行合同；未实现后端拒绝",
        }
    if job.backend == "mujoco" and job.assets.get("recovery_mujoco"):
        return {
            "service": "generic",
            "module": "native_stand.simulation.recovery_mujoco",
            "reason": "Explicit Recovery original-XML collection contract",
        }
    if job.backend != "isaac":
        return {
            "service": "generic",
            "module": BACKENDS.get(job.backend, BACKENDS["cpu"]),
            "reason": "非Isaac后端",
        }
    if "runtime_template" in job.assets:
        return {
            "service": "production_motion",
            "module": PRODUCTION_MOTION,
            "reason": "配置声明assets.runtime_template：生产支撑中心Motion适配器",
        }
    if "legacy_spec_path" in job.assets:
        return {
            "service": "generic",
            "module": BACKENDS["isaac"],
            "reason": "配置声明assets.legacy_spec_path：通用Isaac适配器",
        }
    return {
        "service": "unsupported",
        "module": None,
        "reason": "Isaac配置既没有runtime_template也没有legacy_spec_path",
    }


def backend_for_evaluation(job: JobConfig):
    """评估用的后端：与训练选择同一个执行服务，不用另一套字段。"""
    service = execution_service(job)
    if service["service"] == "unsupported":
        raise DispatchError(service["reason"])
    module = __import__(service["module"], fromlist=["make_env"])
    ok, reason = backend_available(module, job)
    if not ok:
        raise DispatchError(reason)
    if not hasattr(module, "make_env"):
        raise DispatchError(f"{service['module']} 没有提供评估环境工厂")
    return Backend(
        name=job.backend,
        make_env=module.make_env,
        semantics=getattr(module, "SEMANTICS", {}),
        condition_hook=getattr(module, "apply_condition", None),
        session_factory=getattr(module, "EvaluationSession", None),
    )


def stage_order(job: JobConfig, budget, *, only: str | None = None) -> list[str]:
    """按预算剩余决定执行顺序；已用额度不重置。"""
    limits = budget.limits()
    used = budget.used()
    if only is not None:
        if only not in limits:
            raise DispatchError(f"未知阶段: {only}")
        if int(used.get(only, 0)) >= int(limits.get(only, 0)):
            return []
        return [only]
    order = [
        stage
        for stage in ("smoke", "formal")
        if int(limits.get(stage, 0)) > 0 and int(used.get(stage, 0)) < int(limits.get(stage, 0))
    ]
    return order


def require_supported(job: JobConfig, operation: str) -> None:
    """在创建run与预约之前核对操作/算法组合。"""
    if operation not in OPERATIONS:
        raise DispatchError(f"未知操作: {operation}")
    if operation == "train":
        backend_for(job)
        return
    if operation == "evaluate":
        if job.task == "jump" and job.action.version == "jump_unified_raw37_wheel48_v1":
            raise DispatchError("Jump课程/选模仅使用正常训练统计；未开放独立评估")
        if job.algorithm != "ppo":
            raise DispatchError(f"evaluate需要algorithm=ppo，当前{job.algorithm}")
        module = backend_module(job)
        ok, reason = backend_available(module, job)
        if not ok:
            raise DispatchError(reason)
        return
    if operation == "distill":
        if job.algorithm not in ("distillation", "ppo"):
            raise DispatchError(f"distill不支持algorithm={job.algorithm}")
        return
    if operation == "diagnose":
        return


def service_description() -> dict[str, Any]:
    """供 --help 之外的只读展示与测试核对。"""
    return {
        "operations": list(OPERATIONS),
        "backends": sorted(BACKENDS),
        "training": sorted(f"{task}/{algorithm}" for task, algorithm in _training_combinations()),
        "evaluation_backends": list(EVALUATION_BACKENDS),
        "isaac_services": ["production_motion", "generic"],
    }


def load_and_require(path, operation: str) -> JobConfig:
    job = load_job_config(path)
    require_supported(job, operation)
    return job
