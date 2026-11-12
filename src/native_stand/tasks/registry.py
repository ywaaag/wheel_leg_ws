"""任务注册表：任务ID → 构造/评估/支持组合的唯一静态声明。

只使用轻量数据与import路径字符串；导入本模块不加载Isaac、Torch或资产。
新增任务 = 新增任务模块 + 在此登记；不需要复制owner/worker/预算/停止恢复实现。
"""

TASKS: dict[str, dict] = {
    "motion": {
        "build_module": "native_stand.tasks.motion.spec",
        "build_symbol": "build_task",
        "evaluator_module": "native_stand.learning.ppo.evaluate",
        "evaluator_symbol": "evaluate_checkpoint",
        "supported_algorithms": ("ppo", "distillation"),
        "supported_backends": ("isaac", "cpu", "mujoco"),
        # 任务定义提供的观测键映射（policy factory 在装配边界完成一次映射）
        "observation_keys": {
            "actor": "student_actor",
            "critic": "student_critic",
            "action_valid": "recovery_actor_valid",
        },
    },
    "recovery": {
        "build_module": "native_stand.tasks.recovery.spec",
        "build_symbol": "build_task",
        "evaluator_module": "native_stand.learning.ppo.evaluate",
        "evaluator_symbol": "evaluate_checkpoint",
        # 按动作/分布契约声明的策略构造入口（物理Gaussian），非task名猜测。
        "policy_contracts": {
            "recovery_cubic_physical_Gaussian_v2": {
                "module": "native_stand.learning.ppo.recovery_cubic",
                "symbol": "build_policy",
                "interface": "recovery_cubic_physical_v2",
            },
        },
        "supported_algorithms": ("ppo",),
        "supported_backends": ("isaac", "isaac_recovery", "mujoco", "cpu"),
        "observation_keys": {
            "actor": "student_actor",
            "critic": "student_critic",
            "action_valid": "recovery_actor_valid",
        },
    },
    "jump": {
        "build_module": "native_stand.tasks.jump.spec",
        "build_symbol": "build_task",
        "evaluator_module": "native_stand.learning.ppo.evaluate",
        "evaluator_symbol": "evaluate_checkpoint",
        "supported_algorithms": ("ppo",),
        "supported_backends": ("mujoco", "isaac"),
        "execution_contracts": {
            "jump_isaac_ABI4_raw37_wheel48_v1": {
                "module": "native_stand.simulation.jump_backend",
                "backends": ("isaac",),
            },
            "jump_unified_raw37_wheel48_v1": {
                "module": "native_stand.simulation.jump_backend",
                "backends": ("mujoco",),
            },
        },
        "policy_contracts": {
            "jump_isaac_ABI4_raw37_wheel48_v1": {
                "module": "native_stand.learning.ppo.jump_isaac",
                "symbol": "build_policy",
                "interface": "jump_isaac_ABI4_raw37_wheel48_v1",
            },
            "jump_unified_raw37_wheel48_v1": {
                "module": "native_stand.learning.ppo.jump_unified",
                "symbol": "build_policy",
                "interface": "jump_unified_raw37_wheel48_v1",
            },
        },
        "observation_keys": {
            "actor": "student_actor",
            "critic": "student_critic",
            "action_valid": "recovery_actor_valid",
        },
    },
}


def get_task_definition(task_id: str) -> dict:
    if task_id not in TASKS:
        raise KeyError(f"未登记的任务: {task_id}；可用: {sorted(TASKS)}")
    return TASKS[task_id]


def build_task_from_registry(job):
    """唯一正式的任务构造入口：worker/dispatch 消费同一份定义。"""
    import importlib

    definition = get_task_definition(job.task)
    if job.task == "jump" and job.action.version not in definition["execution_contracts"]:
        raise ValueError("旧Jump ABI不再提供正式任务入口")
    module = importlib.import_module(definition["build_module"])
    return getattr(module, definition["build_symbol"])(job)


def observation_keys(task_id: str) -> dict:
    """任务声明的观测键映射；policy factory 在装配边界完成一次映射。"""
    return get_task_definition(task_id).get("observation_keys", {})


def supports(task_id: str, algorithm: str, backend: str) -> bool:
    definition = get_task_definition(task_id)
    return (
        algorithm in definition["supported_algorithms"]
        and backend in definition["supported_backends"]
    )


def policy_definition(job):
    """由显式动作/分布合同选择装配；相同任务名不决定Actor类型。"""
    contracts = get_task_definition(job.task).get("policy_contracts", {})
    selected = contracts.get(job.action.version)
    if selected is not None and job.model.interface != selected["interface"]:
        raise ValueError("策略输入与动作/分布合同不同")
    if selected is None and any(job.model.interface == x["interface"] for x in contracts.values()):
        raise ValueError("该策略接口缺少明确的动作/分布合同")
    return selected
