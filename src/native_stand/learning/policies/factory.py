"""策略构造入口：由配置决定网络结构，与技能名称解耦。

当前只支持MLP；未知网络类型明确拒绝。观测/动作键名映射在装配边界完成，
公共factory不强制 "student_actor" 等任务键名；探索尺度来自动作契约（逐通道包络），
不按通道名称猜测物理单位。
"""

from typing import Any

from ..ppo import assembly


def build_policy(job, observation_sample, *, device: str = "cpu", source=None) -> dict[str, Any]:
    """按配置构造Actor/Critic/优化器装配；观测样例由任务环境提供。

    观测键名来自任务registry声明（装配边界完成一次canonical映射）；
    网络结构由配置决定（仅mlp），不按技能名称或通道名称猜测。
    """
    from native_stand.tasks.registry import policy_definition

    selected = policy_definition(job)
    if selected:
        import importlib

        module = importlib.import_module(selected["module"])
        return getattr(module, selected["symbol"])(
            job, observation_sample, device=device, source=source
        )
    if job.model.kind != "mlp":
        raise ValueError(f"不支持的策略网络类型: {job.model.kind}；当前仅支持mlp")
    from native_stand.tasks.registry import observation_keys as task_keys

    keys = task_keys(job.task)
    # canonical映射：任务键名 → 冻结算法内部键名，一次完成；缺失即拒绝。
    internal = {
        "actor": "student_actor",
        "critic": "student_critic",
        "action_valid": "recovery_actor_valid",
    }  # 冻结装配消费的内部键名
    missing = [
        task_key
        for canonical, task_key in keys.items()
        if canonical in internal and task_key not in observation_sample
    ]
    if missing:
        raise ValueError(f"观测缺少任务声明的键: {missing}")
    sample_for_assembly = observation_sample
    if any(keys[c] != internal[c] for c in internal if c in keys):
        # 任务键名与内部键名不同：构造映射后的样例供冻结装配消费。
        from tensordict import TensorDict

        renamed = {}
        for canonical, task_key in keys.items():
            if task_key in observation_sample:
                renamed[internal.get(canonical, task_key)] = observation_sample[task_key]
        sample_for_assembly = TensorDict(renamed, batch_size=observation_sample.batch_size)
    ppo = assembly.assemble_ppo(
        assembly.build_actor(job),
        assembly.build_critic(job),
        sample_for_assembly,
        job,
        device=device,
    )
    # 推理边界映射：任务键名观测 → 内部键名；直接替换实例属性，
    # 保持 nn.Module 参数、设备与保存加载行为不变（不重建算法对象）。
    if sample_for_assembly is not observation_sample:

        class _MappedActor:
            """任务键名 → 内部键名的推理包装；参数与状态来自内部actor。"""

            def __init__(self, inner, mapping):
                self._inner = inner
                self._mapping = mapping

            def __call__(self, obs, stochastic_output=False):
                renamed = type(obs)(
                    {ik: obs[tk] for ik, tk in self._mapping.items() if tk in obs},
                    batch_size=obs.batch_size,
                )
                return self._inner(renamed, stochastic_output=stochastic_output)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        mapping = {internal[c]: keys[c] for c in internal if c in keys}
        ppo.actor = _MappedActor(ppo.actor, mapping)
    return {
        "ppo": ppo,
        # 任务声明的键映射（公开接口），冻结内部键名不外泄到新任务
        "observation_keys": dict(keys),
        "action_valid_key": keys.get("action_valid"),
    }
