"""PPO装配：显式网络、显式观测列、显式学习率与完整恢复状态。

算法数学沿用受控PPO（`native_stand.recovery.controlled_ppo`），本模块只负责：
- 按配置构造Actor（部署列）与Critic（声明特权列）；
- 建立rollout存储，要求 `recovery_actor_valid` 控制mask；
- 给Actor与Critic分别设置学习率；
- 保存/恢复Actor、Critic、Adam、std、RNG与课程状态。

torch与rsl_rl在函数内延迟导入，导入本模块本身不需要Isaac或GPU。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from native_stand.experiments.config import JobConfig

from .networks import StudentNetwork, initial_std

ACTOR_KEY = "student_actor"
CRITIC_KEY = "student_critic"
CONTROL_KEY = "recovery_actor_valid"


def observation_columns(job: JobConfig) -> dict[str, int]:
    """本次运行要求的观测列与宽度；缺失即拒绝，不用零填充凑形状。"""
    return {
        ACTOR_KEY: job.model.actor_input,
        CRITIC_KEY: job.model.critic_input,
        CONTROL_KEY: 1,
    }


def build_actor(job: JobConfig):
    """Actor只吃部署可得的列。"""
    return StudentNetwork(
        actor=True,
        input_dim=job.model.actor_input,
        hidden_dims=job.model.actor_hidden_dims,
        output_dim=job.action.dim,
        init_std=initial_std(job),
    )


def build_critic(job: JobConfig):
    """Critic只吃声明的特权列。"""
    if job.model.critic_privileged_version is None:
        raise ValueError("Critic需要明确的特权信息版本")
    return StudentNetwork(
        actor=False,
        input_dim=job.model.critic_input,
        hidden_dims=job.model.critic_hidden_dims,
        output_dim=1,
    )


def build_storage(observations, job: JobConfig, *, device: str):
    """按配置建立rollout存储；步数来自runtime，不硬编码。"""
    from rsl_rl.storage import RolloutStorage

    for column, width in observation_columns(job).items():
        if column not in observations:
            raise ValueError(f"观测缺少{column}列")
        actual = observations[column].shape[-1]
        if actual != width:
            raise ValueError(f"观测列{column}宽度{actual}与声明{width}不一致")
    return RolloutStorage(
        "rl",
        int(observations.batch_size[0]),
        job.runtime.steps_per_env,
        observations,
        [job.action.dim],
        device=device,
    )


def split_learning_rates(ppo, job: JobConfig) -> None:
    """用两个参数组替换默认单组优化器：Actor与Critic各用自己的学习率。"""
    import torch

    from native_stand.learning.ppo.controlled import ControlledPPO  # noqa: F401  (类型提示用)

    defaults = dict(ppo.optimizer.defaults)
    defaults.pop("lr", None)
    defaults.pop("differentiable", None)
    groups = [
        {
            "params": [p for p in ppo.actor.parameters() if p.requires_grad],
            "lr": job.ppo.learning_rate,
        },
        {
            "params": [p for p in ppo.critic.parameters() if p.requires_grad],
            "lr": job.ppo.critic_learning_rate,
        },
    ]
    ppo.optimizer = torch.optim.Adam(groups, **defaults)
    ppo.learning_rate = job.ppo.learning_rate


def assemble_ppo(actor, critic, observations, job: JobConfig, *, device: str | None = None):
    """装配受控PPO：固定std、两路学习率、warmup初态。"""
    from native_stand.learning.ppo.controlled import ControlledPPO

    target = device or job.runtime.device
    storage = build_storage(observations, job, device=target)
    ppo_config = job.ppo
    ppo = ControlledPPO(
        actor,
        critic,
        storage,
        learning_rate=ppo_config.learning_rate,
        schedule="fixed",
        num_learning_epochs=ppo_config.epochs,
        num_mini_batches=ppo_config.minibatches,
        entropy_coef=ppo_config.entropy_coef,
        clip_param=ppo_config.clip,
        gamma=ppo_config.gamma,
        lam=ppo_config.lam,
        value_loss_coef=ppo_config.value_coef,
        max_grad_norm=ppo_config.max_grad_norm,
        use_clipped_value_loss=False,
        device=target,
    )
    split_learning_rates(ppo, job)
    # 训练固定std：不把探索噪声当可学习参数更新。
    ppo.actor.log_std.requires_grad_(False)
    ppo.actor_update_enabled = ppo_config.critic_warmup_updates == 0
    return ppo


# ---- 完整恢复状态 ---------------------------------------------------
def _std_tensor(ppo) -> Any:
    import torch

    return (
        ppo.actor.log_std.detach().clone()
        if isinstance(getattr(ppo.actor, "log_std", None), torch.nn.Parameter)
        else torch.zeros(1)
    )


def capture_full(
    ppo,
    *,
    counters: Mapping[str, Any],
    curriculum: Mapping[str, Any] | None = None,
    mixing: Mapping[str, Any] | None = None,
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """导出完整恢复状态：两个参数集合、Adam、std、RNG、计数与课程。"""
    import torch

    return {
        "actor": {key: value.detach().clone() for key, value in ppo.actor.state_dict().items()},
        "critic": {key: value.detach().clone() for key, value in ppo.critic.state_dict().items()},
        "optimizer": ppo.optimizer.state_dict(),
        "std": _std_tensor(ppo),
        "rng": torch.get_rng_state(),
        "counters": dict(counters),
        "curriculum": dict(curriculum or {}),
        "mixing": dict(mixing or {}),
        "source_identity": dict(source_identity or {}),
        "actor_update_enabled": bool(getattr(ppo, "actor_update_enabled", True)),
    }


def restore_full(ppo, payload: Mapping[str, Any]) -> dict[str, Any]:
    """加载完整恢复状态；形状或键不符时直接失败，不静默跳过。"""
    import torch

    ppo.actor.load_state_dict(payload["actor"], strict=True)
    ppo.critic.load_state_dict(payload["critic"], strict=True)
    ppo.optimizer.load_state_dict(payload["optimizer"])
    std = payload.get("std")
    if std is not None and hasattr(ppo.actor, "log_std"):
        if std.shape != ppo.actor.log_std.shape:
            raise ValueError(
                f"std形状{tuple(std.shape)}与网络{tuple(ppo.actor.log_std.shape)}不一致"
            )
        with torch.no_grad():
            ppo.actor.log_std.copy_(std)
    if payload.get("rng") is not None:
        torch.set_rng_state(payload["rng"])
    ppo.actor_update_enabled = bool(payload.get("actor_update_enabled", True))
    return {
        "counters": dict(payload.get("counters") or {}),
        "curriculum": dict(payload.get("curriculum") or {}),
        "mixing": dict(payload.get("mixing") or {}),
    }


# ---- warmup语义 -----------------------------------------------------
def warmup_updates(job: JobConfig, stage: str, *, completed_updates: int) -> int:
    """Critic-only warmup次数由配置决定。

    smoke阶段从头开始按配置执行；formal接续已完成状态时为0，不重复warmup。
    """
    configured = job.ppo.critic_warmup_updates
    if stage == "formal" and completed_updates > 0:
        return 0
    return configured


def actor_update_allowed(
    job: JobConfig, stage: str, update: int, *, offset: int = 0, completed_updates: int = 0
) -> bool:
    """第update次（含offset）是否允许Actor更新。"""
    warmup = warmup_updates(job, stage, completed_updates=completed_updates)
    return offset + update > warmup


def loss_coefficients(job: JobConfig) -> dict[str, float]:
    """暴露给指标记录的系数，避免多处重复声明。"""
    return {
        "entropy_coef": job.ppo.entropy_coef,
        "value_loss_coef": job.ppo.value_coef,
        "clip_param": job.ppo.clip,
        "gamma": job.ppo.gamma,
        "lam": job.ppo.lam,
        "max_grad_norm": job.ppo.max_grad_norm,
        "actor_lr": job.ppo.learning_rate,
        "critic_lr": job.ppo.critic_learning_rate,
    }
