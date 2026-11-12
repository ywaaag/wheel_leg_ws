"""冻结同契约来源，只在已知名义命令上做on-policy策略保留。"""

from __future__ import annotations

import copy

import torch

KEY = "supported_source_retention_cohort"
VERSION = "supported_nominal_source_retention_v1"
CAUSAL_VERSION = "supported_C0_qualified_nominal_source_KL_v2"


def metadata(observations, targets, profiles, valid, commands):
    ids = torch.full_like(valid, -1, dtype=torch.long)
    for i, command in enumerate(commands):
        ids = torch.where(
            valid & (profiles < 0) & (targets == targets.new_tensor(command)).all(-1), i, ids
        )
    observations[KEY] = ids[:, None]
    return observations


class SourceAnchor:
    def __init__(self, actor, config):
        if (
            config["version"] not in (VERSION, CAUSAL_VERSION)
            or not 0 < config["coefficient"] <= 0.1
        ):
            raise ValueError("来源保留配置无效")
        self.config = copy.deepcopy(config)
        self.teacher = copy.deepcopy(actor).eval().requires_grad_(False)
        self.std = actor.log_std.detach().exp().clone()
        self.reset()

    def reset(self):
        self.calls = 0
        self.total = 0.0
        self.uses = [0] * len(self.config["commands"])

    def loss(self, actor, observations, valid):
        ids = observations[KEY].squeeze(-1)
        if ids.dtype != torch.long:
            raise ValueError("保留cohort索引必须为整数")
        x = observations["student_actor"]
        losses = []
        for i in range(len(self.uses)):
            selected = (valid & (ids == i)).nonzero(as_tuple=True)[0]
            if not selected.numel():
                continue
            n = min(self.config["maximum_states_per_cohort"], selected.numel())
            positions = torch.div(
                torch.arange(n, device=x.device) * selected.numel(), n, rounding_mode="floor"
            )
            sample = x[selected[positions]].detach()
            with torch.no_grad():
                reference = self.teacher.mean(sample)
            mean = actor.mean(sample)  # 不更新采样分布缓存，不消耗PPO RNG。
            losses.append(0.5 * ((mean - reference) / self.std).square().mean())
            self.uses[i] += n
        result = torch.stack(losses).mean() if losses else x.sum() * 0.0
        self.total += float(result.detach())
        self.calls += 1
        return self.config["coefficient"] * result

    def summary(self):
        return dict(
            version=self.config["version"],
            coefficient=self.config["coefficient"],
            source_sha256=self.config["source_sha256"],
            commands=self.config["commands"],
            weighted_loss_mean=self.config["coefficient"] * self.total / max(1, self.calls),
            actual_selected_visits_by_cohort=self.uses,
            actor_minibatches=self.calls,
            inputs="current on-policy nominal old-command states; no replay or teacher control",
            independent_certificate=False,
        )


def qualified_valid(valid, *, event_enabled, age_s, pitch, roll, wheel_support):
    """仅因果、名义、无事件的学生样本；profiles/commands由metadata另筛。"""
    return (
        valid
        & ~event_enabled
        & (age_s >= 2.0)
        & (pitch.abs() <= 0.1)
        & (roll.abs() <= 0.05)
        & (wheel_support > 2.0).all(-1)
    )


def gradient_comparison(actor, policy_loss, retention_loss):
    """每次update首个Actor minibatch的真实梯度；不写.grad/采样缓存/RNG。"""
    params = [p for p in actor.parameters() if p.requires_grad]

    def gradients(loss):
        return (
            torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            if loss.requires_grad
            else [None] * len(params)
        )

    gp, gr = gradients(policy_loss), gradients(retention_loss)
    zero = policy_loss.new_zeros(())
    p2 = sum((g.square().sum() for g in gp if g is not None), zero)
    r2 = sum((g.square().sum() for g in gr if g is not None), zero)
    dot = sum(
        ((p * r).sum() for p, r in zip(gp, gr, strict=False) if p is not None and r is not None),
        zero,
    )
    p, r = float(p2.sqrt()), float(r2.sqrt())
    return dict(
        PPO_gradient_norm=p,
        weighted_source_KL_gradient_norm=r,
        source_to_PPO_gradient_norm_ratio=r / p if p else None,
        cosine=float(dot / (p2 * r2).sqrt()) if p and r else None,
        scope="first valid Actor minibatch, before clipping; no optimizer or RNG mutation",
    )
