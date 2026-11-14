"""显式支撑中心PPO：因果控制mask、物理辅助损失和更新后KL约束。"""

from __future__ import annotations

import torch

from native_stand.tasks.motion.scale_corrections import boundary_loss, huber

from .guarded_step import bounded_actor_step

DQ = (10, 11, 13, 14)
LEGS = (0, 1, 3, 4)


def physical_gain(actor, observations, *, delta=0.05, kp=40.0, leg_scale=0.15):
    shift = observations.new_zeros(4, 31)
    for j, column in enumerate(DQ):
        shift[j, column] = delta * 0.05
    plus = actor.mean((observations[:, None] + shift).flatten(0, 1)).reshape(-1, 4, 6)
    minus = actor.mean((observations[:, None] - shift).flatten(0, 1)).reshape(-1, 4, 6)
    return ((plus[:, :, LEGS] - minus[:, :, LEGS]) * (kp * leg_scale / (2 * delta))).transpose(1, 2)


def auxiliary_loss(actor, observations, valid, cfg, *, leg_scale=0.15):
    if cfg.dq_gain_coefficient == 0 and cfg.boundary_coefficient == 0:
        return observations.new_zeros(()), {"states": 0, "gain_skipped": True}
    ids = valid.nonzero(as_tuple=True)[0]
    if not ids.numel():
        return None, {"states": 0}
    count = min(cfg.auxiliary_states, ids.numel())
    positions = torch.div(
        torch.arange(count, device=ids.device) * ids.numel(), count, rounding_mode="floor"
    )
    x = observations[ids[positions]].detach()
    mean = actor.mean(x)
    boundary, probability = boundary_loss(
        mean,
        actor.log_std.exp().expand_as(mean),
        allowed_probability=cfg.boundary_clip_probability,
        coefficient=cfg.boundary_coefficient,
    )
    sigma = None
    gain = mean.new_zeros(())
    if cfg.dq_gain_coefficient != 0:
        matrices = physical_gain(actor, x, leg_scale=leg_scale)
        sigma = torch.linalg.matrix_norm(matrices, ord=2)
        gain = cfg.dq_gain_coefficient * huber(torch.relu(sigma / cfg.dq_gain_threshold - 1)).mean()
    return boundary + gain, {
        "states": count,
        "gain_weighted_loss": float(gain.detach()),
        "boundary_weighted_loss": float(boundary.detach()),
        "gain_spectral_mean": float(sigma.detach().mean()) if sigma is not None else None,
        "gain_spectral_max": float(sigma.detach().max()) if sigma is not None else None,
        "expected_leg_clip_probability": probability.detach().mean(0).cpu().tolist(),
    }


def ppo_class():
    # 已安装的冻结数值模块在引擎适配后导入；本模块不选择环境或读配置文件。
    from native_stand.learning.ppo.controlled import KEY, ControlledPPO

    class SupportedPPO(ControlledPPO):
        def __init__(self, *args, scale_config, **kwargs):
            super().__init__(*args, **kwargs)
            self.scale_config = scale_config
            self.physical_leg_scale = 0.15
            actor, critic = list(self.actor.parameters()), list(self.critic.parameters())
            self.optimizer = torch.optim.Adam(
                [
                    dict(params=actor, lr=scale_config.learning_rate, owner="Actor"),
                    dict(params=critic, lr=scale_config.critic_learning_rate, owner="Critic"),
                ]
            )
            self.critic_control_only = True
            self.record_gradient_norms = True

        def update(self):
            storage = self.storage
            valid = storage.observations[KEY]
            if valid.dtype != torch.bool or valid.shape != storage.advantages.shape:
                raise ValueError("PPO控制mask缺失或不同步")
            totals = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0}
            actor_steps = critic_steps = skipped = visits = 0
            kl_values, clips, actor_norms, critic_norms, guards, auxiliary, steps = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            retention = getattr(self, "source_anchor", None)
            if retention is not None:
                retention.reset()
            retention_gradient = None
            blocked = False
            batches = 0
            for batch in storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            ):
                mask = batch.observations[KEY].squeeze(-1)
                if not bool(mask.any()):
                    skipped += 1
                    continue
                self.actor(batch.observations, stochastic_output=True)
                logp = self.actor.get_output_log_prob(batch.actions)
                values = self.critic(batch.observations)
                value_loss = (batch.returns[mask] - values[mask]).square().mean()
                with torch.no_grad():
                    old = tuple(p[mask] for p in batch.old_distribution_params)
                    current = tuple(p[mask] for p in self.actor.output_distribution_params)
                    before = float(self.actor.get_kl_divergence(old, current).mean())
                if not torch.isfinite(values).all() or not torch.isfinite(torch.tensor(before)):
                    raise RuntimeError("非有限Critic或Actor KL")
                guards.append(before)
                blocked = blocked or before > self.scale_config.kl_limit
                update_actor = self.actor_update_enabled and not blocked
                surrogate = values.new_zeros(())
                entropy = values.new_zeros(())
                loss = self.value_loss_coef * value_loss
                if update_actor:
                    ratio = (logp[mask] - batch.old_actions_log_prob.squeeze(-1)[mask]).exp()
                    advantage = batch.advantages.squeeze(-1)[mask]
                    surrogate = torch.maximum(
                        -advantage * ratio,
                        -advantage * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
                    ).mean()
                    entropy = self.actor.output_entropy[mask].mean()
                    extra, detail = auxiliary_loss(
                        self.actor,
                        batch.observations["student_actor"],
                        mask,
                        self.scale_config,
                        leg_scale=self.physical_leg_scale,
                    )
                    auxiliary.append(detail)
                    policy_loss = surrogate - self.entropy_coef * entropy + extra
                    retained = values.new_zeros(())
                    if retention is not None:
                        retained = retention.loss(self.actor, batch.observations, mask)
                        if (
                            retention.config.get("record_first_gradient")
                            and retention_gradient is None
                        ):
                            from .source_anchor import gradient_comparison

                            retention_gradient = gradient_comparison(
                                self.actor, policy_loss, retained
                            )
                    loss = loss + policy_loss + retained
                    clips.append(float(((ratio - 1).abs() > self.clip_param).float().mean()))
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not update_actor:
                    for parameter in self.actor.parameters():
                        parameter.grad = None
                a_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                c_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                if not torch.isfinite(a_norm) or not torch.isfinite(c_norm):
                    raise RuntimeError("非有限PPO梯度")
                actor_norms.append(float(a_norm))
                critic_norms.append(float(c_norm))
                if update_actor:
                    inputs = batch.observations["student_actor"][mask]

                    def measure(inputs=inputs, old=old):
                        # 显式绑定当前迭代值，避免闭包读到后续迭代的变量。
                        with torch.no_grad():
                            distribution = self.actor.distribution(inputs)
                            return self.actor.get_kl_divergence(
                                old, (distribution.mean, distribution.stddev)
                            ).mean()

                    detail = bounded_actor_step(
                        self.optimizer,
                        self.actor,
                        measure,
                        limit=self.scale_config.kl_limit,
                        max_backtracks=self.scale_config.max_actor_backtracks,
                    )
                    steps.append(detail)
                    if detail["accepted"]:
                        actor_steps += 1
                        visits += int(mask.sum())
                        kl_values.append(detail["post_kl"])
                    else:
                        blocked = True
                        skipped += 1
                else:
                    self.optimizer.step()
                    skipped += 1
                critic_steps += 1
                batches += 1
                totals["value"] += float(value_loss.detach())
                totals["surrogate"] += float(surrogate.detach())
                totals["entropy"] += float(entropy.detach())

            def average(rows):
                return sum(rows) / len(rows) if rows else 0.0

            self.mask_summary = {
                "version": "supported015_poststep_KL_physical_aux_v1",
                "total_transitions": valid.numel(),
                "valid_actor_samples": int(valid.sum()),
                "effective_actor_update_samples": visits,
                "actor_minibatch_sample_visits": visits,
                "actor_optimizer_steps": actor_steps,
                "critic_optimizer_steps": critic_steps,
                "actor_skipped_minibatches": skipped,
                "kl_valid_mean": average(kl_values),
                "Actor_post_Adam_accepted_KL_peak": max(kl_values, default=0.0),
                "Actor_guard_KL_pre_minibatch_peak": max(guards, default=0.0),
                "Actor_backtracks": sum(r["backtracks"] for r in steps),
                "Actor_proposed_optimizer_steps": sum(r["attempts"] for r in steps),
                "Actor_rejected_minibatches": sum(not r["accepted"] for r in steps),
                "clip_fraction_valid_mean": average(clips),
                "actor_gradient_norm_mean": average(actor_norms),
                "critic_gradient_norm_mean": average(critic_norms),
                "critic_warmup": not self.actor_update_enabled,
                "BC_loss": 0.0,
                "BC_updates": 0,
                "actor_lr": self.optimizer.param_groups[0]["lr"],
                "critic_lr": self.optimizer.param_groups[1]["lr"],
                "auxiliary": auxiliary,
            }
            if retention is not None:
                self.mask_summary["source_policy_retention"] = retention.summary()
                self.mask_summary["source_policy_retention"]["gradient_comparison"] = (
                    retention_gradient
                )
            storage.clear()
            return {key: value / max(1, batches) for key, value in totals.items()}

    return SupportedPPO


def assemble_from_full(source, observation, job, *, fresh=False):
    from rsl_rl.storage import RolloutStorage

    from .assembly import build_actor, build_critic

    actor = build_actor(job)
    critic = build_critic(job)
    actor.load_state_dict(source["actor"], strict=True)
    critic.load_state_dict(source["critic"], strict=True)
    stored_std = source.get("std")
    if stored_std is not None:
        # 历史Motion同时存在log_std与sigma字段；不让通用恢复误把sigma写入log_std。
        kind = source.get("std_parameterization")
        log_std = actor.log_std.detach()
        expected = {"log_std": log_std, "sigma": log_std.exp()}
        if kind not in (None, *expected):
            raise ValueError("未知std参数化")
        matched = [
            name
            for name, value in expected.items()
            if torch.allclose(torch.as_tensor(stored_std).to(value), value, atol=1e-7, rtol=1e-6)
        ]
        if (kind is not None and kind not in matched) or (kind is None and len(matched) != 1):
            raise ValueError("完整来源std与Actor实际采样分布不一致")
    actor.log_std.requires_grad_(False)
    device = str(observation["student_actor"].device)
    actor.to(device)
    critic.to(device)
    storage = RolloutStorage(
        "rl",
        len(observation["student_actor"]),
        job.runtime.steps_per_env,
        observation,
        [6],
        device=device,
    )
    if job.assets.get("source_policy_retention") is not None:
        from .source_anchor import KEY as RETENTION_KEY

        storage.observations[RETENTION_KEY] = torch.full_like(
            storage.observations[RETENTION_KEY], -1, dtype=torch.long
        )
    cfg = job.ppo
    algorithm = ppo_class()(
        actor,
        critic,
        storage,
        scale_config=cfg,
        learning_rate=cfg.learning_rate,
        schedule="fixed",
        num_learning_epochs=cfg.epochs,
        num_mini_batches=cfg.minibatches,
        entropy_coef=cfg.entropy_coef,
        clip_param=cfg.clip,
        gamma=cfg.gamma,
        lam=cfg.lam,
        value_loss_coef=cfg.value_coef,
        max_grad_norm=cfg.max_grad_norm,
        use_clipped_value_loss=False,
        device=device,
    )
    algorithm.physical_leg_scale = job.action.leg_scale_rad
    from .dual_adam import DualAdam

    if fresh:
        algorithm.optimizer = DualAdam(
            actor.parameters(),
            critic.parameters(),
            actor_lr=cfg.learning_rate,
            critic_lr=cfg.critic_learning_rate,
        )
    else:
        schema = source["optimizer"].get("version")
        if schema is not None:
            if schema != DualAdam.VERSION:
                raise ValueError("未知optimizer状态版本")
            algorithm.optimizer = DualAdam(
                actor.parameters(),
                critic.parameters(),
                actor_lr=cfg.learning_rate,
                critic_lr=cfg.critic_learning_rate,
            )
        elif "param_groups" not in source["optimizer"]:
            raise ValueError("缺少明确optimizer schema")
        algorithm.optimizer.load_state_dict(source["optimizer"])
    if (
        algorithm.optimizer.param_groups[0]["lr"] != cfg.learning_rate
        or algorithm.optimizer.param_groups[1]["lr"] != cfg.critic_learning_rate
    ):
        raise ValueError("完整恢复的Actor/Critic学习率与新声明不同")
    algorithm.actor_update_enabled = (
        int(source.get("warmup_updates_done", 0)) >= cfg.critic_warmup_updates
    )
    retention = job.assets.get("source_policy_retention")
    if retention is not None:
        from native_stand.artifacts.retention_identity import (
            require_same_retention,
            resolve_retention,
        )

        retention = resolve_retention(retention, assets=job.assets)
        if source.get("source_anchor_actor") is not None:
            require_same_retention(source.get("source_policy_retention"), retention)
        from .command_bank import VERSION as COMMAND_BANK
        from .command_bank import CommandBankAnchor
        from .source_anchor import SourceAnchor

        if retention["version"] == COMMAND_BANK:
            import copy

            import numpy as np

            from native_stand.artifacts.io import file_sha256

            for key in ("bank", "bank_metadata"):
                ref = retention[key]
                if file_sha256(ref["path"]) != ref["sha256"]:
                    raise ValueError("固定来源bank身份不同: " + key)
            teacher = copy.deepcopy(actor)
            teacher.load_state_dict(source["source_anchor_actor"], strict=True)
            from native_stand.artifacts.io import read_json

            meta = read_json(retention["bank_metadata"]["path"])
            if meta["data_role"] != "train_only" or meta["commands"] != retention["commands"]:
                raise ValueError("bank资格/命令声明不一致")
            with np.load(retention["bank"]["path"], allow_pickle=False) as data:
                algorithm.source_anchor = CommandBankAnchor(teacher, retention, data)
            runtime = source.get("source_anchor_runtime_state")
            if runtime is not None:
                algorithm.source_anchor.load_state_dict(runtime)
        else:
            algorithm.source_anchor = SourceAnchor(actor, retention)
        anchor = source.get("source_anchor_actor")
        if anchor is not None:
            algorithm.source_anchor.teacher.load_state_dict(anchor, strict=True)
        elif retention["source_sha256"] != job.initialization.sha256:
            raise ValueError("非初始化来源Teacher缺少冻结状态")
    return algorithm
