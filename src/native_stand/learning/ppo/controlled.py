"""公共受控PPO；保留Critic/GAE时间线，任务通过显式Actor批次判定扩展。"""

from contextlib import nullcontext

import torch
from rsl_rl.algorithms import PPO

from .performance import Measurement

KEY = "recovery_actor_valid"


def declaration():
    return {
        "version": "recovery_controlled_actor_v1",
        "scope": "pre_action_policy_control",
        "advantage_normalization": "valid_global_rollout",
        "critic": "all_transitions",
        "invalid_actor_grad": "None",
    }


def pre_action_valid(initializing, initialization_checked):
    if initializing.shape != initialization_checked.shape:
        raise ValueError("Actor control mask shape mismatch")
    return (~initializing.bool()) & initialization_checked.bool()


def normalize_valid(raw, valid):
    if raw.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("Actor mask missing/misaligned")
    out = torch.zeros_like(raw)
    if valid.any():
        v = raw[valid]
        out[valid] = (v - v.mean()) / (v.std() + 1e-8) if v.numel() > 1 else 0.0
    return out


class ControlledPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self.actor.is_recurrent
            or self.critic.is_recurrent
            or self.rnd
            or self.symmetry
            or self.is_multi_gpu
            or self.schedule != "fixed"
        ):
            raise ValueError("Controlled Recovery PPO supports fixed feedforward only")
        if set(map(id, self.actor.parameters())) & set(map(id, self.critic.parameters())):
            raise ValueError("Actor/critic shared parameters unsupported")
        self.storage.observations[KEY] = torch.zeros_like(
            self.storage.observations[KEY], dtype=torch.bool
        )
        self.mask_summary = {}
        self.actor_update_enabled = True
        self.record_gradient_norms = False
        self.critic_control_only = False
        self.temporal_regularizer = None

    def act(self, obs):
        if (
            KEY not in obs
            or obs[KEY].dtype != torch.bool
            or obs[KEY].shape != (obs.batch_size[0], 1)
        ):
            raise ValueError("Pre-action valid metadata required")
        return super().act(obs)

    @torch.no_grad()
    def compute_returns(self, obs):
        # Same RSL-RL GAE recursion. Its unmasked normalization was immediately
        # overwritten; normalize only the globally valid Actor samples once.
        st = self.storage
        last_values = self.critic(obs).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = (
                last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            )
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = (
                st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            )
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        if KEY not in st.observations:
            raise ValueError("Storage actor mask missing")
        st.advantages = normalize_valid(st.returns - st.values, st.observations[KEY])

    def actor_auxiliary_loss(self, batch, valid):
        """Dormant extension point; default Recovery/PPO objective unchanged."""
        return None

    def actor_batch_allowed(self, batch, valid, *, valid_count=None):
        return self.actor_update_enabled

    def valid_actor_sample_count(self, valid):
        return int(valid.sum())

    def update(self):
        st = self.storage
        valid = st.observations.get(KEY)
        if valid is None or valid.shape != st.advantages.shape:
            raise ValueError("Storage mask missing/misaligned")
        telemetry = torch.zeros(6, dtype=torch.float64, device=st.values.device)
        timer = Measurement(getattr(self, "performance_measurement", False), st.values.device)
        count = 0
        actor_steps = 0
        critic_steps = 0
        empty = 0
        actor_grad = []
        critic_grad = []
        temporal_pairs = 0
        temporal_grads = []
        for b in timer.batches(
            st.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        ):
            m = b.observations[KEY].squeeze(-1)
            valid_count = int(m.sum())
            has_valid = valid_count > 0
            update_actor = self.actor_update_enabled and self.actor_batch_allowed(
                b, m, valid_count=valid_count
            )
            # Preserve the random draw and final distribution cache even when
            # no Actor update is allowed, but do not construct its unused graph.
            with (
                timer.section("actor_forward"),
                nullcontext() if has_valid and update_actor else torch.no_grad(),
            ):
                self.actor(b.observations, stochastic_output=True)
                logp = self.actor.get_output_log_prob(b.actions)
                entropy = self.actor.output_entropy
            with timer.section("critic_forward"):
                values = self.critic(b.observations)
            if self.critic_control_only and not has_valid:
                self.optimizer.zero_grad(set_to_none=True)
                empty += 1
                continue  # Neither Adam parameter nor momentum state advances.
            vloss = (
                (b.returns - values).square().mean()
                if not self.use_clipped_value_loss
                else torch.maximum(
                    (b.returns - values).square(),
                    (
                        b.returns
                        - (b.values + (values - b.values).clamp(-self.clip_param, self.clip_param))
                    ).square(),
                ).mean()
            )
            if self.critic_control_only:
                if self.use_clipped_value_loss:
                    raise ValueError("controlled critic requires declared unclipped value loss")
                vloss = (b.returns[m] - values[m]).square().mean()
            loss = self.value_loss_coef * vloss
            sur = values.new_zeros(())
            ent = values.new_zeros(())
            if has_valid and update_actor:
                ratio = torch.exp(logp[m] - b.old_actions_log_prob.squeeze(-1)[m])
                adv = b.advantages.squeeze(-1)[m]
                sur = torch.maximum(
                    -adv * ratio, -adv * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
                ).mean()
                ent = entropy[m].mean()
                loss = loss + sur - self.entropy_coef * ent
                if self.temporal_regularizer is not None:
                    term, npairs = self.temporal_regularizer.loss(self.actor)
                    if self.record_gradient_norms and npairs:
                        pars = [p for p in self.actor.parameters() if p.requires_grad]
                        gs = torch.autograd.grad(
                            self.temporal_regularizer.cfg["coefficient"] * term,
                            pars,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        temporal_grads.append(
                            float(
                                torch.sqrt(
                                    sum(
                                        (g.square().sum() for g in gs if g is not None),
                                        values.new_zeros(()),
                                    )
                                )
                            )
                        )
                    loss = loss + self.temporal_regularizer.cfg["coefficient"] * term
                    telemetry[5] += term.detach().double()
                    temporal_pairs += npairs
                with timer.section("diagnostics"), torch.no_grad():
                    kl = self.actor.get_kl_divergence(
                        tuple(p[m] for p in b.old_distribution_params),
                        tuple(p[m] for p in self.actor.output_distribution_params),
                    ).mean()
                    telemetry[3] += kl.detach().double()
                    telemetry[4] += (abs(ratio - 1) > self.clip_param).float().mean().double()
                with timer.section("auxiliary"):
                    auxiliary = self.actor_auxiliary_loss(b, m)
                if auxiliary is not None:
                    loss = loss + auxiliary
                actor_steps += 1
            else:
                empty += 1
            self.optimizer.zero_grad(set_to_none=True)
            with timer.section("backward"):
                loss.backward()
            if not has_valid or not update_actor:
                for p in self.actor.parameters():
                    p.grad = None
            if self.record_gradient_norms:
                actor_grad.append(
                    float(
                        torch.sqrt(
                            sum(
                                (
                                    p.grad.square().sum()
                                    for p in self.actor.parameters()
                                    if p.grad is not None
                                ),
                                values.new_zeros(()),
                            )
                        )
                    )
                )
                critic_grad.append(
                    float(
                        torch.sqrt(
                            sum(
                                (
                                    p.grad.square().sum()
                                    for p in self.critic.parameters()
                                    if p.grad is not None
                                ),
                                values.new_zeros(()),
                            )
                        )
                    )
                )
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            with timer.section("adam"):
                self.optimizer.step()
            critic_steps += 1
            telemetry[:3] += torch.stack((vloss.detach(), sur.detach(), ent.detach())).double()
            count += 1
        value_sum, surrogate_sum, entropy_sum, kl_sum, clip_sum, temporal_sum = (
            telemetry.cpu().tolist()
        )
        total_valid = self.valid_actor_sample_count(valid)
        self.performance_timings = timer.seconds
        self.mask_summary = {
            "total_transitions": valid.numel(),
            "valid_actor_samples": total_valid,
            "invalid_actor_samples": valid.numel() - total_valid,
            "actor_optimizer_steps": actor_steps,
            "actor_skipped_minibatches": empty,
            "kl_valid_mean": kl_sum / max(actor_steps, 1),
            "clip_fraction_valid_mean": clip_sum / max(actor_steps, 1),
            "version": declaration()["version"],
        }
        self.mask_summary.update(
            effective_actor_update_samples=total_valid if self.actor_update_enabled else 0,
            actor_gradient_norm_mean=sum(actor_grad) / len(actor_grad) if actor_grad else None,
            critic_gradient_norm_mean=sum(critic_grad) / len(critic_grad) if critic_grad else None,
            BC_loss=0.0,
            BC_gradient_norm=None,
            BC_updates=0,
            critic_warmup=not self.actor_update_enabled,
        )
        self.mask_summary.update(
            critic_optimizer_steps=critic_steps, critic_control_only=self.critic_control_only
        )
        if self.temporal_regularizer is not None:
            self.mask_summary.update(
                temporal_loss_mean=temporal_sum / max(actor_steps, 1),
                temporal_pairs_sampled=temporal_pairs,
                temporal_coefficient=self.temporal_regularizer.cfg["coefficient"],
                temporal_weighted_gradient_norm_mean=sum(temporal_grads) / len(temporal_grads)
                if temporal_grads
                else None,
            )
        st.clear()
        return {
            k: v / max(count, 1)
            for k, v in zip(
                ("value", "surrogate", "entropy"),
                (value_sum, surrogate_sum, entropy_sum),
                strict=True,
            )
        }
