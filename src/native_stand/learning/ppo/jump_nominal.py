"""37D latent Normal，单次Tanh交给真实C；复用公共受控PPO和独立Adam。"""

import copy

import torch

from native_stand.tasks.jump.nominal import VERSION

from .controlled import KEY, ControlledPPO
from .networks import StudentNetwork
from .recovery_cubic import IndependentAdam


class NominalActor(StudentNetwork):
    def __init__(self, hidden_dims=(2048, 1024, 512), *, std=0.12):
        super().__init__(
            actor=True,
            input_dim=37,
            hidden_dims=hidden_dims,
            output_dim=6,
            init_std=torch.full((6,), std),
        )
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()
        self.log_std.requires_grad_(False)

    def latent_mean(self, x):
        return self.mlp(x)

    def distribution(self, x):
        return torch.distributions.Normal(self.latent_mean(x), self.log_std.exp())

    def forward(self, observation, stochastic_output=False):
        x = self._select(observation)
        if stochastic_output:
            self._current_distribution = self.distribution(x)
            return self._current_distribution.sample()
        return self.latent_mean(x)


class NominalPPO(ControlledPPO):
    def __init__(self, *args, minimum_unique_samples=128, minimum_minibatch_samples=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.minimum_unique_samples = minimum_unique_samples
        self.minimum_minibatch_samples = minimum_minibatch_samples
        self.minibatches = []

    def actor_batch_allowed(self, batch, valid, *, valid_count=None):
        count = int(valid.sum()) if valid_count is None else valid_count
        allowed = (
            self.unique_samples >= self.minimum_unique_samples
            and count >= self.minimum_minibatch_samples
        )
        self.minibatches.append(dict(valid=count, Actor_enabled=allowed))
        return allowed

    def update(self):
        valid = self.storage.observations[KEY].squeeze(-1)
        phases = self.storage.observations["student_actor"][..., 31:36]
        if bool(((phases.sum(-1) != 1) & valid).any()):
            raise ValueError("Jump控制样本必须包含真实C阶段one-hot")
        self.unique_samples = int(valid.sum())
        self.minibatches = []
        result = super().update()
        self.mask_summary.update(
            version=VERSION,
            unique_actor_samples=self.unique_samples,
            minibatches=self.minibatches,
            raw_action_semantics="pre_Tanh_latent_Normal_residual_around_same_C_nominal",
        )
        return result


def build_policy(
    job,
    observations,
    *,
    device="cpu",
    source=None,
    actor_class=NominalActor,
    ppo_class=NominalPPO,
    version=VERSION,
    action_semantics="pre_Tanh_latent_Normal_residual_around_same_C_nominal",
):
    from native_stand.tasks.jump.spec import build_task

    from .assembly import build_critic, build_storage

    task = build_task(job)
    actor = actor_class(
        job.model.actor_hidden_dims, std=job.diagnostics.get("latent_std", 0.12)
    ).to(device)
    critic = build_critic(job).to(device)
    storage = build_storage(observations, job, device=device)
    cfg = job.ppo
    ppo = ppo_class(
        actor,
        critic,
        storage,
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
        minimum_unique_samples=job.diagnostics.get("minimum_unique_actor_samples", 128),
        minimum_minibatch_samples=job.diagnostics.get("minimum_minibatch_samples", 8),
    )
    ppo.optimizer = IndependentAdam(actor, critic, cfg.learning_rate, cfg.critic_learning_rate)

    def capture(counters, curriculum=None, execution_state=None):
        from native_stand.students.resume import rng_state

        return copy.deepcopy(
            dict(
                version=version,
                contract=task.contract,
                actor=actor.state_dict(),
                critic=critic.state_dict(),
                optimizer=ppo.optimizer.state_dict(),
                optimizer_parameter_binding=ppo.optimizer.binding(),
                rng=torch.get_rng_state(),
                training_rng=rng_state(),
                std=actor.log_std.exp().detach(),
                counters=counters,
                curriculum=curriculum,
                execution_state=execution_state,
                actor_update_enabled=ppo.actor_update_enabled,
                gates=[ppo.minimum_unique_samples, ppo.minimum_minibatch_samples],
                raw_action_semantics=action_semantics,
            )
        )

    def restore(state):
        from native_stand.students.resume import restore_rng

        if state.get("version") != version or state.get("contract") != task.contract:
            raise ValueError("Jump拒绝旧绝对37D/72D/Motion完整恢复")
        if state["optimizer_parameter_binding"] != ppo.optimizer.binding() or state["gates"] != [
            ppo.minimum_unique_samples,
            ppo.minimum_minibatch_samples,
        ]:
            raise ValueError("Jump优化器或采样门槛身份不同")
        actor.load_state_dict(state["actor"], strict=True)
        critic.load_state_dict(state["critic"], strict=True)
        ppo.optimizer.load_state_dict(state["optimizer"])
        ppo.actor_update_enabled = state["actor_update_enabled"]
        restore_rng(state["training_rng"])

    return dict(
        ppo=ppo,
        capture=capture,
        restore=restore,
        contract=task.contract,
        observation_keys={
            "actor": "student_actor",
            "critic": "student_critic",
            "action_valid": KEY,
        },
        action_valid_key=KEY,
    )
