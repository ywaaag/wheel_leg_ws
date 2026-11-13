"""Recovery physical-target Gaussian; deterministic Tanh/cubic export unchanged.

No normalized log_std parameter and no second Gaussian. ControlledPPO math is
reused with two independent Adam instances through a minimal optimizer facade.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from native_stand.learning.ppo.networks import StudentNetwork

VERSION = "recovery_cubic_physical_Gaussian_v2"
LEGS = (0, 1, 3, 4)


class PhysicalActor(StudentNetwork):
    def __init__(self, contract, hidden_dims=(2048, 1024, 512)):
        super().__init__(
            actor=True, input_dim=31, hidden_dims=hidden_dims, output_dim=6, init_std=torch.ones(6)
        )
        del self.log_std
        self.contract = copy.deepcopy(contract)
        self.register_buffer("physical_std", torch.tensor([0.02, 0.02, 2.0, 0.02, 0.02, 2.0]))
        self.register_buffer("center", torch.tensor(contract["center_rad"]))
        self.register_buffer("lower", torch.tensor(contract["leg_lower_rad"]))
        self.register_buffer("upper", torch.tensor(contract["leg_upper_rad"]))

    def physical_mean(self, x):
        return self.decode_mean(self.mean(x))

    def decode_mean(self, a):
        v = a[..., LEGS]
        reach = torch.where(v >= 0, self.upper - self.center, self.center - self.lower)
        q = self.center + 0.15 * v + (reach - 0.15) * v.pow(3)
        out = a * 100.0
        out = out.clone()
        out[..., list(LEGS)] = q
        if not bool(torch.isfinite(out).all()):
            raise ValueError("Nonfinite physical mean")
        return out

    def distribution(self, x):
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Nonfinite Actor input")
        return torch.distributions.Normal(self.physical_mean(x), self.physical_std)

    def forward(self, observation, stochastic_output=False):
        x = self._select(observation)
        if stochastic_output:
            self._current_distribution = self.distribution(x)
            return self._current_distribution.sample()  # raw physical sample
        return self.physical_mean(x)

    def protect(self, raw):
        if raw.shape[-1] != 6 or not bool(torch.isfinite(raw).all()):
            raise ValueError("Finite raw physical sample required")
        lo = raw.new_tensor([-1.5, -1.5, -100, -1.5, -1.5, -100])
        hi = -lo
        return torch.maximum(lo, torch.minimum(hi, raw))

    def get_output_log_prob(self, raw):
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("Nonfinite raw physical sample")
        return super().get_output_log_prob(raw)


def initialize_from_C0(actor, source):
    state = {k: v.clone() for k, v in source.items() if k.startswith("mlp.")}
    state["mlp.0.bias"] += state["mlp.0.weight"][:, 25] - state["mlp.0.weight"][:, 26]
    actor.mlp.load_state_dict({k.removeprefix("mlp."): v for k, v in state.items()}, strict=True)


class IndependentAdam:
    def __init__(self, actor, critic, actor_lr=1e-5, critic_lr=9e-5):
        self.models = {"actor": actor, "critic": critic}
        self.actor = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.param_groups = self.actor.param_groups + self.critic.param_groups

    def binding(self):
        result = {}
        all_ids = []
        for owner, model in self.models.items():
            named = {id(p): (name, list(p.shape)) for name, p in model.named_parameters()}
            parameters = [p for g in getattr(self, owner).param_groups for p in g["params"]]
            ids = [id(p) for p in parameters]
            all_ids += ids
            if len(set(ids)) != len(ids) or set(ids) != set(named):
                raise ValueError("Optimizer owner parameter set differs")
            result[owner] = [
                {"name": named[id(p)][0], "shape": named[id(p)][1]} for p in parameters
            ]
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("Multiple optimizer owners")
        return dict(
            version="named_optimizer_order_v1",
            owners=result,
            hyperparameters={
                name: [
                    {k: g[k] for k in ["lr", "betas", "eps", "weight_decay"]}
                    for g in getattr(self, name).param_groups
                ]
                for name in self.models
            },
        )

    def zero_grad(self, **kw):
        self.actor.zero_grad(**kw)
        self.critic.zero_grad(**kw)

    def step(self):
        self.actor.step()
        callback = getattr(self, "audit_after_actor_step", None)
        if callback is not None:
            self.audit_after_actor_step = None
            callback()
        self.critic.step()

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict())

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])


def assemble(
    actor,
    critic,
    storage,
    *,
    device="cpu",
    actor_lr=1e-5,
    critic_lr=9e-5,
    epochs=2,
    minibatches=2,
    retention=None,
    action_prior=None,
    ppo_config=None,
):
    from native_stand.learning.ppo.controlled import ControlledPPO

    class PhysicalRetentionPPO(ControlledPPO):
        def actor_auxiliary_loss(self, batch, valid):
            # Called only after the shared PPO gate has established a valid
            # Actor batch; do not synchronize the same mask a second time.
            # retention与action_prior相互独立：两者可单独启用/停用。
            losses = []
            if retention is not None:
                x, y, coefficient = retention() if callable(retention) else retention
                if coefficient < 0:
                    raise ValueError("Nonnegative physical retention coefficient")
                # Frozen quality-qualified nominal states, not unknown catch/DR
                # means forcibly labelled from another decoder.
                if coefficient > 0:
                    residual = (self.actor.physical_mean(x) - y) / x.new_tensor(
                        [0.05, 0.05, 10, 0.05, 0.05, 10]
                    )
                    losses.append(
                        coefficient
                        * torch.nn.functional.huber_loss(residual, torch.zeros_like(residual))
                    )
            if action_prior is not None:
                prior_loss = action_prior(self.actor, batch, valid)
                self.action_prior_readback = action_prior.last
                if prior_loss is not None:
                    losses.append(prior_loss)
            if not losses:
                return None
            aux = losses[0]
            for extra in losses[1:]:
                aux = aux + extra
            if getattr(self, "recovery_objective_diagnostics", False) and not getattr(
                self, "retention_readback_done", False
            ):
                self.retention_readback_done = True
                pars = list(self.actor.parameters())
                before = [p.detach().clone() for p in pars]
                logp = self.actor.get_output_log_prob(batch.actions)
                ratio = torch.exp(logp[valid] - batch.old_actions_log_prob.squeeze(-1)[valid])
                adv = batch.advantages.squeeze(-1)[valid]
                surrogate = torch.maximum(
                    -adv * ratio, -adv * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
                ).mean()

                def vector(loss):
                    gs = torch.autograd.grad(loss, pars, retain_graph=True, allow_unused=True)
                    return torch.cat(
                        [
                            (torch.zeros_like(p) if g is None else g).detach().flatten()
                            for p, g in zip(pars, gs, strict=False)
                        ]
                    )

                pg = vector(surrogate)
                bg = vector(aux)

                def after():
                    move = torch.cat(
                        [(p.detach() - old).flatten() for p, old in zip(pars, before, strict=False)]
                    )

                    def cosine(a, b):
                        denom = a.norm() * b.norm()
                        return float(torch.dot(a, b) / denom) if denom > 1e-20 else None

                    self.recovery_objective_audit = dict(
                        PPO_gradient_norm=float(pg.norm()),
                        weighted_BC_gradient_norm=float(bg.norm()),
                        BC_to_PPO_gradient_ratio=float(bg.norm() / pg.norm())
                        if pg.norm() > 1e-20
                        else None,
                        actual_update_norm=float(move.norm()),
                        actual_update_cosine_PPO_descent=cosine(move, -pg),
                        actual_update_cosine_BC_descent=cosine(move, -bg),
                        retention_scope="physical-mean standing anchor plus explicit weak UNKNOWN old-Recovery action prior when configured; no variance retention or behavior certificate",
                    )

                self.optimizer.audit_after_actor_step = after
            return aux

    ppo = PhysicalRetentionPPO(
        actor,
        critic,
        storage,
        learning_rate=actor_lr,
        schedule="fixed",
        num_learning_epochs=epochs,
        num_mini_batches=minibatches,
        entropy_coef=ppo_config.entropy_coef if ppo_config is not None else 0.0,
        clip_param=ppo_config.clip if ppo_config is not None else 0.2,
        gamma=ppo_config.gamma if ppo_config is not None else 0.9980019986673331,
        lam=ppo_config.lam if ppo_config is not None else 0.95,
        value_loss_coef=ppo_config.value_coef if ppo_config is not None else 1.0,
        max_grad_norm=ppo_config.max_grad_norm if ppo_config is not None else 1.0,
        use_clipped_value_loss=False,
        device=device,
    )
    ppo.optimizer = IndependentAdam(actor, critic, actor_lr, critic_lr)
    return ppo


def capture(ppo, *, updates, curriculum, execution_state=None):
    return copy.deepcopy(
        dict(
            version=VERSION,
            actor=ppo.actor.state_dict(),
            critic=ppo.critic.state_dict(),
            optimizer=ppo.optimizer.state_dict(),
            optimizer_parameter_binding=ppo.optimizer.binding(),
            distribution=dict(
                raw_units=["rad", "rad", "rad/s", "rad", "rad", "rad/s"],
                fixed_std=ppo.actor.physical_std.tolist(),
                entropy_coefficient=0.0,
                second_mean_Gaussian=False,
            ),
            contract=ppo.actor.contract,
            rng=torch.get_rng_state(),
            updates=updates,
            curriculum=curriculum,
            execution_state=execution_state,
            actor_update_enabled=ppo.actor_update_enabled,
        )
    )


def restore(ppo, state):
    if state["version"] != VERSION or state["contract"] != ppo.actor.contract:
        raise ValueError("Recovery distribution/contract mismatch")
    if state["distribution"]["fixed_std"] != ppo.actor.physical_std.tolist():
        raise ValueError("Physical std changed")
    for key, expected in [
        ("physical_std", ppo.actor.physical_std),
        ("center", ppo.actor.center),
        ("lower", ppo.actor.lower),
        ("upper", ppo.actor.upper),
    ]:
        if not torch.equal(state["actor"][key].detach().cpu(), expected.detach().cpu()):
            raise ValueError("Frozen physical Actor buffer mismatch " + key)
    if state.get("optimizer_parameter_binding") != ppo.optimizer.binding():
        raise ValueError("Optimizer parameter names/order/shape/scope binding mismatch")
    ppo.actor.load_state_dict(state["actor"], strict=True)
    ppo.critic.load_state_dict(state["critic"], strict=True)
    ppo.optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"])
    ppo.actor_update_enabled = state["actor_update_enabled"]
    return state["updates"], state["curriculum"]


def build_policy(job, observation_sample, *, device: str = "cpu", source=None):
    """Recovery物理动作装配：合同来自job.assets，PPO参数全部来自job.ppo。

    source只在actor_only初始化时使用：Recovery格式Actor（带output_contract）
    直接加载；C0来源经initialize_from_C0做一次显式任务列偏置转换。
    full恢复走capture/restore适配器，不在这里裁剪checkpoint。
    """
    import json as _json

    from . import assembly as _assembly

    contract_ref = job.assets.get("recovery_training_contract") or {}
    if not contract_ref.get("path"):
        raise ValueError("Recovery物理动作需要assets.recovery_training_contract")
    contract_path = Path(contract_ref["path"])
    contract = _json.loads(contract_path.read_text())
    actor = PhysicalActor(contract, hidden_dims=job.model.actor_hidden_dims)

    mode = job.initialization.mode
    payload = getattr(source, "payload", source) or {}
    if mode == "actor_only" and "optimizer" not in payload:
        if not payload:
            raise ValueError("actor_only初始化需要来源checkpoint")
        if payload.get("output_contract") is not None:
            # 已是Recovery格式：直接加载Actor，不做C0转换。
            weights = payload["actor"]
            if payload.get("actor_only") and set(weights) == {
                "mlp." + k for k in actor.mlp.state_dict()
            }:
                actor.mlp.load_state_dict(
                    {k.removeprefix("mlp."): v for k, v in weights.items()}, strict=True
                )
            else:
                actor.load_state_dict(weights, strict=True)
        else:
            # C0来源：一次显式任务列偏置转换。
            initialize_from_C0(actor, payload["actor"])
    elif mode == "full" and payload:
        pass  # full由restore适配器处理，不在此处部分加载。

    critic = _assembly.build_critic(job)
    storage = _assembly.build_storage(observation_sample, job, device=device)
    retention = None
    anchor = job.assets.get("qualified_physical_BC_anchor")
    if anchor is not None:
        import numpy as np

        from native_stand.artifacts.io import file_sha256

        if file_sha256(anchor["path"]) != anchor["sha256"]:
            raise ValueError("Recovery合格物理BC来源SHA不同")
        sampling = anchor.get("sampling")
        if sampling not in (None, "full_pool_height_dq_stratified"):
            raise ValueError("未知Recovery BC采样版本")
        with np.load(anchor["path"], allow_pickle=False) as data:
            ids = (
                np.arange(len(data["observation"]))
                if sampling
                else np.linspace(
                    0, len(data["observation"]) - 1, min(128, len(data["observation"])), dtype=int
                )
            )
            x = torch.tensor(data["observation"][ids], dtype=torch.float32, device=device)
            y = torch.tensor(data["physical_target"][ids], dtype=torch.float32, device=device)
        if not bool((x[:, 25:31] == x.new_tensor([0, 1, 0, 1, 0, 0])).all()):
            raise ValueError("Recovery BC任务标志不一致")
        if sampling:
            from .physical_anchor import PhysicalAnchor

            retention = PhysicalAnchor(x, y, anchor["coefficient"], anchor["batch_size"])
        else:
            retention = (x, y, anchor["coefficient"])
    action_prior = None
    prior = job.assets.get("old_Recovery_action_prior")
    if prior is not None:
        from native_stand.artifacts.io import file_sha256

        from .networks import StudentNetwork
        from .recovery_action_prior import ActionPrior

        if file_sha256(prior["path"]) != prior["sha256"]:
            raise ValueError("Recovery action prior SHA不同")
        old = torch.load(prior["path"], map_location=device, weights_only=False)
        teacher = StudentNetwork(
            actor=True,
            input_dim=31,
            output_dim=6,
            hidden_dims=tuple(prior["actor_hidden_dims"]),
            init_std=torch.ones(6),
        ).to(device)
        teacher.load_state_dict(old["actor"], strict=True)
        action_prior = ActionPrior(teacher, prior["coefficient"], prior["max_rows"])
    ppo = assemble(
        actor,
        critic,
        storage,
        device=device,
        actor_lr=job.ppo.learning_rate,
        critic_lr=job.ppo.critic_learning_rate,
        epochs=job.ppo.epochs,
        minibatches=job.ppo.minibatches,
        ppo_config=job.ppo,
        retention=retention,
        action_prior=action_prior,
    )

    auxiliary_identity = dict(
        BC=None
        if anchor is None
        else {key: anchor.get(key) for key in ("sha256", "coefficient", "sampling", "batch_size")},
        prior=None
        if prior is None
        else {
            key: prior[key] for key in ("sha256", "coefficient", "max_rows", "actor_hidden_dims")
        },
        sampling_rng="checkpointed Torch CPU/CUDA generator; no independent untracked sampler",
    )

    def _capture(counters, curriculum=None, execution_state=None):
        state = capture(
            ppo,
            updates=counters.get("updates", 0),
            curriculum=curriculum,
            execution_state=execution_state,
        )
        # 对齐完整checkpoint契约字段：std是固定物理std（不是归一化log_std）、
        # rng在正确最终边界捕获、counters记录轮次/样本。
        state["std"] = ppo.actor.physical_std.detach().clone()
        state["rng"] = torch.get_rng_state()
        if torch.cuda.is_initialized():
            state["CUDA_rng"] = torch.cuda.get_rng_state_all()
        elif payload.get("CUDA_rng") is not None:
            # CPU assembly can preserve device state without executing it.
            state["CUDA_rng"] = copy.deepcopy(payload["CUDA_rng"])
        if (
            payload.get("optimizer") is not None
            and payload.get("actor_update_enabled")
            and payload["updates"] >= job.ppo.critic_warmup_updates
        ):
            state["warmup_updates_done"] = job.ppo.critic_warmup_updates
        state["recovery_auxiliary_identity"] = copy.deepcopy(auxiliary_identity)
        if action_prior is not None:
            state["action_prior_total_queried_rows"] = action_prior.total_queried_rows
        state["counters"] = dict(counters)
        return state

    def _restore(state):
        if (
            state.get("recovery_auxiliary_identity") is not None
            and state["recovery_auxiliary_identity"] != auxiliary_identity
        ):
            raise ValueError("Recovery BC/prior来源或采样合同不同")
        restore(ppo, state)
        if action_prior is not None:
            action_prior.total_queried_rows = state.get("action_prior_total_queried_rows", 0)
        if state.get("rng") is not None:
            # RNG在对象/环境恢复之后设置，避免被后续初始化消耗。
            torch.set_rng_state(state["rng"])

    return {
        "ppo": ppo,
        "actor_class": type(actor).__name__,
        "observation_keys": {
            "actor": "student_actor",
            "critic": "student_critic",
            "action_valid": "recovery_actor_valid",
        },
        "action_valid_key": "recovery_actor_valid",
        "capture": _capture,
        "restore": _restore,
        "contract": contract,
        "contract_sha256": __import__("hashlib").sha256(contract_path.read_bytes()).hexdigest(),
    }
