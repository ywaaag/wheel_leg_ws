"""ABI4 raw Gaussian; common controlled PPO, storage and two Adam optimizers."""

import torch

from native_stand.tasks.jump.unified import VERSION

from .controlled import KEY, ControlledPPO
from .jump_nominal import NominalPPO
from .jump_nominal import build_policy as assemble_policy
from .networks import StudentNetwork


class DirectActor(StudentNetwork):
    def __init__(self, hidden_dims=(2048, 1024, 512), *, std=0.12):
        super().__init__(
            actor=True,
            input_dim=37,
            hidden_dims=hidden_dims,
            output_dim=6,
            init_std=torch.full((6,), std),
        )

    def mean(self, x):
        return self.mlp(x)


def validate_saved_std(saved, log_std):
    """The authoritative log_std is exact; exp differs across CPU/CUDA kernels."""
    saved, log_std = saved.detach().cpu(), log_std.detach().cpu()
    expected = log_std.exp()
    if saved.shape != expected.shape or saved.dtype != expected.dtype:
        raise ValueError("Jump raw std与Actor状态冲突")
    spacing = torch.nextafter(expected, torch.full_like(expected, float("inf"))) - expected
    if not bool(
        (
            torch.isfinite(saved)
            & torch.isfinite(expected)
            & (saved > 0)
            & (expected > 0)
            & ((saved - expected).abs() <= 2 * spacing)
        ).all()
    ):
        raise ValueError("Jump raw std与Actor状态冲突")


def validate_phases(observations, valid):
    actor = observations["student_actor"]
    bits = actor[..., 31:36]
    phase = observations["jump_C_phase"].squeeze(-1)
    owner = observations["jump_C_owner"].squeeze(-1)
    balance = (phase == 9) & (bits == 0).all(-1) & (actor[..., 36] == 0)
    active = (phase >= 1) & (phase <= 5) & ((bits == 0) | (bits == 1)).all(-1) & (bits.sum(-1) == 1)
    encoded = torch.nn.functional.one_hot((phase - 1).clamp(0, 4).long(), 5)
    active &= (bits == encoded).all(-1)
    flags = (actor[..., 25:31] == actor.new_tensor([0, 0, 1, 0, 0, 1])).all(-1)
    if bool((valid & ~((balance | active) & (owner == 2) & flags)).any()):
        raise ValueError("Jump真实C阶段/owner与37D编码不一致")


class DirectPPO(NominalPPO):
    def valid_actor_sample_count(self, valid):
        return self.unique_samples

    def update(self):
        valid = self.storage.observations[KEY].squeeze(-1).bool()
        validate_phases(self.storage.observations, valid)
        self.unique_samples = int(valid.sum())
        self.minibatches = []
        result = ControlledPPO.update(self)
        self.mask_summary.update(
            version=VERSION,
            unique_actor_samples=self.unique_samples,
            minibatches=self.minibatches,
            raw_action_semantics="raw_Normal_direct_linear_no_Tanh",
        )
        return result


def build_policy(job, observations, *, device="cpu", source=None, version=VERSION):
    from native_stand.tasks.jump import skill_course

    handle = assemble_policy(
        job,
        observations,
        device=device,
        actor_class=DirectActor,
        ppo_class=DirectPPO,
        version=version,
        action_semantics="raw_Normal_direct_linear_no_Tanh",
    )
    payload = getattr(source, "payload", source)
    if payload is not None and "optimizer" not in payload:
        if not payload.get("actor_only") or payload.get("output_contract") != handle["contract"]:
            raise ValueError("Jump首次PPO必须是匹配ABI4的Actor-only来源；BC状态不是PPO恢复点")
        handle["ppo"].actor.load_state_dict(payload["actor"], strict=True)
        handle["ppo"].actor.requires_grad_(True)
        from native_stand.tasks.jump.outcome import enabled

        if enabled(job):
            from .actor_initialization import physical_std

            with torch.no_grad():
                handle["ppo"].actor.log_std.copy_(physical_std(job).to(device).log())
    fixed_std = None
    if skill_course.enabled(job):
        from .actor_initialization import physical_std

        fixed_std = physical_std(job)
        with torch.no_grad():
            handle["ppo"].actor.log_std.copy_(fixed_std.log().to(device))
        handle["ppo"].actor.log_std.requires_grad_(False)
    semantics = dict(
        reward=job.reward.version,
        course=job.assets.get("jump_course"),
        objective=job.assets.get("jump_objective"),
    )
    if fixed_std is not None:
        semantics["exploration"] = dict(
            version=skill_course.EXPLORATION, raw_std=fixed_std.tolist()
        )
    capture = handle["capture"]
    restore = handle["restore"]
    source_supervised_updates = int(
        (payload or {}).get(
            "source_supervised_updates",
            (payload or {}).get(
                "supervised_updates",
                job.assets.get("source_supervised_updates", 0)
                if (payload or {}).get("actor_only")
                else 0,
            ),
        )
    )

    def capture_state(*args, **kwargs):
        value = capture(*args, **kwargs)
        value["jump_learning_semantics"] = semantics
        value["critic_schema"] = job.assets["jump_critic_schema"]
        value["source_supervised_updates"] = source_supervised_updates
        value["output_contract"] = handle["contract"]
        return value

    def restore_state(state):
        nonlocal source_supervised_updates
        if (semantics["objective"] is not None or "jump_learning_semantics" in state) and (
            state.get("jump_learning_semantics") != semantics
        ):
            raise ValueError("Jump learning reward/course/controller semantics differ")
        if (
            state.get("critic_schema") != job.assets["jump_critic_schema"]
            or state.get("raw_action_semantics") != "raw_Normal_direct_linear_no_Tanh"
        ):
            raise ValueError("Jump完整恢复Critic/raw合同不同")
        validate_saved_std(state["std"], state["actor"]["log_std"])
        if fixed_std is not None and not torch.allclose(
            state["std"].detach().cpu(), fixed_std, atol=0, rtol=2e-6
        ):
            raise ValueError("Jump fixed physical exploration differs from saved distribution")
        restore(state)
        source_supervised_updates = int(state.get("source_supervised_updates", 0))

    handle.update(capture=capture_state, restore=restore_state)
    return handle
