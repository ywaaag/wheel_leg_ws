"""v9 actual lifecycle/adapter: v8 positives unchanged, one finite reward per env."""

import copy
import dataclasses

import pytest
import torch
from test_recovery_continuation import fixture

from native_stand.artifacts.checkpoint import load_source, save_full
from native_stand.simulation.driver import Backend
from native_stand.simulation.recovery_adapter import RecoveryAdapter
from native_stand.simulation.recovery_isaac.preentry_lifecycle import Lifecycle as V8
from native_stand.simulation.recovery_isaac.settle_lifecycle import (
    VERSION,
    Lifecycle,
    settling_density,
)


def inputs(n=2):
    s = torch.zeros(n, 13)
    s[:, 3] = torch.tensor([0.34, 0.168])[:n]
    s[:, 9:11] = 3
    s[:, 12] = 1
    return dict(
        state=s,
        time_s=torch.ones(n),
        target=torch.zeros(n, 6),
        motor=torch.zeros(n, 6),
        student=torch.ones(n, dtype=torch.bool),
        terminal=torch.zeros(n, dtype=torch.bool),
        truncated=torch.zeros(n, dtype=torch.bool),
        hp10=torch.zeros(n, 4),
        angular_velocity=torch.ones(n, 3),
        physical_q=torch.zeros(n, 6),
        clearance=torch.ones(n),
    )


def test_v8_reward_is_preserved_at_low_swing_and_no_extra_positive_credit():
    v8, v9 = V8(2), Lifecycle(2)
    x = inputs()
    for life in (v8, v9):
        life.reset(torch.arange(2), torch.tensor([0.34, 0.168]), torch.zeros(2))
    a, b = v8.observe(**x), v9.observe(**x)
    assert b["reward"].shape == (2,)
    assert b["reward"][1] == a["reward"][1]
    assert b["reward"][0] < a["reward"][0]
    for name, value in a["density"].items():
        assert torch.equal(value, b["density"][name])
    assert torch.equal(a["success_tail_credit"], b["success_tail_credit"])
    assert torch.equal(a["entry_bonus"], b["entry_bonus"])
    assert torch.equal(v9.total_reward, b["reward"])


def test_continuous_gate_does_not_wait_for_entry_or_require_contact():
    x = inputs()
    s = x["state"]
    s[:, 3] = torch.tensor([0.27, 0.32])
    s[:, 9:11] = 0
    gate, costs = settling_density(s, x["angular_velocity"], torch.full((2,), 0.34), x["student"])
    assert 0 < gate[0] < gate[1] <= 1
    assert costs["early_settle_contact_cost"][1] < 0
    _, masked = settling_density(s, x["angular_velocity"], torch.full((2,), 0.34), ~x["student"])
    assert all(torch.count_nonzero(v) == 0 for v in masked.values())


def test_hf_remains_diagnostic_only_and_timeout_not_bootstrapped():
    x = inputs()
    a, b = Lifecycle(2), Lifecycle(2)
    y = copy.deepcopy(x)
    y["hp10"][:] = 1000
    x["truncated"][:] = True
    y["truncated"][:] = True
    u, v = a.observe(**x), b.observe(**y)
    assert torch.equal(u["reward"], v["reward"])
    assert u["done"].all()


def test_critic_adaptation_formal_build_save_and_resume_does_not_repeat(tmp_path):
    _, context, io = fixture(tmp_path)
    context.updates_used = 0
    context.job = dataclasses.replace(
        context.job,
        reward=dataclasses.replace(context.job.reward, version=VERSION),
        assets=dict(context.job.assets, reward_critic_adaptation=dict(version=VERSION, updates=5)),
    )
    context.total_updates_before = lambda stage: 5
    backend = Backend("CPU-I/O", lambda job: io)
    adapter = RecoveryAdapter(backend)
    adapter.build(context, "formal", None)
    assert adapter._prior_warmup == 5 and adapter._warmup == 5
    for count in [3, 5]:
        context.updates_used = count
        state = adapter.capture(context, {"updates": count})
        assert state["reward_critic_adaptation"]["completed"] == count
        state["run_identity"] = str(context.run_dir)
        cp = context.run_dir / f"resume{count}.pt"
        save_full(cp, state)
        resumed = RecoveryAdapter(backend)
        resumed.build(context, "formal", load_source(cp))
        assert resumed._warmup == 5
        assert resumed._prior_warmup == 5
    del state["reward_critic_adaptation"]
    cp = context.run_dir / "bad.pt"
    save_full(cp, state)
    with pytest.raises(ValueError, match="missing reward"):
        RecoveryAdapter(backend).build(context, "formal", load_source(cp))


def test_five_new_critic_updates_preserve_actor_adam_then_sixth_updates_actor(tmp_path):
    from tensordict import TensorDict

    _, context, io = fixture(tmp_path)
    context.job = dataclasses.replace(
        context.job,
        reward=dataclasses.replace(context.job.reward, version=VERSION),
        assets=dict(context.job.assets, reward_critic_adaptation=dict(version=VERSION, updates=5)),
    )
    context.updates_used = 0
    context.total_updates_before = lambda stage: 5
    obs = io.observations()
    obs["student_actor"] = torch.randn(2, 31)
    obs["student_critic"] = torch.randn(2, 909)
    io.observations = lambda: TensorDict({k: v.clone() for k, v in obs.items()}, [2])
    io.infinite_horizon = False
    io.num_envs = 2
    io.step = lambda raw: dict(
        rewards=torch.tensor([1.0, -0.3]),
        dones=torch.zeros(2, dtype=torch.bool),
        terminated=torch.zeros(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
        extras={},
    )
    adapter = RecoveryAdapter(Backend("CPU-I/O", lambda job: io))
    adapter.build(context, "formal", None)
    before = copy.deepcopy(adapter.ppo.actor.state_dict())
    optimizer_before = copy.deepcopy(adapter.ppo.optimizer.state_dict()["actor"])
    for i in range(6):
        context.updates_used = i
        metric = adapter.update_once(context, i + 1)
        assert metric["critic_optimizer_steps"] > 0
        if i < 5:
            assert metric["actor_optimizer_steps"] == 0
            assert all(torch.equal(v, adapter.ppo.actor.state_dict()[k]) for k, v in before.items())
            assert adapter.ppo.optimizer.state_dict()["actor"] == optimizer_before
        else:
            assert metric["actor_optimizer_steps"] > 0
            assert any(
                not torch.equal(v, adapter.ppo.actor.state_dict()[k]) for k, v in before.items()
            )
