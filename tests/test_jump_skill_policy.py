"""New Jump recipe through the real shared PPO, checkpoint and dispatch boundaries."""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_jump_unified import observation

from native_stand.artifacts.checkpoint import load_full, save_full
from native_stand.experiments.config import load_job_config
from native_stand.experiments.plan import build_plan
from native_stand.learning.policies.factory import build_policy
from native_stand.learning.ppo.actor_initialization import physical_std
from native_stand.learning.ppo.jump_unified import DirectActor
from native_stand.simulation.dispatch import execution_service
from native_stand.tasks.jump.spec import build_task

FROZEN_TEST_SCOPE = "PORTABLE"
ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs/jobs/jump_first_takeoff_isaac_prepared.json"


def small_job():
    job = load_job_config(RECIPE)
    return replace(
        job,
        model=replace(job.model, actor_hidden_dims=(8,), critic_hidden_dims=(8,)),
        runtime=replace(job.runtime, envs=2, steps_per_env=4),
        ppo=replace(job.ppo, epochs=2, minibatches=1),
        diagnostics=dict(
            job.diagnostics, minimum_unique_actor_samples=1, minimum_minibatch_samples=1
        ),
    )


def sample():
    obs = observation()
    obs["student_critic"] = torch.arange(2 * 5593, dtype=torch.float32).reshape(2, 5593) / 11186
    return obs


def update(handle, obs):
    ppo = handle["ppo"]
    with torch.inference_mode():
        for _ in range(4):
            action = ppo.act(obs)
            rewards = -(action[:, :1] - 1).square() + action[:, 2:3]
            ppo.process_env_step(obs, rewards, torch.zeros(2, dtype=torch.bool), {})
        ppo.compute_returns(obs)
    return ppo.update()


def equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, tuple | list):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


def test_actor_only_preserves_mean_uses_fresh_two_adam_and_frozen_physical_std():
    job, obs = small_job(), sample()
    source_actor = DirectActor((8,))
    source = dict(
        actor_only=True,
        actor=copy.deepcopy(source_actor.state_dict()),
        output_contract=build_task(job).contract,
    )
    original = copy.deepcopy(source)
    handle = build_policy(job, obs, source=source)
    ppo = handle["ppo"]
    assert torch.equal(ppo.actor(obs), source_actor(obs))
    equal(source, original)
    assert not ppo.optimizer.actor.state and not ppo.optimizer.critic.state
    std = ppo.actor.log_std.detach().clone()
    torch.testing.assert_close(std.exp(), physical_std(job), atol=0, rtol=2e-7)
    assert not ppo.actor.log_std.requires_grad
    assert all(p.requires_grad for name, p in ppo.actor.named_parameters() if name != "log_std")
    before = copy.deepcopy(ppo.actor.state_dict())
    critic_before = copy.deepcopy(ppo.critic.state_dict())
    ppo.actor_update_enabled = False
    update(handle, obs)
    equal(before, ppo.actor.state_dict())
    assert any(not torch.equal(v, ppo.critic.state_dict()[k]) for k, v in critic_before.items())
    assert not ppo.optimizer.actor.state and ppo.optimizer.critic.state
    ppo.actor_update_enabled = True
    update(handle, obs)
    assert any(not torch.equal(v, ppo.actor.state_dict()[k]) for k, v in before.items())
    assert torch.equal(std, ppo.actor.log_std) and ppo.actor.log_std.grad is None
    assert ppo.optimizer.actor.state and ppo.optimizer.critic.state


def test_full_resume_repeats_raw_samples_optimizer_and_rng_exactly(tmp_path):
    job, obs = small_job(), sample()
    torch.manual_seed(442)
    source = dict(
        actor_only=True,
        actor=DirectActor((8,)).state_dict(),
        output_contract=build_task(job).contract,
    )
    handle = build_policy(job, obs, source=source)
    update(handle, obs)
    saved = handle["capture"]({"updates": 1})
    save_full(tmp_path / "full.pt", saved)
    update(handle, obs)
    continuous = handle["capture"]({"updates": 2})
    restored = build_policy(job, obs)
    restored["restore"](load_full(tmp_path / "full.pt"))
    update(restored, obs)
    equal(continuous, restored["capture"]({"updates": 2}))
    assert not restored["ppo"].actor.log_std.requires_grad
    # An old critic or a changed distribution is a new experiment, not an implicit resume.
    for field, value in (("critic_schema", {"width": 5591}), ("jump_learning_semantics", {})):
        with pytest.raises(ValueError):
            restored["restore"](dict(saved, **{field: value}))
    bad = copy.deepcopy(saved)
    bad["actor"]["log_std"] += 0.01
    bad["std"] = bad["actor"]["log_std"].exp()
    with pytest.raises(ValueError, match="fixed physical exploration"):
        restored["restore"](bad)
    bad = dict(saved, std=saved["std"] * 1.1)
    with pytest.raises(ValueError, match="std"):
        restored["restore"](bad)
    other_lr = replace(job, ppo=replace(job.ppo, learning_rate=3e-5))
    with pytest.raises(ValueError, match="优化器"):
        build_policy(other_lr, obs)["restore"](saved)


def test_prepared_recipes_have_zero_allocation_and_a_single_lr_difference():
    job = load_job_config(RECIPE)
    other = load_job_config(ROOT / "configs/jobs/jump_first_takeoff_isaac_lr3e5_prepared.json")
    first, second = job.as_run_record(), other.as_run_record()
    for key in ("name", "config_path", "config_sha256"):
        first.pop(key)
        second.pop(key)
    assert first.pop("ppo")["learning_rate"] == 6e-5
    assert second.pop("ppo")["learning_rate"] == 3e-5
    assert replace(job.ppo, learning_rate=3e-5) == other.ppo
    assert first == second
    plan = build_plan(job)
    assert plan["total_updates"] == 0 and not plan["stages"]
    assert job.runtime.samples_per_update == 131072
    assert job.storage.checkpoint_every_updates == 100
    assert job.assets["jump_ground"] == "native_plane_v1"
    assert build_task(job).contract["control"]["final_motor_caps"] == [40, 40, 4.8, 40, 40, 4.8]
    assert execution_service(job)["module"] == "native_stand.simulation.jump_backend"


@pytest.mark.parametrize(
    "invalid", ["objective", "exploration", "entropy", "critic", "horizon", "clearance"]
)
def test_semantic_mismatch_is_rejected_before_policy_creation(invalid):
    job = small_job()
    assets = copy.deepcopy(job.assets)
    if invalid == "objective":
        assets["jump_course"]["levels"][0]["objective"] = 2
    elif invalid == "exploration":
        assets["jump_exploration"] = "trainable_std"
    elif invalid == "entropy":
        job = replace(job, ppo=replace(job.ppo, entropy_coef=0.001))
    elif invalid == "horizon":
        assets["jump_course"]["config"]["active_horizon_s"] = 16
    elif invalid == "clearance":
        assets["jump_course"]["levels"][0]["clearance_m"] = 0.004
    else:
        job = replace(
            job,
            model=replace(
                job.model, critic_privileged_version="jump_isaac_full_execution_critic_v1"
            ),
        )
    with pytest.raises(ValueError):
        build_task(replace(job, assets=assets))
