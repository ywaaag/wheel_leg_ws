"""Actual local BC60/C/V4/assets; no physical integration or CUDA permitted."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from fixtures.jump_isaac_schema import schema
from test_jump_batch_bridge_cpu import bridge  # noqa: F401
from test_jump_cpu_reuse import no_physics_or_cuda  # noqa: F401
from test_jump_skill_course import scorer
from test_jump_skill_policy import RECIPE, sample
from test_jump_unified import observation, packet

from native_stand.artifacts.checkpoint import load_actor_source
from native_stand.artifacts.io import json_digest
from native_stand.experiments.config import load_job_config
from native_stand.learning.policies.factory import build_policy
from native_stand.learning.ppo.jump_unified import DirectActor
from native_stand.learning.ppo.performance import Measurement
from native_stand.simulation.jump_backend import validate_job
from native_stand.simulation.jump_isaac import IsaacJump
from native_stand.tasks.jump.adaptive_course import AdaptiveCourse
from native_stand.tasks.jump.skill_course import SkillTaskSpec

FROZEN_TEST_SCOPE = "HOST"


def test_actual_C_V4_schema_and_all_bound_dependencies_match_recipe(tmp_path):
    job = load_job_config(RECIPE)
    validate_job(job)
    actual = schema(job, tmp_path)
    assert actual["width"] == 5593 and len(actual["fields"]) == 228
    assert json_digest(actual) == json_digest(job.assets["jump_critic_schema"])


def test_actual_BC60_actor_initialization_preserves_mean_with_fresh_learning_state():
    job = load_job_config(RECIPE)
    job = replace(job, runtime=replace(job.runtime, envs=2, steps_per_env=1))
    source = load_actor_source(job.initialization.checkpoint)
    assert source.sha256 == job.initialization.sha256
    fit = Path(job.initialization.checkpoint).parents[2] / "supervision/phase_009_fit"
    from native_stand.artifacts.io import file_sha256

    assert source.payload["source_sha256"] == file_sha256(fit / "student_actor.pt")
    bc_state = torch.load(fit / "distillation_latest.pt", map_location="cpu", weights_only=False)
    assert bc_state["round"] == 60
    assert {int(row["step"]) for row in bc_state["optimizer"]["state"].values()} == {3060}
    actor = DirectActor(job.model.actor_hidden_dims)
    actor.load_state_dict(source.payload["actor"], strict=True)
    handle = build_policy(job, sample(), source=source)
    for key, value in source.payload["actor"].items():
        if key != "log_std":
            assert torch.equal(handle["ppo"].actor.state_dict()[key], value)
    with torch.inference_mode():
        assert torch.equal(actor(sample()), handle["ppo"].actor(sample()))
    assert not handle["ppo"].optimizer.actor.state and not handle["ppo"].optimizer.critic.state
    assert not handle["ppo"].actor.log_std.requires_grad
    assert handle["capture"]({"updates": 0})["source_supervised_updates"] == 3060


def test_actual_Isaac_reset_dispatches_objective_and_leaves_other_real_C_rows_untouched(bridge):  # noqa: F811
    env = IsaacJump.__new__(IsaacJump)
    job = load_job_config(RECIPE)
    env.job, env.num_envs, env.device = job, 3, "cpu"
    env.execution = bridge()
    env.measurement = Measurement(False, "cpu")
    env.courses = [AdaptiveCourse(seed=i, **job.assets["jump_course"]) for i in range(3)]
    env.courses[0]._move(1, "scripted course evidence")
    env.courses[1]._move(2, "scripted course evidence")
    env.requests = [None] * 3
    env.rewards = [scorer()] * 3
    env.starts, env.balance_s, env.balance_good = np.zeros(3), np.zeros(3), np.zeros(3)
    env.motor, env.NET, env.passive = (torch.zeros(3, 6) for _ in range(3))
    env.teaching_commands, env.prepared_tick, env._obs = None, -1, None
    env.reward_config = scorer().config
    env.inverse = SimpleNamespace(initial=None)
    env.scene = SimpleNamespace(
        reset=lambda *_: None,
        ids=list(range(6)),
        physics=SimpleNamespace(mass=torch.full((3, 1), 20)),
        robot=SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=SimpleNamespace(torch=torch.zeros(3, 6)),
                joint_vel=SimpleNamespace(torch=torch.zeros(3, 6)),
            )
        ),
    )
    env.host = lambda: {}
    env.packet = lambda *_: packet(dt_s=0)
    env.observations = lambda: observation(3)
    env.reset()
    assert all(isinstance(r.spec, SkillTaskSpec) for r in env.rewards)
    assert env.device_reward.goals["objective"].tolist() == [1, 2, 0]
    before = [bytes(s) for s in env.execution.c.states]
    other_rewards = [r.state_dict() for r in env.rewards[1:]]
    env.courses[0]._move(2, "scripted promotion")
    env.reset([0])
    assert env.requests[0]["objective"] == env.rewards[0].spec.objective == 2
    assert [bytes(s) for s in env.execution.c.states[1:]] == before[1:]
    assert [r.state_dict() for r in env.rewards[1:]] == other_rewards
