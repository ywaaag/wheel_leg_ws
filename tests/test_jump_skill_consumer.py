"""Real Isaac step/adapter consumers with explicit scripted physical I/O only."""

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import torch
from test_jump_skill_course import jump_packet, scorer
from test_jump_skill_policy import sample, small_job
from test_jump_unified import packet

from native_stand.learning.policies.factory import build_policy
from native_stand.learning.ppo.performance import Measurement
from native_stand.simulation.engine import GenericPPOAdapter
from native_stand.simulation.jump_isaac import IsaacJump
from native_stand.tasks.jump.adaptive_course import AdaptiveCourse, CourseConfig
from native_stand.tasks.jump.device_reward import DeviceReward
from native_stand.tasks.jump.skill_course import SkillLevel

FROZEN_TEST_SCOPE = "PORTABLE"


def scripted_env(*, fault=False):
    env = IsaacJump.__new__(IsaacJump)
    env.num_envs, env.device = 2, "cpu"
    env.measurement = Measurement(False, "cpu")
    env.outcome_objective = True
    env.reward_config = scorer().config
    env.teaching_commands = None
    env.courses = [
        AdaptiveCourse(
            seed=i,
            plant_id="scripted",
            DR_id="nominal",
            config=CourseConfig(
                window_episodes=1, promote_windows=1, require_landing_rehearsal=False
            ),
            levels=[SkillLevel(str(k), 1, 0.02, 0.005, 0.18, k) for k in range(3)],
        )
        for i in range(2)
    ]
    env.requests = [c.next_episode() for c in env.courses]
    env.rewards = [scorer(0), scorer(0)]
    for tick in range(39):
        env.rewards[0].step(jump_packet(tick))
    env.device_reward = DeviceReward(env.rewards, "cpu")
    env.device_phase_seconds = torch.zeros(2, 10, dtype=torch.float64)
    env.device_balance_s = torch.zeros(2, dtype=torch.float64)
    env.device_balance_good = torch.zeros(2, dtype=torch.float64)
    env.balance_s, env.balance_good = np.zeros(2), np.zeros(2)
    env.motor, env.NET, env.passive = (torch.zeros(2, 6) for _ in range(3))
    env.completed, env.serial = [], 0
    env.prepare = lambda *_: None
    env.validate_device_measurement = lambda: None
    env.critic_snapshot = lambda: None
    states = [
        SimpleNamespace(phase=3, owner=2, fault=0, request_id=1, epoch=1),
        SimpleNamespace(phase=9, owner=2, fault=0, request_id=0, epoch=0),
    ]

    def observations():
        obs = sample()
        for i, state in enumerate(states):
            obs["jump_C_phase"][i] = state.phase
            obs["student_actor"][i, 31:36] = 0
            if 1 <= state.phase <= 5:
                obs["student_actor"][i, 30 + state.phase] = 1
            obs["student_critic"][i, 0] = env.rewards[i].elapsed_s
        return obs

    env.observations = observations
    # Scripted I/O deliberately replaces only instance boundaries. No engine,
    # global implementation replacement, hidden controller or task evaluator.
    env.execution = SimpleNamespace(
        tick=0,
        publish_raw=lambda *_: np.zeros(2),
        c=SimpleNamespace(
            states=states,
            observe=lambda: (observations()["student_actor"].numpy(), np.ones(2, bool)),
        ),
    )

    def step(*_):
        env.execution.tick += 1
        return torch.zeros(2, 6), torch.zeros(2, 6)

    env.execution.step = step
    env.scene = SimpleNamespace(
        ids=list(range(6)),
        robot=SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=SimpleNamespace(torch=torch.zeros(2, 6)),
                joint_vel=SimpleNamespace(torch=torch.zeros(2, 6)),
            )
        ),
        apply_effort=lambda *_: torch.zeros(2, 6),
    )

    def values():
        packets = [jump_packet(39, fault=fault), packet(phase=9)]
        return {
            f.name: torch.as_tensor(np.array([getattr(p, f.name) for p in packets]))
            for f in fields(packets[0])
        }

    env.device_packet = values
    env.host = lambda: dict(
        root=np.tile([0, 0, 0.34] + [0] * 10, (2, 1)),
        spring=np.full((2, 2), 0.18),
        length=np.full((2, 2), 0.28),
        closure=np.zeros((2, 4)),
        restricted=np.zeros(2, bool),
        normal=np.ones((2, 2)),
    )
    env.row_observation = lambda i, *_args, **_kwargs: observations()["student_critic"][i].numpy()
    env.reset_ids = []

    def reset(ids):
        env.reset_ids.extend(ids)
        for i in ids:
            env.requests[i] = env.courses[i].next_episode()
            env.rewards[i] = scorer(env.requests[i]["objective"])
            states[i].phase = 9
        env.device_reward.reset(ids, env.rewards)

    env.reset = reset
    return env


def test_mid_policy_success_is_recorded_once_resets_only_ended_row_and_keeps_terminal():
    env = scripted_env()
    out = env.step(torch.zeros(2, 6))
    assert out["dones"].tolist() == [True, False]
    assert out["extras"]["episode_physics_steps"] == [1, 10]
    assert out["extras"]["transition_valid"].tolist() == [True, True]
    assert env.reset_ids == [0] and len(env.completed) == 1
    ended = env.completed[0]
    assert ended["course_success"] and not ended["success"] and ended["objective"] == 0
    assert env.courses[0].level == 1 and env.courses[1].level == 0
    assert env.requests[0]["objective"] == 1 and not env.rewards[0].has_taken_off
    assert out["terminal_observations"]["student_critic"][0, 0] > 0
    assert out["observations"]["student_critic"][0, 0] == 0
    assert out["terminal_observations"]["jump_C_phase"][0] == 3
    assert out["observations"]["jump_C_phase"][0] == 9


def test_short_fault_remains_invalid_and_cannot_become_course_success():
    env = scripted_env(fault=True)
    out = env.step(torch.zeros(2, 6))
    assert out["extras"]["transition_valid"].tolist() == [False, True]
    assert env.completed[0]["fault"] and not env.completed[0]["course_success"]
    assert env.courses[0].level == 0


def test_public_adapter_actually_stores_the_success_action_as_valid_for_ppo():
    env = scripted_env()
    job = small_job()
    job = replace(job, runtime=replace(job.runtime, steps_per_env=1))
    handle = build_policy(job, env.observations())
    adapter = GenericPPOAdapter.__new__(GenericPPOAdapter)
    adapter.job, adapter.env, adapter.ppo = job, env, handle["ppo"]
    adapter.measurement = Measurement(False, "cpu")
    adapter._warmup, adapter._steps_per_env = 0, 1
    result = adapter.update_once(SimpleNamespace(updates_used=0), 0)
    assert adapter.ppo.storage.observations["recovery_actor_valid"][0, :, 0].tolist() == [
        True,
        True,
    ]
    assert result["actor_optimizer_steps"] > 0 and result["critic_optimizer_steps"] > 0
    assert env.completed[0]["course_success"] and not env.completed[0]["success"]
