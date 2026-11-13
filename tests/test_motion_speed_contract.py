"""Production Motion diagnostic cadence; no engine/GPU or behavioral training."""

from __future__ import annotations

import copy
from types import SimpleNamespace as NS

import pytest
import torch
from test_motion_production_integration import equal_tree, full_source, make_job, observations

from native_stand.learning.ppo.supported import assemble_from_full
from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter
from native_stand.students.motion_hotpath import runner_flags
from native_stand.students.motion_statistics import MotionStatistics
from native_stand.students.resume import restore_rng
from native_stand.students.standing_survival_task import bootstrap_reward
from native_stand.students.step_checks import StepChecks
from native_stand.tasks.motion.value_diagnostics import from_storage

FROZEN_TEST_SCOPE = "PORTABLE"


@pytest.mark.parametrize("profile", [False, True])
def test_report_only_diagnostics_keep_raw_logprob_rewards_optimizer_and_RNG_identical(
    tmp_path, profile
):
    job, contract = make_job(tmp_path, retention=False)
    job.diagnostics["performance_measurement"] = profile
    obs = observations(job)
    source = full_source(job, contract, obs)
    results = []
    for prior_updates in (98, 99):
        algorithm = assemble_from_full(source, obs, job)
        restore_rng(source["training_rng"])
        calls = dict(stat_allocations=0, stat_observations=0, value_diagnostics=0)

        class Statistics(MotionStatistics):
            def __init__(self, *args, _calls=calls, **kwargs):
                self.calls = _calls
                self.calls["stat_allocations"] += 1
                super().__init__(*args, **kwargs)

            def observe(self, *args, **kwargs):
                self.calls["stat_observations"] += 1
                return super().observe(*args, **kwargs)

        def value_diagnostics(*args, _calls=calls, **kwargs):
            _calls["value_diagnostics"] += 1
            return from_storage(*args, **kwargs)

        class EngineIO:
            device = "cpu"
            num_envs = 4
            stage_spec = {"commands": [[0.0, 0.0]]}
            command_index = torch.zeros(4, dtype=torch.long)
            rise_hold = NS(episode=NS(steps=torch.zeros(4, dtype=torch.long)))
            student_actor_valid = torch.ones(4, dtype=torch.bool)
            recovery_snapshot = torch.zeros(4, 31)
            standing_snapshot = torch.zeros(4, 6)
            wheel_geometry_snapshot = torch.zeros(4, 9)
            motion_action_command = torch.tensor([[0.0, 0.0, 0.34]]).expand(4, -1)
            supported_requested_target_post = torch.zeros(4, 6)
            supported_adopted_target_post = torch.zeros(4, 6)
            supported_saturation_post = {
                name: torch.zeros(4, 6)
                for name in (
                    "request_over_cap",
                    "driver_at_cap",
                    "motor_at_cap",
                    "motor_capacity_fraction",
                )
            }
            ticks = 0

            def step(self, raw):
                self.ticks += 1
                self.rise_hold.episode.steps.add_(1)
                ended = torch.zeros(4, dtype=torch.bool)
                return obs.clone(), 1.0 - raw.square().mean(-1), ended, ended.clone(), {}

        env = EngineIO()
        adapter = MotionProductionAdapter()
        adapter.job, adapter.env, adapter.algorithm, adapter.obs = job, env, algorithm, obs
        adapter.source, adapter.restore_point_updates = source, 0
        adapter.fixed_std = algorithm.actor.log_std.detach().clone()
        adapter.spec = {
            "student_training": {"train_schedule_s": [2.0, 8.0, 8.0, 8.0], "parking_s": 4.0}
        }
        adapter.modules = dict(
            KEY="recovery_actor_valid",
            MotionStatistics=Statistics,
            StepChecks=StepChecks,
            runner_flags=runner_flags,
            bootstrap_reward=bootstrap_reward,
            observations=lambda env, value: value,
            value_diagnostics=value_diagnostics,
        )
        saved = {}
        actual_update = algorithm.update

        def capture_then_update(_saved=saved, _algorithm=algorithm, _update=actual_update):
            for key in (
                "actions",
                "actions_log_prob",
                "rewards",
                "returns",
                "advantages",
                "values",
            ):
                _saved[key] = getattr(_algorithm.storage, key).detach().clone()
            return _update()

        algorithm.update = capture_then_update
        outcome = adapter.update_once(
            NS(stop_requested=lambda: False, updates_used=prior_updates), 1
        )
        phases = ("rollout_seconds", "returns_seconds", "diagnostics_seconds", "learning_seconds")
        if profile:
            assert all(outcome[key] >= 0 for key in phases)
            assert sum(outcome[key] for key in phases) == pytest.approx(
                outcome["seconds_per_update"]
            )
        else:
            assert not any(key in outcome for key in phases)
        results.append(
            dict(
                calls=copy.deepcopy(calls),
                storage=saved,
                outcome=outcome,
                actor=copy.deepcopy(algorithm.actor.state_dict()),
                critic=copy.deepcopy(algorithm.critic.state_dict()),
                optimizer=copy.deepcopy(algorithm.optimizer.state_dict()),
                rng=torch.get_rng_state().clone(),
                scope=adapter._update_scope,
            )
        )
        assert env.ticks == job.runtime.steps_per_env
    nonreport, report = results
    assert nonreport["calls"] == dict(stat_allocations=0, stat_observations=0, value_diagnostics=0)
    assert report["calls"] == dict(
        stat_allocations=1, stat_observations=job.runtime.steps_per_env, value_diagnostics=1
    )
    assert nonreport["scope"]["stats"] is None and nonreport["scope"]["diagnostics"] == {}
    assert report["scope"]["stats"].export()["controlled"]["cohort_0"]["samples"] > 0
    assert report["scope"]["diagnostics"]["rows"][0]["samples"] > 0
    assert nonreport["scope"]["execution_stats"] is None
    execution = report["scope"]["execution_stats"].export()
    assert execution["rows"][0]["samples"] == 4 * job.runtime.steps_per_env
    assert execution["policy_steps"] == job.runtime.steps_per_env
    for key in ("storage", "actor", "critic", "optimizer", "rng"):
        equal_tree(nonreport[key], report[key])
    for key in ("reward_mean", "actor_steps", "critic_only", "value_loss", "surrogate", "entropy"):
        assert nonreport["outcome"][key] == report["outcome"][key]
