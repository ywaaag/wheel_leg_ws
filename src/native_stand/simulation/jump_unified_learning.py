"""CPU learning interface for the unified actor; common physics and course reused.

The public MuJoCo CPU lifecycle consumes this environment. Isaac remains
unimplemented. Actor37 is separate from full C phase9 / wheel4.8 Critic state.
"""

from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
import torch
from tensordict import TensorDict

from native_stand.artifacts.io import json_digest
from native_stand.simulation import jump_direct_critic as critic
from native_stand.simulation.jump_dense_env import DenseCritic, DenseJumpEnv
from native_stand.simulation.jump_direct_env import DirectEngineeringEnv
from native_stand.tasks.jump import unified
from native_stand.tasks.jump.controller_critic import (
    NormalizedCritic,
    extend_fields,
)
from native_stand.tasks.jump.dense_reward import RewardConfig, TaskSpec
from native_stand.tasks.jump.pretraining import TaskMemory
from native_stand.tasks.jump.unified_reward import (
    VERSION,
    UnifiedPacket,
    UnifiedReward,
    command_relative,
)


class UnifiedCritic(critic.FullCritic):
    def contract(self):
        return {
            **super().contract(),
            "version": "jump_unified_full_causal_CPU_critic_v1",
            "wheel_cap_Nm": 4.8,
            "phase_count": 10,
        }


class UnifiedDenseCritic(DenseCritic):
    def contract(self):
        return {
            **super().contract(),
            "version": "jump_unified_dense_causal_CPU_critic_v1",
            "wheel_cap_Nm": 4.8,
            "phase_count": 10,
            "completion": "same Jump actor; truth-gated",
        }


def pack(env, actor):
    return critic.pack(env, actor, wheel_cap=4.8, phase_count=10, schema_class=UnifiedCritic)


def packet(plant, dt_s):
    return plant.scoring_packet(dt_s, packet_type=UnifiedPacket)


class UnifiedBase(DirectEngineeringEnv):
    def __init__(self, plant):
        self.plant = plant
        self.contract = dict(
            interface=unified.contract(),
            reward=VERSION,
            support="simulation full-state inverse model; production adapter unavailable",
            backend="MuJoCo CPU",
        )
        self.interface_digest = json_digest(self.contract)
        self._obs = None
        self.prepared_tick = -1
        self.done = True
        self.schema = None
        self.total_episodes = 0

    def observations(self):
        if self._obs is not None:
            return self._obs
        p, e = self.plant, self.plant.execution
        if self.prepared_tick != e.tick:
            e.update([p.feedback()])
            self.prepared_tick = e.tick
        actor, valid = e.observe()
        values, self.schema = pack(self, actor[0])
        self._obs = TensorDict(
            dict(
                student_actor=torch.tensor(actor),
                student_critic=torch.tensor(values[None]),
                recovery_actor_valid=torch.tensor(valid[:, None]),
            ),
            batch_size=[1],
        )
        return self._obs


class UnifiedLearningEnv(DenseJumpEnv):
    def __init__(self, plant, *, course, reward_config=None):
        self.base = UnifiedBase(plant)
        self.course = course
        self.reward_config = reward_config or RewardConfig()
        self.identity = json_digest(
            dict(
                version=VERSION,
                base=self.base.interface_digest,
                cohort=[course.plant_id, course.DR_id],
                wheel_cap=4.8,
                reward_config=asdict(self.reward_config),
                course_config=asdict(course.config),
            )
        )
        self.reward = self.request = self.schema = self._obs = None
        self.episode_recorded = False

    def observations(self):
        if self._obs is not None:
            return self._obs
        previous_schema = self.schema
        self.schema = None
        obs = super().observations(schema_class=UnifiedDenseCritic)
        p = self.plant
        after = p.jump_request_after_ms
        extras = dict(
            jump_request_enabled=(np.asarray(after is not None), 1.0, "bool"),
            jump_request_relative_s=(
                np.asarray(
                    0.0
                    if after is None
                    else (p.execution.tick - p.episode_start_tick - after) * 0.001
                ),
                10.0,
                "signed seconds",
            ),
            actual_episode_elapsed_s=(
                np.asarray((p.execution.tick - p.episode_start_tick) * 0.001),
                10.0,
                "seconds",
            ),
        )
        values, fields = extend_fields({}, [], extras)
        self._request_tail_schema = NormalizedCritic.reuse(
            fields, getattr(self, "_request_tail_schema", None)
        )
        tail = self._request_tail_schema.encode(values)
        self.schema = UnifiedDenseCritic.reuse(list(self.schema.fields) + fields, previous_schema)
        obs["student_critic"] = torch.cat((obs["student_critic"], torch.tensor(tail[None])), dim=1)
        obs["jump_C_phase"] = torch.tensor([[int(p.execution.c.states[0].phase)]])
        obs["jump_C_owner"] = torch.tensor([[int(p.execution.c.states[0].owner)]])
        self._obs = obs
        return obs

    def reset_next(self):
        if not self.base.done:
            raise RuntimeError("explicit natural end/reset required")
        self.request = self.course.next_episode()
        r = self.request
        case = dict(
            command=[0, 0, r["entry_height_m"]],
            leg_length_m=r["leg_length_m"],
            drop_m=r["drop_m"],
            jump_request_after_ms=1500 if r["kind"] == "ground_active_jump" else None,
        )
        self.plant.initialize_case(case)
        self.base.spec = copy.deepcopy(r)
        self.base.memory = TaskMemory(
            r["kind"], r["seconds"]
        )  # frozen packer's explicit legacy fields
        self.base.done = False
        self.base._obs = None
        self.base.prepared_tick = -1
        self.base.total_episodes += 1
        spec = TaskSpec(
            kind=r["kind"],
            horizon_s=r["seconds"],
            mass_kg=self.plant.mass,
            target_rise_m=r["target_rise_m"],
            target_clearance_m=r["target_clearance_m"],
            air_leg_m=r["air_leg_m"],
            entry_height_m=r["entry_height_m"],
        )
        self.reward = UnifiedReward.start(spec, packet(self.plant, 0.0), self.reward_config)
        self._obs = None
        self.episode_recorded = False
        self.balance_seconds = self.balance_good_seconds = 0.0
        return self.observations()

    def step(self, raw):
        if self.base.done:
            raise RuntimeError("episode reset required")
        p, e = self.plant, self.plant.execution
        obs = self.observations()
        if (e.publish_raw(raw, obs["recovery_actor_valid"][:, 0].numpy()) < 0).any():
            raise RuntimeError("C raw publication rejected")
        from dataclasses import replace

        outcomes, samples, failure = [], [], None
        for substep in range(10):
            before = p.physics_steps
            try:
                if substep:
                    e.update([p.feedback()])
                p.integrate_control()
                if p.solved_restricted:
                    raise RuntimeError("restricted non-wheel contact")
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                failure = str(exc)
            advanced = p.physics_steps - before
            if advanced or failure:
                raw_packet = packet(p, 0.001 if advanced else 0.0)
                forward = p.data.xmat[p.root].reshape(3, 3) @ np.array([0.0, 0.0, -1.0])
                state = e.c.states[0]
                sample = command_relative(
                    raw_packet,
                    p.command[0] * forward[:2],
                    own_completed=state.phase == unified.BALANCE and bool(state.success),
                )
                if failure:
                    sample = replace(sample, fault=True)
                samples.append(sample)
                outcomes.append(self.reward.step(sample))
                if sample.phase == unified.BALANCE and sample.owner == 2:
                    cfg = self.reward_config
                    self.balance_seconds += sample.dt_s
                    good = (
                        all(sample.contact)
                        and abs(sample.root_height_m - self.reward.spec.entry_height_m)
                        <= cfg.stable_height_error_m
                        and max(map(abs, sample.tilt_rad)) <= cfg.stable_tilt_rad
                        and np.linalg.norm(sample.COM_velocity_m_s[:2]) <= cfg.stable_vxy_m_s
                        and np.linalg.norm(sample.omega_rad_s) <= cfg.stable_omega_rad_s
                    )
                    self.balance_good_seconds += sample.dt_s * bool(good)
            if failure or outcomes and outcomes[-1]["terminated"]:
                break
        names = set().union(*(v["terms"] for v in outcomes))
        terms = {k: sum(v["terms"].get(k, 0.0) for v in outcomes) for k in names}
        result = {**outcomes[-1], "terms": terms, "total": float(sum(terms.values()))}
        self.base.done = bool(result["terminated"])
        self.base._obs = self._obs = None
        if self.base.done and not self.episode_recorded:
            self.course.record(
                self.request,
                jump_control_s=self.reward.jump_control_s,
                success=result["active_jump_success"] or result["landing_only_success"],
                fault=self.reward.failure_paid,
            )
            self.episode_recorded = True
        out = self.observations()
        return dict(
            observations=out,
            rewards=torch.tensor([result["total"]]),
            dones=torch.tensor([self.base.done]),
            terminated=torch.tensor([self.base.done]),
            truncated=torch.tensor([False]),
            terminal_critic=out["student_critic"].clone(),
            extras=dict(
                reward=result,
                failure=failure,
                physical_steps=sum(v.dt_s > 0 for v in samples),
                actual_C_owner=int(e.c.states[0].owner),
            ),
        )

    def runtime_state(self):
        p = self.plant
        return copy.deepcopy(
            dict(
                version=VERSION,
                identity=self.identity,
                base=self.base.runtime_state(),
                reward=self.reward.state_dict(),
                course=self.course.state_dict(),
                request=self.request,
                episode_recorded=self.episode_recorded,
                balance_seconds=self.balance_seconds,
                balance_good_seconds=self.balance_good_seconds,
                force=p.scoring_force_N,
                slip=p.scoring_slip_m_s,
                force_valid=p.scoring_force_valid,
                actual_command=p.command,
                jump_request_after_ms=p.jump_request_after_ms,
            )
        )

    def restore_runtime_state(self, value):
        if value["version"] != VERSION or value["identity"] != self.identity:
            raise ValueError("unified learning full-state identity")
        reward = UnifiedReward.restore(value["reward"])
        course = type(self.course).restore(value["course"])
        request = value["request"]
        if (
            reward.config != self.reward_config
            or course.config != self.course.config
            or (course.plant_id, course.DR_id) != (self.course.plant_id, self.course.DR_id)
        ):
            raise ValueError("Jump恢复奖励或课程配置不同")
        expected = TaskSpec(
            kind=request["kind"],
            horizon_s=request["seconds"],
            mass_kg=self.plant.mass,
            target_rise_m=request["target_rise_m"],
            target_clearance_m=request["target_clearance_m"],
            air_leg_m=request["air_leg_m"],
            entry_height_m=request["entry_height_m"],
        )
        if (
            reward.spec != expected
            or not value["episode_recorded"]
            and course.pending.get(request["request_id"]) != request
        ):
            raise ValueError("Jump恢复目标/在途请求身份冲突")
        self.base.restore_runtime_state(value["base"])
        self.reward, self.course = reward, course
        self.request = copy.deepcopy(value["request"])
        self.episode_recorded = value["episode_recorded"]
        self.balance_seconds = value["balance_seconds"]
        self.balance_good_seconds = value["balance_good_seconds"]
        p = self.plant
        p.scoring_force_N = value["force"].copy()
        p.scoring_slip_m_s = value["slip"].copy()
        p.scoring_force_valid = value["force_valid"]
        p.command = value["actual_command"].copy()
        p.jump_request_after_ms = value["jump_request_after_ms"]
        self.schema = None
        self._obs = None
