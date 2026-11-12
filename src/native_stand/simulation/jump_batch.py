"""Vector lifecycle for ABI4; partial reset never touches another plant."""

import copy

import torch

from native_stand.artifacts.io import json_digest
from native_stand.tasks.jump.adaptive_course import AdaptiveCourse


class JumpBatch:
    device = "cpu"
    infinite_horizon = False

    def __init__(self, envs, job):
        self.envs = envs
        self.job = job
        self.num_envs = len(envs)
        self.completed = []
        self.serial = 0
        self.identity = json_digest(
            dict(
                contract=job.assets["jump_contract_data"],
                schema=job.assets["jump_critic_schema"],
                bindings=job.assets["jump_bindings"],
                envs=self.num_envs,
            )
        )
        self.schema_verified = False
        self.reset()

    def reset(self, ids=None):
        self._obs = None
        ids = range(self.num_envs) if ids is None else ids
        for i in ids:
            env = self.envs[int(i)]
            if not env.base.done:
                # Explicit reset abandons the unfinished request, never records a synthetic success.
                env.course.pending.pop(env.request["request_id"])
                env.base.done = True
            env.reset_next()
        return self.observations()

    def observations(self):
        if getattr(self, "_obs", None) is not None:
            return self._obs
        obs = torch.cat([e.observations() for e in self.envs], dim=0)
        for env in self.envs if not self.schema_verified else ():
            if json_digest(env.schema.contract()) != json_digest(
                self.job.assets["jump_critic_schema"]
            ):
                raise ValueError("Jump实际6370D/235字段schema与冻结布局不同")
        self.schema_verified = True
        self._obs = obs
        return obs

    def step(self, raw):
        rewards = []
        dones = []
        terminal = []
        durations = []
        reasons = []
        next_observations = []
        for i, env in enumerate(self.envs):
            result = env.step(raw[i : i + 1])
            rewards.append(result["rewards"].reshape(1))
            done = bool(result["dones"][0])
            dones.append(done)
            terminal.append(result["terminal_critic"])
            steps = result["extras"]["physical_steps"]
            durations.append(steps)
            if env.reward.timeout_paid:
                reason = "task_timeout"
            elif env.reward.failure_paid:
                failure = result["extras"]["failure"] or ""
                physical = any(
                    name in failure
                    for name in (
                        "Jump C safety owner",
                        "restricted non-wheel",
                        "spring stroke",
                        "closure failure",
                    )
                )
                reason = "task_failure" if physical or not failure else "engineering_fault"
            else:
                reason = "task_end" if done else None
            reasons.append(reason)
            if done:
                request = copy.deepcopy(env.request)
                self.serial += 1
                self.completed.append(
                    dict(
                        id=self.serial,
                        request=request,
                        success=bool(
                            result["extras"]["reward"]["active_jump_success"]
                            if request["kind"] == "ground_active_jump"
                            else result["extras"]["reward"]["landing_only_success"]
                        ),
                        fault=bool(env.reward.failure_paid),
                        jump_control_s=env.reward.jump_control_s,
                        reason=reason,
                        failure=result["extras"]["failure"],
                        balance_seconds=env.balance_seconds,
                        balance_good_seconds=env.balance_good_seconds,
                    )
                )
                next_observations.append(env.reset_next())
            else:
                next_observations.append(result["observations"])
        self._obs = torch.cat(next_observations, dim=0)
        obs = self._obs
        return dict(
            observations=obs,
            rewards=torch.cat(rewards),
            dones=torch.tensor(dones),
            terminated=torch.tensor(dones),
            truncated=torch.zeros(self.num_envs, dtype=torch.bool),
            terminal_observations={"student_critic": torch.cat(terminal)},
            extras=dict(
                actual_physics_steps=durations,
                control_owner=torch.zeros(self.num_envs, dtype=torch.long),
                collection_conditions={},
                transition_valid=torch.tensor([n == 10 for n in durations]),
                termination_reasons=reasons,
            ),
        )

    def runtime_state(self):
        return copy.deepcopy(
            dict(
                version="jump_ABI4_batch_v1",
                identity=self.identity,
                environments=[e.runtime_state() for e in self.envs],
                completed=self.completed,
                serial=self.serial,
            )
        )

    def load_runtime_state(self, state):
        if state["version"] != "jump_ABI4_batch_v1" or state["identity"] != self.identity:
            raise ValueError("Jump后端/模型/合同/schema完整恢复身份不同")
        if self.job.assets.get("jump_restore_mode", "exact") == "exact":
            for env, value in zip(self.envs, state["environments"], strict=True):
                env.restore_runtime_state(value)
        elif self.job.assets["jump_restore_mode"] == "fresh_reset":
            for env, value in zip(self.envs, state["environments"], strict=True):
                env.course = AdaptiveCourse.restore(value["course"])
                env.course.pending.clear()  # unfinished dwell/request is abandoned, completed evidence retained
                env.base.done = True
                env.reset_next()
        else:
            raise ValueError("未知Jump恢复模式")
        self._obs = None
        self.completed = copy.deepcopy(state["completed"])
        self.serial = state["serial"]

    def initialize_from_training_state(self, state):
        if state["identity"] != self.identity:
            raise ValueError("Jump新run full继承执行合同不同")
        for env, value in zip(self.envs, state["environments"], strict=True):
            env.course = AdaptiveCourse.restore(value["course"])
            env.course.pending.clear()
            env.base.done = True
            env.reset_next()
        self.completed = []
        self.serial = 0
        self._obs = None

    def close(self):
        for env in self.envs:
            env.close()
