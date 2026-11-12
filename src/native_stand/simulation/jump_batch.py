"""Vector lifecycle for ABI4; partial reset never touches another plant."""

import copy

import torch

from native_stand.artifacts.io import json_digest
from native_stand.tasks.jump.adaptive_course import AdaptiveCourse


class JumpBatch:
    device = "cpu"
    infinite_horizon = False

    def __init__(self, envs, job, *, measurement=None):
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
        self.schema_digest = json_digest(self.job.assets["jump_critic_schema"])
        self.set_measurement(measurement)
        self.reset()

    def set_measurement(self, measurement):
        from native_stand.learning.ppo.performance import Measurement

        self.measurement = measurement if measurement is not None else Measurement(False, "cpu")
        for env in self.envs:
            env.measurement = self.measurement
            plant = getattr(env, "plant", None)
            if plant is not None:
                plant.measurement = plant.execution.c.measurement = self.measurement

    def invalidate_observations(self, ids=None, *, parameters_changed=False):
        self._obs = None
        for i in range(self.num_envs) if ids is None else ids:
            self.envs[int(i)].invalidate_observations(parameters_changed=parameters_changed)

    def reset(self, ids=None):
        with self.measurement.section("jump_reset", synchronize=False):
            return self._reset(ids)

    def _reset(self, ids):
        self._obs = None
        full = ids is None
        ids = range(self.num_envs) if ids is None else ids
        for i in ids:
            env = self.envs[int(i)]
            if not env.base.done:
                # Explicit reset abandons the unfinished request, never records a synthetic success.
                env.course.pending.pop(env.request["request_id"])
                env.base.done = True
            env.reset_next()
            self.measurement.count("jump_environment_resets", reason="full" if full else "partial")
        return self.observations()

    def observations(self):
        if getattr(self, "_obs", None) is not None:
            return self._obs
        obs = torch.cat([e.observations() for e in self.envs], dim=0)
        for env in self.envs if not self.schema_verified else ():
            if json_digest(env.schema.contract()) != self.schema_digest:
                raise ValueError("Jump实际6370D/235字段schema与冻结布局不同")
        self.schema_verified = True
        self._obs = obs
        return obs

    def step(self, raw):
        with self.measurement.section("jump_batch_step", synchronize=False):
            return self._step(raw)

    def _step(self, raw):
        # One host conversion per batch, rather than one detach per environment.
        if torch.is_tensor(raw):
            if raw.device.type != "cpu":
                raise ValueError("Jump MuJoCo CPU不接受CUDA动作；不是GPU后端")
            raw = raw.detach().numpy()
        rewards = []
        dones = []
        terminal = []
        durations = []
        reasons = []
        owners = []
        next_observations = []
        for i, env in enumerate(self.envs):
            result = env.step(raw[i : i + 1])
            rewards.append(result["rewards"].reshape(1))
            done = bool(result["dones"][0])
            dones.append(done)
            terminal.append(result["observations"])
            steps = result["extras"]["physical_steps"]
            durations.append(steps)
            self.measurement.count("jump_script_or_physics_ticks", steps)
            owners.append(result["extras"].get("actual_C_owner", -1))
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
                self.measurement.count("jump_environment_resets", reason="terminal")
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
            terminal_observations=torch.cat(terminal, dim=0) if any(dones) else obs,
            extras=dict(
                actual_physics_steps=durations,
                # Shared collection encoding:0 policy,2 unavailable/safety.
                control_owner=torch.tensor([0 if owner == 2 else 2 for owner in owners]),
                actual_C_owner=torch.tensor(owners),
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
