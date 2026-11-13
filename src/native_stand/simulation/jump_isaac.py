"""ABI4 PhysX consumer with the common owner/driver/PPO and existing task math.

The first version deliberately uses an explicit CPU C/inverse-model bridge.
It measures its synchronization cost and never claims that C is a CUDA kernel.
"""

from __future__ import annotations

import atexit
import copy
import ctypes as C
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from native_stand.artifacts.io import json_digest, read_json, write_json
from native_stand.learning.ppo.performance import Measurement
from native_stand.simulation.jump_build import build
from native_stand.simulation.jump_isaac_bridge import InverseFeedback, critic_execution
from native_stand.simulation.jump_isaac_critic import RowCritic
from native_stand.simulation.jump_response import Response
from native_stand.simulation.jump_unified import UnifiedV4Execution
from native_stand.tasks.jump.adaptive_course import AdaptiveCourse
from native_stand.tasks.jump.dense_reward import RewardConfig, TaskSpec
from native_stand.tasks.jump.geometry import CHANNELS
from native_stand.tasks.jump.isaac_contract import CRITIC, TRAINING
from native_stand.tasks.jump.unified import BALANCE
from native_stand.tasks.jump.unified_reward import VERSION as REWARD_VERSION
from native_stand.tasks.jump.unified_reward import UnifiedPacket, UnifiedReward

TRAINING_SEMANTICS = dict(
    reward=REWARD_VERSION,
    request="single_request_first_BALANCE_at_or_after_1500ms",
)


def scheduled_request(request, elapsed_ms, state):
    """Delay a single episode request while busy; C retains all entry gates.

    RLJumpInit on physical reset clears request_id. Once C consumes the
    rising edge, its request_id prevents retries, including rejected requests.
    """
    return (
        request["kind"] == "ground_active_jump"
        and elapsed_ms >= 1500
        and state.phase == BALANCE
        and state.request_id == 0
    )


def validate_job(job):
    from native_stand.tasks.jump.spec import build_task

    build_task(job)
    if job.reward.version != REWARD_VERSION:
        raise ValueError("Explicit current Jump reward version required")
    if job.backend != "isaac" or job.action.version != TRAINING or job.runtime.device != "cuda:0":
        raise ValueError("Explicit Jump Isaac ABI4/CUDA0 contract required")
    if (job.control.physics_dt_s, job.control.pd_dt_s, job.control.policy_dt_s) != (
        0.001,
        0.005,
        0.01,
    ):
        raise ValueError("Jump1/5/10ms")
    if job.model.critic_privileged_version != CRITIC:
        raise ValueError("Jump Isaac Critic identity")
    if job.assets.get("jump_restore_mode") != "fresh_reset":
        raise ValueError(
            "PhysX full hidden solver restoration is unavailable; fresh_reset required"
        )
    if not job.assets.get("jump_isaac", {}).get("gpu_uuid"):
        raise ValueError("Explicit physical GPU UUID required")
    if job.limits.smoke_updates:
        raise ValueError(
            "Jump Isaac uses one formal stage; bound initial updates with --max-updates"
        )
    policy = job.assets["jump_isaac"].get("resource_policy", {})
    if policy.get("bounded_physical_io_only") is True and (
        job.runtime.envs > 2 or job.limits.formal_updates or job.limits.sample_limit is not None
    ):
        raise ValueError("Bounded shared-GPU diagnosis cannot carry PPO work")


class IsaacJump:
    device = "cuda:0"
    infinite_horizon = False

    def __init__(self, job, *, measurement=None, output=None):
        validate_job(job)
        self.job = job
        self.num_envs = job.runtime.envs
        self.measurement = measurement or Measurement(False, "cpu")
        if output is None:
            durable = Path(job.assets["jump_isaac"]["execution_root"])
            durable.mkdir(parents=True, exist_ok=True)
            output = tempfile.mkdtemp(prefix=job.name + "-", dir=durable)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.scene = self.app = None
        self._app_exit_code = 0
        self._obs = None
        self.completed = []
        self.serial = 0
        from native_stand.simulation.jump_isaac_scene import PHYSICAL_CONTRACT

        self.physical_contract = copy.deepcopy(PHYSICAL_CONTRACT)
        self.identity = json_digest(
            dict(
                contract=job.assets["jump_contract_data"],
                schema=job.assets["jump_critic_schema"],
                course=job.assets["jump_course"],
                backend=TRAINING,
                physics=self.physical_contract,
                training=TRAINING_SEMANTICS,
                envs=self.num_envs,
            )
        )
        try:
            self._build()
        except BaseException as exc:
            self._app_exit_code = 1
            import traceback

            write_json(
                self.output / "construction_failure.json",
                dict(error=repr(exc), traceback=traceback.format_exc()),
            )
            try:
                self.close()
            except BaseException as cleanup:
                exc.add_note("Jump construction cleanup: " + repr(cleanup))
            raise

    def _build(self):
        from native_stand.simulation.recovery_isaac.backend import require_idle_gpu
        from native_stand.simulation.recovery_isaac.runtime_safety import preserve_worker_signals

        resources = self.job.assets["jump_isaac"]
        readback = require_idle_gpu(
            resources["gpu_uuid"],
            resources["resource_policy"],
            allow_bounded_diagnostic_owner=True,
            shared_training_task=resources["resource_policy"].get("shared_training_task"),
        )
        # Runtime imports below occur only after the existing AppLauncher.
        from isaaclab.app import AppLauncher

        with preserve_worker_signals():
            self.app = AppLauncher(
                {
                    "headless": True,
                    "device": self.device,
                    "multi_gpu": False,
                    "fast_shutdown": True,
                }
            ).app
        # Kit6.2 fast shutdown calls os._exit. Defer it until the common worker
        # has finalized its budget/report; never terminate inside adapter.close.
        atexit.register(self._exit_app)
        from native_stand.simulation.jump_isaac_scene import IsaacScene

        refs = self.job.assets["jump_bindings"]
        inputs = {
            name: read_json(refs[name]["path"])
            for name in ("description", "spring_geometry", "mechanics", "V4")
        }
        self.description = inputs["description"]
        self.springs = inputs["spring_geometry"]["springs"]
        self.mechanics = inputs["mechanics"]["mechanics"]
        V4 = inputs["V4"]
        with self.measurement.section("jump_C_build", synchronize=False):
            self.binding = build(self.output / "jump_abi4.so", measurement=self.measurement)
        self.inverse = InverseFeedback(self.description, self.springs, self.mechanics, V4)
        self.scene = IsaacScene(
            self.job, self.description, self.springs, self.mechanics, V4, self.inverse.initial
        )
        self._root_mass_COM = (
            torch.cat(
                (
                    self.scene.physics.mass[:, self.scene.root : self.scene.root + 1],
                    self.scene.physics.com[:, self.scene.root, :3],
                ),
                -1,
            )
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        material = self.scene.physics.material.detach().cpu().numpy()
        self._material_bounds = np.stack((material.min(1), material.max(1)), 1)
        self.execution = UnifiedV4Execution(
            self.binding["path"],
            V4,
            Response,
            self.num_envs,
            device=self.device,
            library_sha=self.binding["sha256"],
            row_faults=True,
        )
        self.execution.c.measurement = self.measurement
        cfg = self.execution.c.config
        cfg.enabled = cfg.model_available = 1
        cfg.model_id = 0x53494D34
        cfg.model_input_id = self.execution.c.INPUT_CONTRACT
        cfg.model_output_id = self.execution.c.OUTPUT_CONTRACT
        self.courses = [
            AdaptiveCourse(seed=self.job.runtime.seed + i, **self.job.assets["jump_course"])
            for i in range(self.num_envs)
        ]
        self.envs = [
            SimpleNamespace(plant=SimpleNamespace(binding=self.binding), course=c)
            for c in self.courses
        ]
        self.requests = [None] * self.num_envs
        self.rewards = [None] * self.num_envs
        self.critics = [RowCritic(len(self.mechanics["joint_names"])) for _ in self.courses]
        self.starts = np.zeros(self.num_envs, dtype=np.int64)
        self.balance_s = np.zeros(self.num_envs)
        self.balance_good = np.zeros(self.num_envs)
        self.motor = torch.zeros(self.num_envs, 6, device=self.device)
        self.NET = torch.zeros_like(self.motor)
        self.passive = torch.zeros_like(self.motor)
        self.prepared_tick = -1
        self.reset()
        write_json(
            self.output / "construction.json",
            dict(
                status="CONSTRUCTED",
                resource_readback=readback,
                backend="Isaac PhysX GPU",
                C_execution="batched CPU C bridge",
                support="PhysX state->CPU inverse model; simulation hypothesis, no contact observation",
                actual_runtime_schema=self.critics[0].schema.contract(),
                physics_restore="fresh_reset",
                startup_engine_ticks=self.scene.actual_ticks,
                physical_contract=self.physical_contract,
            ),
        )

    def set_measurement(self, measurement):
        self.measurement = measurement
        if hasattr(self, "execution"):
            self.execution.c.measurement = measurement

    def host(self):
        key = (self.scene.actual_ticks, self.scene.reset_epoch)
        if getattr(self, "_host_key", None) == key:
            return self._host_cache
        with self.measurement.section("jump_PhysX_readback", synchronize=False):
            measured = self.scene.measured()
            measured = {**measured, "motor": self.motor, "NET": self.NET, "passive": self.passive}
            widths = [v[0].numel() for v in measured.values()]
            packed = (
                torch.cat([v.reshape(self.num_envs, -1).float() for v in measured.values()], -1)
                .detach()
                .cpu()
                .numpy()
            )
            self.measurement.count("jump_host_batch_transfers")
            host = {}
            start = 0
            for (name, value), width in zip(measured.items(), widths, strict=True):
                row = packed[:, start : start + width].reshape(tuple(value.shape))
                host[name] = row.astype(bool) if value.dtype == torch.bool else row
                start += width
        from native_stand.simulation.jump_isaac_scene import validate_physical_measurement

        validate_physical_measurement(host)
        self._host_key, self._host_cache = key, host
        return host

    def prepare(self, host, *, reset_ids=None):
        e = self.execution
        if self.prepared_tick == e.tick and reset_ids is None:
            return
        if reset_ids is None and e.tick % 5:
            self.prepared_tick = e.tick
            return
        indices = range(self.num_envs) if reset_ids is None else reset_ids
        with self.measurement.section("jump_inverse_C_feedback", synchronize=False):
            rows = []
            for i in indices:
                request = self.requests[i]
                elapsed = e.tick - self.starts[i]
                key = scheduled_request(request, elapsed, e.c.states[i])
                rows.append(
                    self.inverse.input(
                        tick=e.tick,
                        state=e.c.states[i],
                        root=host["root"][i],
                        joint=host["joint"][i],
                        acceleration=host["acceleration"][i],
                        motor=host["motor"][i],
                        NET=host["NET"][i],
                        command=[0.0, 0.0, 0.34],
                        request=key,
                    )
                )
            if reset_ids is None:
                e.update(rows)
            else:
                for i, row in zip(indices, rows, strict=True):
                    e.input_rows[i] = row
                    e.c.inputs[i] = row
                    if e.tick % 5 == 0:
                        e.c.lib.RLJumpStep(
                            C.byref(e.c.states[i]), C.byref(e.c.inputs[i]), C.byref(e.c.config)
                        )
                        self.measurement.count("jump_C_FSM_calls", reason="partial_reset_row")
        self.prepared_tick = e.tick

    def packet(self, i, h, dt, *, fault=False):
        state = self.execution.c.states[i]
        return UnifiedPacket(
            dt_s=dt,
            phase=int(state.phase),
            owner=int(state.owner),
            COM_height_m=float(h["COM"][i, 2]),
            COM_velocity_m_s=tuple(h["COMv"][i]),
            root_height_m=float(h["root"][i, 2]),
            tilt_rad=tuple(h["tilt"][i]),
            omega_rad_s=tuple(h["omega"][i]),
            leg_length_m=tuple(h["length"][i]),
            leg_swing_rad=tuple(h["swing"][i]),
            spring_length_m=tuple(h["spring"][i]),
            contact=tuple(h["contact"][i]),
            tire_clearance_m=tuple(h["clearance"][i]),
            wheel_normal_force_N=tuple(h["normal"][i]),
            contact_slip_m_s=tuple(h["slip"][i]),
            actual_motor_Nm=tuple(h["motor"][i]),
            actual_dq_rad_s=tuple(
                h["joint"][i, 1, [self.mechanics["joint_names"].index(n) for n in CHANNELS]]
            ),
            adopted_target=tuple(state.adopted),
            force_valid=True,
            motion_confirmed=bool(state.phase == BALANCE and state.success),
            fault=bool(fault or state.fault or h["restricted"][i] or h["closure"][i].max() > 0.005),
        )

    def physical(self, i, h):
        r = self.requests[i]
        mass_COM = self._root_mass_COM[i]
        elapsed = (self.execution.tick - self.starts[i]) * 0.001
        return dict(
            root_relative_state=h["root"][i],
            all_joint_q_dq_ddq=h["joint"][i],
            COM_height_velocity=np.r_[h["COM"][i, 2], h["COMv"][i]],
            leg_length_swing_spring=np.r_[h["length"][i], h["swing"][i], h["spring"][i]],
            closure_residuals=h["closure"][i],
            wheel_normal_slip_clearance=np.r_[h["normal"][i], h["slip"][i], h["clearance"][i]],
            motor_NET_passive=np.stack((h["motor"][i], h["NET"][i], h["passive"][i])),
            root_mass_COM=mass_COM,
            material_min_max=self._material_bounds[i],
            elapsed_request=np.array(
                [
                    elapsed,
                    1.5 if r["kind"] == "ground_active_jump" else 0.0,
                    r["kind"] == "ground_active_jump",
                ]
            ),
        )

    def row_observation(self, i, h, actor, *, execution=None):
        return self.critics[i].pack(
            self.execution if execution is None else execution,
            i,
            actor,
            self.physical(i, h),
            self.rewards[i],
            self.requests[i],
        )

    def critic_snapshot(self):
        with self.measurement.section("jump_critic_readback", synchronize=False):
            result = critic_execution(self.execution)
            self.measurement.count("jump_critic_batch_transfers", result.transfer_batches)
            return result

    def observations(self):
        if self._obs is not None:
            return self._obs
        with self.measurement.section("jump_observation_pack", synchronize=False):
            h = self.host()
            self.prepare(h)
            actor, valid = self.execution.observe()
            execution = self.critic_snapshot()
            critic = np.stack(
                [
                    self.row_observation(i, h, actor[i], execution=execution)
                    for i in range(self.num_envs)
                ]
            )
            schema = self.critics[0].schema.contract()
            if json_digest(schema) != json_digest(self.job.assets["jump_critic_schema"]):
                raise ValueError("Actual Isaac Critic schema differs from frozen job")
            self._obs = TensorDict(
                dict(
                    student_actor=torch.from_numpy(actor).to(self.device),
                    student_critic=torch.from_numpy(critic).to(self.device),
                    recovery_actor_valid=torch.from_numpy(valid[:, None]).to(self.device),
                    jump_C_phase=torch.tensor(
                        [[s.phase] for s in self.execution.c.states], device=self.device
                    ),
                    jump_C_owner=torch.tensor(
                        [[s.owner] for s in self.execution.c.states], device=self.device
                    ),
                ),
                batch_size=[self.num_envs],
            )
            return self._obs

    def reset(self, ids=None):
        previous = self._obs
        partial = ids is not None and len(ids) < self.num_envs
        prepared = self.prepared_tick == self.execution.tick
        ids = list(range(self.num_envs)) if ids is None else [int(i) for i in ids]
        with self.measurement.section("jump_reset", synchronize=False):
            for i in ids:
                if self.requests[i] is not None:
                    self.courses[i].pending.pop(self.requests[i]["request_id"], None)
                self.requests[i] = self.courses[i].next_episode()
            self.scene.reset(ids, [self.requests[i] for i in ids], self.inverse.initial)
            indices = torch.tensor(ids, device=self.device)
            q = self.scene.robot.data.joint_pos.torch[:, self.scene.ids][indices]
            dq = self.scene.robot.data.joint_vel.torch[:, self.scene.ids][indices]
            self.execution.reset(indices, q, dq)
            for i in ids:
                state = self.execution.c.states[i]
                state.adopted_valid = 1
                state.adopted_sequence = 1
                state.adopted_tick = state.phase_tick = self.execution.tick
                state.adopted[:] = self.execution.v4.adopted[i].detach().cpu().numpy()
                state.first_hold = 1
                self.starts[i] = self.execution.tick
                self.balance_s[i] = self.balance_good[i] = 0
                self.motor[i] = 0
                self.NET[i] = 0
                self.passive[i] = 0
            self.prepared_tick = -1
            self._obs = None
            h = self.host()
            if partial and prepared:
                self.prepare(h, reset_ids=ids)
            for i in ids:
                r = self.requests[i]
                spec = TaskSpec(
                    kind=r["kind"],
                    horizon_s=r["seconds"],
                    mass_kg=float(self.scene.physics.mass[i].sum()),
                    target_rise_m=r["target_rise_m"],
                    target_clearance_m=r["target_clearance_m"],
                    air_leg_m=r["air_leg_m"],
                    entry_height_m=r["entry_height_m"],
                )
                self.rewards[i] = UnifiedReward.start(spec, self.packet(i, h, 0.0))
                self.measurement.count("jump_environment_resets")
        result = self.observations()
        if partial and previous is not None:
            untouched = [i for i in range(self.num_envs) if i not in ids]
            for name in list(result.keys()):
                result[name][untouched] = previous[name][untouched]
        return result

    def step(self, raw):
        start = self.observations()
        if (
            self.execution.publish_raw(
                raw, start["recovery_actor_valid"][:, 0].detach().cpu().numpy()
            )
            < 0
        ).any():
            raise RuntimeError("Actual C rejected raw publication")
        done = np.zeros(self.num_envs, dtype=bool)
        valid_steps = np.zeros(self.num_envs, dtype=np.int32)
        reward = np.zeros(self.num_envs)
        terminal = {}
        outcomes = [None] * self.num_envs
        for _ in range(10):
            h = self.host()
            self.prepare(h)
            q = self.scene.robot.data.joint_pos.torch[:, self.scene.ids]
            dq = self.scene.robot.data.joint_vel.torch[:, self.scene.ids]
            motor, net = self.execution.step(q, dq)
            self.motor.copy_(motor)
            self.NET.copy_(net)
            with self.measurement.section("jump_PhysX_step", synchronize=False):
                self.passive.copy_(self.scene.apply_effort(motor, net))
            self.measurement.count("jump_actual_engine_ticks", self.num_envs)
            h = self.host()
            ended = []
            for i in range(self.num_envs):
                if done[i]:
                    continue
                p = self.packet(i, h, 0.001)
                result = self.rewards[i].step(p)
                outcomes[i] = result
                reward[i] += result["total"]
                valid_steps[i] += 1
                self.measurement.count("jump_reward_packets")
                if p.phase == BALANCE and p.owner == 2:
                    self.balance_s[i] += 0.001
                    cfg = RewardConfig()
                    good = (
                        all(p.contact)
                        and abs(p.root_height_m - 0.34) <= cfg.stable_height_error_m
                        and max(map(abs, p.tilt_rad)) <= cfg.stable_tilt_rad
                        and np.linalg.norm(p.COM_velocity_m_s[:2]) <= cfg.stable_vxy_m_s
                        and np.linalg.norm(p.omega_rad_s) <= cfg.stable_omega_rad_s
                    )
                    self.balance_good[i] += 0.001 * bool(good)
                if result["terminated"]:
                    done[i] = True
                    ended.append(i)
            if ended:
                actor, _ = self.execution.c.observe()
                execution = self.critic_snapshot()
                for i in ended:
                    terminal[i] = dict(
                        critic=self.row_observation(i, h, actor[i], execution=execution).copy(),
                        actor=actor[i].copy(),
                        phase=int(self.execution.c.states[i].phase),
                        owner=int(self.execution.c.states[i].owner),
                        fault=int(self.execution.c.states[i].fault),
                        physical=dict(
                            root_height_m=float(h["root"][i, 2]),
                            spring_m=h["spring"][i].tolist(),
                            leg_m=h["length"][i].tolist(),
                            closure_m=h["closure"][i].tolist(),
                            restricted=bool(h["restricted"][i]),
                            wheel_normal_N=h["normal"][i].tolist(),
                        ),
                    )
        self._obs = None
        terminal_next = self.observations()
        phases = terminal_next["jump_C_phase"].clone()
        owners = terminal_next["jump_C_owner"].clone()
        term_actor = terminal_next["student_actor"].clone()
        term_critic = terminal_next["student_critic"].clone()
        for i in np.flatnonzero(done):
            result = outcomes[i]
            r = self.requests[i]
            rew = self.rewards[i]
            success = bool(
                result["active_jump_success"]
                if r["kind"] == "ground_active_jump"
                else result["landing_only_success"]
            )
            self.courses[i].record(
                r, jump_control_s=rew.jump_control_s, success=success, fault=rew.failure_paid
            )
            self.serial += 1
            self.completed.append(
                dict(
                    id=self.serial,
                    request=copy.deepcopy(r),
                    success=success,
                    fault=bool(rew.failure_paid),
                    jump_control_s=rew.jump_control_s,
                    reason="task_failure"
                    if rew.failure_paid
                    else "task_timeout"
                    if rew.timeout_paid
                    else "task_end",
                    failure=(
                        "C_fault="
                        + str(terminal[i]["fault"])
                        + "; physical="
                        + repr(terminal[i]["physical"])
                    )
                    if rew.failure_paid
                    else None,
                    balance_seconds=float(self.balance_s[i]),
                    balance_good_seconds=float(self.balance_good[i]),
                    C_requests=int(self.execution.c.states[i].request_id),
                    C_accepted_jumps=int(self.execution.c.states[i].epoch),
                    C_final_phase=int(terminal[i]["phase"]),
                )
            )
            phases[i] = terminal[i]["phase"]
            owners[i] = terminal[i]["owner"]
            term_actor[i] = torch.from_numpy(terminal[i]["actor"]).to(self.device)
            term_critic[i] = torch.from_numpy(terminal[i]["critic"]).to(self.device)
        terminal_obs = TensorDict(
            dict(
                student_actor=term_actor,
                student_critic=term_critic,
                jump_C_phase=phases,
                jump_C_owner=owners,
                recovery_actor_valid=terminal_next["recovery_actor_valid"].clone(),
            ),
            batch_size=[self.num_envs],
        )
        if done.any():
            self.reset(np.flatnonzero(done))
        return dict(
            observations=self.observations(),
            rewards=torch.tensor(reward, dtype=torch.float32, device=self.device),
            dones=torch.tensor(done, device=self.device),
            terminated=torch.tensor(done, device=self.device),
            truncated=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            terminal_observations=terminal_obs,
            extras=dict(
                actual_physics_steps=[10] * self.num_envs,
                episode_physics_steps=valid_steps.tolist(),
                transition_valid=torch.tensor(valid_steps == 10, device=self.device),
                control_owner=torch.where(start["recovery_actor_valid"][:, 0], 0, 2),
                actual_C_owner=owners[:, 0],
                collection_conditions={},
            ),
        )

    def runtime_state(self):
        return dict(
            version=TRAINING,
            identity=self.identity,
            physical_contract=self.physical_contract,
            training_semantics=copy.deepcopy(TRAINING_SEMANTICS),
            restore="fresh_reset",
            execution=self.execution.state_dict(),
            courses=[c.state_dict() for c in self.courses],
            completed=copy.deepcopy(self.completed),
            serial=self.serial,
            visible_physics={k: v.copy() for k, v in self.host().items()},
            actual_engine_ticks=self.scene.actual_ticks,
            physical_hidden_solver_restored=False,
        )

    def load_runtime_state(self, value):
        if value["version"] != TRAINING or value["identity"] != self.identity:
            raise ValueError("Jump Isaac restore identity")
        self.courses = [AdaptiveCourse.restore(v) for v in value["courses"]]
        for c in self.courses:
            c.pending.clear()
        for i, c in enumerate(self.courses):
            self.envs[i].course = c
        self.requests = [None] * self.num_envs
        self.completed = copy.deepcopy(value["completed"])
        self.serial = value["serial"]
        self.reset()

    def initialize_from_training_state(self, value):
        if value["identity"] != self.identity:
            transition = self.job.assets.get("jump_physics_transition", {})
            source_envs = len(value["courses"])
            expansion = transition.get("environment_expansion")
            count_matches = source_envs == self.num_envs
            if not count_matches:
                count_matches = (
                    0 < source_envs < self.num_envs
                    and isinstance(expansion, dict)
                    and expansion
                    == dict(
                        source_envs=source_envs,
                        target_envs=self.num_envs,
                        existing_rows="preserve_course_rng_and_counters",
                        added_rows="job_seed_plus_row_index",
                        seed=self.job.runtime.seed,
                    )
                )
            if (
                self.job.initialization.mode != "full"
                or transition.get("source_checkpoint_sha256") != self.job.initialization.sha256
                or not self.job.initialization.sha256
                or transition.get("source_execution_identity") != value["identity"]
                or transition.get("source_physical_contract") != value.get("physical_contract")
                or transition.get("target_physical_contract") != self.physical_contract
                or transition.get("course_evidence")
                != "fresh_physics_cohort_preserve_rng_and_counters"
                or value["version"] != TRAINING
                or value["restore"] != "fresh_reset"
                or not count_matches
                or any(c["level"] != 0 for c in value["courses"])
            ):
                raise ValueError("Explicit full-state Jump physics transition required")
            # A new run may preserve ALL learning state while changing the plant.
            # The old physical world and incomplete episodes are never resumed.
            # Do not pass a forged target identity into same-run restoration.
            write_json(
                self.output / "physics_transition.json",
                dict(**transition, target_execution_identity=self.identity, restore="fresh_reset"),
            )
            for course, prior in zip(self.courses[:source_envs], value["courses"], strict=True):
                # Old failed-plant windows cannot certify the corrected plant.
                # Preserve scheduling RNG and cumulative episode identities,
                # while starting explicit new-plant evidence at level zero.
                course.rng.bit_generator.state = copy.deepcopy(prior["rng"])
                course.serial = prior["serial"]
                course.episodes = prior["episodes"]
                course.generation = prior["generation"] + 1
                course.pending.clear()
            for course in self.courses[source_envs:]:
                # Constructor seeded these independent rows with job.seed+i.
                # They have no invented source episode history or success.
                course.generation = max(c["generation"] for c in value["courses"]) + 1
            self.requests = [None] * self.num_envs
            self.reset()
        else:
            self.load_runtime_state(value)
        self.completed = []
        self.serial = 0

    def close(self):
        if sys.exc_info()[0] is not None:
            self._app_exit_code = 1
        if self.scene is not None:
            self.scene.close()
            self.scene = None

    def _exit_app(self):
        if self.app is not None:
            self.app.close(exit_code=self._app_exit_code)
            self.app = None


def make_env(job, *, measurement=None):
    return IsaacJump(job, measurement=measurement)
