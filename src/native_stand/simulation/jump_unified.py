"""Actual ABI4 C/V4 assembly; frozen ABI3 experiments remain unchanged."""

from __future__ import annotations

import ctypes as C

import mujoco
import numpy as np
import torch

from native_stand.simulation.jump_dense_env import ScoringPlant
from native_stand.simulation.jump_shared_c import SharedJump
from native_stand.simulation.jump_shared_v4 import SharedV4Execution
from native_stand.tasks.jump import unified


class UnifiedJump(SharedJump):
    VALID_PHASES = tuple(range(10))
    ABI_VERSION = unified.ABI
    INPUT_CONTRACT = unified.INPUT_ID
    OUTPUT_CONTRACT = unified.OUTPUT_ID
    CONFIG_CONTRACT = unified.CONFIG_ID

    def __init__(self, *args, library_sha=None, **kwargs):
        if not library_sha:
            raise ValueError("verified ABI4 library SHA required")
        super().__init__(*args, library_sha=library_sha, **kwargs)
        from native_stand.tasks.jump.outcome import CONFIG_ID

        if self.config.config_id not in (unified.CONFIG_ID, CONFIG_ID):
            raise ValueError("Unknown Jump controller config")
        self.CONFIG_CONTRACT = self.config.config_id
        self.lib.RLJumpUpstreamMotorCommand.argtypes = [C.c_void_p] * 3


class UnifiedV4Execution(SharedV4Execution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, control_class=UnifiedJump)
        self.v4.caps[:, [2, 5]] = 4.8
        self.timing_version = "jump_unified_ABI4_wheel48_target5_fullV4_v1"

    def publish_raw(self, raw, mask=None):
        a = raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
        return self.c.publish(a, mask)

    publish_mean = publish_raw
    publish_latent = publish_raw


class UnifiedPlant(ScoringPlant):
    def __init__(self, *args, unified_library, unified_library_sha, **kwargs):
        super().__init__(
            *args,
            library=unified_library,
            library_sha=unified_library_sha,
            execution_factory=UnifiedV4Execution,
            **kwargs,
        )
        cfg = self.execution.c.config
        cfg.enabled = cfg.model_available = 1
        cfg.model_id = 0x53494D34  # SIM only
        cfg.model_input_id = unified.INPUT_ID
        cfg.model_output_id = unified.OUTPUT_ID
        self.caps[[2, 5]] = 4.8
        for a in self.aids[[2, 5]]:
            self.model.actuator_ctrlrange[a] = [-4.8, 4.8]
            self.model.actuator_forcerange[a] = [-4.8, 4.8]
            self.model.actuator_ctrllimited[a] = self.model.actuator_forcelimited[a] = 1
        self.command = np.array([0.0, 0.0, 0.34])
        self.jump_request_after_ms = None

    def feedback(self):
        i = super().feedback()
        i.command[:] = self.command
        i.upstream_owner = 2
        elapsed = self.execution.tick - self.episode_start_tick
        i.key_down = int(
            self.jump_request_after_ms is not None
            and self.jump_request_after_ms <= elapsed < self.jump_request_after_ms + 20
        )
        return i

    def initialize_case(self, case):
        # Explicit episode reset creates a new local clock, never a phase transition.
        # Cumulative charged physics_steps is preserved separately.
        self.execution.tick = self.execution.v4.tick = self.execution.v4.leg.tick = 0
        self.execution.input_rows = None
        self.execution.c.last_snapshot = None
        self.execution.last_frames = None
        length = case.get("leg_length_m", 0.28)
        if length not in self._reset_shapes:
            self._reset_shapes[length] = self.geometry.solve(length)
        shape = self._reset_shapes[length]
        self.invalidate_measurements()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = shape.root_support_height_m + case.get("drop_m", 0.0)
        for name, joint in self.refs.items():
            self.data.qpos[self.model.jnt_qposadr[joint]] = shape.q[name]
        self.command[:] = case["command"]
        forward = np.array([0.0, 1.0, 0.0])  # CAD -Z rotated +90deg around X
        self.data.qvel[:3] = forward * case.get("initial_vx_m_s", 0.0)
        self.data.time = self.execution.tick * 0.001
        mujoco.mj_forward(self.model, self.data)
        q = torch.tensor(self.data.qpos[self.qids][None], dtype=torch.float32)
        dq = torch.tensor(self.data.qvel[self.vids][None], dtype=torch.float32)
        self.execution.reset(self.execution.v4.ids, q, dq)
        s = self.execution.c.states[0]
        s.adopted_valid = 1
        s.adopted_sequence = 1
        s.adopted_tick = s.phase_tick = self.execution.tick
        s.adopted[:] = self.execution.v4.adopted[0].numpy()
        s.first_hold = 1
        self.actual_motor[:] = 0
        self.actual_NET[:] = 0
        self.solved_ground_Fz_N = 0.0
        self.solved_ground_contact[:] = False
        self.solved_restricted = False
        self.scoring_force_N[:] = self.scoring_slip_m_s[:] = 0
        self.scoring_force_valid = False
        self.episode_start_tick = self.execution.tick
        self.request_jump = False
        self.jump_request_after_ms = case.get("jump_request_after_ms")
        return self.readback()
