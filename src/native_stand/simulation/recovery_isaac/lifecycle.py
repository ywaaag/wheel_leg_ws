"""Vectorized Recovery clocks/reward. Own control is distinct from C0 prefix.

Same declared physical rewards as CPU v2; v3 explicitly absorbs potential on
SUCCESS as well as fault termination (CPU v2 did so only for fault). No gate is
relaxed. True truncations retain successor potential/value bootstrap.
"""

from __future__ import annotations

import copy

import torch

from .execution import CAPS, LEGS

VERSION = "recovery_physical_entry_hold_v3_isaac"


def huber(x):
    a = abs(x)
    return torch.where(a < 1, 0.5 * a.square(), a - 0.5)


class Lifecycle:
    def __init__(self, n, device="cpu", dtype=torch.float32):
        self.n = n
        self.device = device
        self.dtype = dtype
        self.max_s = 23.0
        z = torch.zeros(n, device=device, dtype=dtype)
        for k in ["h0", "prefix_s", "dwell", "hold", "last_phi", "elapsed", "total_reward"]:
            setattr(self, k, z.clone())
        self.entry = z.new_full((n,), -1)
        self.success = torch.zeros(n, device=device, dtype=torch.bool)
        self.last_target = z.new_zeros(n, 6)
        self.last_delta = self.last_target.clone()
        self.valid_target = self.success.clone()
        self.reward = z.clone()
        self.last_result = {}

    def reset(self, ids, h0, prefix_s):
        self.h0[ids] = h0.to(self.h0)
        self.prefix_s[ids] = prefix_s.to(self.prefix_s)
        for k in [
            "dwell",
            "hold",
            "last_phi",
            "elapsed",
            "total_reward",
            "last_target",
            "last_delta",
            "reward",
        ]:
            getattr(self, k)[ids] = 0
        self.entry[ids] = -1
        self.success[ids] = False
        self.valid_target[ids] = False

    def observe(
        self,
        *,
        time_s,
        state,
        target,
        motor,
        student,
        terminal,
        truncated,
        hp10,
        angular_velocity,
        dt_s=0.01,
    ):
        s = state
        dt = s.new_tensor(dt_s)
        if not bool(torch.isfinite(torch.cat((s, target, motor), -1)).all()):
            raise ValueError("Nonfinite lifecycle")
        height, pitch, roll = s[:, 3], s[:, 4], s[:, 5]
        dual = (s[:, 9:11] > 2).all(-1)
        commanded = getattr(self, "command_height", torch.full_like(height, 0.34))
        eligible = (
            (height >= 0.30)
            & (height <= 0.40)
            & (abs(pitch) <= 0.1)
            & (abs(roll) <= 0.1)
            & dual
            & (s[:, 11] <= 0.005)
        )
        if getattr(self, "height_conditioned", False):
            eligible = eligible & (abs(height - commanded) <= 0.02)
        self.dwell = torch.where(eligible, self.dwell + dt, 0.0)
        entered = (self.entry < 0) & (self.dwell >= 0.2 - 1e-7)
        self.entry = torch.where(entered, time_s, self.entry)
        latched = self.entry >= 0
        age = torch.where(latched, (time_s - self.entry).clamp_min(0), 0.0)
        hold_quality = (
            (height >= 0.30)
            & (height <= 0.40)
            & (abs(pitch) <= 0.18)
            & (abs(roll) <= 0.18)
            & dual
            & (s[:, 11] <= 0.005)
        )
        if getattr(self, "height_conditioned", False):
            hold_quality = hold_quality & (abs(height - commanded) <= 0.02)
        self.hold = torch.where(latched & hold_quality & student, self.hold + dt, 0.0)
        succeeded = (
            (~self.success)
            & (self.hold >= 10.0 - 1e-5)
            & (time_s >= getattr(self, "minimum_success_time", torch.zeros_like(time_s)) - 1e-6)
        )
        self.success |= succeeded
        progress = ((height - self.h0) / (commanded - self.h0).clamp_min(0.02)).clamp(0, 1)
        broad = torch.exp(-((pitch / 0.5).square() + (roll / 0.35).square()))
        narrow = torch.exp(-((pitch / 0.1).square() + (roll / 0.05).square()))
        phi = progress * broad
        potential = 2 * (
            torch.exp(-dt / 5) * torch.where(terminal | succeeded, 0.0, phi) - self.last_phi
        )
        self.last_phi = phi
        tighten = torch.where(latched, ((age - 0.5) / 1.5).clamp(0, 1), 0.0)
        delta = torch.where(self.valid_target[:, None], target - self.last_target, 0.0)
        dd = delta - self.last_delta
        self.last_target.copy_(target)
        self.last_delta.copy_(delta)
        self.valid_target.fill_(True)
        scale = s.new_tensor([0.05, 0.05, 10, 0.05, 0.05, 10])
        caps = getattr(self, "effort_caps", s.new_tensor(CAPS))
        terms = dict(
            supported_posture=3 * (0.4 * broad + 0.6 * narrow) * (0.25 + 0.75 * dual),
            height_precision=(
                0.4 * torch.exp(-((height - commanded) / 0.1).square())
                + 0.6 * torch.exp(-((height - commanded) / 0.025).square())
            )
            * broad,
            mirror_precision=0.5
            * torch.exp(-((s[:, 6] / 0.05).square() + (s[:, 8] / 0.1).square()))
            * dual,
            rotation_cost=-0.1
            * huber(
                (
                    torch.linalg.vector_norm(angular_velocity, dim=-1)
                    - torch.where(tighten > 0, 0.2, 2.0)
                ).clamp_min(0)
            ),
            motor_effort_cost=-0.02 * (motor / caps).square().mean(-1),
            physical_target_delta_cost=-0.002 * (delta / scale).square().mean(-1),
            target_second_difference_cost=-tighten
            * (
                0.005 * (dd[:, LEGS] / 0.05).square().mean(-1)
                + 0.001 * (dd[:, [2, 5]] / 10).square().mean(-1)
            ),
            standing_contact_deficit_cost=-0.5 * tighten * (~dual),
            actual_dq_HP10_cost=-0.1 * tighten * huber(hp10 / 0.2).mean(-1),
            saturation_cost=-0.1 * tighten * (abs(motor) >= caps * 0.98).to(s.dtype).mean(-1),
            drift_cost=-0.05 * tighten * huber((abs(s[:, 0]) - 0.3).clamp_min(0)),
            pre_entry_time_cost=-0.05 * (~latched),
        )
        self.reward = (
            dt * sum(terms.values()) + potential + entered.to(s.dtype) + 10 * succeeded.to(s.dtype)
        )
        self.elapsed.copy_(time_s)
        self.total_reward += self.reward
        terminated = terminal | succeeded
        truncated = (~terminated) & (truncated | (time_s >= self.max_s - 1e-6))
        self.last_result = dict(
            reward=self.reward,
            entered=entered,
            success=succeeded,
            terminated=terminated,
            truncated=truncated,
            done=terminated | truncated,
            entry_eligible=eligible,
            hold_quality=hold_quality,
            first_entry_age_s=age,
            density=terms,
            potential=potential,
        )
        return self.last_result

    def extra(self, state, clearance, restricted, initialized):
        h = state[:, 3]
        commanded = getattr(self, "command_height", torch.full_like(h, 0.34))
        phase = torch.where(self.entry >= 0, 2, torch.where(h < 0.27, 0, 1))
        return dict(
            initial_root_height=self.h0[:, None] / 0.5,
            height_progress=((h - self.h0) / (commanded - self.h0).clamp_min(0.02)).clamp(0, 1)[
                :, None
            ],
            chassis_clearance=clearance[:, None] / 0.5,
            closure=state[:, 11:12] / 0.005,
            restricted_contact=restricted[:, None] / 100,
            stage_onehot=torch.nn.functional.one_hot(phase, 3).to(state.dtype),
            entry_latched=(self.entry >= 0).to(state.dtype)[:, None],
            entry_age=torch.where(self.entry >= 0, self.elapsed - self.entry, 0.0)[:, None] / 10,
            hold_age=self.hold[:, None] / 10,
            student_age=(self.elapsed - self.prefix_s).clamp_min(0)[:, None] / 23,
            time_remaining=(self.max_s - self.elapsed).clamp_min(0)[:, None] / 23,
            entry_dwell_progress=(self.dwell / 0.2).clamp_max(1)[:, None],
            initialization_complete=initialized.to(state.dtype)[:, None],
        )

    def state_dict(self):
        return copy.deepcopy(
            {
                k: v
                for k, v in vars(self).items()
                if torch.is_tensor(v) or k in ["max_s", "last_result", "height_conditioned"]
            }
        )

    def load_state_dict(self, value):
        for k, v in value.items():
            setattr(self, k, copy.deepcopy(v.to(self.device) if torch.is_tensor(v) else v))
