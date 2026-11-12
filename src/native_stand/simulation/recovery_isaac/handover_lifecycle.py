"""Recovery raises and hands over; frozen Motion confirms ten seconds.

Entry conditions retain the former geometric entry envelope and .2s debounce.
Recovery no longer has to execute the subsequent ten-second standing hold.
"""

import torch

from .execution import CAPS
from .lifecycle import Lifecycle as Base
from .lifecycle import huber

VERSION = "recovery_rise_then_frozen_Motion_confirm_v1"


class Lifecycle(Base):
    def __init__(self, n, device="cpu", dtype=torch.float32):
        super().__init__(n, device, dtype)
        self.handover_active = torch.zeros(n, device=device, dtype=torch.bool)
        self.handover_time = self.entry.clone()
        self.motion_hold = self.hold.clone()
        self.entry_events = self.hold.clone()

    def reset(self, ids, h0, prefix_s):
        super().reset(ids, h0, prefix_s)
        self.handover_active[ids] = False
        self.handover_time[ids] = -1.0
        self.motion_hold[ids] = 0.0
        self.entry_events[ids] = 0.0

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
        h, pitch, roll = s[:, 3], s[:, 4], s[:, 5]
        if not bool(torch.isfinite(torch.cat((s, target, motor, angular_velocity), -1)).all()):
            raise ValueError("Nonfinite handover lifecycle")
        dual = (s[:, 9:11] > 2).all(-1)
        was_motion = self.handover_active.clone()
        eligible = (
            (h >= 0.30)
            & (h <= 0.40)
            & (abs(pitch) <= 0.1)
            & (abs(roll) <= 0.1)
            & dual
            & (s[:, 11] <= 0.005)
            & student
            & ~terminal
        )
        self.dwell = torch.where(eligible, self.dwell + dt, 0.0)
        entered = (~was_motion) & student & (self.dwell >= 0.2 - 1e-7)
        self.entry = torch.where(entered, time_s, self.entry)
        self.handover_time = torch.where(entered, time_s, self.handover_time)
        self.handover_active |= entered
        self.entry_events += entered
        # Only actual downstream-control transitions count towards confirmation.
        quality = (
            (h >= 0.30)
            & (h <= 0.40)
            & (abs(pitch) <= 0.18)
            & (abs(roll) <= 0.18)
            & dual
            & (s[:, 11] <= 0.005)
            & ~terminal
        )
        self.motion_hold = torch.where(was_motion & ~student & quality, self.motion_hold + dt, 0.0)
        self.hold.copy_(self.motion_hold)
        succeeded = ~self.success & (self.motion_hold >= 10.0 - 1e-5)
        self.success |= succeeded
        progress = ((h - self.h0) / (0.34 - self.h0).clamp_min(0.02)).clamp(0, 1)
        broad = torch.exp(-((pitch / 0.5).square() + (roll / 0.35).square()))
        phi = progress * broad
        potential = 2 * (
            torch.exp(-dt / 5) * torch.where(terminal | succeeded, 0.0, phi) - self.last_phi
        )
        self.last_phi.copy_(phi)
        delta = torch.where(self.valid_target[:, None], target - self.last_target, 0.0)
        self.last_delta.copy_(delta)
        self.last_target.copy_(target)
        self.valid_target.fill_(True)
        scale = s.new_tensor([0.05, 0.05, 10, 0.05, 0.05, 10])
        caps = s.new_tensor(CAPS)
        # Broad rise/contact shaping; no fine-standing, HF or long-own-hold gate.
        terms = dict(
            rise_progress=3 * progress * broad,
            support_acquisition=2 * dual * broad,
            posture_broad=broad,
            leg_geometry=0.5 * torch.exp(-((s[:, 6] / 0.10).square() + (s[:, 8] / 0.35).square())),
            angular_cost=-0.03
            * huber((torch.linalg.vector_norm(angular_velocity, dim=-1) - 2.0).clamp_min(0)),
            motor_effort_cost=-0.005 * (motor / caps).square().mean(-1),
            physical_target_delta_cost=-0.0005 * (delta / scale).square().mean(-1),
            recovery_time_cost=s.new_full((len(s),), -0.1),
        )
        terms = {k: v * student for k, v in terms.items()}
        # Dense rewards stop once Motion takes control; the delayed confirmation
        # bonus/failure still propagates to prior Recovery actions through GAE.
        self.reward = (
            dt * sum(terms.values())
            + potential * student
            + 5 * entered.to(s.dtype)
            + 20 * succeeded.to(s.dtype)
            - 5 * terminal.to(s.dtype)
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
            hold_quality=quality,
            first_entry_age_s=(time_s - self.entry).clamp_min(0),
            density=terms,
            potential=potential,
            Motion_confirmation=self.motion_hold.clone(),
        )
        return self.last_result
