"""V4-adapted old successful reward with finite-horizon success settlement.

No negative baseline/remaining worst-cost subsidy. Keep original positive
lift/support terms and explicit stop shaping. Successful terminal receives
the discounted upper bound of remaining 23s positive reward, avoiding loss
of survival reward by completing ten-second hold. Timeout has no bootstrap.
"""

import torch

from native_stand.tasks.recovery.legacy_reward import MAX_DENSITY, density

from .execution import CAPS
from .lifecycle import Lifecycle as Base

GAMMA = 0.9980019986673331


class Lifecycle(Base):
    def observe(self, *, physical_q, clearance, **kwargs):
        previous = self.last_target.clone()
        initialized = self.valid_target.clone()
        out = super().observe(**kwargs)
        old = self.reward.clone()
        target = kwargs["target"]
        s = kwargs["state"]
        dt = kwargs.get("dt_s", 0.01)
        if abs(dt - 0.01) > 1e-12:
            raise ValueError("Versioned legacy physical reward requires 10ms")
        previous = torch.where(initialized[:, None], previous, target)
        # Only standing-retention episodes use a declared standing reward
        # origin; true Swing starts use their actual initial physical height.
        origin = torch.where(self.h0 >= 0.30, torch.full_like(self.h0, 0.168), self.h0)
        latched = self.entry >= 0
        age = torch.where(latched, (kwargs["time_s"] - self.entry).clamp_min(0), 0.0)
        result = density(
            state=s,
            h0=origin,
            command_height=getattr(self, "command_height", torch.full_like(self.h0, 0.34)),
            leg_q=physical_q[:, [0, 1, 3, 4]],
            angular_pitch_roll=kwargs["angular_velocity"][:, [0, 2]],
            motor=kwargs["motor"],
            previous_target=previous,
            adopted_target=target,
            entered=latched,
            entry_age=age,
            clearance=clearance,
            controlled=kwargs["student"],
            valid=s[:, 11] <= 0.005,
            fault=kwargs["terminal"],
            effort_caps=getattr(self, "effort_caps", s.new_tensor(CAPS)),
        )
        eligible = result["eligible"]
        # Certificate entry is not cut off after2.5s. Only its extra speed
        # bonus decays to zero; a late legal rise remains a valid entry.
        entry_bonus = out["entered"] * (1 + (1 - kwargs["time_s"] / 2.5).clamp(0, 1)) * eligible
        succeeded = out["success"] & eligible
        fault = kwargs["terminal"] & ~succeeded
        timeout = out["truncated"] & ~succeeded
        remaining = torch.round((self.max_s - kwargs["time_s"]).clamp_min(0) / dt)
        success_tail = GAMMA * dt * MAX_DENSITY * (1 - GAMMA**remaining) / (1 - GAMMA) * succeeded
        failure = -5 * (fault | timeout).to(s.dtype)
        self.reward = (
            dt * result["weighted_density"].sum(-1)
            + entry_bonus
            + 10 * succeeded
            + success_tail
            + failure
        )
        self.total_reward += self.reward - old
        self.last_phi.zero_()  # No potential shaping in this reward version.
        out.update(
            reward=self.reward,
            density=result["density"],
            potential=torch.zeros_like(self.reward),
            reward_height_progress=result["height_progress"],
            success_tail_credit=success_tail,
            entry_bonus=entry_bonus,
            failure_event_cost=failure,
            mirror_error_rad=result["mirror_error_rad"],
            precision_gate=result["precision_gate"],
        )
        self.last_result = out
        return out
