"""Legacy v7 plus ONE pre-entry high-translation soft cost; no speed limiter."""

import torch

from .legacy_lifecycle import Lifecycle as Legacy
from .lifecycle import huber

VERSION = "recovery_preentry_runaway48_physical_v8"
PARAMETERS = dict(deadband_m_s=1.0, scale_m_s=1.0, coefficient_per_s=0.1)


def runaway_density(vx, entered, eligible):
    if not bool(torch.isfinite(vx).all()):
        raise ValueError("Finite true body-forward velocity")
    excess = (vx.abs() - PARAMETERS["deadband_m_s"]).clamp_min(0) / PARAMETERS["scale_m_s"]
    return -PARAMETERS["coefficient_per_s"] * huber(excess) * (~entered) * eligible


class Lifecycle(Legacy):
    def observe(self, **kwargs):
        out = super().observe(**kwargs)
        s = kwargs["state"]
        eligible = kwargs["student"] & (s[:, 11] <= 0.005) & ~kwargs["terminal"]
        cost = runaway_density(s[:, 0], self.entry >= 0, eligible)
        delta = kwargs.get("dt_s", 0.01) * cost
        self.reward = self.reward + delta
        self.total_reward += delta
        out["reward"] = self.reward
        out["density"]["pre_entry_runaway_cost"] = cost
        return out
