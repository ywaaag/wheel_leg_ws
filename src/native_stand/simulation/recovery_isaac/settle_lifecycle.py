"""v8 plus small continuous pre-entry settling costs, without HF or new gates."""

import torch

from .lifecycle import huber
from .preentry_lifecycle import Lifecycle as Preentry

VERSION = "recovery_early_settle48_physical_v9"
PARAMETERS = dict(
    height_below_command_begin_m=0.08,
    height_ramp_m=0.06,
    tilt_width_rad=0.20,
    angular_scale_rad_s=1.0,
    lateral_scale_m_s=0.20,
    angular_coefficient_per_s=0.025,
    contact_coefficient_per_s=0.10,
    lateral_coefficient_per_s=0.025,
    HF_coefficient_per_s=0.0,
)


def settling_density(state, angular_velocity, command_height, eligible):
    """Measured body velocity/contact; gate does not require entry or both contacts."""
    height = ((state[:, 3] - command_height + 0.08) / 0.06).clamp(0, 1)
    height = height.square() * (3 - 2 * height)
    posture = torch.exp(-(state[:, 4:6] / 0.20).square().sum(-1))
    gate = height * posture * eligible
    terms = dict(
        early_settle_angular_cost=-0.025 * huber(angular_velocity / 1.0).mean(-1),
        early_settle_contact_cost=-0.10 * ((2 - state[:, 9:11]) / 2).clamp(0, 1).square().mean(-1),
        early_settle_lateral_cost=-0.025 * huber(state[:, 1] / 0.20),
    )
    return gate, {name: value * gate for name, value in terms.items()}


class Lifecycle(Preentry):
    def observe(self, **kwargs):
        out = super().observe(**kwargs)
        s = kwargs["state"]
        eligible = kwargs["student"] & (s[:, 11] <= 0.005) & ~kwargs["terminal"]
        command = getattr(self, "command_height", torch.full_like(self.h0, 0.34))
        gate, terms = settling_density(s, kwargs["angular_velocity"], command, eligible)
        delta = kwargs.get("dt_s", 0.01) * sum(terms.values())
        self.reward = self.reward + delta
        self.total_reward += delta
        out["reward"] = self.reward
        out["density"].update(terms)
        out["early_settle_gate"] = gate
        return out
