"""Explicit Motion motor limits; historical jobs keep their frozen limits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

VERSION = "motion_motor48_leg40_v1"
WHEELS = (2, 5)
CURRENT_PER_NM = 19.0 / 0.00036 / (268.0 / 17.0)
HISTORICAL_WHEEL_NM = 8192.0 / CURRENT_PER_NM


@dataclass(frozen=True)
class MotorLimits:
    version: str = VERSION
    leg_nm: float = 40.0
    wheel_nm: float = 4.8

    def __post_init__(self):
        if self.version != VERSION or self.leg_nm != 40.0 or self.wheel_nm != 4.8:
            raise ValueError("Explicit 40Nm leg / 4.8Nm wheel contract required")

    @property
    def caps(self):
        return (self.leg_nm, self.leg_nm, self.wheel_nm, self.leg_nm, self.leg_nm, self.wheel_nm)

    @property
    def wheel_current_raw(self):
        return self.wheel_nm * CURRENT_PER_NM

    def as_dict(self):
        return {
            **asdict(self),
            "motor_caps_nm": list(self.caps),
            "wheel_current_raw": self.wheel_current_raw,
            "NET_and_passive_force_inside_motor_cap": False,
            "input_scales": [1.5, 100.0],
            "action_scales": [0.15, 80.0],
        }


def validate(config):
    expected = MotorLimits().as_dict()
    if config != expected:
        raise ValueError("Motion execution limit declaration differs")
    return MotorLimits()


def clip_motor(request, limits):
    """Final motor clipping only; do not pass NET or passive forces here."""
    import torch

    cap = request.new_tensor(limits.caps)
    if request.shape[-1] != 6 or not torch.isfinite(request).all():
        raise ValueError("Finite six-channel motor request required")
    return request.clamp(-cap, cap)


def check_wheel_cap(value):
    if not math.isfinite(value) or value not in (HISTORICAL_WHEEL_NM, 4.8):
        raise ValueError("Only declared historical / 4.8Nm comparison profiles")
    return value
