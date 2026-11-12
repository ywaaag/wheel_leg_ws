"""Stage rewards in SI, once-only events, independent active-jump provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RewardMemory:
    apex_paid: bool = False
    takeoff_paid: bool = False
    handover_paid: bool = False
    previous_COM_vz: float = 0.0
    takeoff_COM_height_m: float | None = None
    peak_COM_height_m: float | None = None
    motor_positive_work_J: float = 0.0
    motor_negative_work_J: float = 0.0
    NET_work_J: float = 0.0
    spring_work_J: float = 0.0
    initial_kind: str = "ground_active_jump"

    def state_dict(self):
        return asdict(self)
