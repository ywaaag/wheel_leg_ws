"""Deployable event FSM, model-side contact order; one six-axis owner per tick."""

from __future__ import annotations

from enum import IntEnum


class Phase(IntEnum):
    IDLE = 0
    CROUCH = 1
    PUSH = 2
    FLIGHT = 3
    LAND = 4
    SETTLE = 5
    CONFIRM = 6
    COOLDOWN = 7
    FAULT = 8
