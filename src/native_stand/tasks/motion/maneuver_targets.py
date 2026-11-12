"""Task targets for level spin, symmetric straight travel and banked turns."""

from __future__ import annotations

import math

import torch

VERSION = "motion_height_then_motion_banked_turn_v1"
MINIMUM_SIGNED_YAW_GAIN = 0.5


def yaw_response(mean_yaw, target_yaw):
    """Physical rad/s, separate from precision MAE and absolute quality gates."""
    if not math.isfinite(mean_yaw) or not math.isfinite(target_yaw):
        raise ValueError("Finite actual/target yaw required")
    required = abs(target_yaw) > 0.05
    gain = mean_yaw / target_yaw if required else None
    return dict(
        required=required,
        minimum_signed_gain=MINIMUM_SIGNED_YAW_GAIN if required else None,
        signed_gain=gain,
        passed=not required or gain >= MINIMUM_SIGNED_YAW_GAIN,
    )


def mode_masks(command, *, tolerance=0.05, is_flat=None):
    if command.shape[-1] != 3 or not torch.isfinite(command).all():
        raise ValueError("Finite vx/yaw/height command required")
    moving = command[..., 0].abs() > tolerance
    turning = command[..., 1].abs() > tolerance
    return {
        "standing": ~moving & ~turning,
        "straight": moving & ~turning,
        "spin": ~moving & turning,
        "travel_turn": moving & turning,
        "symmetry_required": ~turning if is_flat is None else (~turning & is_flat),
    }


def roll_target(command, *, roll_limit_rad=0.30, gravity_mps2=9.81):
    """FLU roll: positive top lean is right; positive vx*yaw turns left.

    Thus bank left requires negative roll. This is a reward reference, never
    an extra actuator command, IMU correction or observed oracle variable.
    """
    mode = mode_masks(command)
    desired = -torch.atan(command[..., 0] * command[..., 1] / gravity_mps2)
    return torch.where(
        mode["travel_turn"],
        desired.clamp(-roll_limit_rad, roll_limit_rad),
        torch.zeros_like(desired),
    )


def spin_rolling_error(left_forward_mps, right_forward_mps):
    """Use hold-mean common/differential rolling, not joint-name polarity."""
    common = (left_forward_mps + right_forward_mps) * 0.5
    differential = (left_forward_mps - right_forward_mps) * 0.5
    tolerance = torch.maximum(common.new_tensor(0.03), differential.abs() * 0.20)
    return torch.relu(common.abs() - tolerance), common, differential


def next_spin_target(current_rad_s, *, passed, factor=1.25):
    if not math.isfinite(current_rad_s) or current_rad_s <= 0 or factor <= 1:
        raise ValueError("Finite positive spin level and growing factor required")
    # No fixed 4rad/s terminal level. Every individual rollout remains bounded.
    return current_rad_s * factor if passed else current_rad_s


def banked_yaw_limit(vx, *, roll_limit_rad=0.30, gravity_mps2=9.81):
    """Finite lateral-acceleration envelope for travelling turns only."""
    return gravity_mps2 * math.tan(roll_limit_rad) / max(abs(vx), 0.10)


def command_program(seconds, case):
    """Change height first, then move; combined yaw starts during travel."""
    from .maneuver_program import EXTENDED_VERSION, extended_numpy_program, numpy_program
    from .maneuver_program import VERSION as SHARED_PROGRAM

    if case.get("command_program_version") == EXTENDED_VERSION:
        return extended_numpy_program(seconds, case["command"])
    if case.get("command_program_version") == SHARED_PROGRAM:
        return numpy_program(seconds, case["command"])
    if case.get("command_program_version") is not None:
        raise ValueError("Unknown command timing version")
    import numpy as np

    vx, yaw, target_height = case["command"]
    if case.get("height_then_motion"):
        h = 0.34 + (target_height - 0.34) * float(np.clip((seconds - 1.0) / 2.0, 0.0, 1.0))
        factor = float(np.clip((seconds - 6.0) / 2.0, 0.0, 1.0))
        return np.array([vx * factor, yaw * factor, h])
    factor = float(np.clip((seconds - 0.5) / 1.0, 0.0, 1.0))
    turn_factor = float(np.clip((seconds - 5.0) / 2.0, 0.0, 1.0)) if vx and yaw else factor
    return np.array([vx * factor, yaw * turn_factor, target_height])
