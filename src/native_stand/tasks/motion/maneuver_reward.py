"""Replace conflicting inherited reward columns, keeping real physical HF."""

from __future__ import annotations

import torch

from .maneuver_targets import mode_masks, roll_target
from .scale_corrections import huber, replace_density, tracking_terms

VERSION = "motion48_height_bank_mode_reward_v1"


def configuration():
    return dict(
        version=VERSION,
        dt=0.01,
        height_weight=4.0,
        height_wide_m=0.06,
        height_narrow_m=0.015,
        height_narrow_mix=0.25,
        height_tail_weight=-0.75,
        height_tail_scale_m=0.04,
        bank_reference="FLU -atan(vx*yaw/g), soft reference only",
        bank_max_rad=0.30,
        symmetry="standing/straight only; reward, quality termination and assessment agree",
        physical_HF="original integrated1ms term retained exactly; no extra policy dt",
        stand_drift_deadband_mps=0.15,
        zero_position_hold=False,
        motor_effort="actual motor/caps, NET excluded",
        actor_and_PD_dq=True,
    )


def height_density(error):
    return 4.0 * (
        0.75 / (1 + (error / 0.06).square()) + 0.25 * torch.exp(-(error / 0.015).square())
    )


def apply(env, config):
    r = env.student_reward_result
    height_valid = getattr(env, "motion_terrain_height_valid", None)
    if height_valid is not None:
        r["eligible"] &= height_valid
        old = r["dt"].sum(-1)
        for key in ("raw", "density", "dt"):
            r[key] *= height_valid[:, None]
        r["total"] += r["dt"].sum(-1) - old
    c = env.motion_action_command
    mode = mode_masks(c, is_flat=getattr(env, "motion_flat_terrain", None))
    ref = roll_target(c)
    p = env.standing_snapshot
    error = env.recovery_snapshot[:, 6] - c[:, 2]
    replace_density(r, "height", height_density(error))
    replace_density(r, "height_continuous_tail", -0.75 * huber(error / 0.04))
    angles = torch.stack((p[:, 0], p[:, 1] - ref), -1)
    replace_density(r, "upright", 2.0 / (1 + (angles / 0.20).square().sum(-1)))
    replace_density(r, "roll_excess", -0.5 * huber(torch.relu(angles[:, 1].abs() - 0.05) / 0.10))
    replace_density(
        r,
        "fullchain_roll",
        -0.6
        * huber(
            angles[:, 1]
            / torch.where(mode["travel_turn"], ref.new_tensor(0.10), ref.new_tensor(0.05))
        ),
    )
    for name in ("mirror", "wheel_alignment", "leg_balance", "leg_swing_symmetry"):
        if name in r["fields"]:
            old = r["density"][:, r["fields"].index(name)].clone()
            replace_density(r, name, old * mode["symmetry_required"])
    if "transfer_vertical_velocity" in r["fields"]:
        vz = env.robot.data.root_lin_vel_w.torch[:, 2]
        if getattr(env, "motion_surface", None) is not None:
            from native_stand.simulation.ground_surface import velocity_from_env

            vz = velocity_from_env(env)
        replace_density(
            r, "transfer_vertical_velocity", -0.5 * (vz - env.maneuver_height_rate).square()
        )
    # All actual command axes get the same wide/narrow tracking semantics.
    for yaw, axis, col, track_name, tail_name in [
        (False, 0, 7, "linear_tracking", "linear_error"),
        (True, 1, 9, "yaw_tracking", "yaw_error"),
    ]:
        track, tail = tracking_terms(
            env.recovery_snapshot[:, col] - c[:, axis], yaw=yaw, config=config
        )
        if not yaw:
            standing = mode["standing"]
            track = torch.where(
                standing,
                4.0 - 0.25 * huber(torch.relu(env.recovery_snapshot[:, 7].abs() - 0.15) / 0.30),
                track,
            )
            tail = torch.where(standing, torch.zeros_like(tail), tail)
        replace_density(r, track_name, track)
        replace_density(r, tail_name, tail)
    # Demand support also for travelling curves; the old spin-specific term
    # is preserved, and this replaces the general existing support column.
    deficit = torch.relu(1 - env.recovery_snapshot[:, 10:12] / 2).clamp_max(1).square().mean(-1)
    replace_density(r, "support_force_deficit", -1.0 * deficit)
    env.maneuver_swing_difference.copy_(
        r["symmetry_diagnostics"]["leg_swing_signed_difference_rad"].squeeze(-1)
    )
    r["maneuver_reward_version"] = VERSION


def quality_inputs(posture, wheel_offset, command, *, is_flat=None):
    """Only task-dependent geometry gates change; absolute physical safety stays."""
    masks = mode_masks(command, is_flat=is_flat)
    p = posture.clone()
    p[:, 1] -= roll_target(command)
    asymmetric = ~masks["symmetry_required"]
    mean = posture[:, 2:4].mean(-1)
    p[:, 2] = torch.where(asymmetric, mean, p[:, 2])
    p[:, 3] = torch.where(asymmetric, mean, p[:, 3])
    p[:, 4:6] = torch.where(asymmetric[:, None], torch.zeros_like(p[:, 4:6]), p[:, 4:6])
    wheel = torch.where(asymmetric, torch.zeros_like(wheel_offset), wheel_offset)
    return p, wheel
