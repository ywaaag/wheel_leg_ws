"""Observable-only Motion Teacher domains for supervision, never deployment."""

import math

VERSION = "motion_observable_teacher_domain_v1"
ROLES = ("quiet_c0", "core_vm1", "new_u2200")


def validate(config):
    expected = {
        "version",
        "role",
        "reference_height_m",
        "command_tolerance",
        "core_max_vx_mps",
        "quiet_gyro_rad_s",
        "quiet_leg_dq_rad_s",
        "quiet_wheel_dq_rad_s",
        "quiet_tilt_rad",
    }
    if set(config) != expected or config["version"] != VERSION or config["role"] not in ROLES:
        raise ValueError("Explicit observable Motion Teacher domain required")
    if any(
        not isinstance(config[k], int | float) or not math.isfinite(config[k]) or config[k] <= 0
        for k in expected - {"version", "role"}
    ):
        raise ValueError("Finite positive observable domain parameters required")


def mask(observation, config):
    import torch

    tolerance = config["command_tolerance"]
    command = observation[:, 22:25] * observation.new_tensor([4.0, 4.0, 0.5])
    same_height = (command[:, 2] - config["reference_height_m"]).abs() <= tolerance
    straight = command[:, 1].abs() <= tolerance
    core = same_height & straight & (command[:, 0].abs() <= config["core_max_vx_mps"] + tolerance)
    quiet = (
        core
        & (command[:, 0].abs() <= tolerance)
        & (torch.linalg.vector_norm(observation[:, :3] * 4, dim=1) <= config["quiet_gyro_rad_s"])
        & ((observation[:, [10, 11, 13, 14]] * 20).abs().amax(1) <= config["quiet_leg_dq_rad_s"])
        & ((observation[:, [12, 15]] * 20).abs().amax(1) <= config["quiet_wheel_dq_rad_s"])
        & (-observation[:, 4] >= math.cos(config["quiet_tilt_rad"]))
    )
    return (
        quiet
        if config["role"] == "quiet_c0"
        else core & ~quiet
        if config["role"] == "core_vm1"
        else ~core
    )
