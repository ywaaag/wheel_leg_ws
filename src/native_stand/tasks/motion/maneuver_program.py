"""One numerical timing contract for new Isaac, MuJoCo and scoring consumers."""

from __future__ import annotations

VERSION = "motion48_height_travel_turn_schedule_v2"
EXTENDED_VERSION = "motion48_height020_rate_limited_schedule_v3"


def extended_tensor_program(age, target_xy, goals):
    import torch

    duration = (goals - 0.34).abs() / 0.02
    moving_height = (goals - 0.34).abs() > 1e-6
    phase = ((age - 1) / duration.clamp_min(0.01)).clamp(0, 1)
    height = 0.34 + (goals - 0.34) * phase
    height = torch.where(phase >= 1, goals, height)
    ready = torch.where(moving_height, 2.0 + duration, age.new_full(age.shape, 0.5))
    linear = ((age - ready) / 1.0).clamp(0, 1)
    mixed = (target_xy[:, 0] != 0) & (target_xy[:, 1] != 0)
    angular = torch.where(mixed, ((age - ready - 4) / 2).clamp(0, 1), linear)
    rate = torch.where((age > 1) & (age < 1 + duration), (goals - 0.34).sign() * 0.02, 0.0)
    return torch.cat(
        (
            target_xy[:, 0:1] * linear[:, None],
            target_xy[:, 1:2] * angular[:, None],
            height[:, None],
        ),
        -1,
    ), rate


def extended_scored_start(target_xy, goals):
    import torch

    duration = (goals - 0.34).abs() / 0.02
    ready = torch.where(duration > 1e-6, 2 + duration, torch.full_like(duration, 0.5))
    return ready + torch.where((target_xy[:, 0] != 0) & (target_xy[:, 1] != 0), 7.0, 3.0)


def extended_phase_flags(age, goals):
    """Causal Critic phase flags for the same rate-limited schedule."""
    import torch

    duration = (goals - 0.34).abs() / 0.02
    ready = torch.where(duration > 1e-6, 2 + duration, torch.full_like(duration, 0.5))
    quiet = (duration > 1e-6) & (age >= 1 + duration) & (age < ready)
    ramp = (age >= ready) & (age < ready + 1)
    return quiet, ramp


def extended_numpy_program(seconds, command):
    import numpy as np

    vx, yaw, goal = command
    duration = abs(goal - 0.34) / 0.02
    phase = float(np.clip((seconds - 1) / max(duration, 0.01), 0, 1))
    height = goal if phase >= 1 else 0.34 + (goal - 0.34) * phase
    ready = 2 + duration if duration > 1e-6 else 0.5
    linear = float(np.clip(seconds - ready, 0, 1))
    angular = float(np.clip((seconds - ready - 4) / 2, 0, 1)) if vx and yaw else linear
    return np.array([vx * linear, yaw * angular, height])


def tensor_program(age, target_xy, goals):
    import torch

    changing = (goals - 0.34).abs() > 1e-6
    height = 0.34 + (goals - 0.34) * ((age - 1.0) / 2.0).clamp(0, 1)
    linear = torch.where(changing, ((age - 6.0) / 2.0).clamp(0, 1), ((age - 0.5) / 1.0).clamp(0, 1))
    mixed = (target_xy[:, 0] != 0) & (target_xy[:, 1] != 0)
    angular = torch.where(
        mixed,
        torch.where(changing, ((age - 11.0) / 2.0).clamp(0, 1), ((age - 5.0) / 2.0).clamp(0, 1)),
        linear,
    )
    command = torch.cat(
        (
            target_xy[:, 0:1] * linear[:, None],
            target_xy[:, 1:2] * angular[:, None],
            height[:, None],
        ),
        -1,
    )
    rate = torch.where((age > 1) & (age < 3), (goals - 0.34) / 2.0, torch.zeros_like(goals))
    return command, rate


def numpy_program(seconds, command):
    import numpy as np

    vx, yaw, goal = command
    changing = abs(goal - 0.34) > 1e-6

    def clip(x):
        return float(np.clip(x, 0.0, 1.0))

    height = 0.34 + (goal - 0.34) * clip((seconds - 1.0) / 2.0)
    linear = clip((seconds - 6.0) / 2.0) if changing else clip((seconds - 0.5) / 1.0)
    angular = (
        (clip((seconds - 11.0) / 2.0) if changing else clip((seconds - 5.0) / 2.0))
        if vx and yaw
        else linear
    )
    return np.array([vx * linear, yaw * angular, height])


def scored_start(command):
    vx, yaw, goal = command
    changing = abs(goal - 0.34) > 1e-6
    return (13.0 if vx and yaw else 10.0) if changing else (7.0 if vx and yaw else 5.0)
