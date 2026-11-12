"""Five real height-schedule sources appended to the causal904 Critic."""

import torch

VERSION = "recovery_standalone_height_causal909_v5"
WIDTH = 909
FIELDS = (
    ("initial_height_command", 1),
    ("current_height_command", 1),
    ("height_lower_waypoint", 1),
    ("height_upper_waypoint", 1),
    ("height_schedule_dynamic", 1),
)
REWARD_VERSION = "recovery_standalone_height_own_hold_v5"


def command(initial, dynamic, time):
    # Match the declared continuous .34 -> .32 -> .36 -> .34 supervision.
    h = initial.clone()
    t = time
    v = torch.where(
        t < 2,
        0.34,
        torch.where(
            t < 4,
            0.34 - (t - 2) * 0.01,
            torch.where(
                t < 6,
                0.32,
                torch.where(
                    t < 8,
                    0.32 + (t - 6) * 0.02,
                    torch.where(t < 10, 0.36, torch.where(t < 12, 0.36 - (t - 10) * 0.01, 0.34)),
                ),
            ),
        ),
    )
    return torch.where(dynamic, v, h)


def append(base, initial, current, dynamic):
    if base.shape[-1] != 904:
        raise ValueError("Height context requires actual causal904 prefix")
    values = torch.stack(
        (
            initial / 0.5,
            current / 0.5,
            initial.new_full(initial.shape, 0.32 / 0.5),
            initial.new_full(initial.shape, 0.36 / 0.5),
            dynamic.to(initial.dtype),
        ),
        -1,
    )
    if values.shape != (len(base), 5) or not bool(torch.isfinite(values).all()):
        raise ValueError("Finite actual height schedule sources required")
    return torch.cat((base, values), -1)
