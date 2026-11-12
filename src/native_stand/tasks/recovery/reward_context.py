"""Explicit v3 Critic suffix; Actor/physical action contract is unchanged."""

import torch

VERSION = "recovery_reward_causal904_v3"
WIDTH = 904
FIELDS = (
    ("reward_last_phi", 1),
    ("reward_last_target", 6),
    ("reward_last_delta", 6),
    ("reward_target_initialized", 1),
    ("physical_HP10_low", 4),
    ("prefix_duration", 1),
    ("next_student_owner", 1),
    ("pulse_onset", 1),
    ("pulse_pattern", 4),
    ("pulse_enabled", 1),
)
assert sum(w for _, w in FIELDS) == WIDTH - 878


def sources(ex, life, *, prefix_s, next_student, pulse_onset, pulse_pattern, pulse_enabled):
    scale = ex.raw.new_tensor([1.5, 1.5, 100.0, 1.5, 1.5, 100.0])
    return dict(
        reward_last_phi=life.last_phi[:, None],
        reward_last_target=life.last_target / scale,
        reward_last_delta=life.last_delta / scale,
        reward_target_initialized=life.valid_target[:, None].to(ex.raw.dtype),
        physical_HP10_low=ex.hp_low * 0.05,
        prefix_duration=prefix_s[:, None] / 23,
        next_student_owner=next_student[:, None].to(ex.raw.dtype),
        pulse_onset=pulse_onset[:, None] / 23,
        pulse_pattern=pulse_pattern,
        pulse_enabled=pulse_enabled[:, None].to(ex.raw.dtype),
    )


def append(base, values):
    if base.shape[-1] != 878 or set(values) != {k for k, _ in FIELDS}:
        raise ValueError("Explicit 878 prefix + all reward context fields required")
    rows = [base]
    for k, w in FIELDS:
        x = values[k]
        if x.shape != (len(base), w) or not bool(torch.isfinite(x).all()):
            raise ValueError("Reward context source " + k)
        rows.append(x)
    return torch.cat(rows, -1)
