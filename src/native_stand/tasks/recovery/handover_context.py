"""904 causal prefix plus observable ownership/confirmation state."""

import torch

VERSION = "recovery_handover_causal908_v4"
WIDTH = 908
FIELDS = (
    ("Motion_owner", 1),
    ("handover_age", 1),
    ("Motion_confirmation_age", 1),
    ("entry_count", 1),
)


def append(base, life):
    if base.shape[-1] != 904:
        raise ValueError("Full reward904 prefix required")
    age = torch.where(life.handover_active, life.elapsed - life.handover_time, 0.0)
    suffix = torch.stack(
        (life.handover_active.to(base.dtype), age / 23, life.motion_hold / 10, life.entry_events),
        -1,
    )
    if not bool(torch.isfinite(suffix).all()):
        raise ValueError("Nonfinite handover state")
    return torch.cat((base, suffix), -1)
