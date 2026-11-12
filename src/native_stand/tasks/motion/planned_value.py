"""Known episode endpoints belong to the privileged value state, never Actor."""

from __future__ import annotations

import torch

VERSION = "motion48_height_bank_planned_causal877_v2"
BASE_DIM = 875
DIM = 877
FIELDS = (("planned_episode_vx_over4", 1), ("planned_episode_yaw_over4", 1))


def append(prefix, targets):
    if (
        prefix.ndim != 2
        or prefix.shape[1] != BASE_DIM
        or targets.shape != (len(prefix), 2)
        or targets.device != prefix.device
        or targets.dtype != prefix.dtype
    ):
        raise ValueError("Planned value context requires875 prefix and two same-device targets")
    return torch.cat((prefix, targets / 4.0), -1)
