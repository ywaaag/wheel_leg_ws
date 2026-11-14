"""Explicit physical-target demonstration shift; never an online controller."""

import math

VERSION = "motion_height_support_teacher_v1"


def validate(record):
    if (
        not isinstance(record, dict)
        or set(record) != {"version", "height_m", "offset_rad", "evidence_sha256"}
        or record["version"] != VERSION
    ):
        raise ValueError("Unknown support Teacher shift declaration")
    nodes, offsets = record["height_m"], record["offset_rad"]
    if len(nodes) < 2 or len(nodes) != len(offsets) or nodes[0] != 0.2 or nodes[-1] != 0.4:
        raise ValueError("Support Teacher shift must cover .20-.40m")
    if not all(
        isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x) for x in nodes
    ) or any(a >= b for a, b in zip(nodes, nodes[1:], strict=False)):
        raise ValueError("Support Teacher heights must be finite and increasing")
    for row in offsets:
        if len(row) != 4 or not all(
            isinstance(x, int | float)
            and not isinstance(x, bool)
            and math.isfinite(x)
            and abs(x) <= 0.35
            for x in row
        ):
            raise ValueError("Support Teacher shift needs bounded four-leg physical offsets")
        if abs(row[0] + row[2]) > 1e-9 or abs(row[1] - row[3]) > 1e-9:
            raise ValueError("Support Teacher standing shift must be mirrored")
    sha = record["evidence_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Support Teacher shift needs evidence content identity")
    return record


def tensor_shift(height, record):
    import torch

    if not bool(torch.isfinite(height).all()) or bool(
        ((height < 0.2 - 1e-6) | (height > 0.4 + 1e-6)).any()
    ):
        raise ValueError("Support Teacher height out of declared domain")
    nodes = height.new_tensor(record["height_m"])
    offsets = height.new_tensor(record["offset_rad"])
    i = torch.searchsorted(nodes, height.contiguous(), right=True).clamp(1, len(nodes) - 1)
    f = (height - nodes[i - 1]) / (nodes[i] - nodes[i - 1])
    return offsets[i - 1] + f[:, None] * (offsets[i] - offsets[i - 1])


def numpy_shift(height, record):
    import numpy as np

    if not np.isfinite(height) or not 0.2 - 1e-6 <= height <= 0.4 + 1e-6:
        raise ValueError("Support Teacher height out of declared domain")
    return np.array(
        [
            np.interp(height, record["height_m"], column)
            for column in zip(*record["offset_rad"], strict=True)
        ]
    )
