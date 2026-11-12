"""Declared Teacher decoder -> physical target -> ABI4 raw labels; no fitting."""

import numpy as np
import torch

from native_stand.tasks.jump.unified import labels_from_physical
from native_stand.tasks.motion.action_contract import ActionDecoder


def convert_teacher_mean(mean, teacher_contract, *, height):
    decoder = ActionDecoder.from_contract(teacher_contract)
    values = torch.as_tensor(mean, dtype=torch.float32)
    physical = decoder.decode(values, torch.as_tensor(height, dtype=torch.float32))[
        "protected_target"
    ]
    labels = labels_from_physical(physical.detach().cpu().numpy())
    if not np.all(labels["representable"]):
        raise ValueError("Teacher物理目标超Jump保护范围；不裁标签")
    return labels


def inspect_BC_source(path, expected_sha):
    """Only inspect supervised state; this function never creates a PPO optimizer."""
    from native_stand.artifacts.io import file_sha256

    if file_sha256(path) != expected_sha:
        raise ValueError("BC来源身份不同")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if set(value) != {
        "actor",
        "optimizer",
        "torch_rng",
        "generator_rng",
        "supervised_updates",
        "config",
    }:
        raise ValueError("BC恢复状态schema不同")
    return dict(
        kind="BC_resume_only",
        supervised_updates=value["supervised_updates"],
        PPO_updates=0,
        optimizer_schema="supervised_Adam_not_PPO",
        actor=value["actor"],
    )
