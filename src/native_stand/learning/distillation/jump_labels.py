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


def motion_teacher_observation(observation, version):
    """Current physical Jump sensor state -> declared Motion31 task flags.

    Shared first25 columns retain actual previous adopted physical targets.
    Phase columns are not silently supplied to the Motion network.
    """
    if version != "jump37_to_motion31_v1" or observation.shape[-1] != 37:
        raise ValueError("Explicit Jump37 to Motion31 adapter required")
    result = observation[:, :31].clone()
    result[:, 25:31] = result.new_tensor([1, 0, 0, 1, 0, 0])
    return result


def executor_condition_names(version):
    if version != "motion_no_C_spring_to_jump_C_spring_v1":
        raise ValueError("Explicit supported Teacher executor transform required")
    return tuple("jump_spring_model_nm_" + str(i) for i in range(4))


def match_jump_executor(physical, conditions, mask, version):
    """Match pre-cap leg torque between Motion no-C-spring and actual Jump C.

    Target correction is a declared supervised label transform, not an added
    deployed controller. The Student continues to emit direct physical targets
    through unchanged C/PD/V4. Conditions are from the pre-action C input state.
    """
    names = executor_condition_names(version)
    compensation = torch.stack(
        [torch.as_tensor(conditions[k], device=physical.device)[mask] for k in names], -1
    ).to(physical.dtype)
    if not torch.isfinite(compensation).all():
        raise ValueError("Finite actual executor compensation required")
    result = physical.clone()
    result[:, [0, 1, 3, 4]] -= compensation / 40.0
    guard = result.new_tensor([1.5, 1.5, 80, 1.5, 1.5, 80])
    if (result.abs() > guard).any():
        raise ValueError("Teacher executor target outside Jump guard; no clipping")
    return result
