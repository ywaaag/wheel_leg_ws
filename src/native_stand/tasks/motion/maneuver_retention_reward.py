"""Versioned coarse drift tail; keep banking, symmetry and physical HF semantics."""

from __future__ import annotations

import torch

from . import maneuver_reward
from .maneuver_targets import mode_masks
from .scale_corrections import huber, replace_density

VERSION = "motion48_bank_anchored_coarse_drift_reward_v2"


def configuration():
    return dict(
        **{**maneuver_reward.configuration(), "version": VERSION},
        stand_drift_tail_weight=1.5,
        stand_drift_tail_scale_mps=0.30,
        nominal_source_retention="balanced qualified training-only source states",
        precision_parking=False,
    )


def standing_density(vx, config):
    excess = torch.relu(vx.abs() - config["stand_drift_deadband_mps"])
    return 4.0 - config["stand_drift_tail_weight"] * huber(
        excess / config["stand_drift_tail_scale_mps"]
    )


def apply(env, scale_config, config):
    if config != configuration():
        raise ValueError("Explicit anchored coarse-drift reward configuration required")
    maneuver_reward.apply(env, scale_config)
    result = env.student_reward_result
    standing = mode_masks(env.motion_action_command)["standing"]
    current = result["density"][:, result["fields"].index("linear_tracking")].clone()
    value = torch.where(standing, standing_density(env.recovery_snapshot[:, 7], config), current)
    replace_density(result, "linear_tracking", value)
    result["maneuver_reward_version"] = VERSION


def readback(env):
    result = env.student_reward_result
    if result.get("maneuver_reward_version") != VERSION:
        raise RuntimeError("The declared coarse-drift reward did not reach the actual result")
    active = mode_masks(env.motion_action_command)["standing"] & result["eligible"]
    actual = result["density"][:, result["fields"].index("linear_tracking")]
    expected = standing_density(env.recovery_snapshot[:, 7], configuration())
    if not torch.allclose(actual[active], expected[active], atol=1e-6, rtol=0):
        raise RuntimeError("Actual zero-mode drift reward differs from declaration")
    return dict(
        version=VERSION,
        actual_density_verified=True,
        standing_samples=int(active.sum()),
        deadband_mps=0.15,
        tail_weight=1.5,
        tail_scale_mps=0.30,
        policy_dt_applied_once=True,
    )
