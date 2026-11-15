"""Unassisted result-based Jump recipe; no trainer, control replacement or IO."""

from dataclasses import replace

from .dense_reward import RewardConfig

PROFILE = "jump_unassisted_outcome_v1"
REWARD = "jump_unassisted_outcome_reward_v3"
CONFIG_ID = 0x4A050402
REQUEST_AFTER_MS = 300


def enabled(job):
    value = job.assets.get("jump_objective")
    if value not in (None, PROFILE):
        raise ValueError("Unknown explicit Jump objective")
    return value == PROFILE


def reward_config():
    return replace(
        RewardConfig(),
        require_crouch=False,
        goal_gate_landing=True,
        no_takeoff_deadline_s=2.0,
        w_crouch=0.0,
        w_push=0.0,
        w_extension=0.0,
        w_air_shape=0.0,
        w_cushion=0.0,
        w_settle=0.0,
        w_velocity_potential=1.0,
        potential_events=True,
        w_takeoff=4.0,
        w_air_rise=6.0,
        w_apex=4.0,
        w_landing=8.0,
        w_handover=42.0,
        failure_cost=24.0,
        timeout_cost=32.0,
        w_time=1.0,
        # Moderate drift is allowed. Real landing/contact/posture remain gates.
        stable_vxy_m_s=0.5,
        w_drift=0.05,
    )


def controller_record():
    return dict(
        version=PROFILE,
        config_id=CONFIG_ID,
        request_after_ms=REQUEST_AFTER_MS,
        request="pending_until_safe_acceptance_or_episode_end",
        crouch="no_mandatory_pose_or_phase",
        push_timeout_ms=1200,
        no_takeoff_deadline_s=2.0,
        external_assist=False,
    )
