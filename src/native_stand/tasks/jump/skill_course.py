"""Reachable Jump objectives with physical event rewards and explicit goals."""

import math
from dataclasses import dataclass, replace

from .adaptive_course import AdaptiveCourse, Level
from .dense_reward import RewardConfig, TaskSpec

PROFILE = "jump_first_takeoff_course_v1"
REWARD = "jump_physical_skill_events_reward_v1"
CRITIC = "jump_isaac_skill_execution_critic_v2"
EXPLORATION = "fixed_physical_Normal_v1"
TAKEOFF, LANDING, COMPLETE = range(3)


@dataclass(frozen=True)
class SkillLevel(Level):
    objective: int = COMPLETE

    def __post_init__(self):
        if type(self.objective) is not int or self.objective not in (TAKEOFF, LANDING, COMPLETE):
            raise ValueError("Explicit takeoff/landing/complete Jump objective required")


@dataclass(frozen=True)
class SkillTaskSpec(TaskSpec):
    objective: int = COMPLETE
    takeoff_clearance_m: float = 0.005

    def __post_init__(self):
        super().__post_init__()
        if type(self.objective) is not int or self.objective not in (TAKEOFF, LANDING, COMPLETE):
            raise ValueError("Unknown Jump skill objective")
        if not 0.002 < self.takeoff_clearance_m <= self.target_clearance_m:
            raise ValueError("Explicit real-wheel clearance for active takeoff required")


def enabled(job):
    return job.assets.get("jump_objective") == PROFILE


def reward_config():
    return replace(
        RewardConfig(),
        require_crouch=False,
        goal_gate_landing=False,
        no_takeoff_deadline_s=3.0,
        airborne_dwell_s=0.020,
        takeoff_min_vz_m_s=0.15,
        w_crouch=0.0,
        w_push=0.0,
        w_extension=0.0,
        w_air_shape=0.0,
        w_cushion=0.0,
        w_settle=0.0,
        w_velocity_potential=1.0,
        potential_events=False,
        w_takeoff=6.0,
        w_air_rise=4.0,
        w_apex=2.0,
        w_landing=10.0,
        w_handover=42.0,
        complete_partial_reward_scale=0.25,
        failure_cost=40.0,
        timeout_cost=40.0,
        w_time=1.0,
        stable_vxy_m_s=0.5,
        w_drift=0.05,
    )


def validate(job):
    if not enabled(job):
        return
    course = job.assets["jump_course"]
    validated_course = AdaptiveCourse(seed=0, **course)
    levels = [SkillLevel(**row) for row in course["levels"]]
    for row in levels:
        SkillTaskSpec(
            "ground_active_jump",
            course.get("config", {}).get("active_horizon_s", 16.0),
            1.0,
            target_rise_m=row.rise_m,
            target_clearance_m=row.clearance_m,
            air_leg_m=row.air_leg_m,
            objective=row.objective,
        )
    if (
        len(levels) < 3
        or [row.objective for row in levels[:2]] != [TAKEOFF, LANDING]
        or any(row.objective != COMPLETE for row in levels[2:])
        or any(row.active_time_share != 1.0 for row in levels)
        or course.get("config", {}).get("require_landing_rehearsal", True)
    ):
        raise ValueError("Jump skill course must progress takeoff -> landing -> complete")
    c = reward_config()
    partial = c.w_takeoff + c.w_air_rise + c.w_apex
    complete_partial = (partial + c.w_landing) * c.complete_partial_reward_scale
    terminal = min(c.failure_cost, c.timeout_cost) * math.exp(
        -c.discount_rate_s * validated_course.config.active_horizon_s
    )
    if terminal <= max(partial, complete_partial):
        raise ValueError("Discounted terminal cost must exceed all incomplete skill credit")
    if job.assets.get("jump_exploration") != EXPLORATION:
        raise ValueError("New Jump skill course requires explicit fixed physical exploration")
    if job.ppo.entropy_coef != 0:
        raise ValueError("Fixed Normal std has no mean entropy gradient; declare coefficient zero")
    if any(
        not math.isfinite(v) or v <= 0
        for v in (job.ppo.exploration_leg_rad, job.ppo.exploration_wheel_rad_s)
    ):
        raise ValueError("Finite positive physical exploration std required")
    if job.reward.version != REWARD or job.model.critic_privileged_version != CRITIC:
        raise ValueError("Jump skill reward and causal Critic must have distinct identities")
    if job.backend != "isaac" or (job.model.critic_input, job.observation.critic_dim) != (
        5593,
        5593,
    ):
        raise ValueError("Jump skill Critic requires its explicit Isaac5593 layout")
