"""Training-only separate standing/motion nominees, compared within DR tier."""

from __future__ import annotations

import copy
import math

from . import motion_candidates
from .maneuver_statistics import VERSION as STATISTICS

VERSION = "motion48_training_standing_and_maneuver_candidates_v1"
RESPONSE_VERSION = "motion48_response_checked_training_candidates_v2"


def configuration(*, version=VERSION):
    if version not in (VERSION, RESPONSE_VERSION):
        raise ValueError("Unknown candidate version")
    return dict(
        version=version,
        every_updates=100,
        maximum_candidates=8,
        objectives=["motion", "standing"],
        primary_objective="motion",
        minimum_ended=16,
        label="训练统计候选",
        independent_certificate=False,
        extra_simulation_steps=0,
        evidence="complete stochastic training episodes, not one deterministic checkpoint",
        compare="fixed review tasks at matching DR tier; achieved maneuver quality reported separately",
    )


def validate_configuration(value):
    if value != configuration(version=value.get("version")):
        raise ValueError("Maneuver candidate configuration mismatch")


def comparison_context(record):
    version = record["diagnostics"]["training_candidate_selection"]["version"]
    return dict(
        version=version,
        contracts={
            k: copy.deepcopy(record[k])
            for k in ("model", "observation", "action", "control", "reward", "curriculum", "ppo")
        },
        physics={
            k: copy.deepcopy(record["assets"][k])
            for k in (
                "output_contract_sha256",
                "runtime_template_sha256",
                "delayed_execution",
                "motor_limits",
            )
        },
        envs=record["runtime"]["envs"],
    )


def empty_index(identity, *, version=VERSION):
    result = motion_candidates.empty_index(identity)
    result["version"] = version
    return result


def summarize(review, *, DR_level, comparison_sha256):
    from .maneuver_statistics import RESPONSE_VERSION as NEW_STATISTICS

    version = RESPONSE_VERSION if review.get("version") == NEW_STATISTICS else VERSION
    r = dict(
        version=version,
        eligible=False,
        label="训练统计候选",
        DR_level=DR_level,
        comparison_sha256=comparison_sha256,
        through_update=review["through_update"],
        stage_revision=review["stage_revision"],
        independent_certificate=False,
        extra_simulation_steps=0,
        reason="insufficient_fixed_review_coverage",
    )
    if review.get("version") not in (STATISTICS, NEW_STATISTICS):
        return r
    fixed = [
        x
        for x in review["rows"]
        if x["axis"] in ("standing", "forward", "backward") and x["slot_count"] > 0
    ]
    if not fixed or any(
        x["ended_episodes"] < 16 or x["pass_fraction"] is None or x["physical_HF_RMS_rad_s"] is None
        for x in fixed
    ):
        return r
    stand = [x for x in fixed if x["axis"] == "standing"]
    motion = [x for x in fixed if x["axis"] in ("forward", "backward")]
    if not stand or len(motion) != 2:
        return r
    values = [x["mean_dual_support_fraction"] for x in fixed]
    if any(not math.isfinite(v) for v in values):
        return r
    eligible = (
        min(values) >= 0.90
        and min(x["completed_episodes"] / x["ended_episodes"] for x in stand) >= 0.90
    )
    hf = max(x["physical_HF_RMS_rad_s"] for x in stand)
    r.update(
        eligible=eligible,
        reason="covered_fixed_review" if eligible else "fixed_review_quality_gap",
        standing_rank=[
            min(x["pass_fraction"] for x in stand),
            sum(x["pass_fraction"] for x in stand) / len(stand),
            min(values),
            -hf,
        ],
        motion_rank=[
            min(x["pass_fraction"] for x in motion),
            min(x["quality_pass_fraction"] for x in motion),
            min(values),
            -hf,
        ],
        achieved_new_task_rows=[
            dict(key=x["key"], pass_fraction=x["pass_fraction"])
            for x in review["rows"]
            if x["axis"] not in ("standing", "forward", "backward")
        ],
    )
    return r


consider = motion_candidates.consider
