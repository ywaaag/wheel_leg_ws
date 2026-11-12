"""分别保护运动与站立的训练统计候选，避免单一站立排名丢掉运动模型。"""

from __future__ import annotations

import copy
import math

from . import training_candidate as standing
from .progressive_course import AXES

VERSION = "motion_training_motion_and_standing_candidates_v2"
FAST_YAW_VERSION = "motion_training_motion_and_standing_yaw1to4_candidates_v3"
FAST_YAW_ANCHORS = ((0.25, 0.0), (-0.25, 0.0), (0.0, 1.0), (0.0, -1.0))


def configuration(*, fast_yaw=False):
    cfg = standing.configuration()
    cfg.update(
        version=VERSION,
        maximum_candidates=8,
        objectives=["motion", "standing"],
        primary_objective="motion",
        motion_rank=[
            "minimum_reached_magnitude",
            "sum_reached_magnitudes",
            "matched_low_speed_pass_fraction",
            "negative_standing_HF_tail_rad_s",
        ],
        milestone100="explicitly retained, even if not ranked best",
    )
    if fast_yaw:
        cfg.update(
            version=FAST_YAW_VERSION,
            motion_anchors=[list(x) for x in FAST_YAW_ANCHORS],
            standing_coverage="all C0 v2 cohorts including nominal/combined modal and continuous handover",
        )
    return cfg


def validate_configuration(value):
    if value != configuration(fast_yaw=value.get("version") == FAST_YAW_VERSION):
        raise ValueError("运动/站立双候选配置不一致")
    return value


def comparison_context(record):
    result = standing.comparison_context(record)
    fast_yaw = record["diagnostics"]["training_candidate_selection"]["version"] == FAST_YAW_VERSION
    result.update(
        version=FAST_YAW_VERSION if fast_yaw else VERSION,
        selection=configuration(fast_yaw=fast_yaw),
        source_policy_retention=record["assets"].get("source_policy_retention"),
    )
    return result


def empty_index(identity, *, fast_yaw=False, version=None):
    return dict(
        version=version or (FAST_YAW_VERSION if fast_yaw else VERSION),
        label=standing.LABEL,
        comparison_sha256=identity,
        candidates_by_objective={"motion": {}, "standing": {}},
        best_candidate=None,
        best_standing_candidate=None,
        independent_certificate=False,
        extra_simulation_steps=0,
    )


def summarize(review, *, DR_level, comparison_sha256):
    from .c0_fast_yaw_course import VERSION as COURSE

    # UNKNOWN is a valid zero/undersampled window; malformed evidence is not.
    for row in review["rows"]:
        for name in (
            "key",
            "command",
            "purpose",
            "profile",
            "failed_check_counts",
            "movement_components",
        ):
            if name not in row:
                raise ValueError(f"训练窗口缺少字段: {name}")
        if not isinstance(row["failed_check_counts"], dict) or not isinstance(
            row["movement_components"], dict
        ):
            raise ValueError("训练窗口行为统计字段损坏")
        for name in ("support", "continuous_loss"):
            if name not in row["failed_check_counts"]:
                raise ValueError("训练窗口缺少接触统计")
        if "episode_means" not in row["movement_components"]:
            raise ValueError("训练窗口缺少回合均值")
        counts = [
            row[name]
            for name in (
                "ended_episodes",
                "completed_episodes",
                "passed_episodes",
                "physical_quality_passed",
            )
        ]
        if (
            any(type(n) is not int or n < 0 for n in counts)
            or not counts[2] <= counts[3] <= counts[1] <= counts[0]
        ):
            raise ValueError("训练候选回合计数损坏")
        for name in ("quality_pass_fraction", "pass_fraction", "mean_HF_tail_rad_s"):
            value = row[name]
            if counts[0] == 0:
                if value is not None:
                    raise ValueError("零回合必须使用UNKNOWN")
            elif value is None or not math.isfinite(value):
                raise ValueError("训练候选物理统计损坏")
        if counts[0]:
            # Validate every nonempty cohort, including undersampled cohorts.
            standing._row(row, 0)
    fast_yaw = review.get("course_version") == COURSE
    result = standing.summarize(
        review,
        DR_level=DR_level,
        comparison_sha256=comparison_sha256,
        **dict(anchors=FAST_YAW_ANCHORS, coverage_version="motion_C0_standing_events50_motion50_v2")
        if fast_yaw
        else {},
    )
    result["version"] = FAST_YAW_VERSION if fast_yaw else VERSION
    reaches = {axis: 0.0 for axis in AXES}
    for row in review["rows"]:
        axis = row.get("axis")
        fraction = row.get("progression_pass_fraction")
        if (
            axis in reaches
            and row["profile"] == -1
            and row["ended_episodes"] >= 32
            and fraction is not None
        ) and (
            fraction >= 0.70
            and row["quality_pass_fraction"] >= 0.70
            and row["completed_episodes"] / row["ended_episodes"] >= 0.95
        ):
            reaches[axis] = max(reaches[axis], max(map(abs, row["command"])))
    result.update(
        valid_ended_episodes=sum(row["ended_episodes"] for row in review["rows"]),
        reached_magnitudes=reaches,
        standing_rank=result.get("rank"),
        motion_rank=[
            min(reaches.values()),
            sum(reaches.values()),
            result.get("matched_low_speed_pass_fraction", 0.0),
            result.get("negative_standing_HF_tail_rad_s", 0.0),
        ],
    )
    return result


def consider(index, evidence, reference, provenance):
    updated = (
        copy.deepcopy(index)
        if index
        else empty_index(
            evidence["comparison_sha256"], fast_yaw=evidence["version"] == FAST_YAW_VERSION
        )
    )
    if (
        updated["version"] != evidence["version"]
        or updated["comparison_sha256"] != evidence["comparison_sha256"]
    ):
        raise ValueError("运动候选比较域不一致")
    changed = []
    if evidence["eligible"]:
        if not reference or len(reference.get("sha256", "")) != 64:
            raise ValueError("候选缺少完整checkpoint")
        for objective in ("motion", "standing"):
            if not evidence.get("eligible_by_objective", {}).get(objective, True):
                continue
            pool = updated["candidates_by_objective"][objective]
            key = str(evidence["DR_level"])
            previous = pool.get(key)
            rank = objective + "_rank"
            if previous is None or tuple(evidence[rank]) > tuple(previous["evidence"][rank]):
                pool[key] = dict(
                    reference,
                    label=standing.LABEL,
                    objective=objective,
                    evidence=copy.deepcopy(evidence),
                    provenance=copy.deepcopy(provenance),
                    independent_certificate=False,
                )
                changed.append(objective)
    for objective, field in (("motion", "best_candidate"), ("standing", "best_standing_candidate")):
        pool = updated["candidates_by_objective"][objective]
        updated[field] = pool[max(pool, key=int)] if pool else None
    updated["last_review"] = dict(
        through_update=evidence["through_update"],
        checkpoint=reference,
        changed=bool(changed),
        changed_objectives=changed,
        reason=evidence["reason"],
        evidence=copy.deepcopy(evidence),
    )
    return updated
