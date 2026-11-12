"""分支内固定复习域的双候选；站立与主任务资格独立，不授独立证书。"""

from __future__ import annotations

import copy
import math

from . import maneuver_candidates, motion_candidates
from .maneuver_statistics import RESPONSE_VERSION

VERSION = "motion48_split_training_candidates_v1"
empty_index = motion_candidates.empty_index
consider = motion_candidates.consider


def configuration():
    return dict(
        version=VERSION,
        every_updates=100,
        maximum_candidates=8,
        objectives=["motion", "standing"],
        primary_objective="motion",
        minimum_ended=16,
        minimum_complete_fraction=0.90,
        minimum_quality_fraction=0.70,
        task_pass_fraction=0.70,
        minimum_dual_support_fraction=0.90,
        label="训练统计候选",
        independent_certificate=False,
        extra_simulation_steps=0,
        compare="branch-specific fixed anchors at matching DR tier; no cross-branch ranking",
    )


def validate_configuration(value):
    if value != configuration():
        raise ValueError("Motion/Spin分支候选配置不同")


def comparison_context(record):
    from native_stand.artifacts.retention_identity import semantic_identity

    result = maneuver_candidates.comparison_context(record)
    result.update(
        branch=record["assets"]["branch_course"]["branch"],
        course=copy.deepcopy(record["assets"]["branch_course"]),
        retention_semantic_sha256=semantic_identity(record["assets"]["source_policy_retention"]),
        selection=copy.deepcopy(record["diagnostics"]["training_candidate_selection"]),
    )
    return result


def summarize(review, *, DR_level, comparison_sha256):
    if review.get("version") != RESPONSE_VERSION:
        raise ValueError("分支候选不能使用旧/不完整统计合同")
    cfg = configuration()
    result = dict(
        version=VERSION,
        comparison_sha256=comparison_sha256,
        through_update=review["through_update"],
        stage_revision=review["stage_revision"],
        DR_level=DR_level,
        label="训练统计候选",
        independent_certificate=False,
        extra_simulation_steps=0,
        eligible=False,
        eligible_by_objective={},
        task_passed_by_objective={},
        objective_evidence={},
        reasons=[],
        thresholds={
            k: cfg[k]
            for k in (
                "minimum_ended",
                "minimum_complete_fraction",
                "minimum_quality_fraction",
                "task_pass_fraction",
                "minimum_dual_support_fraction",
            )
        },
    )
    for row in review["rows"]:
        n, completed = row["ended_episodes"], row["completed_episodes"]
        if type(n) is not int or type(completed) is not int or not 0 <= completed <= n:
            raise ValueError("分支候选结束回合计数错误")
        for field in (
            "pass_fraction",
            "quality_pass_fraction",
            "mean_dual_support_fraction",
            "physical_HF_RMS_rad_s",
        ):
            value = row.get(field)
            if n == 0 and value is not None:
                raise ValueError("零回合候选必须是UNKNOWN")
            if value is not None and (
                not math.isfinite(value)
                or value < 0
                or (field != "physical_HF_RMS_rad_s" and value > 1)
            ):
                raise ValueError("候选实际物理统计超范围")
        passed, quality = row["pass_fraction"], row["quality_pass_fraction"]
        if (
            n
            and passed is not None
            and quality is not None
            and (passed > quality + 1e-6 or quality > completed / n + 1e-6)
        ):
            raise ValueError("候选任务/物理质量通过数与完整回合矛盾")
    active = [r for r in review["rows"] if r["slot_count"] > 0]
    groups = dict(
        standing=[r for r in active if r["axis"] == "standing"],
        motion=[r for r in active if r["purpose"].startswith("anchor_")],
    )
    for objective, rows in groups.items():
        covered = bool(rows) and all(
            r["ended_episodes"] >= cfg["minimum_ended"]
            and all(
                r.get(name) is not None and math.isfinite(r[name])
                for name in (
                    "pass_fraction",
                    "quality_pass_fraction",
                    "mean_dual_support_fraction",
                    "physical_HF_RMS_rad_s",
                )
            )
            for r in rows
        )
        eligible = covered and all(
            r["completed_episodes"] / r["ended_episodes"] >= cfg["minimum_complete_fraction"]
            and r["quality_pass_fraction"] >= cfg["minimum_quality_fraction"]
            and r["mean_dual_support_fraction"] >= cfg["minimum_dual_support_fraction"]
            for r in rows
        )
        result["eligible_by_objective"][objective] = eligible
        task_passed = covered and all(r["pass_fraction"] >= cfg["task_pass_fraction"] for r in rows)
        result["task_passed_by_objective"][objective] = task_passed
        failures = []
        for row in rows:
            n = row["ended_episodes"]
            checks = [
                ("coverage_insufficient", "ended_episodes", n, cfg["minimum_ended"]),
                (
                    "quality_insufficient",
                    "complete_fraction",
                    row["completed_episodes"] / n if n else None,
                    cfg["minimum_complete_fraction"],
                ),
                (
                    "quality_insufficient",
                    "quality_pass_fraction",
                    row["quality_pass_fraction"],
                    cfg["minimum_quality_fraction"],
                ),
                (
                    "quality_insufficient",
                    "mean_dual_support_fraction",
                    row["mean_dual_support_fraction"],
                    cfg["minimum_dual_support_fraction"],
                ),
                (
                    "task_insufficient",
                    "pass_fraction",
                    row["pass_fraction"],
                    cfg["task_pass_fraction"],
                ),
            ]
            for reason, metric, actual, threshold in checks:
                if reason != "coverage_insufficient" and n < cfg["minimum_ended"]:
                    continue  # A small measured sample is not a known task/quality failure.
                if actual is None or actual < threshold:
                    failures.append(
                        dict(
                            reason="coverage_insufficient" if actual is None else reason,
                            key=copy.deepcopy(row["key"]),
                            metric=metric,
                            actual=actual,
                            minimum=threshold,
                            status="UNKNOWN" if actual is None else "BELOW_THRESHOLD",
                        )
                    )
            if row.get("physical_HF_RMS_rad_s") is None:
                failures.append(
                    dict(
                        reason="coverage_insufficient",
                        key=copy.deepcopy(row["key"]),
                        metric="physical_HF_RMS_rad_s",
                        actual=None,
                        required="finite measurement",
                        status="UNKNOWN",
                    )
                )
        if not rows:
            failures.append(
                dict(
                    reason="coverage_insufficient",
                    metric="active_groups",
                    actual=0,
                    minimum=1,
                    status="UNKNOWN",
                )
            )
        result["objective_evidence"][objective] = dict(
            covered=covered,
            physical_candidate_eligible=eligible,
            training_task_passed=task_passed,
            status="UNKNOWN" if not covered else ("TASK_PASS" if task_passed else "TASK_GAP"),
            unpassed_targets=[
                dict(
                    key=r["key"],
                    pass_fraction=r["pass_fraction"],
                    ended_episodes=r["ended_episodes"],
                    completed_episodes=r["completed_episodes"],
                )
                for r in rows
                if r["pass_fraction"] is None or r["pass_fraction"] < cfg["task_pass_fraction"]
            ],
            independent_certificate=False,
            reasons=failures,
        )
        result["reasons"].extend(dict(objective=objective, **r) for r in failures)
        result[objective + "_rank"] = (
            [
                min(r["pass_fraction"] for r in rows),
                min(r["quality_pass_fraction"] for r in rows),
                min(r["mean_dual_support_fraction"] for r in rows),
                -max(r["physical_HF_RMS_rad_s"] for r in rows),
            ]
            if covered
            else None
        )
    result["eligible"] = any(result["eligible_by_objective"].values())
    if result["eligible"]:
        result["reason"] = (
            "covered_branch_fixed_review"
            if all(result["task_passed_by_objective"].values())
            else "covered_objective_candidates_with_separate_task_status"
        )
    else:
        result["reason"] = next(
            (
                name
                for name in ("coverage_insufficient", "quality_insufficient", "task_insufficient")
                if any(r["reason"] == name for r in result["reasons"])
            ),
            "no_eligible_objective",
        )
    result["frontier_diagnostics"] = [
        dict(key=r["key"], pass_fraction=r["pass_fraction"], ended_episodes=r["ended_episodes"])
        for r in active
        if r["purpose"].startswith("focus_")
    ]
    return result
