"""已有on-policy窗口的候选排序；不授予独立能力结论、不推进物理。"""

from __future__ import annotations

import copy
import math

from .progressive_course import AMPLITUDES, FAMILIES

VERSION = "motion_training_candidate_v1"
LABEL = "训练统计候选"
ANCHORS = ((0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25))
RANK_FIELDS = (
    "worst_standing_pass_fraction",
    "balanced_standing_pass_fraction",
    "standing_quality_fraction",
    "standing_dual_support_fraction",
    "negative_standing_HF_tail_rad_s",
    "matched_low_speed_pass_fraction",
)


def configuration():
    return dict(
        version=VERSION,
        every_updates=100,
        maximum_candidates=len(AMPLITUDES),
        minimum_nominal_ended=32,
        minimum_other_ended=16,
        minimum_standing_complete_fraction=0.95,
        minimum_nominal_quality_fraction=0.70,
        minimum_standing_support_pass_fraction=0.95,
        rank_fields=list(RANK_FIELDS),
        motion_anchors=[list(x) for x in ANCHORS],
        label=LABEL,
        evidence="stochastic on-policy window spanning policy updates",
        independent_certificate=False,
        extra_simulation_steps=0,
        DR_comparison="one candidate per strength; primary is highest covered eligible strength",
    )


def validate_configuration(value):
    if value != configuration():
        raise ValueError("训练统计候选配置与显式版本不一致")
    return value


def comparison_context(record):
    """只纳入行为/采样契约，不把run名、预算、模型更新号当比较条件。"""
    assets = record["assets"]
    return dict(
        version=VERSION,
        contracts={
            k: copy.deepcopy(record[k])
            for k in ("model", "observation", "action", "control", "reward", "ppo", "curriculum")
        },
        runtime={k: record["runtime"][k] for k in ("envs", "steps_per_env")},
        assets={
            k: copy.deepcopy(assets.get(k))
            for k in (
                "center",
                "runtime_template_sha256",
                "frozen_source_root",
                "frozen_bundle",
                "delayed_execution",
                "wheel_identification_sha256",
            )
        },
        movement_components_version=record["diagnostics"].get("movement_components_version"),
    )


def _number(value):
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("nonfinite_or_missing_statistic")
    return value


def _row(row, minimum):
    ended = _number(row["ended_episodes"])
    if ended < minimum or int(ended) != ended:
        raise ValueError("insufficient_ended_episodes")
    for name in ("completed_episodes", "passed_episodes", "physical_quality_passed"):
        value = _number(row[name])
        if int(value) != value or not 0 <= value <= ended:
            raise ValueError("invalid_episode_count")
    if (
        row["passed_episodes"] > row["physical_quality_passed"]
        or row["physical_quality_passed"] > row["completed_episodes"]
    ):
        raise ValueError("inconsistent_episode_counts")
    failures = row["failed_check_counts"]
    for name in ("support", "continuous_loss"):
        if not 0 <= _number(failures[name]) <= ended:
            raise ValueError("invalid_support_count")
    means = row["movement_components"]["episode_means"]
    dual = _number(means["dual_support_fraction"])
    hf = _number(row["mean_HF_tail_rad_s"])
    if not 0 <= dual <= 1 or hf < 0:
        raise ValueError("invalid_physical_statistic")
    return dict(
        ended=ended,
        complete=row["completed_episodes"] / ended,
        passed=row["passed_episodes"] / ended,
        quality=row["physical_quality_passed"] / ended,
        support=1 - min(ended, failures["support"] + failures["continuous_loss"]) / ended,
        dual=dual,
        HF=hf,
    )


def summarize(review, *, DR_level, comparison_sha256, anchors=ANCHORS, coverage_version=None):
    """固定站立任务与固定低速锚点，避免课程中新端点稀释总体通过率。"""
    result = dict(
        version=VERSION,
        label=LABEL,
        eligible=False,
        comparison_sha256=comparison_sha256,
        DR_level=DR_level,
        through_update=review.get("through_update"),
        stage_revision=review.get("stage_revision"),
        independent_certificate=False,
        extra_simulation_steps=0,
    )
    try:
        if (
            review.get("version") != "motion_progressive_complete_episode_v1"
            or review.get("independent_evaluation") is not False
            or review.get("extra_simulation_steps") != 0
        ):
            raise ValueError("unsupported_training_window")
        through = _number(review["through_update"])
        if through < 100 or int(through) != through:
            raise ValueError("invalid_training_window_update")
        if type(DR_level) is not int or not 0 <= DR_level < len(AMPLITUDES):
            raise ValueError("invalid_DR_level")
        if not isinstance(comparison_sha256, str) or len(comparison_sha256) != 64:
            raise ValueError("missing_comparison_identity")
        expected = {"clean_standing": -1, "dq_plus_general_DR": 0}
        course_version = coverage_version or review.get("course_version")
        if course_version in (
            "motion_C0_standing_events50_motion50_v1",
            "motion_C0_standing_events50_motion50_v2",
        ):
            expected.update(continuous_C0_handover_m1=-1, continuous_C0_handover_12=12)
        if course_version == "motion_C0_standing_events50_motion50_v2":
            expected.update(
                {f"modal_combined_{mode}_{sign}": 0 for mode in range(4) for sign in (-1, 1)}
            )
        expected.update({f"modal_{mode}_{sign}": -1 for mode in range(4) for sign in (-1, 1)})
        expected.update({f"dq_standing_{profile}": profile for profile in (10, 11, 12)})
        expected.update({"recovery_" + name: i + 1 for i, name in enumerate(FAMILIES)})
        expected_anchors = tuple(anchors)
        standing, found_anchors, keys = {}, {}, set()
        for row in review["rows"]:
            key = tuple(row["key"])
            if key in keys:
                raise ValueError("duplicate_cohort_key")
            keys.add(key)
            if (
                len(key) != 5
                or list(key[:2]) != row["command"]
                or key[3:] != (row["profile"], row["purpose"])
            ):
                raise ValueError("cohort_identity_mismatch")
            purpose = row["purpose"]
            if purpose in expected:
                if (
                    row["command"] != [0.0, 0.0]
                    or row["profile"] != expected[purpose]
                    or key[2] != 0.34
                ):
                    raise ValueError("standing_contract_mismatch")
                standing[purpose] = _row(row, 32 if purpose == "clean_standing" else 16)
            elif tuple(row["command"]) in expected_anchors and row["profile"] == -1:
                anchor = tuple(row["command"])
                if anchor in found_anchors or key[2] != 0.34:
                    raise ValueError("motion_anchor_identity_mismatch")
                found_anchors[anchor] = _row(row, 16)
        if set(standing) != set(expected) or set(found_anchors) != set(expected_anchors):
            raise ValueError("incomplete_fixed_cohort_coverage")
        values = list(standing.values())
        count = sum(x["ended"] for x in values)

        def average(field):
            return sum(x[field] for x in values) / len(values)

        complete = sum(x["complete"] * x["ended"] for x in values) / count
        support = min(x["support"] for x in values)
        result.update(
            DR_strength=AMPLITUDES[DR_level],
            standing_ended_episodes=int(count),
            standing_cohorts=len(values),
            fixed_motion_anchors=len(found_anchors),
            nominal_quality_fraction=standing["clean_standing"]["quality"],
            standing_complete_fraction=complete,
            minimum_support_pass_fraction=support,
            worst_standing_pass_fraction=min(x["passed"] for x in values),
            balanced_standing_pass_fraction=average("passed"),
            standing_quality_fraction=average("quality"),
            standing_dual_support_fraction=average("dual"),
            negative_standing_HF_tail_rad_s=-average("HF"),
            matched_low_speed_pass_fraction=sum(x["passed"] for x in found_anchors.values())
            / len(found_anchors),
            combined_DR_pass_fraction=standing["dq_plus_general_DR"]["passed"],
            nominal_HF_tail_rad_s=standing["clean_standing"]["HF"],
        )
        if complete < 0.95 or support < 0.95 or standing["clean_standing"]["quality"] < 0.70:
            raise ValueError("standing_guard_not_met")
        result.update(
            eligible=True,
            reason="covered_training_window",
            rank=[result[name] for name in RANK_FIELDS],
        )
    except (KeyError, TypeError, ValueError) as exc:
        result.update(eligible=False, reason=str(exc))
    return result


def empty_index(comparison_sha256):
    return dict(
        version=VERSION,
        label=LABEL,
        comparison_sha256=comparison_sha256,
        candidates_by_DR_level={},
        best_candidate=None,
        independent_certificate=False,
        extra_simulation_steps=0,
    )


def consider(index, evidence, reference, provenance):
    """每个DR档仅保留一个候选；严格改善才替换，等分保留较早证据。"""
    updated = copy.deepcopy(index) if index else empty_index(evidence["comparison_sha256"])
    if (
        updated["version"] != VERSION
        or updated["comparison_sha256"] != evidence["comparison_sha256"]
    ):
        raise ValueError("训练候选比较域不一致，不能混用Plant/接口/采样/奖励")
    changed = False
    reason = evidence["reason"]
    if evidence["eligible"]:
        if not reference or len(reference.get("sha256", "")) != 64:
            raise ValueError("训练统计候选缺少已保存checkpoint身份")
        level = str(evidence["DR_level"])
        previous = updated["candidates_by_DR_level"].get(level)
        if previous is None or tuple(evidence["rank"]) > tuple(previous["evidence"]["rank"]):
            updated["candidates_by_DR_level"][level] = dict(
                reference,
                label=LABEL,
                evidence=copy.deepcopy(evidence),
                provenance=copy.deepcopy(provenance),
                independent_certificate=False,
            )
            changed, reason = (
                True,
                "first_covered_candidate" if previous is None else "rank_improved",
            )
        else:
            reason = "incumbent_not_outperformed"
    entries = updated["candidates_by_DR_level"]
    if len(entries) > len(AMPLITUDES):
        raise ValueError("训练候选数量超出四个DR档上限")
    updated["best_candidate"] = entries[max(entries, key=int)] if entries else None
    updated["last_review"] = dict(
        through_update=evidence["through_update"],
        checkpoint=reference,
        changed=changed,
        reason=reason,
        evidence=copy.deepcopy(evidence),
    )
    return updated
