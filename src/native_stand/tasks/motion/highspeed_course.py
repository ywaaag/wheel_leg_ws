"""高速优先、保留DR与站立的四轴独立课程；晋级证据来自训练完整回合。"""

from __future__ import annotations

import copy
import math

from .progressive_course import (
    AMPLITUDES,
    AXES,
    MAGNITUDES,
    command,
    split,
    stage_from_rows,
)

VERSION = "motion_supported_highspeed70_standing30_v2"
PROGRESSION = "motion_hold_mean_rolling_progression_v1"


def configuration():
    return dict(
        version=VERSION,
        magnitudes=list(MAGNITUDES),
        standing_fractions=dict(clean=0.15, modal=0.05, dq=0.05, combined=0.03, singles=0.02),
        motion_fraction=0.70,
        endpoint_fraction=0.60,
        motion_directions_equal=True,
        motion_profile_fractions=dict(nominal=0.60, dq=0.20, general_DR=0.20),
        fixed_height_m=0.34,
        prefix_intervals=0,
        review_every_updates=100,
        confirmations=2,
        regression_confirmations=3,
        minimum_nominal_episodes=32,
        nominal_pass_fraction=0.70,
        stress_pass_fraction=0.65,
        regression_scope="only highest mastered endpoint, never any slower rehearsal command",
        progression_version=PROGRESSION,
        progression_rolling=dict(
            hold_mean_opposite=True,
            common_absolute_tolerance_mps=0.03,
            common_relative_tolerance=0.20,
            common_mean_abs_mps_max=0.12,
        ),
        historical_gates="preserved and reported; new progression is explicitly separate",
        evidence="complete stochastic on-policy episodes, not independent certificates",
    )


def initial_state(previous):
    """从冻结400的已有证据选探索起点，不凭编号授予新能力。"""
    state = copy.deepcopy(previous)
    state.update(
        version=VERSION,
        stage=0,
        stage_started_update=0,
        last_training_update=-1,
        standing_passed=False,
        highspeed_paused=False,
        general_DR_frontier=0,
        standing_DR_frontier=0,
        standing_DR_level=-1,
        dq_frontier=0,
        dq_level=0,
        levels=dict(forward=1, backward=5, yaw_positive=-1, yaw_negative=-1),
        frontiers=dict(forward=2, backward=6, yaw_positive=1, yaw_negative=1),
        pass_streak={a: 0 for a in AXES},
        regression_streak={a: 0 for a in AXES},
        DR_streak=0,
        accepted_by_stage={},
        assessment_keys=[],
        parent_course=copy.deepcopy(previous),
    )
    # DR流的episode序号接续，不重复抽取来源的同一随机回合。
    return state


def task_rows(state, envs, families):
    if envs < 1 or any(not 1 <= state["frontiers"][a] < len(MAGNITUDES) for a in AXES):
        raise ValueError("高速课程环境数或档位无效")
    rows = []
    amp = AMPLITUDES[state["general_DR_frontier"]]

    def add(target, profile, purpose, count, *, mode=-1, sign=0, axis=None, endpoint=False):
        rows.append(
            dict(
                target=list(target),
                profile=profile,
                purpose=purpose,
                count=count,
                prefix=0,
                mode=mode,
                sign=sign,
                axis=axis,
                endpoint=endpoint,
                amplitude=amp if profile >= 0 else 0.0,
                dq_amplitude=amp if profile in (0, 10, 11, 12) else 0.0,
            )
        )

    quotas = [round(envs * f) for f in (0.15, 0.05, 0.05, 0.03, 0.02)]
    add((0.0, 0.0), -1, "clean_standing", quotas[0])
    for i, n in enumerate(split(quotas[1], 8)):
        sign = -1 if i % 2 == 0 else 1
        add((0.0, 0.0), -1, f"modal_{i // 2}_{sign}", n, mode=i // 2, sign=sign)
    for profile, n in zip((10, 11, 12), split(quotas[2], 3), strict=False):
        add((0.0, 0.0), profile, f"dq_standing_{profile}", n)
    add((0.0, 0.0), 0, "dq_plus_general_DR", quotas[3])
    for i, n in enumerate(split(quotas[4], len(families))):
        add((0.0, 0.0), i + 1, "recovery_" + families[i], n)
    for axis, quota in zip(AXES, split(envs - sum(quotas), 4), strict=False):
        level = state["frontiers"][axis]
        endpoint = round(quota * 0.6)
        counts = split(quota - endpoint, level) + [endpoint]
        for j, total in enumerate(counts):
            nominal = round(total * 0.6)
            dq = round(total * 0.2)
            for profile, n, label in (
                (-1, nominal, "nominal"),
                (12, dq, "dq"),
                (0, total - nominal - dq, "DR"),
            ):
                add(
                    command(axis, MAGNITUDES[j]),
                    profile,
                    f"motion_{axis}_{MAGNITUDES[j]:g}_{label}",
                    n,
                    axis=axis,
                    endpoint=j == level,
                )
    if envs == 1:
        for i, row in enumerate(rows):
            row["count"] = int(i == 0)
    if sum(r["count"] for r in rows) != envs:
        raise ValueError("高速课程槽位数错误")
    return rows


def stage_config(stage, envs, *, state, families):
    if stage != state["stage"]:
        raise ValueError("高速课程revision不匹配")
    return stage_from_rows(stage, envs, task_rows(state, envs, families), VERSION)


def bind_keys(state, envs, families):
    state["DR_levels"] = {f: state["general_DR_frontier"] for f in families}
    state["dq_level"] = state["general_DR_frontier"]
    state["assessment_keys"] = stage_config(state["stage"], envs, state=state, families=families)[
        "cohort_keys"
    ]
    return state


def advance(state, report, update, *, envs, families):
    return advance_curriculum(
        state,
        report,
        update,
        envs=envs,
        families=families,
        stage_builder=stage_config,
        binder=bind_keys,
        version=VERSION,
        minimum_frontier=1,
    )


def advance_curriculum(
    state, report, update, *, envs, families, stage_builder, binder, version, minimum_frontier
):
    """共用相同晋退级规则，采样和新任务初始档位由显式课程提供。"""
    s = copy.deepcopy(state)
    expected = stage_builder(s["stage"], envs, state=s, families=families)
    if (
        update <= s["last_training_update"]
        or report.get("stage_revision") != s["stage"]
        or [r["key"] for r in report["rows"]] != expected["cohort_keys"]
    ):
        raise ValueError("高速课程窗口身份过期或不完整")

    for row in report["rows"]:
        ended, complete = row["ended_episodes"], row["completed_episodes"]
        if type(ended) is not int or type(complete) is not int or not 0 <= complete <= ended:
            raise ValueError("课程回合计数损坏")
        for name in (
            "quality_pass_fraction",
            "progression_pass_fraction" if "progression_pass_fraction" in row else "pass_fraction",
        ):
            value = row[name]
            if ended == 0:
                if value is not None:
                    raise ValueError("零回合必须使用UNKNOWN")
            elif value is None or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("课程行为比例损坏")

    def good(row, threshold, minimum=32):
        fraction = row.get("progression_pass_fraction", row.get("pass_fraction"))
        return (
            row["ended_episodes"] >= minimum
            and row["completed_episodes"] >= minimum // 2
            and fraction is not None
            and math.isfinite(fraction)
            and fraction >= threshold
        )

    rows = report["rows"]
    clean = next(r for r in rows if r["purpose"] == "clean_standing")
    standing_ok = good(clean, 0.7)
    changed = False
    decisions = {}
    for axis in AXES:
        endpoint = next(
            r for r in rows if r.get("axis") == axis and r["profile"] == -1 and r["endpoint"]
        )
        passed = standing_ok and good(endpoint, 0.70)
        s["pass_streak"][axis] = s["pass_streak"][axis] + 1 if passed else 0
        advanced = regressed = False
        if s["pass_streak"][axis] >= 2:
            s["levels"][axis] = max(s["levels"][axis], s["frontiers"][axis])
            if s["frontiers"][axis] < len(MAGNITUDES) - 1:
                s["frontiers"][axis] += 1
                changed = advanced = True
            s["pass_streak"][axis] = 0
        # 低速不准不应清除高速学习。仅最高已掌握档的物理质量明显恶化才退档。
        mastered = [
            r
            for r in rows
            if r.get("axis") == axis
            and r["profile"] == -1
            and s["levels"][axis] >= 0
            and max(map(abs, r["command"])) == MAGNITUDES[s["levels"][axis]]
        ]
        bad = (
            bool(mastered)
            and mastered[0]["ended_episodes"] >= 32
            and mastered[0]["quality_pass_fraction"] < 0.35
        )
        s["regression_streak"][axis] = s["regression_streak"][axis] + 1 if bad else 0
        if not advanced and s["regression_streak"][axis] >= 3:
            floor = (
                minimum_frontier[axis] if isinstance(minimum_frontier, dict) else minimum_frontier
            )
            old = s["frontiers"][axis]
            s["frontiers"][axis] = max(floor, old - 1)
            s["levels"][axis] = min(s["levels"][axis], s["frontiers"][axis] - 1)
            if isinstance(minimum_frontier, dict) and s["levels"][axis] < floor:
                s["levels"][axis] = -1
            s["regression_streak"][axis] = 0
            regressed = old != s["frontiers"][axis]
            changed |= regressed
            s["pass_streak"][axis] = 0
        decisions[axis] = dict(
            window_passed=passed,
            advanced=advanced,
            regressed=regressed,
            target=MAGNITUDES[s["frontiers"][axis]],
            certified_level=s["levels"][axis],
        )
    robust = [r for r in rows if not any(r["command"]) and r["purpose"] != "clean_standing"]
    covered = bool(robust) and all(good(r, 0.65, 16) for r in robust)
    s["DR_streak"] = s["DR_streak"] + 1 if standing_ok and covered else 0
    dr_advanced = False
    if not changed and s["DR_streak"] >= 2:
        s["standing_DR_level"] = s["general_DR_frontier"]
        if s["general_DR_frontier"] < 3:
            s["general_DR_frontier"] += 1
            changed = dr_advanced = True
        s["DR_streak"] = 0
    s["standing_passed"] = standing_ok
    s["last_training_update"] = update
    s["standing_DR_frontier"] = s["general_DR_frontier"]
    s["dq_frontier"] = s["general_DR_frontier"]
    if changed:
        s["stage"] += 1
        s["stage_started_update"] = update
        # 其他方向晋级计数保留；只有物理DR换域才全部重新确认。
        if dr_advanced:
            s["pass_streak"] = {a: 0 for a in AXES}
    binder(s, envs, families)
    return s, dict(
        version=version,
        state_revision_changed=changed,
        advanced=any(d["advanced"] for d in decisions.values()) or dr_advanced,
        axes=decisions,
        standing_window_passed=standing_ok,
        robust_coverage_passed=covered,
        general_DR_frontier=s["general_DR_frontier"],
        DR_advanced=dr_advanced,
        training_evidence_only=True,
        independent_certificate=False,
        all_stages_passed=all(s["levels"][a] == 6 for a in AXES) and s["standing_DR_level"] == 3,
        stop=False,
    )
