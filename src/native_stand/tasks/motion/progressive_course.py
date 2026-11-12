"""固定高度的站立与四方向独立课程；只消费完整的on-policy训练窗口。"""

from __future__ import annotations

import copy
import math

VERSION = "motion_supported_standing50_motion50_progressive_v1"
AXES = ("forward", "backward", "yaw_positive", "yaw_negative")
MAGNITUDES = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
AMPLITUDES = (0.25, 0.5, 0.75, 1.0)
FAMILIES = (
    "observation_noise",
    "sensor_bias",
    "timing_response",
    "net_load_capacity",
    "mass_inertia",
    "COM",
    "friction",
    "initial_push",
    "ground_slope",
)


def configuration():
    return dict(
        version=VERSION,
        magnitudes=list(MAGNITUDES),
        standing_fractions=dict(clean=0.25, modal=0.10, dq=0.05, combined=0.05, singles=0.05),
        motion_fraction=0.50,
        motion_directions_equal=True,
        endpoint_fraction=0.60,
        nominal_fraction_per_motion_target=0.50,
        fixed_height_m=0.34,
        prefix_intervals=0,
        review_every_updates=100,
        confirmations=2,
        minimum_nominal_episodes=32,
        minimum_stress_episodes=16,
        nominal_pass_fraction=0.70,
        stress_pass_fraction=0.65,
        regression_confirmations=2,
        regression_pass_fraction=0.35,
        progression="independent four axes; nominal standing protects motion; DR upgrades independently after two covered standing windows; never stop for ordinary behavior regression",
        evidence="complete stochastic on-policy episodes only; no independent certificate",
        source_of_reuse="9995: persistent old-command rehearsal, gradual task/uncertainty progression; no old action/plant or certificate transfer",
    )


def initial_state(previous=None):
    prior = previous or {}
    frontier = min(3, max(0, int(prior.get("general_DR_frontier", 0))))
    return dict(
        version=VERSION,
        stage=0,
        stage_started_update=0,
        last_training_update=-1,
        standing_passed=False,
        general_DR_frontier=frontier,
        standing_DR_frontier=frontier,
        standing_DR_level=-1,
        DR_streak=0,
        dq_frontier=frontier,
        dq_level=frontier,
        DR_levels={family: frontier for family in FAMILIES},
        highspeed_paused=False,
        levels={a: -1 for a in AXES},
        frontiers={a: 1 for a in AXES},
        pass_streak={a: 0 for a in AXES},
        regression_streak={a: 0 for a in AXES},
        accepted_by_stage={},
        assessment_keys=[],
        parent_course=copy.deepcopy(prior),
        DR_episode_counters=copy.deepcopy(prior.get("DR_episode_counters", [])),
    )


def split(total, count):
    q, r = divmod(total, count)
    return [q + int(i < r) for i in range(count)]


def command(axis, magnitude):
    return {
        "forward": [magnitude, 0.0],
        "backward": [-magnitude, 0.0],
        "yaw_positive": [0.0, magnitude],
        "yaw_negative": [0.0, -magnitude],
    }[axis]


def task_rows(state, envs, families):
    if envs < 1 or any(not 0 <= state["frontiers"][a] < len(MAGNITUDES) for a in AXES):
        raise ValueError("环境数或运动课程档位无效")
    rows = []
    amplitude = AMPLITUDES[state["general_DR_frontier"]]

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
                amplitude=amplitude if profile >= 0 else 0.0,
                dq_amplitude=amplitude if profile in (0, 10, 11, 12) else 0.0,
            )
        )

    quotas = [round(envs * f) for f in (0.25, 0.10, 0.05, 0.05, 0.05)]
    add((0.0, 0.0), -1, "clean_standing", quotas[0])
    for j, n in enumerate(split(quotas[1], 8)):
        sign = -1 if j % 2 == 0 else 1
        add((0.0, 0.0), -1, f"modal_{j // 2}_{sign}", n, mode=j // 2, sign=sign)
    for profile, n in zip((10, 11, 12), split(quotas[2], 3), strict=False):
        add((0.0, 0.0), profile, f"dq_standing_{profile}", n)
    add((0.0, 0.0), 0, "dq_plus_general_DR", quotas[3])
    for i, n in enumerate(split(quotas[4], len(families))):
        add((0.0, 0.0), i + 1, "recovery_" + families[i], n)
    for axis, quota in zip(AXES, split(envs - sum(quotas), 4), strict=False):
        level = state["frontiers"][axis]
        endpoint = round(quota * 0.60)
        counts = split(quota - endpoint, level) + [endpoint]
        for j, total in enumerate(counts):
            magnitude = MAGNITUDES[j]
            for profile, n in zip((-1, 12), split(total, 2), strict=False):
                add(
                    command(axis, magnitude),
                    profile,
                    f"motion_{axis}_{magnitude:g}_" + ("nominal" if profile < 0 else "dq"),
                    n,
                    axis=axis,
                    endpoint=j == level,
                )
    if envs == 1:
        for i, row in enumerate(rows):
            row["count"] = int(i == 0)
    if sum(r["count"] for r in rows) != envs:
        raise ValueError("固定槽位总量不等于环境数")
    return rows


def stage_config(stage, envs, *, state, families):
    if state["stage"] != stage:
        raise ValueError("课程revision不匹配")
    rows = task_rows(state, envs, families)
    return stage_from_rows(stage, envs, rows, VERSION)


def stage_from_rows(stage, envs, rows, name):
    """不同显式课程复用同一物理cohort编码。"""
    counts = [r["count"] for r in rows]
    return dict(
        id=stage,
        name=name,
        commands=[r["target"] for r in rows],
        slot_counts=counts,
        weights=[n / envs for n in counts],
        sampling_mode="fixed_slots",
        sampling_slots=[i for i, n in enumerate(counts) for _ in range(n)],
        enforce_whitelist=True,
        cohort_keys=[[*r["target"], 0.34, r["profile"], r["purpose"]] for r in rows],
        height_targets_m=[0.34] * len(rows),
        mcu_variants=[0] * len(rows),
        cohort_purposes=[r["purpose"] for r in rows],
        identified_profiles=[r["profile"] for r in rows],
        identified_amplitudes=[r["amplitude"] for r in rows],
        dq_amplitudes=[r["dq_amplitude"] for r in rows],
        prefix_intervals=[0] * len(rows),
        modal_modes=[r["mode"] for r in rows],
        modal_signs=[r["sign"] for r in rows],
        speed_width_mix=[0.0] * len(rows),
        height_width_mix=[0.75] * len(rows),
        PD_feedback_max_ms=12,
        motion_axes=[r["axis"] for r in rows],
        motion_endpoints=[r["endpoint"] for r in rows],
    )


def bind_keys(state, envs, families):
    state["DR_levels"] = {family: state["general_DR_frontier"] for family in families}
    state["dq_level"] = state["general_DR_frontier"]
    state["assessment_keys"] = stage_config(state["stage"], envs, state=state, families=families)[
        "cohort_keys"
    ]
    return state


def mixtures(stage, *, symmetry=False):
    return [
        dict(
            target=target,
            mix=0.0,
            **({"confirmations": 0, "regressions": 0} if symmetry else {"consecutive_eligible": 0}),
        )
        for target in stage["commands"]
    ]


def advance(state, report, update, *, envs, families):
    """每方向晋级不等待另一方向；缺失/旧revision的证据绝不触发晋级或降级。"""
    cfg = configuration()
    s = copy.deepcopy(state)
    expected = stage_config(s["stage"], envs, state=s, families=families)
    if update <= s["last_training_update"]:
        raise ValueError("重复或倒退的训练窗口")
    if (
        report.get("stage_revision") != s["stage"]
        or [r["key"] for r in report["rows"]] != expected["cohort_keys"]
    ):
        raise ValueError("课程窗口身份不完整或过期")
    rows = report["rows"]

    def good(row, threshold, minimum):
        return (
            row["ended_episodes"] >= minimum
            and row["completed_episodes"] >= minimum // 2
            and row["pass_fraction"] is not None
            and math.isfinite(row["pass_fraction"])
            and row["pass_fraction"] >= threshold
        )

    standing = next(r for r in rows if r["purpose"] == "clean_standing")
    standing_ok = good(standing, cfg["nominal_pass_fraction"], cfg["minimum_nominal_episodes"])
    decisions = {}
    changed = False
    for axis in AXES:
        current = MAGNITUDES[s["frontiers"][axis]]
        endpoint = [
            r
            for r in rows
            if r.get("axis") == axis
            and r["profile"] < 0
            and max(abs(x) for x in r["command"]) == current
        ]
        passed = standing_ok and len(endpoint) == 1 and good(endpoint[0], 0.70, 32)
        s["pass_streak"][axis] = s["pass_streak"][axis] + 1 if passed else 0
        advanced = False
        regressed = False
        if s["pass_streak"][axis] >= 2:
            s["levels"][axis] = max(s["levels"][axis], s["frontiers"][axis])
            if s["frontiers"][axis] < len(MAGNITUDES) - 1:
                s["frontiers"][axis] += 1
                advanced = changed = True
            s["pass_streak"][axis] = 0
        mastered = [
            r
            for r in rows
            if r.get("axis") == axis
            and r["profile"] < 0
            and s["levels"][axis] >= 0
            and max(abs(x) for x in r["command"]) <= MAGNITUDES[s["levels"][axis]]
        ]
        bad = (
            bool(mastered)
            and all(r["ended_episodes"] >= 16 for r in mastered)
            and any(r["pass_fraction"] is not None and r["pass_fraction"] < 0.35 for r in mastered)
        )
        s["regression_streak"][axis] = s["regression_streak"][axis] + 1 if bad else 0
        if not advanced and s["regression_streak"][axis] >= 2:
            old = s["frontiers"][axis]
            s["frontiers"][axis] = max(1, old - 1)
            s["levels"][axis] = min(s["levels"][axis], s["frontiers"][axis] - 1)
            s["regression_streak"][axis] = 0
            regressed = old != s["frontiers"][axis]
            changed |= regressed
        decisions[axis] = dict(
            window_passed=passed,
            advanced=advanced,
            regressed=regressed,
            target=MAGNITUDES[s["frontiers"][axis]],
            certified_level=s["levels"][axis],
        )
    robust = [r for r in rows if r["command"] == [0.0, 0.0] and r["purpose"] != "clean_standing"]
    covered = bool(robust) and all(good(r, 0.65, 16) for r in robust)
    s["DR_streak"] = s["DR_streak"] + 1 if standing_ok and covered else 0
    dr_advanced = False
    # 不在同一次revision中同时提高运动上限与DR强度。
    if not changed and s["DR_streak"] >= 2:
        s["standing_DR_level"] = s["general_DR_frontier"]
        if s["general_DR_frontier"] < 3:
            s["general_DR_frontier"] += 1
            changed = dr_advanced = True
        s["DR_streak"] = 0
    s["standing_DR_frontier"] = s["general_DR_frontier"]
    s["dq_frontier"] = s["general_DR_frontier"]
    s["standing_passed"] = standing_ok
    s["last_training_update"] = update
    if changed:
        s["stage"] += 1
        s["stage_started_update"] = update
        s["pass_streak"] = {a: 0 for a in AXES}
        s["DR_streak"] = 0
    bind_keys(s, envs, families)
    return s, dict(
        version=VERSION,
        state_revision_changed=changed,
        advanced=changed,
        axes=decisions,
        standing_window_passed=standing_ok,
        robust_coverage_passed=covered,
        general_DR_frontier=s["general_DR_frontier"],
        DR_advanced=dr_advanced,
        training_evidence_only=True,
        independent_certificate=False,
        all_stages_passed=all(s["levels"][a] == len(MAGNITUDES) - 1 for a in AXES)
        and s["standing_DR_level"] == 3,
        stop=False,
    )
