"""Independent fixed-height travel, banking and unbounded finite spin levels."""

from __future__ import annotations

import copy
import math

from .progressive_course import AMPLITUDES, FAMILIES, split, stage_from_rows
from .progressive_course import mixtures as mixtures

VERSION = "motion48_height_then_travel_banked_turn_spin_v1"
CRITIC_VERSION = "motion48_height_bank_causal875_v1"
DIM = 875
PROGRESSION = "mode_specific_support_posture_height_tracking_v1"
HEIGHTS = ((0.32, 0.36), (0.30, 0.32, 0.36, 0.38), (0.30, 0.32, 0.36, 0.38, 0.40))
TURN_SPEEDS = (1.0, 1.0, 2.0, 2.0, 3.6)
TURN_YAWS = (0.25, 0.5, 0.25, 0.5, 0.5)


def configuration():
    return dict(
        version=VERSION,
        review_fraction=0.40,
        new_fraction=0.60,
        review=dict(
            DR_forward36=0.10, DR_backward36=0.10, clean=0.05, dq=0.05, push=0.05, modal=0.05
        ),
        focuses=dict(height=0.20, turn=0.20, spin=0.20),
        straight_max_mps=3.6,
        height_levels=[list(x) for x in HEIGHTS],
        turn_speeds=list(TURN_SPEEDS),
        turn_yaws=list(TURN_YAWS),
        spin_terminal_cap=None,
        spin_growth=1.25,
        confirmations=2,
        regression_confirmations=3,
        nominal_fraction_new_task=0.5,
        review_every_updates=100,
        minimum_ended=16,
        minimum_complete_fraction=0.90,
        task_pass_fraction=0.70,
        DR_amplitudes=list(AMPLITUDES),
        DR_enabled=True,
        teacher_prefix_fraction=0.0,
        evidence="completed on-policy episodes; fixed checkpoint independent evidence separate",
        symmetry="standing and straight only",
        height_switch="stationary before movement, no state reset",
    )


def initial_state():
    return dict(
        version=VERSION,
        stage=0,
        last_training_update=-1,
        stage_started_update=0,
        general_DR_frontier=0,
        standing_DR_frontier=0,
        standing_DR_level=-1,
        height_level=0,
        turn_level=0,
        spin_targets=dict(positive=1.0, negative=1.0),
        streaks=dict(height=0, turn=0, positive=0, negative=0, DR=0),
        regressions=dict(height=0, turn=0, positive=0, negative=0),
        accepted_by_stage={},
        assessment_keys=[],
        DR_levels={f: 0 for f in FAMILIES},
        dq_level=0,
        DR_episode_counters=[],
    )


def task_rows(state, envs, families=FAMILIES):
    if state["version"] != VERSION or envs < 1:
        raise ValueError("Explicit maneuver version and environment count required")
    amplitude = AMPLITUDES[state["general_DR_frontier"]]
    rows = []

    def add(n, vx, yaw, profile, purpose, *, height=0.34, mode=-1, sign=0, axis=None):
        rows.append(
            dict(
                target=[vx, yaw],
                profile=profile,
                purpose=purpose,
                count=n,
                prefix=0,
                mode=mode,
                sign=sign,
                axis=axis,
                endpoint=True,
                amplitude=amplitude if profile >= 0 else 0.0,
                dq_amplitude=amplitude if profile in (0, 10, 11, 12) else 0.0,
                goal_height=height,
            )
        )

    quotas = [round(envs * f) for f in (0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.20, 0.20)]
    quotas.append(envs - sum(quotas))
    add(quotas[0], 3.6, 0.0, 0, "review_forward36", axis="forward")
    add(quotas[1], -3.6, 0.0, 0, "review_backward36", axis="backward")
    add(quotas[2], 0.0, 0.0, -1, "clean_standing")
    for p, n in zip((10, 11, 12), split(quotas[3], 3), strict=False):
        add(n, 0.0, 0.0, p, f"dq_standing_{p}")
    add(quotas[4], 0.0, 0.0, 8, "recovery_initial_push")
    for j, n in enumerate(split(quotas[5], 16)):
        mode = (j // 2) // 2
        sign = -1 if (j // 2) % 2 == 0 else 1
        profile = -1 if j % 2 == 0 else 0
        add(n, 0.0, 0.0, profile, f"modal_{mode}_{sign}_{profile}", mode=mode, sign=sign)
    hlist = HEIGHTS[state["height_level"]]
    speed = 1.8 if state["height_level"] == 0 else 3.6
    cases = [(h, v, p) for h in hlist for v in (0.0, speed, -speed) for p in (-1, 0)]
    for n, (h, v, p) in zip(split(quotas[6], len(cases)), cases, strict=False):
        add(n, v, 0.0, p, f"height_{h:g}_{v:g}_{p}", height=h, axis="height")
    level = state["turn_level"]
    v = TURN_SPEEDS[level]
    w = TURN_YAWS[level]
    # Expand only the physically bounded lateral acceleration domain.
    turn_heights = (0.34,)
    if state["height_level"] >= 1 and level >= 2:
        turn_heights = (0.32, 0.34, 0.36)
    cases = [
        (sv * v, sw * w, p, h)
        for sv in (-1, 1)
        for sw in (-1, 1)
        for p in (-1, 0)
        for h in turn_heights
    ]
    for n, (v, w, p, h) in zip(split(quotas[7], len(cases)), cases, strict=False):
        add(n, v, w, p, f"turn_{v:g}_{w:g}_{p}_h{h:g}", height=h, axis="turn")
    cases = [
        (name, sgn * state["spin_targets"][name], p)
        for name, sgn in [("positive", 1), ("negative", -1)]
        for p in (-1, 0)
    ]
    for n, (name, w, p) in zip(split(quotas[8], len(cases)), cases, strict=False):
        if not math.isfinite(w) or not abs(w) > 0:
            raise ValueError("Every sampled spin endpoint must be finite")
        add(n, 0.0, w, p, f"spin_{name}_{p}", axis=name)
    if envs < 64:
        # A small engineering fixture covers real consumers, without claiming
        # its sparse rows are a complete training/evaluation matrix.
        chosen = [2, 0, 1, 3, 6, 7]
        chosen += [
            next(
                i
                for i, r in enumerate(rows)
                if r["axis"] == "height"
                and r["goal_height"] == h
                and r["target"][0] == v
                and r["profile"] == -1
            )
            for h, v in [(hlist[0], speed), (hlist[-1], 0.0)]
        ]
        chosen += [
            next(i for i, r in enumerate(rows) if r["axis"] == a and r["profile"] == -1)
            for a in ("turn", "positive", "negative")
        ]
        chosen += [
            next(
                i
                for i, r in enumerate(rows)
                if r["mode"] == 3 and r["sign"] == 1 and r["profile"] == 0
            )
        ]
        for r in rows:
            r["count"] = 0
        for i in range(envs):
            rows[chosen[i % len(chosen)]]["count"] += 1
    if sum(r["count"] for r in rows) != envs or len(rows) > 128:
        raise ValueError("Maneuver cohort capacity/quota mismatch")
    return rows


def stage_config(stage, envs, *, state, families=FAMILIES):
    if stage != state["stage"]:
        raise ValueError("Maneuver revision mismatch")
    rows = task_rows(state, envs, families)
    s = stage_from_rows(stage, envs, rows, VERSION)
    s["goal_heights_m"] = [r["goal_height"] for r in rows]
    s["cohort_keys"] = [
        [*r["target"], 0.34, r["goal_height"], r["profile"], r["purpose"]] for r in rows
    ]
    s["maneuver_tasks"] = [r["axis"] or "standing" for r in rows]
    s["height_width_mix"] = [0.25] * len(rows)
    return s


def bind_keys(state, envs, families=FAMILIES):
    state["standing_DR_frontier"] = state["general_DR_frontier"]
    state["DR_levels"] = {f: state["general_DR_frontier"] for f in families}
    state["dq_level"] = state["general_DR_frontier"]
    state["assessment_keys"] = stage_config(state["stage"], envs, state=state, families=families)[
        "cohort_keys"
    ]
    return state


def advance(state, report, update, *, envs, families=FAMILIES, independent_DR=False):
    s = copy.deepcopy(state)
    if update <= s["last_training_update"] or report["stage_revision"] != s["stage"]:
        raise ValueError("Stale maneuver report")
    expected = stage_config(s["stage"], envs, state=s, families=families)["cohort_keys"]
    if [r["key"] for r in report["rows"]] != expected:
        raise ValueError("Maneuver report keys do not match the actual sampler")

    for row in report["rows"]:
        ended = row["ended_episodes"]
        complete = row["completed_episodes"]
        passed = row["pass_fraction"]
        if (
            ended < 0
            or complete < 0
            or complete > ended
            or (ended == 0 and passed is not None)
            or (ended > 0 and (passed is None or not math.isfinite(passed) or not 0 <= passed <= 1))
        ):
            raise ValueError("Maneuver episode evidence malformed")

    def covered(r):
        return r["ended_episodes"] >= 16 and r["pass_fraction"] is not None

    def good(r):
        return (
            covered(r)
            and r["completed_episodes"] / r["ended_episodes"] >= 0.90
            and r["pass_fraction"] >= 0.70
        )

    decisions = {}
    changed = False
    for focus in ("height", "turn", "positive", "negative"):
        rs = [r for r in report["rows"] if r["axis"] == focus]
        eligible = bool(rs) and all(covered(r) for r in rs)
        passed = eligible and all(good(r) for r in rs)
        s["streaks"][focus] = s["streaks"][focus] + 1 if passed else 0
        promoted = False
        regressed = False
        if s["streaks"][focus] >= 2:
            if focus in ("height", "turn"):
                field = focus + "_level"
                maximum = len(HEIGHTS if focus == "height" else TURN_SPEEDS) - 1
                if s[field] < maximum:
                    s[field] += 1
                    promoted = True
            else:
                s["spin_targets"][focus] *= 1.25
                promoted = True
            s["streaks"][focus] = 0
        changed |= promoted
        bad = eligible and any(r["pass_fraction"] < 0.35 for r in rs)
        s["regressions"][focus] = s["regressions"][focus] + 1 if bad else 0
        if not promoted and s["regressions"][focus] >= 3:
            if focus in ("height", "turn"):
                field = focus + "_level"
                old = s[field]
                s[field] = max(0, old - 1)
                regressed = old != s[field]
            else:
                old = s["spin_targets"][focus]
                s["spin_targets"][focus] = max(1.0, old / 1.25)
                regressed = old != s["spin_targets"][focus]
            s["regressions"][focus] = 0
            s["streaks"][focus] = 0
        changed |= regressed
        decisions[focus] = dict(
            covered=eligible, passed=passed, promoted=promoted, regressed=regressed
        )
    robust = [
        r
        for r in report["rows"]
        if r["axis"] in ("standing", "forward", "backward") and r["slot_count"] > 0
    ]
    dr_good = bool(robust) and all(good(r) for r in robust)
    s["streaks"]["DR"] = s["streaks"]["DR"] + 1 if dr_good else 0
    dr_up = (
        (independent_DR or not changed) and s["streaks"]["DR"] >= 2 and s["general_DR_frontier"] < 3
    )
    if dr_up:
        s["general_DR_frontier"] += 1
        s["streaks"]["DR"] = 0
        changed = True
    s["last_training_update"] = update
    if changed:
        s["stage"] += 1
        s["stage_started_update"] = update
    # A different focus progressing never clears another focus's confirmation.
    bind_keys(s, envs, families)
    return s, dict(
        version=VERSION,
        state_revision_changed=changed,
        advanced=changed,
        focuses=decisions,
        DR_advanced=dr_up,
        general_DR_frontier=s["general_DR_frontier"],
        training_evidence_only=True,
        independent_certificate=False,
        all_stages_passed=False,
    )
