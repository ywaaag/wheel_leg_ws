"""C0接口重新确认四方向能力；站立/连续交接/受扰/DR与运动各占一半。"""

from __future__ import annotations

from . import highspeed_course
from .progressive_course import (
    AMPLITUDES,
    AXES,
    MAGNITUDES,
    command,
    split,
    stage_from_rows,
)

VERSION = "motion_C0_standing_events50_motion50_v1"
PROGRESSION = highspeed_course.PROGRESSION


def configuration():
    return dict(
        version=VERSION,
        magnitudes=list(MAGNITUDES),
        fixed_height_m=0.34,
        standing_fractions=dict(
            clean=0.20, handover=0.10, modal=0.10, dq=0.05, combined=0.03, singles=0.02
        ),
        motion_fraction=0.50,
        motion_directions_equal=True,
        endpoint_fraction=0.60,
        motion_profile_fractions=dict(nominal=0.60, dq=0.20, general_DR=0.20),
        prefix_intervals=500,
        prefix_identity="frozen C0, same new decoder; not physical LQR",
        review_every_updates=100,
        confirmations=2,
        regression_confirmations=3,
        minimum_nominal_episodes=32,
        nominal_pass_fraction=0.70,
        stress_pass_fraction=0.65,
        progression_version=PROGRESSION,
        initial_frontier=0,
        initial_evidence="none; previous-center levels never transferred",
        evidence="complete student-controlled on-policy episodes; not independent certificates",
    )


def initial_state():
    from .progressive_course import initial_state as fresh_state

    state = fresh_state()
    state.update(
        version=VERSION,
        frontiers={axis: 0 for axis in AXES},
        parent_course={},
        last_training_update=-1,
        DR_episode_counters=[],
    )
    return state


def task_rows(state, envs, families):
    if envs < 1 or any(not 0 <= state["frontiers"][a] < len(MAGNITUDES) for a in AXES):
        raise ValueError("C0课程环境数或档位无效")
    rows = []
    amplitude = AMPLITUDES[state["general_DR_frontier"]]

    def add(
        target, profile, purpose, count, *, mode=-1, sign=0, axis=None, endpoint=False, prefix=0
    ):
        rows.append(
            dict(
                target=list(target),
                profile=profile,
                purpose=purpose,
                count=count,
                mode=mode,
                sign=sign,
                axis=axis,
                endpoint=endpoint,
                prefix=prefix,
                amplitude=amplitude if profile >= 0 else 0.0,
                dq_amplitude=amplitude if profile in (0, 10, 11, 12) else 0.0,
            )
        )

    q = [round(envs * f) for f in (0.20, 0.10, 0.10, 0.05, 0.03, 0.02)]
    q[0] += round(envs * 0.50) - sum(q)
    add((0.0, 0.0), -1, "clean_standing", q[0])
    for profile, n in zip((-1, 12), split(q[1], 2), strict=False):
        add(
            (0.0, 0.0),
            profile,
            "continuous_C0_handover_" + ("m1" if profile < 0 else str(profile)),
            n,
            prefix=500,
        )
    for j, n in enumerate(split(q[2], 8)):
        sign = -1 if j % 2 == 0 else 1
        add((0.0, 0.0), -1, f"modal_{j // 2}_{sign}", n, mode=j // 2, sign=sign)
    for profile, n in zip((10, 11, 12), split(q[3], 3), strict=False):
        add((0.0, 0.0), profile, f"dq_standing_{profile}", n)
    add((0.0, 0.0), 0, "dq_plus_general_DR", q[4])
    for j, n in enumerate(split(q[5], len(families))):
        add((0.0, 0.0), j + 1, "recovery_" + families[j], n)
    for axis, quota in zip(AXES, split(envs - sum(q), 4), strict=False):
        level = state["frontiers"][axis]
        endpoint = round(quota * 0.60) if level else quota
        counts = split(quota - endpoint, level) + [endpoint] if level else [quota]
        for j, total in enumerate(counts):
            nominal = round(total * 0.60)
            dq = round(total * 0.20)
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
        for j, row in enumerate(rows):
            row["count"] = int(j == 0)
    if sum(r["count"] for r in rows) != envs:
        raise ValueError("C0课程槽位数不匹配")
    return rows


def stage_config(stage, envs, *, state, families):
    if state["stage"] != stage:
        raise ValueError("C0课程revision不匹配")
    rows = task_rows(state, envs, families)
    result = stage_from_rows(stage, envs, rows, VERSION)
    result["prefix_intervals"] = [r["prefix"] for r in rows]
    return result


def bind_keys(state, envs, families):
    state["DR_levels"] = {f: state["general_DR_frontier"] for f in families}
    state["dq_level"] = state["general_DR_frontier"]
    state["assessment_keys"] = stage_config(state["stage"], envs, state=state, families=families)[
        "cohort_keys"
    ]
    return state


def advance(state, report, update, *, envs, families):
    return highspeed_course.advance_curriculum(
        state,
        report,
        update,
        envs=envs,
        families=families,
        stage_builder=stage_config,
        binder=bind_keys,
        version=VERSION,
        minimum_frontier=0,
    )
