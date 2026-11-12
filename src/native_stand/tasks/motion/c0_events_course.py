"""C0事件覆盖修订；保留v1配额/运动晋级，只拆分模态受扰的执行条件。"""

from __future__ import annotations

from . import c0_course, highspeed_course
from .progressive_course import (
    AMPLITUDES,
    split,
    stage_from_rows,
)
from .transient_statistics import VERSION as EVENTS_VERSION

VERSION = "motion_C0_standing_events50_motion50_v2"
PROGRESSION = c0_course.PROGRESSION


def configuration():
    return {
        **c0_course.configuration(),
        "version": VERSION,
        "modal_profile_fractions": {"nominal": 0.50, "combined_DR": 0.50},
        "event_statistics_version": EVENTS_VERSION,
        "event_time_origin": "pulse force-off; pre window is before force-on",
        "reward_changed": False,
    }


def initial_state():
    return {**c0_course.initial_state(), "version": VERSION}


def task_rows(state, envs, families):
    old = c0_course.task_rows(state, envs, families)
    modal = [r for r in old if r["mode"] >= 0]
    nominal, combined = split(sum(r["count"] for r in modal), 2)
    counts = {p: split(n, 8) for p, n in ((-1, nominal), (0, combined))}
    rows = []
    for row in old:
        if row["mode"] < 0:
            rows.append(row)
            continue
        index = row["mode"] * 2 + int(row["sign"] > 0)
        for profile in (-1, 0):
            rows.append(
                {
                    **row,
                    "profile": profile,
                    "count": counts[profile][index],
                    "purpose": row["purpose"]
                    if profile == -1
                    else f"modal_combined_{row['mode']}_{row['sign']}",
                    "amplitude": 0.0 if profile < 0 else AMPLITUDES[state["general_DR_frontier"]],
                    "dq_amplitude": 0.0
                    if profile < 0
                    else AMPLITUDES[state["general_DR_frontier"]],
                }
            )
    if sum(r["count"] for r in rows) != envs or len(rows) > 128:
        raise ValueError("C0 v2课程槽位或存储上限不匹配")
    return rows


def stage_config(stage, envs, *, state, families):
    if state["stage"] != stage:
        raise ValueError("C0 v2课程revision不匹配")
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
