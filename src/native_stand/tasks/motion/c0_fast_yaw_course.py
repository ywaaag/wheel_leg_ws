"""保留站立/DR与直线进度，自转只采样1～4rad/s的独立四轴课程。"""

from __future__ import annotations

import copy

from . import c0_events_course as base
from . import highspeed_course
from .progressive_course import (
    AXES,
    MAGNITUDES,
    command,
    split,
    stage_from_rows,
)

VERSION = "motion_C0_events50_motion50_yaw1to4_v4"
PROGRESSION = base.PROGRESSION
MINIMUM_FRONTIERS = dict(forward=0, backward=0, yaw_positive=2, yaw_negative=2)


def configuration():
    return {
        **base.configuration(),
        "version": VERSION,
        "axis_magnitudes": {a: list(MAGNITUDES[MINIMUM_FRONTIERS[a] :]) for a in AXES},
        "initial_frontier": "full_checkpoint_progress_with_yaw_floor_1rad_s",
        "initial_evidence": "preserve mastered source levels; no new certificate from raising a frontier",
        "minimum_frontiers": dict(MINIMUM_FRONTIERS),
        "yaw_below_1rad_s_rehearsal": False,
        "precise_low_speed_or_position_hold_required": False,
        "reward_changed": False,
    }


def migrate_state(previous, *, source_update):
    if previous.get("version") not in ("motion_C0_source_boundary_events50_motion50_v3", VERSION):
        raise ValueError("自转课程只接受明确的同接口完整来源")
    state = copy.deepcopy(previous)
    state["parent_course"] = {
        k: copy.deepcopy(previous[k])
        for k in (
            "version",
            "stage",
            "levels",
            "frontiers",
            "general_DR_frontier",
            "last_training_update",
        )
    }
    state.update(
        version=VERSION,
        stage=0,
        stage_started_update=0,
        last_training_update=-1,
        source_checkpoint_updates=int(source_update),
        assessment_keys=[],
        accepted_by_stage={},
        pass_streak=dict.fromkeys(AXES, 0),
        regression_streak=dict.fromkeys(AXES, 0),
        DR_streak=0,
    )
    for axis in AXES:
        floor = MINIMUM_FRONTIERS[axis]
        state["frontiers"][axis] = max(floor, state["frontiers"][axis])
        if state["levels"][axis] < floor:
            state["levels"][axis] = -1
    return state


def task_rows(state, envs, families):
    if state.get("version") != VERSION or any(
        not MINIMUM_FRONTIERS[a] <= state["frontiers"][a] < len(MAGNITUDES) for a in AXES
    ):
        raise ValueError("自转1～4课程版本或档位不一致")
    old = base.task_rows(state, envs, families)
    rows = [r for r in old if r["axis"] not in ("yaw_positive", "yaw_negative")]
    for axis in ("yaw_positive", "yaw_negative"):
        quota = sum(r["count"] for r in old if r["axis"] == axis)
        level = state["frontiers"][axis]
        indices = list(range(MINIMUM_FRONTIERS[axis], level + 1))
        endpoint = round(quota * 0.60) if len(indices) > 1 else quota
        counts = (
            split(quota - endpoint, len(indices) - 1) + [endpoint] if len(indices) > 1 else [quota]
        )
        for index, total in zip(indices, counts, strict=False):
            nominal, dq = round(total * 0.60), round(total * 0.20)
            for profile, count, label in (
                (-1, nominal, "nominal"),
                (12, dq, "dq"),
                (0, total - nominal - dq, "DR"),
            ):
                amplitude = 0.0 if profile < 0 else base.AMPLITUDES[state["general_DR_frontier"]]
                rows.append(
                    dict(
                        target=list(command(axis, MAGNITUDES[index])),
                        profile=profile,
                        purpose=f"motion_{axis}_{MAGNITUDES[index]:g}_{label}",
                        count=count,
                        prefix=0,
                        mode=-1,
                        sign=0,
                        axis=axis,
                        endpoint=index == level,
                        amplitude=amplitude,
                        dq_amplitude=amplitude,
                    )
                )
    if sum(r["count"] for r in rows) != envs or len(rows) > 128:
        raise ValueError("自转课程环境配额或Critic条件容量不一致")
    return rows


def stage_config(stage, envs, *, state, families):
    if stage != state["stage"]:
        raise ValueError("自转课程revision不同")
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
        minimum_frontier=MINIMUM_FRONTIERS,
    )
