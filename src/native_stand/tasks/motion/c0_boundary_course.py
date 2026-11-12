"""同C0事件课程，从显式来源摸底边界开始；不导入旧训练通过证书。"""

from . import c0_events_course as base
from . import highspeed_course
from .progressive_course import AXES, MAGNITUDES

VERSION = "motion_C0_source_boundary_events50_motion50_v3"
PROGRESSION = base.PROGRESSION


def configuration():
    return {
        **base.configuration(),
        "version": VERSION,
        "initial_frontier": "per_axis_measured_source_boundary",
        "initial_evidence": "bound deterministic source diagnostic; Native mastery still unconfirmed",
    }


def initial_state(frontiers):
    if set(frontiers) != set(AXES) or any(
        type(v) is not int or not 0 <= v < len(MAGNITUDES) for v in frontiers.values()
    ):
        raise ValueError("来源边界需要四个明确的课程索引")
    s = base.initial_state()
    s.update(version=VERSION, frontiers=dict(frontiers), source_boundary_frontiers=dict(frontiers))
    return s


task_rows = base.task_rows


def stage_config(stage, envs, *, state, families):
    result = base.stage_config(stage, envs, state=state, families=families)
    result["name"] = VERSION
    return result


def bind_keys(state, envs, families):
    base.bind_keys(state, envs, families)
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
