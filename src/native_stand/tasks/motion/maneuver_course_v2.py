"""Versioned response gate and independent DR advancement; sampler unchanged."""

from __future__ import annotations

import copy

from . import maneuver_course as legacy
from .progressive_course import FAMILIES

VERSION = "motion48_response_checked_independent_DR_v2"
CRITIC_VERSION = legacy.CRITIC_VERSION
DIM = legacy.DIM
mixtures = legacy.mixtures


def configuration():
    value = legacy.configuration()
    value.update(
        version=VERSION,
        independent_DR_advancement=True,
        nonzero_yaw_minimum_signed_gain=0.5,
        command_program_version="motion48_height_travel_turn_schedule_v2",
        legacy_confirmations_are_not_new_evidence=True,
    )
    return value


def _legacy(state):
    if state.get("version") != VERSION:
        raise ValueError("Explicit maneuver v2 state required")
    result = copy.deepcopy(state)
    result["version"] = legacy.VERSION
    return result


def initial_state():
    state = legacy.initial_state()
    state["version"] = VERSION
    return state


def migrate_state(state, *, source_update):
    if state.get("version") == VERSION:
        return copy.deepcopy(state)
    if state.get("version") != legacy.VERSION:
        raise ValueError("Only explicit maneuver v1 to v2 continuation is supported")
    result = copy.deepcopy(state)
    result.update(version=VERSION, stage=state["stage"] + 1, stage_started_update=source_update)
    result["protocol_transition"] = dict(
        from_version=legacy.VERSION,
        source_update=source_update,
        invalidated_confirmations={k: state["streaks"][k] for k in ("turn", "DR")},
        previous_turn_regression=state["regressions"]["turn"],
        old_evidence_is_legacy=True,
        frontiers_retained_as_learning_targets=True,
    )
    result["streaks"]["turn"] = result["streaks"]["DR"] = 0
    result["regressions"]["turn"] = 0
    return result


def task_rows(state, envs, families=FAMILIES):
    return legacy.task_rows(_legacy(state), envs, families)


def stage_config(stage, envs, *, state, families=FAMILIES):
    value = legacy.stage_config(stage, envs, state=_legacy(state), families=families)
    value["name"] = VERSION
    value["command_program_version"] = configuration()["command_program_version"]
    return value


def bind_keys(state, envs, families=FAMILIES):
    value = legacy.bind_keys(_legacy(state), envs, families)
    value["version"] = VERSION
    state.clear()
    state.update(value)
    return state


def advance(state, report, update, *, envs, families=FAMILIES):
    from .maneuver_statistics import RESPONSE_VERSION

    if report.get("version") != RESPONSE_VERSION:
        raise ValueError("New response gate cannot consume legacy statistics")
    value, decision = legacy.advance(
        _legacy(state), report, update, envs=envs, families=families, independent_DR=True
    )
    value["version"] = VERSION
    decision["version"] = VERSION
    return value, decision
