"""Actual split-course confirmation identity and finite Spin endpoint regressions."""

from __future__ import annotations

import copy
import math

import pytest
import torch
from test_motion_spin_branches import course_configuration, report

from native_stand.tasks.motion import split_candidates
from native_stand.tasks.motion.course_strategy import resolve
from native_stand.tasks.motion.split_course import MAX_COMMAND_MAGNITUDE

FROZEN_TEST_SCOPE = "PORTABLE"


def initial(branch="spin", **overrides):
    cfg = course_configuration(branch)
    cfg.update(overrides)
    strategy = resolve(cfg["version"])
    return strategy, strategy.initial_state({"branch_course": cfg})


def window(strategy, state, update, *, good=(), bad=(), robust=False):
    review = report(state, state["branch"], unknown=True)
    for row in review["rows"]:
        is_focus = row["purpose"].startswith("focus_")
        passed = (is_focus and row["axis"] in good) or (not is_focus and robust)
        failed = is_focus and row["axis"] in bad
        if passed or failed:
            row.update(
                ended_episodes=16,
                completed_episodes=16 if passed else 0,
                pass_fraction=1.0 if passed else 0.0,
                quality_pass_fraction=1.0 if passed else 0.0,
                mean_dual_support_fraction=1.0,
                physical_HF_RMS_rad_s=0.01,
            )
    review["through_update"] = update
    review["window"] = dict(
        run=state["branch"],
        comparison_sha256="fixed-domain",
        start_update=state.get("consumed_window", {}).get("end_update", 0) + 1,
        end_update=update,
        stage_revision=state["stage"],
    )
    new, decision = strategy.advance(state, review, update, envs=2560)
    return new, decision, review


def test_unrelated_direction_revision_keeps_valid_confirmation():
    strategy, state = initial()
    state, _, _ = window(strategy, state, 100, good=["positive"])
    negative_condition = copy.deepcopy(state["confirmation_conditions"]["negative"])
    state, decision, _ = window(strategy, state, 200, good=["positive", "negative"])
    assert decision["decisions"]["positive"]["promoted"]
    assert state["stage"] == 1 and state["general_DR_frontier"] == 1
    assert state["streaks"]["negative"] == 1
    assert state["confirmation_conditions"]["negative"] == negative_condition
    state, decision, _ = window(strategy, state, 300, good=["negative"])
    assert decision["decisions"]["negative"]["promoted"]
    assert state["levels"] == {"positive": 1, "negative": 1}


def test_DR_change_cannot_reuse_previous_distribution_confirmation():
    strategy, state = initial()
    state, _, _ = window(strategy, state, 100, good=["positive"], robust=True)
    old_negative = state["confirmation_conditions"]["negative"]["sha256"]
    state, decision, _ = window(strategy, state, 200, good=["positive", "negative"], robust=True)
    assert decision["DR_promoted"] and decision["decisions"]["positive"]["promoted"]
    assert state["general_DR_frontier"] == 2 and state["levels"]["negative"] == 0
    assert state["streaks"]["negative"] == 0
    assert state["confirmation_conditions"]["negative"]["sha256"] != old_negative
    state, decision, _ = window(strategy, state, 300, good=["negative"])
    assert not decision["decisions"]["negative"]["promoted"]
    assert state["streaks"]["negative"] == 1
    state, decision, _ = window(strategy, state, 400, good=["negative"])
    assert decision["decisions"]["negative"]["promoted"]


@pytest.mark.parametrize("change", ["height", "target", "execution"])
def test_actual_condition_change_restarts_only_affected_chain(change):
    strategy, state = initial("motion")
    axis = "height_lower_forward"
    other = "turn_fp"
    state, _, _ = window(strategy, state, 100, good=[axis, other])
    old = state["confirmation_conditions"][axis]["sha256"]
    if change == "height":
        state["configuration"]["targets"][axis][0][2] = 0.31
    elif change == "target":
        state["configuration"]["targets"][axis][0][0] = 1.6
    else:
        state["configuration"]["confirmation_execution_sha256"] = "c" * 64
    state, decision, _ = window(strategy, state, 200, good=[axis, other])
    assert not decision["decisions"][axis]["promoted"]
    assert state["streaks"][axis] == 1
    assert state["confirmation_conditions"][axis]["sha256"] != old
    if change != "execution":
        assert decision["decisions"][other]["promoted"]
    else:
        assert state["streaks"][other] == 1


def test_unknown_breaks_confirmation_without_failure_or_level_regression():
    strategy, state = initial()
    state, _, _ = window(strategy, state, 100, good=["positive"])
    before = dict(state["levels"])
    state, decision, _ = window(strategy, state, 200)
    assert state["levels"] == before and not decision["state_revision_changed"]
    assert state["streaks"]["positive"] == state["regressions"]["positive"] == 0
    state, decision, _ = window(strategy, state, 300, good=["positive"])
    assert not decision["decisions"]["positive"]["promoted"]


def test_serialized_confirmation_restores_next_window_and_duplicate_is_idempotent(tmp_path):
    strategy, state = initial()
    state, _, _ = window(strategy, state, 100, good=["positive"])
    state, _, review = window(strategy, state, 200, good=["positive", "negative"])
    replayed, decision = strategy.advance(state, review, 200, envs=2560)
    assert replayed == state and decision["duplicate_window"]
    path = tmp_path / "course_state.pt"
    torch.save(state, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    continuous, _, _ = window(strategy, state, 300, good=["negative"])
    resumed, _, _ = window(strategy, restored, 300, good=["negative"])
    assert resumed == continuous
    assert resumed["levels"]["negative"] == 1
    assert len(resumed["progress_history"]) == 2


def test_three_early_failure_windows_only_regress_the_failing_direction():
    strategy, state = initial()
    state, _, _ = window(strategy, state, 100, good=["positive", "negative"])
    state, _, _ = window(strategy, state, 200, good=["positive", "negative"])
    state, _, _ = window(strategy, state, 300, good=["negative"], bad=["positive"])
    state, _, _ = window(strategy, state, 400, good=["negative"], bad=["positive"])
    state, decision, _ = window(strategy, state, 500, bad=["positive"])
    assert decision["decisions"]["positive"]["regressed"]
    assert not decision["decisions"]["negative"]["regressed"]
    assert state["levels"] == {"positive": 0, "negative": 2}
    assert state["spin_targets"]["positive"] == state["configuration"]["targets"]["positive"][0][1]
    assert state["progress_history"][-1]["regressed"]


def test_finite_spin_cap_clamps_next_endpoint_and_does_not_repeat_revision():
    strategy, state = initial(spin_terminal_cap=10.0)
    state, _, _ = window(strategy, state, 100, good=["positive", "negative"])
    state, decision, _ = window(strategy, state, 200, good=["positive", "negative"])
    assert decision["state_revision_changed"]
    assert state["spin_targets"] == {"positive": 10.0, "negative": -10.0}
    revision = state["stage"]
    for update in (300, 400, 500, 600):
        state, decision, _ = window(strategy, state, update, good=["positive", "negative"])
        assert not decision["state_revision_changed"] and state["stage"] == revision
    stage = strategy.stage_config(state["stage"], 2560, state=state)
    assert max(abs(yaw) for _, yaw in stage["commands"]) == 10.0


def test_exact_cap_never_creates_a_new_level():
    cfg = course_configuration("spin")
    cap = abs(cfg["targets"]["positive"][0][1])
    strategy, state = initial(spin_terminal_cap=cap)
    for update in (100, 200, 300, 400):
        state, decision, _ = window(strategy, state, update, good=["positive", "negative"])
        assert not decision["state_revision_changed"]
    assert state["levels"] == {"positive": 0, "negative": 0}


def test_null_spin_cap_and_separate_directions():
    strategy, state = initial(spin_terminal_cap=None)
    base = state["spin_targets"]["positive"]
    state, _, _ = window(strategy, state, 100, good=["positive"])
    state, _, _ = window(strategy, state, 200, good=["positive"])
    assert state["spin_targets"]["positive"] == pytest.approx(base * 1.15)
    assert state["spin_targets"]["negative"] == -base
    state, _, _ = window(strategy, state, 300, good=["positive"])
    state, _, _ = window(strategy, state, 400, good=["positive"])
    assert state["spin_targets"]["positive"] > 10.0


def test_growth_cannot_overflow_before_finite_check():
    strategy, state = initial(spin_growth=1e308)
    state, _, _ = window(strategy, state, 100, good=["positive"])
    state, _, _ = window(strategy, state, 200, good=["positive"])
    assert state["spin_targets"]["positive"] == MAX_COMMAND_MAGNITUDE
    assert math.isfinite(state["spin_targets"]["positive"])
    assert torch.isfinite(torch.tensor(state["spin_targets"]["positive"], dtype=torch.float32))
    before = state["stage"]
    state, _, _ = window(strategy, state, 300, good=["positive"])
    state, decision, _ = window(strategy, state, 400, good=["positive"])
    assert not decision["state_revision_changed"] and state["stage"] == before


@pytest.mark.parametrize("growth", [0, 1, -1, float("inf"), float("nan"), True, None, "bad"])
def test_invalid_spin_growth_rejected(growth):
    with pytest.raises(ValueError, match="增长倍率"):
        initial(spin_growth=growth)


@pytest.mark.parametrize(
    "change", ["positive_sign", "negative_sign", "vx", "cap", "cap_small", "cap_nan"]
)
def test_invalid_spin_direction_and_bounds_rejected(change):
    cfg = course_configuration("spin")
    if change == "positive_sign":
        cfg["targets"]["positive"][0][1] *= -1
    elif change == "negative_sign":
        cfg["targets"]["negative"][0][1] *= -1
    elif change == "vx":
        cfg["targets"]["positive"][0][0] = 0.5
    elif change == "cap":
        cfg["spin_terminal_cap"] = 1e40
    elif change == "cap_small":
        cfg["spin_terminal_cap"] = 8
    else:
        cfg["spin_terminal_cap"] = float("nan")
    with pytest.raises(ValueError):
        resolve(cfg["version"]).initial_state({"branch_course": cfg})


def test_candidate_reports_physical_quality_separately_from_spin_task_success():
    _, state = initial()
    review = report(state, "spin")
    for row in review["rows"]:
        if row["purpose"].startswith("anchor_"):
            row["pass_fraction"] = 0.0
    evidence = split_candidates.summarize(review, DR_level=1, comparison_sha256="a" * 64)
    assert evidence["eligible_by_objective"]["motion"]
    assert not evidence["task_passed_by_objective"]["motion"]
    status = evidence["objective_evidence"]["motion"]
    assert status["status"] == "TASK_GAP" and status["unpassed_targets"]
    assert not status["independent_certificate"]


def test_unknown_window_cannot_publish_a_qualified_candidate():
    _, state = initial()
    evidence = split_candidates.summarize(
        report(state, "spin", unknown=True), DR_level=1, comparison_sha256="a" * 64
    )
    assert not evidence["eligible"]
    assert all(row["status"] == "UNKNOWN" for row in evidence["objective_evidence"].values())
    index = split_candidates.empty_index("a" * 64, version=split_candidates.VERSION)
    updated = split_candidates.consider(index, evidence, None, {})
    assert updated["best_candidate"] is updated["best_standing_candidate"] is None


@pytest.mark.parametrize("bad", ["fake_zero", "impossible_pass_fraction"])
def test_candidate_rejects_missing_or_contradictory_evidence(bad):
    _, state = initial()
    review = report(state, "spin", unknown=bad == "fake_zero")
    row = review["rows"][0]
    if bad == "fake_zero":
        row["pass_fraction"] = 0.0
    else:
        row["quality_pass_fraction"] = 0.0
    with pytest.raises(ValueError):
        split_candidates.summarize(review, DR_level=1, comparison_sha256="a" * 64)


def test_candidates_keep_separate_DR_pools():
    _, state = initial()
    identity = "a" * 64
    index = split_candidates.empty_index(identity, version=split_candidates.VERSION)
    for level, update in ((1, 100), (2, 200)):
        review = report(state, "spin")
        review["through_update"] = update
        evidence = split_candidates.summarize(review, DR_level=level, comparison_sha256=identity)
        index = split_candidates.consider(
            index,
            evidence,
            dict(checkpoint=f"/tmp/fixture_{update}.pt", sha256=str(level) * 64),
            {},
        )
    assert set(index["candidates_by_objective"]["motion"]) == {"1", "2"}
    assert index["candidates_by_objective"]["motion"]["1"]["evidence"]["DR_level"] == 1
    assert index["candidates_by_objective"]["motion"]["2"]["evidence"]["DR_level"] == 2
