"""Actual Motion statistics/export/adapter boundaries; physics I/O is scripted."""

import copy
from types import SimpleNamespace as NS

import pytest
import torch

from native_stand.tasks.motion import split_candidates
from native_stand.tasks.motion.maneuver_statistics import RESPONSE_VERSION, episode_class

FROZEN_TEST_SCOPE = "PORTABLE"


def window(start=1, end=100, revision=0):
    return dict(
        run="fixture",
        comparison_sha256="a" * 64,
        stage_revision=revision,
        start_update=start,
        end_update=end,
    )


def environment():
    stage = dict(
        id=0,
        commands=[[0, 0], [0, 0]],
        cohort_keys=[["clean"], ["DR"]],
        goal_heights_m=[0.34, 0.34],
        cohort_purposes=["clean_standing", "dq_DR"],
        identified_profiles=[-1, 0],
        maneuver_tasks=["standing"] * 2,
        slot_counts=[1, 1],
    )
    e = NS(
        num_envs=2,
        device="cpu",
        stage_spec=stage,
        command_index=torch.arange(2),
        student_actor_valid=torch.tensor([True, False]),
        rise_hold=NS(episode=NS(steps=torch.full((2,), 3000)), offset=torch.zeros(2)),
        motion_action_command=torch.tensor([[0.0, 0.0, 0.34]] * 2),
        recovery_snapshot=torch.zeros(2, 31),
        standing_snapshot=torch.zeros(2, 6),
        maneuver_goal_height=torch.full((2,), 0.34),
        antiosc_physical_stats_post=torch.zeros(2, 9),
        motion_spin_measurement_post={"rolling_speed_forward_mps": torch.zeros(2, 2)},
        maneuver_swing_difference=torch.zeros(2),
    )
    e.recovery_snapshot[:, 6] = 0.34
    e.recovery_snapshot[:, 10:12] = 10
    e.onpolicy_episodes = episode_class(object, version=RESPONSE_VERSION)(e)
    return e


def step(e, *, ended=False):
    done = torch.full((2,), ended, dtype=torch.bool)
    e.onpolicy_episodes.observe_final((done, torch.zeros_like(done)))


def test_continuous_windows_split_student_all_transitions_and_lifetime():
    e = environment()
    s = e.onpolicy_episodes
    s.bind_sample_window(window())
    step(e, ended=True)
    first = s.export(100, window=window())
    assert first["valid_student_steps"] == 1 and first["all_transitions"] == 2
    assert [r["actual_student_policy_samples"] for r in first["rows"]] == [1, 0]
    assert [r["actual_all_transitions"] for r in first["rows"]] == [1, 1]
    assert first["rows"][0]["ended_episodes"] == 1  # Early failure remains in denominator.
    assert first["rows"][1]["pass_fraction"] is None
    s.reset(torch.tensor([0]))  # Partial reset changes no window or lifetime counters.
    e.student_actor_valid[:] = True
    step(e, ended=True)
    second = s.export(200, window=window(101, 200))
    assert second["valid_student_steps"] == second["all_transitions"] == 2
    assert second["sample_accounting"]["cumulative_student_policy_samples"] == 3
    assert second["sample_accounting"]["cumulative_all_transitions"] == 4
    assert second["observed_policy_steps"] == 1
    assert second["sample_accounting"]["cumulative_observed_policy_steps"] == 2


def test_duplicate_export_does_not_consume_the_following_window():
    e = environment()
    s = e.onpolicy_episodes
    s.bind_sample_window(window())
    step(e, ended=True)
    first = s.export(100, window=window())
    step(e, ended=True)
    assert s.export(100, window=window()) == first
    with pytest.raises(ValueError, match="conflicting model update"):
        s.export(101, window=window())
    second = s.export(200, window=window(101, 200))
    assert second["valid_student_steps"] == 1
    with pytest.raises(ValueError, match="origin differs"):
        s.export(100, window=window())


def test_partial_window_checkpoint_restore_and_duplicate_report(tmp_path):
    e = environment()
    s = e.onpolicy_episodes
    s.bind_sample_window(window())
    step(e, ended=True)
    path = tmp_path / "statistics.pt"
    torch.save(s.completed_state(), path)
    restored = environment().onpolicy_episodes
    restored.restore_completed(torch.load(path, weights_only=False))
    restored.bind_sample_window(window())
    assert restored.export(100, window=window()) == s.export(100, window=window())
    again = environment().onpolicy_episodes
    again.restore_completed(restored.completed_state())
    assert again.export(100, window=window()) == restored.export(100, window=window())


def test_legacy_checkpoint_window_unknown_until_next_declared_window():
    e = environment()
    s = e.onpolicy_episodes
    s.bind_sample_window(window())
    step(e, ended=True)
    record = s.completed_state()
    record.pop("sample_accounting")
    record.pop("last_export_identity")
    record.pop("last_export_report")
    new = environment()
    restored = new.onpolicy_episodes
    restored.restore_completed(record)
    restored.bind_sample_window(window())
    step(new, ended=True)
    first = restored.export(100, window=window())
    assert first["sample_accounting"]["window_status"] == "UNKNOWN"
    assert first["sample_accounting"]["window_origin"] is None
    assert first["valid_student_steps"] is first["all_transitions"] is None
    assert first["sample_accounting"]["cumulative_student_policy_samples"] == 2
    assert first["sample_accounting"]["cumulative_all_transitions"] is None
    assert first["sample_accounting"]["cumulative_observed_policy_steps"] is None
    step(new, ended=True)
    second = restored.export(200, window=window(101, 200))
    assert second["sample_accounting"]["window_status"] == "KNOWN"
    assert second["valid_student_steps"] == 1 and second["all_transitions"] == 2


def test_rebuild_preserves_global_counts_but_starts_new_layout_window():
    e = environment()
    old = e.onpolicy_episodes
    old.bind_sample_window(window())
    step(e)
    old.export(100, window=window())
    state = old.sample_accounting.state_dict()
    new = environment()
    new.stage_spec["id"] = 1
    stats = new.onpolicy_episodes
    stats.inherit_sample_totals(state)
    stats.bind_sample_window(window(101, 200, revision=1))
    step(new)
    report = stats.export(200, window=window(101, 200, revision=1))
    assert report["valid_student_steps"] == 1 and report["all_transitions"] == 2
    assert report["sample_accounting"]["cumulative_student_policy_samples"] == 2
    assert report["sample_accounting"]["cumulative_all_transitions"] == 4


@pytest.mark.parametrize("kind", ["origin", "dtype", "negative", "version"])
def test_corrupt_or_conflicting_sample_state_is_rejected(kind):
    e = environment()
    s = e.onpolicy_episodes
    s.bind_sample_window(window())
    if kind == "origin":
        with pytest.raises(ValueError, match="origin differs"):
            s.bind_sample_window(window(2, 100))
        return
    record = s.completed_state()
    a = record["sample_accounting"]
    if kind == "dtype":
        a["total"] = a["total"].float()
    elif kind == "negative":
        a["total"][0] = -1
    else:
        a["version"] = "unknown"
    with pytest.raises(ValueError, match="sample"):
        environment().onpolicy_episodes.restore_completed(record)


def candidate_review():
    from test_motion_spin_branches import course_configuration, report

    from native_stand.tasks.motion.course_strategy import resolve

    cfg = course_configuration("motion")
    state = resolve(cfg["version"]).initial_state({"branch_course": cfg})
    return report(state, "motion")


def test_candidate_multiple_reasons_thresholds_and_unchanged_ranking():
    report = candidate_review()
    for row in report["rows"]:
        if row["axis"] == "standing":
            row.update(quality_pass_fraction=0.6, pass_fraction=0.4)
    result = split_candidates.summarize(report, DR_level=1, comparison_sha256="a" * 64)
    standing = result["objective_evidence"]["standing"]
    assert standing["covered"] and standing["status"] == "TASK_GAP"
    assert {r["reason"] for r in standing["reasons"]} == {
        "quality_insufficient",
        "task_insufficient",
    }
    assert result["thresholds"]["minimum_quality_fraction"] == 0.7
    assert result["standing_rank"][:2] == [0.4, 0.6]
    assert not result["eligible_by_objective"]["standing"]


def test_candidate_quality_failure_is_not_missing_coverage():
    report = candidate_review()
    for row in report["rows"]:
        row.update(quality_pass_fraction=0.5, pass_fraction=0.4)
    result = split_candidates.summarize(report, DR_level=1, comparison_sha256="a" * 64)
    assert not result["eligible"] and result["reason"] == "quality_insufficient"
    assert all(o["covered"] for o in result["objective_evidence"].values())


def test_candidate_task_gap_keeps_the_original_quality_eligibility_and_rank():
    report = candidate_review()
    for row in report["rows"]:
        if row["axis"] == "standing":
            row["pass_fraction"] = 0.4
    result = split_candidates.summarize(report, DR_level=1, comparison_sha256="a" * 64)
    assert result["eligible_by_objective"]["standing"]
    assert not result["task_passed_by_objective"]["standing"]
    assert result["standing_rank"] == [0.4, 1.0, 1.0, -0.01]
    assert {r["reason"] for r in result["objective_evidence"]["standing"]["reasons"]} == {
        "task_insufficient"
    }
    assert result["thresholds"]["task_pass_fraction"] == 0.7


def test_candidate_unknown_is_not_a_measured_quality_or_task_failure():
    report = candidate_review()
    for row in report["rows"]:
        row.update(
            ended_episodes=0,
            completed_episodes=0,
            pass_fraction=None,
            quality_pass_fraction=None,
            mean_dual_support_fraction=None,
            physical_HF_RMS_rad_s=None,
        )
    before = copy.deepcopy(report)
    result = split_candidates.summarize(report, DR_level=1, comparison_sha256="a" * 64)
    assert result["reason"] == "coverage_insufficient" and report == before
    assert {o["status"] for o in result["objective_evidence"].values()} == {"UNKNOWN"}
    assert {r["reason"] for r in result["reasons"]} == {"coverage_insufficient"}


def test_candidate_small_sample_keeps_quality_and_task_unknown():
    report = candidate_review()
    for row in report["rows"]:
        row.update(
            ended_episodes=8, completed_episodes=0, pass_fraction=0.0, quality_pass_fraction=0.0
        )
    result = split_candidates.summarize(report, DR_level=1, comparison_sha256="a" * 64)
    assert {r["reason"] for r in result["reasons"]} == {"coverage_insufficient"}
    assert all(r["status"] == "UNKNOWN" for r in result["objective_evidence"].values())


def test_real_adapter_capture_save_load_preserves_window_origin(tmp_path, monkeypatch):
    from test_motion_spin_branches import equal_tree, fixture
    from test_phase3_fixes import _production_context

    from native_stand.artifacts.checkpoint import load_complete_state
    from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter
    from native_stand.students.resume import rng_state

    job, source, obs = fixture(tmp_path)
    context = _production_context(tmp_path / "run", job, smoke=1)

    def physical_boundary(adapter):
        adapter.env = NS(
            num_envs=4,
            device="cpu",
            stage_spec=adapter.spec["curriculum"]["stages"][0],
            identified_dr=NS(episodes=torch.ones(4, dtype=torch.long)),
        )
        adapter.env.onpolicy_episodes = episode_class(object, version=RESPONSE_VERSION)(adapter.env)
        adapter.obs = obs
        adapter.modules = {"rng_state": rng_state}

    monkeypatch.setattr(MotionProductionAdapter, "_open_environment", physical_boundary)
    monkeypatch.setattr(MotionProductionAdapter, "_write_restoration", lambda self: None)
    first = MotionProductionAdapter()
    first.build(context, "smoke", None)
    first.env.onpolicy_episodes.sample_accounting.observe(
        torch.zeros(4, dtype=torch.long), torch.tensor([True, False, True, False])
    )
    saved = first.capture(context, {})
    reference = context.save(saved, update=0, stage="smoke")
    second = MotionProductionAdapter()
    second.build(context, "smoke", load_complete_state(reference["checkpoint"]))
    try:
        equal_tree(
            saved["motion_sample_accounting"],
            second.env.onpolicy_episodes.sample_accounting.state_dict(),
        )
        equal_tree(source["optimizer"], second.algorithm.optimizer.state_dict())
        assert (
            second.env.onpolicy_episodes.sample_accounting.summary()["student_policy_samples"] == 2
        )
        assert second.env.onpolicy_episodes.sample_accounting.summary()["all_transitions"] == 4
    finally:
        context.metrics.close()
