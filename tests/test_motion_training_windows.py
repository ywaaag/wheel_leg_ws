"""Normal PPO window consumption and shared reference authority; CPU only."""

import copy
from pathlib import Path

import pytest
import torch

from native_stand.artifacts.io import file_sha256, read_json
from native_stand.artifacts.model_refs import ModelIndex, ModelReference
from native_stand.artifacts.retention import SavePolicy, plan_prune, prune, write_inventory
from native_stand.tasks.motion import c0_boundary_course
from native_stand.tasks.motion.course_strategy import resolve
from test_motion_course_resume import source95, review

FROZEN_TEST_SCOPE = "PORTABLE"


def window_report(state, end, axes=()):
    result = review(state, axes)
    result["window"] = dict(
        run="/tmp/example-run",
        comparison_sha256="c" * 64,
        stage_revision=state["stage"],
        start_update=end - 99,
        end_update=end,
    )
    return result


def test_window_is_idempotent_and_resume_equivalent(tmp_path):
    strategy = resolve(c0_boundary_course.VERSION)
    state = strategy.prepare(source95(), 4096)
    state["last_training_update"] = -1
    report = window_report(state, 100, ["forward"])
    state, _ = strategy.advance(state, report, 100, envs=4096)
    duplicate, decision = strategy.advance(state, report, 100, envs=4096)
    assert duplicate == state and decision["duplicate_window"]
    bad = copy.deepcopy(report)
    bad["rows"][0]["ended_episodes"] += 1
    with pytest.raises(ValueError, match="冲突"):
        strategy.advance(state, bad, 100, envs=4096)
    path = tmp_path / "state.pt"
    torch.save(state, path)
    restored = torch.load(path, weights_only=False)
    second = window_report(state, 200, ["forward", "yaw_positive"])
    uninterrupted, _ = strategy.advance(state, second, 200, envs=4096)
    resumed, _ = strategy.advance(restored, second, 200, envs=4096)
    assert uninterrupted == resumed
    assert resumed["pass_streak"]["yaw_positive"] == 1
    assert resumed["frontiers"]["forward"] == state["frontiers"]["forward"] + 1


def test_unknown_interrupts_confirmations_without_regression():
    strategy = resolve(c0_boundary_course.VERSION)
    state = strategy.prepare(source95(), 4096)
    state["last_training_update"] = -1
    state, _ = strategy.advance(state, window_report(state, 100, ["forward"]), 100, envs=4096)
    before = copy.deepcopy(state)
    report = window_report(state, 200)
    for row in report["rows"]:
        row.update(
            ended_episodes=0, completed_episodes=0, pass_fraction=None, quality_pass_fraction=None
        )
    after, _ = strategy.advance(state, report, 200, envs=4096)
    assert after["frontiers"] == before["frontiers"] and after["levels"] == before["levels"]
    assert not any(after["pass_streak"].values()) and not any(after["regression_streak"].values())
    report["rows"][0]["quality_pass_fraction"] = float("nan")
    with pytest.raises(ValueError):
        strategy.advance(before, report, 200, envs=4096)


def checkpoints(tmp_path):
    run = tmp_path / "runs" / "parent"
    (run / "checkpoints").mkdir(parents=True)
    rows = []
    for update in range(1, 5):
        path = run / "checkpoints" / f"model_update_{update}.pt"
        path.write_bytes(bytes([update]))
        rows.append(dict(checkpoint=str(path.resolve()), sha256=file_sha256(path), update=update))
    write_inventory(run, rows)
    return run, rows, ModelIndex(tmp_path / "runs" / "models.json")


def test_cross_run_multiple_consumers_and_plan_race(tmp_path):
    run, rows, index = checkpoints(tmp_path)
    first = rows[0]
    plan = plan_prune(run, SavePolicy())
    assert first["checkpoint"] in plan["removed"]
    index.register(
        ModelReference(first["sha256"], first["checkpoint"], ("initialization", "teacher"), "child")
    )
    index.register(
        ModelReference(first["sha256"], first["checkpoint"], ("deployment",), "deploy-operation")
    )
    index.release(consumer="child", role="initialization")
    assert first["checkpoint"] in index.consumer_paths()
    # Execute the stale plan through the actual deletion boundary: reference wins.
    result = index.delete_owned(first["checkpoint"], first["sha256"], run_dir=run)
    assert not result["deleted"] and Path(first["checkpoint"]).is_file()
    report = prune(run, SavePolicy(), index=index)
    assert report["removed"] == [rows[1]["checkpoint"]]
    assert len(index.rows()[0]["consumers"]) == 2
    index.release(consumer="child", role="teacher")
    assert first["checkpoint"] in index.consumer_paths()
    index.release(consumer="deploy-operation", role="deployment")
    result = index.delete_owned(first["checkpoint"], first["sha256"], run_dir=run)
    assert result["deleted"] and result["logical_bytes"] == 1
    with pytest.raises(ValueError, match="删除"):
        index.register(
            ModelReference(first["sha256"], first["checkpoint"], ("teacher",), "late-child")
        )


def test_atomic_publication_recovery_and_generation(tmp_path):
    from native_stand.tasks.motion import motion_candidates

    run, rows, index = checkpoints(tmp_path)
    candidates = motion_candidates.empty_index("c" * 64)
    candidates["last_review"] = dict(through_update=1)
    candidates["candidates_by_objective"]["motion"]["0"] = rows[0]
    index.publish_candidates("parent", candidates)
    assert index.candidate_index("parent") == candidates
    assert rows[0]["checkpoint"] in index.consumer_paths()
    generation = read_json(index.path)["generation"]
    index.publish_candidates("parent", copy.deepcopy(candidates))
    assert read_json(index.path)["generation"] == generation
    conflicting = copy.deepcopy(candidates)
    conflicting["comparison_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="比较域"):
        index.publish_candidates("parent", conflicting)


def test_publication_crash_after_commit_replays_without_generation_change(tmp_path):
    from types import SimpleNamespace as NS
    from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter
    from native_stand.tasks.motion import motion_candidates

    run, rows, index = checkpoints(tmp_path)
    state = resolve(c0_boundary_course.VERSION).prepare(source95(), 4096)
    report = review(state)
    report.update(
        version="motion_progressive_complete_episode_v1",
        independent_evaluation=False,
        extra_simulation_steps=0,
        through_update=100,
        course_version=c0_boundary_course.VERSION,
    )
    for row in report["rows"]:
        row.update(
            ended_episodes=0,
            completed_episodes=0,
            passed_episodes=0,
            physical_quality_passed=0,
            pass_fraction=None,
            quality_pass_fraction=None,
            mean_HF_tail_rad_s=None,
            failed_check_counts={"support": 0, "continuous_loss": 0},
            movement_components={"episode_means": None},
        )
    evidence = motion_candidates.summarize(report, DR_level=0, comparison_sha256="c" * 64)
    evidence["through_update"] = 1
    adapter = MotionProductionAdapter()
    adapter.candidates = motion_candidates.empty_index("c" * 64)
    adapter._candidate_pending = evidence
    adapter.source = {}
    adapter.source_path = tmp_path / "source.pt"
    adapter.source_sha = "a" * 64

    class View:
        failed = True

        def update(self, **kwargs):
            if self.failed:
                raise OSError("crash before view write")

    view = View()
    context = NS(refs=index, run_dir=run, store=view)
    with pytest.raises(OSError, match="crash"):
        adapter.after_save(context, rows[0])
    generation = read_json(index.path)["generation"]
    assert index.candidate_index(run.name)["last_review"]["evidence"] == evidence
    view.failed = False
    # Simulate recovery's minimal file reference (not a new training update).
    adapter.after_save(context, {k: rows[0][k] for k in ("checkpoint", "sha256", "update")})
    assert adapter._candidate_pending is None
    assert read_json(index.path)["generation"] == generation
    assert read_json(run / "training_candidates.json") == index.candidate_index(run.name)


def test_hardlink_deletion_reports_no_released_blocks(tmp_path):
    import os

    run, rows, index = checkpoints(tmp_path)
    path = Path(rows[0]["checkpoint"])
    os.link(path, tmp_path / "other-link.pt")
    result = index.delete_owned(path, rows[0]["sha256"], run_dir=run)
    assert result["logical_bytes"] == 1 and result["released_bytes"] == 0
    assert (tmp_path / "other-link.pt").read_bytes() == b"\x01"


def test_owner_persists_shared_index_and_runs_root_isolation(tmp_path):
    from test_motion_production_integration import make_job
    from native_stand.experiments.owner import create_run

    job, _ = make_job(tmp_path)
    first = create_run(job, runs_root=tmp_path / "first", run_id="parent")
    child = create_run(job, runs_root=tmp_path / "first", run_id="child")
    isolated = create_run(job, runs_root=tmp_path / "second", run_id="parent")
    assert (
        read_json(first / "run.json")["model_index"] == read_json(child / "run.json")["model_index"]
    )
    assert (
        read_json(first / "run.json")["model_index"]
        != read_json(isolated / "run.json")["model_index"]
    )


def test_delete_intent_survives_interruption_and_blocks_acquisition(tmp_path, monkeypatch):
    run, rows, index = checkpoints(tmp_path)
    row = rows[0]
    original = Path.unlink

    def interrupted(path, *args, **kwargs):
        if str(path) == row["checkpoint"]:
            raise OSError("interrupted unlink")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        index.delete_owned(row["checkpoint"], row["sha256"], run_dir=run)
    key = f"{row['sha256']}\t{row['checkpoint']}"
    assert read_json(index.path)["deletions"][key]["status"] == "intent"
    with pytest.raises(ValueError, match="删除"):
        index.register(ModelReference(row["sha256"], row["checkpoint"], ("teacher",), "child"))
    monkeypatch.setattr(Path, "unlink", original)
    assert index.delete_owned(row["checkpoint"], row["sha256"], run_dir=run)["deleted"]


def test_output_retains_multiple_real_dependencies_after_role_release(tmp_path):
    run, rows, index = checkpoints(tmp_path)
    for row in rows[:2]:
        index.register(ModelReference(row["sha256"], row["checkpoint"], ("teacher",), "child"))
    output = tmp_path / "child-output.pt"
    output.write_bytes(b"output")
    index.register_output(
        ModelReference(file_sha256(output), str(output), (), owner="child"), consumer="child"
    )
    index.release(consumer="child", role="teacher")
    assert {r["checkpoint"] for r in rows[:2]} <= index.consumer_paths()
    assert not index.delete_owned(rows[0]["checkpoint"], rows[0]["sha256"], run_dir=run)["deleted"]


def test_concurrent_reference_and_delete_share_the_same_lock(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    run, rows, index = checkpoints(tmp_path)
    row = rows[0]
    barrier = Barrier(2)

    def acquire():
        barrier.wait()
        try:
            index.register(ModelReference(row["sha256"], row["checkpoint"], ("teacher",), "child"))
            return True
        except ValueError:
            return False

    def delete():
        barrier.wait()
        return index.delete_owned(row["checkpoint"], row["sha256"], run_dir=run)["deleted"]

    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(acquire)
        second = executor.submit(delete)
        acquired, deleted = first.result(), second.result()
    assert acquired != deleted
    assert Path(row["checkpoint"]).exists() == acquired
