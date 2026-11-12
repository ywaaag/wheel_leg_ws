"""真实worker的已结算PPO恢复边界；仅tmp小网络CPU fixture。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from native_stand.artifacts.checkpoint import load_source
from native_stand.artifacts.io import read_json
from native_stand.experiments import worker
from native_stand.experiments.budget import Budget
from native_stand.experiments.config import load_job_config
from native_stand.experiments.owner import create_run
from native_stand.experiments.state import RunStateStore, StateError

FROZEN_TEST_SCOPE = "PORTABLE"


def _stopped_cpu_run(tmp_path):
    data = json.loads((Path(__file__).parent / "fixtures/cpu_motion_job.json").read_text())
    data["limits"].update(smoke_updates=0, formal_updates=4)
    data["runtime"].update(envs=2, steps_per_env=2)
    path = tmp_path / "job.json"
    path.write_text(json.dumps(data))
    job = load_job_config(path)
    run = create_run(job, runs_root=tmp_path / "runs", run_id="resume_cpu")
    Budget.create(run / "budget.json", smoke_updates=0, formal_updates=4)
    RunStateStore(run).create(run_id=run.name, owner={"pid": 0}, stage="formal")
    assert worker.main(["--run-dir", str(run), "--stage", "formal", "--max-updates", "1"]) == 0
    source = load_source(RunStateStore(run).read()["latest_model"]["checkpoint"])
    return run, source


def test_real_worker_resumes_after_normal_stop_without_resetting_budget(tmp_path):
    run, source = _stopped_cpu_run(tmp_path)
    before = read_json(run / "budget.json")
    assert (
        worker.main(
            [
                "--run-dir",
                str(run),
                "--stage",
                "formal",
                "--resume-checkpoint",
                str(source.path),
                "--max-updates",
                "1",
            ]
        )
        == 0
    )
    state, book = RunStateStore(run).read(), read_json(run / "budget.json")
    assert state["status"] == "STOPPED_BOUNDED" and state["failure"] is None
    assert state["updates_used"] == book["used"]["formal"] == 2
    assert state["samples_used"] == 8
    assert book["limits"] == before["limits"]
    assert len(book["attempts"]) == 2
    assert all(row["updates"] == 1 for row in book["attempts"].values())
    restored = load_source(state["latest_model"]["checkpoint"])
    assert restored.payload["warmup_updates_done"] == source.payload["warmup_updates_done"] == 1
    assert restored.payload["counters"]["updates"] == 2


def test_terminal_claim_without_verified_resume_still_rejected(tmp_path):
    run, _source = _stopped_cpu_run(tmp_path)
    context = worker._new_context(run, "formal", suffix="unverified")
    try:
        with pytest.raises(StateError, match="终态"):
            context.startup_stage("formal")
    finally:
        context.budget.finalize(context.attempt_id, context.output, "FAILED", reason="fixture")
        context.metrics.close()


def test_motion_resume_uses_local_counter_instead_of_ancestor_progress(tmp_path):
    run, source = _stopped_cpu_run(tmp_path)
    source.payload["counters"] = {"new_run_updates": 1, "source_online_updates": 5097}
    source.payload["motion_course_state"] = {"source_history": "unchanged"}
    context = worker._new_context(run, "formal", suffix="motion_counter")
    try:
        context.prepare_resume(source)
        assert context.resume_verified
        assert context.updates_used == 1
        assert context.store.read()["status"] == "STOPPED_BOUNDED"
    finally:
        context.budget.finalize(context.attempt_id, context.output, "FAILED", reason="fixture")
        context.metrics.close()


@pytest.mark.parametrize("mismatch", ["directory", "counter", "failed"])
def test_resume_rejects_foreign_stale_or_failed_state(tmp_path, mismatch):
    run, source = _stopped_cpu_run(tmp_path)
    context = worker._new_context(run, "formal", suffix=mismatch)
    if mismatch == "directory":
        source.payload["run_identity"] = str(tmp_path / "other_run")
    elif mismatch == "counter":
        source.payload["counters"]["updates"] = 0
    else:
        state = read_json(run / "state.json")
        state["status"] = "FAILED"
        (run / "state.json").write_text(json.dumps(state))
    try:
        with pytest.raises(ValueError):
            context.prepare_resume(source)
        assert not context.resume_verified
        assert RunStateStore(run).read()["updates_used"] == 1
    finally:
        context.budget.finalize(context.attempt_id, context.output, "FAILED", reason="fixture")
        context.metrics.close()
