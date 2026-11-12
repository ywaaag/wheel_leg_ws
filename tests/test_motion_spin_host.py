"""Explicit host tests of selected archives; not part of the portable suite."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

FROZEN_TEST_SCOPE = "HOST"
ROOT = Path(__file__).resolve().parents[1]


def test_real_u3000_both_full_branches_capture_resume():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests/fixtures/motion_spin_branch_probe.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipts = json.loads(result.stdout)
    assert [row["branch"] for row in receipts] == ["motion", "spin"]
    for row in receipts:
        assert (
            row["source_sha256"]
            == "abe10db68b0ad62a3e0ceac5285ebb0041ff084134bc313edb235ae46d427d7e"
        )
        assert row["actor_equal"] and row["critic_equal"] and row["two_Adam_equal"]
        assert row["training_RNG_preserved"] and row["checkpoint_saved_and_restored"]
        assert row["branch_updates"] == row["physical_steps"] == row["PPO_updates"] == 0
        assert row["model_progress"] == 5097 and row["consumed_authorization"] == 5111
        assert row["shared_balance_unallocated"] == 4889 and row["warmup_done"] == 50


@pytest.mark.parametrize("branch", ["motion", "spin"])
def test_real_prepared_spec_keeps_enabled_speed_options(branch):
    from native_stand.artifacts.runtime_paths import execution_json
    from native_stand.experiments.config import load_job_config
    from native_stand.simulation.isaac.motion_backend import _resolve_source
    from native_stand.students.performance import settings
    from native_stand.tasks.motion.course_strategy import resolve
    from native_stand.tasks.motion.legacy_spec_adapter import build_environment_spec

    job = load_job_config(ROOT / f"configs/jobs/{branch}48_u3000_prepared.json")
    loaded = _resolve_source(job, None)
    spec = build_environment_spec(
        execution_json(job.assets["runtime_template"], job.assets),
        loaded.payload,
        job,
        stage_builder=resolve(job.curriculum.version).stage_config,
    )
    performance = settings(spec)
    for name in (
        "motion_hotpath",
        "motion_runtime_reduction",
        "combined_step_checks",
        "fixed_shape_statistics",
        "contact_history_disabled",
    ):
        assert performance[name] is True
    assert spec["student_training"]["no_independent_evaluation"] is True
    assert job.storage.save_traces is False and job.runtime.headless is True
    assert job.runtime.steps_per_env == 64
    assert job.storage.checkpoint_every_updates == 100
