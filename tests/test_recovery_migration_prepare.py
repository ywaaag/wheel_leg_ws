"""Local preparation never changes the source ledger or starts a worker."""

import importlib.util
from pathlib import Path

import pytest

from native_stand.artifacts.io import read_json
from native_stand.experiments.config import job_from_record

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "prepare", ROOT / "scripts/prepare_recovery_continuation.py"
    )
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def test_actual_settled_source_preparation_and_unavailable_candidate(tmp_path):
    m = module()
    receipt = m.prepare(
        Path("/home/kellen/wheel_leg_native_train/runs") / m.SOURCE_RUN,
        ROOT / "configs/jobs/recovery_preentry48_remaining.json",
        "/home/kellen/wheel_leg_training_data",
        tmp_path / "prepared",
    )
    assert not receipt["ready_for_submission"] and not receipt["worker_started"]
    assert receipt["remaining"] == 6534 and receipt["cumulative_used"] == 3466
    roles = {row["role"]: row for row in receipt["protected_models"]}
    assert roles["best_candidate"]["status"] == "UNAVAILABLE"
    assert (
        roles["latest_model"]["status"] == roles["best_standing_candidate"]["status"] == "AVAILABLE"
    )
    job = job_from_record(read_json(tmp_path / "prepared/job_record.json"))
    assert job.initialization.mode == "full" and job.limits.formal_updates == 8334
    assert job.ppo.critic_warmup_updates == 5
    assert job.assets.get("physical_gpu_uuid") is None
    assert str(job.initialization.checkpoint).startswith("/home/kellen/wheel_leg_training_data/")
    with pytest.raises(ValueError, match="existing output"):
        m.prepare(
            Path("/home/kellen/wheel_leg_native_train/runs") / m.SOURCE_RUN,
            ROOT / "configs/jobs/recovery_preentry48_remaining.json",
            "/home/kellen/wheel_leg_training_data",
            tmp_path / "prepared",
        )
