"""Readback and real-observation export boundaries without physical integration."""

import numpy as np
from test_public_migration import model_file

from native_stand.artifacts.checkpoint import load_actor_source
from native_stand.artifacts.deployment import export_package, verify_package
from native_stand.artifacts.io import file_sha256, write_npz
from native_stand.tasks.recovery.distillation_metrics import summarize

FROZEN_TEST_SCOPE = "PORTABLE"


def test_incomplete_episode_unknown_and_reset_cannot_hide_first_failure():
    data = dict(
        physical_episode_keys=np.array(["a", "a", "b", "c"]),
        episodes=np.array([0, 0, 0, 1]),
        terminated=np.array([0, 1, 0, 1], bool),
        truncated=np.zeros(4, bool),
        condition_entered=np.array([1, 1, 1, 1]),
        condition_dual_contact=np.array([1, 0, 1, 1]),
        condition_continuous_hold_s=np.array([0.1, 0, 9.99, 10.0]),
        condition_HF_peak=np.array([0.1, 0.2, 0.1, 0.1]),
        condition_validation_cohort=np.array([0, 0, 1, 0]),
        condition_initial_Swing=np.ones(4),
    )
    report = summarize(data)
    assert report["completed"] == 1 and report["UNKNOWN"] == 1 and report["first_episodes"] == 2
    assert report["continuous_10s_rate"] == 0 and report["settling_contact_loss_rate"] == 1


def test_export_uses_supplied_actual_rows_and_rejects_rebound_source(tmp_path):
    import pytest

    spec, path = model_file(tmp_path)
    data = tmp_path / "observed.npz"
    inputs = np.arange(5 * 31, dtype=np.float32).reshape(5, 31) / 100
    write_npz(data, dict(states=inputs))
    golden = dict(path=str(data), sha256=file_sha256(data), indices=[1, 3])
    output = tmp_path / "deployment"
    export_package(load_actor_source(path), spec, output, execution={}, golden=golden)
    record = verify_package(output)
    assert record["golden_source"] == golden
    with np.load(output / "golden.npz") as raw:
        np.testing.assert_array_equal(raw["inputs"], inputs[[1, 3]])
    with pytest.raises(ValueError, match="golden source differs"):
        export_package(
            load_actor_source(path), spec, output, execution={}, golden=dict(golden, indices=[0])
        )
