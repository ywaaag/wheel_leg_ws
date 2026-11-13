"""Cohort isolation and production adapter cadence are CPU-only diagnostics."""

import torch

from native_stand.tasks.motion.execution_statistics import ExecutionStatistics

FROZEN_TEST_SCOPE = "PORTABLE"


def test_cohort_signs_student_mask_empty_and_snapshot_ownership():
    stats = ExecutionStatistics([[2, 0.5], [2, -0.5], [-2, 0.5]], "cpu")
    raw = torch.tensor([[2.0] * 6, [-0.5] * 6, [float("nan")] * 6])
    requested = torch.tensor([[10.0] * 6, [-6.0] * 6, [float("nan")] * 6])
    adopted = torch.tensor([[8.0] * 6, [-5.0] * 6, [float("nan")] * 6])
    saturation = {
        "request_over_cap": torch.tensor([[True] * 6, [False] * 6, [True] * 6]),
        "driver_at_cap": torch.zeros(3, 6, dtype=torch.bool),
        "motor_at_cap": torch.ones(3, 6, dtype=torch.bool),
        "motor_capacity_fraction": torch.tensor([[0.9] * 6, [0.2] * 6, [float("nan")] * 6]),
    }
    stats.observe(
        torch.tensor([0, 1, 0]),
        torch.tensor([True, True, False]),
        raw,
        requested,
        adopted,
        saturation,
    )
    # 累计值不持有执行缓冲区引用；reset/后续写入不更改历史。
    requested.fill_(100)
    adopted.fill_(100)
    result = stats.export()
    a, b, empty = result["rows"]
    assert [r["samples"] for r in result["rows"]] == [1, 1, 0]
    assert a["mean"]["requested_target_mean"] == [10.0] * 6
    assert b["mean"]["adopted_target_mean"] == [-5.0] * 6
    assert a["mean"]["target_gap_abs_mean"] == [2.0] * 6
    assert b["mean"]["target_gap_abs_mean"] == [1.0] * 6
    assert a["mean"]["raw_clip_fraction"] == [1.0] * 6
    assert b["mean"]["request_over_cap_fraction"] == [0.0] * 6
    assert empty["mean"] is None
    assert result["not_review_window"] and result["not_physics_substep_integration"]
    assert stats.export() == result
