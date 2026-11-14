"""Real C evidence for Teacher/Jump execution mismatch; no physical integration."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from native_stand.simulation.jump_build import build
from native_stand.simulation.jump_unified import UnifiedJump

FROZEN_TEST_SCOPE = "PORTABLE"


@pytest.fixture(scope="module")
def boundary(tmp_path_factory):
    path = Path(__file__).resolve().parents[1] / "scripts/audit_jump_execution_boundary.py"
    spec = importlib.util.spec_from_file_location("jump_execution_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    binding = build(tmp_path_factory.mktemp("boundary") / "control.so")
    control = UnifiedJump(binding["path"], 1, library_sha=binding["sha256"])
    return module.audit(control)


def test_zero_PD_is_not_zero_motor_and_mapping_is_not_symmetric(boundary):
    frame = boundary["cases"]["zero_PD"]["original"]
    np.testing.assert_array_equal(frame["pd_nm"], np.zeros(6))
    np.testing.assert_allclose(
        frame["final_model_nm"], [14.427342, 14.427342, 0, -14.427342, 14.427342, 0], atol=2e-6
    )
    assert boundary["ABI"] == 4


@pytest.mark.parametrize("case", ["zero_PD", "asymmetric_with_D", "saturated_representable"])
def test_true_C_same_input_correction_matches_leg_boundary_only(boundary, case):
    result = boundary["cases"][case]
    assert result["corrected_target_inside_guard"]
    assert result["corrected_leg_max_error_nm"] < 2e-6
    assert result["input_and_FSM_unchanged"]


def test_same_instant_identity_does_not_qualify_future_or_unrepresentable_targets(boundary):
    assert boundary["changed_geometry_with_stale_correction"]["max_leg_error_nm"] > 1.0
    assert not boundary["cases"]["outside_target_guard"]["corrected_target_inside_guard"]
    assert boundary["C_motor_calls"] == 13
    for key in (
        "new_physics_ticks",
        "new_GPU_runs",
        "new_PPO_updates",
        "new_BC_updates",
        "V4_steps",
    ):
        assert boundary[key] == 0
