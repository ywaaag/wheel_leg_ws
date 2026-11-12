"""迁移前后数值一致：输入编码、动作解码、previous与探索尺度。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from native_stand.students import interface as legacy  # noqa: E402
from native_stand.students.motion_centered_output import (  # noqa: E402
    HeightEpisodeOutput,
)
from native_stand.students.motion_kinematic_output import (  # noqa: E402
    VERSION as GEOMETRIC_VERSION,
)
from native_stand.students.motion_kinematic_output import KinematicMotionOutput  # noqa: E402
from native_stand.students.motion_supported_output import (  # noqa: E402
    VERSION as SUPPORTED_VERSION,
)
from native_stand.students.motion_supported_output import (  # noqa: E402
    SupportedKinematicMotionOutput,
)
from native_stand.tasks.motion import encoding  # noqa: E402

FROZEN_TEST_SCOPE = "PORTABLE"  # 自包含CPU工程契约：tmp fixture + 真实代码，无本机runs/资产依赖

GEOMETRY = {
    "version": "flat_two_link_geometry_v1",
    "hip_above_root_m": 2.4e-05,
    "wheel_radius_m": 0.06,
    "equivalent_upper_m": 0.21,
    "equivalent_lower_m": 0.25,
    "hip_zero_deg": 47.1033888,
    "active_zero_deg": 47.1080837,
    "clearance_m": 0.0,
}
OFFSET = (-0.24229908821701296, -0.3063761712607683, 0.24229908821701296, -0.3063761712607683)


def _inputs(rows: int = 3):
    generator = torch.Generator().manual_seed(7)
    q = torch.rand((rows, 6), generator=generator)
    dq = torch.rand((rows, 6), generator=generator) - 0.5
    reference = torch.zeros((rows, 6))
    previous_legs = torch.rand((rows, 4), generator=generator)
    previous_wheels = torch.rand((rows, 2), generator=generator) * 3 - 1.5
    gravity = torch.zeros((rows, 3))
    gravity[:, 2] = 1.0
    angular = torch.rand((rows, 3), generator=generator)
    return angular, gravity, q, dq, previous_legs, previous_wheels, reference


def test_frame_encoding_matches_legacy_exactly():
    angular, gravity, q, dq, previous_legs, previous_wheels, reference = _inputs()
    expected = legacy.frame(
        angular, gravity, q, dq, previous_legs, previous_wheels, reference
    )
    actual = encoding.encode_frame(
        angular, gravity, q, dq, previous_legs, previous_wheels, reference
    )
    assert torch.equal(actual, expected)
    assert actual.shape[-1] == encoding.FRAME_DIM == 22


def test_legacy_targets_and_new_decode_differ_by_declared_envelope():
    """旧1.5/100与新的0.15/40是两套语义，不能混用。"""
    angular, gravity, q, dq, previous_legs, previous_wheels, reference = _inputs()
    raw = torch.zeros((3, 6))
    raw[:, 0] = 0.5
    raw[:, 2] = 0.5
    old = legacy.targets(raw, reference)
    center = torch.zeros((3, 4))
    new = encoding.decode_targets(raw, center)
    assert torch.allclose(old["leg_executed"][:, 0], torch.full((3,), 0.75))
    assert torch.allclose(new["requested_target"][:, 0], torch.full((3,), 0.075))
    assert torch.allclose(new["requested_target"][:, 2], torch.full((3,), 20.0))


@pytest.mark.parametrize("height", [0.30, 0.34, 0.40])
def test_kinematic_center_matches_frozen_geometry(height):
    declaration = {
        "action_version": GEOMETRIC_VERSION,
        "input_version": "multiskill_mcu_v2",
        "leg_halfwidth_rad": 0.15,
        "wheel_halfwidth_rad_s": 40.0,
        "height_command_range_m": [0.30, 0.40],
        "center_formula": GEOMETRY,
        "support_target_offset_rad": [0.0, 0.0, 0.0, 0.0],
    }
    frozen = KinematicMotionOutput(declaration).centers(torch.tensor([height], dtype=torch.float64))
    ours = encoding.kinematic_center(
        torch.tensor([height], dtype=torch.float64),
        hip_above_root_m=GEOMETRY["hip_above_root_m"],
        wheel_radius_m=GEOMETRY["wheel_radius_m"],
        equivalent_upper_m=GEOMETRY["equivalent_upper_m"],
        equivalent_lower_m=GEOMETRY["equivalent_lower_m"],
        hip_zero_deg=GEOMETRY["hip_zero_deg"],
        active_zero_deg=GEOMETRY["active_zero_deg"],
    )
    assert torch.allclose(ours, frozen, atol=1e-12, rtol=0)


def test_support_offset_center_matches_frozen_supported_output():
    declaration = {
        "action_version": SUPPORTED_VERSION,
        "input_version": "multiskill_mcu_v2",
        "leg_halfwidth_rad": 0.15,
        "wheel_halfwidth_rad_s": 40.0,
        "height_command_range_m": [0.30, 0.40],
        "center_formula": GEOMETRY,
        "support_target_offset_rad": list(OFFSET),
    }
    height = torch.tensor([0.34], dtype=torch.float64)
    frozen = SupportedKinematicMotionOutput(declaration).centers(height)
    ours = encoding.kinematic_center(
        height,
        hip_above_root_m=GEOMETRY["hip_above_root_m"],
        wheel_radius_m=GEOMETRY["wheel_radius_m"],
        equivalent_upper_m=GEOMETRY["equivalent_upper_m"],
        equivalent_lower_m=GEOMETRY["equivalent_lower_m"],
        hip_zero_deg=GEOMETRY["hip_zero_deg"],
        active_zero_deg=GEOMETRY["active_zero_deg"],
        support_offset_rad=OFFSET,
    )
    assert torch.allclose(ours, frozen, atol=1e-12, rtol=0)


def test_decode_matches_frozen_centered_output():
    height = torch.tensor([0.34, 0.34])
    center = torch.tensor([[0.1, -0.2, 0.1, -0.2], [0.05, -0.1, 0.05, -0.1]])
    frozen_bound = HeightEpisodeOutput(height, center)
    generator = torch.Generator().manual_seed(11)
    raw = torch.rand((2, 6), generator=generator) * 3 - 1.5
    limits = torch.tensor(
        [[[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]] * 2
    )
    frozen = frozen_bound.decode(raw, limits)
    ours = encoding.decode_targets(raw, center, leg_limits_rad=limits)
    for key in ("requested_target", "protected_target", "clipped_action"):
        assert torch.equal(ours[key], frozen[key]), key
    assert torch.equal(ours["mechanical_clipped"], frozen["mechanical_clipped"])


def test_previous_from_adopted_matches_frozen_semantics():
    adopted = torch.tensor([[0.3, -0.2, 12.0, 0.1, -0.4, -7.0]])
    frozen = HeightEpisodeOutput.previous_from_adopted(adopted)
    ours = encoding.previous_from_adopted(adopted)
    assert torch.equal(ours, frozen)
    # 输出布局是四腿在前、两轮在后。
    assert torch.allclose(ours[0, 2], torch.tensor(0.1 / 1.5))
    assert torch.allclose(ours[0, 4], torch.tensor(0.12))


def test_normalized_std_matches_frozen():
    frozen = HeightEpisodeOutput.normalized_std(0.02, 2.0)
    ours = encoding.normalized_std(0.02, 2.0)
    assert torch.allclose(ours, frozen)


def test_task_vector_layout():
    command = torch.tensor([[0.5, 0.0, 0.34]])
    task = encoding.encode_task(command, skill="Motion", jump="NONE")
    assert task.shape[-1] == 9
    assert task[0, 3] == 1 and task[0, 6] == 1
    assert torch.allclose(task[0, 0], torch.tensor(0.125))


def test_center_rejects_out_of_range_height():
    with pytest.raises(ValueError, match="超出"):
        encoding.kinematic_center(
            torch.tensor([0.5]),
            hip_above_root_m=GEOMETRY["hip_above_root_m"],
            wheel_radius_m=GEOMETRY["wheel_radius_m"],
            equivalent_upper_m=GEOMETRY["equivalent_upper_m"],
            equivalent_lower_m=GEOMETRY["equivalent_lower_m"],
            hip_zero_deg=GEOMETRY["hip_zero_deg"],
            active_zero_deg=GEOMETRY["active_zero_deg"],
        )
