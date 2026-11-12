"""Real primitive support math; no physics engine or GPU execution."""

from types import SimpleNamespace as NS

import pytest
import torch

from native_stand.simulation.terrain_geometry import vertical_support_points
from native_stand.students.motion_runtime_reduction import GeometryGaps

FROZEN_TEST_SCOPE = "PORTABLE"


def rotate(q, value):
    # SDK3 XYZW convention; injected math does not replace the support algorithm.
    v = q[..., :3]
    twice = 2 * torch.cross(v, value, dim=-1)
    return value + q[..., 3:] * twice + torch.cross(v, twice, dim=-1)


def inverse(q, value):
    return rotate(torch.cat((-q[..., :3], q[..., 3:]), -1), value)


def test_all_primitive_support_z_matches_existing_plane_guard_with_transforms():
    rng = torch.Generator().manual_seed(441)
    geometry = []
    for i, kind in enumerate(("Cube", "Sphere", "Capsule", "Cylinder")):
        q = torch.randn(4, generator=rng, dtype=torch.float64)
        q /= q.norm()
        r = rotate(q.expand(3, 4), torch.eye(3, dtype=torch.float64)).T
        matrix = torch.eye(4, dtype=torch.float64)
        matrix[:3, :3] = r
        matrix[:3, 3] = torch.randn(3, generator=rng, dtype=torch.float64)
        geometry.append(
            (
                dict(body="robot", type=kind, size=0.6, radius=0.12, height=0.42, axis="Y"),
                i,
                matrix,
            )
        )
    helper = GeometryGaps(geometry, {}, "cpu", torch.float64)
    positions = torch.randn(5, 4, 3, generator=rng, dtype=torch.float64)
    q = torch.randn(5, 4, 4, generator=rng, dtype=torch.float64)
    q /= q.norm(dim=-1, keepdim=True)
    points = vertical_support_points(helper, positions, q, inverse, rotate)
    gaps, _ = helper.measure(positions, q, inverse, 0.0)
    torch.testing.assert_close(points[..., 2], gaps, rtol=1e-12, atol=1e-12)


def test_translated_capsule_queries_terrain_at_support_xy_not_body_origin():
    from native_stand.simulation.terrain_surface import TriangleRoads

    helper = GeometryGaps(
        [
            (
                dict(body="robot", type="Capsule", radius=0.02, height=0.42, axis="Y"),
                0,
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.1],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=torch.float64,
                ),
            )
        ],
        {},
        "cpu",
        torch.float64,
    )
    angle = torch.tensor(0.4, dtype=torch.float64)
    q = torch.tensor([[[torch.sin(angle / 2), 0.0, 0.0, torch.cos(angle / 2)]]])
    p = torch.tensor([[[0.0, 0.0, 0.1]]], dtype=torch.float64)
    point = vertical_support_points(helper, p, q, inverse, rotate)
    # Mesh height = 0.08 + 0.5*worldY; worldY = assetX-spawnX.
    x = torch.arange(31, dtype=torch.float64) * 0.1
    h = (0.08 + 0.5 * (x - 1.0)).expand(31, 31)
    surface = TriangleRoads(
        torch.stack((h * 0, h)),
        torch.zeros(1, 3, dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        torch.tensor([1]),
        cell=0.1,
        length=3.0,
        width=3.0,
        repetitions=1,
        spawn_x=1.0,
    )
    wrong_ground, _ = surface.sample(p[..., :2])
    correct_ground, _ = surface.sample(point[..., :2])
    old_gap = point[..., 2] - wrong_ground
    correct_gap = point[..., 2] - correct_ground
    assert old_gap.item() < -0.005 and correct_gap.item() > 0.005
    assert point[..., 1].item() != pytest.approx(p[..., 1].item())


def test_runtime_failure_report_keeps_infinite_limits_as_explicit_evidence(tmp_path):
    from native_stand.artifacts.io import read_json
    from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter

    adapter = MotionProductionAdapter()
    snapshot = torch.zeros(1, 32)
    snapshot[0, 4] = 1
    adapter.context = NS(run_dir=tmp_path, updates_used=3)
    adapter.env = NS(
        recovery_snapshot=snapshot,
        motion_timing=NS(tick=2210),
        student_fault_joint_data={"limits": torch.tensor([[[-torch.inf, torch.inf]]])},
        cfg=NS(closure_fault_m=0.005),
        recovery_config={"initialization": {"severe_penetration_m": 0.005}},
    )
    adapter.record_runtime_fault()
    saved = read_json(tmp_path / "runtime_fault_detail.json")
    assert saved["pre_reset_joints"]["limits"] == [[["-inf", "inf"]]]
    assert saved["severe_penetration_m"] == 0.005
