"""Actual hull support guards differ from the analytic task cylinder."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from native_stand.artifacts.io import json_digest
from native_stand.simulation.jump_collision_readback import (
    convex_clearance,
    convex_mesh_readback,
    cooked_cylinder_envelope,
    physics_execution_readback,
)
from native_stand.simulation.jump_isaac import IsaacJump, execution_readback
from native_stand.simulation.jump_isaac_scene import (
    validate_physical_measurement,
    wheel_convex_geometry,
)

FROZEN_TEST_SCOPE = "PORTABLE"


def test_receipt_uses_actual_physics_scene_and_manager_instead_of_requested_device():
    values = {"physxScene:enableGPUDynamics": False, "physxScene:broadphaseType": "MBP"}
    prim = SimpleNamespace(
        GetAttribute=lambda name: SimpleNamespace(Get=lambda: values[name]),
        GetPath=lambda: "/physicsScene",
    )
    stage = SimpleNamespace(GetPrimAtPath=lambda path: prim if path == "/physicsScene" else None)
    sim = SimpleNamespace(
        stage=stage,
        cfg=SimpleNamespace(physics_prim_path="/physicsScene", device="cuda:0"),
        device="cpu",
        backend="warp",
    )
    actual = physics_execution_readback(sim)
    assert actual["actual_device"] == "cpu"
    assert actual["scene_gpu_dynamics"] is False
    assert actual["scene_broadphase"] == "MBP"
    assert "device=cpu" in actual["backend_label"]
    assert actual["narrowphase_device"].startswith("UNVERIFIED")
    sim.cfg.physics_prim_path = "/missing"
    with pytest.raises(RuntimeError, match="Actual physics scene unavailable"):
        physics_execution_readback(sim)


def inverse_rotate(quaternions, vectors):
    values = (
        Rotation.from_quat(quaternions.numpy().reshape(-1, 4))
        .inv()
        .apply(vectors.numpy().reshape(-1, 3))
    )
    return torch.as_tensor(values.reshape(vectors.shape), dtype=vectors.dtype)


def test_original_900um_gap_passes_but_actual_convex_gap_rejects_without_threshold_change():
    vertices, _ = wheel_convex_geometry(0.06, 0.0165, 0)
    positions = torch.tensor([[[0.0, 0.0, 0.0591]] * 2], dtype=torch.float64)
    quaternions = torch.tensor([[[0.0, 0.0, 0.0, 1.0]] * 2], dtype=torch.float64)
    gaps = convex_clearance(positions, quaternions, [torch.as_tensor(vertices)] * 2, inverse_rotate)
    packet = dict(clearance=np.full((1, 2), -0.0009), closure=np.zeros((1, 4)))
    validate_physical_measurement(packet)
    assert gaps[0, 0].item() == pytest.approx(-0.001230496773810994, abs=1e-14)
    packet["collision_clearance"] = gaps.numpy()
    with pytest.raises(RuntimeError, match="wheel penetration"):
        validate_physical_measurement(packet)


def test_cooked_support_matches_world_vertex_min_for_tilted_offset_link_shapes():
    rng = np.random.default_rng(144)
    q = Rotation.random(24, random_state=rng).as_quat().reshape(12, 2, 4)
    positions = rng.normal(size=(12, 2, 3))
    vertices = []
    expected = np.zeros((12, 2))
    for wheel in range(2):
        p = rng.normal(size=(17 + wheel, 3)) * 0.06 + [0.012, -0.004, 0.006]
        vertices.append(torch.as_tensor(p))
        for row in range(12):
            world = Rotation.from_quat(q[row, wheel]).apply(p) + positions[row, wheel]
            expected[row, wheel] = world[:, 2].min()
    actual = convex_clearance(
        torch.as_tensor(positions), torch.as_tensor(q), vertices, inverse_rotate
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)


def cooked_mesh():
    points, faces = wheel_convex_geometry(0.06, 0.0165, 2)
    polygons, indices = [], []
    for face in faces:
        polygons.append(
            SimpleNamespace(index_base=len(indices), num_vertices=len(face), plane=[0, 0, 1, 0])
        )
        indices.extend(face)
    return SimpleNamespace(vertices=points, indices=indices, polygons=polygons)


def test_public_cooked_mesh_faces_are_real_data_and_count_limits_are_not_GPU_certificate():
    mesh = cooked_mesh()
    actual = convex_mesh_readback(mesh)
    assert actual["counts"] == dict(vertices=60, polygons=32, maximum_polygon_vertices=30)
    assert actual["within_documented_GPU_count_limits"]
    assert actual["polygons"][0]["num_vertices"] == 30
    assert isinstance(actual["polygons"][0]["plane"], list)
    assert actual["buildGPUData"] == "NOT_EXPOSED_BY_PUBLIC_REPRESENTATION_API"
    assert actual["narrowphase_device"] == "UNVERIFIED"
    assert json_digest(convex_mesh_readback(mesh)) == json_digest(actual)
    mesh.vertices[0, 0] += 0.0001
    assert convex_mesh_readback(mesh)["geometry_sha256"] != actual["geometry_sha256"]


@pytest.mark.parametrize("bad", ["vertices", "index", "face"])
def test_invalid_actual_cooked_geometry_cannot_fall_back_to_authored_mesh(bad):
    mesh = cooked_mesh()
    if bad == "vertices":
        mesh.vertices[0, 0] = np.nan
    elif bad == "index":
        mesh.indices[0] = 1000
    else:
        mesh.polygons[0].index_base = len(mesh.indices)
    with pytest.raises(ValueError):
        convex_mesh_readback(mesh)


def test_cooked_envelope_uses_actual_vertices_and_planes_and_detects_inward_simplification():
    points, _ = wheel_convex_geometry(0.06, 0.0165, 2)
    from scipy.spatial import ConvexHull

    planes = ConvexHull(points).equations
    mesh = dict(vertices=points, polygons=[dict(plane=p) for p in planes])
    envelope = cooked_cylinder_envelope(mesh, 0.06, 0.0165, 2)
    assert envelope["maximum_radial_excess_m"] == pytest.approx(0.000330496773810994)
    assert envelope["maximum_axial_excess_m"] == 0
    assert abs(envelope["maximum_cylinder_outside_hull_plane_m"]) < 1e-15
    # A simplifier may stay inside the radial bound while cutting into the cylinder.
    for polygon in mesh["polygons"]:
        polygon["plane"][3] += 0.0002
    assert (
        cooked_cylinder_envelope(mesh, 0.06, 0.0165, 2)["maximum_cylinder_outside_hull_plane_m"]
        > 0.00019
    )


def test_device_physical_guard_rejects_actual_hull_before_rollout_can_continue():
    env = IsaacJump.__new__(IsaacJump)
    env.num_envs = 1
    values = dict(
        clearance=torch.full((1, 2), -0.0009),
        collision_clearance=torch.full((1, 2), -0.0012304968),
        closure=torch.zeros(1, 4),
    )
    env.scene = SimpleNamespace(measured=lambda: values)
    env.motor = env.NET = env.passive = torch.zeros(1, 6)

    def host():
        validate_physical_measurement({k: v.numpy() for k, v in values.items()})

    env.host = host
    with pytest.raises(RuntimeError, match="wheel penetration"):
        env.validate_device_measurement()


def test_execution_receipt_tracks_constructed_firmware_and_inverse_not_config_device():
    from native_stand.simulation.jump_cuda_execution import CudaExecution
    from native_stand.simulation.jump_isaac_bridge import InverseFeedback
    from native_stand.simulation.jump_unified import UnifiedV4Execution

    cpu = execution_readback(
        UnifiedV4Execution.__new__(UnifiedV4Execution), InverseFeedback.__new__(InverseFeedback)
    )
    execution = CudaExecution.__new__(CudaExecution)
    execution.gpu = object()

    # A receipt-only fixture avoids importing the optional Warp runtime in portable CI.
    class DeviceInverse:
        version = "jump_PhysX_Warp_inverse_float32_v1"

    gpu = execution_readback(execution, DeviceInverse())
    assert cpu["C_execution"] == "batched CPU C bridge"
    assert "CPU inverse" in cpu["support"]
    assert gpu["C_execution"] == "CUDA firmware C consumer"
    assert "Warp device inverse" in gpu["support"]
    assert gpu["inverse_implementation"].endswith("DeviceInverse")
    assert "independent" in gpu["contact_generation_device"]
    mixed = execution_readback(execution, InverseFeedback.__new__(InverseFeedback))
    assert mixed["C_execution"] == "CUDA firmware C consumer"
    assert "CPU inverse" in mixed["support"]
