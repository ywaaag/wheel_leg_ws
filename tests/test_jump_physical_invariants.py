"""Physical failures reject host packets, without confusing ordinary task failure."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from native_stand.learning.ppo.performance import Measurement
from native_stand.simulation.jump_isaac import IsaacJump
from native_stand.simulation.jump_isaac_scene import validate_physical_measurement

FROZEN_TEST_SCOPE = "PORTABLE"


@pytest.mark.parametrize("envs", [1, 32, 128, 256, 4096, 8192])
def test_tiled_ground_covers_clone_grid_with_declared_margin(envs):
    from math import ceil, sqrt

    from native_stand.simulation.jump_isaac_scene import physical_contract

    contract = physical_contract(envs)
    half_span = (ceil(sqrt(envs)) - 1) * contract["env_spacing_m"] / 2
    assert contract["ground_size_m"][0] / 2 - half_span >= 32
    assert contract["ground_size_m"][0] % contract["maximum_tile_side_m"] == 0
    if envs <= 256:
        assert contract["ground_size_m"] == [128, 128, 0.1]
    if envs == 4096:
        assert contract["ground_size_m"] == [256, 256, 0.1]


def packet():
    return {
        "root": np.array([[0, 0, 0.17, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]),
        "clearance": np.array([[-1.2e-6, 0.0]]),
        "closure": np.full((1, 4), 4e-5),
        "normal": np.zeros((1, 2)),
        "restricted": np.array([True]),
    }


def test_explicit_contact_solver_changes_identity_without_relaxing_geometry_guard():
    from native_stand.simulation.jump_isaac_scene import physical_contract

    old = physical_contract(4096)
    candidate = physical_contract(4096, "wheel_convex30_v1")
    assert candidate["version"] != old["version"]
    assert candidate["wheel_collision"] == "circumscribed_convex30"
    assert candidate["wheel_radial_envelope_error_bound_m"] == 0.000331
    for key in ("ground_size_m", "maximum_wheel_penetration_m", "maximum_closure_residual_m"):
        assert candidate[key] == old[key]
    with pytest.raises(ValueError, match="Unsupported Jump wheel collision"):
        physical_contract(4096, "unknown")


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_authored_convex_wheel_bounds_true_cylinder_at_all_contact_directions(axis):
    from native_stand.simulation.jump_isaac_scene import wheel_convex_geometry

    points, faces = wheel_convex_geometry(0.06, 0.0165, axis)
    rng = np.random.default_rng(140)
    directions = rng.normal(size=(2000, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    axial = abs(directions[:, axis])
    exact = 0.0165 * axial + 0.06 * np.sqrt(1 - axial**2)
    support = (directions @ points.T).max(-1)
    assert (support >= exact - 1e-14).all()
    assert (support - exact).max() < 0.000331
    assert points.shape == (60, 3)
    assert min(i for face in faces for i in face) == 0
    assert max(i for face in faces for i in face) == 59


def test_untrained_falling_and_nonwheel_contact_are_task_events():
    # The free-drop traces reach .17m without broken wheel/closure geometry.
    validate_physical_measurement(packet())


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("clearance", [[-0.038, 0.0]], "wheel penetration"),
        ("closure", [[0.003, 0, 0, 0]], "closure residual"),
        ("root", [[float("nan")] * 13], "nonfinite readback"),
    ],
)
def test_formal_host_consumer_rejects_physical_failure_before_caching(
    field, value, message, tmp_path
):
    values = packet()
    values[field] = np.asarray(value)
    env = IsaacJump.__new__(IsaacJump)
    env.num_envs = 1
    env.output = tmp_path
    env.measurement = Measurement(False, "cpu")
    env.scene = SimpleNamespace(
        actual_ticks=21,
        reset_epoch=1,
        measured=lambda: {k: torch.as_tensor(v) for k, v in values.items()},
    )
    env.motor = env.NET = env.passive = torch.zeros(1, 6)
    with pytest.raises(RuntimeError, match=message):
        env.host()
    assert not hasattr(env, "_host_cache")
    from native_stand.artifacts.io import read_json

    failure = read_json(tmp_path / "physical_failure.json")
    assert failure["actual_engine_tick"] == 21
    assert message in failure["error"]
    assert failure["minimum_clearance_row"] == 0
