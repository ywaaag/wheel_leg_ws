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
def test_formal_host_consumer_rejects_physical_failure_before_caching(field, value, message):
    values = packet()
    values[field] = np.asarray(value)
    env = IsaacJump.__new__(IsaacJump)
    env.num_envs = 1
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
