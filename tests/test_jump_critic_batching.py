"""Large Critic packing preserves columns, numeric values and rejection gates."""

import json
from pathlib import Path

import numpy as np
import pytest

from native_stand.tasks.jump.controller_critic import Field, NormalizedCritic, PackedCritic

FROZEN_TEST_SCOPE = "PORTABLE"


def encoders():
    path = Path(__file__).parents[1] / "configs/contracts/jump_isaac_full_execution_critic_v1.json"
    fields = [
        Field(x["name"], tuple(x["shape"]), np.asarray(x["scale"]), x["unit"])
        for x in json.loads(path.read_text())["fields"]
    ]
    rng = np.random.default_rng(41)
    values = {f.name: rng.normal(size=f.shape) for f in fields}
    return NormalizedCritic(fields), PackedCritic(fields), values


def test_full_5591_columns_are_bitwise_equal_and_outputs_are_owned():
    old, packed, values = encoders()
    assert old.contract() == packed.contract()
    for multiplier in (0.0, 1.0, -3.5, 1e6):
        current = {k: v * multiplier for k, v in values.items()}
        actual = packed.encode(current)
        np.testing.assert_array_equal(actual, old.encode(current))
        assert actual.shape == (5591,) and actual.dtype == np.float32
        actual[:] = 123
        np.testing.assert_array_equal(packed.encode(current), old.encode(current))


@pytest.mark.parametrize("bad", ["missing", "extra", "shape", "nan", "inf"])
def test_packed_critic_retains_dynamic_field_shape_and_finite_checks(bad):
    old, packed, values = encoders()
    name = next(k for k, v in values.items() if v.size > 1)
    if bad == "missing":
        del values[name]
    elif bad == "extra":
        values["unknown"] = np.zeros(1)
    elif bad == "shape":
        values[name] = np.zeros((123,))
    else:
        values[name].flat[0] = float(bad)
    for encoder in (old, packed):
        with pytest.raises(ValueError, match="missing/extra|shape/finite"):
            encoder.encode(values)
