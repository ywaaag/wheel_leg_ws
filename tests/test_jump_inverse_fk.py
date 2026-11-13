"""Actual authored FK and byte-exact inverse feedback, without integration."""

import os
from pathlib import Path

import numpy as np
import pytest
from fixtures.jump_cpu_io import mapped_job
from test_jump_cpu_reuse import no_physics_or_cuda  # noqa: F401

from native_stand.artifacts.io import read_json
from native_stand.simulation.jump_isaac_bridge import InverseFeedback
from native_stand.simulation.jump_shared_c import State
from native_stand.tasks.jump.geometry import ClosedChain

FROZEN_TEST_SCOPE = "HOST"


@pytest.fixture
def inverse_args():
    root = Path(__file__).parents[1]
    data = Path(os.environ.get("WHEEL_LEG_DATA_ROOT", "/home/kellen/wheel_leg_training_data"))
    refs = mapped_job(root, data).assets["jump_bindings"]
    return (
        read_json(refs["description"]["path"]),
        read_json(refs["spring_geometry"]["path"])["springs"],
        read_json(refs["mechanics"]["path"])["mechanics"],
        read_json(refs["V4"]["path"]),
    )


def test_minimal_FK_matches_authored_full_geometry_for_broad_joint_angles(inverse_args):
    inverse = InverseFeedback(*inverse_args)
    geometry = inverse.geometry
    rng = np.random.default_rng(60)
    for span in (0.1, 3.14, 10000):
        for _ in range(100):
            q = dict(
                zip(
                    inverse.joint_names,
                    rng.uniform(-span, span, len(inverse.joint_names)),
                    strict=True,
                )
            )
            np.testing.assert_array_equal(geometry.leg_lengths(q), geometry.measure(q).length)
    assert len(geometry._leg_chain) == 6
    assert len(geometry.joints) == 38


def test_inverse_C_input_is_bitwise_equal_to_full_FK_without_RNG_or_engine(inverse_args):
    class ReferenceGeometry(ClosedChain):
        def leg_lengths(self, q):
            return self.measure(q).length

    fast, reference = InverseFeedback(*inverse_args), InverseFeedback(*inverse_args)
    reference.geometry = ReferenceGeometry(*inverse_args[:2], stop_margin_m=0)
    rng = np.random.default_rng(61)
    q = np.array([fast.initial.q[n] for n in fast.joint_names])
    state = State()
    state.adopted_sequence = 123
    state.adopted[:] = [0.1, 0.2, 1, -0.1, 0.2, -1]
    for tick in range(0, 640, 5):
        joint = np.stack(
            (
                q + rng.uniform(-0.02, 0.02, len(q)),
                rng.normal(size=len(q)) * 0.1,
                rng.normal(size=len(q)),
            )
        )
        kwargs = dict(
            tick=tick,
            state=state,
            root=np.array([0, 0, 0.34, 2**-0.5, 0, 0, 2**-0.5, 0.01, -0.02, 0, 0.03, 0.01, 0]),
            joint=joint,
            acceleration=rng.normal(size=6) * 0.1,
            motor=rng.uniform(-4, 4, 6),
            NET=rng.uniform(-1, 1, 6),
            command=[0, 0, 0.34],
            request=tick == 500,
        )
        assert bytes(fast.input(**kwargs)) == bytes(reference.input(**kwargs))
