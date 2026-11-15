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
    import mujoco

    class FullForwardInverse(InverseFeedback):
        def forward_feedback(self):
            mujoco.mj_forward(self.model, self.data)

    class ReferenceGeometry(ClosedChain):
        def leg_lengths(self, q):
            return self.measure(q).length

    fast, reference = InverseFeedback(*inverse_args), FullForwardInverse(*inverse_args)
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
        from native_stand.simulation.jump_support_model import estimate

        cached = estimate(fast, workspace=fast.support_workspace)
        fresh = estimate(fast)
        for key in ("internal_tau", "driver_tau", "support_J"):
            np.testing.assert_array_equal(cached[key], fresh[key])


def test_device_ancestor_chain_lengths_match_original_geometry_on_cpu(inverse_args):
    import torch

    from native_stand.simulation.jump_device_inverse import DeviceInverse

    ref = InverseFeedback(*inverse_args)
    ref.geometry.leg_lengths(ref.initial.q)
    device = DeviceInverse.__new__(DeviceInverse)
    device.n, device.device = 64, "cpu"
    device.hips = torch.as_tensor(np.asarray(ref.geometry._leg_hips))
    device.leg_chain = [
        (j, ref.joint_names.index(j["name"]), torch.as_tensor(a), torch.as_tensor(b))
        for j, a, b in ref.geometry._leg_chain
    ]
    rng = np.random.default_rng(116)
    joint = torch.as_tensor(rng.uniform(-3, 3, (64, 3, len(ref.joint_names))))
    lengths = device.leg_lengths(joint).numpy()
    expected = np.stack(
        [
            ref.geometry.leg_lengths(dict(zip(ref.joint_names, row[0].numpy(), strict=True)))
            for row in joint
        ]
    )
    np.testing.assert_allclose(lengths, expected, atol=2e-15, rtol=0)


def test_device_mass_legacy_dense_and_symmetric_CSR_preserve_full_double_product():
    import torch

    from native_stand.simulation.jump_device_inverse import dense_mass

    rows = torch.tensor([0, 1, 1, 2, 2])
    columns = torch.tensor([0, 0, 1, 0, 2])
    packed = torch.tensor([[4, -1, 5, 2, 6], [7, 3, 8, -2, 9]], dtype=torch.float32)
    expected = torch.tensor(
        [[[4, -1, 2], [-1, 5, 0], [2, 0, 6]], [[7, 3, -2], [3, 8, 0], [-2, 0, 9]]],
        dtype=torch.float64,
    )
    actual = dense_mass(packed, 3, rows=rows, columns=columns)
    assert torch.equal(actual, expected)
    legacy = torch.zeros(2, 4, 4)
    legacy[:, :3, :3] = expected
    assert torch.equal(dense_mass(legacy, 3), expected)
    acceleration = torch.tensor([1.123456789, -2.345678912, 3.456789123], dtype=torch.float64)
    assert torch.equal(actual @ acceleration, expected @ acceleration)
    with pytest.raises(ValueError, match="CSR"):
        dense_mass(packed[:, :-1], 3, rows=rows, columns=columns)
