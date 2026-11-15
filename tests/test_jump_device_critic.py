"""Actual C/V4 histories packed in batch, without physics or CUDA."""

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_jump_batch_bridge_cpu import advance, bridge  # noqa: F401, F811
from test_jump_cpu_reuse import no_physics_or_cuda  # noqa: F401

from native_stand.learning.ppo.performance import Measurement
from native_stand.simulation.jump_isaac import IsaacJump
from native_stand.simulation.jump_isaac_critic import BatchCritic, RowCritic
from native_stand.tasks.jump.dense_reward import TaskSpec
from native_stand.tasks.jump.unified_reward import UnifiedPacket, UnifiedReward

FROZEN_TEST_SCOPE = "HOST"


def test_batch_device_fields_match_scalar_full_contract_at_ring_wrap_and_reset(bridge):  # noqa: F811
    e = bridge()
    env = IsaacJump.__new__(IsaacJump)
    env.execution, env.num_envs, env.device = e, 3, "cpu"
    env.request_after_ms = 1500
    env.measurement = Measurement(False, "cpu")
    rng = np.random.default_rng(91)
    shapes = dict(
        root=(13,),
        joint=(3, 38),
        COM=(3,),
        COMv=(3,),
        length=(2,),
        swing=(2,),
        spring=(2,),
        closure=(4,),
        normal=(2,),
        slip=(2,),
        clearance=(2,),
    )
    h = {k: rng.normal(size=(3, *shape)).astype(np.float32) for k, shape in shapes.items()}
    m = {k: torch.from_numpy(v) for k, v in h.items()}
    env.scene = SimpleNamespace(measured=lambda: m)
    env.motor, env.NET, env.passive = [torch.randn(3, 6) for _ in range(3)]
    h.update(motor=env.motor.numpy(), NET=env.NET.numpy(), passive=env.passive.numpy())
    env._root_mass_COM = rng.normal(size=(3, 4)).astype(np.float32)
    env._material_bounds = rng.normal(size=(3, 2, 3)).astype(np.float32)
    env.starts = [0, 3, 21]
    env.requests = [
        dict(kind=k, seconds=5.0 + i)
        for i, k in enumerate(("landing_only", "ground_active_jump", "landing_only"))
    ]
    initial = UnifiedPacket(
        dt_s=0,
        phase=9,
        owner=2,
        COM_height_m=0.3,
        COM_velocity_m_s=(0, 0, 0),
        root_height_m=0.34,
        tilt_rad=(0, 0),
        omega_rad_s=(0, 0, 0),
        leg_length_m=(0.28, 0.28),
        leg_swing_rad=(0, 0),
        spring_length_m=(0.02, 0.02),
        contact=(True, True),
        tire_clearance_m=(0, 0),
        wheel_normal_force_N=(50, 50),
        contact_slip_m_s=(0, 0),
        actual_motor_Nm=(0,) * 6,
        actual_dq_rad_s=(0,) * 6,
        adopted_target=(0,) * 6,
        force_valid=True,
    )
    env.rewards = [
        UnifiedReward.start(TaskSpec(r["kind"], r["seconds"], 10), initial) for r in env.requests
    ]
    env.critics = [RowCritic(38) for _ in range(3)]
    packer = None
    for tick in range(171):
        advance(e)
        if tick == 132:
            e.reset(torch.tensor([1]), torch.zeros(1, 6), torch.zeros(1, 6))
        if tick not in (0, 11, 127, 132, 139, 170):
            continue
        actor, _ = e.observe()
        if packer is None:
            packer = BatchCritic(env, actor, h)
        snapshot = env.critic_snapshot()
        expected = np.stack(
            [env.row_observation(i, h, actor[i], execution=snapshot) for i in range(3)]
        )
        actual = packer.pack(env, actor)
        np.testing.assert_array_equal(actual.numpy(), expected)
        assert actual.shape == (3, 5591)
        # Every row view must die with the temporary terminal snapshot.
        # Otherwise asynchronous terminal rows pin full-batch CPU allocations.
        storage = next(v for v in vars(snapshot.v4).values() if torch.is_tensor(v))
        reference = weakref.ref(storage._base)
        assert all(not hasattr(critic.base, "plant") for critic in env.critics)
        del storage, snapshot
        gc.collect()
        assert reference() is None
        if tick == 0:
            env.critics[0].base._pack_signature = ("invalid",)
            with pytest.raises(ValueError, match="Critic packing contract changed"):
                env.row_observation(0, h, actor[0], execution=env.critic_snapshot())
            assert not hasattr(env.critics[0].base, "plant")
            del env.critics[0].base._pack_signature


def test_device_controller_encoding_preserves_uint32_wrap_and_all_fields(bridge):  # noqa: F811
    from native_stand.simulation.jump_device_c import controller_device_fields
    from native_stand.tasks.jump.controller_critic import controller_fields

    e = bridge()
    for _ in range(10):
        advance(e)
    for k, (state, inputs) in enumerate(zip(e.c.states, e.c.inputs, strict=True)):
        inputs.now = 3 + k
        state.phase_tick = 0xFFFFFFF0 + k
        state.candidate[:] = [0, 2]
        state.candidate_tick[:] = [0xFFFFFFFE, 1]
        state.off_valid[:] = [1, 0]
        state.off_tick[:] = [0xFFFFFFFD, 2]
        state.command_sequence = 0xFFFFFFFE
        state.adopted_sequence = 2
        state.command_valid = 1
        state.command_tick = 0xFFFFFFFA
        inputs.feedback_tick[:] = [0xFFFFFFFF, 0, 1, 2, 3, 4]
    actual = controller_device_fields(e, "cpu")
    for i in range(3):
        expected, _ = controller_fields(
            e.c.states[i], e.c.inputs[i], fault_count=14, phase_count=10
        )
        expected["C.raw_output"] = expected.pop("C.mean")
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            np.testing.assert_array_equal(actual[name][i].numpy(), value, err_msg=name)
