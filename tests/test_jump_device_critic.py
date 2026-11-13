"""Actual C/V4 histories packed in batch, without physics or CUDA."""

from types import SimpleNamespace

import numpy as np
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
