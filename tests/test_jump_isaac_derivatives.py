"""Physical-time acceleration survives other rows' nonphysical reset epochs."""

import pytest
import torch

from native_stand.simulation.jump_isaac_scene import TickDerivatives

FROZEN_TEST_SCOPE = "PORTABLE"


def test_partial_reset_retains_other_rows_root_and_joint_acceleration():
    d = TickDerivatives(0.001)
    initial = torch.zeros(3, 44)
    d.sample(128, initial)
    velocity = torch.arange(132, dtype=torch.float32).reshape(3, 44) / 1000
    expected = d.sample(129, velocity).clone()
    reset_velocity = velocity.clone()
    reset_velocity[1] = -0.125
    d.reset_rows(torch.tensor([1]), 129, reset_velocity)
    result = d.sample(129, reset_velocity)
    assert torch.equal(result[[0, 2]], expected[[0, 2]])
    assert torch.count_nonzero(result[1]) == 0
    # A same-tick readback must not consume a second 1ms of physical time.
    assert d.sample(129, reset_velocity) is result
    assert torch.equal(d.sample(130, reset_velocity), torch.zeros_like(result))


def test_reset_velocity_is_baseline_not_an_acceleration_impulse():
    d = TickDerivatives(0.001)
    velocity = torch.full((3, 44), 2.0)
    d.sample(0, velocity)
    velocity[0] = -50
    d.reset_rows(torch.tensor([0]), 0, velocity)
    velocity[0] += 0.001
    actual = d.sample(1, velocity)
    assert torch.allclose(actual[0], torch.ones(44), atol=0.001)
    assert torch.count_nonzero(actual[1:]) == 0


def test_multiple_reset_epochs_do_not_advance_other_rows_physical_history():
    d = TickDerivatives(0.001)
    velocity = torch.zeros(3, 44, dtype=torch.float64)
    d.sample(7, velocity)
    velocity += 0.001
    expected = d.sample(8, velocity).clone()
    for row in (0, 2, 0):
        velocity[row] = -row - 1
        d.reset_rows(torch.tensor([row]), 8, velocity)
    assert torch.equal(d.sample(8, velocity)[1], expected[1])
    velocity += 0.003
    assert torch.allclose(d.sample(9, velocity), torch.full_like(velocity, 3.0))


def test_sparse_readback_uses_actual_tick_duration_and_rejects_clock_reversal():
    d = TickDerivatives(0.001)
    v = torch.zeros(2, 44, dtype=torch.float64)
    d.sample(100, v)
    actual = d.sample(105, v + 0.025)
    assert torch.equal(actual, torch.full_like(v, 5))
    with pytest.raises(ValueError, match="backwards"):
        d.sample(104, v)


def test_full_reset_starts_new_velocity_baseline_without_moving_physics_tick():
    d = TickDerivatives(0.001)
    velocity = torch.zeros(3, 44)
    d.sample(50, velocity)
    d.sample(51, velocity + 1)
    d.reset_rows(torch.arange(3), 51, velocity)
    assert d.tick == 51
    assert torch.count_nonzero(d.sample(52, velocity)) == 0


def test_scene_partial_reset_wires_only_selected_physical_history_rows():
    from types import SimpleNamespace

    from native_stand.simulation.jump_isaac_scene import IsaacScene

    scene = IsaacScene.__new__(IsaacScene)
    scene.device = "cpu"
    scene.dt = 0.001
    scene.reset_epoch = 0
    scene.actual_ticks = 129
    scene.all_ids = list(range(38))
    scene.derivatives = TickDerivatives(scene.dt)
    velocity = torch.zeros(2, 44)
    scene.derivatives.sample(128, velocity)
    velocity[0] = 0.001
    velocity[1] = 0.002
    scene.derivatives.sample(129, velocity)
    data = SimpleNamespace(
        default_joint_pos=torch.zeros(2, 38),
        default_root_state=torch.zeros(2, 13),
        root_link_lin_vel_w=velocity[:, :3].clone(),
        root_link_ang_vel_w=velocity[:, 3:6].clone(),
        joint_vel=velocity[:, 6:].clone(),
    )

    def root_velocity(*, root_velocity, env_ids):
        data.root_link_lin_vel_w[env_ids.long()] = root_velocity[:, :3]
        data.root_link_ang_vel_w[env_ids.long()] = root_velocity[:, 3:]

    def joint_state(*, position, velocity, env_ids):
        data.joint_vel[env_ids.long()] = velocity

    scene.robot = SimpleNamespace(
        data=data,
        joint_names=[f"joint_{i}" for i in range(38)],
        write_root_link_pose_to_sim_index=lambda **kw: None,
        write_root_com_velocity_to_sim_index=root_velocity,
        write_joint_state_to_sim_index=joint_state,
    )
    scene.scene = SimpleNamespace(
        env_origins=torch.zeros(2, 3),
        reset=lambda ids: None,
        write_data_to_sim=lambda: None,
        update=lambda dt: None,
    )
    scene.sim = SimpleNamespace(forward=lambda: None)
    scene.reset([0], [{"drop_m": 0.0}], SimpleNamespace(q={}, root_support_height_m=0.34))
    assert scene.reset_epoch == 1 and scene.actual_ticks == 129
    actual = scene.derivatives.acceleration
    assert torch.count_nonzero(actual[0]) == 0
    assert torch.equal(actual[1], torch.full((44,), 2.0))
