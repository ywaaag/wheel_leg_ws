"""物理尺度、有效限矩与Actor更新回退的CPU行为回归。"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from native_stand.experiments.config import load_job_config
from native_stand.learning.ppo.guarded_step import bounded_actor_step
from native_stand.tasks.motion.scale_corrections import (
    apply_runtime,
    boundary_loss,
    clip_probability,
    configuration,
    saturation,
    tracking_terms,
    wheel_target_from_trace,
)

FROZEN_TEST_SCOPE = 'PORTABLE'
ROOT = Path(__file__).resolve().parents[1]


def test_bridge_field_never_uses_new_raw_scale():
    physical = torch.tensor([[0., 0., 20., 0., 0., -20.]])
    bridge = physical / torch.tensor([1.5, 1.5, 100., 1.5, 1.5, 100.])
    raw = torch.tensor([[0., 0., .5, 0., 0., -.5]])
    expected = physical[:, [2, 5]]
    for value, encoding in ((physical, 'physical_target'), (bridge, 'legacy_bridge_1p5_100'), (raw, 'supported_raw_015_40')):
        torch.testing.assert_close(wheel_target_from_trace(value, encoding=encoding), expected)
    with pytest.raises(ValueError):
        wheel_target_from_trace(raw, encoding='inferred_from_shape')


def test_saturation_consumes_effective_caps_and_motor_layer():
    caps = torch.tensor([[40., 40., 2.44694786, 40., 40., 2.44694786],
                         [36., 36., 2.0, 36., 36., 2.0]])
    motor = torch.zeros_like(caps)
    motor[:, [2, 5]] = caps[:, [2, 5]]
    request = motor * 1.2
    measured = saturation(request, motor, motor, caps)
    assert measured['request_over_cap'][:, [2, 5]].all()
    assert measured['motor_at_cap'][:, [2, 5]].all()
    assert not measured['motor_at_cap'][:, [0, 1, 3, 4]].any()
    torch.testing.assert_close(measured['motor_capacity_fraction'][:, [2, 5]], torch.ones(2, 2))
    assert not (motor.abs() >= 5.5).any()


@pytest.mark.parametrize('yaw', [False, True])
def test_low_speed_wide_narrow_and_tail_have_signed_signal(yaw):
    error = torch.tensor([.05, .1, .3, .5, 1., 2., 4.], requires_grad=True)
    a, b = tracking_terms(error, yaw=yaw)
    derivative = torch.autograd.grad((a+b).sum(), error)[0]
    assert (derivative < 0).all()
    pos = tracking_terms(-error.detach(), yaw=yaw)
    torch.testing.assert_close(pos[0], a.detach())
    torch.testing.assert_close(pos[1], b.detach())


def test_clip_tail_is_differentiable_and_does_not_penalize_interior():
    mean = torch.tensor([[0., .2, 0., .95, -.95, 0.]], requires_grad=True)
    std = torch.full_like(mean, .133333333)
    loss, probability = boundary_loss(mean, std)
    loss.backward()
    assert mean.grad[0, 3] > 0 and mean.grad[0, 4] < 0
    assert mean.grad[0, 0] == 0 and mean.grad[0, 1] == 0
    assert probability[0, 2] > .1
    inside, _ = boundary_loss(torch.zeros_like(mean), std)
    assert inside == 0
    torch.testing.assert_close(clip_probability(mean.detach(), std), clip_probability(-mean.detach(), std))


def make_optimizer():
    actor = torch.nn.Linear(1, 1, bias=False)
    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        actor.weight.zero_()
        critic.weight.zero_()
    optimizer = torch.optim.Adam([dict(params=actor.parameters(), lr=1., owner='Actor'),
                                  dict(params=critic.parameters(), lr=1., owner='Critic')])
    actor.weight.grad = -torch.ones_like(actor.weight)
    critic.weight.grad = -torch.ones_like(critic.weight)
    return actor, critic, optimizer


def test_rejected_actor_restores_weights_and_adam_critic_advances_once():
    actor, critic, optimizer = make_optimizer()
    result = bounded_actor_step(optimizer, actor, lambda: actor.weight.square().sum(), limit=1e-6, max_backtracks=2)
    assert not result['accepted'] and result['attempts'] == 3
    assert torch.equal(actor.weight, torch.zeros_like(actor.weight))
    assert actor.weight not in optimizer.state
    assert optimizer.state[critic.weight]['step'] == 1
    assert critic.weight > .9
    assert optimizer.param_groups[0]['lr'] == 1.


def test_backtracking_accepts_small_step_without_extra_adam_moments():
    actor, critic, optimizer = make_optimizer()
    result = bounded_actor_step(optimizer, actor, lambda: actor.weight.square().sum(), limit=.01, max_backtracks=4)
    assert result['accepted'] and result['backtracks'] == 4
    assert result['post_kl'] <= .01
    assert optimizer.state[actor.weight]['step'] == 1
    assert optimizer.state[critic.weight]['step'] == 1
    assert optimizer.param_groups[0]['lr'] == 1.


def test_reward_replacement_preserves_dt_events_and_nominal_zero():
    fields = ('effort', 'transfer_effort', 'wheel_torque_change', 'wheel_request_excess',
              'linear_tracking', 'linear_error', 'yaw_tracking', 'yaw_error')
    density = torch.ones(2, len(fields))
    result = dict(fields=fields, raw=density.clone(), density=density.clone(), dt=density*.01,
                  total=(density*.01).sum(-1)+torch.tensor([2., -1.]), eligible=torch.tensor([True, False]))
    caps = torch.tensor([[40., 40., 2., 40., 40., 2.]]).expand(2, -1)
    motor = torch.zeros(2, 6)
    motor[:, 2] = 2.
    base = SimpleNamespace(motor_actual=motor, driver=motor, requested=torch.ones(2, 6), adopted=torch.ones(2, 6))
    timing = SimpleNamespace(base=base, dr=SimpleNamespace(caps=caps), config={'smooth_mix': 0.})
    env = SimpleNamespace(student_reward_result=result, motion_timing=timing,
        target_xy=torch.tensor([[.25, 0.], [0., 0.]]), motion_action_command=torch.zeros(2, 3),
        recovery_snapshot=torch.zeros(2, 39), effort_pre=motor)
    event = result['total']-result['dt'].sum(-1)
    apply_runtime(env, torch.zeros(2, 6), torch.ones(2), configuration())
    torch.testing.assert_close(result['total']-result['dt'].sum(-1), event)
    torch.testing.assert_close(result['dt'], result['density']*.01)
    assert env.recovery_snapshot[:, 17].eq(1).all()
    assert result['density'][0, 0] == pytest.approx(-.02/6)
    with pytest.raises(ValueError, match='重复'):
        apply_runtime(env, torch.zeros(2, 6), torch.ones(2), configuration())


def test_resume_budget_has_no_repeated_warmup_and_scope_is_explicit():
    job = load_job_config(ROOT/'configs/jobs/motion442_scale_corrected_resume.json')
    assert job.ppo.critic_warmup_updates == job.limits.smoke_updates == 0
    assert job.assets['historical_online_updates'] + job.limits.formal_updates == 10000
    assert job.observation.leg_input_scale_rad == 1.5 and job.action.leg_scale_rad == .15
    assert job.observation.wheel_input_scale_rad_s == 100 and job.action.wheel_scale_rad_s == 40
    from native_stand.simulation.isaac.motion_backend import validate_job
    from native_stand.tasks.motion.course_strategy import LEGACY_VERSION
    assert job.curriculum.version == LEGACY_VERSION
    assert validate_job(job)
    with pytest.raises(ValueError, match='未知Motion课程版本'):
        validate_job(replace(job, curriculum=replace(job.curriculum, version='unregistered')))
    with pytest.raises(ValueError, match='超过'):
        validate_job(replace(job, limits=replace(job.limits, formal_updates=9547)))
