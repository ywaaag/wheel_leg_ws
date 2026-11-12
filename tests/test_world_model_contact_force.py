"""Contact-force supervision: synthetic engineering tests, zero physics steps.

These tests only exercise declared formulas on artificial tensors plus one small
fixture replay written under tmp_path. No robot data, no MuJoCo step, no GPU.
"""
FROZEN_TEST_SCOPE = 'PORTABLE'

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from native_stand.world_model.state import initial
from native_stand.world_model.models import (Ensemble, mechanics_targets, mechanical_features,
    contact_force_config, force_supervision, wheel_force, CONTACT_FORCE_FROZEN)
from native_stand.world_model.training import supervised_loss, fit_world
from native_stand.world_model.geometry import integrate, rotation_error
from native_stand.world_model.owner import load_world


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(n=2):
    from native_stand.students.motion_mcu_execution import profile, spring_compensation
    p = profile()
    q = torch.zeros(n, 13)
    q[:, 2] = .34
    q[:, 3:7] = q.new_tensor([2**-.5, 2**-.5, 0., 0.])
    sensors = torch.zeros(n, 18)
    sensors[:, 4] = -1
    comp, _, _, _ = spring_compensation(torch.zeros(n, 4))
    return initial(q, torch.zeros(n, 12), sensors, torch.zeros(n, 3),
                   torch.tensor([[0., 0., .34, 0.]]).expand(n, -1), compensation=comp,
                   cap=torch.tensor(p['driver_limits_nm']).expand(n, -1),
                   delay=torch.tensor(p['motor_delay_ms']).expand(n, -1),
                   tau=torch.tensor(p['motor_tau_ms'], dtype=torch.float32).expand(n, -1))


class FlatGeometry(torch.nn.Module):
    """Artificial differentiable closure geometry; no robot capability evidence."""
    active_q = list(range(7, 13))
    active_v = list(range(6, 12))

    def forward(self, q, v):
        return dict(closure=q.new_zeros(len(q), 1, 3))


def single_step_batch(contact, body_contact=None):
    s = fixture(len(contact))
    if body_contact is None:
        body_contact = torch.zeros(len(contact), 1)
    return dict(q=s.q, v=s.v, warm=s.warm, applied=s.generalized, next_q=s.q, next_v=s.v,
                next_warm=s.warm, contact=contact, body_contact=body_contact, context=s.context)


def legacy_loss(wm, member, batch, geometry):
    """Frozen pre-existing objective, re-derived here as the compatibility oracle."""
    features = mechanical_features(batch['q'], batch['v'], batch['warm'], batch['applied'], batch['context'])
    predicted = wm.raw(features, member)
    target = mechanics_targets(batch['next_v'], batch['v'], batch['next_warm'], batch['contact'], batch.get('body_contact'))
    error = predicted-wm.target(target)
    value = predicted*wm.target.scale+wm.target.mean
    v = batch['v']+value[:, :wm.nv]
    q = integrate(batch['q'], v)
    measured = geometry(q, v)
    closure = (measured['closure']/.001).square().mean()
    qloss = (q[:, 7:]-batch['next_q'][:, 7:]).square().mean()
    rloss = rotation_error(q[:, 3:7], batch['next_q'][:, 3:7]).mean()
    return torch.nn.functional.smooth_l1_loss(error, torch.zeros_like(error))+closure*.01+qloss+rloss


def head_for_force(force, nv):
    """Pre-softplus head value that yields the requested physical wheel force."""
    target = mechanics_targets(torch.zeros(len(force), nv), torch.zeros(len(force), nv),
                               torch.zeros(len(force), nv), force, torch.zeros(len(force), 1))
    return target[:, 2*nv:2*nv+2].detach()


def values_for_force(force, nv):
    """Full output vector with only the two wheel head rows set to the requested force."""
    values = torch.zeros(len(force), 2*nv+3, dtype=force.dtype)
    values[:, 2*nv:2*nv+2] = head_for_force(force, nv)
    return values


def test_wheel_force_matches_prediction_softplus_rows():
    torch.manual_seed(11)
    wm = Ensemble(13, 12, 3, members=2, width=16, blocks=1)
    s = fixture(4)
    with torch.no_grad():
        predicted = wm.predict(s, s.generalized, 0)
        values = wm.raw(wm.features(s, s.generalized), 0)*wm.target.scale+wm.target.mean
        force = wheel_force(values, wm.nv)
    torch.testing.assert_close(force, predicted.contact)
    assert force.shape == (4, 2) and bool((force >= 0).all())


def test_inverse_softplus_target_round_trip_is_newtons():
    forces = torch.tensor([[0., 2.], [90., 150.], [3000., 0.5]])
    nv = 12
    target = mechanics_targets(torch.zeros(3, nv), torch.zeros(3, nv), torch.zeros(3, nv),
                               forces, torch.zeros(3, 1))
    recovered = torch.nn.functional.softplus(target[:, 2*nv:2*nv+2])
    torch.testing.assert_close(recovered, forces.clamp_min(1e-6), rtol=1e-5, atol=1e-6)


def test_force_loss_gradient_points_from_prediction_towards_truth():
    config = contact_force_config(dict(version='contact_force_v1', scale_n=10., lambda_force=.3))
    nv = 12
    cases = [  # (predicted force, real force): false support, missed support, impact tail
        (torch.tensor([[90., 20.]]), torch.tensor([[0., 0.]])),
        (torch.tensor([[0., 1.]]), torch.tensor([[90., 120.]])),
        (torch.tensor([[90., 90.]]), torch.tensor([[3000., 0.]]))]
    for predicted_force, real in cases:
        head = values_for_force(predicted_force, nv).requires_grad_(True)
        loss, detail = force_supervision(head, real, nv, config)
        loss.backward()
        gradient = head.grad[:, 2*nv:2*nv+2]
        force = wheel_force(head, nv).detach()
        assert bool(torch.isfinite(gradient).all())
        # dl/dhead has the sign of (prediction - truth): a descent step reduces the error.
        assert torch.equal(torch.sign(gradient), torch.sign(force-real))
        assert detail['contact_force_mae_N'] >= 0
    # The Huber tail is linear, so one 3000N impact cannot dominate the batch.
    tail, _ = force_supervision(values_for_force(torch.tensor([[90., 90.]]), nv),
                                torch.tensor([[3000., 3000.]]), nv, config)
    assert 1. < float(tail) < 300.


def test_force_gradient_matches_central_finite_difference():
    config = contact_force_config(dict(version='contact_force_v1', scale_n=10., lambda_force=.5))
    nv = 12
    # float64 so central differences are not dominated by cancellation in the Huber tail.
    real = torch.tensor([[0., 90.]], dtype=torch.float64)
    head = values_for_force(torch.tensor([[25., 40.]], dtype=torch.float64), nv)
    head.requires_grad_(True)
    loss, _ = force_supervision(head, real, nv, config)
    analytic = torch.autograd.grad(loss, head)[0][:, 2*nv:2*nv+2]
    epsilon = 1e-6
    numeric = torch.zeros_like(analytic)
    for row in range(head.shape[0]):
        for column in range(2):
            shifted = head.detach().clone()
            shifted[0, 2*nv+column] += epsilon
            shifted.requires_grad_(True)
            plus, _ = force_supervision(shifted, real, nv, config)
            shifted = head.detach().clone()
            shifted[0, 2*nv+column] -= epsilon
            shifted.requires_grad_(True)
            minus, _ = force_supervision(shifted, real, nv, config)
            numeric[row, column] = (plus-minus)/(2*epsilon)
    torch.testing.assert_close(analytic, numeric, rtol=1e-5, atol=1e-9)


def test_lambda_zero_is_literally_the_old_objective():
    assert contact_force_config(dict(version='contact_force_v1', lambda_force=0.)) is None
    assert contact_force_config(CONTACT_FORCE_FROZEN) == CONTACT_FORCE_FROZEN
    torch.manual_seed(47)
    source = Ensemble(13, 12, 3, members=1, width=16, blocks=1)
    batch = single_step_batch(torch.tensor([[90., 0.], [0., 120.]]))
    reference = legacy_loss(source, 0, batch, FlatGeometry())
    copy = Ensemble(**source.config)
    copy.load_state_dict(source.state_dict())
    zero, detail = supervised_loss(copy, 0, batch, FlatGeometry(),
                                   contact_force=contact_force_config(dict(version='contact_force_v1',
                                                                           lambda_force=0.)))
    torch.testing.assert_close(zero, reference, atol=0, rtol=0)
    assert 'contact_force_weighted' not in detail


def test_body_contact_dimension_is_not_supervised_in_force_space():
    config = contact_force_config(CONTACT_FORCE_FROZEN)
    wheel = torch.tensor([[0., 0.]])
    quiet = single_step_batch(wheel, body_contact=torch.zeros(1, 1))
    loud = single_step_batch(wheel, body_contact=torch.ones(1, 1)*123.)
    wm = Ensemble(13, 12, 3, members=1, width=16, blocks=1)
    first, detail_first = supervised_loss(wm, 0, quiet, FlatGeometry(), contact_force=config)
    second, detail_second = supervised_loss(wm, 0, loud, FlatGeometry(), contact_force=config)
    assert detail_first['contact_force_huber'] == pytest.approx(detail_second['contact_force_huber'])
    assert float(first.detach()) != pytest.approx(float(second.detach()))


def test_disabled_or_absent_declaration_reproduces_legacy_loss_and_gradient():
    torch.manual_seed(23)
    source = Ensemble(13, 12, 3, members=1, width=16, blocks=1)
    batch = single_step_batch(torch.tensor([[0., 90.], [90., 0.]]))
    reference = legacy_loss(source, 0, batch, FlatGeometry())
    first = torch.autograd.grad(reference, list(source.parameters()))
    for declaration in (None, dict(version='none')):
        copy = Ensemble(**source.config)
        copy.load_state_dict(source.state_dict())
        loss, detail = supervised_loss(copy, 0, batch, FlatGeometry(),
                                       contact_force=contact_force_config(declaration))
        torch.testing.assert_close(loss, reference, atol=0, rtol=0)
        for want, got in zip(first, torch.autograd.grad(loss, list(copy.parameters()))):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        assert 'contact_force_huber' not in detail and 'contact_force_weighted' not in detail


def test_nonzero_lambda_changes_only_the_wheel_head_rows_and_shared_trunk():
    torch.manual_seed(31)
    source = Ensemble(13, 12, 3, members=1, width=16, blocks=1)
    copy = Ensemble(**source.config)
    copy.load_state_dict(source.state_dict())
    batch = single_step_batch(torch.tensor([[0., 0.], [90., 0.]]))
    plain, _ = supervised_loss(source, 0, batch, FlatGeometry())
    forced, detail = supervised_loss(copy, 0, batch, FlatGeometry(),
                                     contact_force=contact_force_config(CONTACT_FORCE_FROZEN))
    assert float(forced.detach()) > float(plain.detach())
    assert detail['contact_force_weighted'] > 0
    g_plain = torch.autograd.grad(plain, (source.members[0].output.weight, source.members[0].input.weight))
    g_forced = torch.autograd.grad(forced, (copy.members[0].output.weight, copy.members[0].input.weight))
    nv = source.nv
    torch.testing.assert_close(g_forced[0][:2*nv], g_plain[0][:2*nv], atol=0, rtol=0)
    assert not torch.allclose(g_forced[0][2*nv:2*nv+2], g_plain[0][2*nv:2*nv+2])
    torch.testing.assert_close(g_forced[0][2*nv+2:], g_plain[0][2*nv+2:], atol=0, rtol=0)
    assert not torch.allclose(g_forced[1], g_plain[1])


def test_declaration_dispatch_is_explicit_and_rejects_unknown_versions():
    assert contact_force_config(None) is None
    assert contact_force_config(dict(version='none')) is None
    frozen = contact_force_config(CONTACT_FORCE_FROZEN)
    assert frozen['version'] == 'contact_force_v1' and frozen['scale_n'] >= 1. and frozen['force_dims'] == 2
    with pytest.raises(ValueError, match='unknown contact force'):
        contact_force_config(dict(version='contact_force_v2'))
    with pytest.raises(ValueError, match='newton scale'):
        contact_force_config(dict(version='contact_force_v1', scale_n=.1))
    with pytest.raises(ValueError, match='finite scalar in'):
        contact_force_config(dict(version='contact_force_v1', lambda_force=-.5))
    with pytest.raises(ValueError, match='two wheel contact forces'):
        contact_force_config(dict(version='contact_force_v1', force_dims=3))
    from native_stand.world_model.experiment import contact_force_supervision
    assert contact_force_supervision({})['contact_force_supervision'] == dict(version='none')
    assert (contact_force_supervision({}, CONTACT_FORCE_FROZEN)['contact_force_supervision']['lambda_force']
            == CONTACT_FORCE_FROZEN['lambda_force'])
    with pytest.raises(ValueError, match='unknown contact force'):
        contact_force_supervision({}, dict(version='guess_from_dimensions'))


def write_episode_under(path, split, contact_pattern, steps=24):
    from native_stand.world_model.dataset import write_episode
    s = fixture(1)
    rows, micro, blobs = [], [], []
    for i in range(steps):
        b = replace(s, tick=torch.tensor([i*10]), episode_step=torch.tensor([i]), owner=torch.tensor([1]))
        a = replace(b, tick=b.tick+10, episode_step=b.episode_step+1)
        rows.append(dict(before=b, after=a, action=torch.zeros(1, 6), observation=torch.zeros(1, 31),
                         reward=torch.ones(1)*.01, terminated=torch.tensor([False]),
                         truncated=torch.tensor([i == steps-1]), reward_raw=torch.ones(1, 1),
                         reward_density=torch.ones(1, 1), reward_dt=torch.ones(1, 1)*.01,
                         reward_fields=['synthetic']))
        micro.extend([dict(q=s.q[0].numpy(), v=s.v[0].numpy(), warm=s.warm[0].numpy(),
                           applied=s.generalized[0].numpy(), next_q=s.q[0].numpy(), next_v=s.v[0].numpy(),
                           next_warm=s.warm[0].numpy(),
                           contact=np.asarray(contact_pattern(j % 10), dtype=np.float64),
                           body_contact=np.zeros(1)) for j in range(10)])
        blobs.append(np.zeros(2))
    write_episode(path, dict(id=split, split=split, tuple_id=split, controller='teacher4500'),
                  rows, micro, blobs, {'synthetic': True})


def build_fixture_replay(tmp_path):
    from native_stand.world_model.dataset import Replay, build_manifest
    paths = []
    for split in ('train', 'validation'):
        write_episode_under(tmp_path/split, split, lambda j: [90., 0.] if j < 5 else [0., 0.])
        paths.append(tmp_path/split)
    return Replay(build_manifest(paths, tmp_path/'dataset.json', {'fixture': 'NO_ROBOT_DATA'}), 'train')


class FixtureBudget:
    def __init__(self):
        self.local = {}

    def check(self):
        pass

    def storage_guard(self, n):
        assert n > 0

    def charge(self, key, n):
        self.local[key] = self.local.get(key, 0) + n


def test_fit_world_records_force_recipe_history_and_resume_state(tmp_path):
    from native_stand.world_model.dataset import Replay
    train = build_fixture_replay(tmp_path)
    validation = Replay(train.manifest, 'validation')
    budget = FixtureBudget()
    result = fit_world(train, validation, FlatGeometry(), budget, tmp_path/'wm', steps=2, batch_size=2,
                       width=16, members=3, fit_window_substeps=8, sampling='uniform_episode_v1',
                       training_revision='supervised_contactforce_force_v5',
                       contact_force=CONTACT_FORCE_FROZEN)
    assert budget.local['wm_optimizer_steps'] == 2
    saved = torch.load(result['checkpoint'], weights_only=False)
    recipe = saved['training_recipe']
    assert recipe['contact_force_supervision']['version'] == 'contact_force_v1'
    assert recipe['contact_force_supervision']['scale_n'] == 10.
    assert 'lambda_force' in recipe['loss_definition']
    resume = saved['resume_state']
    assert resume['next_member'] == 2 % 3 and 'numpy_rng' in resume and resume['torch_rng'].numel() > 0
    assert resume['consumed_by'].startswith('NOT_CONSUMED')
    assert saved['episode_ids'] == [row['episode_id'] for row in train.rows]
    history = result['history'][-1]
    assert {'contact_force_huber', 'contact_force_mae_N', 'contact_force_false_support',
            'contact_force_base_loss'} <= set(history)
    assert history['contact_force_mae_N'] > 0
    assert len(history['sampler_state_sha']) == 64 and history['member'] == (2-1) % 3
    last = torch.load(Path(result['checkpoint']).with_name('last_world.pt'), weights_only=False)
    assert last['resume_state']['next_member'] == 2 % 3
    assert last['training_recipe']['validation_sampling'] == 'uniform_episode_v1'


def test_disabled_arm_checkpoint_has_no_force_term(tmp_path):
    from native_stand.world_model.dataset import Replay
    train = build_fixture_replay(tmp_path)
    validation = Replay(train.manifest, 'validation')
    budget = FixtureBudget()
    result = fit_world(train, validation, FlatGeometry(), budget, tmp_path/'wm_control', steps=2,
                       batch_size=2, width=16, members=3, fit_window_substeps=8,
                       training_revision='supervised_contactforce_control_v5',
                       contact_force=dict(version='none'))
    saved = torch.load(result['checkpoint'], weights_only=False)
    assert saved['training_recipe']['contact_force_supervision'] is None
    assert 'contact_force_huber' not in result['history'][-1]
    assert 'lambda_force' not in saved['training_recipe']['loss_definition']
    assert load_world(result['checkpoint'], 'cpu').config['members'] == 3


def test_declaration_reaches_fit_world_through_the_owner_readback():
    from native_stand.world_model.owner import world_fit_arguments
    from native_stand.world_model.experiment import contact_force_supervision
    declared = contact_force_supervision(dict(members=3, width=1024, blocks=2, batch_size=256, seed=3017,
        fit_window_substeps=80, activation_checkpoint=False, training_revision='r', sampling='uniform_episode_v1',
        supervision_geometry='closure_fk_v1'), CONTACT_FORCE_FROZEN)
    arguments = world_fit_arguments(dict(world_model=declared))
    assert arguments['contact_force'] == declared['contact_force_supervision']
    assert contact_force_config(arguments['contact_force']) == CONTACT_FORCE_FROZEN
    control = contact_force_supervision(dict(members=3), dict(version='none'))
    assert world_fit_arguments(dict(world_model=control))['contact_force'] is None
    # A legacy frozen spec has no such key at all and must still be usable.
    assert world_fit_arguments(dict(world_model=dict(members=3)))['contact_force'] is None


def test_owner_handoff_is_strict_and_one_shot(tmp_path, monkeypatch):
    from native_stand.world_model.experiment import (executing_session, record_owner_handoff,
                                                     verify_owner_handoff)
    monkeypatch.delenv('HAPI_SESSION_ID', raising=False)
    monkeypatch.setenv('CODEBUDDY_SESSION_ID', 'session-under-test')
    assert executing_session() == dict(session_id='session-under-test', kind='ide_session',
                                       hapi_session=None, source_env='CODEBUDDY_SESSION_ID')
    parent = tmp_path/'spec.json'
    parent.write_text(json.dumps({'run': str(tmp_path/'run'), 'authorization': {'owner_session': 'dead-owner'}}))
    analysis = tmp_path/'analysis.json'
    analysis.write_text(json.dumps({'event_id': 'evt', 'status': 'STOPPED_GATE'}))
    out = tmp_path/'handoff.json'
    scope = dict(arms=['control', 'force'], actions=['prepare', 'execute'], member_dispatches_per_arm=300)
    with pytest.raises(ValueError, match='delegation'):
        record_owner_handoff(parent, 'short', scope=scope, output=tmp_path/'x.json')
    record = record_owner_handoff(parent, 'user delegated the route to this session',
                                  scope=scope, output=out, analysis=analysis)
    frozen = json.loads(out.read_text())
    assert frozen['parent']['owner_session'] == 'dead-owner'
    assert frozen['new_owner']['session_id'] == 'session-under-test'
    assert frozen['constraints']['no_hapi_session_fabrication'] is True
    assert frozen['constraints']['parent_spec_read_only'] is True
    record.update(path=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                  owner_session='session-under-test')
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    assert verify_owner_handoff(record, parent_spec_sha=parent_sha, session_id='session-under-test',
                                action='execute')['version'] == 'wm_owner_handoff_v1'
    with pytest.raises(ValueError, match='another executing session'):
        verify_owner_handoff(record, parent_spec_sha=parent_sha, session_id='other', action='execute')
    with pytest.raises(ValueError, match='different parent'):
        verify_owner_handoff(record, parent_spec_sha='0'*64, session_id='session-under-test', action='execute')
    with pytest.raises(ValueError, match='does not cover'):
        verify_owner_handoff(record, parent_spec_sha=parent_sha, session_id='session-under-test', action='collect')
    out.write_text(out.read_text().replace('dead-owner', 'someone-else'))
    with pytest.raises(ValueError, match='changed or missing'):
        verify_owner_handoff(record, parent_spec_sha=parent_sha, session_id='session-under-test', action='execute')
    with pytest.raises(ValueError, match='one-shot'):
        record_owner_handoff(parent, 'user delegated the route to this session', scope=scope, output=out)


def test_interrupted_attempt_settles_only_when_its_pid_is_dead(tmp_path):
    import os
    from native_stand.world_model.experiment import settle_interrupted_attempt
    spec = tmp_path/'spec.json'
    spec.write_text(json.dumps({'version': 'v'}))
    digest = hashlib.sha256(spec.read_bytes()).hexdigest()
    book = dict(spec_sha256=digest, attempts={
        'gone': dict(stage='predict', output=str(tmp_path/'o1'), status='RUNNING', pid=2**22+7),
        'live': dict(stage='predict', output=str(tmp_path/'o2'), status='RUNNING', pid=os.getpid()),
        'done': dict(stage='train-world', output=str(tmp_path/'o3'), status='COMPLETED', pid=1)})
    (tmp_path/'budget.json').write_text(json.dumps(book))
    with pytest.raises(ValueError, match='still alive'):
        settle_interrupted_attempt(spec, 'live', error='worker died without settling')
    with pytest.raises(ValueError, match='already settled'):
        settle_interrupted_attempt(spec, 'done', error='worker died without settling')
    with pytest.raises(ValueError, match='unknown attempt'):
        settle_interrupted_attempt(spec, 'missing', error='worker died without settling')
    result = settle_interrupted_attempt(spec, 'gone', error='worker died without settling its attempt')
    assert result['status'] == 'FAILED'
    settled = json.loads((tmp_path/'budget.json').read_text())['attempts']['gone']
    assert settled['status'] == 'FAILED' and settled['interrupted'] is True
    assert 'used' not in json.loads((tmp_path/'budget.json').read_text())


def test_force_window_never_crosses_an_episode_end():
    from native_stand.world_model.dataset import mechanics_indices
    data = [{'micro_q': np.zeros((12, 45)), 'micro_contact': np.zeros((12, 2))}]
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match='complete window'):
        mechanics_indices(data, 1, rng, episode_pool=[0], window=13)
    selected, ids, events = mechanics_indices(data, 4, rng, episode_pool=[0], window=12)
    assert events == 0 and all(i == 0 for i in ids) and len(selected) == 4
