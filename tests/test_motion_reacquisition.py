"""CPU contracts for one deferred standing -> Motion transition."""
FROZEN_TEST_SCOPE = 'PORTABLE'
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as S

import numpy as np
import pytest
import torch

from native_stand.students import motion_reacquisition as m
from native_stand.students.data import sha


def controller():
    p = Path.cwd() / 'scripts/transition_motion_after_standing.py'
    spec = importlib.util.spec_from_file_location('motion_transition_test', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('stage', range(6))
@pytest.mark.parametrize('envs', (1, 64, 4096))
def test_fixed_quota_all_unlocked_and_signs_equal(stage, envs):
    s = m.stage_config(stage, envs)
    assert len(s['commands']) == 1 + 4 * (stage + 1)
    assert sum(s['slot_counts']) == len(s['sampling_slots']) == envs
    assert len(set(map(tuple, s['commands']))) == len(s['commands'])
    if envs != 1:
        assert s['slot_counts'][0] == envs // 4
        for index in range(1, len(s['commands']), 2):
            assert s['slot_counts'][index] == s['slot_counts'][index + 1]
        assert s['slot_counts'][-4:] == [envs * (3 if stage == 0 else 2) // 16] * 4
    if stage:
        assert s['commands'][:len(m.commands(stage - 1))] == m.commands(stage - 1)


def candidate(stage, update, accepted=True):
    return {'stage': stage, 'update': update, 'checkpoint': f'/fixture/model_update_{update}.pt',
            'sha256': f'{update:064x}', 'complete_matrix': True,
            'per_env': [{'accepted': accepted, 'survival_passed': True}],
            'evaluations': [{'case': {'command': v + [.34], '_timing_level': 4},
                             'path': f'eval_{i}.json', 'sha256': 'b' * 64,
                             'per_env': [{'env': j, 'accepted': accepted} for j in range(8)]}
                            for i, v in enumerate(m.commands(stage))]}


def state():
    return {'stage': 0, 'stage_started_update': 0, 'last_evaluation_update': -1,
            'consecutive_passes': 0, 'accepted_by_stage': {}}


def test_two_full_matrices_unlock_without_behavior_stop_and_never_skip():
    s = state()
    for stage in range(6):
        s, decision = m.advance(s, candidate(stage, stage * 500 + 250))
        assert not decision['advanced'] and not decision['stop']
        s, decision = m.advance(s, candidate(stage, stage * 500 + 500))
        assert decision['advanced'] == (stage < 5)
        assert decision['all_stages_passed'] == (stage == 5)
        assert not decision['stop']
    assert set(s['accepted_by_stage']) == set(map(str, range(6)))


@pytest.mark.parametrize('fault', ('unknown', 'failed', 'wrong_level', 'wrong_command', 'short', 'duplicate_env', 'missing_case'))
def test_incomplete_or_failed_evidence_does_not_promote(fault):
    s, _ = m.advance(state(), candidate(0, 250))
    c = candidate(0, 500)
    r = c['evaluations'][0]
    if fault == 'unknown': r['per_env'][0]['accepted'] = None
    if fault == 'failed': r['per_env'][0]['accepted'] = False
    if fault == 'wrong_level': r['case']['_timing_level'] = 0
    if fault == 'wrong_command': r['case']['command'] = [0, 1, .34]
    if fault == 'short': r['per_env'].pop()
    if fault == 'duplicate_env': r['per_env'][0]['env'] = 1
    if fault == 'missing_case': c['evaluations'].pop()
    result, d = m.advance(s, c)
    assert result['stage'] == 0 and result['consecutive_passes'] == 0
    assert not d['stop'] and not d['advanced']


def test_repeated_checkpoint_or_gap_is_not_consecutive_confirmation():
    s, _ = m.advance(state(), candidate(0, 250))
    c = candidate(0, 500); c['sha256'] = s['last_checkpoint_sha256']
    out, _ = m.advance(s, c)
    assert out['stage'] == 0 and out['consecutive_passes'] == 1
    out, _ = m.advance(s, candidate(0, 750))
    assert out['stage'] == 0 and out['consecutive_passes'] == 1
    with pytest.raises(ValueError): m.advance(s, candidate(1, 500))


def test_command_stage_cannot_change_physical_inertia_and_no_rng_effect():
    fake_path = Path.cwd() / 'tests/test_motion_timing.py'
    fs = importlib.util.spec_from_file_location('motion_timing_fake_fixture', fake_path)
    fixture_module = importlib.util.module_from_spec(fs); fs.loader.exec_module(fixture_module)
    fake_env = fixture_module.fake_env
    from native_stand.students.motion_timing_runtime import Timing
    from native_stand.students.motion_wheel_matching import configuration
    e = fake_env(8); e.cfg = S(leg_kp=40., leg_kd=2., wheel_kv=.2)
    e.motion_config = {'course_version': m.VERSION}
    a = torch.ones(8, 6) * .01
    e.robot.write_joint_armature_to_sim_index = lambda *, armature, joint_ids: a.__setitem__((slice(None), joint_ids.long()), armature)
    e.robot.root_view = S(get_dof_armatures=lambda: a, get_dof_dampings=lambda: torch.zeros_like(a))
    rng = torch.get_rng_state().clone()
    t = Timing(e, configuration(), 4)
    for override in (None, 0, 3, 4):
        t.configure(override)
        assert (t.armature_levels == 4).all()
        torch.testing.assert_close(a[:, [0, 1, 3, 4]], torch.full((8, 4), .012))
    assert torch.equal(rng, torch.get_rng_state())


def test_recipe_keeps_full140_and_rejects_drift():
    from native_stand.students.motion_training import declaration
    from native_stand.students.motion_tracking_reward import PARALLEL_VERSION
    from native_stand.students.motion_antiosc_privileged import VERSION, network
    from native_stand.students.motion_wheel_matching import configuration
    from native_stand.students.training_recipe import validate_instance
    r = declaration('a' * 64, 0, reward_version=PARALLEL_VERSION)
    r.update(course_version=m.VERSION, course_stage=0, timing_configuration=configuration(),
             critic_privileged_version=VERSION, network_architecture=network(),
             critic='full_motion140_wheel_match', optimizer='source_Adam_full_state',
             checkpoint_every_updates=100, milestones=list(range(250, 10000, 250)))
    original = copy.deepcopy(r)
    assert validate_instance(r, 'a' * 64) is r and r == original
    for key, value in [('optimizer', 'new_Adam'), ('unknown_option', 1), ('BC_coefficient', .1)]:
        bad = copy.deepcopy(r); bad[key] = value
        with pytest.raises(ValueError): validate_instance(bad, 'a' * 64)


def arrays():
    a = {'state': np.zeros((3000, 1, 39)), 'posture': np.zeros((3000, 1, 10)),
         'geometry': np.zeros((3000, 1, 9)), 'controller': np.full((3000, 1, 1), 3),
         'first_episode_valid': np.ones((3000, 1), bool)}
    a['state'][:, :, 6] = .34
    a['posture'][:, :, 2:4] = .34
    a['geometry'][:, :, 7] = -1.
    return a


@pytest.mark.parametrize('fault,gate', [('height', 'reacquisition_height'), ('legs', 'reacquisition_leg_mae'),
                                      ('wheel', 'reacquisition_wheel_mae'), ('short', 'reacquisition_height')])
def test_geometry_screen_uses_complete_hold_not_survival_only(fault, gate):
    a = arrays()
    r = {'checks': {'pitch_absolute_quality': True}, 'accepted': True, 'failed_checks': []}
    assert m.assess(copy.deepcopy(r), a, 0)['accepted']
    if fault == 'height': a['state'][:, :, 6] = .31
    if fault == 'legs': a['posture'][:, :, 2] += .03
    if fault == 'wheel': a['geometry'][:, :, 2] += .03
    if fault == 'short': a['first_episode_valid'][0] = False
    result = m.assess(r, a, 0)
    assert not result['accepted'] and result['checks'][gate] is False


def test_install_checks_all_base_hashes_before_any_write(tmp_path):
    c = controller()
    root, bundle = tmp_path / 'root', tmp_path / 'bundle'
    (root / 'src').mkdir(parents=True); (bundle / 'files/src').mkdir(parents=True)
    files = []
    for name in ('one.py', 'two.py'):
        path = root / 'src' / name; path.write_text('old')
        q = bundle / 'files/src' / name; q.write_text('new')
        files.append({'path': f'src/{name}', 'before_sha256': c.sha(path), 'after_sha256': c.sha(q)})
    c.write(bundle / 'manifest.json', {'files': files})
    c.write(bundle / 'bundle_ready.json', {'status': 'CPU_PASSED', 'manifest_sha256': c.sha(bundle / 'manifest.json')})
    (root / 'src/two.py').write_text('another engineer changed it')
    with pytest.raises(ValueError, match='refusing overwrite'): c.install_bundle(bundle, root)
    assert (root / 'src/one.py').read_text() == 'old'


def test_event_wait_handles_atomic_replace_without_timed_poll(tmp_path):
    c = controller(); events = c.Events([tmp_path])
    try:
        c.write(tmp_path / 'target.json', {'ready': True})
        changed = events.wait(lambda p: p == tmp_path / 'target.json')
        assert changed == [tmp_path / 'target.json']
    finally: events.close()


def test_settlement_rejects_failed_and_running_parent(tmp_path):
    c = controller(); run = tmp_path / 'run'; (run / 'formal').mkdir(parents=True)
    p = tmp_path / 'spec.json'; p.write_text('{}')
    plan = {'parent_run': str(run), 'parent_spec': str(p), 'attempt_id': 'fixture'}
    c.write(run / 'chain_terminal_report.json', {'status': 'STOPPED_BOUNDED', 'failure': None})
    c.write(run / 'formal/terminal_report.json', {'status': 'STOPPED_BOUNDED', 'failure': None,
             'attempt_id': 'fixture', 'actual_updates': 2500, 'stop_kind': 'user_signal'})
    c.write(tmp_path / 'budget.json', {'used': {'formal': 2500, 'smoke': 5}, 'attempts': {'x': {'status': 'RUNNING'}}})
    with pytest.raises(ValueError): c.verify_settlement(plan)
    c.write(tmp_path / 'budget.json', {'used': {'formal': 2500, 'smoke': 5}, 'attempts': {'x': {'status': 'STOPPED_BOUNDED'}}})
    assert c.verify_settlement(plan)['formal_updates'] == 2500
    c.write(run / 'chain_terminal_report.json', {'status': 'FAILED', 'failure': 'infrastructure'})
    with pytest.raises(ValueError): c.verify_settlement(plan)


def standing_parent(tmp_path):
    """Synthetic full140 checkpoint and immutable passing raw evidence, no project runs."""
    c = controller()
    from native_stand.students.motion_training import declaration
    from native_stand.students.motion_tracking_reward import PARALLEL_VERSION
    from native_stand.students.motion_antiosc_privileged import VERSION, network, contract
    from native_stand.students.motion_wheel_matching import configuration
    from native_stand.students.motion_guard import evidence_configuration
    from native_stand.students.motion_antioscillation import protocol as parent_protocol
    from native_stand.students.motion_parallel_assessment import safety_protocol
    from native_stand.students.model import Student
    from native_stand.students.resume import rng_state
    name = '2026_10_22-00_00__fixture_standing'
    p = tmp_path / 'configs/experiments' / name / 'spec.json'; p.parent.mkdir(parents=True)
    run = tmp_path / 'runs' / name; formal = run / 'formal'; formal.mkdir(parents=True)
    r = declaration('a' * 64, 0, reward_version=PARALLEL_VERSION)
    r.update(course_version=m.SOURCE_COURSE, course_stage=0, timing_configuration=configuration(),
             critic_privileged_version=VERSION, network_architecture=network(),
             critic='full_motion140_wheel_match', optimizer='source_Adam_full_state',
             checkpoint_every_updates=100, milestones=list(range(250, 10000, 250)))
    template = {'kind': 'student', 'case_id': 'fixture', 'checkpoint': '/fixture', 'sha256': 'a' * 64,
                'role': 'MotionZeroStudent', 'family': 'motion_native_normal',
                'command': [0., 0., .34], 'seed': 123, 'yaw_rad': 0., 'seconds': 31.52}
    old = {'experiment': name, 'interface_version': 'multiskill_mcu_v2', 'student_training': r,
           'reward': r['reward'], 'network_architecture': network(), 'student_interface': contract(),
           'observation': {'frame_dim': 22, 'actor_dim': 31, 'critic_dim': 140},
           'motion_timing': {'version': m.SOURCE_COURSE, 'configuration': configuration(), 'stage': 0,
                             'wheel_match_restart': {'historical_online_used': 397}},
           'motion_training': {'version': r['version'], 'stage': 0, 'resume': None,
                               'spec_path': str(p), 'early_stop_policy': evidence_configuration()},
           'motion_course': {'source_recipe_updates': 1, 'configuration': {}},
           'motion_assessment': parent_protocol(),
           'motion_symmetry': {'assessment': {'swing_strict_max_rad': .05}},
           'motion_differential': {}, 'motion_spin_quality': {},
           'student_evaluation': {'version': parent_protocol()['version'], 'cases': [template], 'max_output_bytes': 128 * 2**20},
           'motion_startup': {}, 'curriculum': {'active_stage': 0, 'stages': []},
           'evaluation': {'envs': 64, 'scored_envs': 8, 'seconds': 31.52},
           'performance': {}, 'budget': {}, 'artifacts': {}, 'student_dependencies': {},
           'ppo': {'envs': 4096, 'steps_per_env': 128}, 'device': {'physical_index': 0},
           'authorization': {}, 'notifications': {}, 'resources': {}}
    c.write(p, old)
    actor = Student(True, version='multiskill_mcu_v2', network_architecture=network())
    critic = Student(False, version='multiskill_mcu_v2', network_architecture=network(), critic_privileged_version=VERSION)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-5)
    # Seed real optimizer moments in a synthetic CPU fixture, not online PPO.
    (actor(torch.ones(2, 31)).sum() + critic(torch.ones(2, 140)).sum()).backward(); optimizer.step()
    registry = []
    for update in (2250, 2500):
        recipe = copy.deepcopy(r); recipe.update(stage=4, course_stage=4)
        cp = formal / f'model_update_{update}.pt'
        data = {'interface_version': 'multiskill_mcu_v2', 'role': 'MotionZeroStudent',
                'actor': actor.state_dict(), 'critic': critic.state_dict(), 'optimizer': optimizer.state_dict(),
                'network_architecture': network(), 'critic_privileged_version': VERSION,
                'training_rng': rng_state(), 'reward_refinement_state': [{'target': [0., 0.], 'mix': 0.}],
                'recipe_version': r['version'], 'training_recipe': recipe,
                'source_sha256': 'a' * 64, 'dataset_sha256': 'b' * 64, 'manifest_sha256': 'c' * 64,
                'asset_sha256': 'd' * 64, 'spec_sha256': sha(p), 'source_code_sha256': 'e' * 64,
                'update': update, 'recipe_updates': update + 100, 'fixed_std': True, 'BC_coefficient': 0.}
        torch.save(data, cp); records = []
        for level in (0, 3, 4):
            case = {**template, 'case_id': f'armature_4_level_{level}', '_timing_level': level,
                    'checkpoint': str(cp), 'sha256': sha(cp)}
            path = formal / f'evaluation_{update}_{level}.json'; trace = path.with_suffix('.npz')
            trace.write_bytes(b'opaque CPU evidence fixture')
            es = [{'env': i, 'accepted': True, 'survival_passed': True, 'controlled_seconds': 30.,
                   'symmetry_metrics': {}, 'checks': {'antiosc_complete': True}} for i in range(8)]
            raw = {'case': case, 'checkpoint_sha256': sha(cp), 'spec_sha256': sha(p),
                   'protocol': safety_protocol(old), 'model_unchanged': True, 'scored_envs': 8,
                   'per_env': es, 'motion_timing': {'override': level, 'actuator_readback': {
                       'levels': [64 if i == level else 0 for i in range(5)]}},
                   'trace': {'path': trace.name, 'sha256': sha(trace)}}
            c.write(path, raw)
            records.append({'case': case, 'path': path.name, 'sha256': sha(path), 'per_env': es})
        registry.append({'checkpoint': str(cp), 'sha256': sha(cp), 'stage': 4, 'update': update,
                         'complete_matrix': True, 'evaluations': records, 'per_env': [e for r in records for e in r['per_env']]})
        c.write(formal / f'reward_refinement_after_{update}.json', {'update': update,
                'evaluation_inputs': [{'path': r['path'], 'sha256': r['sha256']} for r in records]})
    c.write(formal / 'candidate_registry.json', registry)
    c.write(formal / 'course_progress.json', {'state': {'stage': 4, 'consecutive_passes': 2, 'last_evaluation_update': 2500},
            'decision': {'all_stages_passed': True}, 'checkpoint_sha256': registry[-1]['sha256']})
    c.write(run / 'chain_terminal_report.json', {'status': 'STOPPED_BOUNDED', 'failure': None})
    c.write(formal / 'terminal_report.json', {'status': 'STOPPED_BOUNDED', 'failure': None,
            'actual_updates': 2500, 'attempt_id': 'fixture', 'stop_kind': 'user_signal'})
    c.write(p.with_name('budget.json'), {'used': {'formal': 2500, 'smoke': 5}, 'attempts': {'fixture': {'status': 'STOPPED_BOUNDED'}}})
    plan = {'parent_spec': str(p), 'parent_run': str(run), 'parent_spec_sha256': sha(p), 'attempt_id': 'fixture'}
    return plan, registry[-1]['checkpoint']


def test_event_proof_requires_raw_sha_trace_and_both_complete_matrices(tmp_path):
    c = controller(); plan, cp = standing_parent(tmp_path)
    proof = c.completed_pair(plan)
    assert proof['status'] == 'CONFIRMED' and proof['checkpoints'][-1]['checkpoint'] == cp
    raw = next(Path(p) for p in proof['inputs'] if p.endswith('evaluation_2250_4.json'))
    raw.write_text(raw.read_text() + ' ')
    with pytest.raises(ValueError, match='SHA'): c.completed_pair(plan)


def test_successor_build_and_full_cpu_restore_no_interface_drift(tmp_path):
    c = controller(); plan, cp = standing_parent(tmp_path); proof = c.completed_pair(plan)
    dest = tmp_path / 'configs/experiments/2026_10_22-01_00__fixture_motion/spec.json'
    spec = m.build(plan['parent_spec'], cp, experiment=dest.parent.name, spec_path=dest,
                   confirmation=proof, owner_session='fixture_session')
    assert m.verify(spec)
    assert spec['budget']['smoke_updates'] + spec['budget']['formal_updates'] == 10000
    assert spec['budget']['formal_wall_s'] is None and spec['budget']['chain_wall_s'] is None
    assert spec['motion_timing']['stage'] == 4 and spec['student_training']['stage'] == 0
    assert len(spec['student_evaluation']['cases']) == 5
    assert m.effective_spec(spec, 5)['motion_timing']['stage'] == 4
    from native_stand.students.requests import validate
    assert not validate(spec)
    from native_stand.students.ppo import assemble
    from native_stand.students.motion_course_requests import restore
    from native_stand.recovery.controlled_ppo import KEY
    from tensordict import TensorDict
    obs = TensorDict({'student_actor': torch.zeros(2, 31), 'student_critic': torch.zeros(2, 140),
                      KEY: torch.ones(2, 1, dtype=torch.bool)}, batch_size=[2])
    ppo = assemble(cp, obs, steps_per_env=2, recipe=spec['student_training'])
    saved = torch.load(cp, weights_only=False, map_location='cpu')
    r = restore(ppo, saved, spec, 'formal')
    assert r['source_Adam_loaded'] and r['source_RNG_loaded']
    for name, value in saved['actor'].items(): assert torch.equal(value, ppo.actor.state_dict()[name])
    for name, value in saved['critic'].items(): assert torch.equal(value, ppo.critic.state_dict()[name])
    actual = ppo.optimizer.state_dict()
    assert actual['param_groups'] == saved['optimizer']['param_groups']
    for pid, row in saved['optimizer']['state'].items():
        for name, value in row.items(): assert torch.equal(actual['state'][pid][name], value)
    assert len(ppo.reward_refinement_state) == 5
    bad = copy.deepcopy(spec); bad['control'] = {'wheel_kv': 1.}
    with pytest.raises(ValueError, match='drift'): m.verify(bad)


def test_storage_bound_covers_future25_commands_and_anti_fields(tmp_path):
    c = controller(); plan, cp = standing_parent(tmp_path)
    out = tmp_path / 'configs/experiments/2026_10_22-01_00__fixture_motion/spec.json'
    s = m.build(plan['parent_spec'], cp, experiment=out.parent.name, spec_path=out,
                confirmation=c.completed_pair(plan), owner_session='fixture_session')
    s['control'] = {'policy_dt_s': .01}
    from native_stand.command_v6.tools.storage_budget import motion_student_estimate
    estimate = motion_student_estimate(s, 'formal')
    minimum = 25 * 40 * (131072 + 3152 * 8 * 114 * 4)
    assert estimate['evaluation_bytes'] == minimum
    assert s['artifacts']['storage_max_bytes'] == 13 * 2**30
    assert (estimate['estimated_artifact_bytes'] + estimate['atomic_temp_peak_bytes']
            + estimate['terminal_reserve_bytes'] <= s['artifacts']['storage_max_bytes'])


def test_invalid_gravity_fails_without_nan_report():
    a = arrays(); a['geometry'][:, :, 7] = 0.
    r = m.assess({'checks': {}, 'accepted': True, 'failed_checks': []}, a, 0)
    assert r['accepted'] is False
    json.dumps(r, allow_nan=False)
