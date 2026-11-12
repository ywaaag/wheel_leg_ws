"""CPU regressions for Motion refinement and controller reporting contracts."""
FROZEN_TEST_SCOPE = 'PORTABLE'
import copy
from types import SimpleNamespace as S

import numpy as np
import pytest
import torch

from native_stand.students import motion_refinement as refine
from native_stand.students import motion_reacquisition as motion
from native_stand.students import motion_robustness as dr
from native_stand.students.motion_tracking_reward import configuration, REFINEMENT_VERSION


def evidence(version=motion.ASSESSMENT):
    policy = configuration(REFINEMENT_VERSION)['refinement']
    rows = []
    for target in (0., .5, -.5):
        checks = {k: True for k in policy['required_checks']
                  if k not in ('pitch_source_guard', 'roll_source_guard')}
        checks.update(pitch_absolute_quality=True, roll_absolute_quality=True,
                      quality_complete_window=True, roll_quality=True,
                      leg_balance_quality=True, wheel_alignment_quality=True,
                      joint_mirror_quality=True)
        rows.append({'case': {'command': [target, 0., .34]}, 'per_env': [
            {'env': i, 'version': version, 'accepted': True,
             'refinement_evidence_version': 'motion31_absolute_refinement_evidence_v1',
             'checks': copy.deepcopy(checks),
             'tracking': {'MAE': .1, 'direction': 1., 'gain': 1.}}
            for i in range(8)]})
    return policy, rows


def test_reacquisition_refinement_uses_absolute_posture_and_keeps_wide_floor():
    policy, rows = evidence()
    before = copy.deepcopy(rows)
    state = refine.initial([[0., 0.], [.5, 0.], [-.5, 0.]])
    rng = torch.get_rng_state().clone()
    state, decisions = refine.advance(state, rows, policy)
    assert [d['eligible'] for d in decisions] == [False, True, True]
    assert [s['mix'] for s in state] == [0., 0., 0.]
    state, _ = refine.advance(state, rows, policy)
    assert [s['mix'] for s in state] == [0., .25, .25]
    for _ in range(12):
        state, _ = refine.advance(state, rows, policy)
    assert [s['mix'] for s in state] == [0., .75, .75]
    assert rows == before and torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('fault', ['rejected', 'unknown', 'missing', 'false', 'quality', 'tracking'])
def test_new_refinement_requires_complete_acceptance_and_each_absolute_gate(fault):
    policy, rows = evidence()
    row = rows[1]['per_env'][0]
    if fault == 'rejected': row['accepted'] = False
    elif fault == 'unknown': row['accepted'] = None
    elif fault == 'missing': row['checks'].pop('pitch_absolute_quality')
    elif fault == 'false': row['checks']['roll_absolute_quality'] = False
    elif fault == 'quality': row['checks']['quality_complete_window'] = None
    else: row['tracking']['MAE'] = .3
    state = refine.initial([[0., 0.], [.5, 0.], [-.5, 0.]])
    state[1].update(mix=.5, consecutive_eligible=1)
    state, decisions = refine.advance(state, rows, policy)
    assert not decisions[1]['eligible']
    assert state[1]['mix'] == .5 and state[1]['consecutive_eligible'] == 0


@pytest.mark.parametrize('version', [None, 'unknown_assessment_v99'])
def test_unrecognized_versions_do_not_infer_schema_from_present_fields(version):
    policy, rows = evidence(version)
    state = refine.initial([[0., 0.], [.5, 0.], [-.5, 0.]])
    _, decisions = refine.advance(state, rows, policy)
    assert not decisions[1]['eligible']
    for record in rows:
        for row in record['per_env']:
            row['checks'].update(pitch_source_guard=True, roll_source_guard=True)
            row.pop('accepted')
    _, decisions = refine.advance(state, rows, policy)
    assert decisions[1]['eligible']  # Existing legacy input contract stays valid.


def test_three_DR_groups_do_not_hide_a_new_geometry_failure_from_refinement():
    policy, records = evidence()
    matrix = []
    for group in range(3):
        for record in records:
            row = copy.deepcopy(record)
            row['case']['_motion_dr'] = {'group': group}
            matrix.append(row)
    # Use one straight target and the real three-group aggregation.
    matrix[4]['per_env'][0]['accepted'] = False
    merged = dr.refinement_records(matrix, 0)
    record = next(r for r in merged if r['case']['command'] == [.5, 0., .34])
    _, decisions = refine.advance(refine.initial([[.5, 0.]]), [record], policy)
    assert not decisions[0]['eligible']


def test_legacy_reacquisition_records_keep_frozen_refinement_history():
    policy, records = evidence()
    for record in records:
        for row in record['per_env']:
            row.pop('refinement_evidence_version')
    state = refine.initial([[0., 0.], [.5, 0.], [-.5, 0.]])
    state, decisions = refine.advance(state, records, policy)
    assert [d['eligible'] for d in decisions] == [False, False, False]
    assert [s['mix'] for s in state] == [0., 0., 0.]


def test_reporting_uses_actual_course_owner_without_mutating_scoring():
    original = motion.protocol()
    before = copy.deepcopy(original)
    reported = dr.reporting_protocol(original)
    assert original == before and reported['confirmations_required'] == 1
    assert reported['course_controller']['version'] == dr.VERSION
    assert reported['course_controller']['behavior_chain_stop'] is False
    assert reported['course_controller']['DR_retreat'] is False
    assert reported['course_controller']['evaluation_every_updates'] == 250
    assert reported['absolute_posture'] == {
        key: original['absolute_posture'][key]
        for key in ('pitch_hold_abs_max_rad', 'roll_hold_abs_max_rad', 'relative_posture')}
    for key, value in original.items():
        if key not in ('confirmations_required', 'absolute_posture'):
            assert reported[key] == value
    assert motion.protocol()['confirmations_required'] == 2


def test_actual_score_arrays_reports_owner_without_changing_scores(monkeypatch):
    from native_stand.students import evaluator, motion_training
    monkeypatch.setattr(evaluator, 'model_contract', lambda *args: {})
    monkeypatch.setattr(motion_training, 'score', lambda *args: {
        'checks': {'survival30': True}, 'accepted': True, 'tracking': {}})
    env = S(motion_training_enabled=True, motion_absolute_protocol=motion.protocol())
    arrays = dict(first_episode_valid=np.ones((3000, 1), bool),
                  state=np.zeros((3000, 1, 39)), posture=np.zeros((3000, 1, 10)),
                  geometry=np.zeros((3000, 1, 9)), root_post=np.zeros((3000, 1, 13)),
                  policy_time_s=np.arange(3000) * .01,
                  controller=np.full((3000, 1, 1), 3),
                  termination_causes=np.zeros((3000, 1, 4)))
    arrays['state'][:, :, 29] = 1
    row = {'command': [0., 0., .34]}
    baseline, _ = evaluator.score_arrays(env, row, arrays, scored_envs=1)
    env.motion_dr = object()
    result, _ = evaluator.score_arrays(env, row, arrays, scored_envs=1)
    stripped = copy.deepcopy(result['per_env'])
    for item in stripped:
        assert item.pop('refinement_evidence_version') == 'motion31_absolute_refinement_evidence_v1'
    assert stripped == baseline['per_env']
    assert result['accepted'] == baseline['accepted']
    assert result['protocol']['confirmations_required'] == 1
    assert 'course_controller' in result['protocol']
    assert env.motion_absolute_protocol['confirmations_required'] == 2
