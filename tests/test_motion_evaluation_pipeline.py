"""PORTABLE: planning and asynchronous evidence identity, no simulation."""
FROZEN_TEST_SCOPE = 'PORTABLE'
import copy
import pytest
from native_stand.students.motion_evaluation_pipeline import plan, snapshot, result_decision


def cases(n=29):
    return [dict(case_id=f'case{i}', kind='student', role='MotionZeroStudent',
                 family='motion_native_normal', seed=17, yaw_rad=0., seconds=31.52,
                 checkpoint='/model.pt', sha256='a'*64, command=[0., i/100, .34],
                 _identity_key=[i, 'physical_profile']) for i in range(n)]


def request():
    return snapshot(cases(), checkpoint='/model.pt', checkpoint_sha256='a'*64,
                    spec_sha256='b'*64, course_revision=3, update=1500)


def response(r):
    return {**{k:r[k] for k in ('snapshot_sha256', 'checkpoint_sha256', 'spec_sha256',
                              'course_revision', 'update')},
            'complete_matrix':True, 'online_PPO':0, 'offline_optimizer':0,
            'records':[dict(case=case, per_env=[dict(env=i, accepted=True) for i in range(8)])
                       for case in cases()]}


def test_29_cases_fit_one_batch_without_removing_windows_or_envs():
    original=cases(); before=copy.deepcopy(original); p=plan(original)
    assert p['legacy_batch_count']==4 and p['proposed_batch_count']==1
    assert p['scored_episodes']==232 and p['batches'][0]['policy_steps']==3152
    assert p['batches'][0]['estimated_score_buffer_bytes']<512*2**20
    assert original==before and p['simulation']==0 and not p['runtime_recipe_ready']


def test_detailed_legacy_capture_cannot_just_expand_capacity():
    with pytest.raises(ValueError, match='byte limit'):
        plan(cases(),float_equivalents=640)


def test_different_seed_duration_and_construction_do_not_merge():
    c=cases(4);c[1]['seed']=18;c[2]['seconds']=32.52;c[3]['_mcu_variant']=1
    assert plan(c)['proposed_batch_count']==4


def test_larger_matrix_is_bounded_and_cases_are_not_dropped():
    p=plan(cases(40));assert p['proposed_batch_count']==2
    assert sum(b['scored_envs'] for b in p['batches'])==320


def test_complete_late_result_only_belongs_to_checkpoint():
    r=request(); d=result_decision(r,response(r),current_revision=3,current_update=1530)
    assert d['evidence_valid'] and d['eligible_for_existing_course_checks']
    assert not d['certifies_current_weights'] and not d['course_advanced']


@pytest.mark.parametrize('revision,update',[(4,1530),(3,1601),(3,1499)])
def test_old_revision_excessive_lag_or_future_result_cannot_promote(revision,update):
    r=request();d=result_decision(r,response(r),current_revision=revision,current_update=update)
    assert d['evidence_valid'] and d['stale_for_curriculum']
    assert not d['eligible_for_existing_course_checks']


@pytest.mark.parametrize('change', ['sha','missing_case','partial','seven_envs','duplicate_env',
                                  'bool_env','case_sha','case_path','key','optimizer','bool_zero'])
def test_bad_results_never_reach_course_checks(change):
    r=request(); out=response(r)
    if change=='sha':out['checkpoint_sha256']='c'*64
    if change=='missing_case':out['records'].pop()
    if change=='partial':out['complete_matrix']=False
    if change=='seven_envs':out['records'][0]['per_env'].pop()
    if change=='duplicate_env':out['records'][0]['per_env'][0]['env']=1
    if change=='bool_env':out['records'][0]['per_env'][0]['env']=False
    if change=='case_sha':out['records'][0]['case']['sha256']='c'*64
    if change=='case_path':out['records'][0]['case']['checkpoint']='/other.pt'
    if change=='key':out['records'][0]['case']['_identity_key'][0]=999
    if change=='optimizer':out['online_PPO']=1
    if change=='bool_zero':out['online_PPO']=False
    d=result_decision(r,out,current_revision=3,current_update=1530)
    assert not d['evidence_valid'] and not d['eligible_for_existing_course_checks']


def test_snapshot_is_independent_and_tampering_rejected():
    r=request(); out=response(r);r['assessment_keys'][0][0]=999
    d=result_decision(r,out,current_revision=3,current_update=1530)
    assert 'request_snapshot_changed' in d['errors']


def test_duplicate_identity_is_not_more_evidence():
    c=cases(2);c[1]['_identity_key']=c[0]['_identity_key']
    with pytest.raises(ValueError,match='duplicate assessment'):
        snapshot(c,checkpoint='/model.pt',checkpoint_sha256='a'*64,
                 spec_sha256='b'*64,course_revision=3,update=1500)
