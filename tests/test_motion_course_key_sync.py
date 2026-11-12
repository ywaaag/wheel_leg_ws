FROZEN_TEST_SCOPE = 'PORTABLE'

import copy
import json
from types import SimpleNamespace
import pytest
from native_stand.students import motion_fixed_height as course
from native_stand.students.motion_runner import advance_course_boundary
from native_stand.students.motion_course_keys_resume import repaired_state


def fixture():
    state = dict(version=course.VERSION, levels=dict(forward=3,backward=3,yaw_positive=3,yaw_negative=3,height=-1,DR=1),
        focus='height',focus_level=0,stage=0,focus_cursor=0,stage_started_update=0,last_evaluation_update=-1,
        DR_level=1,certified_DR_level=1,accepted_by_stage={},regression_streaks={},repair_queue=[])
    spec = dict(motion_decoupled=dict(state=state),motion_robustness=dict(state=state),motion_training={},
        student_training=dict(course_version=course.VERSION),curriculum={},ppo=dict(envs=4096),motion_timing={},
        student_evaluation=dict(cases=[dict(command=[0.,0.,.34],seed=1,seconds=31.52)]))
    spec = course.effective_spec(spec,0,state)
    return spec, SimpleNamespace(motion_course_state=spec['motion_decoupled']['state'])


def candidate(spec,update):
    return dict(checkpoint=f'/model_update_{update}.pt',sha256='a'*64,update=update,
        stage=spec['motion_decoupled']['state']['stage'],complete_matrix=True,
        evaluations=[dict(case=copy.deepcopy(c),per_env=[dict(env=i,accepted=True) for i in range(8)])
                     for c in spec['student_evaluation']['cases']])


def test_actual_runner_boundary_rebinds_keys_before_progress_and_checkpoint():
    spec,ppo=fixture();old=copy.deepcopy(spec);c=candidate(spec,250)
    for r in c['evaluations']:
        if r['case']['_focus_required']:r['per_env'][0]['accepted']=False
    spec,state,d=advance_course_boundary(spec,ppo,c)
    assert old['motion_decoupled']['state']['assessment_keys']!=state['assessment_keys']
    assert len(state['assessment_keys'])==35 and state['focus']=='DR' and d['evidence_valid']
    assert state is ppo.motion_course_state and state is spec['motion_decoupled']['state']
    assert state is spec['motion_robustness']['state'] and state is spec['student_training']['decoupled_state']
    # JSON persistence and a subsequent evaluation follow the exact owner path.
    persisted=json.loads(json.dumps(dict(state=state)))
    ppo.motion_course_state=persisted['state'];spec,state,d=advance_course_boundary(spec,ppo,candidate(spec,500))
    assert d['stage_passed'] and d['evidence_valid'] and state['levels']['DR']==2
    assert state['focus']=='forward' and len(state['assessment_keys'])==18
    assert state['assessment_keys']==[r['_identity_key'] for r in spec['student_evaluation']['cases']]


def test_repeated_focus_changes_do_not_freeze_regression_or_promotion():
    spec,ppo=fixture()
    for update in range(250,1751,250):
        spec,state,d=advance_course_boundary(spec,ppo,candidate(spec,update))
        assert d['evidence_valid'] and d['stage_passed']
        assert ppo.motion_course_state['assessment_keys']==[r['_identity_key'] for r in spec['student_evaluation']['cases']]
    assert state['levels']['DR']>=3 and state['levels']['forward']>=4


@pytest.mark.parametrize('mode',['missing','wrong_group','unknown','wrong_height','wrong_revision'])
def test_boundary_fix_does_not_accept_wrong_or_incomplete_evidence(mode):
    spec,ppo=fixture();c=candidate(spec,250)
    if mode=='missing':c['evaluations'].pop()
    elif mode=='unknown':c['evaluations'][-1]['per_env'][0]['accepted']=None
    elif mode=='wrong_revision':c['evaluations'][0]['case']['_course_revision']=9
    else:c['evaluations'][0]['case']['_identity_key'][5 if mode=='wrong_group' else 2]=9
    if mode=='wrong_revision':
        with pytest.raises(ValueError):advance_course_boundary(spec,ppo,c)
    else:
        _,state,d=advance_course_boundary(spec,ppo,c)
        assert not d['evidence_valid'] and not d['stage_passed'] and state['stage']==0


def test_historical_stale_boundary_is_repaired_once_without_mutating_checkpoint():
    original,ppo=fixture();c=candidate(original,250)
    for r in c['evaluations']:
        if r['case']['_focus_required']:r['per_env'][0]['accepted']=False
    stale,_=course.advance(ppo.motion_course_state,c)
    active=course.effective_spec(original,stale['stage'],stale);passed=candidate(active,750)
    stale['last_evaluation_update']=750
    source=dict(update=822,motion_course_state=stale);before=copy.deepcopy(source)
    state,decision=repaired_state(original,source,passed)
    assert source==before and decision['evidence_valid'] and decision['stage_passed']
    assert state['levels']['DR']==2 and state['last_evaluation_update']==-1 and state['stage_started_update']==0
    assert state['accepted_by_stage']['1']['update']==750 and state['focus']=='forward'
    with pytest.raises(ValueError):repaired_state(original,dict(update=822,motion_course_state=state),passed)
