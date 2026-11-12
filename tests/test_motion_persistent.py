FROZEN_TEST_SCOPE = 'PORTABLE'

import copy
from types import SimpleNamespace
import pytest
from native_stand.students import motion_persistent as course, motion_fixed_height as old
from native_stand.students.motion_runner import advance_course_boundary


def fixture(dr_level=5):
    previous=dict(version=old.VERSION,levels=dict(forward=3,backward=4,yaw_positive=4,yaw_negative=3,height=-1,DR=dr_level),
        stage=8,focus='yaw_negative',focus_level=4,stage_started_update=1500,last_evaluation_update=1500,
        accepted_by_stage={},regression_streaks={},repair_queue=[])
    state=course.initial_state(previous)
    s=dict(motion_decoupled=dict(state=state),motion_robustness=dict(state=state),motion_training={},
        student_training=dict(course_version=course.VERSION),curriculum={},ppo=dict(envs=4096),motion_timing={},
        student_evaluation=dict(cases=[dict(command=[0.,0.,.34],seed=1,seconds=31.52)]))
    s=course.effective_spec(s,state['stage'],state)
    return s,SimpleNamespace(motion_course_state=s['motion_decoupled']['state'])


def candidate(s,update,fail=()):
    rows=[]
    for c in s['student_evaluation']['cases']:
        ok=not(set(c['_frontier_axes'])&set(fail))
        rows.append(dict(case=copy.deepcopy(c),per_env=[dict(env=i,accepted=ok) for i in range(8)]))
    return dict(checkpoint=f'/model{update}.pt',sha256='a'*64,update=update,
        stage=s['motion_decoupled']['state']['stage'],complete_matrix=True,evaluations=rows)


def test_all_frontiers_have_nonzero_fixed_quotas_and_no_combined_commands():
    s,_=fixture();stage=s['curriculum']['stages'][0]
    assert len(stage['sampling_slots'])==sum(stage['slot_counts'])==4096
    assert stage['frontiers']==dict(forward=4,backward=5,yaw_positive=5,yaw_negative=4,height=0)
    for axis in stage['frontiers']:
        assert sum(n for a,p,n in zip(stage['cohort_axes'],stage['cohort_purposes'],stage['slot_counts']) if a==axis and p=='frontier')>=409
    assert all(not(vx and yaw) for vx,yaw in stage['commands'])
    assert all(x[2]==x[3] and x[4] is False for x in stage['height_scenarios'])
    assert len(stage['group_slots'])==len(stage['DR_level_slots'])==4096


def test_failed_forward_is_never_removed_while_other_axes_promote():
    s,ppo=fixture();frozen=copy.deepcopy(s)
    for update in (250,500,750):
        s,state,d=advance_course_boundary(s,ppo,candidate(s,update,fail=['forward']))
        assert d['evidence_valid'] and state['levels']['forward']==3
        stage=s['curriculum']['stages'][0]
        assert sum(n for v,n in zip(stage['commands'],stage['slot_counts']) if v==[3.,0.])>0
        assert state is ppo.motion_course_state and state is s['student_training']['decoupled_state']
        assert state['assessment_keys']==[r['_identity_key'] for r in s['student_evaluation']['cases']]
    assert frozen['motion_decoupled']['state']['levels']['backward']==4
    assert state['levels']['backward']==5 and state['levels']['yaw_positive']==5


def test_new_dr_applies_only_to_mastered_tasks_not_new_speed_or_height():
    s,_=fixture(dr_level=2);state=s['motion_decoupled']['state'];rows=s['student_evaluation']['cases']
    for r in rows:
        a=r['_frontier_axes'];p=r['_motion_dr']
        if 'DR' in a:assert p==dict(level=3,group=2,seed=20261024+state['stage']) and r['command'][:2] in [[0.,0.],[.5,0.],[1.,0.],[1.5,0.],[2.,0.],[-.5,0.],[-1.,0.],[-1.5,0.],[-2.,0.],[-3.,0.],[0.,.5],[0.,1.],[0.,1.5],[0.,2.],[0.,3.],[0.,-.5],[0.,-1.],[0.,-1.5],[0.,-2.]]
        elif 'height' in a:assert p['level']==2 and p['group']==0
        elif a:assert p['level']==2 and p['group']==2


@pytest.mark.parametrize('mode',['missing','unknown','wrong_group','wrong_height','wrong_revision','wrong_roles'])
def test_missing_or_wrong_evidence_never_promotes(mode):
    s,ppo=fixture();c=candidate(s,250)
    if mode=='missing':c['evaluations'].pop()
    elif mode=='unknown':c['evaluations'][0]['per_env'][0]['accepted']=None
    elif mode=='wrong_group':c['evaluations'][0]['case']['_identity_key'][-1]=2
    elif mode=='wrong_height':c['evaluations'][0]['case']['_identity_key'][2]=.4
    elif mode=='wrong_revision':c['evaluations'][0]['case']['_course_revision']+=1
    else:c['evaluations'][0]['case']['_frontier_axes']=['forward']
    _,state,d=advance_course_boundary(s,ppo,c)
    assert not d['evidence_valid'] and not d['promoted_axes'] and state['levels']==s['motion_decoupled']['state']['levels']


def test_two_known_failures_lower_only_affected_axis_and_keep_learning_target():
    s,ppo=fixture();initial=s['motion_decoupled']['state']['levels'].copy()
    for update in (250,500):
        c=candidate(s,update,fail=list(course.frontiers(s['motion_decoupled']['state'])))
        for r in c['evaluations']:
            if 'forward' in r['case']['_retention_axes']:
                for e in r['per_env']:e['accepted']=False
        s,st,d=advance_course_boundary(s,ppo,c)
    assert d['lowered_tasks']==['forward'] and st['levels']['forward']==2
    assert {k:v for k,v in st['levels'].items() if k!='forward'}=={k:v for k,v in initial.items() if k!='forward'}
    assert course.frontiers(st)['forward']==3


def test_all_pass_finishes_axes_without_changing_scoring_or_old_default():
    s,ppo=fixture(dr_level=1)
    for update in range(250,1501,250):
        s,st,d=advance_course_boundary(s,ppo,candidate(s,update))
    assert d['all_stages_passed'] and st['levels']==dict(forward=5,backward=5,yaw_positive=5,yaw_negative=5,height=4,DR=5)
    assert course.protocol()==old.protocol() and course.network()==old.network() and course.contract()==old.contract()
    assert old.configuration()['version']==old.VERSION


def test_actor_critic_and_algorithm_declarations_are_unchanged_in_cpu_plan():
    from native_stand.students.motion_course import implementation
    assert implementation(course.VERSION) is course
    assert course.configuration()['no_combined_vx_yaw'] and course.configuration()['no_moving_height']


def test_height_bonus_only_doubles_height_frontier_with_single_dt_and_no_event_changes():
    import torch
    raw=torch.tensor([[1.],[.5],[0.]])
    result=dict(fields=('height',),raw=raw,density=raw*2,dt=raw*.02,
                total=raw[:,0]*.02-1,events=torch.full((3,3),-1.),eligible=torch.tensor([True,True,False]))
    bonus=course.apply_height_bonus(result,torch.tensor([True,False,True]))
    assert bonus['fields']==('height','height_frontier_bonus')
    assert torch.allclose(bonus['total'],torch.tensor([-.96,-.99,-1.]))
    assert torch.equal(bonus['events'],result['events'])
    assert torch.equal(bonus['raw'][:,-1],torch.tensor([1.,0.,0.]))
    assert result['fields']==('height',) and result['total'][0].item()==pytest.approx(-.98)


def test_height_bonus_curve_has_correct_direction_and_no_clipping():
    snap=course.height_reward_snapshot()
    for r in snap['curves']:
        assert r['height_frontier_density']==pytest.approx(2*r['base_density'])
        assert r['dt_contribution']==pytest.approx(.01*r['height_frontier_density'])
        if r['error_m']:assert r['error_m']*r['derivative_per_m']<0
    assert snap['total_clip'] is None and not snap['convergence_guaranteed']
