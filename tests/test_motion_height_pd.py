"""Real course boundaries: identity, task isolation, reward and reset histories."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy
from types import SimpleNamespace

import pytest
import torch

from native_stand.students import motion_height_pd as course


def spec_fixture():
    template=dict(kind='student',role='MotionZeroStudent',checkpoint='x',sha256='a'*64,
        family='motion_native_normal',seed=1,yaw_rad=0.,seconds=31.52)
    return dict(motion_height_dr=dict(state=course.initial_state()),motion_training={},
        student_training={},motion_course={},ppo=dict(envs=4096),curriculum={},
        student_evaluation=dict(cases=[template]))


def matrix(spec,*,fail_pd=False,fail_height=False):
    es=lambda good:[dict(env=i,accepted=good,survival_passed=True,controlled_seconds=30.) for i in range(8)]
    records=[dict(case=copy.deepcopy(r),per_env=es(not((fail_pd and r.get('_pd_focus')) or
        (fail_height and r.get('_height_focus'))))) for r in spec['student_evaluation']['cases']]
    return dict(stage=spec['motion_training']['stage'],update=500,checkpoint='candidate',sha256='b'*64,
        complete_matrix=True,evaluations=records,per_env=[e for r in records for e in r['per_env']])


def test_initial_quotas_and_no_strong_delay_at_new_heights():
    x=course.stage_config();assert sum(x['slot_counts'])==4096
    totals={}
    for p,n in zip(x['cohort_purposes'],x['slot_counts']):totals[p]=totals.get(p,0)+n
    assert totals=={'PD_zero':205,'PD_motion':819,'height':1024,'nominal_zero':614,'nominal_motion':1434}
    assert x['height_targets_m'][5:7]==[.32,.36]
    assert all(h==.34 for h,v in zip(x['height_targets_m'],x['mcu_variants']) if v==3)
    assert all(p==0 for p,v in zip(x['prefix_intervals'],x['mcu_variants']) if v==4)
    assert sum(course.stage_config(envs=64)['slot_counts'])==64


@pytest.mark.parametrize('failed',('pd','height'))
def test_independent_promotion_and_no_failure_rotation(failed):
    s=course.effective_spec(spec_fixture(),0)
    state,decision=course.advance(s['motion_height_dr']['state'],matrix(s,fail_pd=failed=='pd',fail_height=failed=='height'))
    assert state[failed+'_level']==0
    other='height' if failed=='pd' else 'pd'
    assert state[other+'_level']==1 and decision['advanced'] and not decision['stop']
    new=course.effective_spec(s,state['stage'],state)
    assert new['motion_height_dr']['state']['assessment_keys']!=s['motion_height_dr']['state']['assessment_keys']
    assert new['student_training']['height_dr_state']['assessment_keys']==new['motion_height_dr']['state']['assessment_keys']
    old=matrix(s);old.update(stage=state['stage'],update=1000)
    _,decision=course.advance(new['motion_height_dr']['state'],old)
    assert not decision['evidence_valid']


def test_no_promotion_on_incomplete_or_unknown_environment_evidence():
    s=course.effective_spec(spec_fixture(),0);candidate=matrix(s)
    candidate['evaluations'][-1]['per_env'][0]['accepted']=None
    state,decision=course.advance(s['motion_height_dr']['state'],candidate)
    assert state['height_level']==0
    candidate=matrix(s);candidate['evaluations'].pop()
    _,decision=course.advance(s['motion_height_dr']['state'],candidate)
    assert not decision['evidence_valid'] and not decision['advanced']


def test_height_references_do_not_reward_return_to034():
    from native_stand.students.motion_height import correct_reward
    fields=('height','transfer_vertical_velocity')
    raw=torch.zeros(3,2);result=dict(fields=fields,raw=raw,density=raw.clone(),dt=raw.clone(),total=torch.zeros(3),eligible=torch.ones(3,dtype=torch.bool))
    actual=torch.tensor([.32,.34,.36]);cmd=torch.tensor([.32,.32,.36])
    r=correct_reward(result,actual,cmd,torch.zeros(3),torch.zeros(3))
    assert r['density'][0,0]==r['density'][2,0]==2
    assert r['density'][1,0]<r['density'][0,0]
    assert torch.equal(result['density'],torch.zeros(3,2))


def test_retention_excludes_height_even_if_variant_metadata_is_wrong():
    from native_stand.students.motion_source_retention import add_metadata,KEY
    from native_stand.recovery.controlled_ppo import KEY as VALID
    obs={VALID:torch.ones(3,1,dtype=torch.bool),'student_actor':torch.zeros(3,31)}
    obs['student_actor'][:,24]=torch.tensor([.68,.64,.72])
    out=add_metadata(obs,torch.zeros(3,2),course.retention(course.handover.SOURCE_SHA),variants=torch.zeros(3,dtype=torch.long))
    assert out[KEY].flatten().tolist()==[0,-1,-1]


def test_width_records_follow_actual_duplicate_command_cohorts():
    s=course.effective_spec(spec_fixture(),0);candidate=matrix(s)
    records=course.refinement_records(candidate['evaluations'],0)
    assert len(records)==len(s['curriculum']['stages'][0]['commands'])
    for row,cmd in zip(records,s['curriculum']['stages'][0]['commands']):assert row['case']['command'][:2]==cmd


def test_existing_recipe_and_hold_remain_separate():
    from native_stand.students.motion_handover_robustness import configuration
    assert configuration()['formal_updates_max']==1000
    assert course.configuration()['formal_updates_max']==5124
    assert 5+4865+1+course.ONLINE==10000


def test_new10000_authorization_keeps_default_remainder_and_exact_cadence():
    recipe=course.recipe(course.handover.SOURCE_SHA,10000)
    assert recipe['authorized_online_limit']==10000
    assert recipe['milestones']==[500,*range(1000,10000,1000),9995]
    assert course.configuration(10000)['formal_updates_max']==9995
    assert course.validate_recipe(recipe,course.handover.SOURCE_SHA) is recipe
    assert course.configuration()['formal_updates_max']==5124
    with pytest.raises(ValueError):course.configuration(10001)


def test_case_checkpoint_identity_uses_selected_source_not_inherited_template():
    s=spec_fixture();s.update(source_checkpoint='selected4500.pt',source_sha256='b'*64)
    rows=course.cases(s)
    assert len(rows)==34
    assert all(r['checkpoint']=='selected4500.pt' and r['sha256']=='b'*64 for r in rows)


def test_actual_density_slopes_point_toward_height_and_symmetry_targets():
    result=course.learning_signal_audit(dict(source_sha256='b'*64))
    assert result['status']=='PASSED' and result['optimizer_updates']==0
    assert {r['target_height_m'] for r in result['curves'] if r['term']=='height'}==set(course.HEIGHTS)
    for r in result['curves']:
        if r['error']:assert r['derivative_per_physical_error']*r['error']<0
        assert r['dt_contribution']==pytest.approx(r['weighted_density']*.01)
    assert len(result['teacher_eligible_cohort_indices'])==22


def test_qualified_source_subset_does_not_accept_unmeasured_new_tasks():
    candidate=dict(update=0,stage=0,complete_matrix=False,per_env=[dict(accepted=True)]*192)
    selection=course.choose([candidate],{})
    assert selection['source'] is candidate
    assert selection['accepted_candidate'] is None and selection['best_new'] is None


def test_source_reuse_keeps_new_height_and_age_out_and_rejects_plant_drift(monkeypatch):
    s=course.effective_spec(spec_fixture(),0)
    pose=dict(height_m=.34,joint_pos={'hip':.2})
    s.update(source_checkpoint='source',mcu_execution=dict(pose=pose),action={},control={'kp':40},ground={},asset_sha256='x')
    s['motion_height_dr'].update(original_spec='old',height_poses=[pose])
    old=copy.deepcopy(s)
    records=[dict(case=copy.deepcopy(row),path='original.json',sha256='a'*64,per_env=[])
             for row in s['student_evaluation']['cases'][:24]]
    monkeypatch.setattr(course.handover,'source_records',lambda *_:(old,records,{}))
    reused,missing=course.startup_evidence(s)
    assert len(reused)==24 and len(missing)==10
    assert all(row.get('_execution_uncertainty') or row.get('_height') for row in missing)
    assert all(row['reuse_reference']['original_case']==source['case'] for row,source in zip(reused,records))
    s['control']['kp']=41
    with pytest.raises(ValueError,match='plant differs'):course.startup_evidence(s)


def test_source_comparison_follows_physical_condition_after_course_revision():
    from native_stand.students.training import course_baseline
    s=course.effective_spec(spec_fixture(),0)
    s.update(mcu_execution=True,motion_execution_robustness=True)
    old=s['student_evaluation']['cases'][0]
    reference=dict(case=old,per_env=[dict(accepted=True)])
    revised=dict(old,_mcu_revision=4)
    assert course_baseline(s,revised,[reference]) is reference
    changed=dict(revised,command=[0.,0.,.32],_height=dict(initial_m=.32,target_m=.32,moving=False))
    assert course_baseline(s,changed,[reference]) is None
