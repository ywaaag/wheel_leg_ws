FROZEN_TEST_SCOPE = 'PORTABLE'
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from native_stand.students.motion_cross_engine import cases,scenario,score,ranking,free_velocity
from native_stand.students.motion_height_pd import initial_state


def fixture_spec():
    return {'motion_height_dr':{'state':initial_state()},'source_checkpoint':'source.pt','source_sha256':'a'*64,
        'student_evaluation':{'cases':[{'kind':'normal_motion_baseline','role':'MotionZeroStudent',
            'checkpoint':'source.pt','sha256':'a'*64,'command':[0.,0.,.34],
            'family':'motion_native_normal','seed':104,'yaw_rad':0.,'seconds':31.52}]}}


def trace(vx=0.,yaw=0.,height=.34,seconds=30.):
    n=round(seconds*100);zero=lambda *tail:np.zeros((n,*tail))
    state=zero(7);state[:,0]=vx;state[:,2]=yaw;state[:,3]=height
    lengths=zero(2)+.34
    return dict(time=np.arange(1,n+1)*.01,state=state,native_posture=zero(2),root_velocity_w=zero(3),
        lengths=lengths,heading_offset=zero(),swing_difference=zero(),contact_resultant=zero(2)+30,
        motor=zero(6),driver_cap=zero(6)+5,driver=zero(6),requested_target=zero(6),adopted_target=zero(6),
        dq=zero(6),support_velocity_xy=zero(2,2),rolling_geometry_valid=np.ones((n,2),dtype=bool),
        undesired_ground_contact=zero(),rolling=zero(2),differential_target=zero())


def test_matrix_is_37_original_plus20_heights_plus3_endpoints_and_does_not_mutate():
    spec=fixture_spec();before=copy.deepcopy(spec);rows=cases(spec)
    assert spec==before and len(rows)==60 and len({r['case_id'] for r in rows})==60
    assert sum(r['group']=='nominal_old' for r in rows)==22
    assert sum(r['group']=='height_motion_probe' for r in rows)==20
    assert [r['command'] for r in rows[-3:]]==[[4.,0.,.34],[-4.,0.,.34],[0.,-4.,.34]]
    assert {r['command'][2] for r in rows if r['_mcu_variant']==4}=={.30,.32,.36,.38,.40}
    assert 'checkpoint' not in rows[0] and rows[0]['_template_source']['checkpoint']=='source.pt'


def test_world_angular_velocity_is_mapped_to_local_free_joint_coordinates():
    rotation=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
    v=np.array([.05,0.,0.,0.,.1,0.]);free=free_velocity(v,rotation)
    np.testing.assert_allclose(free[:3],v[:3],atol=0)
    np.testing.assert_allclose(rotation@free[3:],v[3:],atol=1e-12)
    assert free[5]==-.1


def test_world_velocity_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError,match='velocity6'):free_velocity([0.]*5,np.eye(3))
    with pytest.raises(ValueError,match='velocity6'):free_velocity([np.nan]*6,np.eye(3))


@pytest.mark.parametrize('variant,age,delay,spring',[(0,10,7,1.),(1,10,7,1.),(2,12,9,1.1)])
def test_world_initial_velocity_and_post_pd_timing(variant,age,delay,spring):
    row=dict(case_id='x',_mcu_variant=variant)
    u,v,p=scenario(row)
    assert (u['PD_age_ms'],u['motor_delay_ms'][2],u['spring_scale'])==(age,delay,spring)
    assert v==([0.]*6 if variant==0 else [.05,0.,0.,0.,.1,0.])
    assert p==0


def test_prefix_and_pd_age_preserve_distinct_zero_conditions():
    row=dict(case_id='x',_mcu_variant=3,_execution_uncertainty={'PD_age_ms':20,'prefix_intervals':100})
    u,v,p=scenario(row)
    assert u['PD_age_ms']==20 and u['motor_delay_ms']==[10,10,7,10,10,11] and p==1.


@pytest.mark.parametrize('height',[.30,.32,.34,.36,.38,.40])
def test_height_score_uses_requested_reference_not_034(height):
    row=dict(command=[0.,0.,height]);a=trace(height=height)
    result=score(a,row)
    assert result['accepted'] and result['metrics']['height_MAE_m']<1e-12
    a['state'][:,3]+=.025
    assert not score(a,row)['checks']['height']


def test_yaw_axis_is_angular_rate_not_vx_and_equal_opposite_is_measured():
    a=trace(vx=0.,yaw=-.5);a['rolling'][:,0]=-.1;a['rolling'][:,1]=.1;a['differential_target'][:]=-.1
    r=score(a,{'command':[0.,-.5,.34]})
    assert r['accepted'] and r['metrics']['gain']==1.
    a['state'][:,2]=+.5
    bad=score(a,{'command':[0.,-.5,.34]})
    assert not bad['checks']['direction'] and bad['metrics']['gain']==-1.


def test_rolling_common_and_translation_are_separate_failures():
    a=trace(yaw=.5);a['rolling'][:]=[.2,0.];a['differential_target'][:]=.1
    r=score(a,{'command':[0.,.5,.34]})
    assert r['checks']['in_place_translation'] and not r['checks']['spin_common']
    assert not r['checks']['opposite_rolling']


def test_incomplete_fall_or_missing_samples_never_pass():
    assert not score(trace(seconds=2.35),{'command':[0.,0.,.34]},termination='FALL')['accepted']
    a=trace();a['time'][50]+=.001
    assert not score(a,{'command':[0.,0.,.34]})['full_window_completed']
    a=trace();a['state'][400,0]=np.nan
    assert not score(a,{'command':[0.,0.,.34]})['accepted']


def test_prefix_excluded_from_actual30second_scoring_window():
    a=trace(seconds=31.);a['state'][:100,4]=1.;a['native_posture'][:100,0]=1.
    assert score(a,{'command':[0.,0.,.34]},prefix=1.)['accepted']


def test_native_posture_swing_loss_contact_and_motor_limits_not_hidden():
    a=trace();a['native_posture'][1000:1800,1]=.06;a['swing_difference'][5]=.05
    a['contact_resultant'][1000:1800,0]=0.;a['motor'][:,2]=5.
    r=score(a,{'command':[0.,0.,.34]})
    assert {'roll','swing_full','contact','motor_limit'}<=set(r['failed_checks'])
    assert r['metrics']['longest_loss_contact_s'][0]==8.


def test_incomplete_model_cannot_win_ranking():
    rows=[]
    for name,n in [('complete',60),('partial',59)]:
        for i in range(n):rows.append(dict(model=name,case={'group':'nominal_old' if i<22 else 'x'},
            status='COMPLETED',termination='WINDOW_COMPLETE',assessment={'accepted':name=='partial'}))
    assert ranking(rows,['partial','complete'])[0]['model']=='complete'


def test_prefix_inference_audit_uses_actual_owner():
    module_path=Path(__file__).parents[1]/'scripts/validate_motion_cross_engine.py'
    spec=importlib.util.spec_from_file_location('cross_engine_cli',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Actor:
        def __init__(self,v):self.v=v
        def mean(self,x):return torch.full((len(x),6),self.v)
    a={'actor_input':np.zeros((3,31)),'mean':np.array([[1.]*6,[2.]*6,[2.]*6]),
       'owner':np.array([0,1,1]),'adopted_target':np.zeros((3,6))}
    assert module.inspect_trace(a,Actor(2.),1.,teacher=Actor(1.))['actor_error']==0.


def test_zero_velocity_recovery_preserves_evidence_and_charges_interrupted_tail(tmp_path):
    import json
    from native_stand.students.data import sha
    module_path=Path(__file__).parents[1]/'scripts/validate_motion_cross_engine.py'
    spec=importlib.util.spec_from_file_location('cross_engine_recovery',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Actor:
        def mean(self,x):return torch.zeros(len(x),6)
    a=trace();a.update(actor_input=np.zeros((3000,31)),mean=np.zeros((3000,6)),owner=np.ones(3000,dtype=int))
    trace_path=tmp_path/'trace.npz';np.savez_compressed(trace_path,**a)
    case=dict(case_id='zero',command=[0.,0.,.34],_mcu_variant=0,group='nominal_old')
    row=dict(model='source4500',case=case,status='COMPLETED',termination='WINDOW_COMPLETE',
             assessment=score(a,case),trace={'path':'trace.npz','sha256':sha(trace_path)})
    models=[{'name':'source4500'}]
    terminal=tmp_path/'terminal_report.json'
    terminal.write_text(json.dumps(dict(status='STOPPED_BOUNDED',models_unchanged=True,results=[row],physical_seconds=31.25,wall_seconds=20.)))
    (tmp_path/'manifest.json').write_text(json.dumps({'models':models}))
    (tmp_path/'budget.json').write_text(json.dumps({'attempts':{'first':{'status':'COMPLETED','settled':True},'partial':{'status':'FAILED','settled':True}}}))
    request={'models':models,'recovery':{'version':'motion31_mujoco_zero_velocity_resume_v1',
        'parent_output':str(tmp_path),'parent_terminal_sha256':sha(terminal)}}
    result,inputs,lo,hi,wall=module.reusable_records(request,[case],{'source4500':Actor()})
    assert len(result)==1 and (lo,hi,wall)==(31.25,31.26,20.)
    assert result[0]['tested_model']==models[0] and result[0]['reused_without_simulation']
    assert str(trace_path.resolve()) in inputs
    case['_mcu_variant']=1
    with pytest.raises(ValueError,match='zero-world-velocity'):module.reusable_records(request,[case],{'source4500':Actor()})
