"""Portable causal execution and curriculum tests; no simulator required."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy
import torch
import pytest
from native_stand.students.motion_mcu_execution import MotorRing,pd_and_motor,profile,spring_compensation,WHEEL_CAP
from native_stand.students import motion_mcu_course as course


def test_motor_delays_are_post_pd_and_independent_per_wheel():
    r=MotorRing(2,'cpu');r.reset(torch.arange(2),torch.zeros(2,6))
    motor=torch.ones(2,6)
    outputs=torch.stack([r.step(t,motor).clone() for t in range(20)])
    assert not outputs[:7,:,2].any() and torch.equal(outputs[7:,:,2],torch.ones(13,2))
    assert not outputs[:11,:,5].any() and torch.equal(outputs[11:,:,5],torch.ones(9,2))
    assert outputs[10,0,0]==pytest.approx(1-torch.exp(torch.tensor(-.2)).item())
    with pytest.raises(ValueError,match='exactly'):r.step(21,motor)


def test_partial_reset_retains_other_motor_history():
    r=MotorRing(2,'cpu');r.reset(torch.arange(2),torch.zeros(2,6))
    for t in range(15):r.step(t,torch.ones(2,6))
    other=r.actual[1].clone();history=r.history[:,1].clone()
    r.reset(torch.tensor([0]),torch.full((1,6),.2))
    assert torch.equal(r.actual[1],other) and torch.equal(r.history[:,1],history)
    assert torch.equal(r.history[:,0],torch.full((64,6),.2))


def test_two_caps_and_spring_compensation_are_separate():
    q=torch.zeros(2,6);q[:,[0,1,3,4]]=torch.tensor([.21648856,.21663799,-.21648856,.21663799])
    dq=torch.zeros_like(q);target=q.clone();target[:,2]=100;target[:,5]=-100
    request,limited,comp,motor,valid=pd_and_motor(q,dq,target)
    assert valid.all() and torch.equal(request[:,2],torch.full((2,),20.))
    assert torch.equal(limited[:,2],torch.full((2,),5.5))
    assert torch.allclose(motor[:,2],torch.full((2,),WHEEL_CAP))
    assert torch.allclose(motor[:,5],torch.full((2,),-WHEEL_CAP))
    assert comp[0,0]>0 and comp[0,1]>0 and comp[0,3]<0 and comp[0,4]>0
    assert torch.allclose((motor-comp)[:,[0,1,3,4]],torch.zeros(2,4))


@pytest.mark.parametrize('sign',[-1.,1.])
def test_runtime_combines_35nm_motor_and_independent_15nm_spring_without_40nm_reclip(monkeypatch,sign):
    from types import SimpleNamespace
    from test_motion_timing import fake_env
    from native_stand.students import motion_mcu_runtime
    from native_stand.students.motion_fixed_timing import configuration
    def compensation(q_leg):
        comp=q_leg.new_zeros(len(q_leg),6);comp[:,0]=-sign*15.
        return comp,q_leg.new_zeros(len(q_leg),2),q_leg.new_zeros(len(q_leg),2),torch.ones(len(q_leg),dtype=torch.bool)
    monkeypatch.setattr(motion_mcu_runtime,'spring_compensation',compensation)
    env=fake_env(1);env.cfg=SimpleNamespace(leg_kp=40.,leg_kd=2.,wheel_kv=1.)
    timing=motion_mcu_runtime.MCUTiming(env,configuration(),0)
    timing.driver.zero_();timing.driver[:,0]=sign*35.
    timing.motor_ring.delay.zero_();timing.motor_ring.tau.zero_()
    timing.tick=1  # Motor delivery between PD ticks; isolate the independent spring.
    published=[]
    env.robot.set_joint_effort_target_index=lambda **kw:published.append(kw['target'].clone())
    timing.apply()
    assert timing.motor_actual[0,0]==sign*35.
    assert timing.spring_actual[0,0]==sign*15.
    assert published[0][0,0]==sign*50.
    assert abs(published[0][0,0])<profile()['solver_limits_nm'][0]


def test_gpu_compatible_model_inference_tensors_do_not_poison_reset():
    r=MotorRing(2,'cpu');r.reset(torch.arange(2),torch.zeros(2,6))
    with torch.inference_mode():r.step(0,torch.ones(2,6))
    r.reset(torch.tensor([0]),torch.zeros(1,6))
    assert not r.actual[0].any()


def test_stage_quotas_standing_first_and_legacy_proofs_not_imported():
    for stage in range(4):
        s=course.stage_config(stage)
        assert len(s['sampling_slots'])==4096 and sum(s['slot_counts'])==4096
        if stage<3:assert s['commands']==[[0.,0.]]
        else:assert s['slot_counts']==[2048,512,512,512,512]
    assert course.configuration()['old_DR_proofs_imported'] is False
    assert profile()['local_target_delay_ms']==0


def candidate(stage=0,passed=True):
    variants=[0] if stage==0 else [0,1] if stage==1 else [0,1,2]
    records=[dict(case=dict(command=cmd+[.34],_mcu_variant=v,_mcu_revision=stage),
             per_env=[dict(env=i,accepted=passed) for i in range(8)]) for v in variants for cmd in course.commands(stage)]
    return dict(update=500,stage=stage,complete_matrix=True,evaluations=records,checkpoint='/source.pt',sha256='a'*64)


def test_advancement_requires_every_actual_variant_and_no_stale_revision():
    state=dict(stage=0,stage_started_update=0,last_evaluation_update=-1,accepted_by_stage={})
    new,d=course.advance(state,candidate());assert d['advanced'] and new['stage']==1
    new,d=course.advance(state,candidate(passed=False));assert not d['advanced'] and new['stage']==0
    bad=candidate();bad['evaluations'][0]['case']['_mcu_revision']=2
    new,d=course.advance(state,bad);assert not d['evidence_valid'] and not d['advanced']
    bad=candidate();bad['evaluations'][0]['per_env'][0]['accepted']=None
    assert not course.advance(state,bad)[1]['advanced']
    with pytest.raises(ValueError):course.advance({**state,'last_evaluation_update':500},candidate())


def test_source_mix_filters_out_other_heights():
    source=[dict(target=[.5,0,.32,.32],mix=.75),dict(target=[.5,0,.34,.34],mix=.25)]
    assert course.extend_mixture(source,[[.5,0.]])[0]['mix']==.25


def test_capability_probe_covers_signed_four_limits_and_groups_nominal_cases():
    from native_stand.students.motion_mcu_probe import cases
    from native_stand.students.grouped_evaluation import plan
    template=dict(kind='student',role='MotionZeroStudent',checkpoint='/source.pt',sha256='a'*64,
                  family='motion_native_normal',seed=20261014,yaw_rad=0.,seconds=31.52)
    rows=cases(template)
    assert len(rows)==26 and len({r['case_id'] for r in rows})==26
    assert {tuple(r['command'][:2]) for r in rows[:24]}=={
        (value,0.) if axis==0 else (0.,value)
        for axis in (0,1) for magnitude in (.5,1.,1.5,2.,3.,4.) for value in (magnitude,-magnitude)}
    assert all(r['_mcu_variant']==0 for r in rows[:24])
    assert [r['_mcu_variant'] for r in rows[24:]]==[1,2]
    assert all(r['command']==[0.,0.,.34] for r in rows[24:])
    assert [len(g) for g in plan(rows,maximum_commands=8)]==[8,8,8,1,1]
    assert template.get('command') is None


def test_critic_variance_is_centered_with_large_return_mean():
    from tensordict import TensorDict
    from types import SimpleNamespace
    from native_stand.recovery.controlled_ppo import KEY
    returns=torch.tensor([1000.,1000.01,999.99,1000.02]).reshape(2,2,1)
    storage=SimpleNamespace(observations=TensorDict({KEY:torch.ones(2,2,1,dtype=torch.bool),
          'motion_diagnostic_cohort':torch.zeros(2,2,1)},batch_size=[2,2]),returns=returns,
          values=returns-.01,advantages=torch.zeros_like(returns))
    r=course.value_diagnostics(storage,[[0.,0.]])[0]
    assert r['return_variance']>1e-5 and r['explained_variance']>.999


def test_motor_saturation_uses_driver_capacity_not_generalized_spring_torque():
    import numpy as np
    from unittest.mock import patch
    state=np.zeros((3000,1,39),np.float32);state[:,:,6]=.34;state[:,:,29]=1
    arrays=dict(first_episode_valid=np.ones((3000,1),bool),controller=np.full((3000,1,1),3),
        state=state,posture=np.zeros((3000,1,10),np.float32),mcu_execution=np.zeros((3000,1,24),np.float32),
        root_post=np.zeros((3000,1,13),np.float32))
    with patch('native_stand.students.motion_reacquisition.assess',lambda *a:None):
        r=dict(checks={'saturation':False})
        course.assess(r,arrays,0,dict(command=[0,0,.34]))
        assert r['checks']['saturation'] is True
        arrays['mcu_execution'][:,:,2]=WHEEL_CAP
        r=dict(checks={'saturation':True})
        course.assess(r,arrays,0,dict(command=[0,0,.34]))
        assert r['checks']['saturation'] is False
