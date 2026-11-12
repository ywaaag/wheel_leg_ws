FROZEN_TEST_SCOPE='PORTABLE'
import copy
from types import SimpleNamespace as S

import pytest
import torch

from native_stand.students import motion_standing_delay as c


def timing_configuration():
    from native_stand.students.motion_antioscillation import configuration
    from native_stand.students.motion_wheel_matching import profile
    old=configuration();old['smooth_mix']=.75
    old['wheel_execution']=profile();old['wheel_kv']=.2
    old['actor_sensor']['critic']='Actor31/Critic140 unchanged; delayed Actor prefix and current physical privileged columns'
    return c.timing_configuration(old)


def fixture():
    return dict(source_checkpoint='source.pt',source_sha256='a'*64,ppo=dict(envs=4096),
        student_evaluation=dict(cases=[dict(kind='student',role='MotionZeroStudent',
            family='motion_native_normal',seed=104,yaw_rad=.01,seconds=31.52)]),
        motion_training={},student_training={},motion_course={},curriculum={},
        motion_height_dr=dict(state=c.initial_state()))


def test_sensor20_has_exact_two_packet_delay_and_causal_pending_states():
    from native_stand.students.motion_sensor_transfer import SensorPackets
    packets=SensorPackets(2,'cpu',delay_ms=20)
    x=torch.ones(2,16)
    assert torch.equal(packets.read(x,0),x)
    assert torch.equal(packets.read(x*2,10),x)
    assert packets.age.tolist()==[10,10]
    assert torch.equal(packets.read(x*3,20),x)
    assert packets.pending()[0,:,0].tolist()==[3,2]
    assert torch.equal(packets.read(x*9,20),x)
    assert packets.pending()[0,:,0].tolist()==[3,2]
    packets.reset(torch.tensor([0]))
    out=packets.read(x*4,20).clone()
    assert out[:,0].tolist()==[4,1]
    assert packets.pending()[0,:,0].tolist()==[4,4]
    assert packets.read(x*5,30)[:,0].tolist()==[4,2]
    assert packets.age.tolist()==[10,20]
    assert packets.read(x*6,40)[:,0].tolist()==[4,3]


def test_target_wait5_and_torque_wait20_are_separate_with_current_PD_feedback():
    from test_motion_timing import fake_env
    from native_stand.students.motion_fixed_timing import configuration
    from native_stand.students.motion_standing_delay_runtime import StandingDelayTiming
    env=fake_env(2);env.cfg=S(leg_kp=40.,leg_kd=2.,wheel_kv=1.)
    t=StandingDelayTiming(env,configuration(),0)
    t.requested[:,2]=2.
    rows=[]
    for tick in range(36):
        if tick>=10:env.robot.data.joint_vel.torch[:,2]=3.
        t.apply();rows.append((float(t.adopted[0,2]),float(t.driver[0,2]),float(t.motor_actual[0,2])))
        assert torch.equal(env.executed_wheel,t.adopted[:,[2,5]])
    assert all(x[0]==0 for x in rows[:5])
    assert rows[5][0]==2. and rows[5][1]==pytest.approx(.4)
    assert rows[10][1]==pytest.approx(-.2)  # Current dq3, not10ms-old dq0.
    assert all(x[2]==0 for x in rows[:25])
    alpha=1-torch.exp(torch.tensor(-.5)).item()
    assert rows[25][2]==pytest.approx(alpha*.4)  # Old torque arrives, then tau2ms response.
    assert rows[30][2]==pytest.approx(.4*(1-torch.exp(torch.tensor(-2.5)).item())*(1-alpha)-alpha*.2)
    other=t.motor_ring.history[:,1].clone()
    t.reset(torch.tensor([0]))
    assert torch.equal(other,t.motor_ring.history[:,1])
    assert (t.delays==5).all() and (t.feedback==0).all() and (t.motor_ring.delay==20).all()


def test_critic411_preserves_prefix_and_contains_complete_causal_queues():
    base=torch.randn(2,312)
    motors=torch.arange(64.).reshape(64,1,1).expand(64,2,6)
    targets=motors[:32].clone();sensors=torch.randn(2,2,16)
    caps=torch.ones(2,6)
    out=c.critic_input(base,motors,targets,sensors,31,caps)
    assert out.shape==(2,411) and torch.equal(out[:,:312],base)
    assert out[0,312:348].reshape(6,6)[:,0].tolist()==[16,15,14,13,12,11]
    assert torch.allclose(out[0,348:378].reshape(5,6)[:,2],torch.tensor([30,29,28,27,26])/100.)
    assert torch.equal(out[:,378:410],sensors.flatten(1))
    assert (out[:,410]==.25).all()
    # The not-yet-executed tick and more recent motor history cannot affect the appended15..20ms block.
    changed=motors.clone();changed[31]=999
    assert torch.equal(c.critic_input(base,changed,targets,sensors,31,caps),out)


def test_state_identity_and_unknown_never_certify_standing():
    spec=c.effective_spec(fixture(),0)
    assert spec['curriculum']['stages'][0]['slot_counts']==[3072,1024]
    assert spec['curriculum']['stages'][0]['commands']==[[0.,0.],[0.,0.]]
    rows=[dict(case=r,per_env=[dict(env=i,accepted=True,survival_passed=True,controlled_seconds=30.) for i in range(8)])
          for r in spec['student_evaluation']['cases']]
    candidate=dict(stage=0,update=100,complete_matrix=True,evaluations=rows,
        checkpoint='x.pt',sha256='b'*64)
    state,decision=c.advance(spec['motion_height_dr']['state'],candidate)
    assert decision['stage_passed'] and not decision['stop'] and not decision['advanced']
    broken=copy.deepcopy(candidate);broken['evaluations'][0]['per_env'][0]['accepted']=None
    assert not c.advance(spec['motion_height_dr']['state'],broken)[1]['evidence_valid']
    broken=copy.deepcopy(candidate);broken['evaluations'][1]['case']=rows[0]['case']
    assert not c.advance(spec['motion_height_dr']['state'],broken)[1]['evidence_valid']


def test_new_recipe_explicit_old_defaults_unchanged_and_bad_timing_rejected():
    from native_stand.students.training_recipe import validate_instance
    from native_stand.students.motion_mcu_execution import profile
    from native_stand.students.motion_course import implementation
    r=c.recipe('a'*64,timing_configuration())
    assert validate_instance(r,'a'*64)==r and implementation(c.VERSION) is c
    assert profile()['actor_age_ms']==10 and profile()['local_target_delay_ms']==0
    assert c.execution_profile()['actor_age_ms']==20 and c.execution_profile()['local_target_delay_ms']==5
    assert c.execution_profile()['PD_age_ms']==0 and c.execution_profile()['motor_delay_ms']==[20]*6
    assert c.execution_profile()['motor_tau_ms']==[2]*6
    r['timing_configuration']['actor_sensor']['joint_delay_ms']=10
    with pytest.raises(ValueError):validate_instance(r,'a'*64)


def test_runtime_protocol_resolution_uses_explicit_course_without_legacy_override():
    from native_stand.students.evaluator import absolute_protocol_for_spec
    from native_stand.students.motion_highspeed_standing import VERSION as OLD,protocol as old_protocol
    for version,expected in ((c.VERSION,c.protocol()),(OLD,old_protocol())):
        actual,forward=absolute_protocol_for_spec(dict(mcu_execution=True,
            student_training=dict(course_version=version)))
        assert actual==expected and forward
    assert c.protocol()!=old_protocol()


def test_actual_reward_curves_correct_direction_and_non_saturating_zero_tail():
    from native_stand.students.motion_tracking_reward import configuration,PARALLEL_VERSION
    audit=c.learning_signal_audit(dict(source_sha256='a'*64,reward=configuration(PARALLEL_VERSION),
        motion_timing=dict(configuration=timing_configuration())))
    assert audit['status']=='PASSED' and audit['teacher_eligible_cohort_indices']==[]
    rows=audit['standing_zero_and_posture_curves']
    assert all(r['state_error_derivative']<0 for r in rows)
    tails=[r for r in rows if r['signal'] in ('linear_zero','yaw_zero') and r['error']>=1]
    assert tails and all(abs(r['state_error_derivative'])>1 for r in tails)


def test_real_adam_and_value_initialization_keep_source_value_and_allow_new_columns_to_learn():
    from native_stand.students.actor_widening import wide_network
    from native_stand.students.motion_mcu_privileged import VERSION as OLD
    from native_stand.students.model import Student
    old=Student(False,version='multiskill_mcu_v2',critic_privileged_version=OLD,
        network_architecture=wide_network(2048))
    optimizer=torch.optim.Adam(old.parameters(),lr=3e-5)
    old(torch.randn(4,312)).square().mean().backward();optimizer.step()
    source=dict(actor={},critic=old.state_dict(),optimizer=optimizer.state_dict(),
        network_architecture=wide_network(2048),critic_privileged_version=OLD)
    migrated=c.expand_source(source)
    new=Student(False,version='multiskill_mcu_v2',critic_privileged_version=c.CRITIC,
        network_architecture=c.network())
    new.load_state_dict(migrated['critic'])
    x=torch.randn(4,312);extra=torch.randn(4,99)
    assert torch.allclose(new(torch.cat((x,extra),-1)),old(x),atol=1e-7)
    opt=torch.optim.Adam(new.parameters(),lr=3e-5);opt.load_state_dict(migrated['optimizer'])
    assert not opt.state[new.mlp[0].weight]['exp_avg'][:,312:].any()
    loss=(new(torch.cat((x,extra),-1))-1).square().mean();loss.backward()
    assert new.mlp[0].weight.grad[:,312:].abs().sum()>0
    opt.step();assert new.mlp[0].weight[:,312:].abs().sum()>0
    assert source['critic']['mlp.0.weight'].shape==(512,312)
