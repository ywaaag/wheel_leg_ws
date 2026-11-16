"""CPU command/event/readback contracts; no MuJoCo integration or CUDA."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from native_stand.simulation.motion_replication_probe import command_at, event_torque, validate
from native_stand.tasks.motion.replication_metrics import band_rms, longest, summarize

FROZEN_TEST_SCOPE = 'PORTABLE'


def case():
    return dict(name='step',scenario='low',role='development',seed=1,duration_s=6,
                score_window_s=[4,6],commands=[[0,0,0,.34],[2,.25,0,.34],[4,0,0,.34]],
                events=[dict(kind='modal',at_s=3,duration_s=.1,mode=2,sign=-1,peak_Nm=6)])


def test_command_and_real_vector_force_boundaries():
    c=case()
    np.testing.assert_array_equal(command_at(c,1.999),[0,0,np.float32(.34)])
    np.testing.assert_array_equal(command_at(c,2),[.25,0,np.float32(.34)])
    assert not event_torque(c,2999).any()
    assert np.isclose(np.linalg.norm(event_torque(c,3050)),6)
    assert event_torque(c,3099).any()
    assert not event_torque(c,3100).any()
    assert not event_torque(c,3000)[[2,5]].any()


def test_validate_limits_before_any_engine():
    cfg=dict(version='motion_replication_probe_v1',cases=[case()],wall_s=60,
             max_physics_steps=6000,max_output_bytes=8*2**20)
    validate(cfg)
    cfg['max_physics_steps']=5999
    with pytest.raises(ValueError,match='budget'):
        validate(cfg)
    cfg['max_physics_steps']=6000
    cfg['actor_packet_age_ms']=7
    with pytest.raises(ValueError,match='stress'):
        validate(cfg)


def test_packet_stress_only_changes_Actor_age_in_existing_player_plan(tmp_path):
    import dataclasses
    from native_stand.experiments.config import load_job_config
    from native_stand.simulation.playback import PlaybackPlan
    from native_stand.simulation.motion_replication_probe import prepare
    root=__import__('pathlib').Path(__file__).resolve().parents[1]
    job=dataclasses.replace(load_job_config(root/'configs/jobs/motion_supported015_dqgain.json'),assets={})
    contract=dict(action_version=job.action.version,leg_halfwidth_rad=.15,
                  height_command_range_m=[.30,.40],mean_activation='Tanh')
    spec=dict(kind='tanh',input_dim=31,hidden_dims=[8],output_dim=6,
              input_contract=dict(version='multiskill_mcu_v2'),output_contract=contract)
    nominal=dict(version='motion48_v4_wheel_reported_delay_actor20_dr_v2',
                 actor_packet_nominal_ms=20,actor_leg_packet_nominal_ms=20,
                 target_nominal_ms=5,wheel_transport_nominal_ms=[7,11])
    plan=PlaybackPlan(tmp_path,tmp_path/'source.pt',job,{},contract,nominal)
    cfg=dict(plant=dict(run='r',checkpoint='c',path_mapping='m',sha256='a'*64,
                       solver_profile='closure_contact_tight_v1'),
             actor=dict(path='actor.pt',sha256='b'*64,inference=spec),
             cases=[case()],actor_packet_age_ms=30)
    with patch('native_stand.simulation.playback.plan',return_value=plan),patch('native_stand.learning.inference.load_model',return_value='actor'):
        result,_=prepare(cfg)
    assert result.timing['actor_packet_nominal_ms']==result.timing['actor_leg_packet_nominal_ms']==30
    assert nominal['actor_packet_nominal_ms']==20
    assert result.timing['target_nominal_ms']==5 and result.timing['wheel_transport_nominal_ms']==[7,11]


def test_frequency_units_support_gap_and_quiet_recovery():
    t=np.arange(6000)*.001
    value=np.c_[np.sin(2*np.pi*20*t),3*np.sin(2*np.pi*3*t)]
    np.testing.assert_allclose(band_rms(value),[2**-.5,0],atol=1e-10)
    assert longest([False,True,True,False,True])==.002
    c=case();c['commands']=[[0,0,0,.34]]
    z=np.zeros(6000)
    trace=dict(time=t+.001,vx=z,vy=z,yaw=z,roll=z,pitch=z,height=z+.34,
               command=np.tile([0,0,.34],(6000,1)),q=np.zeros((6000,6)),
               dq=np.zeros((6000,6)),normal_force_N=np.ones((6000,2))*30,
               position_xy=np.zeros((6000,2)),raw=np.zeros((6000,6)),
               motor=np.zeros((6000,6)),requested=np.zeros((6000,6)),adopted=np.zeros((6000,6)))
    gates=dict(support_force_N=2,longest_support_loss_s=.15,height_MAE_m=.035,dual_support_fraction=.9,attitude_rad=.35,
               vx_MAE_floor_mps=.3,vx_MAE_fraction=.25,yaw_MAE_rad_s=.5,
               absolute_stationary_speed_mps=.15,response_tolerance_mps=.15,
               quiet_dq_rad_s=.25,quiet_hf_rad_s=.25,quiet_speed_mps=.30,quiet_roll_rad=.05,quiet_pitch_rad=.10)
    r=summarize(trace,c,None,gates)
    assert r['teacher_eligible'] and r['events'][0]['recovery_status']=='PASS'
    trace['normal_force_N'][4500:4700]=0
    r=summarize(trace,c,None,gates)
    assert r['longest_support_loss_s']==.2
    assert not r['teacher_eligible']


def test_explicit_support_gate_is_strictly_greater_than_sixty_percent():
    c=case();c['events']=[];t=np.arange(6000)*.001+.001;z=np.zeros(6000)
    trace=dict(time=t,vx=z,vy=z,yaw=z,roll=z,pitch=z,height=z+.34,
               command=np.tile([0,0,.34],(6000,1)),q=np.zeros((6000,6)),
               dq=np.zeros((6000,6)),normal_force_N=np.ones((6000,2))*30,
               position_xy=np.zeros((6000,2)),raw=np.zeros((6000,6)),
               motor=np.zeros((6000,6)),requested=np.zeros((6000,6)),adopted=np.zeros((6000,6)))
    indices=np.flatnonzero((t>=4)&(t<6));lost=indices[np.arange(len(indices))%5<2]
    trace['normal_force_N'][lost]=0
    gates=dict(support_force_N=2,longest_support_loss_s=.15,height_MAE_m=.035,
               dual_support_fraction=.6,support_fraction_comparison='gt',attitude_rad=.35,
               vx_MAE_floor_mps=.3,vx_MAE_fraction=.25,yaw_MAE_rad_s=.5,
               absolute_stationary_speed_mps=.15,response_tolerance_mps=.15,
               quiet_dq_rad_s=.25,quiet_hf_rad_s=.25,quiet_speed_mps=.30,quiet_roll_rad=.05,quiet_pitch_rad=.10)
    result=summarize(trace,c,None,gates)
    assert result['dual_support_fraction']==.6 and not result['teacher_eligible']
    trace['normal_force_N'][lost[0]]=30
    assert summarize(trace,c,None,gates)['teacher_eligible']


def test_external_torque_added_after_NET_without_changing_motor_or_caches():
    from native_stand.simulation.mujoco_playback import MotionPlayer

    player=object.__new__(MotionPlayer)
    motor=np.arange(6,dtype=float);net=np.arange(6,dtype=float)*.1;external=np.array([1,2,0,3,4,0.])
    class Execution:
        tick=1
        def step(self,q,dq):
            self.tick+=1
            return motor.copy(),net.copy()
    player.execution=Execution()
    player.data=SimpleNamespace(qpos=np.zeros(6),qvel=np.zeros(6),ctrl=np.zeros(6),qfrc_applied=np.ones(6),time=0.)
    player.model=SimpleNamespace(neq=0)
    player.qids=player.vids=np.arange(6)
    player.actuator_ids=np.arange(6);player.actuator_gears=np.ones(6);player.loop_sites=[]
    player.terrain=None;player.plan=SimpleNamespace(job=SimpleNamespace(task='motion'))
    player.recording=False;player.last_record=None;player.actual_physics_steps=0
    observed=[]
    player.mj=SimpleNamespace(mj_step=lambda m,d:observed.append((d.ctrl.copy(),d.qfrc_applied.copy())),mj_forward=lambda m,d:None)
    player.step(SimpleNamespace(),external_torque_nm=external)
    np.testing.assert_array_equal(observed[0][0],motor)
    np.testing.assert_array_equal(observed[0][1],net+external)
    assert player.execution.tick==2


def test_dry_run_never_builds_C_creates_player_or_output(tmp_path):
    import json
    from native_stand.simulation.motion_replication_probe import from_file

    cfg=dict(version='motion_replication_probe_v1',cases=[case()],wall_s=60,
             max_physics_steps=6000,max_output_bytes=8*2**20,actor={'sha256':'a'*64},output=str(tmp_path/'absent'))
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    with patch('native_stand.simulation.motion_replication_probe.prepare',return_value=(SimpleNamespace(record=lambda:{}),None)),patch('native_stand.simulation.control_core.build',side_effect=AssertionError),patch('native_stand.simulation.mujoco_playback.MotionPlayer',side_effect=AssertionError):
        result=from_file(path,dry_run=True)
    assert not result['starts_engine'] and not result['writes'] and not (tmp_path/'absent').exists()


def test_support_force_includes_loaded_positive_gap_contact_without_changing_legacy_touch():
    from native_stand.simulation.mujoco_playback import MotionPlayer

    player = object.__new__(MotionPlayer)
    player.data = SimpleNamespace(
        xmat=np.array([[[0, 0, -1], [-1, 0, 0], [0, 1, 0]]]), qpos=np.zeros(6), qvel=np.zeros(6), time=0.,
        contact=[SimpleNamespace(geom=(1, 0), dist=.0002),
                 SimpleNamespace(geom=(2, 0), dist=-.0001)],
    )
    player.model = SimpleNamespace(geom_bodyid=np.array([0, 1, 2]))
    def force(model, data, index, output):
        output[0] = [23., 29.][index]
    player.mj = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_BODY=1),
        mj_id2name=lambda model, kind, index: ['world', 'left_wheel', 'right_wheel'][index],
        mj_contactForce=force,
        mj_objectVelocity=lambda model, data, kind, body, output, local: output.fill(0),
    )
    player.execution = SimpleNamespace(tick=0, requested=np.zeros(6),
                                      core=SimpleNamespace(adopted=np.zeros(6)),
                                      motor=np.zeros(6), NET=np.zeros(6))
    player.robot_body=player.reset_count=0
    player.qids=player.vids=np.arange(6)
    player.last_record=None; player.terrain=None; player.spawn='test'
    player.solver_readback={'version': 'test'}
    player.ground_height=lambda xy: np.zeros(len(xy))
    result=player.telemetry()
    assert result['normal_force_geometric_right_N']==23.
    assert result['positive_gap_normal_force_geometric_right_N']==23.
    assert result['normal_force_geometric_left_N']==29.
    assert result['positive_gap_normal_force_geometric_left_N']==0.
    assert not result['contact_geometric_right'] and result['contact_geometric_left']


def test_physical_huber_contract_and_gradient_are_not_raw_MSE():
    import torch
    from native_stand.learning.distillation.loop import supervised_loss
    prediction=torch.tensor([[.1,.1,.125,.1,.1,.125]],requires_grad=True)
    labels=torch.zeros(1,6)
    loss=supervised_loss(prediction,labels,config=dict(kind='motion_physical_huber',leg_scale_rad=.05,wheel_scale_rad_s=10),output_contract=dict(leg_halfwidth_rad=.5,wheel_halfwidth_rad_s=80))
    torch.testing.assert_close(loss,torch.tensor(.5))
    loss.backward()
    assert torch.isfinite(prediction.grad).all() and (prediction.grad>0).all()
    assert supervised_loss(prediction.detach(),labels)<loss.detach()


def test_dataset_window_quotas_keep_complete_episode_role_and_sample_identity(tmp_path):
    from native_stand.artifacts.dataset import read_source
    from native_stand.artifacts.io import file_sha256,write_npz
    file=tmp_path/'data.npz'
    n=200
    write_npz(file,dict(states=np.zeros((n,31),np.float32),episodes=np.repeat([4,5],100),time=np.tile(np.arange(100)*.01,2),physical_episode_keys=np.repeat(['epA','epB'],100),roles=np.repeat(['train','development'],100),sample_ids=np.array([f's{i}' for i in range(n)]),condition_scenario=np.repeat(['quiet','push'],100)))
    ref=dict(path=str(file),sha256=file_sha256(file),sampling=dict(time_key='time',windows=[[0,.5,10],[.5,1.,20]]))
    result=read_source(ref)
    assert len(result['states'])==60
    assert sum(result['roles']=='train')==sum(result['roles']=='development')==30
    assert len(set(result['sample_ids']))==60
    assert set(result['physical_episode_keys'])=={'epA','epB'}
    ref['sampling']['windows']=[[0,.6,10],[.5,1.,20]]
    with pytest.raises(ValueError,match='disjoint'):
        read_source(ref)


def test_teacher_envelope_marks_unknown_before_routing_and_physical_conversion(tmp_path):
    import torch
    from native_stand.artifacts.dataset import read_source
    from native_stand.artifacts.io import file_sha256, write_npz
    from native_stand.learning.distillation.routing import ActionLabels, Teacher, label

    file=tmp_path/'student_states.npz'
    write_npz(file,dict(states=np.array([[.1]*31,[2.]*31],np.float32),episodes=np.zeros(2),
                        roles=np.full(2,'train'),sample_ids=np.array(['a','b']),
                        transition_valid=np.ones(2,bool)))
    envelope=dict(version='teacher_observation_box_v1',lower=[-1.]*31,upper=[1.]*31)
    ref=dict(path=str(file),sha256=file_sha256(file),observation_envelope=envelope)
    chunk=read_source(ref)
    np.testing.assert_array_equal(chunk['behavior_valid'],[True,False])
    assert list(chunk['sample_ids'])==['a','b']
    seen=[]
    def mean(states):
        seen.append(states.clone())
        assert bool((states<=1).all())
        return states[:,:6]
    teacher=Teacher('qualified',SimpleNamespace(mean=mean),ActionLabels('physical'),1,{})
    result,owner,valid=label([teacher],torch.from_numpy(chunk['states']),{},ActionLabels('physical'),
                             row_valid=chunk['behavior_valid'])
    assert len(seen)==1 and seen[0].shape==(1,31)
    assert owner.tolist()==[0,-1] and valid.tolist()==[True,False]
    assert torch.isnan(result[1]).all()
    ref['observation_envelope']={**envelope,'upper':[1.]*30}
    with pytest.raises(ValueError,match='envelope'):
        read_source(ref)


def test_vm1_model_constructor_keeps_wheel_transmission_and_motor_only_caps(tmp_path):
    from native_stand.tasks.motion.encoding import CHANNELS
    from native_stand.simulation.motion_vm1_plant import make_model
    names=list(CHANNELS)
    bodies=''.join(f'<body name="b{i}" pos="0 0 .2"><joint name="{n}"/><geom type="sphere" size=".01"/></body>' for i,n in enumerate(names))
    motors=''.join(f'<motor joint="{n}" gear="{16 if i in (2,5) else 1}"/>' for i,n in enumerate(names))
    p=tmp_path/'plant.xml';p.write_text('<mujoco><worldbody>'+bodies+'</worldbody><actuator>'+motors+'</actuator></mujoco>')
    contract=dict(members=[dict(role='mjcf',path=str(p))])
    mechanics=dict(joint_names=names,armature=dict.fromkeys(names,.012),
                   damping=dict.fromkeys(names,0.),friction_loss_Nm=dict.fromkeys(names,0.))
    desc=dict(joints=[dict(name=n,limited=True,limits_rad=[-1.5,1.5]) for n in names])
    v4=dict(sides={k:dict(body_changes=[],parameters=dict(leg_mass_inertia_scale=1.)) for k in ['L','R']})
    with patch('native_stand.simulation.motion_vm1_plant.load_contract',return_value=contract):
        m,refs,meta=make_model({},description=desc,mechanics=mechanics,V4_contract=v4,
                               motor_caps=[40,40,4.8,40,40,4.8])
    assert len(refs)==6 and meta['version']=='motion_vm1_mujoco_v1'
    np.testing.assert_allclose(m.actuator_ctrlrange[[2,5]],[[-.3,.3]]*2)
    np.testing.assert_allclose(m.actuator_forcerange[[2,5]],[[-.3,.3]]*2)
    np.testing.assert_array_equal(m.geom_friction[:,1:],0.)


def test_observable_teacher_routing_cannot_use_future_scenario_or_push_velocity():
    import torch
    from native_stand.learning.distillation.routing import ActionLabels,Teacher,label
    from native_stand.tasks.motion.teacher_domain import validate

    base=dict(version='motion_observable_teacher_domain_v1',reference_height_m=.34,
              command_tolerance=1e-6,core_max_vx_mps=1.,quiet_gyro_rad_s=.25,
              quiet_leg_dq_rad_s=.25,quiet_wheel_dq_rad_s=5.,quiet_tilt_rad=.04)
    teachers=[]
    for i,role in enumerate(['quiet_c0','core_vm1','new_u2200']):
        cfg=dict(base,role=role);validate(cfg)
        model=SimpleNamespace(mean=lambda x,i=i:torch.full((len(x),6),float(i)))
        teachers.append(Teacher(role,model,ActionLabels('physical'),1,{},observation_domain=cfg))
    obs=torch.zeros(7,31);obs[:,4]=-1;obs[:,24]=.68
    obs[2,22]=.025  # actual command+.1m/s -> vM1
    obs[3,12]=.5    # wheel10rad/s, zero command after a push -> vM1, never u2200
    obs[4,22]=.9    # command3.6m/s -> u2200
    obs[5,23]=.125  # commandyaw.5rad/s -> u2200
    obs[6,24]=.4    # commandheight.20m -> u2200
    before=obs.clone()
    actions,owner,valid=label(teachers,obs,{'future_scenario':['low','high','a','b','c','d','e']},ActionLabels('physical'))
    assert owner.tolist()==[0,0,1,1,2,2,2] and valid.all()
    torch.testing.assert_close(actions,owner[:,None].expand(-1,6).float())
    torch.testing.assert_close(obs,before)
    base['role']='new_u2200';base['future_scenario']='high'
    with pytest.raises(ValueError,match='observable'):
        validate(base)
