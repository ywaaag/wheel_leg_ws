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
