"""Actual ABI2 C, portable nominal/residual policy and common state interfaces."""
import copy
import ctypes as C
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from native_stand.experiments.config import load_job_config,job_from_record
from native_stand.learning.policies.factory import build_policy
from native_stand.simulation.control_core import build
from native_stand.simulation.jump_nominal_c import NominalJump
from native_stand.simulation.jump_shared_c import array,declared_input
from native_stand.tasks.jump.nominal import RESIDUAL,VERSION
from native_stand.tasks.registry import build_task_from_registry

FROZEN_TEST_SCOPE='PORTABLE'
FIXTURE=Path(__file__).parent/'fixtures/cpu_jump_job.json'


@pytest.fixture(scope='module')
def binding(tmp_path_factory):return build(tmp_path_factory.mktemp('jump-core')/'control.so',jump=True)


def active(binding,phase=1):
    c=NominalJump(binding['path'],1,library_sha=binding['sha256'])
    cfg=c.config;cfg.enabled=cfg.model_available=1;cfg.model_id=0x54455354
    cfg.model_input_id=c.INPUT_CONTRACT;cfg.model_output_id=c.OUTPUT_CONTRACT
    i=declared_input(now=100,gyro=[0,0,0],gravity=[0,-1,0],q=[-.13,-.15,.13,-.15],dq=[0]*6,
        command=[0,0,.34],adopted=[-.13,-.15,0,.13,-.15,0],adopted_sequence=10,
        sensor_tick=100,imu_tick=100,feedback_ticks=[100]*6,accel=[0,-9.81,0],driver_tau=[0]*4,
        internal_tau=[0]*4,support_J=np.tile(np.eye(2),(2,1)),support_tick=100,support_valid=3,
        leg_length=[.28,.28],leg_spread=[1.,1.],motion_entry=1)
    c.inputs[0]=i;s=c.states[0];s.phase=phase;s.owner=2;s.phase_sequence=1;s.phase_tick=40
    s.jump_tick=40;s.last_tick=100;s.adopted_valid=1;s.adopted_sequence=10;s.adopted_tick=100
    s.adopted[:]=i.upstream_target;s.contact[:]=s.contact_valid[:]=[1,1];s.candidate[:]=[1,1]
    c.lib.RLJumpNominalUpdate(C.byref(s),100)
    return c


@pytest.mark.parametrize('phase',range(1,6))
def test_actual_C_ABI2_stage_residual_and_physical_previous(binding,phase):
    c=active(binding,phase);s=c.states[0]
    assert c.config.abi==2 and c.config.trajectory_id==2830348563
    obs,valid=c.observe();assert valid[0]
    np.testing.assert_array_equal(obs[0,31:36],np.eye(5)[phase-1])
    nominal=array(c.last_snapshot[0].nominal)
    mean=np.full((1,6),.2,np.float32)
    assert c.publish(mean)[0]==0
    np.testing.assert_allclose(array(s.requested),nominal+.2*RESIDUAL[phase-1],atol=1e-7)
    c.inputs[0].now=105;rc,frame=c.pd();assert rc[0]==0
    np.testing.assert_allclose(frame[0]['residual'],.2*RESIDUAL[phase-1],atol=1e-7)
    o,_=c.observe()
    np.testing.assert_allclose(o[0,16:20],array(s.adopted)[[0,1,3,4]]/1.5,atol=1e-7)
    np.testing.assert_allclose(o[0,20:22],array(s.adopted)[[2,5]]/100,atol=1e-7)


def test_actual_C_full_state_resume_and_old_ABI_rejection(binding):
    first=active(binding);first.observe();first.publish(np.full((1,6),.1,np.float32))
    saved=first.state_dict();second=active(binding);second.load_state_dict(saved)
    assert bytes(first.states)==bytes(second.states)
    assert bytes(first.inputs)==bytes(second.inputs)
    bad=copy.deepcopy(saved);bad['abi']=1
    with pytest.raises(ValueError,match='identity'):second.load_state_dict(bad)


def test_retired_ABI2_registry_and_historical_policy_loss_state_restore():
    job=load_job_config(FIXTURE)
    assert job_from_record(job.as_run_record()).as_run_record()==job.as_run_record()
    with pytest.raises(ValueError,match='旧Jump ABI'):
        build_task_from_registry(job)
    from native_stand.tasks.jump.spec import build_task
    from native_stand.learning.ppo.jump_nominal import build_policy
    assert build_task(job).contract['version']==VERSION
    actor=torch.zeros(4,37);actor[:,31]=1
    obs=TensorDict({'student_actor':actor,'student_critic':torch.randn(4,64),
        'recovery_actor_valid':torch.ones(4,1,dtype=torch.bool)},batch_size=[4])
    handle=build_policy(job,obs,device='cpu');ppo=handle['ppo']
    assert not ppo.actor.mlp[-1].weight.any()
    with torch.inference_mode():
        for _ in range(job.runtime.steps_per_env):
            raw=ppo.act(obs)
            ppo.process_env_step(obs,1.-raw.square().mean(-1),torch.zeros(4,dtype=torch.bool),{})
        ppo.compute_returns(obs)
    ppo.update()
    assert ppo.mask_summary['actor_optimizer_steps']>0
    assert ppo.mask_summary['version']==VERSION
    saved=handle['capture']({'updates':1,'samples':8},execution_state={'C_state':'explicit fixture'})
    other=build_policy(job,obs,device='cpu');other['restore'](saved)
    for k,v in ppo.actor.state_dict().items():assert torch.equal(v,other['ppo'].actor.state_dict()[k])
    assert other['ppo'].optimizer.binding()==ppo.optimizer.binding()
    bad=dict(saved,version='jump_simple_absolute_37_v1')
    with pytest.raises(ValueError,match='旧绝对'):other['restore'](bad)


def test_jump_upstream_motion_cap_order_preserved(binding):
    c=active(binding);i=c.inputs[0];i.q[:]=[0]*4;i.dq[:]=[0]*6
    i.leg_length[:]=[.28,.28];i.leg_spread[:]=[np.pi/2]*2
    target=np.array([-1.5,0,0,0,0,0],np.float32)
    jump=c.FrameType();motion=c.FrameType()
    assert not c.lib.RLJumpMotorCommand(C.byref(i),target.ctypes.data,C.byref(jump))
    assert not c.lib.RLJumpUpstreamMotorCommand(C.byref(i),target.ctypes.data,C.byref(motion))
    assert jump.final_model_nm[0]==-40
    assert motion.final_model_nm[0]==pytest.approx(-23.5)
