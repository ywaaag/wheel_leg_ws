FROZEN_TEST_SCOPE='PORTABLE'
import copy
import torch
import pytest

from native_stand.students.motion_large_combo import network,stage_config,VERSION
from native_stand.students.motion_tracking_reward import reward,configuration,COMBO_VERSION,YAW_NARROW_VERSION
from native_stand.students.model import Student


def data(vx=0.,target=(0.,.5)):
 z=torch.zeros(3);yes=torch.ones(3,dtype=torch.bool)
 return dict(command=torch.tensor([[*target,.34]]).expand(3,-1).float(),target=torch.tensor([target]).expand(3,-1).float(),
  velocity=torch.tensor([[vx,0.,0.]]).expand(3,-1).float(),yaw=z+.5,height=z+.34,angles=torch.zeros(3,2),contacts=torch.ones(3,2)*20,
  leg_q=torch.zeros(3,4),wheel_offset=z,leg_lengths=torch.ones(3,2)*.3,applied_torque=torch.zeros(3,6),
  previous_targets=torch.zeros(3,6),executed_targets=torch.zeros(3,6),controlled=yes,valid=yes,physical_failure=~yes,fault=~yes)


def test_gate_is_episode_yaw_only_and_has_finite_gradient():
 c=configuration(COMBO_VERSION);old=configuration(YAW_NARROW_VERSION)
 for target in ((0.,0.),(.5,0.),(1.,0.)):
  d=data(.5,target);torch.testing.assert_close(reward(**d,config=c)['raw'],reward(**d,config=old)['raw'])
 d=data(.5);d['velocity']=d['velocity'].clone().requires_grad_()
 out=reward(**d,config=c);assert torch.allclose(out['raw'][:,2],torch.exp(torch.tensor(-.25)).expand(3))
 g=torch.autograd.grad(out['total'].sum(),d['velocity'])[0];assert torch.isfinite(g).all() and (g[:,0]<0).all()
 d=data(.5);d['command']=d['command'].clone();d['command'][:,1]=0
 assert (reward(**d,config=c)['raw'][:,2]<reward(**d,config=old)['raw'][:,2]).all()


def test_large_stage_keeps_integer_quotas_and_separate64eval():
 from native_stand.command_v6.domain.command_sampling import resolve
 s=stage_config(0);assert s['slot_counts']==[1024,384,384,384,384,768,768]
 assert len(resolve(s,4096)['fixed_slots'])==4096
 assert len(resolve(stage_config(0,64),64)['fixed_slots'])==64
 with pytest.raises(ValueError):stage_config(0,512)


def test_explicit_network_roundtrip_and_old_defaults(tmp_path):
 a=Student(True,version='multiskill_mcu_v2',network_architecture=network())
 c=Student(False,version='multiskill_mcu_v2',critic_privileged_version='motion_privileged49_v1',network_architecture=network())
 assert a.mlp[0].weight.shape==(512,31) and c.mlp[0].weight.shape==(512,49)
 assert Student(True,version='multiskill_mcu_v2').mlp[0].weight.shape==(128,31)
 from native_stand.students.model import load
 # Full trained checkpoints require matching recipe; bare explicit checkpoint is rejected.
 p=tmp_path/'bad.pt';torch.save({'interface_version':'multiskill_mcu_v2','actor':a.state_dict(),'network_architecture':network()},p)
 with pytest.raises(ValueError,match='architecture differs'):load(p)


def test_reward_snapshot_actual_gate_version_is_differentiable():
 from native_stand.students.motion_reward_audit import audit
 r=audit(configuration(COMBO_VERSION));assert r['simulation']==0
 assert r['configuration']['yaw_translation_gate']['denominator_m2_s2']==1.


def test_real_ppo_large_assembly_restore_step_save_load(tmp_path,monkeypatch):
 from tensordict import TensorDict
 from native_stand.students.ppo import assemble
 from native_stand.students.motion_source_retention import declaration as retention,add_metadata
 from native_stand.students.motion_training import declaration as recipe
 from native_stand.students.motion_startup import smoke_profile
 from native_stand.students.motion_course_requests import restore
 from native_stand.students.model import load
 from native_stand.recovery.controlled_ppo import KEY
 small=Student(True,version='multiskill_mcu_v2');small.log_std.requires_grad_(False)
 critic=Student(False,version='multiskill_mcu_v2',critic_privileged_version='motion_privileged49_v1')
 optimizer=torch.optim.Adam([*small.parameters(),*critic.parameters()],lr=3e-5)
 source={'interface_version':'multiskill_mcu_v2','role':'MotionZeroStudent','actor':small.state_dict(),
         'critic':critic.state_dict(),'optimizer':optimizer.state_dict(),'training_rng':{},
         'reward_refinement_state':[{'target':c,'mix':0.,'consecutive_eligible':0} for c in [[0,0],[.5,0],[1,0],[1.5,0],[2,0],[0,.5],[0,-.5]]]}
 cp=tmp_path/'source.pt';torch.save(source,cp)
 cfg=recipe('a'*64,reward_version=COMBO_VERSION)
 cfg.update(network_architecture=network(),course_version=VERSION,course_stage=0,
            source_policy_retention=retention('a'*64),smoke_wiring=smoke_profile(),checkpoint_every_updates=100)
 obs=TensorDict({'student_actor':torch.zeros(8,31),'student_critic':torch.zeros(8,49),KEY:torch.ones(8,1,dtype=torch.bool)},batch_size=[8])
 obs=add_metadata(obs,torch.zeros(8,2),cfg['source_policy_retention'])
 p=assemble(cp,obs,steps_per_env=4,recipe=cfg)
 monkeypatch.setattr('native_stand.students.resume.restore_rng',lambda state:None)
 restore(p,source,{'network_architecture':network(),'student_training':cfg},'test')
 assert p.actor.mlp[0].weight.shape==(512,31) and p.critic.mlp[0].weight.shape==(512,49)
 assert p.source_retention.teacher.mlp[0].weight.shape==(128,31)
 for _ in range(4):
  p.act(obs);p.process_env_step(obs,torch.ones(8)*.1,torch.zeros(8,dtype=torch.bool),{})
 p.compute_returns(obs);p.update()
 assert p.mask_summary['actor_optimizer_steps']==16
 saved={**source,'actor':p.actor.state_dict(),'critic':p.critic.state_dict(),'optimizer':p.optimizer.state_dict(),
        'network_architecture':network(),'training_recipe':cfg}
 target=tmp_path/'large.pt';torch.save(saved,target)
 reloaded,_=load(target,'MotionZeroStudent')
 assert torch.equal(reloaded.mean(torch.zeros(1,31)),p.actor.mean(torch.zeros(1,31)))


def test_value_diagnostics_is_within_cohort_and_handles_constant_targets():
 from types import SimpleNamespace
 from native_stand.students.motion_large_combo import value_diagnostics
 from native_stand.recovery.controlled_ppo import KEY
 s=SimpleNamespace(observations={KEY:torch.ones(1,7,1,dtype=torch.bool),'motion_diagnostic_cohort':torch.arange(7).reshape(1,7,1)},
                   returns=torch.arange(7.).reshape(1,7,1),values=torch.zeros(1,7,1),advantages=torch.ones(1,7,1))
 r=value_diagnostics(s)
 assert all(x['explained_variance'] is None for x in r)
 assert r[2]['value_error_MSE']==4 and r[2]['raw_advantage_mean']==2
