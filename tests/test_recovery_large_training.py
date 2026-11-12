"""Explicit large Recovery1024 training boundaries, no Isaac/physical execution."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy
import pytest
import torch
from tensordict import TensorDict
from native_stand.students.recovery_large_training import recipe,configuration,reward,SOURCE,FIELDS,selection_rank
from native_stand.students.recovery_broad_narrow_reward import reward as old_reward
from native_stand.students.training_recipe import validate_instance
from test_recovery_broad_narrow_reward import inputs

def test_reward_preserves_pre_entry_and_ramps_once_with_correct_dt():
 x=inputs();x['velocity_vx_vy_yaw'][:,0]=torch.tensor([0.,.39,1.])
 original=old_reward(**x);rng=torch.get_rng_state().clone()
 r=reward(**x,entry_elapsed=torch.tensor([0.,.5,1.]),reward_configuration=configuration())
 torch.testing.assert_close(r['total'],original['total']);assert torch.equal(rng,torch.get_rng_state())
 x['entered'][:]=True
 start=reward(**x,entry_elapsed=torch.zeros(3),reward_configuration=configuration())
 end=reward(**x,entry_elapsed=torch.ones(3),reward_configuration=configuration())
 assert start['weighted_density'][:,7].sum()==0 and start['raw'][:,11:].sum()==0
 assert end['total'][0]>end['total'][1]>end['total'][2]
 assert end['weighted_density'][0,11:].sum()==2
 # Continuous terms get dt once; Certificate events unchanged and unscaled.
 x['entry_event'][0]=True;x['success_event'][1]=True
 events=reward(**x,entry_elapsed=torch.ones(3),reward_configuration=configuration())
 torch.testing.assert_close(events['total'],events['weighted_density'].sum(-1)*.01+events['events'].sum(-1))
 assert events['events'][1,1]==10
 with pytest.raises(ValueError,match='elapsed'):reward(**x,entry_elapsed=torch.tensor([0.,float('nan'),1.]),reward_configuration=configuration())
 x['controlled'][:]=False
 assert reward(**x,entry_elapsed=torch.ones(3),reward_configuration=configuration())['total'].sum()==0

def test_recipe_and_full_selection_reject_quick_and_unbound_source():
 r=recipe();assert validate_instance(r,SOURCE)==r
 with pytest.raises(ValueError,match='source SHA'):validate_instance(r,'e'*63)
 changed=copy.deepcopy(r);changed['reward']['stop_shaping']['ramp_s']=2
 with pytest.raises(ValueError):validate_instance(changed,SOURCE)
 e={'accepted':True,'entry_index':20,'valid_hold_prefix_s':10.,'velocity_means':[.4,0,0]}
 a={'per_env':[copy.deepcopy(e) for i in range(32)],'update':0}
 b=copy.deepcopy(a);b['update']=100
 for v in b['per_env']:v['velocity_means'][0]=.1
 assert selection_rank(b)>selection_rank(a)
 b['per_env'][0]['accepted']=False
 assert selection_rank(a)>selection_rank(b)
 with pytest.raises(ValueError,match='32 matrix'):selection_rank({'per_env':[e]})

def test_public_assemble_large_mask_warmup_and_fresh_critic(tmp_path):
 from native_stand.students.model import Student,load
 from native_stand.students.recovery_large import expand
 from test_recovery_large import candidate
 from native_stand.students.ppo import assemble
 from native_stand.recovery.controlled_ppo import KEY
 torch.set_num_threads(4)
 small=Student(version='multiskill_mcu_v2');critic=Student(False,version='multiskill_mcu_v2',critic_privileged_version='recovery_privileged64_v1')
 sa,sc=expand(small.state_dict(),critic.state_dict());cp=tmp_path/'initial.pt';torch.save(candidate(sa,sc),cp)
 o=TensorDict({'student_actor':torch.randn(4,31),'student_critic':torch.randn(4,64),KEY:torch.ones(4,1,dtype=torch.bool)},batch_size=[4])
 p=assemble(cp,o,steps_per_env=4,recipe=recipe());assert p.actor.mlp[0].out_features==512 and p.critic.mlp[0].in_features==64
 assert p.optimizer.param_groups[0]['lr']==3e-5 and not p.optimizer.state
 assert all(torch.equal(v,p.actor.state_dict()[k]) for k,v in sa.items())
 assert any(not torch.equal(v,p.critic.state_dict()[k]) for k,v in sc.items())
 def step(enable,valid):
  before=copy.deepcopy(p.actor.state_dict());p.actor_update_enabled=enable;o[KEY][:]=valid
  with torch.inference_mode():
   for i in range(4):
    raw=p.act(o);torch.testing.assert_close(p.transition.actions_log_prob,p.actor.log_prob(o['student_actor'],raw))
    p.process_env_step(o,torch.rand(4),torch.zeros(4,dtype=torch.bool),{})
   p.compute_returns(o)
  p.update();return any(not torch.equal(v,p.actor.state_dict()[k]) for k,v in before.items())
 assert not step(False,True)
 assert step(True,True)
 state=copy.deepcopy(p.optimizer.state_dict());before=copy.deepcopy(p.actor.state_dict())
 assert not step(True,False)
 for param in p.actor.parameters():
  assert param.grad is None
 assert p.mask_summary['actor_optimizer_steps']==0 and p.mask_summary['BC_updates']==0
 assert torch.equal(before['log_std'],p.actor.log_std)


def test_public_config_and_model_factory1024_budget():
 from native_stand.students.training_recipe import validate
 from native_stand.command_v6.tools.model_factory import model
 from native_stand.command_v6.domain.execution_limits import validate_budget,ExecutionLimits
 from native_stand.students.recovery_large import network
 r=recipe()
 cases=[dict(case_id=name,kind='student',seed=seed,yaw_rad=yaw,family=r['training_family'],role='RecoveryStudent') for name,seed,yaw in [('student_native',20261010,0.),('student_yaw10',20261010,.01),('student_yaw11',20261011,.01),('student_yaw12',20261012,.01)]]
 s=dict(interface_version='multiskill_mcu_v2',student_training=r,source_sha256=SOURCE,network_architecture=network(),
 ppo=dict(envs=1024,steps_per_env=128,hidden_dims=[512,256,128],learning_rate=3e-5,epochs=4,minibatches=4,gamma=.99,lam=.95,clip=.2,value_coef=.5,entropy_coef=.001,max_grad_norm=1.),
 recovery=dict(sampling={'version':'family_weights_v1','family_weights':{r['training_family']:1.}},assessment={'families':[r['training_family']],'retained_families':[]}),restore_source_rng=False,restore_recovery_curriculum=False,
 reward=r['reward'],student_evaluation={'cases':cases},budget=dict(smoke_updates=12,formal_updates=2488,smoke_wall_s=None,formal_wall_s=None,chain_wall_s=None,sample_limit=327680000))
 assert validate(s)==[] and validate_budget(s)['samples']==327680000
 assert ExecutionLimits.resolve(s,'formal').wall_s is None
 assert model({},actor=True,spec=s).mlp[0].weight.shape==(512,31)
 assert model({},actor=False,spec=s).mlp[0].weight.shape==(512,64)
 bad=copy.deepcopy(s);bad['ppo']['envs']=64;assert validate(bad)
 bad=copy.deepcopy(s);bad['reward']['stop_shaping']['ramp_s']=9;assert validate(bad)

def test_checkpoint_roundtrip_carries_large_network_and_fresh_recipe(tmp_path):
 from native_stand.students.model import Student,load
 from native_stand.students.recovery_large import expand,network
 from test_recovery_large import candidate
 from native_stand.students.ppo import assemble
 from native_stand.students.training import save_checkpoint
 from native_stand.recovery.controlled_ppo import KEY
 import json
 a=Student(version='multiskill_mcu_v2');c=Student(False,version='multiskill_mcu_v2',critic_privileged_version='recovery_privileged64_v1')
 sa,sc=expand(a.state_dict(),c.state_dict());cp=tmp_path/'initial.pt';source=candidate(sa,sc);torch.save(source,cp)
 obs=TensorDict({'student_actor':torch.randn(4,31),'student_critic':torch.randn(4,64),KEY:torch.ones(4,1,dtype=torch.bool)},batch_size=[4])
 p=assemble(cp,obs,steps_per_env=4,recipe=recipe())
 s={'source_sha256':SOURCE,'interface_version':'multiskill_mcu_v2','student_training':recipe(),'asset_sha256':'f'*64,'task':'recovery'}
 (tmp_path/'source_snapshot').mkdir();(tmp_path/'source_snapshot/spec.json').write_text(json.dumps(s))
 saved=save_checkpoint(tmp_path,s,p.actor,p.critic,p,source,6)
 actor,loaded=load(saved['checkpoint'],'RecoveryStudent')
 assert loaded['network_architecture']==network() and loaded['training_recipe']==recipe()
 assert loaded['training_rng'] and loaded['online_updates_this_stage']==6
 assert 'initialization_kind' not in loaded
 assert all(torch.equal(v,actor.state_dict()[k]) for k,v in p.actor.state_dict().items())


def test_quick_schedule_never_replaces_full_or_enters_selection():
 from native_stand.students.recovery_large_training import quick_due
 s={'student_training':recipe()}
 assert not quick_due(s,'formal',0,full_assessment=False)
 assert quick_due(s,'formal',100,full_assessment=False)
 assert not quick_due(s,'formal',500,full_assessment=True)
 assert not quick_due(s,'smoke',100,full_assessment=False)
 assert not quick_due(s,'formal',2488,full_assessment=True)
 assert not quick_due({},'formal',100,full_assessment=False)
