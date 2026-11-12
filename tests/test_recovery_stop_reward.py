FROZEN_TEST_SCOPE='PORTABLE'
import torch
from native_stand.students.recovery_broad_narrow_reward import reward,configuration,STOP_VERSION
from native_stand.students.recovery_privileged import recipe,STOP_RECIPE,STOP_SOURCE
from native_stand.students.training_recipe import validate_instance
from test_recovery_broad_narrow_reward import inputs

def test_only_forward_cost_changes_after_entry():
 old=configuration();new=configuration(STOP_VERSION)
 assert {k for k in old if old[k]!=new[k]}=={'version','forward_deadband_mps','post_entry_velocity_scale'}
 x=inputs();x['velocity_vx_vy_yaw'][:,0]=torch.tensor([0.,.1,.4])
 a=reward(**x,reward_configuration=old);b=reward(**x,reward_configuration=new)
 torch.testing.assert_close(a['total'],b['total'])
 x['entered'][:]=True
 a=reward(**x,reward_configuration=old);b=reward(**x,reward_configuration=new)
 assert b['raw'][1,7]>0 and a['raw'][1,7]==0
 assert b['total'][2]<a['total'][2]
 torch.testing.assert_close(a['raw'][:,:7],b['raw'][:,:7]);torch.testing.assert_close(a['raw'][:,8:],b['raw'][:,8:])
 assert torch.allclose(b['total']-a['total'],(b['weighted_density'][:,7]-a['weighted_density'][:,7])*.01)
 r=recipe(STOP_SOURCE,version=STOP_RECIPE);assert validate_instance(r,STOP_SOURCE)==r

def test_public_new_recipe_actual_cpu_optimizer_mask_warmup(tmp_path):
 from native_stand.students.model import Student
 from native_stand.students.ppo import assemble
 from native_stand.recovery.controlled_ppo import KEY
 from tensordict import TensorDict
 a=Student(version='multiskill_mcu_v2');cp=tmp_path/'source.pt'
 torch.save({'interface_version':'multiskill_mcu_v2','role':'RecoveryStudent','actor':a.state_dict()},cp)
 o=TensorDict({'student_actor':torch.randn(4,31),'student_critic':torch.randn(4,64),KEY:torch.ones(4,1,dtype=torch.bool)},batch_size=[4])
 p=assemble(cp,o,steps_per_env=4,recipe=recipe(STOP_SOURCE,version=STOP_RECIPE))
 before={k:v.clone() for k,v in p.actor.state_dict().items()}
 def step(enabled,valid):
  p.actor_update_enabled=enabled;o[KEY][:]=valid
  with torch.inference_mode():
   for i in range(4):
    raw=p.act(o);p.process_env_step(o,torch.rand(4),torch.zeros(4,dtype=torch.bool),{})
   p.compute_returns(o)
  p.update()
 step(False,True);assert all(torch.equal(v,p.actor.state_dict()[k]) for k,v in before.items())
 step(True,False);assert all(torch.equal(v,p.actor.state_dict()[k]) for k,v in before.items())
 step(True,True);assert any(not torch.equal(v,p.actor.state_dict()[k]) for k,v in before.items())
 assert p.mask_summary['BC_updates']==0 and torch.equal(before['log_std'],p.actor.log_std)
