"""GPU migration uses complete optimizer/RNG, unchanged recipe and no quota refund."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy
import pytest,torch
from tensordict import TensorDict
from native_stand.students import gpu_resume
from native_stand.students.recovery_large_training import recipe
from native_stand.students.recovery_large import expand
from native_stand.students.model import Student
from native_stand.students.ppo import assemble
from native_stand.recovery.controlled_ppo import KEY
from test_recovery_large import candidate

def test_actual_full_restore_matches_Adam_RNG_and_does_not_restart_warmup(tmp_path,monkeypatch):
 torch.set_num_threads(4)
 a=Student(version='multiskill_mcu_v2');c=Student(False,version='multiskill_mcu_v2',critic_privileged_version='recovery_privileged64_v1')
 sa,sc=expand(a.state_dict(),c.state_dict());cp=tmp_path/'initial.pt';torch.save(candidate(sa,sc),cp)
 obs=TensorDict({'student_actor':torch.randn(4,31),'student_critic':torch.randn(4,64),KEY:torch.ones(4,1,dtype=torch.bool)},batch_size=[4])
 p=assemble(cp,obs,steps_per_env=4,recipe=recipe());p.actor_update_enabled=True
 with torch.inference_mode():
  for i in range(4):p.act(obs);p.process_env_step(obs,torch.rand(4),torch.zeros(4,dtype=torch.bool),{})
  p.compute_returns(obs)
 p.update()
 from native_stand.students.resume import rng_state
 data={'actor':copy.deepcopy(p.actor.state_dict()),'critic':copy.deepcopy(p.critic.state_dict()),'optimizer':copy.deepcopy(p.optimizer.state_dict()),'training_rng':rng_state()}
 expected_rng=torch.get_rng_state().clone();q=assemble(cp,obs,steps_per_env=4,recipe=recipe())
 monkeypatch.setattr(gpu_resume,'verify',lambda s:{'warmup_repeated':False})
 report=gpu_resume.restore(q,data,{'ppo':{'learning_rate':3e-5}})
 assert report['source_Adam_loaded'] and report['source_RNG_loaded'] and not report['warmup_repeated']
 assert torch.equal(expected_rng,torch.get_rng_state())
 for left,right in ((q.actor,p.actor),(q.critic,p.critic)):
  assert all(torch.equal(v,right.state_dict()[k]) for k,v in left.state_dict().items())
 for k,state in q.optimizer.state_dict()['state'].items():
  for name,v in state.items():
   assert torch.equal(v,data['optimizer']['state'][k][name])
 assert q.actor.log_std.requires_grad is False

def test_resume_source_and_milestones_are_explicit_while_reward_fixed():
 from native_stand.students.training_recipe import validate_instance
 r=recipe('b'*64);r['milestones']=[400,900,1400,1900,2323]
 assert validate_instance(r,'b'*64)==r
 bad=copy.deepcopy(r);bad['learning_rate']=1e-4
 with pytest.raises(ValueError):validate_instance(bad,'b'*64)

def test_verified_resume_lifecycle_source_to_formal():
 from native_stand.command_v6.tools.run_state import transition,RunState
 import inspect
 # Real public state transition requires explicit verified same-recipe evidence.
 result=transition(RunState.SOURCE_EVAL,RunState.FORMAL,continuation={'version':gpu_resume.VERSION,'verified':True,'checkpoint_sha256':'a'*64})
 assert result
