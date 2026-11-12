"""R1/R2 DR real contract, shared PD, public planner and backend adapter boundaries."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest,torch
from native_stand.students import recovery_dr as dr


def test_zero_strength_makes_no_draw_and_leaves_global_rng(monkeypatch):
 state=torch.get_rng_state().clone();nstate=np.random.get_state()
 monkeypatch.setattr(dr,'_uniform',lambda *a:pytest.fail('zero strength consumed a draw'))
 out=dr.sample(12,[0,1],[10,20],[0,0])
 np.testing.assert_array_equal(out,[[.5,60,2,1],[.5,60,2,1]])
 assert torch.equal(state,torch.get_rng_state())
 assert all(np.array_equal(a,b) for a,b in zip(nstate,np.random.get_state()))


def test_independent_stream_reorder_partial_resets_and_family_isolation():
 ids=np.arange(100);ep=ids%3+1
 a=dr.sample(44,ids,ep,[1,1]);b=dr.sample(44,ids[::-1],ep[::-1],[1,1])
 np.testing.assert_array_equal(a,b[::-1])
 np.testing.assert_array_equal(a[[2]],dr.sample(44,[2],[ep[2]],[1,1]))
 np.testing.assert_array_equal(a[:,0],dr.sample(44,ids,ep,[1,0])[:,0])
 np.testing.assert_array_equal(a[:,1:],dr.sample(44,ids,ep,[0,1])[:,1:])
 assert not np.array_equal(a,dr.sample(45,ids,ep,[1,1]))
 assert (a[:,0]>=.3).all() and (a[:,0]<=.7).all()
 assert (a[:,1:]>=np.array([57,1.9,.95])-1e-6).all()
 assert (a[:,1:]<=np.array([63,2.1,1.05])+1e-6).all()
 np.testing.assert_allclose(dr.pair_friction([.3,.5,.7]),[.4,.5,.6])


def test_nominal_shared_pd_bit_exact_and_gain_broadcast_limits():
 from native_stand.command_v6.domain.contract import effort
 q=torch.zeros(4,6);dq=torch.zeros_like(q);leg=torch.ones(4,4)*.01;wheel=torch.ones(4,2)*.2
 a,b=effort(q,dq,leg,wheel)
 c,d=effort(q,dq,leg,wheel,leg_kp=torch.full((4,1),60.),leg_kd=torch.full((4,1),2.),wheel_kv=torch.ones(4,1))
 assert torch.equal(a,c) and torch.equal(b,d)
 _,d=effort(q,dq,leg*1000,wheel*1000,leg_kp=torch.full((4,1),63.),leg_kd=torch.full((4,1),2.1),wheel_kv=torch.full((4,1),1.05))
 np.testing.assert_array_equal(d.numpy(),np.tile([40,40,5.5,40,40,5.5],(4,1)))


def rows(n,passed=True):
 return [{'accepted':passed,'entry_index':40 if passed else None,'valid_hold_prefix_s':10. if passed else 0.,'dr_nominal_quality':{'mean_abs_vx':.09,'entry_s':.55}} for _ in range(n)]


def test_gate_requires_independent_update_complete_quality_and_no_skip():
 s=dr.initial_state();n=rows(32);c=rows(24);p=rows(24)
 s,r=dr.advance(s,n,c,p,250);assert s['level']==0 and s['confirmations']==1
 with pytest.raises(ValueError):dr.advance(s,n,c,p,250)
 s,r=dr.advance(s,n,c,p,500);assert s['level']==1 and s['certified_levels']==[0]
 n[0]['dr_nominal_quality']['mean_abs_vx']=None
 s,r=dr.advance(s,n,c,p,750);assert not r['gate_passed'] and not r['quality_known']
 assert dr.exposure(0,1)==[0,0] and dr.exposure(3,2)==[1,.25]
 s=dr.initial_state();s,_=dr.advance(s,rows(32,False),c,p,250);assert s['severe']==1
 s,_=dr.advance(s,rows(32),c,p,500);assert s['severe']==0
 s,_=dr.advance(s,rows(32,False),c,p,750);assert s['severe']==1


def test_runtime_adapter_writes_both_wheels_reads_back_and_only_reset_ids(monkeypatch):
 from native_stand.students.recovery_dr_runtime import Adapter
 class View:
  count=4;max_shapes=1
  def __init__(self):self.data=torch.tensor([.5,.5,0.]).reshape(1,1,3).repeat(4,1,1)
  def get_material_properties(self):return self.data.clone()
  def set_material_properties(self,data,ids):self.data[ids.long()]=data[ids.long()]
 views={s:View() for s in ('left','right')}
 sim=SimpleNamespace(create_rigid_body_view=lambda path:views['left' if 'left' in path else 'right'])
 env=SimpleNamespace(num_envs=4,device='cpu',curriculum_training=False,robot=SimpleNamespace(_physics_sim_view=sim))
 monkeypatch.setattr(Adapter,'verify_binding',lambda self:setattr(self,'bindings',[{'combine':'average'}]))
 a=Adapter(env,dr.configuration());a.configure(override={'strength':[1,1],'seed':44});a.reset(torch.tensor([3,1]))
 assert np.array_equal(a.episodes,[0,1,0,1]);assert a.params[0,0]==.5
 assert torch.equal(views['left'].data,views['right'].data)
 assert a.last_readback['readback_matches_requested']
 before=a.params.clone();a.configure(override=None);a.reset(torch.tensor([3]))
 assert a.params[3,0]==.5 and torch.equal(a.params[1],before[1])
 assert set(a.gains())=={'leg_kp','leg_kd','wheel_kv'}
 # Same API must fail on backend silently ignoring set, rather than trusting requested.
 monkeypatch.setattr(views['left'],'set_material_properties',lambda data,ids:None)
 a.configure(override={'strength':[1,0],'seed':42})
 with pytest.raises(RuntimeError,match='write/readback'):a.reset(torch.tensor([0]))


def test_public_request_contract_and_recipe_do_not_relax_old_paths():
 from native_stand.students.contact_quality import recipe
 from native_stand.students.training_recipe import validate_instance
 r=recipe(dr.SOURCE);r['critic']='source_same_semantic_contact64_v2'
 assert validate_instance(r,dr.SOURCE)==r
 s={'source_sha256':dr.SOURCE,'asset_sha256':dr.ASSET,'student_training':r,'recovery_domain_randomization':dr.configuration()}
 assert dr.validate(s)==dr.configuration()
 for mutation in ('unknown','source','teacher'):
  bad=copy.deepcopy(s)
  if mutation=='unknown':bad['recovery_domain_randomization']['anything']=True
  if mutation=='source':bad['source_sha256']='0'*64
  if mutation=='teacher':bad['recovery_course']={'teacher':True}
  with pytest.raises(ValueError):dr.validate(bad)
 # Old contact recipe still has a fresh random Critic and original fixed SHA guard.
 old=recipe();assert old['critic']=='new_random_contact64_v2';assert validate_instance(old,old['source_actor_sha256'])==old


def test_real_ppo_factory_loads_same_semantic_critic_fresh_optimizer(tmp_path):
 from native_stand.students.model import Student
 from native_stand.students.contact_quality import recipe,network
 from native_stand.students.ppo import assemble
 from test_recovery_large import candidate
 from tensordict import TensorDict
 from native_stand.recovery.controlled_ppo import KEY
 a=Student(version='multiskill_mcu_v2',network_architecture=network())
 c=Student(False,version='multiskill_mcu_v2',critic_privileged_version='recovery_contact64_v2',network_architecture=network())
 data=candidate(a.state_dict(),c.state_dict())
 data.pop('initialization_kind',None)
 data.update(network_architecture=network(),critic_privileged_version='recovery_contact64_v2',recipe_version=recipe()['version'],training_recipe=recipe(),source_sha256=recipe()['source_actor_sha256'],spec_sha256='a'*64,source_code_sha256='a'*64,BC_coefficient=0.,fixed_std=True)
 cp=tmp_path/'source.pt';torch.save(data,cp)
 obs=TensorDict({'student_actor':torch.randn(4,31),'student_critic':torch.randn(4,64),KEY:torch.ones(4,1,dtype=torch.bool)},batch_size=[4])
 r=recipe(dr.SOURCE);r['critic']='source_same_semantic_contact64_v2'
 p=assemble(cp,obs,steps_per_env=4,recipe=r)
 assert not p.optimizer.state and not p.actor.log_std.requires_grad
 for k,v in c.state_dict().items():assert torch.equal(v,p.critic.state_dict()[k])
 for k,v in a.state_dict().items():assert torch.equal(v,p.actor.state_dict()[k])


def test_real_warp_cpu_material_conversion_roundtrip():
 import warp as wp
 from native_stand.students.recovery_dr_runtime import material_tensor,set_material_tensor
 class WarpView:
  def __init__(self):self.data=wp.array(np.array([[[.5,.5,0.]],[[.5,.5,0.]]],dtype=np.float32),device='cpu')
  def get_material_properties(self):return self.data
  def set_material_properties(self,values,indices):
   assert isinstance(values,wp.array) and isinstance(indices,wp.array)
   ids=wp.to_torch(indices).long();dest=wp.to_torch(self.data);dest[ids]=wp.to_torch(values)[ids]
 v=WarpView();m=material_tensor(v).clone();m[1,0,:2]=.7
 set_material_tensor(v,m,torch.tensor([1],dtype=torch.int32))
 assert torch.equal(material_tensor(v),m)
 # Discrete reusable friction buckets avoid unbounded PhysX material creation.
 draw=dr.sample(42,np.arange(10000),np.ones(10000,dtype=int),[1,0])
 assert len(np.unique(draw[:,0]))<=64
