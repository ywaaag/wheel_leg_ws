FROZEN_TEST_SCOPE='PORTABLE'
import copy
from types import SimpleNamespace as S
import pytest
import torch
from native_stand.students import motion_antiosc_privileged as p
from native_stand.students.model import Student


def test_feature_layout_preserves_old_prefix_and_signed_units():
 b=torch.arange(220,dtype=torch.float32).reshape(2,110);a=torch.tensor([[.35],[.012]]);f=torch.ones(2,16)*2
 request=torch.tensor([[40.,-40.,5.5,-40.,40.,-5.5]]).expand(2,-1);previous=request*.5;mix=torch.tensor([[0.],[.75]])
 r=p.critic_input(b,a,f,request,previous,mix)
 assert r.shape==(2,140) and torch.equal(r[:,:110],b)
 assert torch.equal(r[:,111:127],f) and torch.equal(r[:,127:133],request/torch.tensor([40,40,5.5,40,40,5.5]))
 assert torch.equal(r[:,133:139],r[:,127:133]*.5) and torch.equal(r[:,139:],mix)
 assert r[0,110]==1 and r[1,110]==pytest.approx(.012/.35)
 with pytest.raises(ValueError):p.critic_input(b,a,f[:,:15],request,previous,mix)
 with pytest.raises(ValueError):p.critic_input(b,a.double(),f,request,previous,mix)


def source():
 from native_stand.students.motion_timing import network
 actor=Student(True,version='multiskill_mcu_v2',network_architecture=network())
 critic=Student(False,version='multiskill_mcu_v2',network_architecture=network(),critic_privileged_version='motion_timing_privileged110_v1')
 opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=3e-5)
 (actor.mean(torch.ones(2,31)).sum()+critic(torch.ones(2,110)).sum()).backward();opt.step()
 return actor,critic,{'actor':actor.state_dict(),'critic':critic.state_dict(),'optimizer':opt.state_dict(),'network_architecture':network(),'critic_privileged_version':'motion_timing_privileged110_v1'}


def test_zero_column_migration_preserves_actor_value_and_all_old_moments():
 actor,critic,s=source();old=copy.deepcopy(s);d=p.expand_source(s)
 big=Student(False,version='multiskill_mcu_v2',network_architecture=p.network(),critic_privileged_version=p.VERSION);big.load_state_dict(d['critic'])
 x=torch.randn(8,110);extra=torch.randn(8,30)*20
 torch.testing.assert_close(big(torch.cat((x,extra),-1)),critic(x),atol=1e-6,rtol=1e-5)
 for k,v in s['actor'].items():assert torch.equal(v,d['actor'][k]) and torch.equal(v,old['actor'][k])
 assert not d['critic']['mlp.0.weight'][:,110:].count_nonzero()
 for pid,state in old['optimizer']['state'].items():
  for k,v in state.items():
   got=d['optimizer']['state'][pid][k]
   if v.shape==(512,110):assert torch.equal(got[:,:110],v) and not got[:,110:].count_nonzero()
   else:assert torch.equal(got,v)
 assert p.expand_source(d) is d
 bad=copy.deepcopy(s);bad['critic_privileged_version']='motion_symmetry_privileged85_v1'
 with pytest.raises(ValueError):p.expand_source(bad)
 with pytest.raises(ValueError):Student(False,version='multiskill_mcu_v2',network_architecture=p.network(),critic_privileged_version='motion_timing_privileged110_v1')
 with pytest.raises(ValueError):Student(False,version='multiskill_mcu_v2',critic_privileged_version=p.VERSION)


def test_runtime_true_frame_delayed_actor_and_actual_armature_are_separate():
 from test_motion_timing import fake_env
 from native_stand.students.motion_timing_runtime import subclass
 from native_stand.students.motion_antioscillation import configuration
 class Base:
  def __init__(self):
   self.__dict__.update(fake_env(2).__dict__);self.cfg=S(leg_kp=40.,leg_kd=2.,wheel_kv=1.)
   self.critic_privileged_version=p.VERSION;self.robot.joint_names=list('abcdef')
   self.armature=torch.tensor([[.35,.35,.01,.35,.35,.01]]).expand(2,-1).clone()
   def write(*,armature,joint_ids):self.armature[:,joint_ids.long()]=armature
   self.robot.write_joint_armature_to_sim_index=write;self.robot.root_view=S(get_dof_armatures=lambda:self.armature)
   self.cache={'student_actor':torch.zeros(2,31),'student_critic':torch.zeros(2,85)}
  def _get_observations(self):return self.cache
  def _reset_idx(self,ids):pass
 e=subclass(Base,configuration(),0)();e._get_observations();e.motion_timing.configure(4)
 e.motion_timing.tick=10;e.cache['student_actor'][:,:16]=2.
 e.effort_pre[:]=5.5;e.motion_timing.prev_torque[:]=1.;e.motion_timing.critic_previous_torque[:]=.5
 r=e._get_observations();assert r['student_critic'].shape==(2,140)
 assert (r['student_actor'][:,:16]==0).all() and (r['student_critic'][:,111:127]==2).all()
 assert torch.equal(r['student_critic'][:,:31],r['student_actor'])
 assert torch.allclose(r['student_critic'][:,110],torch.full((2,),.012/.35))
 assert r['student_critic'][0,135]==pytest.approx(1/5.5) # pre-reward terminal call sees previous actual, not older cache.
 e.motion_timing.last_reward_tick=10
 r=e._get_observations();assert r['student_critic'][0,135]==pytest.approx(.5/5.5)
 e.motion_timing.reset(torch.tensor([0]));assert (e.motion_timing.critic_previous_torque[0]==0).all() and (e.motion_timing.critic_previous_torque[1]==.5).all()
 assert (e.cache['student_actor'][:,:16]==2).all()


def test_value_diagnostics_grouped_by_armature_metadata():
 from native_stand.students.motion_large_combo import value_diagnostics
 from native_stand.recovery.controlled_ppo import KEY
 s=S(observations={KEY:torch.ones(4,1,dtype=torch.bool),'motion_inertia_cohort':torch.tensor([[0],[0],[1],[1]])},
  returns=torch.tensor([[1.],[2.],[3.],[3.]]),values=torch.tensor([[1.],[2.],[2.],[2.]]),advantages=torch.ones(4,1))
 r=value_diagnostics(s,commands_override=[.35,.11],cohort_key='motion_inertia_cohort')
 assert [v['samples'] for v in r]==[2,2] and r[0]['explained_variance']==1 and r[1]['explained_variance'] is None
 assert r[1]['value_error_MSE']==1
