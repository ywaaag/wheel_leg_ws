"""Standing-only reward, signed rigid kinematics, actual Critic/PPO boundary."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy,sys
from types import SimpleNamespace
import pytest,torch
from native_stand.students import contact_quality as q
from test_recovery_broad_narrow_reward import inputs

def measured(n=3,speed=0.):
 zero=torch.zeros(n,2,3);center=zero.clone();center[...,2]=.06
 vel=zero.clone();vel[...,0]=speed;axis=zero.clone();axis[...,1]=1
 omega=zero.clone();omega[...,1]=speed/.06
 return q.point_velocity(center,center,vel,omega,axis)

def test_signed_pure_rolling_reverse_locked_airborne_and_com_offset():
 for speed in (-1.,1.):
  r=measured(speed=speed);torch.testing.assert_close(r['tangent_velocity_w'],torch.zeros(3,2,2),atol=1e-6,rtol=0)
  n=1;center=torch.tensor([[[0.,0.,.06],[0.,0.,.06]]]);v=torch.zeros_like(center);v[...,0]=speed
  axis=torch.zeros_like(center);axis[...,1]=1;omega=torch.zeros_like(center);omega[...,1]=speed/.06
  wrong=q.point_velocity(center,center,v,-omega,axis)
  torch.testing.assert_close(wrong['tangent_velocity_w'][...,0],torch.full((1,2),2*speed))
  locked=q.point_velocity(center,center,v,torch.zeros_like(v),axis)
  torch.testing.assert_close(locked['tangent_velocity_w'][...,0],torch.full((1,2),speed))
  offset=center+center.new_tensor([.01,.02,0.]);vcom=v+torch.cross(omega,offset-center,dim=-1)
  shifted=q.point_velocity(center,offset,vcom,omega,axis)
  torch.testing.assert_close(shifted['tangent_velocity_w'],torch.zeros(1,2,2),atol=1e-6,rtol=0)
 vertical=torch.zeros(3,2,3);vertical[...,2]=1
 z=torch.zeros_like(vertical)
 assert not q.point_velocity(z,z,z,z,vertical)['geometry_valid'].any()
 with pytest.raises(ValueError,match='finite'):q.point_velocity(z,z,z+float('nan'),z,vertical)

def test_reward_changes_only_after_entry_dt_masks_contact_and_slip_disabled():
 from native_stand.students.recovery_large_training import reward as parent,configuration
 x=inputs();x['executed_targets'][:,[2,5]]=10;x['velocity_vx_vy_yaw'][:,0]=.4
 elapsed=torch.ones(3);m=measured()
 old=parent(**x,entry_elapsed=elapsed,reward_configuration=configuration())
 rng=torch.get_rng_state().clone();r=q.reward(**x,entry_elapsed=elapsed,measurement=m,reward_configuration=q.configuration())
 torch.testing.assert_close(old['total'],r['total']);assert torch.equal(rng,torch.get_rng_state())
 assert r['fields'][-3:]==q.FIELDS and r['raw'].shape==(3,16)
 x['entered'][:]=True;x['contacts'][1]=0
 a=q.reward(**x,entry_elapsed=torch.zeros(3),measurement=m,reward_configuration=q.configuration())
 assert a['raw'][:,-3:].sum()==0
 r=q.reward(**x,entry_elapsed=elapsed,measurement=m,reward_configuration=q.configuration())
 assert r['weighted_density'][0,-3].item()==pytest.approx(-.2)
 assert r['weighted_density'][1,-2].item()==pytest.approx(-.5)
 # Slip raw is retained diagnostically but reward cannot silently enable it.
 slipping={**m,'tangent_velocity_w':torch.ones(3,2,2)*2}
 b=q.reward(**x,entry_elapsed=elapsed,measurement=slipping,reward_configuration=q.configuration())
 torch.testing.assert_close(b['total'],r['total']);assert b['raw'][0,-1]>0 and b['raw'][1,-1]==0
 assert (b['weighted_density'][:,-1]==0).all()
 torch.testing.assert_close(r['total'],r['continuous_dt'].sum(-1)+r['events'].sum(-1))
 c=q.configuration();c['contact_quality']['weights']['post_entry_slip']=-.2
 with pytest.raises(ValueError,match='differs'):q.reward(**x,entry_elapsed=elapsed,measurement=m,reward_configuration=c)
 x['controlled'][:]=False
 assert q.reward(**x,entry_elapsed=elapsed,measurement=m,reward_configuration=q.configuration())['total'].sum()==0

def test_critic_actual_forward_actor_isolated_and_old64_strict():
 from native_stand.students.model import Student
 m=measured();base=torch.randn(3,64);extra=q.critic_input(base,m,torch.ones(3,2))
 assert extra.shape==(3,64);torch.testing.assert_close(extra[:,:52],base[:,:52]);assert not torch.equal(extra[:,52:],base[:,52:])
 model=Student(False,version='multiskill_mcu_v2',critic_privileged_version=q.CRITIC,network_architecture=q.network())
 assert model(extra).shape==(3,1)
 with pytest.raises(ValueError,match='contact64'):Student(False,version='multiskill_mcu_v2',critic_privileged_version='recovery_privileged64_v1',network_architecture=q.network())
 assert q.contract()['actor_dim']==31 and q.contract()['critic_dim']==64

def test_phase_metrics_do_not_mix_initialization_or_reset_target(monkeypatch):
 from native_stand.students.standing_phase_metrics import PhaseMetrics
 x=inputs(5);args=dict(elapsed=torch.tensor([0.,0.,.5,1.5,3.]),entered=torch.tensor([False,False,True,True,True]),controlled=torch.tensor([False,True,True,True,True]),contacts=x['contacts'],angles=x['angles'],velocity=x['velocity_vx_vy_yaw'],previous=x['previous_targets'],executed=x['executed_targets'],requested_torque=x['applied_torque'],clipped_torque=x['applied_torque'],applied_torque=x['applied_torque'])
 a=PhaseMetrics('cpu');a.observe(**args,measurement=measured(5));r=a.result()
 assert [p['controlled_samples'] for p in r['phases'].values()]==[1,1,1,1]
 args['clipped_torque'][:,2]=5.5;a=PhaseMetrics('cpu');a.observe(**args)
 assert a.result()['phases']['pre_entry']['channel_limit_fraction'][2]==1
 assert a.result()['phases']['entry_2_10s']['contact_point_velocity_xy_rms_mps'] is None

def test_runtime_reader_uses_explicit_com_fields_and_total_body_omega(monkeypatch):
 n=2;center=torch.zeros(n,2,3);center[...,2]=.06;velocity=torch.zeros_like(center);velocity[...,1]=-.6
 omega=torch.zeros_like(center);omega[...,0]=10
 data={name:SimpleNamespace(torch=value) for name,value in dict(body_link_pos_w=center,body_com_pos_w=center,body_com_lin_vel_w=velocity,body_com_ang_vel_w=omega,body_link_quat_w=torch.tensor([0.,0.,0.,1.]).expand(n,2,4)).items()}
 data['body_com_ang_vel_w']=SimpleNamespace(torch=torch.cat((omega,torch.zeros_like(omega)),1))
 data['joint_vel']=SimpleNamespace(torch=torch.tensor([0.,0.,10.,0.,0.,10.]).expand(n,-1))
 env=SimpleNamespace(ids=list(range(6)),wheel_body_ids=[0,1],robot=SimpleNamespace(body_names=['left_wheel','right_wheel','left_lower_leg','right_lower_leg'],data=SimpleNamespace(**data)))
 monkeypatch.setitem(sys.modules,'isaaclab.utils.math',SimpleNamespace(quat_apply=lambda quat,v:v))
 r=q.runtime_measurement(env);torch.testing.assert_close(r['tangent_velocity_w'],torch.zeros(n,2,2),atol=1e-6,rtol=0)
 delattr(env.robot.data,'body_com_lin_vel_w')
 with pytest.raises(ValueError,match='missing'):q.runtime_measurement(env)

def test_real_ppo_factory_transfers_only_actor_and_new_contact_critic(tmp_path):
 from native_stand.students.model import Student
 from native_stand.students.recovery_large import expand
 from native_stand.students.ppo import assemble
 from test_recovery_large import candidate
 from tensordict import TensorDict
 from native_stand.recovery.controlled_ppo import KEY
 a=Student(version='multiskill_mcu_v2');c=Student(False,version='multiskill_mcu_v2',critic_privileged_version='recovery_privileged64_v1')
 sa,sc=expand(a.state_dict(),c.state_dict());cp=tmp_path/'source.pt';torch.save(candidate(sa,sc),cp)
 obs=TensorDict({'student_actor':torch.randn(4,31),'student_critic':torch.randn(4,64),KEY:torch.ones(4,1,dtype=torch.bool)},batch_size=[4])
 p=assemble(cp,obs,steps_per_env=4,recipe=q.recipe())
 assert p.critic.network_architecture==q.network() and p.critic.mlp[0].in_features==64
 assert all(torch.equal(v,p.actor.state_dict()[k]) for k,v in sa.items())
 assert any(not torch.equal(v,p.critic.state_dict()[k]) for k,v in sc.items())
 assert not p.optimizer.state and not p.actor.log_std.requires_grad
 before=copy.deepcopy(p.actor.state_dict());p.actor_update_enabled=False
 def update(valid):
  obs[KEY][:]=valid
  with torch.inference_mode():
   for i in range(4):p.act(obs);p.process_env_step(obs,torch.rand(4),torch.zeros(4,dtype=torch.bool),{})
   p.compute_returns(obs)
  p.update()
 update(True);assert all(torch.equal(v,p.actor.state_dict()[k]) for k,v in before.items())
 p.actor_update_enabled=True;update(False);assert all(torch.equal(v,p.actor.state_dict()[k]) for k,v in before.items())
 update(True);assert any(not torch.equal(v,p.actor.state_dict()[k]) for k,v in before.items())

def test_actual_runtime_post_physics_reward_and_contact_critic_getter(monkeypatch):
 import importlib
 from native_stand.students.recovery_task import Certificate
 from native_stand.students.recovery_privileged import HoldExtrema
 monkeypatch.setitem(sys.modules,'native_stand.recovery.runtime',SimpleNamespace(RecoveryEnv=object))
 monkeypatch.delitem(sys.modules,'native_stand.students.runtime',raising=False)
 mod=importlib.import_module('native_stand.students.runtime')
 monkeypatch.setattr(mod.StudentEnv,'_get_dones',lambda self:(torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool)))
 monkeypatch.setattr(q,'runtime_measurement',lambda e:measured(2))
 e=mod.TrainingStudentEnv.__new__(mod.TrainingStudentEnv);e.task_ready=True;e.num_envs=2;e.device='cpu';e.ids=list(range(6));e.student_reward_configuration=q.configuration()
 e.rise_hold=Certificate(2,'cpu');e.rise_hold.entry[:]=0;e.rise_hold.count[:]=200
 e._motion_state=lambda:(torch.zeros(2,3),torch.tensor([[0.,-1.,0.]]).expand(2,-1))
 e.vx=e.vy=e.yaw=torch.zeros(2);e.student_h0=torch.ones(2)*.168;e.student_actor_valid=torch.ones(2,dtype=torch.bool)
 e.student_previous_physical=torch.zeros(2,6);e.executed_leg=torch.zeros(2,4);e.executed_wheel=torch.ones(2,2)*10
 e.recovery_snapshot=torch.zeros(2,39);e.recovery_snapshot[:,6]=.34;e.recovery_snapshot[:,10:12]=3;e.recovery_snapshot[:,13]=.03;e.recovery_snapshot[:,29]=1
 e.standing_snapshot=torch.zeros(2,10);e.recovery_hold_extrema=HoldExtrema(2,'cpu')
 e.robot=SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.zeros(2,6)),applied_torque=SimpleNamespace(torch=torch.zeros(2,6))))
 e._get_dones()
 assert e.student_reward_result['raw'].shape==(2,16)
 torch.testing.assert_close(e.student_reward_result['weighted_density'][:,-3],torch.full((2,),-.2))
 assert e.contact_measurement_post['geometry_valid'].all()
 # Actual subclass getter consumes measurements into Critic, not only metadata.
 monkeypatch.delitem(sys.modules,'native_stand.students.contact_runtime',raising=False)
 adapter=importlib.import_module('native_stand.students.contact_runtime')
 base=torch.randn(2,64);actor=base[:,:31].clone();e.student_ready=True;e.critic_privileged_version=q.CRITIC
 monkeypatch.setattr(mod.TrainingStudentEnv,'_get_observations',lambda self:{'student_actor':actor,'student_critic':base})
 linked=adapter.ContactStudentEnv.__new__(adapter.ContactStudentEnv);linked.__dict__.update(e.__dict__)
 result=linked._get_observations()
 assert torch.equal(result['student_actor'],actor)
 assert torch.equal(result['student_critic'][:,:52],base[:,:52])
 assert not torch.equal(result['student_critic'][:,52:],base[:,52:])
 monkeypatch.delitem(sys.modules,'native_stand.students.contact_runtime',raising=False)
 monkeypatch.delitem(sys.modules,'native_stand.students.runtime',raising=False)

def test_independent_reset_observations_have_real_contact64_fields(monkeypatch):
 import importlib
 from native_stand.students.interface import History
 from native_stand.students.mcu_interface import PolicyClock
 from native_stand.students.recovery_task import Certificate
 from native_stand.students.recovery_privileged import HoldExtrema
 from native_stand.recovery.domain import Mode
 class Base:
  def _get_observations(self):return {}
 monkeypatch.setitem(sys.modules,'native_stand.recovery.runtime',SimpleNamespace(RecoveryEnv=Base))
 monkeypatch.setitem(sys.modules,'isaaclab.utils.math',SimpleNamespace(quat_apply_inverse=lambda q,v:v,quat_apply=lambda q,v:v))
 monkeypatch.delitem(sys.modules,'native_stand.students.runtime',raising=False);monkeypatch.delitem(sys.modules,'native_stand.students.contact_runtime',raising=False)
 cls=importlib.import_module('native_stand.students.contact_runtime').ContactStudentEnv
 e=cls.__new__(cls);e.num_envs=2;e.device='cpu';e.student_ready=e.task_ready=True;e.interface_version='multiskill_mcu_v2'
 e.student_role='RecoveryStudent';e.student_command_height=.34;e.step_dt=.01;e.ids=list(range(6));e.wheel_body_ids=[0,1];e.critic_privileged_version=q.CRITIC
 e.fsm=SimpleNamespace(mode=torch.full((2,),int(Mode.INITIALIZING)))
 e.reference=torch.zeros(2,6);e.executed_leg=torch.zeros(2,4);e.executed_wheel=torch.zeros(2,2);e.student_history=History(2);e.mcu_clock=PolicyClock(2);e.episode_length_buf=torch.zeros(2,dtype=torch.long)
 e.student_obs=None;e.student_h0=torch.zeros(2);e.root_height=torch.ones(2)*.168;e.contact_force=torch.zeros(2,2)
 e.recovery_config={'standing_task':{'hip_anchors_root_m':[[0.,0.,0.],[0.,0.,0.]]}}
 e.recovery_length_valid=torch.zeros(2,dtype=torch.bool);e.recovery_length_time=torch.zeros(2,dtype=torch.float64);e.recovery_previous_length=torch.zeros(2,2);e.recovery_length_rate=torch.zeros(2,2)
 e.rise_hold=Certificate(2,'cpu');e.recovery_hold_extrema=HoldExtrema(2,'cpu')
 z=torch.zeros(2,2,3);quat=torch.tensor([0.,0.,0.,1.]).expand(2,2,4)
 values=dict(joint_pos=torch.zeros(2,6),joint_vel=torch.zeros(2,6),root_pos_w=torch.zeros(2,3),root_quat_w=torch.zeros(2,4),root_lin_vel_w=torch.zeros(2,3),body_pos_w=z,body_link_pos_w=z,body_link_quat_w=quat,body_com_pos_w=z,body_com_lin_vel_w=z,body_com_ang_vel_w=z,applied_torque=torch.zeros(2,6))
 values['body_com_ang_vel_w']=torch.zeros(2,4,3)
 e.robot=SimpleNamespace(body_names=['left_wheel','right_wheel','left_lower_leg','right_lower_leg'],data=SimpleNamespace(**{k:SimpleNamespace(torch=v) for k,v in values.items()}));e._motion_state=lambda:(torch.zeros(2,3),torch.tensor([[0.,-1.,0.]]).expand(2,-1));e._geometry_gaps=lambda:(torch.zeros(2,1),torch.ones(2)*.012)
 obs=e._get_observations();assert obs['student_actor'].shape==(2,31) and obs['student_critic'].shape==(2,64)
 assert (obs['student_critic'][:,-2:]==1).all()
 first=obs['student_critic'].clone();e._get_observations();torch.testing.assert_close(e.student_obs['student_critic'],first)
 e.mcu_clock.reset(torch.tensor([0]));e.student_history.reset(torch.tensor([0]));e.recovery_length_valid[0]=False
 e._get_observations();assert not e.recovery_length_rate.any()
 monkeypatch.delitem(sys.modules,'native_stand.students.contact_runtime',raising=False);monkeypatch.delitem(sys.modules,'native_stand.students.runtime',raising=False)

def test_source_readback_gate_reconstructs_and_rejects_preentry_cost(tmp_path):
 import json,numpy as np
 from native_stand.students.contact_readback import verify_records
 from native_stand.students.data import sha
 records=[]
 for i in range(4):
  vectors=np.zeros((3,8,2,3),np.float32)
  density=np.zeros((3,8,16),np.float32);events=np.zeros((3,8,3),np.float32)
  arrays={'wheel_contact_parent_angular_velocity_w':vectors,'wheel_contact_joint_spin_rad_s':np.zeros((3,8,2),np.float32),'wheel_contact_relative_spin_rad_s':np.zeros((3,8,2),np.float32),'wheel_contact_axle_w':np.tile([1.,0.,0.],(3,8,2,1)),'wheel_contact_com_velocity_w':vectors,'wheel_contact_angular_velocity_w':vectors,'wheel_contact_support_point_w':vectors,'wheel_contact_com_w':vectors,'wheel_contact_velocity_w':vectors,'wheel_contact_geometry_valid':np.ones((3,8,2),bool),'contact_entered':np.array([False,True,True])[:,None,None].repeat(8,axis=1),'contact_reward_weighted_density':density,'contact_reward_dt':density*.01,'contact_reward_events':events,'contact_reward_total':np.zeros((3,8)),'actor_valid':np.ones((3,8,1),bool),'first_episode_valid':np.ones((3,8),bool)}
  path=tmp_path/f'trace_{i:02}.npz';np.savez_compressed(path,**arrays)
  row={'case':{'case_id':str(i)},'per_env':[{'accepted':True} for k in range(8)],'trace':{'path':path.name,'sha256':sha(path)}}
  p=tmp_path/f'evaluation_{i:02}.json';p.write_text(json.dumps(row));records.append({'path':p.name,'sha256':sha(p),'case':row['case']})
 assert verify_records(tmp_path,records)['status']=='PASSED'
 arrays['contact_reward_weighted_density'][0,:,-3]=-1
 np.savez_compressed(path,**arrays);row['trace']['sha256']=sha(path);p.write_text(json.dumps(row));records[-1]['sha256']=sha(p)
 with pytest.raises(ValueError,match='pre-entry'):verify_records(tmp_path,records)
