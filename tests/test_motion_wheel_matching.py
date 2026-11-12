"""Explicit wheel profile, actuator readback, recipe and legacy regression."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy
from types import SimpleNamespace as S
import pytest
import torch
from native_stand.students import motion_wheel_matching as m


def test_explicit_configuration_does_not_change_legacy_defaults():
 from native_stand.students.motion_antioscillation import configuration,validate_configuration,reward_snapshot
 c=m.configuration();assert c['wheel_kv']==.2 and configuration()['wheel_kv']==1.
 assert 'wheel_execution' not in configuration()
 assert validate_configuration(c) is c
 assert reward_snapshot(c)['configuration']['wheel_execution']==m.profile()
 for name,value in [('wheel_kv',.5),('unknown',1)]:
  bad=copy.deepcopy(c);bad[name]=value
  with pytest.raises(ValueError):validate_configuration(bad)
 bad=copy.deepcopy(c);bad['wheel_execution']['wheel_extra_armature_kg_m2_ch2_ch5'][0]+=.01
 with pytest.raises(ValueError):validate_configuration(bad)


def test_setup_uses_named_binding_and_rejects_effective_damping():
 from test_motion_timing import fake_env
 e=fake_env(2);e.ids=[5,0,4,2,1,3];e.cfg=S(wheel_kv=.2)
 a=torch.full((2,6),.01);damping=torch.zeros_like(a)
 def write(*,armature,joint_ids):a[:,joint_ids.long()]=armature
 e.robot.write_joint_armature_to_sim_index=write
 e.robot.root_view=S(get_dof_armatures=lambda:a,get_dof_dampings=lambda:damping)
 old=a.clone();r=m.setup(e,m.profile())
 assert r['channels']==[2,5] and r['geometric_sides']==['right','left']
 torch.testing.assert_close(a[:,[4,3]],torch.tensor(m.profile()['wheel_extra_armature_kg_m2_ch2_ch5']).expand(2,-1))
 assert torch.equal(a[:,[5,0,2,1]],old[:,[5,0,2,1]])
 damping[:,4]=.1
 with pytest.raises(RuntimeError,match='damping'):m.setup(e,m.profile())


def test_timing_kv_and_wheel_inertias_persist_across_leg_course_reset():
 from test_motion_timing import fake_env
 from native_stand.students.motion_timing_runtime import Timing
 e=fake_env(8);e.cfg=S(leg_kp=40.,leg_kd=2.,wheel_kv=.2)
 a=torch.full((8,6),.01)
 def write(*,armature,joint_ids):a[:,joint_ids.long()]=armature
 e.robot.write_joint_armature_to_sim_index=write
 e.robot.root_view=S(get_dof_armatures=lambda:a,get_dof_dampings=lambda:torch.zeros_like(a))
 rng=torch.get_rng_state().clone();t=Timing(e,m.configuration(),0)
 t.requested[:,[2,5]]=2.
 for _ in range(16):t.apply()
 torch.testing.assert_close(e.effort[:,[2,5]],torch.full((8,2),.4))
 wheels=a[:,[2,5]].clone()
 for stage in range(5):
  t.configure(stage);t.reset(torch.arange(8));assert torch.equal(a[:,[2,5]],wheels)
 assert torch.equal(rng,torch.get_rng_state())
 assert t.summary()['wheel_execution_readback']['wheel_kv']==.2
 e.cfg.wheel_kv=1.
 with pytest.raises(ValueError):Timing(e,m.configuration(),0)


def test_new_recipe_keeps140d_schema_and_exact_reward():
 from native_stand.students.motion_training import declaration
 from native_stand.students.motion_tracking_reward import PARALLEL_VERSION
 from native_stand.students.motion_antiosc_privileged import network,VERSION
 from native_stand.students.motion_timing import validate_recipe
 r=declaration('a'*64,0,reward_version=PARALLEL_VERSION)
 r.update(course_version='motion31_standing_antiosc_armature_v4',course_stage=0,timing_configuration=m.configuration(),
  critic_privileged_version=VERSION,network_architecture=network(),critic='full_motion140_wheel_match',optimizer='source_Adam_full_state',
  checkpoint_every_updates=100,milestones=list(range(250,10000,250)))
 old=copy.deepcopy(r);assert validate_recipe(r,'a'*64) is r and r==old
 r['optimizer']='fresh_Adam'
 with pytest.raises(ValueError):validate_recipe(r,'a'*64)
