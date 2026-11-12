FROZEN_TEST_SCOPE='PORTABLE'
import copy
from types import SimpleNamespace as S
import pytest
import torch
from test_motion_timing import fake_env
from native_stand.students.motion_fixed_timing import configuration,apply_actuator,verify_armature,cases,refinement_records,advance,build
from native_stand.students.motion_timing_runtime import Timing

def test_fixed_delays_all_envs_resets_overrides_and_pd_are_effective():
 e=fake_env(8);e.cfg=S(leg_kp=40.,leg_kd=2.,wheel_kv=1.)
 rng=torch.get_rng_state().clone();t=Timing(e,configuration(),0)
 for override in (None,-1,3):
  t.configure(override);t.reset(torch.arange(8))
  assert torch.equal(t.delays,torch.tensor([10,10,7,10,10,11]).expand(8,-1))
  assert (t.feedback==10).all() and torch.equal(t.tau,torch.tensor([5,5,0,5,5,0]).expand(8,-1))
 assert torch.equal(rng,torch.get_rng_state())
 t.requested[:,[0,1,3,4]]=.1
 for _ in range(11):t.apply()
 assert torch.allclose(e.effort_pre[:,[0,1,3,4]],torch.full((8,4),4.))
 assert (e.effort[:,[0,1,3,4]]>0).all() and (e.effort[:,[0,1,3,4]]<4).all()
 assert t.summary()['leg_kp']==40
 e.cfg.leg_kp=60
 with pytest.raises(ValueError):Timing(e,configuration(),0)


def test_armature_only_four_legs_and_no_default_mutation():
 from native_stand.contract import JOINT_NAMES,LEG_COLUMNS
 groups={k:S(joint_names_expr=[JOINT_NAMES[i] for i in cols],armature=value) for k,cols,value in
         [('leg_motors',LEG_COLUMNS,.01),('wheel_motors',[2,5],.003),('passive_linkage',[],.004)]}
 robot=S(actuators=groups);cfg=S(robot_cfg=robot)
 s={'motion_timing':{'configuration':configuration()},'control':{'leg_kp':40.,'leg_kd':2.,'wheel_kv':1.}}
 apply_actuator(cfg,s)
 assert cfg.robot_cfg.actuators['leg_motors'].armature==.35 and robot.actuators['leg_motors'].armature==.01
 assert cfg.robot_cfg.actuators['wheel_motors'].armature==.003 and cfg.robot_cfg.actuators['passive_linkage'].armature==.004
 cfg2=S(robot_cfg=robot);apply_actuator(cfg2,{})
 assert cfg2.robot_cfg is robot
 bad=copy.deepcopy(s);bad['motion_timing']['configuration']['leg_armature_kg_m2']=.11
 with pytest.raises(ValueError):apply_actuator(S(robot_cfg=robot),bad)
 robot.actuators['leg_motors'].joint_names_expr.append(JOINT_NAMES[2])
 with pytest.raises(ValueError):apply_actuator(S(robot_cfg=robot),s)


def test_backend_armature_readback_not_config_substitution():
 e=fake_env();e.cfg=S(leg_kp=40.,leg_kd=2.);e.robot.joint_names=list('abcdef')
 values=torch.tensor([[.35,.35,.01,.35,.35,.01]])
 e.robot.root_view=S(get_dof_armatures=lambda:values)
 assert verify_armature(e)['leg_kp']==40
 values[0,3]=.01
 with pytest.raises(RuntimeError):verify_armature(e)


def test_single_target_condition_and_unknown_never_passes():
 r=cases({'student_evaluation':{'cases':[{}]}})[0]
 rows=[{'case':r,'per_env':[{'env':i,'accepted':True} for i in range(8)]}]
 assert len(refinement_records(rows,0)[0]['per_env'])==8
 assert refinement_records(rows*3,0)[0]['per_env']==[]
 state={'stage':0,'last_evaluation_update':-1,'consecutive_passes':0,'accepted_by_stage':{}}
 c={'stage':0,'update':250,'complete_matrix':True,'evaluations':rows,'checkpoint':'fixture','sha256':'a'*64}
 state,d=advance(state,c);assert not d['stop'] and state['stage']==0
 state,d=advance(state,{**c,'update':500});assert d['target_condition_confirmed'] and not d['stop']
 with pytest.raises(ValueError):advance(state,{**c,'update':500})
 state,d=advance(state,{**c,'update':750,'evaluations':[]});assert not d['target_condition_confirmed']
