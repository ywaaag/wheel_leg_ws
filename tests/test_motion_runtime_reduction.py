"""Actual collider math/initialization predicates/state epochs, no Isaac launch."""
FROZEN_TEST_SCOPE='PORTABLE'
from types import SimpleNamespace as NS
import ast
from pathlib import Path
import pytest
import torch
from native_stand.students.motion_runtime_reduction import GeometryGaps,settled_initial_subset,diagnostic_snapshot


def test_real_performance_boundary_requires_explicit_motion_hotpath_and_keeps_old_default():
 from native_stand.students.performance import settings
 from native_stand.students.motion_hotpath import readback
 spec={'student_training':{'version':'student_motion31_training_v1'},'performance':{'version':'student_performance_v2','motion_runtime_reduction':True}}
 with pytest.raises(ValueError,match='explicit hotpath'):settings(spec)
 spec['performance'].update(motion_hotpath=True,combined_step_checks=True)
 assert settings(spec)['motion_runtime_reduction']
 spec['student_training']['version']='student_recovery_broad_narrow_ppo_v1'
 with pytest.raises(ValueError,match='explicit Motion'):settings(spec)
 assert not settings({'performance':{'version':'student_performance_v2'}})['motion_runtime_reduction']
 env=NS(cfg=NS(motion_hotpath=True,motion_runtime_reduction=True,scene=NS(contact_forces=NS(history_length=3))),sensor=NS(cfg=NS(history_length=3),body_names=['robot','left_wheel','right_wheel']),wheel_sensor_ids=[1,2],forbidden_sensor_ids=[0])
 assert 'not evaluated' in readback(env)['settled_diagnostic_scope']


def test_new_defaults_and_parent_compatibility_keep_explicit_fallback_and_legacy_profiles():
 from native_stand.students.performance import new_motion_defaults,parent_motion_performance_matches,settings
 old={'version':'student_performance_v1'}
 new=new_motion_defaults(old)
 assert new['motion_hotpath'] and new['motion_runtime_reduction']
 assert parent_motion_performance_matches(new,old)
 legacy=dict(new);legacy.pop('motion_hotpath');legacy.pop('motion_runtime_reduction')
 assert parent_motion_performance_matches(legacy,old)
 fallback={**new,'motion_hotpath':False,'motion_runtime_reduction':False}
 assert parent_motion_performance_matches(fallback,old)
 assert not settings({'performance':legacy,'student_training':{'version':'student_motion31_training_v1'}})['motion_runtime_reduction']
 assert not parent_motion_performance_matches({**new,'combined_step_checks':False},old)


def inverse(q,v):
 # Quaternion native math exercised with nonzero unit xyzw input.
 xyz=q[...,:3];w=q[...,3:4]
 return v-2*w*torch.cross(xyz,v,dim=-1)+2*torch.cross(xyz,torch.cross(xyz,v,dim=-1),dim=-1)


def old_gaps(env):
 tree=ast.parse(Path('src/native_stand/recovery/runtime.py').read_text());cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and any(isinstance(n,ast.FunctionDef) and n.name=='_geometry_gaps' for n in c.body));f=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_geometry_gaps');ns={'torch':torch,'quat_apply_inverse':inverse}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),'actual_recovery_geometry','exec'),ns)
 return ns['_geometry_gaps'](env)


def test_batched_all_collider_types_nonzero_pose_match_actual_original_loop():
 torch.manual_seed(42);n=7;geometry=[]
 for i,kind in enumerate(['Cube','Sphere','Capsule','Cylinder']):
  matrix=torch.eye(4);a=.2*(i+1);matrix[:2,:2]=torch.tensor([[torch.cos(torch.tensor(a)),-torch.sin(torch.tensor(a))],[torch.sin(torch.tensor(a)),torch.cos(torch.tensor(a))]]);matrix[:3,3]=torch.rand(3)*.05
  shape={'type':kind,'size':.1,'radius':.06,'height':.12,'axis':'XYZ'[i%3],'body':'robot' if i==0 else 'link'};geometry.append((shape,i,matrix))
 q=torch.randn(n,4,4);q=q/q.norm(dim=-1,keepdim=True);p=torch.rand(n,4,3)
 e=NS(device='cpu',num_envs=n,cfg=NS(ground_height_m=.01,motion_runtime_reduction=False),geometry=geometry,robot=NS(data=NS(body_pos_w=NS(torch=p),body_quat_w=NS(torch=q))))
 expected=old_gaps(e);actual=GeometryGaps(geometry,{},'cpu',p.dtype).measure(p,q,inverse,.01)
 for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
 assert actual[0].shape==(n,4) # no collider dropped from penetration checks


@pytest.mark.parametrize('count',[0,1,5])
def test_initializing_subset_preserves_acceptance_not_controlled_eligibility(count):
 n=5;torch.manual_seed(7);mask=torch.arange(n)<count
 v=torch.randn(n,3)*.01;w=v.clone();dq=torch.randn(n,6)*.01;up=torch.ones(n);height=torch.full((n,),.34);gap=torch.full((n,),.1);rb=torch.tensor([[.3,.4]]).repeat(n,1);cb=torch.tensor([[.05,.2]]).repeat(n,1);fault=torch.zeros(n,dtype=torch.bool);fault[-1]=True
 config={'settled_linear_speed_max_mps':.1,'settled_angular_speed_max_rad_s':.1,'settled_joint_speed_max_rad_s':.1,'tilt_max_rad':.2}
 out=settled_initial_subset(initializing=mask,velocity=v,angular=w,joint_velocity=dq,settlement_joint_ids=[0,1,3,4],up=up,height=height,chassis=gap,root_bounds=rb,chassis_bounds=cb,fault=fault,restricted=torch.zeros_like(mask),config=config)
 expected=(v.norm(dim=-1)<=.1)&(w.norm(dim=-1)<=.1)&(dq[:,[0,1,3,4]].abs().amax(-1)<=.1)&~fault
 torch.testing.assert_close(out,expected&mask)
 # Old downstream transition is always intersected with initializing.
 torch.testing.assert_close(out&mask,expected&mask)


def test_initialization_predicate_uses_no_dynamic_shape_operations(monkeypatch):
 # These operations introduce CUDA synchronization even for empty masks.
 def forbidden(*args,**kwargs):raise AssertionError('dynamic shape operation in per-step predicate')
 monkeypatch.setattr(torch.Tensor,'nonzero',forbidden)
 test_initializing_subset_preserves_acceptance_not_controlled_eligibility(0)
 test_initializing_subset_preserves_acceptance_not_controlled_eligibility(5)


def test_preallocated_diagnostics_preserve_latest_pre_reset_after_source_changes():
 e=NS();q=torch.ones(4,6);limits=torch.zeros(4,6,2)
 a=diagnostic_snapshot(e,{'q':q,'limits':limits});ptr=a['q'].data_ptr();q.zero_();assert a['q'].sum()==24
 b=diagnostic_snapshot(e,{'q':q,'limits':limits});assert b['q'].data_ptr()==ptr and b['q'].sum()==0 and b['limits'].data_ptr()==limits.data_ptr()


def test_spin_cache_requires_same_physics_and_reset_epoch():
 from native_stand.students.motion_yaw_quality import runtime_measurement
 n=2;q=torch.tensor([0.,0.,0.,1.]).expand(n,2,4);root=q[:,0]
 # Stub only the simulator quaternion helper; actual runtime measurement executes.
 import sys,types
 module=types.ModuleType('isaaclab.utils.math');module.quat_apply=lambda q,v:v.clone()
 from unittest.mock import patch
 data=NS(body_link_quat_w=NS(torch=q),root_quat_w=NS(torch=root),body_link_pos_w=NS(torch=torch.zeros(n,2,3)),body_com_pos_w=NS(torch=torch.zeros(n,2,3)),body_com_lin_vel_w=NS(torch=torch.zeros(n,2,3)),body_com_ang_vel_w=NS(torch=torch.zeros(n,2,3)))
 e=NS(cfg=NS(motion_runtime_reduction=True),_sim_step_counter=10,_motion_reset_epoch=0,robot=NS(data=data),wheel_body_ids=[0,1])
 with patch.dict(sys.modules,{'isaaclab.utils.math':module}):
  a=runtime_measurement(e);assert runtime_measurement(e) is a
  e._sim_step_counter+=1;b=runtime_measurement(e);assert b is not a
  e._motion_reset_epoch+=1;assert runtime_measurement(e) is not b


def test_subset_transition_events_match_original_over_initialization_partial_resets_and_faults():
 # Compile the original expression still used by the flag-off runtime.
 source=ast.parse(Path('src/native_stand/recovery/runtime.py').read_text());expr=None
 for n in ast.walk(source):
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='settled' for t in n.targets) and isinstance(n.value,ast.BinOp):expr=n.value;break
 assert expr is not None
 code=compile(ast.fix_missing_locations(ast.Expression(expr)),'actual_legacy_settled','eval')
 n=8;mask=torch.ones(n,dtype=torch.bool);up=torch.ones(n);height=torch.full((n,),.34);chassis=torch.full((n,),.1);bounds=torch.tensor([[.3,.4]]).repeat(n,1);cb=torch.tensor([[.05,.2]]).repeat(n,1);velocity=torch.zeros(n,3);angular=velocity.clone();dq=torch.zeros(n,6)
 config={'settled_linear_speed_max_mps':.1,'settled_angular_speed_max_rad_s':.1,'settled_joint_speed_max_rad_s':.1,'tilt_max_rad':.2}
 age=torch.zeros(n);dwell1=age.clone();dwell2=age.clone();zeroed=torch.ones(n,dtype=torch.bool)
 for step in range(50):
  if step==20:mask[2:4]=True;age[2:4]=0;dwell1[2:4]=0;dwell2[2:4]=0
  fault=torch.zeros(n,dtype=torch.bool);fault[0]=step==3;restricted=torch.zeros_like(fault);restricted[1]=step==4
  self=NS(settlement_joint_ids=[0,1,3,4],up=up,root_height=height);data=NS(root_ang_vel_w=NS(torch=angular),joint_vel=NS(torch=dq))
  old=eval(code,{'torch':torch},{'self':self,'data':data,'velocity':velocity,'root_bounds':bounds,'chassis_bounds':cb,'chassis':chassis,'fault':fault,'restricted':restricted,'initial':config})
  new=settled_initial_subset(initializing=mask,velocity=velocity,angular=angular,joint_velocity=dq,settlement_joint_ids=self.settlement_joint_ids,up=up,height=height,chassis=chassis,root_bounds=bounds,chassis_bounds=cb,fault=fault,restricted=restricted,config=config)
  age+=mask.float()*.01
  dwell1=torch.where(mask&zeroed&old,dwell1+.01,0.);dwell2=torch.where(mask&zeroed&new,dwell2+.01,0.)
  accepted1=mask&(age>=.1)&(dwell1>=.05);accepted2=mask&(age>=.1)&(dwell2>=.05)
  rejected1=mask&((age>=.5)|fault|restricted)&~accepted1;rejected2=mask&((age>=.5)|fault|restricted)&~accepted2
  torch.testing.assert_close(dwell1,dwell2);torch.testing.assert_close(accepted1,accepted2);torch.testing.assert_close(rejected1,rejected2)
  mask&=~(accepted1|rejected1)
