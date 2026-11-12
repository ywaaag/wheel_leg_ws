"""Exercise real base measurement methods without importing Isaac runtime."""
FROZEN_TEST_SCOPE='PORTABLE'
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import pytest
from native_stand.students.motion_hotpath import runner_flags,body_consumers
from native_stand.students.step_checks import StepChecks
from native_stand.students.performance import settings


def methods():
 source=Path('src/native_stand/env.py').read_text();tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativeStandEnv')
 nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_motion_state','_closure')]
 loops=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LOOPS' for t in n.targets)))
 calls=[]
 def rotate(q,x):calls.append(1);return x.clone()
 ns={'torch':torch,'quat_apply':rotate,'quat_apply_inverse':rotate,'LOOPS':loops}
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual_env_methods','exec'),ns)
 return ns,calls,loops


def fake(fast):
 ns,calls,loops=methods();names=list(dict.fromkeys(n for a,b,_,_ in loops for n in (a,b)));n=2
 d=NS(root_quat_w=NS(torch=torch.tensor([[0.,0.,0.,1.]]).repeat(n,1)),root_ang_vel_w=NS(torch=torch.ones(n,3)),root_lin_vel_w=NS(torch=torch.ones(n,3)*2),root_pos_w=NS(torch=torch.ones(n,3)),body_pos_w=NS(torch=torch.zeros(n,len(names),3)),body_quat_w=NS(torch=torch.zeros(n,len(names),4)))
 e=NS(cfg=NS(motion_hotpath=fast),robot=NS(data=d),num_envs=n,device='cpu',_sim_step_counter=10,_motion_reset_epoch=0,body_ids={k:i for i,k in enumerate(names)},closure_each=torch.zeros(n,4))
 for k in ('root_height','up','vx','vy','yaw'):setattr(e,k,torch.zeros(n))
 return e,ns,calls


def test_actual_cached_measurement_matches_baseline_and_invalidates_on_physics_and_partial_reset():
 fast,ns,calls=fake(True);slow,sns,_=fake(False)
 for name in ('_motion_state','_closure'):
  ns[name](fast);sns[name](slow)
 for k in ('root_height','up','vx','vy','yaw','closure_each'):torch.testing.assert_close(getattr(fast,k),getattr(slow,k))
 count=len(calls);ns['_motion_state'](fast);ns['_closure'](fast);assert len(calls)==count
 # Pre-action to post-physics advances actual simulator step, unlike policy-only keys.
 fast._sim_step_counter+=1;fast.robot.data.root_lin_vel_w.torch[0,2]=7
 ns['_motion_state'](fast);assert fast.vx[0]==-7 and len(calls)>count
 # Same physical counter, one environment reset changes epoch; all cached measurements refresh safely.
 fast._motion_reset_epoch+=1;fast.robot.data.root_lin_vel_w.torch[1,2]=9
 ns['_motion_state'](fast);assert fast.vx[1]==-9
 constants=fast._closure_local_constants;fast._sim_step_counter+=1;ns['_closure'](fast);assert fast._closure_local_constants is constants


def test_timeout_flag_and_all_safety_checks_share_transfer_without_dropping_faults():
 c=StepChecks();c.require(torch.tensor([True,True]),'finite');assert runner_flags(c,torch.tensor([False,True]))
 c.require(torch.tensor([True,False]),'physical fault')
 with pytest.raises(RuntimeError,match='physical fault'):runner_flags(c,torch.tensor([False,False]))


def test_contact_scope_audit_preserves_restricted_guides_and_never_enables_subset():
 names=['robot','left_wheel','right_wheel','left_upper_leg','guide_wheel_0','guide_wheel_1']
 d=body_consumers(names,['left_wheel','right_wheel'],['robot','guide_wheel_0'])
 assert 'guide_wheel_0' in d['required'] and d['removable_if_no_other_consumer']==['guide_wheel_1']
 with pytest.raises(ValueError):body_consumers(names,[],['missing_link'])


def test_opt_in_motion_only_and_safety_compatibility():
 s={'student_training':{'version':'student_motion31_training_v1'},'performance':{'version':'student_performance_v2','motion_hotpath':True,'combined_step_checks':True,'contact_history_disabled':True}}
 assert settings(s)['motion_hotpath'];s['performance']['combined_step_checks']=False
 with pytest.raises(ValueError,match='combined'):settings(s)
 assert not settings({}).get('motion_hotpath',False)


def test_history_readback_explicit_and_sensor_mapping_preserved():
 from native_stand.students.motion_hotpath import readback
 e=NS(cfg=NS(motion_hotpath=True,scene=NS(contact_forces=NS(history_length=0))),sensor=NS(cfg=NS(history_length=0),body_names=['robot','left_wheel','right_wheel']),wheel_sensor_ids=[1,2],forbidden_sensor_ids=[0],allowed_sensor=[0,1,2],restricted_sensor=[])
 out=readback(e)
 assert out['contact_history_length_effective']==0 and out['sensor_bodies']==e.sensor.body_names
 e.sensor.cfg.history_length=3
 with pytest.raises(RuntimeError,match='history'):readback(e)
