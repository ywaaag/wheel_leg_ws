FROZEN_TEST_SCOPE='PORTABLE'
import copy
import pytest
import torch
from native_stand.students.motion_yaw_quality import measurement,configuration,apply,FIELDS,VERSION
from native_stand.students.motion_tracking_reward import configuration as reward_config,reward,reward_fields,COMBO_VERSION

def kinematics(spin=(1.,-1.)):
 center=torch.tensor([[[0.,-.18,.06],[0.,.18,.06]]],dtype=torch.float64)
 axis=torch.tensor([[[0.,1.,0.],[0.,1.,0.]]],dtype=torch.float64)
 angular=axis*torch.tensor(spin,dtype=torch.float64)[None,:,None]
 # Forward +X and axle +Y: omega cross (0,0,-r) = -r*omega X.
 linear=torch.zeros_like(center);linear[:,:,0]=.06*torch.tensor(spin,dtype=torch.float64)
 return center,axis,angular,linear

def test_opposed_pure_rolling_has_zero_slip_and_common_mode():
 c,a,w,v=kinematics();m=measurement(c,c,v,w,a,c.new_tensor([[1.,0.,0.]]))
 assert m['geometry_valid'].all()
 assert m['tangent_velocity_w'].abs().max()<1e-12
 assert m['rolling_speed_forward_mps'].mean(-1).abs().max()<1e-12
 # Explicit COM offset must not change physical support point velocity.
 com=c+c.new_tensor([.01,0.,.02]);vc=v+torch.cross(w,com-c,dim=-1)
 n=measurement(c,com,vc,w,a,c.new_tensor([[1.,0.,0.]]))
 torch.testing.assert_close(m['tangent_velocity_w'],n['tangent_velocity_w'])

def inputs():
 z=torch.zeros(1,dtype=torch.float64);yes=torch.ones(1,dtype=torch.bool)
 return dict(command=torch.tensor([[0.,.5,.34]],dtype=torch.float64),target=torch.tensor([[0.,.5]],dtype=torch.float64),
  velocity=torch.zeros(1,3,dtype=torch.float64),yaw=z+.5,height=z+.34,angles=torch.zeros(1,2,dtype=torch.float64),
  contacts=torch.ones(1,2,dtype=torch.float64)*20,leg_q=torch.zeros(1,4,dtype=torch.float64),wheel_offset=z,
  leg_lengths=torch.ones(1,2,dtype=torch.float64)*.3,applied_torque=torch.zeros(1,6,dtype=torch.float64),
  previous_targets=torch.zeros(1,6,dtype=torch.float64),executed_targets=torch.zeros(1,6,dtype=torch.float64),
  controlled=yes,valid=yes,physical_failure=~yes,fault=~yes)

def measured():
 c,a,w,v=kinematics();return measurement(c,c,v,w,a,c.new_tensor([[1.,0.,0.]]))

def test_spin_reward_prefers_clean_turn_and_preserves_old_recipe():
 kw=inputs();m=measured();c=reward_config(VERSION)
 clean=reward(**kw,config=c,spin_measurement=m)
 assert tuple(clean['fields'])==tuple(reward_fields(c))
 assert clean['raw'][0,-6:].abs().max()==0
 dirty=copy.deepcopy(kw);dirty['velocity'][0,0]=.5;dirty['executed_targets'][:]=1
 worse=reward(**dirty,config=c,spin_measurement=m)
 assert clean['total']>worse['total']
 assert torch.isfinite(worse['total']).all()
 for target in ([0.,0.],[1.,0.]):
  old=inputs();old['target'][0]=old['target'].new_tensor(target)
  new=reward(**old,config=c,spin_measurement=m);base=reward(**old,config=reward_config(COMBO_VERSION))
  torch.testing.assert_close(new['total'],base['total'])
 with pytest.raises(ValueError):reward(**kw,config=c)

def test_slip_contact_mask_does_not_reward_airborne():
 kw=inputs();m=measured();m['tangent_velocity_w']+=.4
 grounded=reward(**kw,config=reward_config(VERSION),spin_measurement=m)
 kw['contacts'].zero_();air=reward(**kw,config=reward_config(VERSION),spin_measurement=m)
 assert grounded['raw'][0,-1]>0 and air['raw'][0,-1]==0
 assert air['raw'][0,-2]==1 and grounded['raw'][0,-2]==0
 assert air['total']<grounded['total']

def test_cohort_identity_and_effective_command_activation():
 kw=inputs();kw['command'][0,1]=0
 out=reward(**kw,config=reward_config(VERSION),spin_measurement=measured())
 assert out['spin_diagnostics']['activation'].item()==0
 kw['command'][0,1]=.25
 out=reward(**kw,config=reward_config(VERSION),spin_measurement=measured())
 assert out['spin_diagnostics']['activation'].item()==.5

def test_slip_tail_has_nonzero_gradient_and_correct_direction():
 kw=inputs();m=measured();x=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
 m['tangent_velocity_w']=m['tangent_velocity_w']+x
 out=reward(**kw,config=reward_config(VERSION),spin_measurement=m)
 grad=torch.autograd.grad(out['total'].sum(),x)[0]
 assert grad<0 and torch.isfinite(grad)

def test_vertical_axle_invalid_not_nan():
 c,a,w,v=kinematics();a[:]=a.new_tensor([0.,0.,1.])
 m=measurement(c,c,v,w,a,c.new_tensor([[1.,0.,0.]]))
 assert not m['geometry_valid'].any()
 assert torch.isfinite(m['tangent_velocity_w']).all()

def test_reward_audit_uses_real_costs_and_all_named_fields():
 from native_stand.students.motion_reward_audit import audit
 a=audit(reward_config(VERSION))
 assert len(a['spin_quality']['curves'])==72
 assert all(k in a['named_weights'] for k in FIELDS)
 assert all(row['state_error_derivative']<0 for row in a['spin_quality']['curves'] if row['field']=='spin_slip' and row['error']>0)

def test_motion61_field_order_and_legacy_factory():
 from native_stand.students.motion_spin_privileged import critic_input,contract,network,VERSION as PRIV
 from native_stand.students.model import Student
 b=torch.zeros(1,49,dtype=torch.float64);q=b.new_tensor([[1.,0.,0.,0.]])
 # Identity robot: Native forward=-world Z, left=-world X.
 m=measured();m['tangent_velocity_w'][:]=m['tangent_velocity_w'].new_tensor([.2,.4])
 result=critic_input(b,m,b.new_zeros(1,6),b.new_ones(1,6),q)
 assert result.shape==(1,61) and result[0,49]==0 and result[0,50]==-.4
 assert contract()['critic_dim']==61
 assert Student(False,version='multiskill_mcu_v2',critic_privileged_version='motion_privileged49_v1').mlp[0].in_features==49
 assert Student(False,version='multiskill_mcu_v2',critic_privileged_version=PRIV,network_architecture=network()).mlp[0].in_features==61

def test_migration_zero_columns_preserve_critic_and_adam():
 from native_stand.students.motion_spin_privileged import expand_source,network,VERSION as PRIV
 from native_stand.students.motion_large_combo import network as base_network
 from native_stand.students.model import Student
 a=Student(True,version='multiskill_mcu_v2',network_architecture=base_network())
 c=Student(False,version='multiskill_mcu_v2',critic_privileged_version='motion_privileged49_v1',network_architecture=base_network())
 opt=torch.optim.Adam(list(a.parameters())+list(c.parameters()),lr=3e-5)
 for p in list(a.parameters())+list(c.parameters()):p.grad=torch.ones_like(p)
 opt.step()
 source={'actor':a.state_dict(),'critic':c.state_dict(),'optimizer':opt.state_dict(),'network_architecture':base_network(),'critic_privileged_version':'motion_privileged49_v1'}
 migrated=expand_source(source)
 new=Student(False,version='multiskill_mcu_v2',critic_privileged_version=PRIV,network_architecture=network());new.load_state_dict(migrated['critic'])
 x=torch.randn(16,49);xx=torch.cat((x,torch.randn(16,12)),1)
 torch.testing.assert_close(new(xx),c(x),rtol=1e-5,atol=1e-6)
 assert migrated['critic']['mlp.0.weight'][:,49:].count_nonzero()==0
 ids=[i for g in migrated['optimizer']['param_groups'] for i in g['params']]
 assert migrated['optimizer']['state'][ids[len(source['actor'])]]['exp_avg'][:,49:].count_nonzero()==0
 assert source['critic']['mlp.0.weight'].shape==(512,49)

def test_spin_statistics_counts_hold_only_and_uses_null_when_missing():
 from native_stand.students.motion_yaw_quality import SpinStatistics
 stats=SpinStatistics(2,'cpu');m=measured();m['activation']=torch.ones(1,dtype=torch.float64)
 # Collector matches runtime float32 accumulator/input precision.
 m={k:v.float() if v.dtype==torch.float64 else v for k,v in m.items()}
 zero=torch.zeros(1,6);contact=torch.ones(1,2)*20
 stats.observe(torch.tensor([0]),torch.tensor([True]),torch.tensor([True]),m,contact,zero,zero)
 stats.observe(torch.tensor([0]),torch.tensor([True]),torch.tensor([False]),m,contact,zero,zero)
 out=stats.export()['cohorts']
 assert out['0']['samples']==1 and out['1']['mean'] is None
 assert out['0']['mean']['left_slip_speed_mps']==0

def test_runtime_only_enables_quality_for_explicit_new_version():
 import ast
 from pathlib import Path
 p=Path(__file__).resolve().parents[1]/'src/native_stand/students/motion_runtime.py'
 ast.parse(p.read_text())
 # Verify supported version membership without depending on tuple formatting/order.
 memberships=[]
 for node in ast.walk(ast.parse(p.read_text())):
  if isinstance(node,ast.Compare) and isinstance(node.ops[0],ast.In) and isinstance(node.comparators[0],ast.Tuple):
   values=[v.value for v in node.comparators[0].elts if isinstance(v,ast.Constant)]
   if 'student_motion31_yaw_contact_quality_v7' in values:memberships.append(set(values))
 expected={'student_motion31_yaw_contact_quality_v7','student_motion31_symmetry_quality_v8','student_motion31_differential_quality_v9','student_motion31_parallel_quality_v10'}
 assert len(memberships)==2 and all(values==expected for values in memberships)

def test_v7_recipe_declares_61_and_preserves_large49_defaults():
 from native_stand.students.motion_training import declaration
 from native_stand.students.motion_spin_privileged import network
 from native_stand.students.training_recipe import validate_instance
 r=declaration('a'*64,0,reward_version=VERSION)
 r.update(course_version='motion31_large_combo4096_v1',course_stage=0,network_architecture=network())
 assert validate_instance(r,'a'*64)['critic_privileged_version']=='motion_spin_privileged61_v1'
 assert r['critic']=='source49_to61_zero_new_columns'
 assert declaration('a'*64,0,reward_version=COMBO_VERSION)['critic_privileged_version']=='motion_privileged49_v1'

def test_quality_wrapper_v7_costs_match_direct_reward():
 from native_stand.students.motion_tracking_reward import COMBO_BASE
 from native_stand.students.motion_quality import apply_reward,configuration as quality_config
 kw=inputs();kw['velocity'][0,0]=.2;kw['executed_targets']+=.02
 c=reward_config(VERSION);m=measured();m['tangent_velocity_w']+=.1
 base=reward(**kw,config=reward_config(COMBO_BASE))
 posture=torch.zeros(1,10,dtype=torch.float64);posture[:,:2]=kw['angles'];posture[:,2:4]=kw['leg_lengths']
 q=apply_reward(base,posture=posture,wheel_offset=kw['wheel_offset'],leg_q=kw['leg_q'],controlled=kw['controlled'],valid=kw['valid'],quality_failed=kw['physical_failure'],physical_failed=kw['physical_failure'],fault=kw['fault'],config=quality_config(),reward_config=c)
 wrapped=apply(q,target=kw['target'],command=kw['command'],velocity=kw['velocity'],contacts=kw['contacts'],previous_targets=kw['previous_targets'],executed_targets=kw['executed_targets'],measurement=m,config=c['spin_quality'])
 direct=reward(**kw,config=c,spin_measurement=m)
 torch.testing.assert_close(wrapped['total'],direct['total'])
 assert len(wrapped['fields'])==22

def test_source_calibration_signed_readback_and_reconstruction(tmp_path):
 import numpy as np,json
 from native_stand.students.motion_spin_calibration import verify_records
 c,a,w,v=kinematics((1.,1.));f=c.new_tensor([[1.,0.,0.]])
 m=measurement(c,c,v,w,a,f);steps=1200
 fields={'center_world':c,'com_world':c,'linear_com_world':v,'angular_world':w,'axis_world':a,'forward_world':f,**m}
 data={'spin_'+k:np.repeat(x.numpy()[None,:],steps,axis=0) for k,x in fields.items() if k in ('center_world','com_world','linear_com_world','angular_world','axis_world','forward_world','tangent_velocity_w','rolling_speed_forward_mps','geometry_valid')}
 data.update(first_episode_valid=np.ones((steps,1),dtype=bool),policy_time_s=np.arange(1,steps+1)*.01,controller=np.full((steps,1,1),3))
 np.savez(tmp_path/'trace.npz',**data);(tmp_path/'raw.json').write_text(json.dumps({'trace':{'path':'trace.npz'},'case':{'command':[.5,0.,.34]}}))
 result=verify_records([{'path':'raw.json'}],tmp_path)
 assert result['status']=='PASSED' and result['capability_gate'] is False

def test_spin_uses_current_smoke_instead_of_legacy_task_resume(monkeypatch):
 from native_stand.command_v6.wiring import completed_smoke
 from native_stand.students import motion_training_requests
 def legacy(_):raise AssertionError('spin must not enter legacy task manifest resume')
 monkeypatch.setattr(motion_training_requests,'verify_continuation',legacy)
 assert completed_smoke({'attempts':{}},{'motion_spin_quality':{'version':'motion31_spin_quality61_v1'},'motion_training':{'resume':{'parent_spec':'fixture'}}}) is False

def test_spin_recovery_dispatch_is_explicit(monkeypatch):
 from native_stand.students import motion_spin_experiment as spin
 from native_stand.students.motion_wiring_recovery import verify
 monkeypatch.setattr(spin,'verify_recovery',lambda s:{'status':'VERIFIED','historical_smoke_updates':5,'remaining_formal_updates':1995})
 s={'motion_training':{'wiring_reuse':{'version':spin.RECOVERY}}}
 assert verify(s)['remaining_formal_updates']==1995
 assert spin.verify(s)['historical_smoke_updates']==5
