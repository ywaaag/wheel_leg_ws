import numpy as np
from native_stand.students.evaluator import _compact_motion_evidence

def test_compact_retains_actual_target_and_physical_rolling_without_full_control():
 from native_stand.command_v6.domain.trace_schema import CONTROL
 c=np.arange(54,dtype=np.float32).reshape(1,1,54)
 a={'control':c,'state':np.zeros((1,1,39)),'actor_input':np.zeros((1,1,31)),'spin_rolling_speed_forward_mps':np.array([[[.1,-.09]]])}
 b=_compact_motion_evidence(a,wheel_scale=100.)
 np.testing.assert_array_equal(b['wheel_actual_joint_velocity_rad_s'],c[...,CONTROL['joint_dq']][...,[2,5]])
 np.testing.assert_array_equal(b['wheel_target_velocity_rad_s'],c[...,CONTROL['clipped_action']][...,[2,5]]*100.)
 np.testing.assert_array_equal(b['spin_rolling_speed_forward_mps'],a['spin_rolling_speed_forward_mps'])
 assert 'control' not in b and 'actor_input' not in b and 'state' in b
 assert 'control' in a and len(a)==4
 legacy=_compact_motion_evidence(a)
 assert set(legacy)=={'state'}

def test_training_source_and_formal_preserve_differential_and_wheel_fields(tmp_path):
 from native_stand.students.training import compact_motion_evidence
 c=np.arange(54,dtype=np.float32).reshape(1,1,54)
 a={'control':c,'state':np.zeros((1,1,39)),'actor_input':np.zeros((1,1,31)),'critic_input':np.zeros((1,1,85)),
    'spin_center_world':np.zeros((1,1,2,3)),'spin_forward_world':np.zeros((1,1,3)),
    'spin_rolling_speed_forward_mps':np.array([[[.1,-.09]]])}
 for k in ('target','difference','error','rolling'):a['differential_'+k]=np.zeros((1,1,2 if k=='rolling' else 1))
 spec={'action':{'wheel_scale_rad_s':100.},'motion_differential':{'version':'fixture'},'motion_symmetry':{'version':'fixture'},'motion_spin_quality':{'version':'fixture'}}
 for phase in ('source_eval','formal'):
  b=compact_motion_evidence(spec,a,phase)
  for key in ('target','difference','error','rolling'):assert 'differential_'+key in b
  assert 'wheel_actual_joint_velocity_rad_s' in b and 'wheel_target_velocity_rad_s' in b
  if phase=='source_eval':assert {'actor_input','critic_input','control','spin_center_world','spin_forward_world'}<=set(b)
  else:assert 'control' not in b and 'actor_input' not in b
  path=tmp_path/(phase+'.npz');np.savez_compressed(path,**b)
  with np.load(path) as restored:
   assert set(restored.files)==set(b)
   np.testing.assert_array_equal(restored['differential_rolling'],a['differential_rolling'])
 bad={k:v for k,v in a.items() if k!='differential_target'}
 import pytest
 with pytest.raises(ValueError,match='before save'):compact_motion_evidence(spec,bad,'formal')
