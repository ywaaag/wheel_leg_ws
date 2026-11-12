"""CPU-only capture analysis; no Isaac imports or optimizer."""
import numpy as np
import pytest
from native_stand.students.motion_control_readback import analyze,COMMANDS,VERSION

def fixture_arrays():
 n=3000
 a={'first_episode_valid':np.ones((n,1),bool),'controller':np.full((n,1,1),3),
 'state':np.zeros((n,1,39)),'root_post':np.zeros((n,1,13)),'control':np.zeros((n,1,54)),
 'raw_action':np.zeros((n,1,6)),'readback_joint_dq_rad_s':np.tile([[-2.,2.]],(n,1,1)),
 'readback_relative_spin_rad_s':np.tile([[-2.,2.]],(n,1,1)),
 'readback_executed_wheel_target_rad_s':np.tile([[-2.,2.]],(n,1,1)),
 'readback_rolling_speed_forward_mps':np.tile([[.12,-.12]],(n,1,1)),
 'readback_tangent_velocity_w':np.zeros((n,1,2,2))}
 a['state'][:,:,29]=1;a['state'][:,:,10:12]=10;a['state'][:,:,9]=.5
 a['control'][:,:,[8,11]]=[-2,2];a['control'][:,:,[20,23]]=[-.02,.02]
 a['raw_action'][:,:,[2,5]]=[-.02,.02]
 return a

def test_readback_opposite_wheels_and_zero_slip():
 r=analyze(fixture_arrays())
 assert r['actual_opposite_sign_fraction']==1 and r['target_opposite_sign_fraction']==1
 assert r['joint_vs_relative_axial_max_abs_rad_s']==0
 assert r['support_slip_mean_mps']==[0,0] and r['horizontal_mean_mps']==0

def test_readback_rejects_wrong_polarity():
 a=fixture_arrays();a['readback_relative_spin_rad_s']*=-1
 with pytest.raises(ValueError,match='polarity'):analyze(a)

def test_readback_rejects_target_mismatch():
 a=fixture_arrays();a['readback_executed_wheel_target_rad_s']+=1
 with pytest.raises(ValueError,match='target readback'):analyze(a)

def test_incomplete_episode_is_not_a_complete_readback():
 a=fixture_arrays();a['first_episode_valid'][2000:]=False
 with pytest.raises(ValueError,match='incomplete'):analyze(a)

def test_torque_can_brake_a_positive_wheel():
 a=fixture_arrays();a['control'][:,:,[8,11]]=2
 a['control'][:,:,[34,37]]=-1;a['control'][:,:,[46,49]]=-1
 r=analyze(a)
 assert r['actual_joint_dq_mean_rad_s']==[2,2] and r['requested_torque_mean_nm']==[-1,-1]

def test_explicit_readback_request_preserves_single_screen_validation(monkeypatch):
 import native_stand.students.motion_single_screen as single
 from native_stand.students.motion_control_readback import validate
 import json
 from pathlib import Path
 parent=Path(__file__).resolve().parents[1]/'configs/experiments/2026_10_17-18_18__motion810_spin61_formal_recovery/spec.json'
 if not parent.exists():pytest.skip('local parent fixture unavailable')
 from native_stand.students.data import sha
 s=json.loads(parent.read_text());cp=Path(s['source_checkpoint']).parent/'model_update_750.pt'
 # validate() itself delegates immutable checkpoint/session checks to base.
 monkeypatch.setenv('HAPI_SESSION_ID','test-owner')
 cp=Path(__file__).resolve().parents[1]/'runs/2026_10_17-18_18__motion810_spin61_formal_recovery/formal/model_update_750.pt'
 if not cp.exists():pytest.skip('local trained checkpoint unavailable')
 r=dict(version=VERSION,authorization='AUTHORIZED',online_PPO=0,physical_envs=1,scored_envs=1,wall_limit_s=1800,
        parent_spec=str(parent),parent_spec_sha256=sha(parent),session='test-owner',physical_gpu=0,
        checkpoints=[dict(checkpoint=str(cp),update=750,sha256=sha(cp))],parent_output=str(cp.parent),output='/unused',commands=COMMANDS,output_max_bytes=256*2**20)
 assert validate(r)['reward']['version']=='student_motion31_yaw_contact_quality_v7'
 with pytest.raises(ValueError,match='schema'):validate(dict(r,online_PPO=1,commands=[]))
 with pytest.raises(ValueError,match='session'):validate(dict(r,session='wrong'))
 with pytest.raises(ValueError,match='identity'):validate(dict(r,checkpoints=[dict(r['checkpoints'][0],sha256='wrong')]))
