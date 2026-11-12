import sys,json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]

def test_friction_diagnostic_only_changes_three_sliding_slots():
 import xml.etree.ElementTree as ET
 r,_=modules()
 xml='<mujoco><worldbody><geom name="ground" friction="1.1 .05 .01"/><body><geom name="left_wheel_contact" friction="1.1 .005 .0001"/><geom name="right_wheel_contact" friction="1.1 .005 .0001"/><joint damping=".01"/></body></worldbody></mujoco>'
 updated,changes=r.wheel_ground_friction(xml,.5)
 assert len(changes)==3
 a,b=ET.fromstring(xml),ET.fromstring(updated)
 for x,y in zip(a.iter(),b.iter()):
  if x.tag=='geom':
   assert y.get('friction').split()[0]=='0.5'
   assert x.get('friction').split()[1:]==y.get('friction').split()[1:]
   y.set('friction',x.get('friction'))
 assert ET.tostring(a)==ET.tostring(b)
 with pytest.raises(ValueError):r.wheel_ground_friction(xml,float('nan'))

def modules():
 sys.path.insert(0,str(ROOT/'scripts'))
 import validate_recovery31_mujoco as r
 import validate_native_mujoco as n
 return r,n

def test_native_limit_copy_and_complete_pose_encoder():
 mj=pytest.importorskip('mujoco');r,n=modules()
 usd=ROOT/'assets/diagnostic/rmuc_robot_closed_loop_knee_stop_0233_v2.usd'
 xml=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml')
 if not usd.exists() or not xml.exists():pytest.skip('local assets unavailable')
 original=xml.read_bytes();text,changes=r.xml_with_native_limits(xml,usd)
 model=mj.MjModel.from_xml_string(text);refs=n.binding(model)
 left=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'left_knee_pitch')
 assert model.jnt_range[left,0]==pytest.approx(-.233,abs=1e-7)
 assert xml.read_bytes()==original
 s=json.loads((ROOT/'configs/experiments/2026_10_15-22_57__recovery31_swing_broad_narrow_10000/spec.json').read_text())
 pose=next(p for p in json.loads(Path(s['recovery']['pose_library']).read_text())['poses'] if p['family']=='swing_candidate_from_historical_r1' and p['split']=='holdout')
 data=r.initialize(model,refs,pose)
 import torch
 legs=torch.tensor(data.qpos[refs['qpos']][[0,1,3,4]],dtype=torch.float32)[None]
 obs,state=r.observation(model,data,refs,legs,torch.zeros(1,2))
 assert obs.shape==(1,31) and obs[0,26]==1 and obs[0,25]==0 and obs[0,24]==pytest.approx(.68)
 np.testing.assert_allclose(obs[0,16:20],legs[0]/1.5)
 assert state['closure_m']<1e-4 and r.clearance(model,data,refs)>0
 assert not data.qvel.any()
 # No simulation step: modifying encoder velocity must refresh derived IMU too.
 data.qvel[3:6]=[.4,.2,-.3]
 x,_=r.observation(model,data,refs,legs,torch.zeros(1,2))
 np.testing.assert_allclose(x[0,:3],np.array([.4,.2,-.3])*.25,atol=1e-7)

def test_body_torque_target_previous_cache_roundtrip():
 import torch
 from native_stand.students.interface import targets
 a=torch.tensor([[.2,.3,.04,-.2,.3,-.04]])
 out=targets(a,torch.zeros(1,6),torch.tensor([[[-3.,3.]]*6]))
 torch.testing.assert_close(out['leg_executed'],torch.tensor([[.3,.45,-.3,.45]]))
 torch.testing.assert_close(out['wheel_executed'],torch.tensor([[4.,-4.]]))


def test_pause_and_single_policy_interval_do_not_advance_while_paused():
 r,_=modules();p=r.Playback(True)
 assert not p.tick()
 p.key(ord('N'))
 assert sum(p.tick() for _ in range(15))==10
 assert not p.tick()
 p.key(32);assert all(p.tick() for _ in range(20))
 p.key(ord('P'));assert not p.tick()
