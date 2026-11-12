import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def test_usd_joint_frame_fk_includes_axis_sign_and_child_frame():
 pytest.importorskip('mujoco');pytest.importorskip('pxr')
 from audit_recovery_same_state import transform,usd_fk
 from scipy.spatial.transform import Rotation
 flip=Rotation.from_euler('y',np.pi).as_matrix()
 l0=transform([.2,0,0],flip);l1=transform([0,0,.3],flip)
 poses=usd_fk([('hip','robot','leg',l0,l1,0)],{'hip':.4},np.eye(4))
 expected=transform([.2,0,0],np.eye(3))@transform([0,0,0],Rotation.from_rotvec([-.4,0,0]).as_matrix())@transform([0,0,-.3],np.eye(3))
 np.testing.assert_allclose(poses['leg'],expected,atol=1e-12)
 assert not np.allclose(poses['leg'],l0@l1)

def test_actual_same_state_audit_prohibits_physics_steps(tmp_path,monkeypatch):
 mj=pytest.importorskip('mujoco');pytest.importorskip('pxr')
 import audit_recovery_same_state as audit
 root=Path(__file__).resolve().parents[1]
 if not (root/'sim2sim/2026_10_18-09_59__recovery1500_mujoco/trajectory.npz').exists():pytest.skip('local trace required')
 def forbidden(*a,**kw):raise AssertionError('no physics integration authorized in offline audit')
 monkeypatch.setattr(mj,'mj_step',forbidden)
 output=tmp_path/'audit.json';monkeypatch.setattr(sys,'argv',['audit','--output',str(output)])
 audit.main()
 import json
 result=json.loads(output.read_text())
 assert result['status']=='PASSED'
 assert len(result['mujoco_samples'])==128 and len(result['native_samples'])==128
 assert result['simulation_steps']==0
