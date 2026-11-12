import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

def test_inertia_comparison_rotates_full_tensor_not_eigenvalue_order():
    pytest.importorskip('mujoco');pytest.importorskip('pxr')
    from audit_recovery_mujoco_parity import inertia
    angle=np.pi/4
    a=inertia([1,2,3],[np.cos(angle),0,0,np.sin(angle)])
    np.testing.assert_allclose(a,np.diag([2,1,3]),atol=1e-12)

def test_readback_compiled_model_audit_never_steps(monkeypatch):
    mj=pytest.importorskip('mujoco');pytest.importorskip('pxr')
    from audit_recovery_mujoco_parity import audit
    root=Path(__file__).resolve().parents[1]
    spec=root/'configs/experiments/2026_10_17-17_59__recovery31_standing_contact_quality/spec.json'
    readback=root/'runs/2026_10_17-17_59__recovery31_standing_contact_quality/source_eval/evaluation_readback_00.json'
    xml=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml')
    if not all(p.exists() for p in (spec,readback,xml)):pytest.skip('local read-only evidence unavailable')
    def forbidden(*a,**kw):raise AssertionError('static audit must never integrate physics')
    monkeypatch.setattr(mj,'mj_step',forbidden)
    r=audit(spec,readback,xml)
    assert r['simulation_steps']==0 and len(r['bodies'])==39
    assert max(b['inertia_relative_frobenius'] for b in r['bodies'])<1e-5
    wheels=[j for j in r['joints'] if j['group']=='wheel_motors']
    assert len(wheels)==2 and all(j['native_explicit_actuator_damping']==0 and j['mj_mechanical_damping']==.01 for j in wheels)

def test_damping_alignment_changes_only_named_damping_attributes():
    from audit_recovery_mujoco_parity import align_joint_damping
    import xml.etree.ElementTree as ET
    xml='<mujoco><default><joint damping=".6"/></default><worldbody><body><joint name="wheel" damping=".01" armature=".01"/><geom friction="1.1 .05 .01"/></body></worldbody></mujoco>'
    rb={'joint_names':['wheel'],'control':{'actuators':{'wheel_motors':{'joint_ids':[0],'damping':[0.]}}}}
    new,changes=align_joint_damping(xml,rb)
    tree=ET.fromstring(new);joint=tree.find('.//body/joint');assert joint.get('damping')=='0'
    joint.set('damping','.01')
    assert ET.tostring(tree)==ET.tostring(ET.fromstring(xml))
    assert changes[0]['aligned_damping']==0
    rb['joint_names']=['missing']
    with pytest.raises(ValueError,match='absent'):align_joint_damping(xml,rb)
