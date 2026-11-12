"""Exercise actual response bridge boundaries without integrating physics."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy
from pathlib import Path
import importlib.util
import numpy as np
import pytest
from native_stand.students.motion_response_probe import VERSION,episode_parameters,declaration,prepare_colleague,bootstrap_case
from native_stand.students.firmware_sampling import source_parameters
from native_stand.students.motion_compliance_training import PROFILES
from native_stand.students.real2sim_delay import torque_response

ROOT=Path(__file__).resolve().parents[1]


def spec(asset='current'):
    return dict(PD=dict(leg_Kp=20.,leg_Kd=2.,wheel_Kv=.15),response_probe=dict(version=VERSION,asset=asset))


def test_old_path_remains_value_identical_and_unaliased():
    p=source_parameters();m=PROFILES[0];original=copy.deepcopy((m,p))
    a,b=episode_parameters({}, {},m,p)
    assert (a,b)==original and (m,p)==original and a is not m and b is not p


def test_real_constructor_default_is_not_a_training_flex_slot():
    from native_stand.students.motion_compliance_training import assignments
    defaults=assignments(0,0)
    assert any(r['profile']!='R' for r in defaults[:4])
    for r in defaults[:4]:
        first=bootstrap_case(spec('colleague'),r)
        m,p=episode_parameters(spec('colleague'),first,PROFILES[0],source_parameters())
        assert m['name']=='R' and p['motor_tau_ms']==[0.]*6
        assert bootstrap_case({},r)==r


@pytest.mark.parametrize('tau',[0.,2.])
def test_tau_only_changes_motor_response(tau):
    p=source_parameters();m=PROFILES[2]
    a,b=episode_parameters(spec(),dict(motor_tau_ms=tau),m,p)
    assert a==m
    assert {k:v for k,v in b.items() if k!='motor_tau_ms'}=={k:v for k,v in p.items() if k!='motor_tau_ms'}
    assert b['motor_tau_ms']==[tau,tau,0.,tau,tau,0.]
    assert p['motor_tau_ms']==[0.,0.,0.,0.,0.,0.]
    out=torque_response(np.zeros(6),np.ones(6),b['motor_tau_ms'])
    assert np.allclose(out[[2,5]],1.)
    assert np.allclose(out[[0,1,3,4]],1. if tau==0 else -np.expm1(-.5))


@pytest.mark.parametrize('change',[dict(version='unknown'),dict(asset='other')])
def test_version_is_explicit(change):
    s=spec();s['response_probe'].update(change)
    with pytest.raises(ValueError):declaration(s)


def test_no_unplanned_tau_or_colleague_flex():
    with pytest.raises(ValueError):episode_parameters(spec(),dict(motor_tau_ms=5.),PROFILES[0],source_parameters())
    with pytest.raises(ValueError):episode_parameters(spec('colleague'),dict(motor_tau_ms=0.),PROFILES[1],source_parameters())
    m,p=episode_parameters(spec('colleague'),dict(motor_tau_ms=2.),PROFILES[0],source_parameters())
    assert m['load_armature']==.01 and PROFILES[0]['load_armature']==.012


def test_real_colleague_model_bridge_preserves_mechanics_without_integration():
    import mujoco,sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from validate_native_mujoco import binding
    path=Path('/home/kellen/桌面/11 (1)/11/wheeled-legged_RL/sim2real_RL/sim2sim/rmuc_peak489_v9.xml')
    if not path.exists():pytest.skip('external read-only asset unavailable')
    model=mujoco.MjModel.from_xml_path(str(path));refs=binding(model)
    keys=('body_mass','body_inertia','dof_damping','dof_armature','jnt_range','geom_friction','geom_solref','eq_solref')
    before={k:getattr(model,k).copy() for k in keys}
    report=prepare_colleague(model,refs)
    for k,v in before.items():assert np.array_equal(v,getattr(model,k)),k
    assert model.opt.timestep==.001 and not report['D5_runtime']
    assert model.actuator_forcerange[refs['actuators'][2],1]==pytest.approx(2.44695,abs=1e-5)


def test_fixed_matrix_pairs_and_episodes():
    path=ROOT/'scripts/compare_motion31_response.py'
    loader=importlib.util.spec_from_file_location('response_probe_script',path)
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    rows=module.matrix();assert len(rows)==24
    assert sum(r['asset']=='current' for r in rows)==16
    for key in {r['pair'] for r in rows}:
        a,b=[r for r in rows if r['pair']==key]
        assert a['motor_tau_ms']==0. and b['motor_tau_ms']==2.
        assert {k:v for k,v in a.items() if k not in ('id','motor_tau_ms')}=={k:v for k,v in b.items() if k not in ('id','motor_tau_ms')}
