"""CPU-only force assembly/forward dynamics, never integrates physical time."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy

import numpy as np
import pytest

from native_stand.students.firmware_mechanics import (
    ForceRouter,MechanicalEnvelopeError,mechanical_cases,followup_cases,reproduction)


def fixture():
    mj=pytest.importorskip('mujoco')
    world=''.join(f'<body pos="{i} 0 0"><joint name="q{i}" type="hinge" axis="0 0 1"/>'
        '<inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/></body>' for i in range(6))
    motors=''.join(f'<motor joint="q{i}" ctrlrange="-67 67" forcerange="-40 40"/>' for i in range(6))
    m=mj.MjModel.from_xml_string('<mujoco><option timestep=".001" gravity="0 0 0"/>'
        '<worldbody>'+world+'</worldbody><actuator>'+motors+'</actuator></mujoco>')
    d=mj.MjData(m);refs=dict(actuators=np.arange(6),dofs=np.arange(6),gears=np.ones(6))
    return mj,m,d,refs


def test_split_force_actually_reaches_engine_without_raising_motor_limit():
    mj,m,d,refs=fixture();motor=np.array([35,35,.2,35,35,-.2]);spring=np.array([15,15,0,15,15,0])
    original=m.actuator_forcerange.copy();router=ForceRouter(m,refs,'separated',False)
    router.apply(d,motor,motor,spring);mj.mj_forward(m,d);r=router.readback(d)
    assert np.allclose(r['actuator'],motor) and np.allclose(r['external'],spring)
    assert np.allclose(r['total'][[0,1,3,4]],50)
    assert np.allclose(d.qacc[[0,1,3,4]],5000)
    assert np.array_equal(m.actuator_forcerange,original) and m.actuator_forcelimited.all()


def test_legacy_route_exposes_actual_secondary40nm_clamp():
    mj,m,d,refs=fixture();motor=np.zeros(6);motor[[0,1,3,4]]=35;spring=np.zeros(6);spring[[0,1,3,4]]=15
    router=ForceRouter(m,refs,'legacy_combined',False)
    router.apply(d,motor,motor,spring);mj.mj_forward(m,d);r=router.readback(d)
    assert np.allclose(d.ctrl[[0,1,3,4]],50)
    assert np.allclose(r['actuator'][[0,1,3,4]],40) and (r['external']==0).all()
    assert np.allclose(d.qacc[[0,1,3,4]],4000)


def test_flexible_transmission_is_independent_passive_force_and_not_double_motor():
    mj,m,d,refs=fixture();motor=np.zeros(6);motor[[0,1,3,4]]=35
    transmitted=motor.copy();transmitted[[0,1,3,4]]=55
    spring=np.zeros(6);spring[[0,1,3,4]]=15
    router=ForceRouter(m,refs,'separated',True)
    router.apply(d,motor,transmitted,spring);mj.mj_forward(m,d);r=router.readback(d)
    assert (r['actuator']==0).all() and np.allclose(r['transmission_external'],transmitted)
    assert np.allclose(r['spring_external'],spring) and np.allclose(r['total'][[0,1,3,4]],70)
    assert np.allclose(r['motor_state'][[0,1,3,4]],35) #35, not70, at upstream motor


def test_original40nm_motor_limit_and_independent_numeric_guard_remain():
    mj,m,d,refs=fixture();router=ForceRouter(m,refs,'separated',True)
    motor=np.zeros(6);motor[0]=40.01
    with pytest.raises(RuntimeError,match='capability'):router.apply(d,motor,motor,np.zeros(6))
    motor[:]=0;trans=motor.copy();trans[0]=201
    with pytest.raises(MechanicalEnvelopeError):router.apply(d,motor,trans,np.zeros(6))
    trans[0]=np.nan
    with pytest.raises(ValueError):router.apply(d,motor,trans,np.zeros(6))


def test_readback_rejects_an_overwrite_of_external_force():
    mj,m,d,refs=fixture();router=ForceRouter(m,refs,'separated',False)
    router.apply(d,np.zeros(6),np.zeros(6),np.ones(6));d.qfrc_applied[0]=0
    mj.mj_forward(m,d)
    with pytest.raises(RuntimeError,match='external force'):router.readback(d)


def test_frozen_matrix_changes_only_declared_mechanics_and_contains_route_pairs():
    rows=mechanical_cases();assert len(rows)==110 and len({r['name'] for r in rows})==110
    for r in rows:
        assert r['parameters']['armature']==.012 and r['parameters']['kd']==2
        assert r['parameters']['friction']==.5 and r['parameters']['zero_bias']==[0.]*4
        assert r['parameters']['actor_age_ms']==r['parameters']['pd_age_ms']==0
        assert r['preload_fraction'] in (0,1)
    for name in ('rigid','lowgain_rigid','flex100','flex300'):
        a=next(r for r in rows if r['name']==name+'_legacy_combined')
        b=next(r for r in rows if r['name']==name+'_separated')
        assert a['parameters']==b['parameters'] and a['sampling']==b['sampling']


def test_followup_only_reads_fit_ranking_and_not_later_window_or_success():
    rows=[r for r in mechanical_cases() if r['family']=='compliance_grid'][:8]
    for i,r in enumerate(rows):r.update(features_fit=dict(coverage=1.),fit=dict(score=i))
    result=followup_cases(rows);assert 0<len(result)<=12
    other=copy.deepcopy(rows)
    for r in other:r['holdout']=dict(score=-1000,oscillation_candidate=True)
    assert result==followup_cases(other)
    assert all(r['parent'] in [x['name'] for x in rows[:4]] for r in result)


def test_reproduction_requires_all_channels_amplitude_pose_and_full_window():
    real=dict(valid=True,coverage=1.,dq_std_rad_s=[5]*4,velocity_peak_hz=[23.4]*4,
        band20_27_fraction=[.7]*4,tilt_mean_rad=[.02,.01])
    assert reproduction(real,real)['matched']
    for key,value in (('dq_std_rad_s',[5,5,5,20]),('velocity_peak_hz',[23,23,23,8]),
        ('tilt_mean_rad',[.4,.01]),('coverage',.5)):
        changed=copy.deepcopy(real);changed[key]=value
        assert not reproduction(real,changed)['matched']


def test_low_stiffness_preload_cannot_schedule_an_unrepresentable_initial_twist():
    rows=[r for r in mechanical_cases() if r['family']=='compliance_grid' and r['sampling']['coupling_stiffness_nm_rad']==20][:4]
    for i,row in enumerate(rows):row.update(features_fit=dict(coverage=1.),fit=dict(score=i))
    # Recorded pre-LQR motor command requires .166rad at K20, beyond the .15rad bound.
    initial=np.array([3.31978035,1.364,.0,-1.266956,1.479923,.0])
    result=followup_cases(rows,initial_motor=initial)
    assert result and not any(r['name'].startswith('preload__') for r in result)
    assert len(followup_cases(rows,initial_motor=np.zeros(6)))>len(result)
