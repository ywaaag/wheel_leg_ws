import copy
from pathlib import Path

import numpy as np
import pytest

from native_stand.students.oscillation_search import (
    CExecution, LEGS, WHEELS, combinations, features, nominal, singles, similarity, validate)


def test_single_matrix_is_explicit_independent_and_gain_controls_are_first():
    cases=singles()
    assert len({r['name'] for r in cases})==len(cases)
    assert cases[0]['parameters']==nominal()
    assert [r['parameters']['kp'] for r in cases[:4]]==[40,20,40,20]
    assert [r['parameters']['kv'] for r in cases[:4]]==[.2,.2,.15,.15]
    cases[4]['parameters']['motor_delay_ms'][0]=99
    assert nominal()['motor_delay_ms'][0]==10
    assert singles()[4]['parameters']['motor_delay_ms'][0]==10


@pytest.mark.parametrize('key,value',[('pd_age_ms',25),('actor_age_ms',.1),('kp',0),
    ('motor_gain',1.2),('zero_bias',[.2]*4),('motor_delay_ms',[.5]*6)])
def test_reject_unsupported_or_guard_bypassing_parameters(key,value):
    p=nominal();p[key]=value
    with pytest.raises(ValueError):validate(p)


def test_combinations_keep_gains_and_change_only_two_different_mechanisms():
    rows=singles();leaders=[rows[4],rows[5],next(r for r in rows if r['family']=='spring'),rows[1]]
    pairs=combinations(leaders)
    assert pairs
    for r in pairs:
        assert r['parameters']['kp']==40
        assert r['parameters']['kv']==.2
        assert len(r['parents'])==2
        assert all(name!='legP20' for name in r['parents'])


def signals(seconds=6., amplitude=5., frequency=23.4375):
    t=np.arange(0,seconds,.01)
    w=2*np.pi*frequency
    v=amplitude*np.sin(w*t)
    dq=np.tile(v[:,None],(1,6))
    q=np.tile((-amplitude/w*np.cos(w*t))[:,None],(1,4))
    action=np.tile((v*.05)[:,None],(1,6))
    tau=dq*2
    gravity=np.tile([0,-1,0],(len(t),1))
    return t,q,dq,action,tau,gravity


def test_real_and_sim_use_identical_frequency_and_units():
    r=features(*signals())
    assert r['valid'] and r['coverage']>.99
    assert np.allclose(r['velocity_peak_hz'],23.4375)
    score=similarity(r,copy.deepcopy(r))
    assert score['score']==0 and score['oscillation_candidate']
    assert not score['physical_parameters_identified']


def test_quiet_motion_or_early_fall_cannot_match_a_real_oscillation():
    real=features(*signals())
    quiet=features(*signals(amplitude=.01))
    assert not similarity(real,quiet)['oscillation_candidate']
    early=features(*signals(seconds=1.))
    assert not similarity(real,early)['oscillation_candidate']
    assert similarity(real,early)['coverage']<.98


def test_saturation_uses_raw_commands_even_with_timestamp_jitter():
    from native_stand.students.motion_mcu_execution import WHEEL_CAP
    t,q,dq,a,tau,g=signals()
    t=t+.0003*np.sin(np.arange(len(t)))
    limits=np.array([40,40,WHEEL_CAP,40,40,WHEEL_CAP])
    tau[:]=np.where(np.arange(len(t))[:,None]%2,1.,-1.)*limits
    f=features(t,q,dq,a,tau,g)
    assert np.array_equal(f['command_saturation'],np.ones(6))
    assert f['score_version']=='raw_moments_fft_grid_v2'


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    repo=Path('/home/kellen/combat-dm-balance-main')
    if not (repo/'application/chassis/rl_control.c').exists():
        pytest.skip('local electrical C source unavailable')
    from importlib.util import spec_from_file_location,module_from_spec
    source=Path(__file__).resolve().parents[1]/'scripts/validate_motion_firmware.py'
    s=spec_from_file_location('search_host_validation',source);m=module_from_spec(s);s.loader.exec_module(m)
    out=tmp_path_factory.mktemp('search_host')/'core'
    return m.build(repo,out)['library']


def start(library,p):
    ex=CExecution(library,p,np.zeros(6))
    native=np.zeros(16,np.float32);native[4]=-1
    native[6:10]=[.24,.206,-.265,.223]
    mean=np.array([.2,.15,.02,-.2,.15,.02],np.float32)
    ex.core.observe(0,-p['actor_age_ms'],native,[0,0,.34])
    ex.core.publish(0,mean)
    return ex,native,mean


def test_nominal_scheduler_matches_full_C_and_preserves_adoption_previous(library):
    p=nominal();p['motor_delay_ms']=[0]*6;p['motor_tau_ms']=[0]*6
    ex,native,mean=start(library,p)
    try:
        torque=ex.step(0,native,native[6:10],[0,0,.34])
        assert ex.nominal_error<2e-5
        assert np.allclose(ex.motor,ex.last['motor_model_nm'],atol=2e-5)
        assert np.allclose(torque,ex.motor-ex.compensation(native[6:10]),atol=2e-5)
        for tick in range(1,10):ex.step(tick,native,native[6:10],[0,0,.34])
        obs=ex.core.observe(10,0,native,[0,0,.34])
        assert np.allclose(obs[16:22],mean[[0,1,3,4,2,5]],atol=1e-6)
        with pytest.raises(ValueError):ex.step(11,native,native[6:10],[0,0,.34])
    finally:ex.close()


def test_gain_variant_changes_PD_request_before_C_compensation_and_limits(library):
    p=nominal();p.update(kp=20.,kv=.15)
    ex,native,mean=start(library,p)
    try:
        ex.step(0,native,native[6:10],[0,0,.34])
        target=mean*np.array([1.5,1.5,100,1.5,1.5,100])
        assert np.allclose(ex.raw[LEGS],20*(target[LEGS]-native[6:10]),atol=2e-6)
        assert np.allclose(ex.raw[WHEELS],.15*target[WHEELS],atol=2e-6)
        assert np.allclose(ex.adopted,target,atol=1e-6)
    finally:ex.close()


def test_post_PD_transport_is_applied_once_per_millisecond(library):
    p=nominal();p['motor_delay_ms']=[10]*6;p['motor_tau_ms']=[0]*6
    ex,native,mean=start(library,p)
    try:
        for tick in range(10):
            ex.step(tick,native,native[6:10],[0,0,.34])
            assert np.array_equal(ex.motor,np.zeros(6))
        ex.core.observe(10,0,native,[0,0,.34]);ex.core.publish(10,mean)
        ex.step(10,native,native[6:10],[0,0,.34])
        assert np.allclose(ex.motor,ex.last['motor_model_nm'],atol=2e-5)
    finally:ex.close()


def test_out_of_branch_motion_is_a_C_input_guard_not_a_relaxed_input(library):
    from importlib.util import spec_from_file_location,module_from_spec
    source=Path(__file__).resolve().parents[1]/'scripts/search_motion31_oscillation.py'
    s=spec_from_file_location('oscillation_search_cli',source);m=module_from_spec(s);s.loader.exec_module(m)
    ex,native,_=start(library,nominal())
    try:
        assert m.c_input_valid(ex,native,[0,0,.34])
        invalid=native.copy();invalid[9]=2.6483349
        assert not m.c_input_valid(ex,invalid,[0,0,.34])
        with pytest.raises(ValueError):ex.core.observe(10,0,invalid,[0,0,.34])
        # The same guard stays active. There is no clipping the input to fit.
        assert invalid[9]>2.6
    finally:ex.close()


def test_dynamic_matrix_and_pair_keep_explicit_measurement_semantics():
    from native_stand.students.oscillation_search import dynamic_cases,dynamic_nominal
    cases=dynamic_cases()
    assert len({r['name'] for r in cases})==len(cases)
    assert cases[0]['parameters']==nominal() and cases[0]['dynamics']==dynamic_nominal()
    assert any(c['dynamics']['dq_gain']==2 for c in cases)
    assert any(c['parameters']['kd']==.5 for c in cases)
    a=next(c for c in cases if c['name']=='dq_gain1.5')
    b=next(c for c in cases if c['name']=='pdage18_P40')
    p=combinations([a,b])[0]
    assert p['dynamics']['dq_gain']==1.5 and p['parameters']['pd_age_ms']==18


def test_default_velocity_sensor_is_identical_and_consumes_no_rng():
    from native_stand.students.oscillation_search import VelocityFeedback,dynamic_nominal
    s=np.arange(18,dtype=float)
    v=VelocityFeedback(dynamic_nominal(),s)
    state=copy.deepcopy(v.rng.bit_generator.state)
    for tick in range(8):
        new=s+tick*.25
        assert np.array_equal(v.step(tick,new),new)
    assert state==v.rng.bit_generator.state


def test_velocity_age_and_packet_hold_do_not_change_position_or_imu():
    from native_stand.students.oscillation_search import VelocityFeedback,dynamic_nominal
    p=dynamic_nominal();p['dq_extra_age_ms']=2
    s=np.zeros(18);v=VelocityFeedback(p,s)
    outputs=[]
    for tick in range(11):
        new=np.full(18,float(tick))
        out=v.step(tick,new);outputs.append(out)
        assert np.array_equal(out[:12],new[:12])
    assert outputs[4][12]==0
    assert outputs[5][12]==3
    assert outputs[9][12]==3
    assert outputs[10][12]==8


def test_dq_noise_is_private_reproducible_and_has_no_position_effect():
    from native_stand.students.oscillation_search import VelocityFeedback,dynamic_nominal
    p=dynamic_nominal();p['dq_noise_std_rad_s']=.5
    s=np.zeros(18);a=VelocityFeedback(p,s);b=VelocityFeedback(p,s)
    np.random.seed(123);state=np.random.get_state()
    for tick in range(11):
        out=a.step(tick,s);assert np.array_equal(out,b.step(tick,s))
        assert np.array_equal(out[:12],s[:12])
    assert np.array_equal(state[1],np.random.get_state()[1])


def test_velocity_gain_packet_is_exact_at_nominal_Actor_and_PD_consumption_ticks():
    from native_stand.students.oscillation_search import VelocityFeedback,dynamic_nominal
    from native_stand.students.real2sim_delay import ObservationHistory
    p=dynamic_nominal();p['dq_gain']=1.5
    s=np.zeros(18);sensor=VelocityFeedback(p,s)
    history=ObservationHistory(s,initial_tick=-32)
    originals={}
    for tick in range(51):
        current=s.copy();current[12:18]=np.sin(tick*.147)
        originals[tick]=current.copy()
        history.append(tick,sensor.step(tick,current))
        if tick>=10 and tick%5==0:
            # Nominal Actor/PD consume an age10ms packet on a5ms boundary.
            assert np.allclose(history.at(tick-10)[12:18],originals[tick-10][12:18]*1.5)


def test_dynamic_motor_slew_and_100hz_pd_are_declared_not_hidden(library):
    from native_stand.students.oscillation_search import dynamic_nominal
    p=nominal();p['motor_delay_ms']=[0]*6;p['motor_tau_ms']=[0]*6
    d=dynamic_nominal();d.update(motor_slew_nm_per_s=200.,pd_period_ms=10)
    ex=CExecution(library,p,np.zeros(6),dynamics=d)
    native=np.zeros(16,np.float32);native[4]=-1;native[6:10]=[.24,.206,-.265,.223]
    mean=np.array([.2,.15,.02,-.2,.15,.02],np.float32)
    try:
        ex.core.observe(0,-10,native,[0,0,.34]);ex.core.publish(0,mean)
        ex.step(0,native,native[6:10],[0,0,.34])
        assert np.max(abs(ex.motor[LEGS]))<=.200001
        raw=ex.raw.copy()
        for tick in range(1,6):
            changed=native.copy();changed[6:10]+=.01
            previous=ex.motor.copy();ex.step(tick,changed,native[6:10],[0,0,.34])
            assert np.max(abs(ex.motor[LEGS]-previous[LEGS]))<=.200001
        assert np.array_equal(ex.raw,raw) # no5ms PD when explicitly100Hz
    finally:ex.close()
