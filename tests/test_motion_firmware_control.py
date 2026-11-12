"""Real compiled electrical source boundary; explicitly needs the external repo."""
FROZEN_TEST_SCOPE='PROJECT_DATA'
from pathlib import Path
import importlib.util
import numpy as np
import pytest
import sys

ROOT=Path(__file__).resolve().parents[1]
FW=Path('/home/kellen/combat-dm-balance-main')

def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.fixture(scope='module')
def compiled(tmp_path_factory):
    if not (FW/'tests/test_rl_execution.c').is_file():pytest.skip('electrical S1 sources absent; run explicit firmware CPU validator on owner host')
    out=tmp_path_factory.mktemp('firmware')/'build'
    v=load_script('validate_motion_firmware').build(FW,out)
    assert v['status']=='CPU_PASS' and len(v['tests'])==5 and v['source_unchanged_during_build']
    return out

def sensor():
    a=np.zeros(16,np.float32);a[4]=-1;return a

def test_real_firmware_pd_and_previous_are_owned_by_adoption(compiled):
    from native_stand.students.firmware_control import FirmwareControl
    with FirmwareControl(compiled/'motion_firmware.so') as c:
        a=sensor();a[6]=.1;a[10]=.2
        obs=c.observe(0,-10,a,[0,0,.34])
        assert obs[16]==pytest.approx(.1/1.5)  # explicit initialization proxy
        c.publish(0,[.2,0,.1,0,0,-.1])
        o=c.pd(0,-10,a,[0,0,.34],[.28,.28],[0,0])
        assert o['metadata'][0]==1 and o['pd_request_nm'][0]==pytest.approx(7.6)
        a[6]=.2;a[10]=1
        q=c.pd(5,-5,a,[0,0,.34],[.28,.28],[0,0])
        assert q['pd_request_nm'][0]==pytest.approx(2)
        assert q['metadata'][3]==o['metadata'][3]
        obs=c.observe(10,0,a,[0,0,.34]);assert obs[16]==pytest.approx(.3/1.5)
        c.publish(10,[.8,0,.1,0,0,-.1])
        # A prediction without PD adoption cannot change previous.
        assert c.observe(11,1,a,[0,0,.34])[16]==pytest.approx(.3/1.5)

def test_real_firmware_wheel_cap_matches_raw_current_conversion(compiled):
    from native_stand.students.firmware_control import FirmwareControl
    with FirmwareControl(compiled/'motion_firmware.so') as c:
        a=sensor();c.observe(0,-10,a,[0,0,.34]);c.publish(0,[0,0,1,0,0,-1])
        o=c.pd(0,-10,a,[0,0,.34],[.28,.28],[0,0])
        np.testing.assert_allclose(o['pd_limited_nm'][[2,5]],[5.5,-5.5])
        cap=8192/(19/.00036/(268/17))
        np.testing.assert_allclose(o['motor_model_nm'][[2,5]],[cap,-cap],rtol=2e-7)

@pytest.mark.parametrize('kind',['stale','invalid_q','invalid_gravity'])
def test_real_firmware_rejects_bad_inputs(compiled,kind):
    from native_stand.students.firmware_control import FirmwareControl
    with FirmwareControl(compiled/'motion_firmware.so') as c:
        a=sensor()
        if kind=='invalid_q':a[6]=1e30
        if kind=='invalid_gravity':a[4]=100
        with pytest.raises(ValueError,match='rejected'):
            c.observe(50,0 if kind=='stale' else 40,a,[0,0,.34])

def test_real_capture_roundtrip_and_torn_frame_refusal(compiled,tmp_path):
    import subprocess
    m=load_script('decode_motion_capture')
    source=compiled/'capture_synthetic.bin'
    h,a=m.decode(source)
    assert len(a)==128 and h['record_bytes']==m.DTYPE.itemsize==440
    assert h['frozen'] and a[-1]['policy_sequence']==99
    raw=bytearray(source.read_bytes());offset=4*len(m.HEADER)
    records=np.frombuffer(raw,dtype=m.DTYPE,offset=offset)
    index=(h['write_index']-1)%h['capacity'];records[index]['frame_begin']|=1
    bad=tmp_path/'torn.bin';bad.write_bytes(raw)
    with pytest.raises(ValueError,match='torn'):m.decode(bad)

def test_shared_fivebar_kinematics_runs_in_real_c_core(compiled):
    from native_stand.students.firmware_control import FirmwareControl
    with FirmwareControl(compiled/'motion_firmware.so') as c:
        length,spread=c.geometry([.21648856,.21663799,-.21648856,.21663799])
        np.testing.assert_allclose(length,[.27902275,.27902275],atol=2e-6)
        np.testing.assert_allclose(spread,[2.077427,2.077427],atol=2e-6)


def test_extended_public_validator_consumes_real_driver_and_snapshot_boundaries(tmp_path):
    if not (FW/'tests/run_motor_driver_host_tests.sh').is_file():
        pytest.skip('extended electrical driver tests are not installed on this host')
    result=load_script('validate_motion_firmware').build(FW,tmp_path/'extended',extended=True)
    assert result['status']=='CPU_PASS' and len(result['tests'])==6
    assert result['driver_boundary']['status']=='PASS'
    assert result['chassis_snapshot_boundary']['result']['returncode']==0
    assert len(result['shadow_modes'])==4
    assert len(result['chassis_output_boundary'])==2
    assert all(x['returncode']==0 for x in result['chassis_output_boundary'])
    assert result['chassis_command_boundary']['returncode']==0
    assert result['source_unchanged_during_build']


def execution_sensors():
    a=np.zeros(18,np.float32);a[4]=-1
    a[6:12]=[.21648856,.21663799,0,-.21648856,.21663799,0]
    return a


def test_explicit_zero_motor_timing_is_bitwise_legacy_compatible(compiled):
    from native_stand.students.firmware_control import FirmwareExecution
    a=execution_sensors();mean=a[6:12]/[1.5,1.5,100,1.5,1.5,100];mean[[2,5]]=.02
    plain=FirmwareExecution(compiled/'motion_firmware.so',a[6:12],'ideal_cancellation')
    zero=FirmwareExecution(compiled/'motion_firmware.so',a[6:12],'ideal_cancellation',motor_delay_ms=[0]*6,motor_tau_ms=[0]*6)
    try:
        for tick in range(21):
            if tick%10==0:
                for ex in (plain,zero):ex.observe(tick,tick,a,[0,0,.34]);ex.publish(tick,mean)
            np.testing.assert_array_equal(plain.step(tick,tick,a,[0,0,.34]),zero.step(tick,tick,a,[0,0,.34]))
            np.testing.assert_array_equal(plain.adopted,zero.adopted)
    finally:plain.close();zero.close()


def test_post_pd_wheel_transport_does_not_delay_local_previous(compiled):
    from native_stand.students.firmware_control import FirmwareExecution
    a=execution_sensors();mean=a[6:12]/[1.5,1.5,100,1.5,1.5,100];mean[[2,5]]=.02
    ex=FirmwareExecution(compiled/'motion_firmware.so',a[6:12],'motor_only',motor_delay_ms=[0,0,7,0,0,11])
    try:
        values=[]
        for tick in range(21):
            if tick%10==0:
                obs=ex.observe(tick,tick,a,[0,0,.34])
                if tick==10:assert obs[21]==pytest.approx(.02)  # ch5 still undelivered, but locally adopted
                ex.publish(tick,mean)
            values.append(ex.step(tick,tick,a,[0,0,.34]))
        values=np.array(values)
        np.testing.assert_array_equal(values[:7,2],0)
        np.testing.assert_array_equal(values[:11,5],0)
        np.testing.assert_allclose(values[7:,2],.4,rtol=2e-6)
        np.testing.assert_allclose(values[11:,5],.4,rtol=2e-6)
        with pytest.raises(ValueError,match='contiguous'):ex.step(23,23,a,[0,0,.34])
    finally:ex.close()


def test_post_pd_torque_response_is_once_per_one_ms_step(compiled):
    from native_stand.students.firmware_control import FirmwareExecution
    a=execution_sensors();mean=a[6:12]/[1.5,1.5,100,1.5,1.5,100]
    ex=FirmwareExecution(compiled/'motion_firmware.so',a[6:12],'motor_only',motor_tau_ms=[5]*6)
    try:
        values=[]
        for tick in range(10):
            if tick==0:ex.observe(0,0,a,[0,0,.34]);ex.publish(0,mean)
            values.append(ex.step(tick,tick,a,[0,0,.34]))
        requested=ex.last['motor_model_nm']
        expected=(1-np.exp(-np.arange(1,11)[:,None]/5))*requested
        np.testing.assert_allclose(values,expected,rtol=2e-6,atol=1e-7)
    finally:ex.close()
