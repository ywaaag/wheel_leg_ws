"""Production C packet/driver/control boundaries, needs read-only electrical repo."""
FROZEN_TEST_SCOPE='PROJECT_DATA'
import ctypes as C
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pytest

from native_stand.students.firmware_control import FP, UP
from native_stand.students.firmware_sampling import DriverPackets, SampledControl, SourceExecution, profile, source_parameters

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    repo=Path('/home/kellen/combat-dm-balance-main')
    if not (repo/'tests/host_driver').is_dir():pytest.skip('electrical source host adapters unavailable')
    sys.path.insert(0,str(ROOT/'scripts'))
    try:
        s=spec_from_file_location('validate_motion_sampling',ROOT/'scripts/validate_motion_sampling.py')
        m=module_from_spec(s);s.loader.exec_module(m)
        result=m.build(repo,tmp_path_factory.mktemp('packet')/'host')
        assert result['status']=='CPU_PASS' and result['source_unchanged']
        return result['library']
    finally:sys.path.pop(0)


def sensors():
    s=np.zeros(18);s[4]=-1;s[6:12][[0,1,3,4]]=[.24,.206,-.265,.223]
    return s


def test_real_leg_decoder_quantization_direction_and_sample_hold(library):
    p=profile();p['leg_packet_period_ms']=5;s=sensors();io=DriverPackets(library,p,s)
    io.update(0,s);first=io.current.copy();receipt=io.receipt.copy();seq=io.sequence.copy()
    assert np.allclose(first[6:10],s[6:12][[0,1,3,4]],atol=2e-4)
    moving=s.copy();moving[6:12][[0,1,3,4]]+=.1;moving[12:18][[0,1,3,4]]=[1.,2.,3.,4.]
    for tick in range(1,5):
        io.update(tick,moving)
        assert np.array_equal(io.current[6:10],first[6:10])
        assert np.array_equal(io.receipt[[0,1,3,4]],receipt[[0,1,3,4]])
        assert np.array_equal(io.sequence[[0,1,3,4]],seq[[0,1,3,4]])
    io.update(5,moving)
    assert np.allclose(io.current[6:10],moving[6:12][[0,1,3,4]],atol=2e-4)
    assert np.allclose(io.current[10:16][[0,1,3,4]],[1,2,3,4],atol=.0111)
    assert (io.receipt[[0,1,3,4]]==5).all()


def test_real_wheel_filter_updates_per_packet_not_per_read_or_millisecond(library):
    p=profile();p['wheel_packet_period_ms']=5;s=sensors();io=DriverPackets(library,p,s)
    moving=s.copy();moving[12:18][[2,5]]=[2.,-3.]
    io.update(0,moving);first=io.current[10:16][[2,5]].copy()
    for tick in range(1,5):
        io.update(tick,moving);assert np.array_equal(io.current[10:16][[2,5]],first)
    io.update(5,moving);second=io.current[10:16][[2,5]].copy()
    assert np.allclose(second,1.15*first,atol=1e-6)
    io.update(6,moving);assert np.array_equal(io.current[10:16][[2,5]],second)


def test_sample_time_can_receipt_and_robot_snapshot_are_not_conflated(library):
    p=profile();p.update(leg_sample_age_ms=5,can_receive_delay_ms=2)
    s=sensors();io=DriverPackets(library,p,s)
    for tick in range(11):
        v=s.copy();v[6]+=.001*tick;io.update(tick,v)
    assert (io.receipt[[0,1,3,4]]==10).all()
    assert (io.sample_tick[[0,1,3,4]]==3).all()
    assert io.current[6]==pytest.approx(s[6]+.003,abs=2e-4)
    assert io.receipt[6]==10 # current synthetic IMU, not the leg sampling time


def test_actual_motor_task_sends_legs_every_other_call_and_wire_torque_is_quantized(library):
    io=DriverPackets(library,profile(),sensors());io.update(0,sensors())
    a=np.array([1.,2.,.5,-3.,-4.,-.7]);b=-a
    first=io.send(0,a).copy();sequence=io.wire_sequence.copy()
    assert np.allclose(first,a,atol=.021)
    second=io.send(1,b)
    assert np.array_equal(second[[0,1,3,4]],first[[0,1,3,4]])
    assert np.array_equal(io.wire_sequence[[0,1,3,4]],sequence[[0,1,3,4]])
    assert np.allclose(second[[2,5]],b[[2,5]],atol=.001)
    third=io.send(2,b);assert np.allclose(third,b,atol=.021)
    assert (io.wire_tick[[0,1,3,4]]==2).all()


def test_individually_stale_feedback_cannot_hide_behind_new_snapshot_time(library):
    with SampledControl(library) as core:
        native=np.r_[sensors()[:6],sensors()[6:12][[0,1,3,4]],sensors()[12:18]]
        receipt=np.ones(7,dtype=int)*100;receipt[1]=75
        with pytest.raises(ValueError,match='-2'):core.observe_packet(100,100,receipt,native,[0,0,.34])
        receipt[1]=76
        assert core.observe_packet(100,100,receipt,native,[0,0,.34]).shape==(31,)


@pytest.mark.parametrize('settings,first_adoption,first_send',[
    ({},5,6),({'motor_send_phase_ms':1},5,5),({'actor_phase_ms':4,'inference_ms':1},10,10),
    ({'motor_before_robot':True},5,6),({'robot_phase_ms':2},12,12)])
def test_actor_robot_and_driver_phases_preserve_real_adoption(library,settings,first_adoption,first_send):
    p=profile();p.update(settings);s=sensors();physical=source_parameters()
    mean=np.array([.15,.15,.01,-.15,.15,-.01],np.float32)
    initial=np.zeros(6);first_changed=None
    ex=SourceExecution(library,physical,initial,p,s,np.zeros(6))
    try:
        for tick in range(31):
            ex.step_sampled(tick,s,[0,0,.34],lambda x:mean[None])
            if ex.first_adoption_tick is not None and first_changed is None:
                if np.any(abs(ex.io.wire[[0,1,3,4]])>.03):first_changed=tick
        assert ex.first_adoption_tick==first_adoption and first_changed==first_send
        assert np.allclose(ex.policy['actor_input'][16:22],mean[[0,1,3,4,2,5]],atol=1e-6)
        assert (ex.policy['receipt_tick_ms'][:6]<=ex.policy['snapshot_tick_ms']).all()
        assert ex.policy['snapshot_tick_ms']<=ex.policy['tick_ms']
        assert ex.nominal_error<2e-5
    finally:ex.close()


def test_legacy_host_core_abi_remains_available_in_new_driver_library(library):
    from native_stand.students.firmware_control import FirmwareControl
    with FirmwareControl(library) as core:
        s=sensors();native=np.r_[s[:6],s[6:12][[0,1,3,4]],s[12:18]]
        o=core.observe(0,0,native,[0,0,.34]);core.publish(0,[.1,.1,0,-.1,.1,0])
        out=core.pd(0,0,native,[0,0,.34],*core.geometry(native[6:10]))
        assert o.shape==(31,) and out['metadata'][0]==1
