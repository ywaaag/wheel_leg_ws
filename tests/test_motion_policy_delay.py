FROZEN_TEST_SCOPE='PORTABLE'
import numpy as np
import pytest
from native_stand.students.motion_policy_delay import PolicyDelay,cases


def observation(tick):
    x=np.zeros((1,31),np.float32);x[:,:16]=tick/100
    x[:,16:]=np.arange(15)/100+tick/1000
    return x


def actor(x):return x[:,:6]*.5


def test_zero_delay_exact_identity_no_alias():
    q=PolicyDelay(actor,0,0,np.zeros(6))
    for tick in range(0,61,10):
        x=observation(tick);before=x.copy();out=q.step(tick,x)
        np.testing.assert_array_equal(out,actor(x));np.testing.assert_array_equal(x,before)
        np.testing.assert_array_equal(q.records[-1]['actor_input'],x[0])


def test_sensor_delay_keeps_previous_and_command_current_without_future():
    q=PolicyDelay(actor,20,0,np.zeros(6))
    for tick in range(0,61,10):
        x=observation(tick);q.step(tick,x);r=q.records[-1]
        np.testing.assert_array_equal(r['actor_input'][:16],observation(max(0,tick-20))[0,:16])
        np.testing.assert_array_equal(r['actor_input'][16:],x[0,16:])
        assert r['sensor_source_policy_tick_ms']==max(0,tick-20)
        assert r['prehistory_clamped']==(tick<20)


def test_target_queue_holds_declared_previous_and_delivers_exact_old_mean():
    previous=np.array([.1,.2,.3,.4,.05,-.05],np.float32)
    q=PolicyDelay(actor,0,20,previous)
    expected=previous[[0,1,4,2,3,5]]
    for tick in range(0,71,10):
        out=q.step(tick,observation(tick))[0]
        np.testing.assert_array_equal(out,expected if tick<20 else actor(observation(tick-20))[0])
        assert q.records[-1]['target_generation_tick_ms']==(-1 if tick<20 else tick-20)


def test_no_silent_clipping_or_skipped_clock():
    with pytest.raises(ValueError):PolicyDelay(actor,25,0,np.zeros(6))
    with pytest.raises(ValueError):PolicyDelay(actor,0,0,np.ones(6)*2)
    q=PolicyDelay(actor,0,0,np.zeros(6));q.step(0,observation(0))
    with pytest.raises(ValueError):q.step(20,observation(20))


def test_matrix_is_fixed_and_transport_mechanisms_not_doubled():
    rows=cases();assert len(rows)==36 and len({r['name'] for r in rows})==36
    assert {r['observation_ms'] for r in rows}=={0,10,20,30}
    assert {r['target_ms'] for r in rows}=={0,10,20,30}
    for r in rows:assert not (r['motor_delay_ms'] and r['target_ms'])
    assert sum(r['profile']=='F40' for r in rows)==8
