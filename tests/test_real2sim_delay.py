FROZEN_TEST_SCOPE='PORTABLE'
import numpy as np
import pytest
from native_stand.students.real2sim_delay import TargetTransport,ObservationHistory,torque_response,scenarios


def test_transport_has_per_channel_clock_and_no_early_delivery():
    q=TargetTransport([0,5,11,0,10,7],np.zeros(6));q.submit(0,np.ones(6))
    np.testing.assert_array_equal(q.delivered(0),[1,0,0,1,0,0])
    np.testing.assert_array_equal(q.delivered(7),[1,1,0,1,0,1])
    q.submit(10,np.full(6,2.))
    np.testing.assert_array_equal(q.delivered(10),[2,1,0,2,1,1])
    np.testing.assert_array_equal(q.delivered(11),[2,1,1,2,1,1])
    np.testing.assert_array_equal(q.delivered(21),np.full(6,2.))


def test_transport_delay_zero_identity_and_invalid_inputs():
    q=TargetTransport([0]*6,np.zeros(6))
    for tick in range(0,100,10):
        q.submit(tick,np.full(6,tick));np.testing.assert_array_equal(q.delivered(tick),np.full(6,tick))
    for d in ([1]*5,[.5]*6,[-1]*6,[101]*6):
        with pytest.raises(ValueError):TargetTransport(d,np.zeros(6))


def test_observation_history_returns_real_sample_and_no_future():
    h=ObservationHistory([0,0])
    for tick in range(1,151):h.append(tick,[tick,-tick])
    np.testing.assert_array_equal(h.at(143),[143,-143])
    np.testing.assert_array_equal(h.at(130),[130,-130])
    with pytest.raises(ValueError):h.append(100,[0,0])


def test_response_tau_zero_is_identity_and_tau_is_not_transport_delay():
    p=np.zeros(6);want=np.ones(6)
    np.testing.assert_allclose(torque_response(p,want,[0]*6),want)
    result=torque_response(p,want,[10]*6)
    np.testing.assert_allclose(result,np.full(6,1-np.exp(-.1)))
    assert (result>0).all() # immediate finite rise, unlike a pure dead-time queue


def test_cases_change_one_mechanism_and_preserve_channel_order():
    rows=scenarios();assert len({x['name'] for x in rows})==len(rows)
    base=next(x for x in rows if x['name']=='baseline_reverse')
    for row in rows:
        diff=[k for k in base if k!='name' and row[k]!=base[k]]
        assert len(diff)<=1
    assert next(x for x in rows if x['name']=='wheel_transport_11_7')['target_delay_ms']==[0,0,11,0,0,7]
