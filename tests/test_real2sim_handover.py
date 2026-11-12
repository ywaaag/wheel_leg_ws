FROZEN_TEST_SCOPE='PORTABLE'
import numpy as np
import pytest
from native_stand.students.real2sim_handover import decode_log, signal_metrics
from native_stand.students.real2sim_delay import ObservationHistory


def test_log_previous_is_physical_target_not_measured_velocity():
    x=np.zeros(31); x[4]=-1; x[6:10]=[.1,.2,-.1,.3]
    x[10:16]=.1; x[16:20]=.4; x[20:22]=[.02,-.03]; x[22:25]=[.1,.125,.68]
    d=decode_log(x)
    np.testing.assert_allclose(d['q'][[0,1,3,4]], [.15,.3,-.15,.45])
    np.testing.assert_allclose(d['previous'], [.6,.6,2.,.6,.6,-3.])
    np.testing.assert_allclose(d['dq'], np.ones(6)*2)
    np.testing.assert_allclose(d['command'], [.4,.5,.34])
    x[4]=0
    with pytest.raises(ValueError):decode_log(x)


def test_frequency_and_amplitude_match_known_sampled_oscillation():
    t=np.arange(200)*.01; y=np.zeros((200,6))
    for j in [0,1,3,4]:y[:,j]=.3*np.sin(2*np.pi*15*t)
    g=np.tile([0.,-1.,0.],(200,1)); r=signal_metrics(t,y,np.zeros_like(y),g)
    np.testing.assert_allclose(r['dominant_leg_target_hz'],15.)
    assert .55<r['leg_target_p95_minus_p5_rad'][0]<=.6
    assert r['upright_failure_time_s'] is None
    g[180:,1]=-.4
    assert signal_metrics(t,y,np.zeros_like(y),g)['upright_failure_time_s']==pytest.approx(1.8)


def test_negative_causal_history_never_uses_future_sample():
    h=ObservationHistory([1.],initial_tick=-20);h.append(-10,[2.]);h.append(0,[3.])
    np.testing.assert_array_equal(h.at(-15),[1.]);np.testing.assert_array_equal(h.at(-5),[2.])
