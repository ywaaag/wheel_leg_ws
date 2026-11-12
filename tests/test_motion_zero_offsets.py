"""Exercise real sensor injection/PD/time-axis boundaries without physics."""
FROZEN_TEST_SCOPE='PORTABLE'
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def module(name):
    from unittest.mock import patch
    with patch.object(sys,'path',[str(Path('scripts').resolve()),*sys.path]):
        spec=importlib.util.spec_from_file_location(name,Path('scripts')/(name+'.py'))
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_real_feedback_injection_does_not_move_true_state_or_change_other_signals():
    m=module('audit_motion_deployment_csv')
    s=np.arange(18,dtype=float);original=s.copy();bias=np.array([.01,-.02,.03,-.04])
    out=m.biased_sensors(s,bias)
    np.testing.assert_array_equal(s,original)
    expected=s.copy();expected[[6,7,9,10]]+=bias
    np.testing.assert_array_equal(out,expected)
    np.testing.assert_array_equal(m.biased_sensors(s,np.zeros(4)),s)
    for bad in ([1,2,3],[np.nan]*4,[[1,2],[3,4]]):
        with pytest.raises(ValueError):m.biased_sensors(s,bad)


def test_actual_executor_uses_biased_feedback_once_and_keeps_reported_previous():
    m=module('audit_motion_deployment_csv');exmod=module('compare_motion_timing_profiles')
    s=np.zeros(18);s[4]=-1;s[6:12]=[.2,.25,0,-.2,.25,0]
    b=np.array([-.04,-.08,.005,-.06]);reported=m.biased_sensors(s,b)
    bridge=m.target_in_reported_coordinates(s[6:12],b)
    cfg=dict(transport_ms=[0]*6,feedback_ms=0,tau_ms=[0]*6,wheel_peak=5.5,kp=40.,kd=2.,kv=.2)
    ex=exmod.AuditExecution(cfg,bridge,[(-32,reported),(0,reported)])
    # Reset bridge is q_true+b, so bias alone creates no reset torque impulse.
    np.testing.assert_allclose(ex.step(0,bridge,reported[6:12],reported[12:18]),0,atol=1e-12)
    request=bridge.copy();request[0]+=.1
    np.testing.assert_allclose(ex.step(10,request,reported[6:12],reported[12:18]),[4,0,0,0,0,0],atol=1e-12)
    np.testing.assert_array_equal(ex.adopted,request)
    # Equivalent physical target differs by exactly b, with no wheel reordering.
    physical=m.target_in_reported_coordinates(ex.adopted,-b)
    np.testing.assert_allclose(physical-s[6:12],[.1,0,0,0,0,0],atol=1e-12)
    # Same target, new 5ms feedback: actual PD is recomputed rather than held.
    nextq=reported[6:12].copy();nextq[0]+=.02
    np.testing.assert_allclose(ex.step(15,request,nextq,reported[12:18]),[3.2,0,0,0,0,0],atol=1e-12)


def test_episode_window_derived_offsets_and_fixed_matrix():
    m=module('diagnose_motion_zero_offsets');t=np.arange(100)*.01
    x=np.zeros((100,31));x[:,4]=-1;x[:,24]=.68
    reference=np.array([.2,.21,-.2,.21]);b=np.array([-.04,-.08,.005,-.06])
    x[:,6:10]=(reference+b)/1.5
    r=m.derive_bias(x,t,[.2,.8],reference)
    np.testing.assert_allclose(r['bias_rad'],b,atol=1e-15)
    cases=m.scenarios(r['bias_rad'],10.)
    assert len(cases)==7 and len({c[0] for c in cases})==7
    assert all(c[1] is None and c[2:5]==('zero','bridge',10.) for c in cases)
    assert cases[0][-1]=={}
    np.testing.assert_allclose(cases[3][-1]['actor_leg_zero_bias_rad'],b)
    np.testing.assert_allclose(cases[3][-1]['pd_leg_zero_bias_rad'],b)
    with pytest.raises(ValueError,match='20'):m.derive_bias(x,t,[0,.1],reference)
    t[10]=t[9]
    with pytest.raises(ValueError,match='increasing'):m.derive_bias(x,t,[.2,.8],reference)
