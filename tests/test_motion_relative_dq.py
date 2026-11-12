"""Relative q/dq timing fit is offline, chronological and not absolute latency."""
FROZEN_TEST_SCOPE='PORTABLE'
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import review_motion_relative_dq as tool


def test_exact_piecewise_linear_integral_and_no_extrapolation():
    t=np.array([0.,.009,.021,.03]);v=2*t+1
    query=np.array([0.,.003,.015,.03])
    np.testing.assert_allclose(tool.integral_piecewise(t,v,query),query+query**2,atol=1e-12)
    with pytest.raises(ValueError,match='outside'):tool.integral_piecewise(t,v,[-.001])


def test_relative_age_sign_when_filter_is_known_disabled(monkeypatch):
    t=np.arange(0,3,.01);age=.008
    q=lambda s:.2*np.sin(2*np.pi*2*s)+.03*np.sin(2*np.pi*4*s)
    v=lambda s:.2*2*np.pi*2*np.cos(2*np.pi*2*s)+.03*2*np.pi*4*np.cos(2*np.pi*4*s)
    monkeypatch.setattr(tool,'TAUS',[0.])
    r=tool.profile(t,q(t),v(t-age),[.2,1.4],[1.4,2.8])
    assert abs(r['selected']['relative_age_ms']-8)<=1
    assert abs(r['selected']['gain']-1)<.02
    assert r['selected']['validation']['interval_position_RMSE_rad']<.001


def test_unknown_filter_can_trade_against_relative_age():
    t=np.arange(0,3,.01);age=.008
    q=.2*np.sin(4*np.pi*t)+.03*np.sin(8*np.pi*t)
    v=.2*4*np.pi*np.cos(4*np.pi*(t-age))+.03*8*np.pi*np.cos(8*np.pi*(t-age))
    r=tool.profile(t,q,v,[.2,1.4],[1.4,2.8]);best=r['selected']
    # Same physical synthetic case: releasing an unknown filter can move the
    # minimum away from the known true age. Never force this into a pure delay.
    assert abs(best['relative_age_ms']+best['observer_tau_ms']-8)<=1
    assert best['validation']['interval_position_RMSE_rad']<.001


def test_validation_values_do_not_choose_profile_and_offset_cancels():
    t=np.arange(0,3,.01);q=np.sin(4*np.pi*t)*.1;v=np.cos(4*np.pi*(t-.003))*4*np.pi*.1
    first=tool.profile(t,q,v,[.2,1.4],[1.4,2.8]);changed=q.copy();changed[t>=1.4]+=.1*np.sin(3*t[t>=1.4])
    second=tool.profile(t,changed,v,[.2,1.4],[1.4,2.8]);offset=tool.profile(t,q+.3,v,[.2,1.4],[1.4,2.8])
    for k in ('relative_age_ms','observer_tau_ms','gain'):
        assert second['selected'][k]==pytest.approx(first['selected'][k],abs=1e-10)
        assert offset['selected'][k]==pytest.approx(first['selected'][k],abs=1e-10)


def test_dq_disabled_is_unidentifiable_not_a_free_latency_fit():
    t=np.arange(0,3,.01)
    r=tool.profile(t,np.sin(t),np.zeros_like(t),[.2,1.4],[1.4,2.8])
    assert not r['identifiable']
