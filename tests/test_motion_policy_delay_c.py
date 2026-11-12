"""Actual C adoption and source packet boundaries; no MuJoCo integration."""
FROZEN_TEST_SCOPE='PROJECT_DATA'
import json
from pathlib import Path
import numpy as np
import pytest
from native_stand.students.firmware_sampling import SourceExecution,source_parameters,profile
from native_stand.students.motion_policy_delay import PolicyDelay

VALIDATION=Path(__file__).resolve().parents[1]/'artifacts/2026_10_29__motion4500_compliance3000_engineering/firmware_cpu/validation.json'
LEGS=[0,1,3,4]


def run(obs_delay,target_delay,wrapped=True):
    if not VALIDATION.exists():pytest.skip('validated CPU C artifact unavailable')
    library=json.loads(VALIDATION.read_text())['library']
    s=np.zeros(18);s[4]=-1;s[6:12][LEGS]=[.24,.206,-.265,.223]
    previous=np.r_[s[6:12][LEGS]/1.5,[0.,0.]]
    ticks=[0]
    def actor(x):
        a=np.array([.15,.15,.01,-.15,.15,-.01],np.float32);a[0]+=ticks[0]*.0001
        return a[None]
    relay=PolicyDelay(actor,obs_delay,target_delay,previous)
    ex=SourceExecution(library,source_parameters(),np.zeros(6),profile(),s,previous)
    records=[]
    try:
        for t in range(71):
            ticks[0]=t
            ex.step_sampled(t,s,[0,0,.34],lambda x:relay.step(t,x) if wrapped else actor(x))
            if t%10==0:records.append(dict(adopted=ex.adopted.copy(),wire=ex.io.wire.copy(),actor=ex.policy['actor_input'].copy()))
        assert ex.nominal_error<2e-5
    finally:ex.close()
    return records,relay.records


def test_zero_relay_matches_real_C_path_exactly():
    a,_=run(0,0,False);b,_=run(0,0)
    for x,y in zip(a,b):
        for key in x:np.testing.assert_array_equal(x[key],y[key])


@pytest.mark.parametrize('delay',[20,30])
def test_real_C_previous_tracks_delivered_target_without_extending_freshness(delay):
    rows,relay=run(delay,delay)
    for i,r in enumerate(relay):
        if i:
            old=relay[i-1]['delivered_mean']
            np.testing.assert_allclose(r['actor_input'][16:22],old[[0,1,3,4,2,5]],atol=2e-8)
        np.testing.assert_array_equal(r['actor_input'][16:],rows[i]['actor'][16:])
        assert r['sensor_source_policy_tick_ms']<=r['tick_ms']
