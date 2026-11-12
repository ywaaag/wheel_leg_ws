FROZEN_TEST_SCOPE='PORTABLE'
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

p=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(p))
spec=importlib.util.spec_from_file_location('handover_search',p/'search_motion_handover.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_grid_unique_bounded_and_no_double_wheel_lag():
    rows=m.coarse_candidates();assert len(rows)==96
    assert len({m.key(c,cache) for c,cache in rows})==96
    for c,cache in rows:
        assert c['target_delay_ms'][2:3]==[11]
        assert c['torque_tau_ms'][2]==0
    c=m.configuration(10,5,5,wheel='rise11_7')
    assert c['target_delay_ms'][2]==0 and c['torque_tau_ms'][2]==11
    with pytest.raises(ValueError):m.configuration(80,0,0)
    local=m.local_candidates([{'configuration':rows[0][0],'cache_hypothesis':rows[0][1]}])
    assert len(local)<=10


def test_signed_score_rejects_opposite_posture_and_missing_tail():
    t=np.arange(200)*.01;x=np.zeros((200,31));x[:,4]=-np.cos(.2);x[:,5]=np.sin(.2)
    action=np.zeros((200,6))
    for j in [0,1,3,4]:action[:,j]=.2*np.sin(2*np.pi*15*t)
    arrays=dict(time=t,target=action*m.SCALE,dq_pre=np.zeros((200,6)),gravity=x[:,3:6].copy())
    good=m.compare_window(t,x,action,arrays,0,.8)
    assert good['score']==pytest.approx(0.,abs=1e-6)
    arrays['gravity'][:,2]*=-1
    bad=m.compare_window(t,x,action,arrays,0,.8)
    assert bad['per_observable']['signed_pitch_roll_rmse']>0
    short={k:v[:5] for k,v in arrays.items()}
    assert m.compare_window(t,x,action,short,.8,1.6)['score']==100
