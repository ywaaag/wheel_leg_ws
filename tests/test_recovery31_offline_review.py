"""Offline ranking must not turn a long but invalid window into acceptance."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/review_recovery31_checkpoints.py'
spec=importlib.util.spec_from_file_location('offline_review',path)
review=importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_capability_ranking_success_then_entry_then_prefix_then_earlier():
    def r(u,passed,entry,prefix):
        return dict(update=u,accepted_envs=passed,entry_envs=entry,physical_hold_prefix_sum_s=prefix)
    rows=[r(500,0,32,320),r(1500,0,32,270),r(100,0,32,320),r(2000,1,1,10)]
    assert [x['update'] for x in sorted(rows,key=review.ranking,reverse=True)]==[2000,100,500,1500]


def test_window_units_order_and_contact_resultant():
    n=3
    a={'state':np.zeros((n,1,39)),'posture':np.zeros((n,1,10)),
       'control':np.zeros((n,1,54)),'joint_q_pre':np.zeros((n,1,6)),
       'root_pre':np.zeros((n,1,13)),'root_post':np.zeros((n,1,13)),
       '_active_indices':[0,1,2,3,4,5]}
    a['state'][:,:,6]=.34;a['state'][:,:,10:12]=3
    a['joint_q_pre'][:,:,0]=.2;a['joint_q_pre'][:,:,3]=-.2
    a['joint_q_pre'][:,:,1]=.4;a['joint_q_pre'][:,:,4]=.4
    a['control'][:,0,20]=[0,.1,.2] # left wheel clipped action -> 0/10/20 rad/s
    r=review.window_metrics(a,0,np.arange(n))
    assert r['wheel_target_delta_rms_rad_s']==[10,0]
    assert r['hip_active_mirror_abs_mean_rad']==[0,0]
    assert r['dual_contact_fraction']==1
    assert r['height_min_mean_max_m'][0]==.34


def test_refuses_existing_or_run_local_output(tmp_path):
    run=tmp_path/'run';run.mkdir()
    with pytest.raises(ValueError,match='Fresh output'):
        review.review(run,run/'review')
    output=tmp_path/'existing';output.mkdir()
    with pytest.raises(ValueError,match='Fresh output'):
        review.review(run,output)


def test_fixed_physical_prefix_does_not_claim_velocity_acceptance():
    s=np.zeros((1200,39));p=np.zeros((1200,10))
    s[:,6]=.34;s[:,10:12]=3;s[:,13]=.02;s[:,7]=2
    assert review.physical_prefix(s,p,20)==10 # speed is a separate certificate item
    p[30,0]=.2
    assert review.physical_prefix(s,p,20)==.09
    assert review.physical_prefix(s,p,None)==0
