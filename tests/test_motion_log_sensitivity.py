FROZEN_TEST_SCOPE='PORTABLE'
import importlib.util
from pathlib import Path
import numpy as np
import pytest

p=Path(__file__).resolve().parents[1]/'scripts/audit_motion_log_sensitivity.py'
import sys
sys.path.insert(0,str(p.parent))
spec=importlib.util.spec_from_file_location('log_sensitivity',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_delay_uses_timestamps_not_fixed_row_offset():
    x=np.zeros((4,31),np.float32);x[:,10]=[0,1,2,3];t=np.array([0.,.009,.021,.03])
    y,age=m.delayed_columns(x,t,[10],5,'past_sample')
    np.testing.assert_array_equal(y[:,10],[0,0,1,2]);np.testing.assert_allclose(age,[0,.009,.012,.009])
    y,age=m.delayed_columns(x,t,[10],5,'linear_estimate')
    assert y[2,10]==pytest.approx(1+7/12)
    y,_=m.delayed_columns(x,t,[10],0,'past_sample');np.testing.assert_array_equal(x,y)


def test_proxy_scaling_recursive_seed_and_input_immutability():
    x=np.zeros((4,31),np.float32);x[:,12]=.5;x[:,15]=-.25;x[:,20:22]=.9;x[:,6:10]=.3
    original=x.copy();infer=lambda v:np.column_stack([v[:,16],v[:,17],v[:,20]+.1,v[:,18],v[:,19],v[:,21]-.1])
    rows={name:(v,y) for name,method,v,y,meta in m.replay_variants(x,np.arange(4)*.01,infer)}
    np.testing.assert_allclose(rows['measured_wheels_proxy'][0][:,20:22],[[.1,-.05]]*4)
    np.testing.assert_allclose(rows['recursive_shadow_previous'][1][:,2],[1.,1.1,1.2,1.3])
    np.testing.assert_array_equal(x,original)
    np.testing.assert_allclose(rows['measured_q_wheels_proxy'][0][:,16:20],x[:,6:10])


def test_physical_roughness_uses_output_channel_scales():
    y=np.zeros((4,6));y[:,0]=[0,.1,-.1,.1];y[:,2]=[0,.1,0,.1]
    r=m.output_metrics(y,np.zeros_like(y),np.arange(4)*.01)
    assert r['physical_step_abs_max'][0]==pytest.approx(.3)
    assert r['physical_step_abs_max'][2]==10
