"""Centered float64 diagnostics; does not change value loss."""
import pytest
import torch

from native_stand.tasks.motion.value_diagnostics import grouped


def test_small_variance_at_return_scale_11_is_preserved():
    r = (11.4+torch.linspace(-1,1,32768)*.001).float()
    error = torch.linspace(-1,1,len(r))*.0002
    v = r-error
    out = grouped(r,v,error,torch.ones_like(r,dtype=torch.bool),torch.zeros_like(r),[[0,0]])['rows'][0]
    reference = 1-torch.var((r-v).double(),unbiased=False)/torch.var(r.double(),unbiased=False)
    assert out['return_variance'] > 3e-7
    assert out['explained_variance'] == pytest.approx(float(reference), abs=1e-12)
    assert .95 < out['explained_variance'] < .97


def test_group_masks_zero_variance_and_empty_rows_are_not_claimed_pass():
    ret = torch.tensor([1.,1.,4.,8.]); v = ret.clone()
    result = grouped(ret,v,ret*0,torch.tensor([True,True,True,False]),torch.tensor([0,0,1,2]),
        [[0,0],[1,0],[-1,0]])
    assert [r['samples'] for r in result['rows']]==[2,1,0]
    assert all(r['explained_variance'] is None for r in result['rows'])
    assert result['rows'][2]['return_variance'] is None
    assert result['rows'][2]['value_error_MSE'] is None


def test_nonconstant_exact_value_is_EV_one_and_other_group_cannot_contaminate():
    ret = torch.tensor([1.,3.,1000.,2000.])
    val = torch.tensor([1.,3.,0.,0.])
    rows=grouped(ret,val,ret-val,torch.ones(4,dtype=torch.bool),torch.tensor([0,0,1,1]),
        [[0,0],[1,0]])['rows']
    assert rows[0]['explained_variance']==1.
    assert rows[1]['explained_variance']==0.
    assert rows[1]['value_error_MSE']==2500000.
