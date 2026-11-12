import pytest
import torch
from native_stand.students.motion_symmetry_design import EXTRA_FIELDS,critic_input,density,audit

def test_named85_and_phase_validation():
 base=torch.zeros(3,61);f={n:torch.zeros(3,w) for n,w,_ in EXTRA_FIELDS};f['schedule_phase_onehot'][:,2]=1
 assert critic_input(base,f).shape==(3,85)
 del f['leg_swing_rad']
 with pytest.raises(ValueError):critic_input(base,f)

def test_no_dead_tail_and_symmetric():
 for mix in (0.,.75):
  for e in (.03,.05,.065,.2,.5,1.):
   x=torch.tensor(e,requires_grad=True);v=density(x,x,torch.stack((x,x*0)),narrow_mix=mix)
   for name,y in v.items():
    assert torch.autograd.grad(y,x,retain_graph=True)[0]<0
    assert torch.isfinite(y)
   opposite=density(-x,-x,torch.stack((-x,x*0)),narrow_mix=mix)
   assert all(torch.allclose(v[k],opposite[k]) for k in v)

def test_zero_is_best_and_mix_bounded():
 z=torch.tensor(0.)
 assert all(x==0 for x in density(z,z,torch.zeros(2)).values())
 with pytest.raises(ValueError):density(z,z,torch.zeros(2),narrow_mix=1.)
 assert audit()['runtime_registered'] is False
