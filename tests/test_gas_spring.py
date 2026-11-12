import torch
import pytest
from native_stand.recovery.gas_spring import SpringLaw,force_pair

def sample(rate=0.,length=.18):
 a=torch.tensor([[0.,0.,0.]],dtype=torch.float64);b=torch.tensor([[0.,length,0.]],dtype=torch.float64)
 ca=a+torch.tensor([[.02,0.,0.]]);cb=b+torch.tensor([[-.03,0.,0.]])
 return dict(position_a=a,position_b=b,com_a=ca,com_b=cb,velocity_com_a=a*0,velocity_com_b=torch.tensor([[0.,rate,0.]],dtype=torch.float64),omega_a=a*0,omega_b=a*0)

def test_balanced_wrench_and_virtual_power():
 r=force_pair(**sample(.1),law=SpringLaw(350,1700,.21,.14,30,40))
 torch.testing.assert_close(r['net_force_world'],torch.zeros(1,3,dtype=torch.float64))
 torch.testing.assert_close(r['net_torque_about_origin_world'],torch.zeros(1,3,dtype=torch.float64))
 assert r['force_a_world'][0,1]<0 and r['force_b_world'][0,1]>0
 assert r['force_n'].item()==pytest.approx(398.)
 assert r['power_w'].item()==pytest.approx(39.8)
 w=sample(.1);w['omega_a']+=torch.tensor([[.2,.3,.4]]);w['omega_b']+=torch.tensor([[.1,.5,.2]])
 r=force_pair(**w,law=SpringLaw(350,1700,.21,.14))
 p=(r['force_a_world']*w['velocity_com_a']).sum()+(r['force_b_world']*w['velocity_com_b']).sum()+(r['torque_a_about_com_world']*w['omega_a']).sum()+(r['torque_b_about_com_world']*w['omega_b']).sum()
 torch.testing.assert_close(p,r['power_w'].sum())

def test_extension_gas_cutoff_does_not_remove_stop_and_rng_unchanged():
 before=torch.random.get_rng_state().clone()
 r=force_pair(**sample(length=.22),law=SpringLaw(350,1700,.20,.14))
 assert not r['stroke_valid'].item() and r['force_n'].item()==0
 assert torch.equal(before,torch.random.get_rng_state())
 z=force_pair(**sample(),law=SpringLaw(0,0,.21,.14))
 assert not z['force_a_world'].any() and not z['torque_a_about_com_world'].any()

def test_damping_opposes_extension_and_compression():
 law=SpringLaw(350,0,.21,.14,30,40)
 assert force_pair(**sample(.1),law=law)['damping_force_n'].item()==pytest.approx(-3.)
 assert force_pair(**sample(-.1),law=law)['damping_force_n'].item()==pytest.approx(4.)
 with pytest.raises(ValueError):SpringLaw(350,0,.14,.21)

def test_two_stops_separate_from_gas_and_abort_on_excess_penetration():
 from native_stand.recovery.gas_spring import EndStops,with_end_stops
 law=SpringLaw(350,0,.215,.155);stops=EndStops(10000,50,.003,1000)
 def calc(length,rate):
  w=sample(rate,length);r=force_pair(**w,law=law)
  return with_end_stops(r,w['position_a'],w['position_b'],w['com_a'],w['com_b'],law,stops)
 lower=calc(.154,-.1);upper=calc(.216,.1)
 assert lower['stop_reaction_n'].item()==pytest.approx(15)
 assert upper['stop_reaction_n'].item()==pytest.approx(-15)
 assert lower['stop_power_w'].item()<0 and upper['stop_power_w'].item()<0
 assert upper['force_n'].item()==0 # gas cut off; mechanical reaction remains
 assert upper['force_b_world'][0,1]<0
 torch.testing.assert_close(upper['net_force_world'],torch.zeros(1,3,dtype=torch.float64))
 with pytest.raises(ValueError,match='penetration'):calc(.220,0)
