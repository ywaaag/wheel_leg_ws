from types import SimpleNamespace
import torch
from native_stand.recovery.gas_spring import SpringLaw,EndStops
from native_stand.recovery.gas_spring_runtime import GasSpringWrenches

def test_wrench_adapter_adds_balanced_world_point_forces_each_substep():
 class Composer:
  def __init__(self):self.calls=[]
  def add_forces_and_torques_index(self,**kwargs):self.calls.append(kwargs)
 def proxy(x):return SimpleNamespace(torch=x)
 positions=torch.tensor([[[0.,0.,0.],[0.,.18,0.],[.2,0.,0.],[.2,.18,0.]]],dtype=torch.float64)
 robot=SimpleNamespace(body_names=['left_lower_leg','left_upper_leg','right_lower_leg','right_upper_leg'],instantaneous_wrench_composer=Composer())
 robot.data=SimpleNamespace(body_link_pos_w=proxy(positions),body_link_quat_w=proxy(torch.tensor([[[0.,0.,0.,1.]]*4],dtype=torch.float64)),body_com_pos_w=proxy(positions.clone()),body_com_lin_vel_w=proxy(torch.zeros_like(positions)),body_com_ang_vel_w=proxy(torch.zeros_like(positions)))
 geometry={'version':'gas_spring_bilateral_planar_geometry_v2','springs':[{'side':side,'anchors':[{'body':side+'_'+link,'local_m':[0.,0.,0.]} for link in ['lower_leg','upper_leg']]} for side in ['left','right']]}
 adapter=GasSpringWrenches(robot,geometry,SpringLaw(350,0,.215,.155),EndStops(10000,50,.003,1000))
 before=torch.random.get_rng_state().clone();adapter.apply();adapter.apply()
 assert len(robot.instantaneous_wrench_composer.calls)==2
 c=robot.instantaneous_wrench_composer.calls[0]
 assert c['is_global'] and c['body_ids'].tolist()==[0,1,2,3]
 torch.testing.assert_close(c['forces'].sum(1),torch.zeros(1,3,dtype=torch.float64))
 torch.testing.assert_close(c['positions'],positions)
 assert torch.equal(before,torch.random.get_rng_state())
 solver=GasSpringWrenches(robot,geometry,SpringLaw(350,0,.215,.155),None,stop_mode='knee_solver')
 result=solver.apply()
 assert solver.stop_mode=='knee_solver' and 'stop_reaction_n' not in result[0]
 import pytest
 with pytest.raises(ValueError,match='double-count'):
  GasSpringWrenches(robot,geometry,SpringLaw(350,0,.215,.155),EndStops(10000,50,.003,1000),stop_mode='knee_solver')
