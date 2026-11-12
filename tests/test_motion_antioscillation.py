FROZEN_TEST_SCOPE='PORTABLE'
import copy
from types import SimpleNamespace as S
import numpy as np
import pytest
import torch
from native_stand.students import motion_antioscillation as m


def test_excess_cost_has_tail_signal_mask_and_dt():
 n=3;r=torch.zeros(n,1);base={'fields':('base',),'raw':r,'density':r.clone(),'dt':r.clone(),'total':torch.zeros(n),'eligible':torch.tensor([True,True,False])}
 request=torch.tensor([[0.,0.,11.,0.,0.,11.],[0.,0.,90.,0.,0.,90.],[0.,0.,90.,0.,0.,90.]],requires_grad=True)
 z=torch.zeros(n,6);contacts=torch.full((n,2),100.)
 out=m.apply_costs(base,request,z,z,z,z,z,contacts,torch.zeros(n,dtype=torch.bool),m.configuration())
 assert out['raw'][1,1]>out['raw'][0,1]>0 and out['total'][1]<out['total'][0]<0
 assert out['total'][2]==0 and out['fields']==('base',*m.FIELDS)
 g=torch.autograd.grad(out['total'].sum(),request)[0]
 assert g[1,2]<0 and g[1,5]<0 and g[2].abs().sum()==0
 assert torch.allclose(out['total'],out['density'].sum(-1)*.01)
 assert all(row['derivative_per_Nm']<0 for row in m.reward_snapshot()['curves'] if row['request_abs_Nm']>4.4)


def test_history_reset_does_not_penalize_initial_curvature():
 z=torch.zeros(1,6);r=torch.zeros(1,1);base={'fields':('base',),'raw':r,'density':r.clone(),'dt':r.clone(),'total':torch.zeros(1),'eligible':torch.ones(1,dtype=torch.bool)}
 out=m.apply_costs(base,z,z+5,z,z+1,z,z,torch.zeros(1,2),torch.zeros(1,dtype=torch.bool),m.configuration())
 assert out['raw'][0,2:5].sum()==0 and out['raw'][0,-1]>0


def record(stage):
 return [{'case':{'case_id':f'armature_{stage}_level_{level}','_timing_level':level,'command':[0.,0.,.34]},
 'per_env':[{'env':i,'accepted':True,'symmetry_metrics':{}} for i in range(8)]} for level in sorted({0,max(0,stage-1),stage})]


def test_course_needs_two_complete_matrices_and_rejects_wrong_levels():
 s={'stage':0,'last_evaluation_update':-1,'consecutive_passes':0,'accepted_by_stage':{}}
 for stage in range(5):
  for step in (1,2):
   update=stage*500+step*250;c={'stage':stage,'update':update,'complete_matrix':True,'evaluations':record(stage),'checkpoint':'fixture','sha256':str(stage)*64}
   s,d=m.advance(s,c)
   assert d['advanced']==(step==2 and stage<4)
  assert d['all_stages_passed']==(stage==4)
 assert s['stage']==4 and set(s['accepted_by_stage'])==set(map(str,range(5)))
 with pytest.raises(ValueError):m.advance(s,c)
 bad=record(4);bad[0]['case']['_timing_level']=4
 assert not m.refinement_records(bad,4)[0]['per_env']
 bad=record(2);bad[0]['per_env'][0]['accepted']=False
 assert not m.refinement_records(bad,2)[0]['per_env'][0]['accepted']


def test_assessment_requires_whole_window_and_nonoscillating_controls():
 def row():return {'accepted':True,'checks':{},'failed_checks':[]}
 c=np.zeros((3152,54));s=np.zeros((3152,39));s[:,29]=1;v=np.ones(3152,dtype=bool)
 assert m.assess(row(),c,v,s)['accepted']
 c[:,[20,23]]=np.arange(len(c))[:,None]%2
 x=m.assess(row(),c,v,s);assert not x['accepted'] and not x['checks']['antiosc_wheel_delta']
 assert not m.assess(row(),None,v,s)['accepted']
 v[:200]=False;assert not m.assess(row(),np.zeros_like(c),v,s)['accepted']


def test_dynamic_recipe_fields_and_old_defaults():
 from native_stand.students.motion_tracking_reward import effective_reward_fields,configuration,PARALLEL_VERSION
 from native_stand.students.motion_sensor_transfer import configuration as old
 s={'reward':configuration(PARALLEL_VERSION),'motion_timing':{'version':m.VERSION,'configuration':m.configuration()}}
 assert effective_reward_fields(s)[-5:]==m.FIELDS and 'shaping' not in old()
 s['motion_timing']['configuration']['smooth_mix']=1
 with pytest.raises(ValueError):effective_reward_fields(s)


def test_runtime_fixed_delay_stage4_and_cohort_backend_readback():
 from test_motion_timing import fake_env
 from native_stand.students.motion_timing_runtime import Timing
 e=fake_env(8);e.cfg=S(leg_kp=40.,leg_kd=2.,wheel_kv=1.)
 values=torch.ones(8,6)*.01
 def write(*,armature,joint_ids):values[:,joint_ids.long()]=armature
 e.robot.write_joint_armature_to_sim_index=write;e.robot.root_view=S(get_dof_armatures=lambda:values)
 rng=torch.get_rng_state().clone();t=Timing(e,m.configuration(),4);t.configure(None)
 assert torch.allclose(values[:,0],torch.tensor([.35,.02,.012,.012,.35,.02,.012,.012]))
 for level in range(5):
  t.configure(level);t.reset(torch.arange(8));assert torch.allclose(values[:,0],torch.full((8,),m.ARMATURES[level]))
  assert (t.feedback==10).all() and (t.delays[:,2]==7).all() and (t.delays[:,5]==11).all()
 assert torch.equal(rng,torch.get_rng_state())
