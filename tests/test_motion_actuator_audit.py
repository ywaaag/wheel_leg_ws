"""CPU contract checks for bounded actuator diagnostics; no engine stepping."""
FROZEN_TEST_SCOPE='PORTABLE'
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

scripts=Path(__file__).resolve().parents[1]/'scripts'
if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('actuator_audit_test',scripts/'compare_motion_timing_profiles.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_matrix_contains_paired_models_and_separates_historical_changes():
 rows=m.actuator_cases()
 assert len(rows)==81
 assert len({(r['policy'],r['initial'],r['configuration']['name']) for r in rows})==81
 physics=[r for r in rows if r['group']=='wheel_physics']
 assert len(physics)==30 and all(r['configuration']['cache']=='adopted' for r in physics)
 for profile in {r['configuration']['name'] for r in physics}:
  assert {(r['policy'],r['initial']) for r in physics if r['configuration']['name']==profile}==set(__import__('itertools').product(('teacher3250','small3250','final500'),('normal','log')))
 base,profiles=m.actuator_profiles()
 for p in profiles:
  changes={k for k in base if k not in ('name','scope') and base[k]!=p[k]}
  assert changes<=({'wheel_B','wheel_C'} if p['name']=='wheel_friction_reported' else {'wheel_J'} if p['name']=='wheel_J_reported' else {'wheel_peak'})
 assert all(r['configuration']['transport_ms'][2::3]==[7,11] for r in rows if r['configuration']['name']!='wheel_transport0')
 followup=m.actuator_cases(confirm=True)
 assert len(followup)==9 and {r['initial'] for r in followup}=={'log'}
 low=next(r['configuration'] for r in followup if r['configuration']['name']=='leg_arm012_only')
 assert {k for k in base if low[k]!=base[k]}=={'name','leg_armature'}


def test_cpu_execution_matches_actual_timing_runtime_and_preserves_rng():
 from native_stand.students.motion_timing_runtime import Timing
 from native_stand.students.motion_fixed_timing import configuration
 from test_motion_timing import fake_env
 e=fake_env(1);e.cfg=SimpleNamespace(leg_kp=40.,leg_kd=2.,wheel_kv=1.)
 runtime=Timing(e,configuration(),0);cfg,_=m.actuator_profiles()
 audit=m.AuditExecution(cfg,np.zeros(6),[(-20,np.zeros(18))]);rng=torch.get_rng_state().clone()
 for tick in range(100):
  q=np.full(6,np.sin(tick*.01)*.03);dq=np.full(6,np.cos(tick*.01)*.03)
  want=np.array([.1,.1,3.,.1,.1,-3.]) if tick>=20 else np.zeros(6)
  e.robot.data.joint_pos.torch[:]=torch.tensor(q);e.robot.data.joint_vel.torch[:]=torch.tensor(dq)
  runtime.requested[:]=torch.tensor(want);runtime.apply()
  actual=audit.step(tick,want,q,dq)
  np.testing.assert_allclose(actual,e.effort[0].numpy(),atol=2e-6)
  np.testing.assert_allclose(audit.pre,e.effort_pre[0].numpy(),atol=2e-6)
  np.testing.assert_allclose(audit.adopted,runtime.adopted[0].numpy(),atol=2e-6)
 assert torch.equal(rng,torch.get_rng_state())


def test_total_wheel_inertia_replaces_extra_and_each_case_restores_base():
 import mujoco
 xml='<mujoco><worldbody><body><joint name="w" axis="1 0 0" armature=".01" damping=".01"/><inertial pos="0 0 0" mass=".4" diaginertia=".0006 .0004 .0004"/></body></worldbody></mujoco>'
 model=mujoco.MjModel.from_xml_string(xml)
 assert m.wheel_axis_inertia(model,0)==pytest.approx(.0006)
 # Repeated diagnostic indices suffice to exercise one joint's replace/restore semantics.
 refs={'dofs':np.zeros(6,dtype=int),'joints':[0]*6}
 original={k:getattr(model,k).copy() for k in ('dof_armature','dof_damping','dof_frictionloss')}
 cfg,rows=m.actuator_profiles();cfg['leg_armature']=.01
 changed={**cfg,'wheel_J':[.0042,.0042],'wheel_B':[.002,.002],'wheel_C':[.05,.05]}
 r=m.apply_actuator_profile(model,refs,changed,original)
 assert model.dof_armature[0]==pytest.approx(.0036)
 assert r['wheel_total_J_kg_m2']==pytest.approx([.0042,.0042])
 m.apply_actuator_profile(model,refs,cfg,original)
 assert model.dof_armature[0]==pytest.approx(.01) and model.dof_damping[0]==pytest.approx(.01)
 with pytest.raises(ValueError):m.apply_actuator_profile(model,refs,{**cfg,'wheel_J':[.0001,.0001]},original)


def test_high_frequency_summary_resolves_125hz_and_limits_tail_claims():
 rows=[]
 for i in range(1000):
  torque=np.zeros(6);torque[[2,5]]=np.sin(2*np.pi*125*i*.001)
  rows.append({'time':(i+1)*.001,'q':np.zeros(6),'dq':torque,'torque':torque,'contact':np.array([3.,3.]) if i<500 else np.array([3.,0.]),'preclip':torque*10,'limits':np.full(6,5.5)})
 a,r=m.audit_tail(rows)
 assert r['spectra']['wheel_applied_torque']['peak_hz']==pytest.approx(125.)
 assert r['single_lost_fraction']==.5 and r['both_lost_fraction']==0
 assert r['first_observed_in_tail_s']['single_or_dual_contact_loss']==.501
 assert r['sample_rate_hz']==1000 and 'tail only' in r['interpretation']


def test_wheel_match_pairs_isolate_kv_and_preserve_source_candidate_conditions():
 rows=m.wheel_match_cases();assert len(rows)==36
 for group in ('source_candidate','deployed_Kv_only'):
  selected=[r for r in rows if r['group']==group]
  for pair in {r['pair_id'] for r in selected}:
   a,b=[r for r in selected if r['pair_id']==pair]
   assert a['initial']==b['initial']
   differences={k for k in a['configuration'] if a['configuration'][k]!=b['configuration'][k]}
   if group=='deployed_Kv_only':
    assert differences=={'name','kv'} and {a['configuration']['kv'],b['configuration']['kv']}=={1.,.2}
   else:
    assert not differences and {a['policy'],b['policy']}=={'source387','candidate'}
    assert a['configuration']['wheel_J']==[.004204,.0042075] and a['configuration']['wheel_B']==[0.,0.]
 assert {r['configuration']['leg_armature'] for r in rows if r['group']=='source_candidate'}=={.35,.02,.012}
 assert any(r['configuration']['feedback_ms']==20 for r in rows if r['group']=='source_candidate')
 reference=m.wheel_match_cases(reference_only=True)
 assert len(reference)==4
 assert all(r['configuration']['kp']==60 and r['configuration']['leg_armature']==.01 and r['configuration']['actor_ms']==0 for r in reference)
 for initial in ('normal','log'):
  a,b=[r['configuration'] for r in reference if r['initial']==initial]
  assert {k for k in a if a[k]!=b[k]}=={'name','kv'}


def test_wheel_match_screen_separates_incomplete_window_from_good_short_metrics():
 low={'time':np.arange(1,3001)*.01,'state':np.zeros((3000,6)),
  'geometry':np.zeros((3000,3)),'target':np.zeros((3000,6)),'preclip':np.zeros((3000,6))}
 low['state'][:,2]=.34
 summary={'termination':'WINDOW_COMPLETED','simulated_seconds':30.,'both_contact_fraction':1.,
  'saturation_any_fraction':0.,'swing_difference_peak_rad':0.}
 r=m.wheel_match_diagnostics(summary,low)
 assert r['diagnostic_screen_passed'] and r['hold_complete']
 short={k:v[:200] for k,v in low.items()}
 r=m.wheel_match_diagnostics({**summary,'termination':'UPRIGHT','simulated_seconds':2.},short)
 assert not r['diagnostic_screen_passed'] and not r['hold_complete']
 assert 'complete30' in r['failed_checks'] and 'pitch_hold_le_0p1' in r['failed_checks']
 low['geometry'][:,2]=.05
 assert not m.wheel_match_diagnostics({**summary,'swing_difference_peak_rad':.05},low)['diagnostic_screen_passed']


def test_command_screen_preserves_control_and_readback_schedule():
 cases=m.motion_command_cases();assert len(cases)==10
 assert sum(c['initial']=='log' for c in cases)==1
 assert all(c['configuration']['leg_armature']==.012 and c['configuration']['kp']==40 and c['configuration']['kv']==.2 for c in cases)
 c=next(c['configuration'] for c in cases if c['configuration']['command_target']==[0.,-.5])
 np.testing.assert_allclose(m.motion_command(c,0),[0,0,.34],atol=1e-7)
 np.testing.assert_allclose(m.motion_command(c,6),[0,-.25,.34],atol=1e-7)
 np.testing.assert_allclose(m.motion_command(c,10),[0,-.5,.34],atol=1e-7)
 np.testing.assert_allclose(m.motion_command(c,26),[0,0,.34],atol=1e-7)
 with pytest.raises(ValueError):m.motion_command_cases([[1.,.5]])
 with pytest.raises(ValueError):m.motion_command_cases([[float('nan'),0.]])


def test_motion_response_uses_yaw_axis_and_zero_checks_abs_velocity():
 low={'time':np.arange(1,3001)*.01,'policy_time':np.arange(3000)*.01,'state':np.zeros((3000,6)),
  'geometry':np.zeros((3000,3)),'target':np.zeros((3000,6)),'preclip':np.zeros((3000,6)),
  'dq':np.zeros((3000,6)),'velocity':np.tile([.6,0.,-.5],(3000,1)),'command':np.tile([0.,-.5,.34],(3000,1))}
 low['state'][:,2]=.34
 s={'termination':'WINDOW_COMPLETED','simulated_seconds':30.,'both_contact_fraction':1.,
  'saturation_any_fraction':0.,'swing_difference_peak_rad':0.}
 r=m.motion_command_metrics(s,low,[0.,-.5])
 assert r['response_axis']=='yaw_rad_s' and r['tracking_MAE']==0 and r['gain']==1 and r['direction_fraction']==1
 assert r['physical_quality_passed'] and not r['tracking_passed']
 assert r['tracking_failed_checks']==['in_place_translation_le_0p1']
 low['command'][:]=[0.,0.,.34];low['velocity'][:]=0;low['velocity'][:,0]=np.where(np.arange(3000)%2, .4,-.4)
 r=m.motion_command_metrics(s,low,[0.,0.])
 assert r['measured_mean']==pytest.approx(0) and r['vx_hold_abs_mean_mps']==pytest.approx(.4)
 assert not r['tracking_passed'] and 'zero_horizontal_p95_le_0p3' in r['tracking_failed_checks']
