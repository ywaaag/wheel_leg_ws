"""Portable CPU new-version integration; no AppLauncher, environment or live PPO."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from native_stand.students import motion_parallel_course as c
from native_stand.students import motion_parallel_design as d
from native_stand.students.motion_tracking_reward import configuration,quality_base_configuration,reward_fields


def test_explicit_runtime_recipe_and_reward_schema():
 from native_stand.students.motion_training import declaration
 from native_stand.students.training_recipe import validate_instance
 from native_stand.students.motion_symmetry import network
 from native_stand.students.motion_startup import smoke_profile
 from native_stand.students.motion_course import implementation
 r=declaration('a'*64,0,reward_version=c.REWARD);r.update(course_version=c.VERSION,course_stage=0,parallel_state=c.initial_state(),network_architecture=network(),smoke_wiring=smoke_profile())
 assert validate_instance(r,'a'*64)==r and implementation(c.VERSION) is c
 assert r['critic_privileged_version']=='motion_symmetry_privileged85_v1'
 assert len(reward_fields(r['reward']))==len(reward_fields(configuration('student_motion31_differential_quality_v9')))
 assert quality_base_configuration(c.REWARD)['version']=='student_motion31_large_gated_refinement_v6'
 bad=copy.deepcopy(r);bad['reward']['weights'][0]=5
 with pytest.raises(ValueError):validate_instance(bad,'a'*64)


def event(state,update,cp='b'*64):
 from native_stand.students.motion_parallel_assessment import protocol,REQUIRED
 p=protocol({'version':'motion31_differential85_assessment_v1'})
 records=[{'case':{'command':cmd+[.34],'sha256':cp},'protocol':p,'path':str(i),'sha256':'f'*64,'per_env':[{'env':j,'controlled_seconds':30.,'accepted':True,'checks':{k:True for k in REQUIRED},'tracking':{'MAE':.1}} for j in range(8)]} for i,cmd in enumerate(c.commands(state['stage'],state))]
 return {'stage':state['stage'],'update':update,'sha256':cp,'checkpoint':'/cpu/'+cp+'.pt','source_sha256':'a'*64,'complete_matrix':True,'evaluations':records,'per_env':[e for r in records for e in r['per_env']]}


def test_independent_promotion_keeps_other_new_failures_local():
 state=c.initial_state();a=event(state,250)
 for r in a['evaluations']:
  if r['case']['command'][0]<0:
   for e in r['per_env']:e['accepted']=False
 state,decision=c.advance(state,a);assert not decision['advanced']
 a['update']=500;a['sha256']='c'*64
 for r in a['evaluations']:r['case']['sha256']='c'*64
 state,decision=c.advance(state,a)
 assert decision['advanced'] and state['lanes']['forward']['level']==state['lanes']['yaw']['level']==1 and state['lanes']['backward']['level']==0
 rows=c.stage_config(state['stage'],4096,state)
 assert sum(rows['slot_counts'])==4096 and [3.,0.] in rows['commands'] and [-1.5,0.] not in rows['commands']
 assert [2.5,0.] in rows['commands'] and [0.,1.] in rows['commands']


def test_no_source_smoke_or_duplicate_evidence_unlock():
 state=c.initial_state();a=event(state,250);a['stage']=1
 with pytest.raises(ValueError):c.advance(state,a)
 a=event(state,250);a['evaluations'][0]['protocol']['version']='wrong'
 with pytest.raises(ValueError):c.advance(state,a)


def test_retention_follows_command_and_env_not_position():
 old=event(c.initial_state(),250)['evaluations'];new=list(reversed(copy.deepcopy(old)))
 assert not c.retained_regression(old,new)
 new[0]['per_env'][0]['accepted']=False;assert c.retained_regression(old,new)


def test_mixture_refinement_rolls_back_and_freezes_unknown():
 state=[{'target':[0.,1.],'mix':0.,'consecutive_eligible':0}]
 records=[{'case':{'command':[0.,1.,.34]},'per_env':[{'env':i,'accepted':True} for i in range(8)]}]
 state,_=c.update_mixture(state,records,{});state,_=c.update_mixture(state,records,{});assert state[0]['mix']==.25
 unknown=copy.deepcopy(records);unknown[0]['per_env']=[];assert c.update_mixture(state,unknown,{})[0]==state
 for e in records[0]['per_env']:e['accepted']=False
 state,_=c.update_mixture(state,records,{});state,_=c.update_mixture(state,records,{});assert state[0]['mix']==0


def test_applied_tracking_preserves_geometry_events_and_dt():
 fields=reward_fields(configuration('student_motion31_differential_quality_v9'));n=3
 density=torch.ones(n,len(fields),dtype=torch.float64);raw=density.clone();result={'fields':fields,'density':density,'raw':raw,'dt':density*.01,'total':density.sum(-1)*.01-1.,'eligible':torch.ones(n,dtype=torch.bool),'events':torch.ones(n,3)}
 env=SimpleNamespace(motion_action_command=torch.tensor([[2.5,0.,.34],[-1.,0.,.34],[0.,1.,.34]],dtype=torch.float64),
 target_xy=torch.tensor([[2.5,0.],[-1.,0.],[0.,1.]],dtype=torch.float64),vx=torch.tensor([0.,0.,0.],dtype=torch.float64),vy=torch.zeros(n,dtype=torch.float64),yaw=torch.zeros(n,dtype=torch.float64),motion_cohort_mix=torch.zeros(n,dtype=torch.float64),command_index=torch.arange(n))
 out=c.apply_tracking(result,env)
 assert out['version']==c.REWARD and torch.allclose(out['total'],out['density'].sum(-1)*.01-1.)
 for name in ('spin_differential_target','leg_swing_symmetry','pitch_excess','roll_excess'):
  assert torch.equal(out['density'][:,fields.index(name)],density[:,fields.index(name)])
 assert torch.equal(result['density'],density)


def test_ramp_zero_keeps_episode_target_semantics():
 from native_stand.students.motion_tracking_reward import reward,configuration,COMBO_VERSION
 z=torch.zeros(1,dtype=torch.float64);v=torch.tensor([[.1,0.,0.]],dtype=torch.float64);target=torch.tensor([[1.,0.]],dtype=torch.float64)
 a=d.signals(torch.zeros(1,2,dtype=torch.float64),v[:,:2],z,new_task=torch.zeros(1,dtype=torch.bool),mix=z,episode_target=target)
 assert a['linear_denominator'].item()==1 and a['fields']['linear_error'].item()==pytest.approx(-.005)


def test_real_source_plan_zero_write_and_frozen_contract(tmp_path):
 from native_stand.students.motion_parallel_experiment import build,verify
 from native_stand.students.requests import validate
 root=Path(__file__).resolve().parents[1];parent=root/'configs/experiments/2026_10_18-23_10__motion1000_differential85_5000_retry/spec.json';cp=root/'runs/2026_10_18-23_10__motion1000_differential85_5000_retry/formal/model_update_3250.pt'
 if not cp.exists():pytest.skip('local u3250 unavailable')
 spec=build(parent,cp,experiment='2026_10_19-18_30__parallel_fixture',spec_path=tmp_path/'spec.json')
 assert verify(spec) and validate(spec)==[] and not (tmp_path/'spec.json').exists()
 assert spec['authorization']['status']=='UNAUTHORIZED' and not spec['resources']['sharing_authorized']
 old=json.loads(parent.read_text())
 for key in ('action','control','ground','ppo','observation','history','network_architecture'):assert spec[key]==old[key]
 assert len(spec['student_evaluation']['cases'])==12
 from native_stand.students.motion_quick_evaluation import validate as quick
 assert quick(spec)
 bad=copy.deepcopy(spec);bad['control']['policy_dt_s']=.008
 with pytest.raises(ValueError):verify(bad)

def test_checkpoint_recipe_json_roundtrip_and_optimizer_restore(tmp_path,monkeypatch):
 from native_stand.students.model import Student,load
 from native_stand.students.motion_symmetry import network
 from native_stand.students.motion_training import declaration
 from native_stand.students.ppo import assemble
 from native_stand.students.motion_course_requests import restore
 from native_stand.recovery.controlled_ppo import KEY
 from tensordict import TensorDict
 a=Student(True,version='multiskill_mcu_v2',network_architecture=network());critic=Student(False,version='multiskill_mcu_v2',critic_privileged_version='motion_symmetry_privileged85_v1',network_architecture=network())
 source={'interface_version':'multiskill_mcu_v2','role':'MotionZeroStudent','actor':a.state_dict(),'critic':critic.state_dict(),'network_architecture':network(),'training_recipe':{'network_architecture':network()},'critic_privileged_version':'motion_symmetry_privileged85_v1','optimizer':torch.optim.Adam([*a.parameters(),*critic.parameters()],lr=3e-5).state_dict(),'training_rng':{},'reward_refinement_state':[{'target':[0.,0.],'mix':.5,'consecutive_eligible':0}]}
 path=tmp_path/'source.pt';torch.save(source,path)
 cfg=declaration('a'*64,reward_version=c.REWARD);cfg.update(course_version=c.VERSION,course_stage=0,parallel_state=c.initial_state(),network_architecture=network())
 obs=TensorDict({'student_actor':torch.zeros(8,31),'student_critic':torch.zeros(8,85),KEY:torch.ones(8,1,dtype=torch.bool)},batch_size=[8])
 p=assemble(path,obs,steps_per_env=4,recipe=cfg);monkeypatch.setattr('native_stand.students.resume.restore_rng',lambda _:None)
 restore(p,source,{'network_architecture':network(),'student_training':cfg},'CPU_FIXTURE')
 assert p.reward_refinement_state[0]['mix']==.5 and len(p.reward_refinement_state)==12
 assert all(torch.equal(v,p.actor.state_dict()[k]) for k,v in source['actor'].items())
 assert json.loads(json.dumps(c.configuration()))==c.configuration()
 # No PPO optimizer step: CPU preparation authorization does not authorize training.
 assert not p.optimizer.state


def test_dynamic_twelve_cohort_critic_diagnostics():
 from native_stand.students.motion_large_combo import value_diagnostics
 from native_stand.recovery.controlled_ppo import KEY
 storage=SimpleNamespace(observations={KEY:torch.ones(1,12,1,dtype=torch.bool),'motion_diagnostic_cohort':torch.arange(12).reshape(1,12,1)},returns=torch.arange(12.).reshape(1,12,1),values=torch.zeros(1,12,1),advantages=torch.ones(1,12,1))
 out=value_diagnostics(storage,commands_override=c.commands());assert len(out)==12 and out[-1]['command']==[0.,-1.]

def test_full_all_lanes_promotion_reaches_final_without_dropping_endpoints():
 state=c.initial_state();update=0
 for _ in range(10):
  update+=250;a=event(state,update,cp=f'{update:064x}');state,decision=c.advance(state,a)
  assert sum(c.stage_config(state['stage'],4096,state)['slot_counts'])==4096
  if decision['all_stages_passed']:break
 assert decision['all_stages_passed'] and [4.,0.] in c.commands(state['stage'],state) and [-4.,0.] in c.commands(state['stage'],state)
 assert [0.,4.] in c.commands(state['stage'],state) and [0.,-4.] in c.commands(state['stage'],state)
 assert len(c.commands(state['stage'],state))==27


def test_source_incomplete_new_case_is_not_a_capability_prerequisite(tmp_path):
 import numpy as np
 from native_stand.students.motion_spin_calibration import verify_records
 from test_motion_yaw_quality import kinematics,measurement
 center,axis,angular,velocity=kinematics((1.,1.));forward=center.new_tensor([[1.,0.,0.]])
 measured=measurement(center,center,velocity,angular,axis,forward)
 def save(name,command,steps):
  fields={'center_world':center,'com_world':center,'linear_com_world':velocity,'angular_world':angular,'axis_world':axis,'forward_world':forward,**measured}
  data={'spin_'+k:np.repeat(x.numpy()[None,:],steps,axis=0) for k,x in fields.items() if k in ('center_world','com_world','linear_com_world','angular_world','axis_world','forward_world','tangent_velocity_w','rolling_speed_forward_mps','geometry_valid')}
  data.update(first_episode_valid=np.ones((steps,1),bool),controller=np.full((steps,1,1),3))
  np.savez(tmp_path/(name+'.npz'),**data);(tmp_path/(name+'.json')).write_text(json.dumps({'trace':{'path':name+'.npz'},'case':{'command':command}}));return {'path':name+'.json'}
 records=[save('forward',[.5,0.,.34],1200),save('reverse',[-1.,0.,.34],100)]
 assert verify_records(records,tmp_path,allow_incomplete_new=True)['records'][1]['holding_behavior'].startswith('NOT_OBSERVED')
 (tmp_path/'spin_measurement_readback.json').unlink()
 with pytest.raises(ValueError,match='holding'):verify_records(records,tmp_path)

def test_behavior_never_stops_parallel_chain_even_when_all_stages_pass():
 for regression in (False,True):
  for passed in (False,True):assert not c.may_stop_for_behavior(regression=regression,all_stages_passed=passed)
 assert c.configuration()['stop_policy']['source_skill_failure']=='record_only'
 assert c.configuration()['stop_policy']['budget_storage_runtime_fault']=='stop'

def test_acceptance_needs_two_distinct_final_matrices_and_does_not_stop():
 state=c.initial_state()
 for lane in d.LEVELS:state['lanes'][lane]['level']=len(d.LEVELS[lane])-1
 first=event(state,1000,'a'*64);second=event(state,1250,'b'*64)
 source={'update':0,'stage':0,'checkpoint':'/source','per_env':[]}
 assert len(first['evaluations'])==27
 assert c.choose([source,first],{})['accepted_candidate'] is None
 assert c.choose([source,first,second],{})['accepted_candidate']['update']==1250
 assert not c.may_stop_for_behavior(all_stages_passed=True)

def test_every100_lr_diagnostics_bind_checkpoint_and_keep_missing_null():
 metric={'update':100,'actual_LR':[3e-5],'cohort_statistics':{'hold':{'cohort0':{'samples':0,'mean':None}}},'cohort_value_diagnostics':[{'command':[0.,1.],'explained_variance':None}],
 'actor_update_scope':{'kl_valid_mean':.01,'clip_fraction_valid_mean':.1,'source_policy_retention':{'loss_mean':.02,'weighted_loss_mean':.0002,'channel_KL_mean':[.01]*6}}}
 cp={'update':100,'checkpoint':'/cpu/model_update_100.pt','sha256':'a'*64}
 out=c.checkpoint_diagnostics(metric,cp)
 assert out['PPO_KL']==.01 and out['clip_fraction']==.1 and out['hold_tracking']['cohort0']['mean'] is None
 assert out['cohort_critic_EV'][0]['explained_variance'] is None and out['optimizer_or_simulation_added']==0
 assert out['source_retention_KL']['weighted_loss_mean']==.0002
 with pytest.raises(ValueError):c.checkpoint_diagnostics({**metric,'update':99},cp)
 with pytest.raises(ValueError):c.checkpoint_diagnostics(metric,{**cp,'update':200})
 with pytest.raises(ValueError):c.checkpoint_diagnostics({**metric,'actual_LR':[float('nan')]},cp)

def test_every100_diagnostics_use_cumulative_boundary_and_local_checkpoint_identity():
 metric={'update':54,'parent_formal_offset':46,'formal_update_including_parent':100,'actual_LR':[3e-5]}
 cp={'update':54,'checkpoint':'/cpu/model_update_54.pt','sha256':'a'*64}
 report=c.checkpoint_diagnostics(metric,cp)
 assert report['update']==100 and report['local_update']==54 and report['parent_formal_offset']==46
 with pytest.raises(ValueError):c.checkpoint_diagnostics({**metric,'formal_update_including_parent':200},cp)
 with pytest.raises(ValueError):c.checkpoint_diagnostics({**metric,'parent_formal_offset':-46},cp)
 with pytest.raises(ValueError):c.checkpoint_diagnostics(metric,{**cp,'update':100})
 second=c.checkpoint_diagnostics({'update':100,'parent_formal_offset':100,'formal_update_including_parent':200,'actual_LR':[3e-5]},
  {'update':100,'checkpoint':'/cpu/resume/model_update_100.pt','sha256':'b'*64})
 assert second['update']==200 and second['local_update']==100


def test_public_health_spin_validator_routes_explicit_parallel_version(monkeypatch):
 from native_stand.students import motion_spin_experiment,motion_parallel_experiment
 marker={'motion_parallel':{'version':c.VERSION}}
 monkeypatch.setattr(motion_parallel_experiment,'verify',lambda spec:spec is marker)
 assert motion_spin_experiment.verify(marker)
