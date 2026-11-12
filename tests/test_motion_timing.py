FROZEN_TEST_SCOPE='PORTABLE'
from types import SimpleNamespace as S
import torch
from native_stand.students.motion_timing import configuration,network,contract,advance
from native_stand.students.motion_timing_runtime import Timing

def fake_env(n=8):
 data=S(joint_pos=S(torch=torch.zeros(n,6)),joint_vel=S(torch=torch.zeros(n,6)))
 return S(num_envs=n,device='cpu',curriculum_training=True,ids=list(range(6)),
  robot=S(data=data,set_joint_effort_target_index=lambda **kw:None),executed_leg=torch.zeros(n,4),
  executed_wheel=torch.zeros(n,2),effort=torch.zeros(n,6),effort_pre=torch.zeros(n,6),pd_counter=0)

def test_nominal_timing_is_exact_shared_pd_and_no_rng_change():
 e=fake_env();rng=torch.get_rng_state().clone();t=Timing(e,configuration(),0)
 assert torch.equal(rng,torch.get_rng_state())
 t.requested[:,[0,1,3,4]]=.1;t.requested[:,[2,5]]=2.;t.apply()
 assert torch.equal(e.effort,torch.tensor([6.,6.,2.,6.,6.,2.]).expand(8,-1))
 assert torch.equal(t.adopted,t.requested);assert t.features().shape==(8,25)
 assert torch.equal(rng,torch.get_rng_state())

def test_reset_parameters_fixed_and_wheel_sides_not_double_counted():
 e=fake_env(32);t=Timing(e,configuration(),3);t.configure(3);t.reset(torch.arange(32))
 assert ((t.delays[:,2]==0)|(t.delays[:,2]==7)).all()
 assert ((t.delays[:,5]==0)|(t.delays[:,5]==11)).all()
 assert ((t.delays[:,2]==7)^(t.tau[:,2]==7)).all()
 assert ((t.delays[:,5]==11)^(t.tau[:,5]==11)).all()
 before=t.delays.clone()
 for _ in range(25):t.apply()
 assert torch.equal(before,t.delays);assert torch.isfinite(e.effort).all()
 assert (e.effort.abs()<=torch.tensor([40,40,5.5,40,40,5.5])).all()

def test_delayed_target_not_adopted_early_and_reset_flushes_history():
 e=fake_env();t=Timing(e,configuration(),0);t.delays[:]=10;t.requested[:]=1
 for _ in range(10):t.apply()
 assert (t.adopted==0).all()
 t.apply();assert (t.adopted==1).all()
 t.reset(torch.arange(8));assert torch.equal(t.targets[0],t.adopted)

def test_critic_contract_and_unknown_does_not_promote():
 assert contract()['actor_dim']==31 and contract()['critic_dim']==110 and network()['critic_input']==110
 s={'stage':0,'stage_started_update':0,'consecutive_passes':0,'accepted_by_stage':{}}
 s,d=advance(s,{'evaluations':[],'update':250,'stage':0});assert s['stage']==0 and not d['advanced']
 candidate={'evaluations':timing_records(), 'stage':0,'complete_matrix':True,
 'update':500,'checkpoint':'cpu-fixture','sha256':'a'*64}
 s,d=advance(s,candidate);assert not d['advanced']
 candidate['update']=750;s,d=advance(s,candidate);assert d['advanced'] and s['stage']==1


def test_timing_rollout_metadata_reaches_value_diagnostics_without_retention():
 from native_stand.students.motion_runner import motion_observations
 from native_stand.students.motion_large_combo import value_diagnostics
 from native_stand.recovery.controlled_ppo import KEY
 e=S(num_envs=3,target_xy=torch.zeros(3,2),command_index=torch.zeros(3,dtype=torch.long),
     motion_config={'course_version':'motion31_standing_timing_course_v1'},
     actor_valid_pre=lambda:torch.tensor([True,False,True]))
 raw={'student_actor':torch.zeros(3,31),'student_critic':torch.zeros(3,110)}
 obs=motion_observations(e,raw)
 assert 'motion_diagnostic_cohort' in obs and 'motion_diagnostic_cohort' not in raw
 assert obs['student_actor'].shape==(3,31) and obs['student_critic'].shape==(3,110)
 stored=torch.stack([obs,obs],0)
 returns=torch.tensor([1.,2.,3.,4.,5.,6.]).reshape(2,3,1)
 storage=S(observations=stored,returns=returns,values=torch.zeros_like(returns),advantages=torch.ones_like(returns))
 rows=value_diagnostics(storage,commands_override=[[0.,0.,.34]])
 assert len(rows)==1 and rows[0]['samples']==4 and rows[0]['explained_variance']==0.
 assert torch.equal(obs[KEY].flatten(),torch.tensor([True,False,True]))



def timing_records(stage=0):
 from native_stand.students.motion_timing import cases
 records=cases({'student_evaluation':{'cases':[{}]}},stage)
 return [{'case':r,'per_env':[{'env':i,'accepted':True,'survival_passed':True,
  'checks':{},'tracking':{'MAE':0.},'symmetry_metrics':{'swing_hold_peak_rad':.01,
  'horizontal_hold_mean_mps':.01,'common_abs_hold_mean_mps':.01}} for i in range(8)]} for r in records]


def test_timing_three_conditions_advance_mixture_once_not_three_times():
 from native_stand.students.motion_timing import refinement_records
 from native_stand.students.motion_symmetry import update_mix
 state=[{'target':[0.,0.],'mix':0.,'confirmations':0,'regressions':0}]
 records=timing_records()
 first=update_mix(state,refinement_records(records,0))
 assert first[0]['confirmations']==1 and first[0]['mix']==0.
 second=update_mix(first,refinement_records(records,0))
 assert second[0]['mix']==.25 and state[0]['mix']==0.
 records[2]['per_env'][4]['accepted']=False
 assert update_mix(state,refinement_records(records,0))[0]['confirmations']==0
 for broken in [records[:2],records+[records[0]],[records[0]]*3]:
  assert refinement_records(broken,0)[0]['per_env']==[]


def test_timing_cannot_accept_source_or_reuse_wrong_stage_or_duplicate_update():
 import pytest
 from native_stand.students.motion_timing import advance,choose
 records=timing_records()
 state={'stage':0,'stage_started_update':0,'last_evaluation_update':-1,'consecutive_passes':0,'accepted_by_stage':{}}
 c={'evaluations':records,'per_env':[e for r in records for e in r['per_env']],
    'stage':0,'update':250,'checkpoint':'cpu','sha256':'a'*64,'complete_matrix':True}
 state,d=advance(state,c);assert state['consecutive_passes']==1
 with pytest.raises(ValueError):advance(state,c)
 assert choose([{**c,'update':0},c],{})['accepted_candidate'] is None
 wrong=timing_records();wrong[2]['case']['_timing_level']=3
 state,d=advance(state,{**c,'update':500,'evaluations':wrong})
 assert state['consecutive_passes']==0 and not d['advanced']


def test_timing_posteval_recovery_replays_bookkeeping_without_optimizer(tmp_path):
 import torch
 from native_stand.students.motion_timing_experiment import restored_posteval_state
 p=tmp_path/'checkpoint.pt';p.write_bytes(b'fixture')
 source={'update':250,'motion_course_state':{'stage':0,'stage_started_update':0,'last_evaluation_update':-1,
 'consecutive_passes':0,'accepted_by_stage':{}},'symmetry_mix_state':[{'target':[0.,0.],'mix':0.,'confirmations':0,'regressions':0}],
 'reward_refinement_state':[{'target':[0.,0.],'mix':0.,'consecutive_eligible':0}]}
 state,mix,refinement,decision=restored_posteval_state(source,{'pending_evaluations':timing_records(),
 'checkpoint':str(p)}, {})
 assert state['consecutive_passes']==1 and state['last_evaluation_update']==0 and state['stage_started_update']==-250
 assert mix[0]['confirmations']==1 and refinement==source['reward_refinement_state']
 assert source['motion_course_state']['consecutive_passes']==0


def test_resume_preserves_absolute_checkpoint_cadence():
 from native_stand.students.motion_runner import checkpoint_plan
 recipe={'milestones':[250,500],'checkpoint_every_updates':100}
 assert checkpoint_plan(recipe,'formal',50,9745,offset=250)==(True,False)
 assert checkpoint_plan(recipe,'formal',100,9745,offset=250)==(False,False)
 assert checkpoint_plan(recipe,'formal',250,9745,offset=250)==(True,True)
 assert checkpoint_plan(recipe,'formal',9745,9745,offset=250)==(True,True)
