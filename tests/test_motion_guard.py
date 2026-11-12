FROZEN_TEST_SCOPE='PORTABLE'
import copy
import pytest,torch
from native_stand.students.motion_guard import configuration,advance,assess,protocol

def record(command,*,bad=False,missing=False):
 rows=[]
 for i in range(8):
  t={'MAE':.8 if bad else .1,'direction':.7 if bad else 1.,'horizontal_steady_p95_mps':.6 if bad else .05,'pitch_hold_abs_max_rad':.03,'roll_hold_abs_max_rad':.00001}
  if missing:t.pop('MAE')
  rows.append({'env':i,'accepted':not bad,'survival_passed':True,'tracking':t,'checks':{}})
 return {'case':{'command':command},'per_env':rows}

def test_observe_only_records_three_regressions_without_behavior_stop():
 from native_stand.students.motion_guard import evidence_configuration,behavior_mode,apply_behavior_mode,validate_behavior_mode
 c=[.5,0,.34];base=[record(c)];state={}
 for update in (500,750,1000):
  state,decision=advance(state,[record(c,bad=True)],base,update,evidence_configuration())
 assert decision['stop']
 observed=apply_behavior_mode(decision,behavior_mode())
 assert not observed['stop'] and observed['behavior_stop_would_trigger']
 assert observed['decisions']==decision['decisions'] and not observed['decisions'][0]['promotion_allowed']
 assert decision['stop'] and apply_behavior_mode(decision) is decision
 fault=copy.deepcopy(decision);fault['decisions'][0]['gate_evidence']['runtime_fault']='FAILED'
 assert apply_behavior_mode(fault,behavior_mode())['stop']
 assert validate_behavior_mode({'motion_symmetry':{'version':'fixture'},'motion_behavior_guard':behavior_mode()})==behavior_mode()
 with pytest.raises(ValueError):validate_behavior_mode({'motion_behavior_guard':behavior_mode()})
 bad=behavior_mode();bad['runtime_fault_action']='ignore'
 with pytest.raises(ValueError):apply_behavior_mode(decision,bad)

def test_guard_adaptation_then_same_command_three_fixed_checks():
 commands=[[.5,0,.34],[-.5,0,.34]];base=[record(c) for c in commands];bad=[record(c,bad=True) for c in commands]
 state,d=advance({},bad,base,250);assert not d['stop'] and list(state['commands'].values())==[0,0]
 for u in (500,750):state,d=advance(state,bad,base,u);assert not d['stop']
 state,d=advance(state,bad,base,1000);assert d['stop']

def test_different_commands_do_not_accumulate_into_global_failure():
 commands=[[.5,0,.34],[-.5,0,.34]];base=[record(c) for c in commands];state={}
 for u,i in ((500,0),(750,1),(1000,0),(1250,1)):
  records=copy.deepcopy(base);records[i]=record(commands[i],bad=True)
  state,d=advance(state,records,base,u);assert not d['stop']
 assert max(state['commands'].values())==1

def test_unknown_neither_promotes_nor_counts_behavior_failure():
 c=[.5,0,.34];base=[record(c)];state,d=advance({},[record(c,missing=True)],base,500)
 assert d['decisions'][0]['evidence']=='UNKNOWN' and not d['decisions'][0]['promotion_allowed'] and not d['stop']
 state,d=advance(state,[record(c,bad=True)],base,750);assert not d['stop']
 with pytest.raises(ValueError,match='advance'):advance(state,base,base,750)

def test_micro_roll_and_small_tracking_fluctuations_do_not_stop():
 c=[.5,0,.34];base=[record(c)];candidate=record(c)
 for row in candidate['per_env']:row['tracking']['roll_hold_abs_max_rad']=.001;row['tracking']['MAE']=.35;row['accepted']=False
 state={}
 for u in (500,750,1000,1250):state,d=advance(state,[candidate],base,u);assert not d['stop']

def test_complete_absolute_maxima_and_failure_window():
 n=3000;state=torch.zeros(n,39);state[:,29]=1;posture=torch.zeros(n,10);posture[:,0]=.03;posture[:,1]=.001;world=torch.zeros(n,3);controlled=torch.ones(n,dtype=torch.bool)
 result={'accepted':False,'checks':{'survival30':True,'roll_source_guard':False},'tracking':{}}
 original=copy.deepcopy(result);r=assess(result,posture,state,controlled,world)
 assert result==original and r['accepted'] and r['tracking']['pitch_hold_abs_max_rad']==pytest.approx(.03)
 posture[1200,0]=.11;r=assess(result,posture,state,controlled,world);assert not r['accepted'] and not r['checks']['pitch_absolute_quality']
 r=assess(result,posture[:2000],state[:2000],controlled[:2000],world[:2000]);assert not r['accepted'] and not r['checks']['absolute_posture_complete_window']
 assert protocol()['schedule_s']==[2.,8.,8.,8.]

def test_runtime_fault_and_survival_paths():
 c=[0.,0.,.34];base=[record(c)];bad=record(c)
 bad['per_env'][0]['runtime_fault']=True
 _,d=advance({},[bad],base,100);assert d['stop']
 bad=record(c)
 for row in bad['per_env'][:3]:row['survival_passed']=False
 state={}
 for u in (500,750,1000):state,d=advance(state,[bad],base,u)
 assert d['stop']


def test_missing_fixed_checkpoint_breaks_consecutive_regression_streak():
 c=[.5,0,.34];base=[record(c)];bad=[record(c,bad=True)]
 state,_=advance({},bad,base,500)
 state,d=advance(state,bad,base,1000)
 assert max(state['commands'].values())==1 and not d['stop']


def test_v2_known_survival_failure_survives_unknown_hold_metrics_v1_unchanged():
 from native_stand.students.motion_guard import evidence_configuration,EVIDENCE_VERSION
 command=[1.5,0.,.34];base=[record(command)];bad=record(command)
 for row in bad['per_env']:row.update(accepted=False,survival_passed=False,tracking={})
 c=evidence_configuration();state={}
 for u in (500,750,1000):state,d=advance(state,[bad],base,u,c)
 row=d['decisions'][0]
 assert state['version']==EVIDENCE_VERSION and row['consecutive']==3 and d['stop']
 assert row['gate_evidence']['survival']=='FAILED'
 assert row['gate_evidence']['MAE']=='UNKNOWN' and row['evidence']=='PARTIAL'
 assert row['known_behavior_failure'] and not row['promotion_allowed']
 assert row['missing_fields']['MAE']==list(range(8))
 old={}
 for u in (500,750,1000):old,legacy=advance(old,[bad],base,u)
 assert not legacy['stop'] and max(old['commands'].values())==0


@pytest.mark.parametrize('missing',[None,float('nan'),float('inf'),True])
def test_v2_unknown_alone_does_not_become_regression_or_pass(missing):
 from native_stand.students.motion_guard import evidence_configuration
 c=[.5,0.,.34];base=[record(c)];candidate=record(c)
 for row in candidate['per_env']:row['tracking']['MAE']=missing
 state={}
 for u in (500,750,1000):state,d=advance(state,[candidate],base,u,evidence_configuration())
 assert not d['stop'] and list(state['commands'].values())==[0]
 decision=d['decisions'][0]
 assert decision['gate_evidence']['survival']=='PASSED' and decision['gate_evidence']['MAE']=='UNKNOWN'
 assert not decision['known_behavior_failure'] and not decision['promotion_allowed']


def test_v2_known_metric_failure_counts_with_other_unknown_metrics():
 from native_stand.students.motion_guard import evidence_configuration
 c=[.5,0.,.34];base=[record(c)];candidate=record(c)
 candidate['per_env'][0]['tracking']['direction']=.6
 candidate['per_env'][1]['tracking'].pop('direction')
 candidate['per_env'][2]['tracking'].pop('MAE')
 state,d=advance({},[candidate],base,500,evidence_configuration())
 row=d['decisions'][0]
 assert row['gate_evidence']['direction']=='FAILED' and row['gate_evidence']['MAE']=='UNKNOWN'
 assert row['consecutive']==1 and not row['promotion_allowed']


def test_v2_tolerance_adaptation_and_incomplete_identity():
 from native_stand.students.motion_guard import evidence_configuration
 c=[0.,0.,.34];base=[record(c)];candidate=record(c)
 for row in candidate['per_env'][:2]:row.update(accepted=False,survival_passed=False)
 state={}
 for u in (250,500,750,1000):state,d=advance(state,[candidate],base,u,evidence_configuration())
 assert not d['stop'] and d['decisions'][0]['gate_evidence']['survival']=='PASSED'
 # Original tolerance survives: six survivors is not severe source regression.
 for row in candidate['per_env']:row.update(accepted=False,survival_passed=False,tracking={})
 _,d=advance({},[candidate],base,250,evidence_configuration())
 assert d['decisions'][0]['consecutive']==0 and not d['stop']
 candidate['per_env'].pop()
 state,d=advance({},[candidate],base,500,evidence_configuration())
 assert not d['stop'] and d['decisions'][0]['gate_evidence']['identity']=='UNKNOWN'
 assert not d['decisions'][0]['known_behavior_failure']


def test_v2_unknown_gap_resets_but_known_survival_failure_keeps_prior_streak():
 from native_stand.students.motion_guard import evidence_configuration
 c=[2.,0.,.34];base=[record(c)];bad=[record(c,bad=True)];policy=evidence_configuration()
 state,_=advance({},bad,base,500,policy)
 state,_=advance(state,bad,base,750,policy)
 failed=record(c)
 for row in failed['per_env']:row.update(accepted=False,survival_passed=False,tracking={})
 next_state,d=advance(state,[failed],base,1000,policy)
 assert d['stop'] and list(next_state['commands'].values())==[3]
 unknown=record(c)
 for row in unknown['per_env']:row['tracking'].pop('MAE')
 state,d=advance(state,[unknown],base,1000,policy)
 assert not d['stop'] and list(state['commands'].values())==[0]
 state,d=advance(state,bad,base,1500,policy)
 assert not d['stop'] and list(state['commands'].values())==[1]


def test_v2_runtime_fault_immediate_and_new_task_failure_not_old_regression():
 from native_stand.students.motion_guard import evidence_configuration
 c=[0.,.5,.34];base=[record(c,bad=True)];bad=record(c,bad=True);state={}
 for u in (500,750,1000):state,d=advance(state,[bad],base,u,evidence_configuration())
 assert not d['stop'] and not d['decisions'][0]['retained_source']
 bad['per_env'][0]['runtime_fault']=True
 _,d=advance({},[bad],base,1,evidence_configuration())
 assert d['stop'] and 'runtime_fault' in d['decisions'][0]['reasons']


def test_v2_policy_identity_state_and_numeric_thresholds_are_explicit():
 from native_stand.students.motion_guard import evidence_configuration,EVIDENCE_VERSION
 original=configuration();new=evidence_configuration()
 for key in original:
  if key not in ('version','unknown'):assert new[key]==original[key]
 assert configuration()['version']!='motion31_per_gate_retention_v2'
 c=[.5,0.,.34];base=[record(c)]
 old,_=advance({},base,base,500)
 with pytest.raises(ValueError,match='state version'):advance(old,base,base,750,new)
 corrupted=copy.deepcopy(new);corrupted['survival_stop_below_envs']=8
 with pytest.raises(ValueError,match='policy'):advance({},base,base,500,corrupted)
 state,d=advance({},base,base,500,new)
 assert state['version']==EVIDENCE_VERSION and d['decisions'][0]['promotion_allowed']
 with pytest.raises(ValueError,match='advance'):advance(state,base,base,500,new)


def test_historical_v1_function_body_is_unchanged():
 import ast,hashlib
 from pathlib import Path
 import native_stand.students.motion_guard as module
 tree=ast.parse(Path(module.__file__).read_text())
 function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_advance_v1')
 function.name='advance_v1'
 assert hashlib.sha256(ast.dump(function,include_attributes=False).encode()).hexdigest()=='b9dc884695f648457966d6cbc498b0b8cd5fc5ba725a0cae619c368d0411edfa'


def test_v2_conflicting_acceptance_cannot_promote_and_missing_survival_is_unknown():
 from native_stand.students.motion_guard import evidence_configuration
 c=[.5,0.,.34];base=[record(c)];candidate=record(c)
 for row in candidate['per_env']:row['survival_passed']=False
 _,d=advance({},[candidate],base,500,evidence_configuration())
 assert d['decisions'][0]['known_behavior_failure'] and not d['decisions'][0]['promotion_allowed']
 for row in candidate['per_env']:row.pop('survival_passed')
 _,d=advance({},[candidate],base,500,evidence_configuration())
 assert d['decisions'][0]['gate_evidence']['survival']=='UNKNOWN' and not d['decisions'][0]['known_behavior_failure']
