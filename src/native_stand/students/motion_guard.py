"""Explicit-data Motion assessment and tolerant per-command retention policy."""
import copy,math
VERSION='motion31_absolute_posture_retention_v1'
EVIDENCE_VERSION='motion31_per_gate_retention_v2'
ASSESSMENT='motion31_absolute_posture_assessment_v4'
BEHAVIOR_MODE='motion31_behavior_regression_observe_only_v1'

def behavior_mode():
 return {'version':BEHAVIOR_MODE,'regression_action':'record_only','acceptance_thresholds':'unchanged','runtime_fault_action':'stop','budget_action':'stop'}

def validate_behavior_mode(spec):
 mode=spec.get('motion_behavior_guard')
 if mode is not None and (mode!=behavior_mode() or not spec.get('motion_symmetry')):
  raise ValueError('observe-only behavior guard requires explicit Motion symmetry declaration; safety/budgets unchanged')
 return mode

def apply_behavior_mode(decision,mode=None):
 """Record regression without stopping; never suppress physical/runtime faults."""
 if mode is None:return decision
 if mode!=behavior_mode():raise ValueError('unknown Motion behavior guard mode')
 out=copy.deepcopy(decision)
 fault=any('runtime_fault' in d.get('reasons',[]) or d.get('gate_evidence',{}).get('runtime_fault')=='FAILED' for d in out.get('decisions',[]))
 out.update(behavior_guard_mode=mode,behavior_stop_would_trigger=bool(decision['stop']),stop=fault)
 return out
def configuration():
 return {'version':VERSION,'adaptation_updates':300,'evaluation_every_updates':250,'consecutive_failures':3,
  'pitch_hold_abs_max_rad':.1,'roll_hold_abs_max_rad':.05,'tracking_MAE_stop_mps':.5,'tracking_MAE_increase_stop_mps':.2,
  'direction_stop_below':.8,'zero_horizontal_p95_stop_mps':.5,'survival_stop_below_envs':6,'expected_envs':8,
  'relative_posture':'diagnostic only','unknown':'blocks promotion; does not count as behavior regression',
  'runtime_fault':'stop immediately; individual ordinary episode failures remain learning samples'}
def protocol():
 from .motion_training import protocol as base
 p=base(quality=True);p.update(version=ASSESSMENT,absolute_posture=configuration(),posture_note='absolute hold maxima pitch<=.1 roll<=.05; relative deltas diagnostic only')
 return p

def assess(result,posture,state,controlled,world_velocity):
 """Same full30s / hold10..18s / zero20..30s, no shortened failure window."""
 p=protocol();r=copy.deepcopy(result);mask=controlled.bool()&state[:,29].bool();q=posture[mask];w=world_velocity[mask]
 r['legacy_checks']=dict(r['checks']);r['legacy_accepted']=r['accepted'];r['version']=ASSESSMENT
 for key in ('pitch_source_guard','roll_source_guard'):r['checks'].pop(key,None)
 complete=len(q)>=3000
 r['checks']['absolute_posture_complete_window']=complete
 if complete:
  q=q[1000:1800];pitch=float(q[:,0].abs().max());roll=float(q[:,1].abs().max())
  r['tracking'].update(pitch_hold_abs_max_rad=pitch,roll_hold_abs_max_rad=roll)
  r['tracking']['horizontal_steady_p95_mps']=float(__import__('torch').quantile(w[2000:3000,:2].norm(dim=-1),.95))
  r['checks'].update(pitch_absolute_quality=pitch<=.1,roll_absolute_quality=roll<=.05)
 else:r['checks'].update(pitch_absolute_quality=False,roll_absolute_quality=False)
 r.update(accepted=all(r['checks'].values()),failed_checks=[k for k,v in r['checks'].items() if not v],relative_posture_scope='diagnostic only')
 return r

def _advance_v1(state,records,source_records,update,policy=None):
 """Never mutate evidence; missing evidence cannot pass or mimic regression."""
 c=configuration() if policy is None else policy
 if c!=configuration():raise ValueError('unknown retention policy')
 if type(update) is not int or update<0 or len(records)!=len(source_records):raise ValueError('guard update/matrix differs')
 if state and update<=state.get('last_update',-1):raise ValueError('guard update must advance')
 if state and set(state)-{'last_update','commands','version'}:raise ValueError('guard state fields differ')
 out={'version':VERSION,'last_update':update,'commands':{}};decisions=[];stop=False
 for candidate,source in zip(records,source_records):
  command=candidate.get('case',{}).get('command');original=source.get('case',{}).get('command')
  if not command or command!=original:raise ValueError('retention command identity differs')
  key=','.join(map(str,command));rows=candidate.get('per_env',[]);baseline=source.get('per_env',[]);reasons=[];known=len(rows)==len(baseline)==8
  if known:
   ids=[r.get('env',r.get('env_id')) for r in rows];bids=[r.get('env',r.get('env_id')) for r in baseline]
   known=ids==bids and len(set(ids))==8
  if known and (any(type(r.get('survival_passed')) is not bool or type(r.get('accepted')) is not bool for r in rows+baseline)):known=False
  if known and any(r.get('checks',{}).get('runtime_fault') is True or r.get('runtime_fault') is True for r in rows):
   stop=True;reasons.append('runtime_fault')
  retained=known and all(r.get('accepted') is True for r in baseline)
  if known and retained:
   if sum(r.get('survival_passed') is True for r in rows)<6:reasons.append('survival')
   for r,b in zip(rows,baseline):
    t=r.get('tracking',{});bt=b.get('tracking',{})
    fields=['MAE','direction'] if command[0] or command[1] else ['horizontal_steady_p95_mps']
    for f in fields:
     value=t.get(f);old=bt.get(f)
     if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in (value,old)):known=False;continue
     if f=='MAE' and value>.5 and value-old>=.2:reasons.append('tracking')
     if f=='direction' and value<.8:reasons.append('direction')
     if f=='horizontal_steady_p95_mps' and value>.5:reasons.append('zero_drift')
    for f,limit in (('pitch_hold_abs_max_rad',.1),('roll_hold_abs_max_rad',.05)):
     v=t.get(f)
     if not isinstance(v,(int,float)) or not math.isfinite(v):known=False
     elif v>limit:reasons.append(f)
  # Only fixed scheduled checks after adaptation may accumulate failures.
  eligible=update>=c['adaptation_updates'] and update%c['evaluation_every_updates']==0
  old=(state or {}).get('commands',{}).get(key,0)
  if eligible and state and state['last_update']!=update-c['evaluation_every_updates']:old=0
  count=old
  if eligible:count=old+1 if known and retained and reasons else 0
  out['commands'][key]=count
  stop|=bool(eligible and count>=3)
  decisions.append({'command':command,'evidence':'KNOWN' if known else 'UNKNOWN','retained_source':retained,'adaptation':update<300,'scheduled_check':eligible,'reasons':sorted(set(reasons)),'consecutive':count,'promotion_allowed':known and all(r.get('accepted') is True for r in rows)})
 return out,{'stop':stop,'decisions':decisions,'policy':c}


def evidence_configuration():
 """New early-stop semantics only; numerical thresholds/scoring stay unchanged."""
 c=configuration()
 c.update(version=EVIDENCE_VERSION,
  unknown='per gate; known failures count despite unrelated unknown metrics; unknown never promotes',
  evidence='independent survival/tracking/posture/acceptance states')
 return c


def _advance_v2(state,records,source_records,update,policy):
 c=policy
 if c!=evidence_configuration():raise ValueError('unknown per-gate retention policy')
 if type(update) is not int or update<0 or len(records)!=len(source_records):raise ValueError('guard update/matrix differs')
 if state and (set(state)!={'last_update','commands','version'} or state.get('version')!=EVIDENCE_VERSION):raise ValueError('per-gate guard state version/fields differ')
 if state and update<=state.get('last_update',-1):raise ValueError('guard update must advance')
 out={'version':EVIDENCE_VERSION,'last_update':update,'commands':{}};decisions=[];stop=False
 scheduled=update>=c['adaptation_updates'] and update%c['evaluation_every_updates']==0
 def finite(x):return type(x) in (int,float) and math.isfinite(x)
 def aggregate(flags):
  return 'FAILED' if False in flags else 'UNKNOWN' if None in flags or not flags else 'PASSED'
 for candidate,source in zip(records,source_records):
  command=candidate.get('case',{}).get('command')
  if not command or command!=source.get('case',{}).get('command'):raise ValueError('retention command identity differs')
  key=','.join(map(str,command));rows=candidate.get('per_env',[]);baseline=source.get('per_env',[])
  ids=[r.get('env',r.get('env_id')) for r in rows];bids=[r.get('env',r.get('env_id')) for r in baseline]
  identity=len(rows)==len(baseline)==c['expected_envs'] and ids==bids and set(ids)==set(range(c['expected_envs']))
  gates={'identity':'PASSED' if identity else 'UNKNOWN'};missing={};reasons=[]
  fault=any(r.get('runtime_fault') is True or r.get('checks',{}).get('runtime_fault') is True for r in rows)
  if fault:stop=True;reasons.append('runtime_fault')
  gates['runtime_fault']='FAILED' if fault else 'NOT_OBSERVED'
  source_flags=[r.get('accepted') if type(r.get('accepted')) is bool else None for r in baseline]
  retained=identity and len(source_flags)==8 and aggregate(source_flags)=='PASSED' and all(r.get('survival_passed') is True for r in baseline)
  gates['source_qualified']='PASSED' if retained else 'NOT_APPLICABLE' if identity and False in source_flags else 'UNKNOWN'
  if identity:
   flags=[r.get('accepted') if type(r.get('accepted')) is bool else None for r in rows]
   gates['acceptance']=aggregate(flags)
   if None in flags:missing['acceptance']=[i for i,x in zip(ids,flags) if x is None]
   flags=[r.get('survival_passed') if type(r.get('survival_passed')) is bool else None for r in rows]
   # Incomplete booleans are UNKNOWN, never a fabricated failed episode.
   gates['survival']='UNKNOWN' if None in flags else 'FAILED' if sum(flags)<c['survival_stop_below_envs'] else 'PASSED'
   if None in flags:missing['survival']=[i for i,x in zip(ids,flags) if x is None]
   if retained:
    if gates['survival']=='FAILED':reasons.append('survival')
    fields=['MAE','direction'] if command[0] or command[1] else ['horizontal_steady_p95_mps']
    fields+=['pitch_hold_abs_max_rad','roll_hold_abs_max_rad']
    for field in fields:
     flags=[];unknown=[]
     for row,b in zip(rows,baseline):
      value=row.get('tracking',{}).get(field);old=b.get('tracking',{}).get(field)
      if not finite(value) or (field=='MAE' and not finite(old)):
       flags.append(None);unknown.append(row.get('env',row.get('env_id')));continue
      if field=='MAE':bad=value>c['tracking_MAE_stop_mps'] and value-old>=c['tracking_MAE_increase_stop_mps']
      elif field=='direction':bad=value<c['direction_stop_below']
      elif field=='horizontal_steady_p95_mps':bad=value>c['zero_horizontal_p95_stop_mps']
      elif field=='pitch_hold_abs_max_rad':bad=value>c['pitch_hold_abs_max_rad']
      else:bad=value>c['roll_hold_abs_max_rad']
      flags.append(not bad)
     gates[field]=aggregate(flags)
     if unknown:missing[field]=unknown
     if gates[field]=='FAILED':reasons.append({'MAE':'tracking','horizontal_steady_p95_mps':'zero_drift'}.get(field,field))
  else:gates.update(acceptance='UNKNOWN',survival='UNKNOWN')
  known_failure=retained and bool(reasons)
  old=(state or {}).get('commands',{}).get(key,0)
  if scheduled and state and state['last_update']!=update-c['evaluation_every_updates']:old=0
  count=old+1 if scheduled and known_failure else 0 if scheduled else old
  out['commands'][key]=count
  stop|=bool(scheduled and count>=c['consecutive_failures'])
  unknown=any(v=='UNKNOWN' for v in gates.values())
  decisions.append({'command':command,'evidence':'PARTIAL' if unknown and known_failure else 'UNKNOWN' if unknown else 'KNOWN',
   'gate_evidence':gates,'missing_fields':missing,'known_behavior_failure':known_failure,'retained_source':retained,
   'adaptation':update<c['adaptation_updates'],'scheduled_check':scheduled,'reasons':sorted(set(reasons)),
   'consecutive':count,'promotion_allowed':retained and not fault and not unknown and not known_failure and gates['acceptance']=='PASSED' and all(r.get('survival_passed') is True for r in rows)})
 return out,{'stop':stop,'decisions':decisions,'policy':c}


def advance(state,records,source_records,update,policy=None):
 if isinstance(policy,dict) and policy.get('version')==EVIDENCE_VERSION:
  return _advance_v2(state,records,source_records,update,policy)
 return _advance_v1(state,records,source_records,update,policy)
