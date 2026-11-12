"""Explicit in-place bidirectional yaw curriculum, v3 training recipe; parking diagnostic only."""
import copy
VERSION='motion31_yaw_progression_v1'
ASSESSMENT='motion31_yaw_capability_v1'
TARGETS=(.5,1.,1.5)

def configuration():
 return {'version':VERSION,'targets':list(TARGETS),'envs':512,'zero_slots':128,'old_slots':192,'yaw_slots':192,
  'formal_updates_max':1995,'smoke_updates_max':5,'stage_updates_max':1500,'evaluation_every':250,'confirmations':2,
  'source_capabilities':[[0.,0.],[.5,0.],[1.,0.],[1.5,0.],[2.,0.]],'parking':'diagnostic only; original training schedule retained',
  'scope':'in-place yaw plus retained positive straight; single evaluation seed, not multi-seed certification'}

def commands(stage):
 if type(stage) is not int or stage not in range(3):raise ValueError('invalid yaw stage')
 return configuration()['source_capabilities']+[[0.,sign*v] for v in TARGETS[:stage+1] for sign in (1,-1)]

def stage_config(stage,envs=512):
 if envs not in (64,512):raise ValueError('yaw64/512 only')
 values=commands(stage);unit=envs//64
 # Keep 24/64 forward slots; distribute the 24/64 yaw slots over unlocked pairs.
 yaw_counts=[0]*(2*(stage+1))
 for i in range(12*unit):
  pair=i%(stage+1);yaw_counts[2*pair]+=1;yaw_counts[2*pair+1]+=1
 counts=[16*unit]+[6*unit]*4+yaw_counts
 return {'id':stage,'name':('Y0p5','Y1','Y1p5')[stage],'commands':values,'weights':[n/envs for n in counts],
  'sampling_mode':'fixed_slots','sampling_slots':[i for i,n in enumerate(counts) for _ in range(n)],'enforce_whitelist':True,'slot_counts':counts}

def protocol():
 from .motion_guard import protocol as absolute
 p=absolute();p.update(version=ASSESSMENT,parking_required=False,parking_scope='diagnostic only; not precise stopping',
  yaw_scope='in-place positive/negative yaw; retain zero and positive straight; original full30s/hold10..18s',
  acceptance_note='yaw MAE<=.3rad/s and gain .8..1.2, direction>=.95; horizontal hold mean<=.3m/s; pitch<=.1 roll<=.05; no precision parking gate')
 return p

def assess(result):
 r=copy.deepcopy(result);r['version']=ASSESSMENT
 r.setdefault('diagnostic_checks',{})['parking']=r['checks'].pop('parking',None)
 r['accepted']=all(r['checks'].values());r['failed_checks']=[k for k,v in r['checks'].items() if not v]
 return r

def cases(spec,stage):
 template=spec['student_evaluation']['cases'][0]
 return [{**template,'case_id':f'yaw_{stage}_command_{i}','command':v+[.34]} for i,v in enumerate(commands(stage))]

def effective_spec(spec,stage):
 s=copy.deepcopy(spec);s['motion_training']['stage']=stage;s['student_training']['stage']=stage;s['student_training']['course_stage']=stage
 s['curriculum']['stages']=[stage_config(stage,s['ppo']['envs'])];s['student_evaluation']['cases']=cases(s,stage)
 return s

def advance(state,candidate):
 stage=state['stage'];update=candidate['update']
 if candidate.get('stage')!=stage or update<=state.get('last_evaluation_update',-1):raise ValueError('yaw stage/update differs')
 records=candidate.get('evaluations',[])
 complete=candidate.get('complete_matrix') is True and len(records)==len(commands(stage))
 if complete:
  complete=all(r.get('case',{}).get('command')==v+[.34] and len(r.get('per_env',[]))==8 and len({e.get('env',e.get('env_id')) for e in r['per_env']})==8 and all(e.get('accepted') is True for e in r['per_env']) for r,v in zip(records,commands(stage)))
 streak=state.get('consecutive_passes',0)+1 if complete else 0;passed=streak>=2
 out={**state,'last_evaluation_update':update,'consecutive_passes':streak}
 if passed:out['accepted_by_stage']={**state.get('accepted_by_stage',{}),str(stage):{'checkpoint':candidate['checkpoint'],'sha256':candidate['sha256'],'update':update,'evaluations':[{'path':r['path'],'sha256':r['sha256']} for r in records]}}
 if passed and stage<2:out.update(stage=stage+1,stage_started_update=update,consecutive_passes=0)
 return out,{'stage_passed':passed,'advanced':passed and stage<2,'all_stages_passed':passed and stage==2,'complete_matrix_passed':complete}

def choose(candidates,accepted_by_stage):
 from .motion_course import choose as base
 r=base(candidates,{})
 stage=r['current_stage'];passed=accepted_by_stage.get(str(stage))
 def rank(candidate):
  records=candidate.get('evaluations',[])
  old=[e for row in records if row['case']['command'][:2] in configuration()['source_capabilities'] for e in row.get('per_env',[])]
  current=[e for row in records if row['case']['command'][:2] in [[0.,TARGETS[stage]],[0.,-TARGETS[stage]]] for e in row.get('per_env',[])]
  return (sum(e.get('accepted') is True for e in old),sum(e.get('accepted') is True for e in current),
   -max((e.get('tracking',{}).get('MAE',999) for e in current),default=999))
 new=[c for c in candidates if c.get('update',0)>0 and c.get('stage')==stage and c.get('complete_matrix') is True]
 best=max(new,key=rank) if new else None
 overall=max([candidates[0]]+new,key=rank)
 accepted=next((c for c in new if stage==2 and passed and c.get('sha256')==passed['sha256']),None)
 r.update(best_new_evaluated_candidate=best,best_new=best,best_overall=overall,best_observed_candidate=overall,
  accepted_candidate=accepted,accepted_by_stage=accepted_by_stage,final_multi_seed_certified=False,
  capability_scope='in-place yaw and retained forward endpoints; no reverse/highspeed yaw certification')
 return r

from .motion_course import extend_mixture
