"""Explicit progressive straight/yaw course; no files, simulator, or RNG."""
import copy
VERSION='motion31_straight_yaw_progression_v1'
TARGETS=((1.5,0.),(2.,0.),(0.,1.),(0.,1.5))
def configuration():
 return {'version':VERSION,'targets':[list(t) for t in TARGETS],'envs':512,'zero_slots':128,'old_slots':192,'new_slots':192,
  'formal_updates_max':1995,'smoke_updates_max':5,'stage_updates_max':1500,'evaluation_every':250,'confirmations':2,
  'source_capabilities':[[0.,0.],[.5,0.],[-.5,0.],[1.,0.],[-1.,0.]],
  'retention':'all previously unlocked endpoints at every fixed full evaluation','reward':'original quality_v3 weights and stopping schedule unchanged','refinement_gate':'new absolute protocol replaces source-relative posture checks; mixture numeric policy and kernels unchanged',
  'unlock':'two complete8-per-command passed matrices with real PPO updates between; source/smoke never unlock',
  'budget':'one formal optimizer lifecycle, global counters never reset on stage advance','final_scope':'single training stream and one evaluation seed per milestone; not final multiseed yaw certification'}
def commands(stage):
 if type(stage) is not int or stage not in range(4):raise ValueError('invalid straight/yaw stage')
 result=configuration()['source_capabilities']
 for vx,yaw in TARGETS[:stage+1]:result.extend([[vx,yaw],[-vx,-yaw]])
 return result

def stage_config(stage,envs=512):
 if envs not in (64,512):raise ValueError('explicit course64/512 envs only')
 values=commands(stage);unit=envs//64;counts=[16*unit]+[0]*(len(values)-1)
 counts[-2:]=[12*unit,12*unit]
 pairs=(len(values)-3)//2
 # Equal +/- within each old endpoint; deterministic integer remainder.
 for j in range(12*unit):
  pair=j%pairs;counts[1+pair*2]+=1;counts[2+pair*2]+=1
 return {'id':stage,'name':('L1p5','L2','Y1','Y1p5')[stage],'commands':values,'weights':[n/envs for n in counts],
  'sampling_mode':'fixed_slots','sampling_slots':[i for i,n in enumerate(counts) for _ in range(n)],'enforce_whitelist':True,'slot_counts':counts}

def effective_spec(spec,stage):
 """Local active-stage view; frozen spec and original evidence remain unchanged."""
 s=copy.deepcopy(spec);s['motion_training']['stage']=stage;s['student_training']['stage']=stage;s['student_training']['course_stage']=stage
 s['curriculum']['stages']=[stage_config(stage,s['ppo']['envs'])];s['student_evaluation']['cases']=cases(s,stage)
 return s

def cases(spec,stage):
 template=spec['student_evaluation']['cases'][0]
 return [{**template,'case_id':f'course_{stage}_command_{i}','command':v+[.34]} for i,v in enumerate(commands(stage))]

def extend_mixture(previous,values):
 by_target={tuple(r['target']):r for r in previous}
 return [copy.deepcopy(by_target[tuple(v)]) if tuple(v) in by_target else {'target':list(v),'mix':0.,'consecutive_eligible':0} for v in values]

def advance(state,candidate):
 """Frozen8-env command matrix only; incomplete/old-stage evidence cannot unlock."""
 stage=state['stage'];update=candidate['update']
 if candidate.get('stage')!=stage or update<=state.get('last_evaluation_update',-1):raise ValueError('course stage/update evidence differs')
 records=candidate.get('evaluations',[]);expected=commands(stage)
 complete=candidate.get('complete_matrix') is True and len(records)==len(expected)
 if complete:
  complete=all(r.get('case',{}).get('command')==v+[.34] and len(r.get('per_env',[]))==8 and len({e.get('env',e.get('env_id')) for e in r['per_env']})==8 and all(e.get('accepted') is True for e in r['per_env']) for r,v in zip(records,expected))
 streak=state.get('consecutive_passes',0)+1 if complete else 0
 out={**state,'last_evaluation_update':update,'consecutive_passes':streak};passed=streak>=2
 if passed:
  out['accepted_by_stage']={**state.get('accepted_by_stage',{}),str(stage):{'checkpoint':candidate['checkpoint'],'sha256':candidate['sha256'],'update':update,'evaluations':[{'path':r['path'],'sha256':r['sha256']} for r in records]}}
 if passed and stage<3:out.update(stage=stage+1,stage_started_update=update,consecutive_passes=0)
 return out,{'stage_passed':passed,'advanced':passed and stage<3,'all_stages_passed':passed and stage==3,'complete_matrix_passed':complete}

def choose(candidates,accepted_by_stage):
 stage=max(c.get('stage',-1) for c in candidates if c.get('update',0)>0) if any(c.get('update',0)>0 for c in candidates) else 0
 new=[c for c in candidates if c.get('update',0)>0 and c.get('stage')==stage and c.get('complete_matrix')]
 def rank(c):return (sum(e['accepted'] for e in c['per_env']),sum(e['survival_passed'] for e in c['per_env']),-max((e.get('tracking',{}).get('MAE',999) for e in c['per_env']),default=999))
 best=max(new,key=rank) if new else None
 certified=accepted_by_stage.get(str(stage));accepted=next((c for c in new if stage==3 and certified and c['sha256']==certified['sha256']),None)
 return {'source_baseline':candidates[0],'source':candidates[0],'fallback_checkpoint':candidates[0],'latest_checkpoint':candidates[-1],
  'best_new_evaluated_candidate':best,'best_new':best,'best_overall':best or candidates[0],'best_observed_candidate':best or candidates[0],
  'accepted_candidate':accepted,'accepted_by_stage':accepted_by_stage,'current_stage':stage,'final_multi_seed_certified':False,'standing_quality_certified':False}


def implementation(version):
 if version=='motion31_standing_sensor20_target5_motor20_v1':
  from . import motion_standing_delay
  return motion_standing_delay
 if version=='motion31_mcu_highspeed_standing2048_v1':
  from . import motion_highspeed_standing
  return motion_highspeed_standing
 if version=='motion31_mcu_height_pd_parallel_v1':
  from . import motion_height_pd
  return motion_height_pd
 if version=='motion31_mcu_handover_pd_age_v1':
  from . import motion_handover_robustness
  return motion_handover_robustness
 if version=='motion31_mcu_posture_rehearsal_v1':
  from . import motion_posture_design
  return motion_posture_design
 if version=='motion31_mcu_measured_frontiers_v2':
  from . import motion_mcu_measured
  return motion_mcu_measured
 if version=='motion31_mcu_standing_adaptation_v1':
  from . import motion_mcu_course
  return motion_mcu_course
 if version=='motion31_persistent_frontiers_v1':
  from . import motion_persistent
  return motion_persistent
 if version=='motion31_fixed_height_motion_v2':
  from . import motion_fixed_height
  return motion_fixed_height
 if version=='motion31_decoupled_height_course_v1':
  from . import motion_decoupled
  return motion_decoupled
 if version=='motion31_alternating_standing_robustness_v1':
  from . import motion_robustness
  return motion_robustness
 if version=='motion31_posture_reacquisition_v1':
  from . import motion_reacquisition
  return motion_reacquisition
 if version=='motion31_standing_antiosc_armature_v4':
  from . import motion_antioscillation
  return motion_antioscillation
 if version=='motion31_standing_sensor_transfer_v3':
  from . import motion_sensor_transfer
  return motion_sensor_transfer
 if version=='motion31_standing_fixed_timing_v2':
  from . import motion_fixed_timing
  return motion_fixed_timing
 if version=='motion31_standing_timing_course_v1':
  from . import motion_timing
  return motion_timing
 if version=='motion31_parallel_bidirectional_course_v1':
  from . import motion_parallel_course
  return motion_parallel_course
 if version==VERSION:
  import sys
  return sys.modules[__name__]
 from . import motion_forward_course,motion_yaw_course,motion_yaw_width_course,motion_yaw_retention_course,motion_large_combo
 if version==motion_large_combo.VERSION:return motion_large_combo
 if version==motion_forward_course.VERSION:return motion_forward_course
 if version==motion_yaw_course.VERSION:return motion_yaw_course
 if version==motion_yaw_width_course.VERSION:return motion_yaw_width_course
 if version==motion_yaw_retention_course.VERSION:return motion_yaw_retention_course
 raise ValueError('unknown motion course version')
