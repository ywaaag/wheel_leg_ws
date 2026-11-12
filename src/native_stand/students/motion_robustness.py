"""Explicit alternating Motion/timing/standing robustness course; no hidden controls."""
import copy
import json
from pathlib import Path
import torch
from . import motion_reacquisition as base
from .data import sha

VERSION='motion31_alternating_standing_robustness_v1'
MATERIAL_BACKEND='motion_friction_lattice64_v1'
MATERIAL_WRITER='articulation_wheel_shapes_v1'
RESUME_VERSION='motion_robustness_material_resume_v1'
DIAGNOSTIC_RESUME_VERSION='motion_robustness_diagnostic_resume_v1'
EVALUATION_RESUME_VERSION='motion_robustness_eval_material_resume_v1'
BASELINE_RESUME_VERSION='motion_robustness_group_baseline_resume_v1'
RESUME_VERSIONS=(RESUME_VERSION,DIAGNOSTIC_RESUME_VERSION,EVALUATION_RESUME_VERSION,BASELINE_RESUME_VERSION)
TARGETS=base.TARGETS
LEVELS=((.5,0,0,.5,.5,.5),(1,0,0,.5,.5,.5),(1,.5,0,1,.5,.5),
        (1,1,0,1,1,.5),(1,1,.5,1,1,1),(1,1,1,1,1,1))
GROUPS=('nominal','previous','current')
commands=base.commands
stage_config=base.stage_config
protocol=base.protocol
scoring_protocol=base.scoring_protocol
assess=base.assess
extend_mixture=base.extend_mixture


def configuration():
 return dict(version=VERSION,levels=[list(x) for x in LEVELS],families=['Actor_packet_age','target_transport','PD_feedback_age','standing_root_mass','standing_wheel_friction','standing_push'],
  seed=20261023,groups=list(GROUPS),group_fractions=[.25,.25,.5],confirmations=1,every_updates=250,
  standing_only=['root_mass_and_inertia','wheel_friction','push'],mass_full_scale=[.9,1.1],wheel_shape_friction_full=[.4,.6],ground_friction=.5,friction_combine='average',
  push=dict(version='standing_velocity_impulse_v1',full_delta_v_mps=.1,controlled_times_s=[12.,22.],axes='world XY, second impulse opposite first',not_a_measured_hardware_force=True),
  actor_age_ms=[8,12],target_common_offset_ms=[-2,2],PD_feedback_age_ms=[8,12],
  sampling='reset-only independent counter stream; exact command x DR quotas; no policy RNG',
  rollback='one known mastered-command failure -> one command level down; retain model/Adam/DR; new task failure stays',
  DR_retreat=False,simultaneous_difficulty_increases=False)


def reporting_protocol(scoring):
 """Report the actual course owner; do not change the scoring declaration."""
 from .motion_refinement import ABSOLUTE_EVIDENCE_VERSION
 result=copy.deepcopy(scoring);course=configuration()
 result['confirmations_required']=course['confirmations']
 result['refinement_evidence_version']=ABSOLUTE_EVIDENCE_VERSION
 if 'absolute_posture' in result:
  # The inherited dictionary mixes scoring limits with an unused legacy guard.
  result['absolute_posture']={k:v for k,v in result['absolute_posture'].items()
                            if k in ('pitch_hold_abs_max_rad','roll_hold_abs_max_rad','relative_posture')}
 result['course_controller']=dict(version=VERSION,confirmations_required=course['confirmations'],
  evaluation_every_updates=course['every_updates'],rollback=course['rollback'],
  DR_retreat=course['DR_retreat'],behavior_chain_stop=False,runtime_fault_action='stop',
  unknown='blocks promotion; not a known behavioral regression')
 return result


def strength(level,group):
 if level not in range(len(LEVELS)) or group not in range(3):raise ValueError('invalid robustness level/group')
 return (0.,)*6 if group==0 or group==1 and level==0 else LEVELS[level-1 if group==1 else level]


def parameters(ids,episodes,strengths,*,seed=20261023,standing=None,eval_corners=False,material_backend=None):
 """Pure torch, stable env/episode/family keys. No random generator is consumed."""
 ids=torch.as_tensor(ids,dtype=torch.long);episodes=torch.as_tensor(episodes,device=ids.device,dtype=torch.long)
 s=torch.as_tensor(strengths,device=ids.device,dtype=torch.float32)
 if s.shape!=(len(ids),6) or ((s<0)|(s>1)).any():raise ValueError('DR strength shape/range differs')
 result=torch.tensor([10,0,10,1,.5,0,0],device=ids.device,dtype=torch.float32).repeat(len(ids),1)
 for family in range(6):
  if not bool((s[:,family]>0).any()):continue
  key=(ids*1103515245+episodes*12345+(family+701)*2654435761+seed)%2147483647
  # Integer hash mixing prevents different families from being simple shifted copies.
  key=((key^(key>>16))*0x45d9f3b)&0x7fffffff
  key=((key^(key>>16))*0x45d9f3b)&0x7fffffff
  key=(key^(key>>16))&0x7fffffff
  u=(key.float()/2147483648).clamp_max(1-1e-7)
  if eval_corners:u=torch.where(ids%8==0,0.,torch.where(ids%8==1,1-1e-7,u))
  amplitude=s[:,family]
  if family<3:
   half=torch.floor(2*amplitude);result[:,family]+=torch.floor(u*(2*half+1))-half
  elif family==3:result[:,3]+=(2*u-1)*.1*amplitude
  elif family==4:result[:,4]+=(2*u-1)*.1*amplitude
  else:
   result[:,5]=.1*amplitude
   # Eight deterministic directions include opposite pairs, independent of policy sampling.
   result[:,6]=(ids%8).float()*(torch.pi/4) if eval_corners else u*(2*torch.pi)
 if standing is not None:
  mask=torch.as_tensor(standing,device=ids.device,dtype=torch.bool)
  nominal=result.new_tensor([1,.5,0,0])
  result[:,3:]=torch.where(mask[:,None],result[:,3:],nominal)
 if material_backend is not None:
  if material_backend!=MATERIAL_BACKEND:raise ValueError('unknown Motion friction backend')
  # A fixed lattice limits repeated PhysX material creation/reuse; preserve .5
  # exactly and the original half/full range endpoints. No additional RNG.
  result[:,4]=torch.round(result[:,4]*320)/320
 return result


def cross_quotas(counts):
 table=torch.zeros(len(counts),3,dtype=torch.long)
 if sum(counts)!=4096:raise ValueError('training command slots must sum4096')
 table[:,:2]=torch.as_tensor(counts)[:,None]//4
 for group in (0,1):
  missing=1024-int(table[:,group].sum())
  if missing%2:raise ValueError('sign-pair remainder differs')
  for j in range(missing//2):
   pair=1+2*(j%((len(counts)-1)//2));table[pair:pair+2,group]+=1
 table[:,2]=torch.as_tensor(counts)-table[:,:2].sum(1)
 if (table<0).any():raise ValueError('negative group quota')
 return table


def initial_state(stage=0):
 return dict(version=VERSION,stage=stage,DR_level=0,certified_DR_level=-1,mode='DR_ACQUIRE',pending_skill_stage=None,
  stage_started_update=0,last_evaluation_update=-1,last_full_matrix_passed=None,accepted_by_stage={},qualified_by_DR={},
  last_checkpoint_sha256=None,last_evaluation_id=None,smooth_mix=0.,mixture_archive={},symmetry_archive={})


def cases(spec,stage=0,state=None):
 s=state or spec['motion_robustness']['state'];level=s['DR_level'];template=spec['student_evaluation']['cases'][0]
 result=[]
 for group,name in enumerate(GROUPS):
  for i,v in enumerate(commands(stage)):
   row={k:copy.deepcopy(x) for k,x in template.items() if not k.startswith('_')}
   row.update(case_id=f'robust_{level}_{name}_{i}',command=v+[.34],_timing_level=4,
     _motion_dr=dict(level=level,group=group,mode=s['mode'],skill_stage=stage,seed=20261023+250*max(0,s['last_evaluation_update'])))
   result.append(row)
 return result


def effective_spec(spec,stage,state=None):
 s=copy.deepcopy(spec);state=copy.deepcopy(state or s['motion_robustness']['state'])
 s['motion_robustness']['state']=state
 s['motion_training']['stage']=stage;s['student_training'].update(stage=stage,course_stage=stage)
 s['motion_timing']['stage']=4;s['curriculum']['stages']=[stage_config(stage,s['ppo']['envs'])]
 s['motion_timing']['configuration']['smooth_mix']=state.get('smooth_mix',s['motion_timing']['configuration']['smooth_mix'])
 s['student_training']['timing_configuration']=copy.deepcopy(s['motion_timing']['configuration'])
 s['student_evaluation']['cases']=cases(s,stage,state)
 return s


def refinement_records(records,stage):
 expected=commands(stage);out=[]
 for target in expected:
  parts=[r for r in records if r.get('case',{}).get('command')==target+[.34]]
  good=(len(parts)==3 and {r['case'].get('_motion_dr',{}).get('group') for r in parts}=={0,1,2}
        and all(len(r.get('per_env',[]))==8 and {e.get('env') for e in r['per_env']}==set(range(8)) for r in parts))
  row=dict(case={'command':target+[.34]},per_env=[])
  if good:
   for i in range(8):
    entries=[next(e for e in r['per_env'] if e['env']==i) for r in parts]
    worst=max(entries,key=lambda e:(e.get('accepted') is not True,e.get('tracking',{}).get('MAE',999)))
    e=copy.deepcopy(worst);e['accepted']=all(x.get('accepted') is True for x in entries) if all(type(x.get('accepted')) is bool for x in entries) else None;row['per_env'].append(e)
  out.append(row)
 return out


def advance(state,candidate):
 s=copy.deepcopy(state);update=candidate['update'];records=candidate.get('evaluations',[])
 if update<=s['last_evaluation_update']:
  return s,dict(advanced=False,stage_passed=False,all_stages_passed=False,stop=False,duplicate=True)
 if candidate.get('stage')!=s['stage']:raise ValueError('robustness candidate command stage differs')
 for r in records:
  c=r.get('case',{}).get('_motion_dr',{})
  if any(c.get(k)!=s[k] for k in ('mode','DR_level') if k in c):raise ValueError('robustness candidate phase differs')
  if c.get('level')!=s['DR_level'] or c.get('skill_stage')!=s['stage']:raise ValueError('robustness candidate DR scope differs')
 merged=refinement_records(records,s['stage'])
 valid=(candidate.get('complete_matrix') is True and len(records)==3*len(commands(s['stage']))
        and all(len(r['per_env'])==8 and all(type(e.get('accepted')) is bool for e in r['per_env']) for r in merged))
 passed=valid and all(e['accepted'] for r in merged for e in r['per_env'])
 before=(s['stage'],s['DR_level'],s['mode']);level=s['DR_level']
 known=set(map(tuple,s['qualified_by_DR'].get(str(level),[])))
 regression=valid and any(tuple(r['case']['command'][:2]) in known and any(e['accepted'] is False for e in r['per_env']) for r in merged)
 s.update(last_evaluation_update=update,last_checkpoint_sha256=candidate['sha256'],last_evaluation_id=f'{update}:{candidate["sha256"]}',last_full_matrix_passed=passed if valid else None)
 decision=dict(stage_passed=passed,complete_matrix_passed=passed,old_command_regression=regression,stop=False)
 if valid and regression and s['mode'] in ('COMMAND_ACQUIRE','COMMAND_REPAIR','FINAL_CONSOLIDATE'):
  if s['pending_skill_stage'] is None:s['pending_skill_stage']=s['stage']
  s.update(stage=max(0,s['stage']-1),mode='COMMAND_REPAIR')
 elif passed:
  s['accepted_by_stage'][str(s['stage'])]={k:candidate[k] for k in ('checkpoint','sha256','update')}
  s['accepted_by_stage'][str(s['stage'])]['DR_level']=level
  s['qualified_by_DR'][str(level)]=[list(v) for v in sorted(known|set(map(tuple,commands(s['stage']))))]
  if s['mode']=='COMMAND_REPAIR':
   s['stage']=min(s['stage']+1,s['pending_skill_stage'])
   if s['stage']==s['pending_skill_stage']:s.update(mode='COMMAND_ACQUIRE',pending_skill_stage=None)
  elif s['mode']=='DR_ACQUIRE':
   s['certified_DR_level']=level
   if s['stage']<5:s.update(stage=s['stage']+1,mode='COMMAND_ACQUIRE')
   elif level<len(LEVELS)-1:s['DR_level']+=1
   else:s['mode']='FINAL_CONSOLIDATE'
  elif s['mode']=='COMMAND_ACQUIRE':
   if level<len(LEVELS)-1:s.update(DR_level=level+1,mode='DR_ACQUIRE')
   elif s['stage']<5:s['stage']+=1
   else:s['mode']='FINAL_CONSOLIDATE'
 if valid:s['smooth_mix']=min(.75,s.get('smooth_mix',0.)+.25) if passed else max(0.,s.get('smooth_mix',0.)-.25)
 changed=before!=(s['stage'],s['DR_level'],s['mode'])
 if changed:s['stage_started_update']=update
 decision.update(advanced=changed,all_stages_passed=s['mode']=='FINAL_CONSOLIDATE',mode=s['mode'],
                 command_stage=s['stage'],DR_level=s['DR_level'],evidence_valid=valid,
                 progression='single complete8-env matrix; alternate one axis; command repair only, DR monotone')
 return s,decision


def choose(candidates,state):
 accepted=state['accepted_by_stage'];result=base.choose(candidates,accepted)
 eligible=[c for c in candidates if c.get('update',0)>0 and c.get('stage')==state['stage'] and c.get('complete_matrix')
           and c.get('evaluations') and all(r['case']['_motion_dr']['level']==state['DR_level'] for r in c['evaluations'])]
 best=max(eligible,key=lambda c:(sum(e.get('accepted') is True for e in c['per_env']),c['update']),default=None)
 final=accepted.get('5')
 certified=next((c for c in candidates if final and final.get('DR_level')==len(LEVELS)-1 and c['sha256']==final['sha256']),None)
 latest_pass=max(accepted.values(),key=lambda row:row['update'],default=None)
 anchor=next((c for c in candidates if latest_pass and c['sha256']==latest_pass['sha256']),None)
 overall=certified or best or anchor or candidates[0]
 result.update(current_stage=state['stage'],current_DR_level=state['DR_level'],current_mode=state['mode'],
               highest_command_stage_seen=max(c.get('stage',0) for c in candidates),
               best_new_evaluated_candidate=best,best_new=best,best_overall=overall,best_observed_candidate=overall,
               accepted_candidate=certified,capability_scope='one checkpoint and its exact skill/DR matrix; active command ceiling may retreat; no multiseed or hardware certification')
 return result


def validate_recipe(recipe,source_sha):
 r=copy.deepcopy(recipe);r['course_version']=base.VERSION
 offset=r.pop('formal_resume_offset',None)
 if offset is not None:
  original=list(range(250,10000,250))
  if type(offset) is not int or not 0<offset<9995 or r['milestones']!=[u-offset for u in original if u>offset]:
   raise ValueError('explicit material-resume evaluation offset differs')
  r['milestones']=original
 base.validate_recipe(r,source_sha);return recipe


def archive_mixtures(state,ppo):
 for name,attr in [('mixture_archive','reward_refinement_state'),('symmetry_archive','symmetry_mix_state')]:
  archive=state.setdefault(name,{})
  for row in getattr(ppo,attr,[]):archive[','.join(map(str,row['target']))]=copy.deepcopy(row)


def restored_mixture(state,name,previous,values):
 archive=state.get(name,{})
 combined=list(archive.values())+list(previous)
 return extend_mixture(combined,values)


def preppo_recovery_evidence(parent_spec):
 """Recover only a settled, identified inference-buffer fault before any PPO."""
 p=Path(parent_spec).resolve();old=json.loads(p.read_text())
 run=p.parents[3]/'runs'/old['experiment']
 book=json.loads(p.with_name('budget.json').read_text())
 chain=json.loads((run/'chain_terminal_report.json').read_text())
 terminal=json.loads((run/'source_eval/terminal_report.json').read_text())
 failure=terminal.get('failure') or {}
 if (old.get('motion_robustness',{}).get('version')!=VERSION or chain['status']!='FAILED'
     or terminal['status']!='FAILED' or not book['attempts']
     or any(a['status']=='RUNNING' for a in book['attempts'].values())
     or any(a['phase']!='source_eval' or a.get('updates')!=0 for a in book['attempts'].values())
     or any(book['used'].get(k,0) for k in ('smoke','formal','evaluate','play'))
     or chain['smoke_updates']!=0 or chain['formal_updates']!=0 or terminal['actual_updates']!=0):
  raise ValueError('settled zero-PPO robustness source failure required')
 if ('Inplace update to inference tensor outside InferenceMode' not in failure.get('exception','')
     or 'motion_robustness_runtime.py' not in failure.get('traceback','')
     or 'first_adoption_ages' not in failure.get('traceback','')):
  raise ValueError('unsupported robustness failure; diagnose before recovery')
 verify(old)
 paths=[p,p.with_name('budget.json'),run/'chain_terminal_report.json',run/'source_eval/terminal_report.json']
 previous=old['motion_training'].get('prior_consumption',{})
 return dict(version='motion_robustness_preppo_buffer_recovery_v1',parent_spec=str(p),parent_run=str(run),
  inputs={str(q):sha(q) for q in paths},historical_online_updates=0,original_online_limit=10000,
  historical_wall_s=previous.get('historical_wall_s',0.)+chain['chain_wall_s'],
  source_sha256=old['source_sha256'],source_parent_spec=old['motion_robustness']['parent_spec'],
  reason='normal reset after inference-mode PD adoption replaced a persistent buffer; repeat engineering readback, no smoke certificate reuse')


def build(parent,checkpoint,*,experiment,spec_path,confirmation,owner_session,storage_max_bytes=8*2**30,recovery_parent=None):
 p=Path(parent).resolve();cp=Path(checkpoint).resolve();old=json.loads(p.read_text());run=p.parents[3]/'runs'/old['experiment']
 term=json.loads((run/'formal/terminal_report.json').read_text());chain=json.loads((run/'chain_terminal_report.json').read_text());book=json.loads(p.with_name('budget.json').read_text())
 if (chain['status'] not in ('COMPLETED','STOPPED_BOUNDED') or chain.get('failure') or term.get('failure')
     or any(a['status']=='RUNNING' for a in book['attempts'].values()) or term['actual_updates']!=book['used']['formal']):raise ValueError('settled normal parent required')
 source=torch.load(cp,map_location='cpu',weights_only=False)
 if old.get('student_training',{}).get('course_version')!=base.VERSION:raise ValueError('this first handoff requires the specific posture-reacquisition parent')
 if confirmation.get('status')!='CONFIRMED' or confirmation.get('parent_spec_sha256')!=sha(p) or confirmation.get('certified_skill_stage') not in range(6):raise ValueError('bound parent promotion proof required')
 if (cp!=run/'formal'/f'model_update_{term["actual_updates"]}.pt' or source['spec_sha256']!=sha(p)
     or source['critic_privileged_version']!='motion_antiosc_privileged140_v1'):raise ValueError('latest complete same140 parent source required')
 for path,digest in confirmation['inputs'].items():
  if sha(path)!=digest:raise ValueError('parent promotion evidence changed')
 s=copy.deepcopy(old);s.update(experiment=experiment,source_checkpoint=str(cp),source_sha256=sha(cp))
 stage=confirmation['certified_skill_stage'];state=initial_state(stage)
 state['smooth_mix']=source['training_recipe']['timing_configuration']['smooth_mix']
 s['motion_robustness']=dict(version=VERSION,configuration=configuration(),state=state,parent_spec=str(p),parent_spec_sha256=sha(p),confirmation=confirmation,
  source_latest_unassessed=True,automatic_engineering_recovery=True,
  source_readback_scope='one nominalzero engineering case, not a full capability baseline',storage_profile='eight_env_metrics_one_env_trace_v1')
 s.pop('motion_reacquisition',None)
 s['student_training'].update(course_version=VERSION,course_stage=stage,stage=stage,source_actor_sha256=sha(cp),storage_max_bytes=storage_max_bytes,
  milestones=list(range(250,10000,250)),checkpoint_every_updates=100)
 s['motion_timing'].update(parent_spec=str(p),parent_spec_sha256=sha(p),source_checkpoint_sha256=sha(cp))
 s['motion_training'].update(spec_path=str(Path(spec_path).resolve()),resume=None,stage=stage)
 for k in ('wiring_reuse','partial_smoke_recovery','prior_consumption','successor'):s['motion_training'].pop(k,None)
 s['motion_course']={'source_recipe_updates':source['recipe_updates'],'configuration':{**configuration(),'formal_updates_max':9995,'stage_updates_max':9995,'smoke_updates_max':5}}
 s['budget'].update(smoke_updates=5,formal_updates=9995,formal_samples=9995*4096*128,sample_limit=10000*4096*128,formal_wall_s=None,chain_wall_s=None,source_eval_wall_s=7200,smoke_wall_s=1800,evaluation_wall_s=21600)
 s['artifacts'].update(run_name=experiment,storage_max_bytes=storage_max_bytes,terminal_reserve_bytes=512*2**20)
 s['notifications']={'enabled':True};s['authorization']={'status':'UNAUTHORIZED','hapi_session':owner_session,
  'user_scope':'User2026-10-23 explicitly authorizes one alternating Motion/DR successor from latest normally stopped parent; standing root mass/friction/push plus timing DR; new inclusive10000 PPO, unlimited training wall, sharedGPU0, current-session notifications and engineering repair/resume within cumulative budget'}
 s['motion_startup'].update(references=[],missing_commands=[[0.,0.,.34]])
 s['student_evaluation']['cases'][0].update(checkpoint=str(cp),sha256=sha(cp))
 s=effective_spec(s,stage,state)
 paths=[p,p.with_name('budget.json'),run/'formal/terminal_report.json',run/'chain_terminal_report.json',cp]
 s['student_dependencies'].update({str(q):sha(q) for q in paths});s['student_dependencies'].update(confirmation['inputs'])
 if recovery_parent is not None:
  recovery=preppo_recovery_evidence(recovery_parent)
  if recovery['source_sha256']!=s['source_sha256'] or recovery['source_parent_spec']!=str(p):
   raise ValueError('pre-PPO recovery must retain exact original source and parent')
  s['motion_robustness']['engineering_recovery']=recovery
  s['motion_training']['prior_consumption']=recovery
  s['student_dependencies'].update(recovery['inputs'])
 return s


def verify(spec):
 if spec.get('motion_training',{}).get('wiring_reuse',{}).get('version') in RESUME_VERSIONS:
  return verify_resume(spec)
 cfg=spec['motion_robustness']
 if cfg['version']!=VERSION or cfg['configuration']!=configuration():raise ValueError('explicit robustness configuration differs')
 expected=build(cfg['parent_spec'],spec['source_checkpoint'],experiment=spec['experiment'],spec_path=spec['motion_training']['spec_path'],confirmation=cfg['confirmation'],owner_session=spec['authorization']['hapi_session'],storage_max_bytes=spec['artifacts']['storage_max_bytes'],recovery_parent=cfg.get('engineering_recovery',{}).get('parent_spec'))
 for key in ('authorization','resources','notifications'):expected[key]=spec[key]
 if spec!=expected:raise ValueError('robustness declaration has undeclared drift')
 validate_recipe(spec['student_training'],spec['source_sha256']);return True


def resume_evidence(parent_spec,checkpoint,material_probe):
 """Only identified material/reset or checkpoint-diagnostic wiring failures."""
 from native_stand.command_v6.wiring import validate_persisted_certificate
 p=Path(parent_spec).resolve();old=json.loads(p.read_text());verify(old)
 run=p.parents[3]/'runs'/old['experiment'];cp=Path(checkpoint).resolve();probe=Path(material_probe).resolve()
 book=json.loads(p.with_name('budget.json').read_text());chain=json.loads((run/'chain_terminal_report.json').read_text())
 terminal=json.loads((run/'formal/terminal_report.json').read_text());progress=json.loads((run/'formal/progress.json').read_text())
 error=terminal.get('failure') or {};n=book['used']['formal'];previous=old['motion_training'].get('wiring_reuse',{})
 material_fault=(error.get('exception','').startswith("RuntimeError('Motion DR friction backend readback differs")
                 and 'motion_robustness_runtime.py' in error.get('traceback','') and not previous)
 diagnostic_fault=(error.get('exception')=="ValueError('diagnostics require actual formal100 boundary')"
                   and 'motion_parallel_course.py' in error.get('traceback','')
                   and previous.get('version') in RESUME_VERSIONS)
 evaluation_fault=(error.get('exception','').startswith('RuntimeError("Motion DR friction backend readback differs')
                   and 'motion_robustness_runtime.py' in error.get('traceback','')
                   and 'evaluate_cases' in error.get('traceback','') and previous.get('version') in RESUME_VERSIONS)
 baseline_fault=(error.get('exception')=="ValueError('duplicate course baseline command')"
                 and 'training.py' in error.get('traceback','') and 'evaluate_cases' in error.get('traceback','')
                 and previous.get('version')==EVALUATION_RESUME_VERSION)
 if (chain['status']!='FAILED' or terminal['status']!='FAILED'
     or any(a['status']=='RUNNING' for a in book['attempts'].values())
     or not (material_fault or diagnostic_fault or evaluation_fault or baseline_fault)):
  raise ValueError('settled identified robustness wiring failure required')
 if cp!=run/'formal'/f'model_update_{n}.pt' or n<=0 or terminal['actual_updates']!=n or progress['actual_updates']!=n:
  raise ValueError('exact charged full formal checkpoint required')
 saved=torch.load(cp,map_location='cpu',weights_only=False)
 required=('actor','critic','optimizer','training_rng','reward_refinement_state','symmetry_mix_state','motion_course_state')
 active=effective_spec(old,saved['motion_course_state']['stage'],saved['motion_course_state']) if baseline_fault else old
 if (saved.get('update')!=n or saved.get('spec_sha256')!=sha(p)
     or saved.get('critic_privileged_version')!='motion_antiosc_privileged140_v1'
     or saved.get('training_recipe')!=active['student_training'] or not all(k in saved for k in required)
     or saved['motion_course_state']['version']!=VERSION
     or not baseline_fault and saved['motion_course_state']['last_evaluation_update']!=-1):
  raise ValueError('complete same-recipe pre-first-assessment checkpoint required')
 metrics=[json.loads(line) for line in (run/'formal/metrics.jsonl').read_text().splitlines()]
 if [r['update'] for r in metrics]!=list(range(1,n+1)):
  raise ValueError('charged optimizer evidence differs from saved update')
 carried=diagnostic_fault or evaluation_fault or baseline_fault
 evidence_run=Path(previous.get('source_evidence_run',previous.get('parent_run',str(run)))) if carried else run
 smoke=evidence_run/'smoke';source=evidence_run/'source_eval'
 source_declaration=json.loads((source/'source_snapshot/spec.json').read_text()) if carried else old
 if validate_persisted_certificate(smoke,source_declaration)['status']!='PASSED':raise ValueError('real prior smoke certificate required')
 source_terminal=json.loads((source/'terminal_report.json').read_text())
 source_probe=json.loads((source/'robustness_engineering_readback.json').read_text())
 if source_terminal['status']!='COMPLETED' or source_terminal.get('failure') or source_probe.get('status')!='PASSED':
  raise ValueError('completed source engineering evidence required')
 proof=json.loads(probe.read_text())
 if (proof.get('status')!='COMPLETED' or proof.get('variant')!=MATERIAL_BACKEND or proof.get('envs')!=4096 or proof.get('PPO_updates')!=0
     or len(proof.get('checks',[]))<31 or any(row.get('passed') is not True
       or any(row[side].get('all_env_max_diff',float('inf'))>2e-6
              or row[side].get('nonreset_max_change',float('inf'))>2e-6 for side in ('left','right')) for row in proof['checks'])):
  raise ValueError('actual4096 all-row and nonreset material readback proof required')
 prior=old['motion_training'].get('prior_consumption',{});used=prior.get('historical_online_updates',0)+book['used']['smoke']+n
 smoke_used=prior.get('historical_smoke_updates',0)+book['used']['smoke']
 global_formal=old['student_training'].get('formal_resume_offset',0)+n
 if not 0<used<10000 or smoke_used!=5 or used!=smoke_used+global_formal:
  raise ValueError('original inclusive10000 and cumulative smoke/formal identity required')
 if evaluation_fault or baseline_fault:
  evaluation_probe=proof.get('evaluation_probe',{})
  if (global_formal%250 or proof.get('material_writer')!=MATERIAL_WRITER
      or evaluation_probe.get('status')!='COMPLETED' or evaluation_probe.get('envs')!=64
      or evaluation_probe.get('reset_calls',0)<486 or evaluation_probe.get('physics_steps',0)<31520
      or evaluation_probe.get('material_writer')!=MATERIAL_WRITER):
   raise ValueError('actual64 grouped/reset/full-window and4096 articulation material proof required')
 if diagnostic_fault:
  from .motion_parallel_course import checkpoint_diagnostics
  metric=metrics[-1]
  if (metric.get('parent_formal_offset')!=old['student_training']['formal_resume_offset']
      or metric.get('formal_update_including_parent')!=global_formal or global_formal%100):
   raise ValueError('failed diagnostic is not the exact cumulative checkpoint boundary')
  checkpoint_diagnostics(metric,dict(update=n,checkpoint=str(cp),sha256=sha(cp)))
 paths=[p,p.with_name('budget.json'),run/'chain_terminal_report.json',run/'formal/terminal_report.json',run/'formal/progress.json',run/'formal/metrics.jsonl',cp,probe,
        smoke/'wiring_certificate.json',smoke/'terminal_report.json',smoke/'progress.json',smoke/'metrics.jsonl',
        source/'candidate_registry.json',source/'terminal_report.json',source/'robustness_engineering_readback.json']
 for r in source_terminal['evaluations']:
  raw=source/r['path'];d=json.loads(raw.read_text());trace=source/d['trace']['path']
  source_sha=sha(source/'source_snapshot/spec.json') if carried else sha(p)
  if (sha(raw)!=r['sha256'] or sha(trace)!=d['trace']['sha256'] or d['checkpoint_sha256']!=source_declaration['source_sha256']
      or d['spec_sha256']!=source_sha or d['per_env']!=r['per_env']):raise ValueError('ancestral source raw identity differs')
  paths.extend((raw,trace))
 if carried:paths.append(source/'source_snapshot/spec.json')
 if evaluation_fault or baseline_fault:
  for name in ('evaluation_probe','training_probe'):
   ref=proof[name]
   if sha(ref['path'])!=ref['sha256']:raise ValueError('actual material probe identity differs')
   paths.append(Path(ref['path']))
 if baseline_fault:
  registry=json.loads((run/'formal/candidate_registry.json').read_text())
  baseline=next((r for r in registry if r.get('update')==0),None)
  records=baseline.get('evaluations',[]) if baseline else []
  if (not baseline or baseline.get('sha256')!=old['source_sha256'] or baseline.get('checkpoint')!=old['source_checkpoint']
      or baseline.get('complete_matrix') is not True or progress.get('pending_assessment',{}).get('status')!='COMPLETED'):
   raise ValueError('completed parent boundary matrix required')
  full_records=[]
  for record in records:
   raw=run/'formal'/record['path'];row=json.loads(raw.read_text());trace=raw.parent/row['trace']['path']
   compact=[dict(per_env=copy.deepcopy(row['per_env']))];compact_historical_records(compact)
   if (sha(raw)!=record['sha256'] or sha(trace)!=row['trace']['sha256'] or row['checkpoint_sha256']!=old['source_sha256']
       or row['spec_sha256']!=sha(p) or record['per_env'] not in (row['per_env'],compact[0]['per_env']) or row['case']!=record['case']):
    raise ValueError('parent completed matrix identity differs')
   full_records.append({**record,'per_env':row['per_env']})
   paths.extend((raw,trace))
  from types import SimpleNamespace
  origin=torch.load(old['source_checkpoint'],map_location='cpu',weights_only=False)
  replay=SimpleNamespace(motion_course_state=copy.deepcopy(old['motion_robustness']['state']),
   reward_refinement_state=extend_mixture(origin['reward_refinement_state'],commands(old['motion_training']['stage'])),
   symmetry_mix_state=extend_mixture(origin['symmetry_mix_state'],commands(old['motion_training']['stage'])))
  _,expected,_,decision=complete_pending_assessment(old,replay,full_records)
  for key,value in expected.items():
   if key!='DR_episode_counters' and saved['motion_course_state'].get(key)!=value:
    raise ValueError('saved course state differs from actual completed matrix')
  if expected['last_evaluation_update']!=0 or not decision['evidence_valid']:
   raise ValueError('one prior completed pending boundary required')
  if (saved['reward_refinement_state']!=replay.reward_refinement_state or saved['symmetry_mix_state']!=replay.symmetry_mix_state):
   raise ValueError('saved refinement differs from actual completed matrix')
  paths.extend((run/'formal/candidate_registry.json',run/'formal/course_progress.json'))
 version=BASELINE_RESUME_VERSION if baseline_fault else EVALUATION_RESUME_VERSION if evaluation_fault else DIAGNOSTIC_RESUME_VERSION if diagnostic_fault else RESUME_VERSION
 result=dict(version=version,parent_spec=str(p),parent_run=str(run),checkpoint=str(cp),checkpoint_sha256=sha(cp),checkpoint_update=n,
  inputs={str(q):sha(q) for q in paths},historical_online_updates=used,historical_smoke_updates=smoke_used,historical_formal_updates=global_formal,
  original_online_limit=10000,remaining_formal_updates=10000-used,historical_wall_s=prior.get('historical_wall_s',0.)+chain['chain_wall_s'],
  material_probe=str(probe),material_backend=MATERIAL_BACKEND,warmup_repeated=False,physical_state_restored=False)
 if carried:result.update(source_evidence_run=str(evidence_run),formal_update_offset=global_formal)
 if evaluation_fault or baseline_fault:result.update(material_writer=MATERIAL_WRITER,pending_assessment_formal_update=global_formal)
 if baseline_fault:result.update(active_command_stage=saved['motion_course_state']['stage'],course_state_rebased=True)
 return result


def resumed_state(source,offset):
 state=copy.deepcopy(source['motion_course_state'])
 # This recovery is restricted to before the first assessment. Use local update
 # coordinates for course timers and a separate offset for global save cadence.
 state['stage_started_update']-=offset
 if state['last_evaluation_update']>=0:
  state['last_evaluation_update']-=offset
  for accepted in state['accepted_by_stage'].values():accepted['update']-=offset
 return state


def build_resume(parent,checkpoint,*,experiment,spec_path,material_probe,owner_session):
 p=Path(parent).resolve();old=json.loads(p.read_text());c=resume_evidence(p,checkpoint,material_probe)
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False);s=copy.deepcopy(old)
 s.update(experiment=experiment,source_checkpoint=c['checkpoint'],source_sha256=c['checkpoint_sha256'])
 s['motion_training'].update(spec_path=str(Path(spec_path).resolve()),resume=None,wiring_reuse=c,prior_consumption=c)
 state=resumed_state(saved,c['checkpoint_update'])
 if c['version']==BASELINE_RESUME_VERSION:s=effective_spec(s,state['stage'],state)
 s['motion_robustness'].update(state=state,material_backend=MATERIAL_BACKEND,failure_tail_saved_envs=8)
 if c['version'] in (EVALUATION_RESUME_VERSION,BASELINE_RESUME_VERSION):s['motion_robustness']['material_writer']=MATERIAL_WRITER
 s['motion_course']['source_recipe_updates']=saved['recipe_updates']
 remaining=c['remaining_formal_updates'];offset=c.get('formal_update_offset',c['checkpoint_update'])
 s['student_training'].update(source_actor_sha256=c['checkpoint_sha256'],formal_resume_offset=offset,milestones=[v-offset for v in range(250,10000,250) if v>offset])
 historical_bytes=sum(q.stat().st_size for q in Path(c['parent_run']).rglob('*') if q.is_file() and not q.is_symlink())
 maximum=old['artifacts']['storage_max_bytes']-historical_bytes
 s['artifacts'].update(run_name=experiment,storage_max_bytes=maximum)
 s['student_training']['storage_max_bytes']=maximum
 s['motion_timing'].update(parent_spec=str(p),parent_spec_sha256=sha(p),source_checkpoint_sha256=c['checkpoint_sha256'])
 s['budget'].update(source_eval_updates=0,smoke_updates=0,formal_updates=remaining,formal_samples=remaining*4096*128,sample_limit=remaining*4096*128)
 s['motion_course']['configuration'].update(formal_updates_max=remaining,smoke_updates_max=0)
 for row in s['student_evaluation']['cases']:row.update(checkpoint=c['checkpoint'],sha256=c['checkpoint_sha256'])
 s['student_dependencies'].update(c['inputs'])
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':owner_session,'user_scope':f'User authorized ordinary engineering repair/resume of the same Motion10000 DR chain; retain actual{c["historical_online_updates"]} updates, resume full formal{offset} state for remaining{remaining}, sharedGPU0, no training wall limit, current-session notifications; fixed friction lattice within original ranges, no reward/PPO/PD changes'}
 return s


def verify_resume(spec):
 c=spec['motion_training']['wiring_reuse']
 expected=build_resume(c['parent_spec'],spec['source_checkpoint'],experiment=spec['experiment'],spec_path=spec['motion_training']['spec_path'],
  material_probe=c['material_probe'],owner_session=spec['authorization']['hapi_session'])
 for key in ('authorization','resources','notifications'):expected[key]=spec[key]
 if expected!=spec:raise ValueError('material recovery has undeclared behavior/budget drift')
 claim=Path(c['parent_spec']).parent.parent/'.motion_robustness_formal_resume_claims'/(sha(c['parent_spec'])+'.json')
 if claim.exists() and json.loads(claim.read_text())['destination']!=spec['experiment']:raise ValueError('failed chain already claimed')
 validate_recipe(spec['student_training'],spec['source_sha256'])
 return dict(status='VERIFIED',historical_online_updates=c['historical_online_updates'],remaining_formal_updates=c['remaining_formal_updates'],capability_accepted=False)


def complete_pending_assessment(spec,ppo,records):
 """Apply the interrupted boundary assessment before any new optimizer step."""
 c=spec['motion_training']['wiring_reuse'];state=ppo.motion_course_state
 if (c['version'] not in (EVALUATION_RESUME_VERSION,BASELINE_RESUME_VERSION) or c['pending_assessment_formal_update']!=c['formal_update_offset']
     or c['formal_update_offset']%250 or state['last_evaluation_update']>=0):
  raise ValueError('explicit pending first full assessment required')
 from .motion_differential import update_mix
 candidate=dict(checkpoint=spec['source_checkpoint'],sha256=spec['source_sha256'],update=0,stage=state['stage'],
  per_env=[e for r in records for e in r['per_env']],evaluations=records,
  complete_matrix=len(records)==3*len(commands(state['stage'])),evaluated=True,
  formal_update_including_parent=c['formal_update_offset'])
 ppo.symmetry_mix_state=update_mix(ppo.symmetry_mix_state,refinement_records(records,state['stage']))
 archive_mixtures(state,ppo)
 state,decision=advance(state,candidate)
 if spec['reward'].get('refinement'):
  from .motion_refinement import advance as refine
  ppo.reward_refinement_state,_=refine(ppo.reward_refinement_state,refinement_records(records,candidate['stage']),spec['reward']['refinement'])
 s=effective_spec(spec,state['stage'],state);values=s['curriculum']['stages'][0]['commands']
 for attr,name in [('reward_refinement_state','mixture_archive'),('symmetry_mix_state','symmetry_archive')]:
  setattr(ppo,attr,restored_mixture(state,name,getattr(ppo,attr),values))
 ppo.motion_course_state=state
 return s,state,candidate,decision


def failure_tail_arrays(tail,maximum_envs,preferred_ids=()):
 """Retain bounded geometry/control histories and every env's final state."""
 last=tail[-1]['state'];n=len(last)
 faults=torch.where(last[:,[3,4,30]].bool().any(-1))[0].cpu().tolist()
 ids=list(dict.fromkeys(list(preferred_ids)+faults+list(range(n))))[:maximum_envs]
 selected=torch.tensor(ids,device=last.device)
 arrays={k:torch.stack([row[k][selected] for row in tail]).cpu().numpy() for k in tail[0]}
 arrays.update(env_ids=selected.cpu().numpy(),all_env_final_state=last.cpu().numpy())
 return arrays


def reward_snapshot(spec):
 result=base.reward_snapshot(spec);result['course']=configuration();result['DR_changes_reward']=False
 result['note']='same source reward; robustness is physical exposure and explicit course, not hidden control'
 return result


def diagnostic_storage(payload,arrays):
 """All8 scored results remain authoritative JSON; keep one full diagnostic trajectory.

 This profile deliberately does NOT claim full8-env trajectory replay. No scoring
 input is changed before assessment. Select a failing environment first, never
 cherry-pick a successful trace to hide other failures.
 """
 count=len(payload['per_env'])
 if count!=8:raise ValueError('robust full assessment must score8 actual environments')
 index=next((i for i,e in enumerate(payload['per_env']) if e.get('accepted') is not True),0)
 result={k:(v[:,index:index+1].copy() if getattr(v,'ndim',0)>=2 and v.shape[1]==count else v) for k,v in arrays.items()}
 payload['diagnostic_storage']=dict(version='eight_env_metrics_one_env_trace_v1',scored_envs=8,stored_trace_envs=1,
  original_scored_env_index=index,all8_per_env_results_retained=True,full8_trajectory_replay_available=False,
  selection='first failed environment, otherwise0; after all8 scoring only',trace_sample_hz=100)
 return result


def compact_historical_records(records):
 """Bound mutable registries by retaining raw-file references and gate essentials."""
 for r in records:
  r['per_env']=[{k:copy.deepcopy(e[k]) for k in ('env','accepted','survival_passed','checks','failed_checks','tracking') if k in e} for e in r['per_env']]


def engineering_probe(spec,actor,output,stop):
 """Bounded PPO0 backend roundtrip, not a standing capability certificate."""
 from .evaluator import environment
 from native_stand.command_v6.completion import write
 env=environment(spec,'MotionZeroStudent',envs=64)
 results=[]
 try:
  env.curriculum_training=False;env.forced_target=(0.,0.)
  env.select_initial_family('motion_native_normal','train')
  for level in range(len(LEVELS)):
   if stop():raise TimeoutError('engineering probe stopped')
   env.motion_dr.configure(dict(level=level,group=2,seed=20261023))
   obs,_=env.reset()
   with torch.inference_mode():
    for _ in range(20):
     obs,reward,*_=env.step(actor(obs))
     if obs['student_actor'].shape!=(64,31) or obs['student_critic'].shape!=(64,140) or not torch.isfinite(reward).all() or env.recovery_snapshot[:,4].bool().any():raise RuntimeError('DR engineering interface/reward/runtime fault')
   results.append(env.motion_dr.summary())
  ids=torch.tensor([0,1,2,3],device=env.device)
  env.motion_dr.impulse(ids,0)
  requested=env.motion_dr.last_push
  if not requested or not any(abs(v)>0 for row in requested['delta_world_XY_mps'] for v in row):raise RuntimeError('standing push was not requested')
  write(output/'robustness_engineering_readback.json',dict(status='PASSED',physics_steps=20*10*len(LEVELS),PPO_updates=0,levels=results,
   explicit_push_readback=requested,scope='real64env actual mass/inertia/friction writes and timestamp paths; short windows only, not ability acceptance'))
 finally:env.close()
