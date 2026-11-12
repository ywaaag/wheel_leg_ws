"""4750 standing-first timing DR declaration; explicit new run and budget only."""
import copy,json
from pathlib import Path
from .data import sha
from . import motion_timing as timing

def build(parent,checkpoint,*,experiment,spec_path):
 from .model import load
 from .motion_training import declaration
 from .motion_startup import smoke_profile
 from .performance import new_motion_defaults
 p=Path(parent).resolve();cp=Path(checkpoint).resolve();old=json.loads(p.read_text())
 if old.get('motion_timing'):return build_resume(p,cp,experiment=experiment,spec_path=spec_path)
 _,source=load(cp,'MotionZeroStudent')
 if source['update']!=4750 or source['spec_sha256']!=sha(p) or source['critic_privileged_version']!='motion_symmetry_privileged85_v1':raise ValueError('exact4750/source spec required')
 s=copy.deepcopy(old)
 for name in ('motion_parallel','motion_parallel_assessment','motion_startup','checkpoint_quick_evaluation',
              'motion_confirmation','motion_large_continuation','motion_differential_retry'):
  s.pop(name,None)
 s.update(experiment=experiment,source_checkpoint=str(cp),source_sha256=sha(cp),
          network_architecture=timing.network(),student_interface=timing.contract(),
          observation={'frame_dim':22,'actor_dim':31,'critic_dim':110})
 recipe=declaration(sha(cp),0,reward_version=s['reward']['version'])
 recipe.update(critic_privileged_version=timing.CRITIC,network_architecture=timing.network(),
  course_version=timing.VERSION,course_stage=0,timing_configuration=timing.configuration(),
  critic='source85_zero_pad_to110',optimizer='source_Adam_zero_new_columns',
  milestones=list(range(250,10000,250)),checkpoint_every_updates=100,storage_max_bytes=8*2**30,
  smoke_wiring=smoke_profile(),selection_scope='standing034, horizontal/symmetric/contact/smooth under timing DR; no new speed task')
 s['student_training']=recipe
 s['motion_timing']={'version':timing.VERSION,'configuration':timing.configuration(),'stage':0,
  'parent_spec':str(p),'parent_spec_sha256':sha(p),'source_checkpoint_sha256':sha(cp)}
 s['motion_training'].update(stage=0,spec_path=str(Path(spec_path).resolve()),resume=None)
 for name in ('successor','wiring_reuse','partial_smoke_recovery','prior_consumption'):s['motion_training'].pop(name,None)
 s['motion_course']={'source_recipe_updates':source['recipe_updates'],'configuration':{
  'version':timing.VERSION,'formal_updates_max':9995,'smoke_updates_max':5,'stage_updates_max':9995,'evaluation_every':250}}
 s['motion_assessment']=timing.protocol()
 t=s['student_evaluation']['cases'][0]
 s['student_evaluation']={'version':timing.protocol()['version'],'cases':[{**t,'checkpoint':str(cp),'sha256':sha(cp)}], 'max_output_bytes':128*2**20}
 s['student_evaluation']['cases']=timing.cases(s)
 s['curriculum']['stages']=[timing.stage_config()];s['curriculum']['active_stage']=0
 s['evaluation'].update(full_every_updates=250,checkpoint_every_updates=100,protocol_identity=timing.protocol()['version'])
 s['performance']=new_motion_defaults(old['performance'],grouped_evaluation=False)
 s['performance'].update(motion_hotpath=True,motion_runtime_reduction=True,contact_history_disabled=True,
                         evaluation_group_commands=False,evaluation_reuse_environment=True)
 s['budget'].update(smoke_updates=5,formal_updates=9995,formal_samples=9995*4096*128,
  sample_limit=10000*4096*128,formal_wall_s=None,chain_wall_s=None,source_eval_wall_s=3600,smoke_wall_s=1800,evaluation_wall_s=3600)
 s['artifacts'].update(run_name=experiment,storage_max_bytes=8*2**30,terminal_reserve_bytes=512*2**20)
 s['motion_startup']={'version':'motion31_minimal_startup_v1','source_policy':'reuse_exact_source_evidence_then_evaluate_only_missing_endpoints',
  'smoke_policy':'motion31_wiring5_no_capability_eval_v2','references':[],'missing_commands':[[0.,0.,.34]]*3,
  'source_evaluation_is_not_wiring':True,'no_capability_prerequisite_for_smoke':True}
 # Source-reuse keys must distinguish timing conditions; this experiment does not reuse raw evals.
 s['motion_symmetry']['stage_scope']='zero standing only, timing nominal/previous/current curriculum'
 s['student_dependencies']={str(p):sha(p),str(cp):sha(cp),s['asset_path']:s['asset_sha256']}
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':None,
  'user_scope':'User2026-10-21 approves4750 big Teacher standing034 timing randomization10000 inclusive updates, unlimited training wall, GPU0 shared, timing privileged Critic, speed optimizations'}
 s['resources']={'mode':'shared','sharing_authorized':True,'authorization_note':'User approves new4750 standing timing DR; GPU0 shared, no preemption'}
 s['notifications']={'enabled':False}
 return s

def verify(spec):
 if spec.get('motion_standing_delay'):
  from .motion_standing_delay import verify
  return verify(spec)
 if spec.get('mcu_execution'):
  from .motion_mcu_course import verify as verify_mcu
  return verify_mcu(spec)
 if spec.get('motion_decoupled',{}).get('version')=='motion31_persistent_frontiers_v1':
  from .motion_persistent import verify as verify_persistent
  return verify_persistent(spec)
 if spec.get('motion_decoupled'):
  from .motion_decoupled import verify as verify_decoupled
  if spec['motion_decoupled'].get('version')=='motion31_fixed_height_motion_v2':
   from .motion_fixed_height import verify as verify_decoupled
  return verify_decoupled(spec)
 if spec.get('motion_robustness'):
  from .motion_robustness import verify as verify_robustness
  return verify_robustness(spec)
 if spec.get('motion_reacquisition'):
  from .motion_reacquisition import verify as verify_reacquisition
  return verify_reacquisition(spec)
 if spec.get('motion_timing',{}).get('version')=='motion31_standing_antiosc_armature_v4':
  from .motion_antioscillation import verify as verify_anti
  return verify_anti(spec)
 from .motion_sensor_transfer import VERSION as SENSOR,verify as verify_sensor
 if spec.get("motion_timing",{}).get("version")==SENSOR:return verify_sensor(spec)
 from .motion_fixed_timing import VERSION as FIXED,verify as verify_fixed
 if spec.get('motion_timing',{}).get('version')==FIXED:return verify_fixed(spec)
 if spec.get('motion_training',{}).get('wiring_reuse',{}).get('version')==RESUME_VERSION:return verify_resume(spec)
 c=spec['motion_timing']
 expected=build(c['parent_spec'],spec['source_checkpoint'],experiment=spec['experiment'],spec_path=spec['motion_training']['spec_path'])
 for k in ('authorization','resources','notifications'):expected[k]=spec[k]
 if spec.get('motion_timing_recovery'):
  expected['motion_timing_recovery']=recovery_evidence(spec['motion_timing_recovery']['parent_spec'])
 if spec!=expected:raise ValueError('timing experiment has undeclared drift')
 timing.validate_recipe(spec['student_training'],spec['source_sha256'])
 return True

def recovery_evidence(parent_spec):
 p=Path(parent_spec).resolve();old=json.loads(p.read_text())
 root=p.parents[3];run=root/'runs'/old['experiment'];source=run/'source_eval'
 book=json.loads(p.with_name('budget.json').read_text());chain=json.loads((run/'chain_terminal_report.json').read_text())
 if old.get('motion_timing_recovery'):
  if old['motion_timing_recovery']['version']!='motion_timing_source_settlement_recovery_v1':
   raise ValueError('nested timing recovery requires new explicit authorization design')
  prior=recovery_evidence(old['motion_timing_recovery']['parent_spec'])
  smoke=run/'smoke/terminal_report.json';failed=json.loads(smoke.read_text())
  if (chain['status']!='FAILED' or failed['status']!='FAILED' or failed['actual_updates']!=0
      or any(book['used'].get(k,0) for k in ('smoke','formal')) or (run/'formal').exists()
      or any(a['status']=='RUNNING' for a in book['attempts'].values())
      or 'motion_diagnostic_cohort' not in str(failed.get('failure'))):
   raise ValueError('diagnostic recovery requires settled pre-optimizer failure and zero updates')
  if json.loads((source/'terminal_report.json').read_text())['status']!='COMPLETED':
   raise ValueError('reused source phase incomplete')
  paths=[p,p.with_name('budget.json'),run/'chain_terminal_report.json',smoke,
         source/'terminal_report.json',source/'source_reuse.json']
  result=copy.deepcopy(prior)
  result.update(version='motion_timing_diagnostic_metadata_recovery_v1',parent_spec=str(p),
   failed_parent_run=str(run),historical_chain_wall_s=prior['historical_chain_wall_s']+chain['chain_wall_s'],
   new_runtime_authorization='User explicitly approves restoring training after diagnostic metadata failure')
  result['inputs'].update({str(q):sha(q) for q in paths})
  return result
 terminal=json.loads((source/'terminal_report.json').read_text())
 if chain['status']!='FAILED' or terminal['status']!='COMPLETED' or terminal.get('failure'):
  raise ValueError('timing settlement recovery requires failed chain with completed source')
 if any(book['used'].get(k,0) for k in ('smoke','formal')) or (run/'smoke').exists() or (run/'formal').exists():
  raise ValueError('timing source settlement recovery cannot conceal PPO consumption')
 log=run/'source_eval.log'
 if "KeyError: 'source_eval_wall_used_s'" not in log.read_text(errors='replace'):
  raise ValueError('expected source settlement failure not found')
 paths=[p,p.with_name('budget.json'),run/'chain_terminal_report.json',source/'terminal_report.json',
        source/'progress.json',source/'candidate_registry.json',log]
 for case,rec in zip(old['student_evaluation']['cases'],terminal['evaluations']):
  raw_path=source/rec['path'];raw=json.loads(raw_path.read_text());trace=source/raw['trace']['path']
  if sha(raw_path)!=rec['sha256'] or raw['case']!=case or raw['checkpoint_sha256']!=old['source_sha256']:
   raise ValueError('reused timing source identity differs')
  if raw.get('model_unchanged') is not True or len(raw.get('per_env',[]))!=8 or sha(trace)!=raw['trace']['sha256']:
   raise ValueError('reused timing source trace/env evidence incomplete')
  paths.extend([raw_path,trace])
 if len(terminal['evaluations'])!=3:raise ValueError('three timing conditions required')
 return {'version':'motion_timing_source_settlement_recovery_v1','parent_spec':str(p),'parent_run':str(run),
  'inputs':{str(q):sha(q) for q in paths},'historical_online_updates':0,'original_online_limit':10000,
  'historical_chain_wall_s':chain['chain_wall_s'],'historical_source_wall_s':terminal['wall_s_before_cleanup'],
  'old_ledger_running_is_stale':'chain FAILED after completed child; missing accumulator caused finalize transaction rollback',
  'source_reuse':'raw evaluations readonly; new source phase only validates references; no physics',
  'new_runtime_authorization':'user explicitly approved restore10000 unlimited training wall'}

RESUME_VERSION='motion_timing_posteval_wiring_resume_v1'

def resume_evidence(parent,checkpoint):
 """Settled known Python wiring failure only; preserve all charged optimizer updates."""
 from .model import load
 from .motion_parallel_assessment import safety_protocol
 from native_stand.command_v6.wiring import validate_persisted_certificate
 p=Path(parent).resolve();old=json.loads(p.read_text());run=p.parents[3]/'runs'/old['experiment']
 book=json.loads(p.with_name('budget.json').read_text());terminal=json.loads((run/'formal/terminal_report.json').read_text())
 chain=json.loads((run/'chain_terminal_report.json').read_text());_,data=load(checkpoint,'MotionZeroStudent')
 if not old.get('motion_timing') or chain['status']!='FAILED' or any(a['status']=='RUNNING' for a in book['attempts'].values()):raise ValueError('settled timing failure required')
 failure=terminal.get('failure',{})
 if failure.get('exception')!="IndexError('list index out of range')" or 'motion_symmetry.py' not in failure.get('traceback',''):raise ValueError('unsupported failure: inspect before resuming')
 n=book['used']['formal'];prior=old.get('motion_training',{}).get('wiring_reuse',{}).get('historical_online_updates',0)
 used=prior+book['used']['smoke']+n
 if n!=terminal['actual_updates'] or data['update']!=n or data['spec_sha256']!=sha(p) or data['critic_privileged_version']!=timing.CRITIC:raise ValueError('exact completed optimizer checkpoint required')
 if Path(checkpoint).resolve()!=run/'formal'/f'model_update_{n}.pt':raise ValueError('checkpoint outside exact failed phase')
 if used>=10000:raise ValueError('no remaining authorized updates')
 if validate_persisted_certificate(run/'smoke',old)['status']!='PASSED':raise ValueError('original smoke certificate required')
 # This recovery resumes post-evaluation processing, never replays an optimizer update.
 records=json.loads((run/'formal/progress.json').read_text())['evaluations'][-3:]
 if len(timing.refinement_records(records,data['motion_course_state']['stage'])[0]['per_env'])!=8:raise ValueError('complete timing matrix required')
 paths=[p,p.with_name('budget.json'),run/'chain_terminal_report.json',run/'formal/terminal_report.json',run/'formal/progress.json',
        run/'smoke/wiring_certificate.json',run/'smoke/terminal_report.json',run/'source_eval/candidate_registry.json',
        Path(checkpoint).resolve()]
 for record in records:
  q=Path(record['path']);q=q if q.is_absolute() else run/'formal'/q
  raw=json.loads(q.read_text());trace=q.parent/raw['trace']['path']
  if sha(q)!=record['sha256'] or raw['checkpoint_sha256']!=sha(checkpoint) or raw['protocol']!=safety_protocol(old) or sha(trace)!=raw['trace']['sha256']:raise ValueError('resume evaluation identity differs')
  paths.extend([q,trace])
 history=old.get('motion_timing_recovery',{}).get('historical_chain_wall_s',0.)
 return {'version':RESUME_VERSION,'parent_spec':str(p),'parent_run':str(run),'inputs':{str(q):sha(q) for q in paths},
  'historical_smoke_updates':book['used']['smoke'],'historical_formal_updates':n,'historical_online_updates':used,
  'original_online_limit':10000,'historical_wall_s':history+chain['chain_wall_s'],
  'checkpoint':str(Path(checkpoint).resolve()),'checkpoint_update':n,
  'pending_evaluations':[{**r,'path':str((run/'formal'/r['path']).resolve())} for r in records],
  'remaining_formal_updates':10000-used,'physical_state_restored':False}

def build_resume(parent,checkpoint,*,experiment,spec_path):
 p=Path(parent).resolve();old=json.loads(p.read_text());c=resume_evidence(p,checkpoint)
 s=copy.deepcopy(old);s.pop('motion_timing_recovery',None)
 s.update(experiment=experiment,source_checkpoint=c['checkpoint'],source_sha256=sha(checkpoint))
 s['motion_training'].update(spec_path=str(Path(spec_path).resolve()),wiring_reuse=c,resume=None)
 s['motion_timing']['parent_spec']=str(p);s['motion_timing']['parent_spec_sha256']=sha(p)
 s['motion_timing']['source_checkpoint_sha256']=sha(checkpoint)
 import torch
 data=torch.load(checkpoint,map_location='cpu',weights_only=False)
 s['motion_course']['source_recipe_updates']=data['recipe_updates']
 s['student_training']['source_actor_sha256']=sha(checkpoint)
 s['student_training']['critic']='source85_zero_pad_to110'  # expand_source is identity for declared110D checkpoints.
 for row in s['student_evaluation']['cases']:row.update(checkpoint=c['checkpoint'],sha256=sha(checkpoint))
 remaining=c['remaining_formal_updates']
 s['budget'].update(smoke_updates=0,formal_updates=remaining,formal_samples=remaining*4096*128,sample_limit=remaining*4096*128)
 s['artifacts']['run_name']=experiment
 s['student_dependencies'].update({str(p):sha(p),str(checkpoint):sha(checkpoint)})
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'User explicitly approves posteval wiring fix and full-state resume from250 within original10000, automatic same-recipe engineering retries, GPU0 shared, unlimited training wall'}
 s['notifications']={'enabled':True}
 return s

def verify_resume(spec):
 c=spec['motion_training']['wiring_reuse']
 parent=Path(c['parent_spec']);claim=parent.parent.parent/'.motion_timing_resume_claims'/(sha(parent)+'.json')
 if claim.exists() and json.loads(claim.read_text())['destination']!=spec['experiment']:raise ValueError('timing parent already claimed by another resume')
 expected=build_resume(c['parent_spec'],spec['source_checkpoint'],experiment=spec['experiment'],spec_path=spec['motion_training']['spec_path'])
 for k in ('authorization','resources','notifications'):expected[k]=spec[k]
 if expected!=spec:raise ValueError('timing resume has undeclared recipe or budget drift')
 return {'status':'VERIFIED','historical_online_updates':c['historical_online_updates'],
         'remaining_formal_updates':c['remaining_formal_updates'],'capability_accepted':False}

def restored_posteval_state(source,certificate,recipe):
 """Rebuild the unfinished CPU bookkeeping from frozen raw evaluations exactly once."""
 from .motion_symmetry import update_mix
 from .motion_refinement import advance as refine
 state=copy.deepcopy(source['motion_course_state']);records=certificate['pending_evaluations']
 candidate={'checkpoint':certificate['checkpoint'],'sha256':sha(certificate['checkpoint']),
            'update':source['update'],'stage':state['stage'],'complete_matrix':True,'evaluations':records}
 mix=update_mix(source['symmetry_mix_state'],timing.refinement_records(records,state['stage']))
 refinement=copy.deepcopy(source['reward_refinement_state'])
 if recipe.get('refinement'):refinement,_=refine(refinement,timing.refinement_records(records,state['stage']),recipe['refinement'])
 state,decision=timing.advance(state,candidate)
 # Child updates are local; keep confirmation, translate stage age and last evaluation.
 state['stage_started_update']-=source['update'];state['last_evaluation_update']=0
 return state,mix,refinement,decision
