"""Exact-source CPU proposal; no budget reservation, runtime or authorization."""
import copy,json
from pathlib import Path
from .data import sha
from . import motion_parallel_course as course
SOURCE='b6720daf80687fe8a90f96e3a3fc7ba8e9244f5cec19ac44b69bc527d00236d1'

def build(parent,checkpoint,*,experiment,spec_path,formal_updates=4995):
 from .model import load
 from .motion_tracking_reward import configuration
 from .motion_parallel_assessment import safety_protocol,protocol
 from .performance import new_motion_defaults
 from .motion_symmetry import network
 p=Path(parent).resolve();cp=Path(checkpoint).resolve();before=json.loads(p.read_text());_,saved=load(cp,'MotionZeroStudent')
 if sha(cp)!=SOURCE or saved['spec_sha256']!=sha(p) or saved.get('network_architecture')!=network() or saved['critic_privileged_version']!='motion_symmetry_privileged85_v1':raise ValueError('exact u3250/full Motion85 source required')
 if type(formal_updates) is not int or not 250<=formal_updates<=20000:raise ValueError('finite proposed formal updates250..20000; not authorization')
 term=cp.parent.parent/'chain_terminal_report.json';t=json.loads(term.read_text())
 if t.get('status')!='STOPPED_BOUNDED' or t.get('failure') is not None:raise ValueError('source chain must be normally settled')
 if any(k not in saved for k in ('actor','critic','optimizer','training_rng','reward_refinement_state')):raise ValueError('full source checkpoint missing')
 if {x['lr'] for x in saved['optimizer']['param_groups']}!={3e-5}:raise ValueError('source LR differs')
 registry=cp.parent/'candidate_registry.json';candidates=json.loads(registry.read_text());matched=[c for c in candidates if c.get('sha256')==SOURCE and c.get('update')==3250]
 if len(matched)!=1 or not matched[0].get('complete_matrix'):raise ValueError('u3250 complete matrix missing')
 expected_commands={(0.,0.,.34),(.5,0.,.34),(1.,0.,.34),(1.5,0.,.34),(2.,0.,.34),(0.,.5,.34),(0.,-.5,.34)}
 if {tuple(r['case']['command']) for r in matched[0]['evaluations']}!=expected_commands or len(matched[0]['evaluations'])!=7:raise ValueError('source seven-command identity differs')
 evidence={str(registry):sha(registry)}
 for rec in matched[0]['evaluations']:
  raw=cp.parent/rec['path'];d=json.loads(raw.read_text())
  if sha(raw)!=rec['sha256'] or d.get('checkpoint_sha256')!=SOURCE or d.get('model_unchanged') is not True or d.get('scored_envs')!=8 or len(d.get('per_env',[]))!=8 or {e.get('env') for e in d['per_env']}!=set(range(8)):raise ValueError('source full capability evidence differs')
  evidence[str(raw)]=sha(raw)
 s=copy.deepcopy(before)
 for field in ('motion_differential_retry',):s.pop(field,None)
 s.update(experiment=experiment,source_checkpoint=str(cp),source_sha256=SOURCE,reward=configuration(course.REWARD))
 s['motion_parallel']={'version':course.VERSION,'parent_spec':str(p),'parent_spec_sha256':sha(p),'source_sha256':SOURCE,'state':course.initial_state(),'formal_updates_proposed':formal_updates,'scope':'parallel lanes, not combined turning/jump/recovery','checkpoint_diagnostics':course.checkpoint_diagnostics_configuration()}
 s['motion_parallel_assessment']=protocol(safety_protocol(before))
 r=s['student_training'];r.update(critic='source_same_semantic85',optimizer='source_Adam_same85',source_actor_sha256=SOURCE,reward=s['reward'],stage=0,course_version=course.VERSION,course_stage=0,parallel_state=course.initial_state(),milestones=list(range(250,formal_updates+1,250)),storage_max_bytes=12*2**30)
 s['motion_training'].update(stage=0,spec_path=str(Path(spec_path).resolve()),resume={'parent_spec':str(p),'parent_spec_sha256':sha(p),'formal_dir':str(cp.parent),'parent_recipe_updates':saved['recipe_updates']})
 for k in ('wiring_reuse','partial_smoke_recovery','successor'):s['motion_training'].pop(k,None)
 s['motion_course']['configuration']={**course.configuration(),'formal_updates_max':formal_updates,'stage_updates_max':formal_updates}
 s['motion_course']['source_recipe_updates']=saved['recipe_updates']
 s['student_evaluation']['cases']=course.cases(s);s['student_evaluation']['version']=s['motion_parallel_assessment']['version']
 s['evaluation']['protocol_identity']=s['motion_parallel_assessment']['version']
 r['selection_scope']='parallel straight/backward/in-place yaw coarse-speed plus physical/absolute posture/symmetry; old retention intact; parking diagnostic only'
 for row in s['student_evaluation']['cases']:row.update(checkpoint=str(cp),sha256=SOURCE)
 s['curriculum']['stages']=[course.stage_config()];s['curriculum']['active_stage']=0
 s['performance']=new_motion_defaults(before['performance'],grouped_evaluation=True);s['performance'].update(evaluation_group_max_commands=4,trace_buffer_max_bytes=512*2**20)
 s['budget'].update(smoke_updates=5,formal_updates=formal_updates,formal_samples=formal_updates*4096*128,sample_limit=(formal_updates+5)*4096*128,formal_wall_s=None,chain_wall_s=None,source_eval_wall_s=7200,evaluation_wall_s=3600)
 s['artifacts'].update(run_name=experiment,storage_max_bytes=12*2**30)
 s['motion_startup'].update(references=[],missing_commands=[r['command'] for r in s['student_evaluation']['cases']])
 s['student_dependencies'].update({str(p):sha(p),str(cp):SOURCE,str(term):sha(term),**evidence})
 s['experiment_variable']='Parallel forward/backward/in-place yaw lanes from u3250; explicit v10 tracking widths; v9 quality/physics preserved; behavior observe-only'
 s['preparation_note']='New CPU proposal; no simulation or online authorization; do not reuse predecessor balance'
 s['authorization_note']='New runtime approval and independent ledger required; proposed budget is not granted'
 s['motion_symmetry']['stage_scope']='parallel independent lanes; all learned endpoints retained; no combined-turn/jump/recovery'
 s['motion_symmetry']['budget']='proposed only; smoke5 plus formal proposal; no inherited balance'
 s['motion_differential'].update(source_checkpoint_sha256=SOURCE,online_limit=formal_updates+5,restore='u3250 same85 Actor/Critic/Adam/std/RNG; new environment; formal critic warmup5 counted',behavior_regression='record only; no behavior-triggered chain stop')
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'CPU draft only; proposed limits are not approved online budget'}
 s['resources']={'mode':'shared','sharing_authorized':False,'authorization_note':'GPU0 sharing requires fresh explicit runtime approval for this draft'}
 s['notifications']={**s.get('notifications',{}),'enabled':False}
 return s

def verify(s):
 c=s['motion_parallel'];p=Path(c['parent_spec'])
 if c['version']!=course.VERSION or sha(p)!=c['parent_spec_sha256']:raise ValueError('parallel parent identity differs')
 expected=build(p,s['source_checkpoint'],experiment=s['experiment'],spec_path=s['motion_training']['spec_path'],formal_updates=c['formal_updates_proposed'])
 # Authorization is checked by public preflight/reserve, not inferred from preparation.
 expected['authorization']=s['authorization'];expected['resources']=s['resources'];expected['notifications']=s['notifications']
 from .motion_quick_evaluation import warm_configuration
 if s.get('checkpoint_quick_evaluation')==warm_configuration():
  expected['checkpoint_quick_evaluation']=warm_configuration()
 for flag in ('motion_hotpath','contact_history_disabled','motion_runtime_reduction'):
  if flag in s.get('performance',{}):
   if type(s['performance'][flag]) is not bool:raise ValueError('explicit boolean Motion hotpath required')
   expected['performance'][flag]=s['performance'][flag]
  elif flag in ('motion_hotpath','motion_runtime_reduction'):
   expected['performance'].pop(flag,None) # historical explicit spec, not a runtime upgrade
 if s!=expected:raise ValueError('parallel recipe/slots/limits/physical drift')
 from .training_recipe import validate_instance
 validate_instance(s['student_training'],s['source_sha256'])
 return True
