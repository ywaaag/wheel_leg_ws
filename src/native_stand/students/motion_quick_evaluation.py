"""Owner-triggered checkpoint screening in a bounded isolated Isaac child; no watcher."""
import copy,json,os,subprocess,sys,time,traceback,atexit
from pathlib import Path
VERSION='motion31_checkpoint_quick_evaluation_v1'
WARM_VERSION='motion31_checkpoint_quick_warm_app_v2'

def configuration():
 return {'version':VERSION,'every_updates':100,'scored_envs_per_command':8,'maximum_commands_per_batch':7,
  'physical_envs':64,'window':'unchanged full parent protocol','wall_limit_s':900,'trace_buffer_max_bytes':512*2**20,
  'output_max_bytes':128*2**20,'selection_eligible':False,'parent_full_evaluation_every':250,
  'isolation':'owned subprocess; no parent physics/reset/RNG/optimizer changes','on_error':'record screening failure; full gates/runtime parent protection unchanged'}

def warm_configuration():
 cfg=configuration();cfg.update(version=WARM_VERSION,execution='existing_AppLauncher_new_evaluation_environment',
  representative_cases='zero,current_max_forward,current_max_backward,current_max_positive_yaw,current_max_negative_yaw,old_forward_0p5',
  scored_envs_per_command=1,physical_envs=1,maximum_commands_per_batch=1,
  isolation='completed PPO update; close training env; restore RNG; rebuild/reset training; no GAE across reset',
  reuse_complete_evaluation=True,on_error='stop chain with evidence; never resume stale training observations')
 return cfg

def validate(spec):
 cfg=spec.get('checkpoint_quick_evaluation')
 if cfg is None:return None
 if cfg not in (configuration(),warm_configuration()):raise ValueError('quick evaluation explicit configuration differs')
 if spec.get('student_training',{}).get('version')!='student_motion31_training_v1':raise ValueError('quick evaluation requires Motion31')
 if spec['student_training'].get('checkpoint_every_updates')!=100 or spec['evaluation']['full_every_updates']!=250:raise ValueError('quick evaluation requires save100/full250')
 from .grouped_evaluation import plan
 if cfg['version']==WARM_VERSION:
  from .motion_evaluation_review import representative_cases
  representative_cases(spec['student_evaluation']['cases'])
  return cfg
 groups=plan(spec['student_evaluation']['cases'],maximum_commands=7)
 if not spec.get('motion_parallel') and (len(groups)!=1 or len(groups[0])!=7):raise ValueError('quick evaluation requires one compatible seven-command matrix')
 if spec.get('motion_parallel'):
  if sum(r['command'][:2]==[0.,.5] for r in spec['student_evaluation']['cases'])!=1:raise ValueError('parallel quick requires retained yaw+.5')
  return cfg
 import math
 peak=math.ceil(max(r['seconds'] for r in groups[0])/.01)*56*640*4
 if peak>cfg['trace_buffer_max_bytes']:raise ValueError('quick evaluation declared trace memory too small')
 return cfg

def due(spec,phase,update):
 cfg=validate(spec)
 return bool(cfg and phase=='formal' and update>0 and update%cfg['every_updates']==0)

def parent_deadline(spec,manifest,deadline):
 """Null chain wall removes only the parent wall cap, not child timeouts."""
 chain_wall=spec['budget'].get('chain_wall_s')
 chain_deadline=float('inf') if chain_wall is None else manifest['submitted_unix_s']+chain_wall
 return min(deadline,manifest.get('global_deadline_unix_s') or float('inf'),chain_deadline)


def run(output,spec,checkpoint,*,update,parent_spec,attempt_id,deadline):
 """Called exactly once by trainer after atomic save. Child consumes no PPO allowance."""
 from .data import sha
 cfg=validate(spec);folder=Path(output)/f'quick_update_{update}'
 if cfg['version']==WARM_VERSION:raise ValueError('warm-App v2 must run at Motion runner completed-update boundary')
 folder.mkdir(exist_ok=False)
 started=time.monotonic()
 manifest_path=Path(output).parent/'manifest.json'
 if manifest_path.exists():
  manifest=json.loads(manifest_path.read_text())
  deadline=parent_deadline(spec,manifest,deadline)
 remaining=min(cfg['wall_limit_s'],deadline-time.time()-90)
 request={'version':VERSION,'parent_spec':str(Path(parent_spec).resolve()),'parent_spec_sha256':sha(parent_spec),
  'parent_output':str(Path(output).resolve()),'attempt_id':attempt_id,'checkpoint':str(Path(checkpoint).resolve()),
  'checkpoint_sha256':sha(checkpoint),'update':update,'deadline_unix_s':time.time()+max(0,remaining),'output':str(folder.resolve())}
 with (folder/'request.json').open('x') as f:json.dump(request,f,indent=2)
 result={'status':'NOT_RUN','selection_eligible':False,'accepted_candidate':None,'checkpoint_sha256':request['checkpoint_sha256'],'update':update,'output':str(folder)}
 if remaining<60:result['reason']='parent wall budget reserve insufficient'
 else:
  command=[sys.executable,'-m',__name__,'--request',str((folder/'request.json').resolve())]
  try:
   with (folder/'runtime.log').open('x') as log:
    # timeout owns child termination; parent does not poll logs or checkpoints.
    child=subprocess.run(['timeout','--signal=TERM','--kill-after=15s',str(max(1,remaining-20))+'s',*command],stdout=log,stderr=subprocess.STDOUT,timeout=remaining,env={**os.environ,'PYTHONPATH':spec['execution_source_root']})
   result.update(status='FAILED',exit_code=child.returncode)
   summary_path=folder/'summary.json'
   if child.returncode==0 and summary_path.is_file():
    summary=json.loads(summary_path.read_text())
    if summary.get('status')=='COMPLETED' and summary.get('checkpoint_sha256')==request['checkpoint_sha256'] and summary.get('update')==update and len(summary.get('commands',[]))==summary.get('condition_count',7) and all(len(r.get('per_env',[]))==summary.get('scored_envs_per_command',8) for r in summary['commands']):
     result.update(status='COMPLETED',summary_sha256=sha(summary_path))
    else:result['reason']='child summary incomplete or identity mismatch'
   elif child.returncode==0:result['reason']='child exited0 without complete summary'
  except (OSError,ValueError,subprocess.TimeoutExpired) as error:result.update(status='FAILED',reason=repr(error))
 if result['status']=='FAILED':
  for name in ('worker_failure.json','worker_exit_failure.json','worker_shutdown_failure.json'):
   failure_path=folder/name
   if failure_path.is_file():
    result['failure_evidence']={'path':name,'sha256':sha(failure_path),'detail':json.loads(failure_path.read_text())};break
 result['wall_s']=time.monotonic()-started
 with (folder/'owner_result.json').open('x') as f:json.dump(result,f,indent=2)
 return result

def worker(request):
 from .data import sha
 from .model import load
 from .requests import validate as validate_spec
 from native_stand.command_v6.tools.source_identity import record_configuration
 p=Path(request['parent_spec']);spec=json.loads(p.read_text())
 if request.get('standalone_request'):
  from .motion_quick_backfill import verify_child_request
  standalone=verify_child_request(request);cfg=standalone['configuration']
  validate(dict(spec,checkpoint_quick_evaluation=cfg))
 else:cfg=validate(spec)
 if sha(p)!=request['parent_spec_sha256'] or sha(request['checkpoint'])!=request['checkpoint_sha256']:raise ValueError('quick source/checkpoint SHA differs')
 errors=validate_spec(spec)
 if errors:raise ValueError(errors)
 book=json.loads(p.with_name('budget.json').read_text());attempt=book['attempts'].get(request['attempt_id'])
 if not request.get('standalone_request') and (not attempt or attempt['status']!='RUNNING' or attempt['phase']!='formal' or attempt['output_dir']!=request['parent_output'] or attempt['updates']<request['update']):raise ValueError('quick evaluation needs active exact parent formal budget')
 cp=Path(request['checkpoint'])
 if cp.parent!=Path(request['parent_output']) or cp.name!=f"model_update_{request['update']}.pt":raise ValueError('quick checkpoint path/update differs')
 actor,data=load(cp,'MotionZeroStudent')
 if data.get('update')!=request['update'] or data.get('spec_sha256')!=request['parent_spec_sha256']:raise ValueError('quick checkpoint internal identity differs')
 out=Path(request['output']);record_configuration(out,p)
 # Runtime-local evaluation layout; parent declaration/metadata stay untouched.
 s=copy.deepcopy(spec)
 if spec.get('motion_parallel'):
  from .motion_parallel_course import effective_spec
  state=data.get('motion_course_state')
  if not isinstance(state,dict):raise ValueError('parallel quick checkpoint course state missing')
  s=effective_spec(s,state['stage'],state)
 s['performance'].update(version='student_performance_v2',evaluation_group_commands=True,evaluation_group_max_commands=7,
  evaluation_gpu_buffer=True,evaluation_reuse_environment=True,combined_step_checks=False,fixed_shape_statistics=False,trace_buffer_max_bytes=cfg['trace_buffer_max_bytes'])
 for row in s['student_evaluation']['cases']:row.update(checkpoint=str(cp),sha256=request['checkpoint_sha256'])
 s['student_evaluation']['max_output_bytes']=cfg['output_max_bytes']
 # User-authorized default: one command / one environment, reused sequentially.
 matrix_screen=bool(request.get('standalone_request')) and standalone['version']=='motion31_saved_checkpoint_matrix_screen_v2'
 single_screen=not bool(request.get('standalone_request')) or matrix_screen
 s['performance']['evaluation_group_commands']=not single_screen
 if single_screen and not matrix_screen:
  targets=[r for r in s['student_evaluation']['cases'] if r['command'][:2]==[0.,.5]]
  if len(targets)!=1:raise ValueError('single condition quick screen requires explicit positive yaw+.5 in parent')
  s['student_evaluation']['cases']=targets
 with (out/'effective_layout.json').open('x') as f:json.dump({'parent_spec_sha256':sha(p),'performance':s['performance'],'protocol':spec.get('motion_parallel_assessment',spec['motion_assessment']),'case_matrix':s['student_evaluation']['cases'],'scored_envs_total':len(s['student_evaluation']['cases']) if single_screen else 56,'physical_envs':1 if single_screen else 64,'layout':'serial_single_condition_single_environment' if single_screen else 'legacy_authorized_grouped8','selection_eligible':False},f,indent=2)
 from .motion_reward_audit import audit
 import native_stand.students.motion_tracking_reward as reward_module
 snap=audit(s['reward']);snap.update(spec_sha256=sha(p),source_sha256=s['source_sha256'],checkpoint_sha256=sha(cp),reward_source_sha256=sha(reward_module.__file__))
 with (out/'reward_snapshot.json').open('x') as f:json.dump(snap,f,indent=2)
 # Assets before runtime import; AppLauncher before evaluator/environment creation.
 sys.path.insert(0,s['asset_module_root'])
 from isaaclab.app import AppLauncher
 app=AppLauncher({'headless':True,'device':'cuda:0','multi_gpu':False}).app
 try:
  from .training import evaluate_cases
  actor=actor.to('cuda:0').eval().requires_grad_(False)
  records=evaluate_cases(s,out,actor,s['student_evaluation']['cases'],checkpoint=str(cp),stop=lambda:time.time()>request['deadline_unix_s']-30,screen_single_env=single_screen)
  summary={'status':'COMPLETED','version':VERSION,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'update':request['update'],
   'selection_eligible':False,'accepted_candidate':None,'condition_count':len(s['student_evaluation']['cases']),'scored_envs_per_command':1 if single_screen else 8,'physical_envs':1 if single_screen else 64,'protocol':spec.get('motion_parallel_assessment',spec['motion_assessment']),'physical_serial_equivalence':'NOT_VERIFIED',
   'scope':'single-environment per command screening; full250 matrix remains candidate/early-stop authority' if single_screen else 'legacy grouped screening',
   'commands':[{'command':r['case']['command'],'passed':sum(e['accepted'] for e in r['per_env']),'scored_envs':len(r['per_env']),'per_env':r['per_env'],'raw':r['path'],'sha256':r['sha256']} for r in records]}
  with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2,allow_nan=False)
 except BaseException:
  # Persist primary exception before Kit cleanup can terminate the interpreter.
  failure={'status':'FAILED','exception':repr(sys.exception()),'traceback':traceback.format_exc(),'checkpoint_sha256':request['checkpoint_sha256'],'update':request['update']}
  with (out/'worker_failure.json').open('x') as f:json.dump(failure,f,indent=2)
  raise
 finally:
  primary=sys.exception()
  try:app.close()
  except BaseException:
   if primary is None:raise


def guarded_worker(request,invoke=None):
 """Convert Kit-swallowed errors/exit0 to persisted nonzero process outcome."""
 out=Path(request['output']);out.mkdir(parents=True,exist_ok=True)
 def interrupted_shutdown():
  if not (out/'summary.json').exists():
   path=out/'worker_shutdown_failure.json'
   if not path.exists():
    with path.open('x') as f:json.dump({'status':'FAILED','reason':'interpreter shutdown before summary; native/Kit exit or unhandled interruption; exact cause UNKNOWN','exit_code':1},f)
   os._exit(1)
 atexit.register(interrupted_shutdown)
 try:
  (invoke or worker)(request)
  summary=out/'summary.json'
  if not summary.is_file():raise RuntimeError('worker returned without complete summary')
  d=json.loads(summary.read_text())
  if d.get('status')!='COMPLETED' or d.get('checkpoint_sha256')!=request['checkpoint_sha256'] or d.get('update')!=request['update']:
   raise ValueError('worker summary identity/status differs')
  atexit.unregister(interrupted_shutdown)
  return 0
 except BaseException as error:
  failure={'status':'FAILED','exception':repr(error),'traceback':traceback.format_exc(),'checkpoint_sha256':request.get('checkpoint_sha256'),'update':request.get('update'),'exit_code':1}
  path=out/'worker_exit_failure.json'
  with path.open('x') as f:json.dump(failure,f,indent=2)
  atexit.unregister(interrupted_shutdown)
  return 1

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description='Internal owned child; public submission remains supervisor')
 parser.add_argument('--request',type=Path,required=True);args=parser.parse_args()
 code=guarded_worker(json.loads(args.request.read_text()))
 sys.stdout.flush();sys.stderr.flush()
 # Kit shutdown may replace SystemExit status; force the recorded owner outcome.
 os._exit(code)
