"""Authorized single-condition, one-environment frozen checkpoint diagnostic."""
import json,os,sys,time,subprocess,traceback
from pathlib import Path
from .data import sha
VERSION='motion31_single_condition_screen_v1'

def validate(request):
 if request.get('version') in ('recovery31_distilled_native_screen_v1','recovery31_distilled_native_student_only_v2'):
  from .recovery_distilled_native import validate as validate_recovery_distilled
  return validate_recovery_distilled(request)
 if request.get('version') in ('motion31_distilled_native_screen_v1','motion31_distilled_native_completion_v2'):
  from .motion_distilled_native import validate as validate_distilled
  return validate_distilled(request)
 if request.get('version')=='recovery_gas_spring_probe_v1':
  from .gas_spring_probe import validate as validate_spring
  return validate_spring(request)
 if request.get('version')=='recovery_physics_readback_v1':
  from .physics_readback import validate as validate_physics
  return validate_physics(request)
 if request.get('version')=='motion31_control_readback_v1':
  from .motion_control_readback import validate as validate_control
  return validate_control(request)
 if request['version']!=VERSION or request['authorization']!='AUTHORIZED' or request['online_PPO']!=0 or request['physical_envs']!=1 or request['scored_envs']!=1 or not 60<=request['wall_limit_s']<=1800:raise ValueError('single screen declaration differs')
 p=Path(request['parent_spec']);s=json.loads(p.read_text())
 if sha(p)!=request['parent_spec_sha256'] or request['session']!=os.environ.get('HAPI_SESSION_ID') or s['device']['physical_index']!=request['physical_gpu']:raise ValueError('screen parent/session/device differs')
 if request['command'] not in [x['command'] for x in s['student_evaluation']['cases']]:raise ValueError('screen command not in parent')
 for row in request['checkpoints']:
  cp=Path(row['checkpoint'])
  if cp.parent!=Path(request['parent_output']) or cp.name!=f"model_update_{row['update']}.pt" or sha(cp)!=row['sha256']:raise ValueError('screen checkpoint identity differs')
 return s

def launch(args):
 from native_stand.command_v6.tools.health_check import gpu_snapshot
 from native_stand.command_v6.domain.resource_policy import resolve_resources
 req=json.loads(Path(args.single_screen_request).read_text());s=validate(req);out=Path(req['output'])
 policy=resolve_resources({},mode=args.resource_mode,sharing_authorization=args.sharing_authorization)
 if policy['mode']!='shared' or not policy['sharing_authorized']:raise ValueError('shared GPU authorization required')
 if out.exists():raise FileExistsError(out)
 if not args.launch:print(json.dumps({'status':'PLAN','files_written':0,'online_PPO':0,'request':req},indent=2));return
 gpu=gpu_snapshot();devices=[x for x in gpu['devices'] if x['index']==req['physical_gpu']]
 if len(devices)!=1 or devices[0]['uuid']!=s['device']['uuid']:raise ValueError('GPU UUID differs')
 if float(devices[0]['memory_total_mib'])-float(devices[0]['memory_used_mib'])<2048:raise ValueError('GPU memory insufficient; no preemption')
 model_count=len(req['checkpoints'])+int(req.get('version')=='recovery31_distilled_native_screen_v1')
 condition_count=len(req.get('cases',req.get('commands',[req.get('command')])))
 out.mkdir();(out/'request.json').write_text(json.dumps(req,indent=2));(out/'budget.json').write_text(json.dumps({'status':'RUNNING','online_PPO':0,'optimizer_steps':0,'limit_wall_s':req['wall_limit_s'],'conditions_per_checkpoint':condition_count,'scored_episode_limit':condition_count*model_count,'scored_envs':1,'checkpoints':model_count,'gpu_snapshot':gpu},indent=2))
 command=[sys.executable,'-m',__name__,'--request',str(out/'request.json')]
 if req['wall_limit_s'] is not None:command=['timeout','--signal=TERM','--kill-after=15s',str(req['wall_limit_s'])+'s']+command
 start=time.monotonic()
 with (out/'runtime.log').open('x') as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(req['physical_gpu']),'PYTHONPATH':s['execution_source_root']}).returncode
 summary=out/'summary.json';status='COMPLETED' if code==0 and summary.exists() else 'FAILED'
 # Some Kit shutdown handlers force OS exit0 even after a caught worker fault.
 semantic_code=code if code else 0 if status=='COMPLETED' else 1
 (out/'terminal_report.json').write_text(json.dumps({'status':status,'exit_code':semantic_code,'process_exit_code':code,'wall_s':time.monotonic()-start,'online_PPO':0,'summary':str(summary)},indent=2))
 b=json.loads((out/'budget.json').read_text());b.update(status=status,wall_s=time.monotonic()-start);(out/'budget.json').write_text(json.dumps(b,indent=2))
 print(json.dumps({'status':status,'output':str(out),'exit_code':semantic_code,'process_exit_code':code,'summary':json.loads(summary.read_text()) if summary.exists() else None},ensure_ascii=False,indent=2))
 if status!='COMPLETED':raise RuntimeError('single-condition screen incomplete; see terminal/worker failure evidence')

def worker(request):
 if request.get('version') in ('recovery31_distilled_native_screen_v1','recovery31_distilled_native_student_only_v2'):
  from .recovery_distilled_native import worker as recovery_distilled_worker
  return recovery_distilled_worker(request)
 if request.get('version') in ('motion31_distilled_native_screen_v1','motion31_distilled_native_completion_v2'):
  from .motion_distilled_native import worker as distilled_worker
  return distilled_worker(request)
 if request.get('version')=='recovery_gas_spring_probe_v1':
  from .gas_spring_probe import worker as spring_worker
  return spring_worker(request)
 if request.get('version')=='recovery_physics_readback_v1':
  from .physics_readback import worker as physics_worker
  return physics_worker(request)
 if request.get('version')=='motion31_control_readback_v1':
  from .motion_control_readback import worker as control_worker
  return control_worker(request)
 s=validate(request);out=Path(request['output']);app=env=None;results=[];code=1;started=time.monotonic()
 # Diagnostic-local transport optimization; parent frozen spec is not mutated.
 import copy
 from .performance import new_motion_defaults
 s=copy.deepcopy(s);s['performance']=new_motion_defaults(s.get('performance'))
 (out/'effective_performance.json').write_text(json.dumps({'performance':s['performance'],'parent_spec_sha256':request['parent_spec_sha256'],'scope':'explicit new single-environment diagnostic only'},indent=2))
 try:
  from .motion_reward_audit import audit
  (out/'reward_snapshot.json').write_text(json.dumps(audit(s['reward']),indent=2))
  sys.path.insert(0,s['asset_module_root'])
  from isaaclab.app import AppLauncher
  app=AppLauncher({'headless':True,'device':'cuda:0','multi_gpu':False}).app
  from .evaluator import environment,rollout
  from .model import load
  env=environment(s,'MotionZeroStudent',envs=1,initial_family='motion_native_normal',autonomous_preview=True,single_condition_screen=True)
  template=next(x for x in s['student_evaluation']['cases'] if x['command']==request['command'])
  for cp in request['checkpoints']:
   actor,data=load(cp['checkpoint'],'MotionZeroStudent');actor=actor.to('cuda:0').eval().requires_grad_(False)
   if data['spec_sha256']!=request['parent_spec_sha256']:raise ValueError('screen internal spec differs')
   case={**template,'checkpoint':cp['checkpoint'],'sha256':cp['sha256']}
   payload,arrays=rollout(env,actor,case,scored_envs=1,stop=lambda:time.monotonic()-started>request['wall_limit_s']-45)
   expected=__import__('numpy').asarray(request['command'][:2])
   actual_command=arrays['effective_command_pre']
   if abs(actual_command[1000:1800,0,:2]-expected).max()>1e-5:raise ValueError('hold command readback mismatch')
   row={'update':cp['update'],'checkpoint':cp['checkpoint'],'sha256':cp['sha256'],'command':request['command'],'physical_envs':1,'scored_envs':1,'per_env':payload['per_env'],'protocol':payload['protocol'],'selection_eligible':False}
   (out/f"update_{cp['update']}.json").write_text(json.dumps(row,indent=2,allow_nan=False));results.append(row)
  (out/'summary.json').write_text(json.dumps({'status':'COMPLETED','version':VERSION,'results':results,'online_PPO':0,'scope':'one condition/one environment screening; not full capability certification'},indent=2,allow_nan=False));code=0
 except BaseException as error:
  (out/'worker_failure.json').write_text(json.dumps({'exception':repr(error),'traceback':traceback.format_exc(),'partial_results':len(results)},indent=2))
 finally:
  if env is not None:
   try:env.close()
   except BaseException as error:
    (out/'cleanup_error.json').write_text(json.dumps({'exception':repr(error)}))
  if app is not None:
   try:app.close()
   except BaseException:pass
 return code

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args();code=worker(json.loads(a.request.read_text()));sys.stdout.flush();sys.stderr.flush();os._exit(code)
