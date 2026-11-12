"""Explicit distilled small Motion Actor PPO0, through existing single-screen owner."""
import copy,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from .data import sha
from .motion_distillation_pipeline import COMMANDS,validate_student,write_json
VERSION='motion31_distilled_native_screen_v1'
COMPLETION='motion31_distilled_native_completion_v2'

def validate_completion(r):
 """Bind four completed prior cases, then run exactly three missing cases fresh."""
 if set(r['completion'])!={'parent_run','inputs'}:raise ValueError('completion provenance schema differs')
 p=Path(r['completion']['parent_run']);inputs=r['completion']['inputs']
 if r['wall_limit_s'] is not None or r['commands']!=COMMANDS[4:]:raise ValueError('completion scope must be three missing conditions with explicit null wall')
 for path,digest in inputs.items():
  if sha(path)!=digest:raise ValueError('completion evidence changed:'+path)
 term=json.loads((p/'terminal_report.json').read_text());failure=json.loads((p/'worker_failure.json').read_text());req=json.loads((p/'request.json').read_text())
 if term['status']!='FAILED' or failure['exception']!="TimeoutError('Student evaluation wall budget exhausted')" or len(failure['partial_results'])!=4:raise ValueError('completion requires settled four-case timeout')
 if req['checkpoints']!=r['checkpoints'] or req['parent_spec_sha256']!=r['parent_spec_sha256']:raise ValueError('completion model/protocol identity differs')
 required=[p/'request.json',p/'terminal_report.json',p/'worker_failure.json']
 for i,c in enumerate(COMMANDS[:4]):
  ep=p/f'evaluation_{i:02}.json';d=json.loads(ep.read_text());tp=p/d['trace']['path'];required.extend((ep,tp))
  if d['case']['command']!=c or d['checkpoint_sha256']!=r['checkpoints'][0]['sha256'] or not d.get('model_unchanged') or len(d['per_env'])!=1 or d['per_env'][0]['controlled_seconds']!=30 or sha(tp)!=d['trace']['sha256']:raise ValueError('prior completed case identity/window differs')
 if set(inputs)!=set(map(str,required)):raise ValueError('completion input manifest differs')
 return req

def validate(r):
 import torch
 from .model import load
 from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
 required={'version','authorization','online_PPO','physical_envs','scored_envs','wall_limit_s','parent_spec','parent_spec_sha256','session','physical_gpu','checkpoints','output','commands','output_max_bytes'}
 completing=r.get('version')==COMPLETION
 if completing:required.add('completion')
 if set(r)!=required or r['version'] not in (VERSION,COMPLETION) or r['authorization']!='AUTHORIZED':raise ValueError('distilled diagnostic schema/authorization differs')
 if completing:validate_completion(r)
 elif type(r['wall_limit_s']) not in (int,float) or not 60<=r['wall_limit_s']<=1800:raise ValueError('distilled diagnostic wall budget differs')
 if r['online_PPO']!=0 or r['physical_envs']!=1 or r['scored_envs']!=1 or r['output_max_bytes']!=256*2**20:raise ValueError('distilled diagnostic budget differs')
 if r['commands']!=(COMMANDS[4:] if completing else COMMANDS) or len(r['checkpoints'])!=1:raise ValueError('distilled command coverage required')
 validate_new_experiment_name(Path(r['output']).name)
 if r['session']!=os.environ.get('HAPI_SESSION_ID') or not r['session']:raise ValueError('current session differs')
 spec_path=Path(r['parent_spec']);s=json.loads(spec_path.read_text())
 if sha(spec_path)!=r['parent_spec_sha256'] or s['device']['physical_index']!=r['physical_gpu']:raise ValueError('parent spec/device differs')
 cp=r['checkpoints'][0]
 if set(cp)!={'checkpoint','sha256'} or sha(cp['checkpoint'])!=cp['sha256']:raise ValueError('Student SHA differs')
 actor,data=load(cp['checkpoint'],'MotionZeroStudent');validate_student(data,spec_path)
 if list(actor.mlp[0].weight.shape)!=[128,31] or len([m for m in actor.mlp if isinstance(m,torch.nn.Linear)])!=4:raise ValueError('small Actor architecture differs')
 if any(c not in [row['command'] for row in s['student_evaluation']['cases']] for c in COMMANDS):raise ValueError('parent command coverage differs')
 return s

def swing_report(arrays,anchors):
 """Root-frame wheel centers from existing named geometry, first controlled episode."""
 from native_stand.command_v6.domain.trace_schema import STATE
 valid=arrays['first_episode_valid'][:,0].astype(bool)&(arrays['controller'][:,0,0]==3)&arrays['state'][:,0,STATE['initial_valid']].astype(bool)
 g=arrays['geometry'][valid,0][:3000,:6].reshape(-1,2,3)
 v=g-np.asarray(anchors)[None,:,:];a=np.arctan2(-v[...,2],-v[...,1]);delta=abs(np.arctan2(np.sin(a[:,0]-a[:,1]),np.cos(a[:,0]-a[:,1])))
 full=len(delta)==3000;hold=delta[1000:1800]
 return {'strict_limit_rad':.05,'sample_rate_hz':100,'controlled_samples':len(delta),'full_window_complete':full,
  'full_max_rad':float(delta.max()) if len(delta) else None,'hold_max_rad':float(hold.max()) if len(hold) else None,
  'hold_mean_rad':float(hold.mean()) if len(hold) else None,'hold_passed':len(hold)==800 and bool((hold<.05).all()),
  'full_passed':full and bool((delta<.05).all()),'scope':'first controlled episode; root-frame hip-anchor to wheel center, atan2(-z,-y); no reset splice'}

def worker(r):
 import torch
 from .model import load
 s=copy.deepcopy(validate(r));out=Path(r['output']);app=env=None;rows=[];started=time.monotonic();code=1
 from .performance import new_motion_defaults
 s['performance']=new_motion_defaults(s.get('performance'))
 try:
  from .motion_reward_audit import audit
  from . import motion_tracking_reward,motion_yaw_quality
  snapshot=audit(s['reward']);snapshot.update(parent_spec_sha256=r['parent_spec_sha256'],checkpoint_sha256=r['checkpoints'][0]['sha256'],actual_formula_sha256={str(Path(m.__file__).resolve()):sha(m.__file__) for m in (motion_tracking_reward,motion_yaw_quality)})
  write_json(out/'reward_snapshot.json',snapshot)
  write_json(out/'effective_diagnostic.json',{'version':r['version'],'actor':[31,128,64,32,6],'critic_loaded':False,'teacher_control':False,'performance':s['performance'],'physical_envs':1,'scored_envs_per_command':1,'commands':r['commands'],'wall_limit_s':r['wall_limit_s'],'seconds_per_case':31.52,'controlled_seconds_required':30,'hold_window_s':[10,18],'online_PPO':0,'optimizer_updates':0,'sampling_hz':100,'storage':'full requested first-episode traces; no arrays embedded in summary'})
  sys.path.insert(0,s['asset_module_root'])
  from isaaclab.app import AppLauncher
  app=AppLauncher({'headless':True,'device':'cuda:0','multi_gpu':False}).app
  from .evaluator import environment,rollout
  cp=r['checkpoints'][0];actor,data=load(cp['checkpoint'],'MotionZeroStudent');actor=actor.to('cuda:0').eval().requires_grad_(False)
  before={k:v.clone() for k,v in actor.state_dict().items()}
  env=environment(s,'MotionZeroStudent',envs=1,initial_family='motion_native_normal',autonomous_preview=True,single_condition_screen=True)
  for command in r['commands']:
   i=COMMANDS.index(command)
   case=copy.deepcopy(next(x for x in s['student_evaluation']['cases'] if x['command']==command));case.update(checkpoint=cp['checkpoint'],sha256=cp['sha256'],seconds=31.52)
   try:payload,arrays=rollout(env,actor,case,scored_envs=1,stop=None if r['wall_limit_s'] is None else lambda:time.monotonic()-started>r['wall_limit_s']-45)
   except BaseException as error:
    partial=getattr(error,'student_partial_arrays',None) or getattr(error,'student_pre_action_arrays',None)
    if partial:np.savez_compressed(out/f'partial_{i:02}.npz',**partial)
    raise
   if any(not torch.equal(v,actor.state_dict()[k]) for k,v in before.items()) or sha(cp['checkpoint'])!=cp['sha256']:raise ValueError('Student changed during PPO0')
   expected=arrays['effective_command_pre']/np.array([4.,4.,.5])
   if not np.allclose(arrays['actor_input'][...,22:25],expected,atol=1e-6):raise ValueError('command/observation readback differs')
   swing=swing_report(arrays,s['recovery']['standing_task']['hip_anchors_root_m'])
   np.savez_compressed(out/f'trace_{i:02}.npz',**arrays)
   payload.update(checkpoint_sha256=cp['sha256'],parent_spec_sha256=r['parent_spec_sha256'],distillation=data['distillation'],trace={'path':f'trace_{i:02}.npz','sha256':sha(out/f'trace_{i:02}.npz')},supplemental_swing_constraint=swing,model_unchanged=True)
   payload['interface'].update(actor_hidden_dims=[128,64,32],critic_loaded=False,actor_input=31)
   write_json(out/f'evaluation_{i:02}.json',payload)
   rows.append({'command':command,'evaluation':f'evaluation_{i:02}.json','passed_envs':payload['passed_envs'],'accepted_old_gate':payload['accepted'],'swing':swing,'per_env':payload['per_env']})
   if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>r['output_max_bytes']:raise RuntimeError('diagnostic storage budget exhausted')
  write_json(out/'summary.json',{'status':'COMPLETED','version':r['version'],'results':rows,'prior_completed_evidence':r.get('completion'),'online_PPO':0,'optimizer_updates':0,'capability_certified':False,'model_unchanged':True,'scope':'requested commands x one environment, one seed; native Student independent PPO0; not multi-env certification'})
  code=0
 except BaseException as error:
  write_json(out/'worker_failure.json',{'exception':repr(error),'traceback':traceback.format_exc(),'partial_results':rows})
 finally:
  if env is not None:
   try:env.close()
   except BaseException as error:write_json(out/'cleanup_error.json',{'exception':repr(error)})
  if app is not None:
   try:app.close()
   except BaseException:pass
 return code
