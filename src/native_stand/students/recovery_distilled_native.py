"""Explicit small Recovery independent PPO0 adapter under the public diagnostic owner."""
import copy,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from .data import sha
VERSION='recovery31_distilled_native_screen_v1'
STUDENT_ONLY='recovery31_distilled_native_student_only_v2'


def validate(r):
 from .model import load
 from .recovery_distillation import VERSION as BC_VERSION,SOURCE,ASSET
 from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
 required={'version','authorization','online_PPO','physical_envs','scored_envs','wall_limit_s','parent_spec','parent_spec_sha256','session','physical_gpu','gpu_uuid','checkpoints','teacher','dataset','dataset_sha256','dataset_manifest_sha256','output','cases','output_max_bytes'}
 if set(r)!=required or r['version'] not in (VERSION,STUDENT_ONLY) or r['authorization']!='AUTHORIZED':raise ValueError('Recovery distilled request schema/authorization differs')
 if r['online_PPO']!=0 or r['physical_envs']!=1 or r['scored_envs']!=1 or r['output_max_bytes']!=256*2**20 or not 300<=r['wall_limit_s']<=1800:raise ValueError('Recovery diagnostic bound differs')
 if r['session']!=os.environ.get('HAPI_SESSION_ID') or not r['session']:raise ValueError('current session differs')
 if r['physical_gpu']!=0 or r['gpu_uuid']!='GPU-e704e7d4-2965-4712-9759-ba6b9910bd72':raise ValueError('explicit current GPU0 shared identity required')
 validate_new_experiment_name(Path(r['output']).name)
 parent=Path(r['parent_spec']);s=json.loads(parent.read_text())
 if sha(parent)!=r['parent_spec_sha256'] or s['asset_sha256']!=ASSET or sha(s['asset_path'])!=ASSET or s['student_training']['version']!='student_recovery_contact_quality_ppo_v1':raise ValueError('parent spec/asset/runtime differs')
 if s.get('recovery_domain_randomization') or s.get('recovery_course'):raise ValueError('nominal independent Recovery only; no DR/teacher prefix')
 expected=[{k:row[k] for k in ('case_id','kind','family','seed','yaw_rad','seconds','role')} for row in s['student_evaluation']['cases']]
 if len(expected)!=4 or r['cases']!=expected:raise ValueError('exact original four Recovery conditions required')
 if len(r['checkpoints'])!=1 or set(r['checkpoints'][0])!={'checkpoint','sha256'} or set(r['teacher'])!={'checkpoint','sha256'}:raise ValueError('one Student and one frozen Teacher required')
 student=r['checkpoints'][0];teacher=r['teacher']
 if sha(student['checkpoint'])!=student['sha256'] or sha(teacher['checkpoint'])!=SOURCE or teacher['sha256']!=SOURCE:raise ValueError('Student/Teacher SHA differs')
 actor,data=load(student['checkpoint'],'RecoveryStudent');t,cp=load(teacher['checkpoint'],'RecoveryStudent')
 if cp['spec_sha256']!=sha(parent) or cp['asset_sha256']!=ASSET:raise ValueError('Teacher exact parent differs')
 dataset=Path(r['dataset']);manifest=dataset.with_name('manifest.json');m=json.loads(manifest.read_text())
 if sha(dataset)!=r['dataset_sha256'] or sha(manifest)!=r['dataset_manifest_sha256'] or data['dataset_sha256']!=sha(dataset) or data['manifest_sha256']!=sha(manifest):raise ValueError('Student dataset lineage differs')
 if m['version']!=BC_VERSION or m['teacher_parent_spec_sha256']!=sha(parent) or data['teacher_sha256']!={str(Path(teacher['checkpoint']).resolve()):SOURCE}:raise ValueError('Student teacher lineage differs')
 for model,target in ((actor,[31,128,64,32,6]),(t,[31,512,256,128,6])):
  dims=[model.mlp[0].in_features]+[v.out_features for v in model.mlp if hasattr(v,'out_features')]
  if dims!=target:raise ValueError('explicit small/large architecture differs')
 # Resource-only override is explicit in this diagnostic, never modifies frozen parent spec.
 s=copy.deepcopy(s);s['device']={'physical_index':r['physical_gpu'],'uuid':r['gpu_uuid'],'logical':'cuda:0','headless':True}
 return s


def phase_metrics(arrays,result):
 valid=arrays['first_episode_valid'][:,0].astype(bool)&arrays['actor_valid'][:,0,0].astype(bool)
 indices=np.flatnonzero(valid);entry=result.get('entry_index');windows={'controlled':indices,'pre_entry':indices if entry is None else indices[indices<=entry]}
 if entry is not None:windows['first_fixed_hold']=indices[(indices>entry)&(indices<=entry+1000)]
 rows={}
 for name,ix in windows.items():
  if not len(ix):continue
  s=arrays['state'][ix,0];p=arrays['posture'][ix,0];c=arrays['control'][ix,0]
  rows[name]={'samples':len(ix),'seconds':len(ix)*.01,'height_max_m':float(s[:,6].max()),'height_min_m':float(s[:,6].min()),'mean_abs_vx_vy_yaw':abs(s[:,7:10]).mean(0).tolist(),'peak_abs_pitch_roll_rad':abs(p[:,:2]).max(0).tolist(),'both_contact_fraction':float((s[:,10:12]>2).all(-1).mean()),'channel_torque_limit_fraction':(abs(c[:,38:44])>=np.array([40,40,5.5,40,40,5.5])-1e-4).mean(0).tolist(),'wheel_targets_abs_mean_rad_s':abs(c[:,18:24][:,[2,5]]*100).mean(0).tolist()}
 return rows


def worker(r):
 import torch
 from .model import load
 from .artifacts import npz as save_npz
 from native_stand.command_v6.completion import write
 s=validate(r);out=Path(r['output']);app=env=None;results=[];started=time.monotonic();code=1
 models=[('student',r['checkpoints'][0])]
 if r['version']==VERSION:models.append(('teacher_baseline',r['teacher']))
 matrix=[{'model':label,'case_id':case['case_id'],'status':'NOT_RUN'} for label,_ in models for case in r['cases']]
 write(out/'evaluation_matrix.json',matrix)
 try:
  sys.path.insert(0,s['asset_module_root'])
  from isaaclab.app import AppLauncher
  app=AppLauncher({'headless':True,'device':'cuda:0','multi_gpu':False}).app
  from .evaluator import environment,rollout
  from native_stand.command_v6.runtime.control_readback import verify,verify_ground
  env=environment(s,'RecoveryStudent',envs=1,autonomous_preview=True,single_condition_screen=True)
  write(out/'runtime_readback.json',{'control':verify(env),'ground':verify_ground(env,s),'interface':'multiskill_mcu_v2','command':[0,0,.34],'effective_device':s['device'],'physical_envs':1,'scored_envs':1,'parent_physical_gpu':json.loads(Path(r['parent_spec']).read_text())['device']['physical_index'],'no_teacher_execution_in_student':True,'teacher_baseline':'separate whole episode; no rescue or action blending'})
  for label,checkpoint in models:
   actor,data=load(checkpoint['checkpoint'],'RecoveryStudent');actor=actor.to('cuda:0').eval().requires_grad_(False)
   before={k:v.clone() for k,v in actor.state_dict().items()}
   for case in r['cases']:
    index=len(results);matrix[index]['status']='INCOMPLETE';write(out/'evaluation_matrix.json',matrix)
    row={**case,**checkpoint};payload,arrays=rollout(env,actor,row,scored_envs=1,stop=lambda:time.monotonic()-started>r['wall_limit_s']-45)
    if not all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items()):raise RuntimeError('evaluation modified model')
    valid=arrays['first_episode_valid'].astype(bool)&arrays['actor_valid'][...,0].astype(bool)
    if not valid.any() or not np.all(arrays['controller'][...,0][valid]==3):raise RuntimeError('student control ownership missing')
    x=arrays['actor_input'][valid]
    if not np.allclose(x[:,22:25],[0,0,.68],atol=1e-6) or not np.all(x[:,25:31]==[0,1,0,1,0,0]):raise RuntimeError('Recovery input/command readback differs')
    dimensions=[31]+[layer.out_features for layer in actor.mlp if hasattr(layer,'out_features')]
    payload.update(evaluated_model=label,checkpoint=checkpoint['checkpoint'],checkpoint_sha256=checkpoint['sha256'],parent_spec_sha256=r['parent_spec_sha256'],actor_actual_network=dimensions,model_unchanged=True,teacher_rescue=False,phase_metrics=phase_metrics(arrays,payload['per_env'][0]))
    payload['interface']['actor_hidden_dims']=dimensions[1:-1]
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if size>r['output_max_bytes']-16*2**20:raise RuntimeError('diagnostic storage reserve reached')
    trace=out/f'evaluation_{index:02}.npz';save_npz(trace,arrays);payload['trace']={'path':trace.name,'sha256':sha(trace)}
    path=out/f'evaluation_{index:02}.json';write(path,payload)
    result=payload['per_env'][0];results.append({'model':label,'case_id':case['case_id'],'path':path.name,'sha256':sha(path),'accepted':result['accepted'],'entry':result.get('entry_index') is not None,'hold_complete':result['hold_complete'],'settle_s':result.get('settle_s'),'failed_checks':result['failed_checks'],'phase_metrics':payload['phase_metrics']})
    matrix[index].update(status='COMPLETED',result=path.name);write(out/'evaluation_matrix.json',matrix)
  write(out/'summary.json',{'status':'COMPLETED','version':r['version'],'results':results,'student_passed':sum(e['accepted'] for e in results if e['model']=='student'),'teacher_passed':sum(e['accepted'] for e in results if e['model']=='teacher_baseline') if len(models)>1 else None,'teacher_evaluation':'RUN' if len(models)>1 else 'SKIPPED_USER_REQUEST','scored_episodes':len(results),'physical_envs':1,'scored_envs_per_condition':1,'online_PPO':0,'optimizer_steps':0,'scope':'four declared conditions,1env each per model; screening, not32/32 certification','teacher_control_in_student_episodes':False});code=0
 except BaseException as error:
  partial=getattr(error,'student_partial_arrays',None)
  if partial:
   try:save_npz(out/'fault_tail.npz',partial)
   except BaseException as secondary:error.add_note(repr(secondary))
  write(out/'worker_failure.json',{'exception':repr(error),'traceback':traceback.format_exc(),'partial_results':results,'secondary_errors':getattr(error,'__notes__',[])})
 finally:
  if env is not None:
   try:env.close()
   except BaseException as error:write(out/'cleanup_error.json',{'exception':repr(error)})
  if app is not None:
   try:app.close()
   except BaseException:pass
 return code
