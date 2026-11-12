"""Public-owner bounded 350N/solver-stop diagnostic, zero optimizer updates."""
import copy,json,os,sys,time,traceback
from pathlib import Path
from .data import sha
VERSION='recovery_gas_spring_probe_v1'

def validate(r):
 from .physics_readback import validate as base
 adapted={k:v for k,v in r.items() if k not in ('geometry','geometry_sha256','limit_manifest','limit_manifest_sha256')};adapted.update(version='recovery_physics_readback_v1',scored_envs=0)
 if r['version']!=VERSION or r['scored_envs']!=1:raise ValueError('spring probe scope differs')
 s=base(adapted)
 for k in ('geometry','limit_manifest'):
  if sha(r[k])!=r[k+'_sha256']:raise ValueError('spring declaration changed')
 limits=json.loads(Path(r['limit_manifest']).read_text());g=json.loads(Path(r['geometry']).read_text())
 if sha(limits['candidate_usd'])!=limits['candidate_sha256'] or g['version']!='gas_spring_bilateral_planar_geometry_v2':raise ValueError('spring candidate identity differs')
 return s

def worker(r):
 from native_stand.command_v6.completion import write
 from native_stand.command_v6.tools.artifact_io import npz
 s=copy.deepcopy(validate(r));out=Path(r['output']);app=env=None;rows=[];code=1;start=time.monotonic()
 limits=json.loads(Path(r['limit_manifest']).read_text());g=json.loads(Path(r['geometry']).read_text())
 # New diagnostic effective asset, source checkpoint keeps original identity.
 s.update(asset_path=str(Path(limits['candidate_usd']).resolve()),asset_sha256=limits['candidate_sha256'])
 write(out/'effective_diagnostic.json',{'version':VERSION,'effective_asset':s['asset_path'],'effective_asset_sha256':s['asset_sha256'],'source_spec_sha256':r['parent_spec_sha256'],'law':'350N constant, k0, gas damping0, cutoffL>.215','stop':'knee_solver; compliant end stop disabled','control':s['control'],'max_episodes':2})
 try:
  sys.path.insert(0,s['asset_module_root'])
  from isaaclab.app import AppLauncher
  app=AppLauncher({'headless':True,'device':'cuda:0','multi_gpu':False}).app
  import torch,numpy as np
  from .evaluator import environment,rollout
  from .model import load
  from .physics_readback import collect
  cp=r['checkpoints'][0];actor,saved=load(cp['checkpoint'],'RecoveryStudent');actor.to('cuda:0').eval().requires_grad_(False)
  if saved['spec_sha256']!=r['parent_spec_sha256']:raise ValueError('source checkpoint spec differs')
  before={k:v.clone() for k,v in actor.state_dict().items()}
  for i,family in enumerate(['motion_native_normal','swing_candidate_from_historical_r1']):
   env=environment(s,'RecoveryStudent',envs=1,initial_family=family,autonomous_preview=True,diagnostic_gas_spring=g)
   env.reset();readback=collect(env)
   backend=readback['articulation']['get_dof_limits']
   if backend['status']!='READ':raise ValueError('solver knee limits unreadable')
   for row in limits['rows']:
    j=readback['joint_names'].index(row['joint'])
    if not np.allclose(backend['values'][0][j],row['saved_limits_rad'],atol=1e-6):raise ValueError('runtime knee limits differ')
   write(out/f'readback_{i:02}.json',readback)
   case={**next(x for x in s['student_evaluation']['cases'] if x['kind']=='student'),'family':family,'case_id':f'spring_{family}','checkpoint':cp['checkpoint'],'sha256':cp['sha256'],'seed':s['seed'],'yaw_rad':0.,'command_height_m':.34}
   payload,arrays=rollout(env,actor,case,scored_envs=1,stop=lambda:time.monotonic()-start>r['wall_limit_s']-45)
   samples=np.asarray(env.gas_spring_samples);calls=env.gas_spring_adapter.calls
   if calls!=len(samples) or calls==0:raise ValueError('spring substep capture mismatch')
   if not np.isfinite(samples).all() or samples[:,:,2:].max()>1e-3:raise ValueError('spring balance or finite failure')
   arrays['gas_spring_substeps']=samples
   npz(out/f'trace_{i:02}.npz',arrays)
   metrics={'spring_calls_1ms':calls,'length_minmax_m':[[float(samples[:,side,0].min()),float(samples[:,side,0].max())] for side in range(2)],'force_minmax_n':[float(samples[:,:,1].min()),float(samples[:,:,1].max())],'net_force_torque_max':samples[:,:,2:].max(axis=(0,1)).tolist(),'compliant_stop_enabled':False}
   payload.update(gas_spring=metrics,source_checkpoint_sha256=cp['sha256'],effective_asset_sha256=s['asset_sha256'])
   write(out/f'evaluation_{i:02}.json',payload);rows.append({'family':family,'per_env':payload['per_env'],'metrics':metrics})
   fault=arrays['state'][...,4].any()
   if fault:raise RuntimeError('runtime physical fault; second case not started')
   if any(not torch.equal(v,actor.state_dict()[k]) for k,v in before.items()):raise RuntimeError('Actor changed')
   env.close();env=None
   if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>r['output_max_bytes']:raise RuntimeError('output cap')
  write(out/'summary.json',{'status':'COMPLETED','version':VERSION,'results':rows,'online_PPO':0,'optimizer':0,'model_unchanged':True});code=0
 except BaseException as exc:write(out/'worker_failure.json',{'exception':repr(exc),'traceback':traceback.format_exc(),'completed_cases':rows,'remaining':'NOT_RUN'})
 finally:
  if env is not None:
   try:env.close()
   except Exception as exc:write(out/'cleanup_error.json',{'exception':repr(exc)})
  if app is not None:
   try:app.close()
   except Exception:pass
 return code
