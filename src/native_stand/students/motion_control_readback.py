"""Explicit bounded deterministic control capture, through public single-screen owner."""
import copy,json,time,traceback
from pathlib import Path
import numpy as np
from .data import sha
VERSION='motion31_control_readback_v1'
COMMANDS=[[0.,0.,.34],[.5,0.,.34],[0.,.5,.34],[0.,-.5,.34]]

def validate(r):
 from .motion_single_screen import validate as base,VERSION as BASE
 required={'version','authorization','online_PPO','physical_envs','scored_envs','wall_limit_s','parent_spec','parent_spec_sha256','session','physical_gpu','checkpoints','parent_output','output','commands','output_max_bytes'}
 if set(r)!=required or r['version']!=VERSION or r['commands']!=COMMANDS or len(r['checkpoints'])!=1 or r['output_max_bytes']!=256*2**20:raise ValueError('control readback schema/coverage/storage differs')
 local={k:v for k,v in r.items() if k not in ('commands','output_max_bytes')};local.update(version=BASE,command=COMMANDS[0])
 s=base(local)
 for c in COMMANDS:
  if c not in [x['command'] for x in s['student_evaluation']['cases']]:raise ValueError('readback command not declared by parent')
 if s.get('reward',{}).get('version')!='student_motion31_yaw_contact_quality_v7':raise ValueError('readback requires explicit spin-v7 measurement')
 return s

def capture_pre(env):
 """Aligned pre-action sample: no cached terminal/reset values or RNG calls."""
 from .motion_yaw_quality import runtime_measurement
 m=runtime_measurement(env);d=env.robot.data
 names=env.robot.body_names
 parents=[names.index(side+'_lower_leg') for side in ('left','right')]
 p=d.body_com_ang_vel_w.torch[:,parents]
 dq=d.joint_vel.torch[:,env.ids][:,[2,5]]
 return {**m,'parent_angular_world':p,'joint_dq_rad_s':dq,
         'relative_spin_rad_s':((m['angular_world']-p)*m['axis_world']).sum(-1),
         'executed_wheel_target_rad_s':env.executed_wheel}

def analyze(a):
 """Uses named capture fields; policy-end torque and speed are not PD-aligned."""
 from native_stand.command_v6.domain.trace_schema import CONTROL
 valid=a['first_episode_valid'][:,0].astype(bool)&(a['controller'][:,0,0]==3)&a['state'][:,0,29].astype(bool)
 idx=np.flatnonzero(valid)[:3000]
 if len(idx)!=3000:raise ValueError('control readback incomplete first controlled30s')
 h=idx[1000:1800];pre=h[1:] # First readback follows previous command's target; exclude boundary.
 dq=a['readback_joint_dq_rad_s'][pre,0];relative=a['readback_relative_spin_rad_s'][pre,0]
 residual=abs(relative-dq)
 if not np.isfinite(residual).all() or not (residual<=.05+.005*abs(dq)).all():raise ValueError('wheel joint/relative-body axial spin polarity differs')
 c=a['control'][h,0];actual=c[:,CONTROL['joint_dq']][:,[2,5]]
 raw=a['raw_action'][h,0][:,[2,5]]
 clipped=c[:,CONTROL['clipped_action']][:,[2,5]];target=clipped*100.
 request=c[:,CONTROL['requested_torque']][:,[2,5]];applied=c[:,CONTROL['applied_torque']][:,[2,5]]
 exec_before=a['readback_executed_wheel_target_rad_s'][pre,0]
 previous_target=target[:-1]
 if abs(exec_before-previous_target).max()>1e-4:raise ValueError('executed target readback versus previous action differs')
 mean=lambda x:np.asarray(x).mean(axis=0).tolist()
 rolling=a['readback_rolling_speed_forward_mps'][pre,0];slip=np.linalg.norm(a['readback_tangent_velocity_w'][pre,0],axis=-1)
 return {'controlled_samples':len(idx),'hold_samples':len(h),'window_s':[10.,18.],
  'wheel_order':['left','right'],'joint_vs_relative_axial_max_abs_rad_s':float(residual.max()),
  'raw_action_wheel_mean':mean(raw),'target_mean_rad_s':mean(target),'actual_joint_dq_mean_rad_s':mean(actual),
  'speed_error_abs_mean_rad_s':mean(abs(target-actual)),
  'requested_torque_mean_nm':mean(request),'applied_torque_mean_nm':mean(applied),
  'applied_torque_abs_mean_nm':mean(abs(applied)),
  'target_opposite_sign_fraction':float((target[:,0]*target[:,1]<0).mean()),
  'actual_opposite_sign_fraction':float((actual[:,0]*actual[:,1]<0).mean()),
  'target_delta_abs_mean_rad_s':mean(abs(np.diff(target,axis=0))),
  'applied_torque_sign_switches':(np.sign(applied[1:])!=np.sign(applied[:-1])).sum(0).tolist(),
  'rolling_forward_mean_mps':mean(rolling),'support_slip_mean_mps':mean(slip),
  'support_slip_p95_mps':np.quantile(slip,.95,axis=0).tolist(),
  'horizontal_mean_mps':float(np.linalg.norm(a['root_post'][h,0,7:9],axis=-1).mean()),
  'yaw_mean_rad_s':float(a['state'][h,0,9].mean()),
  'dual_contact_fraction':float((a['state'][h,0,10:12]>2).all(-1).mean()),
  'wheel_torque_saturation_fraction':mean(abs(applied)>=5.5-1e-4),
  'pd_endstep_residual_abs_mean_nm':mean(abs(request-(target-actual))),
  'timing_note':'control torque is latest held5ms PD request; joint dq is policy-end state. Their residual alone is not wiring error.',
  'slip_note':'100Hz finite-cylinder support proxy, not solver contact patch; pre-action sampled separately from policy-end control'}

def worker(r):
 import sys,torch
 s=copy.deepcopy(validate(r));out=Path(r['output']);app=env=None;rows=[];started=time.monotonic();code=1
 from .performance import new_motion_defaults
 s['performance']=new_motion_defaults(s.get('performance'))
 try:
  from .motion_reward_audit import audit
  from . import motion_tracking_reward,motion_yaw_quality
  snapshot=audit(s['reward']);snapshot.update(parent_spec_sha256=r['parent_spec_sha256'],checkpoint_sha256=r['checkpoints'][0]['sha256'],
    actual_formula_sha256={str(Path(m.__file__).resolve()):sha(m.__file__) for m in (motion_tracking_reward,motion_yaw_quality)})
  (out/'reward_snapshot.json').write_text(json.dumps(snapshot,indent=2))
  (out/'effective_capture.json').write_text(json.dumps({'version':VERSION,'performance':s['performance'],'sample_rate_hz':100,'conditions':r['commands'],
    'fields':'full rollout control/actor_input plus named pre-action wheel COM/axis/velocity/parent spin/target; no compact trimming','max_output_bytes':r['output_max_bytes'],'online_PPO':0,'optimizer_updates':0},indent=2))
  sys.path.insert(0,s['asset_module_root'])
  from isaaclab.app import AppLauncher
  app=AppLauncher({'headless':True,'device':'cuda:0','multi_gpu':False}).app
  from .evaluator import environment,rollout
  from .model import load
  cp=r['checkpoints'][0];actor,data=load(cp['checkpoint'],'MotionZeroStudent')
  if data['spec_sha256']!=r['parent_spec_sha256']:raise ValueError('checkpoint internal spec identity differs')
  actor=actor.to('cuda:0').eval().requires_grad_(False)
  before={k:v.clone() for k,v in actor.state_dict().items()}
  env=environment(s,'MotionZeroStudent',envs=1,initial_family='motion_native_normal',autonomous_preview=True,single_condition_screen=True)
  env.motion_control_readback=True
  for i,command in enumerate(r['commands']):
   case=copy.deepcopy(next(x for x in s['student_evaluation']['cases'] if x['command']==command));case.update(checkpoint=cp['checkpoint'],sha256=cp['sha256'])
   payload,arrays=rollout(env,actor,case,scored_envs=1,stop=lambda:time.monotonic()-started>r['wall_limit_s']-45)
   for name,arr in arrays.items():
    if np.issubdtype(arr.dtype,np.number) and not np.isfinite(arr).all():raise ValueError('nonfinite capture '+name)
   for k,v in before.items():
    if not torch.equal(v,actor.state_dict()[k]):raise ValueError('Actor changed during PPO0')
   if abs(arrays['effective_command_pre'][1000:1800,0,:2]-np.asarray(command[:2])).max()>1e-5:raise ValueError('hold command readback differs')
   np.savez_compressed(out/f'trace_{i:02d}.npz',**arrays)
   try:analysis=analyze(arrays);analysis_status='COMPLETE'
   except ValueError as error:analysis={'error':str(error)};analysis_status='INCOMPLETE_OR_INVALID'
   payload.update(checkpoint_sha256=cp['sha256'],parent_spec_sha256=r['parent_spec_sha256'],trace={'path':f'trace_{i:02d}.npz'},readback=analysis,readback_status=analysis_status,model_unchanged=True)
   (out/f'evaluation_{i:02d}.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
   rows.append({'command':command,'analysis':analysis,'readback_status':analysis_status,'evaluation':f'evaluation_{i:02d}.json'})
   if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>r['output_max_bytes']:raise RuntimeError('readback output budget exhausted')
   if analysis_status!='COMPLETE':raise RuntimeError('readback incomplete or invalid; raw trace retained')
  summary={'status':'COMPLETED','version':VERSION,'checkpoint_sha256':cp['sha256'],'results':rows,'online_PPO':0,'optimizer_updates':0,'physical_envs':1,'scored_envs_per_condition':1,'capability_certification':False,'model_unchanged':True}
  (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));code=0
 except BaseException as error:
  (out/'worker_failure.json').write_text(json.dumps({'exception':repr(error),'traceback':traceback.format_exc(),'partial_results':rows},indent=2))
 finally:
  if env is not None:
   try:env.close()
   except BaseException as error:(out/'cleanup_error.json').write_text(json.dumps({'exception':repr(error)}))
  if app is not None:
   try:app.close()
   except BaseException:pass
 return code
