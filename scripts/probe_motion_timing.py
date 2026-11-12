"""Authorized bounded pre-launch timing wiring probe; no optimizer/PPO."""
import argparse,json,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 s=json.loads(a.spec.read_text());a.output_dir.mkdir(parents=True,exist_ok=False)
 def write(name,d):
  with (a.output_dir/name).open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
 sys.path.insert(0,s['asset_module_root'])
 from isaaclab.app import AppLauncher
 app=AppLauncher({'headless':True,'device':'cuda:0','multi_gpu':False}).app
 import torch
 from native_stand.students.evaluator import environment
 from native_stand.students.model import load
 from native_stand.students.data import sha
 started=time.monotonic();env=None;status='FAILED';evidence=[];failure=None
 try:
  actor,_=load(s['source_checkpoint'],'MotionZeroStudent');actor=actor.to('cuda:0').eval()
  env=environment(s,'MotionZeroStudent',envs=64);env.curriculum_training=True
  if s.get('motion_timing',{}).get('configuration',{}).get('wheel_execution'):
   from native_stand.control_readback import tensor
   view=env.robot.root_view;names=env.robot.body_names
   ids=[names.index(side+'_wheel') for side in ('left','right')]
   write('wheel_physics_readback.json',{'model_body_names':[names[i] for i in ids],
    'geometric_sides':['right','left'],
    'body_inertia_backend':torch.as_tensor(tensor(view.get_inertias()))[0,ids].tolist(),
    'body_com_backend':torch.as_tensor(tensor(view.get_coms()))[0,ids].tolist(),
    'body_mass_backend_kg':torch.as_tensor(tensor(view.get_masses()))[0,ids].tolist(),
    'execution':env.motion_timing.wheel_readback,
    'scope':'backend values at setup; inertia tensor frame must be checked before treating an entry as wheel-axis J'})
  anti=s.get('motion_timing',{}).get('version')=='motion31_standing_antiosc_armature_v4'
  levels=list(range(5))+[None] if anti else [-1,3]
  steps_per_level=50 if anti else 100;actual_steps=0
  from native_stand.students.motion_tracking_reward import effective_reward_fields
  for level in levels:
   if anti and level is None:env.motion_timing.stage=4
   env.motion_timing.configure(level);obs,_=env.reset()
   before=env.motion_timing.summary()
   for step in range(steps_per_level):
    if time.monotonic()-started>180:raise TimeoutError('bounded timing probe')
    with torch.inference_mode():obs,reward,terminated,truncated,extras=env.step(actor(obs))
    if obs['student_actor'].shape!=(64,31) or obs['student_critic'].shape!=(64,s['observation']['critic_dim']):raise ValueError('actual timing IO differs')
    if not torch.isfinite(obs['student_critic']).all():raise ValueError('actual timing input nonfinite')
   actual_steps+=steps_per_level*10
   if anti:
    result=env.student_reward_result
    if tuple(result['fields'])!=tuple(effective_reward_fields(s)) or not torch.isfinite(result['raw']).all():raise ValueError('actual new reward schema or finite readback differs')
   summary=env.motion_timing.summary()
   if not summary['finite'] or summary['physics_ticks']<steps_per_level*10:raise ValueError('timing clock/readback invalid')
   if level==3 and (summary['delay_ms_max'][2]!=7 or summary['delay_ms_max'][5]!=11):raise ValueError('geometric wheel delay readback differs')
   evidence.append({'override':level,'before':before,'after':summary,'actor_dim':31,'critic_dim':s['observation']['critic_dim'],
    'observed_reward_nonzero':bool((reward!=0).any()),'actual_reward_fields':list(env.student_reward_result['fields']),'reward_raw_mean':env.student_reward_result['raw'].mean(0).cpu().tolist(),'status':'ENGINEERING_PASSED_NOT_CAPABILITY'})
  status='COMPLETED'
 except BaseException as exc:
  failure={'type':type(exc).__name__,'message':str(exc)};raise
 finally:
  write('terminal_report.json',{'status':status,'failure':failure,'evidence':evidence,'PPO':0,
    'physical_steps':actual_steps if status=='COMPLETED' else None,'wall_s':time.monotonic()-started,
    'spec_sha256':sha(a.spec),'source_sha256':s['source_sha256'],'capability_certified':False})
  if env is not None:env.close()
  try:app.close()
  except SystemExit:
   if status!='COMPLETED':raise SystemExit(2)
 print(str(a.output_dir.resolve()))
if __name__=='__main__':main()
