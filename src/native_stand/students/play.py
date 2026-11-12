"""Explicit student preview via shared environment; never launches by default."""
import json,time,os,sys
from pathlib import Path
from .data import sha
from .requests import validate

MOTION_TRAINING_VERSION='student_motion31_training_v1'
MOTION_ROLES=('MotionStudent','MotionZeroStudent')

def independent_preview_action(env,actor,obs):
 """Match independent PPO0: no actor requests during passive initialization."""
 import torch
 from native_stand.recovery.domain import Mode
 controlled=(env.fsm.mode!=int(Mode.INITIALIZING))&env.initialization_checked
 if hasattr(env,'actor_valid_pre'):controlled=env.actor_valid_pre()
 action=torch.zeros(env.num_envs,6,device=env.device)
 if controlled.any():
  action[controlled]=actor({k:v[controlled] for k,v in obs.items()},stochastic_output=False)
 return action

def motion_declaration(spec):
 """Explicit Motion manual-preview scope; None keeps every other spec on its own path."""
 training=spec.get('student_training') or {}
 if training.get('version')!=MOTION_TRAINING_VERSION:return None
 if not spec.get('motion_training'):
  raise ValueError('Motion training spec without the frozen motion_training stage declaration')
 return training

def motion_preview(args,spec,training):
 """Keyboard/panel preview of a Motion student; the checkpoint role is read, never inferred."""
 from .model import load
 from . import motion_panel
 panel=bool(getattr(args,'panel',False)) or motion_panel.enabled(False)
 panel_port=motion_panel.port_number(getattr(args,'panel_port',None))
 if not args.preview:raise ValueError('Motion preview requires --preview: no ledger or output writes')
 if getattr(args,'handoff_condition',None) is not None:
  raise ValueError('Motion preview never uses a teacher/handoff condition')
 actor,data=load(args.checkpoint)
 role=data.get('role')
 if role not in MOTION_ROLES:raise ValueError('checkpoint role is not a Motion student: %r'%(role,))
 if data.get('spec_sha256')!=sha(args.spec):raise ValueError('Motion preview checkpoint spec binding differs')
 if data.get('asset_sha256')!=spec['asset_sha256']:raise ValueError('Motion preview checkpoint asset binding differs')
 if data.get('recipe_version')!=spec['student_training']['version']:raise ValueError('Motion preview recipe version differs')
 if data.get('interface_version')!=spec['interface_version']:raise ValueError('Motion preview interface version differs')
 from native_stand.command_v6.tools.experiment_preparation import verify_publication
 from .motion_preview import plan as preview_plan,run as preview_run
 bound={**data,'checkpoint_sha256':sha(args.checkpoint)}
 result=preview_plan(args,spec,bound,panel=panel,panel_port=panel_port,role=role)
 # Registration legitimately replaces spec/ledger against its exact receipt; same strict check as the Recovery path.
 result['publication']=verify_publication(args.spec,authorized=True)
 if not args.run:print(json.dumps(result,indent=2));return
 if args.physical_gpu not in (0,1):raise ValueError('Explicit physical GPU0 or GPU1 required')
 if spec.get('device',{}).get('physical_index')!=args.physical_gpu:raise ValueError('preview physical GPU differs from declared spec')
 return preview_run(args,spec,role,actor,bound,panel=panel,panel_port=panel_port)

def execute(args,spec):
 errors=validate(spec)
 if errors:raise ValueError(errors)
 motion=motion_declaration(spec)
 if motion is not None:return motion_preview(args,spec,motion)
 family=getattr(args,'preview_family',None) or spec['recovery']['active_family']
 independent=[r for r in spec['student_evaluation']['cases'] if r['kind']=='student' and r['family']==family]
 diagnostic=getattr(args,'diagnostic_initialization',False)
 if diagnostic:
  if not getattr(args,'preview_family',None):raise ValueError('diagnostic initialization requires explicit preview-family')
  library_path=spec['recovery']['pose_library']
  if sha(library_path)!=spec['recovery']['pose_library_sha256']:raise ValueError('preview pose library SHA differs')
  library=json.loads(Path(library_path).read_text())
  from .requests import evaluation_recovery
  evaluation_recovery(spec['recovery'],library['poses'],family)
  if not independent:
   baseline=next((r for r in spec['student_evaluation']['cases'] if r['kind']=='student'),None)
   if baseline is None:raise ValueError('student diagnostic baseline absent')
   independent=[{**baseline,'family':family}]
 row=next((r for r in independent if r['checkpoint']==str(args.checkpoint.resolve())),None)
 if row is None:
  from .model import load
  _,candidate=load(args.checkpoint,'RecoveryStudent')
  if candidate.get('spec_sha256')!=sha(args.spec) or candidate.get('asset_sha256')!=spec['asset_sha256'] or candidate.get('recipe_version')!=spec.get('student_training',{}).get('version') or candidate.get('interface_version')!=spec['interface_version']:
   raise ValueError('Student preview candidate spec/asset/recipe/interface identity differs')
  row=next(iter(independent),None)
  if row is None:raise ValueError('Student preview family not declared')
  row={**row,'checkpoint':str(args.checkpoint.resolve()),'sha256':sha(args.checkpoint)}
 if sha(args.checkpoint)!=row['sha256']:raise ValueError('Student preview checkpoint SHA differs')
 if not args.preview or args.handoff_condition is not None or args.keyboard:raise ValueError('Student preview must be independent, without teacher/keyboard override')
 result={'interface_version':spec['interface_version'],'checkpoint':str(args.checkpoint),'checkpoint_sha256':sha(args.checkpoint),'initial_family':row['family'],'teacher_control':False,'teacher_prefix_intervals':0,'simulation':0,'online_PPO':0}
 result.update(diagnostic_initialization=diagnostic,capability_certification=False,
  assessment_family_declared=any(r['family']==family for r in spec['student_evaluation']['cases']),
  target_height_m=row.get('command_height_m',spec.get('student_training',{}).get('target_height_m')))
 from . import motion_panel
 panel=motion_panel.enabled(bool(getattr(args,'panel',False)))
 panel_port=motion_panel.port_number(getattr(args,'panel_port',None))
 result['panel']={'enabled':panel,'mode':'Recovery','url':f'http://127.0.0.1:{panel_port}/' if panel else None}
 from native_stand.command_v6.tools.experiment_preparation import verify_publication
 # Registration legitimately replaces spec/ledger against its exact receipt.
 # Perform the same strict identity check in CPU preview and GUI execution.
 result['publication']=verify_publication(args.spec,authorized=True)
 if not args.run:print(json.dumps(result,indent=2));return
 if args.physical_gpu not in (0,1):raise ValueError('Explicit physical GPU0 or GPU1 required')
 # Preview device is explicit runtime intent, independent of training placement.
 # Keep the original spec/checkpoint/publication identities unchanged.
 from native_stand.command_v6.domain.gui_device import resolve
 device_plan=resolve(args.physical_gpu,os.environ.get('CUDA_VISIBLE_DEVICES'))
 os.environ['CUDA_VISIBLE_DEVICES']=str(args.physical_gpu);sys.path.insert(0,spec['asset_module_root'])
 from isaaclab.app import AppLauncher
 app=AppLauncher({**device_plan['app_options'],'headless':False,'visualizer':['kit'],'visualizer_explicit':True}).app;env=None;server=None
 try:
  from .evaluator import environment
  from .model import load
  import torch
  env=environment(spec,row['role'],envs=1,initial_family=row.get('family'),autonomous_preview=True);env.student_yaw_amplitude=row['yaw_rad'];env.select_initial_family(row['family'],'holdout')
  if row.get('command_height_m') is not None:env.student_command_height=row['command_height_m']
  actor,_=load(row['checkpoint'],row['role']);actor=actor.to(env.device).eval()
  torch.manual_seed(row['seed'])
  obs,_=env.reset();count=0;start=time.monotonic()
  from .motion_preview import PreviewTiming,_telemetry
  from types import SimpleNamespace
  timing=PreviewTiming(env.step_dt,start) if panel else None;resets=0;panel_plan=None
  if panel:
   panel_plan=motion_panel.Plan(1.,0.,mode='Recovery')
   server,_=motion_panel.start(panel_plan,panel_port)
   print(f'Recovery panel: http://127.0.0.1:{panel_port}/',flush=True)
  from native_stand.command_v6.tools.ground_visual import setup,camera_view
  try:
   import omni.usd
   result.update(setup(omni.usd.get_context().get_stage(),env.sim,style=args.ground_style,
     extent_m=args.ground_extent_m,grid_spacing_m=args.ground_grid_spacing_m))
  except Exception as error:
   print(f'[Visual Notice] Visual setup skipped: {error}',flush=True)
  camera_view(env.sim)
  print('GUI running: independent %s; close window or Ctrl-C.' % row['role'],flush=True)
  with torch.inference_mode():
   while app.is_running() and (not args.max_policy_steps or count<args.max_policy_steps) and time.monotonic()-start<args.max_wall_seconds:
    if panel_plan is not None:
     command=panel_plan.take()
     if command['stop']:break
     if command['reset']:obs,_=env.reset();resets+=1
    obs,*_=env.step(independent_preview_action(env,actor,obs));count+=1
    if panel_plan is not None and count%2==0:
     panel_plan.publish({**_telemetry(env,SimpleNamespace(speed=0.,yaw_speed=0.),count,resets,'Recovery autonomous',panel_plan),**timing.sample(count,time.monotonic())})
    if count%200==0:
     root=env.robot.data.root_pos_w.torch[0].cpu().tolist()
     camera_view(env.sim,root[:2])
 finally:
  original=sys.exception()
  if server is not None:
   try:motion_panel.stop(server)
   except BaseException:
    if original is None:raise
  for obj in (env,app):
   if obj:
    try:obj.close()
    except BaseException:
     if original is None:raise
