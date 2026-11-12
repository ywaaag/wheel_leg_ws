"""31D Motion adapter behind validate_native_mujoco; reference MJCF stays read-only."""
import json,time
from pathlib import Path
import numpy as np
import torch
import mujoco
from native_stand.students.data import sha


PRIVILEGED_VERSIONS=('motion_privileged49_v1','motion_spin_privileged61_v1','motion_symmetry_privileged85_v1',
 'motion_timing_privileged110_v1','motion_antiosc_privileged140_v1','motion_height_robust_privileged184_v1',
 'motion_mcu_motorqueue_privileged312_v1','motion_standing_delay_privileged411_v1')
ARM_LEVELS=(.35,.11,.04,.02,.012)


class OnnxActor:
 """Read-only ONNX actor for the same explicit 31D->6D contract.

 The export manifest is required: it binds the graph SHA, the parent spec SHA, the
 distilled student and the teacher it was exported from, so a preview cannot silently
 load an ONNX file that belongs to a different recipe.
 """
 def __init__(self,path,manifest_path,spec_path):
  import onnx
  from onnx.reference import ReferenceEvaluator
  self.kind='onnx';self.path=Path(path);self.manifest_path=Path(manifest_path)
  self.manifest=json.loads(self.manifest_path.read_text())
  # Two export schemas are accepted and each one must still bind the graph itself.
  expected_onnx=self.manifest.get('actor_onnx_sha256') or (self.manifest.get('onnx') or {}).get('sha256')
  if not expected_onnx:raise ValueError('ONNX manifest records no graph SHA')
  if sha(self.path)!=expected_onnx:raise ValueError('ONNX actor SHA does not match its manifest')
  expected_spec=self.manifest.get('spec_sha256')
  self.spec_identity='manifest spec_sha256'
  if expected_spec is None:
   # actor_onnx_export_v1 names the source checkpoint instead of the parent spec; the
   # checkpoint stores the spec SHA that the loader already enforces.
   source=Path((self.manifest.get('checkpoint') or {}).get('path',''))
   if source.is_file():
    expected_spec=torch.load(source,map_location='cpu',weights_only=False).get('spec_sha256')
    self.spec_identity='source checkpoint spec_sha256'
  if expected_spec is None:self.spec_identity='UNVERIFIED (manifest records no spec identity)'
  elif expected_spec!=sha(spec_path):raise ValueError('ONNX actor was exported from a different spec')
  graph=onnx.load(str(self.path))
  dims=[d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
  out=[d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim]
  if dims[-1]!=31 or out[-1]!=6:raise ValueError('ONNX actor must map 31D observation to 6D mean action')
  self.input=graph.graph.input[0].name;self.output=graph.graph.output[0].name
  self.engine=ReferenceEvaluator(str(self.path))
  self.network_architecture=self.manifest.get('architecture')
 def mean(self,obs,name=None):
  out=self.engine.run([self.output],{self.input:np.asarray(obs.detach().numpy() if hasattr(obs,'detach') else obs,dtype=np.float32)})[0]
  return torch.from_numpy(np.asarray(out,dtype=np.float32))
 def state_dict(self):return {}
 def eval(self):return self


class ActorSet:
 """Explicit named models sharing one plant; the panel picks which one drives the actions."""
 def __init__(self,actors,records,default):
  self.actors=dict(actors);self.records=list(records);self.active=default
  self.network_architecture=getattr(self.actors[default],'network_architecture',None)
 def names(self):return list(self.actors)
 def mean(self,obs,name=None):
  key=name or self.active
  if key not in self.actors:raise ValueError('unknown preview model %r' % (key,))
  self.active=key
  return self.actors[key].mean(obs)
 def record(self,name=None):
  key=name or self.active
  return next(r for r in self.records if r['name']==key)
 def snapshot(self):return {n:{k:v.clone() for k,v in a.state_dict().items()} for n,a in self.actors.items()}
 def unchanged(self,snap):
  return all(torch.equal(v,a.state_dict()[k]) for n,a in self.actors.items() for k,v in snap[n].items())
 def __getattr__(self,name):
  if name.startswith('_') or name=='actors':raise AttributeError(name)
  return getattr(self.actors[self.active],name)


def model_specs(args):
 """NAME=PATH pairs from --models, else the single --checkpoint named after its file."""
 items=getattr(args,'models',None)
 if items:
  rows=[]
  for item in items:
   if '=' not in item:raise ValueError('--models entries must be NAME=PATH')
   name,path=item.split('=',1);name=name.strip();path=path.strip()
   if not name or not path:raise ValueError('--models entries must be NAME=PATH')
   if name in [r[0] for r in rows]:raise ValueError('--models names must be unique')
   if not Path(path).exists():raise ValueError(f'--models path missing: {path}')
   rows.append((name,Path(path)))
  return rows
 return [(Path(args.checkpoint).stem,Path(args.checkpoint))]


def panel_host(args):
 """Explicit bind address; loopback unless the caller asks for a LAN interface."""
 return str(getattr(args,'panel_host','127.0.0.1'))


def actuator_cfg(args,spec):
 """Spec-declared execution chain, chosen by explicit version.

 ``mcu_execution.profile.version`` selects the post-PD MCU chain and its own
 configuration; every other spec keeps the legacy motion_timing hypothesis and
 the legacy 60/2/1 no-delay preview (None) so old playbacks keep their physics.
 """
 from motion_mcu_playback import configuration,declared,forced
 execution=declared(spec)
 if execution is not None:return configuration(args,execution,spec)
 if getattr(args,'mcu_execution',False):
  # Explicit user override: same recipe/weights/plant start, MCU execution chain instead.
  return configuration(args,forced(spec),spec)
 armature=getattr(args,'leg_armature_m',None)
 if armature is None:return None
 timing=spec.get('motion_timing',{}).get('configuration')
 if not timing:raise ValueError('--leg-armature-m needs a spec with motion_timing.configuration')
 levels=timing.get('armature_levels_kg_m2') or list(ARM_LEVELS)
 if float(armature) not in [float(v) for v in levels]:raise ValueError(f'leg armature must be one declared level {levels}')
 wheel=timing.get('wheel_execution') or {}
 total=wheel.get('wheel_total_J_candidate_kg_m2_ch2_ch5')
 declared={'transport_ms':[int(v) for v in timing['target_delay_ms_ch0_ch5']],'feedback_ms':int(timing['PD_feedback_ms']),
  'tau_ms':[int(v) for v in timing['torque_tau_ms_ch0_ch5']],
  'actor_sensor_ms':int((timing.get('actor_sensor') or {}).get('sample_period_ms',10))}
 cfg={'leg_armature':float(armature),'kp':float(timing['leg_kp']),'kd':float(timing['leg_kd']),
  'kv':float(timing['wheel_kv']),'wheel_peak':float(wheel.get('wheel_peak_Nm',spec['control']['effort_limits_nm'][2])),
  'wheel_J':[float(v) for v in total] if total else None,'wheel_B':[0.,0.],'wheel_C':None,**declared}
 if not getattr(args,'zero_latency',False):return cfg
 # Explicit diagnostic: keep the spec armature/PD but play the plant with no delay anywhere.
 cfg.update(zero_latency=True,declared=declared,transport_ms=[0]*6,feedback_ms=0,tau_ms=[0]*6,actor_sensor_ms=0)
 return cfg


def initial_target(model,data,refs,leg_index):
 """Six-channel target cache at start: hold the current leg pose, zero wheel speed."""
 target=np.zeros(6);target[leg_index]=np.asarray(data.qpos[refs['qpos']],dtype=float)[leg_index]
 return target


def plan(args,spec):
 from native_stand.students.model import load
 from native_stand.students.mcu_interface import contract
 from validate_native_mujoco import binding,state,JOINTS
 if spec.get('interface_version')!='multiskill_mcu_v2' or spec.get('student_training',{}).get('version')!='student_motion31_training_v1':raise ValueError('Explicit31D Motion training interface required')
 if not 0<args.wall_seconds<=1800:raise ValueError('bounded wall0..1800 required')
 if getattr(args,'zero_latency',False):
  if getattr(args,'leg_armature_m',None) is None:raise ValueError('--zero-latency needs an explicit --leg-armature-m')
  if not getattr(args,'read_only',False):raise ValueError('--zero-latency is a read-only preview diagnostic; drop --output-dir')
 expected_critic=spec['student_training'].get('critic_privileged_version')
 specs=model_specs(args);actors={};records=[];primary=None
 # An ONNX export is a distilled Actor only: the Critic is never loaded, so a recipe whose
 # Critic semantics this preview does not implement cannot leak into the preview. Checkpoints
 # still have to name a Critic this code knows.
 onnx_only=bool(specs) and all(Path(p).suffix.lower()=='.onnx' for _,p in specs)
 if expected_critic not in PRIVILEGED_VERSIONS and not onnx_only:raise ValueError('explicit Motion Critic semantics required')
 for name,path in specs:
  if path.suffix.lower()=='.onnx':
   manifest_path=Path(getattr(args,'onnx_manifest',None) or path.with_name('manifest.json'))
   if not manifest_path.exists():raise ValueError(f'ONNX model {name} needs its export manifest.json or --onnx-manifest')
   actor=OnnxActor(path,manifest_path,args.spec)
   records.append({'name':name,'kind':'onnx','path':str(path),'sha256':sha(path),
    'manifest_sha256':sha(manifest_path),'teacher_sha256':actor.manifest.get('teacher_sha256'),
    'spec_identity':actor.spec_identity,'source_checkpoint':(actor.manifest.get('checkpoint') or {}).get('path'),
    'student_sha256':actor.manifest.get('student_sha256'),'architecture':actor.network_architecture})
   actors[name]=actor
   if primary is None:primary=(name,actor,{'distilled':True,'critic_version':None})
   continue
  actor,saved=load(path,'MotionZeroStudent');actor.eval()
  distilled=saved.get('distillation') is not None
  if expected_critic=='motion_symmetry_privileged85_v1':
   from native_stand.students.motion_symmetry import network
   if spec.get('network_architecture')!=network() or (not distilled and saved.get('network_architecture')!=network()):raise ValueError('explicit Motion85 Actor architecture differs')
  if distilled:
   from native_stand.students.motion_distillation_pipeline import validate_student
   validate_student(saved,args.spec)
  elif saved.get('spec_sha256')!=sha(args.spec) or saved.get('asset_sha256')!=spec['asset_sha256'] or saved.get('critic_privileged_version')!=expected_critic:raise ValueError('Exact trained checkpoint/spec/asset required')
  records.append({'name':name,'kind':'checkpoint','path':str(path),'sha256':sha(path),
   'distilled':bool(distilled),'critic_version':None if distilled else expected_critic,
   'network_architecture':getattr(actor,'network_architecture',None)})
  actors[name]=actor
  if primary is None:primary=(name,actor,{'distilled':distilled,'critic_version':None if distilled else expected_critic,'saved':saved})
 default_name=specs[0][0]
 actor=ActorSet(actors,records,default_name)
 distilled=bool(primary[2]['distilled']);saved=primary[2].get('saved') or {}
 checkpoint_path=Path(specs[0][1])
 for later in records[1:]:
  if later.get('teacher_sha256') and records[0]['kind']=='checkpoint' and later['teacher_sha256']!=records[0]['sha256']:
   later['teacher_matches_primary']=False
  elif later.get('teacher_sha256'):later['teacher_matches_primary']=later['teacher_sha256']==records[0]['sha256']
 records[0]['default']=True
 if sha(spec['asset_path'])!=spec['asset_sha256']:raise ValueError('USD changed')
 cfg=actuator_cfg(args,spec)
 control=spec['control']
 if [control[k] for k in ('physics_dt_s','pd_dt_s','policy_dt_s')]!=[.001,.005,.01] or control['effort_limits_nm']!=[40,40,5.5,40,40,5.5]:raise ValueError('Control drift')
 if cfg is None and (control['leg_kp'],control['leg_kd'],control['wheel_kv'])!=(60.,2.,1.):raise ValueError('Control drift')
 if cfg is not None and (control['leg_kp'],control['leg_kd'],control['wheel_kv'])!=(cfg['kp'],cfg['kd'],cfg['kv']):raise ValueError('spec control and motion_timing PD differ')
 if spec['action']['leg_scale_rad']!=1.5 or spec['action']['wheel_scale_rad_s']!=100. or spec['action']['leg_pair_radius_rad']!=0 or spec['action']['order']!=list(JOINTS):raise ValueError('Action drift')
 commands=args.commands or [0.,.5,1.,1.5,2.]
 if len(set(commands))!=len(commands) or any(not np.isfinite(c) or abs(c)>4 for c in commands):raise ValueError('Invalid commands')
 yaw_commands=getattr(args,'yaw_commands',None) or []
 if len(set(yaw_commands))!=len(yaw_commands) or any(not np.isfinite(v) or v==0 or abs(v)>4 for v in yaw_commands):raise ValueError('Invalid yaw targets')
 command_matrix=[[v,0.] for v in commands]+[[0.,w] for w in yaw_commands]
 if getattr(args,'keyboard',False) or getattr(args,'panel',False):
  from native_stand.students.mujoco_keyboard import MotionKeyboard
  MotionKeyboard(args.max_speed,args.max_yaw)
  if not args.viewer:raise ValueError('manual keyboard needs viewer')
  if getattr(args,'panel',False):
   import ipaddress
   from native_stand.students.motion_panel import port_number
   port_number(args.panel_port)
   ipaddress.IPv4Address(panel_host(args))
  command_matrix=[[0.,0.]]
 model=mujoco.MjModel.from_xml_path(str(args.xml));refs=binding(model);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
 from native_stand.students.mujoco_solver import apply as apply_solver,resolve as resolve_solver
 from motion_mcu_playback import plant_profile as select_plant,apply_current_plant
 solver_version=resolve_solver(spec,getattr(args,'solver_profile',None))
 requested_plant=getattr(args,'plant_profile',None)
 from motion_v4_plant import selected as v4_selected
 v4_plant=v4_selected(requested_plant)
 plant_version=select_plant(spec,solver_version,None if v4_plant else requested_plant)
 profile=None;plant_readback={'version':plant_version}
 if plant_version=='motion_mcu_currentplant_v1':
  plant_readback=apply_current_plant(model,refs,spec,cfg);profile=plant_readback['actuator']
 elif cfg is not None:
  from compare_motion_timing_profiles import LEGS as PROFILE_LEGS,apply_actuator_profile
  original={k:getattr(model,k).copy() for k in ('dof_armature','dof_damping','dof_frictionloss')}
  profile=apply_actuator_profile(model,refs,cfg,original)
  profile['leg_armature_kg_m2']=[float(model.dof_armature[int(refs['dofs'][i])]) for i in PROFILE_LEGS]
 if getattr(args,'leg_physics','spec')=='bilateral_modal_v1':
  from motion_mcu_playback import bilateral_leg_physics,leg_response_contract
  plant_readback['leg_physics']=bilateral_leg_physics(model,refs,leg_response_contract())
 solver_profile=apply_solver(model,refs,solver_version)
 if v4_plant:
  from motion_v4_plant import apply as apply_v4, contract as v4_contract
  data_v4,v4_meta=v4_contract()
  plant_readback['v4']=apply_v4(model,refs,spec,cfg,data_v4)
  plant_readback['v4']['contract_sha256']=v4_meta['sha256']
  plant_readback['v4']['contract_pinned']=v4_meta['pinned']
  plant_readback['v4']['contract_path']=v4_meta['path']
 if cfg is not None:
  cfg['solver_profile']=solver_version;cfg['plant_profile']=('motion_v4_identified_v1' if v4_plant else plant_version)
  if v4_plant:cfg['v4_plant']=plant_readback['v4']
 angular,gravity,initial=state(model,data,refs)
 if model.nu!=6 or model.neq!=4 or not np.isclose(model.opt.timestep,.001) or initial['closure_m']>1e-6 or not np.allclose(gravity,[0,-1,0],atol=1e-6):raise ValueError('MJCF topology/frame invalid')
 limits=np.array(spec['control']['effort_limits_nm']);ctrl=limits/np.abs(refs['gears'])
 if v4_plant:
  # The v4 plant declares its own 2.447 Nm wheel motor cap; the spec keeps the pre-v4 5.5 Nm
  # reference limit, which is reported instead of silently re-imposed on the actuator.
  from motion_v4_plant import MOTOR_CAP_NM
  ctrl=np.minimum(ctrl,MOTOR_CAP_NM/np.abs(refs['gears']))
 if np.any(ctrl>np.min(np.abs(model.actuator_ctrlrange[refs['actuators']]),axis=1)):raise ValueError('Actuator range insufficient')
 from native_stand.students.mcu_interface import encode
 import xml.etree.ElementTree as ET
 inputs={str(p.resolve()):sha(p) for p in [args.spec,checkpoint_path,args.xml,Path(__file__),Path(__file__).with_name('validate_native_mujoco.py'),Path(spec['asset_path'])]+[Path(r['path']) for r in records[1:]]}
 inputs.update(plant_readback.get('input_sha256',{}))
 inputs[str(Path(__file__).with_name('motion_mcu_playback.py').resolve())]=sha(Path(__file__).with_name('motion_mcu_playback.py'))
 tree=ET.parse(args.xml);meshdir=tree.getroot().find('compiler').get('meshdir','')
 for mesh in tree.findall('./asset/mesh'):
  p=args.xml.parent/meshdir/mesh.get('file');inputs[str(p.resolve())]=sha(p)
 from pxr import Usd,UsdGeom,UsdPhysics
 usd=Usd.Stage.Open(spec['asset_path']);cache=UsdGeom.XformCache();parity=[]
 for body in ['robot','left_wheel','right_wheel','left_upper_leg','right_upper_leg']:
  prim=next(p for p in usd.Traverse() if p.GetName()==body and p.HasAPI(UsdPhysics.RigidBodyAPI))
  i=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body)
  parity.append({'body':body,'zero_pose_position_error_m':float(np.linalg.norm(np.array(cache.GetLocalToWorldTransform(prim).ExtractTranslation())-data.xpos[i])),
      'usd_mass':float(UsdPhysics.MassAPI(prim).GetMassAttr().Get()),'mj_mass':float(model.body_mass[i])})
 poses=json.loads(Path(spec['recovery']['pose_library']).read_text())['poses'];pose=next(p for p in poses if p['family']=='motion_native_normal')
 if abs(initial['height']-pose['root_height_m'])>1e-4 or not np.allclose(data.qpos[refs['qpos']],0,atol=1e-6):raise ValueError('Normal pose origin mismatch')
 declared_initial=None
 from motion_mcu_playback import start_pose
 start=start_pose(spec,getattr(args,'init_height_m',None))
 if start is None:
  declared_initial={'source':'recipe / MJCF default start','height_m':float(initial['height']),
   'requested_height_m':getattr(args,'init_height_m',None),
   'note':'no verified standing pose at the requested height; a pose is never scaled or clipped to fit'}
 else:
  # The playback start pose replaces the zero-joint origin. The zero-pose parity and the
  # normal-pose origin check above already ran on the unmodified MJCF, so this cannot hide an
  # asset/pose mismatch; the closed-chain and height readback below still apply.
  from motion_mcu_playback import apply_initial_pose
  joints=apply_initial_pose(model,data,refs,{'pose':start})
  mujoco.mj_forward(model,data)
  angular,gravity,initial=state(model,data,refs)
  limit=float(start.get('closure_max_m') or 0.)
  if initial['closure_m']>limit+1e-6:raise ValueError('playback start pose closure differs')
  if abs(initial['height']-float(start['height_m']))>1e-3:raise ValueError('playback start pose height differs')
  declared_initial={'source':start['source'],'joint_pos':joints.tolist(),'height_m':float(start['height_m']),
   'root_quaternion_xyzw':[float(v) for v in start['root_quaternion_xyzw']],
   'closure_m':float(initial['closure_m']),'closed_loop_verified':True,
   'requested_height_m':start.get('requested_height_m'),
   'not_used':'asset zero-joint origin is not a playback initial state'}
 q=torch.tensor(data.qpos[refs['qpos']],dtype=torch.float32)[None];dq=torch.tensor(data.qvel[refs['dofs']],dtype=torch.float32)[None]
 height_cmd=float(getattr(args,'target_height_m',.34))
 reference=torch.zeros(1,6)
 previous=torch.zeros(1,6)
 if declared_initial is not None and 'joint_pos' in declared_initial:
  # The Actor's carried target is the executed/adopted start pose, not a zero request.
  previous[0,[0,1,3,4]]=torch.tensor(declared_initial['joint_pos'],dtype=torch.float32)[[0,1,3,4]]
 obs=encode(torch.tensor(angular,dtype=torch.float32)[None],torch.tensor(gravity,dtype=torch.float32)[None],q,dq,previous[:,[0,1,3,4]],previous[:,[2,5]],reference,'Motion',torch.tensor([[0.,0.,height_cmd]]))
 with torch.no_grad():mean=actor.mean(obs)
 if obs.shape!=(1,31) or not torch.isfinite(mean).all():raise ValueError('Initial Actor invalid')
 from native_stand.students import mujoco_solver
 inputs[str(Path(mujoco_solver.__file__).resolve())]=sha(mujoco_solver.__file__)
 manifest={'version':'motion31_mujoco_transfer_v1','checkpoint_sha256':sha(checkpoint_path),'input_sha256':inputs,'solver_profile':solver_profile,'plant_profile':plant_readback,
  'models':records,'default_model':default_name,
  'actor_contract':contract(),'actor_network_architecture':getattr(actor,'network_architecture',None),'distilled_student':saved.get('distillation'),'critic_not_loaded':True,'checkpoint_critic_version':None if distilled else expected_critic,'height_command_m':height_cmd,'actor_frame':'CAD body; no upright transform in Actor',
  'velocity':'root body COM; vx=-CAD_z,vy=-CAD_x,yaw=CAD_y',
  'initial_actor_input':obs[0].tolist(),'initial_mean':mean[0].tolist(),'initial_state':initial,'zero_pose_parity':parity,
  'protocol':{'commands':command_matrix,'controlled_seconds':30.,'prepare_s':2.,'ramp_s':8.,'hold_s':[10.,18.],'return_s':8.,'parking_s':[26.,30.],
   'height_band':[height_cmd-.04,height_cmd+.04],'contact_threshold_N':2.,'contact_rate_min':.95,'pitch_max':.1,'roll_max':.05,'zero_horizontal_p95':.3,
   'nonzero_MAE_max':.3,'direction_min':.95,'gain_range':[.8,1.2],'parking':'diagnostic only','envs_per_command':1},
  'differences':{'MJ_ground_friction':model.geom_friction[refs['ground']].tolist(),'native_friction':[.5,.5,0],
   'MJ_active_damping':model.dof_damping[refs['dofs']].tolist(),'MJ_active_armature':model.dof_armature[refs['dofs']].tolist(),
   'collision_solver':solver_version+'; reference collision proxies; not PhysX/USD parity certified','initialization':pose.get('initialization_control','passive_settle')+'; exact normal zero geometry/velocities; engine collider/solver differences retained'},
  'simulation':0,'online_PPO':0,'certification':False,'wall_limit_s':args.wall_seconds}
 if declared_initial is not None:manifest['declared_initial_state']=declared_initial
 manifest['playback_start']=start
 if cfg is not None:
  manifest.update(actuator_configuration=cfg,physical_readback=profile,
   actuator_profile_note='spec motion_timing.configuration applied as transport/feedback/tau; leg armature is the explicit --leg-armature-m course level, never inferred from training state')
  if cfg.get('mcu'):
   from motion_mcu_playback import summary as mcu_summary
   manifest['mcu_execution']={**mcu_summary(cfg['mcu_version'],cfg['profile']),'leg_armature_kg_m2':cfg['leg_armature'],
   'armature_source':cfg['armature_source'],'forced_by_cli':bool(cfg.get('forced_by_cli')),
   'initial_state':(declared_initial or {}).get('source'),
    'actor_age_ms':cfg['actor_sensor_ms'],'PD_age_ms':cfg['feedback_ms'],
    'transport_ms':cfg['transport_ms'],'tau_ms':cfg['tau_ms'],
    'execution_source':cfg['execution_source'],'plant_source':cfg['plant_source'],
    'scope':'declared firmware contract and timing hypotheses; not identified hardware latency, capability, spring or measured LQR',
    'not_applied':'the 7/11 ms wheel array is not also applied to pre-PD targets; AuditExecution keeps that separate historical hypothesis'}
   manifest['actuator_profile_note']=('execution forced by --mcu-execution on a spec that declares no MCU chain; the recipe keeps its own initial state and armature, so this is a chain-only comparison and not a capability claim'
   if cfg.get('forced_by_cli') else 'execution from spec mcu_execution.profile; armature/J are plant properties and the leg armature is the training-fixed MCU level, never inferred from training state')
  if getattr(args,'leg_response','spec')=='bilateral_modal_v1':
   from motion_mcu_playback import leg_response_contract
   manifest['leg_response']=leg_response_contract()
   manifest['differences']['leg_chain']=('legs: per-side modal (common,differential) transport and first-order '
    'response from the delivered bilateral handoff; wheels keep the spec chain; PD40/2, +-40Nm and the plant '
    'inertia/spring/closed chain are unchanged, so J/B/K are not overlaid')
  if cfg.get('zero_latency'):
   manifest['latency_override']={'applied':True,'source':'--zero-latency (user-requested diagnostic)',
    'spec_declared':cfg['declared'],'played':{'transport_ms':[0]*6,'feedback_ms':0,'tau_ms':[0]*6,'actor_sensor_ms':0},
    'unchanged':{'leg_armature':cfg['leg_armature'],'kp':cfg['kp'],'kd':cfg['kd'],'kv':cfg['kv']},
    'scope':'latency-free playback of the same weights; not measured hardware latency and not a capability claim'}
 if getattr(args,'keyboard',False) or getattr(args,'panel',False):
  manifest.update(mode='manual_keyboard_preview',capability_certified=False,keyboard={'max_speed_mps':args.max_speed,'max_yaw_rad_s':args.max_yaw,'step':.25,'keys':'W/S vx; A/D yaw; Space zero; R reset; close window exits','known_scope':'native source verified forward0..2 and yaw±.5 only; reverse/combined commands unverified'})
  manifest['protocol']={'mode':'manual_keyboard_preview','commands':command_matrix,'scoring':'NOT_RUN; manual varying commands cannot use fixed-case gates','height_command_m':.34,'policy_dt_s':.01}
 if getattr(args,'panel',False):
  manifest['panel']={'backend':'MuJoCo','url':f'http://{panel_host(args)}:{args.panel_port}/','bind':panel_host(args),'keyboard_enabled':getattr(args,'keyboard',False),'telemetry_hz':10,'limit_policy':'editable finite session caps; CLI values are initial','rendering':'native MuJoCo window; browser is command/telemetry panel','exposure':'loopback' if panel_host(args).startswith('127.') else 'LAN: any host that can reach this address may POST commands',
  'editable_command':{'vx_mps':[-float(args.max_speed),float(args.max_speed)],'yaw_rad_s':[-float(args.max_yaw),float(args.max_yaw)],
   'height_m':[.20,.45],'skill':['Motion','Recovery','Jump'],'jump':['NONE','SMALL','BIG'],'model':[r['name'] for r in records]},
  'trained_task':'Motion/NONE; changing skill or jump only rewrites the last six Actor inputs and is a diagnostic, not a capability claim'}
  manifest['mode']='manual_browser_preview'
  manifest['input_sha256'].update({str(p.resolve()):sha(p) for p in [Path(__file__).parents[1]/'src/native_stand/students'/name for name in ('motion_panel.py','motion_panel.html','mujoco_keyboard.py')]})
 return model,refs,actor,manifest


def manual_command(keyboard,default_model='main'):
 """Explicit manual command: vx, yaw, reset, height, 31D skill, 31D jump, active model.

 A GLFW-only keyboard keeps the trained height/Motion/NONE and the default model; the
 browser panel can edit the height, the last six Actor dimensions and the model.
 """
 if keyboard is None:return (0.,0.,False,.34,'Motion','NONE',default_model)
 if hasattr(keyboard,'read_full'):
  row=keyboard.read_full();return (row['vx'],row['yaw'],row['reset'],row['height'],row['skill'],row['jump'],row['model'])
 vx,yaw,reset=keyboard.read();return (vx,yaw,reset,.34,'Motion','NONE',default_model)


_START_UNSET=object()


def case(spec,model,refs,actor,target,deadline,viewer=False,keyboard=None,cfg=None,pause=None,
         start=_START_UNSET,height_cmd=.34,leg_response=None):
 from validate_native_mujoco import state
 from native_stand.students.mcu_interface import encode
 from native_stand.students.interface import targets
 from native_stand.command_v6.domain.contract import effort
 from native_stand.command_v6.domain.command_schedule import effective
 data=mujoco.MjData(model);mujoco.mj_forward(model,data)
 if start is _START_UNSET:
  from motion_mcu_playback import DEFAULT_START_HEIGHT_M,start_pose
  start=start_pose(spec,DEFAULT_START_HEIGHT_M)
 if start is not None:
  # Every playback starts from the declared standing pose at the training height, never the
  # asset's neutral origin pose; the chain below only decides how torque is produced.
  from motion_mcu_playback import apply_initial_pose
  apply_initial_pose(model,data,refs,{'pose':start})
  mujoco.mj_forward(model,data)
 execution=history=sensed=mcu_initial=None;last_model=None;demand=np.zeros(6)
 if cfg is not None:
  from diagnose_motion_real2sim import sensors
  from native_stand.students.real2sim_delay import ObservationHistory
  from compare_motion_timing_profiles import LEGS as P_LEGS,WHEELS as P_WHEELS
  if cfg.get('mcu'):
   # New chain: motor/adoption cache initialised from the start pose, never from zeros.
   joints=np.asarray(data.qpos[refs['qpos']],dtype=float)
   target6=np.zeros(6);target6[P_LEGS]=joints[P_LEGS]
   mcu_initial={'joint_q6':joints,'target6':target6}
   if cfg.get('v4_plant'):
    # Full-v4 identified plant: same loop, but the leg kernel is the frozen v4 response.
    from motion_v4_plant import V4Execution,contract as v4_contract
    v4_data,_=v4_contract()
    execution=V4Execution(v4_data,joints,np.asarray(data.qvel[refs['dofs']],dtype=float),target6,tick=0)
   else:
    from motion_mcu_playback import MCUPlayback
    execution=MCUPlayback({'version':cfg['mcu_version'],'profile':cfg['profile']},joints,target6,tick=0,leg_response=leg_response)
  else:
   from compare_motion_timing_profiles import AuditExecution
   sample=sensors(model,data,refs)
   execution=AuditExecution(cfg,initial_target(model,data,refs,P_LEGS),[(-20,sample),(-10,sample)])
  history=ObservationHistory(sensors(model,data,refs),initial_tick=0);sensed=history.at(0)
 control_=spec['control']
 physics_rows=[('腿 armature (kg·m²)',f"{cfg['leg_armature']:.3f}" if cfg else 'MJCF 默认'),
  ('Kp / Kd / 轮 Kv',f"{control_['leg_kp']:.0f} / {control_['leg_kd']:.0f} / {control_['wheel_kv']:.2f}"),
  ('限矩 腿 / 轮 (N·m)',f"{control_['effort_limits_nm'][0]:.1f} / {control_['effort_limits_nm'][2]:.1f}"),
  ('dt physics / PD / policy (ms)',f"{control_['physics_dt_s']*1000:.0f} / {control_['pd_dt_s']*1000:.0f} / {control_['policy_dt_s']*1000:.0f}"),
  ('机体总质量 (kg)',f"{float(model.body_mass.sum()):.2f}"),
  ('地面摩擦 μ',"/".join(f"{v:.2f}" for v in model.geom_friction[refs['ground']].tolist()))]
 if cfg is not None:
  physics_rows+=[('轮转动惯量 J (kg·m²)','/'.join(f"{v:.4f}" for v in cfg['wheel_J']) if cfg['wheel_J'] else '—'),
   ('MuJoCo solver 配置',cfg.get('solver_profile','legacy_xml_v1')),
   ('MuJoCo Plant 配置',cfg.get('plant_profile','source_xml_v1')),
   ('轮峰值力矩 (N·m)',f"{cfg['wheel_peak']:.1f}"),
   ('传达时延 ch0–5 (ms)','/'.join(str(v) for v in cfg['transport_ms'])),
   ('PD 反馈时延 (ms)',str(cfg['feedback_ms'])),
   ('力矩 tau ch0–5 (ms)','/'.join(str(v) for v in cfg['tau_ms'])),
   ('Actor 传感包 (ms)',str(cfg['actor_sensor_ms']))]
 if start is not None:
  physics_rows+=[('播放初态高度 (m)',f"{start['height_m']:.2f} · "+('spec mcu_execution.pose' if start['source']=='spec mcu_execution.pose' else 'recorded standing pose'))]
 if cfg is not None and cfg.get('mcu'):
  physics_rows+=[('执行链版本',cfg['mcu_version']),
   ('运输作用于','PD 后电机矩命令（本地目标运输 %d ms）' % cfg['local_target_transport_ms']),
   ('电机运输 ch0–5 (ms)',"/".join(str(v) for v in cfg['transport_ms'])),
   ('电机响应 tau ch0–5 (ms)',"/".join(str(v) for v in cfg['tau_ms'])),
   ('Actor / PD 反馈年龄 (ms)',f"{cfg['actor_sensor_ms']} / {cfg['feedback_ms']}"),
   ('PD 计算限幅 ch0–5 (N·m)',"/".join(f"{v:g}" for v in cfg['pd_limits_nm'])),
   ('最终电机能力 ch0–5 (N·m)',"/".join(f"{v:.4f}" for v in cfg['driver_limits_nm'])+'（源码换算假设，非实测）'),
   ('solver 合力矩 ch0–5 (N·m)',"/".join(f"{v:g}" for v in cfg['solver_limits_nm'])+'（含假定独立弹簧，不是电机能力）')]
 reference=torch.zeros(1,6);legs=torch.zeros(1,4);wheels=torch.zeros(1,2);requested=applied=torch.zeros(1,6)
 rows={};fail=None;initial=spec['recovery']['initialization'];settled=0.;zeroed=False;control_start=None;display=None
 poses=json.loads(Path(spec['recovery']['pose_library']).read_text())['poses']
 pose=next(p for p in poses if p['family']=='motion_native_normal')
 if pose.get('initialization_control')=='direct_zero_reset':control_start=0.
 if mcu_initial is not None:control_start=0.
 if viewer:
  from mujoco import viewer as v
  display=v.launch_passive(model,data,key_callback=keyboard.key) if keyboard is not None else v.launch_passive(model,data);display.cam.trackbodyid=refs['root'];display.cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING;display.cam.distance=2.
 started=time.monotonic();resets=0
 try:
  last_telemetry=None
  for step in range(1800000 if keyboard is not None else 31000):
   if pause is not None and pause.paused:
    paused_begin=time.monotonic();ticks=0
    while pause.paused:
     if display:display.sync()
     if keyboard is not None and hasattr(keyboard,'publish') and last_telemetry and ticks%10==0:
      keyboard.publish({**last_telemetry,'paused':True})
     ticks+=1;time.sleep(.01)
    deadline+=time.monotonic()-paused_begin
   if time.monotonic()>deadline:fail='WALL_BUDGET';break
   if display and not display.is_running():fail='VIEWER_CLOSED';break
   manual=manual_command(keyboard,actor.names()[0] if hasattr(actor,'names') else 'main')
   if manual[2]:
    mujoco.mj_resetData(model,data);joints=None
    if start is not None:
     from motion_mcu_playback import apply_initial_pose
     joints=apply_initial_pose(model,data,refs,{'pose':start})
    mujoco.mj_forward(model,data)
    legs.zero_();wheels.zero_();requested=torch.zeros(1,6);applied=torch.zeros(1,6);demand=np.zeros(6)
    if execution is not None:
     sample=sensors(model,data,refs);history=ObservationHistory(sample,initial_tick=step);sensed=sample.copy()
     if mcu_initial is not None:
      if joints is None:joints=np.asarray(data.qpos[refs['qpos']],dtype=float)
      mcu_initial['joint_q6']=joints;execution.reset(step,joints,mcu_initial['target6'])
     else:execution=AuditExecution(cfg,initial_target(model,data,refs,P_LEGS),[(-20,sample),(-10,sample)])
    zeroed=False;settled=0.;control_start=0. if (mcu_initial is not None or pose.get('initialization_control')=='direct_zero_reset') else None
    started=time.monotonic();resets+=1;last_model=manual[6]
   elif execution is not None and last_model is not None and manual[6]!=last_model:
    # Model switch: another model must not inherit a queue, previous, packet or motor response.
    sample=sensors(model,data,refs);history=ObservationHistory(sample,initial_tick=step);sensed=sample.copy()
    if mcu_initial is not None:execution.reset(step,mcu_initial['joint_q6'],mcu_initial['target6'])
    else:execution=AuditExecution(cfg,initial_target(model,data,refs,P_LEGS),[(-20,sample),(-10,sample)])
    requested=torch.zeros(1,6);applied=torch.zeros(1,6);demand=np.zeros(6);last_model=manual[6]
   elif last_model is None:last_model=manual[6]
   controlled=control_start is not None;clock=0. if not controlled else data.time-control_start
   if keyboard is None and controlled and clock>=30-1e-8:break
   if history is not None:history.append(step,sensors(model,data,refs))
   if step%10==0:
    if history is not None:
     # All sixteen measured Actor dims (angular, gravity, leg q, dq) come from one 1ms-indexed
     # sample at the declared age, not a delayed q/dq combined with current IMU.
     # The v4 contract keeps the 20ms queued sensor packet of the fitted chain; the spec's own
     # actor age stays untouched for the other playback plants.
     sensed=history.at(step-(20 if getattr(execution,'v4',False) else cfg['actor_sensor_ms']))
     if getattr(execution,'v4',False):
      # The v4 observer holds the leg q/dq for5ms and quantises them; the Actor must see those.
      sensed=np.asarray(sensed,dtype=float).copy()
      sensed[[6,7,9,10]]=np.asarray(execution.held_leg_q(),dtype=float)/1.5
      sensed[[12,13,15,16]]=np.asarray(execution.held_leg_dq(),dtype=float)*.05
     ang,grav,qpos6,dq6=sensed[:3],sensed[3:6],sensed[6:12],sensed[12:18]
    else:
     ang,grav,_=state(model,data,refs);qpos6,dq6=data.qpos[refs['qpos']],data.qvel[refs['dofs']]
    q=torch.tensor(qpos6,dtype=torch.float32)[None];dq=torch.tensor(dq6,dtype=torch.float32)[None]
    command=torch.tensor([[manual[0],manual[1],manual[3]]],dtype=torch.float32) if keyboard is not None else effective(torch.tensor([target]),torch.tensor([clock]),height=height_cmd,entry=2,ramp=8,hold=8,exit=8)
    if execution is not None:
     previous=torch.as_tensor(np.asarray(execution.adopted),dtype=torch.float32)[None]
     prev_legs,prev_wheels=previous[:,[0,1,3,4]],previous[:,[2,5]]
    else:prev_legs,prev_wheels=legs,wheels
    obs=encode(torch.tensor(ang,dtype=torch.float32)[None],torch.tensor(grav,dtype=torch.float32)[None],q,dq,prev_legs,prev_wheels,reference,manual[4],command,jump=manual[5])
    raw=actor.mean(obs,manual[6]) if hasattr(actor,'actors') else actor.mean(obs)
    if not torch.isfinite(raw).all():fail='NONFINITE_ACTION';break
    if controlled:
     ranges=torch.tensor(model.jnt_range[refs['joints']],dtype=torch.float32)[None]
     # Unlimited wheel range is irrelevant; mechanical projection only affects authored limited legs.
     for col,j in enumerate(refs['joints']):
      if not model.jnt_limited[j]:ranges[0,col]=torch.tensor([-float('inf'),float('inf')])
     output=targets(raw,reference,ranges);legs=output['leg_executed'];wheels=output['wheel_executed']
    if execution is not None:
     demand=np.zeros(6)
     if controlled:
      demand[P_LEGS]=legs[0].numpy();demand[P_WHEELS]=wheels[0].numpy()
   if execution is not None:
    # One call per 1ms physics step: motor transport, response and the spring reaction advance
    # every step; only PD/adoption are the internal 5ms events.
    execution.step(step,demand,data.qpos[refs['qpos']],data.qvel[refs['dofs']])
    requested=torch.as_tensor(np.asarray(execution.pre),dtype=torch.float32)[None];applied=torch.as_tensor(np.asarray(execution.actual),dtype=torch.float32)[None]
    data.ctrl[refs['actuators']]=applied[0].numpy()/refs['gears']
    if getattr(execution,'v4',False):
     # Motor only through the actuator; the fitted NET load stays a separate external force.
     data.qfrc_applied[:]=0.;data.qfrc_applied[refs['dofs']]=np.asarray(execution.net,dtype=float)
   elif step%5==0:
    q=torch.tensor(data.qpos[refs['qpos']],dtype=torch.float32)[None];dq=torch.tensor(data.qvel[refs['dofs']],dtype=torch.float32)[None]
    if controlled:requested,applied=effort(q,dq,legs,wheels,leg_kp=spec['control']['leg_kp'],leg_kd=spec['control']['leg_kd'],wheel_kv=spec['control']['wheel_kv'])
    else:requested=torch.zeros(1,6);applied=requested
    data.ctrl[refs['actuators']]=applied[0].numpy()/refs['gears']
   mujoco.mj_step(model,data)
   if step%10==9:
    _,_,s=state(model,data,refs)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():fail='NONFINITE_STATE'
    elif s['closure_m']>.005:fail='CLOSURE'
    elif s['up']<.5:fail='UPRIGHT'
    if fail:break
    if not controlled:
     if not zeroed and data.time>=initial['settle_min_s']:
      data.qvel[:]=0.;mujoco.mj_forward(model,data);zeroed=True;settled=0.
     elif zeroed:
      good=np.linalg.norm(s['COM_linear_velocity_w'])<=initial['settled_linear_speed_max_mps'] and np.linalg.norm(s['angular_velocity_w'])<=initial['settled_angular_speed_max_rad_s'] and np.abs(data.qvel[6:]).max()<=initial['settled_joint_speed_max_rad_s'] and s['tilt']<=initial['tilt_max_rad']
      settled=settled+.01 if good else 0.
      if settled>=initial['settle_dwell_s']-1e-8:control_start=data.time
     if control_start is None and data.time>=initial['settle_max_s']:fail='INITIALIZATION_NOT_SETTLED';break
    else:
     g=s['posture_geometry']
     # Same body-frame sagittal geometry as native offline swing audit.
     rotation=data.xmat[refs['root']].reshape(3,3)
     anchors=np.asarray(spec['recovery']['standing_task']['hip_anchors_root_m'])
     centers=(data.xpos[refs['wheels']]-data.xpos[refs['root']])@rotation
     vectors=centers-anchors
     swing=np.arctan2(-vectors[:,2],-vectors[:,1])
     swing_difference=abs(np.arctan2(np.sin(swing[0]-swing[1]),np.cos(swing[0]-swing[1])))
     record={'leg_swing_rad':swing,'leg_swing_difference_rad':np.array(swing_difference),'wheel_centers_root':centers,
      'time':np.array(clock+.01),'actor_input':obs[0].numpy(),'command':command[0].numpy(),'raw_action':raw[0].numpy(),
      'clipped_action':output['clipped_action'][0].numpy(),'leg_target':legs[0].numpy(),'wheel_target':wheels[0].numpy(),
      'joint_q':data.qpos[refs['qpos']].copy(),'joint_dq':data.qvel[refs['dofs']].copy(),'PD_request':requested[0].numpy(),'applied_torque':applied[0].numpy(),
      'actuator_force':data.qfrc_actuator[refs['dofs']].copy(),
      'state':np.array([s[k] for k in ['vx','vy','yaw_rate','height','pitch','roll','tilt','closure_m','wheel_offset_m']]),
      'geometry':np.array([g[k] for k in ['left_leg_length_m','right_leg_length_m','leg_length_difference_m','wheel_heading_offset_m']]),
      'contact':np.array(s['contact_force_n']),'root_position':np.array(s['root_position_w'])}
     if mcu_initial is not None:
      record.update(mcu_adopted=execution.adopted.numpy().copy(),mcu_pd_limited_nm=execution.pd_limited.numpy().copy(),
       mcu_driver_nm=execution.driver.numpy().copy(),mcu_motor_nm=execution.motor_actual.numpy().copy(),
       mcu_compensation_nm=execution.compensation.numpy().copy(),mcu_spring_nm=execution.spring_actual.numpy().copy(),
       mcu_geometry_valid=np.array(execution.geometry_valid))
     for k,v in record.items():
      if keyboard is not None:
       from collections import deque
       rows.setdefault(k,deque(maxlen=3000)).append(v)
      else:rows.setdefault(k,[]).append(v)
    if keyboard is not None and hasattr(keyboard,'publish') and step%100==99:
     elapsed=max(time.monotonic()-started,1e-9)
     rotation=data.xmat[refs['root']].reshape(3,3)
     anchors=np.asarray(spec['recovery']['standing_task']['hip_anchors_root_m'])
     centers=(data.xpos[refs['wheels']]-data.xpos[refs['root']])@rotation
     vectors=centers-anchors
     swing=np.arctan2(-vectors[:,2],-vectors[:,1])
     swing_diff=float(abs(np.arctan2(np.sin(swing[0]-swing[1]),np.cos(swing[0]-swing[1]))))
     swing_mean=float((swing[0]+swing[1])/2)
     telemetry_physics=physics_rows
     if mcu_initial is not None:
      adopted=execution.adopted.numpy();driver=execution.driver.numpy()
      telemetry_physics=physics_rows+[
       ('MCU 采用目标 腿','/'.join(f"{v:+.3f}" for v in adopted[[0,1,3,4]])+' rad'),
       ('MCU 采用目标 轮','/'.join(f"{v:+.1f}" for v in adopted[[2,5]])+' rad/s'),
       ('MCU 驱动/电机/弹簧 (N·m)','/'.join(f"{v:+.2f}" for v in driver[[2,5]])+' | '
        +'/'.join(f"{v:+.2f}" for v in execution.motor_actual.numpy()[[2,5]])+' | '
        +'/'.join(f"{v:+.2f}" for v in execution.spring_actual.numpy()[[2,5]])),
       ('MCU 弹簧构型有效',str(bool(execution.geometry_valid)))]
     last_telemetry={'t':time.time(),'steps':step//10+1,'resets':resets,'source':'MuJoCo browser / window keys',
      'cmd_vx':float(command[0,0]),'cmd_yaw':float(command[0,1]),'cmd_height':float(command[0,2]),
      'vx':s['vx'],'vy':s['vy'],'yaw':s['yaw_rate'],
      'height':s['height'],'pitch':s['pitch'],'roll':s['roll'],'vz':s['COM_linear_velocity_w'][2],
      'swing_diff_rad':swing_diff,'swing_mean_rad':swing_mean,
      'leg_length_diff_m':s['posture_geometry']['leg_length_difference_m'],
     'wheel_fore_aft_m':s['posture_geometry']['wheel_heading_offset_m'],
      'wheel_qdot_l':float(data.qvel[refs['dofs'][2]]),'wheel_qdot_r':float(data.qvel[refs['dofs'][5]]),
      'wheel_tau_l':float(applied[0,2]),'wheel_tau_r':float(applied[0,5]),
      'leg_tau_0':float(applied[0,0]),'leg_tau_1':float(applied[0,1]),
      'leg_tau_2':float(applied[0,3]),'leg_tau_3':float(applied[0,4]),
      'act_0':float(raw[0,0]),'act_1':float(raw[0,1]),'act_2':float(raw[0,2]),
      'act_3':float(raw[0,3]),'act_4':float(raw[0,4]),'act_5':float(raw[0,5]),
      'actor_input':[float(v) for v in obs[0]],'physics':telemetry_physics,
      'simulation_time_s':data.time,
      'wall_time_s':elapsed,'realtime_factor':None,'realtime_factor_total':data.time/elapsed,
      'task_mode':manual[4],'skill':manual[4],'jump':manual[5],'cmd_height_bounds':[.20,.45],
      'model':manual[6],'models':list(actor.names()) if hasattr(actor,'names') else ['main'],
      'actor_task_tail':[float(v) for v in obs[0,25:31]],
      'initialized':controlled,'actor_valid':controlled};keyboard.publish(last_telemetry)
    if display:
     display.sync();delay=data.time-(time.monotonic()-started)
     if delay>0:time.sleep(min(delay,.01))
 finally:
  if display:display.close()
 arrays={k:np.stack(v) for k,v in rows.items()}
 count=len(rows.get('time',[]));summary={'target':target,'controlled_samples':count,'continuous_s':count*.01,'initialization_s':control_start,'termination':fail,'capability_certified':False}
 if keyboard is not None:return {**summary,'mode':'manual_keyboard_preview','passed':False,'failed_checks':[],'task_scoring':'NOT_RUN','resets':resets,'recorded_tail_seconds':count*.01,'note':'at most final30s trace; not a fixed-command capability test'},arrays
 if not count:return {**summary,'passed':False,'failed_checks':[fail or 'NO_CONTROLLED_SAMPLES']},arrays
 t=arrays['time'];s=arrays['state'];hold=(t>10)&(t<=18+1e-6);park=(t>26)&(t<=30+1e-6)
 checks={'full_window':count==3000,'runtime':fail is None,'full_hold':int(hold.sum())>=799}
 if hold.any():
  h=s[hold];vx=h[:,0];contact=(arrays['contact'][hold]>2).all(-1)
  yaw_mode=target[1]!=0;desired=target[1] if yaw_mode else target[0];speed=h[:,2] if yaw_mode else vx
  mae=float(np.abs(speed-desired).mean());gain=float(speed.mean()/desired) if desired else None
  checks.update(height=height_cmd-.04<=h[:,3].mean()<=height_cmd+.04,contact=contact.mean()>=.95,pitch=np.abs(h[:,4]).max()<=.1,roll=np.abs(h[:,5]).max()<=.05,
   closure=s[:,7].max()<=.005,tilt=s[:,6].max()<=.5,saturation=(np.abs(arrays['PD_request'][hold])>=np.array([40,40,5.5,40,40,5.5])).any(-1).mean()<=.05)
  if desired:checks.update(tracking=mae<=.3,direction=(speed*desired>0).mean()>=.95,gain=.8<=gain<=1.2,sustained=(speed*np.sign(desired)>=abs(desired)*.7).mean()>=.8)
  if yaw_mode:checks['in_place_translation']=np.linalg.norm(h[:,:2],axis=-1).mean()<=.3
  if not desired:checks.update(zero_horizontal=np.quantile(np.linalg.norm(h[:,:2],axis=-1),.95)<=.3,zero_yaw=np.abs(h[:,2]).mean()<=.3)
  summary.update(vx_mean=float(vx.mean()),MAE=mae,gain=gain,direction_rate=float((speed*desired>0).mean()) if desired else None,yaw_mean=float(h[:,2].mean()),horizontal_mean=float(np.linalg.norm(h[:,:2],axis=-1).mean()),tracking_axis='yaw' if yaw_mode else 'vx',
   both_contact_rate=float(contact.mean()),height_mean=float(h[:,3].mean()),pitch_abs_max=float(np.abs(h[:,4]).max()),roll_abs_max=float(np.abs(h[:,5]).max()),
   wheel_offset_abs_mean=float(np.abs(h[:,8]).mean()),leg_difference_abs_mean=float(np.abs(arrays['geometry'][hold,2]).mean()))
 swing=arrays['leg_swing_difference_rad']
 summary['supplemental_swing_constraint']={'definition':'hip-anchor to wheel-center, root sagittal angle atan2(-z,-y); wrapped left-minus-right absolute',
  'strict_limit_rad':.05,'sampling_hz':100,'full_max_rad':float(swing.max()),'full_fraction_ge_limit':float((swing>=.05).mean()),
  'hold_mean_rad':float(swing[hold].mean()) if hold.any() else None,'hold_max_rad':float(swing[hold].max()) if hold.any() else None,
  'hold_fraction_ge_limit':float((swing[hold]>=.05).mean()) if hold.any() else None,
  'hold_passed':bool(checks['full_hold'] and (swing[hold]<.05).all()),'full_passed':bool(checks['full_window'] and (swing<.05).all())}
 summary.update(checks={k:bool(v) for k,v in checks.items()},passed=all(checks.values()),failed_checks=[k for k,v in checks.items() if not v],
  parking_abs_vx=float(np.abs(s[park,0]).mean()) if park.any() else None,closure_max=float(s[:,7].max()),
  net_displacement=float(np.linalg.norm(arrays['root_position'][-1,:2]-arrays['root_position'][0,:2])),
  raw_clip_fraction=(np.abs(arrays['raw_action'])>1).mean(0).tolist(),
  torque_saturation_fraction=(np.abs(arrays['PD_request'])>=np.array([40,40,5.5,40,40,5.5])).mean(0).tolist())
 return summary,arrays


def execute(args):
 from validate_native_mujoco import save
 from native_stand.command_v6.completion import write
 from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
 from native_stand.command_v6.tools.artifact_io import npz
 started=time.monotonic();torch.set_num_threads(1);spec=json.loads(args.spec.read_text());model,refs,actor,manifest=plan(args,spec)
 height_cmd=float(getattr(args,'target_height_m',.34))
 if not args.run:print(json.dumps(manifest,indent=2));return
 read_only=bool(getattr(args,'read_only',False))
 if not read_only:
  validate_new_experiment_name(args.output_dir.name);args.output_dir.mkdir(parents=True,exist_ok=False)
  save(args.output_dir/'manifest.json',manifest)
 def save_out(name,payload):
  if not read_only:save(args.output_dir/name,payload)
 def write_out(name,payload):
  if not read_only:write(args.output_dir/name,payload)
 def npz_out(name,arrays):
  if not read_only:npz(args.output_dir/name,arrays)
 summaries=[];attempts={};status='FAILED'
 before=actor.snapshot() if hasattr(actor,'snapshot') else {k:v.clone() for k,v in actor.state_dict().items()};panel_server=None;panel_plan=None
 def budget():return {'schema_version':'mujoco_diagnostic_budget_v1','authorization':'USER_REQUESTED_FROZEN_MOTION_SIM2SIM','checkpoint_sha256':manifest['checkpoint_sha256'],'wall_limit_s':args.wall_seconds,'wall_used_s':time.monotonic()-started,'online_PPO':0,'attempts':attempts}
 try:
  if getattr(args,'panel',False):
   from native_stand.students.motion_panel import Plan,start
   panel_plan=Plan(args.max_speed,args.max_yaw,backend='MuJoCo',
    models=actor.names() if hasattr(actor,'names') else None,
    model=actor.active if hasattr(actor,'active') else None);panel_plan.pause_supported=True
   panel_server,_=start(panel_plan,args.panel_port,panel_host(args))
   print(f'MuJoCo browser controls: {manifest["panel"]["url"]}',flush=True)
  with torch.inference_mode():
   for i,target in enumerate(manifest['protocol']['commands']):
    if time.monotonic()>started+args.wall_seconds-5:status='STOPPED_BOUNDED';break
    key=f'command_{i:02}';attempts[key]={'status':'RUNNING','target':target};write_out('budget.json',budget())
    keyboard=None
    if getattr(args,'keyboard',False) or getattr(args,'panel',False):
     from native_stand.students.mujoco_keyboard import MotionKeyboard
     if panel_plan is not None:
      from native_stand.students.mujoco_keyboard import MotionPanelControls
      keyboard=MotionPanelControls(panel_plan,getattr(args,'keyboard',False))
     else:keyboard=MotionKeyboard(args.max_speed,args.max_yaw)
     if getattr(args,'keyboard',False):print('MuJoCo keyboard: W/S forward/backward; A/D yaw; Space zero; R reset. Click viewer first. Policy10ms/PD5ms unchanged.',flush=True)
    summary,arrays=case(spec,model,refs,actor,target,started+args.wall_seconds-5,args.viewer,keyboard=keyboard,
     cfg=manifest.get('actuator_configuration'),pause=panel_plan if getattr(args,'panel',False) else None,
     start=manifest.get('playback_start'),height_cmd=height_cmd,
     leg_response=(manifest.get('leg_response') if getattr(args,'leg_response','spec')=='bilateral_modal_v1' else None))
    if arrays:npz_out(key+'.npz',arrays)
    save_out(key+'.json',summary);summaries.append(summary)
    attempts[key].update(status='COMPLETED',controlled_steps=summary['controlled_samples'],settled=True)
    print(json.dumps(summary),flush=True)
    if summary['termination'] in ['NONFINITE_STATE','CLOSURE']:status='FAILED';break
   else:status='COMPLETED'
 except BaseException as exc:
  save_out('runtime_failure.json',{'error':repr(exc)})
  for a in attempts.values():
   if a['status']=='RUNNING':a.update(status='FAILED',settled=True)
  raise
 finally:
  if panel_server is not None:
   from native_stand.students.motion_panel import stop
   stop(panel_server)
  unchanged=actor.unchanged(before) if hasattr(actor,'unchanged') else all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
  result={'status':status,'model_unchanged':unchanged,'wall_s':time.monotonic()-started,'online_PPO':0,'results':summaries,
   'all_passed':len(summaries)==len(manifest['protocol']['commands']) and all(r['passed'] for r in summaries),'capability_certified':False,
   'scope':'manual keyboard preview, no capability scoring' if (getattr(args,'keyboard',False) or getattr(args,'panel',False)) else 'single deterministic reference-MJCF migration diagnostic; not native protocol certification'}
  save_out('terminal_report.json',result);write_out('budget.json',budget())
  if read_only:print(json.dumps({**result,'spec':str(args.spec),'model':getattr(actor,'active',None),
  'models':list(actor.names()) if hasattr(actor,'names') else [str(args.checkpoint)],'read_only':True,'files_written':0},ensure_ascii=False),flush=True)
  else:
   with (args.output_dir/'report.md').open('x') as f:
    f.write('# Motion31 frozen checkpoint MuJoCo sim2sim\n\n单环境迁移诊断，非正式认证。摆角硬约束独立记录于supplemental_swing_constraint，不冒充旧协议通过。\n\n|command|seconds|vx|MAE|failures|\n|---|---:|---|---|---|\n')
    for r in summaries:f.write(f"|{r['target']}|{r['continuous_s']}|{r.get('vx_mean')}|{r.get('MAE')}|{r['failed_checks']}|\n")
  for path,h in manifest['input_sha256'].items():
   if sha(path)!=h:raise RuntimeError('Read-only input changed during diagnostic')
