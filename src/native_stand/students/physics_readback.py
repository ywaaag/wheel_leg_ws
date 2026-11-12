"""Bounded no-Actor PhysX parameter readback through the public diagnostic owner."""
import json,os,sys,traceback
from pathlib import Path
from .data import sha
VERSION='recovery_physics_readback_v1'

def validate(r):
 allowed={'version','authorization','online_PPO','physical_envs','scored_envs','wall_limit_s','output','parent_spec','parent_spec_sha256','session','physical_gpu','checkpoints','output_max_bytes'}
 if set(r)!=allowed:raise ValueError('physics readback unknown/missing fields')
 if r['version']!=VERSION or r['authorization']!='AUTHORIZED' or r['online_PPO']!=0 or r['physical_envs']!=1 or r['scored_envs']!=0 or not 60<=r['wall_limit_s']<=900:raise ValueError('physics readback scope differs')
 if r['session']!=os.environ.get('HAPI_SESSION_ID'):raise ValueError('readback session differs')
 p=Path(r['parent_spec']);s=json.loads(p.read_text())
 if sha(p)!=r['parent_spec_sha256'] or s['device']['physical_index']!=r['physical_gpu']:raise ValueError('readback spec/device differs')
 for path_key,sha_key in [('asset_path','asset_sha256'),('asset_module_path','asset_module_sha256'),('colleague_config_path','colleague_config_sha256')]:
  if sha(s[path_key])!=s[sha_key]:raise ValueError('readback source identity differs: '+path_key)
 if s['student_training']['version']!='student_recovery_contact_quality_ppo_v1':raise ValueError('explicit contact-quality environment required')
 if len(r['checkpoints'])!=1:raise ValueError('one identity-only checkpoint required')
 cp=r['checkpoints'][0]
 if sha(cp['checkpoint'])!=cp['sha256']:raise ValueError('checkpoint identity differs')
 if not 0<r['output_max_bytes']<=16777216:raise ValueError('readback output cap differs')
 return s

def read_view(view,methods):
 from native_stand.control_readback import tensor
 out={}
 for method in methods:
  try:out[method]={'status':'READ','values':tensor(getattr(view,method)())}
  except Exception as exc:out[method]={'status':'UNAVAILABLE','error':repr(exc)}
 return out

def collect(env):
 from native_stand.command_v6.runtime.control_readback import verify,verify_ground
 from pxr import UsdPhysics,UsdShade,PhysxSchema
 rv=env.robot.root_view
 result={'version':VERSION,'joint_names':env.robot.joint_names,'body_names':env.robot.body_names,'active_indices':env.ids,
  'articulation':read_view(rv,['get_dof_limits','get_dof_max_velocities','get_dof_max_forces','get_dof_armatures','get_dof_friction_coefficients','get_dof_stiffnesses','get_dof_dampings','get_masses','get_coms','get_inertias','get_material_properties','get_contact_offsets','get_rest_offsets']),
  'control':verify(env),'binding_semantics':'composed USD physics-purpose material binding; view values separately from authored attrs','collision_materials':[],'wheel_views':{},'simulation_policy_steps':0,'actor_calls':0,'optimizer_steps':0}
 stage=env.sim.stage
 for prim in stage.Traverse():
  path=str(prim.GetPath())
  if not prim.HasAPI(UsdPhysics.CollisionAPI) or not (path=='/World/ground' or '/env_0/' in path):continue
  material,rel=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
  row={'path':path,'bound_material':str(material.GetPath()) if material else None,'binding_relation':str(rel.GetPath()) if rel else None,'authored_contact':{a.GetName():str(a.Get()) for a in prim.GetAttributes() if any(k in a.GetName() for k in ['contactOffset','restOffset'])}}
  if material:
   mp=material.GetPrim();p=PhysxSchema.PhysxMaterialAPI(mp)
   row['composed_material']={a.GetName():str(a.Get()) for a in mp.GetAttributes() if any(k in a.GetName().lower() for k in ['friction','restitution','combine'])}
   row['schema_fallback_combine']={'friction':p.GetFrictionCombineModeAttr().Get(),'restitution':p.GetRestitutionCombineModeAttr().Get()}
  result['collision_materials'].append(row)
 # Rigid body views provide unambiguous per-wheel shape slots (not articulation-wide padding).
 simview=env.robot._physics_sim_view
 for side in ('left','right'):
  paths=[p for p in stage.Traverse() if '/env_0/' in str(p.GetPath()) and p.GetName()==side+'_wheel' and p.HasAPI(UsdPhysics.RigidBodyAPI)]
  if len(paths)!=1:raise ValueError('unique wheel body absent')
  view=simview.create_rigid_body_view(str(paths[0].GetPath()))
  result['wheel_views'][side]={'path':str(paths[0].GetPath()),'count':view.count,'max_shapes':view.max_shapes,'properties':read_view(view,['get_material_properties','get_contact_offsets','get_rest_offsets','get_masses','get_coms','get_inertias'])}
 return result

def worker(r):
 from native_stand.command_v6.completion import write
 s=validate(r);out=Path(r['output']);app=env=None;code=1
 try:
  sys.path.insert(0,s['asset_module_root'])
  from isaaclab.app import AppLauncher
  app=AppLauncher({'headless':True,'device':'cuda:0','multi_gpu':False}).app
  from .evaluator import environment
  env=environment(s,'RecoveryStudent',envs=1,autonomous_preview=True)
  env.reset()
  payload=collect(env)
  if hasattr(env,'recovery_dr'):payload['domain_randomization']=env.recovery_dr.summary()
  # Missing backend properties remain explicitly unavailable, never replaced by config values.
  payload['scope']='fresh shared environment initialized; no Actor loaded/called, no env.step; constructor/reset can advance initialization physics'
  payload['parent_spec_sha256']=r['parent_spec_sha256']
  payload['checkpoint_identity_only']=r['checkpoints'][0]
  payload['ground_backend_offsets']={'status':'UNAVAILABLE','reason':'static Plane has no articulation/rigid-body tensor view in this capture; composed material binding recorded separately'}
  if len(json.dumps(payload).encode())>r['output_max_bytes']-1048576:raise RuntimeError('readback output cap exhausted')
  write(out/'physics_readback.json',payload)
  write(out/'summary.json',{'status':'COMPLETED','version':VERSION,'report':'physics_readback.json','online_PPO':0,'scored_episodes':0,'actor_calls':0})
  code=0
 except BaseException as e:write(out/'worker_failure.json',{'exception':repr(e),'traceback':traceback.format_exc()})
 finally:
  if env is not None:
   try:env.close()
   except Exception as e:write(out/'cleanup_error.json',{'exception':repr(e)})
  if app is not None:
   try:app.close()
   except Exception:pass
 return code
