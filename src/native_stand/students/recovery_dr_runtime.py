"""Opt-in R1/R2 adapter; original scene, target cache, PD and limits remain shared."""
import numpy as np
import torch
from . import recovery_dr as dr


def setup_materials(env):
 """Called before cloning/PhysX initialization; explicit average on both surfaces."""
 from pxr import UsdPhysics,PhysxSchema,UsdShade
 import isaaclab.sim as sim_utils
 stage=env.sim.stage
 ground=stage.GetPrimAtPath(env.cfg.ground_material_path)
 api=PhysxSchema.PhysxMaterialAPI.Apply(ground)
 api.CreateFrictionCombineModeAttr().Set('average')
 path='/World/recovery_dr_wheel_material'
 mat=sim_utils.RigidBodyMaterialCfg(static_friction=.5,dynamic_friction=.5,restitution=0.,friction_combine_mode='average',restitution_combine_mode='average')
 mat.func(path,mat)
 found=set()
 for prim in stage.Traverse():
  p=str(prim.GetPath())
  if '/env_0/' not in p or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
  for side in ('left','right'):
   if '/'+side+'_wheel/' in p or p.endswith('/'+side+'_wheel'):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(stage,path),bindingStrength=UsdShade.Tokens.strongerThanDescendants,materialPurpose='physics')
    found.add(side)
 if found!={'left','right'}:raise RuntimeError('DR wheel collision material binding missing')


def material_tensor(view):
 value=view.get_material_properties()
 if isinstance(value,torch.Tensor):return value
 import warp as wp
 if isinstance(value,wp.array):return wp.to_torch(value)
 raise TypeError('unsupported physics material backend '+type(value).__name__)


def set_material_tensor(view,values,indices):
 original=view.get_material_properties()
 if isinstance(original,torch.Tensor):view.set_material_properties(values,indices);return
 import warp as wp
 if not isinstance(original,wp.array):raise TypeError('unsupported material setter backend')
 view.set_material_properties(wp.from_torch(values.contiguous(),dtype=wp.float32),wp.from_torch(indices.contiguous(),dtype=wp.int32))


class Adapter:
 def __init__(self,env,c):
  self.env=env;self.config=c;self.state=dr.initial_state();self.override=None
  self.episodes=np.zeros(env.num_envs,dtype=np.int64)
  self.params=torch.tensor([.5,60.,2.,1.],device=env.device).repeat(env.num_envs,1)
  self.group_ids=torch.tensor([0]*256+[1]*256+[2]*512,device=env.device) if env.num_envs==1024 else torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
  self.strengths=np.zeros((env.num_envs,2));self.reset_count=0;self.last_readback=None
  self.views={side:env.robot._physics_sim_view.create_rigid_body_view('/World/envs/env_*/Robot/'+side+'_wheel') for side in ('left','right')}
  for side,v in self.views.items():
   if v.count!=env.num_envs or v.max_shapes!=1:raise RuntimeError('DR wheel tensor view mapping differs: '+side)
  self.verify_binding()
  self.pd_readback=None
  self.material_probe=self.probe_material_endpoints()
 def probe_material_endpoints(self):
  rows=[]
  for coefficient in (.3,.7):
   for side,v in self.views.items():
    original=material_tensor(v).clone();changed=original.clone()
    changed[:,0,0:2]=coefficient
    index=torch.arange(v.count,dtype=torch.int32,device='cpu')
    try:
     set_material_tensor(v,changed,index)
     actual=material_tensor(v).clone()
     if not torch.allclose(actual[:,:,:2],torch.full_like(actual[:,:,:2],coefficient),atol=2e-6,rtol=0):raise RuntimeError('DR endpoint backend material readback differs')
     rows.append({'wheel':side,'shape_requested':coefficient,'shape_actual_min':float(actual[:,:,:2].min()),'shape_actual_max':float(actual[:,:,:2].max()),'pair_actual':float(dr.pair_friction(coefficient)),'env_count':v.count})
    finally:
     set_material_tensor(v,original,index)
     if not torch.equal(original,material_tensor(v)):raise RuntimeError('DR endpoint probe nominal restoration differs')
  return {'status':'PASSED','rows':rows,'physics_steps':0,'scope':'backend coefficient roundtrip and explicit composed combine on both materials; no measured force coefficient'}
 def verify_pd(self,q,dq,legs,wheels,request,effort,controlled):
  if self.pd_readback is not None or not bool(controlled.any()):return
  eligible=torch.where(controlled)[0];ids=torch.cat((eligible[:2],eligible[-6:])).unique();v=self.params[ids]
  expected=torch.zeros_like(q[ids]);expected[:,[0,1,3,4]]=v[:,1:2]*(legs[ids]-q[ids][:,[0,1,3,4]])-v[:,2:3]*dq[ids][:,[0,1,3,4]]
  expected[:,[2,5]]=v[:,3:4]*(wheels[ids]-dq[ids][:,[2,5]])
  limits=expected.new_tensor([40,40,5.5,40,40,5.5]);clipped=expected.clamp(-limits,limits)
  if not torch.equal(request[ids],expected) or not torch.equal(effort[ids],clipped):raise RuntimeError('DR actual PD execution differs from per-env gains/limits')
  self.pd_readback={'status':'PASSED','env_ids':ids.cpu().tolist(),'gains':v[:,1:].cpu().tolist(),'q':q[ids].cpu().tolist(),'dq':dq[ids].cpu().tolist(),'leg_targets':legs[ids].cpu().tolist(),'wheel_targets':wheels[ids].cpu().tolist(),'requested':request[ids].cpu().tolist(),'clipped':effort[ids].cpu().tolist(),'torque_limits':limits.cpu().tolist(),'PD_dt_s':.005,'physics_dt_s':.001}
 def verify_binding(self):
  from pxr import UsdPhysics,UsdShade,PhysxSchema
  stage=self.env.sim.stage;bindings=[]
  for prim in stage.Traverse():
   path=str(prim.GetPath())
   if not prim.HasAPI(UsdPhysics.CollisionAPI) or not (path=='/World/ground' or '/env_0/' in path and ('/left_wheel/' in path or '/right_wheel/' in path)):continue
   m,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
   if not m:raise RuntimeError('DR composed material binding missing '+path)
   api=PhysxSchema.PhysxMaterialAPI(m.GetPrim());mode=api.GetFrictionCombineModeAttr().Get()
   physics=UsdPhysics.MaterialAPI(m.GetPrim())
   row={'shape':path,'material':str(m.GetPath()),'friction_combine':mode,'static':physics.GetStaticFrictionAttr().Get(),'dynamic':physics.GetDynamicFrictionAttr().Get()}
   if mode!='average' or row['static']!=.5 or row['dynamic']!=.5:raise RuntimeError('DR nominal material/combine differs '+repr(row))
   bindings.append(row)
  if len(bindings)<3:raise RuntimeError('DR ground/two wheel binding evidence absent')
  self.bindings=bindings
 def configure(self,state=None,override=None):
  if state is not None:self.state=state
  self.override=override
 def reset(self,ids):
  ids=np.asarray(ids.detach().cpu(),dtype=np.int64);self.episodes[ids]+=1
  training=self.env.curriculum_training
  if self.override is not None:
   strength=np.broadcast_to(self.override['strength'],(len(ids),2));seed=self.override['seed']
  elif training:
   groups=self.group_ids[torch.as_tensor(ids,device=self.env.device)].cpu().numpy()
   strength=np.asarray([dr.exposure(self.state['level'],int(g)) for g in groups]);seed=self.config['seed']
  else:strength=np.zeros((len(ids),2));seed=self.config['seed']
  self.strengths[ids]=strength
  values=dr.sample(seed,ids,self.episodes[ids],strength)
  self.params[torch.as_tensor(ids,device=self.env.device)]=torch.as_tensor(values,device=self.env.device)
  # PhysX tensor material API uses host tensors; update only reset environments.
  index=torch.as_tensor(ids,dtype=torch.int32,device='cpu')
  for side,v in self.views.items():
   material=material_tensor(v).clone()
   material[ids,0,0]=torch.as_tensor(values[:,0]);material[ids,0,1]=torch.as_tensor(values[:,0]);material[ids,0,2]=0.
   set_material_tensor(v,material,index)
   actual=material_tensor(v)[ids,0].cpu().numpy()
   expected=np.column_stack((values[:,0],values[:,0],np.zeros(len(ids))))
   if not np.allclose(actual,expected,rtol=0,atol=2e-6):raise RuntimeError('PhysX wheel material write/readback differs '+side)
  self.reset_count+=len(ids)
  self.last_readback={'env_ids':ids[:16].tolist(),'requested_wheel_static_dynamic':values[:16,0].tolist(),'actual_pair_static_dynamic':dr.pair_friction(values[:16,0]).tolist(),'pd_gains':values[:16,1:].tolist(),'combine':'average','ground':.5,'readback_matches_requested':True}
 def gains(self):
  return {'leg_kp':self.params[:,1:2],'leg_kd':self.params[:,2:3],'wheel_kv':self.params[:,3:4]}
 def summary(self):
  p=self.params.detach().cpu().numpy()
  return {'version':dr.VERSION,'level':self.state['level'],'family':dr.LEVELS[self.state['level']][0],'strength':dr.strengths(self.state['level']),
   'parameters':['wheel_static_dynamic','leg_kp','leg_kd','wheel_kv'],'min':p.min(0).tolist(),'max':p.max(0).tolist(),'mean':p.mean(0).tolist(),
   'pair_friction_min':float(dr.pair_friction(p[:,0]).min()),'pair_friction_max':float(dr.pair_friction(p[:,0]).max()),'reset_count':self.reset_count,'last_readback':self.last_readback,
   'bindings':self.bindings,'endpoint_probe':self.material_probe,'executed_PD_readback':self.pd_readback,'pair_interpretation':'computed from backend wheel readback + composed fixed Plane .5, explicit average on both; not measured friction-force identification'}
 def snapshot(self):
  return {'state':self.state,'episodes':self.episodes.tolist(),'strengths':self.strengths.tolist(),'params':self.params.detach().cpu(),'reset_count':self.reset_count}


def subclass(base):
 class DRStudentEnv(base):
  def _reset_idx(self,env_ids):
   super()._reset_idx(env_ids)
   if hasattr(self,'recovery_dr'):
    ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
    self.recovery_dr.reset(ids)
 return DRStudentEnv
