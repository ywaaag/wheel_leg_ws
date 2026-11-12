"""Opt-in Motion robustness: true timestamped sensing and standing-only events."""
import torch
from . import motion_robustness as dr
from native_stand.control_readback import tensor


class DelayedPackets:
 def __init__(self,adapter):
  self.adapter=adapter;self.age=torch.zeros(adapter.n,device=adapter.device)
 def reset(self,ids):
  self.age[ids]=0
 def read(self,current,tick):
  a=self.adapter;e=a.env;t=e.motion_timing
  ages=a.params[:,0].long();available=(tick-a.reset_tick).clamp_min(0)
  ages=torch.minimum(ages,available);self.age.copy_(ages)
  roots=a.root_history[(tick-ages)%32,a.ids];joints=t.sensors[(tick-ages)%32,a.ids]
  from isaaclab.utils.math import quat_apply_inverse
  # The query may be at the current reset tick before any1ms write.
  angular=quat_apply_inverse(roots[:,:4],roots[:,4:7])
  down=torch.zeros_like(angular);down[:,2]=-1
  gravity=quat_apply_inverse(roots[:,:4],down)
  result=torch.cat((angular*.25,gravity,(joints[:,[0,1,3,4]]-e.reference[:,[0,1,3,4]])/1.5,joints[:,6:]*.05),-1)
  return torch.where((available==0)[:,None],current,result)


class Adapter:
 def __init__(self,env,cfg):
  self.env=env;self.config=cfg;self.state=cfg['state'];self.override=None
  self.n=env.num_envs;self.device=env.device;self.ids=torch.arange(self.n,device=self.device)
  saved=self.state.get('DR_episode_counters',[])
  self.episodes=torch.tensor(saved,device=self.device,dtype=torch.long) if len(saved)==self.n else torch.zeros(self.n,device=self.device,dtype=torch.long)
  self.params=torch.tensor([10,0,10,1,.5,0,0],device=self.device).repeat(self.n,1)
  self.reset_tick=torch.zeros(self.n,device=self.device,dtype=torch.long)
  self.root_history=torch.zeros(32,self.n,7,device=self.device)
  self.request_ticks=torch.full((32,self.n,6),-1,device=self.device,dtype=torch.long)
  self.adopted_ticks=torch.full((self.n,6),-1,device=self.device,dtype=torch.long)
  self.first_adoption_ages=torch.full((self.n,6),-1,device=self.device,dtype=torch.long)
  self.pushed=torch.zeros(self.n,2,device=self.device,dtype=torch.bool)
  self.push_count=0;self.last_push=None;self.reset_count=0;self.last_readback=None;self.material_failure=None
  if self.n==4096 and env.motion_config.get('course_version') in ('motion31_decoupled_height_course_v1','motion31_fixed_height_motion_v2','motion31_persistent_frontiers_v1'):
   self.group_ids=torch.tensor(env.stage_spec['group_slots'],device=self.device)
   if env.motion_config.get('course_version')=='motion31_persistent_frontiers_v1':
    self.level_ids=torch.tensor(env.stage_spec['DR_level_slots'],device=self.device)
    self.strength_table=torch.tensor([[dr.strength(l,g) for g in range(3)] for l in range(len(dr.LEVELS))],device=self.device)
  elif self.n==4096:
   table=dr.cross_quotas(env.stage_spec['slot_counts']);groups=[]
   for row in table.tolist():
    for group,count in enumerate(row):groups += [group]*count
   self.group_ids=torch.tensor(groups,device=self.device)
  else:self.group_ids=torch.zeros(self.n,device=self.device,dtype=torch.long)
  self.default_masses=torch.as_tensor(tensor(env.robot.root_view.get_masses()),device=self.device).clone()
  self.default_inertias=torch.as_tensor(tensor(env.robot.root_view.get_inertias()),device=self.device).clone()
  self.root_body=0
  if env.robot.body_names[0] not in ('base_link','base','root','body','robot','base_link_0'):
   # Physical root convention is explicit and recorded; never guess by side-name.
   if getattr(env.robot,'root_body_id',0)!=0:raise RuntimeError('unsupported articulation root-body mapping')
  from .recovery_dr_runtime import material_tensor
  self.views={side:env.robot._physics_sim_view.create_rigid_body_view('/World/envs/env_*/Robot/'+side+'_wheel') for side in ('left','right')}
  for side,view in self.views.items():
   if view.count!=self.n or view.max_shapes!=1:raise RuntimeError('Motion DR wheel material view differs: '+side)
   mat=material_tensor(view)
   if not torch.allclose(mat[:,:,:2],torch.full_like(mat[:,:,:2],.5),atol=2e-6):raise RuntimeError('Motion DR nominal wheel material differs')
  self.material_shape_columns=None
  writer=cfg.get('material_writer')
  if writer is not None:
   if writer!=dr.MATERIAL_WRITER:raise ValueError('unknown Motion material writer')
   root=env.robot.root_view;columns={};offset=0
   for path in root.link_paths[0]:
    view=env.robot._physics_sim_view.create_rigid_body_view(path)
    for side in self.views:
     if path.endswith('/'+side+'_wheel'):
      if view.max_shapes!=1 or side in columns:raise RuntimeError('Motion wheel shape mapping differs')
      columns[side]=offset
    offset+=view.max_shapes
   if root.count!=self.n or offset!=root.max_shapes or set(columns)!=set(self.views):raise RuntimeError('Motion articulation material shape mapping differs')
   self.material_shape_columns=columns
  self.root_body_name=env.robot.body_names[0]
  self.reset(self.ids)
 def configure(self,override=None):self.override=override
 def reset(self,ids):
  if ids is None:ids=self.ids
  self.episodes[ids]+=1
  e=self.env;t=e.motion_timing
  if self.override is not None:
   level=self.override['level'];group=self.override['group'];seed=self.override['seed']
   strengths=torch.tensor(dr.strength(level,group),device=self.device).expand(len(ids),-1)
  elif e.curriculum_training:
   level=self.state['DR_level'];groups=self.group_ids[ids]
   options=torch.tensor([dr.strength(level,g) for g in range(3)],device=self.device)
   strengths=self.strength_table[self.level_ids[ids],groups] if hasattr(self,'level_ids') else options[groups];seed=self.config['configuration']['seed']
  else:strengths=torch.zeros(len(ids),6,device=self.device);seed=self.config['configuration']['seed']
  standing=e.target_xy[ids].abs().sum(-1)<1e-6
  self.params[ids]=dr.parameters(ids,self.episodes[ids],strengths,seed=seed,standing=standing,eval_corners=self.override is not None,material_backend=self.config.get('material_backend'))
  p=self.params[ids];t.delays[ids]=p[:,1:2].long()+torch.tensor([10,10,7,10,10,11],device=self.device)
  t.feedback[ids]=p[:,2].long()
  self.reset_tick[ids]=t.tick;self.pushed[ids]=False
  self.request_ticks[:,ids]=-1;self.adopted_ticks[ids]=-1;self.first_adoption_ages[ids]=-1
  d=e.robot.data;root=torch.cat((d.root_quat_w.torch,d.root_ang_vel_w.torch),-1)
  self.root_history[:,ids]=root[ids][None]
  self._physical_parameters(ids)
  self.reset_count+=len(ids)
  self.push_enabled=bool((self.params[:,5]>0).any())
 def _physical_parameters(self,ids):
  from .recovery_dr_runtime import material_tensor,set_material_tensor
  e=self.env;scales=self.params[ids,3]
  body=torch.tensor([self.root_body],device=self.device,dtype=torch.int32)
  mass=self.default_masses[ids,self.root_body]*scales
  inertia=self.default_inertias[ids,self.root_body]*scales[:,None]
  e.robot.set_masses_index(masses=mass[:,None].contiguous(),body_ids=body,env_ids=ids.to(torch.int32))
  e.robot.set_inertias_index(inertias=inertia[:,None].contiguous(),body_ids=body,env_ids=ids.to(torch.int32))
  actual_mass=torch.as_tensor(tensor(e.robot.root_view.get_masses()),device=self.device)[ids,self.root_body]
  actual_inertia=torch.as_tensor(tensor(e.robot.root_view.get_inertias()),device=self.device)[ids,self.root_body]
  if not torch.allclose(actual_mass,mass,atol=2e-6,rtol=1e-5) or not torch.allclose(actual_inertia,inertia,atol=2e-7,rtol=1e-5):raise RuntimeError('Motion DR mass/inertia backend readback differs')
  host_ids=ids.detach().cpu().long();coefficient=self.params[ids,4].detach().cpu()
  if self.material_shape_columns is not None:
   root=e.robot.root_view;values=material_tensor(root).clone()
   columns=torch.tensor(list(self.material_shape_columns.values()))
   values[host_ids[:,None],columns[None,:],:2]=coefficient[:,None,None]
   set_material_tensor(root,values,host_ids.to(torch.int32))
   # The complete articulation tensor preserves every non-wheel shape and every
   # non-reset row. Verify those too, independently of the wheel-body views below.
   if not torch.allclose(material_tensor(root),values,atol=2e-6,rtol=0):
    raise RuntimeError('Motion DR articulation material readback differs')
  for side,v in self.views.items():
   if self.material_shape_columns is None:
    values=material_tensor(v).clone();values[host_ids,0,0]=coefficient;values[host_ids,0,1]=coefficient
    set_material_tensor(v,values,host_ids.to(torch.int32))
   actual=material_tensor(v)[host_ids,0,:2]
   different=(actual-coefficient[:,None]).abs().amax(-1)>2e-6
   if different.any():
    bad=torch.where(different)[0][:8]
    self.material_failure=dict(side=side,env_ids=host_ids[bad].tolist(),expected=coefficient[bad].tolist(),actual=actual[bad].tolist(),
     maximum_error=float((actual-coefficient[:,None]).abs().max()),backend=self.config.get('material_backend','legacy_continuous'))
    raise RuntimeError('Motion DR friction backend readback differs '+side+' '+str(self.material_failure))
  self.last_readback=dict(root_body=self.root_body_name,env_ids=ids[:8].cpu().tolist(),mass_scale=self.params[ids[:8],3].cpu().tolist(),
   wheel_shape_coefficient=self.params[ids[:8],4].cpu().tolist(),mass_inertia_readback_verified=True,friction_readback_verified=True,
   combine='average on ground and wheel',material_backend=self.config.get('material_backend','legacy_continuous'),material_writer=self.config.get('material_writer','rigid_body_view'),
   wheel_shape_columns=self.material_shape_columns,scope='standing target only; no motor/armature randomization')
 def capture(self):
  e=self.env;d=e.robot.data;t=e.motion_timing;slot=t.tick%32
  self.root_history[slot]=torch.cat((d.root_quat_w.torch,d.root_ang_vel_w.torch),-1)
  self.request_ticks[slot]=10*(t.tick//10)
 def adoption_readback(self):
  t=self.env.motion_timing;tick=t.tick-1
  if tick%5:return
  origin=self.request_ticks[(tick-t.delays)%32,self.ids[:,None],t.channels[None,:]]
  changed=(origin>=0)&(origin!=self.adopted_ticks)
  # Keep the normal reset buffer: rollout may run inside inference_mode while
  # the next explicit reset runs outside it.
  self.first_adoption_ages.copy_(torch.where(changed,tick-origin,self.first_adoption_ages))
  self.adopted_ticks.copy_(origin)
 def impulse(self,ids,pulse=0):
  if not len(ids):return
  e=self.env;before=e.robot.data.root_vel_w.torch[ids].clone();after=before.clone()
  angle=self.params[ids,6];dv=self.params[ids,5]*(-1 if pulse else 1)
  delta=torch.stack((torch.cos(angle)*dv,torch.sin(angle)*dv),-1);after[:,:2]+=delta
  e.robot.write_root_velocity_to_sim_index(root_velocity=after,env_ids=ids.to(torch.int32))
  actual=e.robot.data.root_vel_w.torch[ids]
  # Some backend caches update lazily; requested root-state change remains separately identified.
  self.last_push=dict(env_ids=ids[:8].cpu().tolist(),delta_world_XY_mps=delta[:8].cpu().tolist(),before=before[:8,:3].cpu().tolist(),
   requested_after=after[:8,:3].cpu().tolist(),pulse=pulse,method='bounded root velocity impulse; not a force or motor torque')
  self.push_count+=len(ids)
 def maybe_push(self):
  if not self.push_enabled:return
  e=self.env;steps=e.rise_hold.episode.steps
  for pulse,tick in enumerate((1200,2200)):
   due=(steps>=tick)&~self.pushed[:,pulse]&(self.params[:,5]>0)&(e.target_xy.abs().sum(-1)<1e-6)
   ids=torch.where(due)[0]
   if len(ids):self.impulse(ids,pulse);self.pushed[ids,pulse]=True
 def summary(self):
  p=self.params.detach().cpu()
  return dict(version=dr.VERSION,state={k:self.state[k] for k in ('stage','DR_level','certified_DR_level','mode')},override=self.override,parameters=['Actor_age_ms','target_bus_offset_ms','PD_age_ms','root_mass_scale','wheel_shape_friction','push_delta_v_mps','push_angle_world_rad'],
   minimum=p.amin(0).tolist(),maximum=p.amax(0).tolist(),reset_count=self.reset_count,backend=self.last_readback,push_count=self.push_count,last_push=self.last_push,
   actor_actual_age_ms=self.env.actor_sensor_packets.age[:8].detach().cpu().tolist(),root_history_clock_ms=1,
   first_PD_adoption_age_ms=self.first_adoption_ages[:8].cpu().tolist(),previous='PD adopted physical target; never sensor-delayed',standing_only=['root mass/inertia','wheel friction','push'],bounds_are_engineering_priors=True)


def subclass(base,cfg):
 class RobustMotionEnv(base):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs)
   self.motion_dr=Adapter(self,cfg);self.actor_sensor_packets=DelayedPackets(self.motion_dr)
  def _reset_idx(self,ids):
   super()._reset_idx(ids)
   if hasattr(self,'motion_dr'):self.motion_dr.reset(ids)
  def _pre_physics_step(self,action):
   if hasattr(self,'motion_dr'):self.motion_dr.maybe_push()
   super()._pre_physics_step(action)
  def _apply_action(self):
   if hasattr(self,'motion_dr'):self.motion_dr.capture()
   super()._apply_action()
   if hasattr(self,'motion_dr'):self.motion_dr.adoption_readback()
 return RobustMotionEnv
