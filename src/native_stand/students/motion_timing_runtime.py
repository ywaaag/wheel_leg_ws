"""GPU1ms execution rings; opt-in Motion timing, legacy paths unchanged."""
import torch
from .motion_timing import TARGETS
LEGS=[0,1,3,4];WHEELS=[2,5]

class Timing:
 def __init__(self,env,configuration,stage):
  self.env=env;self.config=configuration;self.stage=stage;self.override=None
  from .motion_fixed_timing import VERSION as FIXED,configuration as fixed_configuration
  from .motion_sensor_transfer import VERSION as SENSOR,configuration as sensor_configuration
  self.antiosc=configuration.get('version')=='motion31_standing_antiosc_armature_v4'
  self.sensor_transfer=configuration.get('version')==SENSOR or self.antiosc
  self.fixed=configuration.get('version') in (FIXED,SENSOR) or self.antiosc
  expected=sensor_configuration() if self.sensor_transfer else fixed_configuration()
  if self.antiosc:
   from .motion_antioscillation import validate_configuration
   validate_configuration(configuration);expected=configuration
  if self.fixed and configuration!=expected:raise ValueError('unknown fixed timing parameters')
  cfg=getattr(env,'cfg',None)
  self.kp=getattr(cfg,'leg_kp',60.);self.kd=getattr(cfg,'leg_kd',2.);self.kv=getattr(cfg,'wheel_kv',1.)
  wheel=configuration.get('wheel_execution');expected_kv=wheel['wheel_kv_Nm_per_rad_s'] if wheel is not None else 1.
  if self.fixed and (self.kp,self.kd,self.kv)!=(40.,2.,expected_kv):raise ValueError('fixed timing effective PD differs')
  self.n=env.num_envs;self.device=env.device;self.tick=0;self.length=32
  self.ids=torch.arange(self.n,device=self.device);self.channels=torch.arange(6,device=self.device)
  self.limits=torch.tensor([40,40,5.5,40,40,5.5],device=self.device)
  self.scales=torch.tensor([1.5,1.5,100,1.5,1.5,100],device=self.device)
  self.episodes=torch.zeros(self.n,device=self.device,dtype=torch.long)
  self.delays=torch.zeros(self.n,6,device=self.device,dtype=torch.long)
  self.feedback=torch.zeros(self.n,device=self.device,dtype=torch.long)
  self.tau=torch.zeros(self.n,6,device=self.device)
  self.targets=torch.zeros(self.length,self.n,6,device=self.device)
  self.sensors=torch.zeros(self.length,self.n,12,device=self.device)
  self.requested=torch.zeros(self.n,6,device=self.device);self.adopted=self.requested.clone();self.actual=self.requested.clone()
  self.demand=self.requested.clone();self.readback=None;self.wheel_readback=None
  if wheel is not None:
   from .motion_wheel_matching import setup
   self.wheel_readback=setup(env,wheel)
  self.reset(self.ids)
 def configure(self,level=None):
  self.override=level
  if self.antiosc:self.configure_armature()
 def configure_armature(self):
  from .motion_antioscillation import ARMATURES
  from native_stand.control_readback import tensor
  e=self.env
  if getattr(e,'motion_config',{}).get('course_version') in ('motion31_posture_reacquisition_v1','motion31_alternating_standing_robustness_v1','motion31_fixed_height_motion_v2','motion31_persistent_frontiers_v1'):
   self.override=4  # Command curriculum stage never selects physical inertia.
  if self.override is not None:
   if self.override not in range(5):raise ValueError('inertia override invalid')
   levels=torch.full((self.n,),self.override,device=self.device,dtype=torch.long)
  elif e.curriculum_training:
   group=torch.arange(self.n,device=self.device)%4;levels=torch.where(group==0,0,torch.where(group==1,max(0,self.stage-1),self.stage))
  else:levels=torch.full((self.n,),self.stage,device=self.device,dtype=torch.long)
  values=torch.tensor(ARMATURES,device=self.device)[levels];ids=[e.ids[i] for i in LEGS]
  e.robot.write_joint_armature_to_sim_index(armature=values[:,None].expand(-1,4).contiguous(),joint_ids=torch.tensor(ids,device=self.device,dtype=torch.int32))
  actual=torch.as_tensor(tensor(e.robot.root_view.get_dof_armatures()),device=self.device)[:,ids]
  if not torch.allclose(actual,values[:,None].expand(-1,4),atol=1e-6,rtol=0):raise RuntimeError('PhysX inertia cohort readback differs')
  self.actual_armature=actual[:,0:1].clone();self.armature_levels=levels.clone()
  self.readback={'levels':levels.cpu().bincount(minlength=5).tolist(),'values_kg_m2':list(ARMATURES),'four_leg_readback_verified':True,'sampling':'setup only, no physical-step CPU write'}
 def reset(self,ids):
  self.episodes[ids]+=1
  # Counter stream independent of policy/torch sampling RNG; reset-only variation.
  v=(ids[:,None]*1103515245+self.episodes[ids,None]*12345+torch.arange(16,device=self.device)[None,:]*2654435761+self.config['seed'])%2147483647
  u=((v*48271)%2147483647).float()/2147483647
  if self.antiosc:strength=torch.zeros(len(ids),device=self.device)
  elif self.override is not None:strength=torch.full((len(ids),),0. if self.override<0 else TARGETS[self.override],device=self.device)
  elif self.env.curriculum_training:
   group=ids%4;strength=torch.full((len(ids),),TARGETS[self.stage],device=self.device)
   strength=torch.where(group==0,0.,strength)
   strength=torch.where(group==1,torch.full_like(strength,TARGETS[self.stage-1] if self.stage else 0.),strength)
  else:strength=torch.zeros(len(ids),device=self.device)
  delay=torch.floor(u[:,:6]*11*strength[:,None]).long()
  # Nominal wheel onset hypotheses: geometric rightch2=7, leftch5=11;
  # split between transportation OR rise, never count the full value twice.
  response=u[:,6]>.5
  delay[:,2]=torch.round(strength*7).long();delay[:,5]=torch.round(strength*11).long()
  delay[:,WHEELS]=torch.where(response[:,None],0,delay[:,WHEELS])
  tau=u[:,7:13]*5*strength[:,None]
  tau[:,2]=torch.where(response,strength*7,0.);tau[:,5]=torch.where(response,strength*11,0.)
  if self.fixed:
   delay[:]=torch.tensor(self.config['target_delay_ms_ch0_ch5'],device=self.device)
   tau[:]=torch.tensor(self.config['torque_tau_ms_ch0_ch5'],device=self.device)
  self.delays[ids]=delay;self.feedback[ids]=10 if self.fixed else torch.floor(u[:,13]*11*strength).long();self.tau[ids]=tau
  self.alpha=-torch.expm1(-1./self.tau.clamp_min(1e-9))
  e=self.env;d=e.robot.data
  q=d.joint_pos.torch[:,e.ids];dq=d.joint_vel.torch[:,e.ids]
  target=torch.zeros(len(ids),6,device=self.device);target[:,LEGS]=e.executed_leg[ids];target[:,WHEELS]=e.executed_wheel[ids]
  self.requested[ids]=target;self.adopted[ids]=target;self.actual[ids]=0.;self.demand[ids]=0.
  self.targets[:,ids]=target[None];self.sensors[:,ids]=torch.cat((q[ids],dq[ids]),-1)[None]
  if self.antiosc and hasattr(self,'history_valid'):
   self.history_valid[ids]=False;self.prev_torque[ids]=0.;self.prev_target[ids]=target;self.older_target[ids]=target
   if hasattr(self,'critic_previous_torque'):self.critic_previous_torque[ids]=0.
 def apply(self):
  e=self.env;d=e.robot.data;slot=self.tick%self.length
  q=d.joint_pos.torch[:,e.ids];dq=d.joint_vel.torch[:,e.ids]
  self.targets[slot]=self.requested;self.sensors[slot,:,:6]=q;self.sensors[slot,:,6:]=dq
  if self.tick%5==0:
   delivered=self.targets[(self.tick-self.delays)%self.length,self.ids[:,None],self.channels[None,:]]
   self.adopted.copy_(delivered);s=self.sensors[(self.tick-self.feedback)%self.length,self.ids]
   request=torch.zeros_like(self.demand);request[:,LEGS]=self.kp*(self.adopted[:,LEGS]-s[:,LEGS])-self.kd*s[:,[6+i for i in LEGS]]
   request[:,WHEELS]=self.kv*(self.adopted[:,WHEELS]-s[:,[6+i for i in WHEELS]])
   self.demand.copy_(request.clamp(-self.limits,self.limits))
   e.effort_pre.copy_(request)
  alpha=self.alpha if self.fixed else -torch.expm1(-1./self.tau.clamp_min(1e-9));self.actual.add_(alpha*(self.demand-self.actual))
  e.effort.copy_(self.actual)
  e.executed_leg.copy_(self.adopted[:,LEGS]);e.executed_wheel.copy_(self.adopted[:,WHEELS])
  e.robot.set_joint_effort_target_index(target=e.effort,joint_ids=e.ids)
  e.pd_counter+=1;self.tick+=1
 def features(self):
  return torch.cat((self.delays.float()/20,self.feedback[:,None].float()/10,self.tau/10,
                    self.actual/self.limits,(self.adopted-self.requested)/self.scales),-1)
 def summary(self):
  return {'version':self.config['version'],'stage':self.stage,'override':self.override,'physics_ticks':self.tick,
   'delay_ms_min':self.delays.amin(0).cpu().tolist(),'delay_ms_max':self.delays.amax(0).cpu().tolist(),
   'feedback_ms_max':int(self.feedback.max()),'tau_ms_max':self.tau.amax(0).cpu().tolist(),
   'actor_sensor':self.config.get('actor_sensor'),'reward_extension':self.config.get('reward_extension'),
   'actuator_readback':self.readback,'finite':bool(torch.isfinite(self.actual).all()),'critic_extra_dim':25,'previous':'last PD adopted physical target',
   'leg_kp':self.kp,'leg_kd':self.kd,'wheel_kv':self.kv,'wheel_execution_readback':self.wheel_readback,
   'sampling':self.config['sampling'] if self.fixed else 'episode reset only; counter stream independent from policy RNG','provenance':'hypothesis, not identified motor model'}

def subclass(base,configuration,stage,*,mcu_execution=False):
 class TimingMotionEnv(base):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs)
   timing_class=Timing
   if mcu_execution:
    from .motion_mcu_runtime import MCUTiming
    timing_class=MCUTiming
    if getattr(self,'mcu_execution',{}).get('profile',{}).get('version')=='motion_mcu_sensor20_target5_motor20_v1':
     from .motion_standing_delay_runtime import StandingDelayTiming
     timing_class=StandingDelayTiming
   self.motion_timing=timing_class(self,configuration,stage)
   if self.motion_timing.sensor_transfer:
    from .motion_sensor_transfer import SensorPackets
    self.actor_sensor_packets=SensorPackets(self.num_envs,self.device,delay_ms=configuration['actor_sensor']['imu_delay_ms'])
   if self.motion_timing.fixed:
    from .motion_fixed_timing import verify_armature
    self.motion_timing.readback=verify_armature(self)
   if self.motion_timing.antiosc:
    t=self.motion_timing;t.history_valid=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
    t.prev_torque=torch.zeros_like(t.actual);t.critic_previous_torque=torch.zeros_like(t.actual);t.prev_target=t.adopted.clone();t.older_target=t.adopted.clone();t.last_reward_tick=None
    t.configure_armature()
  def _reset_idx(self,ids):
   super()._reset_idx(ids)
   if hasattr(self,'motion_timing'):
    ids=torch.arange(self.num_envs,device=self.device) if ids is None else ids
    self.motion_timing.reset(ids)
    if hasattr(self,'actor_sensor_packets'):self.actor_sensor_packets.reset(ids)
  def _pre_physics_step(self,action):
   super()._pre_physics_step(action)
   if hasattr(self,'motion_timing'):
    t=self.motion_timing;t.requested[:,LEGS]=self.leg_target;t.requested[:,WHEELS]=self.wheel_target
    self.executed_leg.copy_(t.adopted[:,LEGS]);self.executed_wheel.copy_(t.adopted[:,WHEELS])
  def _apply_action(self):
   if hasattr(self,'motion_timing'):self.motion_timing.apply()
   else:super()._apply_action()
  def _get_observations(self):
   original=self.critic_privileged_version
   if hasattr(self,'motion_timing'):self.critic_privileged_version='motion_symmetry_privileged85_v1'
   try:obs=dict(super()._get_observations())
   finally:self.critic_privileged_version=original
   if hasattr(self,'motion_timing') and 'student_critic' in obs:
    obs['student_critic']=torch.cat((obs['student_critic'],self.motion_timing.features()),-1)
   current_frame=obs['student_actor'][:,:16] if original=='motion_antiosc_privileged140_v1' and 'student_actor' in obs else None
   if hasattr(self,'actor_sensor_packets') and 'student_actor' in obs:
    delayed=self.actor_sensor_packets.read(obs['student_actor'][:,:16],self.motion_timing.tick)
    result=dict(obs);result['student_actor']=obs['student_actor'].clone();result['student_actor'][:,:16]=delayed
    if 'student_critic' in obs:
     result['student_critic']=obs['student_critic'].clone();result['student_critic'][:,:31]=result['student_actor']
    obs=result
   if original=='motion_antiosc_privileged140_v1' and 'student_critic' in obs:
    from .motion_antiosc_privileged import critic_input
    t=self.motion_timing
    mix=obs['student_critic'].new_full((self.num_envs,1),t.config['smooth_mix'])
    obs['student_critic']=critic_input(obs['student_critic'],t.actual_armature,current_frame,self.effort_pre,t.critic_previous_torque if t.last_reward_tick==t.tick else t.prev_torque,mix)
   return obs
  def _get_dones(self):
   result=super()._get_dones()
   if hasattr(self,'actor_sensor_packets') and getattr(self,'motion_quality_ready',False):
    from .motion_sensor_transfer import apply_costs,configuration as sensor_configuration
    t=self.motion_timing;d=self.robot.data;angular,_=self._motion_state()
    self.student_reward_result=apply_costs(self.student_reward_result,targets=self.motion_spin_executed_targets,
      previous=self.motion_spin_previous_targets,dq=d.joint_vel.torch[:,self.ids],vertical_velocity=d.root_lin_vel_w.torch[:,2],
      angular_body=angular,torque=d.applied_torque.torch[:,self.ids],config=sensor_configuration()['reward_extension'] if t.antiosc else t.config['reward_extension'])
    if t.antiosc:
     from .motion_antioscillation import apply_costs as anti_costs
     self.student_reward_result=anti_costs(self.student_reward_result,self.effort_pre,t.actual,t.prev_torque,t.adopted,t.prev_target,t.older_target,self.contact_force,t.history_valid,t.config)
     if t.last_reward_tick!=t.tick:
      t.critic_previous_torque.copy_(t.prev_torque)
      t.older_target.copy_(t.prev_target);t.prev_target.copy_(t.adopted);t.prev_torque.copy_(t.actual);t.history_valid.fill_(True);t.last_reward_tick=t.tick
   return result
 return TimingMotionEnv
