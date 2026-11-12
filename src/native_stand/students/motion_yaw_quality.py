"""Opt-in in-place spin quality; explicit world-space cylinder kinematics, no IO."""
import torch
VERSION='student_motion31_yaw_contact_quality_v7'
FIELDS=('spin_translation','spin_common_rolling','spin_leg_smooth','spin_wheel_smooth',
        'spin_single_contact_deficit','spin_slip')

def configuration():
 return {'version':'motion31_spin_quality_cost_v1','cohort':'episode target vx==0 and yaw!=0',
  'activation':'smoothstep by effective abs(yaw)/.5; zero in prep and completed parking',
  'translation_scale_mps':.3,'common_rolling_scale_mps':.3,
  'leg_target_delta_scale_rad':.1,'wheel_target_delta_scale_rad_s':10.,
  'slip_scale_mps':.2,'contact_threshold_n':2.,'radius_m':.06,'half_width_m':.0165,
  'weights':[-.5,-.1,-.02,-.02,-.5,-.1],
  'contact_semantics':'per-wheel resultant N; not certified normal load',
  'slip_semantics':'finite cylinder geometric support point velocity in stationary Z-up Plane; not solver contact patch',
  'sampling':'100Hz post-physics/pre-reset; not 200Hz PD or 1kHz physics',
  'total_clip':None,'source_checkpoint':'parent810; do not override Actor or wheel targets'}

def measurement(center,com,linear,angular,axis,forward,*,radius=.06,half_width=.0165):
 """Absolute body angular velocity includes axle spin and base/leg rotation.

 forward is the robot forward direction in world, projected onto Plane. No joint
 speed polarity assumption; center/COM distinction is explicit. All outputs stay GPU.
 """
 if center.ndim!=3 or center.shape[1:]!=(2,3) or any(t.shape!=center.shape for t in (com,linear,angular,axis)) or forward.shape!=(len(center),3):raise ValueError('spin geometry shape differs')
 if radius<=0 or half_width<0:raise ValueError('spin cylinder dimensions invalid')
 a=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-8)
 up=torch.zeros_like(a);up[...,2]=1.;dot=(up*a).sum(-1,keepdim=True)
 radial=up-dot*a;norm=radial.norm(dim=-1,keepdim=True)
 offset=-radius*radial/norm.clamp_min(1e-8)-half_width*torch.where(dot.abs()>1e-6,dot.sign(),0.)*a
 support=center+offset;v=linear+torch.cross(angular,support-com,dim=-1)
 center_v=linear+torch.cross(angular,center-com,dim=-1)
 f=forward.clone();f[:,2]=0.;fn=f.norm(dim=-1,keepdim=True);f=f/fn.clamp_min(1e-8)
 rotational=torch.cross(angular,offset,dim=-1)
 rolling=-(rotational*f[:,None,:]).sum(-1)
 valid=(norm.squeeze(-1)>1e-5)&(axis.norm(dim=-1)>1e-6)&(fn>1e-6)
 return {'tangent_velocity_w':v[...,:2],'center_velocity_w':center_v,
         'rolling_speed_forward_mps':rolling,'geometry_valid':valid,'center_world':center,'com_world':com,
         'linear_com_world':linear,'angular_world':angular,'axis_world':a,'forward_world':forward}

def runtime_measurement(env):
 reduced=getattr(env.cfg,'motion_runtime_reduction',False)
 key=(getattr(env,'_sim_step_counter',None),getattr(env,'_motion_reset_epoch',0))
 if reduced and key[0] is not None and getattr(env,'_motion_spin_cache_key',None)==key:
  return env._motion_spin_cache
 d=env.robot.data;ids=env.wheel_body_ids
 def value(name):
  if not hasattr(d,name):raise ValueError('spin quality runtime field unavailable: '+name)
  v=getattr(d,name);return v.torch if hasattr(v,'torch') else v
 from isaaclab.utils.math import quat_apply
 q=value('body_link_quat_w')[:,ids]
 axis=quat_apply(q,q.new_tensor([1.,0.,0.]).expand(*q.shape[:-1],3))
 root_q=value('root_quat_w');forward=quat_apply(root_q,root_q.new_tensor([0.,0.,-1.]).expand(len(root_q),3))
 result=measurement(value('body_link_pos_w')[:,ids],value('body_com_pos_w')[:,ids],
  value('body_com_lin_vel_w')[:,ids],value('body_com_ang_vel_w')[:,ids],axis,forward,
  radius=configuration()['radius_m'],half_width=configuration()['half_width_m'])
 if reduced and key[0] is not None:env._motion_spin_cache_key=key;env._motion_spin_cache=result
 return result

def apply(result,*,target,command,velocity,contacts,previous_targets,executed_targets,measurement,config):
 from .motion_tracking_reward import huber
 if config!=configuration():raise ValueError('spin quality declaration differs')
 n=len(target)
 if target.shape!=(n,2) or command.shape!=(n,3) or velocity.shape!=(n,3) or contacts.shape!=(n,2) or previous_targets.shape!=(n,6) or executed_targets.shape!=(n,6):raise ValueError('spin cost shape differs')
 m=measurement
 if m['tangent_velocity_w'].shape!=(n,2,2) or m['rolling_speed_forward_mps'].shape!=(n,2) or m['geometry_valid'].shape!=(n,2):raise ValueError('spin measured fields differ')
 yaw_cohort=(target[:,0]==0)&(target[:,1]!=0)
 # Nonzero during scheduled ramp/hold, zero during prep and after return to zero.
 u=(command[:,1].abs()/.5).clamp(0,1);alpha=u.square()*(3-2*u)
 gate=(yaw_cohort&result['eligible']).to(velocity.dtype)*alpha
 delta=executed_targets-previous_targets
 contact_deficit=((config['contact_threshold_n']-contacts)/config['contact_threshold_n']).clamp(0,1).square().mean(-1)
 touching=(contacts>config['contact_threshold_n'])&m['geometry_valid']
 slip=(huber(m['tangent_velocity_w']/config['slip_scale_mps']).sum(-1)*touching).mean(-1)
 common=m['rolling_speed_forward_mps'].mean(-1)
 raw=torch.stack((huber(velocity[:,:2]/config['translation_scale_mps']).sum(-1),
  huber(common/config['common_rolling_scale_mps']),
  huber(delta[:,[0,1,3,4]]/config['leg_target_delta_scale_rad']).mean(-1),
  huber(delta[:,[2,5]]/config['wheel_target_delta_scale_rad_s']).mean(-1),contact_deficit,slip),-1)*gate[:,None]
 density=raw*raw.new_tensor(config['weights']);dt=density*.01
 out=dict(result);out.update(version=VERSION,fields=tuple(result['fields'])+FIELDS,
  raw=torch.cat((result['raw'],raw),-1),density=torch.cat((result['density'],density),-1),
  dt=torch.cat((result['dt'],dt),-1),total=result['total']+dt.sum(-1))
 out['spin_diagnostics']={**m,'activation':gate,'raw':raw}
 return out

def audit():
 """Curves from apply itself; not historical PPO or a convergence claim."""
 rows=[]
 for field in FIELDS:
  for error in (0.,.005,.01,.02,.05,.1,.2,.3,.5,1.,2.,4.):
   x=torch.tensor(error,dtype=torch.float64,requires_grad=True);z=x.new_zeros(1);yes=torch.ones(1,dtype=torch.bool)
   base={'eligible':yes,'fields':(),'raw':x.new_zeros(1,0),'density':x.new_zeros(1,0),'dt':x.new_zeros(1,0),'total':z}
   velocity=x.new_zeros(1,3);contacts=x.new_full((1,2),20.);executed=x.new_zeros(1,6)
   m={'tangent_velocity_w':x.new_zeros(1,2,2),'rolling_speed_forward_mps':x.new_zeros(1,2),'geometry_valid':torch.ones(1,2,dtype=torch.bool)}
   if field=='spin_translation':velocity=torch.stack((z+x,z,z),-1)
   if field=='spin_common_rolling':m['rolling_speed_forward_mps']=x.expand(1,2)
   if field=='spin_leg_smooth':executed=torch.stack((z+x,z+x,z,z+x,z+x,z),-1)
   if field=='spin_wheel_smooth':executed=torch.stack((z,z,z+x,z,z,z+x),-1)
   if field=='spin_single_contact_deficit':contacts=(2.-x).expand(1,2)
   if field=='spin_slip':m['tangent_velocity_w']=torch.stack((x,z[0])).expand(1,2,2)
   out=apply(base,target=x.new_tensor([[0.,.5]]),command=x.new_tensor([[0.,.5,.34]]),velocity=velocity,contacts=contacts,
    previous_targets=x.new_zeros(1,6),executed_targets=executed,measurement=m,config=configuration())
   i=FIELDS.index(field);y=out['density'][0,i];g=torch.autograd.grad(y,x)[0]
   rows.append({'field':field,'error':error,'raw':float(out['raw'][0,i].detach()),'weighted_density':float(y.detach()),
    'dt_contribution':float(y.detach()*.01),'state_error_derivative':float(g),'unit':'N deficit' if field.endswith('deficit') else 'rad' if field=='spin_leg_smooth' else 'rad/s' if field=='spin_wheel_smooth' else 'm/s'})
 return {'configuration':configuration(),'curves':rows,'interpretation':'actual tensor cost curves; not Actor gradients; no convergence guarantee'}

STAT_FIELDS=('left_slip_speed_mps','right_slip_speed_mps','left_rolling_speed_mps','right_rolling_speed_mps',
 'common_rolling_abs_mps','leg_target_delta_abs_rad','wheel_target_delta_abs_rad_s','left_contact_N','right_contact_N','activation')
class SpinStatistics:
 """Fixed cohort reductions; no per-physics-step CPU transfer, one export/update."""
 def __init__(self,cohorts,device):
  self.count=torch.zeros(cohorts,device=device);self.sum=torch.zeros(cohorts,len(STAT_FIELDS),device=device)
 def observe(self,index,valid,hold,diagnostics,contacts,previous_targets,executed_targets):
  m=diagnostics;delta=(executed_targets-previous_targets).abs()
  slip=m['tangent_velocity_w'].norm(dim=-1)
  data=torch.cat((slip,m['rolling_speed_forward_mps'],m['rolling_speed_forward_mps'].mean(-1).abs()[:,None],
   delta[:,[0,1,3,4]].mean(-1)[:,None],delta[:,[2,5]].mean(-1)[:,None],contacts,m['activation'][:,None]),-1)
  mask=valid&hold
  self.count.scatter_add_(0,index,mask.to(self.count.dtype))
  self.sum.scatter_add_(0,index[:,None].expand(-1,len(STAT_FIELDS)),torch.where(mask[:,None],data,0.))
 def export(self):
  packed=torch.cat((self.count[:,None],self.sum/self.count.clamp_min(1)[:,None]),-1).cpu().tolist()
  return {'scope':'pre-reset policy100Hz hold samples, contact point is geometric approximation',
   'cohorts':{str(i):{'samples':int(row[0]),'mean':dict(zip(STAT_FIELDS,row[1:])) if row[0] else None} for i,row in enumerate(packed)}}
