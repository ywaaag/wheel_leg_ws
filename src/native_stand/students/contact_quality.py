"""Pure Plane/cylinder contact velocity and versioned Recovery quality recipe."""
import torch
from .recovery_large_training import SOURCE
VERSION='student_recovery_contact_quality_ppo_v1'
REWARD='student_recovery_contact_quality_reward_v1'
CRITIC='recovery_contact64_v2'
FIELDS=('post_entry_wheel_smooth','post_entry_contact_deficit','post_entry_slip')

def network():
 from .recovery_large import network as base
 n=base();n['version']='recovery31_contact_large_actor_critic_v2';return n

def contract():
 from .recovery_privileged import contract as base
 c=base();c.update(critic_privileged_version=CRITIC,actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,512,256])
 c['critic_privileged']=c['critic_privileged'][:9]+['left_right_contact_point_vxy_world/.5:4','left_right_wheel_center_vxy_world/4:4','left_right_applied_wheel_torque/5.5:2','left_right_contact_geometry_valid:2']
 c['critic_layout']='Actor31 + existing Recovery privileged first21 + explicit contact12; old cumulative hold summaries removed only in v2'
 return c

def configuration():
 from .recovery_large_training import configuration as base
 c=base();c['version']=REWARD
 c['contact_quality']={'version':'plane_cylinder_contact_quality_v1','entry_ramp_s':1.,
  'wheel_delta_scale_rad_s':10.,'contact_threshold_n':2.,'slip_scale_mps':.2,
  'weights':dict(zip(FIELDS,[-.2,-.5,0.])),
  'slip_penalty_enabled':False,'slip_activation':'requires new explicit recipe and runtime signed velocity validation evidence',
  'contact_measurement':'per-wheel sensor resultant, not certified normal load',
  'velocity':'COM linear world + absolute angular world cross (geometric support point minus COM); Plane stationary',
  'radius_m':.06,'half_width_m':.0165,'axle_local':[1.,0.,0.],
  'sampling':'100Hz held policy sample; not full PD/physics frequency'}
 c['weights'].update(c['contact_quality']['weights']);return c

def recipe(source_sha256=None):
 from .recovery_large_training import recipe as base
 r=base(source_sha256);r.update(version=VERSION,reward=configuration(),critic_privileged_version=CRITIC,
  network_architecture=network(),critic='new_random_contact64_v2',selected_rollouts_formal=[])
 r['checkpoint_quick_evaluation']['version']='recovery_single_native_screen_v1'
 return r

def point_velocity(center_world,com_world,com_velocity_world,angular_world,axle_world,*,radius=.06,half_width=.0165):
 """Support point of finite cylinder above stationary Z-up Plane, world units.

 Uses total body angular velocity, so no left/right joint sign assumption. The
 support point approximates rigid collider geometry, not solver contact patch.
 """
 tensors=(center_world,com_world,com_velocity_world,angular_world,axle_world)
 if any(t.shape!=center_world.shape or t.shape[-1]!=3 or not torch.isfinite(t).all() for t in tensors):raise ValueError('contact velocity explicit finite world3 fields required')
 if radius<=0 or half_width<0:raise ValueError('invalid cylinder geometry')
 norm=axle_world.norm(dim=-1,keepdim=True)
 if (norm<1e-6).any():raise ValueError('zero wheel axle')
 axis=axle_world/norm;up=torch.zeros_like(axis);up[...,2]=1
 dot=(up*axis).sum(-1,keepdim=True);radial=up-dot*axis
 radial_norm=radial.norm(dim=-1,keepdim=True);valid=radial_norm.squeeze(-1)>1e-5
 # On exactly vertical axle the lowest cap has no unique rolling direction.
 offset=-radius*radial/radial_norm.clamp_min(1e-5)-half_width*torch.where(dot.abs()>1e-6,dot.sign(),0.)*axis
 point=center_world+offset
 velocity=com_velocity_world+torch.cross(angular_world,point-com_world,dim=-1)
 center_velocity=com_velocity_world+torch.cross(angular_world,center_world-com_world,dim=-1)
 return {'center_w':center_world,'com_w':com_world,'com_velocity_w':com_velocity_world,'angular_velocity_w':angular_world,'support_point_w':point,'velocity_w':velocity,'tangent_velocity_w':velocity[...,:2],
  'center_velocity_w':center_velocity,'geometry_valid':valid,'axle_w':axis}

def runtime_measurement(env):
 """No finite-difference COM substitute or guessed joint polarity is allowed."""
 d=env.robot.data;ids=env.wheel_body_ids
 def value(name):
  if not hasattr(d,name):raise ValueError('contact measurement runtime field missing: '+name)
  v=getattr(d,name);return v.torch[:,ids]
 quat=value('body_link_quat_w')
 from isaaclab.utils.math import quat_apply
 axis=quat_apply(quat,quat.new_tensor([1.,0.,0.]).expand(*quat.shape[:-1],3))
 result=point_velocity(value('body_link_pos_w'),value('body_com_pos_w'),
  value('body_com_lin_vel_w'),value('body_com_ang_vel_w'),axis)
 names=env.robot.body_names
 parent_ids=[names.index(side+'_lower_leg') for side in ('left','right')]
 parent=d.body_com_ang_vel_w.torch[:,parent_ids]
 dq=d.joint_vel.torch[:,env.ids][:,[2,5]]
 relative=value('body_com_ang_vel_w')-parent
 result.update(parent_angular_velocity_w=parent,joint_spin_rad_s=dq,relative_spin_rad_s=(relative*axis).sum(-1))
 return result

def critic_input(base64,measurement,applied_wheel_torque):
 n=len(base64);m=measurement
 if base64.shape!=(n,64) or applied_wheel_torque.shape!=(n,2):raise ValueError('contact64 input shape differs')
 extra=torch.cat((m['tangent_velocity_w'].reshape(n,4)/.5,m['center_velocity_w'][...,:2].reshape(n,4)/4.,
  applied_wheel_torque/5.5,m['geometry_valid'].to(base64.dtype)), -1)
 out=torch.cat((base64[:,:52],extra),-1)
 if out.shape!=(n,64) or not torch.isfinite(out).all():raise ValueError('invalid contact64 input')
 return out

def reward(*,measurement,entry_elapsed,**kwargs):
 from .recovery_large_training import configuration as parent_config,reward as parent_reward
 from .recovery_broad_narrow_reward import huber
 c=kwargs.pop('reward_configuration')
 if c!=configuration():raise ValueError('contact-quality reward differs')
 r=parent_reward(**kwargs,entry_elapsed=entry_elapsed,reward_configuration=parent_config())
 alpha=r['diagnostics']['stop_alpha'];gate=alpha*kwargs['entered']*r['eligible']
 delta=(kwargs['executed_targets'][:,[2,5]]-kwargs['previous_targets'][:,[2,5]])/10.
 deficit=((2.-kwargs['contacts'])/2.).clamp(0,1).square().mean(-1)
 touching=(kwargs['contacts']>2)&measurement['geometry_valid']
 # Mean over two wheels avoids higher weight when only one wheel contacts.
 slip=(huber(measurement['tangent_velocity_w']/.2).sum(-1)*touching).mean(-1)
 extra=torch.stack((gate*delta.square().mean(-1),gate*deficit,gate*slip),-1)
 weighted=extra*extra.new_tensor([c['contact_quality']['weights'][k] for k in FIELDS])
 r.update(version=REWARD,fields=tuple(r['fields'])+FIELDS,raw=torch.cat((r['raw'],extra),-1),
  weighted_density=torch.cat((r['weighted_density'],weighted),-1),continuous_dt=torch.cat((r['continuous_dt'],weighted*.01),-1),total=r['total']+weighted.sum(-1)*.01)
 r['diagnostics']['contact_quality']=measurement
 return r
