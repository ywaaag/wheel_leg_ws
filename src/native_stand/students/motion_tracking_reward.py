"""Motion density: independent tracking scores plus unsaturated Huber tails."""
import torch
VERSION='student_motion31_tracking_reward_v1'
REFINEMENT_VERSION='student_motion31_tracking_reward_refinement_v2'
QUALITY_VERSION='student_motion31_quality_reward_v3'
SUCCESSOR_VERSION='student_motion31_hold_level_reward_v4'
YAW_NARROW_VERSION='student_motion31_yaw_narrow_quality_v5'
YAW_NARROW_BASE='student_motion31_yaw_narrow_refinement_v5'
COMBO_VERSION='student_motion31_large_gated_quality_v6'
COMBO_BASE='student_motion31_large_gated_refinement_v6'
SPIN_VERSION='student_motion31_yaw_contact_quality_v7'
DIFFERENTIAL_VERSION='student_motion31_differential_quality_v9'
SYMMETRY_VERSION='student_motion31_symmetry_quality_v8'
PARALLEL_VERSION='student_motion31_parallel_quality_v10'
QUALITY_VERSIONS=(PARALLEL_VERSION,QUALITY_VERSION,SUCCESSOR_VERSION,YAW_NARROW_VERSION,COMBO_VERSION,SPIN_VERSION,SYMMETRY_VERSION,DIFFERENTIAL_VERSION)
FIELDS=('support','linear_tracking','yaw_tracking','lateral_tracking','height','upright',
 'linear_error','yaw_error','lateral_error','mirror','wheel_alignment','leg_balance','effort','target_change')
def configuration(version=VERSION):
 if version==PARALLEL_VERSION:
  c=configuration(DIFFERENTIAL_VERSION);c.update(version=PARALLEL_VERSION,parallel_tracking={'version':'motion31_parallel_bidirectional_design_v1','new_denominator':'max(1,(.5*abs(effective_command))**2)','yaw_narrow_denominator':.25,'old_widths':'restored by episode target','geometry_costs':'v9 unchanged'})
  return c
 if version==DIFFERENTIAL_VERSION:
  from .motion_differential import configuration as differential
  return differential()
 if version==SYMMETRY_VERSION:
  from .motion_symmetry import configuration as symmetry
  return symmetry()
 if version==SPIN_VERSION:
  from .motion_yaw_quality import configuration as spin
  result=configuration(COMBO_VERSION);result.update(version=SPIN_VERSION,spin_quality=spin());return result
 if version in (COMBO_VERSION,COMBO_BASE):
  result=configuration(YAW_NARROW_VERSION if version==COMBO_VERSION else YAW_NARROW_BASE)
  result.update(version=version,yaw_translation_gate={'denominator_m2_s2':1.,'cohort':'episode target vx==0 and yaw!=0','measurement':'native horizontal speed squared vx^2+vy^2'})
  result['tracking_combination']='weighted sum; in-place yaw tracking multiplied by exp(-horizontal_speed_squared/1.0)'
  return result
 if version in (YAW_NARROW_VERSION,YAW_NARROW_BASE):
  result=configuration(QUALITY_VERSION if version==YAW_NARROW_VERSION else REFINEMENT_VERSION)
  result.update(version=version,yaw_denominator=.25)
  return result
 if version==SUCCESSOR_VERSION:
  result=configuration(QUALITY_VERSION);result['version']=SUCCESSOR_VERSION;result['weights'][5]=4.
  result['posture_tail']['weights']=[-1.,-1.]
  return result
 if version==QUALITY_VERSION:
  from .motion_quality import configuration as quality_configuration
  result=configuration(REFINEMENT_VERSION);result['version']=QUALITY_VERSION
  result.update(quality_termination=quality_configuration(),geometry_cost='Huber normalized error, same scales/weights, no bounded cap',
   posture_tail={'pitch_deadband_rad':.10,'pitch_scale_rad':.20,'roll_deadband_rad':.05,'roll_scale_rad':.10,'weights':[-.5,-.5]},terminal_event=-1.)
  return result
 if version not in (VERSION,REFINEMENT_VERSION):raise ValueError('unknown Motion reward version')
 result={'version':version,'dt':.01,'linear_denominator':1.,'yaw_denominator':1.,'lateral_denominator':.09,
 'zero_error_scale_mps':.30,'zero_yaw_error_scale_rad_s':.30,'linear_huber_scale_mps':1.,'yaw_huber_scale_rad_s':1.,'lateral_huber_scale_mps':.30,
 'height_target_m':.34,'height_scale_m':.04,'posture_scale_rad':.20,'mirror_scale_rad':.10,
 'wheel_deadband_m':.01,'wheel_scale_m':.03,'leg_balance_scale_m':.02,
 'weights':[.5,4.,2.,1.,2.,2.,-1.,-.5,-.5,-.2,-.3,-.15,-.02,-.002],
 'effort_limits_nm':[40.,40.,5.5,40.,40.,5.5],'target_delta_scales':[.1,.1,10.,.1,.1,10.],
 'cohort_semantics':'episode target determines zero/straight/yaw; ramps do not relabel cohort',
 'total_clip':None,'terminal_event':0.,'tracking_combination':'weighted sum; no velocity-times-yaw gate'}
 if version==REFINEMENT_VERSION:
  result['refinement']={'version':'motion31_evidence_width_v1','axis':'nonzero_linear_only','narrow_denominator':.04,
   'initial_mix':0.,'maximum_mix':.75,'mix_increment':.25,'confirmations':2,'MAE_max':.20,'direction_min':.95,
   'gain_range':[.8,1.2],'update_boundary':'after complete fixed evaluation, before new environment reset',
   'required_checks':['survival30','height','contact','saturation','closure','clearance','tilt','pitch_source_guard','roll_source_guard'],
   'regression':'freeze mixture and reset confirmation streak; no extra budget or automatic retry',
   'huber_tail':'unchanged','zero_cohort':'unchanged','yaw_cohort':'unchanged'}
 return result
def quality_base_configuration(version):
 if version not in QUALITY_VERSIONS:raise ValueError('quality base requires explicit quality reward version')
 return configuration(COMBO_BASE if version in (COMBO_VERSION,SPIN_VERSION,SYMMETRY_VERSION,DIFFERENTIAL_VERSION,PARALLEL_VERSION) else YAW_NARROW_BASE if version==YAW_NARROW_VERSION else REFINEMENT_VERSION)

def huber(x):
 a=x.abs();return torch.where(a<=1,.5*x.square(),a-.5)
def reward(*,command,target,velocity,yaw,height,angles,contacts,leg_q,wheel_offset,leg_lengths,
 applied_torque,previous_targets,executed_targets,controlled,valid,physical_failure,fault,config=None,linear_narrow_mix=None,spin_measurement=None):
 n=len(height);c=configuration() if config is None else config
 if c.get('version')==SPIN_VERSION:
  if c!=configuration(SPIN_VERSION) or spin_measurement is None:raise ValueError('spin quality requires explicit measured kinematics')
  base=reward(command=command,target=target,velocity=velocity,yaw=yaw,height=height,angles=angles,contacts=contacts,
   leg_q=leg_q,wheel_offset=wheel_offset,leg_lengths=leg_lengths,applied_torque=applied_torque,
   previous_targets=previous_targets,executed_targets=executed_targets,controlled=controlled,valid=valid,
   physical_failure=physical_failure,fault=fault,config=configuration(COMBO_VERSION),linear_narrow_mix=linear_narrow_mix)
  from .motion_yaw_quality import apply
  return apply(base,target=target,command=command,velocity=velocity,contacts=contacts,previous_targets=previous_targets,
   executed_targets=executed_targets,measurement=spin_measurement,config=c['spin_quality'])
 if c!=configuration(c.get('version')):raise ValueError('Motion reward declaration differs')
 for t,shape in ((command,(n,3)),(target,(n,2)),(velocity,(n,3)),(yaw,(n,)),(angles,(n,2)),(contacts,(n,2)),(leg_q,(n,4)),(wheel_offset,(n,)),(leg_lengths,(n,2)),(applied_torque,(n,6)),(previous_targets,(n,6)),(executed_targets,(n,6))):
  if t.shape!=shape or t.dtype!=height.dtype or t.device!=height.device:raise ValueError('Motion reward explicit shape/dtype/device differs')
 for t in (controlled,valid,physical_failure,fault):
  if t.shape!=(n,) or t.dtype!=torch.bool or t.device!=height.device:raise ValueError('Motion reward masks differ')
 no_translation=target[:,0]==0;no_yaw=target[:,1]==0;e=velocity[:,0]-command[:,0];ey=yaw-command[:,1];lat=velocity[:,1]
 lin_den=torch.where(no_translation,height.new_tensor(c['zero_error_scale_mps']**2),height.new_tensor(c['linear_denominator']))
 yaw_den=torch.where(no_yaw,height.new_tensor(c['zero_yaw_error_scale_rad_s']**2),height.new_tensor(c['yaw_denominator']))
 lin_scale=torch.where(no_translation,height.new_tensor(c['zero_error_scale_mps']),height.new_tensor(c['linear_huber_scale_mps']))
 yaw_scale=torch.where(no_yaw,height.new_tensor(c['zero_yaw_error_scale_rad_s']),height.new_tensor(c['yaw_huber_scale_rad_s']))
 support=(contacts.amin(-1)/2).clamp(0,1)*((height-.20)/.10).clamp(0,1)
 linear_tracking=torch.exp(-e.square()/lin_den)
 if c['version'] in (REFINEMENT_VERSION,YAW_NARROW_BASE,COMBO_BASE,*QUALITY_VERSIONS):
  mix=height.new_zeros(n) if linear_narrow_mix is None else linear_narrow_mix
  if mix.shape!=(n,) or mix.dtype!=height.dtype or mix.device!=height.device:raise ValueError('reward refinement mix shape/dtype/device differs')
  # Nonzero episode target owns the mixture; passing through zero during a ramp does not relabel.
  mix=mix.clamp(0,c['refinement']['maximum_mix'])*(~no_translation).float()
  linear_tracking=(1-mix)*linear_tracking+mix*torch.exp(-e.square()/c['refinement']['narrow_denominator'])
 elif linear_narrow_mix is not None:raise ValueError('v1 reward cannot consume refinement')
 mirror=torch.stack((leg_q[:,0]+leg_q[:,2],leg_q[:,1]-leg_q[:,3]),-1)
 from .standing_survival_reward import bounded_cost
 yaw_tracking=torch.exp(-ey.square()/yaw_den)
 if c['version'] in (COMBO_VERSION,COMBO_BASE):
  gate=torch.exp(-(velocity[:,0].square()+velocity[:,1].square())/c['yaw_translation_gate']['denominator_m2_s2'])
  yaw_tracking=yaw_tracking*torch.where(no_translation&~no_yaw,gate,torch.ones_like(gate))
 raw=torch.stack((support,linear_tracking,yaw_tracking,
  torch.exp(-lat.square()/c['lateral_denominator']),1/(1+((height-.34)/.04).square()),
  1/(1+(angles/.20).square().sum(-1)),huber(e/lin_scale),
  huber(ey/yaw_scale),huber(lat/.30),bounded_cost(mirror/.10).mean(-1),bounded_cost(torch.relu(wheel_offset.abs()-.01)/.03),
  bounded_cost((leg_lengths[:,0]-leg_lengths[:,1])/.02),(applied_torque/height.new_tensor(c['effort_limits_nm'])).square().mean(-1),
  bounded_cost((executed_targets-previous_targets)/height.new_tensor(c['target_delta_scales'])).mean(-1)),-1)
 mask=controlled&valid&~physical_failure&~fault;raw*=mask[:,None];density=raw*height.new_tensor(c['weights']);dt=density*.01
 result={'version':c['version'],'fields':FIELDS,'raw':raw,'density':density,'dt':dt,'total':dt.sum(-1),'events':height.new_zeros(n,3),'eligible':mask}
 if c['version'] in QUALITY_VERSIONS:
  from .motion_quality import apply_reward
  posture=torch.zeros(n,10,dtype=height.dtype,device=height.device);posture[:,:2]=angles;posture[:,2:4]=leg_lengths
  result=apply_reward(result,posture=posture,wheel_offset=wheel_offset,leg_q=leg_q,controlled=controlled,valid=valid,
   quality_failed=height.new_zeros(n,dtype=torch.bool),physical_failed=physical_failure,fault=fault,config=c['quality_termination'],reward_config=c)
 return result

def reward_fields(config):
 if config.get('version')==PARALLEL_VERSION:return reward_fields(configuration(DIFFERENTIAL_VERSION))
 if config.get("version")==DIFFERENTIAL_VERSION:return (*reward_fields(configuration(SYMMETRY_VERSION)),"spin_differential_target")
 if config.get("version")==SYMMETRY_VERSION:return (*reward_fields(configuration(SPIN_VERSION)),"leg_swing_symmetry")
 if config.get('version')==SPIN_VERSION:
  from .motion_yaw_quality import FIELDS as SPIN_FIELDS
  return (*reward_fields(configuration(COMBO_VERSION)),*SPIN_FIELDS)
 return (*FIELDS,'pitch_excess','roll_excess') if config.get('version') in QUALITY_VERSIONS else FIELDS


def effective_reward_fields(spec):
 """Exact declared schema shared by the runtime producer and smoke consumer."""
 names=reward_fields(spec['reward'])
 if spec.get('motion_timing',{}).get('version')=='motion31_standing_sensor_transfer_v3':
  from .motion_sensor_transfer import FIELDS,configuration
  if spec['motion_timing'].get('configuration')!=configuration():raise ValueError('sensor reward extension declaration differs')
  names=(*names,*FIELDS)
 if spec.get('motion_timing',{}).get('version')=='motion31_standing_antiosc_armature_v4':
  from .motion_antioscillation import validate_configuration,FIELDS as ANTI
  from .motion_sensor_transfer import FIELDS as SENSOR
  validate_configuration(spec['motion_timing']['configuration']);names=(*names,*SENSOR,*ANTI)
 if spec.get('student_training',{}).get('course_version')=='motion31_persistent_frontiers_v1':names=(*names,'height_frontier_bonus')
 return names
