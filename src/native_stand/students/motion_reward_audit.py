"""CPU curves from the actual reward implementation; no optimizer or simulation."""
import torch
from .motion_tracking_reward import configuration,reward,FIELDS,QUALITY_VERSION,QUALITY_VERSIONS

def audit(config):
 if config.get('version')=='student_motion31_parallel_quality_v10':
  if config!=configuration(config['version']):raise ValueError('parallel reward config differs')
  result=audit(configuration('student_motion31_differential_quality_v9'));result['configuration']=config
  from .motion_parallel_course import reward_audit
  result['parallel_tracking']={**config['parallel_tracking'],'curves':reward_audit()};result['interpretation']+='; new parallel tracking curves captured separately from actual motion_parallel_design.signals'
  return result
 if config.get("version")=="student_motion31_differential_quality_v9":
  from .motion_differential import configuration as cfg,audit as differential_audit
  if config!=cfg():raise ValueError("differential reward configuration differs")
  result=audit(configuration("student_motion31_symmetry_quality_v8"));result["configuration"]=config
  result["differential"]=differential_audit();result["named_weights"]["spin_differential_target"]=-.5
  return result
 if config.get("version")=="student_motion31_symmetry_quality_v8":
  from .motion_symmetry import configuration as cfg
  from .motion_symmetry_design import audit as prototype
  if config!=cfg():raise ValueError("symmetry reward configuration differs")
  result=audit(configuration("student_motion31_yaw_contact_quality_v7"));result["configuration"]=config
  result["symmetry85"]=prototype();result["named_weights"]["leg_swing_symmetry"]=1.
  from .data import sha
  from . import motion_symmetry,motion_symmetry_design
  result["symmetry_formula_sha256"]={str(m.__file__):sha(m.__file__) for m in (motion_symmetry,motion_symmetry_design)}
  return result
 if config.get('version')=='student_motion31_yaw_contact_quality_v7':
  from .motion_tracking_reward import COMBO_VERSION
  from .motion_yaw_quality import audit as spin_audit,FIELDS as SPIN_FIELDS
  if config!=configuration(config.get('version')):raise ValueError('spin reward declaration differs')
  result=audit(configuration(COMBO_VERSION));result['configuration']=config
  result['spin_quality']=spin_audit();result['named_weights'].update(zip(SPIN_FIELDS,config['spin_quality']['weights']))
  return result
 if config!=configuration(config.get('version')):raise ValueError('reward audit must match actual versioned implementation')
 points=[0.,.005,.01,.02,.03,.05,.1,.2,.3,.5,1.,2.,4.]
 curves=[];mixture_curves=[]
 cases=[('linear_active','linear_tracking','linear_error',1.),('linear_zero','linear_tracking','linear_error',0.),
  ('yaw_active','yaw_tracking','yaw_error',1.),('yaw_zero','yaw_tracking','yaw_error',0.),
  ('pitch','upright',None,0.),('roll','upright',None,0.),('height_error','height',None,0.),
  ('leg_difference','leg_balance',None,0.),('wheel_offset','wheel_alignment',None,0.),('hip_mirror','mirror',None,0.)]
 for signal,positive,negative,target in cases:
  for value in points:
   x=torch.tensor(value,dtype=torch.float64,requires_grad=True);z=torch.zeros(1,dtype=torch.float64);yes=torch.ones(1,dtype=torch.bool)
   data=dict(command=torch.tensor([[target,0.,.34]],dtype=torch.float64),target=torch.tensor([[target,0.]],dtype=torch.float64),velocity=torch.stack((z+target+x if signal.startswith('linear') else z,z,z),-1),yaw=z,height=z+.34+(x if signal=='height_error' else 0),
    angles=torch.stack((z+x if signal=='pitch' else z,z+x if signal=='roll' else z),-1),contacts=torch.ones(1,2,dtype=torch.float64)*20,
    leg_q=torch.stack((z+x if signal=='hip_mirror' else z,z,z,z),-1),wheel_offset=z+x if signal=='wheel_offset' else z,
    leg_lengths=torch.stack((z+.3+x if signal=='leg_difference' else z+.3,z+.3),-1),applied_torque=torch.zeros(1,6,dtype=torch.float64),previous_targets=torch.zeros(1,6,dtype=torch.float64),executed_targets=torch.zeros(1,6,dtype=torch.float64),controlled=yes,valid=yes,physical_failure=~yes,fault=~yes)
   if signal.startswith('yaw'):
    data['command'][0,0]=0.;data['target'][0,0]=0.
    data['command'][0,1]=target;data['target'][0,1]=target;data['velocity']=torch.zeros(1,3,dtype=torch.float64);data['yaw']=z+target+x
   out=reward(**data,config=config);indices=[FIELDS.index(positive)]+([FIELDS.index(negative)] if negative else [])
   if config['version'] in QUALITY_VERSIONS and signal in ('pitch','roll'):indices.append(out['fields'].index(signal+'_excess'))
   contribution=out['density'][0,indices].sum();derivative=torch.autograd.grad(contribution,x)[0]
   curves.append({'signal':signal,'error':value,'unit':'m/s' if signal.startswith('linear') else 'rad/s' if signal.startswith('yaw') else 'm' if signal in ('height_error','leg_difference','wheel_offset') else 'rad',
    'raw':{out['fields'][i]:float(out['raw'][0,i].detach()) for i in indices},'weighted_density':float(contribution.detach()),'dt_contribution':float(contribution.detach()*config['dt']),'state_error_derivative':float(derivative),
    'near_flat_nonzero_error':value>0 and abs(float(derivative))<1e-4})
   if signal=='linear_active' and 'refinement' in config:
    for mix in (0.,.25,.5,.75):
     mixed=reward(**data,config=config,linear_narrow_mix=x.new_tensor([mix]))
     total=mixed['density'][0,indices].sum();gradient=torch.autograd.grad(total,x)[0]
     mixture_curves.append({'error_mps':value,'narrow_mix':mix,'weighted_density':float(total.detach()),'state_error_derivative':float(gradient)})
 warnings=[]
 for name,scale,field in [('leg_difference',config['leg_balance_scale_m'],'leg_balance'),('hip_mirror',config['mirror_scale_rad'],'mirror'),('wheel_offset',config['wheel_scale_m'],'wheel_alignment')]:
  maximum=abs(config['weights'][FIELDS.index(field)])
  if config['version'] not in QUALITY_VERSIONS:warnings.append({'signal':name,'kind':'BOUNDED_COST_TAIL','maximum_density_penalty':maximum,'scale':scale,
   'raw_at_5_scales':25/26,'raw_at_10_scales':100/101,'note':'large errors approach the same cost; inspect physical error and derivative, not only bounded raw reward'})
 width_comparison=[]
 for denominator in (1.,.25,.09,.04):
  for error in points:
   x=torch.tensor(error,dtype=torch.float64,requires_grad=True)
   score=torch.exp(-x.square()/denominator);derivative=torch.autograd.grad(score,x)[0]
   width_comparison.append({'denominator_mps_squared':denominator,'error_mps':error,'score':float(score.detach()),'state_error_derivative':float(derivative)})
 weights=dict(zip(FIELDS,config['weights']))
 if config['version'] in QUALITY_VERSIONS:weights.update(zip(('pitch_excess','roll_excess'),config['posture_tail']['weights']))
 gate_curves=[]
 if config.get('yaw_translation_gate'):
  for speed in points:
   x=torch.tensor(speed,dtype=torch.float64,requires_grad=True)
   gate=torch.exp(-x.square()/config['yaw_translation_gate']['denominator_m2_s2'])
   gate_curves.append({'horizontal_speed_mps':speed,'gate':float(gate.detach()),'derivative':float(torch.autograd.grad(gate,x)[0])})
 return {'schema_version':'motion_reward_curve_audit_v1','configuration':config,'named_weights':weights,'curves':curves,'warnings':warnings,'yaw_translation_gate_curves':gate_curves,
  'proposed_width_comparison':{'status':'DESIGN_ONLY_NOT_EXECUTED','formula':'exp(-error_squared/denominator); denominator is not sigma', 'curves':width_comparison},
  'declared_refinement_curves':mixture_curves,
  'interpretation':'state-error derivative from actual CPU reward; not Actor gradient, historical PPO replay or convergence guarantee',
  'simulation':0,'optimizer_updates':0,'training_behavior_changed':False}
