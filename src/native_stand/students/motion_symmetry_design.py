"""CPU-only next-recipe prototype. Not registered with runtime or authorized to train."""
import torch
VERSION='motion31_symmetry85_design_v1'
EXTRA_FIELDS=(('leg_swing_rad',2,.1),('leg_swing_signed_difference_rad',1,.05),
 ('leg_lengths_m',2,.34),('leg_length_signed_difference_m',1,.02),
 ('leg_swing_rates_rad_s',2,2.),('rolling_common_differential_mps',2,1.),
 ('wheel_center_heading_forward_mps',2,4.),('root_heading_horizontal_mps',2,4.),
 ('contact_flags',2,1.),('signed_support_balance',1,1.),
 ('schedule_phase_onehot',4,1.),('phase_remaining_s',1,30.),('episode_target_vx_yaw',2,4.))

def declaration():
 return {'version':VERSION,'status':'CPU_DESIGN_ONLY_NOT_RUNTIME','actor_dim':31,'critic_dim':85,
 'critic_base':'motion_spin_privileged61_v1 unchanged first61','extra_fields':[
 {'name':n,'width':w,'divide_by':s} for n,w,s in EXTRA_FIELDS],
 'actor_hidden_dims':[512,256,128],'critic_hidden_dims':[512,512,256],
 'source':'spin61 model_update_750; source Actor/std preserved',
 'migration':'zero24 new Critic columns and matching Adam moments; preserve remaining tensors/RNG; reset real environment',
 'geometry':'hip-to-wheel body sagittal theta=atan2(-z,-y), same sign both legs; wrapped difference',
 'rates':'causal wrapped finite difference / policy dt; reset to0 with validity mask; not future data',
 'phase':'prep/ramp/hold/return_or_parking; exact effective schedule, no future stochastic preview',
 'unused_or_invalid':'no silent zeros for missing runtime data; reject before enabling recipe',
 'normalization':'explicit scales only, no running observation normalization or advantage changes',
 'reward':{'dt':.01,'swing_deadband_rad':.02,'swing_tail_scale_rad':.03,'swing_tail_weight':-.5,
 'swing_kernel_weight':.25,'swing_wide_denominator_rad2':.0064,'swing_narrow_denominator_rad2':.0009,
 'spin_common_weight_old':-.1,'spin_common_weight_new':-.5,
 'spin_common_scale_old_mps':.3,'spin_common_wide_scale_mps':.15,'spin_common_narrow_scale_mps':.08,
 'spin_translation_weight':-.5,'spin_translation_scale_old_mps':.3,'spin_translation_wide_scale_mps':.2,'spin_translation_narrow_scale_mps':.1,
 'maximum_narrow_mix':.75,'total_clip':None,
 'replacement':'replace existing v7 common/translation density; do not append duplicate copies',
 'other_rewards':'unchanged v7; no stronger slip/action smoothness costs in first experiment',
 'mixture':'per episode-target cohort; no update-number switch; minimum25% wide signal'},
 'assessment':{'swing_difference_strict_max_rad':.05,'pitch_hold_max_rad':.1,'roll_hold_max_rad':.05,
 'spin_horizontal_hold_mean_max_mps':.1,'spin_common_abs_hold_mean_max_mps':.05,
 'swing_scope':'hold mandatory; full0..30s separately required for final acceptance, startup violation never hidden',
 'wheel_equality':'physical signed rolling speed common approaches0; no hard overwrite of motor targets',
 'training_termination':'retain old severe quality/physical faults initially; .05 violation remains learning sample, blocks acceptance'},
 'refinement':{'confirmations':2,'mix_step':.25,'max_mix':.75,
 'requires':'all old capability checks and full30s; hold swing peak<.045, spin translation<=.1, common_abs<=.05 when yaw; verified per env',
 'regression':'two consecutive valid fixed evaluations revoke one mix step; UNKNOWN freezes, never advances; persistence required'},
 'course':[{'stage':'S0','targets':'existing zero,+.5/+1/+1.5/+2,yaw±.5','slots512':[128,48,48,48,48,96,96],'goal':'symmetry+center control; no new speed'},
 {'stage':'S1','targets':'introduce backward -.5 then -1 then -1.5 then -2; retain existing forward/yaw','slots512':{'zero':64,'old':320,'new':128}},
 {'stage':'S2','targets':'bidirectional2.5→3→3.5→4; retain old/yaw','slots512':{'zero':64,'old':320,'new':128}},
 {'stage':'S3','targets':'yaw±1→±1.5→±2→±3→±4; retain straight endpoints','slots512':{'zero':64,'old':320,'new':128}}],
 'slot_scaling':'4096 env uses8×512 counts; old cohort balanced quotient/remainder allocation frozen per stage',
 'training':'retain4096env,LR/std/PPO/physics/source-policy coefficient.01; retarget teacher to frozen750 only on its certified seven old cohorts; new endpoints no BC',
 'authorization':'NO_NEW_RUN_OR_BUDGET; implementation, CPU checks and physical readback precede any launch',
 'non_claims':['more privileged features do not guarantee convergence','state derivative of reward is not Actor policy gradient','hard gate does not ensure all training samples obey constraint','single-model skills must pass together; no checkpoint stitching']}

def critic_input(base61,fields):
 if base61.ndim!=2 or base61.shape[1]!=61:raise ValueError('explicit base61 required')
 if set(fields)!={n for n,_,_ in EXTRA_FIELDS}:raise ValueError('privileged fields missing/unknown')
 n=len(base61);pieces=[base61]
 for name,width,scale in EXTRA_FIELDS:
  t=fields[name]
  if t.shape!=(n,width) or t.device!=base61.device or t.dtype!=base61.dtype or not torch.isfinite(t).all():raise ValueError('invalid explicit privileged field:'+name)
  pieces.append(t/scale)
 phase=fields['schedule_phase_onehot']
 if not ((phase==0)|(phase==1)).all() or not (phase.sum(-1)==1).all():raise ValueError('phase must be onehot')
 return torch.cat(pieces,-1)

def huber(x):
 a=x.abs();return torch.where(a<=1,.5*x.square(),a-.5)

def density(signed_swing_difference,common_rolling,horizontal_velocity,*,narrow_mix=0.):
 """New swing and replacement spin densities, separate eligibility belongs to runtime."""
 if not 0<=narrow_mix<=.75:raise ValueError('narrow mixture outside frozen design')
 e=signed_swing_difference.abs();m=narrow_mix
 swing=-.5*huber(torch.relu(e-.02)/.03)+.25*((1-m)*torch.exp(-e.square()/.0064)+m*torch.exp(-e.square()/.0009)-1)
 common=-.5*((1-m)*huber(common_rolling/.15)+m*huber(common_rolling/.08))
 translation=-.5*((1-m)*huber(horizontal_velocity/.2).sum(-1)+m*huber(horizontal_velocity/.1).sum(-1))
 return {'swing':swing,'spin_common_replacement':common,'spin_translation_replacement':translation}

def audit():
 rows=[]
 for mix in (0.,.25,.5,.75):
  for value in (0.,.01,.02,.03,.04,.045,.05,.06,.065,.08,.1,.2,.5,1.):
   x=torch.tensor(value,dtype=torch.float64,requires_grad=True)
   out=density(x,x,torch.stack((x,x*0)),narrow_mix=mix)
   for name,y in out.items():
    gradient=torch.autograd.grad(y,x,retain_graph=True)[0]
    rows.append({'field':name,'narrow_mix':mix,'error':value,'density':float(y.detach()),'dt_contribution':float(y.detach()*.01),'state_error_derivative':float(gradient)})
 return {'declaration':declaration(),'curves':rows,'runtime_registered':False,'simulation':0,'PPO':0}
