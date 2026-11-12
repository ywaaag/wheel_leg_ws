"""Explicit v8 symmetry recipe,85D named Critic and non-actuating geometry."""
import copy
import torch
from .motion_symmetry_design import density,EXTRA_FIELDS
VERSION='student_motion31_symmetry_quality_v8'
CRITIC='motion_symmetry_privileged85_v1'

def network():
 from .motion_spin_privileged import network as base
 n=base();n.update(version='motion31_symmetry_large_actor_critic_v3',critic_input=85);return n

def contract():
 from .motion_spin_privileged import contract as base
 c=base();c.update(critic_dim=85,critic_privileged_version=CRITIC)
 c['critic_privileged']+= [f'{n}:{w}/{s}' for n,w,s in EXTRA_FIELDS];return c

def configuration():
 from .motion_tracking_reward import configuration as base,SPIN_VERSION
 c=base(SPIN_VERSION);c['version']=VERSION
 c['symmetry']={'version':'motion_symmetry_cost_v1','swing_deadband_rad':.02,'swing_tail_scale_rad':.03,'swing_tail_weight':-.5,
 'swing_kernel_weight':.25,'wide_denominator_rad2':.0064,'narrow_denominator_rad2':.0009,'max_mix':.75,
 'spin_common_weight':-.5,'spin_common_scales_mps':[.15,.08],'spin_translation_weight':-.5,'spin_translation_scales_mps':[.2,.1],
 'scope':'all controlled swing; spin replacements only episode-target yaw; no motor override'}
 return c

def expand_source(source):
 from .motion_spin_privileged import network as old
 if source.get('network_architecture')==network() and source.get('critic_privileged_version')==CRITIC:return source
 if source.get('network_architecture')!=old() or source.get('critic_privileged_version')!='motion_spin_privileged61_v1':raise ValueError('symmetry source explicit61 required')
 out=copy.deepcopy(source);weight=out['critic']['mlp.0.weight']
 if weight.shape!=(512,61):raise ValueError('source61 critic shape differs')
 def pad(x):return torch.cat((x,torch.zeros(x.shape[0],24,dtype=x.dtype,device=x.device)),1)
 out['critic']['mlp.0.weight']=pad(weight)
 ids=[p for g in out['optimizer']['param_groups'] for p in g['params']]
 names=list(out['actor']);cn=list(out['critic'])
 if len(ids)!=len(names)+len(cn):raise ValueError('Adam parameter count differs')
 state=out['optimizer']['state'].get(ids[len(names)+cn.index('mlp.0.weight')],{})
 for k in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
  if k in state:
   if state[k].shape!=weight.shape:raise ValueError('Adam source shape differs')
   state[k]=pad(state[k])
 out.update(network_architecture=network(),critic_privileged_version=CRITIC);return out

def runtime_fields(env,m):
 from isaaclab.utils.math import quat_apply_inverse,quat_apply
 d=env.robot.data;q=d.root_quat_w.torch;root=d.root_pos_w.torch
 anchors=root.new_tensor(env.recovery_config['standing_task']['hip_anchors_root_m'])
 hip=root[:,None,:]+quat_apply(q[:,None,:].expand(-1,2,-1),anchors.expand(len(root),-1,-1))
 v=quat_apply_inverse(q[:,None,:].expand(-1,2,-1),m['center_world']-hip)
 swing=torch.atan2(-v[...,2],-v[...,1]);diff=torch.atan2(torch.sin(swing[:,0]-swing[:,1]),torch.cos(swing[:,0]-swing[:,1]))
 lengths=v.norm(dim=-1)
 rootcom=d.root_com_pos_w.torch
 rootlin=d.root_com_lin_vel_w.torch;omega=d.root_com_ang_vel_w.torch
 hipvel=rootlin[:,None,:]+torch.cross(omega[:,None,:].expand_as(hip),hip-rootcom[:,None,:],dim=-1)
 relative=quat_apply_inverse(q[:,None,:].expand(-1,2,-1),m['center_velocity_w']-hipvel)
 localw=quat_apply_inverse(q,omega)
 dv=relative-torch.cross(localw[:,None,:].expand_as(v),v,dim=-1)
 rate=(v[...,1]*dv[...,2]-v[...,2]*dv[...,1])/(v[...,1].square()+v[...,2].square()).clamp_min(1e-8)
 f=m['forward_world'].clone();f[:,2]=0;f=f/f.norm(dim=-1,keepdim=True).clamp_min(1e-8)
 left=torch.stack((-f[:,1],f[:,0],f[:,2]*0),-1)
 horizontal=torch.stack(((rootlin*f).sum(-1),(rootlin*left).sum(-1)),-1)
 center=(m['center_velocity_w']*f[:,None,:]).sum(-1);roll=m['rolling_speed_forward_mps']
 age=env.rise_hold.episode.steps.float()*.01
 times=env.motion_config['train_schedule_s'] if env.curriculum_training else [2.,8.,8.,8.]
 boundaries=age.new_tensor([times[0],sum(times[:2]),sum(times[:3]),sum(times)+ (env.motion_config['parking_s'] if env.curriculum_training else 4.)])
 phase=(age[:,None]>=boundaries[:3]).sum(-1);onehot=torch.nn.functional.one_hot(phase,4).to(root.dtype)
 forces=env.contact_force
 fields={'leg_swing_rad':swing,'leg_swing_signed_difference_rad':diff[:,None],'leg_lengths_m':lengths,
 'leg_length_signed_difference_m':(lengths[:,0]-lengths[:,1])[:,None],'leg_swing_rates_rad_s':rate,
 'rolling_common_differential_mps':torch.stack((roll.mean(-1),(roll[:,0]-roll[:,1])*.5),-1),
 'wheel_center_heading_forward_mps':center,'root_heading_horizontal_mps':horizontal,'contact_flags':(forces>2).to(root.dtype),
 'signed_support_balance':((forces[:,0]-forces[:,1])/forces.sum(-1).clamp_min(1.))[:,None],
 'schedule_phase_onehot':onehot,'phase_remaining_s':(boundaries[phase]-age).clamp_min(0)[:,None],
 'episode_target_vx_yaw':env.target_xy}
 return fields

def apply(result,env):
 m=env.motion_spin_measurement_post;fields=runtime_fields(env,m)
 mix=getattr(env,'symmetry_mix',0.)
 # Runtime tensor mixture per episode cohort; CPU prototype scalar formulas expanded here.
 e=fields['leg_swing_signed_difference_rad'][:,0].abs()
 from .motion_tracking_reward import huber
 wide=torch.exp(-e.square()/.0064);narrow=torch.exp(-e.square()/.0009)
 swing=-.5*huber(torch.relu(e-.02)/.03)+.25*((1-mix)*wide+mix*narrow-1)
 common=fields['rolling_common_differential_mps'][:,0];vel=fields['root_heading_horizontal_mps']
 cd=-.5*((1-mix)*huber(common/.15)+mix*huber(common/.08))
 td=-.5*((1-mix)*huber(vel/.2).sum(-1)+mix*huber(vel/.1).sum(-1))
 names=result['fields'];den=result['density'].clone();raw=result['raw'].clone()
 gate=result['spin_diagnostics']['activation']
 for name,val in [('spin_common_rolling',cd),('spin_translation',td)]:
  i=names.index(name);den[:,i]=val*gate;raw[:,i]=val*gate/-.5
 eligible=result['eligible'].to(den.dtype);swing*=eligible
 den=torch.cat((den,swing[:,None]),-1);raw=torch.cat((raw,(-swing)[:,None]),-1)
 out=dict(result);out.update(fields=(*names,'leg_swing_symmetry'),density=den,raw=raw,dt=den*.01,total=result['total']+(den.sum(-1)-result['density'].sum(-1))*.01,version=VERSION)
 out['symmetry_diagnostics']=fields
 return out

def assess(result,geometry,root_velocity,rolling,controlled):
 from .motion_symmetry_design import density
 mask=controlled.bool();g=geometry[mask][:3000];v=root_velocity[mask][:3000];r=rolling[mask][:3000]
 valid=len(g)==3000
 extra={'swing_hold':False,'swing_full':False};metrics={}
 if valid:
  anchors=g.new_tensor([[.17749999463558197,2.4000000848900527e-05,-1.4000000192027073e-05],[-.17749999463558197,2.4000000848900527e-05,-1.4000000192027073e-05]])
  # Exact anchors are checked against this USD family in runtime readback.
  legs=g[:,:6].reshape(-1,2,3)-anchors
  a=torch.atan2(-legs[...,2],-legs[...,1]);delta=torch.atan2(torch.sin(a[:,0]-a[:,1]),torch.cos(a[:,0]-a[:,1])).abs()
  h=delta[1000:1800];extra.update(swing_hold=bool((h<.05).all()),swing_full=bool((delta<.05).all()))
  metrics={'swing_hold_peak_rad':float(h.max()),'swing_full_peak_rad':float(delta.max()),'swing_hold_mae_rad':float(h.mean()),
   'horizontal_hold_mean_mps':float(v[1000:1800,:2].norm(dim=-1).mean()),'common_abs_hold_mean_mps':float(r[1000:1800].mean(-1).abs().mean())}
 result['symmetry_metrics']=metrics
 result['checks'].update(extra)
 result['accepted']=all(result['checks'].values());result['failed_checks']=[k for k,x in result['checks'].items() if not x]
 return result

def update_mix(state,records):
 out=copy.deepcopy(state)
 for i,record in enumerate(records):
  es=record.get('per_env',[]);target=record['case']['command'];old=out[i]
  valid=len(es)==8 and all(e.get('symmetry_metrics') for e in es)
  if not valid:continue
  good=all(e['accepted'] and e['symmetry_metrics']['swing_hold_peak_rad']<.045 and
   (not target[1] or (e['symmetry_metrics']['horizontal_hold_mean_mps']<=.1 and e['symmetry_metrics']['common_abs_hold_mean_mps']<=.05)) for e in es)
  old['confirmations']=old.get('confirmations',0)+1 if good else 0
  old['regressions']=0 if good else old.get('regressions',0)+1
  if old['confirmations']>=2:old.update(mix=min(.75,old['mix']+.25),confirmations=0)
  elif old['regressions']>=2:old.update(mix=max(0.,old['mix']-.25),regressions=0)
 return out

def verify_source_features(records,output,*,allow_incomplete_new=False):
 import numpy as np,json
 from pathlib import Path
 rows=[]
 for record in records:
  p=Path(record['path']);p=p if p.is_absolute() else Path(output)/p
  d=json.loads(p.read_text());tp=p.parent/d['trace']['path']
  with np.load(tp) as a:
   critic=a['critic_input'];actor=a['actor_input']
   if critic.shape[-1]!=85 or not np.isfinite(critic).all() or not np.array_equal(critic[...,:31],actor):raise ValueError('85D runtime input/actor prefix differs')
   # Appended onehot occupies base61+17..20, normalized target final2.
   phase=critic[...,78:82]
   if not ((phase==0)|(phase==1)).all() or not (phase.sum(-1)==1).all():raise ValueError('85D phase readback differs')
   valid=a['first_episode_valid'];controlled=a['controller'][...,0]==3
   mask=valid&controlled
   if not mask.any():
    if not allow_incomplete_new:raise ValueError('85D episode target readback differs')
    rows.append({'command':d['case']['command'],'critic_dim':85,'samples':0,'target_readback':'NOT_OBSERVED; not behavior acceptance'});continue
   if np.max(abs(critic[...,83:85][mask]-np.asarray(d['case']['command'][:2])/4))>1e-6:raise ValueError('85D episode target readback differs')
   rows.append({'command':d['case']['command'],'critic_dim':85,'samples':int(mask.sum()),'finite':True})
 r={'status':'PASSED','scope':'85D schema/prefix/phase/target finite readback; not behavior certificate','records':rows}
 with (Path(output)/'symmetry_readback.json').open('x') as f:json.dump(r,f,indent=2)
 return r
