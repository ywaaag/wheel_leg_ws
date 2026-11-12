"""Versioned geometric differential rolling target; no action overwrite or IO in reward."""
import copy
import torch
VERSION='student_motion31_differential_quality_v9'

def configuration():
 from .motion_symmetry import configuration as base
 c=base();c.update(version=VERSION,differential={'version':'heading_spin_differential_v1','weight':-.5,'scales_mps':[.20,.08],'max_narrow_mix':.75,
 'target':'((omega_z * world_up cross (center_left-center_right)) dot heading_forward)/2',
 'measurement':'actual signed geometric rolling velocity, (left-right)/2; no raw joint polarity assumption',
 'cohort':'episode vx==0 and yaw!=0; existing smooth activation during ramp/hold','dt':.01,
 'assessment':{'difference_MAE_max_mps':.05,'opposite_fraction_min':.95,'relative_imbalance_mean_max':.20},
 'refinement':'two complete accepted eight-env evaluations; max.75 narrow; two regressions remove.25; UNKNOWN freezes',
 'no_potential_shaping':True,'no_action_override':True})
 return c

def target_from_geometry(centers,forward,yaw_command):
 f=forward.clone();f[...,2]=0;f=f/f.norm(dim=-1,keepdim=True).clamp_min(1e-8)
 delta=centers[:,0]-centers[:,1];up=torch.zeros_like(delta);up[:,2]=1
 return .5*yaw_command*(torch.cross(up,delta,dim=-1)*f).sum(-1)

def terms(rolling,target,mix):
 from .motion_tracking_reward import huber
 common=rolling.mean(-1);difference=(rolling[:,0]-rolling[:,1])*.5;error=difference-target
 mix=torch.as_tensor(mix,dtype=rolling.dtype,device=rolling.device).clamp(0,.75)
 raw=(1-mix)*huber(error/.20)+mix*huber(error/.08)
 return {'raw':raw,'density':-.5*raw,'common':common,'difference':difference,'target':target,'error':error,
 'opposite':(rolling[:,0]*rolling[:,1]<0).to(rolling.dtype),
 'imbalance':(rolling[:,0]+rolling[:,1]).abs()/(rolling.abs().sum(-1).clamp_min(.02))}

def apply(result,env):
 m=env.motion_spin_measurement_post
 target=target_from_geometry(m['center_world'],m['forward_world'],env.motion_action_command[:,1])
 t=terms(m['rolling_speed_forward_mps'],target,getattr(env,'symmetry_mix',0.));gate=result['spin_diagnostics']['activation']
 raw=t['raw']*gate;den=t['density']*gate
 out=dict(result);out.update(fields=(*result['fields'],'spin_differential_target'),raw=torch.cat((result['raw'],raw[:,None]),-1),
  density=torch.cat((result['density'],den[:,None]),-1),dt=torch.cat((result['dt'],den[:,None]*.01),-1),total=result['total']+den*.01)
 out['differential_diagnostics']={k:v[:,None] for k,v in t.items() if k not in ('raw','density')}
 out['differential_diagnostics']['rolling']=m['rolling_speed_forward_mps']
 return out

def assess(result,arrays,mask,eid,controlled,command):
 r=copy.deepcopy(result);metrics={};checks={}
 if command[1]:
  valid=controlled.cpu().numpy().astype(bool)
  def held(key):return torch.as_tensor(arrays['differential_'+key][mask,eid])[valid][:3000][1000:1800]
  error=held('error');opposite=held('opposite');imbalance=held('imbalance')
  complete=len(error)==800 and int(controlled.sum())>=3000
  if complete:
   metrics={'error_hold_MAE_mps':float(error.abs().mean()),'opposite_hold_fraction':float(opposite.mean()),'relative_imbalance_hold_mean':float(imbalance.mean()),'rolling_hold_mean_mps':held('rolling').mean(0).tolist(),'difference_target_hold_mean_mps':float(held('target').mean())}
  checks={'difference_complete':complete,'difference_tracking':complete and metrics['error_hold_MAE_mps']<=.05,
   'opposite_rolling':complete and metrics['opposite_hold_fraction']>=.95,'rolling_equality':complete and metrics['relative_imbalance_hold_mean']<=.20}
 r['differential_metrics']=metrics;r['checks'].update(checks);r.update(accepted=all(r['checks'].values()),failed_checks=[k for k,v in r['checks'].items() if not v])
 return r

def update_mix(state,records):
 # Symmetry owner requires complete accepted evidence; differential gates now included.
 from .motion_symmetry import update_mix as base
 return base(state,records)

def audit():
 from .data import sha
 from pathlib import Path
 points=[]
 for mix in (0.,.25,.5,.75):
  for error in (0.,.01,.02,.05,.1,.2,.5,1.,2.):
   e=torch.tensor(error,dtype=torch.float64,requires_grad=True);r=torch.stack((e,-e))[None,:];v=terms(r,e.new_zeros(1),mix)['density'].sum();g=torch.autograd.grad(v,e)[0]
   points.append({'error_mps':error,'narrow_mix':mix,'weighted_density':float(v.detach()),'derivative_per_mps':float(g),'dt_contribution':float(v.detach())*.01})
 return {'configuration':configuration()['differential'],'curves':points,'formula_sha256':sha(Path(__file__)),'saturation':'Huber unbounded linear tail; derivative of density is not PPO policy gradient'}

def verify_records(records,output,spec):
 import json,numpy as np
 from pathlib import Path
 from .data import sha
 rows=[]
 for rec in records:
  p=Path(rec['path']);p=p if p.is_absolute() else Path(output)/p;d=json.loads(p.read_text());tp=p.parent/d['trace']['path']
  if sha(tp)!=d['trace']['sha256']:raise ValueError('differential readback trace changed')
  with np.load(tp) as a:
   required=['differential_target','differential_difference','differential_error','differential_rolling','spin_center_world','spin_forward_world']
   if any(k not in a for k in required):raise ValueError('differential measurement missing')
   target=target_from_geometry(torch.as_tensor(a['spin_center_world'].reshape(-1,2,3)),torch.as_tensor(a['spin_forward_world'].reshape(-1,3)),torch.as_tensor(a['effective_command_pre'][...,1].reshape(-1))).numpy().reshape(a['differential_target'].shape)
   if not np.allclose(target,a['differential_target'],atol=2e-5):raise ValueError('difference target geometry/command readback differs')
   rolling=a['differential_rolling'];diff=(rolling[...,0]-rolling[...,1])*.5
   if not np.allclose(diff,a['differential_difference'][...,0],atol=1e-5):raise ValueError('difference sign readback differs')
   for k in required:
    if not np.isfinite(a[k]).all():raise ValueError('nonfinite differential readback')
   rows.append({'command':d['case']['command'],'target_range_mps':[float(target.min()),float(target.max())],'signed_difference_readback':True,'capability_passed':False})
 result={'status':'PASSED','scope':'measured sign/geometry/formula wiring only; not behavior gate','records':rows}
 with (Path(output)/'differential_readback.json').open('x') as f:json.dump(result,f,indent=2)
 return result
