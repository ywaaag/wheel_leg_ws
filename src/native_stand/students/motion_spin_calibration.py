"""Source-evaluation measurement readback, not a policy capability certificate."""
import json
from pathlib import Path
import numpy as np
import torch
from .motion_yaw_quality import measurement

def verify_records(records,output,*,allow_incomplete_new=False):
 rows=[]
 for record in records:
  raw_path=Path(record['path']);raw_path=raw_path if raw_path.is_absolute() else Path(output)/raw_path
  raw=json.loads(raw_path.read_text());trace=raw_path.parent/raw['trace']['path']
  with np.load(trace,allow_pickle=False) as data:
   keys=('center_world','com_world','linear_com_world','angular_world','axis_world','forward_world','tangent_velocity_w','rolling_speed_forward_mps','geometry_valid')
   if any('spin_'+k not in data for k in keys):raise ValueError('spin source measurement trace fields missing')
   valid=data['first_episode_valid']
   # Bounded CPU validation; original trace retained, no GPU/optimizer changes.
   ix=np.arange(0,len(valid),25);n=valid.shape[1];m={k:torch.from_numpy(data['spin_'+k][ix].copy()) for k in keys}
   recomputed=measurement(*(m[k].reshape(-1,2,3) for k in keys[:5]),m['forward_world'].reshape(-1,3))
   err=float((recomputed['tangent_velocity_w']-m['tangent_velocity_w'].reshape(-1,2,2)).abs().max())
   finite=all(torch.isfinite(m[k]).all() for k in keys if k!='geometry_valid')
   if not finite or err>1e-4 or not m['geometry_valid'].all():raise ValueError('spin readback nonfinite/geometry/velocity reconstruction differs')
   # Hold interval uses the existing fixed30s evaluation policy clock10..18s.
   controlled=(data['controller'][...,0]==3)&valid
   controlled_age=np.cumsum(controlled,axis=0)*.01
   hold=controlled&(controlled_age>=10)&(controlled_age<18)
   if not hold.any():
    if not allow_incomplete_new or raw['case']['command'][:2]==[.5,0.]:raise ValueError('spin signed readback has no controlled holding samples')
    rows.append({'command':raw['case']['command'],'reconstruction_max_abs_mps':err,'geometry_valid':True,'samples':0,'capability_gate':False,'holding_behavior':'NOT_OBSERVED; new task failure does not reject finite pre-terminal geometry'})
    continue
   rolling=data['spin_rolling_speed_forward_mps'][hold]
   forward=data['spin_forward_world'];norm=np.linalg.norm(forward[...,:2],axis=-1,keepdims=True)
   f=forward[...,:2]/np.maximum(norm,1e-8)
   # Signed center speed from COM fields, same exact kinematic convention.
   c=data['spin_center_world'];com=data['spin_com_world'];lin=data['spin_linear_com_world'];omega=data['spin_angular_world']
   center=lin+np.cross(omega,c-com)
   center_forward=(center[...,:2]*f[:,:,None,:]).sum(-1)[hold]
   item={'command':raw['case']['command'],'reconstruction_max_abs_mps':err,'geometry_valid':True,
    'rolling_signed_mean_mps':rolling.mean(0).tolist(),'center_signed_mean_mps':center_forward.mean(0).tolist(),
    'samples':int(hold.sum()),'scope':'geometric support point, not actual PhysX patch/slip certification'}
   if raw['case']['command'][:2]==[.5,0.]:
    if not (rolling.mean(0)>.02).all() or not (center_forward.mean(0)>.02).all():raise ValueError('forward source rolling/center signed readback inconsistent')
   rows.append(item)
 if not any(r['command'][:2]==[.5,0.] for r in rows):raise ValueError('forward signed source calibration case missing')
 result={'version':'motion_spin_source_measurement_readback_v1','status':'PASSED','capability_gate':False,'records':rows,'actual_contact_patch':'NOT_MEASURED','sampling_hz':100,'online_PPO':0}
 with (Path(output)/'spin_measurement_readback.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 return result
