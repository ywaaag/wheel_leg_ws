"""Source-stage physical readback gate; original files untouched, no optimizer."""
import json
from pathlib import Path
import numpy as np
from .data import sha

def verify_records(folder,records):
 folder=Path(folder);rows=[]
 for record in records:
  path=folder/record['path'];result=json.loads(path.read_text())
  if sha(path)!=record['sha256']:raise ValueError('contact readback evaluation SHA differs')
  tp=folder/result['trace']['path']
  if sha(tp)!=result['trace']['sha256']:raise ValueError('contact readback trace SHA differs')
  with np.load(tp) as a:
   needed=('wheel_contact_parent_angular_velocity_w','wheel_contact_joint_spin_rad_s','wheel_contact_relative_spin_rad_s','wheel_contact_axle_w','wheel_contact_com_velocity_w','wheel_contact_angular_velocity_w','wheel_contact_support_point_w','wheel_contact_com_w','wheel_contact_velocity_w','wheel_contact_geometry_valid','contact_entered','contact_reward_weighted_density','contact_reward_dt','contact_reward_events','contact_reward_total','actor_valid','first_episode_valid')
   if any(k not in a for k in needed):raise ValueError('contact source readback field missing')
   controlled=a['actor_valid'][...,0].astype(bool)&a['first_episode_valid'].astype(bool)
   if not controlled.any():raise ValueError('contact readback has no controlled samples')
   v=a['wheel_contact_com_velocity_w'];w=a['wheel_contact_angular_velocity_w'];p=a['wheel_contact_support_point_w'];c=a['wheel_contact_com_w'];observed=a['wheel_contact_velocity_w']
   expected=v+np.cross(w,p-c);error=float(np.abs(expected-observed)[controlled].max())
   if error>1e-5 or not np.isfinite(expected[controlled]).all():raise ValueError('contact point world velocity reconstruction differs')
   relative=((w-a['wheel_contact_parent_angular_velocity_w'])*a['wheel_contact_axle_w']).sum(-1)
   spin_error=np.abs(relative-a['wheel_contact_joint_spin_rad_s'])
   tolerance=.05+.005*np.abs(a['wheel_contact_joint_spin_rad_s'])
   if not np.isfinite(spin_error[controlled]).all() or (spin_error[controlled]>tolerance[controlled]).any():raise ValueError('wheel angular/joint spin polarity readback differs')
   spin_peak=float(spin_error[controlled].max())
   pre=controlled&~a['contact_entered'][...,0].astype(bool)
   density=a['contact_reward_weighted_density']
   if not pre.any() or (density[pre,-3:]!=0).any():raise ValueError('new standing costs leaked into pre-entry')
   if (density[...,-1]!=0).any():raise ValueError('slip reward enabled without separate approval/validation')
   total=a['contact_reward_dt'].sum(-1)+a['contact_reward_events'].sum(-1)
   if not np.allclose(total[controlled],a['contact_reward_total'][controlled],atol=1e-6,rtol=1e-5):raise ValueError('reward dt/events readback differs')
   # Finite difference of saved contact point is NOT the velocity of the same
   # material point: wheel rotation changes the geometric support point.
   valid=a['wheel_contact_geometry_valid'].astype(bool)
   if not valid[controlled].any():raise ValueError('no valid wheel geometry measured')
   rows.append({'case_id':record['case']['case_id'],'scored_envs':len(result['per_env']),'controlled_samples':int(controlled.sum()),'pre_entry_samples':int(pre.sum()),'velocity_reconstruction_max_abs_mps':error,'wheel_spin_polarity_error_peak_rad_s':spin_peak,'spin_error_tolerance':'.05rad/s + .5percent abs joint dq','valid_wheel_geometry_samples':int(valid[controlled].sum()),'evaluation_sha256':sha(path),'trace_sha256':sha(tp),'passed_envs':sum(e['accepted'] for e in result['per_env'])})
 if len(rows)!=4 or sum(r['scored_envs'] for r in rows)!=32:raise ValueError('full source four-condition32 readback required')
 return {'version':'recovery_contact_runtime_readback_v1','status':'PASSED','rows':rows,'new_costs_pre_entry_zero':True,'slip_reward_enabled':False,'velocity_semantics':'total body omega world; geometric cylinder support, not solver contact patch','scope':'runtime fields/finite/unit/reconstruction/gating; does not certify no real tire slip','optimizer_updates':0}
