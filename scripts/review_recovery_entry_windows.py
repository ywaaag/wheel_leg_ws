"""Read-only fixed post-entry windows: force saturation is not action clipping."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from native_stand.command_v6.domain.trace_schema import STATE,CONTROL,VERSION


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def measure(a,e,ids,active):
 if not len(ids):return None
 s=a['state'][ids,e];c=a['control'][ids,e];p=a['posture'][ids,e,:2]
 q=a['joint_q_pre'][ids,e][:,active];raw=a['raw_action'][ids,e];mean=a['mean'][ids,e]
 limits=np.array([40,40,5.5,40,40,5.5]);torque=c[:,CONTROL['physical_clipped_torque']]
 wheel=c[:,CONTROL['clipped_action']][:,[2,5]]*100
 mirror=np.stack((q[:,0]+q[:,3],q[:,1]-q[:,4]),-1)
 return dict(samples=len(ids),abs_velocity_vx_vy_yaw=np.abs(s[:,7:10]).mean(0).tolist(),
  saturation_any=float(s[:,STATE['saturated']].mean()),
  channel_torque_at_limit=(np.abs(torque)>=limits-1e-4).mean(0).tolist(),
  raw_action_outside_envelope=(np.abs(raw)>1).mean(0).tolist(),
  mean_abs_max=np.abs(mean).max(0).tolist(),mean_abs_mean=np.abs(mean).mean(0).tolist(),
  physical_mechanical_clip_rad_abs_mean=np.abs(c[:,CONTROL['leg_target_before_limit']]-c[:,CONTROL['leg_target_after_limit']]).mean(0).tolist(),
  pitch_roll_abs_mean=np.abs(p).mean(0).tolist(),pitch_roll_peak=np.abs(p).max(0).tolist(),
  mirror_mae_rad=np.abs(mirror).mean(0).tolist(),height_mean=float(s[:,6].mean()),
  wheel_target_delta_rms=(np.sqrt(np.mean(np.diff(wheel,axis=0)**2,axis=0)).tolist() if len(ids)>1 else None),
  torque_rms=np.sqrt((torque**2).mean(0)).tolist())


def review(files,output):
 output=Path(output)
 if output.exists():raise FileExistsError(output)
 rows=[];sources={}
 for filename in files:
  f=Path(filename);d=json.loads(f.read_text());trace=f.parent/d['trace']['path']
  if d['field_schema']['state']!=VERSION or sha(trace)!=d['trace']['sha256']:raise ValueError('trace identity differs')
  if sha(d['case']['checkpoint'])!=d['checkpoint_sha256']:raise ValueError('checkpoint drift')
  sources[str(f)]=sha(f);sources[str(trace)]=sha(trace)
  with np.load(trace) as z:a={k:z[k] for k in z.files}
  if not np.allclose(np.diff(a['policy_time_s']),.01,rtol=0,atol=2e-6):raise ValueError('invalid timestamps')
  u=int(Path(d['case']['checkpoint']).stem.split('_')[-1])
  for e,r in enumerate(d['per_env']):
   entry=r.get('entry_index')
   if entry is None:continue
   windows={}
   for name,lo,hi in [('entry0_1',0,100),('entry1_2',100,200),('entry0_2',0,200),('entry2_10',200,1000),('entry0_10',0,1000)]:
    ids=np.arange(entry+1+lo,min(entry+1+hi,len(a['state'])))
    if len(ids)!=hi-lo or not a['first_episode_valid'][ids,e].all():windows[name]=None;continue
    windows[name]=measure(a,e,ids,d['field_schema']['active_indices'])
   rows.append(dict(update=u,case=d['case']['case_id'],env=e,accepted=r['accepted'],windows=windows))
 groups={}
 for u in sorted({r['update'] for r in rows}):
  groups[str(u)]={}
  for name in rows[0]['windows']:
   data=[r['windows'][name] for r in rows if r['update']==u and r['windows'][name] is not None]
   groups[str(u)][name]={'episodes':len(data),'mean':{k:np.mean([v[k] for v in data],axis=0).tolist() for k in data[0]}} if data else None
 result={'version':'recovery_post_entry_window_diagnostic_v1','historical_scores_unchanged':True,'PPO':0,'optimizer':0,'new_physics_episodes':0,'sources':sources,'summary':groups,'per_env':rows,
 'limits':['Entry2..10 is an8-second diagnostic, not a new10-second certificate.','Deterministic evaluation: raw action equals bounded mean; no exploratory Gaussian noise.','Saturation is ANY physical torque at limit per100Hz policy sample; not action clipping.']}
 output.mkdir(parents=True);(output/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 lines=['# 入口后分窗诊断','', '不修改历史评分；2～10秒仅8秒诊断，不代表完整10秒保持。','', '| update | 入口后窗口 | episode | abs(vx) | abs(yaw) | 任意通道力矩触限 | pitch/roll MAE |','|---|---|---|---|---|---|---|']
 for u,windows in groups.items():
  for name,x in windows.items():
   if not x:continue
   m=x['mean'];v=m['abs_velocity_vx_vy_yaw'];p=m['pitch_roll_abs_mean']
   lines.append(f'| {u} | {name} | {x["episodes"]} | {v[0]:.4f} | {v[2]:.4f} | {m["saturation_any"]:.2%} | {p[0]:.4f}/{p[1]:.4f} |')
 (output/'REPORT.md').write_text('\n'.join(lines)+'\n');return groups

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluations',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 print(json.dumps(review(a.evaluations,a.output),indent=2))
