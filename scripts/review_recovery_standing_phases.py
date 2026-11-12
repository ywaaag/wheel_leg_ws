"""Read-only named-schema first-episode phase review; no new simulation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from native_stand.students.standing_phase_metrics import PhaseMetrics
from native_stand.students.data import sha
from native_stand.command_v6.domain.trace_schema import STATE,CONTROL,VERSION

def review(folder,output):
 folder,output=Path(folder),Path(output)
 if output.exists():raise FileExistsError(output)
 rows=[]
 for ep in sorted(folder.glob('evaluation_*.json')):
  d=json.loads(ep.read_text())
  if 'per_env' not in d or 'trace' not in d:continue
  if d.get('field_schema',{}).get('state')!=VERSION:raise ValueError('explicit trace schema required')
  tp=folder/d['trace']['path']
  if sha(tp)!=d['trace']['sha256']:raise ValueError('trace SHA differs')
  a=np.load(tp);s=a['state'];c=a['control'];p=a['posture'];T,E=s.shape[:2]
  if not all(k in a for k in ('first_episode_valid','actor_valid','previous_targets','policy_time_s')):raise ValueError('missing exact control alignment fields')
  times=a['policy_time_s'];delta=np.diff(times,axis=0)
  if not np.isfinite(times).all() or not np.allclose(delta,.01,atol=2e-6):raise ValueError('policy timestamp missing/repeated')
  entries={r['env']:r.get('entry_index') for r in d['per_env']};stat=PhaseMetrics('cpu');envs=[]
  for e in range(E):
   one=PhaseMetrics('cpu');idx=np.arange(T);entry=entries[e]
   entered=np.zeros(T,bool) if entry is None else idx>=entry
   elapsed=np.zeros(T,np.float32) if entry is None else np.maximum(idx-entry,0).astype(np.float32)*.01
   targets=np.zeros((T,6),np.float32);targets[:,[0,1,3,4]]=c[:,e,CONTROL['leg_target_after_limit']];targets[:,[2,5]]=c[:,e,CONTROL['clipped_action']][:,[2,5]]*100
   args=dict(elapsed=torch.tensor(elapsed),entered=torch.tensor(entered),controlled=torch.tensor(a['actor_valid'][:,e].reshape(T).astype(bool)&a['first_episode_valid'][:,e].astype(bool)),contacts=torch.tensor(s[:,e,10:12]),angles=torch.tensor(p[:,e,:2]),velocity=torch.tensor(s[:,e,7:10]),previous=torch.tensor(a['previous_targets'][:,e][:,[0,1,4,2,3,5]]),executed=torch.tensor(targets),requested_torque=torch.tensor(c[:,e,CONTROL['requested_torque']]),clipped_torque=torch.tensor(c[:,e,CONTROL['physical_clipped_torque']]),applied_torque=torch.tensor(c[:,e,CONTROL['applied_torque']]))
   stat.observe(**args);one.observe(**args);envs.append({'env':e,**one.result()})
  rows.append({'evaluation':str(ep),'evaluation_sha256':sha(ep),'trace':str(tp),'trace_sha256':sha(tp),'checkpoint':d['case']['checkpoint'],'case':d['case']['case_id'],'passed':sum(r['accepted'] for r in d['per_env']),'scored':len(d['per_env']),'summary':stat.result(),'per_env':envs,'slip_status':'NOT_MEASURABLE: old trace lacks wheel COM pose/velocity and total wheel angular velocity; not reconstructed from joint speed minus root vx'})
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps({'rows':rows,'simulation':0,'optimizer_updates':0,'old_scoring_unchanged':True},indent=2));return rows
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',required=True);p.add_argument('--output',required=True);a=p.parse_args();rows=review(a.folder,a.output);print('reviewed',len(rows),'evaluations')
