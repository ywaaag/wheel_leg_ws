"""Read-only saved-input reward replay, never produces physical capability evidence."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from native_stand.students.data import sha
from native_stand.students.recovery_large_training import reward,configuration,FIELDS
from native_stand.command_v6.domain.trace_schema import STATE,CONTROL,VERSION

def replay(run,output):
 run,output=Path(run),Path(output)
 if output.exists():raise FileExistsError(output)
 rows=[]
 for index in range(4):
  ep=run/f'evaluation_{index:02}.json';raw=json.loads(ep.read_text());tp=run/f'trace_{index:02}.npz'
  if raw['field_schema']['state']!=VERSION or raw['checkpoint_sha256']!='0f4bebb2a763d8a44b863bf6017f3388a999e475af8c58f1dee07857d7aa27cd':raise ValueError('source/schema differs')
  a=np.load(tp);s=a['state'];p=a['posture'];c=a['control'];T,E=s.shape[:2]
  tensor=lambda v:torch.tensor(v,dtype=torch.float32).reshape(-1,*np.asarray(v).shape[2:])
  body=Rotation.from_quat(a['root_post'][:,:,3:7].reshape(-1,4)).inv().apply(a['root_post'][:,:,10:13].reshape(-1,3)).reshape(T,E,3)
  entered=np.zeros((T,E),bool);elapsed=np.zeros((T,E),np.float32);h0=np.zeros((T,E),np.float32)
  for e,r in enumerate(raw['per_env']):
   first=np.flatnonzero(a['actor_valid'][:,e,0])
   if not len(first):raise ValueError('controlled baseline missing')
   h0[:,e]=a['root_pre'][first[0],e,2]
   k=r['entry_index']
   if k is not None:entered[k:,e]=True;elapsed[k:,e]=np.arange(T-k)*.01
  # All physical channels authored by the named control schema; previous_targets
  # is four legs then two wheels, not left-first. Explicit remapping here.
  previous=a['previous_targets'][:,:,[0,1,4,2,3,5]]
  target=np.zeros((T,E,6),np.float32);target[:,:,[0,1,3,4]]=c[:,:,CONTROL['leg_target_after_limit']]
  target[:,:,[2,5]]=c[:,:,CONTROL['clipped_action']][:,:,[2,5]]*100
  events=a['student_task_events'];b=lambda x:torch.tensor(x.reshape(-1),dtype=torch.bool)
  r=reward(height=tensor(s[:,:,STATE['height_m']]),h0=tensor(h0),angles=tensor(p[:,:,:2]),contacts=tensor(s[:,:,10:12]),clearance=tensor(s[:,:,STATE['clearance_m']]),leg_q=tensor(c[:,:,CONTROL['joint_q']][:,:,[0,1,3,4]]),angular_pitch_roll=tensor(body[:,:,[0,2]]),velocity_vx_vy_yaw=tensor(s[:,:,7:10]),applied_torque=tensor(c[:,:,CONTROL['applied_torque']]),previous_targets=tensor(previous),executed_targets=tensor(target),entered=b(entered),controlled=b(a['actor_valid'][:,:,0]&a['first_episode_valid']),valid=b(s[:,:,STATE['initial_valid']]>0),fault=b(s[:,:,STATE['runtime_fault']]>0),entry_event=b(events[:,:,0]),success_event=b(events[:,:,2]),failure_event=b(events[:,:,3]),age=tensor(s[:,:,STATE['recovery_clock_s']]),entry_elapsed=tensor(elapsed),reward_configuration=configuration())
  density=r['weighted_density'].reshape(T,E,len(FIELDS));hold=torch.tensor(entered&(elapsed>=1))&r['eligible'].reshape(T,E)
  rows.append({'evaluation':str(ep),'evaluation_sha256':sha(ep),'trace':str(tp),'trace_sha256':sha(tp),'samples':int(r['eligible'].sum()),'full_hold_samples_after_ramp':int(hold.sum()),'weighted_density_mean':dict(zip(FIELDS,density[hold].mean(0).tolist())),'new_vx_reward_density_mean':float(density[:,:,11:].sum(-1)[hold].mean()),'offline_reward_return_sum':float(r['total'].sum())})
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps({'scope':'fixed saved teacher-policy trajectory reward replay; no PPO/physics; not a learning forecast','configuration':configuration(),'rows':rows},indent=2));return rows
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(replay(a.run,a.output),indent=2))
