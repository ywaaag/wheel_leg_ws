#!/usr/bin/env python3
"""CPU-only Motion810 spin quality/Critic61 draft; never authorizes simulation/PPO."""
import argparse,json
from pathlib import Path
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.students.motion_spin_experiment import build,verify
from native_stand.students.motion_reward_audit import audit
from native_stand.students.requests import validate
from native_stand.students.data import sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--parent-spec',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--formal-updates',type=int,default=1000,help='proposal only, not approved budget')
 p.add_argument('--write-draft',action='store_true',help='exclusive-create artifact draft only; no config/run/ledger')
 p.add_argument('--publish',action='store_true',help='publish fresh UNAUTHORIZED config and zero-used ledger; never launch')
 p.add_argument('--unlimited-wall',action='store_true',help='declare no formal/chain wall cap; still requires explicit authorization')
 a=p.parse_args();root=Path(__file__).resolve().parents[1];name=new_experiment_name('motion810_spin_quality61_draft')
 folder=root/'artifacts'/name
 if a.publish and a.write_draft:raise ValueError('publish and artifact draft are separate outputs')
 if a.publish:
  name=new_experiment_name('motion810_spin_quality61_2000');folder=root/'configs/experiments'/name
 s=build(a.parent_spec,a.checkpoint,experiment=name,spec_path=folder/('spec.json' if a.publish else 'spec_draft.json'),formal_updates=a.formal_updates,unlimited_wall=a.unlimited_wall)
 verify(s);errors=validate(s)
 if errors:raise ValueError(errors)
 result={'status':'CPU_DRAFT_VALID','submit_ready':False,'reason_codes':['AUTHORIZATION_MISSING','MISSING_PHYSICAL_CONTACT_CALIBRATION'],
  'actor_dim':31,'critic_dim':61,'source_checkpoint':s['source_checkpoint'],'source_sha256':s['source_sha256'],
  'proposed_smoke':5,'proposed_formal':a.formal_updates,'proposed_budget_not_authorized':True,'simulation':0,'PPO':0,'files_written':0}
 if a.publish:
  from native_stand.command_v6.tools.experiment_preparation import publish
  phases=('source_eval','smoke','formal','evaluate','play')
  book={'authorization':'UNAUTHORIZED','experiment':name,'hapi_session':None,'limits':dict(zip(phases,(1,5,a.formal_updates,0,0))),
   'used':{k:0 for k in phases},'attempts':{},'total_online_PPO_limit':a.formal_updates+5,'source_eval_wall_used_s':0.}
  result.update(output=str(folder),spec=str(folder/'spec.json'),run=str(root/'runs'/name),files_written=5)
  publish(folder,{'spec.json':s,'budget.json':book,'reward_snapshot.json':audit(s['reward']),'preparation.json':result},s['student_dependencies'],completion_marker=True)
 if a.write_draft:
  folder.mkdir(exist_ok=False)
  def write(name,data):
   with (folder/name).open('x') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)
  write('spec_draft.json',s);write('reward_snapshot.json',{'source_checkpoint_sha256':s['source_sha256'],'parent_spec_sha256':sha(a.parent_spec),'audit':audit(s['reward']),
   'implementation_sha256':{str(x.relative_to(root)):sha(x) for x in [root/'src/native_stand/students/motion_tracking_reward.py',root/'src/native_stand/students/motion_yaw_quality.py',root/'src/native_stand/students/motion_spin_privileged.py']}})
  result.update(files_written=3,output=str(folder));write('preparation.json',result)
 print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
