"""Fresh750 S0 symmetry3000 declaration; no simulation or submission."""
import argparse,json
from pathlib import Path
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish
from native_stand.students.motion_symmetry_experiment import build,verify
from native_stand.students.motion_reward_audit import audit
from native_stand.students.requests import validate
ROOT=Path(__file__).resolve().parents[1]
def prepare(parent,checkpoint,write=False):
 name=new_experiment_name('motion750_symmetry85_3000');folder=ROOT/'configs/experiments'/name
 s=build(parent,checkpoint,experiment=name,spec_path=folder/'spec.json');verify(s)
 errors=validate(s)
 if errors:raise ValueError(errors)
 phases=('source_eval','smoke','formal','evaluate','play')
 book={'authorization':'UNAUTHORIZED','experiment':name,'hapi_session':None,'limits':dict(zip(phases,(1,5,2995,0,0))),
 'used':{p:0 for p in phases},'attempts':{},'source_eval_wall_used_s':0.,'total_online_PPO_limit':3000}
 result={'spec':str(folder/'spec.json'),'run':str(ROOT/'runs'/name),'source_sha256':s['source_sha256'],'smoke':5,'formal':2995,
 'online_total':3000,'wall_s':None,'storage_max_bytes':4*2**30,'simulation':0,'files_written':0}
 if write:publish(folder,{'spec.json':s,'budget.json':book,'preparation.json':result,'reward_snapshot.json':audit(s['reward'])},s['student_dependencies'],completion_marker=True);result['files_written']=5
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-spec',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();print(json.dumps(prepare(a.parent_spec,a.checkpoint,a.write),indent=2))
