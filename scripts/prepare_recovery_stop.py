"""Recovery4100 stop refinement; zero-write default, public supervisor only."""
import argparse,copy,json
from pathlib import Path
from native_stand.students.data import sha
from native_stand.students.model import load
from native_stand.students.requests import validate
from native_stand.students.recovery_privileged import recipe,STOP_RECIPE,STOP_SOURCE
from native_stand.command_v6.tools.experiment_preparation import publish
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
ROOT=Path(__file__).resolve().parents[1]

def prepare(parent,checkpoint,session,write=False):
 parent,checkpoint=Path(parent).resolve(),Path(checkpoint).resolve();actor,data=load(checkpoint,'RecoveryStudent')
 if sha(checkpoint)!=STOP_SOURCE or data['spec_sha256']!=sha(parent):raise ValueError('Exact4100/source-spec required')
 s=copy.deepcopy(json.loads(parent.read_text()));name=new_experiment_name('recovery31_stop_from4100_2000');folder=ROOT/'configs/experiments'/name
 r=recipe(STOP_SOURCE,version=STOP_RECIPE)
 s.update(experiment=name,source_checkpoint=str(checkpoint),source_sha256=STOP_SOURCE,student_training=r,reward=r['reward'],seed=20261017,
  authorization={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'new2000 updates pending public registration'},
  notifications={'enabled':True,'target_session':session},restore_source_rng=False,restore_recovery_curriculum=False,
  resources={'mode':'shared','sharing_authorized':True,'authorization_note':'用户同意4100入口后零速奖励微调2000次GPU0共享'},execution_source_root=str(ROOT/'src'))
 s['budget'].update(smoke_updates=12,formal_updates=1988,sample_limit=16384000,formal_samples=1988*8192,source_eval_updates=0,source_eval_wall_s=1800,smoke_wall_s=1800,formal_wall_s=86400,chain_wall_s=90000,evaluation_wall_s=1800)
 s['evaluation'].update(checkpoint_every_updates=100,full_every_updates=100,full_at_final=True)
 s['student_evaluation']['max_output_bytes']=128*2**20
 s['artifacts'].update(run_name=name,storage_max_bytes=2*2**30,terminal_reserve_bytes=256*2**20)
 for case in s['student_evaluation']['cases']:case.update(checkpoint=str(checkpoint),sha256=STOP_SOURCE)
 # Provenance source libraries remain immutable; never inherit mutable progress/ledger.
 s['student_dependencies']={**s['student_dependencies'],str(parent):sha(parent),str(checkpoint):STOP_SOURCE}
 from native_stand.students.performance import new_recovery_defaults
 s['performance']=new_recovery_defaults(s.get('performance'))
 errors=validate(s)
 if errors:raise ValueError(errors)
 book={'authorization':'UNAUTHORIZED','experiment':name,'limits':{'source_eval':1,'smoke':12,'formal':1988,'evaluate':0,'play':0},'used':{'source_eval':0,'smoke':0,'formal':0,'evaluate':0,'play':0},'attempts':{},'sample_limit':16384000,'source_eval_wall_used_s':0.,'total_online_PPO_limit':2000,'hapi_session':None}
 report={'spec':str(folder/'spec.json'),'run':str(ROOT/'runs'/name),'source_sha256':STOP_SOURCE,'online_updates':2000,'smoke':12,'formal':1988,'samples':16384000,'wall_s':90000,'output_bytes':2*2**30,'simulation':0,'optimizer':0,'reward_changes':{'forward_deadband_mps':[.30,0.],'post_entry_vx_huber_scale':[.50,.20]},'source_state':'Actor/std4100; freshCritic64/PPOAdam/RNG; smoke/formal independent','full_evaluation_every_updates':100}
 report['effective_performance']=s['performance']
 if write:publish(folder,{'spec.json':s,'budget.json':book,'preparation.json':report},s['student_dependencies'],completion_marker=True)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-spec',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--session',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();print(json.dumps(prepare(a.parent_spec,a.checkpoint,a.session,a.write),indent=2))
