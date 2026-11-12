"""Prepare trained Motion spin three-seed PPO0; default CPU zero-write plan."""
import argparse,copy,json,os
from pathlib import Path
from native_stand.students.data import sha
from native_stand.students.motion_confirmation import SEEDS,verify
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish
ROOT=Path(__file__).resolve().parents[1]

def prepare(parent,checkpoint,write=False):
 parent=Path(parent).resolve();checkpoint=Path(checkpoint).resolve();s=copy.deepcopy(json.loads(parent.read_text()))
 name=new_experiment_name('motion750_spin_three_seed_ppo0');folder=ROOT/'configs/experiments'/name
 s.update(experiment=name,source_checkpoint=str(checkpoint),source_sha256=sha(checkpoint),evaluation_only=True,
  motion_confirmation=dict(version='motion31_trained_spin_three_seed_v1',parent_spec=str(parent),parent_spec_sha256=sha(parent),checkpoint_sha256=sha(checkpoint),seeds=SEEDS),
  authorization={'status':'UNAUTHORIZED','hapi_session':os.environ['HAPI_SESSION_ID'],'user_scope':'750 three-seed seven-command PPO0 only; no optimizer or continuation'},notifications={'enabled':True})
 for key in ('checkpoint_quick_evaluation','motion_startup','motion_large_continuation','student_storage_resume'):s.pop(key,None)
 s['student_evaluation']={'version':s['student_evaluation']['version'],'cases':[
  dict(row,case_id=f'seed_{seed}_command_{i}',seed=seed,checkpoint=str(checkpoint),sha256=sha(checkpoint))
  for seed in SEEDS for i,row in enumerate(s['student_evaluation']['cases'])],'max_output_bytes':512*2**20}
 s['budget'].update(source_eval_updates=0,smoke_updates=0,formal_updates=0,formal_samples=0,sample_limit=0,
  source_eval_wall_s=0,smoke_wall_s=0,formal_wall_s=0,evaluation_wall_s=7200,chain_wall_s=7200)
 s['artifacts'].update(run_name=name)
 s['student_dependencies'].update({str(parent):sha(parent),str(checkpoint):sha(checkpoint)})
 s['resources']={'mode':'shared','sharing_authorized':True,'authorization_note':'User approves750 three-seed PPO0; inherited explicit GPU0 sharing, no preemption'}
 verify(s)
 from native_stand.students.requests import validate
 errors=validate(s)
 if errors:raise ValueError(errors)
 book={'authorization':'UNAUTHORIZED','experiment':name,'limits':{'source_eval':0,'smoke':0,'formal':0,'evaluate':1,'play':0},'used':{'source_eval':0,'smoke':0,'formal':0,'evaluate':0,'play':0},'attempts':{},'source_eval_wall_used_s':0.,'total_online_PPO_limit':0}
 report={'spec':str(folder/'spec.json'),'run':str(ROOT/'runs'/name),'checkpoint_sha256':sha(checkpoint),'seeds':SEEDS,'conditions':21,'scored_envs_per_condition':8,'scored_episodes':168,'physical_envs':64,'layout':'seven-command grouped per seed; same scoring, not serial trajectory equivalence','wall_s':7200,'max_output_bytes':512*2**20,'PPO':0,'simulation':0,'files_written':0}
 if write:
  from native_stand.students.motion_reward_audit import audit
  snapshot=audit(s['reward']);snapshot.update(source_sha256=sha(checkpoint),parent_spec_sha256=sha(parent))
  publish(folder,{'spec.json':s,'budget.json':book,'preparation.json':report,'reward_snapshot.json':snapshot},s['student_dependencies'],completion_marker=True);report['files_written']=5
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-spec',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();print(json.dumps(prepare(a.parent_spec,a.checkpoint,a.write),indent=2))
