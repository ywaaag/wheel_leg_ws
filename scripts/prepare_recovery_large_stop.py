"""Large Recovery2500 stop learning declaration; no-write unless --write."""
import argparse,copy,json
from pathlib import Path
from native_stand.students.data import sha
from native_stand.students.model import load
from native_stand.students.requests import validate
from native_stand.students.recovery_large_training import recipe,SOURCE
from native_stand.students.recovery_large import network
from native_stand.students.recovery_privileged import contract
from native_stand.students.performance import new_recovery_defaults
from native_stand.command_v6.tools.experiment_preparation import publish
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'isolated/2026_10_17-07_55__recovery31_large_from4100/initialization_candidate.pt'
PARENT=ROOT/'runs/2026_10_15-22_57__recovery31_swing_broad_narrow_10000/formal/source_snapshot/spec.json'

def prepare(checkpoint=DEFAULT,parent=PARENT,write=False):
 checkpoint,parent=Path(checkpoint).resolve(),Path(parent).resolve()
 actor,cp=load(checkpoint,'RecoveryStudent')
 if sha(checkpoint)!=SOURCE or sha(parent)!=cp['source_spec_sha256']:raise ValueError('large source/parent SHA differs')
 s=copy.deepcopy(json.loads(parent.read_text()));name=new_experiment_name('recovery31_large_stop1024_2500');folder=ROOT/'configs/experiments'/name
 r=recipe();interface=contract();interface.update(actor_hidden_dims=network()['actor_hidden_dims'],critic_hidden_dims=network()['critic_hidden_dims'])
 s.update(experiment=name,source_checkpoint=str(checkpoint),source_sha256=SOURCE,student_training=r,
  network_architecture=network(),student_interface=interface,reward=r['reward'],seed=20261017,
  authorization={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'pending new2500 public registration'},
  notifications={'enabled':False},restore_source_rng=False,restore_recovery_curriculum=False,
  resources={'mode':'shared','sharing_authorized':True,'authorization_note':'用户授权Recovery2500更新/1024env/不限时/GPU0共享'},execution_source_root=str(ROOT/'src'))
 s.update(phase_initialization='source Actor/std only; fresh Recovery64 Critic/PPO Adam/RNG independently for smoke and formal',
  recipe_parent_spec=str(parent),recipe_parent_spec_sha256=sha(parent),preparation_note='No teacher execution or Motion49 context; independent large Recovery stop learning',authorization_note='New2500 inclusive updates; unlimited PPO wall; no historical balance/retry')
 s['ppo'].update(envs=1024,hidden_dims=[512,256,128],learning_rate=3e-5)
 s['budget'].update(smoke_updates=12,formal_updates=2488,sample_limit=2500*1024*128,formal_samples=2488*1024*128,
  source_eval_updates=0,source_eval_wall_s=3600,smoke_wall_s=None,formal_wall_s=None,chain_wall_s=None,evaluation_wall_s=1800)
 s['evaluation'].update(checkpoint_every_updates=100,full_every_updates=500,full_at_final=True,protocol_identity='student_autonomous_swing_v1')
 s['student_evaluation']['max_output_bytes']=128*2**20
 s['artifacts'].update(run_name=name,storage_max_bytes=r['storage_max_bytes'],terminal_reserve_bytes=512*2**20)
 s['curriculum']['stages'][0].update(sampling_slots=[0]*1024,slot_counts=[1024])
 for case in s['student_evaluation']['cases']:case.update(checkpoint=str(checkpoint),sha256=SOURCE)
 s['student_dependencies'].update({str(parent):sha(parent),str(checkpoint):SOURCE,
  str(checkpoint.with_name('manifest.json')):sha(checkpoint.with_name('manifest.json')),
  str(cp['source_checkpoint']):cp['source_sha256']})
 s['performance']=new_recovery_defaults(s.get('performance'))
 s['experiment_variable']='large Recovery stop reward;1024 parallel;LR3e-5;new Critic/Adam; source function-preserving4100'
 s['reward_recipe_source']={'kind':'new_version','version':r['reward']['version'],'baseline':str(parent),'baseline_sha256':sha(parent),'pre_entry':'unchanged','post_entry':'first-entry1s ramp; vx Huber-.1 plus wide .5/narrow1.5 rewards'}
 errors=validate(s)
 if errors:raise ValueError(errors)
 book={'authorization':'UNAUTHORIZED','experiment':name,'limits':{'source_eval':1,'smoke':12,'formal':2488,'evaluate':0,'play':0},'used':{'source_eval':0,'smoke':0,'formal':0,'evaluate':0,'play':0},'attempts':{},'sample_limit':2500*1024*128,'source_eval_wall_used_s':0.,'total_online_PPO_limit':2500,'hapi_session':None}
 report={'spec':str(folder/'spec.json'),'run':str(ROOT/'runs'/name),'source_sha256':SOURCE,'online_updates':2500,'smoke':12,'formal':2488,'samples':2500*1024*128,'wall_s':None,'output_bytes':r['storage_max_bytes'],'simulation':0,'optimizer':0,'source_state':'large initialization Actor/std; fresh Critic64/Adam/RNG in each phase','checkpoint_every':100,'quick':r['checkpoint_quick_evaluation'],'full_milestones':r['milestones'],'effective_performance':s['performance']}
 if write:publish(folder,{'spec.json':s,'budget.json':book,'preparation.json':report},s['student_dependencies'],completion_marker=True)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,default=DEFAULT);p.add_argument('--parent-spec',type=Path,default=PARENT);p.add_argument('--write',action='store_true');a=p.parse_args();print(json.dumps(prepare(a.checkpoint,a.parent_spec,a.write),indent=2))
