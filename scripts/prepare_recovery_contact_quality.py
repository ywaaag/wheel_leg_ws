"""Recovery post-entry smooth/contact draft, slip reward disabled pending runtime evidence; no-write unless --write."""
import argparse,copy,json
from pathlib import Path
from native_stand.students.data import sha
from native_stand.students.model import load
from native_stand.students.requests import validate
from native_stand.students.contact_quality import recipe,contract,network
from native_stand.students.recovery_large_training import SOURCE
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
 s=copy.deepcopy(json.loads(parent.read_text()));name=new_experiment_name('recovery31_standing_contact_quality');folder=ROOT/'configs/experiments'/name
 r=recipe();r['milestones']=[500,1000,1500,1988];interface=contract();interface.update(actor_hidden_dims=network()['actor_hidden_dims'],critic_hidden_dims=network()['critic_hidden_dims'])
 s.update(experiment=name,source_checkpoint=str(checkpoint),source_sha256=SOURCE,student_training=r,
  network_architecture=network(),student_interface=interface,reward=r['reward'],seed=20261017,
  authorization={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'UNAUTHORIZED new standing-quality recipe; no active-chain balance or automatic submission'},
  notifications={'enabled':False},restore_source_rng=False,restore_recovery_curriculum=False,
  resources={'mode':'shared','sharing_authorized':True,'authorization_note':'GPU1共享仅资源意向；新配方物理/PPO尚未授权'},execution_source_root=str(ROOT/'src'))
 s.update(phase_initialization='source Actor/std only; fresh Recovery64 Critic/PPO Adam/RNG independently for smoke and formal',
  recipe_parent_spec=str(parent),recipe_parent_spec_sha256=sha(parent),preparation_note='No teacher execution or Motion49 context; independent large Recovery stop learning',authorization_note='New2000 inclusive updates; unlimited PPO wall; no historical balance/retry')
 s['device'].update(physical_index=1,uuid='GPU-dfc076b6-b082-3065-512a-e21064067db1',logical='cuda:0')
 s['ppo'].update(envs=1024,hidden_dims=[512,256,128],learning_rate=3e-5)
 s['budget'].update(smoke_updates=12,formal_updates=1988,sample_limit=2000*1024*128,formal_samples=1988*1024*128,
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
 s['experiment_variable']='standing-only quality reward changes after first entry;contact64 Critic;source unchanged4100;no runtime alteration of takeoff'
 s['reward_recipe_source']={'kind':'new_version','version':r['reward']['version'],'baseline':str(parent),'baseline_sha256':sha(parent),'pre_entry':'unchanged','post_entry':'first-entry1s ramp; added actual wheel delta-.2/contact deficit-.5; slip0 pending physical validation'}
 errors=validate(s)
 if errors:raise ValueError(errors)
 book={'authorization':'UNAUTHORIZED','experiment':name,'limits':{'source_eval':1,'smoke':12,'formal':1988,'evaluate':0,'play':0},'used':{'source_eval':0,'smoke':0,'formal':0,'evaluate':0,'play':0},'attempts':{},'sample_limit':2000*1024*128,'source_eval_wall_used_s':0.,'total_online_PPO_limit':2000,'hapi_session':None}
 report={'spec':str(folder/'spec.json'),'run':str(ROOT/'runs'/name),'source_sha256':SOURCE,'status':'PREPARED_UNAUTHORIZED','slip_penalty':'DISABLED_PENDING_RUNTIME_VALIDATION','critic_version':'recovery_contact64_v2','online_updates':2000,'smoke':12,'formal':1988,'samples':2000*1024*128,'wall_s':None,'output_bytes':r['storage_max_bytes'],'simulation':0,'optimizer':0,'source_state':'large initialization Actor/std; fresh Critic64/Adam/RNG in each phase','checkpoint_every':100,'quick':r['checkpoint_quick_evaluation'],'full_milestones':r['milestones'],'effective_performance':s['performance']}
 if write:publish(folder,{'spec.json':s,'budget.json':book,'preparation.json':report},s['student_dependencies'],completion_marker=True)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,default=DEFAULT);p.add_argument('--parent-spec',type=Path,default=PARENT);p.add_argument('--write',action='store_true');a=p.parse_args();print(json.dumps(prepare(a.checkpoint,a.parent_spec,a.write),indent=2))
