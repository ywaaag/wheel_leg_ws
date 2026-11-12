"""Public-chain Recovery31/64 recipe proposal, zero-write unless --write."""
import argparse,copy,json
from pathlib import Path
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish,checked_input
from native_stand.students.data import sha
from native_stand.students.model import load
from native_stand.students.recovery_privileged import recipe,contract,SOURCE
from native_stand.students.requests import validate,evaluation_recovery
ROOT=Path(__file__).resolve().parents[1]

def prepare(checkpoint,parent_spec,write=False):
 cp,parent=Path(checkpoint).resolve(),Path(parent_spec).resolve();actor,data=load(cp,'RecoveryStudent')
 if sha(cp)!=SOURCE:raise ValueError('explicit migrated Recovery source differs')
 manifest=cp.with_name('manifest.json');publication=cp.with_name('publication_complete.json')
 mark=json.loads(publication.read_text());meta=json.loads(manifest.read_text())
 if mark['files']['recovery_actor_initial.pt']!=sha(cp) or mark['files']['manifest.json']!=sha(manifest):raise ValueError('migration publication differs')
 s=copy.deepcopy(json.loads(parent.read_text()));name=new_experiment_name('recovery31_swing_broad_narrow_10000');folder=ROOT/'configs/experiments'/name
 for key in ('motion_training','motion_assessment','motion_confirmation','student_motion_validation','student_storage_resume','student_survival_probe','recovery_course','student_temporal_regularization','motion_course','wiring_smoke','motion_handoff','source_evidence_reference','initialization_candidate','receiver_training','retention','migration','source_update','source_input_dim','source_asset_sha256','source_spec_sha256','evaluation_requests'):
  s.pop(key,None)
 r=recipe(SOURCE);s.update(experiment=name,source_checkpoint=str(cp),source_sha256=SOURCE,evaluation_only=False,
  student_training=r,student_interface=contract(),reward=r['reward'],observation={'frame_dim':22,'actor_dim':31,'critic_dim':64},student_teachers=[],
  restore_source_rng=False,restore_recovery_curriculum=False,execution_source_root=str(ROOT/'src'),seed=20261016,
  authorization={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'Recovery10000 updates/no wall cap; CPU preparation only'},
  resources={'mode':'shared','sharing_authorized':True,'authorization_note':'用户授权GPU0共享Recovery长训10000次，不停止其他任务'},notifications={'enabled':False})
 poses=Path(s['recovery']['pose_library']);library=json.loads(poses.read_text());family='swing_candidate_from_historical_r1'
 s['recovery']=evaluation_recovery(s['recovery'],library['poses'],family)
 for key in ('support_record_reward','drawback_guidance','progress_guard','retention_regression_policy','actor_update_scope','rising_task'):s['recovery'].pop(key,None)
 for key in ('retention','training_episode'):s['recovery'].get('standing_task',{}).pop(key,None)
 s['recovery']['fsm']['timeout_s']=13.52
 s['ppo'].update(envs=64,steps_per_env=128,hidden_dims=[128,64,32],learning_rate=1e-4,epochs=4,minibatches=4,gamma=.99,lam=.95,clip=.2,value_coef=.5,entropy_coef=.001,max_grad_norm=1.)
 s['budget'].update(source_eval_updates=0,source_eval_wall_s=900,smoke_updates=12,smoke_wall_s=None,formal_updates=9988,formal_wall_s=None,chain_wall_s=None,sample_limit=81920000,formal_samples=81821696,evaluation_wall_s=900)
 s['curriculum']['episode_length_s']=13.52
 s['curriculum']['stages']=[{'id':0,'name':'RecoveryZero','commands':[[0.,0.]],'weights':[1.],'sampling_mode':'fixed_slots','sampling_slots':[0]*64,'slot_counts':[64],'enforce_whitelist':True}]
 s['evaluation'].update(seconds=13.52,envs=64,scored_envs=8,checkpoint_every_updates=100,full_every_updates=500,light_every_updates=0,full_at_final=True,run_final_durability=False)
 s['artifacts'].update(run_name=name,storage_max_bytes=8*2**30,terminal_reserve_bytes=256*2**20)
 for key in ('optimizer_capture','checkpoint_retention'):s['artifacts'].pop(key,None)
 from native_stand.students.performance import new_recovery_defaults
 s['performance']=new_recovery_defaults(s.get('performance'))
 cases=[]
 for label,seed,yaw in [('native',20261010,0.),('yaw10',20261010,.01),('yaw11',20261011,.01),('yaw12',20261012,.01)]:
  cases.append({'case_id':'student_'+label,'kind':'student','checkpoint':str(cp),'sha256':SOURCE,'role':'RecoveryStudent','family':family,'seed':seed,'yaw_rad':yaw,'seconds':13.52,'command_height_m':.34})
 s['student_evaluation']={'version':'student_evaluation_v1','cases':cases,'max_output_bytes':512*2**20}
 # Preserve teaching provenance once; no old task resume or unrelated dependency graph.
 base=json.loads(Path(data['migration']['source_spec']).read_text())
 teacher_manifest=next(Path(p) for p,h in base['student_dependencies'].items() if h==data['manifest_sha256'])
 inputs={str(p):sha(p) for p in (cp,parent,manifest,publication,teacher_manifest,poses,Path(s['asset_path']))}
 s['student_dependencies']=inputs
 errors=validate(s)
 if errors:raise ValueError(errors)
 book={'authorization':'UNAUTHORIZED','experiment':name,'limits':{'source_eval':1,'smoke':12,'formal':9988,'evaluate':0,'play':0},'used':{'source_eval':0,'smoke':0,'formal':0,'evaluate':0,'play':0},'attempts':{},'sample_limit':81920000,'source_eval_wall_used_s':0.,'total_online_PPO_limit':10000,'hapi_session':None}
 report={'spec':str(folder/'spec.json'),'run':str(ROOT/'runs'/name),'checkpoint':str(cp),'checkpoint_sha256':SOURCE,'online_updates':10000,'smoke_updates':12,'formal_updates':9988,'samples':81920000,'wall_s':None,'checkpoint_every_formal_updates':100,'full_evaluation_updates':r['milestones'],'source_episodes':32,'smoke_episodes':8,'output_bytes':8*2**30,'simulation':0,'files_written':0}
 report['effective_performance']=s['performance']
 if write:
  publish(folder,{'spec.json':s,'budget.json':book,'preparation.json':report},inputs,completion_marker=True);report['files_written']=4
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--parent-spec',required=True);p.add_argument('--write',action='store_true');a=p.parse_args()
 print(json.dumps(prepare(a.checkpoint,a.parent_spec,a.write),indent=2))
