"""Prepare explicit large Motion combined first block; never launches."""
import argparse,copy,json,os
from pathlib import Path
from native_stand.students.data import sha
from native_stand.students.motion_large_combo import network,configuration,stage_config,verify,VERSION
from native_stand.students.motion_tracking_reward import configuration as reward,COMBO_VERSION
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish

ROOT=Path(__file__).resolve().parents[1]

def prepare(parent,*,write=False):
 parent=Path(parent).resolve();s=copy.deepcopy(json.loads(parent.read_text()))
 from native_stand.students.motion_yaw_retention_requests import verify as verify_parent
 verify_parent(s)
 name=new_experiment_name('motion31_large_combo4096');folder=ROOT/'configs/experiments'/name
 s.update(experiment=name,network_architecture=network(),motion_large_combo={'version':VERSION,'parent_spec':str(parent),'parent_spec_sha256':sha(parent)})
 s['artifacts']['run_name']=name;s['motion_training']['spec_path']=str(folder/'spec.json')
 s['motion_course']['configuration']=configuration();s['curriculum']['stages']=[stage_config(0)]
 s['ppo'].update(envs=4096,hidden_dims=[512,256,128])
 s['reward']=reward(COMBO_VERSION);s['student_training'].update(course_version=VERSION,network_architecture=network(),reward=copy.deepcopy(s['reward']))
 s['budget'].update(sample_limit=1005*4096*128,formal_samples=1000*4096*128)
 s['student_dependencies'][str(parent)]=sha(parent)
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':os.environ.get('HAPI_SESSION_ID'),'user_scope':'user approved combined large Actor/Critic, yaw gate and4096env; unlimited total intent, first block5smoke+1000formal,9h/2GiB, GPU0shared, retry allowed with new attempts and accounting; no change to active predecessor'}
 from native_stand.students.motion_quick_evaluation import configuration as quick
 s['checkpoint_quick_evaluation']=quick()
 from native_stand.students.performance import new_motion_defaults
 s['performance']=new_motion_defaults(s.get('performance'))
 verify(s)
 from native_stand.students.requests import validate
 errors=validate(s)
 if errors:raise ValueError(errors)
 phases=('source_eval','smoke','formal','evaluate','play')
 book={'authorization':'UNAUTHORIZED','experiment':name,'hapi_session':os.environ.get('HAPI_SESSION_ID'),
       'limits':dict(zip(phases,(1,5,1000,0,0))),'used':{k:0 for k in phases},'attempts':{},'total_online_PPO_limit':1005,'source_eval_wall_used_s':0.}
 from native_stand.students.motion_reward_audit import audit
 snapshot=audit(s['reward'])
 report={'spec':str(folder/'spec.json'),'run':str(ROOT/'runs'/name),'source_sha256':s['source_sha256'],
         'simulation':0,'optimizer_updates':0,'budget':s['budget'],'runtime_authorized':False,'combined_variables':['Actor width','Critic width','yaw translation gate','4096env'],
         'capacity_smoke':'five actual updates includes4Critic+1Actor; no skill gate; no silent env fallback'}
 report['effective_performance']=s['performance']
 if write:publish(folder,{'spec.json':s,'budget.json':book,'reward_snapshot.json':snapshot,'preparation.json':report},s['student_dependencies'],completion_marker=True)
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-spec',required=True);p.add_argument('--write',action='store_true');a=p.parse_args()
 print(json.dumps(prepare(a.parent_spec,write=a.write),ensure_ascii=False,indent=2))
