"""Prepare a full27-command Motion85 independent PPO0 declaration; no launch."""
import argparse,copy,json,os
from pathlib import Path
from native_stand.students.data import sha
from native_stand.students.motion_confirmation import PARALLEL_VERSION,parallel_commands,verify
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-spec',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--write',action='store_true');p.add_argument('--authorize',action='store_true');a=p.parse_args()
 root=Path(__file__).resolve().parents[1];parent=a.parent_spec.resolve();cp=a.checkpoint.resolve();s=copy.deepcopy(json.loads(parent.read_text()));name=new_experiment_name('motion4995_full27_ppo0');folder=root/'configs/experiments'/name;session=os.environ['HAPI_SESSION_ID']
 auth='UNAUTHORIZED'
 s.update(experiment=name,source_checkpoint=str(cp),source_sha256=sha(cp),evaluation_only=True,
  motion_confirmation={'version':PARALLEL_VERSION,'parent_spec':str(parent),'parent_spec_sha256':sha(parent),'checkpoint_sha256':sha(cp),'seeds':[20261014]},
  authorization={'status':auth,'hapi_session':session,'user_scope':'User explicitly requested full4995 performance evaluation detached, no polling. One27-command native PPO0 matrix,8scored per command, original protocol/control, GPU0 shared,4h/4GiB; no PPO/retry/notification.'},notifications={'enabled':False})
 for k in ('checkpoint_quick_evaluation','motion_startup','motion_large_continuation','student_storage_resume'):s.pop(k,None)
 template=s['student_evaluation']['cases'][0]
 s['student_evaluation']={'version':s['student_evaluation']['version'],'cases':[dict(template,case_id=f'full27_{i}',seed=20261014,command=v,checkpoint=str(cp),sha256=sha(cp)) for i,v in enumerate(parallel_commands())],'max_output_bytes':512*2**20}
 s['budget'].update(source_eval_updates=0,smoke_updates=0,formal_updates=0,formal_samples=0,sample_limit=0,source_eval_wall_s=0,smoke_wall_s=0,formal_wall_s=0,evaluation_wall_s=14400,chain_wall_s=14400)
 s['artifacts'].update(run_name=name,storage_max_bytes=4*2**30);s['student_training']['storage_max_bytes']=4*2**30
 s['student_dependencies'].update({str(parent):sha(parent),str(cp):sha(cp)})
 s['resources']={'mode':'shared','sharing_authorized':True,'authorization_note':'User authorizes complete4995 evaluation; inherited explicit GPU0 sharing; no preemption'}
 verify(s)
 from native_stand.students.requests import validate
 errors=validate(s)
 if errors:raise ValueError(errors)
 book={'authorization':auth,'experiment':name,'hapi_session':session,'limits':{'source_eval':0,'smoke':0,'formal':0,'evaluate':1,'play':0},'used':{'source_eval':0,'smoke':0,'formal':0,'evaluate':0,'play':0},'attempts':{},'total_online_PPO_limit':0}
 report={'spec':str(folder/'spec.json'),'run':str(root/'runs'/name),'commands':27,'scored_episodes':216,'scored_envs':8,'physical_envs':64,'wall_s':14400,'PPO':0,'simulation':0,'files_written':0}
 if a.write:
  publish(folder,{'spec.json':s,'budget.json':book,'preparation.json':report},s['student_dependencies'],completion_marker=True);report['files_written']=4
  if a.authorize:
   from native_stand.command_v6.tools.register_authorization import register
   request={'approval_id':'user_motion4995_full27_20261021_'+name,'user_scope':s['authorization']['user_scope'],
    'session':session,'physical_gpu':s['device']['physical_index'],'gpu_uuid':s['device']['uuid'],
    'max_online_updates':0,'max_phase_wall_s':14400,'arms':[{'spec':str(folder/'spec.json'),'spec_sha256':sha(folder/'spec.json'),'budget_sha256':sha(folder/'budget.json')}]}
   report['registration']=register(request,current_session=session,write=True)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
