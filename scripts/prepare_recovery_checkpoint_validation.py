"""Publish a fresh PPO0 declaration for explicitly unmeasured Recovery checkpoints."""
import argparse,copy,json
from pathlib import Path
from native_stand.students.data import sha
from native_stand.students.requests import validate
from native_stand.students.model import load
from native_stand.students.checkpoint_validation import VERSION,FAST_VERSION
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish
ROOT=Path(__file__).resolve().parents[1]


def prepare(parent,review,write=False,*,wall_s=7200,output_bytes=2*2**30,parallel_native=False,notify_session=None):
    if type(wall_s) is not int or not 300<=wall_s<=7200:raise ValueError('wall budget must be300..7200seconds')
    if type(output_bytes) is not int or not 128*2**20<=output_bytes<=2*2**30:raise ValueError('output budget must be128MiB..2GiB')
    parent,review=Path(parent).resolve(),Path(review).resolve()
    s=copy.deepcopy(json.loads(parent.read_text()));report=json.loads(review.read_text())
    missing=[r for r in report['inventory'] if r['closed_loop_status']=='NOT_EVALUATED']
    if not missing or len(missing)>15:raise ValueError('Explicit unmeasured inventory must contain1..15 checkpoints')
    name=new_experiment_name('recovery31_saved_checkpoints_ppo0');folder=ROOT/'configs/experiments'/name
    original_cases=s['student_evaluation']['cases'];cases=[]
    dependencies={**s['student_dependencies'],str(parent):sha(parent),str(review):sha(review)}
    for r in missing:
        cp=Path(r['checkpoint']);actual=sha(cp)
        if actual!=r['sha256']:raise ValueError('Inventory checkpoint drift')
        load(cp,'RecoveryStudent');dependencies[str(cp)]=actual
        for case in (original_cases[:1] if parallel_native else original_cases):
            cases.append({**case,'case_id':f"u{r['update']}_{case['case_id']}",'checkpoint':str(cp),'sha256':actual})
    s.update(experiment=name,evaluation_only=True,execution_source_root=str(ROOT/'src'),
             authorization={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'Checkpoint PPO0 pending registration'},
             resources={'mode':'shared','sharing_authorized':True,'authorization_note':f'用户要求补测{len(missing)}个Recovery checkpoint闭环轨迹，GPU0共享'},
             notifications={'enabled':False})
    s['artifacts'].update(run_name=name)
    if notify_session:s['notifications']={'enabled':True,'target_session':notify_session}
    s['student_evaluation']={'version':FAST_VERSION if parallel_native else VERSION,'cases':cases,'max_output_bytes':output_bytes}
    if parallel_native:s['evaluation'].update(envs=len(cases),scored_envs=1)
    s['budget'].update(source_eval_updates=0,smoke_updates=0,formal_updates=0,sample_limit=0,
                       formal_samples=0,evaluation_wall_s=wall_s,chain_wall_s=wall_s)
    s['student_dependencies']=dependencies
    errors=validate(s)
    if errors:raise ValueError(errors)
    book={'authorization':'UNAUTHORIZED','experiment':name,'limits':{'evaluate':1,'source_eval':0,'smoke':0,'formal':0,'play':0},
          'used':{'evaluate':0,'source_eval':0,'smoke':0,'formal':0,'play':0},'attempts':{},'sample_limit':0,'hapi_session':None}
    result={'spec':str(folder/'spec.json'),'run':str(ROOT/'runs'/name),'ledger':str(folder/'budget.json'),
            'models':len(missing),'cases':len(cases),'scoring_episodes':len(cases)*(1 if parallel_native else 8),'simulated_envs':len(cases) if parallel_native else 64,'scored_envs':1 if parallel_native else 8,
            'PPO':0,'optimizer':0,'wall_s':wall_s,'output_bytes':output_bytes,'status':'UNAUTHORIZED_NOT_RUN'}
    if write:publish(folder,{'spec.json':s,'budget.json':book,'preparation.json':result},dependencies,completion_marker=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-spec',required=True)
    p.add_argument('--review',required=True);p.add_argument('--write',action='store_true')
    p.add_argument('--parallel-native',action='store_true')
    p.add_argument('--notify-session')
    p.add_argument('--wall-seconds',type=int,default=7200);p.add_argument('--output-mib',type=int,default=2048);a=p.parse_args()
    print(json.dumps(prepare(a.parent_spec,a.review,a.write,wall_s=a.wall_seconds,output_bytes=a.output_mib*2**20,parallel_native=a.parallel_native,notify_session=a.notify_session),indent=2))
