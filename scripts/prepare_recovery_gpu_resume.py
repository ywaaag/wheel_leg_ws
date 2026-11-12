"""Publish same-recipe Recovery GPU1 migration with exact remaining budget."""
import argparse,copy,json
from pathlib import Path
from native_stand.students.data import sha
from native_stand.students.model import load
from native_stand.students.gpu_resume import verify,VERSION
from native_stand.students.requests import validate
from native_stand.command_v6.tools.experiment_preparation import publish
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
ROOT=Path(__file__).resolve().parents[1]

def prepare(parent,*,write=False):
 parent=Path(parent).resolve();old=json.loads(parent.read_text());run=ROOT/'runs'/old['experiment'];book=json.loads(parent.with_name('budget.json').read_text())
 if any(a['status']=='RUNNING' for a in book['attempts'].values()):raise ValueError('parent still running; stop normally first')
 cp=max((run/'formal').glob('model_update_*.pt'),key=lambda p:int(p.stem.rsplit('_',1)[1]));_,data=load(cp,'RecoveryStudent')
 used=book['used']['formal']+book['used']['smoke'];limit=book['limits']['formal']+book['limits']['smoke'];remaining=limit-used;n=data['update']
 if remaining<=0:raise ValueError('no remaining updates')
 name=new_experiment_name('recovery31_large_stop_gpu1_resume');folder=ROOT/'configs/experiments'/name;s=copy.deepcopy(old)
 s.update(experiment=name,source_checkpoint=str(cp),source_sha256=sha(cp),authorization={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'newGPU1 migration registration pending'},
 device={'physical_index':1,'uuid':'GPU-dfc076b6-b082-3065-512a-e21064067db1','logical':'cuda:0','headless':True},
 resources={'mode':'shared','sharing_authorized':True,'authorization_note':'用户明确将本Recovery训练迁移GPU1，其他配方及总2500额度不变'},
 phase_initialization='restore Actor/Critic/PPO Adam/std/RNG from saved full checkpoint; no repeated warmup; real PhysX reset',notifications={'enabled':False})
 s['student_training']['source_actor_sha256']=sha(cp)
 s['student_training']['milestones']=sorted(set([u-n for u in old['student_training']['milestones'] if n<u<n+remaining]+[remaining]))
 s['budget'].update(smoke_updates=0,formal_updates=remaining,sample_limit=remaining*1024*128,formal_samples=remaining*1024*128)
 s['artifacts']['run_name']=name
 for row in s['student_evaluation']['cases']:row.update(checkpoint=str(cp),sha256=sha(cp))
 files=[parent,parent.with_name('budget.json'),cp,run/'chain_terminal_report.json',run/'report_ready.json',run/'formal/terminal_report.json',run/'formal/source_snapshot/spec.json',run/'smoke/wiring_certificate.json']
 inputs={str(p):sha(p) for p in files}
 c={'version':VERSION,'parent_spec':str(parent),'parent_run':str(run),'inputs':inputs,'parent_checkpoint_update':n,'historical_online_updates':used,'original_online_limit':limit,'unused_parent_formal_updates':remaining,'uncheckpointed_updates':book['used']['formal']-n}
 s['student_gpu_resume']=c;s['student_dependencies'].update(inputs)
 errors=validate(s)
 if errors:raise ValueError(errors)
 verify(s)
 newbook={'authorization':'UNAUTHORIZED','experiment':name,'limits':{'source_eval':1,'smoke':0,'formal':remaining,'evaluate':0,'play':0},'used':dict.fromkeys(('source_eval','smoke','formal','evaluate','play'),0),'attempts':{},'sample_limit':remaining*1024*128,'source_eval_wall_used_s':0.,'total_online_PPO_limit':remaining,'historical_online_updates':used,'original_task_online_limit':limit,'hapi_session':None}
 report={'spec':str(folder/'spec.json'),'run':str(ROOT/'runs'/name),'checkpoint':str(cp),'source_sha256':sha(cp),'saved_formal_update':n,'charged_parent_smoke':book['used']['smoke'],'charged_parent_formal':book['used']['formal'],'discarded_but_charged_updates':c['uncheckpointed_updates'],'remaining_updates':remaining,'remaining_samples':remaining*1024*128,'original_task_updates':limit,'simulation':0,'optimizer_updates':0,'full_restore':'Actor/Critic/Adam/std/RNG; no warmup','physical_reset':True,'gpu':'physical1 -> logicalcuda0','milestones_local':s['student_training']['milestones']}
 if write:publish(folder,{'spec.json':s,'budget.json':newbook,'preparation.json':report},s['student_dependencies'],completion_marker=True)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-spec',required=True);p.add_argument('--write',action='store_true');a=p.parse_args();print(json.dumps(prepare(a.parent_spec,write=a.write),indent=2))
