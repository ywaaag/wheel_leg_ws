"""Prepare one old-command retention experiment; default plan writes nothing, never launches."""
import copy,json,os
from pathlib import Path
from native_stand.students.data import sha
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish
ROOT=Path(__file__).resolve().parents[1]

def build(parent,name,session):
 from native_stand.students.motion_yaw_width_requests import verify
 from native_stand.students import motion_yaw_retention_course as course
 from native_stand.students.motion_source_retention import declaration
 from native_stand.students.requests import validate
 parent=Path(parent).resolve();s=copy.deepcopy(json.loads(parent.read_text()));verify(s)
 folder=ROOT/'configs/experiments'/name
 s['experiment']=name;s['artifacts']['run_name']=name
 s['student_training'].update(course_version=course.VERSION,source_policy_retention=declaration(s['source_sha256'],.01))
 s['motion_training']['spec_path']=str(folder/'spec.json')
 s['motion_course'].update(configuration=course.configuration(),retention_parent_spec=str(parent),retention_parent_spec_sha256=sha(parent))
 s['student_dependencies'][str(parent)]=sha(parent)
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':session,'user_scope':'2026-10-16 user approved next old-command retention experiment from original1600: coefficient.01, yawPPOonly, same narrowB recipe; agreed proposed5smoke+1000formal,9h,2GiB,GPU0shared; no retry or promotion.'}
 from native_stand.students.performance import new_motion_defaults
 s['performance']=new_motion_defaults(s.get('performance'))
 errors=validate(s)
 if errors:raise ValueError(errors)
 return s

def prepare(parent,*,write=False,quick_evaluation=True):
 from native_stand.students.motion_reward_audit import audit
 from native_stand.students import motion_tracking_reward
 from native_stand.students.motion_startup import resolve
 name=new_experiment_name('motion31_yaw_retention512');s=build(parent,name,os.environ.get('HAPI_SESSION_ID'));folder=ROOT/'configs/experiments'/name
 if quick_evaluation:
  from native_stand.students.motion_quick_evaluation import configuration,validate
  s['checkpoint_quick_evaluation']=configuration();validate(s)
 records,missing=resolve(s)
 phases=('source_eval','smoke','formal','evaluate','play')
 book={'authorization':'UNAUTHORIZED','experiment':name,'hapi_session':os.environ.get('HAPI_SESSION_ID'),'limits':dict(zip(phases,(1,5,1000,0,0))),'used':{k:0 for k in phases},'attempts':{},'source_eval_wall_used_s':0.,'total_online_PPO_limit':1005}
 snapshot=audit(s['reward']);snapshot.update(source_sha256=s['source_sha256'],reward_source_sha256=sha(motion_tracking_reward.__file__),optimizer_auxiliary_losses={'source_policy_retention':s['student_training']['source_policy_retention']})
 report={'spec':str(folder/'spec.json'),'run':str(ROOT/'runs'/name),'source':s['source_checkpoint'],'source_sha256':s['source_sha256'],'control_spec':str(Path(parent).resolve()),'retention':s['student_training']['source_policy_retention'],'source_reused':len(records),'source_missing':[x['command'] for x in missing],'checkpoint_quick_evaluation':s.get('checkpoint_quick_evaluation'),'budget_proposal':s['budget'],'runtime_authorized':False,'authorization_note':'new ledger remains UNAUTHORIZED until explicit approval registration; no implicit reuse of parent balance','simulation':0,'ppo':0,'files_written':0}
 if write:
  publish(folder,{'spec.json':s,'budget.json':book,'reward_snapshot.json':snapshot,'preparation.json':report},s['student_dependencies'],completion_marker=True)
  report['files_written']=5
 return report

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-spec',required=True);p.add_argument('--write',action='store_true');p.add_argument('--quick-evaluation',action=argparse.BooleanOptionalAction,default=True,help='Next declaration: parallel diagnostics at every100 saved updates, never changes active runs');a=p.parse_args()
 print(json.dumps(prepare(a.parent_spec,write=a.write,quick_evaluation=a.quick_evaluation),ensure_ascii=False,indent=2))
