"""Publish explicitly authorized yaw course with forward retention; no launch on preparation."""
import argparse,copy,json,os
from pathlib import Path
from native_stand.students.data import sha
from native_stand.students import motion_yaw_course as course
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish
ROOT=Path(__file__).resolve().parents[1]

def prepare(parent,source_run,*,write=False,performance_v2=True,grouped_evaluation=False,source_policy_retention=False,retention_coefficient=.01,early_stop_v2=True,checkpoint_every_updates=100):
 parent=Path(parent).resolve();source_run=Path(source_run).resolve();s=copy.deepcopy(json.loads(parent.read_text()))
 name=new_experiment_name('motion31_yaw_source_retention512' if source_policy_retention else 'motion31_yaw512_course');folder=ROOT/'configs/experiments'/name
 s['experiment']=name;s['motion_course']['configuration']=course.configuration()
 s['motion_course'].update(source_parent_spec=str(parent),source_parent_spec_sha256=sha(parent))
 # Yaw verification binds the previous exact course, not a rewritten historical declaration.
 s['student_training'].update(course_version=course.VERSION,course_stage=0,stage=0)
 s['student_training']['checkpoint_every_updates']=checkpoint_every_updates
 # A fresh preparation must not silently inherit an experimental optimizer loss.
 s['student_training'].pop('source_policy_retention',None)
 if source_policy_retention:
  from native_stand.students.motion_source_retention import declaration
  s['student_training']['source_policy_retention']=declaration(s['source_sha256'],retention_coefficient)
 from native_stand.students.motion_startup import smoke_profile,build
 s['student_training']['smoke_wiring']=smoke_profile()
 s['motion_training'].update(stage=0,spec_path=str(folder/'spec.json'))
 if early_stop_v2:
  from native_stand.students.motion_guard import evidence_configuration
  s['motion_training']['early_stop_policy']=evidence_configuration()
 s['motion_assessment']=course.protocol();s['student_evaluation']['version']=course.ASSESSMENT
 s['student_evaluation']['cases']=course.cases(s,0)
 s['curriculum']['stages']=[course.stage_config(0)]
 s['evaluation']['protocol_identity']=course.ASSESSMENT
 s['evaluation']['checkpoint_every_updates']=checkpoint_every_updates
 s['artifacts']['run_name']=name
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':os.environ['HAPI_SESSION_ID'],'user_scope':'User authorizes2000 inclusive updates in-place bidirectional yaw course with old forward retention; no precision stopping; original1600'}
 s['notifications']={'enabled':True}
 s['resources']={'mode':'shared','sharing_authorized':True,'authorization_note':'Current user authorizes yaw training on inherited GPU0 shared; no preemption'}
 if source_policy_retention:
  s['authorization']['user_scope']='UNAUTHORIZED new experiment proposal: old-command source policy retention; requires new explicit runtime budget and resource authorization'
 if grouped_evaluation and not performance_v2:raise ValueError('grouped evaluation requires performance v2')
 if performance_v2:
  from native_stand.students.performance import new_motion_defaults
  s['performance']=new_motion_defaults(s.get('performance'),grouped_evaluation=grouped_evaluation)
 s['student_dependencies'].update({str(parent):sha(parent)})
 reuse_folders=[source_run]+[Path(r['raw_path']).parent for r in s.get('motion_startup',{}).get('references',[])]
 s['motion_startup']=build(s,list(dict.fromkeys(reuse_folders)))
 from native_stand.students.requests import validate
 errors=validate(s)
 if errors:raise ValueError(errors)
 book={'authorization':'UNAUTHORIZED','experiment':name,'hapi_session':os.environ['HAPI_SESSION_ID'],
 'limits':{'source_eval':1,'smoke':5,'formal':1995,'evaluate':0,'play':0},'used':{'source_eval':0,'smoke':0,'formal':0,'evaluate':0,'play':0},'attempts':{},'source_eval_wall_used_s':0.,'total_online_PPO_limit':2000}
 report={'spec':str(folder/'spec.json'),'run':str(ROOT/'runs'/name),'source':s['source_checkpoint'],'source_sha256':s['source_sha256'],
 'stages':[course.stage_config(i) for i in range(3)],'source_reuse':s['motion_startup'],'simulation':0,'online_limit_inclusive':2000,
 'budget':s['budget'],'files_written':0,
 'effective_performance':__import__('native_stand.students.performance',fromlist=['settings']).settings(s),
 'source_policy_retention':s['student_training'].get('source_policy_retention'),
 'effective_early_stop_policy':s['motion_training']['early_stop_policy'],
 'checkpoint_every_updates':checkpoint_every_updates,'full_evaluation_every_updates':s['evaluation']['full_every_updates']}
 if write:
  from native_stand.students.motion_reward_audit import audit
  from native_stand.students import motion_tracking_reward
  snapshot=audit(s['reward']);snapshot.update(source_sha256=s['source_sha256'],reward_source_sha256=sha(motion_tracking_reward.__file__))
  if source_policy_retention:snapshot['optimizer_auxiliary_losses']={'source_policy_retention':s['student_training']['source_policy_retention']}
  publish(folder,{'spec.json':s,'budget.json':book,'preparation.json':report,'reward_snapshot.json':snapshot},s['student_dependencies'],completion_marker=True)
  report['files_written']=5
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-spec',required=True);p.add_argument('--source-run',required=True);p.add_argument('--write',action='store_true');p.add_argument('--performance-v2',action=argparse.BooleanOptionalAction,default=True,help='Next-run safe optimizations default on; use --no-performance-v2 for inherited profile');p.add_argument('--grouped-evaluation',action='store_true',help='Opt-in changed physical evaluation layout; requires separate parity validation');p.add_argument('--source-policy-retention',action='store_true',help='Opt-in old-command source policy KL; does not authorize a run');p.add_argument('--retention-coefficient',type=float,default=.01);p.add_argument('--early-stop-v2',action=argparse.BooleanOptionalAction,default=True,help='New specs default to per-gate evidence; --no-early-stop-v2 retains parent policy');p.add_argument('--checkpoint-every-updates',type=int,default=100);a=p.parse_args()
 print(json.dumps(prepare(a.parent_spec,a.source_run,write=a.write,performance_v2=a.performance_v2,grouped_evaluation=a.grouped_evaluation,source_policy_retention=a.source_policy_retention,retention_coefficient=a.retention_coefficient,early_stop_v2=a.early_stop_v2,checkpoint_every_updates=a.checkpoint_every_updates),indent=2))
