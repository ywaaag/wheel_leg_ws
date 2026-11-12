"""Publish same-source yaw denominator A/B; preparation never launches simulation."""
import copy,json,os
from pathlib import Path
from native_stand.students.data import sha
from native_stand.students import motion_yaw_width_course as course
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish
ROOT=Path(__file__).resolve().parents[1]

def prepare(parent,*,write=False):
 parent=Path(parent).resolve();original=json.loads(parent.read_text())
 from native_stand.students.motion_yaw_requests import verify
 verify(original)
 name=new_experiment_name('motion31_yaw_width_ab512');folder=ROOT/'configs/experiments'/name
 report={'experiment':name,'arms':{},'online_updates_total':2010,'wall_limit_s':64800,'simulation':0,'files_written':0}
 from native_stand.students.motion_tracking_reward import configuration,QUALITY_VERSION,YAW_NARROW_VERSION
 from native_stand.students.motion_startup import build
 from native_stand.students.motion_guard import evidence_configuration
 from native_stand.students.motion_reward_audit import audit
 from native_stand.students.requests import validate
 import native_stand.students.motion_tracking_reward as reward_module
 for label,version in [('A',QUALITY_VERSION),('B',YAW_NARROW_VERSION)]:
  s=copy.deepcopy(original);arm_name=name+'__'+label.lower();arm_folder=ROOT/'configs/experiments'/arm_name
  s['experiment']=arm_name;s['artifacts']['run_name']=arm_name
  s['motion_course'].update(configuration=course.configuration(),source_parent_spec=str(parent),source_parent_spec_sha256=sha(parent))
  s['student_training'].update(course_version=course.VERSION,course_stage=0,stage=0,reward=configuration(version),milestones=[250,500,750,1000],checkpoint_every_updates=100)
  s['student_training'].pop('source_policy_retention',None);s['reward']=configuration(version)
  s['motion_training'].update(stage=0,spec_path=str(arm_folder/'spec.json'),early_stop_policy=evidence_configuration())
  s['motion_assessment']=course.protocol();s['student_evaluation'].update(version=course.ASSESSMENT,cases=course.cases(s,0))
  s['curriculum']['stages']=[course.stage_config(0)]
  s['evaluation'].update(checkpoint_every_updates=100,full_every_updates=250,run_final_durability=False)
  s['budget'].update(formal_updates=1000,smoke_updates=5,formal_samples=1000*65536,sample_limit=1005*65536,chain_wall_s=32400,formal_wall_s=28800,source_eval_wall_s=1800,smoke_wall_s=1800)
  s['authorization']={'status':'UNAUTHORIZED','hapi_session':os.environ['HAPI_SESSION_ID'],'user_scope':'2026-10-16 user authorizes same-source yaw denominator1/.25 A/B; 1000formal+5smoke per arm,2010online total; GPU0shared; no promotion/retention/retry'}
  s['resources']={'mode':'shared','sharing_authorized':True,'authorization_note':'Current user approved GPU0shared Motion A/B; preserve Recovery tasks, no preemption'}
  s['notifications']={'enabled':True}
  s['student_dependencies'][str(parent)]=sha(parent)
  folders=[ROOT/'runs/2026_10_15-21_02__motion31_yaw512_course/source_eval',ROOT/'runs/2026_10_15-10_26__motion31_straight_yaw512_course/source_eval',ROOT/'runs/2026_10_15-16_56__motion31_forward512_course/source_eval']
  s['motion_startup']=build(s,folders)
  from native_stand.students.performance import new_motion_defaults
  s['performance']=new_motion_defaults(s.get('performance'))
  errors=validate(s)
  if errors:raise ValueError(errors)
  if s['motion_startup']['missing_commands']:raise ValueError('reuse matrix incomplete: '+str(s['motion_startup']['missing_commands']))
  phases=('source_eval','smoke','formal','evaluate','play')
  book={'authorization':'UNAUTHORIZED','experiment':arm_name,'hapi_session':os.environ['HAPI_SESSION_ID'],'limits':dict(zip(phases,(1,5,1000,0,0))),'used':{k:0 for k in phases},'attempts':{},'source_eval_wall_used_s':0.,'total_online_PPO_limit':1005}
  snapshot=audit(s['reward']);snapshot.update(source_sha256=s['source_sha256'],reward_source_sha256=sha(reward_module.__file__))
  row={'spec':str(arm_folder/'spec.json'),'run':str(ROOT/'runs'/arm_name),'yaw_denominator':s['reward']['yaw_denominator'],'source_sha256':s['source_sha256'],'source_reused':len(s['motion_startup']['references']),'budget':s['budget']}
  if write:publish(arm_folder,{'spec.json':s,'budget.json':book,'reward_snapshot.json':snapshot,'preparation.json':row},s['student_dependencies'],completion_marker=True)
  report['arms'][label]=row
 if write:publish(folder,{'comparison.json':report},{str(parent):sha(parent)},completion_marker=True);report['files_written']=12
 return report
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-spec',required=True);p.add_argument('--write',action='store_true');a=p.parse_args()
 print(json.dumps(prepare(a.parent_spec,write=a.write),indent=2))
