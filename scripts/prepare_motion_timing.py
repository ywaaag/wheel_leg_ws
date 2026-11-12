"""Prepare4750 standing-first timing DR; default zero writes, no implicit launch."""
import argparse,json,os
from pathlib import Path
from native_stand.students.motion_timing_experiment import build,verify
from native_stand.students.motion_reward_audit import audit
from native_stand.students.data import sha
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--parent-spec',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--critic140',action='store_true',help='same10000 budget, stopped110 chain to explicit140 Critic')
 p.add_argument('--wheel-match',action='store_true',help='new10000 inclusive PPO from stopped140 latest, wheel J candidate and Kv.2')
 p.add_argument('--antioscillation',action='store_true',help='new10000 inclusive PPO and evidence-gated .35 to .012 inertia course')
 p.add_argument('--sensor-wiring-recovery',action='store_true',help='reuse corrected real smoke, no repeated PPO')
 p.add_argument('--sensor-transfer',action='store_true',help='new10ms Actor sensing plus V14-inspired smoothness/effort costs')
 p.add_argument('--fixed-retry-parent',type=Path,help='settled zero-PPO fixed protocol dispatch failure only')
 p.add_argument('--fixed-leg-actuator',action='store_true',help='new fixed target delay with four leg armature.35/Kp40/Kd2')
 p.add_argument('--write',action='store_true');p.add_argument('--authorize',action='store_true')
 p.add_argument('--retry-parent',type=Path);a=p.parse_args()
 if a.wheel_match and any((a.critic140,a.antioscillation,a.sensor_wiring_recovery,a.sensor_transfer,a.fixed_leg_actuator,a.fixed_retry_parent,a.retry_parent)):p.error('wheel matching is a separate new configuration and authorization')
 if a.sensor_wiring_recovery and (a.sensor_transfer or a.fixed_leg_actuator or a.fixed_retry_parent or a.retry_parent):p.error('schema recovery is a separate constrained operation')
 if a.sensor_transfer and (a.fixed_leg_actuator or a.fixed_retry_parent or a.retry_parent):p.error('sensor transfer is a separate new recipe')
 if a.fixed_leg_actuator and a.retry_parent:p.error('fixed actuator is a new recipe, not wiring retry')
 if a.critic140 and any((a.antioscillation,a.sensor_wiring_recovery,a.sensor_transfer,a.fixed_leg_actuator,a.fixed_retry_parent,a.retry_parent)):p.error('Critic140 is a separate checked migration')
 if a.antioscillation and any((a.sensor_wiring_recovery,a.sensor_transfer,a.fixed_leg_actuator,a.fixed_retry_parent,a.retry_parent)):p.error('anti-oscillation is a separate new recipe')
 root=Path(__file__).resolve().parents[1];name=new_experiment_name('motion_fixed_timing_armature035_kp40' if a.fixed_leg_actuator else 'motion4750_standing_timingDR_10000')
 builder=build
 if a.fixed_leg_actuator:
  from native_stand.students.motion_fixed_timing import build as builder
 if a.sensor_transfer:
  from native_stand.students.motion_sensor_transfer import build as builder
  name=new_experiment_name('motion526_sensor10ms_smooth_9188')
 if a.sensor_wiring_recovery:
  from native_stand.students.motion_sensor_transfer import build_recovery as builder
  name=new_experiment_name('motion526_sensor_schema_formal9183')
 if a.antioscillation:
  from native_stand.students.motion_antioscillation import build as builder
  name=new_experiment_name('motion500_antiosc_inertia_10000')
 if a.critic140:
  from native_stand.students.motion_antiosc_privileged import build as builder
  name=new_experiment_name('motion_antiosc_critic140_resume')
 if a.wheel_match:
  from native_stand.students.motion_wheel_matching import build as builder
  name=new_experiment_name('motion387_wheelmatch_kv02_10000')
 folder=root/'configs/experiments'/name;s=builder(a.parent_spec,a.checkpoint,experiment=name,spec_path=folder/'spec.json')
 if a.fixed_retry_parent:
  if not a.fixed_leg_actuator:p.error('--fixed-retry-parent requires --fixed-leg-actuator')
  from native_stand.students.motion_fixed_timing import retry_evidence
  s['motion_timing']['fixed_retry']=retry_evidence(a.fixed_retry_parent)
 if a.retry_parent:
  from native_stand.students.motion_timing_experiment import recovery_evidence
  s['motion_timing_recovery']=recovery_evidence(a.retry_parent)
 session=os.environ['HAPI_SESSION_ID'];s['authorization']['hapi_session']=session
 verify(s)
 from native_stand.students.requests import validate
 errors=validate(s)
 if errors:raise ValueError(errors)
 phases=['source_eval','smoke','formal','evaluate','play']
 smoke=s['budget']['smoke_updates'];formal=s['budget']['formal_updates']
 book={'authorization':'UNAUTHORIZED','experiment':name,'hapi_session':session,'limits':dict(zip(phases,[0 if s['motion_training'].get('wiring_reuse') else 1,smoke,formal,0,0])),
       'used':{k:0 for k in phases},'attempts':{},'source_eval_wall_used_s':0.,'total_online_PPO_limit':10000}
 if s['motion_training'].get('wiring_reuse'):book['historical_consumption']=s['motion_training']['wiring_reuse']
 if a.critic140:book['historical_consumption']=s['motion_timing']['critic_upgrade']
 if a.wheel_match:book['previous_experiment_consumption']=s['motion_timing']['wheel_match_restart']
 if a.sensor_transfer:book['historical_consumption']=s['motion_timing']['fixed_parent']
 if a.fixed_leg_actuator:book['historical_consumption']=s['motion_timing']['fixed_parent']
 if a.retry_parent:book['historical_consumption']=s['motion_timing_recovery']
 result={'spec':str(folder/'spec.json'),'run':str(root/'runs'/name),'source_sha256':s['source_sha256'],
         'online_updates':smoke+formal,'smoke':smoke,'formal':formal,'PPO':0,'simulation':0,'files_written':0,
         'actuator_configuration':s['motion_timing']['configuration'],
         'critic':s['observation']['critic_dim'],'actor':31,'performance':s['performance']}
 if a.write:
  reward_snapshot=audit(s['reward'])
  if a.sensor_transfer or a.sensor_wiring_recovery:
   from native_stand.students.motion_sensor_transfer import reward_snapshot as extension_snapshot
   reward_snapshot['runtime_reward_extension']=extension_snapshot()
  if a.antioscillation or a.critic140 or a.wheel_match:
   from native_stand.students.motion_antioscillation import reward_snapshot as anti_snapshot
   reward_snapshot['runtime_reward_extension']=anti_snapshot(s['motion_timing']['configuration'])
  payloads={'spec.json':s,'budget.json':book,'preparation.json':result,'reward_snapshot.json':reward_snapshot}
  if a.sensor_wiring_recovery:payloads['corrected_wiring_certificate.json']=s['motion_training']['wiring_reuse']['corrected_certificate']
  publish(folder,payloads,s['student_dependencies'],completion_marker=True)
  result['files_written']=6 if a.sensor_wiring_recovery else 5
  if a.wheel_match:
   parent=a.parent_spec.resolve();claim=parent.parent.parent/'.motion_wheel_match_claims'/(sha(parent)+'.json');claim.parent.mkdir(exist_ok=True)
   with claim.open('x') as stream:json.dump({'destination':name,'spec':str(folder/'spec.json')},stream)
  if a.critic140:
   parent=Path(s['motion_timing']['parent_spec']);claim=parent.parent.parent/'.motion_critic140_claims'/(sha(parent)+'.json');claim.parent.mkdir(exist_ok=True)
   with claim.open('x') as stream:json.dump({'destination':name,'spec':str(folder/'spec.json')},stream)
  if a.sensor_transfer:
   parent=Path(s['motion_timing']['parent_spec']);claim=parent.parent.parent/'.motion_sensor_transfer_claims'/(sha(parent)+'.json')
   claim.parent.mkdir(exist_ok=True)
   with claim.open('x') as stream:json.dump({'destination':name,'spec':str(folder/'spec.json')},stream)
  if a.fixed_leg_actuator:
   parent=a.fixed_retry_parent or Path(s['motion_timing']['parent_spec'])
   claim=parent.parent.parent/'.motion_fixed_timing_claims'/(sha(parent)+'.json')
   claim.parent.mkdir(exist_ok=True)
   with claim.open('x') as stream:json.dump({'destination':name,'spec':str(folder/'spec.json')},stream)
  if a.sensor_wiring_recovery:
   parent=Path(s['motion_training']['wiring_reuse']['parent_spec']);claim=parent.parent.parent/'.motion_sensor_schema_recovery_claims'/(sha(parent)+'.json')
   claim.parent.mkdir(exist_ok=True)
   with claim.open('x') as stream:json.dump({'destination':name,'spec':str(folder/'spec.json')},stream)
  elif s['motion_training'].get('wiring_reuse'):
   parent=Path(s['motion_training']['wiring_reuse']['parent_spec'])
   claim=parent.parent.parent/'.motion_timing_resume_claims'/(sha(parent)+'.json')
   claim.parent.mkdir(exist_ok=True)
   with claim.open('x') as stream:json.dump({'destination':name,'spec':str(folder/'spec.json')},stream)
  if a.authorize:
   from native_stand.command_v6.tools.register_authorization import register
   request={'approval_id':'user_motion4750_timing10000_20261021_'+name,'user_scope':s['authorization']['user_scope'],
    'session':session,'physical_gpu':s['device']['physical_index'],'gpu_uuid':s['device']['uuid'],
    'max_online_updates':smoke+formal,'max_phase_wall_s':None,'arms':[{'spec':str(folder/'spec.json'),
     'spec_sha256':sha(folder/'spec.json'),'budget_sha256':sha(folder/'budget.json')}]}
   result['registration']=register(request,current_session=session,write=True)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
