FROZEN_TEST_SCOPE='PORTABLE'
import copy
import pytest
from native_stand.students import motion_yaw_course as f
from native_stand.students.motion_course import implementation

def test_yaw_slots_whitelist_and_forward_retention():
 for stage in range(3):
  c=f.stage_config(stage);assert sum(c['slot_counts'])==512 and len(c['sampling_slots'])==512
  assert c['slot_counts'][:5]==[128,48,48,48,48]
  assert sum(c['slot_counts'][5:])==192
  assert c['commands'][:5]==[[0.,0.],[.5,0.],[1.,0.],[1.5,0.],[2.,0.]]
  for j,v in enumerate(f.TARGETS[:stage+1]):
   assert c['commands'][5+2*j:7+2*j]==[[0.,v],[0.,-v]]
   assert c['slot_counts'][5+2*j]==c['slot_counts'][6+2*j]
  assert max(abs(yaw) for vx,yaw in c['commands'])==f.TARGETS[stage]
  assert implementation(f.VERSION) is f
 with pytest.raises(ValueError):implementation('wrong')

def candidate(stage,update):
 return {'stage':stage,'update':update,'checkpoint':f'/fake/{update}.pt','sha256':str(update),'complete_matrix':True,
  'evaluations':[{'case':{'command':v+[.34]},'path':'raw.json','sha256':'a'*64,'per_env':[{'env':i,'accepted':True} for i in range(8)]} for v in f.commands(stage)]}

def test_two_complete_checks_advance_and_never_final_early():
 s={'stage':0,'stage_started_update':0,'last_evaluation_update':-1,'consecutive_passes':0,'accepted_by_stage':{}}
 s,d=f.advance(s,candidate(0,250));assert not d['advanced']
 s,d=f.advance(s,candidate(0,500));assert d['advanced'] and s['stage']==1 and not d['all_stages_passed']
 c=candidate(1,750);c['evaluations'][-1]['per_env'][0]['accepted']=False
 s,d=f.advance(s,c);assert not d['complete_matrix_passed'] and s['consecutive_passes']==0
 s.update(stage=2,consecutive_passes=1,last_evaluation_update=1250)
 s,d=f.advance(s,candidate(2,1500));assert d['all_stages_passed'] and not d['advanced']

def test_parking_diagnostic_cannot_hide_other_failure():
 original={'version':'old','checks':{'parking':False,'contact':True,'tracking':True},'accepted':False,'failed_checks':['parking']}
 r=f.assess(original);assert r['accepted'] and r['diagnostic_checks']['parking'] is False and original['accepted'] is False
 original['checks']['tracking']=False
 assert not f.assess(original)['accepted']

def test_recipe_explicit_forward_only_old_unchanged():
 from native_stand.students.motion_training import declaration
 from native_stand.students.training_recipe import validate_instance
 from native_stand.students.motion_startup import smoke_profile
 from native_stand.students.motion_tracking_reward import QUALITY_VERSION
 r=declaration(reward_version=QUALITY_VERSION);old=copy.deepcopy(r)
 r.update(course_version=f.VERSION,course_stage=0,smoke_wiring=smoke_profile())
 assert validate_instance(r,r['source_actor_sha256'])==r
 assert validate_instance(old,old['source_actor_sha256'])==old
 r['course_stage']=3;r['stage']=3
 with pytest.raises(ValueError):validate_instance(r,r['source_actor_sha256'])


def test_yaw_reward_has_nonzero_sign_correct_gradient_at_zero_translation():
 import torch
 from test_motion31_training import inputs
 from native_stand.students.motion_tracking_reward import reward,configuration,QUALITY_VERSION
 for target in (-.5,.5,-1.,1.):
  data=inputs(0.,0.);data['command']=data['command'].clone();data['target']=data['target'].clone()
  data['command'][:,1]=target;data['target'][:,1]=target
  data['yaw']=torch.zeros(3,requires_grad=True)
  result=reward(**data,config=configuration(QUALITY_VERSION))
  result['total'].sum().backward()
  assert torch.isfinite(data['yaw'].grad).all() and (data['yaw'].grad*target>0).all()
  idx=result['fields'].index('yaw_tracking')
  assert (result['density'][:,idx]>0).all()


def test_source_protocol_allowlist_is_explicit_not_generic():
 from native_stand.students.motion_startup import source_protocols
 from native_stand.students import motion_forward_course
 from native_stand.students.motion_guard import protocol
 assert source_protocols({'student_training':{'course_version':f.VERSION}})==[protocol(),motion_forward_course.protocol()]
 assert source_protocols({'student_training':{'course_version':motion_forward_course.VERSION}})==[protocol()]


def test_gpu_statistics_distinguish_signed_yaw_error_direction_and_drift():
 import torch
 from native_stand.students.motion_statistics import MotionStatistics
 stats=MotionStatistics([[0.,.5],[0.,-.5]],'cpu',schedule_s=[.5,1.,25.5,1.],parking_s=2.)
 s=torch.zeros(2,39);s[:,7]=.1;s[:,9]=torch.tensor([.4,-.3]);s[:,6]=.34;s[:,10:12]=10
 p=torch.zeros(2,10);g=torch.zeros(2,13);g[:,7]=-1
 stats.observe(torch.tensor([0,1]),torch.ones(2,dtype=torch.bool),torch.ones(2)*10,
  torch.tensor([[0.,.5,.34],[0.,-.5,.34]]),s,p,g)
 r=stats.export()['hold']
 for i,value in enumerate((.4,-.3)):
  row=r[f'cohort_{i}'];assert row['mean']['yaw_signed_rad_s']==pytest.approx(value)
  assert row['mean']['yaw_mae_rad_s']==pytest.approx((.1,.2)[i])
  assert row['yaw_direction_rate']==1 and row['vx_direction_rate'] is None
  assert row['mean']['horizontal_speed_mps']==pytest.approx(.1)


def test_actual_yaw_curves_saved_and_nonflat_with_huber_tail():
 from native_stand.students.motion_reward_audit import audit
 from native_stand.students.motion_tracking_reward import configuration,QUALITY_VERSION
 rows=audit(configuration(QUALITY_VERSION))['curves']
 for name in ('yaw_active','yaw_zero'):
  curves=[r for r in rows if r['signal']==name]
  assert len(curves)==13 and all(r['unit']=='rad/s' for r in curves)
  assert all(r['state_error_derivative']<0 for r in curves if r['error']>0)


def test_candidate_keeps_source_over_new_that_loses_forward_capability():
 def c(update):
  r=candidate(0,update)
  for row in r['evaluations']:
   for e in row['per_env']:e.update(survival_passed=True,tracking={'MAE':.1})
  r['per_env']=[e for row in r['evaluations'] for e in row['per_env']]
  return r
 source=c(0)
 for row in source['evaluations'][5:]:
  for e in row['per_env']:e['accepted']=False
 new=c(250)
 for e in new['evaluations'][1]['per_env']:e['accepted']=False
 selection=f.choose([source,new],{})
 assert selection['best_new'] is new and selection['best_overall'] is source
 assert selection['accepted_candidate'] is None
