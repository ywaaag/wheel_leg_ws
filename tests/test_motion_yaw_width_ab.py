FROZEN_TEST_SCOPE='PORTABLE'
import copy
import pytest
import torch
from native_stand.students import motion_yaw_width_course as course
from native_stand.students.motion_tracking_reward import configuration,QUALITY_VERSION,YAW_NARROW_VERSION,YAW_NARROW_BASE,reward

def test_only_yaw_kernel_changes_and_old_defaults_preserved():
 a=configuration(QUALITY_VERSION);b=configuration(YAW_NARROW_VERSION)
 assert {k for k in a if a[k]!=b[k]}=={'version','yaw_denominator'}
 assert a['yaw_denominator']==1 and b['yaw_denominator']==.25
 base=configuration(YAW_NARROW_BASE)
 assert base['yaw_denominator']==b['yaw_denominator'] and base['refinement']==b['refinement']

def test_reward_actual_zero_and_linear_unchanged_yaw_has_correct_gradient():
 from test_motion31_training import inputs
 for target in (0.,-.5,.5):
  outputs=[]
  for version in (QUALITY_VERSION,YAW_NARROW_VERSION):
   data=inputs(0.,0.);data['command']=data['command'].clone();data['target']=data['target'].clone();data['command'][:,1]=target;data['target'][:,1]=target
   data['yaw']=torch.zeros(3,requires_grad=True)
   result=reward(**data,config=configuration(version));result['total'].sum().backward()
   assert torch.isfinite(data['yaw'].grad).all()
   if target:assert (target*data['yaw'].grad>0).all()
   outputs.append(result)
  for i,field in enumerate(outputs[0]['fields']):
   if field!='yaw_tracking' or not target:assert torch.equal(outputs[0]['dt'][:,i],outputs[1]['dt'][:,i])
  if target:assert (outputs[1]['raw'][:,2]<outputs[0]['raw'][:,2]).all()

def test_course_stays_fixed_even_after_repeated_complete_passes():
 from test_motion_yaw_course import candidate
 state={'stage':0,'stage_started_update':0,'last_evaluation_update':-1,'consecutive_passes':0,'accepted_by_stage':{}}
 for update in (250,500,750,1000):
  state,decision=course.advance(state,candidate(0,update))
  assert state['stage']==0 and state['stage_started_update']==0
  assert not decision['advanced'] and not decision['all_stages_passed']
 assert state['accepted_by_stage']['0']['update']==1000
 with pytest.raises(ValueError):course.commands(1)
 assert course.stage_config(0)['slot_counts']==[128,48,48,48,48,96,96]

def test_narrow_wrapper_uses_explicit_narrow_base():
 from native_stand.students.motion_tracking_reward import quality_base_configuration,REFINEMENT_VERSION,SUCCESSOR_VERSION
 assert quality_base_configuration(QUALITY_VERSION)==configuration(REFINEMENT_VERSION)
 assert quality_base_configuration(SUCCESSOR_VERSION)==configuration(REFINEMENT_VERSION)
 assert quality_base_configuration(YAW_NARROW_VERSION)==configuration(YAW_NARROW_BASE)
 with pytest.raises(ValueError):quality_base_configuration('unknown')

def test_queued_arm_failure_never_launches_and_never_retries(tmp_path,monkeypatch):
 import json
 from native_stand.command_v6.submission_queue import launch_successor
 (tmp_path/'manifest.json').write_text(json.dumps({'authorized_next_arm':{'spec':'not-used'}}))
 monkeypatch.setattr('native_stand.command_v6.submission_queue.subprocess.run',lambda *a,**k:pytest.fail('must not submit after infrastructure failure'))
 launch_successor(tmp_path,'FAILED','broken runtime')
 event=json.loads((tmp_path/'next_arm_submission.json').read_text())
 assert event['status']=='NOT_SUBMITTED' and 'runtime' in event['reason']
 with pytest.raises(FileExistsError):launch_successor(tmp_path,'FAILED','broken runtime')

def test_queue_accepts_only_same_source_same_budget_single_variable(tmp_path,monkeypatch):
 import json
 from native_stand.command_v6.submission_queue import validate
 from native_stand.command_v6.completion import sha
 monkeypatch.setattr('native_stand.students.motion_yaw_width_requests.verify',lambda s:True)
 paths=[]
 for label,den in [('a',1.),('b',.25)]:
  folder=tmp_path/label;folder.mkdir();p=folder/'spec.json';paths.append(p)
  s={'reward':{'version':label,'yaw_denominator':den},'experiment':label,'source_sha256':'a'*64,'authorization':{'hapi_session':'current','approval_id':'ab'}}
  p.write_text(json.dumps(s));(folder/'budget.json').write_text(json.dumps({'authorization':'AUTHORIZED','limits':{'smoke':5,'formal':1000}}))
  (folder/'authorization_receipt.json').write_text(json.dumps({'spec_sha256':sha(p)}))
 q=validate(*paths,tmp_path/'new_run','current');assert q['retries']==0 and q['spec_sha256']==sha(paths[1])
 s=json.loads(paths[1].read_text());s['source_sha256']='b'*64;paths[1].write_text(json.dumps(s))
 with pytest.raises(ValueError,match='additional variable'):validate(*paths,tmp_path/'new_run','current')
