FROZEN_TEST_SCOPE='PORTABLE'
import copy,json
from types import SimpleNamespace
import pytest
from native_stand.students.motion_quick_evaluation import configuration,validate,due,run
from native_stand.students.grouped_evaluation import plan
from test_motion_performance_v2 import cases

def spec():
 return {'student_training':{'version':'student_motion31_training_v1','checkpoint_every_updates':100},'evaluation':{'full_every_updates':250},'student_evaluation':{'cases':cases()},'checkpoint_quick_evaluation':configuration(),'execution_source_root':'/fake/src'}

def test_explicit_default_off_and_schedule_independent_from_full():
 assert validate({}) is None and not due({},'formal',100)
 s=spec();assert validate(s)==configuration()
 assert [i for i in range(1,1001) if due(s,'formal',i)]==list(range(100,1001,100))
 assert not due(s,'smoke',100) and not due(s,'formal',250)
 for key,value in [('selection_eligible',True),('scored_envs_per_command',1),('every_updates',10)]:
  bad=copy.deepcopy(s);bad['checkpoint_quick_evaluation'][key]=value
  with pytest.raises(ValueError):validate(bad)
 assert len(plan(cases(),maximum_commands=7))==1
 assert list(map(len,plan(cases())))==[4,3]
 assert [r['case_id'] for r in plan(cases(),maximum_commands=7)[0]]==list(map(str,range(7)))
 with pytest.raises(ValueError):plan(cases(),maximum_commands=True)
 with pytest.raises(ValueError):plan(cases(),maximum_commands=9)

def test_trace_memory_is_explicit_not_silently_overcommitted():
 from native_stand.students.performance import settings
 s=spec();s['performance']={'version':'student_performance_v2','evaluation_group_commands':True,'evaluation_group_max_commands':7,'evaluation_gpu_buffer':True,'evaluation_reuse_environment':True,'trace_buffer_max_bytes':512*2**20}
 assert settings(s)['evaluation_group_max_commands']==7
 s['performance']['trace_buffer_max_bytes']=256*2**20
 with pytest.raises(ValueError,match='conservative peak'):settings(s)

def test_owner_zero_remaining_does_not_launch_or_poll(tmp_path,monkeypatch):
 import time
 p=tmp_path/'spec.json';p.write_text('{}');c=tmp_path/'model_update_100.pt';c.write_bytes(b'checkpoint')
 monkeypatch.setattr('native_stand.students.motion_quick_evaluation.subprocess.run',lambda *a,**k:pytest.fail('must not launch'))
 r=run(tmp_path,spec(),c,update=100,parent_spec=p,attempt_id='id',deadline=time.time())
 assert r['status']=='NOT_RUN' and r['accepted_candidate'] is None and not r['selection_eligible']
 with pytest.raises(FileExistsError):run(tmp_path,spec(),c,update=100,parent_spec=p,attempt_id='id',deadline=time.time())

def test_owned_subprocess_result_has_no_candidate_authority(tmp_path,monkeypatch):
 import time,torch
 p=tmp_path/'spec.json';p.write_text('{}');c=tmp_path/'model_update_100.pt';c.write_bytes(b'checkpoint');calls=[]
 def child(command,**kwargs):calls.append((command,kwargs));return SimpleNamespace(returncode=0)
 monkeypatch.setattr('native_stand.students.motion_quick_evaluation.subprocess.run',child)
 rng=torch.get_rng_state().clone();r=run(tmp_path,spec(),c,update=100,parent_spec=p,attempt_id='id',deadline=time.time()+1000)
 assert torch.equal(rng,torch.get_rng_state()) and len(calls)==1
 assert '--request' in calls[0][0] and 'timeout'==calls[0][0][0]
 assert r['status']=='FAILED' and r['accepted_candidate'] is None
 assert 'without complete summary' in r['reason']


def test_zero_exit_complete_summary_is_hash_bound(tmp_path,monkeypatch):
 import time
 from native_stand.students.data import sha
 p=tmp_path/'spec.json';p.write_text('{}');c=tmp_path/'model_update_100.pt';c.write_bytes(b'checkpoint')
 def child(command,**kwargs):
  request=json.loads((tmp_path/'quick_update_100/request.json').read_text())
  (tmp_path/'quick_update_100/summary.json').write_text(json.dumps({'status':'COMPLETED','checkpoint_sha256':request['checkpoint_sha256'],'update':100,'commands':[{'per_env':[{}]*8}]*7}))
  return SimpleNamespace(returncode=0)
 monkeypatch.setattr('native_stand.students.motion_quick_evaluation.subprocess.run',child)
 r=run(tmp_path,spec(),c,update=100,parent_spec=p,attempt_id='id',deadline=time.time()+1000)
 assert r['status']=='COMPLETED' and r['summary_sha256']==sha(tmp_path/'quick_update_100/summary.json')
 assert r['accepted_candidate'] is None


def test_seven_group_collection_dispatches_once_and_restores_per_command_schema(monkeypatch):
 import numpy as np
 from native_stand.students import evaluator
 from native_stand.students.grouped_evaluation import CommandGroups
 rows=cases();calls=[]
 arrays={'state':np.zeros((3,56,39)),'first_episode_valid':np.ones((3,56),dtype=bool),'policy_time_s':np.arange(3)*.01}
 arrays['state'][:,:,0]=np.arange(56)
 def collect(env,actor,row,**kwargs):
  calls.append(kwargs);assert len(row['_group_cases'])==7 and kwargs['scored_envs']==56
  return arrays
 def score(env,case,part,**kwargs):
  assert kwargs['scored_envs']==8
  return {'per_env':[{'env':i,'accepted':False} for i in range(8)]},part
 monkeypatch.setattr(evaluator,'rollout',collect);monkeypatch.setattr(evaluator,'score_arrays',score)
 groups=CommandGroups(rows,maximum_commands=7);env=SimpleNamespace(num_envs=64)
 for index,row in enumerate(rows):
  payload,part=groups.acquire(env,None,row)
  assert len(payload['per_env'])==8
  assert list(part['state'][0,:,0])==list(range(index*8,index*8+8))
  assert payload['execution_layout']['command_group_size']==7
  assert payload['execution_layout']['all_simulated_env_steps_attributed']==(192 if index==0 else 0)
 assert len(calls)==1 and not groups.pending
