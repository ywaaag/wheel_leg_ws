import copy,json
import numpy as np
import pytest
from native_stand.students import motion_distilled_native as n
from native_stand.students.data import sha

def test_swing_uses_controlled_first_episode_and_strict_limit():
 a={'first_episode_valid':np.ones((3000,1),bool),'controller':np.full((3000,1,1),3),'state':np.zeros((3000,1,39)),'geometry':np.zeros((3000,1,13))}
 a['state'][...,29]=1;a['geometry'][...,1]=-1;a['geometry'][...,4]=-1
 r=n.swing_report(a,[[0,0,0],[0,0,0]])
 assert r['full_passed'] and r['hold_passed'] and r['full_max_rad']==0
 a['geometry'][0,0,2]=-.1
 r=n.swing_report(a,[[0,0,0],[0,0,0]])
 assert not r['full_passed'] and r['hold_passed']
 a['first_episode_valid'][2000:]=False
 assert not n.swing_report(a,[[0,0,0],[0,0,0]])['full_passed']

def test_native_request_identity_and_budget_rejections(tmp_path,monkeypatch):
 from native_stand.students import model
 cp=tmp_path/'student.pt';cp.write_bytes(b'fixture')
 s=tmp_path/'spec.json';s.write_text(json.dumps({'device':{'physical_index':0},'student_evaluation':{'cases':[{'command':c} for c in n.COMMANDS]}}))
 monkeypatch.setenv('HAPI_SESSION_ID','fixture-session')
 monkeypatch.setattr(model,'load',lambda *a:(model.Student(True,version='multiskill_mcu_v2'),{}))
 monkeypatch.setattr(n,'validate_student',lambda *a:{})
 r=dict(version=n.VERSION,authorization='AUTHORIZED',online_PPO=0,physical_envs=1,scored_envs=1,wall_limit_s=1800,parent_spec=str(s),parent_spec_sha256=sha(s),session='fixture-session',physical_gpu=0,checkpoints=[{'checkpoint':str(cp),'sha256':sha(cp)}],output=str(tmp_path/'2026_10_18-22_00__fixture'),commands=copy.deepcopy(n.COMMANDS),output_max_bytes=256*2**20)
 before=set(tmp_path.rglob('*'));n.validate(r);assert set(tmp_path.rglob('*'))==before
 for key,value in [('commands',n.COMMANDS[:-1]),('online_PPO',1),('session','wrong'),('wall_limit_s',1801),('physical_envs',8),('parent_spec_sha256','x'*64)]:
  bad=copy.deepcopy(r);bad[key]=value
  with pytest.raises(ValueError):n.validate(bad)
 bad=copy.deepcopy(r);bad['checkpoints'][0]['sha256']='x'*64
 with pytest.raises(ValueError):n.validate(bad)
 from native_stand.students.motion_single_screen import validate
 validate(r)

def test_completion_exact_three_commands_null_wall_and_frozen_prior(tmp_path,monkeypatch):
 from native_stand.students import model
 cp=tmp_path/'student.pt';cp.write_bytes(b'fixture')
 spec=tmp_path/'spec.json';spec.write_text(json.dumps({'device':{'physical_index':0},'student_evaluation':{'cases':[{'command':c} for c in n.COMMANDS]}}))
 monkeypatch.setenv('HAPI_SESSION_ID','fixture-session')
 monkeypatch.setattr(model,'load',lambda *a:(model.Student(True,version='multiskill_mcu_v2'),{}));monkeypatch.setattr(n,'validate_student',lambda *a:{})
 parent=tmp_path/'old';parent.mkdir();checkpoints=[{'checkpoint':str(cp),'sha256':sha(cp)}]
 req={'checkpoints':checkpoints,'parent_spec_sha256':sha(spec)}
 (parent/'request.json').write_text(json.dumps(req));(parent/'terminal_report.json').write_text(json.dumps({'status':'FAILED'}))
 (parent/'worker_failure.json').write_text(json.dumps({'exception':"TimeoutError('Student evaluation wall budget exhausted')",'partial_results':[{}]*4}))
 for i,c in enumerate(n.COMMANDS[:4]):
  tp=parent/f'trace_{i:02}.npz';np.savez_compressed(tp,state=np.zeros((1,1,39)))
  (parent/f'evaluation_{i:02}.json').write_text(json.dumps({'case':{'command':c},'checkpoint_sha256':sha(cp),'model_unchanged':True,'per_env':[{'controlled_seconds':30.}],'trace':{'path':tp.name,'sha256':sha(tp)}}))
 inputs={str(p):sha(p) for p in parent.iterdir()}
 r=dict(version=n.COMPLETION,authorization='AUTHORIZED',online_PPO=0,physical_envs=1,scored_envs=1,wall_limit_s=None,parent_spec=str(spec),parent_spec_sha256=sha(spec),session='fixture-session',physical_gpu=0,checkpoints=checkpoints,output=str(tmp_path/'2026_10_19-00_00__fixture'),commands=copy.deepcopy(n.COMMANDS[4:]),output_max_bytes=256*2**20,completion={'parent_run':str(parent),'inputs':inputs})
 before=set(tmp_path.rglob('*'));n.validate(r);assert set(tmp_path.rglob('*'))==before
 from native_stand.students.motion_single_screen import validate
 validate(r)
 for key,value in [('commands',n.COMMANDS),('wall_limit_s',1800),('online_PPO',1)]:
  bad=copy.deepcopy(r);bad[key]=value
  with pytest.raises(ValueError):n.validate(bad)
 (parent/'evaluation_00.json').write_text('{}')
 with pytest.raises(ValueError,match='changed'):n.validate(r)
