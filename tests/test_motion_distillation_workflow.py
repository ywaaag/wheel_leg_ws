"""Workflow orchestration fixtures: no simulator or optimizer invocation."""
import copy,json
from pathlib import Path
import numpy as np
import pytest,torch
from native_stand.students import motion_distillation_workflow as w
from native_stand.students.data import sha

def put(path,data):path.write_text(json.dumps(data));return path

def fake_sim(folder,checkpoint_sha,response_offset=0.):
 folder.mkdir();rows=[]
 for c in w.p.COMMANDS:
  rows.append({'target':c[:2],'continuous_s':30.,'termination':None,'vx_mean':c[0]+response_offset,'yaw_mean':c[1]+response_offset,'pitch_abs_max':.01,'roll_abs_max':.01,'passed':True,'supplemental_swing_constraint':{'full_passed':True}})
 put(folder/'manifest.json',{'checkpoint_sha256':checkpoint_sha,'protocol':{'hold':[10,18]},'input_sha256':{'reference.xml':'fixed-physical-SHA'}})
 put(folder/'terminal_report.json',{'status':'COMPLETED','model_unchanged':True,'results':rows})
 return folder

@pytest.fixture
def req(tmp_path,monkeypatch):
 teacher=tmp_path/'teacher.pt';teacher.write_bytes(b'teacher')
 spec=tmp_path/'spec.json';spec.write_text('{}')
 student=tmp_path/'student.pt';torch.save({'distillation':{'teacher_sha256':sha(teacher)}},student)
 dataset=tmp_path/'dataset.pt';torch.save({'fixture':True},dataset)
 put(dataset.with_name('manifest.json'),{'dataset_sha256':sha(dataset),'teacher_sha256':sha(teacher),'parent_spec_sha256':sha(spec)})
 monkeypatch.setattr(w.p,'validate_teacher',lambda *a:(None,None,None));monkeypatch.setattr(w.p,'validate_student',lambda *a:{})
 t=fake_sim(tmp_path/'teacher_sim',sha(teacher));s=fake_sim(tmp_path/'student_sim',sha(student),.1)
 return {'version':w.VERSION,'authorization':'UNAUTHORIZED','teacher':str(teacher),'parent_spec':str(spec),'dataset':str(dataset),'initial_student':str(student),'initial_student_mujoco':str(s),'teacher_mujoco':str(t),'prior_replays':[],'output':str(tmp_path/'2026_10_19-12_00__fixture_cycle'),'limits':{'max_rounds':2,'max_offline_updates':10,'updates_per_round':5,'fit_wall_s':10,'sim_wall_s':10,'total_wall_s':100,'storage_max_bytes':2**20},'criteria':{'linear_gap_mps':.05,'yaw_gap_rad_s':.05,'posture_excess_rad':.01,'minimum_score_improvement':.01,'per_command_regression_margin':.1}}

def test_plan_zero_write_and_fidelity_per_command(req,tmp_path):
 before=set(tmp_path.rglob('*'));p=w.validate(req)
 assert set(tmp_path.rglob('*'))==before and p['files_written']==0 and not p['initial_fidelity']['fidelity_passed']
 bad=copy.deepcopy(req);bad['limits']['max_rounds']=6
 with pytest.raises(ValueError):w.validate(bad)
 with pytest.raises(ValueError):w.run(req,'old permission cannot authorize')
 assert not Path(req['output']).exists()

def test_assets_protocol_and_matrix_conflict_rejected(req):
 path=Path(req['initial_student_mujoco'])/'manifest.json';d=json.loads(path.read_text());d['protocol']={'hold':[2,8]};put(path,d)
 with pytest.raises(ValueError,match='protocol'):w.validate(req)

@pytest.mark.parametrize('field',['solver_profile','plant_profile','actuator_configuration'])
def test_same_xml_cannot_hide_different_effective_simulator_settings(req,field):
 path=Path(req['initial_student_mujoco'])/'manifest.json'
 d=json.loads(path.read_text());d[field]={'version':'different_effective_settings'};put(path,d)
 with pytest.raises(ValueError,match=field):w.validate(req)

def test_source_teacher_incomplete_rejected(req):
 path=Path(req['teacher_mujoco'])/'terminal_report.json';d=json.loads(path.read_text());d['results'][0]['continuous_s']=2.;put(path,d)
 with pytest.raises(ValueError,match='Teacher baseline incomplete'):w.validate(req)

def test_aggregate_preserves_prior_states_and_rejects_teacher_or_validation(tmp_path):
 paths=[]
 for i in range(2):
  folder=tmp_path/str(i);folder.mkdir();path=folder/'replay.pt'
  torch.save({'x':torch.ones(3,31)*i,'y':torch.ones(3,6),'cohort':torch.zeros(3,dtype=torch.long),'phase':torch.zeros(3,dtype=torch.long),'split':torch.zeros(3,dtype=torch.long)},path)
  put(folder/'replay_manifest.json',{'version':w.p.VERSION,'teacher_sha256':'teacher','replay_sha256':sha(path)});paths.append(path)
 rp=w.aggregate_replays(paths,tmp_path/'aggregate','teacher');d=torch.load(rp,weights_only=False)
 assert len(d['x'])==6 and d['x'][:3].sum()==0 and d['x'][3:].sum()==93
 with pytest.raises(ValueError,match='Teacher'):w.aggregate_replays(paths,tmp_path/'bad','different')
 with pytest.raises(FileExistsError):w.aggregate_replays(paths,tmp_path/'aggregate','teacher')

def test_converged_input_runs_zero_new_rounds(req,monkeypatch):
 req['authorization']='AUTHORIZED'
 p=Path(req['initial_student_mujoco'])/'terminal_report.json';d=json.loads(p.read_text())
 for row,c in zip(d['results'],w.p.COMMANDS):row.update(vx_mean=c[0],yaw_mean=c[1])
 put(p,d)
 monkeypatch.setattr(w.p,'fit',lambda *a,**k:pytest.fail('already sufficient candidate must not optimize'))
 result=w.run(req,'fixture explicit scope');assert result['status']=='NATIVE_CONFIRMATION_REQUIRED' and result['offline_updates_used']==0 and not result['native_certified']
 with pytest.raises(FileExistsError):w.run(req,'duplicate submission')

def mock_round(req,monkeypatch,*,improve):
 req['authorization']='AUTHORIZED'
 monkeypatch.setattr(w.p,'collect_replay',lambda *a:None)
 monkeypatch.setattr(w,'aggregate_replays',lambda *a:Path('fixture-replay'))
 def fit(dataset,out,**kwargs):
  out.mkdir();(out/'best_validation.pt').write_bytes(b'candidate');put(out/'budget.json',{'used':kwargs['steps']});return {'offline_optimizer_updates':kwargs['steps']}
 monkeypatch.setattr(w.p,'fit',fit)
 monkeypatch.setattr(w.p,'mujoco_command',lambda cp,parent,out,wall:[str(cp),str(out)])
 def child(cmd,**kwargs):
  from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
  validate_new_experiment_name(Path(cmd[1]).name)
  fake_sim(Path(cmd[1]),sha(cmd[0]),0. if improve else .2)
  return type('Child',(),{'returncode':0})()
 monkeypatch.setattr(w.subprocess,'run',child)

def test_accepted_round_accounts_budget_and_requires_native(req,monkeypatch):
 mock_round(req,monkeypatch,improve=True);r=w.run(req,'fixture authorized')
 assert r['offline_updates_used']==5 and len(r['history'])==1 and r['status']=='NATIVE_CONFIRMATION_REQUIRED'
 assert r['retained_candidate']!=req['initial_student'] and not r['deployment_certified']

def test_regression_retains_previous_and_stops_without_extra_round(req,monkeypatch):
 mock_round(req,monkeypatch,improve=False);r=w.run(req,'fixture authorized')
 assert r['status']=='NO_IMPROVEMENT_RETAIN_PREVIOUS' and r['retained_candidate']==req['initial_student'] and len(r['history'])==1

def test_optimizer_failure_still_counts_used_updates_no_retry(req,monkeypatch):
 mock_round(req,monkeypatch,improve=True)
 def fail(dataset,out,**kwargs):
  out.mkdir();put(out/'budget.json',{'used':2});raise RuntimeError('fixture optimizer failed after2')
 monkeypatch.setattr(w.p,'fit',fail);r=w.run(req,'fixture authorized')
 assert r['status']=='FAILED' and r['offline_updates_used']==2 and r['retained_candidate']==req['initial_student'] and not r['history']


def test_early_student_fall_is_bootstrap_evidence_not_fabricated_tracking(req,monkeypatch):
 path=Path(req['initial_student_mujoco'])/'terminal_report.json';d=json.loads(path.read_text())
 for row in d['results']:
  row.update(continuous_s=1.77,termination='UPRIGHT',passed=False)
  for key in ('vx_mean','yaw_mean','pitch_abs_max','roll_abs_max','supplemental_swing_constraint'):row.pop(key)
 put(path,d)
 result=w.validate(req)['initial_fidelity']
 assert result['score'] is None and not result['all_survived']
 assert all(row['student_response'] is None and not row['fidelity_passed'] for row in result['rows'])
 mock_round(req,monkeypatch,improve=True);result=w.run(req,'fixture authorized')
 assert result['status']=='NATIVE_CONFIRMATION_REQUIRED' and result['retained_candidate']!=req['initial_student']

def test_complete_student_missing_metric_still_rejected(req):
 path=Path(req['initial_student_mujoco'])/'terminal_report.json';d=json.loads(path.read_text());d['results'][0].pop('vx_mean');put(path,d)
 with pytest.raises(ValueError,match='comparison metric'):w.validate(req)
