"""Warm evaluator boundaries without Isaac; no simulated samples in these fixtures."""
FROZEN_TEST_SCOPE='PORTABLE'
import json,random
from pathlib import Path
import numpy as np
import pytest,torch
from native_stand.students import motion_evaluation_review as r
from native_stand.students.motion_quick_evaluation import warm_configuration,configuration,validate
from native_stand.students.data import sha


def cases():
 return [{'case_id':str(i),'kind':'student','family':'motion_native_normal','role':'MotionZeroStudent','seconds':31.52,'command':[x,y,.34]} for i,(x,y) in enumerate([(0,0),(.5,0),(1,0),(2,0),(-.5,0),(-1,0),(0,.5),(0,-.5),(0,1),(0,-1)])]


def spec():return {'student_training':{'version':'student_motion31_training_v1','checkpoint_every_updates':100},'evaluation':{'full_every_updates':250},'student_evaluation':{'cases':cases()},'checkpoint_quick_evaluation':warm_configuration(),'performance':{'version':'student_performance_v2'}}


def test_new_explicit_version_and_only_declared_cases():
 s=spec();assert validate(s)==warm_configuration()
 selected=r.representative_cases(cases())
 assert [z['command'][:2] for z in selected]==[[0,0],[2,0],[-1,0],[0,1],[0,-1],[.5,0]]
 assert all(z['seconds']==31.52 for z in selected)
 assert configuration()['version']=='motion31_checkpoint_quick_evaluation_v1'
 s['checkpoint_quick_evaluation']['selection_eligible']=True
 with pytest.raises(ValueError):validate(s)


def records(folder,selected,cp):
 output=[]
 for i,row in enumerate(selected):
  p=folder/f'raw_{i}.json';p.write_text(json.dumps({'identity':'complete fixture'}))
  output.append({'case':dict(row,sha256=sha(cp)),'path':p.name,'sha256':sha(p),'per_env':[{'accepted':False,'failed_checks':['vx']} ]})
 return output


@pytest.mark.parametrize('fail',[False,True])
def test_warm_boundary_restores_all_rng_modes_and_passes_single_env_no_short_window(tmp_path,fail):
 a=torch.nn.Linear(31,6);cp=tmp_path/'checkpoint.pt';cp.write_bytes(b'exact')
 parent=tmp_path/'source_snapshot/spec.json';parent.parent.mkdir();parent.write_text(json.dumps(spec()))
 state=torch.get_rng_state().clone();python=random.getstate();numpy=np.random.get_state();weights={k:v.clone() for k,v in a.state_dict().items()}
 def evaluate(s,folder,actor,selected,**kwargs):
  assert kwargs['screen_single_env'] and len(selected)==6
  assert all(z['seconds']==31.52 for z in selected)
  # Real evaluate_cases reads this exact dependency when serializing each case.
  assert sha(folder/'source_snapshot/spec.json')==sha(parent)
  layout=json.loads((folder/'effective_layout.json').read_text())
  assert layout['parent_spec_sha256']==sha(parent) and layout['physical_envs']==1
  assert json.loads((folder/'source_snapshot/spec.json').read_text())==spec()
  actor.eval();torch.rand(5);random.random();np.random.rand()
  if fail:raise RuntimeError('fixture physics fault')
  return records(folder,selected,cp)
 if fail:
  with pytest.raises(RuntimeError,match='fixture physics fault'):r.warm_screen(spec(),tmp_path,a,cp,100,evaluate=evaluate)
  assert (tmp_path/'quick_update_100/worker_failure.json').exists()
 else:
  s=r.warm_screen(spec(),tmp_path,a,cp,100,evaluate=evaluate)
  assert s['new_scored_episodes']==6 and not s['selection_eligible'] and s['accepted_candidate'] is None
 assert a.training and torch.equal(torch.get_rng_state(),state) and random.getstate()==python
 np.testing.assert_array_equal(np.random.get_state()[1],numpy[1])
 for k,v in weights.items():torch.testing.assert_close(a.state_dict()[k],v)


def test_warm_missing_parent_identity_fails_before_any_evaluation(tmp_path):
 a=torch.nn.Linear(31,6);cp=tmp_path/'cp.pt';cp.write_bytes(b'exact')
 with pytest.raises(FileNotFoundError):
  r.warm_screen(spec(),tmp_path,a,cp,100,evaluate=lambda *a,**kw:pytest.fail('must fail before simulation'))
 assert (tmp_path/'quick_update_100/worker_failure.json').exists()


def test_full_evaluation_reused_without_any_new_rollout(tmp_path):
 a=torch.nn.Linear(31,6);cp=tmp_path/'cp.pt';cp.write_bytes(b'exact');rs=records(tmp_path,cases(),cp)
 s=r.warm_screen(spec(),tmp_path,a,cp,500,records=rs,evaluate=lambda *a,**k:pytest.fail('must not sample'))
 assert s['new_scored_episodes']==0 and len(s['commands'])==6
 assert s['execution'].startswith('derived')
 rs[0]['case']['sha256']='wrong'
 with pytest.raises(ValueError,match='checkpoint mismatch'):r.warm_screen(spec(),tmp_path,a,cp,1000,records=rs)


def test_report_weighted_trends_do_not_mix_curriculum_stages_or_claim_acceptance(tmp_path):
 rows=[]
 for i,stage,value,n in [(1,0,10,1),(2,1,.5,10),(3,1,.3,30)]:
  rows.append({'update':i,'course':{'stage':stage},'cohort_statistics':{'hold':{'c':{'target_vx':1.,'target_yaw':0.,'samples':n,'mean':{'vx_mae_mps':value,'vx_mps':1-value}}}}})
 (tmp_path/'metrics.jsonl').write_text('\n'.join(json.dumps(v) for v in rows)+'\n{"partial":')
 report=r.report(tmp_path,1)
 assert report['latest_update']==3 and report['training_hold'][0]['mean']['vx_mae_mps']==.3
 assert report['hold_mean_delta_current_minus_previous']['(1.0, 0.0)']['vx_mae_mps']==pytest.approx(-.2)
 assert report['latest_deterministic_screen'] is None and report['partial_jsonl_rows_ignored']
 assert 'UNKNOWN' in r.markdown(report)


@pytest.mark.parametrize('fail',[False,True])
def test_runner_boundary_closes_before_screen_and_rebuilds_fresh_obs_only_on_success(fail):
 from types import SimpleNamespace
 events=[];state=torch.get_rng_state().clone()
 def screen():
  events.append('screen');torch.rand(3)
  if fail:raise RuntimeError('physics failure')
  return {'selection_eligible':False}
 def rebuild():
  assert torch.equal(torch.get_rng_state(),state)
  events.append('rebuild');return 'fresh_env',{'student_actor':'fresh_obs'}
 env=SimpleNamespace(close=lambda:events.append('close'))
 if fail:
  with pytest.raises(RuntimeError,match='physics failure'):r.screen_training_boundary(env,screen,rebuild)
  assert events==['close','screen']
 else:
  env,obs,result=r.screen_training_boundary(env,screen,rebuild)
  assert env=='fresh_env' and obs['student_actor']=='fresh_obs'
  assert events==['close','screen','rebuild'] and result['training_resume']['explicit_reset']
 assert torch.equal(torch.get_rng_state(),state)


def test_atomic_human_review_reuses_metrics_only(tmp_path):
 (tmp_path/'metrics.jsonl').write_text(json.dumps({'update':100,'cohort_statistics':{'hold':{}}})+'\n')
 summary=r.write_review(tmp_path)
 assert summary['simulation']==0 and (tmp_path/'evaluation_review.md').exists()
 assert json.loads((tmp_path/'evaluation_review.json').read_text())['latest_update']==100
