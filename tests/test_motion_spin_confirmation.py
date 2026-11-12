"""CPU identity/coverage checks; no physics or optimizer."""
import copy,json
from pathlib import Path
import pytest
from native_stand.students.motion_confirmation import verify
ROOT=Path(__file__).resolve().parents[1]

def prepared():
 import importlib.util
 parent=ROOT/'configs/experiments/2026_10_17-18_18__motion810_spin61_formal_recovery/spec.json'
 cp=ROOT/'runs/2026_10_17-18_18__motion810_spin61_formal_recovery/formal/model_update_750.pt'
 if not parent.exists() or not cp.exists():pytest.skip('local750 parent/checkpoint unavailable')
 spec=importlib.util.spec_from_file_location('spinprep',ROOT/'scripts/prepare_motion_spin_confirmation.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 result=module.prepare(parent,cp)
 # Capture publication payload without writing files.
 captured={}
 module.publish=lambda folder,payload,*args,**kwargs:captured.update(payload)
 module.prepare(parent,cp,True)
 return captured['spec.json'],result

def test_confirmation_same_recipe21_cases168_envs(monkeypatch):
 monkeypatch.setenv('HAPI_SESSION_ID','cpu-test')
 s,r=prepared();assert verify(s) and len(s['student_evaluation']['cases'])==21
 assert r['files_written']==0 and r['scored_episodes']==168
 for seed in s['motion_confirmation']['seeds']:
  assert len([x for x in s['student_evaluation']['cases'] if x['seed']==seed])==7

@pytest.mark.parametrize('kind',['missing','ppo','reward','sha','count','seconds'])
def test_confirmation_rejects_contract_drift(monkeypatch,kind):
 monkeypatch.setenv('HAPI_SESSION_ID','cpu-test');s,_=prepared();s=copy.deepcopy(s)
 if kind=='missing':s['student_evaluation']['cases'].pop()
 if kind=='ppo':s['budget']['formal_updates']=1
 if kind=='reward':s['reward']['dt']=.02
 if kind=='sha':s['source_sha256']='x'*64
 if kind=='count':s['evaluation']['scored_envs']=1
 if kind=='seconds':s['student_evaluation']['cases'][0]['seconds']=10
 with pytest.raises(ValueError):verify(s)

def test_public_spin_verifier_routes_trained61_confirmation(monkeypatch):
 monkeypatch.setenv('HAPI_SESSION_ID','cpu-test');s,_=prepared()
 from native_stand.students.motion_spin_experiment import verify as spin_verify
 assert spin_verify(s)

def test_health_accepts_explicit_trained61_source(monkeypatch):
 monkeypatch.setenv('HAPI_SESSION_ID','cpu-test')
 path=ROOT/'configs/experiments/2026_10_18-10_37__motion750_spin_three_seed_ppo0/spec.json'
 if not path.exists():pytest.skip('local declaration unavailable')
 from native_stand.command_v6.tools.health_check import inspect_project
 h=inspect_project(ROOT,spec_path=path,include_processes=False)
 assert not [x for x in h.get('findings',[]) if x['reason_code']=='IDENTITY_ERROR' and 'model_update_750.pt' in str(x)]
