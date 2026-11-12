FROZEN_TEST_SCOPE='PORTABLE'
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from native_stand.command_v6.tools import storage_budget
from native_stand.command_v6.domain.evidence_summary import compact,result
from native_stand.command_v6.domain.metrics import motion_reward_fields,RolloutMetrics
from native_stand.command_v6.domain.reward import compute


def test_motion_quick_storage_counts_conditions_and_full_reuse(tmp_path):
 from native_stand.students.motion_quick_evaluation import configuration,warm_configuration
 cp=tmp_path/'model.pt';cp.write_bytes(b'fixture')
 spec={'source_checkpoint':str(cp),'budget':{'formal_updates':1000},
  'student_training':{'milestones':[250,500,750,1000]},
  'student_evaluation':{'cases':[{'seconds':31.52}]},
  'evaluation':{'scored_envs':8},'control':{'policy_dt_s':.01},
  'ppo':{'envs':4096},'artifacts':{},'motion_differential':True,'motion_parallel':True}
 base=storage_budget.motion_student_estimate(spec,'formal')
 assert base['quick_evaluation_bytes']==0
 spec['checkpoint_quick_evaluation']=configuration()
 legacy=storage_budget.motion_student_estimate(spec,'formal')
 assert legacy['quick_evaluation_events']==10 and legacy['quick_conditions_bound']==1
 spec['checkpoint_quick_evaluation']=warm_configuration()
 warm=storage_budget.motion_student_estimate(spec,'formal')
 assert warm['quick_evaluation_events']==8 and warm['quick_conditions_bound']==6
 assert warm['quick_evaluation_bytes']==legacy['quick_evaluation_bytes']*8*6//10
 assert warm['evaluation_bytes']-base['evaluation_bytes']==warm['quick_evaluation_bytes']
 assert storage_budget.motion_student_estimate(spec,'smoke')['quick_evaluation_bytes']==0
 # The quick estimate also applies to older Motion recipes without differential costs.
 spec.pop('motion_differential');spec.pop('motion_parallel')
 spec['student_evaluation']['cases']=[{'case_id':str(i),'kind':'student','family':'motion_native_normal',
  'seconds':31.52,'command':[x,y,.34]} for i,(x,y) in enumerate([(0,0),(.5,0),(2,0),(-1,0),(0,1),(0,-1)])]
 assert storage_budget.motion_student_estimate(spec,'formal')['quick_evaluation_bytes']==warm['quick_evaluation_bytes']


def test_storage_gate_accounts_peak_and_terminal_reserve(tmp_path,monkeypatch):
    spec={'budget':{'formal_updates':50},'evaluation':{'seconds':30,'full_every_updates':25},
        'ppo':{'envs':64,'steps_per_env':128},'control':{'policy_dt_s':.01}}
    monkeypatch.setattr(storage_budget.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
    r=storage_budget.check(spec,tmp_path/'new','formal')
    assert r['status']=='INSUFFICIENT_STORAGE' and r['atomic_temp_peak_bytes']>0 and r['terminal_reserve_bytes']>0
    assert not (tmp_path/'new').exists()
    monkeypatch.setattr(storage_budget.shutil,'disk_usage',lambda p:SimpleNamespace(free=r['required_bytes']))
    assert storage_budget.check(spec,tmp_path/'new','formal')['status']=='PASS'


@pytest.mark.parametrize('free_delta,ceiling_delta,status', [
    (0, 0, 'PASS'), (-1, 0, 'INSUFFICIENT_STORAGE'), (0, -1, 'DECLARED_STORAGE_TOO_SMALL')])
def test_motion_stage_capacity_uses_estimate_and_reserve_once(tmp_path,monkeypatch,free_delta,ceiling_delta,status):
    from native_stand.students.requests import output_storage_budget
    # Existing source/metrics already consume disk; the phase needs200+30+50,
    # not the whole300-byte hard ceiling plus another50-byte terminal reserve.
    spec={'motion_decoupled':{'storage_preflight_version':'motion_stage_capacity_v1'},
          'artifacts':{'storage_max_bytes':280+ceiling_delta}}
    monkeypatch.setattr(storage_budget,'estimate',lambda *a:dict(estimated_artifact_bytes=200,
                        atomic_temp_peak_bytes=30,terminal_reserve_bytes=50))
    monkeypatch.setattr(storage_budget.shutil,'disk_usage',lambda p:SimpleNamespace(free=280+free_delta))
    result=output_storage_budget(spec,tmp_path/'not_created','formal',training=True)
    assert result['required_bytes']==280 and result['status']==status
    assert not (tmp_path/'not_created').exists()


def test_legacy_student_storage_policy_does_not_implicitly_change(tmp_path,monkeypatch):
    from native_stand.students.requests import output_storage_budget
    from native_stand.students import artifacts
    monkeypatch.setattr(artifacts,'storage',lambda s:dict(chain_max_bytes=300,terminal_reserve_bytes=50))
    monkeypatch.setattr(storage_budget.shutil,'disk_usage',lambda p:SimpleNamespace(free=300))
    result=output_storage_budget({},tmp_path/'new','smoke',training=True)
    assert result['required_bytes']==350 and result['status']=='INSUFFICIENT_STORAGE'


def test_reserve_rechecks_storage_before_attempt_mutation(tmp_path,monkeypatch):
    from native_stand.command_v6.ledger import Ledger
    (tmp_path/'spec.json').write_text(json.dumps({'budget':{}}))
    path=tmp_path/'budget.json';path.write_text(json.dumps({'authorization':'AUTHORIZED','limits':{'source_eval':1},
        'used':{'source_eval':0},'attempts':{}}));before=path.read_bytes()
    monkeypatch.setattr(storage_budget,'check',lambda *a:{'status':'INSUFFICIENT_STORAGE'})
    with pytest.raises(ValueError,match='storage'):Ledger(path).reserve('exact','source_eval',tmp_path/'run')
    assert path.read_bytes()==before


def test_recursive_summary_removes_arrays_not_geometry_metrics():
    data={'per_env':[{'trace':[[0]*18]*1000,'mean_roll':.02}],
        'retained_evaluations':[{'per_env':[{'compact_trace_100hz':[[0]*39]*1000}],
          'control_diagnostics':{'rows':[[[0]*54]*8]*1000}}],
        'paired_state':{'rows':[[[0]*197]*8]*1000}}
    summary=result(data,reference='evaluation.json',sha256='a'*64)
    assert len(json.dumps(summary))<2000 and summary['per_env'][0]['mean_roll']==.02
    assert summary['raw_evidence']['sha256']=='a'*64
    assert len(data['per_env'][0]['trace'])==1000
    assert compact(summary)==summary


def test_real_posture_reward_metrics_summary_share_declaration():
    n=2;zero=torch.zeros(n);one=torch.ones(n);z6=torch.zeros(n,6)
    cfg={'version':'bounded_geometry_v1','weight':-.1,
        'deadbands':{'roll_rad':.02,'wheel_heading_offset_m':.02,'leg_length_difference_m':.02},
        'denominators':{'roll_rad':.01,'wheel_heading_offset_m':.01,'leg_length_difference_m':.01}}
    from native_stand.command_v6.domain.posture_geometry import validate_balance
    # Use actual field names from the shared validated declaration.
    validate_balance(cfg)
    weights={k:1. for k in motion_reward_fields({})}
    geometry={'roll_rad':zero,'wheel_heading_offset_m':zero,'leg_length_difference_m':zero,'valid':torch.ones(n,dtype=torch.bool)}
    _,raw,weighted=compute(one,one,zero,zero,zero,torch.stack((zero,zero,one),-1),
        torch.ones(n,2,dtype=torch.bool),z6,z6,z6,torch.zeros(n,4),torch.zeros(n,dtype=torch.bool),
        weights=weights,height_width=.1,linear_denominator=.04,yaw_denominator=.09,terminal_penalty=-1,
        posture_config=cfg,posture_geometry=geometry)
    stats=RolloutMetrics(motion_reward_fields({'posture_balance':cfg}),['fault'],n,'cpu')
    stats.add(raw,weighted,{'fault':torch.zeros(n,dtype=torch.bool)},torch.ones(n,2,dtype=torch.bool),
        torch.zeros(n,4),z6,[40,40,5.5,40,40,5.5])
    assert 'posture_balance' in stats.summary()['weighted_mean']


def test_source_reuse_certificate_identity_complete_and_missing(tmp_path,monkeypatch):
    from native_stand.command_v6.tools import source_evidence as tool
    from native_stand.command_v6.tools import evaluate_checkpoint as evaluator
    from native_stand.command_v6.tools.experiment_preparation import digest
    folder=tmp_path/'source';(folder/'source_snapshot').mkdir(parents=True)
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'frozen')
    s={'source_checkpoint':str(checkpoint),'source_sha256':digest(checkpoint),'asset_sha256':'a'*64}
    sp=folder/'source_snapshot/spec.json';sp.write_text(json.dumps(s))
    identity={'execution_dependencies':{'sha256':'taskcode'},'location_independent_modules_sha256':'code','PPO_sha256':'ppo','dependencies':{'torch':'frozen'}}
    (folder/'manifest.json').write_text(json.dumps({'spec_sha256':digest(sp),'asset_sha256':'a'*64,'execution_identity':identity,'evaluator_sha256':'evaluator'}))
    (folder/'terminal_report.json').write_text(json.dumps({'status':'COMPLETED','actual_updates':0}))
    protocol={'requested_seconds':30,'scored_envs':1,'steady_window_s':[2,30],'motion_window_s':[10,18]}
    payload={'checkpoint_sha256':digest(checkpoint),'spec_sha256':digest(sp),'asset_sha256':'a'*64,
        'target':[0,0],'seconds':30,'effective_protocol':protocol,'per_env':[{'env_id':0,'continuous_s':30,'outcome':'completed'}]}
    path=folder/'evaluation_source.json';path.write_text(json.dumps(payload))
    entry={'role':'source','checkpoint_sha256':digest(checkpoint),'evaluation':path.name,'evaluation_sha256':digest(path)}
    (folder/'candidate_registry.json').write_text(json.dumps([entry]))
    monkeypatch.setattr(tool,'recipe',lambda spec:spec)
    monkeypatch.setattr(tool,'identity',lambda spec:identity)
    monkeypatch.setattr(evaluator,'protocol_for_spec',lambda *a:protocol)
    assert tool.check(s,folder)['status']=='REUSABLE'
    other={**s,'asset_sha256':'c'*64}
    assert tool.check(other,folder)['status']=='CONFLICTING'
    payload['per_env'][0]['continuous_s']=10;path.write_text(json.dumps(payload));entry['evaluation_sha256']=digest(path)
    (folder/'candidate_registry.json').write_text(json.dumps([entry]))
    assert tool.check(s,folder)['status']=='MISSING'


def test_actual_chain_and_registry_compaction_are_wired():
    from native_stand.command_v6.tools.test_resources import source_file
    import ast
    for relative in ('command_v6/train.py','command_v6/stand_chain.py','command_v6/completion.py'):
        text=source_file(relative).read_text();ast.parse(text)
        assert 'compact(' in text
    text=source_file('command_v6/stand_chain.py').read_text()
    assert "if certificate['status']!='REUSABLE'" in text and "historical_wall_not_charged_as_new" in text
