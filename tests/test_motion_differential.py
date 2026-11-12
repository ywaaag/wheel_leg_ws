FROZEN_TEST_SCOPE='PORTABLE'
import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest,torch
from native_stand.students import motion_differential as d

def test_signed_geometric_target_swaps_with_side_and_yaw():
 centers=torch.tensor([[[0.,.2,.06],[0.,-.2,.06]]]);f=torch.tensor([[1.,0.,0.]])
 target=d.target_from_geometry(centers,f,torch.tensor([.5]))
 torch.testing.assert_close(target,torch.tensor([-.1]))
 torch.testing.assert_close(d.target_from_geometry(centers.flip(1),f,torch.tensor([.5])),-target)
 torch.testing.assert_close(d.target_from_geometry(centers,f,torch.tensor([-.5])),-target)
 # Arbitrary rigid yaw rotation preserves projected target.
 q=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
 torch.testing.assert_close(d.target_from_geometry(centers@q.T,f@q.T,torch.tensor([.5])),target)

def test_matching_difference_beats_idle_and_wrong_direction_and_tail_has_gradient():
 for mix in (0.,.75):
  target=torch.tensor([.1]);correct=d.terms(torch.tensor([[.1,-.1]]),target,mix)
  assert correct['density'].item()==0 and correct['opposite'].item()==1 and correct['imbalance'].item()==0
  assert d.terms(torch.zeros(1,2),target,mix)['density']<0
  assert d.terms(torch.tensor([[-.1,.1]]),target,mix)['density']<d.terms(torch.zeros(1,2),target,mix)['density']
  rolling=torch.tensor([[2.,-2.]],requires_grad=True);loss=d.terms(rolling,target,mix)['density'].sum();loss.backward()
  assert torch.isfinite(rolling.grad).all() and rolling.grad.abs().min()>0


def test_apply_gates_to_yaw_and_dt_exactly_once():
 m={'center_world':torch.tensor([[[0.,.2,.06],[0.,-.2,.06]]]*2),'forward_world':torch.tensor([[1.,0.,0.]]*2),'rolling_speed_forward_mps':torch.zeros(2,2)}
 env=SimpleNamespace(motion_spin_measurement_post=m,motion_action_command=torch.tensor([[0.,.5,.34],[1.,0.,.34]]),symmetry_mix=torch.zeros(2))
 base={'fields':('prior',),'raw':torch.zeros(2,1),'density':torch.zeros(2,1),'dt':torch.zeros(2,1),'total':torch.zeros(2),'spin_diagnostics':{'activation':torch.tensor([1.,0.])}}
 out=d.apply(base,env)
 assert out['fields'][-1]=='spin_differential_target' and out['total'][0]<0 and out['total'][1]==0
 torch.testing.assert_close(out['total'],out['density'].sum(-1)*.01)
 assert base['fields']==('prior',) and not base['total'].any()


def test_recipe_and_audit_explicit_v9_old_v8_unchanged():
 from native_stand.students.motion_tracking_reward import configuration,reward_fields,quality_base_configuration
 from native_stand.students.motion_training import declaration
 from native_stand.students.motion_startup import smoke_profile
 from native_stand.students.motion_symmetry import network,configuration as old
 from native_stand.students.training_recipe import validate_instance
 from native_stand.students.motion_reward_audit import audit
 sha='a'*64;r=declaration(sha,0,reward_version=d.VERSION)
 r.update(course_version='motion31_large_combo4096_v1',course_stage=0,network_architecture=network(),smoke_wiring=smoke_profile(),checkpoint_every_updates=100)
 assert validate_instance(r,sha)==r and r['critic_privileged_version']=='motion_symmetry_privileged85_v1'
 assert configuration(d.VERSION)==d.configuration() and 'differential' not in old()
 assert reward_fields(r['reward'])[-1]=='spin_differential_target'
 assert quality_base_configuration(d.VERSION)['version']=='student_motion31_large_gated_refinement_v6'
 a=audit(r['reward']);assert len(a['differential']['curves'])==36 and a['named_weights']['spin_differential_target']==-.5
 bad=copy.deepcopy(r);bad['reward']['differential']['weight']=-5
 with pytest.raises(ValueError):validate_instance(bad,sha)


def test_refinement_requires_complete_accepted_evidence_and_can_revoke():
 state=[{'target':[0.,.5],'mix':0.,'confirmations':0,'regressions':0}]
 e={'accepted':True,'symmetry_metrics':{'swing_hold_peak_rad':.02,'horizontal_hold_mean_mps':.01,'common_abs_hold_mean_mps':.01}}
 rec=[{'case':{'command':[0.,.5,.34]},'per_env':[copy.deepcopy(e) for _ in range(8)]}]
 s=d.update_mix(state,rec);assert s[0]['mix']==0
 s=d.update_mix(s,rec);assert s[0]['mix']==.25
 unknown=[{'case':{'command':[0.,.5,.34]},'per_env':[]}]
 assert d.update_mix(s,unknown)==s
 for r in rec[0]['per_env']:r['accepted']=False
 s=d.update_mix(s,rec);s=d.update_mix(s,rec);assert s[0]['mix']==0


def test_assess_axis_and_full_window_no_idle_pass():
 n=3000;a={}
 for name,value in [('error',0.),('opposite',1.),('imbalance',0.),('target',.1)]:a['differential_'+name]=np.full((n,1,1),value)
 a['differential_rolling']=np.tile([.1,-.1],(n,1,1))
 base={'checks':{'survival30':True},'accepted':True}
 r=d.assess(base,a,np.ones(n,bool),0,torch.ones(n,dtype=torch.bool),[0,.5,.34]);assert r['accepted']
 a['differential_opposite'][:]=0
 assert not d.assess(base,a,np.ones(n,bool),0,torch.ones(n,dtype=torch.bool),[0,.5,.34])['accepted']
 assert base['checks']=={'survival30':True}

def test_actual_u1000_new_recipe_zero_write_and_health_contract():
 from native_stand.students.motion_differential_experiment import build,verify
 from native_stand.students.requests import validate
 from native_stand.command_v6.tools.storage_budget import check
 root=Path(__file__).resolve().parents[1];parent=root/'configs/experiments/2026_10_18-15_45__motion750_symmetry85_reboot_resume/spec.json';cp=root/'runs/2026_10_18-15_45__motion750_symmetry85_reboot_resume/formal/model_update_1000.pt'
 if not parent.exists() or not cp.exists():pytest.skip('local settled u1000 fixture unavailable')
 # Build/verify only; no output creation, simulator or optimizer.
 dest=root/'configs/experiments/2026_10_18-23_59__cpu_diff_fixture/spec.json'
 s=build(parent,cp,experiment=dest.parent.name,spec_path=dest);verify(s);assert validate(s)==[] and not dest.parent.exists()
 assert s['budget']['smoke_updates']+s['budget']['formal_updates']==5000 and s['budget']['formal_wall_s'] is None
 assert s['motion_behavior_guard']['regression_action']=='record_only'
 assert s['student_training']['continuation_teacher']['sha256']!=s['source_sha256']
 assert check(s,root/'runs'/dest.parent.name/'formal','formal')['status']=='PASS'
 bad=copy.deepcopy(s);bad['reward']['differential']['scales_mps'][0]=2.
 with pytest.raises(ValueError):verify(bad)

@pytest.fixture
def settled_retry_bundle(tmp_path):
    from native_stand.students.data import sha

    parent = tmp_path / 'configs/experiments/failed/spec.json'
    run = tmp_path / 'runs/failed'
    evaluation = run / 'source_eval'
    parent.parent.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    parent.write_text(json.dumps({'experiment': 'failed'}))
    parent.with_name('budget.json').write_text(json.dumps({
        'used': {'smoke': 0, 'formal': 0}, 'attempts': {'fixture': {'status': 'FAILED'}},
    }))
    (run / 'chain_terminal_report.json').write_text(json.dumps({
        'status': 'FAILED', 'chain_wall_s': 3.25,
    }))
    (evaluation / 'terminal_report.json').write_text(json.dumps({
        'failure': {'exception': "ValueError('differential measurement missing')"},
        'actual_updates': 0,
    }))
    for i in range(7):
        trace = evaluation / f'trace_{i:02}.bin'
        trace.write_bytes(f'CPU evidence fixture {i}'.encode())
        (evaluation / f'evaluation_{i:02}.json').write_text(json.dumps({
            'trace': {'path': trace.name, 'sha256': sha(trace)},
        }))
    return parent, run


def test_retry_evidence_has_exact_tmp_inputs_and_zero_online_cost(settled_retry_bundle):
    from native_stand.students.data import sha
    from native_stand.students.motion_differential_experiment import retry_evidence

    parent, run = settled_retry_bundle
    result = retry_evidence(parent)
    assert result['parent_run'] == str(run)
    assert result['historical_online_updates'] == 0
    assert result['historical_wall_s'] == 3.25
    assert result['original_online_limit'] == 5000
    assert len(result['inputs']) == 18
    assert all(sha(path) == digest for path, digest in result['inputs'].items())


@pytest.mark.parametrize('fault', ['missing_terminal', 'trace_changed', 'nonzero_online', 'active_attempt'])
def test_retry_evidence_rejects_incomplete_changed_or_unsettled_fixture(settled_retry_bundle, fault):
    from native_stand.students.motion_differential_experiment import retry_evidence

    parent, run = settled_retry_bundle
    if fault == 'missing_terminal':
        (run / 'chain_terminal_report.json').unlink()
        error, message = FileNotFoundError, 'chain_terminal_report'
    elif fault == 'trace_changed':
        (run / 'source_eval/trace_00.bin').write_bytes(b'changed')
        error, message = ValueError, 'parent trace changed'
    else:
        path = parent.with_name('budget.json')
        book = json.loads(path.read_text())
        if fault == 'nonzero_online':
            book['used']['formal'] = 1
        else:
            book['attempts']['fixture']['status'] = 'RUNNING'
        path.write_text(json.dumps(book))
        error, message = ValueError, 'settled zero-PPO'
    with pytest.raises(error, match=message):
        retry_evidence(parent)


def test_retry_bound_to_exact_settled_zero_ppo_failure():
 from native_stand.students.motion_differential_experiment import retry_evidence,build,verify
 root=Path(__file__).resolve().parents[1];failed=root/'configs/experiments/2026_10_18-22_36__motion1000_differential85_5000/spec.json'
 if not failed.exists():pytest.skip('local settled failure unavailable')
 old=json.loads(failed.read_text());run=root/'runs'/old['experiment']
 required=[failed.with_name('budget.json'),run/'chain_terminal_report.json',run/'source_eval/terminal_report.json']
 required += [run/f'source_eval/evaluation_{i:02}.json' for i in range(7)]
 if not all(path.is_file() for path in required):pytest.skip('historical retry terminal/evaluation bundle unavailable')
 required += [run/'source_eval'/json.loads(path.read_text())['trace']['path'] for path in required[3:]]
 parent=Path(old['motion_differential']['parent_spec']);source=Path(old['source_checkpoint'])
 required += [parent,parent.with_name('budget.json'),source,source.parent.parent/'chain_terminal_report.json']
 if not all(path.is_file() for path in required):pytest.skip('historical retry trace/source bundle unavailable')
 r=retry_evidence(failed);assert r['historical_online_updates']==0 and r['historical_wall_s']>0 and len(r['inputs'])==18
 c=old['motion_differential'];dest=root/'configs/experiments/2026_10_18-23_58__cpu_retry_fixture/spec.json'
 s=build(c['parent_spec'],old['source_checkpoint'],experiment=dest.parent.name,spec_path=dest)
 s['motion_differential_retry']=r;s['student_dependencies'].update(r['inputs']);verify(s)
 assert not dest.exists() and s['budget']['smoke_updates']+s['budget']['formal_updates']==5000
 bad=copy.deepcopy(s);bad['motion_differential_retry']['historical_online_updates']=1
 with pytest.raises(ValueError):verify(bad)
