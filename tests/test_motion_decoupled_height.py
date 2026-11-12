FROZEN_TEST_SCOPE = 'PORTABLE'

import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from native_stand.students import motion_decoupled as course, motion_height as height


def spec_state(focus='height', level=0):
    source = dict(certified_DR_level=1, smooth_mix=.75,
                  qualified_by_DR={'1': [[0.,0.]] + [course.axis_command(a,i) for a in course.AXES for i in range(4)]})
    state = course.initial_state(source); state.update(focus=focus, focus_level=level)
    if focus == 'DR': state['DR_level'] = level
    template = dict(kind='student', role='MotionZeroStudent', family='motion_native_normal', seed=1,
                    seconds=31.52, yaw_rad=0., checkpoint='/fake.pt', sha256='a'*64)
    s = dict(motion_decoupled=dict(state=state), motion_robustness=dict(state=state),
             motion_training={}, student_training={}, curriculum={}, ppo=dict(envs=4096),
             motion_timing=dict(configuration={}), student_evaluation=dict(cases=[template]))
    s = course.effective_spec(s,0,state)
    return s, s['motion_decoupled']['state']


def candidate(spec, update=250):
    return dict(checkpoint='/new.pt', sha256='b'*64, stage=spec['motion_decoupled']['state']['stage'],
                update=update, complete_matrix=True,
                evaluations=[dict(case=copy.deepcopy(c), per_env=[dict(env=i, accepted=True) for i in range(8)])
                             for c in spec['student_evaluation']['cases']])


def test_exact_three_sampling_quotas_and_full_height_cohort_identity():
    spec,state=spec_state('height',3)
    stage=course.stage_config(0,state=state)
    assert len(stage['sampling_slots'])==4096
    totals={p:sum(n for n,q in zip(stage['slot_counts'],stage['cohort_purposes']) if p==q)
            for p in ('nominal_zero','rehearsal','focus')}
    assert totals==dict(nominal_zero=614,rehearsal=1434,focus=2048)
    assert stage['group_slots'][:614]==[0]*614
    assert len({tuple(k) for k in stage['cohort_keys'] if k[2]!=k[3]})>10
    assert all(len(k)==4 for k in stage['cohort_keys'])


@pytest.mark.parametrize('level', [0, 1, 3])
def test_actual_runtime_reset_preserves_fixed_slot_height_cohort_identity(level):
    import ast
    from pathlib import Path
    from native_stand.command_v6.domain import command_schedule
    # Execute the actual reset sampling method with a no-op physical parent;
    # this CPU test exercises the former argmax alias bug without Isaac imports.
    tree=ast.parse(Path('src/native_stand/command_v6/runtime/env.py').read_text())
    method=copy.deepcopy(next(n for n in next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CommandEnv').body
                              if isinstance(n,ast.FunctionDef) and n.name=='_reset_idx'))
    base=type('Base',(),{'_reset_idx':lambda self,ids:None})
    cls=ast.ClassDef(name='Reset',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
    scope=dict(Base=base,torch=torch,command_schedule=command_schedule)
    exec(compile(module,'<actual reset sampling>','exec'),scope)
    _,state=spec_state('height',level);stage=course.stage_config(0,state=state)
    env=scope['Reset']();env.num_envs=4096;env.device='cpu';env.forced_target=None;env.stage_spec=stage
    env.target_xy=torch.zeros(4096,2);env.command=torch.zeros(4096,3);env.command_height=.34
    env.command_index=torch.zeros(4096,dtype=torch.long);env.command_counts=torch.zeros(len(stage['commands']),dtype=torch.long)
    env.schedule_s=[.5,1.,28.5,1.];env.history_needs_init=torch.zeros(4096,dtype=torch.bool)
    rng=torch.get_rng_state().clone();env._reset_idx(None)
    assert torch.equal(rng,torch.get_rng_state())
    assert env.command_index.tolist()==stage['sampling_slots']
    assert torch.equal(env.target_xy,torch.tensor(stage['commands'])[env.command_index])
    assert torch.bincount(env.command_index,minlength=len(stage['commands'])).tolist()==stage['slot_counts']
    subset=torch.tensor([4095,699,615,0]);env.command_index[subset]=0;env._reset_idx(subset)
    assert env.command_index[subset].tolist()==[stage['sampling_slots'][i] for i in subset]


def test_h0_legacy_coverage_may_alias_identical_tasks_but_height_transition_may_not():
    _,state=spec_state('height',0);stage=course.stage_config(0,state=state)
    counts=[n*128 for n in stage['slot_counts']]
    counts[0]+=counts[1]+counts[-1];counts[1]=counts[-1]=0
    result=course.holding_coverage(stage,[dict(hold_sample_counts=counts)])
    assert result['status']=='PASSED' and result['scope']=='legacy_H0_identical_task_keys_only'
    assert result['separate_alias_DR_coverage_verified'] is False
    counts[2]=0
    assert course.holding_coverage(stage,[dict(hold_sample_counts=counts)])['status']=='FAILED'
    _,state=spec_state('height',1);stage=course.stage_config(0,state=state)
    counts=[n*128 for n in stage['slot_counts']];counts[-1]=0
    assert course.holding_coverage(stage,[dict(hold_sample_counts=counts)])['status']=='FAILED'
    assert course.holding_coverage(stage,[dict(hold_sample_counts=[-1])])['status']=='FAILED'


def test_height_pair_counter_covers_all_twenty_pairs_without_policy_rng():
    ids=torch.arange(4096);episodes=torch.ones(4096,dtype=torch.long)
    rng=torch.get_rng_state().clone()
    choice=((ids*1103515245+episodes*12345+20261023)%2147483647)%20
    assert set(choice.tolist())==set(range(20))
    assert torch.equal(rng,torch.get_rng_state())


def test_direction_failure_does_not_veto_forward_promotion():
    s,state=spec_state('forward',4); c=candidate(s)
    for r in c['evaluations']:
        if r['case']['_task_axis']=='yaw_negative':r['per_env'][0]['accepted']=False
    out,d=course.advance(state,c)
    assert d['stage_passed'] and out['levels']['forward']==4
    assert out['levels']['yaw_negative']==3 and out['focus']=='DR'


def test_two_failures_lower_only_failed_task_and_keep_dr():
    s,state=spec_state(); c=candidate(s)
    for r in c['evaluations']:
        if r['case']['_focus_required'] or r['case']['_task_axis']=='yaw_negative':
            r['per_env'][0]['accepted']=False
    state,d=course.advance(state,c)
    assert state['levels']['yaw_negative']==3 and state['levels']['height']==-1
    s=course.effective_spec(s,state['stage'],state);state=s['motion_decoupled']['state'];c=candidate(s,500)
    for r in c['evaluations']:
        if r['case']['_focus_required'] or r['case']['_task_axis']=='yaw_negative':
            r['per_env'][0]['accepted']=False
    state,d=course.advance(state,c)
    assert d['lowered_tasks']==['yaw_negative']
    assert state['levels']==dict(forward=3,backward=3,yaw_positive=3,yaw_negative=2,height=-1,DR=1)
    assert not d['stop']


def test_unknown_missing_and_duplicate_cases_never_promote():
    s,state=spec_state()
    for mode in ('missing','unknown','duplicate'):
        c=candidate(s)
        if mode=='missing':c['evaluations'].pop()
        elif mode=='unknown':c['evaluations'][-1]['per_env'][0]['accepted']=None
        else:c['evaluations'][-1]=copy.deepcopy(c['evaluations'][0])
        out,d=course.advance(state,c)
        assert not d['evidence_valid'] and not d['stage_passed'] and out['levels']==state['levels']


def test_uncertified_dr_failure_not_old_skill_regression():
    s,state=spec_state('DR',2);c=candidate(s)
    for r in c['evaluations']:
        if r['case']['_motion_dr']['group']==2:r['per_env'][0]['accepted']=False
    state,d=course.advance(state,c)
    assert not d['lowered_tasks'] and state['levels']['DR']==1
    assert state['focus']=='forward' and state['levels']['yaw_negative']==3


def test_height_progress_then_locomotion_uses_same_episode():
    age=torch.tensor([0.,2.,4.,6.,8.,10.,18.]);initial=torch.full_like(age,.34);target=torch.full_like(age,.36)
    xy=torch.tensor([[.5,0.]]).repeat(len(age),1)
    cmd,rate=height.command(xy,age,initial,target,torch.zeros_like(age,dtype=torch.bool),False)
    assert torch.allclose(cmd[:4,0],torch.zeros(4))
    assert cmd[4,0]>.0 and cmd[5,0]==.5
    assert cmd[0,2]==pytest.approx(.34) and cmd[3,2]==pytest.approx(.36)
    assert rate[0]==0 and rate[2]>0 and rate[3]==0


def test_height_reward_increases_toward_command_without_penalizing_requested_rate():
    names=('height','transfer_vertical_velocity');n=3
    original=dict(fields=names,raw=torch.zeros(n,2),density=torch.zeros(n,2),dt=torch.zeros(n,2),total=torch.zeros(n),eligible=torch.ones(n,dtype=torch.bool))
    result=height.correct_reward(original,torch.tensor([.32,.34,.36]),torch.full((n,),.36),torch.full((n,),.01),torch.full((n,),.01))
    assert result['density'][:,0].tolist()==sorted(result['density'][:,0].tolist())
    assert result['raw'][:,1].sum()==0
    assert torch.equal(original['total'],torch.zeros(n))
    assert torch.allclose(result['total'],result['dt'].sum(-1))


def test_explicit_height_critic144_and_legacy140_value_preserved():
    from native_stand.students.model import Student
    from native_stand.students.motion_antiosc_privileged import network as old_network, VERSION as old_version
    actor=Student(True,version='multiskill_mcu_v2',network_architecture=old_network())
    critic=Student(False,version='multiskill_mcu_v2',critic_privileged_version=old_version,network_architecture=old_network())
    opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=3e-5)
    x=torch.randn(5,140);critic(x).sum().backward();opt.step()
    source=dict(actor=actor.state_dict(),critic=critic.state_dict(),optimizer=opt.state_dict(),network_architecture=old_network(),critic_privileged_version=old_version)
    new=height.expand_source(source)
    target=Student(False,version='multiskill_mcu_v2',critic_privileged_version=height.CRITIC,network_architecture=height.network())
    target.load_state_dict(new['critic'])
    augmented=height.critic_input(x,torch.full((5,),.34),torch.full((5,),.34),torch.full((5,),.34),torch.zeros(5))
    assert augmented.shape==(5,144)
    assert torch.allclose(target(augmented),critic(x),atol=1e-6)
    assert new['critic']['mlp.0.weight'][:,140:].abs().sum()==0
    pid=new['optimizer']['param_groups'][0]['params'][len(new['actor'])]
    assert new['optimizer']['state'][pid]['exp_avg'][:,140:].abs().sum()==0
    assert source['critic']['mlp.0.weight'].shape==(512,140)


def test_nominal_mixtures_only_inherit_matching_height_pair():
    old=[dict(target=[1.,0.],mix=.75)]
    states=course.extend_mixture(old,[[1.,0.,.34,.34],[1.,0.,.34,.36]])
    assert [s['mix'] for s in states]==[.75,0.]


def test_grouped_evaluator_keeps_different_height_schedules_separate():
    from native_stand.students.grouped_evaluation import plan
    spec,_=spec_state('height',1)
    groups=plan(spec['student_evaluation']['cases'])
    assert all(len({str(c['_height']) for c in group})==1 for group in groups)


@pytest.mark.parametrize('kind,expected', [('complete', True), ('short', False), ('nonfinite', False)])
def test_height_gate_uses_json_native_bool_and_preserves_incomplete_failure(monkeypatch, kind, expected):
    import json
    from native_stand.students import motion_reacquisition

    def base_assess(result, arrays, env_id):
        result['checks'] = dict(height=True, reacquisition_height=True)

    monkeypatch.setattr(motion_reacquisition, 'assess', base_assess)
    n = 2999 if kind == 'short' else 3000
    states = np.zeros((n, 1, 16), dtype=np.float64)
    states[:, :, 6] = .34
    if kind == 'nonfinite': states[-1, 0, 6] = np.nan
    arrays = dict(state=states, first_episode_valid=np.ones((n, 1), dtype=bool),
                  controller=np.full((n, 1, 1), 3, dtype=np.int64))
    case = dict(_height=dict(initial_m=.34, target_m=.34, moving=False))
    result = course.assess({}, arrays, 0, case)
    assert result['accepted'] is expected
    assert result['checks']['height_transition_complete'] is expected
    assert result['height_task']['complete_30s'] is expected
    assert all(type(value) is bool for value in result['checks'].values())
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    if not expected: assert 'height_transition_complete' in result['failed_checks']


@pytest.mark.parametrize('formal_updates,error_text', [
    (0, 'TypeError: Object of type bool is not JSON serializable'),
    (1, 'TypeError: Object of type bool is not JSON serializable'),
    (0, 'RuntimeError: unrelated physics fault'),
])
def test_preppo_recovery_rejects_consumed_budget_and_unrelated_fault(tmp_path, monkeypatch, formal_updates, error_text):
    import json
    folder = tmp_path / 'configs/experiments/failed'
    folder.mkdir(parents=True)
    run = tmp_path / 'runs/failed'
    (run / 'source_eval').mkdir(parents=True)
    old = dict(experiment='failed', motion_decoupled=dict(version=course.VERSION, parent_spec='/origin/spec.json'),
               motion_training={}, source_sha256='a' * 64)
    def write(path, data): path.write_text(json.dumps(data))
    write(folder / 'spec.json', old)
    write(folder / 'budget.json', dict(used=dict(source_eval=1, smoke=0, formal=formal_updates),
          attempts={'one': dict(phase='source_eval', status='FAILED', updates=0)}))
    write(run / 'chain_terminal_report.json', dict(status='FAILED', smoke_updates=0, formal_updates=formal_updates, chain_wall_s=208.))
    write(run / 'source_eval/terminal_report.json', dict(status='FAILED', actual_updates=0))
    write(run / 'source_eval/height_engineering_readback.json', dict(status='PASSED', PPO_updates=0, results=[{}] * 5))
    (run / 'source_eval.log').write_text('motion_runner.py\n' + error_text)
    monkeypatch.setattr(course, 'verify', lambda *args, **kwargs: True)
    if formal_updates or 'unrelated' in error_text:
        with pytest.raises(ValueError): course.preppo_recovery_evidence(folder / 'spec.json')
    else:
        proof = course.preppo_recovery_evidence(folder / 'spec.json')
        assert proof['historical_online_updates'] == 0 and proof['original_online_limit'] == 10000
        assert proof['historical_wall_s'] == 208. and proof['historical_artifact_bytes'] > 0


def test_completed_source_storage_recovery_reuses_exact_evidence_and_keeps_history(tmp_path, monkeypatch):
    import json
    from native_stand.students.data import sha
    folder = tmp_path / 'configs/experiments/failed'
    folder.mkdir(parents=True)
    run = tmp_path / 'runs/failed'; source = run / 'source_eval'; source.mkdir(parents=True)
    case = dict(command=[0., 0., .34], _motion_dr=dict(group=0), _height=dict(initial_m=.34, target_m=.34, moving=False))
    history = dict(historical_wall_s=208., historical_artifact_bytes=100, historical_source_eval_attempts=1, inputs={})
    old = dict(experiment='failed', asset_sha256='c'*64, source_sha256='a'*64,
        motion_decoupled=dict(version=course.VERSION, parent_spec='/origin/spec.json',
                             engineering_recovery=dict(version='motion_height_preppo_bool_recovery_v1')),
        motion_training=dict(prior_consumption=history), student_evaluation=dict(cases=[case]))
    def write(path, data): path.write_text(json.dumps(data))
    sp = folder / 'spec.json'; write(sp, old)
    write(folder / 'budget.json', dict(used=dict(source_eval=1, smoke=0, formal=0),
        attempts={'one': dict(phase='source_eval', status='COMPLETED', updates=0)}))
    write(run / 'chain_terminal_report.json', dict(status='FAILED', smoke_updates=0, formal_updates=0, chain_wall_s=204.6,
        failure=repr(ValueError('smoke preflight: '+repr({'errors':['student output filesystem reserve insufficient']})))))
    (run / 'source_eval.log').write_text('source completed without optimizer')
    write(source / 'height_engineering_readback.json', dict(status='PASSED', PPO_updates=0, results=[{}]*5))
    np.savez(source / 'trace_00.npz', state=np.zeros((3000, 8, 39)))
    rows = [dict(env=i, accepted=True, height_task=dict(complete_30s=True)) for i in range(8)]
    raw = dict(case=case, per_env=rows, model_unchanged=True, spec_sha256=sha(sp), asset_sha256='c'*64,
               checkpoint_sha256='a'*64, protocol=course.protocol(), trace=dict(path='trace_00.npz',sha256=sha(source/'trace_00.npz')))
    write(source/'evaluation_00.json',raw)
    records = [dict(case=case, per_env=rows, path='evaluation_00.json', sha256=sha(source/'evaluation_00.json'))]
    write(source/'terminal_report.json',dict(status='COMPLETED', actual_updates=0, evaluations=records))
    write(source/'progress.json',dict(status='COMPLETED', actual_updates=0, engineering_valid=True, evaluations=records))
    write(source/'manifest.json',dict(source_sha256='a'*64, spec_sha256=sha(sp)))
    write(source/'candidate_registry.json',[dict(update=0, sha256='a'*64, evaluations=records, complete_matrix=False)])
    monkeypatch.setattr(course,'verify',lambda *a, **k: True)
    monkeypatch.setattr(course,'assess',lambda result,*a: result)
    before={str(p):sha(p) for p in source.iterdir()}
    proof=course.preppo_recovery_evidence(sp)
    assert proof['version']=='motion_height_preppo_storage_recovery_v1'
    assert proof['historical_wall_s']==pytest.approx(412.6) and proof['historical_source_eval_attempts']==2
    assert proof['historical_online_updates']==0 and proof['original_online_limit']==10000
    assert proof['historical_artifact_bytes']>history['historical_artifact_bytes']
    reused,candidates=course.reused_source(dict(source_sha256='a'*64,motion_decoupled=dict(engineering_recovery=proof)))
    assert reused[0]['reused'] and reused[0]['path']==str(source/'evaluation_00.json')
    assert candidates[0]['complete_matrix'] is False
    assert {str(p):sha(p) for p in source.iterdir()}==before
    with (source/'trace_00.npz').open('ab') as file: file.write(b'tampered')
    with pytest.raises(ValueError,match='identity differs'):
        course.preppo_recovery_evidence(sp)
