"""PORTABLE integration checks for measured-stage state and312D contracts."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy
import torch
import pytest
from native_stand.students import motion_mcu_measured as course,motion_mcu_privileged as privileged
from native_stand.students.model import Student
from native_stand.students.motion_fixed_height import network as old_network,CRITIC as OLD_CRITIC


def test_versioned312_network_keeps_deployment_actor_identical():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(761)
        old=Student(True,version='multiskill_mcu_v2',network_architecture=old_network())
        new=Student(True,version='multiskill_mcu_v2',network_architecture=privileged.network())
        new.load_state_dict(old.state_dict())
        x=torch.randn(9,31)
        torch.testing.assert_close(old(x),new(x),atol=0,rtol=0)
        critic=Student(False,version='multiskill_mcu_v2',network_architecture=privileged.network(),critic_privileged_version=privileged.VERSION)
        assert critic.mlp[0].in_features==312 and privileged.contract()['actor_dim']==31
        with pytest.raises(ValueError):Student(False,version='multiskill_mcu_v2',network_architecture=privileged.network(),critic_privileged_version=OLD_CRITIC)


def test_measured_recipe_keeps_ppo_reward_and_explicit_sampling_only_smoke():
    from native_stand.students.motion_mcu_measured_io import recipe,validate_recipe
    from native_stand.students.motion_wheel_matching import configuration
    from native_stand.students.training_recipe import validate_instance
    c=configuration();c['smooth_mix']=.75
    source='9de0a63d46548bd100882929ef19a4dd16897d91a44d3a250b833fd4350559a7'
    r=recipe(source,course.encode([3,4,2,2],3),4870,2*2**30,c)
    assert validate_recipe(r,source)==r and validate_instance(r,source)==r
    assert r['BC_coefficient']==0 and r['warmup_updates_per_phase']==20
    assert 'all_cohort_pre_action_samples' in r['smoke_wiring']['mandatory']
    bad=copy.deepcopy(r);bad['physical_std_wheel_rad_s']=1.
    with pytest.raises(ValueError):validate_recipe(bad,source)


def test_effective_spec_stage_and_evaluation_revision_stay_bound():
    template=dict(kind='student',role='MotionZeroStudent',checkpoint='/source.pt',sha256='a'*64,
                  family='motion_native_normal',seed=1,yaw_rad=0.,seconds=31.52)
    spec=dict(motion_training=dict(stage=0),student_training=dict(stage=0,course_stage=0),
        motion_timing=dict(stage=4),curriculum={},ppo=dict(envs=4096),student_evaluation=dict(cases=[template]))
    value=course.encode([1,2,3,4],3);out=course.effective_spec(spec,value)
    assert spec['motion_training']['stage']==0
    assert out['motion_training']['stage']==out['student_training']['course_stage']==value
    assert all(r['_mcu_revision']==value for r in out['student_evaluation']['cases'])
    assert all(r['command'][2]==.34 for r in out['student_evaluation']['cases'])
    assert sum(out['curriculum']['stages'][0]['slot_counts'])==4096


def test_owner_refresh_after_stress_promotion_uses_measured_two_argument_contract():
    from types import SimpleNamespace
    from native_stand.students.motion_runner import advance_course_boundary,effective_course_spec
    template=dict(kind='student',role='MotionZeroStudent',checkpoint='/source.pt',sha256='a'*64,
                  family='motion_native_normal',seed=1,yaw_rad=0.,seconds=31.52)
    stage=course.encode([5,5,6,5],2)
    spec=dict(mcu_execution={},motion_training=dict(stage=stage),
        student_training=dict(stage=stage,course_stage=stage,course_version=course.VERSION),
        motion_timing=dict(stage=4,version='motion31_standing_antiosc_armature_v4'),
        curriculum={},ppo=dict(envs=4096),student_evaluation=dict(cases=[template]))
    spec=course.effective_spec(spec,stage)
    state=dict(stage=stage,stage_started_update=0,last_evaluation_update=-1,consecutive_passes=0,accepted_by_stage={})
    owner=SimpleNamespace(motion_course_state=copy.deepcopy(state))
    records=[dict(case=row,per_env=[dict(env=i,accepted=True,controlled_seconds=30.) for i in range(8)])
             for row in spec['student_evaluation']['cases']]
    candidate=dict(stage=stage,update=500,complete_matrix=True,evaluations=records,checkpoint='/u500.pt',sha256='b'*64)
    out,state,decision=advance_course_boundary(spec,owner,candidate)
    assert decision['advanced'] and state['stage']==course.encode([5,5,6,5],3)
    assert owner.motion_course_state is state
    # Both the advanced-stage refresh and recurring anti-oscillation refresh
    # must work; the latter was the actual update500 failure.
    out=effective_course_spec(out,state)
    out=effective_course_spec(out,state)
    assert len(out['student_evaluation']['cases'])==27
    assert all(row['_mcu_revision']==state['stage'] for row in out['student_evaluation']['cases'])
    assert not any(row['command']==[4.,0.,.34] for row in spec['student_evaluation']['cases'])
    assert any(row['command']==[4.,0.,.34] for row in out['student_evaluation']['cases'])


def test_measured_recovery_recipe_cannot_repeat_warmup_or_add_budget():
    from native_stand.students.motion_mcu_measured_io import recipe,validate_recipe
    from native_stand.students.motion_wheel_matching import configuration
    from native_stand.students.motion_training import phase_warmup
    c=configuration();c['smooth_mix']=.75
    r=recipe('b'*64,course.encode([5,5,6,5],3),4370,2*2**30,c,recovery=True)
    assert validate_recipe(r,'b'*64)==r and phase_warmup(r,'formal')==0
    assert r['milestones']==list(range(500,4371,500)) and r['formal_resume_offset']==500
    for key,value in (('warmup_updates_per_phase',20),('physical_std_wheel_rad_s',1.)):
        bad=copy.deepcopy(r);bad[key]=value
        with pytest.raises(ValueError):validate_recipe(bad,'b'*64)
    bad=recipe('b'*64,r['stage'],4870,2*2**30,c,recovery=True)
    with pytest.raises(ValueError):validate_recipe(bad,'b'*64)
    block=recipe('b'*64,r['stage'],1500,2*2**30,c,recovery=True)
    assert validate_recipe(block,'b'*64)==block


def test_measured_storage_counts_all27_conditions_and_rejects_oversized_phase(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from native_stand.command_v6.tools import storage_budget as storage
    cp=tmp_path/'source.pt';cp.write_bytes(b'checkpoint')
    spec=dict(mcu_execution=True,mcu_measured=True,source_checkpoint=str(cp),
        student_evaluation=dict(cases=[{}]*27),student_training=dict(version='student_motion31_training_v1',milestones=list(range(500,4371,500))),
        budget=dict(formal_updates=4370),artifacts=dict(storage_max_bytes=2528544307,terminal_reserve_bytes=512*2**20))
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda path:SimpleNamespace(free=20*2**30))
    plan=storage.check(spec,tmp_path/'formal','formal')
    assert plan['status']=='DECLARED_STORAGE_TOO_SMALL'
    assert plan['evaluation_bytes']==27*9*(131072+3152*8*150*4)
    spec['budget']['formal_updates']=1500
    spec['student_training']['milestones']=[500,1000,1500]
    assert storage.check(spec,tmp_path/'smaller_formal','formal')['status']=='PASS'


def test_posteval_recovery_reconstructs_state_and_extends_only_unseen_commands():
    from native_stand.students.motion_mcu_measured_io import reconstructed_boundary
    from native_stand.students.motion_symmetry import update_mix
    from native_stand.students.motion_refinement import advance,initial
    from native_stand.students.motion_tracking_reward import configuration
    stage=course.encode([5,5,6,5],2)
    policy=configuration('student_motion31_parallel_quality_v10')['refinement']
    source=dict(update=500,motion_course_state=dict(stage=stage,stage_started_update=0,
        last_evaluation_update=-1,consecutive_passes=0,accepted_by_stage={}),
        symmetry_mix_state=initial(course.commands(stage)),reward_refinement_state=initial(course.commands(stage)))
    rows=[]
    commands=[(0,cmd) for cmd in course.commands(stage)]+[(1,[0.,0.]),(2,[0.,0.])]
    for variant,cmd in commands:
        es=[dict(env=i,accepted=True,controlled_seconds=30.,tracking=dict(MAE=0.,gain=1.,direction=1.),
            checks={k:True for k in policy['required_checks']},
            symmetry_metrics=dict(swing_hold_peak_rad=0.,horizontal_hold_mean_mps=0.,common_abs_hold_mean_mps=0.)) for i in range(8)]
        rows.append(dict(case=dict(command=cmd+[.34],_mcu_variant=variant,_mcu_revision=stage),per_env=es))
    candidate=dict(update=500,stage=stage,checkpoint='/u500.pt',sha256='b'*64,complete_matrix=True,evaluations=rows)
    state,decision=course.advance(source['motion_course_state'],candidate)
    old=dict(reward=dict(refinement=policy))
    nominal=course.refinement_records(rows,stage)
    sym=update_mix(source['symmetry_mix_state'],nominal)
    ref,_=advance(source['reward_refinement_state'],nominal,policy)
    result=reconstructed_boundary(old,source,candidate,dict(state=state,decision=decision),sym,ref)
    assert result['state']['stage']==course.encode([5,5,6,5],3)
    assert result['state']['stage_started_update']==result['state']['last_evaluation_update']==0
    assert source['motion_course_state']['last_evaluation_update']==-1
    assert len(result['reward_mix'])==25 and len(ref)==22
    unseen=next(r for r in result['reward_mix'] if r['target']==[4.,0.])
    assert unseen['mix']==0. and unseen['consecutive_eligible']==0
    with pytest.raises(ValueError):
        reconstructed_boundary(old,source,candidate,dict(state=source['motion_course_state'],decision=decision),sym,ref)


def test_new_critic_initial_value_and_adam_zero_columns_without_project_files():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(167)
        actor=Student(True,version='multiskill_mcu_v2',network_architecture=old_network())
        critic=Student(False,version='multiskill_mcu_v2',network_architecture=old_network(),critic_privileged_version=OLD_CRITIC)
        optimizer=torch.optim.Adam([*actor.parameters(),*critic.parameters()],lr=3e-5)
        # Populate Adam statistics without applying a gradient/update.
        for parameter in [*actor.parameters(),*critic.parameters()]:
            optimizer.state[parameter]=dict(step=torch.tensor(12.),exp_avg=torch.ones_like(parameter)*.01,
                exp_avg_sq=torch.ones_like(parameter)*.001)
        source=dict(actor=actor.state_dict(),critic=critic.state_dict(),optimizer=optimizer.state_dict(),
                    network_architecture=old_network(),critic_privileged_version=OLD_CRITIC)
        expanded=privileged.expand_source(source)
        new=Student(False,version='multiskill_mcu_v2',network_architecture=privileged.network(),critic_privileged_version=privileged.VERSION)
        new.load_state_dict(expanded['critic'])
        x=torch.randn(16,184);y=torch.randn(16,128)
        torch.testing.assert_close(critic(x),new(torch.cat((x,y),-1)),atol=1e-6,rtol=1e-5)
        key=next(k for k,v in expanded['optimizer']['state'].items() if v['exp_avg'].shape==(512,312))
        assert not expanded['optimizer']['state'][key]['exp_avg'][:,184:].any()
        assert not expanded['optimizer']['state'][key]['exp_avg_sq'][:,184:].any()
