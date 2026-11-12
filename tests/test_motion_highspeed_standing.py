FROZEN_TEST_SCOPE='PORTABLE'
import copy

import pytest
import torch

from native_stand.students import motion_highspeed_standing as c


def fixture():
    return dict(source_checkpoint='wide.pt',source_sha256='a'*64,ppo=dict(envs=4096),
        student_evaluation=dict(cases=[dict(kind='student',role='MotionZeroStudent',
            family='motion_native_normal',seed=104,yaw_rad=.01,seconds=31.52)]),
        motion_training={},student_training={},motion_course={},curriculum={},
        motion_height_dr=dict(state=c.initial_state()))


def candidate(spec):
    records=[dict(case=copy.deepcopy(row),per_env=[dict(env=i,accepted=True,
        survival_passed=True,controlled_seconds=30.,checks=dict(tracking=True,gain=True,contact=True,direction=True)) for i in range(8)]) for row in spec['student_evaluation']['cases']]
    return dict(stage=0,update=500,checkpoint='new.pt',sha256='b'*64,complete_matrix=True,
        evaluations=records,per_env=[e for r in records for e in r['per_env']])


@pytest.mark.parametrize('envs',(1,64,4096))
def test_allocation_preserves_task_identity_and_equal_direction_samples(envs):
    x=c.stage_config(envs=envs)
    assert sum(x['slot_counts'])==envs and len(x['cohort_keys'])==len(set(map(tuple,x['cohort_keys'])))
    if envs==4096:
        sums={}
        for purpose,n in zip(x['cohort_purposes'],x['slot_counts']):sums[purpose]=sums.get(purpose,0)+n
        assert sums['nominal_standing']+sums['height_standing']==819
        assert sums['stress_standing']+sums['PD_standing']==819
        assert sums['low_transition']==410
        assert all(sums['highspeed_'+d]==512 for d in c.DIRECTIONS)
        assert set(x['height_targets_m'])==set(c.HEIGHTS)
        assert all(h==.34 for h,p in zip(x['height_targets_m'],x['cohort_purposes']) if p.startswith('highspeed'))
    if envs==64:assert min(x['slot_counts'])>0


def test_low_precision_failure_does_not_erase_primary_pass_but_safety_failure_does():
    spec=c.effective_spec(fixture(),0);row=candidate(spec)
    low=next(r for r in row['evaluations'] if not r['case']['_primary_required'])
    low['per_env'][0]['accepted']=False
    low['per_env'][0]['checks']['gain']=False
    state,decision=c.advance(spec['motion_height_dr']['state'],row)
    assert decision['stage_passed'] and not decision['stop'] and not decision['all_stages_passed']
    assert c.choose([dict(update=0),row],state['accepted_by_stage'])['accepted_candidate'] is row
    high=next(r for r in row['evaluations'] if r['case']['command']==[-4.,0.,.34])
    high['per_env'][0]['accepted']=False
    _,decision=c.advance(spec['motion_height_dr']['state'],row)
    assert not decision['direction_passed']['backward'] and not decision['stage_passed']
    assert decision['direction_passed']['forward'] and not decision['stop']
    low['per_env'][0]['checks']['contact']=False
    assert not c.low_safety_passed(low)


def test_wrong_matrix_unknown_or_duplicate_cannot_confirm_primary():
    s=c.effective_spec(fixture(),0);r=candidate(s)
    r['evaluations'][0]['per_env'][0]['accepted']=None
    _,decision=c.advance(s['motion_height_dr']['state'],r)
    assert not decision['evidence_valid'] and not decision['stage_passed']
    r=candidate(s);r['evaluations'][-1]['case']=r['evaluations'][0]['case']
    _,decision=c.advance(s['motion_height_dr']['state'],r)
    assert not decision['evidence_valid']


def test_only_source_qualified_nominal034_gets_teacher_constraint():
    from native_stand.students.motion_source_retention import add_metadata,KEY
    from native_stand.recovery.controlled_ppo import KEY as VALID
    cfg=c.retention(c.SOURCE_SHA)
    assert [0.,4.] not in cfg['commands'] and [-4.,0.] not in cfg['commands']
    x=torch.zeros(5,31);x[:,24]=torch.tensor([.68,.68,.64,.68,.68])
    obs={VALID:torch.ones(5,1,dtype=torch.bool),'student_actor':x}
    targets=torch.tensor([[2.,0.],[2.,0.],[0.,0.],[0.,4.],[0.,0.]])
    out=add_metadata(obs,targets,cfg,variants=torch.tensor([0,3,0,0,0]))
    assert (out[KEY].flatten()>=0).tolist()==[True,False,False,False,True]


def test_recipe_width_is_explicit_and_old_512_dispatch_unchanged():
    from native_stand.students.training_recipe import validate_instance
    from native_stand.students.motion_height_pd import recipe as old
    r=c.recipe('a'*64);assert validate_instance(r,'a'*64)==r
    assert r['network_architecture']['actor_hidden_dims']==[2048,1024,512]
    assert old('0be977b308cfe493fdddee54823e21c713508fa582ce1618ab34e57b13fbc619',10000)['network_architecture']['actor_hidden_dims']==[512,256,128]
    r['network_architecture']['actor_hidden_dims']=[512,256,128]
    with pytest.raises(ValueError):validate_instance(r,'a'*64)


def test_high_speed_reward_density_has_signed_nonzero_correction_in_tail():
    audit=c.learning_signal_audit(dict(source_sha256='a'*64))
    rows=[r for r in audit['curves'] if r['term'].startswith('highspeed_')]
    assert len(rows)==72
    assert all(r['error']*r['derivative_per_physical_error']<0 for r in rows)
    assert c.recipe('a'*64)['smoke_wiring']['version']=='motion31_mcu312_sample_wiring5_v1'


def test_restored_wide_adam_and_critic_support_real_ppo_update(tmp_path):
    from tensordict import TensorDict
    from native_stand.recovery.controlled_ppo import KEY
    from native_stand.students.motion_source_retention import KEY as RET
    from native_stand.students.model import Student,load
    from native_stand.students.motion_mcu_privileged import network,VERSION
    from native_stand.students.actor_widening import expand_mcu_checkpoint
    from native_stand.students.ppo import assemble
    from native_stand.students.motion_course_requests import restore
    from native_stand.students.resume import rng_state
    a=Student(True,version='multiskill_mcu_v2',network_architecture=network())
    v=Student(False,version='multiskill_mcu_v2',critic_privileged_version=VERSION,network_architecture=network())
    a.log_std.requires_grad_(False)
    opt=torch.optim.Adam([*a.parameters(),*v.parameters()],lr=3e-5)
    (a.mean(torch.zeros(3,31)).square().mean()+v(torch.zeros(3,312)).square().mean()).backward();opt.step()
    source=dict(actor=a.state_dict(),critic=v.state_dict(),optimizer=opt.state_dict(),network_architecture=network(),
        critic_privileged_version=VERSION,interface_version='multiskill_mcu_v2',role='MotionZeroStudent',
        spec_sha256='b'*64,asset_sha256='c'*64,training_recipe={},training_rng=rng_state(),
        reward_refinement_state=[],fixed_std=True)
    wide=expand_mcu_checkpoint(source,source_checkpoint_sha256=c.SOURCE_SHA,width=2048)
    wp=tmp_path/'wide.pt';tp=tmp_path/'teacher.pt';torch.save(wide,wp)
    # The teacher file is an initialization-free Actor checkpoint; no training
    # recipe is claimed in this synthetic CPU assembly fixture.
    source['training_recipe']={'network_architecture':network()};torch.save(source,tp)
    from native_stand.students.data import sha
    r=c.recipe(sha(wp));r.pop('source_policy_retention')
    # Keep this test focused on real PPO/restoration, with a validated recipe
    # and the actual source teacher covered by the independent metadata tests.
    r=c.recipe(sha(wp));r['continuation_teacher']['checkpoint']=str(tp)
    r['continuation_teacher']['sha256']=sha(tp)
    r['source_policy_retention']['source_sha256']=sha(tp)
    import native_stand.students.training_recipe as recipes
    import native_stand.students.motion_source_retention as retention
    # Production validators remain strict. Synthetic source identity has no
    # filesystem9995 provenance; patch just identity validation in this fixture.
    original=recipes.validate_instance;original_ret=retention.validate
    try:
        recipes.validate_instance=lambda value,_:value
        retention.validate=lambda value,_:value
        obs=TensorDict({'student_actor':torch.zeros(4,31),'student_critic':torch.zeros(4,312),
                       KEY:torch.ones(4,1,dtype=torch.bool),RET:torch.zeros(4,1,dtype=torch.long)},batch_size=[4])
        ppo=assemble(wp,obs,steps_per_env=2,recipe=r)
        s=dict(mcu_execution=True,motion_execution_robustness=True,network_architecture=wide['network_architecture'],
            student_training=r,curriculum=dict(stages=[c.stage_config(envs=64)]))
        restore(ppo,wide,s,'smoke')
        assert ppo.actor.mlp[0].weight.shape==(2048,31) and ppo.critic_control_only
        assert len(ppo.optimizer.state)==len(opt.state)
        with torch.inference_mode():
            for _ in range(2):
                ppo.act(obs);ppo.process_env_step(obs,torch.ones(4),torch.zeros(4,dtype=torch.bool),{})
            ppo.compute_returns(obs)
        loss=ppo.update()
        assert all(torch.isfinite(p).all() for model in (ppo.actor,ppo.critic) for p in model.parameters())
        assert loss and ppo.mask_summary['actor_optimizer_steps']>0
    finally:
        recipes.validate_instance=original;retention.validate=original_ret
