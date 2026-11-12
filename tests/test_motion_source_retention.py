FROZEN_TEST_SCOPE = 'PORTABLE'
import copy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.storage import RolloutStorage

from native_stand.recovery.controlled_ppo import ControlledPPO, KEY as VALID
from native_stand.students.model import Student
from native_stand.students.motion_source_retention import (
    KEY, OLD_COMMANDS, SourcePolicyRetention, add_metadata, cohort_ids,
    declaration, ppo_class, validate,
)

SOURCE = 'a' * 64


def observations():
    x = torch.randn(8, 31) * .05
    x[:, 22:25] = torch.tensor([0., 0., .68])
    x[:, 25:28] = torch.tensor([1., 0., 0.])
    targets = torch.tensor(OLD_COMMANDS + [[0., .5], [0., -.5], [0., .5]])
    obs = TensorDict({'student_actor': x, 'student_critic': torch.randn(8, 49)*.05,
                     VALID: torch.ones(8, 1, dtype=torch.bool)}, batch_size=[8])
    return add_metadata(obs, targets, declaration(SOURCE)), targets


def actor():
    model = Student(True, version='multiskill_mcu_v2')
    model.log_std.requires_grad_(False)
    return model


def build(enabled=True):
    obs, _ = observations()
    a = actor()
    c = Student(False, version='multiskill_mcu_v2', critic_privileged_version='motion_privileged49_v1')
    st = RolloutStorage('rl', 8, 4, obs, [6], device='cpu')
    cls = ppo_class() if enabled else ControlledPPO
    ppo = cls(a, c, st, learning_rate=3e-5, schedule='fixed', num_learning_epochs=2,
              num_mini_batches=2, entropy_coef=.001, value_loss_coef=.5,
              use_clipped_value_loss=False, device='cpu')
    if enabled:
        ppo.source_retention = SourcePolicyRetention(a, declaration(SOURCE))
    return ppo, obs


def fill(ppo, obs):
    for step in range(4):
        current = obs.clone()
        current['student_actor'] += torch.randn_like(current['student_actor'])*.001
        ppo.act(current)
        ppo.process_env_step(current, torch.randn(8)*.05,
                             torch.zeros(8, dtype=torch.bool), {})
    ppo.compute_returns(current)


def test_declaration_default_off_and_strict_motion_version():
    from native_stand.students.motion_training import declaration as recipe
    from native_stand.students.training_recipe import validate_instance
    from native_stand.students.motion_tracking_reward import QUALITY_VERSION
    r = recipe(SOURCE, reward_version=QUALITY_VERSION)
    assert 'source_policy_retention' not in r
    original = copy.deepcopy(r)
    r.update(course_version='motion31_yaw_progression_v1', course_stage=0)
    r['source_policy_retention'] = declaration(SOURCE)
    assert validate_instance(r, SOURCE) == r
    assert validate_instance(original, SOURCE) == original
    r['course_version'] = 'motion31_forward_progression_v1'
    with pytest.raises(ValueError, match='yaw course'):
        validate_instance(r, SOURCE)
    for value in (0, -1, float('inf'), float('nan'), True, '0.01'):
        with pytest.raises(ValueError):
            declaration(SOURCE, value)
    with pytest.raises(ValueError):
        validate(declaration(SOURCE), 'b'*64)
    c = declaration(SOURCE); c['unknown'] = 1
    with pytest.raises(ValueError):
        validate(c, SOURCE)


def test_episode_target_not_zero_ramp_and_invalid_reset():
    cfg = declaration(SOURCE)
    targets = torch.tensor([[0., .5], [.5, 0.], [0., 0.], [0., -.5], [1., 0.]])
    valid = torch.tensor([True, True, True, True, False])
    assert cohort_ids(targets, valid, cfg).squeeze(-1).tolist() == [-1, 1, 0, -1, -1]
    obs, _ = observations()
    # All instantaneous command columns are zero: yaw ramp must remain excluded.
    assert obs[KEY].squeeze(-1).tolist() == [0, 1, 2, 3, 4, -1, -1, -1]
    assert add_metadata(obs, targets, None) is obs


def test_source_proof_requires_all_old_commands_no_yaw_teacher_prerequisite():
    from native_stand.students.motion_source_retention import verify_source_records
    records = [{'case':{'command':c+[.34]},'per_env':[{'env':i,'accepted':True} for i in range(8)]} for c in OLD_COMMANDS]
    assert verify_source_records(records, declaration(SOURCE))
    with pytest.raises(ValueError,match='8/8'):
        verify_source_records(records[:-1], declaration(SOURCE))
    invalid = copy.deepcopy(records);invalid[0]['per_env'][0]['accepted'] = False
    with pytest.raises(ValueError,match='8/8'):
        verify_source_records(invalid, declaration(SOURCE))
    invalid = copy.deepcopy(records);invalid[0]['per_env'][0]['env'] = 1
    with pytest.raises(ValueError,match='8/8'):
        verify_source_records(invalid, declaration(SOURCE))


def test_equal_weights_zero_loss_no_teacher_grad_or_sampling_side_effects():
    torch.manual_seed(10)
    a = actor(); obs, _ = observations()
    regularizer = SourcePolicyRetention(a, declaration(SOURCE))
    a(obs, stochastic_output=True)
    distribution = a._current_distribution
    rng = torch.get_rng_state().clone()
    loss = regularizer.loss(a, obs, obs[VALID].squeeze(-1))
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(rng, torch.get_rng_state())
    assert a._current_distribution is distribution and a.training
    assert not regularizer.teacher.training
    assert all(not p.requires_grad and p.grad is None for p in regularizer.teacher.parameters())
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in a.parameters())


def test_perturbation_finite_gradient_and_small_descent_towards_source():
    torch.manual_seed(8)
    a = actor(); obs, _ = observations()
    regularizer = SourcePolicyRetention(a, declaration(SOURCE))
    with torch.no_grad():
        a.mlp[-1].bias[0] += .02
    loss = regularizer.loss(a, obs, obs[VALID].squeeze(-1))
    before = loss.item(); assert before > 0
    loss.backward()
    assert a.mlp[-1].bias.grad[0] > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.parameters())
    with torch.no_grad():
        for p in a.parameters():
            if p.grad is not None:
                p -= p.grad*1e-4
    assert regularizer.loss(a, obs, obs[VALID].squeeze(-1)).item() < before


def test_only_old_rows_contribute_and_balanced_cohort_aggregation():
    torch.manual_seed(3)
    a = actor(); obs, _ = observations()
    regularizer = SourcePolicyRetention(a, declaration(SOURCE))
    with torch.no_grad():
        a.mlp[-1].bias[0] += .01
    obs['student_actor'].requires_grad_(True)
    loss = regularizer.loss(a, obs, obs[VALID].squeeze(-1))
    loss.backward()
    assert obs['student_actor'].grad[5:].abs().max() == 0
    assert obs['student_actor'].grad[:5].abs().max() > 0
    # Repeating one identical old cohort should not increase its loss weight.
    duplicate = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    repeated = obs.detach()[duplicate]
    torch.testing.assert_close(regularizer.loss(a, repeated, repeated[VALID].squeeze(-1)), loss)
    only_new = obs.detach()[5:]
    assert regularizer.loss(a, only_new, only_new[VALID].squeeze(-1)).item() == 0
    with pytest.raises(ValueError, match='metadata'):
        regularizer.loss(a, obs.exclude(KEY), obs[VALID].squeeze(-1))


def test_real_synthetic_cpu_update_records_loss_and_teacher_outside_optimizer():
    torch.manual_seed(5)
    ppo, obs = build()
    teacher_before = copy.deepcopy(ppo.source_retention.teacher.state_dict())
    teacher_ids = {id(p) for p in ppo.source_retention.teacher.parameters()}
    optimizer_ids = {id(p) for g in ppo.optimizer.param_groups for p in g['params']}
    assert not teacher_ids & optimizer_ids
    fill(ppo, obs)
    assert ppo.storage.observations[KEY].shape == (4, 8, 1)
    assert torch.equal(ppo.storage.observations[KEY][:, 5:], torch.full((4, 3, 1), -1))
    ppo.update()
    row = ppo.mask_summary['source_policy_retention']
    assert row['eligible_rollout_transitions'] == 20
    assert row['selected_sample_uses_over_epochs'] == 40
    assert row['selected_by_cohort_over_epochs'] == [8]*5
    assert row['weighted_loss_mean'] >= 0 and row['status'] == 'COLLECTED'
    assert ppo.mask_summary['actor_optimizer_steps'] == 4
    assert all(torch.equal(v, ppo.source_retention.teacher.state_dict()[k]) for k, v in teacher_before.items())


def test_critic_warmup_skips_retention_and_leaves_actor_and_teacher_unchanged():
    torch.manual_seed(7)
    ppo, obs = build(); ppo.actor_update_enabled = False
    before = copy.deepcopy(ppo.actor.state_dict()); teacher = copy.deepcopy(ppo.source_retention.teacher.state_dict())
    fill(ppo, obs); ppo.update()
    assert ppo.mask_summary['source_policy_retention']['selected_sample_uses_over_epochs'] == 0
    assert ppo.mask_summary['source_policy_retention']['status'] == 'SKIPPED_ACTOR_UPDATE'
    assert all(torch.equal(v, ppo.actor.state_dict()[k]) for k, v in before.items())
    assert all(torch.equal(v, ppo.source_retention.teacher.state_dict()[k]) for k, v in teacher.items())


def test_zero_loss_hook_preserves_ppo_rng_returns_and_critic_update():
    # All rows are new yaw: retention is exactly zero, not an accidental objective change.
    torch.manual_seed(12); a, obs = build(); b, _ = build(False)
    b.actor.load_state_dict(a.actor.state_dict()); b.critic.load_state_dict(a.critic.state_dict())
    obs[KEY][:] = -1
    rng = torch.get_rng_state(); fill(a, obs); torch.set_rng_state(rng); fill(b, obs)
    assert torch.equal(a.storage.returns, b.storage.returns)
    assert torch.equal(a.storage.advantages, b.storage.advantages)
    rng = torch.get_rng_state(); a.update(); after = torch.get_rng_state(); torch.set_rng_state(rng); b.update()
    assert torch.equal(after, torch.get_rng_state())
    for left, right in zip(a.actor.parameters(), b.actor.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(a.critic.parameters(), b.critic.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_public_assembly_and_checkpoint_declaration_roundtrip(tmp_path):
    from native_stand.students.ppo import assemble
    from native_stand.students.model import load
    from native_stand.students.motion_training import declaration as recipe
    from native_stand.students.motion_tracking_reward import QUALITY_VERSION
    obs, _ = observations(); a = actor()
    source = tmp_path/'source.pt'
    torch.save({'interface_version': 'multiskill_mcu_v2', 'role': 'MotionZeroStudent',
                'actor': a.state_dict()}, source)
    r = recipe(SOURCE, reward_version=QUALITY_VERSION)
    r.update(course_version='motion31_yaw_progression_v1', course_stage=0,
             source_policy_retention=declaration(SOURCE))
    ppo = assemble(source, obs, steps_per_env=4, recipe=r)
    assert hasattr(ppo, 'source_retention')
    torch.testing.assert_close(ppo.source_retention.teacher.mean(obs['student_actor']), a.mean(obs['student_actor']))
    old = copy.deepcopy(r); old.pop('source_policy_retention')
    assert not hasattr(assemble(source, obs.exclude(KEY), steps_per_env=4, recipe=old), 'source_retention')
    with pytest.raises(ValueError, match='metadata'):
        assemble(source, obs.exclude(KEY), steps_per_env=4, recipe=r)
    data = {'interface_version':'multiskill_mcu_v2','role':'MotionZeroStudent','actor':a.state_dict(),
            'recipe_version':r['version'],'training_recipe':r,'BC_coefficient':0.,'fixed_std':True,
            'source_sha256':SOURCE,'dataset_sha256':'b'*64,'manifest_sha256':'c'*64,
            'asset_sha256':'d'*64,'spec_sha256':'e'*64,'source_code_sha256':'f'*64}
    path = tmp_path/'trained.pt';torch.save(data,path)
    with pytest.raises(ValueError,match='retention declaration'):
        load(path)
    data['source_policy_retention'] = r['source_policy_retention'];torch.save(data,path)
    restored,_ = load(path)
    torch.testing.assert_close(restored.mean(obs['student_actor']),a.mean(obs['student_actor']))


def test_fixed_retention_course_requires_explicit_version_and_keeps_warmup():
    from native_stand.students.motion_training import declaration as recipe, phase_warmup
    from native_stand.students.motion_tracking_reward import YAW_NARROW_VERSION
    from native_stand.students.motion_startup import smoke_profile
    from native_stand.students.training_recipe import validate_instance
    from native_stand.students import motion_yaw_retention_course as course
    from native_stand.students.motion_course import implementation
    r=recipe(SOURCE,reward_version=YAW_NARROW_VERSION)
    r.update(course_version=course.VERSION,course_stage=0,smoke_wiring=smoke_profile(),source_policy_retention=declaration(SOURCE))
    assert validate_instance(r,SOURCE)==r
    assert implementation(course.VERSION) is course
    assert phase_warmup(r,'smoke')==4 and phase_warmup(r,'formal')==5
    assert course.stage_config(0)['slot_counts']==[128,48,48,48,48,96,96]
    r['course_version']='motion31_yaw_width_fixed_v1'
    with pytest.raises(ValueError,match='yaw course'):validate_instance(r,SOURCE)


def test_retention_does_not_backpropagate_into_critic_or_std():
    torch.manual_seed(29);ppo,obs=build()
    with torch.no_grad():ppo.actor.mlp[-1].bias[0]+=.01
    value=ppo.source_retention.loss(ppo.actor,obs,obs[VALID].squeeze(-1));value.backward()
    assert value.item()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in ppo.actor.parameters())
    assert all(p.grad is None for p in ppo.critic.parameters())
    assert ppo.actor.log_std.grad is None
    assert all(p.grad is None for p in ppo.source_retention.teacher.parameters())


def test_retention_source_allowlist_and_assessment_unchanged():
    from native_stand.students.motion_startup import source_protocols,source_rows
    from native_stand.students import motion_yaw_retention_course as retained,motion_yaw_width_course as control
    a={'student_training':{'course_version':control.VERSION},'motion_assessment':control.protocol()}
    b={'student_training':{'course_version':retained.VERSION},'motion_assessment':retained.protocol()}
    assert source_protocols(a)==source_protocols(b)
    rows=[{'checks':{'parking':False,'survival30':True},'accepted':False}]
    assert source_rows(a,rows)==source_rows(b,rows)
