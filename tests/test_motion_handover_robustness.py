"""Portable behavioral checks of the new public MCU/PPO extension."""
FROZEN_TEST_SCOPE = 'PORTABLE'

import copy
from types import SimpleNamespace

import pytest
import torch

from native_stand.students import motion_handover_robustness as course
from native_stand.students.motion_handover_runtime import sample_pd_age, subclass


def test_same_reward_interface_and_strict_recipe():
    from native_stand.students.motion_posture_preparation import recipe as old
    from native_stand.students.motion_posture_design import SOURCE_SHA
    from native_stand.students.training_recipe import validate_instance
    r = course.recipe(course.SOURCE_SHA)
    before = old(SOURCE_SHA)
    for key in ('reward', 'timing_configuration', 'posture_reward', 'network_architecture',
                'physical_std_leg_rad', 'physical_std_wheel_rad_s', 'BC_coefficient'):
        assert r[key] == before[key]
    assert validate_instance(r, course.SOURCE_SHA) == r
    with pytest.raises(ValueError): validate_instance(dict(r, BC_coefficient=.1), course.SOURCE_SHA)


def test_per_env_age_stream_no_global_rng_and_reset_order_independent():
    state = torch.get_rng_state().clone()
    ids, episodes = torch.tensor([5, 2, 10]), torch.tensor([2, 20, 7])
    a = sample_pd_age(ids, episodes)
    assert torch.equal(a[[2, 0, 1]], sample_pd_age(ids[[2, 0, 1]], episodes[[2, 0, 1]]))
    sample_pd_age(torch.tensor([100]), torch.tensor([999]))
    assert torch.equal(a, sample_pd_age(ids, episodes))
    assert torch.equal(state, torch.get_rng_state())
    values=sample_pd_age(torch.arange(200), torch.ones(200,dtype=torch.long))
    assert int(values.min()) == 8 and int(values.max()) == 20


def test_nominal_retention_excludes_handover_even_same_zero_command():
    from native_stand.students.motion_source_retention import add_metadata,KEY
    from native_stand.recovery.controlled_ppo import KEY as VALID
    config = course.retention(course.SOURCE_SHA)
    obs={VALID:torch.ones(4,1,dtype=torch.bool)}
    result=add_metadata(obs,torch.zeros(4,2),config,variants=torch.tensor([0,1,2,3]))
    assert result[KEY].flatten().tolist() == [0,-1,-1,-1]


def test_variant_and_case_identities_survive_same_command():
    s=course.stage_config(0,4096)
    assert sum(s['slot_counts']) == 4096 and s['mcu_variants'][:2] == [3,0]
    assert s['cohort_keys'][0] != s['cohort_keys'][1]
    template=dict(kind='student',role='MotionZeroStudent',checkpoint='x',sha256='a'*64,
                  family='motion_native_normal',seed=1,yaw_rad=0.,seconds=31.52)
    rows=course.cases({'student_evaluation':{'cases':[template]}})
    assert len(rows)==30 and len({r['case_id'] for r in rows})==30
    assert [r['_execution_uncertainty']['PD_age_ms'] for r in rows[-6:]]==[8,8,14,14,20,20]
    from native_stand.students.training import course_baseline
    records=[{'case':r,'per_env':[]} for r in rows]
    spec={'mcu_execution':True,'motion_execution_robustness':True}
    for row,record in zip(rows[-6:],records[-6:]):
        assert course_baseline(spec,row,records) is record
    with pytest.raises(ValueError):course_baseline(spec,rows[-1],records+[records[-1]])


def fake_env_class():
    class Parent:
        def __init__(self):
            self.device='cpu'; self.num_envs=64
            self.robot=SimpleNamespace(joint_names=list(range(6)),data=SimpleNamespace(
                root_state_w=SimpleNamespace(torch=torch.zeros(64,13)),
                joint_pos=SimpleNamespace(torch=torch.zeros(64,6)),joint_vel=SimpleNamespace(torch=torch.zeros(64,6))))
            self.fsm=SimpleNamespace(mode=torch.ones(64,dtype=torch.long))
            self.initialization_checked=torch.ones(64,dtype=torch.bool)
            self.student_obs={'student_actor':torch.zeros(64,31)}
            self.student_actor_valid=torch.ones(64,dtype=torch.bool)
            self.rise_hold=SimpleNamespace(motion=torch.ones(64,dtype=torch.bool),
                observe=lambda *args,active,**kw: active.clone())
            self.motion_timing=SimpleNamespace(feedback=torch.full((64,),10),variant=torch.zeros(64,dtype=torch.long),
                                              adopted=torch.randn(64,6),queue=torch.randn(64,14,6))
        def _pre_physics_step(self,actions):
            self.executed=actions.clone()
            self.student_actor_valid.fill_(True)
        def _get_dones(self):return torch.zeros(64,dtype=torch.bool),torch.zeros(64,dtype=torch.bool)
    return Parent


def test_actual_prefix_action_mask_and_handover_preserve_queue_state(monkeypatch):
    from native_stand.students.model import Student
    from native_stand.recovery.domain import Mode
    actor=Student(True,version='multiskill_mcu_v2')
    monkeypatch.setattr('native_stand.students.model.load',lambda *args:(copy.deepcopy(actor),{}))
    env=subclass(fake_env_class(),{'source_checkpoint':'x','source_sha256':course.SOURCE_SHA,'version':course.VERSION})()
    env.fsm.mode.fill_(int(Mode.RECOVERY))
    env.prefix_remaining[0]=2
    before=[v.clone() for v in (env.robot.data.root_state_w.torch,env.motion_timing.adopted,env.motion_timing.queue)]
    request=torch.full((64,6),.4)
    assert not env.actor_valid_pre()[0]
    env._pre_physics_step(request)
    assert torch.equal(env.executed[0],env.handover_teacher.mean(env.student_obs['student_actor'])[0])
    assert not env.student_actor_valid[0] and env.course_owner_snapshot[0] == 2
    assert not env.rise_hold.observe(active=torch.ones(64,dtype=torch.bool))[0]
    env._pre_physics_step(request)
    assert env.actor_valid_pre()[0] and not env.student_actor_valid[0]
    env._pre_physics_step(request)
    assert torch.equal(env.executed[0],request[0]) and env.student_actor_valid[0]
    assert env.rise_hold.observe(active=torch.ones(64,dtype=torch.bool))[0]
    assert env.course_handoff_snapshot[0] and env.robust_handoffs == 1
    assert all(torch.equal(a,b) for a,b in zip(before,(env.robot.data.root_state_w.torch,env.motion_timing.adopted,env.motion_timing.queue)))
    assert not any(p.requires_grad for p in env.handover_teacher.parameters())


def test_user_hold_blocks_public_recipe_before_any_simulation(tmp_path):
    from native_stand.students.motion_mcu_course import verify
    marker=tmp_path/'submission_hold.json'
    marker.write_text('{"status":"USER_HOLD_BEFORE_SUBMISSION"}')
    with pytest.raises(ValueError,match='user requested no training submission'):
        verify({'motion_execution_robustness':{'version':course.VERSION},'motion_training':{'spec_path':str(tmp_path/'spec.json')}})
