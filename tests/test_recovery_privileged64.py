FROZEN_TEST_SCOPE='PORTABLE'
import copy,importlib,sys
from types import SimpleNamespace
import pytest,torch
from tensordict import TensorDict
from native_stand.students.recovery_privileged import VERSION,recipe,contract,HoldExtrema,critic_input
from native_stand.students.model import Student
from native_stand.students.recovery_task import Certificate
from native_stand.recovery.controlled_ppo import KEY


def test_named64_input_actor_isolated_hold_extrema_partial_reset():
    n=2;actor=torch.zeros(n,31);clock=Certificate(n,'cpu');extrema=HoldExtrema(n,'cpu')
    z=lambda k:torch.zeros(n,k)
    h=torch.ones(n)*.34;contacts=torch.ones(n,2)*3
    extrema.observe(h,torch.ones(n)*.05,contacts,torch.ones(n)*.001,torch.tensor([True,False]))
    x=critic_input(actor,h,z(3),contacts,torch.ones(n)*.03,z(6),torch.ones(n,2)*.3,z(2),h,clock,extrema)
    assert x.shape==(2,64) and torch.equal(x[:,:31],actor)
    assert x[0,58]==pytest.approx(.68) and x[1,58]==0
    extrema.reset(torch.tensor([0]));assert not extrema.seen.any() and not extrema.values.any()
    assert contract()['critic_dim']==64
    assert Student(False,version='multiskill_mcu_v2').mlp[0].in_features==40
    assert Student(False,version='multiskill_mcu_v2',critic_privileged_version=VERSION).mlp[0].in_features==64


def test_public_assemble_actual_cpu_updates_new_critic_lr_mask_and_std(tmp_path):
    from native_stand.students.ppo import assemble
    from native_stand.students.training_recipe import validate_instance
    torch.manual_seed(12);a=Student(version='multiskill_mcu_v2');cp=tmp_path/'actor.pt'
    torch.save({'interface_version':'multiskill_mcu_v2','role':'RecoveryStudent','actor':a.state_dict()},cp)
    r=recipe();assert validate_instance(r,r['source_actor_sha256'])==r
    bad=copy.deepcopy(r);bad['learning_rate']=3e-5
    with pytest.raises(ValueError):validate_instance(bad,bad['source_actor_sha256'])
    obs=TensorDict({'student_actor':torch.randn(4,31),'student_critic':torch.randn(4,64),KEY:torch.ones(4,1,dtype=torch.bool)},batch_size=[4])
    p=assemble(cp,obs,steps_per_env=8,recipe=r)
    assert p.optimizer.param_groups[0]['lr']==1e-4 and p.critic.mlp[0].in_features==64 and not p.optimizer.state
    before={k:v.clone() for k,v in p.actor.state_dict().items()}
    def update(enabled,valid=True):
        p.actor_update_enabled=enabled
        with torch.inference_mode():
            for i in range(8):
                o=obs.clone();o[KEY][:]=valid;raw=p.act(o)
                torch.testing.assert_close(p.transition.actions_log_prob,p.actor.log_prob(o['student_actor'],raw))
                p.process_env_step(o,torch.arange(4,dtype=torch.float32)*.01,torch.tensor([i==7,False,False,False]),{})
            p.compute_returns(o)
        p.update()
    update(False)
    assert all(torch.equal(v,p.actor.state_dict()[k]) for k,v in before.items())
    update(True)
    assert any(not torch.equal(v,p.actor.state_dict()[k]) for k,v in before.items())
    assert torch.equal(before['log_std'],p.actor.log_std)
    saved={k:v.clone() for k,v in p.actor.state_dict().items()};update(True,False)
    assert all(torch.equal(v,p.actor.state_dict()[k]) for k,v in saved.items())
    assert p.mask_summary['BC_updates']==0


@pytest.mark.parametrize('reward_version,field_count',[('student_recovery_broad_narrow_reward_v1',11),('student_recovery_large_stop_reward_v1',13)])
def test_real_training_env_reward_branch_returns_named11_nonzero(monkeypatch,reward_version,field_count):
    # No Isaac import: replace physical superclass, exercise actual domain/runtime adapter.
    monkeypatch.setitem(sys.modules,'native_stand.recovery.runtime',SimpleNamespace(RecoveryEnv=object))
    monkeypatch.delitem(sys.modules,'native_stand.students.runtime',raising=False)

    mod=importlib.import_module('native_stand.students.runtime')
    monkeypatch.setattr(mod.StudentEnv,'_get_dones',lambda self:(torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool)))
    e=mod.TrainingStudentEnv.__new__(mod.TrainingStudentEnv);e.task_ready=True;e.num_envs=2;e.device='cpu';e.ids=list(range(6))
    from native_stand.students.recovery_task import configuration
    from native_stand.students.recovery_broad_narrow_reward import FIELDS
    if field_count==13:
        from native_stand.students.recovery_large_training import FIELDS
    e.student_reward_configuration=configuration(reward_version);e.rise_hold=Certificate(2,'cpu')
    e._motion_state=lambda:(torch.zeros(2,3),torch.tensor([[0.,-1.,0.]]).repeat(2,1))
    e.vx=e.vy=e.yaw=torch.zeros(2);e.student_h0=torch.ones(2)*.168
    e.student_actor_valid=torch.ones(2,dtype=torch.bool);e.student_previous_physical=torch.zeros(2,6)
    e.executed_leg=torch.zeros(2,4);e.executed_wheel=torch.zeros(2,2)
    e.recovery_snapshot=torch.zeros(2,39);e.recovery_snapshot[:,6]=.34;e.recovery_snapshot[:,10:12]=3;e.recovery_snapshot[:,13]=.03;e.recovery_snapshot[:,29]=1
    e.standing_snapshot=torch.zeros(2,10);e.recovery_hold_extrema=HoldExtrema(2,'cpu')
    e.robot=SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.zeros(2,6)),applied_torque=SimpleNamespace(torch=torch.zeros(2,6))))
    mod.TrainingStudentEnv._get_dones(e)
    assert e.student_reward_result['fields']==FIELDS and e.student_reward_result['raw'].shape==(2,field_count)
    torch.testing.assert_close(e.student_reward_result['total'],torch.full((2,),.07))
    assert e.student_event_snapshot.shape==(2,5)
    monkeypatch.delitem(sys.modules,'native_stand.students.runtime',raising=False)


def test_actual64_getter_reset_before_first_physics_has_no_clearance_cache(monkeypatch):
    from native_stand.students.interface import History
    from native_stand.students.mcu_interface import PolicyClock
    from native_stand.recovery.domain import Mode
    class Base:
        def _get_observations(self):return {}
    monkeypatch.setitem(sys.modules,'native_stand.recovery.runtime',SimpleNamespace(RecoveryEnv=Base))
    monkeypatch.setitem(sys.modules,'isaaclab.utils.math',SimpleNamespace(quat_apply_inverse=lambda q,v:v))
    monkeypatch.delitem(sys.modules,'native_stand.students.runtime',raising=False)
    mod=importlib.import_module('native_stand.students.runtime');e=mod.StudentEnv.__new__(mod.StudentEnv)
    e.num_envs=2;e.device='cpu';e.student_ready=True;e.task_ready=True;e.interface_version='multiskill_mcu_v2'
    e.student_role='RecoveryStudent';e.student_command_height=.34;e.step_dt=.01;e.ids=list(range(6));e.wheel_body_ids=[0,1]
    e.critic_privileged_version=VERSION;e.fsm=SimpleNamespace(mode=torch.tensor([int(Mode.INITIALIZING)]*2))
    e.reference=torch.zeros(2,6);e.executed_leg=torch.zeros(2,4);e.executed_wheel=torch.zeros(2,2)
    e.student_history=History(2);e.mcu_clock=PolicyClock(2);e.episode_length_buf=torch.zeros(2,dtype=torch.long)
    e.student_obs=None;e.student_h0=torch.zeros(2);e.root_height=torch.ones(2)*.168;e.contact_force=torch.zeros(2,2)
    e.recovery_config={'standing_task':{'hip_anchors_root_m':[[0.,0.,0.],[0.,0.,0.]]}}
    e.recovery_length_valid=torch.zeros(2,dtype=torch.bool);e.recovery_length_time=torch.zeros(2,dtype=torch.float64)
    e.recovery_previous_length=torch.zeros(2,2);e.recovery_length_rate=torch.zeros(2,2)
    e.rise_hold=Certificate(2,'cpu');e.recovery_hold_extrema=HoldExtrema(2,'cpu')
    d=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.zeros(2,6)),joint_vel=SimpleNamespace(torch=torch.zeros(2,6)),
        root_pos_w=SimpleNamespace(torch=torch.zeros(2,3)),root_quat_w=SimpleNamespace(torch=torch.zeros(2,4)),
        root_lin_vel_w=SimpleNamespace(torch=torch.zeros(2,3)),body_pos_w=SimpleNamespace(torch=torch.zeros(2,2,3)))
    e.robot=SimpleNamespace(data=d);e._motion_state=lambda:(torch.zeros(2,3),torch.tensor([[0.,-1.,0.]]).repeat(2,1))
    calls=[]
    e._geometry_gaps=lambda:(calls.append('current') or torch.zeros(2,1),torch.ones(2)*.012)
    assert not hasattr(e,'chassis_clearance')
    obs=e._get_observations()
    assert obs['student_critic'].shape==(2,64) and calls==['current']
    torch.testing.assert_close(obs['student_critic'][:,37],torch.full((2,),.12))
    # Repeated getter doesn't manufacture a rate or access a stale post-step cache.
    e._get_observations();assert not e.recovery_length_rate.any()
    monkeypatch.delitem(sys.modules,'native_stand.students.runtime',raising=False)
