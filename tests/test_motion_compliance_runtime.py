"""CPU reward, PPO and evaluation checks; no robot simulation in these tests."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy
import numpy as np
import pytest
import torch
from native_stand.students.motion_compliance_training import CRITIC,declaration


def test_new3000_budget_is_inclusive_and_unlimited_wall():
    d=declaration(updates=3000);b=d['budget_proposal']
    assert b['online_updates']==3000 and b['smoke']==12 and b['formal']==2988
    assert b['wall_s'] is None and b['raw_rollout_samples']==3072000


def reward_fixture():
    from native_stand.students.motion_tracking_reward import configuration,PARALLEL_VERSION
    from native_stand.students.motion_antioscillation import configuration as anti
    source=dict(reward=configuration(PARALLEL_VERSION),motion_timing=dict(configuration=anti()))
    z=torch.zeros(1);q=torch.tensor([[.2,.2,0.,-.2,.2,0.]])
    kw=dict(command=torch.tensor([[0.,0.,.34]]),target=torch.zeros(1,2),velocity=torch.zeros(1,3),
        angular=torch.zeros(1,3),height=torch.tensor([.34]),angles=torch.zeros(1,2),contacts=torch.full((1,2),20.),
        q=q,dq=torch.zeros(1,6),motor=torch.zeros(1,6),request=torch.zeros(1,6),previous_motor=torch.zeros(1,6),
        adopted=q.clone(),previous_target=q.clone(),older_target=q.clone(),lengths=torch.full((1,2),.34),
        wheel_offset=z.clone(),swing_difference=z.clone(),controlled=True,failed=False,history_valid=torch.tensor([True]),
        spin=dict(tangent_velocity_w=torch.zeros(1,2,2),rolling_speed_forward_mps=torch.zeros(1,2),
            geometry_valid=torch.ones(1,2,dtype=torch.bool),center_world=torch.tensor([[[.22,0.,.06],[-.22,0.,.06]]]),
            forward_world=torch.tensor([[0.,1.,0.]])))
    return source,kw


def test_reward_density_dt_event_and_posture_gradients():
    from native_stand.students.motion_compliance_reward import compute
    source,kw=reward_fixture();normal=compute(source,**kw)
    assert torch.allclose(normal['total'],normal['density'].sum(-1)*.01)
    fail=compute(source,**{**kw,'failed':True})
    assert fail['total'].item()==pytest.approx(-1.)
    bad=copy.deepcopy(kw);bad['angles']=torch.tensor([[0.,.10]],requires_grad=True)
    result=compute(source,**bad);i=result['fields'].index('roll_excess')
    derivative=torch.autograd.grad(result['density'][0,i],bad['angles'])[0][0,1]
    assert derivative<0 and result['total']<normal['total']
    changed=copy.deepcopy(kw);changed['motor'][:]=20.
    high=compute(source,**changed)
    assert high['density'][0,high['fields'].index('effort')]<normal['density'][0,normal['fields'].index('effort')]


def test_small_actor_new_critic64_controlled_ppo_warmup_and_raw_gaussian():
    from tensordict import TensorDict
    from rsl_rl.storage import RolloutStorage
    from native_stand.students.model import Student
    from native_stand.recovery.controlled_ppo import ControlledPPO,KEY
    torch.manual_seed(4)
    actor=Student(True,version='multiskill_mcu_v2');actor.log_std.requires_grad_(False)
    critic=Student(False,version='multiskill_mcu_v2',critic_privileged_version=CRITIC)
    obs=TensorDict({'student_actor':torch.randn(8,31),'student_critic':torch.randn(8,64),KEY:torch.ones(8,1,dtype=torch.bool)},batch_size=[8])
    st=RolloutStorage('rl',8,4,obs,[6],device='cpu')
    ppo=ControlledPPO(actor,critic,st,learning_rate=3e-5,schedule='fixed',num_learning_epochs=1,
        num_mini_batches=2,use_clipped_value_loss=False,device='cpu');ppo.critic_control_only=True
    before={k:v.clone() for k,v in actor.state_dict().items()}
    for actor_enabled in (False,True):
        ppo.actor_update_enabled=actor_enabled
        for _ in range(4):
            with torch.no_grad():raw=ppo.act(obs)
            assert torch.allclose(actor.get_output_log_prob(raw),actor.log_prob(obs['student_actor'],raw))
            ppo.process_env_step(obs,torch.arange(8).float()*.01,torch.zeros(8,dtype=torch.bool),{})
        ppo.compute_returns(obs);ppo.update()
        changed=any(not torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
        assert changed is actor_enabled
    assert actor.log_std.requires_grad is False


def test_evaluation_yaw_uses_angular_rate_and_keeps_failed_window():
    from native_stand.students.motion_compliance_eval import summarize
    rows=[]
    for i in range(3000):
        rows.append(dict(t=(i+1)*.01,controlled=True,velocity=np.zeros(3),yaw=.5,command=np.array([0.,.5,.34]),
            height=.34,angles=np.zeros(2),contact=np.ones(2)*20,lengths=np.ones(2)*.34,
            encoder_dq=np.zeros(6),load_dq=np.zeros(6),swing=0.,wheel_offset=0.,saturated=np.zeros(6),slip=np.zeros(2)))
    case=dict(command=[0.,.5,.34],profile='R')
    r=summarize(rows,dict(timeout=True,reason=None),case)
    assert r['metrics']['tracking_MAE']==0 and r['metrics']['response_mean']==.5 and r['accepted']
    short=summarize(rows[:100],dict(timeout=False,reason='fallen'),case)
    assert not short['accepted'] and not short['survival30']


def test_full_state_checkpoint_has_explicit_new_semantics_and_roundtrips(tmp_path):
    from types import SimpleNamespace as S
    from native_stand.students.model import Student,load
    from native_stand.students.motion_compliance_owner import save
    actor=Student(True,version='multiskill_mcu_v2');critic=Student(False,version='multiskill_mcu_v2',critic_privileged_version=CRITIC)
    optimizer=torch.optim.Adam([*actor.parameters(),*critic.parameters()],lr=3e-5)
    ppo=S(actor=actor,critic=critic,optimizer=optimizer)
    record=save(tmp_path,dict(source_sha256='a'*64,inputs={}),ppo,12,dict(stage=0),'b'*64)
    restored,d=load(record['checkpoint'],'MotionZeroStudent')
    x=torch.randn(2,31)
    assert torch.equal(actor(x),restored(x)) and d['critic_privileged_version']==CRITIC
    assert 'optimizer' in d and 'training_rng' in d and d['update']==12
