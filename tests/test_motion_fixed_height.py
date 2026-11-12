FROZEN_TEST_SCOPE = 'PORTABLE'

import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from native_stand.students import motion_fixed_height as task, motion_decoupled as base, motion_height as height


def state():
    return dict(version=task.VERSION, levels=dict(forward=3,backward=3,yaw_positive=3,yaw_negative=3,height=-1,DR=1),
                focus='height',focus_level=0,stage=0,focus_cursor=0,stage_started_update=0,last_evaluation_update=-1,
                DR_level=1,certified_DR_level=1,accepted_by_stage={},regression_streaks={},repair_queue=[])


def test_fixed_episode_commands_and_range_do_not_reintroduce_h0_or_height_switch():
    s=state()
    for level in range(5):
        scenarios=task.height_scenarios(level,s)
        assert all(.30<=r[2]<=.40 and r[2]==r[3] and r[4] is False for r in scenarios)
        assert all(r[2]!=.34 for r in scenarios) if level<3 else True
    assert {r[2] for r in task.height_scenarios(4,s)}==set(task.HEIGHTS)
    age=torch.linspace(0.,31.,3101);n=len(age)
    for h in (.30,.34,.40):
        command,rate=task.command(torch.tensor([1.,0.]).repeat(n,1),age,torch.full((n,),h),torch.full((n,),h),torch.zeros(n,dtype=torch.bool),True)
        assert torch.all(command[:,2]==h) and torch.count_nonzero(rate)==0
        assert command[1000,0] == 1.


def test_sampling_identity_includes_height_and_all_quotas_are_preserved():
    s=state();stage=task.stage_config(0,state=s)
    total={p:sum(n for n,q in zip(stage['slot_counts'],stage['cohort_purposes']) if p==q) for p in ('nominal_zero','rehearsal','focus')}
    assert total==dict(nominal_zero=614,rehearsal=1434,focus=2048)
    focus=[r for r,p in zip(stage['height_scenarios'],stage['cohort_purposes']) if p=='focus']
    assert {r[2] for r in focus}=={.32,.36}
    s['levels']['height']=3;s.update(focus='forward',focus_level=4)
    assert len(task.focus_scenarios(s))==11
    assert all(r[0]==3 and r[2]==r[3] for r in task.focus_scenarios(s))


def test_critic_migration_preserves_actor_value_and_old_adam_moments():
    from native_stand.students.model import Student
    actor=Student(True,version='multiskill_mcu_v2',network_architecture=height.network())
    old=Student(False,version='multiskill_mcu_v2',critic_privileged_version=height.CRITIC,network_architecture=height.network())
    opt=torch.optim.Adam(list(actor.parameters())+list(old.parameters()),lr=3e-5)
    x=torch.randn(4,144);old(x).square().sum().backward();opt.step()
    source=dict(actor=copy.deepcopy(actor.state_dict()),critic=old.state_dict(),optimizer=opt.state_dict(),network_architecture=height.network(),critic_privileged_version=height.CRITIC)
    new=task.expand_source(source)
    target=Student(False,version='multiskill_mcu_v2',critic_privileged_version=task.CRITIC,network_architecture=task.network())
    target.load_state_dict(new['critic'])
    assert all(torch.equal(v,new['actor'][k]) for k,v in source['actor'].items())
    assert torch.allclose(old(x),target(torch.cat((x,torch.randn(4,40)),dim=-1)),atol=1e-6)
    pid=next(k for k,v in new['optimizer']['state'].items() if v.get('exp_avg',torch.empty(0)).shape==(512,184))
    for key in ('exp_avg','exp_avg_sq'):
        assert torch.equal(new['optimizer']['state'][pid][key][:,:144],source['optimizer']['state'][pid][key])
        assert torch.count_nonzero(new['optimizer']['state'][pid][key][:,144:])==0
    with pytest.raises(ValueError):Student(False,version='multiskill_mcu_v2',critic_privileged_version=height.CRITIC,network_architecture=task.network())


def test_plant_features_use_causal_queued_commands_actual_feedback_and_applied_push():
    from native_stand.students.motion_height_runtime import plant_features
    n=2;tick=40
    t=SimpleNamespace(tick=tick,targets=torch.zeros(32,n,6),sensors=torch.zeros(32,n,12),feedback=torch.tensor([10,8]))
    for offset,value in ((1,1.),(11,2.),(21,3.)):t.targets[(tick-offset)%32]=value
    t.sensors[(35-10)%32,0]=torch.arange(12.)
    t.sensors[(35-8)%32,1]=torch.arange(12.)+1
    d=SimpleNamespace(adopted_ticks=torch.tensor([[30]*6,[-1]*6]),ids=torch.arange(n),
        params=torch.tensor([[10.,0.,10.,1.1,.45,.1,0.],[10.,0.,8.,.9,.55,.1,0.]]),
        pushed=torch.tensor([[True,False],[False,False]]))
    e=SimpleNamespace(device='cpu',motion_timing=t,motion_dr=d,actor_sensor_packets=SimpleNamespace(age=torch.tensor([10.,8.])),
        rise_hold=SimpleNamespace(episode=SimpleNamespace(steps=torch.tensor([1300,1300]))))
    rng=torch.get_rng_state().clone();out=plant_features(e)
    assert torch.equal(torch.get_rng_state(),rng)
    assert out.shape==(n,40) and torch.isfinite(out).all()
    assert torch.allclose(out[0,[2,8,14]],torch.tensor([.01,.02,.03]))
    assert torch.equal(out[0,18:24],torch.full((6,),.5)) and not out[1,18:24].any()
    assert torch.allclose(out[:,24:27],torch.tensor([[1.1,.45,.5],[.9,.55,.4]]))
    assert torch.allclose(out[0,37:39],torch.tensor([1.,0.])) and not out[1,37:].any()


@pytest.mark.parametrize('h',[.30,.40])
def test_fixed_height_assessment_does_not_keep_old034_height_band(monkeypatch,h):
    from native_stand.students import motion_reacquisition
    def stub(result,*args):result['checks']={'height':False,'reacquisition_height':False,'contact':True}
    monkeypatch.setattr(motion_reacquisition,'assess',stub)
    arrays=dict(first_episode_valid=np.ones((3000,1),dtype=bool),controller=np.full((3000,1,1),3),state=np.zeros((3000,1,30)))
    arrays['state'][:,:,6]=h
    result=task.assess({},arrays,0,dict(_height=dict(initial_m=h,target_m=h,moving=False)))
    assert result['accepted'] is True and result['height_task']['height_stable_mae_m']<1e-6


def test_fixed_height_case_keys_and_full_diagnostic_budget():
    s=state();s.update(focus_level=4);s['levels']['height']=3
    spec=dict(motion_decoupled=dict(state=s),student_evaluation=dict(cases=[dict(seconds=31.52,seed=1,command=[0.,0.,.34])]))
    cases=task.cases(spec,state=s)
    assert len({tuple(c['_identity_key']) for c in cases})==len(cases)
    assert sum(c.get('_full_diagnostic',False) for c in cases)==3
    assert len([c for c in cases if c['_focus_required']])==11*17
