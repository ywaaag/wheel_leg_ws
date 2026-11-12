"""Portable production control routing; expected physical values independent of decoder."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from native_stand.simulation.isaac.handover import Handover
from native_stand.tasks.motion.action_contract import ActionDecoder
from native_stand.tasks.motion.encoding import previous_from_adopted
from native_stand.tasks.motion.delayed_execution import configuration, reset_values
from native_stand.simulation.isaac.delayed_execution import dr_factory

FROZEN_TEST_SCOPE = 'PORTABLE'
FIXTURES=Path(__file__).parent/'fixtures'


def contract():
    return json.loads((FIXTURES/'motion_C0_control_contract.json').read_text())


class Teacher(torch.nn.Module):
    def mean(self, x):
        return x.new_tensor([.2,.2,.5,.2,.2,-.5]).expand(len(x),-1)


def router_env():
    decoder=ActionDecoder.from_contract(contract())
    router=Handover(decoder,decoder,Teacher(),fixed_prefix=True)
    env=NS(num_envs=1,device='cpu',fsm=NS(mode=torch.tensor([2])),
        initialization_checked=torch.ones(1,dtype=torch.bool),height_adapter=NS(target=torch.tensor([.34])),
        student_obs={'student_actor':torch.zeros(1,31)})
    router.initialize(env,{})
    env.r4_prefix_remaining.fill_(1)
    return router,env


def test_actual_handover_router_teacher_student_and_previous():
    router,env=router_env()
    raw=torch.tensor([[0.,0.,1.,0.,0.,-1.]])
    units=torch.tensor([1.5,1.5,100,1.5,1.5,100])
    before=torch.get_rng_state().clone()
    teacher,valid=router.actions(env,raw)
    assert not valid.item()
    assert torch.allclose(teacher*units,torch.tensor([[-.10097402,-.12117675,40.,.16097402,-.12117675,-40.]]),atol=2e-7)
    student,valid=router.actions(env,raw)
    assert valid.item() and env.r4_handoff_count.item()==1
    physical=student*units
    assert torch.allclose(physical,torch.tensor([[-.13097402,-.15117675,80.,.13097402,-.15117675,-80.]]),atol=2e-7)
    assert torch.allclose(previous_from_adopted(physical)[:,-2:],torch.tensor([[.8,-.8]]))
    assert torch.equal(raw,env.supported_raw_action)
    assert torch.equal(before,torch.get_rng_state())


def test_teacher_decodes_own_wheel40_contract():
    router,env=router_env()
    own=contract();own.update(action_version='motion31_kinematic_support_centered_015_40_v1',wheel_halfwidth_rad_s=40.)
    own.pop('support_offset_table')
    own['support_target_offset_rad']=[-.34415705961703547,-.36427784427970067,.34415705961703547,-.36427784427970067]
    router.teacher_decoder=ActionDecoder.from_contract(own)
    actual,_=router.actions(env,torch.zeros(1,6))
    assert actual[0,2]*100==20
    assert actual[0,5]*100==-20


def test_contract_rejects_continuous_formula_and_scale_mismatch():
    bad=contract();bad['center_formula']['version']='firmware_continuous_v1'
    with pytest.raises(ValueError,match='中心公式'):ActionDecoder.from_contract(bad)
    bad=contract();bad['wheel_halfwidth_rad_s']=40
    with pytest.raises(ValueError,match='包络'):ActionDecoder.from_contract(bad)


@pytest.mark.parametrize('profiles',[[ -1,3,0,1],[3,3,0,0]])
def test_timing_nominal_single_combined_partial_reset(profiles):
    config=configuration()
    class Base:
        def reset(self,ids):
            self.episodes[ids]+=1
            self.t.base.motor_ring.delay[ids]=20
            self.t.base.motor_ring.tau[ids]=2
    def noise(episodes,ids,width,stream):
        return torch.linspace(-1,1,width).expand(len(ids),-1)
    injected=dr_factory(Base,noise,config)()
    injected.episodes=torch.zeros(4,dtype=torch.long)
    injected.profiles=torch.tensor(profiles)
    injected.amplitude=torch.tensor([[0.],[1.],[1.],[0.]])
    injected.packet_age=torch.zeros(4,dtype=torch.long)
    injected.leg_packet_age=torch.zeros(4,dtype=torch.long)
    delay=torch.zeros(4,6,dtype=torch.long)
    tau=torch.zeros(4,6)
    injected.t=NS(base=NS(motor_ring=NS(delay=delay,tau=tau)))
    injected.reset(torch.arange(4))
    assert injected.packet_age[0]==injected.leg_packet_age[0]==20
    assert delay[0,[2,5]].tolist()==[7,11]
    assert not tau[:,[2,5]].any()
    assert ((delay[:,[2,5]]>=3)&(delay[:,[2,5]]<=15)).all()
    before=copy.deepcopy(injected.t.base.motor_ring)
    injected.reset(torch.tensor([1]))
    assert torch.equal(delay[[0,2,3]],before.delay[[0,2,3]])
    assert injected.episodes.tolist()==[1,2,1,1]


def test_timing_random_stream_is_separate_from_policy_rng():
    def noise(e,ids,width,stream):return torch.zeros(len(ids),width)
    before=torch.get_rng_state()
    value=reset_values(torch.arange(2),torch.zeros(2,dtype=torch.long),torch.tensor([0,3]),torch.ones(2,1),noise,config=configuration())
    assert value['wheel_delay'].tolist()==[[7,11],[7,11]]
    assert torch.equal(before,torch.get_rng_state())


def test_real_motor_ring_7_11_arrival_wrap_and_partial_reset():
    from native_stand.students.motion_mcu_execution import MotorRing
    ring=MotorRing(2,'cpu')
    ring.delay[:,[2,5]]=torch.tensor([7,11]);ring.tau.zero_()
    command=torch.zeros(2,6)
    arrivals=[]
    for tick in range(150):
        command.zero_()
        if tick==65:command[:,[2,5]]=1
        actual=ring.step(tick,command).clone()
        if actual[:,[2,5]].any():arrivals.append((tick,actual[:,[2,5]].tolist()))
        if tick==68:ring.reset(torch.tensor([0]),torch.zeros(1,6))
    assert arrivals==[(72,[[0.,0.],[1.,0.]]),(76,[[0.,0.],[0.,1.]])]


def test_actual_packet_ring_age_wrap_partial_reset_and_previous_independence():
    from native_stand.simulation.isaac.sensor_packets import Packets
    dr=NS(episodes=torch.zeros(2,dtype=torch.long),bias=torch.zeros(2,16),scales=torch.zeros(2,5),
        packet_age=torch.full((2,),20),leg_packet_age=torch.full((2,),20),dq=NS(read=lambda x,t:x))
    timing=NS(n=2,device='cpu',ids=torch.arange(2),dr=dr,tick=0)
    timing.measurement=lambda:(torch.full((2,6),float(timing.tick)),torch.full((2,6),float(timing.tick)))
    packets=Packets(timing,noise=lambda e,ids,w,s:torch.zeros(len(ids),w))
    for tick in range(0,100,10):
        timing.tick=tick
        packets.read(torch.full((2,16),float(tick)),tick)
    assert packets.result[1,6]==pytest.approx(70/1.5)
    packets.reset(torch.tensor([0]));timing.tick=100
    value=packets.read(torch.full((2,16),100.),100)
    assert packets.age.tolist()==[0.,20.] and packets.leg_age.tolist()==[0.,20.]
    assert value[0,6]==pytest.approx(100/1.5) and value[1,6]==pytest.approx(80/1.5)
    # previous is outside the16 physical sensors; this object never receives execution caches.
    again=packets.read(torch.full((2,16),999.),100)
    assert torch.equal(value,again)


def test_real_episode_clock_excludes_teacher_prefix_without_resetting_state():
    from native_stand.students.standing_survival_task import Clock
    router,env=router_env()
    clock=Clock(1,'cpu');env.rise_hold=clock
    router.initialize(env,{})
    env.r4_prefix_remaining.fill_(2)
    q=torch.zeros(1);boolean=torch.zeros(1,dtype=torch.bool)
    for _ in range(4):
        _,student=router.actions(env,torch.zeros(1,6))
        env.student_actor_valid=student
        env.rise_hold.observe(q,torch.zeros(1,2),q,torch.ones(1,2)*3,q,q,boolean,
            active=~boolean,valid=~boolean,safety_failed=boolean,fault=boolean)
    assert clock.episode.steps.tolist()==[2]
    assert env.rise_hold.clock is clock and env.r4_handoff_count.item()==1
