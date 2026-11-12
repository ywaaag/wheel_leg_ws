from types import SimpleNamespace as NS
import json
import sys
from pathlib import Path
import torch
from native_stand.experiments.config import load_job_config
from native_stand.simulation.isaac.legacy_support import activate_view
from native_stand.simulation.playback_execution import unpair
from native_stand.simulation.v4_response import Response
from native_stand.simulation.isaac.sensor_packets import Packets
from native_stand.artifacts.io import write_json
job=load_job_config('/home/kellen/wheel_leg_refactor_ws/configs/jobs/motion48_turn_response_DR_full_resume.json')
activate_view(job.assets['frozen_source_root'],job.assets['frozen_bundle'])
from native_stand.simulation.isaac.fullchain import base511_critic
from native_stand.simulation.isaac.delayed_execution import readback,dr_factory
from native_stand.simulation.isaac.transient_events import event_class
from dq_feedback import Feedback
from critic_tensor import FIELDS
from privileged_dr import append
from runtime_support import noise
import transient_events
n=1
leg=Response(torch.tensor([[1.,.1,.2,4.,6.,1.,-1.,2.,3.,.4,0.,0.,0.]]).expand(2,-1).clone(),[dict(transport_ms=8,q_age_ms=2,v_age_ms=4,q_reference=[0.,0.],velocity_tau_ms=3.)]*2)
leg.reset(torch.zeros(2,2),torch.zeros(2,2),torch.zeros(2,2))
for _ in range(30):leg.step(torch.zeros(2,2),torch.zeros(2,2),torch.zeros(2,2))
base=NS(tick=30,targets=torch.zeros(32,n,6),adopted=torch.zeros(n,6),requested=torch.zeros(n,6),last_pd_sensors=torch.zeros(n,12),motor_actual=torch.zeros(n,6),compensation=torch.zeros(n,6),
    motor_ring=NS(history=torch.zeros(64,n,6),actual=torch.zeros(n,6),delay=torch.tensor([[0,0,7,0,0,11]]),tau=torch.zeros(n,6)))
t=NS(n=n,device='cpu',ids=torch.arange(n),tick=30,base=base,leg=leg,last_external=torch.zeros(n,6),unpair=lambda x:unpair(x)[None])
d=NS(t=t,n=n,device='cpu',ids=t.ids,episodes=torch.ones(n,dtype=torch.long),profiles=torch.full((n,),-1),amplitude=torch.zeros(n,1),
    caps=torch.tensor([[40,40,4.8,40,40,4.8]]),kp=torch.full((n,4),40.),kd=torch.full((n,4),2.),kv=torch.full((n,2),.2),
    packet_age=torch.full((n,),20),leg_packet_age=torch.full((n,),20),actual_armatures=torch.zeros(n,6),wheel_B=torch.zeros(n,2),wheel_Tc=torch.zeros(n,2),parameter_tensor=lambda:torch.zeros(n,32),
    bias=torch.zeros(n,16),scales=torch.zeros(n,5),target_delay=torch.full((n,),5),strength=torch.ones(n,6),slope=torch.zeros(n,2),normal=torch.tensor([[0.,0.,1.]]),restitution=torch.zeros(n,2),
    push_steps=torch.full((n,2),500,dtype=torch.long),push_events=torch.zeros(n,dtype=torch.long),push=torch.zeros(n,3),pushed=torch.zeros(n,2,dtype=torch.bool))
t.dr=d;d.dq=Feedback(d)
t.measurement=lambda:(torch.zeros(n,6),torch.zeros(n,6))
p=Packets(t,noise=noise)
for tick in (0,10,20,30):t.tick=tick;p.read(torch.zeros(n,16),tick)
env=NS(num_envs=n,device='cpu',motion_timing=t,identified_dr=d,identified_packets=p,
    robot=NS(data=NS(root_pos_w=NS(torch=torch.tensor([[0.,0.,.34]])),joint_pos=NS(torch=torch.zeros(n,6)),joint_vel=NS(torch=torch.zeros(n,6)))),ids=list(range(6)),
    scene=NS(env_origins=torch.zeros(n,3)),DR_static_ground_normal=torch.tensor([[0.,0.,1.]]),contact_force=torch.ones(n,2)*3,
    transient_events=NS(current=torch.zeros(n,6)),rise_hold=NS(episode=NS(steps=torch.zeros(n,dtype=torch.long))),
    command=torch.tensor([[0.,0.,.34]]),physical_hf=NS(low=torch.zeros(n,4)))
critic511=base511_critic(env,torch.zeros(n,31),body_velocity=torch.zeros(n,3))
# The51 reward-memory columns are the engine/reward boundary fixture; actual DR/timing append is unchanged production code.
critic=append(env,torch.cat((critic511,torch.zeros(n,51)),-1))
receipt=readback(env,{'student_critic':critic})
offset=sum(w for k,w in FIELDS[:next(i for i,x in enumerate(FIELDS) if x[0]=='effective_wheel_timing')])
assert torch.allclose(critic[:,offset:offset+4],torch.tensor([[7/30,11/30,0.,0.]]))

from native_stand.simulation.isaac.maneuver_runtime import attach,append_critic
from native_stand.simulation.isaac.maneuver_planned import environment_class
from native_stand.tasks.motion.planned_value import append as planned_append
base.limits=d.caps.clone()
env.stage_spec={'goal_heights_m':[.38]};env.command_index=torch.zeros(1,dtype=torch.long)
env.height_adapter=NS(target=torch.tensor([.36]),rate=torch.tensor([.02]))
env.quality_monitor=NS();env.command=torch.tensor([[1.,.5,.36]])
env.standing_snapshot=torch.zeros(1,6);env.target_xy=torch.tensor([[3.6,.5]])
attach(env,job.reward.fields['scale_corrections'],reward_config=job.reward.fields['maneuver'],command_program_version='motion48_height_travel_turn_schedule_v2')
env.maneuver_height_rate[:]=.02
value875=append_critic(env,critic)
assert value875.shape==(1,875)
class PhysicalIO:
 num_envs=1
 planned_value_ready=True
 target_xy=env.target_xy
 def _get_observations(self):return {'student_critic':value875}
 def _get_dones(self):return torch.tensor([False]),torch.tensor([True])
wrapped=environment_class(PhysicalIO)()
ordinary=wrapped._get_observations()['student_critic']
wrapped._get_dones()
assert ordinary.shape==wrapped.student_terminal_critic.shape==(1,877)
assert torch.equal(ordinary,wrapped.student_terminal_critic)
assert torch.allclose(ordinary[:,875:],torch.tensor([[.9,.125]]))
assert torch.allclose(ordinary[:,863],torch.tensor([.72]))
assert torch.allclose(ordinary[:,env.maneuver_critic_columns['height']],torch.tensor([-.02]),atol=1e-6)
from native_stand.simulation.isaac.motion48_execution import readback as cap_readback
assert cap_readback(t,job.assets['motor_limits'],{'student_critic':ordinary})['actual_Critic_caps_verified']
print(json.dumps({'Critic_width':877,'terminal_width':877,'wheel_caps':d.caps[:,[2,5]].tolist(),'physics_steps':0}))
