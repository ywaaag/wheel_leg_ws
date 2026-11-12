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
job=load_job_config('/home/kellen/wheel_leg_native_train/configs/jobs/motion_C0_yaw1to4_full_KL10000.json')
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
    caps=torch.tensor([[40,40,2.44694786377709,40,40,2.44694786377709]]),kp=torch.full((n,4),40.),kd=torch.full((n,4),2.),kv=torch.full((n,2),.2),
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
# Real frozen pulse generation, new matcher/statistics, no physical integrator.
purposes=[f'modal_{m}_{sign}' for m in range(4) for sign in (-1,1)]+['continuous_C0_handover_m1']
n=len(purposes)
e=NS(num_envs=n,device='cpu',identified_dr=NS(episodes=torch.ones(n,dtype=torch.long)),command_index=torch.arange(n),r4_owner=torch.full((n,),3),
    stage_spec=dict(commands=[[0.,0.]]*n,modal_modes=[m for m in range(4) for _ in (-1,1)]+[-1],modal_signs=[s for _ in range(4) for s in (-1,1)]+[1],cohort_purposes=purposes,
        cohort_keys=[[0.,0.,.34,-1,p] for p in purposes]),recovery_snapshot=torch.cat((torch.zeros(n,10),torch.ones(n,2)*3),-1),standing_snapshot=torch.zeros(n,2),
    antiosc_physical_stats_post=torch.ones(n,9)*.1,motion_timing=NS(base=NS(adopted=torch.zeros(n,6))))
events=event_class(transient_events.Events,noise)(e);events.reset(torch.arange(n))
for _ in range(500):
    events.policy(torch.ones(n,dtype=torch.bool))
    for _ in range(10):events.physical()
    events.observe()
assert (events.count[:8]>0).all()
receipt.update(Critic_width=critic.shape[-1],modal_events_by_cohort=events.count.tolist(),physics_steps=0,PPO_updates=0)
write_json(sys.argv[1],receipt) if len(sys.argv)>1 else None
print(json.dumps(receipt))
