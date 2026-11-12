"""Diagnostic upstream sensor/target queues, never enabled by training defaults.

Only Actor sensor slots0:16 age. Previous adopted targets and commands/task stay
current. A target relay queues deterministic means before the live C publisher;
its generation/arrival clock is recorded separately from C-local timestamps.
This is an explicit hypothetical relay, not a replica of MCU message transport.
"""
from collections import deque
import numpy as np
from .real2sim_delay import TargetTransport

VERSION='motion31_actor_target_delay_probe_v1'
LEGS=[0,1,3,4]


class PolicyDelay:
    def __init__(self,actor,observation_ms,target_ms,initial_previous):
        for value in (observation_ms,target_ms):
            if type(value) is not int or value not in (0,10,20,30):
                raise ValueError('explicit0/10/20/30ms policy-aligned delay required')
        prev=np.asarray(initial_previous,np.float32)
        if prev.shape!=(6,) or not np.isfinite(prev).all():raise ValueError('normalized previous4legs+2wheels required')
        initial=np.empty(6,np.float32);initial[LEGS]=prev[:4];initial[[2,5]]=prev[4:]
        if (abs(initial)>1).any():raise ValueError('initial target cannot be represented without clipping')
        self.actor=actor;self.observation_ms=observation_ms;self.target_ms=target_ms
        self.transport=TargetTransport([target_ms]*6,initial)
        self.history=deque(maxlen=5);self.records=[];self.last_tick=-10

    def step(self,tick,observation):
        x=np.asarray(observation,np.float32)
        if x.shape!=(1,31) or not np.isfinite(x).all():raise ValueError('finite current31D required')
        if tick!=self.last_tick+10:raise ValueError('contiguous10ms policy clock required')
        self.last_tick=tick;self.history.append((tick,x.copy()))
        request_tick=tick-self.observation_ms
        stamp,old=next(((s,z) for s,z in reversed(self.history) if s<=request_tick),self.history[0])
        delayed=x.copy();delayed[:,:16]=old[:,:16]
        mean=np.asarray(self.actor(delayed),np.float32)
        if mean.shape!=(1,6) or not np.isfinite(mean).all() or (abs(mean)>1.00001).any():
            raise ValueError('deterministic bounded mean6 required')
        self.transport.submit(tick,mean[0]);delivered=self.transport.delivered(tick).astype(np.float32)
        self.records.append(dict(tick_ms=tick,sensor_source_policy_tick_ms=stamp,
            prehistory_clamped=request_tick<0,target_generation_tick_ms=tick-self.target_ms if tick>=self.target_ms else -1,
            target_from_initial=tick<self.target_ms,current_input=x[0].copy(),actor_input=delayed[0].copy(),
            requested_mean=mean[0].copy(),delivered_mean=delivered.copy()))
        return delivered[None]


def cases():
    """One-axis progression, then fixed combinations; no adaptive search."""
    pairs=[(0,0),(10,0),(20,0),(30,0),(0,10),(0,20),(0,30),
           (10,10),(20,20),(30,30),(10,20),(20,10)]
    rows=[]
    for gain,kp,kv in [('original',40.,.2),('lower',20.,.15)]:
        for obs,target in pairs:
            rows.append(dict(name=f'R_{gain}_obs{obs}_target{target}',profile='R',gain=gain,
                kp=kp,kd=2.,kv=kv,observation_ms=obs,target_ms=target,motor_delay_ms=0))
        for obs in (0,20):
            rows.append(dict(name=f'R_{gain}_obs{obs}_postPD20',profile='R',gain=gain,
                kp=kp,kd=2.,kv=kv,observation_ms=obs,target_ms=0,motor_delay_ms=20))
        for obs,target in ((0,0),(20,0),(0,20),(20,20)):
            rows.append(dict(name=f'F40_{gain}_obs{obs}_target{target}',profile='F40',gain=gain,
                kp=kp,kd=2.,kv=kv,observation_ms=obs,target_ms=target,motor_delay_ms=0))
    return rows
