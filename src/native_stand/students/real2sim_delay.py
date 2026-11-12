"""Bounded diagnostic timing primitives, independent of simulator and filesystem.

Transport delay and torque-rise time are separate hypotheses, never auto-fitted
from a combined response-onset delay. Channels always use native model order.
"""
from collections import deque
import numpy as np


class TargetTransport:
    def __init__(self, delay_ms, initial):
        delay=np.asarray(delay_ms,dtype=float)
        if delay.shape!=(6,) or not np.isfinite(delay).all() or (delay<0).any() or (delay>100).any() or not np.equal(delay,np.round(delay)).all():
            raise ValueError('six integer delays 0..100ms required')
        self.delay=delay.astype(int);self.current=np.array(initial,dtype=float,copy=True)
        if self.current.shape!=(6,) or not np.isfinite(self.current).all():raise ValueError('finite initial target6 required')
        self.pending=[deque() for _ in range(6)];self.last_tick=-1
    def submit(self,tick,target):
        value=np.asarray(target,dtype=float)
        if value.shape!=(6,) or not np.isfinite(value).all() or tick<self.last_tick:raise ValueError('target/time invalid')
        self.last_tick=tick
        for i,v in enumerate(value):self.pending[i].append((tick+self.delay[i],v))
    def delivered(self,tick):
        for i,q in enumerate(self.pending):
            while q and q[0][0]<=tick:self.current[i]=q.popleft()[1]
        return self.current.copy()


class ObservationHistory:
    """1ms time-indexed sensor snapshots. Clamp prehistory to the initial sample."""
    def __init__(self,initial,*,initial_tick=0):
        self.rows=deque([(initial_tick,np.array(initial,dtype=float,copy=True))],maxlen=102)
    def append(self,tick,sample):
        if tick<self.rows[-1][0]:raise ValueError('observation time reversed')
        self.rows.append((tick,np.array(sample,dtype=float,copy=True)))
    def at(self,tick):
        for stamp,row in reversed(self.rows):
            if stamp<=tick:return row.copy()
        return self.rows[0][1].copy()


def torque_response(previous,demand,tau_ms,dt_ms=1.):
    tau=np.asarray(tau_ms,dtype=float)
    if tau.shape!=(6,) or not np.isfinite(tau).all() or (tau<0).any():raise ValueError('finite nonnegative torque rise times6 required')
    alpha=-np.expm1(-dt_ms/np.maximum(tau,1e-12))
    return np.asarray(previous)+alpha*(np.asarray(demand)-np.asarray(previous))


def scenarios():
    """One changed mechanism per row. Values are hypotheses, not identification."""
    base={'target_delay_ms':[0]*6,'dq_delay_ms':0,'imu_delay_ms':0,'pd_feedback_delay_ms':0,'torque_tau_ms':[0]*6,'cache':'executed','command_vx':-.286}
    rows=[]
    def add(name,**kw):rows.append({'name':name,**base,**kw})
    add('baseline_zero',command_vx=0.)
    add('baseline_reverse')
    add('wheel_transport_11_7',target_delay_ms=[0,0,11,0,0,7])
    add('wheel_transport_7_11',target_delay_ms=[0,0,7,0,0,11])
    for ms in (5,10,20):add(f'leg_transport_{ms}',target_delay_ms=[ms,ms,0,ms,ms,0])
    for ms in (30,40,60,80):add(f'leg_transport_{ms}',target_delay_ms=[ms,ms,0,ms,ms,0])
    for ms in (5,10,20):add(f'dq_observation_{ms}',dq_delay_ms=ms)
    add('imu_observation_10',imu_delay_ms=10)
    add('wheel_torque_rise_11_7',torque_tau_ms=[0,0,11,0,0,7])
    for ms in (5,10,20):
        add(f'PD_feedback_{ms}',pd_feedback_delay_ms=ms)
        add(f'leg_torque_rise_{ms}',torque_tau_ms=[ms,ms,0,ms,ms,0])
    add('shadow_cache_at_handover',cache='shadow_seed_once')
    add('measured_wheels_previous',cache='measured_wheels')
    return rows
