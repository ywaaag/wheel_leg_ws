"""Causal electrical packet/scheduler hypotheses, isolated from training defaults.

The host library executes the real DM/DJI drivers and the real control owner.
Packet production times, CAN delivery, inference duration and flexible mechanics
remain declared hypotheses. Integer clocks are local milliseconds, not wall time.
"""
import copy
import ctypes as C
import heapq

import numpy as np
from scipy.linalg import expm

from .firmware_control import FirmwareControl, FP, UP, _array
from .oscillation_search import CExecution, LEGS, WHEELS, nominal, validate
from .real2sim_delay import ObservationHistory, torque_response

VERSION='motion31_source_packet_schedule_v1'


class PacketEnvelopeError(ValueError):
    """Finite hypothetical sensor value outside the declared wire range.

    This rejects an unrepresentable physical hypothesis, without clipping or
    changing the electrical contract. Nonfinite states remain runtime faults.
    """


def profile():
    return dict(leg_packet_period_ms=2,wheel_packet_period_ms=1,
        leg_packet_phase_ms=[0,0,0,0],wheel_packet_phase_ms=[0,0],
        robot_phase_ms=0,actor_phase_ms=0,motor_send_phase_ms=0,motor_before_robot=False,
        inference_ms=0,can_receive_delay_ms=0,leg_sample_age_ms=0,
        internal_dq_age_ms=0,internal_dq_tau_ms=0.,
        coupling_stiffness_nm_rad=0.,coupling_damping_nm_s_rad=.2,rotor_inertia=.006)


def validate_profile(value):
    p=copy.deepcopy(value)
    if set(p)!=set(profile()):raise ValueError('explicit source sampling profile fields required')
    integers=dict(leg_packet_period_ms=(1,10),wheel_packet_period_ms=(1,10),robot_phase_ms=(0,4),
        actor_phase_ms=(0,9),motor_send_phase_ms=(0,1),inference_ms=(0,3),
        can_receive_delay_ms=(0,4),leg_sample_age_ms=(0,15),internal_dq_age_ms=(0,15))
    for k,(lo,hi) in integers.items():
        v=p[k]
        if not np.isfinite(v) or v!=int(v) or not lo<=v<=hi:raise ValueError('integer sampling bound: '+k)
    for k,n,period in (('leg_packet_phase_ms',4,p['leg_packet_period_ms']),
                       ('wheel_packet_phase_ms',2,p['wheel_packet_period_ms'])):
        a=np.asarray(p[k],float)
        if a.shape!=(n,) or not np.isfinite(a).all() or (a<0).any() or (a>=period).any() or not np.equal(a,np.round(a)).all():
            raise ValueError('packet phase outside period: '+k)
    if type(p['motor_before_robot']) is not bool:raise ValueError('explicit event order bool required')
    for k,(lo,hi) in dict(internal_dq_tau_ms=(0,20),coupling_stiffness_nm_rad=(0,1000),
            coupling_damping_nm_s_rad=(0,2),rotor_inertia=(.001,.009)).items():
        if not np.isfinite(p[k]) or not lo<=p[k]<=hi:raise ValueError('physical hypothesis bound: '+k)
    return p


def source_parameters():
    p=nominal();p.update(actor_age_ms=0,pd_age_ms=0,motor_delay_ms=[0]*6,motor_tau_ms=[0.]*6)
    return p


def cases():
    """Frozen one-factor/source-phase matrix, then finite unknown interactions.

    No mass/friction/zero/std/reward search; no hardware identification claim.
    """
    rows=[]
    def add(name,family,changes=None,physical=None):
        s=profile();s.update(changes or {});p=source_parameters();p.update(physical or {})
        rows.append(dict(name=name,family=family,parameters=validate(p),sampling=validate_profile(s)))
    add('source_nominal','baseline')
    add('source_P20','gain',physical=dict(kp=20.))
    add('source_Kv015','gain',physical=dict(kv=.15))
    add('source_P20_Kv015','gain',physical=dict(kp=20.,kv=.15))
    for i in (1,2,4,7):add(f'phase_actor{i}','known_software_phase',dict(actor_phase_ms=i))
    add('phase_robot2','known_software_phase',dict(robot_phase_ms=2))
    add('phase_send1','known_software_phase',dict(motor_send_phase_ms=1))
    add('phase_motor_before_robot','known_software_phase',dict(motor_before_robot=True))
    add('phase_interleaved_packets','known_software_phase',dict(leg_packet_phase_ms=[0,1,0,1]))
    for n in (1,5,10):add(f'packet_period{n}','unknown_packet_period',dict(leg_packet_period_ms=n))
    for n in (1,3):add(f'inference{n}','unknown_inference',dict(inference_ms=n))
    for n in (1,2):add(f'can_delivery{n}','unknown_CAN',dict(can_receive_delay_ms=n))
    for n in (2,5,10):add(f'leg_sample_age{n}','unknown_sample_age',dict(leg_sample_age_ms=n))
    for n in (2,5,10):add(f'internal_dq_age{n}','unknown_estimator_age',dict(internal_dq_age_ms=n))
    for n in (2.,5.,10.):add(f'internal_dq_tau{n:g}','unknown_estimator_filter',dict(internal_dq_tau_ms=n))
    for n in (2,5,10):add(f'motor_delay{n}','unknown_motor_transport',physical=dict(motor_delay_ms=[n]*6))
    for n in (2.,5.,10.):add(f'motor_tau{n:g}','unknown_motor_response',physical=dict(motor_tau_ms=[n,n,0,n,n,0]))
    for k,b in ((100.,.2),(300.,.2),(900.,.2),(300.,.05),(300.,.8)):
        add(f'flexK{k:g}_B{b:g}','unknown_transmission_compliance',dict(coupling_stiffness_nm_rad=k,coupling_damping_nm_s_rad=b))
    for age,tau in ((5,5.),(10,5.)):
        add(f'sample{age}_dq_tau{tau:g}','unknown_interaction',dict(leg_sample_age_ms=age,internal_dq_tau_ms=tau))
    for k,age,tau in ((100.,2,2.),(300.,5,2.),(300.,2,5.)):
        s=dict(coupling_stiffness_nm_rad=k,leg_sample_age_ms=age,internal_dq_tau_ms=tau)
        add(f'flex{k:g}_age{age}_tau{tau:g}','unknown_interaction',s)
        add(f'lowgain_flex{k:g}_age{age}_tau{tau:g}','gain_confirmation',s,dict(kp=20.,kv=.15))
    return rows


class SampledControl(FirmwareControl):
    def __init__(self,library):
        super().__init__(library)
        self.lib.motion_fw_observe_packet.argtypes=[C.c_void_p,C.c_uint32,C.c_uint32,C.c_uint32,UP,FP,FP,FP]
        self.lib.motion_fw_pd_packet.argtypes=[C.c_void_p,C.c_uint32,UP,FP,FP,FP,FP,FP,UP]

    def observe_packet(self,now,snapshot_tick,receipt7,native,command,*,sequence=1):
        x,c=_array(native,16),_array(command,3);o=np.empty(31,np.float32)
        r=np.asarray(receipt7,dtype=np.int64).astype(np.uint32)
        rc=self.lib.motion_fw_observe_packet(self.context,now,snapshot_tick,sequence,r.ctypes.data_as(UP),
            x.ctypes.data_as(FP),c.ctypes.data_as(FP),o.ctypes.data_as(FP))
        if rc:raise ValueError('firmware packet observation rejected: '+str(rc))
        return o

    def pd_packet(self,now,receipt7,native,command):
        length,spread=self.geometry(native[6:10]);values=[_array(v,n) for v,n in ((native,16),(command,3),(length,2),(spread,2))]
        r=np.asarray(receipt7,dtype=np.int64).astype(np.uint32);out=np.empty(30,np.float32);meta=np.zeros(8,np.uint32)
        rc=self.lib.motion_fw_pd_packet(self.context,now,r.ctypes.data_as(UP),*[v.ctypes.data_as(FP) for v in values],
            out.ctypes.data_as(FP),meta.ctypes.data_as(UP))
        if rc:raise RuntimeError(f'firmware packet PD rejected: {rc}, metadata={meta.tolist()}')
        return dict(motor_model_nm=out[:6].copy(),spring_compensation_model_nm=out[6:12].copy(),
            pd_request_nm=out[12:18].copy(),pd_limited_nm=out[18:24].copy(),adopted_target=out[24:30].copy(),metadata=meta.copy())


class DriverPackets:
    """Actual C decoder/filter state. Wire production/rounding are hypotheses.

    One library instance is used serially. No RNG, true q substituted for an
    unreceived packet, or synthetic receipt-time refresh on a repeated read.
    """
    def __init__(self,library,settings,initial):
        self.p=validate_profile(settings);self.lib=C.CDLL(str(library));lib=self.lib
        lib.motion_io_motor_coordinates.argtypes=[FP,FP,FP]
        lib.motion_io_leg_receive.argtypes=[C.c_uint,C.c_uint,C.c_uint,C.c_uint32]
        lib.motion_io_wheel_receive.argtypes=[C.c_uint,C.c_int,C.c_uint32]
        lib.motion_io_read.argtypes=[FP,UP,UP];lib.motion_io_send.argtypes=[C.c_uint32,FP,FP,UP,UP]
        if lib.motion_io_reset():raise RuntimeError('driver host reset failed')
        self.history=ObservationHistory(initial,initial_tick=-64)
        self.estimated=ObservationHistory(initial,initial_tick=-64)
        self.filtered=np.asarray(initial)[12:18].copy();self.queue=[];self.serial=0
        self.sample_tick=np.full(6,-64,dtype=int);self.last_tick=-33
        self.current=None;self.receipt=np.zeros(7,dtype=np.int64);self.sequence=np.zeros(6,np.uint32)
        self.wire=np.zeros(6);self.wire_tick=np.zeros(6,dtype=np.int64);self.wire_sequence=np.zeros(6,np.uint32)
        self.counts=dict(leg_packets=0,wheel_packets=0,motor_task_calls=0)
        # Constant, explicitly unknown prehistory. Warm the actual per-packet
        # wheel filter before takeover, without integrating a single physics step.
        for tick in range(-32,0):self.update(tick,initial)
        self.bootstrap_counts=self.counts.copy();self.counts={k:0 for k in self.counts}

    @staticmethod
    def word(value,bound,bits):
        if not np.isfinite(value):raise ValueError('nonfinite hypothetical encoder')
        if abs(value)>bound+1e-6:
            raise PacketEnvelopeError(f'hypothetical encoder {value} outside declared +/-{bound} CAN range')
        # Internal motor encoder rounding is not public; nearest-bin generation
        # is a declared assumption. Decoding always runs the unmodified driver.
        return int(np.clip(np.floor((value+bound)/(2*bound)*((1<<bits)-1)+.5),0,(1<<bits)-1))

    def coordinates(self,s):
        q=_array(np.asarray(s)[6:12][LEGS],4);dq=_array(np.asarray(s)[12:18],6);out=np.empty(6,np.float32)
        if self.lib.motion_io_motor_coordinates(q.ctypes.data_as(FP),dq.ctypes.data_as(FP),out.ctypes.data_as(FP)):
            raise ValueError('C inverse motor mapping failed')
        return out

    def update(self,tick,sensors):
        if tick!=self.last_tick+1:raise ValueError('packet clock must advance by1ms')
        self.last_tick=tick;s=np.asarray(sensors,float)
        if s.shape!=(18,) or not np.isfinite(s).all():raise ValueError('finite physical sensor18 required')
        self.history.append(tick,s)
        value=self.history.at(tick-self.p['internal_dq_age_ms'])[12:18]
        tau=self.p['internal_dq_tau_ms']
        if tau:self.filtered[LEGS]+=-np.expm1(-1/tau)*(value[LEGS]-self.filtered[LEGS])
        else:self.filtered[LEGS]=value[LEGS]
        estimated=s.copy();estimated[12:18][LEGS]=self.filtered[LEGS];self.estimated.append(tick,estimated)
        sample_tick=tick-self.p['leg_sample_age_ms'];packet=self.estimated.at(sample_tick)
        coordinates=self.coordinates(packet);rates=packet[12:18][[4,3,1,0]]*[-1,-1,1,-1]
        for i in range(4):
            if (tick-self.p['leg_packet_phase_ms'][i])%self.p['leg_packet_period_ms']==0:
                words=(self.word(coordinates[i],6.283,16),self.word(rates[i],45.,12))
                self.serial+=1;heapq.heappush(self.queue,(tick+self.p['can_receive_delay_ms'],self.serial,'leg',i,words,sample_tick))
        coordinates_now=self.coordinates(s)
        for side in range(2):
            if (tick-self.p['wheel_packet_phase_ms'][side])%self.p['wheel_packet_period_ms']==0:
                rpm=int(np.rint(coordinates_now[4+side]))
                if not -32768<=rpm<=32767:raise ValueError('wheel RPM exceeds packet range')
                self.serial+=1;heapq.heappush(self.queue,(tick+self.p['can_receive_delay_ms'],self.serial,'wheel',side,(rpm,),tick))
        while self.queue and self.queue[0][0]<=tick:
            receive,_,kind,index,words,origin=heapq.heappop(self.queue)
            if kind=='leg':
                rc=self.lib.motion_io_leg_receive(index,*words,C.c_uint32(receive));ch=[4,3,1,0][index]
                self.counts['leg_packets']+=1
            else:
                rc=self.lib.motion_io_wheel_receive(index,*words,C.c_uint32(receive));ch=[5,2][index]
                self.counts['wheel_packets']+=1
            if rc:raise ValueError('real driver rejected generated packet')
            self.sample_tick[ch]=origin
        out=np.empty(10,np.float32);r=np.empty(6,np.uint32);seq=np.empty(6,np.uint32)
        if self.lib.motion_io_read(out.ctypes.data_as(FP),r.ctypes.data_as(UP),seq.ctypes.data_as(UP)):
            raise RuntimeError('driver feedback read failed')
        self.current=np.r_[s[:6],out].astype(np.float32)
        self.receipt[:6]=r.astype(np.int32).astype(np.int64);self.receipt[6]=tick
        self.sequence=seq.copy()
        return self.current.copy()

    def send(self,tick,demand):
        d=_array(demand,6);wire=np.empty(6,np.float32);r=np.empty(6,np.uint32);seq=np.empty(6,np.uint32)
        if self.lib.motion_io_send(C.c_uint32(tick),d.ctypes.data_as(FP),wire.ctypes.data_as(FP),r.ctypes.data_as(UP),seq.ctypes.data_as(UP)):
            raise RuntimeError('real driver send failed')
        self.wire=wire.copy();self.wire_tick=r.astype(np.int32).astype(np.int64);self.wire_sequence=seq.copy()
        self.counts['motor_task_calls']+=1
        return self.wire.copy()


class CompliantTransmission:
    """Explicit optional two-inertia hypothesis, encoder upstream of compliance.

    Rotor inertia is removed from plant armature, so total declared armature is
    preserved. Exact held-load linear integration; no hidden output filtering or
    extra drive force. Unbounded spring demand is checked by the physics caller.
    """
    def __init__(self,settings,q4,dq4):
        p=validate_profile(settings);self.k=p['coupling_stiffness_nm_rad'];self.b=p['coupling_damping_nm_s_rad']
        self.j=p['rotor_inertia'];self.theta=np.asarray(q4,float).copy();self.omega=np.asarray(dq4,float).copy()
        if self.k:
            a=np.array([[0,1],[-self.k/self.j,-self.b/self.j]])
            self.e=expm(a*.001);self.u=np.linalg.solve(a,(self.e-np.eye(2))@np.array([0,1/self.j]))

    def step(self,motor6,q4,dq4):
        result=np.asarray(motor6,float).copy()
        if not self.k:return result
        q,dq=np.asarray(q4),np.asarray(dq4)
        relative=np.stack([self.theta-q,self.omega-dq]);next_relative=self.e@relative+self.u[:,None]*result[LEGS]
        omega_next=dq+next_relative[1]
        result[LEGS]=result[LEGS]-self.j*(omega_next-self.omega)/.001
        self.theta=q+.001*dq+next_relative[0];self.omega=omega_next
        return result


class SourceExecution(CExecution):
    def __init__(self,library,p,initial_motor,settings,initial_sensors,first_previous):
        super().__init__(library,p,initial_motor,core_type=SampledControl)
        self.settings=validate_profile(settings);self.io=DriverPackets(library,settings,initial_sensors)
        self.flex=CompliantTransmission(settings,initial_sensors[6:12][LEGS],initial_sensors[12:18][LEGS])
        self.driver=np.asarray(initial_motor,np.float32).copy();self.raw=np.zeros(6,np.float32)
        self.snapshot=None;self.pending=None;self.published=False;self.policy_count=0;self.pd_count=0
        self.first_previous=np.asarray(first_previous,np.float32);self.policy=None;self.pd=None
        self.first_adoption_tick=None;self.motor_sequence_before=np.zeros(6,np.uint32)
        self.events=[];self.snapshot_sequence=0

    def step_sampled(self,tick,current,command,actor):
        if tick!=self.last_tick+1:raise ValueError('source execution needs contiguous1ms clock')
        self.last_tick=tick;p=self.settings;reported=np.asarray(current).copy()
        if self.flex.k:
            reported[6:12][LEGS]=self.flex.theta;reported[12:18][LEGS]=self.flex.omega
        self.io.update(tick,reported)
        def send():
            # Calling the real source body on a shifted1ms task clock gives the
            # production DM every-other-call phase; it is not a2ms PD update.
            if tick>=p['motor_send_phase_ms']:
                self.io.send(tick,self.driver);self.fifo.submit(tick,self.io.wire)
        if p['motor_before_robot']:send()
        if (tick-p['robot_phase_ms'])%5==0:
            native=self.io.current.copy();receipt=self.io.receipt.copy()
            self.snapshot=(tick,native,receipt,np.asarray(command,np.float32).copy(),self.io.sample_tick.copy())
            self.snapshot_sequence+=1
            if self.published:
                result=self.core.pd_packet(tick,receipt,native,command)
                self.update_pd(tick,native,command,packet_result=result);self.pd_count+=1
                if self.first_adoption_tick is None:self.first_adoption_tick=tick
                self.pd=dict(tick_ms=tick,native16=native.copy(),receipt_tick_ms=receipt.copy(),
                    sample_tick_ms=self.io.sample_tick.copy(),metadata=result['metadata'].copy())
        if not p['motor_before_robot']:send()
        # Lower-priority Actor publications at a coincident tick occur after
        # the normal-priority Robot/Motor tasks in this explicit ordering.
        if self.pending is not None and tick>=self.pending[0]:
            self.core.publish(tick,self.pending[1]);self.published=True;self.pending=None
            self.events.append((tick,'published',self.policy_count))
        if (tick-p['actor_phase_ms'])%10==0 and tick>=p['actor_phase_ms'] and self.snapshot is not None:
            if self.pending is not None:raise RuntimeError('overlapping inferred Actor jobs')
            stamp,native,receipt,cmd,origin=self.snapshot
            obs=self.core.observe_packet(tick,stamp,receipt,native,cmd,sequence=self.snapshot_sequence)
            if self.policy_count==0:obs[16:22]=self.first_previous
            mean=np.asarray(actor(obs[None])[0],np.float32)
            if mean.shape!=(6,) or not np.isfinite(mean).all():raise RuntimeError('nonfinite or invalid Actor output')
            self.policy_count+=1;self.policy=dict(tick_ms=tick,snapshot_tick_ms=stamp,
                publish_tick_ms=tick+p['inference_ms'],receipt_tick_ms=receipt.copy(),sample_tick_ms=origin.copy(),
                actor_input=obs.copy(),mean=mean.copy())
            if p['inference_ms']==0:
                self.core.publish(tick,mean);self.published=True;self.events.append((tick,'published',self.policy_count))
            else:self.pending=(tick+p['inference_ms'],mean.copy())
        self.motor=torque_response(self.motor,self.fifo.delivered(tick),self.p['motor_tau_ms'])
        transmitted=self.flex.step(self.motor,np.asarray(current)[6:12][LEGS],np.asarray(current)[12:18][LEGS])
        self.spring=-self.p['spring_scale']*self.compensation(np.asarray(current)[6:12][LEGS])
        return transmitted+self.spring
