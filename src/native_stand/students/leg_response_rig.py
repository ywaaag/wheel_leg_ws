"""Diagnostic local two-mode rig; not the robot's training environment.

Caller owns source contracts, initialization/context and all disk IO. The rig
only receives explicit parameters/targets and advances MuJoCo by1ms substeps.
"""
from collections import deque
import numpy as np

VERSION='local_leg_equivalent_rig_v2'


class LocalLegResponseRig:
    def __init__(self,params,delays_ms,xml):
        import mujoco
        self.mj=mujoco;self.params=np.array(params,float,copy=True);delay=np.asarray(delays_ms,float)
        if self.params.shape!=(2,4) or not np.isfinite(self.params).all():raise ValueError('Two finite J/B/K/tau rows required')
        if np.any(self.params[:,0]<=0) or np.any(self.params[:,1]<0) or np.any(self.params[:,3]<0):raise ValueError('Invalid inertia/damping/tau')
        if delay.shape!=(2,) or not np.isfinite(delay).all() or np.any(delay<0) or np.any(delay>30) or np.any(delay!=np.round(delay)):raise ValueError('Two integer transport ages0..30ms required')
        self.delays=delay.astype(int);self.model=mujoco.MjModel.from_xml_string(xml)
        if self.model.nq!=2 or self.model.nu!=2 or self.model.opt.timestep!=.001:raise ValueError('Local2DOF/1ms contract differs')
        if not np.all(self.model.actuator_forcelimited) or not np.allclose(self.model.actuator_forcerange,[[-40,40]]*2):raise ValueError('Physical40Nm motor caps required')
        if not np.allclose(self.model.dof_damping,2*self.params[:,1]):raise ValueError('Modal damping mismatch')
        if not np.allclose(self.model.body_inertia[1:,2],2*self.params[:,0]):raise ValueError('Modal inertia mismatch')
        self.alpha=np.where(self.params[:,3]==0,1.,-np.expm1(-1/np.maximum(self.params[:,3],1e-12)))
        self.physical_steps=0;self.state_writes=0;self.max_motor_nm=0.;self.max_route_error_nm=0.;self.ready=False

    def reset(self,q,v,motor,loads):
        values=[np.array(x,float,copy=True) for x in (q,v,motor,loads)]
        if any(x.shape!=(2,) or not np.isfinite(x).all() for x in values):raise ValueError('Finite2D initialization required')
        self.data=self.mj.MjData(self.model);q,v,self.motor,self.loads=values
        self.data.qpos[:]=q;self.data.qvel[:]=v;self.mj.mj_forward(self.model,self.data)
        self.queues=[deque([self.motor[i]]*self.delays[i]) for i in range(2)]
        self.state_writes+=1;self.ready=True

    def state(self):
        if not self.ready:raise ValueError('Explicit reset required')
        return np.stack([self.data.qpos.copy(),self.data.qvel.copy(),self.motor.copy()])

    def step(self,target):
        if not self.ready:raise ValueError('Explicit reset required')
        target=np.asarray(target,float)
        if target.shape!=(2,) or not np.isfinite(target).all():raise ValueError('Finite modal target2 required')
        demand=40*(target-self.data.qpos)-2*self.data.qvel
        raw=np.clip([demand[0]-demand[1],demand[0]+demand[1]],-40,40)
        demand=np.array([np.mean(raw),(raw[1]-raw[0])*.5])
        for tick in range(5):
            delivered=demand.copy()
            for i,queue in enumerate(self.queues):
                if queue:delivered[i]=queue.popleft();queue.append(demand[i])
            self.motor+=self.alpha*(delivered-self.motor)
            physical=np.clip([self.motor[0]-self.motor[1],self.motor[0]+self.motor[1]],-40,40)
            self.data.ctrl[:]=physical
            external=-2*(self.params[:,2]*self.data.qpos+self.loads)
            self.data.qfrc_applied[:]=external
            self.mj.mj_step(self.model,self.data);self.physical_steps+=1
            expected=np.array([sum(physical),physical[1]-physical[0]])
            error=float(np.max(abs(self.data.qfrc_actuator-expected)))
            self.max_route_error_nm=max(self.max_route_error_nm,error)
            self.max_motor_nm=max(self.max_motor_nm,float(np.max(abs(self.data.actuator_force))))
            if not np.isfinite(np.r_[self.data.qpos,self.data.qvel]).all() or error>1e-8:raise RuntimeError('Nonfinite state or force-route mismatch')
            if tick==0:self.first_substep_force=np.r_[physical,self.data.qfrc_actuator,external,self.data.qfrc_passive]
        return self.state()
