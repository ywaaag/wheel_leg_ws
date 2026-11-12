"""Opt-in GPU MCU execution wrapper around the existing Motion environment."""
import torch
from .motion_timing_runtime import Timing
from .motion_mcu_execution import MotorRing, pd_and_motor, spring_compensation, profile, LEGS, WHEELS


class MCUTiming(Timing):
    def __init__(self,env,configuration,stage):
        super().__init__(env,configuration,stage)
        self.mcu_profile=profile();self.motor_ring=MotorRing(self.n,self.device)
        self.driver=torch.zeros_like(self.actual);self.motor_actual=self.actual.clone()
        self.compensation=self.actual.clone();self.spring_actual=self.actual.clone()
        self.pd_limited=self.actual.clone();self.last_pd_sensors=self.sensors[0].clone()
        self.adopted_origin=torch.zeros(self.n,6,device=self.device,dtype=torch.long)
        self.geometry_valid=torch.ones(self.n,device=self.device,dtype=torch.bool)
        self.variant=torch.zeros(self.n,device=self.device,dtype=torch.long)
        self.tail=torch.zeros(1000,44,device=self.device)
        self.contact_runs=torch.zeros(self.n,2,device=self.device)
        self.stats_reset();self.last_stats_tick=-1
        self.reset(self.ids)

    def stats_reset(self):
        self.statistics=torch.zeros(5,device=self.device)
        self.contact_max=torch.zeros(2,device=self.device)

    def configure(self,level=None):
        self.override=4;self.configure_armature()

    def reset(self,ids):
        super().reset(ids)
        self.delays[ids]=0;self.feedback[ids]=10
        if not hasattr(self,'motor_ring'):return
        q=self.env.robot.data.joint_pos.torch[:,self.env.ids]
        comp,_,_,valid=spring_compensation(q[ids][:,LEGS])
        self.motor_ring.reset(ids,comp)
        self.driver[ids]=comp;self.motor_actual[ids]=comp;self.compensation[ids]=comp
        self.spring_actual[ids]=-comp;self.actual[ids]=0
        self.adopted_origin[ids]=self.tick;self.geometry_valid[ids]=valid
        self.last_pd_sensors[ids]=self.sensors[0,ids]
        self.contact_runs[ids]=0
        self.motor_ring.delay[ids]=torch.tensor(profile()['motor_delay_ms'],device=self.device)
        # One declared stress corner only, applied independently from local target adoption.
        stress=self.variant[ids]==2
        self.feedback[ids]=torch.where(stress,12,10)
        self.motor_ring.delay[ids]+=stress[:,None].long()*2

    def apply(self):
        e=self.env;d=e.robot.data;slot=self.tick%32
        q=d.joint_pos.torch[:,e.ids];dq=d.joint_vel.torch[:,e.ids]
        self.targets[slot]=self.requested;self.sensors[slot,:,:6]=q;self.sensors[slot,:,6:]=dq
        if self.tick%5==0:
            self.adopted.copy_(self.requested)
            self.adopted_origin.fill_(10*(self.tick//10))
            s=self.sensors[(self.tick-self.feedback)%32,self.ids]
            request,limited,comp,motor,valid=pd_and_motor(s[:,:6],s[:,6:],self.adopted)
            self.geometry_valid.copy_(valid);self.last_pd_sensors.copy_(s)
            self.pd_limited.copy_(limited);self.compensation.copy_(comp);self.driver.copy_(motor)
            self.demand.copy_(limited);e.effort_pre.copy_(request)
        actual=self.motor_ring.step(self.tick,self.driver)
        comp_now,_,_,valid=spring_compensation(q[:,LEGS])
        scale=1+.1*(self.variant==2).to(q.dtype)
        self.spring_actual.copy_(-comp_now*scale[:,None])
        self.geometry_valid &= valid
        self.motor_actual.copy_(actual);self.actual.copy_(actual+self.spring_actual)
        e.effort.copy_(self.actual)
        e.executed_leg.copy_(self.adopted[:,LEGS]);e.executed_wheel.copy_(self.adopted[:,WHEELS])
        e.robot.set_joint_effort_target_index(target=e.effort,joint_ids=e.ids)
        self.tail[self.tick%1000]=torch.cat((q.new_tensor([self.tick*.001]),self.episodes[:1].float(),
            self.motor_actual[0],self.driver[0],self.actual[0],q[0],dq[0],e.effort_pre[0],self.adopted[0]))
        e.pd_counter+=1;self.tick+=1

    def summary(self):
        r=super().summary()
        r.update(execution_profile=self.mcu_profile,actual_local_target_delay_ms=[0]*6,
                 motor_delay_ms_min=self.motor_ring.delay.amin(0).cpu().tolist(),
                 motor_delay_ms_max=self.motor_ring.delay.amax(0).cpu().tolist(),
                 geometry_valid=bool(self.geometry_valid.all()),
                 solver_limit_scope='algebraic motor plus hypothetical independent spring; not hardware authority',
                 motor_queue_in_critic=False,initial_motor_state=profile()['initial_motor'])
        return r


def subclass(base,execution):
    class MCUMotionEnv(base):
        def __init__(self,*args,**kwargs):
            self.mcu_execution=execution;self.mcu_ready=False
            super().__init__(*args,**kwargs)
            self.mcu_q=torch.tensor([execution['pose']['joint_pos'].get(n,0.) for n in self.robot.joint_names],device=self.device)
            self.mcu_override=None
            self.mcu_snapshot=torch.zeros(self.num_envs,24,device=self.device)
            self.mcu_ready=True

        def configure_mcu_cases(self,cases):
            self.mcu_override=torch.full((self.num_envs,),cases[0].get('_mcu_variant',0),device=self.device,dtype=torch.long)
            for group,row in enumerate(cases):self.mcu_override[group*8:(group+1)*8]=row.get('_mcu_variant',0)

        def _reset_idx(self,ids):
            super()._reset_idx(ids)
            if not self.mcu_ready:return
            if ids is None:ids=torch.arange(self.num_envs,device=self.device)
            e=self;d=e.robot.data
            variant=(ids%4!=0).long() if self.motion_config['course_stage']>=1 else torch.zeros_like(ids)
            if self.motion_config['course_stage']>=2:variant=torch.where(ids%4>=2,2,variant)
            if self.motion_config.get('course_version')=='motion31_mcu_measured_frontiers_v2':
                from .motion_mcu_measured import decode
                ready=decode(self.motion_config['course_stage'])['readiness']
                # Missing standing capability gets half the envs; mastered motion
                # remains nominal rehearsal rather than silently inheriting stress.
                focus=(ids<self.num_envs//2)&(self.command_index[ids]==0)
                variant=torch.where(focus,ready,0) if ready in (1,2) else torch.zeros_like(ids)
            if self.motion_config.get('course_version') in ('motion31_mcu_posture_rehearsal_v1','motion31_mcu_height_pd_parallel_v1','motion31_mcu_highspeed_standing2048_v1','motion31_standing_sensor20_target5_motor20_v1'):
                variant=torch.tensor(self.stage_spec['mcu_variants'],device=self.device,dtype=torch.long)[self.command_index[ids]]
            if not self.curriculum_training:variant=torch.zeros_like(ids)
            if self.mcu_override is not None:variant=self.mcu_override[ids]
            root=d.root_state_w.torch[ids,:7].clone();root[:,:3]=self.scene.env_origins[ids];root[:,2]+=.34
            root[:,3:]=root.new_tensor(execution['pose']['root_quaternion_xyzw'])
            q=self.mcu_q[None].expand(len(ids),-1).contiguous()
            velocity=torch.zeros(len(ids),6,device=self.device)
            velocity[:,0]=.05*(variant>0);velocity[:,4]=.10*(variant>0)
            self.robot.write_root_pose_to_sim_index(root_pose=root,env_ids=ids)
            self.robot.write_root_velocity_to_sim_index(root_velocity=velocity,env_ids=ids)
            self.robot.write_joint_position_to_sim_index(position=q,env_ids=ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q),env_ids=ids)
            for name in ('executed_leg','initial_leg_targets','leg_target'):getattr(self,name)[ids]=q[:,self.ids][:,LEGS]
            self.executed_wheel[ids]=0.;self.wheel_target[ids]=0.
            self.student_previous_physical[ids]=0.;self.student_previous_physical[ids[:,None],torch.tensor(LEGS,device=self.device)]=self.executed_leg[ids]
            self.student_h0[ids]=.34
            self.motion_timing.variant[ids]=variant.long();self.motion_timing.reset(ids)
            self.actor_sensor_packets.reset(ids)

        def _get_observations(self):
            version=self.critic_privileged_version
            if self.mcu_ready:self.critic_privileged_version='motion_antiosc_privileged140_v1'
            try:obs=dict(super()._get_observations())
            finally:self.critic_privileged_version=version
            if self.mcu_ready and 'student_critic' in obs:
                from .motion_height import critic_input
                from .motion_fixed_height import critic_features
                t=self.motion_timing;h=torch.full((self.num_envs,),.34,device=self.device)
                base=critic_input(obs['student_critic'],h,h,h,torch.zeros_like(h))
                ages=torch.tensor([1,11,21],device=self.device)
                queued=t.targets[(t.tick-ages)%32].permute(1,0,2)
                age=(t.tick-t.adopted_origin).clamp_min(0).float()
                s=t.last_pd_sensors
                plant=critic_features(queued,age,torch.ones_like(h),h.new_full(h.shape,.5),self.actor_sensor_packets.age,
                     s[:,LEGS],s[:,6:],h.new_zeros(self.num_envs,2),torch.zeros_like(h))
                obs['student_critic']=torch.cat((base,plant),-1)
                if version=='motion_mcu_motorqueue_privileged312_v1':
                    from .motion_mcu_privileged import critic_input as motor_critic,past_motor_commands
                    caps=base.new_tensor(profile()['driver_limits_nm']).expand(self.num_envs,-1)
                    obs['student_critic']=motor_critic(obs['student_critic'],
                        published_commands=past_motor_commands(t.motor_ring.history,t.tick),
                        motor_actual=t.motor_actual,driver_command=t.driver,compensation=t.compensation,
                        spring_torque=t.spring_actual,transport_ms=t.motor_ring.delay.float(),
                        tau_ms=t.motor_ring.tau,driver_limits=caps,PD_feedback_ms=t.feedback[:,None].float(),
                        PD_phase_ms=base.new_full((self.num_envs,1),t.tick%5))
                self.student_obs=obs
            return obs

        def _get_dones(self):
            result=super()._get_dones()
            if self.mcu_ready and getattr(self,'motion_quality_ready',False):
                t=self.motion_timing
                # Snapshot BEFORE reset: evaluator never mixes a new episode's motor state.
                self.mcu_snapshot.copy_(torch.cat((t.motor_actual,t.driver,t.compensation,t.spring_actual),-1))
                rr=self.student_reward_result
                index=rr['fields'].index('wheel_request_excess')
                from .motion_mcu_execution import WHEEL_CAP
                from .motion_antioscillation import huber
                excess=torch.relu(self.effort_pre[:,WHEELS].abs()/WHEEL_CAP-.8)
                mix=t.config['smooth_mix']
                new=((1-mix)*huber(excess/2)+mix*huber(excess)).mean(-1)*rr['eligible']
                old=rr['dt'][:,index].clone();rr['raw'][:,index]=new;rr['density'][:,index]=new*(-.25);rr['dt'][:,index]=rr['density'][:,index]*.01
                rr['total']+=rr['dt'][:,index]-old
                if self.motion_config.get('course_version')=='motion31_mcu_posture_rehearsal_v1':
                    from .motion_posture_design import apply_runtime_reward
                    apply_runtime_reward(rr,self.standing_snapshot)
                if t.last_stats_tick!=t.tick:
                    valid=rr['eligible'];lost=self.contact_force<=2
                    t.contact_runs.copy_(torch.where(lost&valid[:,None],t.contact_runs+1,0))
                    t.contact_max.copy_(torch.maximum(t.contact_max,t.contact_runs.amax(0)))
                    cap=t.driver.new_tensor(profile()['driver_limits_nm'])
                    t.statistics+=torch.stack((valid.sum(),((lost.sum(-1)==1)&valid).sum(),
                        ((lost.sum(-1)==2)&valid).sum(),(((t.driver.abs()>=cap-1e-4).any(-1))&valid).sum(),
                        ((~t.geometry_valid)&valid).sum())).float()
                    t.last_stats_tick=t.tick
                # Geometry invalidity is an engineering fault, not a policy ability gate.
                checks=getattr(self,'motion_post_physics_checks',None)
                if checks is not None:checks.require(t.geometry_valid,'MCU spring geometry invalid')
                elif not bool(t.geometry_valid.all()):raise RuntimeError('MCU spring geometry invalid')
            return result
    return MCUMotionEnv
