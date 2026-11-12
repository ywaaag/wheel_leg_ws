"""Opt-in20/5/20ms MCU standing plant and its causal411D value input."""
import torch

from .motion_mcu_runtime import MCUTiming
from .motion_mcu_execution import LEGS,WHEELS,pd_and_motor,spring_compensation
from .motion_standing_delay import execution_profile,CRITIC


class StandingDelayTiming(MCUTiming):
    def __init__(self,env,configuration,stage):
        super().__init__(env,configuration,stage)
        self.mcu_profile=execution_profile()
        self.reset(self.ids)

    def reset(self,ids):
        super().reset(ids)
        self.delays[ids]=5
        self.feedback[ids]=0
        self.tau[ids]=2.
        if hasattr(self,'motor_ring'):
            self.motor_ring.delay[ids]=20
            self.motor_ring.tau[ids]=2.

    def apply(self):
        e=self.env;d=e.robot.data;slot=self.tick%32
        q=d.joint_pos.torch[:,e.ids];dq=d.joint_vel.torch[:,e.ids]
        self.targets[slot]=self.requested
        self.sensors[slot,:,:6]=q;self.sensors[slot,:,6:]=dq
        if self.tick%5==0:
            self.adopted.copy_(self.targets[(self.tick-5)%32])
            self.adopted_origin.fill_(10*((self.tick-5)//10))
            # Feedback is current at local PD time. Only the resulting torque waits20ms.
            s=self.sensors[slot]
            request,limited,comp,motor,valid=pd_and_motor(s[:,:6],s[:,6:],self.adopted)
            self.geometry_valid.copy_(valid);self.last_pd_sensors.copy_(s)
            self.pd_limited.copy_(limited);self.compensation.copy_(comp);self.driver.copy_(motor)
            self.demand.copy_(limited);e.effort_pre.copy_(request)
        actual=self.motor_ring.step(self.tick,self.driver)
        comp_now,_,_,valid=spring_compensation(q[:,LEGS])
        self.spring_actual.copy_(-comp_now)
        self.geometry_valid &= valid
        self.motor_actual.copy_(actual);self.actual.copy_(actual+self.spring_actual)
        e.effort.copy_(self.actual)
        e.executed_leg.copy_(self.adopted[:,LEGS]);e.executed_wheel.copy_(self.adopted[:,WHEELS])
        e.robot.set_joint_effort_target_index(target=e.effort,joint_ids=e.ids)
        self.tail[self.tick%1000]=torch.cat((q.new_tensor([self.tick*.001]),self.episodes[:1].float(),
            self.motor_actual[0],self.driver[0],self.actual[0],q[0],dq[0],e.effort_pre[0],self.adopted[0]))
        e.pd_counter+=1;self.tick+=1

    def summary(self):
        result=super().summary()
        result.update(actual_local_target_delay_ms=[5]*6,motor_queue_in_critic=True,
            critic_privileged_version=CRITIC,critic_motor_history_ms=20,
            critic_target_history_ms=5,critic_queued_sensor_packets=2,
            PD_feedback_age_ms=0,delay_semantics='sensor20 -> Actor -> target5 -> current PD -> motor20 -> response')
        return result


def subclass(base):
    class StandingDelayEnv(base):
        def _get_observations(self):
            version=self.critic_privileged_version
            if version==CRITIC:self.critic_privileged_version='motion_mcu_motorqueue_privileged312_v1'
            try:obs=super()._get_observations()
            finally:self.critic_privileged_version=version
            if version==CRITIC and getattr(self,'mcu_ready',False) and 'student_critic' in obs:
                from .motion_standing_delay import critic_input
                t=self.motion_timing;b=obs['student_critic']
                obs=dict(obs)
                obs['student_critic']=critic_input(b,t.motor_ring.history,t.targets,
                    self.actor_sensor_packets.pending(),t.tick,
                    b.new_tensor(execution_profile()['driver_limits_nm']).expand(self.num_envs,-1))
                self.student_obs=obs
            return obs

        def _get_dones(self):
            result=super()._get_dones()
            if getattr(self,'motion_quality_ready',False):
                from .motion_posture_design import apply_runtime_reward
                apply_runtime_reward(self.student_reward_result,self.standing_snapshot)
            return result
    return StandingDelayEnv


def runtime_readback(env,obs):
    t=env.motion_timing;p=env.actor_sensor_packets
    checks=dict(actor31=obs['student_actor'].shape==(env.num_envs,31),
        critic411=obs['student_critic'].shape==(env.num_envs,411),
        target5=bool((t.delays==5).all()),PD_current=bool((t.feedback==0).all()),
        motor20=bool((t.motor_ring.delay==20).all()),sensor20=p.delay_ms==20,
        zero_command=bool((env.target_xy==0).all()),height034=bool(torch.isclose(env.height_adapter.target,
            env.height_adapter.target.new_tensor(.34)).all()),
        finite=bool(torch.isfinite(obs['student_critic']).all()))
    if not all(checks.values()):raise RuntimeError('standing delay411 runtime wiring: '+str(checks))
    return dict(status='PASSED',before_first_optimizer=True,checks=checks,
        actual_profile=t.summary(),slot_counts=torch.bincount(env.command_index,minlength=2).cpu().tolist(),
        scope='execution engineering only; independent full-window ability not claimed')
