"""Opt-in fixed-height adapter outside the shared MCU/handover plant."""
import torch


class FixedHeightAdapter:
    def __init__(self,env,cfg):
        self.env=env;self.override=None;self.initial=torch.full((env.num_envs,),.34,device=env.device)
        self.target=self.initial.clone();self.rate=torch.zeros_like(self.initial)
        self.heights=torch.tensor([p['height_m'] for p in cfg['height_poses']],device=env.device)
        self.poses=torch.tensor([[p['joint_pos'].get(n,0.) for n in env.robot.joint_names]
                                for p in cfg['height_poses']],device=env.device)
        self.reset(torch.arange(env.num_envs,device=env.device))

    def configure_cases(self,cases):
        self.override=torch.full_like(self.target,cases[0]['command'][2])
        for i,row in enumerate(cases):self.override[i*8:(i+1)*8]=row['command'][2]

    def reset(self,ids):
        e=self.env
        if ids is None:ids=torch.arange(e.num_envs,device=e.device)
        targets=(torch.tensor(e.stage_spec['height_targets_m'],device=e.device)[e.command_index[ids]]
                 if self.override is None else self.override[ids])
        index=(targets[:,None]-self.heights[None]).abs().argmin(-1)
        if not torch.allclose(self.heights[index],targets,atol=1e-6,rtol=0):
            raise ValueError('fixed height has no closure-solved reset')
        self.initial[ids]=targets;self.target[ids]=targets;self.rate[ids]=0.
        q=self.poses[index];root=e.robot.data.root_state_w.torch[ids,:7].clone()
        root[:,:3]=e.scene.env_origins[ids];root[:,2]+=targets
        e.robot.write_root_pose_to_sim_index(root_pose=root,env_ids=ids)
        e.robot.write_joint_position_to_sim_index(position=q,env_ids=ids)
        e.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q),env_ids=ids)
        # New-height cohorts have no artificial initial velocity or teacher.
        high=targets!=.34;chosen=ids[high]
        if len(chosen):e.robot.write_root_velocity_to_sim_index(root_velocity=root.new_zeros(len(chosen),6),env_ids=chosen)
        leg=q[:,e.ids][:,[0,1,3,4]]
        for field in ('executed_leg','initial_leg_targets','leg_target'):getattr(e,field)[ids]=leg
        e.executed_wheel[ids]=0.;e.wheel_target[ids]=0.
        e.student_previous_physical[ids]=0.
        e.student_previous_physical[ids[:,None],torch.tensor([0,1,3,4],device=e.device)]=leg
        e.student_h0[ids]=targets
        # Preserve reset-only PD-age draws while flushing histories from new q.
        ages=e.motion_timing.feedback[ids].clone()
        e.motion_timing.reset(ids);e.motion_timing.feedback[ids]=ages
        e.actor_sensor_packets.reset(ids)
        e._motion_reset_epoch=getattr(e,'_motion_reset_epoch',0)+1

    def command(self,target,age,training):
        from .motion_training import effective
        result=effective(target,age,training=training,config=self.env.motion_config)
        result[:,2]=self.target
        return result


def subclass(parent,cfg):
    class ParallelHeightMotion(parent):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.height_adapter=FixedHeightAdapter(self,cfg)

        def configure_mcu_cases(self,cases):
            super().configure_mcu_cases(cases)
            self.height_adapter.configure_cases(cases)

        def _reset_idx(self,ids):
            super()._reset_idx(ids)
            if hasattr(self,'height_adapter'):self.height_adapter.reset(ids)

        def _get_observations(self):
            obs=super()._get_observations()
            if hasattr(self,'height_adapter') and 'student_critic' in obs:
                a=self.height_adapter;obs=dict(obs);value=obs['student_critic'].clone()
                value[:,31]+=.34-self.command[:,2]
                value[:,140:144]=torch.stack((a.initial/.5,a.target/.5,self.command[:,2]/.5,a.rate),-1)
                obs['student_critic']=value;self.student_obs=obs
            return obs

        def _get_dones(self):
            result=super()._get_dones()
            if hasattr(self,'height_adapter') and getattr(self,'motion_quality_ready',False):
                from .motion_height import correct_reward
                self.student_reward_result=correct_reward(self.student_reward_result,
                    self.recovery_snapshot[:,6],self.motion_action_command[:,2],
                    self.robot.data.root_link_lin_vel_w.torch[:,2],self.height_adapter.rate)
            return result
    return ParallelHeightMotion
