"""Height-aware reset and commands around the existing Motion/timing/DR plant."""
import torch
from . import motion_height as height


class HeightAdapter:
    def __init__(self, env, cfg):
        self.env = env; self.cfg = cfg; self.override = None
        self.initial = torch.full((env.num_envs,), .34, device=env.device)
        self.target = self.initial.clone()
        self.moving = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.rate = torch.zeros(env.num_envs, device=env.device)
        self.action_rate = self.rate.clone()
        self.episodes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.scenarios = torch.tensor(env.stage_spec['height_scenarios'], device=env.device)
        if env.motion_config.get('course_version') == 'motion31_persistent_frontiers_v1':
            self.height_frontier = torch.tensor([a == 'height' and p == 'frontier' for a,p in
                zip(env.stage_spec['cohort_axes'],env.stage_spec['cohort_purposes'])],device=env.device)
        self.initial_readback = None
        poses = cfg['height_poses']
        self.pose_heights = torch.tensor([p['height_m'] for p in poses], device=env.device)
        self.q = torch.tensor([[p['joint_pos'].get(n, 0.) for n in env.robot.joint_names] for p in poses], device=env.device)
        self.reset(torch.arange(env.num_envs, device=env.device))

    def configure_cases(self, cases):
        rows = torch.tensor([[r['_height']['initial_m'], r['_height']['target_m'], r['_height']['moving']]
                             for r in cases], device=self.env.device)
        self.override = rows[0].repeat(self.env.num_envs, 1)
        for group, row in enumerate(rows): self.override[group*8:(group+1)*8] = row

    def reset(self, ids):
        e = self.env
        if ids is None: ids = torch.arange(e.num_envs, device=e.device)
        self.episodes[ids] += 1
        if self.override is not None:
            values = self.override[ids]
        else:
            state = self.cfg['state']
            if state['focus'] == 'height' and state['focus_level'] >= 3:
                purposes = e.stage_spec['cohort_purposes']; start = purposes.index('focus')
                motions = 1 + sum(state['levels'][a] >= 0 for a in ('forward','backward','yaw_positive','yaw_negative'))
                if self.cfg.get('fixed_episode_height') and state['focus_level'] == 4:
                    motions = 1 + sum(state['levels'][a] + 1 for a in ('forward','backward','yaw_positive','yaw_negative'))
                pairs = (len(purposes) - start) // motions
                current = e.command_index[ids]
                # Counter stream is independent of Torch/CUDA policy sampling RNG.
                # Reduce by the LCG prime first: modulo20 directly would visit
                # only four height pairs because both multipliers divide by5.
                choice = ((ids * 1103515245 + self.episodes[ids] * 12345 + 20261023) % 2147483647) % pairs
                changed = start + choice * motions + (current - start).remainder(motions)
                e.command_index[ids] = torch.where(current >= start, changed, current)
            values = self.scenarios[e.command_index[ids], 2:5]
        self.initial[ids], self.target[ids] = values[:, 0], values[:, 1]
        self.moving[ids] = values[:, 2].bool()
        heights = self.pose_heights
        index = (values[:, :1] - heights).abs().argmin(-1)
        if ((heights[index] - values[:, 0]).abs() > 1e-5).any(): raise ValueError('height reset lacks closure-solved branch')
        q = self.q[index]
        root = e.robot.data.root_state_w.torch[ids, :7].clone()
        root[:, :3] = e.scene.env_origins[ids]
        root[:, 2] += self.initial[ids]
        # Preserve the parent's declared reset yaw; it does not change support height.
        e.robot.write_root_pose_to_sim_index(root_pose=root, env_ids=ids)
        e.robot.write_root_velocity_to_sim_index(root_velocity=root.new_zeros(len(ids), 6), env_ids=ids)
        e.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
        e.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=ids)
        leg = q[:, e.ids][:, [0, 1, 3, 4]]
        for name in ('executed_leg', 'initial_leg_targets', 'leg_target'):
            getattr(e, name)[ids] = leg
        e.executed_wheel[ids] = 0.; e.wheel_target[ids] = 0.
        e.student_previous_physical[ids] = 0.
        e.student_previous_physical[ids[:, None], torch.tensor([0, 1, 3, 4], device=e.device)] = leg
        e.student_h0[ids] = self.initial[ids]
        # Flush all delayed execution and sensing histories from the actual reset.
        if hasattr(e, 'motion_timing'): e.motion_timing.reset(ids)
        if hasattr(e, 'motion_dr'):
            d = e.robot.data
            e.motion_dr.root_history[:, ids] = torch.cat((d.root_quat_w.torch[ids], d.root_ang_vel_w.torch[ids]), -1)[None]
        if hasattr(e, 'actor_sensor_packets'): e.actor_sensor_packets.reset(ids)
        self.rate[ids] = 0.; self.action_rate[ids] = 0.
        if self.initial_readback is None:
            self.initial_readback = dict(requested_height_m=self.initial[ids[:8]].detach().cpu().tolist(),
                q_leg_rad=leg[:8].detach().cpu().tolist(), previous_units='physical rad/rad_s',
                offline_closure_valid=True, runtime_contacts_not_yet_certified=True)

    def command(self, target, age, training):
        command = height.command
        if self.cfg.get('fixed_episode_height'):
            from .motion_fixed_height import command
        result, rate = command(target, age, self.initial, self.target, self.moving, training)
        self.rate.copy_(rate)
        return result


def subclass(parent, cfg):
    class HeightMotionEnv(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.height_adapter = HeightAdapter(self, cfg)

        def _reset_idx(self, ids):
            super()._reset_idx(ids)
            if hasattr(self, 'height_adapter'): self.height_adapter.reset(ids)

        def _get_observations(self):
            original = self.critic_privileged_version
            if hasattr(self, 'height_adapter'): self.critic_privileged_version = 'motion_antiosc_privileged140_v1'
            try: obs = dict(super()._get_observations())
            finally: self.critic_privileged_version = original
            if hasattr(self, 'height_adapter') and 'student_critic' in obs:
                a = self.height_adapter
                obs['student_critic'] = height.critic_input(obs['student_critic'], a.initial, a.target,
                                                          self.command[:, 2], a.rate)
                if cfg.get('fixed_episode_height'):
                    obs['student_critic'] = torch.cat((obs['student_critic'], plant_features(self)), -1)
                self.student_obs = obs
            return obs

        def _pre_physics_step(self, action):
            if hasattr(self, 'height_adapter'): self.height_adapter.action_rate.copy_(self.height_adapter.rate)
            super()._pre_physics_step(action)

        def _get_dones(self):
            result = super()._get_dones()
            if hasattr(self, 'height_adapter') and getattr(self, 'motion_quality_ready', False):
                from .motion_height import correct_reward
                self.student_reward_result = correct_reward(self.student_reward_result,
                    self.recovery_snapshot[:, 6], self.motion_action_command[:, 2],
                    self.robot.data.root_link_lin_vel_w.torch[:, 2], self.height_adapter.action_rate)
                if hasattr(self.height_adapter,'height_frontier'):
                    from .motion_persistent import apply_height_bonus
                    mask=self.height_adapter.height_frontier[self.command_index] & bool(self.curriculum_training)
                    self.student_reward_result=apply_height_bonus(self.student_reward_result,mask)
            return result
    return HeightMotionEnv


def plant_features(env):
    """Read only existing GPU histories; no sampling, writes or CPU transfers."""
    from .motion_fixed_height import critic_features
    t = env.motion_timing; d = env.motion_dr
    ages = torch.tensor([1, 11, 21], device=env.device)
    queued = t.targets[(t.tick - ages) % 32].permute(1, 0, 2)
    adopted_age = torch.where(d.adopted_ticks >= 0, t.tick - d.adopted_ticks, 0).float()
    last_pd = ((t.tick - 1) // 5) * 5
    sensors = t.sensors[(last_pd - t.feedback) % 32, d.ids]
    any_push = d.pushed.any(-1); second = d.pushed[:, 1]
    direction = torch.stack((d.params[:, 6].cos(), d.params[:, 6].sin()), -1)
    applied_push = direction * d.params[:, 5:6] * torch.where(second, -1., 1.)[:, None] * any_push[:, None]
    age = env.rise_hold.episode.steps.float() * .01
    push_age = torch.where(any_push, (age - torch.where(second, 22., 12.)).clamp_min(0.), 0.)
    return critic_features(queued, adopted_age, d.params[:, 3], d.params[:, 4],
                           env.actor_sensor_packets.age.float(), sensors[:, [0,1,3,4]],
                           sensors[:, 6:], applied_push, push_age)


def engineering_probe(spec, actor, output, stop):
    from .evaluator import environment
    from native_stand.command_v6.completion import write
    env = environment(spec, 'MotionZeroStudent', envs=64)
    results = []
    try:
        env.curriculum_training = False; env.forced_target = (0., 0.)
        env.select_initial_family('motion_native_normal', 'train')
        fixed = spec['motion_decoupled'].get('fixed_episode_height', False)
        heights = [p['height_m'] for p in spec['motion_decoupled']['height_poses']]
        for initial in heights:
            if stop(): raise TimeoutError('height engineering probe stopped')
            env.height_adapter.configure_cases([dict(_height=dict(initial_m=initial, target_m=initial if fixed else .34, moving=False))])
            env.motion_dr.configure(dict(level=spec['motion_robustness']['state']['DR_level'], group=2, seed=20261023))
            obs, _ = env.reset()
            root = env.robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
            if not torch.allclose(root, torch.full_like(root, initial), atol=1e-5):
                raise RuntimeError('height initial root readback differs')
            q = env.robot.data.joint_pos.torch[:, env.ids][:, [0,1,3,4]]
            if not torch.allclose(q, env.executed_leg, atol=1e-5): raise RuntimeError('height reset previous target differs')
            if not torch.allclose(obs['student_actor'][:,24], torch.full_like(root, initial/.5), atol=1e-5):
                raise RuntimeError('height Actor command readback differs')
            with torch.inference_mode():
                for _ in range(20):
                    obs, reward, *_ = env.step(actor(obs))
                    if obs['student_actor'].shape != (64,31) or obs['student_critic'].shape != (64,184 if fixed else 144):
                        raise RuntimeError('height Actor/Critic runtime shape differs')
                    if fixed and (not torch.isfinite(obs['student_critic']).all()
                                  or not torch.allclose(env.command[:,2], torch.full_like(root,initial),atol=1e-5)):
                        raise RuntimeError('fixed height/Critic physical readback differs')
                    if not torch.isfinite(reward).all() or env.recovery_snapshot[:,4].bool().any():
                        raise RuntimeError('height engineering reward/runtime fault')
            results.append(dict(initial_height_m=initial, actor_height_readback=True,
                                physical_reset_previous_readback=True, DR=env.motion_dr.summary()))
        write(output/'height_engineering_readback.json',dict(status='PASSED', PPO_updates=0,
            physics_steps=20*10*len(heights), results=results,
            scope='height reset/Actor/Critic/reward/timing wiring only; ability failure does not block learning'))
    finally:
        env.close()
