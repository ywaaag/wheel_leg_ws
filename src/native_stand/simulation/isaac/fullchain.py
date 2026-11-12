"""Full v4 execution and explicit causal511D Critic, opt-in sparse view only."""

import json
from pathlib import Path

import torch
from critic_tensor import input_tensor
from robot_adapter import LEGS, WHEELS, Timing
from runtime_support import noise

from .sensor_packets import Packets


def base511_critic(env, actor, *, body_velocity):
    t = env.motion_timing
    b = t.base
    leg = t.leg
    d = t.dr
    pack = env.identified_packets
    n = env.num_envs
    from .critic_cache import CriticCache

    cache = getattr(env, "_motion_value_cache", None)
    if cache is None:
        cache = env._motion_value_cache = CriticCache(actor)
    scales = cache.scales
    history = cache.history
    static = cache.reset_fields(env)
    sh = torch.cat(
        (
            history(leg.qhist, leg.tick, list(range(1, 11))),
            history(leg.vhist, leg.tick, list(range(1, 11))),
        ),
        -1,
    ).reshape(n, -1)
    parameters = static["parameters"]
    age = torch.cat(
        (
            leg.delay.reshape(n, 2).float() / 40,
            leg.qage.reshape(n, 2).float() / 10,
            leg.vage.reshape(n, 2).float() / 10,
            pack.age[:, None] / 30,
            actor.new_full((n, 1), b.tick % 5 / 5),
        ),
        -1,
    )
    values = dict(
        actor_observation=actor,
        current_true_sensor_frame=pack.true_frame,
        height_error_m=(__import__("ground_height").from_env(env)[:, None] - 0.34),
        body_linear_velocity_mps=body_velocity,
        wheel_contact_over100N=env.contact_force / 100,
        adopted_target=b.adopted / scales,
        requested_target=b.requested / scales,
        target_queue5ms=history(b.targets, b.tick, range(1, 6)) / cache.target_scales,
        PD_reported_q_dq=torch.cat(
            (b.last_pd_sensors[:, :6] / scales, b.last_pd_sensors[:, 6:] * 0.05), -1
        ),
        actual_motor_over_cap=b.motor_actual / d.caps,
        net_external_load_over40=(t.last_external + env.transient_events.current)[:, LEGS] / 40,
        leg_modal_motor=leg.actual.reshape(n, 4) / 40,
        leg_modal_demand=leg.des.reshape(n, 4) / 40,
        leg_motor_history24ms=history(leg.queue, leg.tick, list(range(1, 25))).reshape(n, -1) / 40,
        leg_sensor_history10ms=sh,
        held_leg_sensor=torch.cat((leg.heldq / 1.5, leg.heldv * 0.05), -1).reshape(n, 8),
        wheel_motor_history24ms=history(
            b.motor_ring.history[:, :, WHEELS], b.tick, list(range(1, 25))
        )
        / d.caps[:, WHEELS].repeat(1, 24),
        wheel_actual_motor=b.motor_ring.actual[:, WHEELS] / d.caps[:, WHEELS],
        pending_actor_sensor_packets3=pack.pending().reshape(n, 48),
        complete_fitted_parameters=parameters.reshape(n, 26),
        fitted_q_reference=static["reference"],
        effective_sensor_ages=age,
        effective_PD_gains=static["gains"],
        effective_motor_caps=static["caps"],
        effective_wheel_timing=static["wheel_timing"],
        actual_DR_context=static["context"],
        active_armatures=d.actual_armatures,
        wheel_B_Tc=static["wheel_B_Tc"],
        episode_context=torch.stack(
            (
                env.rise_hold.episode.steps.float() * 0.01 / 30,
                (env.rise_hold.episode.steps >= 150).float(),
                d.pushed[:, 0].float(),
                d.pushed[:, 1].float(),
            ),
            -1,
        ),
    )
    return input_tensor(values)


def subclass(
    base,
    info,
    state,
    *,
    handover,
    dr_factory,
    episodes_factory,
    events_factory,
    terminal_critic_owner="fullchain",
):
    from ground_height import VERSION

    if info.get("ground_height", {}).get("version") != VERSION:
        raise ValueError("corrected ground-height runtime requires explicit version declaration")
    if terminal_critic_owner not in ("fullchain", "outer"):
        raise ValueError("unknown terminal Critic owner")

    class FullEnv(base):
        def _motion_state(self):
            result = super()._motion_state()
            if hasattr(self, "DR_static_ground_normal"):
                from ground_height import from_env

                self.root_height.copy_(from_env(self))
            return result

        def _geometry_gaps(self):
            if not self.identified_ready:
                return super()._geometry_gaps()
            from slope_tiles import geometry_gaps

            gaps, chassis = geometry_gaps(self)
            surface = getattr(self, "motion_surface", None)
            if surface is not None:
                from isaaclab.utils.math import quat_apply, quat_apply_inverse

                from ..terrain_geometry import vertical_support_points

                helper = self.DR_geometry_support
                points = vertical_support_points(
                    helper,
                    self.robot.data.body_pos_w.torch,
                    self.robot.data.body_quat_w.torch,
                    quat_apply_inverse,
                    quat_apply,
                )
                positions = points[..., :2]
                self.motion_road_exit |= ~surface.contains(
                    positions, margin=info["terrain_scene"]["boundary_margin_m"]
                ).all(-1)
                z, _ = surface.sample(positions, check=False)
                rough = (surface.family_ids > 0)[:, None]
                gaps = torch.where(rough, points[..., 2] - z, gaps)
                chassis = gaps[:, helper.chassis].amin(-1)
            return gaps, chassis

        def _setup_scene(self):
            super()._setup_scene()
            from slope_tiles import setup

            setup(self, info)
            if info.get("terrain_scene") is not None:
                from .terrain_scene import setup as setup_terrain

                setup_terrain(self, info["terrain_scene"])

        def __init__(self, *args, **kwargs):
            self.identified_ready = False
            super().__init__(*args, **kwargs)
            self.mcu_override = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            c = json.loads(Path(info["contract"]).read_text())
            from .execution_buffers import ExecutionBuffers

            self.motion_timing = ExecutionBuffers(
                self,
                info["execution_configuration"],
                armature_level=info["armature_level"],
                effort_caps=info.get("motor_effort_caps", (40.0, 40.0, 5.5, 40.0, 40.0, 5.5)),
            )
            self.motion_timing = Timing(self, c)
            optimization = info.get("execution_optimization")
            if optimization is not None:
                from .response_acceleration import accelerate

                self.motion_surface_query_cache = True
                self.motion_timing.leg = accelerate(
                    self.motion_timing.leg, compile_tick=optimization["compile_tensor_tick"]
                )
            self.identified_packets = Packets(self.motion_timing, noise=noise)
            self.actor_sensor_packets = self.identified_packets
            self.identified_dr = dr_factory(self, self.motion_timing, state)
            self.motion_dr = self.identified_dr
            self.motion_timing.dr = self.identified_dr
            self.fullchain_prev_motor = torch.zeros(self.num_envs, 6, device=self.device)
            self.fullchain_prev_targets = self.fullchain_prev_motor.clone()
            self.fullchain_history_valid = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            self.fullchain_r3_info = info
            from antiosc_reward import PhysicalHF

            self.physical_hf = PhysicalHF(self.num_envs, self.device, torch.float32)
            self.antiosc_physical_stats_post = torch.zeros(self.num_envs, 9, device=self.device)
            handover.initialize(self, info)
            self.identified_ready = True
            self.onpolicy_episodes = episodes_factory(self)
            self.transient_events = events_factory(self)

        def configure_mcu_cases(self, cases):
            super().configure_mcu_cases(cases)
            profiles = torch.tensor(
                [
                    cases[min(i // 8, len(cases) - 1)].get("_identified_profile", -1)
                    for i in range(self.num_envs)
                ],
                device=self.device,
                dtype=torch.long,
            )
            handover.configure(self, cases)
            self.identified_dr.override = profiles
            self.identified_dr.dq.override_amplitude = torch.tensor(
                [
                    cases[min(i // 8, len(cases) - 1)].get("_dq_amplitude", 0.0)
                    for i in range(self.num_envs)
                ],
                device=self.device,
            )
            self.identified_dr.override_amplitudes = torch.tensor(
                [
                    cases[min(i // 8, len(cases) - 1)].get("_dr_amplitude", 0.1)
                    for i in range(self.num_envs)
                ],
                device=self.device,
            )

        def _reset_idx(self, ids):
            if not self.identified_ready:
                return super()._reset_idx(ids)
            t = self.motion_timing
            d = t.dr
            t.dr = None
            try:
                super()._reset_idx(ids)
            finally:
                t.dr = d
            ids = t.ids if ids is None else ids
            if hasattr(self, "motion_road_exit"):
                self.motion_road_exit[ids] = False
                self.motion_terrain_height_valid[ids] = True
                from .terrain_scene import align_reset_height

                align_reset_height(self, ids)
            t.reset(ids)
            value_cache = getattr(self, "_motion_value_cache", None)
            if value_cache is not None:
                value_cache.invalidate()
            self.identified_packets.reset(ids)
            self.fullchain_history_valid[ids] = False
            self.physical_hf.reset(ids)
            from ground_height import from_env

            self.initial_height[ids] = from_env(self)[ids]
            self.fullchain_prev_targets[ids] = t.base.adopted[ids]
            self.fullchain_prev_motor[ids] = t.base.motor_actual[ids]
            handover.reset(self, ids)
            if hasattr(self, "onpolicy_episodes"):
                self.onpolicy_episodes.reset(ids)
            if hasattr(self, "transient_events"):
                self.transient_events.reset(ids)

        def actor_valid_pre(self):
            return handover.ordinary(self) & (self.r4_prefix_remaining == 0)

        def _pre_physics_step(self, actions):
            if not self.identified_ready:
                return super()._pre_physics_step(actions)

            self.identified_dr.maybe_push()
            actual, student = handover.actions(self, actions)
            super()._pre_physics_step(actual)
            self.motion_timing.base.requested[:, [0, 1, 3, 4]] = self.leg_target
            self.motion_timing.base.requested[:, [2, 5]] = self.wheel_target
            # 执行路径用物理目标换算；raw日志和PPO仍保留网络采样动作。
            self.raw_action.copy_(actions)
            self.student_actor_valid.copy_(student)
            self.rise_hold.motion.copy_(student)
            self.transient_events.policy(student)

        def _apply_action(self):
            if self.identified_ready:
                self.motion_timing.apply()
            else:
                super()._apply_action()

        def _get_observations(self):
            version = self.critic_privileged_version
            if self.identified_ready:
                self.critic_privileged_version = "motion_antiosc_privileged140_v1"
            try:
                obs = super()._get_observations()
            finally:
                self.critic_privileged_version = version
            if self.identified_ready and "student_actor" in obs:
                from runtime_support import noisy_actor_metadata

                obs = dict(obs)
                actor = obs["student_actor"].clone()
                actor[:, :16] = self.identified_packets.read(actor[:, :16], self.motion_timing.tick)
                obs["student_actor"] = noisy_actor_metadata(self, actor)
                obs["student_critic"] = critic(self, obs["student_actor"])
                self.student_obs = obs
            return obs

        def _get_dones(self):
            result = super()._get_dones()
            if self.identified_ready and getattr(self, "motion_quality_ready", False):
                from reward_extension import apply

                from native_stand.students.motion_antioscillation import apply_costs as anti_costs
                from native_stand.students.motion_posture_design import apply_runtime_reward
                from native_stand.students.motion_sensor_transfer import apply_costs

                t = self.motion_timing.base
                data = self.robot.data
                angular, _ = self._motion_state()
                self.student_reward_result = apply_costs(
                    self.student_reward_result,
                    targets=self.motion_spin_executed_targets,
                    previous=self.motion_spin_previous_targets,
                    dq=data.joint_vel.torch[:, self.ids],
                    vertical_velocity=data.root_lin_vel_w.torch[:, 2],
                    angular_body=angular,
                    torque=data.applied_torque.torch[:, self.ids],
                    config=t.config["reward_extension"],
                )
                self.student_reward_result = anti_costs(
                    self.student_reward_result,
                    self.effort_pre,
                    t.actual,
                    t.prev_torque,
                    t.adopted,
                    t.prev_target,
                    t.older_target,
                    self.contact_force,
                    t.history_valid,
                    t.config,
                )
                if t.last_reward_tick != t.tick:
                    t.critic_previous_torque.copy_(t.prev_torque)
                    t.older_target.copy_(t.prev_target)
                    t.prev_target.copy_(t.adopted)
                    t.prev_torque.copy_(t.actual)
                    t.history_valid.fill_(True)
                    t.last_reward_tick = t.tick
                apply_runtime_reward(self.student_reward_result, self.standing_snapshot)
                apply(self)
                from runtime_r3 import apply_reward

                apply_reward(self)
                from antiosc_reward import apply_antiosc

                apply_antiosc(self)
                from speed_reward import apply as apply_speed

                apply_speed(self)
                if hasattr(self, "onpolicy_episodes"):
                    self.onpolicy_episodes.observe(result)
                self.transient_events.observe()
                # Base snapshot predates quality timers and reward memory.
                # Bootstrap must use the final PRE-RESET successor state.
                if terminal_critic_owner == "fullchain":
                    terminated, truncated = result
                    if bool((truncated & ~terminated).any()):
                        self.student_terminal_critic = (
                            self._get_observations()["student_critic"].detach().clone()
                        )
            return result

    return FullEnv


def runtime_readback(env, obs):
    t = env.motion_timing
    b = t.base
    from ground_height import contract, from_env

    h = from_env(env)
    checks = dict(
        actor31=obs["student_actor"].shape == (env.num_envs, 31),
        critic863=obs["student_critic"].shape == (env.num_envs, 863),
        finite=bool(torch.isfinite(obs["student_critic"]).all()),
        full_tuple=True,
        separate_NET=True,
        no_old_C_spring=bool((b.compensation == 0).all()),
        target_queue_valid=bool(((b.delays >= 0) & (b.delays <= 20)).all()),
        normal_height034=bool(
            torch.isclose(env.height_adapter.target, torch.tensor(0.34, device=env.device)).all()
        ),
        caps_not_raised=bool((env.identified_dr.caps[:, LEGS] <= 40).all()),
        physics_clock_synchronized=t.leg.tick == b.tick,
        root_height_matches_current_plane=bool(
            torch.allclose(env.root_height, h, atol=2e-6, rtol=0)
        ),
        actual_ground_normal_matches_context=bool(
            torch.allclose(env.DR_static_ground_normal, env.identified_dr.normal, atol=1e-6, rtol=0)
        ),
    )
    if not all(checks.values()):
        raise RuntimeError("fullchain runtime wiring: " + str(checks))
    return dict(
        status="PASSED",
        before_first_optimizer=True,
        checks=checks,
        execution=t.summary(),
        execution_optimization=getattr(t.leg, "acceleration_readback", None),
        height_measurement=contract(),
        scope="engineering only; source/new task ability not a prerequisite",
    )


def critic(env, actor):
    from isaaclab.utils.math import quat_apply_inverse
    from privileged_dr import append
    from runtime_r3 import extra_value

    data = env.robot.data
    velocity = quat_apply_inverse(data.root_quat_w.torch, data.root_lin_vel_w.torch)
    return append(env, extra_value(env, base511_critic(env, actor, body_velocity=velocity)))
