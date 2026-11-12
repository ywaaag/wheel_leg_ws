"""Independent Recovery DirectRLEnv subclass. Import runtime only after launcher.

Only immutable NativeStandEnv scene/geometry and the bound V4 tensor response
are reused. Motion reward/863/course/runtime monkeypatches are never installed.
"""

from __future__ import annotations

import numpy as np
import torch

from native_stand.tasks.recovery.cubic_course import RecoveryCourse
from native_stand.tasks.recovery.cubic_randomization import draw

from .execution import LEGS, WHEELS, Execution
from .lifecycle import Lifecycle
from .physics import PhysicalParameters
from .reset import InitialStates, ended_snapshot, phase_record
from .routing import physical_control


def environment_class(
    initials, contract, response_class, source_C0, source_center, *, height_provider=None
):
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from pxr import Usd, UsdGeom, UsdPhysics

    from native_stand.env import NativeStandEnv

    class RecoveryEnv(NativeStandEnv):
        def __init__(self, cfg, **kwargs):
            from native_stand.simulation.scenes import FlatHeight, SceneConfig

            self.ground_height = height_provider or FlatHeight(SceneConfig())
            self.recovery_ready = False
            try:
                super().__init__(cfg, **kwargs)
                self._initialize_recovery(cfg)
            except BaseException as exc:
                try:
                    self.close()
                except BaseException as cleanup:
                    exc.add_note("Partial Recovery environment cleanup: " + repr(cleanup))
                raise

        def _initialize_recovery(self, cfg):
            from native_stand.learning.ppo.performance import Measurement

            self.reset_measurement = Measurement(
                getattr(cfg, "recovery_performance_measurement", False), self.device
            )
            n = self.num_envs
            self.execution = Execution(
                contract,
                response_class,
                n,
                device=self.device,
                effort_caps=getattr(
                    cfg,
                    "recovery_effort_caps",
                    [40.0, 40.0, 2.44694786377709, 40.0, 40.0, 2.44694786377709],
                ),
            )
            self.handover_mode = (
                getattr(cfg, "recovery_task_mode", None) == "Swing_to_frozen_Motion_v1"
            )
            life_class = Lifecycle
            course_class = RecoveryCourse
            if self.handover_mode:
                from native_stand.tasks.recovery.handover_course import (
                    RecoveryCourse as course_class,
                )

                from .handover_lifecycle import Lifecycle as life_class
            if getattr(cfg, "recovery_standalone_course", False):
                if self.handover_mode:
                    raise ValueError("Standalone course cannot enable Motion handover")
                from native_stand.tasks.recovery.standalone_course import (
                    RecoveryCourse as course_class,
                )
            if getattr(cfg, "recovery_swing_priority", False):
                from native_stand.tasks.recovery.swing_priority_course import (
                    RecoveryCourse as course_class,
                )
            if getattr(cfg, "recovery_preentry_reward", False):
                from .preentry_lifecycle import Lifecycle as life_class
            self.life = life_class(n, self.device)
            self.phase_readback = None
            if getattr(cfg, "recovery_preentry_reward", False):
                from native_stand.tasks.recovery.phase_readback import PhaseReadback

                self.phase_readback = PhaseReadback(n, self.device)
            self.life.effort_caps = self.execution.caps
            self.height_conditioned = getattr(cfg, "recovery_height_conditioned", False)
            if self.height_conditioned:
                self.height_initial = torch.full((n,), 0.34, device=self.device)
                self.height_dynamic = torch.zeros(n, device=self.device, dtype=torch.bool)
                self.life.height_conditioned = True
                self.life.command_height = self.height_initial.clone()
            self.physics = PhysicalParameters(self, contract, initials["mechanics"])
            self.init_catalog = initials
            self.initial_states = InitialStates(initials, self.robot.joint_names, self.device)
            self.courses = [course_class(cfg.seed + i) for i in range(n)]
            if getattr(cfg, "recovery_standalone_course", False):
                for course in self.courses:
                    course.stage = getattr(cfg, "recovery_standalone_initial_stage", 0)
            self.DR_rng = [np.random.default_rng(cfg.seed + 170000 + i) for i in range(n)]
            self.draws = [draw(0.0, rng) for rng in self.DR_rng]
            self.episode_spec = [None] * n
            self.episode_steps = torch.zeros(n, device=self.device, dtype=torch.long)
            self.initialized = torch.zeros(n, device=self.device, dtype=torch.bool)
            self.prefix_s = torch.zeros(n, device=self.device)
            self.teacher_seconds = self.prefix_s.clone()
            self.student_seconds = self.prefix_s.clone()
            self.policy_raw = torch.zeros(n, 6, device=self.device)
            self.control_physical = self.policy_raw.clone()
            self.student_mask = self.initialized.clone()
            self.last_external = self.policy_raw.clone()
            self.last_passive = self.policy_raw.clone()
            self.actual_physics_steps = 0
            self.HP10_history = torch.zeros(500, n, 4, device=self.device)
            self.HP10_sum = torch.zeros(n, 4, device=self.device)
            self.HP10_count = torch.zeros(n, device=self.device, dtype=torch.long)
            self.HP10_cursor = 0
            self.C0 = source_C0.to(self.device).eval().requires_grad_(False)
            self.C0_center = torch.tensor(
                source_center, device=self.device, dtype=self.policy_raw.dtype
            )
            self.fine_ring = torch.zeros(1000, n, 65, device=self.device)
            self.fine_probe = []
            self._obs = None
            self.terminal_critic = None
            self.transition = None
            self.completed = []
            self.completed_count = 0
            self._geometry_bindings()
            self.pulse_enabled = torch.zeros(n, device=self.device, dtype=torch.bool)
            self.pulse_onset = torch.zeros(n, device=self.device)
            self.pulse_pattern = torch.zeros(n, 4, device=self.device)
            self.recovery_ready = True

        def _geometry_bindings(self):
            joints = {
                p.GetName(): UsdPhysics.RevoluteJoint(p)
                for p in self.sim.stage.Traverse()
                if p.IsA(UsdPhysics.RevoluteJoint) and "/env_0/" in str(p.GetPath())
            }
            from native_stand.contract import JOINT_NAMES

            self.routes = []
            self.hip_anchors = []
            for channel, name in enumerate(JOINT_NAMES):
                joint = joints[name]
                parent = self.body_ids[joint.GetBody0Rel().GetTargets()[0].name]
                child = self.body_ids[joint.GetBody1Rel().GetTargets()[0].name]
                rotation = joint.GetLocalRot0Attr().Get()
                q = torch.tensor(
                    [[*rotation.GetImaginary(), rotation.GetReal()]], device=self.device
                )
                axis = torch.zeros(1, 3, device=self.device)
                axis[0, "XYZ".index(joint.GetAxisAttr().Get())] = 1
                local = quat_apply(q, axis)[0]
                self.routes.append((parent, child, local))
                if channel in [0, 3]:
                    self.hip_anchors.append(
                        (
                            parent,
                            torch.tensor(list(joint.GetLocalPos0Attr().Get()), device=self.device),
                        )
                    )
            # Real chassis colliders, conservative local bounding corners.
            root_path = self.robot.root_view.link_paths[0][0]
            cache = UsdGeom.XformCache()
            bbox = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy],
            )
            if root_path.rsplit("/", 1)[-1] != self.robot.body_names[0]:
                raise RuntimeError("Actual PhysX root-link body identity")
            self.root_sensor_id = self.sensor.body_names.index(self.robot.body_names[0])
            root = self.sim.stage.GetPrimAtPath(root_path)
            inverse = cache.GetLocalToWorldTransform(root).GetInverse()
            points = []
            for prim in self.sim.stage.Traverse():
                if not prim.HasAPI(UsdPhysics.CollisionAPI) or not str(prim.GetPath()).startswith(
                    root_path + "/"
                ):
                    continue
                body = prim
                while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    body = body.GetParent()
                if not body.IsValid() or str(body.GetPath()) != root_path:
                    continue
                box = bbox.ComputeLocalBound(prim).ComputeAlignedRange()
                if box.IsEmpty():
                    continue
                matrix = cache.GetLocalToWorldTransform(prim)
                for i in range(8):
                    points.append(list(inverse.Transform(matrix.Transform(box.GetCorner(i)))))
            if not points:
                raise RuntimeError("No actual chassis collider bounds")
            self.chassis_corners = torch.tensor(points, device=self.device)

        def measured(self):
            self._closure()
            data = self.robot.data
            q = data.joint_pos.torch[:, self.ids]
            v = data.joint_vel.torch[:, self.ids]
            quaternion = data.root_link_quat_w.torch
            position = data.root_link_pos_w.torch
            velocity = data.root_link_lin_vel_w.torch
            angular = quat_apply_inverse(quaternion, data.root_link_ang_vel_w.torch)
            gravity = quat_apply_inverse(
                quaternion, q.new_tensor([0.0, 0.0, -1.0]).expand(self.num_envs, -1)
            )
            frame = torch.cat((angular * 0.25, gravity, q[:, LEGS] / 1.5, v * 0.05), -1)
            centers = data.body_pos_w.torch[:, self.wheel_body_ids]
            anchors = []
            for parent, local in self.hip_anchors:
                anchors.append(
                    data.body_pos_w.torch[:, parent]
                    + quat_apply(data.body_quat_w.torch[:, parent], local.expand(self.num_envs, -1))
                )
            vectors = quat_apply_inverse(
                quaternion[:, None, :].expand(-1, 2, -1), centers - torch.stack(anchors, 1)
            )
            length = vectors.norm(dim=-1)
            swing = torch.atan2(-vectors[:, :, 2], -vectors[:, :, 1])
            self.leg_geometry = torch.cat((length, swing), -1)
            difference = abs(
                torch.atan2(
                    torch.sin(swing[:, 0] - swing[:, 1]), torch.cos(swing[:, 0] - swing[:, 1])
                )
            )
            forward = quat_apply(
                quaternion, q.new_tensor([0.0, 0.0, -1.0]).expand(self.num_envs, -1)
            )
            forward[:, 2] = 0
            forward /= forward.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            left = torch.cross(
                q.new_tensor([0.0, 0.0, 1.0]).expand(self.num_envs, -1), forward, dim=-1
            )
            matrix = self.sensor.data.force_matrix_w
            if matrix is None:
                raise RuntimeError(
                    "Ground-filtered normal contact matrix required; norm fallback forbidden"
                )
            matrix = matrix.torch
            if matrix.ndim != 4 or matrix.shape[-1] != 3:
                raise RuntimeError("Unexpected Isaac contact force matrix shape")
            contacts = matrix[:, self.wheel_sensor_ids].sum(-2)[:, :, 2].clamp_min(0)
            state = torch.stack(
                (
                    (velocity * forward).sum(-1),
                    (velocity * left).sum(-1),
                    angular[:, 1],
                    position[:, 2] - self.ground_height(position[:, :2]),
                    torch.atan2(gravity[:, 2], -gravity[:, 1]),
                    torch.atan2(gravity[:, 0], -gravity[:, 1]),
                    length[:, 0] - length[:, 1],
                    ((centers[:, 0] - centers[:, 1]) * forward).sum(-1),
                    difference,
                    contacts[:, 0],
                    contacts[:, 1],
                    self.closure_each.amax(-1),
                    -gravity[:, 1],
                ),
                -1,
            )
            corners = (
                quat_apply(
                    quaternion[:, None, :].expand(-1, len(self.chassis_corners), -1),
                    self.chassis_corners[None].expand(self.num_envs, -1, -1),
                )
                + position[:, None]
            )
            clearance = corners[:, :, 2].amin(-1)
            root_contact = matrix[:, self.root_sensor_id].sum(-2).norm(dim=-1)
            return frame, q, v, state, position, velocity, clearance, root_contact, angular

        def commanded_height(self):
            from native_stand.tasks.recovery.height_context import command

            return command(
                self.height_initial, self.height_dynamic, self.episode_steps.float() * 0.01
            )

        def recovery_control_owner(self):
            owner = self.initialized & (self.episode_steps * 0.01 >= self.prefix_s)
            return owner & ~self.life.handover_active if self.handover_mode else owner

        def _pre_physics_step(self, actions):
            if not self.recovery_ready:
                return super()._pre_physics_step(actions)
            self.policy_raw.copy_(actions)
            actual = actions.clone()
            self.student_mask.copy_(self.recovery_control_owner())
            actual = physical_control(
                actions, self._obs["student_actor"], self.student_mask, self.C0, self.C0_center
            )
            self.control_physical.copy_(actual)
            self.execution.submit(actual)

        def _apply_action(self):
            if not self.recovery_ready:
                return super()._apply_action()
            q = self.robot.data.joint_pos.torch[:, self.ids]
            v = self.robot.data.joint_vel.torch[:, self.ids]
            pre_tick = self.execution.tick
            age = self.episode_steps.float() * 0.01 + (pre_tick % 10) * 0.001
            motor, net = self.execution.step(q, v)
            self.effort.copy_(motor)
            self.robot.set_joint_effort_target_index(target=motor, joint_ids=self.ids)
            self.last_passive.copy_(
                -self.physics.damping[:, self.ids] * v
                - self.physics.coulomb[:, self.ids] * torch.sign(v)
            )
            # Active-joint viscous and candidate kinetic friction are external.
            # Passive-joint damping remains PhysX; neither enters motor caps.
            self.last_external.zero_()
            # Finite scalar external pulses are causal, requested by reset
            # episode specification, not hidden additional motor commands.
            dt = age - self.pulse_onset
            active = self.pulse_enabled & (dt > 0) & (dt < 0.1)
            self.last_external[:, LEGS] = (
                self.pulse_pattern * (2 * torch.sin(torch.pi * dt / 0.1) * active)[:, None]
            )
            forces = torch.zeros(self.num_envs, len(self.robot.body_names), 3, device=self.device)
            torques = torch.zeros_like(forces)
            total = net + self.last_external + self.last_passive
            for channel, (parent, child, local) in enumerate(self.routes):
                axis = quat_apply(
                    self.robot.data.body_quat_w.torch[:, parent], local.expand(self.num_envs, -1)
                )
                tau = axis * total[:, channel : channel + 1]
                torques[:, parent] -= tau
                torques[:, child] += tau
            self.robot.set_external_force_and_torque(forces, torques, is_global=True)
            ex = self.execution
            pdq = ex.unpair(ex.leg.heldq)
            pdq[:, LEGS] += ex.bias
            pdv = ex.unpair(ex.leg.heldv)
            pdq[:, WHEELS] = q[:, WHEELS]
            pdv[:, WHEELS] = v[:, WHEELS]
            raw_pd = ex.unpair(ex.leg.des_unclipped)
            raw_pd[:, WHEELS] = ex.wheel_raw
            fine = torch.cat(
                (
                    q.new_full((self.num_envs, 1), pre_tick * 0.001),
                    age[:, None],
                    self.reset_count[:, None].to(q.dtype),
                    q,
                    v,
                    motor,
                    net,
                    self.last_external,
                    self.last_passive,
                    ex.adopted,
                    ex.request_tick[:, None].to(q.dtype),
                    ex.adopted_origin[:, None].to(q.dtype),
                    pdq,
                    pdv,
                    raw_pd,
                ),
                -1,
            )
            self.fine_ring[pre_tick % 1000] = fine
            if getattr(self.cfg, "recovery_probe_cases", None) is not None:
                self.fine_probe.append(fine.detach().cpu())
            self.actual_physics_steps += self.num_envs

        def _get_dones(self):
            if not self.recovery_ready:
                return super()._get_dones()
            frame, q, v, state, pos, vel, clearance, restricted, angular = self.measured()
            self.episode_steps += 1
            fault = (
                (~torch.isfinite(torch.cat((state, q, v), -1)).all(-1))
                | (state[:, 11] > 0.005)
                | (state[:, 12] < 0.5)
                | (state[:, 3] < 0.10)
            )
            time = self.episode_steps.float() * 0.01
            hp = self.execution.physical_HP10()
            slot = self.HP10_cursor % 500
            self.HP10_cursor += 1
            self.HP10_sum += hp.square() - self.HP10_history[slot]
            self.HP10_history[slot] = hp.square()
            self.HP10_count = (self.HP10_count + 1).clamp_max(500)
            if self.height_conditioned:
                self.life.command_height = self.commanded_height()
                self.life.minimum_success_time = torch.where(
                    self.height_dynamic,
                    self.height_initial.new_full((self.num_envs,), 12.0),
                    self.height_initial.new_zeros(self.num_envs),
                )
            reward_inputs = (
                dict(physical_q=q, clearance=clearance)
                if getattr(self.cfg, "recovery_preentry_reward", False)
                else {}
            )
            result = self.life.observe(
                time_s=time,
                state=state,
                target=self.execution.adopted if reward_inputs else self.control_physical,
                motor=self.execution.motor,
                student=self.student_mask,
                terminal=fault,
                truncated=time >= 23.0,
                hp10=hp,
                angular_velocity=angular,
                **reward_inputs,
            )
            if self.phase_readback is not None:
                self.phase_readback.update(
                    self.leg_geometry, state[:, 3], self.life.entry, self.life.hold
                )
            self.student_seconds += 0.01 * self.student_mask
            self.teacher_seconds += 0.01 * (~self.student_mask)
            self.terminal_critic = self._get_observations()["student_critic"].clone()
            ex = self.execution
            self.transition = {
                name: getattr(ex, name).detach().clone()
                for name in [
                    "raw",
                    "range_request",
                    "requested",
                    "adopted",
                    "motor",
                    "NET",
                    "adopted_origin",
                ]
            }
            self.transition.update(
                policy_raw=self.policy_raw.detach().clone(),
                external=self.last_external.detach().clone(),
                passive=self.last_passive.detach().clone(),
                physical_state=state.detach().clone(),
                reward_terms={k: v.detach().clone() for k, v in result["density"].items()},
                student_mask=self.student_mask.clone(),
                episode_id=self.reset_count.clone(),
                actual_actuator=self.robot.data.applied_torque.torch[:, self.ids].clone(),
            )
            self.transition["Motion_handover_active"] = (
                self.life.handover_active.clone()
                if self.handover_mode
                else torch.zeros_like(self.student_mask)
            )
            self.transition["handover_event"] = (
                result["entered"].clone()
                if self.handover_mode
                else torch.zeros_like(self.student_mask)
            )
            self.transition["initial_Swing"] = self.life.h0 < 0.27
            self.transition["task_success"] = result["success"].clone()
            terminated = result["terminated"]
            if getattr(self.cfg, "recovery_probe_full23", False):
                terminated = terminated & ~result["success"]
            return terminated, result["truncated"]

        def refresh_pulses(self, ids=None, *, rows=None):
            ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
            patterns = self.effort.new_tensor(
                [
                    [0.5, 0.5, 0.5, 0.5],
                    [0.5, -0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5, 0.5],
                ]
            )
            rows = ids.cpu().tolist() if rows is None else rows
            self.pulse_enabled[ids] = torch.tensor(
                [bool(self.episode_spec[i] and self.episode_spec[i]["pulse"]) for i in rows],
                device=self.device,
            )
            self.pulse_onset[ids] = (
                self.prefix_s[ids].where(
                    torch.tensor(
                        [self.episode_spec[i]["group"] == "dynamic_catch" for i in rows],
                        device=self.device,
                    ),
                    self.prefix_s.new_full((len(rows),), 2.95),
                )
                + 0.05
            )
            mode = (ids + self.reset_count[ids]) % 4
            sign = torch.where(ids % 2 == 0, 1.0, -1.0)
            self.pulse_pattern[ids] = patterns[mode] * sign[:, None]

        def _get_rewards(self):
            return self.life.reward.clone() if self.recovery_ready else super()._get_rewards()

        def physical_profile(self):
            from .observations import physical_profile

            return physical_profile(self)

        def _get_observations(self):
            if not self.recovery_ready:
                return super()._get_observations()
            from .observations import observe

            return observe(self)

        def _reset_idx(self, env_ids):
            if not self.recovery_ready:
                return super()._reset_idx(env_ids)
            with self.reset_measurement.section("reset"):
                return self._reset_recovery(env_ids)

        def _reset_recovery(self, env_ids):
            ids = (
                torch.arange(self.num_envs, device=self.device)
                if env_ids is None
                else env_ids.long()
            )
            if len(ids) == 0:
                return
            with self.reset_measurement.section("reset_statistics"):
                ended = ended_snapshot(self, ids)
            rows = [int(row[0]) for row in ended]
            for row in ended:
                i, steps = int(row[0]), int(row[1])
                if self.episode_spec[i] is not None and steps > 0 and bool(row[2]):
                    success, quality = bool(row[3]), bool(row[4])
                    self.courses[i].record(
                        self.episode_spec[i]["group"],
                        student_seconds=row[5],
                        teacher_seconds=row[6],
                        passed=success,
                        nominal=self.episode_spec[i]["nominal_anchor"],
                        physical_quality=quality,
                    )
                    self.completed.append(
                        dict(
                            Motion_confirmed=success if self.handover_mode else None,
                            Recovery_handover_reached=bool(row[7]) if self.handover_mode else None,
                            Recovery_handover_time_s=row[8] if self.handover_mode else None,
                            env_id=i,
                            command_height_m=self.episode_spec[i].get("height_m", 0.34),
                            height_dynamic=self.episode_spec[i].get("height_dynamic", False),
                            group=self.episode_spec[i]["group"],
                            nominal_anchor=self.episode_spec[i]["nominal_anchor"],
                            DR_strength=self.episode_spec[i]["DR_strength"],
                            physical_quality=quality,
                            success=success,
                            student_seconds=row[5],
                            teacher_seconds=row[6],
                            steps=steps,
                        )
                    )
                    if self.phase_readback is not None:
                        self.completed[-1].update(
                            phase_readback=phase_record(row),
                            end_reason="success"
                            if success
                            else "timeout"
                            if steps >= 2300
                            else "fault",
                        )
                    self.completed_count += 1
                    if len(self.completed) > 4096:
                        self.completed = self.completed[-4096:]
                spec = self.courses[i].choose()
                fixtures = getattr(self.cfg, "recovery_probe_cases", None)
                if fixtures is not None:
                    case = fixtures[i]
                    strength = case.get("DR_strength", 0.0)
                    spec = dict(
                        group=case.get("group", "standing_review"),
                        standing=case.get("standing", True),
                        prefix_s=case.get("prefix_s", 0.0),
                        pulse=case.get("pulse", False),
                        DR_strength=strength,
                        nominal_anchor=strength == 0.0,
                    )
                    if "height_m" in case:
                        spec["height_m"] = float(case["height_m"])
                    if "height_dynamic" in case:
                        spec["height_dynamic"] = bool(case["height_dynamic"])
                if self.height_conditioned:
                    count = self.courses[i].episodes
                    spec["height_dynamic"] = spec.get(
                        "height_dynamic", bool(spec["standing"] and count % 4 == 3)
                    )
                    spec["height_m"] = spec.get(
                        "height_m",
                        0.34
                        if not spec["standing"] or count % 2 == 0 or spec["height_dynamic"]
                        else [0.30, 0.32, 0.34, 0.36, 0.38][(i + count // 2) % 5],
                    )
                self.episode_spec[i] = spec
                self.draws[i] = draw(spec["DR_strength"], self.DR_rng[i])
                self.draws[i]["friction"] *= self.cfg.ground_dynamic_friction / 0.5
            # A real episode reset only. Continuous prefix->student change
            # never calls this method, clears a queue, or overwrites q/dq.
            super()._reset_idx(ids)
            specs = [self.episode_spec[i] for i in rows]
            if self.height_conditioned:
                self.height_initial[ids] = self.height_initial.new_tensor(
                    [p["height_m"] for p in specs]
                )
                self.height_dynamic[ids] = self.height_dynamic.new_tensor(
                    [p["height_dynamic"] for p in specs]
                )
            q, dq, root, velocity = self.initial_states.select(
                specs, self.scene.env_origins[ids], height_conditioned=self.height_conditioned
            )
            from native_stand.simulation.scenes import validate_reset_pose

            validate_reset_pose(root, q, dq, self.ground_height)
            self.robot.write_root_link_pose_to_sim_index(root_pose=root, env_ids=ids)
            self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)
            self.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=dq, env_ids=ids)
            draws = [self.draws[i] for i in rows]
            with self.reset_measurement.section("reset_physics"):
                self.physics.reset(ids, draws, host_ids=rows)
            ex = self.execution
            ex.bias[ids] = ex.bias.new_tensor(np.stack([d["encoder_bias_rad"] for d in draws]))
            ex.packet_age[ids] = ex.packet_age.new_tensor([d["Actor_age_ms"] for d in draws])
            ex.target_delay[ids] = ex.target_delay.new_tensor([d["target_delay_ms"] for d in draws])
            ex.wheel_delay[ids] = ex.wheel_delay.new_tensor(
                np.stack([d["wheel_delay_ms"] for d in draws])
            )
            self.refresh_pulses(ids, rows=rows)
            self.sim.forward()
            self._closure()
            if not bool((self.closure_each[ids].amax(-1) <= 0.005).all()):
                raise RuntimeError("Exported legal reset does not close in actual Isaac scene")
            ex.reset(ids, q[:, self.ids], dq[:, self.ids])
            self.episode_steps[ids] = 0
            self.initialized[ids] = True
            self.prefix_s[ids] = self.prefix_s.new_tensor(
                [self.episode_spec[i]["prefix_s"] for i in rows]
            )
            self.life.reset(ids, root[:, 2] - self.scene.env_origins[ids, 2], self.prefix_s[ids])
            if self.phase_readback is not None:
                self.measured()
                self.phase_readback.reset(
                    ids, self.leg_geometry[ids], root[:, 2] - self.scene.env_origins[ids, 2]
                )
            if self.height_conditioned:
                self.life.command_height[ids] = self.height_initial[ids]
            self.student_seconds[ids] = 0
            self.teacher_seconds[ids] = 0
            self.HP10_history[:, ids] = 0
            self.HP10_sum[ids] = 0
            self.HP10_count[ids] = 0

    return RecoveryEnv
