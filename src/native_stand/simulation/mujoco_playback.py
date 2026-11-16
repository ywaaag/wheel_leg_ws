"""Motion播放器的唯一物理循环；默认完整V4，软件C与物理NET分路。"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from native_stand.artifacts.io import file_sha256, read_json
from native_stand.learning.inference import build_actor, spec_for_job
from native_stand.tasks.motion.encoding import CHANNELS

from .control_core import ControlCore
from .mujoco_model import model_from_description
from .playback_execution import WHEELS, PlaybackExecution


class MotionPlayer:
    def __init__(self, plan, payload, core, *, actor=None, diagnostic_target_transform=None):
        import mujoco

        self.mj = mujoco
        self.plan = plan
        self.diagnostic_target_transform = diagnostic_target_transform
        self.physics_dt = plan.job.control.physics_dt_s
        refs = plan.assets["bindings"]
        from native_stand.simulation.scenes import initial_catalog, scene_for

        self.scene_config, self.ground_height = scene_for(plan.job)
        self.initials = initial_catalog(
            self.scene_config, read_json(refs["initial_states"]["path"])
        )
        self.terrain = None
        if plan.assets.get("terrain_map"):
            from .playback_terrain import load_atlas

            self.terrain = load_atlas(plan.job.assets["terrain_scene"])
            self.ground_height = self.terrain.height
        self.spawn = "flat"
        self.recording = True
        self.reset_count = -1
        self.model, self.refs = model_from_description(
            read_json(refs["description"]["path"]),
            V4_contract=read_json(refs["V4"]["path"]),
            mechanics=self.initials["mechanics"],
            physical_probe=True,
            scene=self.scene_config,
            terrain=self.terrain,
            motor_caps=list(plan.job.control.effort_limits_nm),
        )
        from native_stand.students.mujoco_solver import apply as apply_solver

        solver_refs = dict(
            ground=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground"),
            wheels=[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in ("left_wheel", "right_wheel")
            ],
        )
        self.solver_readback = apply_solver(
            self.model, solver_refs, plan.assets.get("solver_profile", "legacy_xml_v1")
        )
        self.data = mujoco.MjData(self.model)
        self.qids = np.array([self.model.jnt_qposadr[self.refs[n]] for n in CHANNELS])
        self.vids = np.array([self.model.jnt_dofadr[self.refs[n]] for n in CHANNELS])
        lower = np.array([self.model.jnt_range[self.refs[n], 0] for n in CHANNELS], np.float32)
        upper = np.array([self.model.jnt_range[self.refs[n], 1] for n in CHANNELS], np.float32)
        lower[WHEELS] = -plan.job.action.wheel_scale_rad_s
        upper[WHEELS] = plan.job.action.wheel_scale_rad_s
        self.execution = PlaybackExecution(
            core, read_json(refs["V4"]["path"]), plan.timing, plan.job.control, lower, upper
        )
        self.actor_spec = spec_for_job(plan.job, plan.output_contract)
        self.actor = actor
        if self.actor is None:
            self.actor = build_actor(self.actor_spec).cpu().eval().requires_grad_(False)
            self.actor.load_state_dict(payload["actor"], strict=True)
        self.teacher = None
        self.teacher_core = None
        if plan.job.task == "motion" and plan.job.assets.get("prefix_onnx"):
            import onnx

            from native_stand.learning.ppo.onnx_prefix import from_onnx

            if file_sha256(plan.job.assets["prefix_onnx"]) != plan.job.assets["prefix_sha256"]:
                raise ValueError("播放器Teacher模型SHA不同")
            contract_path = plan.job.assets["prefix_output_contract"]
            if file_sha256(contract_path) != plan.job.assets["prefix_output_contract_sha256"]:
                raise ValueError("播放器Teacher合同SHA不同")
            self.teacher = (
                from_onnx(onnx.load(plan.job.assets["prefix_onnx"])).eval().requires_grad_(False)
            )
            self.teacher_core = ControlCore(core.binding, read_json(contract_path))
        self.robot_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        self.policy_ticks = 0
        self.actual_physics_steps = 0
        self.last_record = None
        self.records = deque(maxlen=10000)
        self.reset()

    def reset(self):
        self.mj.mj_resetData(self.model, self.data)
        height = self.plan.job.control.height_command_m
        key = "Swing" if self.plan.job.task == "recovery" else f"standing_h{height:.2f}"
        pose = self.initials["poses"].get(key)
        if pose is None:
            raise ValueError("播放高度缺少闭链合法初态")
        root = np.array(pose["root_pose_xyzw"])
        if self.terrain is not None:
            root[:2] = self.terrain.spawns[self.spawn]
        self.data.qpos[:3] = root[:3]
        self.data.qpos[2] += float(self.ground_height(root[None, :2])[0])
        self.data.qpos[3:7] = root[[6, 3, 4, 5]]
        for name, j in self.refs.items():
            self.data.qpos[self.model.jnt_qposadr[j]] = pose["joint_position"][name]
            self.data.qvel[self.model.jnt_dofadr[j]] = pose["joint_velocity"][name]
        self.mj.mj_forward(self.model, self.data)
        self.execution.reset(self.data.qpos[self.qids], self.data.qvel[self.vids])
        self.policy_ticks = 0
        self.actual_physics_steps = 0
        self.last_record = None
        self.reset_count += 1
        self.records.clear()
        if self.teacher_core is not None:
            self.teacher_core.adopted = self.execution.core.adopted

    def step(self, controls, *, external_torque_nm=None):
        e = self.execution
        d = self.data
        m = self.model
        if self.terrain is not None:
            self.terrain.check_bounds(d.qpos[None, :2])
        if self.plan.job.task == "recovery" and (controls.vx or controls.yaw):
            raise ValueError("Recovery播放只接受冻结静止命令")
        if e.tick % 10 == 0:
            rotation = d.xmat[self.robot_body].reshape(3, 3)
            gyro = rotation.T @ d.cvel[self.robot_body, :3]
            gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
            command = np.array(
                [
                    controls.vx,
                    controls.yaw,
                    self.plan.job.control.height_command_m
                    if controls.height is None
                    else controls.height,
                ],
                np.float32,
            )
            observation = e.observation(
                gyro, gravity, command, skill=0 if self.plan.job.task == "motion" else 1
            )
            if controls.teacher:
                if self.teacher is None:
                    raise ValueError("该模型没有绑定Teacher")
                network = self.teacher
                decoder = self.teacher_core
            else:
                network = self.actor
                decoder = e.core
            with torch.inference_mode():
                inputs = torch.from_numpy(observation[None])
                value = (
                    network.mean(inputs)
                    if controls.teacher
                    else network({"student_actor": inputs}, stochastic_output=False)
                )
                raw = value[0].numpy()
            requested = (
                raw.copy()
                if not controls.teacher and self.actor_spec.kind == "recovery_physical"
                else decoder.decode(raw, float(command[2]))
            )
            source_requested = requested.copy()
            if self.diagnostic_target_transform is not None and not controls.teacher:
                requested = self.diagnostic_target_transform(raw, float(command[2]), requested)
                if requested.shape != (6,) or not np.isfinite(requested).all():
                    raise ValueError("诊断目标变换必须返回有限六通道物理目标")
            e.submit(requested)
            self.policy_ticks += 1
            self.last_record = dict(
                command=command.copy(),
                raw=raw.copy(),
                requested=requested.copy(),
                source_requested=source_requested,
                previous=e.core.adopted.copy(),
                observation=observation.copy(),
                owner="Teacher" if controls.teacher else "Student",
            )
        motor, NET = e.step(d.qpos[self.qids], d.qvel[self.vids])
        d.ctrl[:] = motor
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[self.vids] = NET
        if external_torque_nm is not None:
            external = np.asarray(external_torque_nm, dtype=float)
            if external.shape != (6,) or not np.isfinite(external).all():
                raise ValueError("外扰必须为显式有限六通道关节力矩")
            d.qfrc_applied[self.vids] += external
        if self.recording and self.last_record is not None and (e.tick - 1) % 10 == 0:
            self.records.append(
                dict(
                    self.last_record,
                    policy_tick=self.policy_ticks,
                    adopted=e.core.adopted.copy(),
                    raw_PD=e.raw_PD.copy(),
                    driver=e.driver.copy(),
                    motor=motor.copy(),
                    NET=NET.copy(),
                )
            )
        self.mj.mj_step(m, d)
        self.actual_physics_steps += 1
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise FloatingPointError("播放器物理状态非有限")
        self.mj.mj_forward(m, d)
        for i in range(self.model.neq):
            a = self.mj.mj_name2id(m, self.mj.mjtObj.mjOBJ_SITE, f"loop{i}a")
            b = self.mj.mj_name2id(m, self.mj.mjtObj.mjOBJ_SITE, f"loop{i}b")
            if np.linalg.norm(d.site_xpos[a] - d.site_xpos[b]) > 0.005:
                raise ValueError("播放器闭链残差超过5mm；未投影状态")

    def telemetry(self):
        from scipy.spatial.transform import Rotation

        d, m, mj = self.data, self.model, self.mj
        rotation = d.xmat[self.robot_body].reshape(3, 3)
        flu = rotation[:, [2, 0, 1]] * np.array([-1.0, -1.0, 1.0])
        velocity = np.empty(6)
        mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, self.robot_body, velocity, 0)
        roll, pitch, _ = Rotation.from_matrix(flu).as_euler("xyz")
        contacts = {"left_wheel": False, "right_wheel": False}
        normal = {"left_wheel": 0.0, "right_wheel": 0.0}
        for index, contact in enumerate(d.contact):
            a, b = contact.geom
            for robot, ground in [(a, b), (b, a)]:
                if m.geom_bodyid[ground] != 0 or contact.dist > 0:
                    continue
                body = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[robot])
                if body in contacts:
                    contacts[body] = True
                    force = np.zeros(6)
                    mj.mj_contactForce(m, d, index, force)
                    normal[body] += max(0.0, float(force[0]))
        try:
            ground = float(self.ground_height(d.qpos[None, :2])[0])
        except ValueError:
            ground = None
        cmd = None if self.last_record is None else self.last_record["command"].tolist()
        return dict(
            simulation_time_s=float(d.time),
            steps=self.execution.tick,
            resets=self.reset_count,
            adopted_command=cmd,
            vx=float(velocity[3:] @ flu[:, 0]),
            vy=float(velocity[3:] @ flu[:, 1]),
            yaw=float(velocity[:3] @ flu[:, 2]),
            height=None if ground is None else float(d.qpos[2] - ground),
            root_z=float(d.qpos[2]),
            ground_z=ground,
            roll=float(roll),
            pitch=float(pitch),
            contact_geometric_right=contacts["left_wheel"],
            contact_geometric_left=contacts["right_wheel"],
            normal_force_geometric_right_N=normal["left_wheel"],
            normal_force_geometric_left_N=normal["right_wheel"],
            q=d.qpos[self.qids].tolist(),
            dq=d.qvel[self.vids].tolist(),
            raw=None if self.last_record is None else self.last_record["raw"].tolist(),
            requested=self.execution.requested.tolist(),
            adopted=self.execution.core.adopted.tolist(),
            motor=self.execution.motor.tolist(),
            NET=self.execution.NET.tolist(),
            position_xy=d.qpos[:2].tolist(),
            spawn=self.spawn,
            terrain="flat" if self.terrain is None else self.terrain.region(d.qpos[:2]),
            solver_profile=self.solver_readback["version"],
            owner="web" if getattr(self, "web_owner", False) else "keyboard",
        )

    def trace(self):
        if not self.records:
            return {}
        return {key: np.array([row[key] for row in self.records]) for key in self.records[0]}

    def close(self):
        self.data = None
        self.model = None
