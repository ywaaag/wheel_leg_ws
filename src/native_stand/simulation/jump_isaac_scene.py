"""Small PhysX scene/physical I/O consumer; runtime imports follow AppLauncher."""

from types import SimpleNamespace

import numpy as np
import torch

from native_stand.tasks.jump.geometry import CHANNELS


def tensor(value):
    if torch.is_tensor(value):
        return value
    return value.torch


class TickDerivatives:
    """Differentiate only physical time; a partial reset changes selected rows."""

    def __init__(self, dt):
        self.dt = dt
        self.tick = None
        self.previous = self.acceleration = None

    def sample(self, tick, velocity):
        if self.tick is not None and tick < self.tick:
            raise ValueError("Physical tick moved backwards")
        if tick != self.tick:
            self.acceleration = (
                torch.zeros_like(velocity)
                if self.tick is None
                else (velocity - self.previous) / ((tick - self.tick) * self.dt)
            )
            self.previous = velocity.clone()
            self.tick = tick
        return self.acceleration

    def reset_rows(self, ids, tick, velocity):
        self.sample(tick, velocity)
        self.previous[ids] = velocity[ids]
        self.acceleration[ids] = 0


class IsaacScene:
    def __init__(self, job, description, springs, mechanics, V4, initial):
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.utils import configclass
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        self.apply, self.inverse = quat_apply, quat_apply_inverse
        self.device, self.num_envs = job.runtime.device, job.runtime.envs
        self.dt = job.control.physics_dt_s
        self.sim = SimulationContext(
            SimulationCfg(dt=self.dt, device=self.device, render_interval=10)
        )

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path="/World/ground",
                collision_group=-1,
                spawn=sim_utils.CuboidCfg(
                    size=(128.0, 128.0, 0.1),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True, disable_gravity=True
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.5, dynamic_friction=0.5, restitution=0.0
                    ),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
            )
            robot = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=job.assets["jump_bindings"]["asset"]["path"],
                    activate_contact_sensors=True,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=8,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, initial.root_support_height_m),
                    rot=(0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
                    joint_pos=initial.q,
                    joint_vel={".*": 0.0},
                ),
                actuators={
                    "leg_motors": IdealPDActuatorCfg(
                        joint_names_expr=[CHANNELS[k] for k in (0, 1, 3, 4)],
                        effort_limit=40.0,
                        effort_limit_sim=40.0,
                        stiffness=0.0,
                        damping=0.0,
                    ),
                    "wheel_motors": IdealPDActuatorCfg(
                        joint_names_expr=[CHANNELS[k] for k in (2, 5)],
                        effort_limit=4.8,
                        effort_limit_sim=4.8,
                        stiffness=0.0,
                        damping=0.0,
                    ),
                    "passive_linkage": IdealPDActuatorCfg(
                        joint_names_expr=[
                            n
                            for n in mechanics["joint_names"]
                            if n not in CHANNELS and not n.startswith("guide_")
                        ],
                        effort_limit=0.0,
                        effort_limit_sim=0.0,
                        stiffness=0.0,
                        damping=0.0,
                    ),
                    "guide_wheels": IdealPDActuatorCfg(
                        joint_names_expr=[
                            n for n in mechanics["joint_names"] if n.startswith("guide_")
                        ],
                        effort_limit=0.0,
                        effort_limit_sim=0.0,
                        stiffness=0.0,
                        damping=0.0,
                    ),
                },
            )
            contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                history_length=0,
                update_period=0.001,
                filter_prim_paths_expr=["/World/ground"],
            )

        self.scene = InteractiveScene(
            SceneCfg(num_envs=self.num_envs, env_spacing=3.0, replicate_physics=True)
        )
        self.robot = self.scene["robot"]
        self.sensor = self.scene["contact_forces"]
        self.cfg = SimpleNamespace(ground_dynamic_friction=0.5)
        self.sim.reset()
        self.ids = [self.robot.joint_names.index(n) for n in CHANNELS]
        self.all_ids = [self.robot.joint_names.index(n) for n in mechanics["joint_names"]]
        if set(self.robot.joint_names) != set(mechanics["joint_names"]):
            raise ValueError("Actual PhysX full articulation joint identity")
        self.body_ids = {n: i for i, n in enumerate(self.robot.body_names)}
        self.root = self.body_ids["robot"]
        self.wheel_ids = [self.body_ids[side + "_wheel"] for side in ("left", "right")]
        self.wheel_sensor_ids = [
            self.sensor.body_names.index(side + "_wheel") for side in ("left", "right")
        ]
        self.restricted_ids = [
            k
            for k, n in enumerate(self.sensor.body_names)
            if n not in ("left_wheel", "right_wheel") and not n.startswith("guide_wheel_")
        ]
        self.anchors = []
        by_name = {j["name"]: j for j in description["joints"]}
        for n in CHANNELS:
            j = by_name[n]
            axis = np.asarray(j["frame0"])[:3, j["axis"]]
            self.anchors.append(
                (
                    self.body_ids[j["parent"]],
                    self.body_ids[j["child"]],
                    torch.tensor(axis, dtype=torch.float32, device=self.device),
                )
            )
        self.leg_anchors = [
            (
                self.body_ids[by_name[side + "_hip_pitch"]["parent"]],
                torch.tensor(
                    by_name[side + "_hip_pitch"]["frame0_local_position"],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
            for side in ("left", "right")
        ]
        self.spring_anchors = [
            [
                (
                    self.body_ids[a["body"].rsplit("/", 1)[-1]],
                    torch.tensor(a["local_m"], dtype=torch.float32, device=self.device),
                )
                for a in spring["anchors"]
            ]
            for spring in springs
        ]
        self.loop_anchors = [
            [
                (
                    self.body_ids[j["parent"]],
                    torch.tensor(j["anchor0"], dtype=torch.float32, device=self.device),
                ),
                (
                    self.body_ids[j["child"]],
                    torch.tensor(j["anchor1"], dtype=torch.float32, device=self.device),
                ),
            ]
            for j in description["loops"]
        ]
        self.geometry = description
        from native_stand.simulation.recovery_isaac.physics import PhysicalParameters
        from native_stand.tasks.recovery.cubic_randomization import draw

        self.physics = PhysicalParameters(self, V4, mechanics)
        self.physics.reset(
            torch.arange(self.num_envs, device=self.device),
            [draw(0.0, np.random.default_rng(0)) for _ in range(self.num_envs)],
        )
        self.forces = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.device)
        self.torques = torch.zeros_like(self.forces)
        self.derivatives = TickDerivatives(self.dt)
        self.actual_ticks = 0
        self.reset_epoch = 0
        self._cached = None

    def anchor(self, body, local):
        d = self.robot.data
        return tensor(d.body_pos_w)[:, body] + self.apply(
            tensor(d.body_quat_w)[:, body], local.expand(self.num_envs, -1)
        )

    def reset(self, ids, requests, initial):
        self.reset_epoch += 1
        device_ids = torch.tensor(ids, device=self.device, dtype=torch.int32)
        q = tensor(self.robot.data.default_joint_pos)[device_ids.long()].clone()
        for name, value in initial.q.items():
            q[:, self.robot.joint_names.index(name)] = value
        root = tensor(self.robot.data.default_root_state)[device_ids.long()].clone()
        root[:, :3] = self.scene.env_origins[device_ids.long()]
        root[:, 2] += initial.root_support_height_m + root.new_tensor(
            [r["drop_m"] for r in requests]
        )
        root[:, 3:7] = root.new_tensor([0.7071067811865476, 0.0, 0.0, 0.7071067811865476])
        root[:, 7:] = 0
        self.robot.write_root_link_pose_to_sim_index(root_pose=root[:, :7], env_ids=device_ids)
        self.robot.write_root_com_velocity_to_sim_index(
            root_velocity=root[:, 7:], env_ids=device_ids
        )
        self.robot.write_joint_state_to_sim_index(
            position=q, velocity=torch.zeros_like(q), env_ids=device_ids
        )
        self.scene.reset(device_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(self.dt)
        self.derivatives.reset_rows(
            device_ids.long(),
            self.actual_ticks,
            torch.cat(
                (
                    tensor(self.robot.data.root_link_lin_vel_w),
                    tensor(self.robot.data.root_link_ang_vel_w),
                    tensor(self.robot.data.joint_vel)[:, self.all_ids],
                ),
                -1,
            ),
        )

    def apply_effort(self, motor, NET):
        q = tensor(self.robot.data.joint_vel)[:, self.ids]
        passive = -self.physics.damping[:, self.ids] * q - self.physics.coulomb[
            :, self.ids
        ] * torch.sign(q)
        self.robot.set_joint_effort_target_index(target=motor, joint_ids=self.ids)
        self.forces.zero_()
        self.torques.zero_()
        for channel, (parent, child, local) in enumerate(self.anchors):
            axis = self.apply(
                tensor(self.robot.data.body_quat_w)[:, parent], local.expand(self.num_envs, -1)
            )
            torque = axis * (NET[:, channel : channel + 1] + passive[:, channel : channel + 1])
            self.torques[:, parent] -= torque
            self.torques[:, child] += torque
        self.robot.set_external_force_and_torque(self.forces, self.torques, is_global=True)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.dt)
        self.actual_ticks += 1
        return passive

    def measured(self):
        key = (self.actual_ticks, self.reset_epoch)
        if self._cached is not None and self._cached[0] == key:
            return self._cached[1]
        d = self.robot.data
        root = torch.cat(
            (
                tensor(d.root_link_pos_w) - self.scene.env_origins,
                tensor(d.root_link_quat_w),
                tensor(d.root_link_lin_vel_w),
                tensor(d.root_link_ang_vel_w),
            ),
            -1,
        )
        q = tensor(d.joint_pos)[:, self.all_ids]
        v = tensor(d.joint_vel)[:, self.all_ids]
        derivatives = self.derivatives.sample(self.actual_ticks, torch.cat((root[:, 7:13], v), -1))
        acceleration = derivatives[:, :6]
        joint = torch.stack((q, v, derivatives[:, 6:]), 1)
        lengths = []
        swings = []
        for k, (parent, local) in enumerate(self.leg_anchors):
            offset = self.inverse(
                root[:, 3:7],
                tensor(d.body_pos_w)[:, self.wheel_ids[k]] - self.anchor(parent, local),
            )
            lengths.append(offset[:, 1:3].norm(dim=-1))
            swings.append(torch.atan2(-offset[:, 2], -offset[:, 1]))
        spring = torch.stack(
            [
                (self.anchor(*pair[0]) - self.anchor(*pair[1])).norm(dim=-1)
                for pair in self.spring_anchors
            ],
            -1,
        )
        closure = torch.stack(
            [
                (self.anchor(*pair[0]) - self.anchor(*pair[1])).norm(dim=-1)
                for pair in self.loop_anchors
            ],
            -1,
        )
        matrix = self.sensor.data.force_matrix_w
        if matrix is None:
            raise RuntimeError("Actual filtered ground force matrix required")
        ground = tensor(matrix).sum(-2)
        normal = ground[:, self.wheel_sensor_ids, 2].clamp_min(0)
        contact = normal > 1.0
        restricted = ground[:, self.restricted_ids].norm(dim=-1).amax(-1) > 1.0
        com_pos = tensor(d.body_com_pos_w)
        com_vel = tensor(d.body_com_lin_vel_w)
        mass = self.physics.mass
        COM = (mass[..., None] * com_pos).sum(1) / mass.sum(1, keepdim=True)
        COMv = (mass[..., None] * com_vel).sum(1) / mass.sum(1, keepdim=True)
        gravity = self.inverse(
            root[:, 3:7], root.new_tensor([0, 0, -1.0]).expand(self.num_envs, -1)
        )
        tilt = torch.stack(
            (
                torch.atan2(gravity[:, 2], -gravity[:, 1]),
                torch.atan2(gravity[:, 0], -gravity[:, 1]),
            ),
            -1,
        )
        omega = self.inverse(root[:, 3:7], root[:, 10:13])
        clearance = []
        slip = []
        for body in self.wheel_ids:
            body_name = self.robot.body_names[body]
            shape = next(
                g
                for g in self.geometry["collision_shapes"]
                if g["body"] == body_name and g["kind"] == "Cylinder"
            )
            frame = np.asarray(shape["body_frame"])
            center = self.anchor(body, root.new_tensor(frame[:3, 3]))
            axis = self.apply(
                tensor(d.body_quat_w)[:, body],
                root.new_tensor(frame[:3, shape["axis"]]).expand(self.num_envs, -1),
            )
            vertical = axis[:, 2].abs()
            radius = shape["radius"]
            clearance.append(
                center[:, 2]
                - (
                    shape["half_height"] * vertical
                    + radius * torch.sqrt((1 - vertical.square()).clamp_min(0))
                )
            )
            local_bottom = center.clone()
            local_bottom[:, 2] -= radius
            velocity_at = tensor(d.body_com_lin_vel_w)[:, body] + torch.cross(
                tensor(d.body_ang_vel_w)[:, body], local_bottom - com_pos[:, body], dim=-1
            )
            slip.append(velocity_at[:, :2].norm(dim=-1))
        result = dict(
            root=root,
            joint=joint,
            acceleration=acceleration,
            COM=COM,
            COMv=COMv,
            length=torch.stack(lengths, -1),
            swing=torch.stack(swings, -1),
            spring=spring,
            closure=closure,
            normal=normal,
            contact=contact,
            restricted=restricted,
            tilt=tilt,
            omega=omega,
            clearance=torch.stack(clearance, -1),
            slip=torch.stack(slip, -1),
        )
        self._cached = (key, result)
        return result

    def close(self):
        self.sim.stop()
        self.sim.clear_instance()
