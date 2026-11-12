"""Dense reward/course adapter around the existing C/V4 MuJoCo environment.

No new integrator or Actor encoder. Extra truth is restricted to scoring and
Critic. This is CPU preparation, not a registered/accepted Isaac backend.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tensordict import TensorDict

from native_stand.simulation.jump_direct_env import EngineeringPlant
from native_stand.tasks.jump.controller_critic import NormalizedCritic, extend_fields
from native_stand.tasks.jump.dense_reward import (
    Packet,
)

ENV_VERSION = "jump_direct_dense_reward_course_cpu_v1"


class ScoringPlant(EngineeringPlant):
    """Read solved force before forward clears it; collect actual 1ms packets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scoring_force_N = np.zeros(2)
        self.scoring_slip_m_s = np.zeros(2)
        self.scoring_force_valid = False
        self.reward_packets = []

    def capture_solved_forces(self, model, data):
        super().capture_solved_forces(model, data)
        force, slip = np.zeros(2), np.zeros(2)
        for index, contact in enumerate(data.contact):
            if self.ground not in (contact.geom1, contact.geom2) or contact.efc_address < 0:
                continue
            other = contact.geom2 if contact.geom1 == self.ground else contact.geom1
            body = int(model.geom_bodyid[other])
            if body not in self.wheel_bodies:
                continue
            side = self.wheel_bodies.index(body)
            wrench = np.zeros(6)
            # Contact index belongs to the still-live solved contact list.
            mujoco.mj_contactForce(model, data, index, wrench)
            force[side] += max(float(wrench[0]), 0.0)
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, contact.pos, body)
            velocity = jac @ data.qvel
            normal = np.asarray(contact.frame).reshape(3, 3)[0]
            tangent = velocity - normal * (normal @ velocity)
            slip[side] = max(slip[side], float(np.linalg.norm(tangent)))
        self.scoring_force_N, self.scoring_slip_m_s = force, slip
        self.scoring_force_valid = True

    def scoring_packet(self, dt_s, *, fault=False, packet_type=Packet):
        readback = self.readback()
        shape = self.geometry.measure(
            {
                name: float(self.data.qpos[self.model.jnt_qposadr[j]])
                for name, j in self.refs.items()
            }
        )
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.root, velocity, 0
        )
        upright_rotation = (
            self.data.xmat[self.root].reshape(3, 3)
            @ Rotation.from_euler("x", -np.pi / 2).as_matrix()
        )
        state = self.execution.c.states[0]
        return packet_type(
            dt_s=dt_s,
            phase=int(state.phase),
            owner=int(state.owner),
            COM_height_m=readback["COM_height_m"],
            COM_velocity_m_s=tuple(self.COM_velocity()),
            root_height_m=readback["root_height_m"],
            tilt_rad=tuple(readback["tilt_rad"]),
            omega_rad_s=tuple(upright_rotation.T @ velocity[:3]),
            leg_length_m=tuple(shape.length),
            leg_swing_rad=tuple(shape.swing),
            spring_length_m=tuple(shape.spring),
            contact=tuple(readback["wheel_contact"]),
            tire_clearance_m=tuple(readback["clearance_m"]),
            wheel_normal_force_N=tuple(self.scoring_force_N),
            contact_slip_m_s=tuple(self.scoring_slip_m_s),
            actual_motor_Nm=tuple(self.data.qfrc_actuator[self.vids]),
            actual_dq_rad_s=tuple(readback["actual_dq"]),
            adopted_target=tuple(readback["adopted"]),
            force_valid=self.scoring_force_valid,
            motion_confirmed=bool(state.success and state.phase == 7),
            fault=bool(fault or state.fault or self.solved_restricted),
            external_takeoff_assist=False,
        )

    def integrate_control(self):
        before = self.physics_steps
        try:
            return super().integrate_control()
        finally:
            if self.physics_steps > before:
                self.reward_packets.append(self.scoring_packet(0.001))


class DenseCritic(NormalizedCritic):
    def contract(self):
        return {
            **super().contract(),
            "version": ENV_VERSION + "_critic",
            "scope": "MuJoCo CPU state plus complete progress/event/goal memory; not Isaac acceptance",
        }


class DenseJumpEnv:
    num_envs = 1
    device = "cpu"

    @property
    def plant(self):
        return self.base.plant

    def observations(self, *, schema_class=DenseCritic):
        if self._obs is not None:
            return self._obs
        obs = self.base.observations()
        values, fields = extend_fields({}, [], self.reward.critic_fields())
        self._reward_tail_schema = NormalizedCritic.reuse(
            fields, getattr(self, "_reward_tail_schema", None)
        )
        tail = self._reward_tail_schema.encode(values)
        schema = schema_class.reuse(
            list(self.base.schema.fields) + fields, getattr(self, "_dense_schema", self.schema)
        )
        self.schema = self._dense_schema = schema
        critic = np.concatenate((obs["student_critic"][0].numpy(), tail))[None]
        self._obs = TensorDict(
            dict(
                student_actor=obs["student_actor"].clone(),
                student_critic=torch.tensor(critic),
                recovery_actor_valid=obs["recovery_actor_valid"].clone(),
            ),
            batch_size=[1],
        )
        return self._obs

    def close(self):
        self.base.close()
