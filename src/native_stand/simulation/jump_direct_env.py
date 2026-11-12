"""Real CPU MuJoCo/C/V4 engineering environment. No optimizer or new physics loop.

Support uses an explicit simulation-only inverse-load hypothesis; production
support/model and the Isaac backend are not certified by this environment.
"""

from __future__ import annotations

import copy
import ctypes as C
from functools import partial

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from native_stand.simulation.jump_probe import JumpProbePlant
from native_stand.simulation.jump_shared_c import array, declared_input
from native_stand.simulation.jump_support_model import estimate
from native_stand.students.mujoco_solver import apply as solver_profile
from native_stand.tasks.jump.geometry import LEGS, ClosedChain
from native_stand.tasks.jump.pretraining import TaskMemory


class EngineeringPlant(JumpProbePlant):
    def __init__(
        self,
        description,
        springs,
        mechanics,
        V4,
        response_class,
        *,
        library,
        library_sha,
        charge_step,
        solver,
        execution_factory,
    ):
        self.geometry = ClosedChain(description, springs, stop_margin_m=0.0)
        shape = self.geometry.solve(0.28)
        super().__init__(
            description,
            springs,
            mechanics,
            V4,
            response_class,
            shape,
            charge_step=charge_step,
            execution_factory=partial(execution_factory, library, library_sha=library_sha),
        )
        self._reset_shapes = {0.28: shape}
        self._measurement_epoch = 0
        self._parameter_epoch = 0
        self.root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        self.ground = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.wheel_bodies = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, s + "_wheel")
            for s in ("left", "right")
        ]
        self.mass = float(self.model.body_mass.sum())
        self.solver = solver_profile(
            self.model, dict(ground=self.ground, wheels=self.wheel_bodies), solver
        )
        self.solved_ground_Fz_N = 0.0
        self.solved_ground_contact = np.zeros(2, dtype=bool)
        self.solved_restricted = False
        self.support_result = None
        # SIM model ID: real C/Actor plumbing, not a deployable trained model.
        cfg = self.execution.c.config
        cfg.enabled = cfg.model_available = 1
        cfg.model_id = 0x53494D33
        cfg.model_input_id = self.execution.c.INPUT_CONTRACT
        cfg.model_output_id = self.execution.c.OUTPUT_CONTRACT

    def invalidate_measurements(self, *, parameters_changed=False):
        """Reset/restore/scene mutation boundary; never a phase transition."""
        self._measurement_epoch += 1
        self._measured_shape = self._kinematics = None
        if parameters_changed:
            self._parameter_epoch += 1

    def measured_shape(self):
        token = (self._measurement_epoch, self.physics_steps, self.execution.tick)
        if getattr(self, "_shape_token", None) != token or self._measured_shape is None:
            self._measured_shape = self.geometry.measure(
                {n: float(self.data.qpos[self.model.jnt_qposadr[j]]) for n, j in self.refs.items()}
            )
            self._shape_token = token
        return self._measured_shape

    def kinematics(self):
        token = (self._measurement_epoch, self.physics_steps, self.execution.tick)
        if getattr(self, "_kinematics_token", None) != token or self._kinematics is None:
            m, d = self.model, self.data
            rotation = d.xmat[self.root].reshape(3, 3)
            upright = rotation @ Rotation.from_euler("x", -np.pi / 2).as_matrix()
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.root, velocity, 0)
            self._kinematics = dict(
                COM=self.COM(),
                COM_velocity=self.COM_velocity(),
                tilt=Rotation.from_matrix(upright).as_euler("xyz")[:2],
                omega=upright.T @ velocity[:3],
                gyro=rotation.T @ velocity[:3],
            )
            self._kinematics_token = token
        return self._kinematics

    def tire_clearance(self):
        m, d = self.model, self.data
        clear = []
        for body in self.wheel_bodies:
            bottoms = []
            for g in np.flatnonzero(m.geom_bodyid == body):
                axis = abs(d.geom_xmat[g].reshape(3, 3)[2, 2])
                size = m.geom_size[g]
                extent = size[1] * axis + size[0] * np.sqrt(max(0.0, 1 - axis * axis))
                bottoms.append(d.geom_xpos[g, 2] - extent)
            clear.append(min(bottoms))
        return np.asarray(clear)

    def capture_solved_forces(self, m, d):
        self.solved_ground_Fz_N = float(d.qfrc_constraint[2])
        contact = np.zeros(2, dtype=bool)
        restricted = False
        for c in d.contact:
            if self.ground not in (c.geom1, c.geom2) or c.efc_address < 0:
                continue
            other = c.geom2 if c.geom1 == self.ground else c.geom1
            body = int(m.geom_bodyid[other])
            if body in self.wheel_bodies:
                contact[self.wheel_bodies.index(body)] = True
            else:
                restricted = True
        self.solved_ground_contact = contact
        self.solved_restricted = restricted

    def COM(self):
        return (self.model.body_mass[:, None] * self.data.xipos).sum(0) / self.mass

    def COM_velocity(self):
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, jac, self.root)
        return jac @ self.data.qvel

    def feedback(self):
        m, d, e = self.model, self.data, self.execution
        self.shape = self.measured_shape()
        q, dq = d.qpos[self.qids], d.qvel[self.vids]
        rotation = d.xmat[self.root].reshape(3, 3)
        spread = (47.1033888 + 47.1080837) * np.pi / 180 + np.array([q[0] + q[1], -q[3] + q[4]])
        i = declared_input(
            now=e.tick,
            gyro=[0] * 3,
            gravity=rotation.T @ [0.0, 0.0, -1.0],
            q=q[list(LEGS)],
            dq=dq,
            command=[0.0, 0.0, 0.34],
            adopted=array(e.c.states[0].adopted),
            adopted_sequence=e.c.states[0].adopted_sequence,
            sensor_tick=e.tick,
            imu_tick=e.tick,
            feedback_ticks=[e.tick] * 6,
            accel=[0] * 3,
            driver_tau=[0] * 4,
            internal_tau=[0] * 4,
            support_J=[0] * 8,
            support_tick=e.tick,
            support_valid=0,
            leg_length=self.shape.length[::-1],
            leg_spread=spread[::-1],
            motion_entry=False,
            upstream_owner=2,
        )
        m, d = self.model, self.data
        rotation = d.xmat[self.root].reshape(3, 3)
        i.gyro[:] = self.kinematics()["gyro"]
        acceleration = np.zeros(6)
        mujoco.mj_rnePostConstraint(m, d)
        mujoco.mj_objectAcceleration(m, d, mujoco.mjtObj.mjOBJ_BODY, self.root, acceleration, 0)
        i.accel[:] = rotation.T @ acceleration[3:]
        self.support_result = estimate(self)
        for key in ["internal_tau", "driver_tau", "support_J"]:
            getattr(i, key)[:] = self.support_result[key]
        i.support_model_valid = int(self.support_result["valid"])
        i.support_mask = 3 if self.support_result["valid"] else 0
        i.motion_entry = int(
            i.gravity[1] < -0.98
            and max(abs(i.dq[2]), abs(i.dq[5])) < 2
            and sum(abs(x) for x in i.gyro) < 0.3
        )
        i.motion_success = 0  # no fabricated ten-second Motion confirmation
        i.upstream_owner = 1 if self.execution.c.states[0].phase in (0, 6, 7) else 2
        i.key_down = int(self.request_jump and self.execution.tick - self.episode_start_tick < 20)
        return i

    def integrate_control(self):
        s = self.execution.c.states[0]
        request = np.asarray(
            s.protected_target if s.owner == 2 else self.execution.c.inputs[0].upstream_target,
            float,
        )
        return self.integrate(request)

    def readback(self):
        d = self.data
        shape = self.measured_shape()
        kinematics = self.kinematics()
        s = self.execution.c.states[0]
        return dict(
            time_s=float(d.time),
            phase=int(s.phase),
            owner=int(s.owner),
            fault=int(s.fault),
            COM_height_m=float(kinematics["COM"][2]),
            COM_vz_m_s=float(kinematics["COM_velocity"][2]),
            wheel_contact=self.solved_ground_contact.copy(),
            clearance_m=self.tire_clearance(),
            restricted_ground_contact=self.solved_restricted,
            spring_m=shape.spring.copy(),
            closure_m=shape.closure_m,
            tilt_rad=kinematics["tilt"].copy(),
            root_height_m=float(d.qpos[2]),
            motor_Nm=self.actual_motor.copy(),
            NET_Nm=self.actual_NET.copy(),
            ground_Fz_N=self.solved_ground_Fz_N,
            actual_q=d.qpos[self.qids].copy(),
            support_estimate_N=array(s.support_N),
            support_valid=array(s.contact_valid),
            actual_dq=d.qvel[self.vids].copy(),
            adopted=array(s.adopted),
            requested=array(s.requested),
            protected_target=array(s.protected_target),
            raw=array(s.mean),
        )


class DirectEngineeringEnv:
    num_envs = 1
    device = "cpu"

    def runtime_state(self):
        p = self.plant
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(p.model, kind))
        mujoco.mj_getState(p.model, p.data, state, kind)
        return copy.deepcopy(
            dict(
                version="jump_direct_full_MuJoCo_engineering_v2",
                interface_digest=self.interface_digest,
                integration=state,
                execution=p.execution.state_dict(),
                physics_steps=p.physics_steps,
                actual_motor=p.actual_motor,
                actual_NET=p.actual_NET,
                spring_power=p.spring_power_W,
                solved_Fz=p.solved_ground_Fz_N,
                solved_contact=p.solved_ground_contact,
                solved_restricted=p.solved_restricted,
                spec=self.spec,
                memory=self.memory.state_dict(),
                done=self.done,
                prepared_tick=self.prepared_tick,
                cached_observations=self._obs,
                total_episodes=self.total_episodes,
                observation_token=bytes(p.execution.c.last_snapshot).hex()
                if p.execution.c.last_snapshot is not None
                else None,
                episode_start_tick=p.episode_start_tick,
                request_jump=p.request_jump,
            )
        )

    def restore_runtime_state(self, value):
        if (
            value["version"] != "jump_direct_full_MuJoCo_engineering_v2"
            or value["interface_digest"] != self.interface_digest
        ):
            raise ValueError("runtime full-state identity")
        p = self.plant
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        p.execution.load_state_dict(value["execution"])
        token = value["observation_token"]
        if token is not None:
            state_type = p.execution.c.StateType * p.execution.c.n
            raw = bytes.fromhex(token)
            if len(raw) != C.sizeof(state_type):
                raise ValueError("observation token ABI size")
            p.execution.c.last_snapshot = state_type.from_buffer_copy(raw)
        mujoco.mj_setState(p.model, p.data, value["integration"], kind)
        mujoco.mj_forward(p.model, p.data)
        mujoco.mj_setState(p.model, p.data, value["integration"], kind)
        p.physics_steps = value["physics_steps"]
        p.actual_motor = value["actual_motor"].copy()
        p.actual_NET = value["actual_NET"].copy()
        p.spring_power_W = value["spring_power"]
        p.solved_ground_Fz_N = value["solved_Fz"]
        p.solved_ground_contact = value["solved_contact"].copy()
        p.solved_restricted = value["solved_restricted"]
        p.episode_start_tick = value["episode_start_tick"]
        p.request_jump = value["request_jump"]
        self.spec = copy.deepcopy(value["spec"])
        self.memory = TaskMemory.restore(value["memory"])
        self.done = value["done"]
        self.prepared_tick = value["prepared_tick"]
        self._obs = copy.deepcopy(value["cached_observations"])
        self.total_episodes = value["total_episodes"]
        p.invalidate_measurements()

    def close(self):
        pass
