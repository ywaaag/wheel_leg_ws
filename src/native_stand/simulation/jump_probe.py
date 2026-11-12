"""Prepared real-integration primitive for an independently authorized Jump block.

Construction/CPU readback never integrates. This is an environment primitive,
not a trainer or background launcher. Public owner/worker must supply the task
controller, causal estimator, accounting, stop and checkpoint boundaries.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import torch

from native_stand.simulation.jump_mechanics import model_from_description, static_data
from native_stand.simulation.recovery_isaac.execution import Execution
from native_stand.tasks.jump.geometry import CHANNELS


def mechanical_contract(source):
    c = copy.deepcopy(source)
    for side in ["L", "R"]:
        row = c["sides"][side]
        for name in c["parameter_names"]:
            row["parameters"][name] = 1.0 if name == "leg_mass_inertia_scale" else 0.0
        for name in ["transport_ms", "q_age_ms", "v_age_ms", "velocity_tau_ms"]:
            row["effective_observer"][name] = 0
    c["version"] = "jump_explicit_mechanics_ideal_leg_response_no_NET_v1"
    return c


class JumpProbePlant:
    def __init__(
        self,
        description,
        springs,
        mechanics,
        V4,
        response_class,
        entry_shape,
        *,
        plant="V4",
        motor_cap_factor=1.0,
        charge_step=None,
    ):
        if plant not in ("V4", "mechanical350") or motor_cap_factor not in (1.0, 0.8):
            raise ValueError("Explicit finite candidate")
        self.description = description
        self.springs = springs
        self.plant = plant
        self.charge_step = charge_step
        self.model, self.refs = model_from_description(
            description,
            V4_contract=V4 if plant == "V4" else None,
            mechanics=mechanics,
            physical_probe=True,
        )
        self.data = static_data(self.model, self.refs, entry_shape)
        self.qids = np.array([self.model.jnt_qposadr[self.refs[n]] for n in CHANNELS])
        self.vids = np.array([self.model.jnt_dofadr[self.refs[n]] for n in CHANNELS])
        self.aids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_motor")
                for n in CHANNELS
            ]
        )
        self.execution = Execution(
            V4 if plant == "V4" else mechanical_contract(V4), response_class, 1, dtype=torch.float64
        )
        self.execution.reset(
            self.execution.ids,
            torch.tensor(self.data.qpos[self.qids][None]),
            torch.tensor(self.data.qvel[self.vids][None]),
        )
        self.caps = np.array([40.0, 40.0, 2.44694786377709, 40.0, 40.0, 2.44694786377709])
        self.caps[[0, 1, 3, 4]] *= motor_cap_factor
        self.physics_steps = 0
        self.actual_motor = np.zeros(6)
        self.actual_NET = np.zeros(6)
        self.spring_power_W = 0.0

    def integrate(self, protected_target):
        """Call ONLY within a separately authorized and accounted worker block."""
        if self.charge_step is None:
            raise RuntimeError("Physical integration requires the owner accounting callback")
        target = np.asarray(protected_target, float)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Physical target6")
        d = self.data
        m = self.model
        mujoco.mj_forward(m, d)  # align FK/anchor velocities with current q/dq
        q = torch.tensor(d.qpos[self.qids][None])
        dq = torch.tensor(d.qvel[self.vids][None])
        self.execution.submit(torch.tensor(target[None]))
        motor, net = self.execution.step(q, dq)
        self.actual_motor = np.clip(motor[0].numpy(), -self.caps, self.caps)
        self.actual_NET = net[0].numpy().copy()
        d.ctrl[self.aids] = self.actual_motor
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[self.vids] = self.actual_NET
        self.spring_power_W = 0.0
        for spring in self.springs:
            points = []
            bodyids = []
            jacobians = []
            for anchor in spring["anchors"]:
                bid = mujoco.mj_name2id(
                    m, mujoco.mjtObj.mjOBJ_BODY, anchor["body"].rsplit("/", 1)[-1]
                )
                bodyids.append(bid)
                point = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ np.asarray(anchor["local_m"])
                points.append(point)
                J = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, J, None, point, bid)
                jacobians.append(J)
            delta = points[1] - points[0]
            length = np.linalg.norm(delta)
            if not 0.155 - 1e-5 <= length <= 0.215 + 1e-5:
                raise ValueError("Actual spring stroke violation; no state projection")
            if self.plant != "mechanical350":
                continue  # V4 NET replaces this load; geometry remains bounded
            force = 350.0 * delta / length
            for sign, point, bid in zip([-1, 1], points, bodyids, strict=False):
                mujoco.mj_applyFT(m, d, sign * force, np.zeros(3), point, bid, d.qfrc_applied)
            self.spring_power_W += force @ (jacobians[1] - jacobians[0]) @ d.qvel
        # Integrate genuine forces exactly once. No q/dq assignment occurs after
        # initialization, no root impulse, restitution boost or trajectory forcing.
        mujoco.mj_step(m, d)
        self.physics_steps += 1
        self.charge_step(1)
        self.capture_solved_forces(m, d)
        if not np.allclose(d.qfrc_actuator[self.vids], self.actual_motor, atol=1e-8, rtol=0):
            raise RuntimeError("Actual actuator route/cap mismatch")
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise FloatingPointError("Actual integration nonfinite")
        mujoco.mj_forward(m, d)
        for spring in self.springs:
            ends = []
            for anchor in spring["anchors"]:
                bid = mujoco.mj_name2id(
                    m, mujoco.mjtObj.mjOBJ_BODY, anchor["body"].rsplit("/", 1)[-1]
                )
                ends.append(d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ np.asarray(anchor["local_m"]))
            if not 0.155 - 1e-5 <= np.linalg.norm(ends[1] - ends[0]) <= 0.215 + 1e-5:
                raise ValueError("Integrated spring stroke violation; charged and retained")
        for index in range(4):
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"loop{index}a")
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"loop{index}b")
            if np.linalg.norm(d.site_xpos[a] - d.site_xpos[b]) > 0.005:
                raise ValueError("Integrated closure failure; charged and retained")
        return dict(
            physics_steps=self.physics_steps,
            motor=self.actual_motor.copy(),
            NET=self.actual_NET.copy(),
            spring_power_W=self.spring_power_W,
        )

    def capture_solved_forces(self, model, data):
        pass

    def snapshot(self):
        return dict(
            physics_steps=self.physics_steps,
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            time_s=self.data.time,
            qacc_warmstart=self.data.qacc_warmstart.copy(),
            execution=self.execution.state_dict(),
            actual_motor=self.actual_motor.copy(),
            actual_NET=self.actual_NET.copy(),
            note="physical and execution snapshot; solver hidden state/bitwise continuation not certified",
        )
