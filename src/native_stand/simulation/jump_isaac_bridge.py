"""CPU inverse-model feedback bridge for actual PhysX state, never integration.

This is an explicit simulation sensor hypothesis. Native contacts are reserved
for reward/validation and never become the C support observation.
"""

from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from native_stand.simulation.jump_mechanics import model_from_description, static_data
from native_stand.simulation.jump_shared_c import declared_input
from native_stand.simulation.jump_support_model import estimate
from native_stand.tasks.jump.geometry import CHANNELS, ClosedChain


class InverseFeedback:
    def __init__(self, description, springs, mechanics, V4):
        self.geometry = ClosedChain(description, springs, stop_margin_m=0.0)
        self.initial = self.geometry.solve(0.28)
        self.model, self.refs = model_from_description(
            description, V4_contract=V4, mechanics=mechanics
        )
        self.data = static_data(self.model, self.refs, self.initial)
        self.root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        self.wheel_bodies = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, side + "_wheel")
            for side in ("left", "right")
        ]
        self.vids = np.array([self.model.jnt_dofadr[self.refs[n]] for n in CHANNELS])
        self.actual_NET = np.zeros(6)
        self.coulomb = np.array([mechanics["friction_loss_Nm"][n] for n in CHANNELS])
        self.joint_names = mechanics["joint_names"]
        self.qids = np.array([self.model.jnt_qposadr[self.refs[n]] for n in self.joint_names])
        self.dofs = np.array([self.model.jnt_dofadr[self.refs[n]] for n in self.joint_names])

    def input(self, *, tick, state, root, joint, acceleration, motor, NET, command, request):
        d = self.data
        d.qpos[:3] = root[:3]
        d.qpos[3:7] = root[[6, 3, 4, 5]]
        d.qpos[self.qids] = joint[0]
        d.qvel[:3] = root[7:10]
        rotation = Rotation.from_quat(root[3:7]).as_matrix()
        d.qvel[3:6] = rotation.T @ root[10:13]
        d.qvel[self.dofs] = joint[1]
        mujoco.mj_forward(self.model, d)
        # The forward call establishes M/bias/passive/FK, not a replacement
        # physical acceleration. Supply the measured PhysX finite differences.
        d.qacc[:3] = acceleration[:3]
        d.qacc[3:6] = rotation.T @ acceleration[3:]
        d.qacc[self.dofs] = joint[2]
        d.qfrc_actuator[:] = 0
        d.qfrc_actuator[self.vids] = motor
        self.actual_NET = NET - self.coulomb * np.sign(d.qvel[self.vids])
        support = estimate(self)
        shape = self.geometry.measure(dict(zip(self.joint_names, joint[0], strict=True)))
        q = d.qpos[[self.model.jnt_qposadr[self.refs[n]] for n in CHANNELS]]
        spread = (47.1033888 + 47.1080837) * np.pi / 180 + np.array([q[0] + q[1], -q[3] + q[4]])
        gyro = rotation.T @ root[10:13]
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        return declared_input(
            now=tick,
            gyro=gyro,
            gravity=gravity,
            q=q[[0, 1, 3, 4]],
            dq=d.qvel[self.vids],
            command=command,
            adopted=np.asarray(state.adopted),
            adopted_sequence=state.adopted_sequence,
            sensor_tick=tick,
            imu_tick=tick,
            feedback_ticks=[tick] * 6,
            accel=rotation.T @ (acceleration[:3] + np.array([0, 0, 9.81])),
            driver_tau=support["driver_tau"],
            internal_tau=support["internal_tau"],
            support_J=support["support_J"],
            support_tick=tick,
            support_valid=3 if support["valid"] else 0,
            leg_length=shape.length,
            leg_spread=spread,
            motion_entry=bool(
                gravity[1] < -0.98
                and max(abs(d.qvel[self.vids][[2, 5]])) < 2.0
                and sum(abs(gyro)) < 0.3
            ),
            upstream_owner=2,
            key=request,
        )


def critic_execution(execution):
    """Read all V4 rows once per dtype, then slice owned CPU tensor views.

    No event, packet, RNG or actuator state advances here. Call after observe
    or at a terminal snapshot, rather than caching across same-tick changes.
    """
    source = (execution.v4, execution.v4.leg)
    values = [{}, {}]
    groups = {}
    for owner, obj in enumerate(source):
        for name, value in vars(obj).items():
            if torch.is_tensor(value):
                groups.setdefault((value.device, value.dtype), []).append((owner, name, value))
    for entries in groups.values():
        packed = torch.cat([v.detach().reshape(-1) for _, _, v in entries]).cpu()
        offset = 0
        for owner, name, value in entries:
            end = offset + value.numel()
            values[owner][name] = packed[offset:end].view(value.shape)
            offset = end
    # Field traversal order defines Critic columns. Dtype grouping must never
    # reorder that original layout, even though the copies are batched.
    values = [
        {name: data[name] for name in vars(obj) if name in data}
        for obj, data in zip(source, values, strict=True)
    ]
    values[0].update(n=execution.v4.n, leg=SimpleNamespace(**values[1]))
    return SimpleNamespace(
        tick=execution.tick,
        c=execution.c,
        v4=SimpleNamespace(**values[0]),
        transfer_batches=len(groups),
    )


def execution_row(execution, index):
    """Views of one batch row, retaining every C/V4/response history field."""
    v = execution.v4
    values = {}
    for name, value in vars(v).items():
        if not torch.is_tensor(value):
            continue
        if name in {"ids", "columns", "sign"}:
            values[name] = value
        elif (
            value.ndim > 1
            and value.shape[1] == v.n
            and name
            in {
                "targets",
                "origins",
                "wheel_queue",
                "motors",
                "NETs",
                "packet",
                "qhistory",
                "dqhistory",
                "target_policy_history",
                "motor_policy_history",
            }
        ):
            values[name] = value[:, index : index + 1]
        elif name in {"theta", "omega"}:
            values[name] = value[2 * index : 2 * index + 2]
        else:
            values[name] = value[index : index + 1]
    response = {}
    for name, value in vars(v.leg).items():
        if torch.is_tensor(value):
            response[name] = (
                value[:, 2 * index : 2 * index + 2]
                if name in {"qhist", "vhist", "queue"}
                else value[2 * index : 2 * index + 2]
            )
    values["leg"] = SimpleNamespace(**response)
    return SimpleNamespace(
        tick=execution.tick,
        v4=SimpleNamespace(**values),
        c=SimpleNamespace(states=[execution.c.states[index]], inputs=[execution.c.inputs[index]]),
    )
