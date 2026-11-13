"""GPU inverse feedback from actual PhysX state, without physical integration.

MuJoCo Warp is used only for kinematics/mass/bias/passive evaluation. PhysX
remains the sole physical integrator. Contact truth never enters support.
"""

import copy

import mujoco
import mujoco_warp as mw
import numpy as np
import torch
import warp as wp

from native_stand.simulation.jump_isaac_bridge import InverseFeedback
from native_stand.tasks.jump.geometry import CHANNELS, LEGS


class DeviceInverse:
    version = "jump_PhysX_Warp_inverse_float32_v1"

    def __init__(self, reference: InverseFeedback, n, device):
        self.reference, self.n, self.device = reference, n, device
        self.initial = reference.initial
        self.model = copy.copy(reference.model)
        # Scratch inverse model only; the actual USD/PhysX model is untouched.
        self.model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        self.wm = mw.put_model(self.model)
        self.wd = mw.put_data(self.model, reference.data, nworld=n, nconmax=1, njmax=128)
        self.qpos = wp.to_torch(self.wd.qpos)
        self.qvel = wp.to_torch(self.wd.qvel)
        self.qacc = torch.zeros(n, self.model.nv, device=device)
        w = reference.support_workspace
        self.active = torch.as_tensor(w.active, device=device)
        self.passive = torch.as_tensor(w.passive, device=device)
        self.vids = torch.as_tensor(reference.vids, device=device)
        self.dofs = torch.as_tensor(reference.dofs, device=device)
        self.qids = torch.as_tensor(reference.qids, device=device)
        self.coulomb = torch.as_tensor(reference.coulomb, device=device)
        self.joint_channel = [reference.joint_names.index(k) for k in CHANNELS]
        self.jacs = [
            wp.empty((n, 3, self.model.nv), dtype=wp.float32, device=device) for _ in range(10)
        ]
        self.points = [torch.empty(n, 3, device=device) for _ in range(10)]
        self.bodies = []
        self.site_ids = []
        for a, b in w.sites:
            self.site_ids.extend((a, b))
        for site in self.site_ids:
            self.bodies.append(
                wp.array(np.full(n, self.model.site_bodyid[site], dtype=np.int32), device=device)
            )
        for body in reference.wheel_bodies:
            self.bodies.append(wp.array(np.full(n, body, dtype=np.int32), device=device))
        self.geom_ids = w.geoms
        self.tangent = torch.zeros(n, self.model.nv, 4, dtype=torch.float64, device=device)
        self.tangent[:, self.active, torch.arange(4, device=device)] = 1
        self.external = torch.zeros(n, self.model.nv, dtype=torch.float64, device=device)
        reference.geometry.leg_lengths(reference.initial.q)
        self.leg_chain = [
            (
                joint,
                reference.joint_names.index(joint["name"]),
                torch.as_tensor(frame0, device=device),
                torch.as_tensor(frame1, device=device),
            )
            for joint, frame0, frame1 in reference.geometry._leg_chain
        ]
        self.hips = torch.as_tensor(np.asarray(reference.geometry._leg_hips), device=device)
        # Initialization uploads may use Warp's allocation stream. Runtime
        # work below shares the current Torch stream, with no per-step global
        # synchronization or races between qpos writes and inverse kernels.
        wp.synchronize_device(device)

    def leg_lengths(self, joint):
        poses = {
            "robot": torch.eye(4, dtype=torch.float64, device=self.device).expand(self.n, -1, -1)
        }
        for record, index, frame0, frame1 in self.leg_chain:
            q = joint[:, 0, index].double()
            cosine, sine = torch.cos(q), torch.sin(q)
            rotation = torch.eye(4, dtype=torch.float64, device=self.device).repeat(self.n, 1, 1)
            axis = record["axis"]
            a, b = (axis + 1) % 3, (axis + 2) % 3
            rotation[:, a, a] = cosine
            rotation[:, b, b] = cosine
            rotation[:, a, b] = -sine
            rotation[:, b, a] = sine
            poses[record["child"]] = poses[record["parent"]] @ frame0 @ rotation @ frame1
        vectors = (
            torch.stack((poses["left_wheel"][:, :3, 3], poses["right_wheel"][:, :3, 3]), 1)
            - self.hips
        )
        return torch.linalg.vector_norm(vectors[:, :, 1:3], dim=-1)

    def evaluate(self, measured, motor, NET):
        def quat_apply(q, vector):
            xyz = q[:, 1:]
            uv = torch.cross(xyz, vector, dim=-1)
            return vector + 2 * (q[:, :1] * uv + torch.cross(xyz, uv, dim=-1))

        def quat_apply_inverse(q, vector):
            conjugate = torch.cat((q[:, :1], -q[:, 1:]), -1)
            return quat_apply(conjugate, vector)

        # Isaac uses xyzw here, while IsaacLab quaternion utility uses wxyz.
        root = measured["root"]
        joint = measured["joint"]
        acc = measured["acceleration"]
        quat = root[:, [6, 3, 4, 5]]
        self.qpos[:, :3] = root[:, :3]
        self.qpos[:, 3:7] = quat
        self.qpos[:, self.qids] = joint[:, 0]
        self.qvel[:, :3] = root[:, 7:10]
        gyro = quat_apply_inverse(quat, root[:, 10:13])
        self.qvel[:, 3:6] = gyro
        self.qvel[:, self.dofs] = joint[:, 1]
        self.qacc[:, :3] = acc[:, :3]
        self.qacc[:, 3:6] = quat_apply_inverse(quat, acc[:, 3:])
        self.qacc[:, self.dofs] = joint[:, 2]
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            mw.kinematics(self.wm, self.wd)
            mw.com_pos(self.wm, self.wd)
            mw.crb(self.wm, self.wd)
            mw.com_vel(self.wm, self.wd)
            mw.passive(self.wm, self.wd)
            mw.rne(self.wm, self.wd)
            site = wp.to_torch(self.wd.site_xpos)
            geoms = wp.to_torch(self.wd.geom_xpos)
            for k in range(10):
                if k < 8:
                    self.points[k].copy_(site[:, self.site_ids[k]])
                else:
                    geom = self.geom_ids[k - 8]
                    self.points[k].copy_(geoms[:, geom])
                    self.points[k][:, 2] -= self.model.geom_size[geom, 0]
                mw.jac(
                    self.wm,
                    self.wd,
                    self.jacs[k],
                    None,
                    wp.from_torch(self.points[k], dtype=wp.vec3),
                    self.bodies[k],
                )
        j = [wp.to_torch(x).double() for x in self.jacs]
        closure = torch.cat([j[k] - j[k + 1] for k in range(0, 8, 2)], 1)
        self.tangent[:, self.passive] = (
            -torch.linalg.pinv(closure[:, :, self.passive], rtol=1e-9) @ closure[:, :, self.active]
        )
        mass = wp.to_torch(self.wd.qM)[:, : self.model.nv, : self.model.nv].double()
        if mass.shape[1:] != (self.model.nv, self.model.nv):
            raise ValueError(f"Warp inverse requires dense mass matrix: {mass.shape}")
        load = (
            (mass @ self.qacc.double().unsqueeze(-1)).squeeze(-1)
            + wp.to_torch(self.wd.qfrc_bias).double()
            - wp.to_torch(self.wd.qfrc_passive).double()
        )
        self.external[:, self.vids] = NET.double() - self.coulomb * torch.sign(
            joint[:, 1, self.joint_channel]
        )
        internal = (self.tangent.transpose(1, 2) @ (load - self.external).unsqueeze(-1)).squeeze(-1)
        # Same model forward direction projected horizontally, not contact data.

        forward = quat_apply(quat, root.new_tensor([0.0, 0.0, -1.0]).expand(self.n, -1))
        forward[:, 2] = 0
        forward = forward / forward.norm(dim=-1, keepdim=True)
        feet = []
        for side in range(2):
            jac = j[8 + side]
            foot = (
                torch.stack((jac[:, 2], (forward.double().unsqueeze(1) @ jac).squeeze(1)), 1)
                @ self.tangent
            )
            feet.append(foot[:, :, 2 * side : 2 * side + 2].reshape(self.n, 4))
        gravity = quat_apply_inverse(quat, root.new_tensor([0.0, 0.0, -1.0]).expand(self.n, -1))
        q = joint[:, 0, self.joint_channel]
        spread = (47.1033888 + 47.1080837) * np.pi / 180 + torch.stack(
            (q[:, 0] + q[:, 1], -q[:, 3] + q[:, 4]), -1
        ).double()
        accel = quat_apply_inverse(quat, acc[:, :3] + acc.new_tensor([0.0, 0.0, 9.81]))
        output = dict(
            gyro=gyro,
            gravity=gravity,
            q=q[:, [0, 1, 3, 4]],
            dq=joint[:, 1, self.joint_channel],
            accel=accel,
            driver_tau=motor[:, list(LEGS)],
            internal_tau=internal,
            support_J=torch.cat(feet, -1),
            leg_length=self.leg_lengths(joint),
            leg_spread=spread,
            motion_entry=(gravity[:, 1] < -0.98)
            & (joint[:, 1, self.joint_channel][:, [2, 5]].abs().amax(-1) < 2)
            & (gyro.abs().sum(-1) < 0.3),
        )

        packed = torch.cat([value.reshape(self.n, -1).double() for value in output.values()], -1)
        if not bool(torch.isfinite(packed).all()):
            raise RuntimeError("Nonfinite GPU inverse feedback")
        return output
