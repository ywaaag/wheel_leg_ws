"""Named reset-only physical setters. No controller, no implicit DR fallback."""

from __future__ import annotations

import torch

from .observations import invalidate_physical_profile


def tensor_view(v):
    if isinstance(v, torch.Tensor):
        return v.detach()
    if hasattr(v, "torch"):
        return v.torch.detach()
    import warp as wp

    if isinstance(v, wp.array):
        return wp.to_torch(v)
    raise TypeError("Unsupported physical tensor frontend " + type(v).__name__)


def value(v, device, ids=None):
    source = tensor_view(v)
    if ids is not None:
        source = source[ids.to(source.device, dtype=torch.long)]
    return source.to(device).clone()


class PhysicalParameters:
    def __init__(self, env, contract, mechanics):
        self.e = env
        r = env.robot
        self.device = env.device
        self.view = r.root_view
        self.mass0 = value(self.view.get_masses(), self.device)
        self.inertia0 = value(self.view.get_inertias(), self.device)
        self.com0 = value(self.view.get_coms(), self.device)
        self.body_ids = torch.arange(len(r.body_names), device=self.device, dtype=torch.int32)
        self.joint_ids = torch.arange(len(r.joint_names), device=self.device, dtype=torch.int32)
        if set(r.joint_names) != set(mechanics["joint_names"]):
            raise ValueError("Complete initial joint identity differs from exported legal scene")
        self.armature = (
            torch.tensor(
                [mechanics["armature"][name] for name in r.joint_names], device=self.device
            )
            .expand(env.num_envs, -1)
            .clone()
        )
        self.damping = (
            torch.tensor([mechanics["damping"][name] for name in r.joint_names], device=self.device)
            .expand(env.num_envs, -1)
            .clone()
        )
        self.coulomb = (
            torch.tensor(
                [mechanics["friction_loss_Nm"][name] for name in r.joint_names], device=self.device
            )
            .expand(env.num_envs, -1)
            .clone()
        )
        touched = set()
        for side in ["L", "R"]:
            row = contract["sides"][side]
            scale = row["parameters"]["leg_mass_inertia_scale"]
            for item in row["body_changes"]:
                i = r.body_names.index(item["name"])
                if i in touched:
                    raise ValueError("V4 body mass scaled twice")
                touched.add(i)
                self.mass0[:, i] *= scale
                self.inertia0[:, i] *= scale
        # PhysX articulation root is link index0; assert at live setup against
        # its actual root-link metadata, never silently choose another body.
        self.root_index = 0
        self.last_readback = {}
        r.write_joint_armature_to_sim_index(
            armature=self.armature, joint_ids=self.joint_ids, env_ids=None
        )
        sim_damping = self.damping.clone()
        sim_damping[:, env.ids] = 0
        r.write_joint_damping_to_sim_index(
            damping=sim_damping, joint_ids=self.joint_ids, env_ids=None
        )
        r.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=torch.zeros_like(self.armature),
            joint_dynamic_friction_coeff=torch.zeros_like(self.armature),
            joint_viscous_friction_coeff=torch.zeros_like(self.armature),
            joint_ids=self.joint_ids,
            env_ids=None,
        )
        # Coulomb torque is explicitly routed outside motor caps. It is a
        # kinetic-friction candidate, NOT MuJoCo's static friction solver.
        material = self.view.get_material_properties()
        self._material_torch = isinstance(material, torch.Tensor)
        self.material = value(material, "cpu")
        self.mass = self.mass0.clone()
        self.inertia = self.inertia0.clone()
        self.com = self.com0.clone()

    def _set_material(self, indices):
        # PhysX requires a full (envs, shapes, 3) tensor plus selected env IDs.
        if self._material_torch:
            self.view.set_material_properties(self.material, indices)
        else:
            import warp as wp

            self.view.set_material_properties(
                wp.from_torch(self.material.contiguous(), dtype=wp.float32),
                wp.from_torch(indices.contiguous(), dtype=wp.int32),
            )

    def reset(self, ids, draws, *, host_ids=None):
        e = self.e
        if len(ids) == 0:
            return
        invalidate_physical_profile(e, host_ids if host_ids is not None else ids.cpu().tolist())
        r = e.robot
        index = ids.to(torch.int32)
        rows = len(ids)
        timer = getattr(e, "reset_measurement", None)
        if timer is not None and timer.enabled:
            for parameter in ("mass", "COM", "inertia", "material"):
                timer.count("recovery." + parameter + ".getters")
                timer.count("recovery." + parameter + ".setters")
                timer.count("recovery." + parameter + ".pool_rows", e.num_envs)
                timer.count("recovery." + parameter + ".used_rows", rows)
        mass = self.mass0[ids].clone()
        inertia = self.inertia0[ids].clone()
        com = self.com0[ids].clone()
        root = self.root_index
        mass[:, root] *= mass.new_tensor([d["root_mass_scale"] for d in draws])
        inertia[:, root] *= inertia.new_tensor([d["root_inertia_scale"] for d in draws])[:, None]
        com[:, root, :3] += com.new_tensor([d["root_COM_m"].tolist() for d in draws])
        r.set_masses_index(masses=mass.contiguous(), body_ids=self.body_ids, env_ids=index)
        # PhysX set_coms rotates the existing inertia when updating the mass frame.
        # Write the declared final inertia AFTER COM, then verify both readbacks.
        r.set_coms_index(coms=com.contiguous(), body_ids=self.body_ids, env_ids=index)
        r.set_inertias_index(inertias=inertia.contiguous(), body_ids=self.body_ids, env_ids=index)
        # Average ground(.5)/shape combine: 2*mu-.5 on wheel material yields
        # the same explicitly requested effective friction as CPU uniform mu.
        host = ids.cpu().long() if host_ids is None else torch.tensor(host_ids, dtype=torch.long)
        materials = self.material
        mu = materials.new_tensor([d["friction"] for d in draws])
        shape_mu = 2 * mu - e.cfg.ground_dynamic_friction
        if bool((shape_mu < 0).any()):
            raise ValueError("Scene material cannot represent this DR effective friction")
        materials[host, :, :2] = shape_mu[:, None, None]
        materials[host, :, 2] = 0
        self._set_material(host.to(torch.int32))
        actual_material = value(self.view.get_material_properties(), "cpu", host)
        if not torch.allclose(actual_material, materials[host], atol=1e-6, rtol=0):
            raise RuntimeError("Actual material reset/readback differs")
        self.mass[ids] = mass
        self.inertia[ids] = inertia
        self.com[ids] = com
        actual = [
            value(self.view.get_masses(), self.device, ids),
            value(self.view.get_inertias(), self.device, ids),
            value(self.view.get_coms(), self.device, ids),
        ]
        for name, x, y in zip(
            ["mass", "inertia", "COM"], actual, [mass, inertia, com], strict=False
        ):
            # PhysX diagonalizes inertia and chooses its principal-axis COM
            # quaternion. Verify physical COM translation and full inertia,
            # not the non-unique principal-axis orientation gauge.
            if name == "COM":
                if not torch.isfinite(x).all() or not torch.allclose(
                    torch.linalg.vector_norm(x[..., 3:7], dim=-1),
                    torch.ones_like(x[..., 0]),
                    atol=2e-6,
                    rtol=0,
                ):
                    raise RuntimeError("Invalid actual COM frame")
                x = x[..., :3]
                y = y[..., :3]
            if not torch.allclose(x, y, atol=2e-6, rtol=1e-5):
                raise RuntimeError(
                    "Isaac reset DR readback " + name + " max=" + str(float((x - y).abs().max()))
                )
        self.com[ids] = actual[2]
        self.last_readback = dict(
            COM_orientation="PhysX principal-axis gauge; translation/full inertia verified",
            rows=rows,
            named_V4_mass_scaled_once=True,
            root_only_DR=True,
            mass_inertia_COM_verified=True,
            passive_model="active-joint viscous+Coulomb body wrench; passive-joint viscous in PhysX; static stiction not certified",
            joint_friction_set_zero=True,
            friction_backend_readback="PENDING_LIVE_ENGINE_CHECK",
        )

    def state_dict(self):
        return {
            k: getattr(self, k).clone()
            for k in ["mass", "inertia", "com", "material", "armature", "damping", "coulomb"]
        }

    def restore(self, state):
        for k, v in state.items():
            getattr(self, k).copy_(v.to(getattr(self, k).device))
        r = self.e.robot
        ids = torch.arange(self.e.num_envs, device=self.device, dtype=torch.int32)
        r.set_masses_index(masses=self.mass, body_ids=self.body_ids, env_ids=ids)
        r.set_coms_index(coms=self.com, body_ids=self.body_ids, env_ids=ids)
        r.set_inertias_index(inertias=self.inertia, body_ids=self.body_ids, env_ids=ids)
        self._set_material(ids.cpu())
        invalidate_physical_profile(self.e)
