"""Native PhysX material API adapter. No emulated external contact forces."""

from __future__ import annotations

import torch

from native_stand.tasks.motion.contact_randomization import sample, validate


def physics_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    import warp as wp

    if isinstance(value, wp.array):
        return wp.to_torch(value)
    raise TypeError("Unsupported native material tensor: " + type(value).__name__)


def write_native(view, name, tensors):
    original = view.get_material_properties()
    if isinstance(original, torch.Tensor):
        getattr(view, name)(*tensors)
        return
    import warp as wp

    if not isinstance(original, wp.array):
        raise TypeError("Unsupported native material frontend")
    converted = [
        wp.from_torch(
            x.contiguous(),
            dtype={torch.float32: wp.float32, torch.uint8: wp.uint8, torch.int32: wp.int32}[
                x.dtype
            ],
        )
        for x in tensors
    ]
    getattr(view, name)(*converted)


def check_scene_materials(env):
    """The tensor setter preserves force/acceleration-spring interpretation."""
    stage = env.sim.stage
    paths = (env.cfg.ground_material_path, "/World/recovery_dr_wheel_material")
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError("Declared contact material is missing: " + path)
        acceleration = prim.GetAttribute("physxMaterial:compliantContactAccelerationSpring").Get()
        if acceleration:
            raise RuntimeError(
                "Contact K/C contract requires force spring, not acceleration spring"
            )
        if (
            path == paths[0]
            and (prim.GetAttribute("physxMaterial:compliantContactStiffness").Get() or 0) != 0
        ):
            raise RuntimeError("Contact DR requires rigid ground; do not combine two soft surfaces")


class ContactMaterials:
    def __init__(self, dr, config, noise, tensor):
        self.dr, self.config, self.noise, self.tensor = dr, validate(config), noise, tensor
        self.view = dr.env.robot.root_view
        for name in ("get_compliant_material_properties", "set_compliant_material_properties"):
            if not callable(getattr(self.view, name, None)):
                raise RuntimeError("Native PhysX compliant-material API required: " + name)
        self.wheels = []
        for side in ("left", "right"):
            paths = [p for p in self.view.link_paths[0] if p.endswith("/" + side + "_wheel")]
            if len(paths) != 1 or "/env_0/" not in paths[0]:
                raise RuntimeError("Wheel collision view needs the actual named environment0 link")
            pattern = paths[0].replace("/env_0/", "/env_*/")
            view = dr.env.robot._physics_sim_view.create_rigid_body_view(pattern)
            if view.count != dr.n or view.max_shapes != 1:
                raise RuntimeError("Wheel collision view row/shape mapping differs")
            self.wheels.append(view)
        self.actual = torch.zeros(dr.n, 4, device=dr.device)
        self.last_readback = None

    def reset(self, ids):
        dr, config = self.dr, self.config
        expected = self.actual.new_zeros(len(ids), 4)
        active = (dr.profiles[ids] == 0) & (dr.amplitude[ids, 0] > 0)
        selected = ids[active]
        if len(selected):
            u = self.noise(dr.episodes[selected], selected, 5, 19031)
            expected[active] = sample(u, dr.amplitude[selected, 0], config)
        host = ids.cpu().long()
        desired = expected.cpu()
        soft_mask = (desired[:, :2] > 0).any(-1)
        soft, rigid = host[soft_mask], host[~soft_mask]
        for side, view in enumerate(self.wheels):
            material = self.tensor(view.get_material_properties()).cpu().clone()
            sd, _ = view.get_compliant_material_properties()
            sd = self.tensor(sd).cpu().clone()
            sd[host, 0, 0] = desired[:, side]
            sd[host, 0, 1] = desired[:, side + 2]
            data = torch.cat((material[..., :2], sd), -1).contiguous()
            # Native RigidBodyView uses [body_count,3] modes (articulation
            # setter uses an extra shape axis). Only wheel links are touched.
            modes = torch.zeros(dr.n, 3, dtype=torch.uint8)
            write_native(
                view, "set_compliant_material_properties", (data, modes, host.to(torch.int32))
            )
            if len(rigid):
                # Clear residual damping first, then restore ordinary bounce.
                write_native(
                    view, "set_material_properties", (material.contiguous(), rigid.to(torch.int32))
                )
            after = self.tensor(view.get_material_properties()).cpu()
            if not torch.allclose(after[host, :, :2], material[host, :, :2], atol=1e-6, rtol=0):
                raise RuntimeError("Compliance update changed wheel friction")
        # Cross-view readback from the original articulation proves the named
        # wheel setters reached the same shapes used by the existing DR.
        actual_sd, actual_modes = self.view.get_compliant_material_properties()
        actual_sd, actual_modes = self.tensor(actual_sd).cpu(), self.tensor(actual_modes).cpu()
        columns = torch.tensor(dr.columns, dtype=torch.long)
        actual = actual_sd[host[:, None], columns[None]]
        raw_stiffness = actual[..., 0]
        ordinary = self.tensor(self.view.get_material_properties()).cpu()[
            host[:, None], columns[None], 2
        ]
        # PhysX6 returns +inf stiffness for a rigid material. Preserve that raw
        # identity and encode disabled compliance as0 in the finite Critic.
        # Only positive-infinite K with ordinary restitution and zero C is
        # admissible; a missing/failed soft write still disagrees with desired.
        rigid_native = torch.isposinf(raw_stiffness) & (ordinary >= 0) & (actual[..., 1] == 0)
        stiffness = torch.where(rigid_native, 0.0, raw_stiffness)
        actual = torch.cat((stiffness, actual[..., 1]), -1)
        if not torch.allclose(actual, desired, rtol=2e-5, atol=1e-5):
            import json

            bad = torch.where(~torch.isclose(actual, desired, rtol=2e-5, atol=1e-5).all(-1))[0][:8]
            rows = host[bad]
            detail = dict(
                selected_envs=rows.tolist(),
                articulation_wheel_shape_columns=dr.columns,
                expected=desired[bad].tolist(),
                actual_articulation=actual[bad].tolist(),
                actual_wheel_views=[
                    self.tensor(v.get_compliant_material_properties()[0]).cpu()[rows].tolist()
                    for v in self.wheels
                ],
                wheel_view_paths=[
                    [v.prim_paths[int(i)] for i in rows] if hasattr(v, "prim_paths") else None
                    for v in self.wheels
                ],
            )
            raise RuntimeError(
                "Actual wheel contact stiffness/damping readback differs: "
                + json.dumps(detail)[:4000]
            )
        if len(soft) and bool((actual_modes[soft[:, None], columns[None]] != 0).any()):
            raise RuntimeError("Actual compliant material combine mode differs")
        self.actual[ids] = actual.to(self.actual)
        # Ordinary restitution is not active for the compliant normal force.
        dr.restitution[ids[soft_mask.to(ids.device)]] = 0
        self.last_readback = dict(
            rows=len(ids),
            compliant_rows=len(soft),
            actual_native_material_values_verified=True,
            contact_force_response_validated=False,
            stiffness_n_m=actual[:, :2].tolist(),
            damping_ns_m=actual[:, 2:].tolist(),
            nonwheel_unchanged=True,
            native_rigid_positive_infinite_stiffness_rows=int(rigid_native.any(-1).sum()),
            critic_zero_stiffness_means="rigid/disabled, not a zero-stiffness soft spring",
            write_scope="two named wheel RigidBodyViews; readback via original ArticulationView",
        )
        dr.last_readback["ordinary_restitution_verified_before_contact_override"] = True
        dr.last_readback["wheel_contact"] = self.last_readback


def dr_factory(base_class, config, noise):
    validate(config)

    class ContactDR(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            check_scene_materials(self.env)
            self.contact = ContactMaterials(self, config, noise, physics_tensor)

        def _physics(self, ids):
            super()._physics(ids)
            self.contact.reset(ids)

        def summary(self):
            result = super().summary()
            actual = self.contact.actual
            result["wheel_contact"] = dict(
                configuration=config,
                actual_stiffness_min_max=[float(actual[:, :2].min()), float(actual[:, :2].max())],
                actual_damping_min_max=[float(actual[:, 2:].min()), float(actual[:, 2:].max())],
                last_readback=self.contact.last_readback,
            )
            return result

    return ContactDR
