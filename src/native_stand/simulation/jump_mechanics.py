"""Static full-USD MuJoCo adapter for collision/inverse-dynamics CPU checks.

No mj_step here. No contact restitution or initialization impulse can create a
jump. Authored body frames, masses, principal inertias and shapes are retained.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from native_stand.tasks.jump.geometry import CHANNELS


def _vector(values):
    return " ".join(format(float(x), ".17g") for x in values)


def _pose(t):
    q = Rotation.from_matrix(t[:3, :3]).as_quat()
    return dict(pos=_vector(t[:3, 3]), quat=_vector(q[[3, 0, 1, 2]]))


def model_from_description(description, *, V4_contract=None, mechanics=None, physical_probe=False):
    import mujoco

    xml = ET.Element("mujoco", model="jump_full_USD_static")
    ET.SubElement(xml, "compiler", angle="radian", fusestatic="false", balanceinertia="false")
    ET.SubElement(xml, "option", timestep=".001", gravity="0 0 -9.81")
    world = ET.SubElement(xml, "worldbody")
    ET.SubElement(
        world,
        "geom",
        name="ground",
        type="plane",
        size="1 1 .1",
        contype="2" if physical_probe else "0",
        conaffinity="1" if physical_probe else "0",
        friction=".5 .005 .0001",
        solref=".008 1",
    )
    root = ET.SubElement(
        world, "body", name="robot", pos="0 0 .34", quat=".7071067811865476 .7071067811865476 0 0"
    )
    ET.SubElement(root, "freejoint", name="root_free")
    bodies = {"robot": root}
    names = []
    pending = list(description["joints"])
    while pending:
        later = []
        for j in pending:
            if j["parent"] not in bodies:
                later.append(j)
                continue
            f1 = np.linalg.inv(j["inverse_frame1"])
            zero = np.asarray(j["frame0"]) @ np.asarray(j["inverse_frame1"])
            node = ET.SubElement(bodies[j["parent"]], "body", name=j["child"], **_pose(zero))
            bodies[j["child"]] = node
            axis = f1[:3, j["axis"]]
            attrs = dict(
                name=j["name"],
                type="hinge",
                pos=_vector(f1[:3, 3]),
                axis=_vector(axis),
                damping="0",
                armature=str(j.get("armature_kg_m2", 0.0)),
            )
            if j.get("limited", True):
                attrs.update(limited="true", range=_vector(j["limits_rad"]))
            else:
                attrs["limited"] = "false"
            ET.SubElement(node, "joint", **attrs)
            names.append(j["name"])
        if len(later) == len(pending):
            raise ValueError("Unconnected body tree")
        pending = later
    scaling = {}
    if V4_contract is not None:
        for side in ["L", "R"]:
            row = V4_contract["sides"][side]
            for body in row["body_changes"]:
                if body["name"] in scaling:
                    raise ValueError("Duplicate V4 body scaling")
                scaling[body["name"]] = row["parameters"]["leg_mass_inertia_scale"]
    for b in description["bodies"]:
        scale = scaling.get(b["name"], 1.0)
        ET.SubElement(
            bodies[b["name"]],
            "inertial",
            mass=str(b["mass_kg"] * scale),
            pos=_vector(b["COM_local_m"]),
            diaginertia=_vector(np.asarray(b["diagonal_inertia_kg_m2"]) * scale),
            quat=_vector(b["principal_axes_wxyz"]),
        )
    for i, g in enumerate(description["collision_shapes"]):
        frame = np.asarray(g["body_frame"]).copy()
        if g["kind"] in ("Capsule", "Cylinder"):
            axis = np.eye(3)[g["axis"]]
            rot = Rotation.align_vectors([axis], [[0, 0, 1]])[0].as_matrix()
            frame[:3, :3] = frame[:3, :3] @ rot
        attrs = dict(
            name="shape_" + str(i),
            **_pose(frame),
            contype="1" if physical_probe else "0",
            conaffinity="2" if physical_probe else "0",
            friction=".5 .005 .0001",
            solref=".008 1",
        )
        if g["kind"] == "Cube":
            attrs.update(type="box", size=_vector(g["half_size"]))
        elif g["kind"] == "Sphere":
            attrs.update(type="sphere", size=str(g["radius"]))
        else:
            attrs.update(type=g["kind"].lower(), size=_vector([g["radius"], g["half_height"]]))
        ET.SubElement(bodies[g["body"]], "geom", **attrs)
    eq = ET.SubElement(xml, "equality")
    # Site constraints use both original local anchors. They are only for
    # static residual/Jacobian inspection in this adapter.
    for i, j in enumerate(description["loops"]):
        ET.SubElement(
            bodies[j["parent"]], "site", name=f"loop{i}a", pos=_vector(j["anchor0"]), size=".001"
        )
        ET.SubElement(
            bodies[j["child"]], "site", name=f"loop{i}b", pos=_vector(j["anchor1"]), size=".001"
        )
        ET.SubElement(eq, "connect", name=j["name"], site1=f"loop{i}a", site2=f"loop{i}b")
    if physical_probe:
        act = ET.SubElement(xml, "actuator")
        for name, cap in zip(
            CHANNELS, [40.0, 40.0, 2.44694786377709, 40.0, 40.0, 2.44694786377709], strict=False
        ):
            ET.SubElement(
                act,
                "motor",
                name=name + "_motor",
                joint=name,
                gear="1",
                ctrllimited="true",
                ctrlrange=f"{-cap} {cap}",
                forcelimited="true",
                forcerange=f"{-cap} {cap}",
            )
        contact = ET.SubElement(xml, "contact")
        # The platform supports ground collisions for every authored primitive;
        # pair only mechanically nonadjacent limbs against the chassis box.
        for i, g0 in enumerate(description["collision_shapes"]):
            if g0["body"] != "robot" or g0["kind"] != "Cube":
                continue
            for j, g1 in enumerate(description["collision_shapes"]):
                if g1["body"] in ("left_lower_leg", "right_lower_leg", "left_wheel", "right_wheel"):
                    ET.SubElement(
                        contact,
                        "pair",
                        geom1="shape_" + str(i),
                        geom2="shape_" + str(j),
                        solref=".008 1",
                    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    refs = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names}
    for n in CHANNELS:
        j = refs[n]
        model.dof_armature[model.jnt_dofadr[j]] = (
            0.012 if n not in ("left_wheel_spin", "right_wheel_spin") else 0.0
        )
    if mechanics is not None:
        if set(mechanics["joint_names"]) != set(refs):
            raise ValueError("Full mechanical joint identity")
        for name, j in refs.items():
            dof = model.jnt_dofadr[j]
            model.dof_armature[dof] = mechanics["armature"][name]
            model.dof_damping[dof] = mechanics["damping"][name]
            model.dof_frictionloss[dof] = mechanics["friction_loss_Nm"][name]
    return model, refs


def static_data(model, refs, shape):
    import mujoco

    d = mujoco.MjData(model)
    d.qpos[2] = shape.root_support_height_m
    for name, j in refs.items():
        d.qpos[model.jnt_qposadr[j]] = shape.q[name]
    mujoco.mj_forward(model, d)
    return d
