"""Load the SHA-bound historical Motion vM1 MJCF into the shared player.

Only model construction differs. Timing, V4 response, PD, policy execution and
integration continue to belong to MotionPlayer/PlaybackExecution.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from native_stand.artifacts.io import file_sha256, read_json
from native_stand.tasks.motion.encoding import CHANNELS

VERSION = "motion_vm1_mujoco_v1"
XML_SHA = "663e121ef9aed4bfab09ff0d3d98321b7ce77d80b63584235e4da691bd3aebf3"


def load_contract(reference):
    if file_sha256(reference["path"]) != reference["sha256"]:
        raise ValueError("Motion vM1 plant contract SHA differs")
    contract = read_json(reference["path"])
    if contract["version"] != VERSION or contract["solver_profile"] != "closure_contact_tight_v1":
        raise ValueError("Explicit historical vM1 plant required")
    names = set()
    for member in contract["members"]:
        name = member["asset_name"]
        if name in names or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid vM1 asset member")
        names.add(name)
        if file_sha256(member["path"]) != member["sha256"]:
            raise ValueError("Motion vM1 asset SHA differs: " + name)
    xml = [r for r in contract["members"] if r["role"] == "mjcf"]
    if len(xml) != 1 or xml[0]["sha256"] != XML_SHA:
        raise ValueError("Motion vM1 source MJCF identity differs")
    return contract


def make_model(reference, *, description, mechanics, V4_contract, motor_caps):
    import mujoco

    contract = load_contract(reference)
    xml = next(r for r in contract["members"] if r["role"] == "mjcf")
    text = Path(xml["path"]).read_text()
    assets = {
        r["asset_name"]: Path(r["path"]).read_bytes()
        for r in contract["members"]
        if r["role"] == "mesh"
    }
    model = mujoco.MjModel.from_xml_string(text, assets=assets)
    refs = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): j
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    }
    if set(refs) != set(mechanics["joint_names"]):
        raise ValueError("vM1/current nominal mechanical joint identities differ")
    joints = {j["name"]: j for j in description["joints"]}
    for name, j in refs.items():
        dof = model.jnt_dofadr[j]
        model.dof_armature[dof] = mechanics["armature"][name]
        model.dof_damping[dof] = mechanics["damping"][name]
        model.dof_frictionloss[dof] = mechanics["friction_loss_Nm"][name]
        model.jnt_limited[j] = joints[name].get("limited", True)
        if model.jnt_limited[j]:
            model.jnt_range[j] = joints[name]["limits_rad"]
    model.geom_friction[:] = [0.5, 0.0, 0.0]
    for side in ("L", "R"):
        row = V4_contract["sides"][side]
        for body in row["body_changes"]:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body["name"])
            if index < 0:
                raise ValueError("Missing vM1 V4 body: " + body["name"])
            scale = row["parameters"]["leg_mass_inertia_scale"]
            model.body_mass[index] *= scale
            model.body_inertia[index] *= scale
    for name, cap in zip(CHANNELS, motor_caps, strict=True):
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == refs[name])
        if len(matches) != 1:
            raise ValueError("vM1 actuator mapping is not one-to-one")
        actuator = matches[0]
        gear = float(model.actuator_gear[actuator, 0])
        if not np.isfinite(gear) or abs(gear) < 1e-12:
            raise ValueError("Invalid vM1 transmission gear")
        model.actuator_ctrlrange[actuator] = [-cap / abs(gear), cap / abs(gear)]
        model.actuator_forcerange[actuator] = model.actuator_ctrlrange[actuator]
        model.actuator_ctrllimited[actuator] = True
        model.actuator_forcelimited[actuator] = True
    pairs = []
    for e in ET.fromstring(text).findall("equality/connect"):
        ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, e.get(k)) for k in ("site1", "site2")
        ]
        if min(ids) < 0:
            raise ValueError("vM1 closure site missing")
        pairs.append(ids)
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return model, refs, dict(version=VERSION, contract=reference, loop_sites=pairs)
