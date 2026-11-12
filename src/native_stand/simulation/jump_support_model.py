"""Simulation-only inverse-dynamics support hypothesis, without contact reads.

Full simulated q/dq/ddq are used here. This is NOT the firmware sensor adapter.
Driver effort is actual simulated actuator effort after response, not PD demand.
"""

import mujoco
import numpy as np

from native_stand.tasks.jump.geometry import CHANNELS, LEGS


def full_mass(model, data, output):
    """MuJoCo3.12 takes MjData; older wheels take its sparse qM buffer."""
    if hasattr(data, "qM"):
        mujoco.mj_fullM(model, output, data.qM)
    else:
        mujoco.mj_fullM(model, data, output)


def estimate(plant):
    m, d = plant.model, plant.data
    active = plant.vids[list(LEGS)]
    passive = np.array(
        [m.jnt_dofadr[j] for n, j in plant.refs.items() if n not in CHANNELS and "guide" not in n]
    )
    rows = []
    for k in range(4):
        a = np.zeros((3, m.nv))
        b = a.copy()
        mujoco.mj_jacSite(
            m, d, a, None, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"loop{k}a")
        )
        mujoco.mj_jacSite(
            m, d, b, None, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"loop{k}b")
        )
        rows.append(a - b)
    closure = np.concatenate(rows)
    tangent = np.zeros((m.nv, 4))
    tangent[active, np.arange(4)] = 1.0
    tangent[passive] = -np.linalg.pinv(closure[:, passive], rcond=1e-9) @ closure[:, active]
    mass = np.zeros((m.nv, m.nv))
    full_mass(m, d, mass)
    load = mass @ d.qacc + d.qfrc_bias - d.qfrc_passive
    external = np.zeros(m.nv)
    external[plant.vids] = plant.actual_NET
    internal = tangent.T @ (load - external)
    driver = d.qfrc_actuator[active].copy()
    jacobians = []
    forward = d.xmat[plant.root].reshape(3, 3) @ np.array([0.0, 0.0, -1.0])
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    for side, body in enumerate(plant.wheel_bodies):
        # Geometric wheel bottom point, independent of actual ground contact.
        geom = next(
            int(g)
            for g in np.flatnonzero(m.geom_bodyid == body)
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
        )
        point = d.geom_xpos[geom].copy()
        point[2] -= m.geom_size[geom, 0]
        jac = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jac, None, point, body)
        foot = np.stack((jac[2], forward @ jac)) @ tangent
        jacobians.append(foot[:, 2 * side : 2 * side + 2].reshape(-1))
    valid = bool(np.isfinite(np.r_[internal, driver, np.concatenate(jacobians)]).all())
    return dict(
        valid=valid,
        fault_reason=None if valid else "nonfinite_simulation_inverse_model",
        internal_tau=internal,
        driver_tau=driver,
        support_J=np.concatenate(jacobians),
        source="simulation_full_state_inverse_model_v1",
        contact_truth_used=False,
        hardware_adapter_accepted=False,
    )
