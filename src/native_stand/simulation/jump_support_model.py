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


class SupportWorkspace:
    """Topology indices and overwritten scratch arrays for one inverse instance."""

    def __init__(self, plant):
        m = plant.model
        self.active = plant.vids[list(LEGS)]
        self.passive = np.array(
            [
                m.jnt_dofadr[j]
                for n, j in plant.refs.items()
                if n not in CHANNELS and "guide" not in n
            ]
        )
        self.sites = [
            tuple(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"loop{k}{s}") for s in "ab")
            for k in range(4)
        ]
        self.geoms = [
            next(
                int(g)
                for g in np.flatnonzero(m.geom_bodyid == body)
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
            )
            for body in plant.wheel_bodies
        ]
        self.a = np.zeros((3, m.nv))
        self.b = self.a.copy()
        self.closure = np.zeros((12, m.nv))
        self.tangent = np.zeros((m.nv, 4))
        self.tangent[self.active, np.arange(4)] = 1.0
        self.mass = np.zeros((m.nv, m.nv))
        self.external = np.zeros(m.nv)


def estimate(plant, *, workspace=None):
    m, d = plant.model, plant.data
    w = workspace or SupportWorkspace(plant)
    active, passive = w.active, w.passive
    a, b = w.a, w.b
    for k in range(4):
        mujoco.mj_jacSite(m, d, a, None, w.sites[k][0])
        mujoco.mj_jacSite(m, d, b, None, w.sites[k][1])
        w.closure[3 * k : 3 * k + 3] = a - b
    closure, tangent = w.closure, w.tangent
    tangent[passive] = -np.linalg.pinv(closure[:, passive], rcond=1e-9) @ closure[:, active]
    mass = w.mass
    full_mass(m, d, mass)
    load = mass @ d.qacc + d.qfrc_bias - d.qfrc_passive
    external = w.external
    external[plant.vids] = plant.actual_NET
    internal = tangent.T @ (load - external)
    driver = d.qfrc_actuator[active].copy()
    jacobians = []
    forward = d.xmat[plant.root].reshape(3, 3) @ np.array([0.0, 0.0, -1.0])
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    for side, body in enumerate(plant.wheel_bodies):
        # Geometric wheel bottom point, independent of actual ground contact.
        geom = w.geoms[side]
        point = d.geom_xpos[geom].copy()
        point[2] -= m.geom_size[geom, 0]
        jac = w.a
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
