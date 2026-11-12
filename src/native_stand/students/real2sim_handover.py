"""Log-constrained handover diagnostics, never a hardware state estimator.

Unknown root velocity, contact and executed targets remain explicit assumptions.
Geometry fitting is initialization only, not a hidden runtime controller.
"""
import numpy as np

LEG = [0, 1, 3, 4]
WHEEL = [2, 5]
SCALE = np.array([1.5, 1.5, 100., 1.5, 1.5, 100.])


def decode_log(observation):
    x = np.asarray(observation, dtype=float)
    if x.shape != (31,) or not np.isfinite(x).all():
        raise ValueError('finite explicit31D observation required')
    norm = np.linalg.norm(x[3:6])
    if not .98 < norm < 1.02:
        raise ValueError('logged unit gravity invalid')
    q = np.zeros(6); q[LEG] = x[6:10] * 1.5
    previous = np.zeros(6)
    previous[LEG] = x[16:20] * 1.5
    previous[WHEEL] = x[20:22] * 100.
    return dict(q=q, dq=x[10:16] * 20., gyro=x[:3] * 4.,
                gravity=x[3:6] / norm, gravity_norm=norm,
                previous=previous, command=x[22:25] * [4., 4., .5])


def signal_metrics(t, targets, dq, gravity):
    """Same sampled observable metrics for real and sim; no inferred contact."""
    t = np.asarray(t); targets = np.asarray(targets); dq = np.asarray(dq)
    gravity = np.asarray(gravity)
    if len(t) < 8 or np.any(np.diff(t) <= 0):
        raise ValueError('at least8 chronological samples required')
    dt = float(np.median(np.diff(t)))
    uniform = np.arange(t[0], t[-1] + dt * .1, dt)
    y = np.stack([np.interp(uniform, t, targets[:, j]) for j in LEG], axis=1)
    # Remove affine drift; FFT peak is descriptive, not a validated modal fit.
    trend = np.column_stack((np.ones(len(uniform)), uniform - uniform[0]))
    y = y - trend @ np.linalg.lstsq(trend, y, rcond=None)[0]
    power = abs(np.fft.rfft(y * np.hanning(len(y))[:, None], axis=0)) ** 2
    freq = np.fft.rfftfreq(len(y), dt)
    eligible = (freq >= 2.) & (freq <= min(45., .45 / dt))
    peaks = freq[eligible][power[eligible].argmax(axis=0)]
    amplitude = np.percentile(targets[:, LEG], 95, axis=0) - np.percentile(targets[:, LEG], 5, axis=0)
    down = -gravity[:, 1]
    fallen = np.flatnonzero(down < .5)
    return dict(window_s=float(t[-1] - t[0]), sample_dt_s=dt,
                dominant_leg_target_hz=peaks.tolist(),
                frequency_note='2..45Hz detrended sampled target peak; <Nyquist only, not mechanical mode certification',
                leg_target_p95_minus_p5_rad=amplitude.tolist(),
                leg_target_step_max_rad=float(abs(np.diff(targets[:, LEG], axis=0)).max()),
                leg_dq_p95_rad_s=np.percentile(abs(dq[:, LEG]), 95, axis=0).tolist(),
                pitch_abs_peak_rad=float(abs(np.arctan2(gravity[:, 2], -gravity[:, 1])).max()),
                roll_abs_peak_rad=float(abs(np.arctan2(gravity[:, 0], -gravity[:, 1])).max()),
                upright_failure_time_s=float(t[fallen[0]] - t[0]) if len(fallen) else None,
                fall_observation='OBSERVED_UP_BELOW_0P5' if len(fallen) else 'NOT_OBSERVED_IN_WINDOW')


def initialize_from_log(model, refs, observation, root_vx):
    """Match observed q/gyro/dq/gravity and solve unobserved passive coordinates.

    Root XY/yaw are gauge choices. Height assumes the lower wheel touches a plane;
    actual contact state and linear velocity were not logged. No physics stepping.
    """
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    s = decode_log(observation)
    d = mujoco.MjData(model)
    free = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
    if len(free) != 1:
        raise ValueError('one free root required')
    root_q = int(model.jnt_qposadr[free[0]])
    rotation, _ = Rotation.align_vectors([[0., 0., -1.]], [s['gravity']])
    quat = rotation.as_quat()
    d.qpos[root_q:root_q+3] = [0., 0., .4]
    d.qpos[root_q+3:root_q+7] = quat[[3, 0, 1, 2]]
    active = np.asarray(refs['qpos'])
    d.qpos[active] = s['q']  # wheel angles arbitrary; absent from the Actor/log.
    passive_joints = [j for j in range(model.njnt)
                      if model.jnt_type[j] == int(mujoco.mjtJoint.mjJNT_HINGE)
                      and j not in refs['joints']]
    passive = model.jnt_qposadr[passive_joints]
    q0 = d.qpos[passive].copy()
    lower = np.array([model.jnt_range[j, 0] if model.jnt_limited[j] else -np.inf for j in passive_joints])
    upper = np.array([model.jnt_range[j, 1] if model.jnt_limited[j] else np.inf for j in passive_joints])
    def residual(q):
        d.qpos[passive] = q; mujoco.mj_forward(model, d)
        closure = np.concatenate([d.site_xpos[a] - d.site_xpos[b] for a, b in refs['loops']])
        return np.r_[closure * 1000., (q - q0) * .001]
    fit = least_squares(residual, np.clip(q0, lower+1e-9, upper-1e-9),
                        bounds=(lower, upper), max_nfev=400)
    residual(fit.x)
    closure = max(np.linalg.norm(d.site_xpos[a]-d.site_xpos[b]) for a, b in refs['loops'])
    if closure > 1e-5:
        raise ValueError(f'log q incompatible with closed chain: {closure}')
    bottoms = []
    for body in refs['wheels']:
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_contype[g]]
        for g in geoms:
            if model.geom_type[g] != int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                raise ValueError('explicit cylindrical wheel support required')
            nz = abs(d.geom_xmat[g].reshape(3, 3)[2, 2])
            extent = model.geom_size[g, 0] * np.sqrt(max(0., 1-nz*nz)) + model.geom_size[g, 1] * nz
            bottoms.append(d.geom_xpos[g, 2] - extent)
    if len(bottoms) != 2:
        raise ValueError('two wheel support shapes required')
    d.qpos[root_q+2] -= min(bottoms)
    mujoco.mj_forward(model, d)
    jacobians = []
    for a, b in refs['loops']:
        ja = np.zeros((3, model.nv)); jb = ja.copy()
        mujoco.mj_jacSite(model, d, ja, None, a); mujoco.mj_jacSite(model, d, jb, None, b)
        jacobians.append(ja-jb)
    eq = np.concatenate(jacobians)
    jp = np.zeros((3, model.nv)); jr = jp.copy()
    mujoco.mj_jacBody(model, d, jp, jr, refs['root'])
    r = d.xmat[refs['root']].reshape(3, 3)
    active_velocity = np.eye(model.nv)[refs['dofs']]
    # Unknown root linear velocity is a declared scan parameter in native axes.
    world_velocity = r @ np.array([0., 0., -root_vx])
    a = np.vstack([eq*100., active_velocity*10., r.T@jr*10., jp])
    b = np.r_[np.zeros(len(eq)), s['dq']*10., s['gyro']*10., world_velocity]
    d.qvel[:] = np.linalg.lstsq(a, b, rcond=None)[0]
    mujoco.mj_forward(model, d)
    error = dict(q_abs_max=float(abs(d.qpos[active][LEG]-s['q'][LEG]).max()),
                 dq_abs_max=float(abs(d.qvel[refs['dofs']]-s['dq']).max()),
                 gyro_abs_max=float(abs(r.T@jr@d.qvel-s['gyro']).max()),
                 gravity_abs_max=float(abs(r.T@np.array([0., 0., -1.])-s['gravity']).max()),
                 closure_m=float(closure), closure_velocity_m_s=float(abs(eq@d.qvel).max()))
    if max(error['dq_abs_max'], error['gyro_abs_max'], error['closure_velocity_m_s']) > 1e-4:
        raise ValueError('initial observed velocity/closure constraints inconsistent')
    return d, dict(measurement_residuals=error, passive_fit_success=bool(fit.success),
                   assumed_root_vx_mps=root_vx, assumed_root_height_m=float(d.qpos[root_q+2]),
                   assumed_wheel_support='lower cylinder tangent to plane; not measured contact',
                   implied_other_wheel_gap_m=float(max(bottoms)-min(bottoms)),
                   unknown=['root linear velocity/height', 'yaw/XY', 'passive branch', 'LQR torque',
                            'executed targets', 'contact', 'spring residual'],
                   solver='initial passive closure least-squares only; no runtime correction')
