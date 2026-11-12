"""Explicit, diagnostic-only bridge for paired motor-response experiments.

The colleague bridge preserves its MJCF mechanics/contact; it deliberately uses
our policy, source C controller and 1/5/10ms schedule. It is NOT the D5 runtime.
"""
import copy
import numpy as np

VERSION = 'motion4500_motor_response_pair_v1'
LEGS = [0, 1, 3, 4]


def declaration(spec):
    value = spec.get('response_probe')
    if value is None:
        return None
    if value.get('version') != VERSION or value.get('asset') not in ('current', 'colleague'):
        raise ValueError('explicit response diagnostic version/asset required')
    if spec['PD'] != dict(leg_Kp=20., leg_Kd=2., wheel_Kv=.15):
        raise ValueError('response pair must keep PD20/2/.15')
    return value


def bootstrap_case(spec, default):
    """Constructor pauses before reset; never inherit a training flex slot."""
    value=declaration(spec)
    if value is None:return copy.deepcopy(default)
    return {**default,'profile':'R','training':False,'motor_tau_ms':0.}


def episode_parameters(spec, row, mechanical, physical):
    """Return independent copies; tau0 is an exact no-op for the existing path."""
    value = declaration(spec)
    result, parameters = copy.deepcopy(mechanical), copy.deepcopy(physical)
    if value is None:
        return result, parameters
    tau = row.get('motor_tau_ms')
    if tau not in (0., 2.):
        raise ValueError('predeclared tau0/2 pair required')
    parameters['motor_tau_ms'] = [tau, tau, 0., tau, tau, 0.]
    if value['asset'] == 'colleague':
        if result['name'] != 'R':
            raise ValueError('colleague bridge only tests its original rigid mechanism')
        result['load_armature'] = .01  # explicit MJCF nominal, not our .012
    return result, parameters


def prepare_colleague(model, refs):
    """Only timing and electrical ceilings change, never geometry or inertia."""
    import mujoco
    from .motion_mcu_execution import WHEEL_CAP
    from .mujoco_solver import readback
    if model.neq != 4 or model.nu != 6:
        raise ValueError('closed-chain six-actuator topology required')
    raw = readback(model, refs)
    actuator = np.asarray(refs['actuators'])
    gear = np.asarray(refs['gears'])
    if not np.allclose(abs(gear), 1.):
        raise ValueError('explicit unit actuator gear required by this bridge')
    cap = np.array([40., 40., WHEEL_CAP, 40., 40., WHEEL_CAP])
    model.opt.timestep = .001
    model.actuator_ctrllimited[actuator] = 1
    model.actuator_forcelimited[actuator] = 1
    model.actuator_ctrlrange[actuator] = np.stack([-cap, cap], -1)
    model.actuator_forcerange[actuator] = np.stack([-cap, cap], -1)
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return dict(raw_solver=raw, effective_solver=readback(model, refs),
        overrides=dict(physics_dt_s=.001, PD_dt_s=.005, policy_dt_s=.01,
            motor_cap_nm=cap.tolist(), source_C_and_our31D=True,
            independent_spring='our unmeasured counter-compensation hypothesis'),
        preserved='MJCF geometry, mass/inertia, joint damping/friction/armature, limits, contact and solver tolerances',
        D5_runtime=False)


def array_summary(model, refs):
    """Small enough to keep once per world; includes actual effective values."""
    keys = ('body_mass', 'body_inertia', 'body_ipos', 'dof_armature',
            'dof_damping', 'dof_frictionloss', 'jnt_range', 'jnt_limited',
            'geom_friction', 'geom_solref', 'geom_solimp', 'eq_solref',
            'eq_solimp', 'actuator_ctrlrange', 'actuator_forcerange')
    return dict(values={key: np.asarray(getattr(model, key)).tolist() for key in keys},
                joint_ids=list(map(int, refs['joints'])),
                dofs=list(map(int, refs['dofs'])), dt_s=float(model.opt.timestep))
