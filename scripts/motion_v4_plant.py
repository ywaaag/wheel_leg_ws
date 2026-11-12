"""Full-v4 identified leg plant for the shared MuJoCo player.

The plant is the delivered equivalent candidate in
``artifacts/2026_10_30__bilateral_detailed_response_v4/candidate_contract.json``:
named-body mass/inertia scaling, motor-only actuator clamp, PD40/2 -> per-side modal
transport and first-order response, a separate NET load on ``qfrc_applied`` and the
5 ms held/quantised leg observer. It is not measured whole-robot truth, and nothing
here writes files or edits the assets.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'motion_v4_identified_v1'
CONTRACT = ROOT / 'artifacts/2026_10_30__bilateral_detailed_response_v4/candidate_contract.json'
CONTRACT_SHA256 = '5510088cc1f57f823718fd36218beb32cabf3c1d8ce708cf75531129178dc1cd'
LEGS = np.array([0, 1, 3, 4])
WHEELS = np.array([2, 5])
# Geometric left is the model's right_* pair; both sides use capture_to_model_signs [-1,+1].
PAIRS = np.array([[3, 4], [0, 1]])
SIGN = np.array([-1., 1.])
MOTOR_CAP_NM = np.array([40., 40., 2.44694786377709, 40., 40., 2.44694786377709])
WHEEL_TRANSPORT_MS = 20
WHEEL_TAU_MS = .5


def contract(path=None):
    source = Path(path or CONTRACT)
    data = json.loads(source.read_text(encoding='utf-8'))
    if data.get('version') != 'fullchain_detailed_response_candidate_v4':
        raise ValueError('not the full-v4 identified response contract')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return data, dict(path=str(source.resolve()), sha256=digest, pinned=digest == CONTRACT_SHA256)


def apply(model, refs, spec, cfg, data):
    """Scale the declared leg bodies and bind the motor-only actuator clamp."""
    import mujoco
    before = dict(body_mass=model.body_mass.copy(), body_inertia=model.body_inertia.copy(),
                  armature=model.dof_armature.copy())
    touched = []
    for side in ('L', 'R'):
        row = data['sides'][side]
        scale = float(row['parameters']['leg_mass_inertia_scale'])
        for body in row['body_changes']:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body['name'])
            if index < 0:
                raise ValueError(f"v4 body {body['name']} is missing from the model")
            model.body_mass[index] *= scale
            model.body_inertia[index] *= scale
            touched.append(dict(side=side, body=body['name'], scale=scale,
                                mass_kg=float(model.body_mass[index]),
                                inertia_kg_m2=model.body_inertia[index].tolist()))
    actuators = np.asarray(refs['actuators'])
    gears = np.abs(np.asarray(refs['gears']))
    model.actuator_ctrlrange[actuators] = np.stack((-MOTOR_CAP_NM / gears, MOTOR_CAP_NM / gears), -1)
    # MuJoCo forcerange is actuator force before gear: motor only, never motor+NET twice.
    model.actuator_forcerange[actuators] = model.actuator_ctrlrange[actuators]
    model.actuator_forcelimited[actuators] = True
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return dict(version=VERSION, contract=data['version'], bodies=touched,
                motor_cap_joint_Nm=MOTOR_CAP_NM.tolist(),
                motor_force_limit_actuator_Nm=(model.actuator_forcerange[actuators]
                                               * gears[:, None]).tolist(),
                armature=model.dof_armature.tolist(),
                observer={side: data['sides'][side]['effective_observer'] for side in ('L', 'R')},
                extra_B={side: [data['sides'][side]['parameters']['extra_modal_B_common'],
                                data['sides'][side]['parameters']['extra_modal_B_difference']]
                         for side in ('L', 'R')},
                net_K={side: [[data['sides'][side]['parameters']['net_K_common'],
                               data['sides'][side]['parameters']['net_K_cross']],
                              [data['sides'][side]['parameters']['net_K_cross'],
                               data['sides'][side]['parameters']['net_K_difference']]]
                       for side in ('L', 'R')},
                separate_NET='net load on qfrc_applied; motor only through the actuator',
                scope='full-v4 equivalent fixture candidate, not measured hardware')


def modal(x):
    return np.stack((x.mean(-1), .5 * (x[..., 1] - x[..., 0])), -1)


def physical(x):
    return np.stack((x[..., 0] - x[..., 1], x[..., 0] + x[..., 1]), -1)


class LegResponse:
    """Frozen v4 leg kernel: held observer -> PD -> modal transport/response -> motor and NET."""

    def __init__(self, data, q, v, target):
        names = data['parameter_names']
        self.x = np.array([[data['sides'][s]['parameters'][k] for k in names] for s in ('L', 'R')])
        observers = [data['sides'][s]['effective_observer'] for s in ('L', 'R')]
        self.delay = np.array([o['transport_ms'] for o in observers], int)
        self.q_age = np.array([o['q_age_ms'] for o in observers], int)
        self.v_age = np.array([o['v_age_ms'] for o in observers], int)
        self.reference = np.array([o['q_reference'] for o in observers])
        self.alpha = np.where(self.x[:, 3:5] > 0, -np.expm1(-1 / np.maximum(self.x[:, 3:5], 1e-12)), 1.)
        velocity_tau = np.array([o['velocity_tau_ms'] for o in observers])[:, None]
        self.velocity_alpha = np.where(velocity_tau > 0,
                                       -np.expm1(-1 / np.maximum(velocity_tau, 1e-12)), 1.)
        self.filtered = v.copy(); self.heldq = q.copy(); self.heldv = v.copy()
        self.desired = np.clip(40 * (target - q) - 2 * v, -40, 40)
        self.actual = modal(self.desired)
        self.q_history = np.stack([q + v * (h - 128) * .001 for h in range(128)])
        self.v_history = np.tile(v, (128, 1, 1)); self.queue = np.tile(self.actual, (128, 1, 1))
        self.tick = 0; self.raw = self.desired.copy()

    def step(self, q, v, target):
        tick = self.tick
        self.filtered += self.velocity_alpha * (v - self.filtered)
        self.q_history[tick % 128] = q
        self.v_history[tick % 128] = self.filtered
        if tick % 5 == 0:
            rows = np.arange(2)
            self.heldq = self.q_history[(tick - self.q_age) % 128, rows].copy()
            self.heldv = self.v_history[(tick - self.v_age) % 128, rows].copy()
            self.heldq = np.round(self.heldq / (12.566 / 65535)) * (12.566 / 65535)
            self.heldv = np.round(self.heldv / (90 / 4095)) * (90 / 4095)
            self.raw = 40 * (target - self.heldq) - 2 * self.heldv
            self.desired = np.clip(self.raw, -40, 40)
        self.queue[tick % 128] = modal(self.desired)
        rows = np.arange(2)
        self.actual += self.alpha * (self.queue[(tick - self.delay) % 128, rows] - self.actual)
        z = modal(q - self.reference); dz = modal(v); x = self.x
        extra = np.stack((x[:, 5] - x[:, 7] * z[:, 0] - x[:, 9] * z[:, 1] - x[:, 1] * dz[:, 0],
                          x[:, 6] - x[:, 8] * z[:, 1] - x[:, 9] * z[:, 0] - x[:, 2] * dz[:, 1]), -1)
        extra -= x[:, 10:12] * z * z
        self.tick += 1
        return np.clip(physical(self.actual), -40, 40), physical(extra)


class V4Execution:
    """Adapter with the attribute surface the shared playback loop already uses."""

    v4 = True

    def __init__(self, data, q6, v6, target6, tick=0):
        self.contract = data
        self.profile = dict(version=VERSION, contract=data['version'],
                            pd_period_ms=5, policy_period_ms=10, physics_period_ms=1,
                            motor_delay_ms=[0, 0, WHEEL_TRANSPORT_MS, 0, 0, WHEEL_TRANSPORT_MS],
                            motor_tau_ms=[0, 0, WHEEL_TAU_MS, 0, 0, WHEEL_TAU_MS],
                            pd_limits_nm=[40, 40, 2.44694786377709, 40, 40, 2.44694786377709],
                            driver_limits_nm=MOTOR_CAP_NM.tolist(),
                            solver_limits_nm=MOTOR_CAP_NM.tolist(),
                            reference_limits_nm=MOTOR_CAP_NM.tolist(),
                            actor_age_ms=20, PD_age_ms=5, local_target_delay_ms=5,
                            spring='v4 NET load on qfrc_applied', initial_motor='zero')
        self.reset(tick, q6, target6, v6)

    def reset(self, tick, q6, target6, v6=None):
        velocity = np.zeros(6) if v6 is None else np.asarray(v6, dtype=float)
        q6 = np.asarray(q6, dtype=float)
        self.targets = np.tile(np.asarray(target6, dtype=float), (32, 1))
        self.leg = LegResponse(self.contract, q6[PAIRS] * SIGN, velocity[PAIRS] * SIGN,
                               np.asarray(target6, dtype=float)[PAIRS] * SIGN)
        self.wheel_queue = np.zeros((128, 2)); self.wheel_state = np.zeros(2); self.wheel_raw = np.zeros(2)
        self._adopted = np.asarray(target6, dtype=float).copy()
        self.adopted = torch.tensor(self._adopted)
        self.pre = torch.zeros(6); self.pd_limited = torch.zeros(6); self.compensation = torch.zeros(6)
        self.driver = torch.zeros(6); self.motor_actual = torch.zeros(6); self.spring_actual = torch.zeros(6)
        self.actual = torch.zeros(6); self.net = torch.zeros(6); self.geometry_valid = True

    def _unpair(self, values):
        out = np.zeros(6); out[PAIRS.flatten()] = (values * SIGN).flatten(); return out

    def held_leg_q(self):
        return self._unpair(self.leg.heldq)[LEGS]

    def held_leg_dq(self):
        return self._unpair(self.leg.heldv)[LEGS]

    def step(self, tick, target, q, dq):
        q = np.asarray(q, dtype=float); dq = np.asarray(dq, dtype=float)
        self.targets[tick % 32] = np.asarray(target, dtype=float)
        if tick % 5 == 0:
            self._adopted = self.targets[(tick - 5) % 32].copy()
            self.adopted = torch.tensor(self._adopted)
            self.wheel_raw = np.clip(.2 * (self._adopted[WHEELS] - dq[WHEELS]),
                                     -MOTOR_CAP_NM[WHEELS], MOTOR_CAP_NM[WHEELS])
        motor_modal, net_modal = self.leg.step(q[PAIRS] * SIGN, dq[PAIRS] * SIGN, self._adopted[PAIRS] * SIGN)
        self.wheel_queue[tick % 128] = self.wheel_raw
        self.wheel_state += (-np.expm1(-1 / WHEEL_TAU_MS)) * (self.wheel_queue[(tick - WHEEL_TRANSPORT_MS) % 128]
                                                             - self.wheel_state)
        motor = self._unpair(motor_modal); motor[WHEELS] = self.wheel_state
        tensor = torch.tensor(motor)
        self.pre = tensor.clone(); self.pd_limited = tensor.clone(); self.driver = tensor.clone()
        self.motor_actual = tensor.clone()
        self.spring_actual = torch.zeros(6)
        self.net = torch.tensor(self._unpair(net_modal))
        self.actual = tensor
        return motor

    def summary(self):
        return dict(self.profile, leg_response=dict(version=VERSION, contract=self.contract['version']),
                    geometry_valid=self.geometry_valid)


def selected(requested):
    return requested == VERSION
