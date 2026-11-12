"""Explicit post-PD MCU execution chain for the read-only MuJoCo preview.

One chain per spec, chosen by version. This module owns
``mcu_execution.profile.version == motion_mcu_post_pd_execution_v1`` and consumes
that block as the execution configuration; it never reads the ancestor
``motion_timing.configuration`` timing as the current execution profile.

Authoritative order, once per 1 ms physics step:

  Actor 10ms physical target -> local 5ms adoption (previous = adopted)
  -> PD with the declared feedback age -> PD limit -> spring compensation
  -> final motor (driver) limit -> motor command transport -> motor response
  -> summed with the independent spring reaction at the current geometry.

Transport here acts on the post-PD motor torque command. ``AuditExecution``
applies the same millisecond array to the pre-PD target, which is a different
historical hypothesis and is not reused just because the numbers match.
PD and spring/compensation math are the training-side pure functions, so the
preview cannot grow a third formula.
"""
import json
from pathlib import Path

import numpy as np
import torch

from native_stand.students.motion_mcu_execution import (LEGS, WHEELS, MotorRing,
                                                        pd_and_motor, profile,
                                                        spring_compensation)
from native_stand.students.real2sim_delay import ObservationHistory

VERSION = 'motion_mcu_post_pd_execution_v1'
# Playback starts at the training standing height, not the asset's neutral origin height.
DEFAULT_START_HEIGHT_M = .34
START_POSE_PATH = Path(__file__).resolve().parents[1] / 'configs/playback/motion_standing_pose_0p34.json'
# Training selects this declared level for every MCU run (MCUTiming.configure_armature, override=4).
MCU_LEG_ARMATURE_KG_M2 = .012
# Delivered bilateral local response: (hip, active) channel pairs in the 6-channel order.
LEG_RESPONSE_VERSION = 'bilateral_modal_leg_response_v1'
# Geometric left is the model's right_* pair: ch3/ch4, and capture_to_model_signs [-1,+1]
# for both sides (V4 contract side_mapping; the 2026_10_30 handoff itself declares no
# mirroring, so the mapping has to come from the contract, never from channel order).
LEG_SIDE_CHANNELS = (('L', (3, 4)), ('R', (0, 1)))
LEG_SIDE_SIGNS = {'L': (-1., 1.), 'R': (-1., 1.)}
LEG_RESPONSE_CONTRACT = (Path(__file__).resolve().parents[1]
                         / 'artifacts/2026_10_30__bilateral_leg_response_handoff/contract.json')
# Fields this chain consumes; the spec must agree with the frozen formula contract.
DECLARED = ('version', 'pd_period_ms', 'policy_period_ms', 'physics_period_ms', 'leg_kp', 'leg_kd',
            'wheel_kv', 'local_target_delay_ms', 'actor_age_ms', 'PD_age_ms', 'motor_delay_ms',
            'motor_tau_ms', 'pd_limits_nm', 'driver_limits_nm', 'reference_limits_nm', 'solver_limits_nm')
PLANT_PROFILES = ('source_xml_v1', 'motion_mcu_currentplant_v1')


def plant_profile(spec, solver_profile, requested=None):
    """New declared MCU playbacks share the distillation plant; old reproduction is explicit."""
    if requested is not None and requested not in PLANT_PROFILES:
        raise ValueError('unknown MuJoCo plant profile: ' + str(requested))
    version = requested or ('motion_mcu_currentplant_v1' if spec.get('mcu_execution') and
                            solver_profile != 'legacy_xml_v1' else 'source_xml_v1')
    if version == 'motion_mcu_currentplant_v1' and not spec.get('mcu_execution'):
        raise ValueError('current MCU plant requires a declared mcu_execution profile')
    return version


def leg_response_contract(path=None):
    """Delivered bilateral leg response, actuator side only.

    The handoff fits J/B/K/tau per side and per modal coordinate. J and K are effective
    stand-ins for the robot's own inertia and gas spring, so only the actuator-side
    transport and first-order response are adopted here; overlaying J/B/K would count
    the articulated plant's own dynamics twice.
    """
    source = Path(path or LEG_RESPONSE_CONTRACT)
    data = json.loads(source.read_text(encoding='utf-8'))
    if data.get('version') != 'bilateral_local_leg_response_handoff_v1':
        raise ValueError('unknown bilateral leg-response handoff version')
    sides = {}
    for side, _ in LEG_SIDE_CHANNELS:
        block = data['sides'][side]
        params = np.asarray(block['parameters_J_B_K_tau_ms'], dtype=float)
        if params.shape != (2, 4):
            raise ValueError(f'side {side} needs two modal J/B/K/tau rows')
        sides[side] = dict(transport_ms=[int(v) for v in block['modal_post_PD_transport_ms']],
                           tau_ms=[float(v) for v in params[:, 3]],
                           J_modal=[float(v) for v in params[:, 0]],
                           B_modal=[float(v) for v in params[:, 1]],
                           K_modal=[float(v) for v in params[:, 2]])
    return dict(version=LEG_RESPONSE_VERSION, source=str(source.resolve()),
                contract_version=data['version'], sides=sides,
                scope='post-PD transport and first-order response in per-side modal coordinates; '
                      'PD40/2, +-40Nm and the plant inertia/spring/closed chain stay unchanged')


class ModalLegRing:
    """Per-side modal transport/response for the four leg channels.

    common=(hip+active)/2 and differential=(active-hip)/2 follow the delivered contract's
    modal order. A modal delay cannot be written as a per-joint delay, so the transform is
    explicit rather than approximated by four joint constants.
    """

    def __init__(self, contract, initial6):
        self.contract = contract
        self.history = np.zeros((64, 4))
        self.delay = np.array([contract['sides'][side]['transport_ms'][m]
                               for side, _ in LEG_SIDE_CHANNELS for m in (0, 1)], dtype=int)
        tau = np.array([contract['sides'][side]['tau_ms'][m]
                        for side, _ in LEG_SIDE_CHANNELS for m in (0, 1)], dtype=float)
        self.tau_ms = tau.copy()
        self.alpha = np.where(tau == 0, 1., -np.expm1(-1. / np.maximum(tau, 1e-12)))
        self.columns = np.array([c for _, pair in LEG_SIDE_CHANNELS for c in pair])
        self.actual = self._modal(np.asarray(initial6, dtype=float))

    @staticmethod
    def _modal(values):
        out = []
        for side, (hip, active) in LEG_SIDE_CHANNELS:
            sign = LEG_SIDE_SIGNS[side]
            first, second = sign[0] * values[hip], sign[1] * values[active]
            out.extend((.5 * (first + second), .5 * (second - first)))
        return np.asarray(out, dtype=float)

    @staticmethod
    def _physical(modal, base):
        out = np.asarray(base, dtype=float).copy()
        for index, (side, (hip, active)) in enumerate(LEG_SIDE_CHANNELS):
            sign = LEG_SIDE_SIGNS[side]
            common, difference = modal[2 * index], modal[2 * index + 1]
            out[hip] = sign[0] * (common - difference)
            out[active] = sign[1] * (common + difference)
        return out

    def step(self, tick, command6):
        command = np.asarray(command6, dtype=float)
        self.history[tick % 64] = self._modal(command)
        delivered = np.array([self.history[(tick - int(delay)) % 64, column]
                              for column, delay in enumerate(self.delay)])
        self.actual = self.actual + self.alpha * (delivered - self.actual)
        out = self._physical(self.actual, command)
        out[self.columns] = np.clip(out[self.columns], -40., 40.)
        return out


def bilateral_leg_physics(model, refs, contract, pose_q=None):
    """Replace the leg plant's damping from the delivered modal B, and report the residual.

    A joint-space diagonal (MuJoCo dof_damping/armature) can only produce EQUAL common and
    differential modal values, because A^T diag(b) A has equal diagonals for
    common=(hip+active)/2 and differential=(active-hip)/2. The delivered B differs per mode,
    so only the modal mean can be written; the split is reported, never hidden. J cannot be
    written at all here: the fitted modal inertia is below the articulated links' own inertia,
    so matching it would require shrinking link inertia in the asset.
    """
    dofs = np.asarray(refs['dofs'], dtype=int)
    readback = {'version': 'bilateral_leg_plant_v1', 'source': contract.get('source'),
                'contract_version': contract.get('contract_version'), 'sides': {}}
    for side, (hip, active) in LEG_SIDE_CHANNELS:
        block = contract['sides'][side]
        b_modes = np.asarray(block['B_modal'], dtype=float)
        joint_damping = float(b_modes.mean() / 2.)
        for channel in (hip, active):
            model.dof_damping[int(dofs[channel])] = joint_damping
        readback['sides'][side] = {
            'delivered_J_modal': block['J_modal'], 'delivered_B_modal': b_modes.tolist(),
            'delivered_K_modal': block['K_modal'], 'tau_ms_modal': block['tau_ms'],
            'written_joint_damping_nm_s_rad': joint_damping,
            'effective_B_modal': [2 * joint_damping, 2 * joint_damping],
            'B_modal_residual': (2 * joint_damping - b_modes).tolist(),
            'J_modal_not_written': 'fitted modal J is below the links own inertia; writing it needs '
                                   'an asset inertia edit, not armature',
            'K_modal_not_written': 'K is the gas spring working-point stiffness and is already '
                                   'represented by the C compensation; overlaying it double counts'}
    readback['joint_damping'] = model.dof_damping[dofs].tolist()
    readback['scope'] = ('only the modal mean of B is representable as joint damping; J, the per-mode '
                         'B split and K need asset inertia and coupled modal springs, not dof arrays')
    return readback


def apply_current_plant(model, refs, spec, cfg):
    """Materialize the SAME nominal plant used in the 4500 precision probe.

    USD mechanical ranges and armatures, passive/guide damping, active timed PD,
    and mu=.5 are shared by preview, Teacher collection and Student evaluation.
    No geometry, mass, PD gain, physical timestep, or motor capability changes.
    Solver compliance is applied separately by mujoco_solver.apply.
    """
    import math
    import mujoco
    import yaml
    from pxr import Usd, UsdPhysics
    from native_stand.students.data import sha
    from compare_motion_timing_profiles import apply_actuator_profile

    if not cfg or not cfg.get('mcu') or cfg.get('forced_by_cli'):
        raise ValueError('current plant needs the spec-declared MCU execution')
    asset = Path(spec['asset_path']); source = Path(spec['colleague_config_path'])
    if sha(asset) != spec['asset_sha256']:
        raise ValueError('current plant USD identity differs')
    expected = spec.get('colleague_config_sha256')
    if expected is not None and sha(source) != expected:
        raise ValueError('current plant actuator configuration identity differs')
    stage = Usd.Stage.Open(str(asset))
    joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    actuators = yaml.safe_load(source.read_text())['robot']['actuators']
    for joint in range(model.njnt):
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        dof = int(model.jnt_dofadr[joint]); prim = joints.get(name)
        model.dof_damping[dof] = 0.
        model.dof_frictionloss[dof] = 0.
        if prim:
            armature = prim.GetAttribute('physxJoint:armature').Get()
            if armature is not None:
                model.dof_armature[dof] = float(armature)
            lo = prim.GetAttribute('physics:lowerLimit').Get()
            hi = prim.GetAttribute('physics:upperLimit').Get()
            if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi):
                model.jnt_limited[joint] = True
                model.jnt_range[joint] = np.deg2rad([lo, hi])
        if name in actuators['passive']['joint_names']:
            model.dof_damping[dof] = actuators['passive']['damping']
        if name in actuators['guide_wheels']['joint_names']:
            model.dof_damping[dof] = actuators['guide_wheels']['damping']
    original = {k: getattr(model, k).copy() for k in ('dof_armature', 'dof_damping', 'dof_frictionloss')}
    actuator = apply_actuator_profile(model, refs, cfg, original)
    cap = np.asarray(cfg['profile']['solver_limits_nm']) / np.abs(refs['gears'])
    model.actuator_ctrlrange[refs['actuators']] = np.stack((-cap, cap), -1)
    model.geom_friction[:] = 0.
    model.geom_friction[:, 0] = .5
    model.dof_damping[np.asarray(refs['dofs'])[WHEELS]] = 0.
    model.dof_frictionloss[np.asarray(refs['dofs'])[WHEELS]] = 0.
    return dict(version='motion_mcu_currentplant_v1', actuator=actuator,
        input_sha256={str(asset.resolve()): sha(asset), str(source.resolve()): sha(source)},
        joint_range=model.jnt_range.tolist(), joint_limited=model.jnt_limited.tolist(),
        armature=model.dof_armature.tolist(), damping=model.dof_damping.tolist(),
        frictionloss=model.dof_frictionloss.tolist(), geom_friction=model.geom_friction.tolist(),
        actuator_ctrlrange=model.actuator_ctrlrange.tolist(),
        scope='current nominal MCU MuJoCo hypothesis, not measured hardware or PhysX certification')


def execution_profile(version):
    """Explicit dispatch: each declared execution version brings its own frozen contract.

    No version is inferred from the numbers; an unlisted one is an error, so a new chain
    cannot be played as if it were the post-PD default.
    """
    if version == VERSION:
        return profile()
    from native_stand.students.motion_standing_delay import EXECUTION, execution_profile as standing_delay
    if version == EXECUTION:
        return standing_delay()
    raise ValueError(f'unknown MCU execution version {version!r}; explicit dispatch required')


def declared(spec):
    """Validated ``mcu_execution`` block, or None when the spec declares no MCU chain."""
    execution = spec.get('mcu_execution')
    if not execution:
        return None
    block = execution.get('profile')
    if not isinstance(block, dict):
        raise ValueError('mcu_execution.profile required')
    version = block.get('version')
    frozen = execution_profile(version)
    drift = [k for k in DECLARED if _value(block.get(k)) != _value(frozen[k])]
    if drift:
        raise ValueError('spec MCU profile differs from the frozen contract: ' + ','.join(drift))
    pose = execution.get('pose')
    if not isinstance(pose, dict) or 'joint_pos' not in pose:
        raise ValueError('mcu_execution.pose with joint_pos required for the declared initial state')
    return {'version': version, 'profile': frozen, 'declared': dict(block), 'pose': pose}


def _value(value):
    if isinstance(value, (list, tuple)):
        return [_value(v) for v in value]
    return float(value) if isinstance(value, (int, float)) else value


def start_pose(spec, height=None):
    """Effective playback start state at the requested height, or None for the recipe's own.

    A spec that declares ``mcu_execution.pose`` always uses its own declared posture. Any
    other recipe falls back to the recorded playback standing pose when the requested height
    matches it; heights without a verified pose keep the recipe start instead of scaling one.
    """
    requested = DEFAULT_START_HEIGHT_M if height is None else float(height)
    declared = (spec.get('mcu_execution') or {}).get('pose')
    if isinstance(declared, dict) and declared.get('joint_pos'):
        return {'height_m': float(declared.get('height_m', requested)),
                'root_quaternion_xyzw': [float(v) for v in declared['root_quaternion_xyzw']],
                'joint_pos': {k: float(v) for k, v in declared['joint_pos'].items()},
                'closure_max_m': declared.get('closure_max_m'), 'source': 'spec mcu_execution.pose',
                'requested_height_m': requested}
    if START_POSE_PATH.is_file():
        block = json.loads(START_POSE_PATH.read_text(encoding='utf-8'))
        if abs(float(block['height_m']) - requested) <= 1e-9:
            return {'height_m': float(block['height_m']),
                    'root_quaternion_xyzw': [float(v) for v in block['root_quaternion_xyzw']],
                    'joint_pos': {k: float(v) for k, v in block['joint_pos'].items()},
                    'closure_max_m': block.get('closure_max_m'),
                    'source': f'recorded playback start pose {START_POSE_PATH.name}',
                    'source_spec': (block.get('source') or {}).get('spec'),
                    'source_spec_sha256': (block.get('source') or {}).get('spec_sha256'),
                    'requested_height_m': requested}
    return None


def forced(spec):
    """Explicit CLI override: play a spec that declares no MCU chain on the MCU chain.

    The recipe keeps its own initial state and plant inertia, so the execution chain is
    the only changed mechanism. Recorded as forced, never as a spec-declared semantics.
    """
    if spec.get('mcu_execution'):
        return declared(spec)
    return {'version': VERSION, 'profile': profile(), 'declared': dict(profile()), 'pose': None,
            'forced_by_cli': True}


def configuration(args, execution, spec):
    """Playback configuration from mcu_execution.profile; plant inertia stays a plant property."""
    forced_chain = bool(execution.get('forced_by_cli'))
    armature = getattr(args, 'leg_armature_m', None)
    if armature is not None and float(armature) != MCU_LEG_ARMATURE_KG_M2 and not forced_chain:
        raise ValueError(f'MCU playback uses the training-fixed leg armature {MCU_LEG_ARMATURE_KG_M2}, '
                         f'not {float(armature)}')
    plant = ((spec.get('motion_timing') or {}).get('configuration') or {}).get('wheel_execution') or {}
    total = plant.get('wheel_total_J_candidate_kg_m2_ch2_ch5')
    p = execution['profile']
    played = {'motor_delay_ms': [int(v) for v in p['motor_delay_ms']],
              'motor_tau_ms': [int(v) for v in p['motor_tau_ms']],
              'PD_age_ms': int(p['PD_age_ms']), 'actor_age_ms': int(p['actor_age_ms'])}
    levels = ((spec.get('motion_timing') or {}).get('configuration') or {}).get('armature_levels_kg_m2') or []
    if forced_chain:
        # Only the execution chain changes; keep the recipe's declared plant inertia unless asked.
        level = MCU_LEG_ARMATURE_KG_M2 if armature is None else float(armature)
        if levels and level not in [float(v) for v in levels]:
            raise ValueError(f'leg armature must be one declared level {levels}')
        armature_value, armature_source = level, '--leg-armature-m explicit (forced MCU chain)'
    else:
        armature_value, armature_source = MCU_LEG_ARMATURE_KG_M2, 'training-fixed MCU level'
    cfg = {'mcu': True, 'mcu_version': p['version'], 'mcu_execution': execution, 'profile': dict(p, **played),
           'mcu_pose': execution.get('pose'), 'forced_by_cli': forced_chain,
           'leg_armature': armature_value, 'armature_source': armature_source,
           'kp': float(p['leg_kp']), 'kd': float(p['leg_kd']),
           'kv': float(p['wheel_kv']),
           'wheel_peak': float(plant.get('wheel_peak_Nm', p['driver_limits_nm'][2])),
           'wheel_J': [float(v) for v in total] if total else None, 'wheel_B': [0., 0.], 'wheel_C': None,
           'transport_ms': played['motor_delay_ms'], 'feedback_ms': played['PD_age_ms'],
           'tau_ms': played['motor_tau_ms'], 'actor_sensor_ms': played['actor_age_ms'],
           'transport_placement': 'post_PD_motor_torque_command',
           'local_target_transport_ms': int(p.get('local_target_delay_ms', 0)),
           'pd_limits_nm': [float(v) for v in p['pd_limits_nm']],
           'driver_limits_nm': [float(v) for v in p['driver_limits_nm']],
           'reference_limits_nm': [float(v) for v in p['reference_limits_nm']],
           'solver_limits_nm': [float(v) for v in p['solver_limits_nm']],
           'spring': p['spring'], 'initial_motor': p['initial_motor'],
           'execution_source': ('user-explicit --mcu-execution on a spec that declares no MCU chain'
                                if forced_chain else 'mcu_execution.profile'),
           'plant_source': 'motion_timing.configuration armature/J only'}
    if not getattr(args, 'zero_latency', False):
        return cfg
    # Explicit diagnostic: same PD/limits/spring, no transport/response/age anywhere.
    cfg.update(zero_latency=True, declared=played, transport_ms=[0] * 6, feedback_ms=0, tau_ms=[0] * 6,
               actor_sensor_ms=0, local_target_transport_ms=0,
               profile=dict(cfg['profile'], motor_delay_ms=[0] * 6, motor_tau_ms=[0] * 6, PD_age_ms=0,
                            actor_age_ms=0, local_target_delay_ms=0))
    return cfg


def pose_joint_positions(model, refs, execution):
    """Declared joint pose in native model order, plus every other declared joint the asset has.

    The four closed-leg loop joints must be written too, otherwise the declared standing
    posture is not reproduced and the closed-chain check below reports a large error.
    A missing ordered joint is an error, never silently zero-filled.
    """
    import mujoco
    declared_pose = execution['pose']['joint_pos']
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in refs['joints']]
    missing = [n for n in names if n not in declared_pose]
    if missing:
        raise ValueError('declared MCU pose is missing joints: ' + ','.join(missing))
    return np.array([declared_pose[n] for n in names], dtype=float)


def apply_declared_joints(model, data, refs, execution):
    """Write the ordered joints plus every other declared joint this asset has, by name."""
    import mujoco
    ordered = pose_joint_positions(model, refs, execution)
    written = {}
    for j in range(model.njnt):
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in execution['pose']['joint_pos']:
            value = float(execution['pose']['joint_pos'][name])
            data.qpos[int(model.jnt_qposadr[j])] = value
            written[name] = value
    data.qpos[refs['qpos']] = ordered
    return ordered, written


def apply_initial_pose(model, data, refs, execution):
    """Declared MCU initial state: root height/quaternion, joint pose, zero velocities."""
    import mujoco
    root = int(refs['root'])
    free = [i for i in range(model.njnt) if int(model.jnt_bodyid[i]) == root
            and model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise ValueError('exactly one root free joint required for the declared MCU pose')
    address = int(model.jnt_qposadr[free[0]])
    pose = execution['pose']
    data.qpos[address:address + 3] = [0., 0., float(pose.get('height_m', .34))]
    x, y, z, w = [float(v) for v in pose['root_quaternion_xyzw']]
    data.qpos[address + 3:address + 7] = [w, x, y, z]
    ordered, written = apply_declared_joints(model, data, refs, execution)
    data.qvel[:] = 0.
    return ordered


class MCUPlayback:
    """One declared MCU chain for a single preview plant.

    ``step`` is the only clock: it must be called once per 1 ms physics step with
    a contiguous tick. PD/adoption run every 5 ms, motor transport and response
    every step, so no caller may hide the 1 ms part inside a 5 ms branch.
    """

    def __init__(self, execution, initial_q6, initial_target6, tick=0, leg_response=None):
        self.execution = execution
        self.profile = execution['profile']
        # Optional delivered bilateral leg response; wheels and PD stay the spec's own chain.
        self.leg_response = leg_response
        self.reset(tick, initial_q6, initial_target6)

    def reset(self, tick, initial_q6, initial_target6):
        p = self.profile
        q6 = np.asarray(initial_q6, dtype=np.float32)
        target = np.asarray(initial_target6, dtype=np.float32)
        compensation, length, spread, valid = spring_compensation(torch.tensor(q6[LEGS])[None])
        self.compensation = compensation[0].clone()
        self.length = length[0].clone()
        self.spread = spread[0].clone()
        self.geometry_valid = bool(valid.all())
        # Declared initial motor state: the current spring compensation, not zero.
        self.ring = MotorRing(1, 'cpu')
        self.ring.delay[0] = torch.tensor([int(v) for v in p['motor_delay_ms']], dtype=torch.long)
        self.ring.tau[0] = torch.tensor([float(v) for v in p['motor_tau_ms']], dtype=torch.float32)
        self.ring.reset(torch.arange(1), self.compensation[None])
        # With the delivered response the leg channels run in modal coordinates, so the
        # joint-wise ring must not add a second delay/response on top of them.
        self.modal = None
        if self.leg_response is not None:
            self.modal = ModalLegRing(self.leg_response, q6.astype(float))
            self.ring.delay[0][LEGS] = 0
            self.ring.tau[0][LEGS] = 0
        self.previous = torch.tensor(target)
        self.adopted = self.previous.clone()
        self.pending = self.previous.clone()
        # Local target transport: the PD adopts the target published this many ms ago.
        # Zero for the post-PD default chain; the standing-delay chain declares 5 ms.
        self.local_delay_ms = int(p.get('local_target_delay_ms', 0))
        self.published = np.tile(self.pending.numpy(), (64, 1)).astype(np.float32)
        self.driver = self.compensation.clone()
        self.motor_actual = self.compensation.clone()
        self.spring_actual = -self.compensation.clone()
        self.pre = torch.zeros(6)
        self.pd_limited = torch.zeros(6)
        # Sum of post-PD transport/response motor torque and the independent spring reaction.
        self.actual = torch.zeros(6)
        # Pre-history row sits one PD age before the first tick, so the first feedback sample
        # clamps to the declared initial state instead of silently using the current one.
        self.history = ObservationHistory(np.r_[q6, np.zeros(6)], initial_tick=int(tick) - int(p['PD_age_ms']))

    def step(self, tick, target, q, dq):
        p = self.profile
        if tick % int(p['policy_period_ms']) == 0:
            self.pending = torch.tensor(np.asarray(target, dtype=np.float32))
            self.published[tick % 64] = self.pending.numpy()
        self.history.append(tick, np.r_[np.asarray(q, dtype=float), np.asarray(dq, dtype=float)])
        if tick % int(p['pd_period_ms']) == 0:
            # Local adoption is separate from any remote delivery; previous is this adopted target.
            self.adopted = (self.pending.clone() if self.local_delay_ms == 0 else
                            torch.tensor(self.published[(tick - self.local_delay_ms) % 64].copy()))
            self.previous = self.adopted.clone()
            feedback = self.history.at(tick - int(p['PD_age_ms']))
            request, limited, compensation, motor, valid = pd_and_motor(
                torch.tensor(feedback[:6], dtype=torch.float32)[None],
                torch.tensor(feedback[6:], dtype=torch.float32)[None], self.adopted[None])
            self.pre = request[0].clone()
            self.pd_limited = limited[0].clone()
            self.compensation = compensation[0].clone()
            self.driver = motor[0].clone()
            self.geometry_valid = bool(valid.all())
        self.motor_actual = self.ring.step(tick, self.driver[None])[0].clone()
        if self.modal is not None:
            self.motor_actual = torch.tensor(self.modal.step(tick, self.motor_actual.numpy()),
                                             dtype=self.motor_actual.dtype)
        spring, length, spread, valid = spring_compensation(
            torch.tensor(np.asarray(q, dtype=np.float32)[LEGS])[None])
        self.spring_actual = -spring[0].clone()
        self.length = length[0].clone()
        self.spread = spread[0].clone()
        self.geometry_valid = self.geometry_valid and bool(valid.all())
        self.actual = self.motor_actual + self.spring_actual
        return self.actual

    def summary(self):
        return summary(self.execution['version'], self.profile, self.geometry_valid, self.leg_response)


def summary(version, profile_, geometry_valid=None, leg_response=None):
    """Effective parameters of one chain, for the manifest and the panel."""
    out = {'version': version, 'transport_placement': 'post_PD_motor_torque_command',
           'motor_delay_ms_ch0_ch5': [int(v) for v in profile_['motor_delay_ms']],
           'motor_tau_ms_ch0_ch5': [int(v) for v in profile_['motor_tau_ms']],
           'PD_age_ms': int(profile_['PD_age_ms']), 'actor_age_ms': int(profile_['actor_age_ms']),
           'local_target_delay_ms': int(profile_['local_target_delay_ms']),
           'pd_limits_nm': list(profile_['pd_limits_nm']),
           'driver_limits_nm': list(profile_['driver_limits_nm']),
           'solver_limits_nm': list(profile_['solver_limits_nm']),
           'initial_motor': profile_['initial_motor'], 'spring': profile_['spring'],
           'geometry_valid': geometry_valid}
    if leg_response is not None:
        out['leg_response'] = leg_response
        out['leg_transport_placement'] = ('legs: post-PD, per-side modal '
                                          '(common,differential) transport/response; wheels: spec chain')
    return out
