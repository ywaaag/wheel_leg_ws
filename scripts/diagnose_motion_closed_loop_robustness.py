"""Bounded CPU closed-loop MCU sensitivity and continuous policy handover.

Plan by default. --run is physical MuJoCo sampling, PPO/optimizer updates are zero.
The existing MCU scheduler, physical decoder and pure compensation are reused.
No LQR controller or measured hardware response is claimed.
"""
import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from native_stand.students.data import sha
from native_stand.students.motion_execution_uncertainty import nominal, screen, combine, validate
from native_stand.students.motion_mcu_execution import LEGS, WHEELS, pd_and_motor, spring_compensation
from native_stand.students.real2sim_delay import ObservationHistory
from motion_mcu_playback import MCUPlayback, configuration, declared, apply_initial_pose

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/2026_10_25-23_52__motion1500_posture_retention312_10000/formal/model_update_4500.pt'
SOURCE_SHA = '0be977b308cfe493fdddee54823e21c713508fa582ce1618ab34e57b13fbc619'
SMALL = ROOT / 'deployment_protocol/2026_10_23-19_53__motion500_currentplant_small31_onnx/actor.onnx'
SMALL_SHA = '5b67781bdaa86b666683e315f685a0f19c45fee918da70439ea3bb43d0fbb03a'
SPEC = ROOT / 'configs/experiments/2026_10_25-23_52__motion1500_posture_retention312_10000/spec.json'
XML = Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml')


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def calibration_biases(uncertainty, calibration=None):
    """Explicit diagnostic split; the existing common-bias path stays unchanged."""
    common = np.asarray(uncertainty['leg_zero_bias_rad'], dtype=np.float32)
    if calibration is None:
        return common.copy(), common.copy()
    expected = {'version', 'actor_leg_zero_bias_rad', 'pd_leg_zero_bias_rad'}
    if set(calibration) != expected or calibration['version'] != 'motion31_split_encoder_offsets_v1':
        raise ValueError('explicit split encoder calibration version/fields required')
    if common.any():
        raise ValueError('common and split calibration cannot be combined')
    result = []
    for name in ('actor_leg_zero_bias_rad', 'pd_leg_zero_bias_rad'):
        value = np.asarray(calibration[name], dtype=np.float32)
        if value.shape != (4,) or not np.isfinite(value).all() or (np.abs(value) > .2).any():
            raise ValueError('four finite diagnostic encoder offsets within .2rad required')
        result.append(value.copy())
    return tuple(result)


class UncertainExecution(MCUPlayback):
    """Reported calibration affects Actor/PD consistently; spring uses true q.

    Commands and previous stay in the reported model coordinates, just like a
    controller whose encoder calibration is offset. PD target-q therefore uses
    one coordinate system. The true plant/mechanical limits remain unchanged.
    """
    def __init__(self, execution, true_q, initial_target, uncertainty, *, calibration=None):
        self.uncertainty = validate(copy.deepcopy(uncertainty))
        p = copy.deepcopy(execution)
        for key in ('actor_age_ms', 'PD_age_ms', 'motor_delay_ms', 'motor_tau_ms'):
            p['profile'][key] = uncertainty[key]
        self.bias = np.zeros(6, np.float32)
        _, self.bias[LEGS] = calibration_biases(uncertainty, calibration)
        super().__init__(p, np.asarray(true_q) + self.bias, initial_target)
        self.driver_cap = torch.tensor(p['profile']['driver_limits_nm'], dtype=torch.float32)
        self.driver_cap[WHEELS] *= uncertainty['wheel_cap_scale']

    def step(self, tick, target, q, dq):
        p = self.profile
        if tick % 10 == 0:
            self.pending = torch.tensor(np.asarray(target, dtype=np.float32))
        reported_q = np.asarray(q, dtype=float) + self.bias
        self.history.append(tick, np.r_[reported_q, np.asarray(dq, dtype=float)])
        if tick % 5 == 0:
            self.adopted = self.pending.clone()
            self.previous = self.adopted.clone()
            feedback = self.history.at(tick - int(p['PD_age_ms']))
            request, limited, comp, motor, valid = pd_and_motor(
                torch.tensor(feedback[:6], dtype=torch.float32)[None],
                torch.tensor(feedback[6:], dtype=torch.float32)[None], self.adopted[None])
            self.pre, self.pd_limited, self.compensation = request[0], limited[0], comp[0]
            self.driver = motor[0].clamp(-self.driver_cap, self.driver_cap)
            self.geometry_valid = bool(valid.all())
        self.motor_actual = self.ring.step(tick, self.driver[None])[0].clone()
        spring, length, spread, valid = spring_compensation(
            torch.tensor(np.asarray(q, dtype=np.float32)[LEGS])[None])
        self.spring_actual = -spring[0] * self.uncertainty['spring_scale']
        self.length, self.spread = length[0], spread[0]
        self.geometry_valid = self.geometry_valid and bool(valid.all())
        self.actual = self.motor_actual + self.spring_actual
        return self.actual


class OnnxMean:
    def __init__(self, path):
        import onnx
        from onnx.reference import ReferenceEvaluator
        graph = onnx.load(str(path))
        self.name = graph.graph.input[0].name
        self.session = ReferenceEvaluator(graph)

    def mean(self, obs):
        return torch.from_numpy(self.session.run(None, {self.name: obs.numpy().astype(np.float32)})[0])


def metrics(arrays, start, stop, origin):
    mask = (arrays['time'] > origin + start + 1e-8) & (arrays['time'] <= origin + stop + 1e-8)
    count = int(mask.sum())
    result = dict(samples=count, required_samples=round((stop - start) * 100),
                  complete=count == round((stop - start) * 100))
    if not count:
        return result
    state, dq = arrays['state'][mask], arrays['dq'][mask]
    request = arrays['requested_target'][mask]
    motor = arrays['motor'][mask]
    from diagnose_motion_mcu_gap import spectrum
    result.update(height_mean_m=float(state[:, 3].mean()), vx_mean_mps=float(state[:, 0].mean()),
        vx_abs_mean_mps=float(np.abs(state[:, 0]).mean()),
        pitch_peak_rad=float(np.abs(state[:, 4]).max()), roll_peak_rad=float(np.abs(state[:, 5]).max()),
        dq_abs_p95_rad_s=np.quantile(np.abs(dq), .95, axis=0).tolist(),
        motor_abs_peak_nm=np.abs(motor).max(0).tolist(),
        wheel_target_step_rms_rad_s=(np.sqrt(np.mean(np.diff(request[:, WHEELS], axis=0)**2, axis=0)).tolist() if count > 1 else None),
        both_contact_fraction=float((arrays['contact'][mask] > 2).all(-1).mean()),
        leg_length_difference_MAE_m=float(np.abs(arrays['leg_difference'][mask]).mean()),
        driver_limit_fraction=(np.abs(arrays['driver'][mask]) >= arrays['driver_cap'][mask] - 1e-4).mean(0).tolist(),
        spectra={key: spectrum(state[:, col], .01) for key, col in [('vx', 0), ('pitch', 4), ('roll', 5)]}
            if count >= 16 else {})
    return result


def run_case(model, refs, actors, cfg, uncertainty, controller, seconds, deadline, *, predecessor=None, warmup=0., target=None, initial_yaw=0., constraint_observer=None, calibration=None, command_age_from_handover=False, initial_velocity_w=None):
    import mujoco
    from diagnose_motion_real2sim import sensors
    from validate_native_mujoco import state
    from native_stand.students.mcu_interface import encode
    from native_stand.students.interface import targets

    data = mujoco.MjData(model)
    q0 = apply_initial_pose(model, data, refs, cfg['mcu_execution'])
    if initial_yaw:
        from scipy.spatial.transform import Rotation
        initial = Rotation.from_quat(data.qpos[3:7][[1, 2, 3, 0]])
        rotated = (Rotation.from_rotvec([0., 0., initial_yaw]) * initial).as_quat()
        data.qpos[3:7] = rotated[[3, 0, 1, 2]]
    free = next(j for j in range(model.njnt) if model.jnt_bodyid[j] == refs['root'] and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    dof = int(model.jnt_dofadr[free])
    mujoco.mj_forward(model, data)
    rotation = data.xmat[refs['root']].reshape(3, 3)
    data.qvel[dof:dof + 3] = rotation @ np.array([0., 0., -uncertainty['initial_vx_mps']])
    data.qvel[dof + 3:dof + 6] = [uncertainty['initial_pitch_rate_rad_s'], 0., 0.]
    if initial_velocity_w is not None:
        from native_stand.students.motion_cross_engine import free_velocity
        data.qvel[dof:dof+6] = free_velocity(initial_velocity_w,rotation)
    mujoco.mj_forward(model, data)
    actor_bias = np.zeros(6, np.float32); bias = np.zeros(6, np.float32)
    actor_bias[LEGS], bias[LEGS] = calibration_biases(uncertainty, calibration)
    requested = np.zeros(6, np.float32); requested[LEGS] = q0[LEGS] + bias[LEGS]
    execution = UncertainExecution(cfg['mcu_execution'], q0, requested, uncertainty, calibration=calibration)
    history = ObservationHistory(sensors(model, data, refs), initial_tick=0)
    bounds = torch.tensor(model.jnt_range[refs['joints']], dtype=torch.float32)[None]
    for col, joint in enumerate(refs['joints']):
        if not model.jnt_limited[joint]:
            bounds[0, col] = torch.tensor([-float('inf'), float('inf')])
    bounds += torch.tensor(bias)[None, :, None]
    reference = torch.zeros(1, 6)
    rows, handover = [], None
    termination = None
    event = round(warmup * 1000)
    for tick in range(round((seconds + warmup) * 1000)):
        if tick % 100 == 0 and time.monotonic() > deadline:
            raise TimeoutError('closed-loop matrix wall limit')
        current = sensors(model, data, refs)
        if not np.isfinite(current).all():
            raise RuntimeError('non-finite physical sensors')
        history.append(tick, current)
        if tick % 10 == 0:
            sensed = history.at(tick - uncertainty['actor_age_ms'])
            reported = sensed[6:12] + actor_bias
            previous = execution.previous.numpy().copy()
            command = torch.tensor([[0., 0., .34]])
            if target is not None:
                from native_stand.command_v6.domain.command_schedule import effective
                command = effective(torch.tensor([target[:2]], dtype=torch.float32),
                    torch.tensor([max(0., tick * .001-warmup) if command_age_from_handover else tick*.001]), height=target[2], entry=2., ramp=8., hold=8., exit=8.)
            obs = encode(torch.tensor(sensed[:3], dtype=torch.float32)[None],
                torch.tensor(sensed[3:6], dtype=torch.float32)[None],
                torch.tensor(reported, dtype=torch.float32)[None],
                torch.tensor(sensed[12:18], dtype=torch.float32)[None],
                torch.tensor(previous[LEGS])[None], torch.tensor(previous[WHEELS])[None],
                reference, 'Motion', command)
            owner = predecessor if tick < event else controller
            mean = actors[owner].mean(obs)
            if not torch.isfinite(mean).all():
                raise RuntimeError('non-finite Actor output')
            decoded = targets(mean, reference, bounds)
            requested[LEGS] = decoded['leg_executed'][0].numpy()
            requested[WHEELS] = decoded['wheel_executed'][0].numpy()
            if tick == event and predecessor:
                handover = dict(tick=tick, predecessor=predecessor, successor=controller,
                    previous_adopted_target=previous.tolist(), first_request=requested.tolist(),
                    target_jump=(requested - previous).tolist(), motor_actual=execution.motor_actual.tolist(),
                    queue_sha256=__import__('hashlib').sha256(execution.ring.history.numpy().tobytes()).hexdigest(),
                    physical_qpos=data.qpos.tolist(), physical_qvel=data.qvel.tolist(),
                    observation=obs[0].tolist(), mean=mean[0].tolist(),
                    physical_state_reset=False, cache_reset=False, queue_reset=False,
                    predecessor_is_hardware_LQR=False)
        pre_q, pre_dq = data.qpos[refs['qpos']].copy(), data.qvel[refs['dofs']].copy()
        execution.step(tick, requested, pre_q, pre_dq)
        data.ctrl[refs['actuators']] = execution.actual.numpy() / refs['gears']
        mujoco.mj_step(model, data)
        if tick % 10 == 9:
            _, _, st = state(model, data, refs)
            rows.append(dict(time=(tick + 1) / 1000., owner=0 if tick < event else 1,
                actor_input=obs[0].numpy().copy(), mean=mean[0].numpy().copy(),
                effective_command=command[0].numpy().copy(),
                state=np.array([st[k] for k in ('vx', 'vy', 'yaw_rate', 'height', 'pitch', 'roll', 'closure_m')]),
                q=data.qpos[refs['qpos']].copy(), dq=data.qvel[refs['dofs']].copy(),
                requested_target=requested.copy(), adopted_target=execution.adopted.numpy().copy(),
                pd_request=execution.pre.numpy().copy(), driver=execution.driver.numpy().copy(),
                driver_cap=execution.driver_cap.numpy().copy(), motor=execution.motor_actual.numpy().copy(),
                spring=execution.spring_actual.numpy().copy(), contact=np.asarray(st['contact_force_n']),
                leg_difference=st['posture_geometry']['leg_length_difference_m'],
                root_xy=np.asarray(st['root_xy'])))
            if calibration is not None:
                pd_tick = (tick // 5) * 5
                feedback = execution.history.at(pd_tick - int(execution.profile['PD_age_ms']))
                rows[-1].update(actor_true_q=sensed[6:12].copy(), actor_true_dq=sensed[12:18].copy(),
                    pd_feedback_q=feedback[:6].copy(), pd_feedback_dq=feedback[6:].copy(),
                    pd_true_q=feedback[:6].copy()-bias, pd_feedback_tick=pd_tick-int(execution.profile['PD_age_ms']),
                    pd_limited=execution.pd_limited.numpy().copy(),
                    spring_compensation=execution.compensation.numpy().copy(),
                    physical_equivalent_adopted=execution.adopted.numpy().copy()-bias)
            if constraint_observer is not None:
                rows[-1].update(constraint_observer(model, data, refs))
            if not execution.geometry_valid:
                raise RuntimeError('invalid spring geometry')
            if st['closure_m'] > .005:
                fault=RuntimeError('closed-chain geometry fault >5mm')
                fault.physical_seconds=float(data.time);fault.physics_tick=tick
                raise fault
            if st['up'] < .5 or st['height'] < .18:
                termination = 'FALL'
                break
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    summary = dict(controller=controller, predecessor=predecessor, uncertainty=uncertainty,
        termination=termination or 'WINDOW_COMPLETE', physical_seconds=float(data.time),
        requested_controlled_seconds=seconds, warmup_seconds=warmup, capability_accepted=None,
        handover=handover, windows={name: metrics(arrays, lo, hi, warmup)
            for name, lo, hi in [('first_0p5', 0., .5), ('catch_0p5_to_2', .5, 2.), ('steady_2_to_30', 2., seconds)]})
    if target is not None:
        summary.update(target=list(target), initial_yaw_rad=initial_yaw,
            schedule=dict(entry=2., ramp=8., hold=8., exit=8.))
    if calibration is not None:
        summary['calibration'] = copy.deepcopy(calibration)
    return summary, arrays


def vulnerability(row):
    steady = row['windows']['steady_2_to_30']
    if not steady.get('samples'):
        return [1., -row['physical_seconds'], 0., 0.]
    return [float(row['termination'] != 'WINDOW_COMPLETE'), -row['physical_seconds'],
            steady['spectra'].get('pitch', {}).get('detrended_rms', 0.), steady['vx_abs_mean_mps']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--wall-seconds', type=float, default=2400.)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not 2.5 <= args.seconds <= 30 or not 60 <= args.wall_seconds <= 3600:
        parser.error('seconds2.5..30 and wall60..3600 required')
    for path, expected in ((SOURCE, SOURCE_SHA), (SMALL, SMALL_SHA)):
        if sha(path) != expected:
            raise ValueError('fixed model identity differs: ' + str(path))
    from native_stand.students.model import load
    # Check the complete source evaluation, including each raw JSON and all envs.
    registry = SOURCE.parent / 'candidate_registry.json'
    candidate = next(r for r in json.loads(registry.read_text()) if r['update'] == 4500)
    if candidate['sha256'] != SOURCE_SHA or candidate.get('complete_matrix') is not True or len(candidate['per_env']) != 192 or not all(e['accepted'] is True for e in candidate['per_env']):
        raise ValueError('4500 complete192/192 evidence required')
    for record in candidate['evaluations']:
        if sha(SOURCE.parent / record['path']) != record['sha256']:
            raise ValueError('4500 raw evaluation SHA differs')
    plan = dict(version='motion_closed_loop_sensitivity_v1', authorization='user2026-10-26 all steps approved',
        models={'Motion500_small': {'path': str(SMALL), 'sha256': SMALL_SHA},
                'baseline4500': {'path': str(SOURCE), 'sha256': SOURCE_SHA}},
        source_native_full_matrix=dict(conditions=24, scored_envs=192, passed=192),
        model_input='multiskill_mcu_v2;31D;Motion/NONE;height0.34;zero command',
        base_spec=str(SPEC), base_spec_sha256=sha(SPEC), xml=str(XML), xml_sha256=sha(XML),
        profile_scope='engineering hypotheses; post-PD transport and independent spring; not identified real robot',
        screens=screen(), pair_policy='two most adverse distinct families per model, one pair each',
        handovers=[['baseline4500', 'Motion500_small'], ['baseline4500', 'baseline4500'],
                   ['Motion500_small', 'Motion500_small'], ['Motion500_small', 'baseline4500']],
        limits=dict(cases=54, seconds_per_case=args.seconds, handover_warmup_s=5,
                    wall_seconds=args.wall_seconds, storage_bytes=256 * 2**20),
        online_PPO_updates=0, offline_optimizer_steps=0, capability_certificate=False)
    if not args.run:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write(args.output_dir / 'manifest.json', plan)
    import mujoco
    from validate_native_mujoco import binding
    from compare_motion_timing_profiles import apply_actuator_profile
    torch.set_num_threads(1)
    spec = json.loads(SPEC.read_text())
    cfg = configuration(SimpleNamespace(leg_armature_m=.012, zero_latency=False), declared(spec), spec)
    model = mujoco.MjModel.from_xml_path(str(XML)); refs = binding(model)
    readback = apply_actuator_profile(model, refs, cfg, {key: getattr(model, key).copy()
        for key in ('dof_armature', 'dof_damping', 'dof_frictionloss')})
    cap = np.asarray(cfg['profile']['solver_limits_nm']) / np.abs(refs['gears'])
    model.actuator_ctrlrange[refs['actuators']] = np.stack((-cap, cap), -1)
    model.dof_damping[np.asarray(refs['dofs'])[WHEELS]] = 0
    model.dof_frictionloss[np.asarray(refs['dofs'])[WHEELS]] = 0
    write(args.output_dir / 'plant_readback.json', dict(profile=cfg, readback=readback,
        mechanical_limits=model.jnt_range[refs['joints']].tolist(), solver_ctrlrange=model.actuator_ctrlrange[refs['actuators']].tolist(),
        equality_solref=model.eq_solref.tolist(), equality_solimp=model.eq_solimp.tolist()))
    large, _ = load(SOURCE, 'MotionZeroStudent'); large.eval()
    actors = {'baseline4500': large, 'Motion500_small': OnnxMean(SMALL)}
    start = time.monotonic(); deadline = start + args.wall_seconds
    results, failure, status = [], None, 'FAILED'

    def execute(name, row, actor, predecessor=None):
        result, arrays = run_case(model, refs, actors, cfg, row, actor, args.seconds, deadline,
                                  predecessor=predecessor, warmup=5. if predecessor else 0.)
        np.savez_compressed(args.output_dir / (name + '.npz'), **arrays)
        result.update(case_id=name, trace=name + '.npz', trace_sha256=sha(args.output_dir / (name + '.npz')))
        write(args.output_dir / (name + '.json'), result)
        results.append(result)
        print(json.dumps(dict(case=name, termination=result['termination'], seconds=result['physical_seconds'],
            vx=result['windows']['steady_2_to_30'].get('vx_mean_mps'))), flush=True)
        if sum(p.stat().st_size for p in args.output_dir.iterdir() if p.is_file()) > 256 * 2**20:
            raise RuntimeError('diagnostic output bound reached')
        return result

    try:
        with torch.inference_mode():
            for actor in actors:
                own = [execute(actor + '__' + row['name'], row, actor) for row in screen()]
                families = {}
                for row in own[1:]:
                    family = row['uncertainty']['family']
                    if family not in families or vulnerability(row) > vulnerability(families[family]):
                        families[family] = row
                worst = sorted(families.values(), key=vulnerability, reverse=True)[:2]
                execute(actor + '__paired', combine(*[r['uncertainty'] for r in worst]), actor)
            for predecessor, successor in plan['handovers']:
                execute(predecessor + '__handover_to__' + successor, nominal(), successor, predecessor)
        status = 'COMPLETED'
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write(args.output_dir / 'result.json', dict(status=status, failure=failure, records=results,
            wall_seconds=time.monotonic() - start, physical_seconds=sum(r['physical_seconds'] for r in results),
            actual_cases=len(results), PPO_updates=0, optimizer_steps=0,
            model_unchanged=sha(SOURCE) == SOURCE_SHA and sha(SMALL) == SMALL_SHA))


if __name__ == '__main__':
    main()
