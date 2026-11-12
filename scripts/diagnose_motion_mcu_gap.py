"""Bounded MCU gap diagnosis, not a trainer or a capability certificate.

Default is a zero-write plan. --run reads immutable native evidence and writes
compact comparisons/spectra. --simulate additionally runs the explicitly listed
MuJoCo counterfactuals on CPU. The frozen PD/compensation and playback scheduler
are reused; no training/default physics, old result, or electrical code is edited.
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


LEGS = [0, 1, 3, 4]
WHEELS = [2, 5]  # geometric RIGHT, LEFT, respectively
REPORTED_B = np.array([.00066957, .0025485])
REPORTED_TC = np.array([.0528843, .054699])


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def spectrum(values, dt):
    """Separate DC/linear drift from oscillation; retain sub-2Hz information."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 16 or not np.isfinite(x).all() or dt <= 0:
        raise ValueError('finite contiguous one-dimensional signal required')
    t = np.arange(len(x)) * dt
    slope, intercept = np.polyfit(t, x, 1)
    residual = x - (intercept + slope * t)
    f = np.fft.rfftfreq(len(x), dt)
    power = np.abs(np.fft.rfft(residual * np.hanning(len(x)))) ** 2
    usable = (f > 0) & (f <= min(45., .5 / dt))
    total = float(power[usable].sum())
    peak = int(np.flatnonzero(usable)[np.argmax(power[usable])])
    return dict(mean=float(x.mean()), std=float(x.std()),
                abs_p95=float(np.quantile(np.abs(x), .95)), linear_slope_per_s=float(slope),
                detrended_rms=float(np.sqrt(np.mean(residual ** 2))),
                peak_hz=float(f[peak]) if total > 1e-18 else None,
                resolution_hz=float(1 / (len(x) * dt)),
                band_power_fraction={name: (float(power[(f >= lo) & (f < hi)].sum()) / total
                                           if total > 1e-18 else None)
                                     for name, lo, hi in [('low_0_to_2', 0.01, 2.),
                                                          ('mid_2_to_10', 2., 10.),
                                                          ('high_10_to_45', 10., 45.)]})


def drag(dq, b, tc, epsilon=.1):
    """Reported wheel drag hypothesis. epsilon is a declared smoothing assumption."""
    return np.asarray(b) * np.asarray(dq) + np.asarray(tc) * np.tanh(np.asarray(dq) / epsilon)


def case_key(case):
    return tuple(case['command']) + (int(case.get('_mcu_variant', 0)),)


def checked(path, digest=None):
    p = Path(path).resolve()
    if digest is not None and sha(p) != digest:
        raise ValueError(f'evidence SHA differs: {p}')
    return p


def read_trace(record, base):
    p = Path(record['path'])
    if not p.is_absolute():
        p = Path(base) / p
    checked(p, record['sha256'])
    raw = json.loads(p.read_text())
    derived = raw.get('derived_from', {})
    trace = derived.get('trace') or str(p.parent / raw['trace']['path'])
    trace = checked(trace, derived.get('trace_sha256') or raw['trace']['sha256'])
    return raw, trace


def compact_row(record):
    envs = record['per_env']
    def values(section, field, maximum=False):
        data = [e.get(section, {}).get(field) for e in envs]
        if any(v is None for v in data): return None
        return float(max(data) if maximum else np.mean(data))
    return dict(command=record['case']['command'], variant=record['case'].get('_mcu_variant', 0),
                passed=sum(e['accepted'] is True for e in envs), envs=len(envs),
                response=values('tracking', 'mean_signed_speed'), MAE=values('tracking', 'MAE'),
                pitch_hold_peak=values('tracking', 'pitch_hold_abs_max_rad', True),
                roll_hold_peak=values('tracking', 'roll_hold_abs_max_rad', True),
                leg_hold_MAE_m=values('reacquisition_geometry', 'leg_length_mae_m'),
                wheel_hold_MAE_m=values('reacquisition_geometry', 'wheel_offset_mae_m'),
                failed_gates=sorted({k for e in envs for k in e['failed_checks']}))


def native_review(args, out):
    assessment = json.loads(args.source_assessment.read_text())
    terminal = json.loads((args.current_run / 'formal/terminal_report.json').read_text())
    if terminal['status'] not in ('STOPPED_BOUNDED', 'COMPLETED') or terminal['failure'] is not None:
        raise ValueError('current training must have settled normally before this diagnosis')
    candidates = terminal['evaluations']
    if {r['case']['sha256'] for r in candidates} != {sha(args.candidate_checkpoint)}:
        raise ValueError('candidate checkpoint does not own the complete native matrix')
    sources = {case_key(r['case']): r for r in assessment['records']}
    rows, spectra, inputs = [], [], {}
    selected = {(0., 0., .34, 0), (3., 0., .34, 0), (-3., 0., .34, 0),
                (4., 0., .34, 0), (0., 4., .34, 0), (0., -4., .34, 0), (0., 0., .34, 2)}
    for candidate in candidates:
        key = case_key(candidate['case'])
        source = sources[key]
        for k in ('seed', 'family', 'yaw_rad', 'seconds'):
            if source['case'][k] != candidate['case'][k]:
                raise ValueError(f'paired case {k} differs')
        rows.append(dict(source=compact_row(source), candidate=compact_row(candidate)))
        for label, record, base in [('original1500', source, args.source_assessment.parent),
                                    ('current500', candidate, args.current_run / 'formal')]:
            raw, path = read_trace(record, base)
            inputs[str(path)] = sha(path)
            if key not in selected:
                continue
            with np.load(path, allow_pickle=False) as z:
                times = z['policy_time_s']
                mask = (times >= 10.) & (times < 18.)
                valid = z['first_episode_valid'][mask]
                if mask.sum() != 800 or not np.allclose(np.diff(times[mask]), .01, atol=2e-6):
                    raise ValueError('hold spectrum requires contiguous 10..18s at100Hz')
                envrows = []
                for env in range(valid.shape[1]):
                    if not valid[:, env].all():
                        envrows.append(dict(env=env, complete=False, reason='first episode ends within window'))
                        continue
                    st, posture = z['state'][mask, env], z['posture'][mask, env]
                    wheel_actual = z['wheel_actual_joint_velocity_rad_s'][mask, env] if 'wheel_actual_joint_velocity_rad_s' in z else z['control'][mask, env, 6:12][:, WHEELS]
                    wheel_target = z['wheel_target_velocity_rad_s'][mask, env] if 'wheel_target_velocity_rad_s' in z else z['control'][mask, env, 18:24][:, WHEELS] * 100.
                    signals = dict(vx_mps=st[:, 7], yaw_rad_s=st[:, 9], pitch_rad=posture[:, 0],
                                   roll_rad=posture[:, 1],
                                   wheel_actual_cm_rad_s=wheel_actual.mean(-1),
                                   wheel_actual_dm_rad_s=(wheel_actual[:, 1] - wheel_actual[:, 0]) * .5,
                                   wheel_target_cm_rad_s=wheel_target.mean(-1),
                                   wheel_target_dm_rad_s=(wheel_target[:, 1] - wheel_target[:, 0]) * .5)
                    if 'mcu_execution' in z:
                        mcu = z['mcu_execution'][mask, env]
                        signals.update(motor_ch2_Nm=mcu[:, 2], motor_ch5_Nm=mcu[:, 5],
                                       driver_ch2_Nm=mcu[:, 8], driver_ch5_Nm=mcu[:, 11])
                    envrows.append(dict(env=env, complete=True,
                                        signals={k: spectrum(v, .01) for k, v in signals.items()}))
                spectra.append(dict(model=label, command=record['case']['command'],
                                    variant=record['case'].get('_mcu_variant', 0), trace=str(path),
                                    window_s=[10., 18.], sample_rate_hz=100, envs=envrows))
    result = dict(version='motion_mcu_gap_review_v1', rows=rows, spectra=spectra, input_trace_sha256=inputs,
                  spectrum_scope='same fixed window; DC and drift removed only for PSD, raw means retained; not a causal test')
    write(out / 'native_review.json', result)
    return result


def friction_profiles():
    return [dict(name='nominal', drag=False, wheel_feedforward=False, spring_scale=1.),
            dict(name='reported_drag', drag=True, wheel_feedforward=False, spring_scale=1.),
            dict(name='reported_drag_compensated', drag=True, wheel_feedforward=True, spring_scale=1.),
            dict(name='spring110pct', drag=False, wheel_feedforward=False, spring_scale=1.1),
            dict(name='spring90pct', drag=False, wheel_feedforward=False, spring_scale=.9)]


def timing_review(spec):
    from native_stand.students.motion_mcu_execution import MotorRing
    p = spec['mcu_execution']['profile']
    ring = MotorRing(1, 'cpu')
    ring.reset(torch.arange(1), torch.zeros(1, 6))
    response = np.array([ring.step(t, torch.ones(1, 6)).numpy()[0].copy() for t in range(40)])
    onset = [int(np.flatnonzero(response[:, ch] > 1e-6)[0]) for ch in range(6)]
    t63 = [int(np.flatnonzero(response[:, ch] >= 1 - np.exp(-1) - 1e-7)[0]) for ch in range(6)]
    return dict(profile=p, zero_to_one_step_onset_ms=onset, step_63pct_ms=t63,
                PD_feedback='parallel feedback path sampled at PD tick-10ms; not an additional serial10ms behind Actor age',
                Actor_packet='physical first16 dims from previous100Hz packet; previous and commands are current',
                previous='local5ms PD adopted target, before motor transport/response; never measured wheel speed',
                first_state='ring prefilled with spring compensation; initial velocities zero; not measured LQR handover',
                timing_status='all delays/response split are hypotheses; 7/11ms is reported merged unloaded onset, not separately measured transport',
                compensation_status='physical spring is negative C compensation at current geometry; control compensation uses delayed PD geometry',
                missing='confirmed flashed build, timestamped actual PD/motor/feedback chain and measured spring residual',
                no_double_count='Actor and PD sensor paths are parallel; tau is first-order response, not another pure dead time',
                test_scope='CPU MotorRing unit step tick at start of each1ms physics interval; 63pct float32 tolerance1e-7, not measured total robot latency')


def simulation_case(model, refs, actor, cfg, experiment, target, seconds, deadline):
    import mujoco
    from diagnose_motion_real2sim import sensors
    from motion_mcu_playback import MCUPlayback, apply_initial_pose
    from native_stand.students.real2sim_delay import ObservationHistory
    from native_stand.students.mcu_interface import encode
    from native_stand.students.interface import targets
    from native_stand.command_v6.domain.command_schedule import effective
    from validate_native_mujoco import state

    d = mujoco.MjData(model)
    q0 = apply_initial_pose(model, d, refs, cfg['mcu_execution'])
    mujoco.mj_forward(model, d)
    demand = np.zeros(6, np.float32); demand[LEGS] = q0[LEGS]
    execution = MCUPlayback(cfg['mcu_execution'], q0, demand)
    history = ObservationHistory(sensors(model, d, refs), initial_tick=0)
    # Only this diagnostic instance changes the motor command before transport.
    # Reuse the base scheduler/PD/compensation, with one optional wheel FF addition
    # between PD clipping and final driver clipping. No global monkeypatch.
    base_step = execution.ring.step
    feedforward = np.zeros(2)

    def motor_step(tick, command):
        nonlocal feedforward
        if experiment['wheel_feedforward'] and tick % 5 == 0:
            feedback = execution.history.at(tick - cfg['profile']['PD_age_ms'])
            feedforward = drag(feedback[6:][WHEELS], REPORTED_B, REPORTED_TC)
            added = execution.pd_limited + execution.compensation
            added[WHEELS] += torch.as_tensor(feedforward, dtype=added.dtype)
            cap = added.new_tensor(cfg['profile']['driver_limits_nm'])
            execution.driver = added.clamp(-cap, cap)
            command = execution.driver[None]
        return base_step(tick, command)

    execution.ring.step = motor_step
    bounds = torch.tensor(model.jnt_range[refs['joints']], dtype=torch.float32)[None]
    for col, joint in enumerate(refs['joints']):
        if not model.jnt_limited[joint]: bounds[0, col] = torch.tensor([-float('inf'), float('inf')])
    reference = torch.zeros(1, 6)
    rows, tail = [], []
    fail = None
    for tick in range(round(seconds * 1000)):
        if time.monotonic() > deadline: raise TimeoutError('bounded diagnosis deadline')
        current = sensors(model, d, refs)
        if not np.isfinite(current).all(): fail = 'NONFINITE'; break
        history.append(tick, current)
        if tick % 10 == 0:
            sensed = history.at(tick - cfg['profile']['actor_age_ms'])
            command = effective(torch.tensor([target]), torch.tensor([tick / 1000.]),
                                height=.34, entry=2, ramp=8, hold=8, exit=8)
            obs = encode(torch.tensor(sensed[:3], dtype=torch.float32)[None],
                         torch.tensor(sensed[3:6], dtype=torch.float32)[None],
                         torch.tensor(sensed[6:12], dtype=torch.float32)[None],
                         torch.tensor(sensed[12:18], dtype=torch.float32)[None],
                         execution.previous[LEGS][None], execution.previous[WHEELS][None],
                         reference, 'Motion', command)
            raw = actor.mean(obs)
            if not torch.isfinite(raw).all(): fail = 'NONFINITE_ACTION'; break
            result = targets(raw, reference, bounds)
            demand[LEGS] = result['leg_executed'][0].numpy()
            demand[WHEELS] = result['wheel_executed'][0].numpy()
        execution.step(tick, demand, d.qpos[refs['qpos']], d.qvel[refs['dofs']])
        # Mechanical drag is an external joint torque; do not charge it to motor output.
        friction = drag(d.qvel[np.asarray(refs['dofs'])[WHEELS]], REPORTED_B, REPORTED_TC) if experiment['drag'] else np.zeros(2)
        d.qfrc_applied[:] = 0
        d.qfrc_applied[np.asarray(refs['dofs'])[WHEELS]] = -friction
        net = execution.motor_actual + experiment['spring_scale'] * execution.spring_actual
        d.ctrl[refs['actuators']] = net.numpy() / refs['gears']
        mujoco.mj_step(model, d)
        if tick >= round((seconds - 1) * 1000):
            tail.append(np.r_[(tick + 1) / 1000., d.qvel[refs['dofs']],
                              execution.motor_actual.numpy(), execution.driver.numpy(), demand])
        if tick % 10 == 9:
            _, _, s = state(model, d, refs)
            g = s['posture_geometry']
            rows.append(dict(time=(tick + 1) / 1000., state=np.array([s[k] for k in ['vx', 'vy', 'yaw_rate', 'height', 'pitch', 'roll', 'closure_m']]),
                             contact=np.asarray(s['contact_force_n']), q=d.qpos[refs['qpos']].copy(),
                             dq=d.qvel[refs['dofs']].copy(), target=demand.copy(),
                             motor=execution.motor_actual.numpy().copy(), driver=execution.driver.numpy().copy(),
                             spring=experiment['spring_scale'] * execution.spring_actual.numpy().copy(),
                             mechanical_drag=friction.copy(), wheel_feedforward=feedforward.copy(),
                             leg_difference=g['leg_length_difference_m'], wheel_offset=s['wheel_offset_m']))
            if not execution.geometry_valid: fail = 'INVALID_SPRING_GEOMETRY'
            elif s['closure_m'] > .005: fail = 'CLOSURE'
            elif s['up'] < .5 or s['height'] < .18: fail = 'FALL'
            if fail: break
    arrays = {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}
    arrays['tail1kHz'] = np.array(tail)
    summary = dict(termination=fail or 'WINDOW_COMPLETE', simulated_seconds=float(d.time),
                   requested_seconds=seconds, full_diagnostic_window=fail is None,
                   holding_window_s=[10., 18.], capability_accepted=None)
    if rows:
        mask = (arrays['time'] > 10) & (arrays['time'] <= 18 + 1e-8)
        summary['complete_hold'] = bool(mask.sum() == 800)
        if summary['complete_hold']:
            st, target_arr = arrays['state'][mask], arrays['target'][mask]
            summary.update(vx_mean=float(st[:, 0].mean()), yaw_mean=float(st[:, 2].mean()),
                           pitch_peak=float(np.abs(st[:, 4]).max()), roll_peak=float(np.abs(st[:, 5]).max()),
                           both_contact_fraction=float((arrays['contact'][mask] > 2).all(-1).mean()),
                           wheel_target_delta_rms=float(np.sqrt(np.mean(np.diff(target_arr[:, WHEELS], axis=0) ** 2))),
                           leg_target_delta_rms=float(np.sqrt(np.mean(np.diff(target_arr[:, LEGS], axis=0) ** 2))),
                           leg_difference_MAE_m=float(np.abs(arrays['leg_difference'][mask]).mean()),
                           wheel_offset_MAE_m=float(np.abs(arrays['wheel_offset'][mask]).mean()),
                           spectra={k: spectrum(st[:, col], .01) for k, col in [('vx', 0), ('yaw', 2), ('pitch', 4), ('roll', 5)]})
    return summary, arrays


def simulations(args, spec, out):
    import mujoco
    from native_stand.students.model import load
    from motion_mcu_playback import configuration, declared
    from compare_motion_timing_profiles import apply_actuator_profile
    from validate_native_mujoco import binding
    cfg = configuration(SimpleNamespace(leg_armature_m=.012, zero_latency=False), declared(spec), spec)
    model = mujoco.MjModel.from_xml_path(str(args.xml)); refs = binding(model)
    original = {k: getattr(model, k).copy() for k in ('dof_armature', 'dof_damping', 'dof_frictionloss')}
    readback = apply_actuator_profile(model, refs, cfg, original)
    # Both models use the same specified plant and initial pose; old source recipe stays immutable.
    cap = np.asarray(cfg['profile']['solver_limits_nm']) / np.abs(refs['gears'])
    model.actuator_ctrlrange[refs['actuators']] = np.stack((-cap, cap), -1)
    model.dof_damping[np.asarray(refs['dofs'])[WHEELS]] = 0
    model.dof_frictionloss[np.asarray(refs['dofs'])[WHEELS]] = 0
    write(out / 'mujoco_plan.json', dict(profile=cfg, plant_readback=readback,
          active_damping=model.dof_damping[refs['dofs']].tolist(),
          active_frictionloss=model.dof_frictionloss[refs['dofs']].tolist(),
          profiles=friction_profiles(), commands=[[0., 0.], [1., 0.], [-1., 0.], [0., .5], [0., -.5]],
          friction_units='joint-side B Nm*s/rad, Tc Nm; model ch2/ch5=geometric right/left',
          wheel_compensation='counterfactual optional friction FF at delayed PD, before final driver limit; not installed electrical control',
          scope='single env/seed reference-MJCF diagnostic, only18seconds, not30second capability certification',
          limits=dict(cases=50, simulated_seconds=900, wall_seconds=args.wall_seconds, storage_bytes=128 * 2 ** 20),
          source_unconfirmed='B/Tc fitted basis/driver compensation unknown; no inference that this equals real robot'))
    started, results = time.monotonic(), []
    status, failure = 'FAILED', None
    try:
        for name, checkpoint in [('original1500', args.source_checkpoint), ('current500', args.candidate_checkpoint)]:
            actor, _ = load(checkpoint, 'MotionZeroStudent'); actor.eval()
            before = {k: v.clone() for k, v in actor.state_dict().items()}
            for target in [[0., 0.], [1., 0.], [-1., 0.], [0., .5], [0., -.5]]:
                for experiment in friction_profiles():
                    with torch.inference_mode():
                        result, arrays = simulation_case(model, refs, actor, cfg, experiment, target, 18., started + args.wall_seconds)
                    key = f"{name}__vx{target[0]:g}_yaw{target[1]:g}__{experiment['name']}"
                    result.update(model=name, command=target, profile=experiment['name'], trace=key + '.npz')
                    np.savez_compressed(out / result['trace'], **arrays)
                    write(out / (key + '.json'), result); results.append(result)
                    print(json.dumps({k: result[k] for k in ['model', 'command', 'profile', 'termination', 'simulated_seconds']}) , flush=True)
                    if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > 128 * 2 ** 20:
                        raise RuntimeError('diagnostic storage bound reached')
            if not all(torch.equal(before[k], v) for k, v in actor.state_dict().items()):
                raise RuntimeError('read-only actor changed')
        status = 'COMPLETED'
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write(out / 'mujoco_summary.json', dict(status=status, failure=failure, cases=results,
              wall_seconds=time.monotonic() - started, physics_seconds=sum(r['simulated_seconds'] for r in results),
              PPO_updates=0, offline_optimizer_updates=0))
    return results


def alignment_model(xml, *, self_collisions=True):
    """Local compiled model only; remove explicit self pairs as well as mask tests."""
    import mujoco
    if self_collisions:
        return mujoco.MjModel.from_xml_path(str(xml))
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml); element = tree.getroot()
    compiler = element.find('compiler')
    compiler.set('meshdir', str((xml.parent / compiler.get('meshdir', '')).resolve()))
    contact = element.find('contact')
    if contact is not None:
        for pair in list(contact):
            # The reference asset's authored pairs are all robot self contacts.
            if pair.tag != 'pair' or 'ground' in (pair.get('geom1'), pair.get('geom2')):
                raise ValueError('unrecognized explicit contact; do not erase ground interaction')
            contact.remove(pair)
    model = mujoco.MjModel.from_xml_string(ET.tostring(element, encoding='unicode'))
    for i in range(model.ngeom):
        if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
            continue  # visual-only meshes remain visual-only
        world = model.geom_bodyid[i] == 0
        model.geom_contype[i] = 2 if world else 1
        model.geom_conaffinity[i] = 1 if world else 2
    return model


def alignment_parameters(model, refs, variant, native):
    import mujoco
    combined = variant in ('all_listed', 'all_declared_and_solver')
    all_damping = variant == 'all_joint_damping' or combined
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name not in native['joint_damping_by_name']:
            continue
        dof = int(model.jnt_dofadr[j])
        group = native['joint_groups'][name]
        selected = all_damping or ((variant == 'active_damping0' and group == 'leg_motors') or
                    (variant == 'passive_damping01' and group == 'passive_linkage') or
                    (variant == 'guide_damping0' and group == 'guide_wheels'))
        if selected:
            model.dof_damping[dof] = native['joint_damping_by_name'][name]
        if (variant == 'knee_limits_native' or combined) and name.endswith('_knee_pitch'):
            model.jnt_range[j] = native['joint_limits_rad'][name]
    if variant == 'sliding05_hypothesis' or combined:
        # Both surfaces must change because equal-priority MuJoCo uses max friction.
        active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        model.geom_friction[active, 0] = .5
    # Engine solver hypotheses, not identified hardware compliance. Keep the1ms
    # step and use a time constant at least2*dt, preserving MuJoCo refsafe.
    if variant in ('closure_tight', 'closure_contact_tight', 'all_declared_and_solver'):
        model.eq_solref[:] = [.002, 1.]
        model.eq_solimp[:] = [.9999, .9999, .001, .5, 2.]
    if variant in ('contact_tight', 'closure_contact_tight', 'all_declared_and_solver'):
        model.geom_solref[:] = [.002, 1.]
        model.geom_solimp[:] = [.999, .9999, .0001, .5, 2.]
    if variant in ('contact_margin0', 'all_declared_and_solver'):
        model.geom_margin[:] = 0
    return dict(damping={mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j):
                         float(model.dof_damping[model.jnt_dofadr[j]])
                         for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE},
                knees={mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): model.jnt_range[j].tolist()
                       for j in range(model.njnt) if '_knee_' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or '')},
                ground_friction=model.geom_friction[refs['ground']].tolist(),
                ground_condim=int(model.geom_condim[refs['ground']]), explicit_self_pairs=int(model.npair),
                equality_solref=model.eq_solref.tolist(), equality_solimp=model.eq_solimp.tolist(),
                ground_solref=model.geom_solref[refs['ground']].tolist(),
                ground_margin_m=float(model.geom_margin[refs['ground']]))


def alignment_audit(args, spec, out):
    import mujoco
    from pxr import Usd, UsdPhysics
    from motion_mcu_playback import configuration, declared, apply_initial_pose
    from compare_motion_timing_profiles import apply_actuator_profile
    from validate_native_mujoco import binding
    from diagnose_motion_real2sim import sensors
    from native_stand.students.mcu_interface import encode
    from native_stand.students.model import load
    snapshot = json.loads(args.native_readback.read_text())
    if snapshot['status'] != 'PASSED':
        raise ValueError('completed native plant readback required')
    control = snapshot['stages'][0]['control']
    native = {'joint_damping_by_name': {}, 'joint_groups': {}, 'joint_limits_rad': {}}
    for group, row in control['actuators'].items():
        for joint_id, damping in zip(row['joint_ids'], row['damping']):
            name = control['joint_names'][joint_id]
            native['joint_damping_by_name'][name] = damping
            native['joint_groups'][name] = group
    cfg = configuration(SimpleNamespace(leg_armature_m=.012, zero_latency=False), declared(spec), spec)
    model = alignment_model(args.xml); refs = binding(model)
    apply_actuator_profile(model, refs, cfg, {k: getattr(model, k).copy()
                                           for k in ('dof_armature', 'dof_damping', 'dof_frictionloss')})
    stage = Usd.Stage.Open(spec['asset_path'])
    body_rows, joint_rows = [], []
    for prim in stage.Traverse():
        name = prim.GetName()
        if prim.IsA(UsdPhysics.RevoluteJoint) and name in native['joint_damping_by_name']:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            limits = [prim.GetAttribute('physics:' + a + 'Limit').Get() for a in ('lower', 'upper')]
            native['joint_limits_rad'][name] = [None if not np.isfinite(v) else float(np.deg2rad(v)) for v in limits]
            arm = prim.GetAttribute('physxJoint:armature').Get()
            joint_rows.append(dict(name=name, group=native['joint_groups'][name],
                    native_effective_damping=native['joint_damping_by_name'][name],
                    native_authored_armature=None if arm is None else float(arm),
                    native_limits_rad=native['joint_limits_rad'][name],
                    mujoco_damping=float(model.dof_damping[model.jnt_dofadr[j]]),
                    mujoco_armature=float(model.dof_armature[model.jnt_dofadr[j]]),
                    mujoco_limits_rad=model.jnt_range[j].tolist()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if i < 0: continue
            mass = UsdPhysics.MassAPI(prim)
            q = mass.GetPrincipalAxesAttr().Get(); diag = mass.GetDiagonalInertiaAttr().Get()
            if q is None or diag is None: continue
            uq = np.r_[float(q.GetReal()), np.asarray(q.GetImaginary())]; uq /= np.linalg.norm(uq)
            ur = np.empty(9); mr = np.empty(9)
            mujoco.mju_quat2Mat(ur, uq); mujoco.mju_quat2Mat(mr, model.body_iquat[i])
            ui = ur.reshape(3, 3) @ np.diag(np.asarray(diag)) @ ur.reshape(3, 3).T
            mi = mr.reshape(3, 3) @ np.diag(model.body_inertia[i]) @ mr.reshape(3, 3).T
            body_rows.append(dict(name=name, mass_delta_kg=float(model.body_mass[i]-mass.GetMassAttr().Get()),
                                  COM_delta_m=(model.body_ipos[i]-np.asarray(mass.GetCenterOfMassAttr().Get())).tolist(),
                                  inertia_tensor_max_delta_kg_m2=float(np.abs(mi-ui).max())))
    source = json.loads(args.source_assessment.read_text())
    record = next(r for r in source['records'] if case_key(r['case']) == (0., 0., .34, 0))
    _, native_trace = read_trace(record, args.source_assessment.parent)
    with np.load(native_trace, allow_pickle=False) as z:
        native_obs = z['actor_input'][0, 0].copy()
    data = mujoco.MjData(model); q = apply_initial_pose(model, data, refs, cfg['mcu_execution'])
    mujoco.mj_forward(model, data); measured = sensors(model, data, refs)
    obs = encode(torch.tensor(measured[:3], dtype=torch.float32)[None],
                 torch.tensor(measured[3:6], dtype=torch.float32)[None],
                 torch.tensor(measured[6:12], dtype=torch.float32)[None],
                 torch.tensor(measured[12:], dtype=torch.float32)[None],
                 torch.tensor(q[LEGS], dtype=torch.float32)[None], torch.zeros(1, 2),
                 torch.zeros(1, 6), 'Motion', torch.tensor([[0., 0., .34]]))
    actor, _ = load(args.source_checkpoint, 'MotionZeroStudent'); actor.eval()
    with torch.inference_mode():
        action_delta = (actor.mean(obs)-actor.mean(torch.tensor(native_obs)[None])).abs().max().item()
    audit = dict(version='motion_mcu_engine_alignment_v1', native=native, joints=joint_rows, bodies=body_rows,
                 native_readback=str(args.native_readback.resolve()), native_readback_sha256=sha(args.native_readback),
                 initial_native_actor_input=native_obs.tolist(), initial_mujoco_actor_input=obs[0].tolist(),
                 initial_input_max_delta=float(np.abs(obs[0].numpy()-native_obs).max()),
                 initial_action_max_delta=action_delta,
                 velocity_scope='both Native root_lin_vel_w alias and MuJoCo mj_objectVelocity use root body COM; height uses root link origin',
                 dormant_contact_parameters='wheel/ground condim3: torsional and rolling coefficients are not active',
                 effective_control='MCU active armature .012 and wheel added inertia replace authored values; native source drive625/.25 is overridden by runtime')
    write(out/'alignment_audit.json', audit)
    return cfg, native, actor


def alignment_cases(args, spec, out):
    from compare_motion_timing_profiles import apply_actuator_profile
    from validate_native_mujoco import binding
    cfg, native, actor = alignment_audit(args, spec, out)
    variants = ['reference', 'active_damping0', 'passive_damping01', 'guide_damping0',
                'all_joint_damping', 'sliding05_hypothesis', 'knee_limits_native',
                'self_collision_off', 'all_listed']
    if args.alignment_set == 'solver':
        variants = ['closure_tight', 'contact_tight', 'closure_contact_tight',
                    'contact_margin0', 'all_declared_and_solver']
    if not args.simulate: return
    started, results, failure = time.monotonic(), [], None
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    try:
        for variant in variants:
            model = alignment_model(args.xml, self_collisions=variant not in ('self_collision_off', 'all_listed', 'all_declared_and_solver'))
            refs = binding(model)
            apply_actuator_profile(model, refs, cfg, {k: getattr(model, k).copy()
                                                   for k in ('dof_armature', 'dof_damping', 'dof_frictionloss')})
            cap = np.asarray(cfg['profile']['solver_limits_nm']) / np.abs(refs['gears'])
            model.actuator_ctrlrange[refs['actuators']] = np.stack((-cap, cap), -1)
            readback = alignment_parameters(model, refs, variant, native)
            with torch.inference_mode():
                summary, arrays = simulation_case(model, refs, actor, cfg,
                        dict(drag=False, wheel_feedforward=False, spring_scale=1.),
                        [0., 0.], 18., started + args.wall_seconds)
            summary.update(variant=variant, parameters=readback, trace=variant+'.npz')
            if summary.get('complete_hold'):
                mask = (arrays['time'] > 10) & (arrays['time'] <= 18 + 1e-8)
                summary.update(height_mean_m=float(arrays['state'][mask, 3].mean()),
                               closure_peak_m=float(arrays['state'][mask, 6].max()))
            np.savez_compressed(out/summary['trace'], **arrays)
            write(out/(variant+'.json'), summary); results.append(summary)
            print(json.dumps({k: summary.get(k) for k in ['variant', 'termination', 'vx_mean', 'roll_peak']}), flush=True)
            if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > 32*2**20:
                raise RuntimeError('alignment storage limit32MiB')
        if not all(torch.equal(before[k], v) for k, v in actor.state_dict().items()):
            raise RuntimeError('actor changed')
    except Exception as exc:
        failure = repr(exc); raise
    finally:
        write(out/'alignment_results.json', dict(status='COMPLETED' if len(results)==len(variants) and failure is None else 'FAILED',
              failure=failure, cases=results, wall_seconds=time.monotonic()-started,
              physics_seconds=sum(r['simulated_seconds'] for r in results),
              source_actor_unchanged=all(torch.equal(before[k], v) for k, v in actor.state_dict().items()),
              PPO_updates=0, capability_certified=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'source-assessment', 'current-run', 'source-checkpoint', 'candidate-checkpoint', 'xml', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--wall-seconds', type=float, default=900.)
    p.add_argument('--run', action='store_true'); p.add_argument('--simulate', action='store_true')
    p.add_argument('--alignment', action='store_true', help='source Actor only: native-vs-MuJoCo plant audit and bounded zero-command ablations')
    p.add_argument('--native-readback', type=Path, help='completed native engineering readback used only with --alignment')
    p.add_argument('--alignment-set', choices=('plant', 'solver'), default='plant',
                   help='plant checks nine physical/config factors; solver checks five closure/contact hypotheses')
    args = p.parse_args()
    if not 0 < args.wall_seconds <= 1800: p.error('wall seconds must be0..1800')
    if args.simulate and not args.run: p.error('--simulate needs --run')
    if args.alignment and args.native_readback is None: p.error('--alignment needs --native-readback')
    inputs = {str(checked(x)): sha(x) for x in [args.spec, args.source_assessment, args.source_checkpoint,
                                               args.candidate_checkpoint, args.xml, Path(__file__)]}
    if args.native_readback is not None: inputs[str(checked(args.native_readback))] = sha(args.native_readback)
    print(json.dumps(dict(output=str(args.output_dir), new_PPO=0, simulation=args.simulate,
                          cases=((5 if args.alignment_set == 'solver' else 9) if args.alignment else 50) if args.simulate else 0,
                          maximum_physics_seconds=((90 if args.alignment_set == 'solver' else 162) if args.alignment else 900) if args.simulate else 0), ensure_ascii=False), flush=True)
    if not args.run: return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    spec = json.loads(args.spec.read_text())
    write(args.output_dir / 'manifest.json', dict(inputs=inputs, command=vars(args) | {'spec': str(args.spec),
          'source_assessment': str(args.source_assessment), 'current_run': str(args.current_run),
          'source_checkpoint': str(args.source_checkpoint), 'candidate_checkpoint': str(args.candidate_checkpoint),
          'xml': str(args.xml), 'output_dir': str(args.output_dir),
          'native_readback': str(args.native_readback) if args.native_readback is not None else None}, scope='user-approved bounded diagnostics; no new PPO'))
    if args.alignment:
        alignment_cases(args, spec, args.output_dir)
        return
    native_review(args, args.output_dir)
    write(args.output_dir / 'timing_audit.json', timing_review(spec))
    if args.simulate: simulations(args, spec, args.output_dir)


if __name__ == '__main__':
    main()
