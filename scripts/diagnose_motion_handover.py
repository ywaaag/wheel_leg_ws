"""Bounded log-constrained MuJoCo handover sweep. Default: zero-write CPU plan.

Uses the deployed3250 Student, not4995. No LQR emulation, fitted hardware model,
Isaac or PPO. --run is physical sampling requiring explicit user authorization.
"""
import argparse
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from diagnose_motion_real2sim import DEFAULT_CP, DEFAULT_SPEC, ROOT, sensors
from replay_motion31_shadow_csv import read_csv
from validate_motion31_mujoco import plan as model_plan
from validate_native_mujoco import state
from native_stand.command_v6.domain.experiment_naming import new_experiment_name, validate_new_experiment_name
from native_stand.students.data import sha
from native_stand.students.real2sim_delay import TargetTransport, ObservationHistory, torque_response
from native_stand.students.real2sim_handover import LEG, WHEEL, SCALE, decode_log, initialize_from_log, signal_metrics


def profiles():
    base = dict(target_delay_ms=[0]*6, dq_delay_ms=0, imu_delay_ms=0,
                pd_feedback_delay_ms=0, torque_tau_ms=[0]*6)
    def row(name, **kw): return dict(name=name, **{**base, **kw})
    wheel = [0, 0, 11, 0, 0, 7]
    rows = [row('ideal'), row('wheel_transport11_7', target_delay_ms=wheel),
            row('wheel_transport7_11', target_delay_ms=[0, 0, 7, 0, 0, 11]),
            row('wheel_response11_7', torque_tau_ms=wheel),
            row('leg10_wheel11_7', target_delay_ms=[10, 10, 11, 10, 10, 7]),
            row('leg20_feedback10', target_delay_ms=[20, 20, 11, 20, 20, 7], pd_feedback_delay_ms=10),
            row('leg40_feedback10', target_delay_ms=[40, 40, 11, 40, 40, 7], pd_feedback_delay_ms=10),
            row('leg20_sensor10', target_delay_ms=[20, 20, 11, 20, 20, 7], dq_delay_ms=10, imu_delay_ms=10),
            row('leg20_response10', target_delay_ms=[20, 20, 11, 20, 20, 7], torque_tau_ms=[10, 10, 0, 10, 10, 0])]
    for ms in (5, 10, 15, 20):
        rows.append(row(f'feedback{ms}_wheel11_7', target_delay_ms=wheel, pd_feedback_delay_ms=ms))
    rows.append(row('dq_feedback10_wheel11_7', target_delay_ms=wheel, pd_feedback_delay_ms=10, pd_position_current=True))
    return rows


def run_case(model, refs, actor, initial, history_samples, row, *, cfg, cache_mode,
             trace_time, observations, sample_time, duration, compare_seconds, deadline,
             record_full_state=False):
    from native_stand.students.mcu_interface import encode
    from native_stand.students.interface import targets
    data = mujoco.MjData(model)
    data.qpos[:] = initial.qpos; data.qvel[:] = initial.qvel
    mujoco.mj_forward(model, data)
    observed = decode_log(row)
    # No logged LQR targets: this bridge is an assumption, not reconstructed execution.
    adopted = observed['q'].copy(); adopted[WHEEL] = observed['dq'][WHEEL]
    if cache_mode == 'logged_as_executed': adopted = observed['previous'].copy()
    cache = observed['previous'].copy() if cache_mode != 'bridge_actual' else adopted.copy()
    transport = TargetTransport(cfg['target_delay_ms'], adopted)
    history = ObservationHistory(history_samples[0][1], initial_tick=history_samples[0][0])
    # Only causal pre-event samples; 100Hz log cannot supply real1ms samples.
    for stamp, sample in history_samples: history.append(stamp, sample)
    bounds = torch.tensor(model.jnt_range[refs['joints']], dtype=torch.float32)[None]
    for i, j in enumerate(refs['joints']):
        if not model.jnt_limited[j]: bounds[0, i] = torch.tensor([-float('inf'), float('inf')])
    limit = np.array([40, 40, 5.5, 40, 40, 5.5]); actual = np.zeros(6); demand = actual.copy()
    results = []; failure = None
    for tick in range(round(duration*1000)):
        if tick % 100 == 0 and time.monotonic() > deadline: failure = 'WALL_BUDGET'; break
        pre = sensors(model, data, refs); history.append(tick, pre)
        if tick % 10 == 0:
            sensed = pre.copy()
            sensed[12:18] = history.at(tick-cfg['dq_delay_ms'])[12:18]
            sensed[:6] = history.at(tick-cfg['imu_delay_ms'])[:6]
            # Logged upper-layer command playback only; never replay future sensors.
            idx = max(0, np.searchsorted(trace_time, sample_time+tick/1000., side='right')-1)
            command = observations[idx, 22:25] * [4., 4., .5]
            obs = encode(torch.tensor(sensed[:3], dtype=torch.float32)[None],
                         torch.tensor(sensed[3:6], dtype=torch.float32)[None],
                         torch.tensor(sensed[6:12], dtype=torch.float32)[None],
                         torch.tensor(sensed[12:18], dtype=torch.float32)[None],
                         torch.tensor(cache[LEG], dtype=torch.float32)[None],
                         torch.tensor(cache[WHEEL], dtype=torch.float32)[None],
                         torch.zeros(1, 6), 'Motion', torch.tensor(command, dtype=torch.float32)[None])
            raw = actor.mean(obs)
            if not torch.isfinite(raw).all(): failure = 'NONFINITE_ACTION'; break
            decoded = targets(raw, torch.zeros(1, 6), bounds)
            requested = np.zeros(6)
            requested[LEG] = decoded['leg_executed'][0].numpy()
            requested[WHEEL] = decoded['wheel_executed'][0].numpy()
            transport.submit(tick, requested)
            if tick == 0: first_obs = obs[0].numpy().copy()
        delivered = transport.delivered(tick)
        if tick % 5 == 0:
            adopted = delivered.copy(); feedback = history.at(tick-cfg['pd_feedback_delay_ms'])
            if cfg.get('pd_position_current', False): feedback[6:12] = pre[6:12]
            demand[LEG] = 60*(adopted[LEG]-feedback[6:12][LEG])-2*feedback[12:18][LEG]
            demand[WHEEL] = adopted[WHEEL]-feedback[12:18][WHEEL]
            demand = np.clip(demand, -limit, limit)
        actual = torque_response(actual, demand, cfg['torque_tau_ms'])
        data.ctrl[refs['actuators']] = actual / refs['gears']
        mujoco.mj_step(model, data)
        # After the first inference both modes use only actual simulated adoption.
        cache = adopted.copy()
        if tick % 10 == 9:
            angular, gravity, s = state(model, data, refs)
            results.append(dict(time=(tick-9)/1000., state_time=tick/1000.+.001, target=requested.copy(), adopted=adopted.copy(),
                                delivered=delivered.copy(), torque=actual.copy(), demand=demand.copy(),
                                q_pre=sensed[6:12].copy(), dq_pre=sensed[12:18].copy(),
                                gravity=sensed[3:6].copy(), actual_gravity_post=gravity.copy(), actor_input=obs[0].numpy().copy(),
                                root_pos=data.xpos[refs['root']].copy(),
                                state=np.array([s['vx'], s['vy'], s['yaw_rate'], s['height'], s['closure_m']]),
                                contact=np.array(s['contact_force_n'])))
            if record_full_state:
                results[-1].update(full_qpos_post=data.qpos.copy(), full_qvel_post=data.qvel.copy())
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all(): failure = 'NONFINITE_STATE'
            elif s['closure_m'] > .005: failure = 'CLOSURE'
            elif s['up'] < .5: failure = 'UPRIGHT'
            if failure: break
    arrays = {k: np.stack([v[k] for v in results]) for k in results[0]} if results else {}
    result = dict(termination=failure, duration_s=len(results)*.01, physics_steps=tick+1,
                  cache_hypothesis=cache_mode, configuration=cfg,
                  initial_actor_input=first_obs.tolist() if results else None,
                  initial_observation_max_error=float(abs(first_obs-row).max()) if results else None)
    if len(results) >= 8:
        n = min(len(results), max(8, round(compare_seconds*100)+1))
        result['first_second'] = signal_metrics(arrays['time'][:n], arrays['target'][:n], arrays['dq_pre'][:n], arrays['gravity'][:n])
        result['both_contact_fraction'] = float((arrays['contact'] > 2).all(1).mean())
        result['upright_failure_time_s'] = float(arrays['state_time'][-1]) if failure == 'UPRIGHT' else None
    return result, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--spec', type=Path, default=DEFAULT_SPEC); p.add_argument('--checkpoint', type=Path, default=DEFAULT_CP)
    p.add_argument('--xml', type=Path, default=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml'))
    p.add_argument('--event-times', type=float, nargs='+', default=[17., 17.06])
    p.add_argument('--root-vx', type=float, nargs='+', default=[0., -.3])
    p.add_argument('--duration-seconds', type=float, default=8.)
    p.add_argument('--wall-seconds', type=float, default=600.)
    p.add_argument('--profiles', nargs='+', choices=[v['name'] for v in profiles()])
    p.add_argument('--output-dir', type=Path); p.add_argument('--run', action='store_true'); a = p.parse_args()
    if not 1 <= a.duration_seconds <= 15 or not 10 <= a.wall_seconds <= 1800:
        raise ValueError('bounded simulation duration/wall required')
    if len(a.event_times) > 4 or len(a.root_vx) > 4 or not np.isfinite(a.event_times+a.root_vx).all() or max(abs(v) for v in a.root_vx)>2:
        raise ValueError('bounded finite initial-state scan required')
    torch.set_num_threads(1)
    _, observations, actions, trace_time, _ = read_csv(a.csv)
    spec = json.loads(a.spec.read_text())
    model, refs, actor, manifest = model_plan(SimpleNamespace(spec=a.spec, checkpoint=a.checkpoint, xml=a.xml,
                                                           wall_seconds=a.wall_seconds, commands=[0.], yaw_commands=[], viewer=False), spec)
    selected = [v for v in profiles() if not a.profiles or v['name'] in a.profiles]
    caches = ['bridge_actual', 'logged_seed', 'logged_as_executed']
    manifest.update(version='motion31_log_constrained_handover_v1', csv=str(a.csv.resolve()), csv_sha256=sha(a.csv),
                    event_times_assumed_s=a.event_times, root_vx_assumed_mps=a.root_vx, profiles=selected, caches=caches,
                    maximum_cases=len(a.event_times)*len(a.root_vx)*len(selected)*len(caches), online_PPO=0,
                    provenance={'wheel_7_11ms':'empty-load combined onset; transport OR rise hypotheses, not identified split',
                                'leg_and_sensor_times':'unmeasured sensitivity hypotheses; never claimed hardware parameters',
                                'J_B_Tc':'NOT_APPLIED: shaft/units/model and wheel-label mapping unconfirmed'},
                    actual_previous_source='UNKNOWN_HARDWARE; three explicitly assumed simulated bridges',
                    missing_real_metrics=['contact', 'root height/linear velocity', 'actual execution/control-owner'],
                    prehistory='causal100Hz real sensor snapshots; negative ticks only, no fabricated sub-ms observations',
                    trace_timing='time/target/q_pre/dq_pre/gravity/actor_input are policy pre-step; state_time/state/contact/actual_gravity_post are10ms later',
                    initial_torque='zero assumed; unlogged compensation and LQR torque not reconstructed',
                    training_export_ready=False)
    manifest['input_sha256'].update({str(a.csv.resolve()):sha(a.csv), str(Path(__file__).resolve()):sha(__file__),
        str(ROOT/'src/native_stand/students/real2sim_handover.py'):sha(ROOT/'src/native_stand/students/real2sim_handover.py'),
        str(ROOT/'src/native_stand/students/real2sim_delay.py'):sha(ROOT/'src/native_stand/students/real2sim_delay.py')})
    starts=[]
    for t in a.event_times:
        ix = int(np.searchsorted(trace_time, t, side='right')-1)
        if ix < 20 or ix+100 >= len(trace_time): raise ValueError('event needs valid prehistory and one-second comparison')
        decode_log(observations[ix])
        starts.append(ix)
    if not a.run: print(json.dumps(manifest, indent=2)); return
    out = a.output_dir or ROOT/'sim2sim'/new_experiment_name('motion3250_log_handover')
    validate_new_experiment_name(out.name); out.mkdir(parents=True, exist_ok=False)
    def write(name, obj):
        with (out/name).open('x') as f: json.dump(obj, f, indent=2, allow_nan=False)
    write('manifest.json', manifest)
    from native_stand.command_v6.completion import write as atomic
    budget=dict(version='mujoco_handover_budget_v1', attempt_id=uuid.uuid4().hex,
                authorization='user approved real-log constrained MuJoCo handover reproduction; no PPO',
                wall_limit_s=a.wall_seconds, storage_max_bytes=128*2**20,
                maximum_cases=manifest['maximum_cases'], online_PPO=0, status='RUNNING', attempts=[])
    started=time.monotonic(); results=[]; real=[]; status='FAILED'; initializations=[]
    before={k:v.clone() for k,v in actor.state_dict().items()}
    atomic(out/'budget.json', budget)
    try:
        with torch.inference_mode():
            for ix in starts:
                at=float(trace_time[ix]); row=observations[ix]
                mask=(trace_time>=at)&(trace_time<at+1.)&(abs(np.linalg.norm(observations[:,3:6], axis=1)-1)<.02)
                real.append(dict(event_sample_time_s=at, metrics=signal_metrics(trace_time[mask]-at, actions[mask]*SCALE,
                    observations[mask,10:16]*20., observations[mask,3:6]), contact='UNKNOWN_NOT_LOGGED',
                    owner='UNKNOWN; alignment before visible oscillation only'))
                history_samples=[]
                for j in range(ix-15, ix):
                    h=decode_log(observations[j]); stamp=int(np.floor((trace_time[j]-at)*1000))
                    history_samples.append((stamp, np.r_[h['gyro'],h['gravity'],h['q'],h['dq']]))
                for vx in a.root_vx:
                    initial, fit=initialize_from_log(model, refs, row, vx)
                    initializations.append(dict(event_sample_time_s=at, **fit))
                    for cfg in selected:
                        for cache_mode in caches:
                            name=f"case_{len(results):03d}"
                            summary, arrays=run_case(model, refs, actor, initial, history_samples, row, cfg=cfg,
                                cache_mode=cache_mode, trace_time=trace_time, observations=observations, sample_time=at,
                                duration=a.duration_seconds, compare_seconds=float(trace_time[mask][-1]-at), deadline=started+a.wall_seconds)
                            summary.update(name=name, event_sample_time_s=at, assumed_root_vx_mps=vx)
                            if arrays: np.savez_compressed(out/(name+'.npz'), **arrays)
                            write(name+'.json', summary); results.append(summary)
                            budget['attempts'].append(dict(name=name, physics_steps=summary['physics_steps'], termination=summary['termination']))
                            atomic(out/'budget.json', budget)
                            if summary['termination']=='WALL_BUDGET': raise TimeoutError('diagnostic wall bound')
                            if sum(v.stat().st_size for v in out.iterdir() if v.is_file()) > budget['storage_max_bytes']:
                                raise RuntimeError('diagnostic storage bound')
                print(json.dumps(dict(event_sample_time_s=at, completed_cases=len(results))), flush=True)
        status='COMPLETED'
    finally:
        unchanged=all(torch.equal(v, actor.state_dict()[k]) for k,v in before.items())
        budget.update(status=status, wall_used_s=time.monotonic()-started,
                      physics_steps_used=sum(v['physics_steps'] for v in results))
        atomic(out/'budget.json', budget)
        write('terminal_report.json', dict(status=status, results=results, real=real, initializations=initializations,
              model_unchanged=unchanged, online_PPO=0, training_export_ready=False))
        if not unchanged: raise RuntimeError('Actor changed')
        for path, digest in manifest['input_sha256'].items():
            if sha(path)!=digest: raise RuntimeError('read-only input changed: '+path)
    print('OUTPUT='+str(out.resolve()), flush=True)


if __name__=='__main__': main()
