"""Paired, bounded Motion31 closed-loop solver/compliance diagnostic.

Uses the existing MCU controller/encoder/run_case; no training, reward changes,
teacher rescue, external MJCF edits, or numerical-precision capability claims.
Without --run, prints a zero-write plan.
"""
import argparse
import importlib.util
import json
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from native_stand.students.data import sha
from native_stand.students.mujoco_solver import PROFILES, ConstraintRecorder, apply

VERSION = 'motion31_solver_accuracy_probe_v1'


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def plan(request):
    if request.get('version') != VERSION:
        raise ValueError('explicit solver diagnostic version required')
    if not request.get('profiles') or any(p not in PROFILES for p in request['profiles']):
        raise ValueError('explicit solver profile list required')
    if len(set(request['profiles'])) != len(request['profiles']):
        raise ValueError('duplicate solver profiles')
    if len({r['name'] for r in request['models']}) != len(request['models']):
        raise ValueError('unique model names required')
    if request['conditions'] != [0, 2]:
        raise ValueError('this version compares nominal and existing stress zero only')
    limits = request['limits']
    count = len(request['profiles']) * len(request['models']) * len(request['conditions'])
    if not (0 < count <= 32 and limits['seconds_per_case'] == 30 and
            60 <= limits['wall_s'] <= 1800 and 0 < limits['storage_bytes'] <= 128 * 2**20):
        raise ValueError('finite paired solver limits required')
    for key in ('parent_spec', 'xml', 'plant_adapter'):
        if sha(request[key]) != request[key + '_sha256']:
            raise ValueError('solver input identity differs: ' + key)
    for model in request['models']:
        if sha(model['checkpoint']) != model['sha256']:
            raise ValueError('solver Actor identity differs: ' + model['name'])
    spec = json.loads(Path(request['parent_spec']).read_text())
    if spec.get('interface_version') != 'multiskill_mcu_v2' or not spec.get('mcu_execution'):
        raise ValueError('declared 31D MCU execution required')
    if sha(spec['asset_path']) != spec['asset_sha256']:
        raise ValueError('bound USD identity differs')
    return dict(version=VERSION, status='PLAN_ONLY', files_written=0,
        cases=count, maximum_physical_seconds=count * limits['seconds_per_case'],
        limits=limits, profiles=request['profiles'], models=request['models'],
        model_input='31D Motion/NONE; height .34m; vx/yaw zero',
        control='1ms physics / 5ms MCU adoption+PD / 10ms policy; existing post-PD transport and independent spring',
        scope='numerical convergence vs constraint compliance; engineering hypotheses, not measured physics',
        source_mjcf_read_only=True, online_PPO_updates=0, offline_optimizer_steps=0,
        hardware_validation='NOT_RUN', capability_certificate=False)


def constraint_metrics(arrays, lo=2., hi=30.):
    mask = (arrays['time'] > lo + 1e-8) & (arrays['time'] <= hi + 1e-8)
    if not mask.any():
        return dict(samples=0)
    closure = np.linalg.norm(arrays['closure_vectors_m'][mask], axis=-1)
    penetration = arrays['wheel_ground_penetration_m'][mask]
    iterations = arrays['solver_iterations_total'][mask]
    return dict(samples=int(mask.sum()),
        closure_per_loop_mean_m=closure.mean(0).tolist(),
        closure_per_loop_p95_m=np.quantile(closure, .95, axis=0).tolist(),
        closure_peak_m=float(closure.max()),
        wheel_penetration_p95_m=np.quantile(penetration, .95, axis=0).tolist(),
        wheel_penetration_peak_m=penetration.max(0).tolist(),
        active_contact_fraction=(arrays['wheel_ground_active_contact_count'][mask] > 0).mean(0).tolist(),
        solver_iterations_mean=float(iterations.mean()), solver_iterations_peak=int(iterations.max()),
        equality_constraint_position_peak_m=float(arrays['equality_position_peak_m'][mask].max()))


def run(request):
    details = plan(request)
    if request.get('authorization') != 'AUTHORIZED':
        raise ValueError('explicit current solver diagnostic authorization required')
    started = time.monotonic()
    deadline = started + request['limits']['wall_s']
    folder = Path(request['output'])
    from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
    validate_new_experiment_name(folder.name)
    if shutil.disk_usage(folder.parent).free < 512 * 2**20:
        raise RuntimeError('solver diagnostic requires 512MiB settlement headroom')
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / 'request.json', request)
    write(folder / 'plan.json', details)
    ledger = dict(version=VERSION, status='RUNNING', attempts=1, completed_cases=0,
        failed_cases=0, physical_seconds=0., unknown_partial_seconds_upper_bound=0.,
        online_PPO_updates=0, offline_optimizer_steps=0, limits=request['limits'])
    write(folder / 'ledger.json', ledger)
    inputs = {request[k]: request[k + '_sha256'] for k in ('parent_spec', 'xml', 'plant_adapter')}
    inputs.update({r['checkpoint']: r['sha256'] for r in request['models']})
    spec = json.loads(Path(request['parent_spec']).read_text())
    inputs[spec['asset_path']] = spec['asset_sha256']
    from native_stand.students import mujoco_solver
    import diagnose_motion_closed_loop_robustness as loop
    import motion_mcu_playback
    code = {str(Path(p).resolve()): sha(p) for p in
            (__file__, mujoco_solver.__file__, loop.__file__, motion_mcu_playback.__file__)}
    write(folder / 'identity.json', dict(inputs_sha256=inputs, code_sha256=code))
    rows = []
    error = None
    recorder = None
    try:
        import mujoco
        from native_stand.students.model import load
        from native_stand.students.motion_distillation_pipeline import validate_student, validate_teacher
        from motion_mcu_playback import configuration, declared
        from compare_motion_timing_profiles import apply_actuator_profile
        from motion_mcu_distillation import uncertainty

        torch.set_num_threads(1)
        actors = {}
        for record in request['models']:
            actor, saved = load(record['checkpoint'], 'MotionZeroStudent')
            if saved.get('distillation') is not None:
                validate_student(saved, request['parent_spec'])
            else:
                validate_teacher(record['checkpoint'], request['parent_spec'])
            actor.eval()
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            actors[record['name']] = actor
        adapter_info = importlib.util.spec_from_file_location('frozen_solver_plant', request['plant_adapter'])
        adapter = importlib.util.module_from_spec(adapter_info)
        adapter_info.loader.exec_module(adapter)
        cfg = configuration(SimpleNamespace(leg_armature_m=.012, zero_latency=False), declared(spec), spec)

        def guard():
            if time.monotonic() >= deadline:
                raise TimeoutError('solver diagnostic wall limit')
            if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) > request['limits']['storage_bytes'] - 2**20:
                raise RuntimeError('solver diagnostic output limit with settlement reserve')
            if shutil.disk_usage(folder).free < 256 * 2**20:
                raise RuntimeError('solver diagnostic free space guard')

        for profile in request['profiles']:
            # Fresh compiled model per profile; no residual tight settings leak.
            guard()
            plant = adapter.Plant(request, spec)
            model, refs = plant.model, plant.refs
            apply_actuator_profile(model, refs, cfg, plant.original)
            cap = np.asarray(cfg['profile']['solver_limits_nm']) / np.abs(refs['gears'])
            model.actuator_ctrlrange[refs['actuators']] = np.stack((-cap, cap), -1)
            model.geom_friction[:] = 0.
            model.geom_friction[:, 0] = .5
            model.dof_damping[np.asarray(refs['dofs'])[[2, 5]]] = 0.
            model.dof_frictionloss[np.asarray(refs['dofs'])[[2, 5]]] = 0.
            readback = apply(model, refs, profile)
            readback.update(mujoco_version=mujoco.__version__, qpos_dtype=str(mujoco.MjData(model).qpos.dtype),
                execution_configuration=cfg, joint_range=model.jnt_range[refs['joints']].tolist(),
                actuator_ctrlrange=model.actuator_ctrlrange[refs['actuators']].tolist())
            write(folder / (profile + '_readback.json'), readback)
            for record in request['models']:
                for variant in request['conditions']:
                    guard()
                    case_id = f'{profile}__{record["name"]}__variant{variant}'
                    recorder = ConstraintRecorder()
                    begin = time.monotonic()
                    try:
                        with torch.inference_mode():
                            summary, arrays = loop.run_case(model, refs, actors, cfg,
                                uncertainty(dict(variant=variant)), record['name'],
                                request['limits']['seconds_per_case'], deadline,
                                constraint_observer=recorder)
                    except BaseException as exc:
                        partial = getattr(exc, 'physical_seconds', None)
                        if partial is None:
                            partial = recorder.last_integrated_time_s
                            ledger['unknown_partial_seconds_upper_bound'] += .01
                        ledger['physical_seconds'] += partial
                        ledger['failed_cases'] += 1
                        write(folder / (case_id + '_fault.json'), dict(case_id=case_id, error=repr(exc),
                            accounted_physical_seconds=partial, actual_clock_verified=hasattr(exc, 'physical_seconds')))
                        raise
                    row = dict(case_id=case_id, solver_profile=profile, model=record['name'], variant=variant,
                        wall_seconds=time.monotonic()-begin, constraints=constraint_metrics(arrays),
                        effective_wheel_ground_contacts=recorder.contact_readback(), **summary)
                    trace = folder / (case_id + '.npz')
                    np.savez_compressed(trace, **arrays)
                    row.update(trace=trace.name, trace_sha256=sha(trace))
                    write(folder / (case_id + '.json'), row)
                    rows.append(row)
                    ledger['completed_cases'] += 1
                    ledger['physical_seconds'] += summary['physical_seconds']
                    ledger['wall_seconds'] = time.monotonic() - started
                    write(folder / 'ledger.json', ledger)
                    print(json.dumps(dict(case=case_id, termination=row['termination'],
                        vx=row['windows']['steady_2_to_30'].get('vx_mean_mps'),
                        closure_peak_m=row['constraints'].get('closure_peak_m')), ensure_ascii=False), flush=True)
        ledger['status'] = 'COMPLETED'
    except BaseException as exc:
        ledger['status'] = 'FAILED'
        error = dict(error=repr(exc), traceback=traceback.format_exc())
    finally:
        changed = [path for path, expected in inputs.items() if sha(path) != expected]
        if changed:
            error = dict(error='input identity changed during diagnostic', paths=changed, earlier_error=error)
            ledger['status'] = 'FAILED'
        ledger.update(wall_seconds=time.monotonic()-started, input_identity_preserved=not changed,
                      output_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()))
        write(folder / 'ledger.json', ledger)
        write(folder / 'result.json', dict(version=VERSION, ledger=ledger, cases=rows, failure=error,
            timestamp_semantics='constraint_time_s is last solved physical state; time is integrated end; 1ms offset, no extra forward call',
            capability_certificate=False, hardware_validation='NOT_RUN', training='NOT_RUN'))
    if error:
        raise RuntimeError(error['error'])
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    print(json.dumps(run(request) if args.run else plan(request), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
