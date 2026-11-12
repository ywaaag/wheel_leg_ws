"""Current-MCU adapter for the existing same31D distillation CLI.

Reuses the existing BC trainer/exporter and the closed-loop MCU diagnostic. No
PPO, Isaac, firmware update, or Teacher intervention in Student rollouts.
"""
import copy
import importlib.util
import json
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from native_stand.students import motion_distillation_pipeline as pipeline
from native_stand.students.data import sha
from native_stand.students.model import load
from native_stand.students.motion_execution_uncertainty import nominal

VERSION = 'motion31_mcu_distillation_cycle_v2'
PRECISE_VERSION = 'motion31_mcu_distillation_cycle_v3'


def solver_profile(request):
    """Historical v2 stays on its original plant; new requests freeze one version."""
    from native_stand.students.mujoco_solver import PROFILES
    if request.get('version') == VERSION:
        if request.get('solver_profile', 'legacy_xml_v1') != 'legacy_xml_v1':
            raise ValueError('historical v2 keeps legacy solver; use explicit v3 for precision')
        return 'legacy_xml_v1'
    if request.get('version') != PRECISE_VERSION:
        raise ValueError('explicit MCU distillation version required')
    version = request.get('solver_profile')
    if version not in PROFILES:
        raise ValueError('new v3 requires an explicit solver_profile')
    if request.get('objective') != 'teacher_fidelity':
        raise ValueError('new v3 objective must be teacher_fidelity')
    if request.get('plant_profile') != 'motion_mcu_currentplant_v1':
        raise ValueError('new v3 requires the current MCU plant profile')
    return version


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate(request):
    selected_solver = solver_profile(request)
    _, source, spec = pipeline.validate_teacher(request['teacher'], request['parent_spec'])
    keys = ('teacher', 'parent_spec', 'xml', 'plant_adapter') if request['version'] == VERSION else ('teacher', 'parent_spec', 'xml')
    for key in keys:
        if sha(request[key]) != request[key + '_sha256']:
            raise ValueError('distillation input SHA differs: ' + key)
    if not spec.get('mcu_execution') or source.get('critic_privileged_version') != 'motion_mcu_motorqueue_privileged312_v1':
        raise ValueError('declared MCU312 Teacher required')
    cases = request['cases']
    if not cases or len({r['case_id'] for r in cases}) != len(cases):
        raise ValueError('unique explicit distillation cases required')
    pipeline.dataset_cohorts({'cohorts': cases}, {'command_set_version': 'motion31_explicit_commands_v2', 'cohorts': cases})
    if len(request['split_yaw_rad']) != 3 or len(set(request['split_yaw_rad'])) != 3:
        raise ValueError('three whole-episode initial yaw groups required')
    limits = request['limits']
    if not (1 <= limits['rounds'] <= 5 and 1 <= limits['steps_per_round'] <= 5000 and
            1 <= limits['fit_wall_s'] <= 1200 and 60 <= limits['wall_s'] <= 14400 and
            0 < limits['storage_bytes'] <= 4 * 2**30):
        raise ValueError('finite distillation/CPU simulation limits required')
    if any(not 0 <= i < len(cases) for i in request['test_case_indices']):
        raise ValueError('test case index outside the declared matrix')
    if request.get('initial_student'):
        if request['version'] != PRECISE_VERSION:
            raise ValueError('initial student continuation requires explicit v3')
        for key in ('initial_student','initial_dataset'):
            if sha(request[key]) != request[key+'_sha256']:
                raise ValueError('initial student/dataset identity differs')
        saved=torch.load(request['initial_student'],map_location='cpu',weights_only=False)
        identity=pipeline.validate_student(saved,request['parent_spec'])
        if identity['teacher_sha256']!=request['teacher_sha256'] or identity['dataset_sha256']!=request['initial_dataset_sha256']:
            raise ValueError('initial student exact Teacher/dataset differs')
    if request.get('standing_batch_fraction') is not None:
        fraction=request['standing_batch_fraction']
        if request['version'] != PRECISE_VERSION or not 0<float(fraction)<1:
            raise ValueError('explicit v3 standing sampling fraction required')
        if not any(c['target'][0]==c['target'][1]==0 for c in cases) or not any(c['target'][0]!=0 or c['target'][1]!=0 for c in cases):
            raise ValueError('standing and moving cohorts both required')
    return dict(version=request['version'], status='PLAN_ONLY', files_written=0, online_PPO=0,
                offline_step_limit=limits['rounds'] * limits['steps_per_round'],
                cases=len(cases), limits=limits, teacher_sha256=request['teacher_sha256'],
                solver_profile=selected_solver, objective='teacher_fidelity',
                plant_profile=request.get('plant_profile', 'historical_frozen_adapter'),
                acceptance_scope='match the SAME Teacher under identical MuJoCo settings; absolute task metrics reported separately')


def uncertainty(case):
    row = nominal()
    if case['variant']:
        row.update(initial_vx_mps=.05, initial_pitch_rate_rad_s=.1)
    if case['variant'] == 2:
        row.update(PD_age_ms=12, motor_delay_ms=[v+2 for v in row['motor_delay_ms']], spring_scale=1.1)
    return row


def episode(arrays, teacher, case, cohort, eid, split):
    """Saved PRE-action input is queried, never reconstructed from post-action q."""
    t = arrays['time'] - .01
    phase = np.select([t < 2, t < 10, t < 18, t < 26], [0, 1, 2, 3], default=4)
    indices = []
    for p in range(5):
        ids = np.flatnonzero(phase == p)
        if len(ids):
            indices.extend(ids[np.linspace(0, len(ids)-1, min(300, len(ids))).astype(int)])
    ids = np.array(sorted(set(indices)))
    x = torch.from_numpy(arrays['actor_input'][ids]).float()
    if not torch.isfinite(x).all() or x.shape[1] != 31:
        raise ValueError('invalid actual pre-action input')
    if not ((x[:, 25] == 1) & (x[:, 26:28] == 0).all(-1) & (x[:, 28] == 1) & (x[:, 29:31] == 0).all(-1)).all():
        raise ValueError('Motion/NONE task encoding differs')
    torch.testing.assert_close(x[:, 22:25], torch.tensor(arrays['effective_command'][ids],dtype=torch.float32) / torch.tensor([4., 4., .5]))
    with torch.inference_mode():
        y = torch.cat([teacher.mean(chunk) for chunk in x.split(1024)])
    n = len(x)
    return dict(x=x, y=y, cohort=torch.full((n,), cohort), phase=torch.tensor(phase[ids]),
                split=torch.full((n,), split), episode=torch.full((n,), eid))


def make_dataset(parts, request, output, provenance):
    data = {k: torch.cat([p[k] for p in parts]) for k in ('x', 'y', 'cohort', 'phase', 'split', 'episode')}
    for a in range(3):
        for b in range(a+1, 3):
            if set(data['episode'][data['split'] == a].tolist()) & set(data['episode'][data['split'] == b].tolist()):
                raise ValueError('whole-episode split overlap')
    data.update(cohorts=request['cases'], teacher_checkpoint=request['teacher'],
                teacher_sha256=request['teacher_sha256'], parent_spec=request['parent_spec'],
                parent_spec_sha256=request['parent_spec_sha256'], provenance=provenance)
    output.mkdir(parents=True, exist_ok=False)
    torch.save(data, output / 'dataset.pt')
    counts = {}
    for ci, case in enumerate(request['cases']):
        counts[case['case_id']] = {str(s): int(((data['cohort'] == ci) & (data['split'] == s)).sum()) for s in range(3)}
    metadata=dict(version=pipeline.VERSION, command_set_version='motion31_explicit_commands_v2',
        **{k: data[k] for k in ('teacher_checkpoint', 'teacher_sha256', 'parent_spec', 'parent_spec_sha256')},
        cohorts=request['cases'], dataset_sha256=sha(output / 'dataset.pt'), samples=len(data['x']),
        episodes=len(data['episode'].unique()), split='whole episodes: yaw0 train, yaw+.015 validation; independent yaw-.015 test later',
        phase='scheduled command: initial0-2/ramp2-10/hold10-18/exit18-26/parking26-30; never Actor fields',
        sample_counts=counts, provenance=provenance, capability_certified=False, online_PPO=0,
        solver_profile=solver_profile(request) if request.get('version') else 'legacy_xml_v1',
        plant_profile=request.get('plant_profile', 'historical_frozen_adapter'),
        objective='teacher_fidelity')
    if request.get('standing_batch_fraction') is not None:
        standing=np.array([c['target'][0]==c['target'][1]==0 for c in request['cases']])
        fraction=float(request['standing_batch_fraction'])
        metadata['cohort_sampling_weights']=np.where(standing,fraction/standing.sum(),(1-fraction)/(~standing).sum()).tolist()
    write(output / 'manifest.json', metadata)
    return output / 'dataset.pt'


def tracking(arrays, case):
    hold = (arrays['time'] > 10) & (arrays['time'] <= 18 + 1e-6)
    result = dict(hold_complete=int(hold.sum()) == 800)
    if not result['hold_complete']:
        return result
    s = arrays['state'][hold]
    axis = 2 if case['target'][1] else 0
    return dict(result, response=float(s[:, axis].mean()),
        tracking_MAE=float(np.abs(s[:, axis] - case['target'][1 if axis == 2 else 0]).mean()),
        height_MAE=float(np.abs(s[:, 3] - .34).mean()), pitch_peak=float(np.abs(s[:, 4]).max()),
        roll_peak=float(np.abs(s[:, 5]).max()), leg_difference_MAE=float(np.abs(arrays['leg_difference'][hold]).mean()),
        both_contact_fraction=float((arrays['contact'][hold] > 2).all(-1).mean()),
        driver_limit_fraction=float((np.abs(arrays['driver'][hold]) >= arrays['driver_cap'][hold]-1e-4).any(-1).mean()))


def compare(teacher, student):
    """Relative fidelity only. Teacher faults are not converted into capabilities."""
    if len(teacher) != len(student) or not teacher:
        raise ValueError('complete paired comparison required')
    rows = []
    for t, s in zip(teacher, student):
        if t['case_id'] != s['case_id'] or t['target'] != s['target']:
            raise ValueError('paired cases differ')
        if t.get('solver_profile') != s.get('solver_profile') or t.get('plant_profile') != s.get('plant_profile'):
            raise ValueError('paired MuJoCo solver/plant differs; resample a matching Teacher baseline')
        full = t['termination'] == s['termination'] == 'WINDOW_COMPLETE'
        both_hold = t.get('hold_complete') and s.get('hold_complete')
        gap = abs(s['response'] - t['response']) if both_hold else None
        posture = max(0., s['pitch_peak'] - t['pitch_peak'], s['roll_peak'] - t['roll_peak']) if both_hold else None
        passed = full and both_hold and gap <= .05 and posture <= .01
        if passed:
            passed = (s['height_MAE'] <= t['height_MAE'] + .005 and
                      s['leg_difference_MAE'] <= t['leg_difference_MAE'] + .003 and
                      s['both_contact_fraction'] >= t['both_contact_fraction'] - .02 and
                      s['driver_limit_fraction'] <= t['driver_limit_fraction'] + .05)
        rows.append(dict(case_id=t['case_id'], target=t['target'], variant=t['variant'],
            teacher_seconds=t['physical_seconds'], student_seconds=s['physical_seconds'],
            teacher_termination=t['termination'], student_termination=s['termination'],
            response_gap=gap, posture_excess=posture, fidelity_passed=bool(passed)))
    return dict(passed=sum(r['fidelity_passed'] for r in rows), count=len(rows), rows=rows,
        all_fidelity_passed=all(r['fidelity_passed'] for r in rows),
        rank=[sum(r['student_termination']=='WINDOW_COMPLETE' for r in rows),
              sum(r['fidelity_passed'] for r in rows), sum(r['student_seconds'] for r in rows),
              -sum(r['response_gap'] if r['response_gap'] is not None else 10. for r in rows)],
        capability_certified=False, native_confirmation='NOT_RUN', hardware_validation='NOT_RUN')


def run(request, authorization):
    validate(request)
    if not authorization or request.get('authorization') != 'AUTHORIZED':
        raise ValueError('current user authorization required')
    from diagnose_motion_closed_loop_robustness import run_case
    from motion_mcu_playback import configuration, declared
    from compare_motion_timing_profiles import apply_actuator_profile
    import mujoco
    folder = Path(request['output']); folder.mkdir(parents=True, exist_ok=False)
    write(folder / 'request.json', request)
    write(folder / 'authorization.json', dict(scope=authorization, online_PPO=0))
    teacher, _, spec = pipeline.validate_teacher(request['teacher'], request['parent_spec'])
    for p in teacher.parameters():
        p.requires_grad_(False)
    cfg = configuration(SimpleNamespace(leg_armature_m=.012, zero_latency=False), declared(spec), spec)
    if request['version'] == PRECISE_VERSION:
        from validate_native_mujoco import binding
        from motion_mcu_playback import apply_current_plant
        model = mujoco.MjModel.from_xml_path(request['xml']); refs = binding(model)
        plant_readback = apply_current_plant(model, refs, spec, cfg)
        rb = plant_readback['actuator']
    else:
        adapter_spec = importlib.util.spec_from_file_location('frozen_mcu_plant', request['plant_adapter'])
        adapter = importlib.util.module_from_spec(adapter_spec); adapter_spec.loader.exec_module(adapter)
        # Frozen v2 keeps the original adapter and original numeric semantics.
        plant = adapter.Plant(request, spec); model, refs = plant.model, plant.refs
        rb = apply_actuator_profile(model, refs, cfg, plant.original)
        cap = np.asarray(cfg['profile']['solver_limits_nm']) / np.abs(refs['gears'])
        model.actuator_ctrlrange[refs['actuators']] = np.stack((-cap, cap), -1)
        model.geom_friction[:] = 0.; model.geom_friction[:, 0] = .5
        model.dof_damping[np.asarray(refs['dofs'])[[2,5]]] = 0
        model.dof_frictionloss[np.asarray(refs['dofs'])[[2,5]]] = 0
        plant_readback = dict(version='historical_frozen_adapter')
    from native_stand.students import mujoco_solver
    selected_solver = solver_profile(request)
    solver_readback = mujoco_solver.apply(model, refs, selected_solver)
    write(folder / 'plant_readback.json', dict(configuration=cfg, readback=rb,
        plant_profile=plant_readback, solver_profile=solver_readback,
        wheel_ground_pair_sliding_friction=.5, contact_scope='MuJoCo approximation; not Native or measured hardware',
        mechanical_limits=model.jnt_range[refs['joints']].tolist(), joint_damping=model.dof_damping[refs['dofs']].tolist(),
        equality_solref=model.eq_solref.tolist(), equality_solimp=model.eq_solimp.tolist(),
        PD_target_transport_ms=0, motor_transport='post-PD', source_native_certification_not_transferred=True))
    start=time.monotonic(); deadline=start+request['limits']['wall_s']
    used=0; physical=0.; episodes=0; parts=[]; teacher_val=[]; history=[]
    best=Path(request['initial_student']) if request.get('initial_student') else None
    best_rank=None; best_dataset=Path(request['initial_dataset']) if best else None
    initial_comparison=None
    paths=[request['teacher'], request['parent_spec'], request['xml'], spec['asset_path']]
    if request['version'] == VERSION: paths.append(request['plant_adapter'])
    else: paths.append(spec['colleague_config_path'])
    if request.get('initial_student'): paths.extend([request['initial_student'],request['initial_dataset']])
    source_hashes={str(p):sha(p) for p in paths}
    write(folder/'input_manifest.json',dict(source_sha256=source_hashes,code_sha256={str(Path(p).resolve()):sha(p) for p in
        (Path(__file__),Path(pipeline.__file__),Path(__file__).with_name('diagnose_motion_closed_loop_robustness.py'),
         Path(__file__).with_name('motion_mcu_playback.py'),Path(mujoco_solver.__file__))}))

    def guard():
        if time.monotonic() >= deadline:
            raise TimeoutError('finite distillation chain wall limit')
        if shutil.disk_usage(folder).free < 512*2**20 or sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) >= request['limits']['storage_bytes']:
            raise RuntimeError('distillation storage guard')

    def rollout(actor, ci, split, out):
        nonlocal physical, episodes
        guard(); torch.set_num_threads(1); case=request['cases'][ci]
        with torch.inference_mode():
            try:
                summary, arrays=run_case(model,refs,{'policy':actor},cfg,uncertainty(case),'policy',30.,deadline,
                    target=case['target'], initial_yaw=request['split_yaw_rad'][split])
            except BaseException as exc:
                partial=getattr(exc,'physical_seconds',None)
                if partial is not None:physical+=partial
                episodes+=1
                write(out.with_suffix('.fault.json'),dict(case_id=case['case_id'],split=split,error=repr(exc),
                    physical_seconds=partial,physics_tick=getattr(exc,'physics_tick',None),runtime_fault=True,
                    scoring_completed=False,not_retried=True))
                raise
        summary.update(case_id=case['case_id'],variant=case['variant'],**tracking(arrays,case))
        summary.update(solver_profile=selected_solver,plant_profile=plant_readback['version'],objective='teacher_fidelity')
        physical+=summary['physical_seconds'];episodes+=1
        out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out.with_suffix('.npz'),**arrays)
        summary.update(trace=out.with_suffix('.npz').name,trace_sha256=sha(out.with_suffix('.npz')))
        write(out.with_suffix('.json'),summary)
        print(json.dumps(dict(stage='CLOSED_LOOP',model='Teacher' if actor is teacher else 'Student',case=ci,split=split,
                              seconds=summary['physical_seconds'],termination=summary['termination'])),flush=True)
        return summary,arrays

    status='FAILED'; error=None; exported=None; test_comparison=None
    try:
        for split in (0,1):
            for ci,case in enumerate(request['cases']):
                row,arrays=rollout(teacher,ci,split,folder/'teacher'/f'split{split}_case{ci:02}')
                parts.append(episode(arrays,teacher,case,ci,split*1000+ci,split))
                if split==1:teacher_val.append(row)
        if best is not None:
            initial,_=load(best,'MotionZeroStudent');initial.eval().requires_grad_(False)
            original_rows=[]
            for ci in range(len(request['cases'])):
                row,_=rollout(initial,ci,1,folder/'initial_comparison'/f'case{ci:02}');original_rows.append(row)
            initial_comparison=compare(teacher_val,original_rows);best_rank=initial_comparison['rank']
            write(folder/'initial_comparison.json',initial_comparison)
            # Validation is never relabelled into training. New yaw0 episodes are
            # collected at the current student's deficient cases and all standing groups.
            for ci,case in enumerate(request['cases']):
                standing=case['target'][0]==case['target'][1]==0
                if standing or not initial_comparison['rows'][ci]['fidelity_passed']:
                    _,arrays=rollout(initial,ci,0,folder/'initial_replay'/f'case{ci:02}')
                    parts.append(episode(arrays,teacher,case,ci,5000+ci,0))
        for ri in range(request['limits']['rounds']):
            guard(); rd=folder/f'round_{ri:02}'
            dataset=make_dataset(parts,request,rd/'dataset',dict(teacher_states=True,student_replay_rounds=ri,
                teacher_controls_student=False,source_trajectories='per-episode NPZ/JSON in this task; SHA preserved',
                teacher_failure_states='diagnostic labels only; no expert-success or hardware claim'))
            fit=pipeline.fit(dataset,rd/'fit',steps=request['limits']['steps_per_round'],
                wall_s=min(request['limits']['fit_wall_s'],max(1,int(deadline-time.monotonic()-60))),
                seed=request['seed'],validation_every=200,patience=5,initial_student=best)
            used+=fit['offline_optimizer_updates']; student, _=load(rd/'fit/best_validation.pt','MotionZeroStudent')
            student.eval().requires_grad_(False); rows=[]
            for ci in range(len(request['cases'])):
                row,_=rollout(student,ci,1,rd/'comparison'/f'case{ci:02}'); rows.append(row)
            comparison=compare(teacher_val,rows); write(rd/'comparison.json',comparison)
            history.append(dict(round=ri,offline_updates=fit['offline_optimizer_updates'],comparison=comparison))
            candidate=rd/'fit/best_validation.pt'
            if best_rank is None or comparison['rank']>best_rank:
                best=candidate;best_rank=comparison['rank'];best_dataset=dataset
            write(folder/'rounds.json',history)
            print(json.dumps(dict(stage='ROUND_COMPLETE',round=ri,passed=comparison['passed'],count=comparison['count'],offline_updates=used)),flush=True)
            if comparison['all_fidelity_passed'] or ri+1==request['limits']['rounds']:
                break
            for ci,case in enumerate(request['cases']):
                _,arrays=rollout(student,ci,0,rd/'replay'/f'case{ci:02}')
                parts.append(episode(arrays,teacher,case,ci,10000+ri*1000+ci,0))
        # The test yaw group is never queried for model/epoch selection or replay.
        selected,_=load(best,'MotionZeroStudent');selected.eval().requires_grad_(False)
        trows=[];srows=[]
        for ci in request['test_case_indices']:
            tr,ta=rollout(teacher,ci,2,folder/'test_teacher'/f'case{ci:02}')
            sr,sa=rollout(selected,ci,2,folder/'test_student'/f'case{ci:02}')
            trows.append(tr);srows.append(sr)
        test_comparison=compare(trows,srows);write(folder/'test_comparison.json',test_comparison)
        exported=pipeline.export_onnx(best,request['parent_spec'],best_dataset,Path(request['export_directory']))
        if any(sha(path)!=digest for path,digest in source_hashes.items()):
            raise RuntimeError('read-only source input changed')
        status='COMPLETED'
    except BaseException as exc:
        error=repr(exc);write(folder/'error.json',dict(error=error,traceback=traceback.format_exc()));raise
    finally:
        result=dict(status=status,error=error,teacher=request['teacher'],teacher_sha256=request['teacher_sha256'],
            solver_profile=selected_solver,plant_profile=plant_readback['version'],objective='teacher_fidelity',
            selected_student=str(best) if best else None,student_sha256=sha(best) if best else None,
            selected_dataset=str(best_dataset) if best_dataset else None,initial_comparison=initial_comparison,rounds=history,offline_optimizer_updates=used,
            online_PPO_updates=0,wall_s=time.monotonic()-start,physical_seconds=physical,scored_episodes=episodes,
            test_comparison=test_comparison,export=exported,source_unchanged=all(sha(p)==s for p,s in source_hashes.items()),
            native_confirmation='NOT_RUN',hardware_validation='NOT_RUN',capability_certified=False)
        write(folder/'result.json',result)
        write(folder/'budget.json',dict(status=status,offline_optimizer_updates=used,online_PPO_updates=0,
            physical_seconds=physical,scored_episodes=episodes,wall_s=result['wall_s'],limits=request['limits']))
    return result
