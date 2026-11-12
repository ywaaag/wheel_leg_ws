"""Finite CPU MuJoCo calibration hypotheses on the shared current MCU chain.

The old Motion500 diagnostic remains read-only. This entry point uses the
current post-PD execution and shared precision/plant, never trains a policy.
"""
import argparse
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from native_stand.students.data import sha
from native_stand.students.motion_execution_uncertainty import nominal, validate
from native_stand.students.motion_mcu_execution import LEGS, pd_and_motor
from native_stand.students import mujoco_solver
from diagnose_motion_closed_loop_robustness import (OnnxMean, calibration_biases,
    run_case, write, SOURCE, SOURCE_SHA, SMALL, SMALL_SHA, SPEC, XML)
from diagnose_motion_zero_offsets import derive_bias, scenarios
from motion_mcu_playback import configuration, declared, apply_current_plant
from replay_motion31_shadow_csv import read_csv

ROOT = Path(__file__).resolve().parents[1]
NEW_SMALL = ROOT/'deployment_protocol/2026_10_26-20_29__motion4500_precise_small31_onnx/actor.onnx'
NEW_SMALL_SHA = '776201d861dbf1f00b6824f2c2615068a3206eb45b38529e65e7591bc150e617'
LOG = ROOT/'logs/Ozone_DataSampling_261024_17_35.csv'
LOG_SHA = '2a83c6eb29e70606e5e99c20d5bd9a2bc79e8fb2fb8eb249c3bf6bde6e9938db'


def cases(bias, matrix='zero_offsets_v1'):
    if matrix == 'combined_stress_v1':
        # Keep the predeclared mild stress identical to the distillation test.
        stress=dict(nominal(),name='mild_combination',family='combined_hypotheses',
            initial_vx_mps=.05,initial_pitch_rate_rad_s=.1,PD_age_ms=12,
            motor_delay_ms=[v+2 for v in nominal()['motor_delay_ms']],spring_scale=1.1)
        validate(stress)
        rows=[]
        for name,physical,scale in [('nominal',nominal(),0.),('log_bias_only',nominal(),1.),
                ('stress_only',stress,0.),('stress_half_bias',stress,.5),
                ('stress_full_bias',stress,1.),('stress_double_bias',stress,2.)]:
            value=(np.asarray(bias,dtype=float)*scale).tolist()
            cal=dict(version='motion31_split_encoder_offsets_v1',
                actor_leg_zero_bias_rad=value,pd_leg_zero_bias_rad=value)
            calibration_biases(physical,cal)
            rows.append(dict(case_id=name,calibration=cal,uncertainty=dict(physical)))
        return rows
    if matrix != 'zero_offsets_v1':raise ValueError('unknown calibration diagnostic matrix')
    rows=[]
    for name,_,_,_,_,change in scenarios(bias,30.):
        calibration=dict(version='motion31_split_encoder_offsets_v1',
            actor_leg_zero_bias_rad=change.get('actor_leg_zero_bias_rad',[0.]*4),
            pd_leg_zero_bias_rad=change.get('pd_leg_zero_bias_rad',[0.]*4))
        calibration_biases(nominal(),calibration)
        rows.append(dict(case_id=name,calibration=calibration))
    return rows


def trajectory_audit(a,case,q0):
    actor,pd=calibration_biases(nominal(),case['calibration'])
    np.testing.assert_allclose(a['actor_input'][0,6:10]*1.5,q0[LEGS]+actor,rtol=0,atol=1e-7)
    np.testing.assert_allclose(a['actor_input'][0,16:20]*1.5,q0[LEGS]+pd,rtol=0,atol=1e-7)
    ae=float(np.max(abs(a['actor_input'][:,6:10]*1.5-a['actor_true_q'][:,LEGS]-actor)))
    pe=float(np.max(abs(a['pd_feedback_q'][:,LEGS]-a['pd_true_q'][:,LEGS]-pd)))
    prev=a['adopted_target'][:-1,[0,1,3,4,2,5]]/np.array([1.5]*4+[100.]*2)
    cache=float(np.max(abs(a['actor_input'][1:,16:22]-prev))) if len(prev) else 0.
    with torch.inference_mode():
        values=pd_and_motor(torch.tensor(a['pd_feedback_q'],dtype=torch.float32),
            torch.tensor(a['pd_feedback_dq'],dtype=torch.float32),torch.tensor(a['adopted_target'],dtype=torch.float32))
    errors={name:float(np.max(abs(a[key]-value.numpy()))) for name,key,value in
        [('PD','pd_request',values[0]),('PD_limited','pd_limited',values[1]),
         ('compensation','spring_compensation',values[2]),('driver','driver',values[3])]}
    assert ae<1e-6 and pe<1e-6 and cache<1e-6 and max(errors.values())<1e-4
    assert (a['owner']==1).all() and np.isfinite(a['state']).all()
    # Compare in FP32: normalization in the real encoder also uses FP32.
    dq_error=float(np.max(abs(a['actor_input'][:,10:16]-a['actor_true_dq'].astype(np.float32)*.05)))
    assert dq_error<1e-6
    return dict(status='PASSED',actor_offset_error_rad=ae,PD_offset_error_rad=pe,
        previous_normalized_error=cache,velocity_encoding_error=dq_error,controller_numeric_errors_nm=errors)


def diagnostics(a):
    delta=np.diff(a['requested_target'][:,LEGS],axis=0)
    return dict(leg_target_step_rms_rad=np.sqrt(np.mean(delta**2,axis=0)).tolist() if len(delta) else None,
        leg_dq_abs_p95_rad_s=np.quantile(abs(a['dq'][:,LEGS]),.95,axis=0).tolist(),
        leg_dq_abs_max_rad_s=np.max(abs(a['dq'][:,LEGS]),axis=0).tolist(),
        first_tilt_over_point1_s=next((float(t) for t,s in zip(a['time'],a['state']) if max(abs(s[4:6]))>.1),None),
        mean_saturation_fraction=(abs(a['mean'])>=.99).mean(0).tolist(),
        sampled_torque_scope='100Hz held samples; not full200Hz PD spectrum or measured hardware torque')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=30.)
    parser.add_argument('--wall-seconds',type=float,default=1200.)
    parser.add_argument('--matrix',choices=['zero_offsets_v1','combined_stress_v1'],default='zero_offsets_v1')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not 10<=args.seconds<=30 or not 60<=args.wall_seconds<=1800:
        parser.error('10..30 physics seconds and60..1800 wall seconds required')
    expected={SOURCE:SOURCE_SHA,SMALL:SMALL_SHA,NEW_SMALL:NEW_SMALL_SHA,LOG:LOG_SHA}
    for path,digest in expected.items():
        if sha(path)!=digest:raise ValueError('input identity differs: '+str(path))
    spec=json.loads(SPEC.read_text());pose=spec['mcu_execution']['pose']['joint_pos']
    q0=np.array([pose[n] for n in spec['action']['order']])
    _,x,_,t,columns=read_csv(LOG);estimate=derive_bias(x,t,[14.13,15.13],q0[LEGS])
    matrix=cases(estimate['bias_rad'],args.matrix)
    yaw_conditions=[0.,.015,-.015] if args.matrix=='combined_stress_v1' else [0.]
    count=3*len(matrix)*len(yaw_conditions)
    paths=list(expected)+[SPEC,Path(spec['asset_path']),Path(spec['colleague_config_path']),XML,
        Path(__file__),ROOT/'scripts/diagnose_motion_closed_loop_robustness.py',
        ROOT/'scripts/motion_mcu_playback.py',Path(mujoco_solver.__file__),
        ROOT/'src/native_stand/students/motion_mcu_execution.py',
        ROOT/'src/native_stand/students/mcu_interface.py',ROOT/'src/native_stand/students/interface.py']
    identities={str(p.resolve()):sha(p) for p in paths}
    models=[('Small4500',NEW_SMALL),('Teacher4500',SOURCE),('DeployedMotion500',SMALL)]
    plan=dict(version='motion31_current_mcu_combined_stress_v1' if args.matrix=='combined_stress_v1' else 'motion31_current_mcu_log_calibration_v1',status='PLAN_ONLY',
        authorization='user explicitly requests CPU closed-loop model capability tests under current MCU execution hypotheses',
        models=[dict(name=n,path=str(p),sha256=sha(p)) for n,p in models],
        source_log_estimate=estimate,cases=matrix,inputs_sha256=identities,
        matrix=args.matrix,initial_yaw_conditions_rad=yaw_conditions,
        solver_profile='closure_contact_tight_v1',plant_profile='motion_mcu_currentplant_v1',
        task=dict(vx=0,yaw=0,height=.34,skill='Motion',jump='NONE'),
        actual_initial_q6_rad=q0.tolist(),actual_initial_height_m=.34,
        previous='PD-adopted target in reported PD coordinates, never raw/shadow; initial physical target plus PD bias',
        compensation='current C-equivalent formula uses PD-reported q; independent physical spring uses true q',
        initial_motor='shared MCU initial compensation from PD-reported q; hypothesis, not measured LQR torque/queue',
        comparisons='same new chain across models; old report differs in solver and execution, not a chain-only A/B',
        limits=dict(cases=count,seconds_per_case=args.seconds,physics_seconds=count*args.seconds,
            wall_seconds=args.wall_seconds,storage_bytes=(256 if args.matrix=='combined_stress_v1' else 128)*2**20),
        online_PPO=0,optimizer_steps=0,certificate=False,hardware_identified=False)
    if not args.run:print(json.dumps(plan,ensure_ascii=False,indent=2));return
    if args.output_dir.exists():raise FileExistsError('refuse to overwrite existing diagnostic')
    if shutil.disk_usage(args.output_dir.parent).free<512*2**20:raise RuntimeError('insufficient diagnostic/settlement reserve')
    args.output_dir.mkdir(parents=True,exist_ok=False);plan['status']='AUTHORIZED_BOUNDED_DIAGNOSTIC'
    write(args.output_dir/'request.json',plan)
    start=time.monotonic();deadline=start+args.wall_seconds;results=[];total=0.;status='FAILED';error=None;partial=None
    try:
        import mujoco
        from validate_native_mujoco import binding
        from native_stand.students.model import load
        torch.set_num_threads(1)
        cfg=configuration(SimpleNamespace(leg_armature_m=.012,zero_latency=False),declared(spec),spec)
        model=mujoco.MjModel.from_xml_path(str(XML));refs=binding(model)
        plant=apply_current_plant(model,refs,spec,cfg)
        solver=mujoco_solver.apply(model,refs,plan['solver_profile'])
        write(args.output_dir/'plant_readback.json',dict(configuration=cfg,plant=plant,solver=solver,
             bounds_leg_rad=model.jnt_range[refs['joints']].tolist(),motor_transport='post-PD'))
        teacher,_=load(SOURCE);teacher.eval().requires_grad_(False)
        actors={'Small4500':OnnxMean(NEW_SMALL),'Teacher4500':teacher,'DeployedMotion500':OnnxMean(SMALL)}
        for name,case,yaw in [(n,c,y) for n,_ in models for c in matrix for y in yaw_conditions]:
                if time.monotonic()>deadline:raise TimeoutError('diagnostic wall budget')
                if sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())>plan['limits']['storage_bytes']-8*2**20:
                    raise RuntimeError('diagnostic storage reserve')
                recorder=mujoco_solver.ConstraintRecorder();tail=[]
                def observe(m,d,r):
                    values=recorder(m,d,r)
                    values.update(qpos_full=d.qpos.copy(),qvel_full=d.qvel.copy())
                    tail.append(dict(time=float(d.time),qpos=d.qpos.tolist(),qvel=d.qvel.tolist()))
                    if len(tail)>20:tail.pop(0)
                    return values
                key=name+'__'+case['case_id']+(f'__yaw{yaw:+.3f}' if len(yaw_conditions)>1 else '')
                partial=dict(case=key,physical_seconds_lower=0.)
                uncertainty=case.get('uncertainty',nominal())
                try:
                    with torch.inference_mode():
                        row,a=run_case(model,refs,actors,cfg,uncertainty,name,args.seconds,deadline,
                            calibration=case['calibration'],constraint_observer=observe,initial_yaw=yaw)
                except BaseException as exc:
                    lower=recorder.last_integrated_time_s
                    physical=getattr(exc,'physical_seconds',None)
                    partial=dict(case=key,physical_seconds=physical,physical_seconds_lower=lower,
                        physical_seconds_upper=(min(args.seconds,lower+.01) if physical is None else physical),
                        error=repr(exc),runtime_fault=True,tail=tail,automatic_retry=False)
                    write(args.output_dir/(key+'.fault.json'),partial);raise
                total+=row['physical_seconds'];partial=None
                row.update(case_id=case['case_id'],model_name=name,initial_yaw_rad=yaw,diagnostics=diagnostics(a),
                    interface_audit=trajectory_audit(a,case,q0),contact_readback=recorder.contact_readback(),
                    solver_profile=plan['solver_profile'],plant_profile=plan['plant_profile'],certificate=False)
                trace=args.output_dir/(key+'.npz');np.savez_compressed(trace,**a)
                row.update(trace=trace.name,trace_sha256=sha(trace));write(trace.with_suffix('.json'),row);results.append(row)
                write(args.output_dir/(key+'.settlement.json'),dict(physical_seconds=row['physical_seconds'],optimizer_steps=0,online_PPO=0))
                print(json.dumps(dict(model=name,case=case['case_id'],seconds=row['physical_seconds'],
                    yaw=yaw,termination=row['termination'],vx=row['windows']['steady_2_to_30'].get('vx_mean_mps')),ensure_ascii=False),flush=True)
        if not all(sha(p)==s for p,s in identities.items()):raise RuntimeError('source identity changed')
        status='COMPLETED'
    except BaseException as exc:
        error=repr(exc)
    finally:
        size=sum(p.stat().st_size for p in args.output_dir.rglob('*') if p.is_file())
        final=dict(status=status,error=error,results=results,completed_cases=len(results),expected_cases=count,
            physical_seconds_completed=total,partial_attempt=partial,wall_s=time.monotonic()-start,
            output_bytes=size,optimizer_steps=0,online_PPO=0,
            inputs_unchanged=all(sha(p)==s for p,s in identities.items()),certificate=False,automatic_retry=False)
        write(args.output_dir/'result.json',final)
        write(args.output_dir/'budget.json',{k:v for k,v in final.items() if k!='results'})
        print(json.dumps({k:v for k,v in final.items() if k!='results'},ensure_ascii=False,indent=2))
    if status!='COMPLETED':raise RuntimeError(error)


if __name__=='__main__':main()
