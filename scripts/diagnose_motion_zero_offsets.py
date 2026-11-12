"""Bounded CPU MuJoCo encoder-zero hypothesis, using the existing log simulator.

The logged stationary mean is ASSUMED to describe the canonical .34m pose.
Its difference from canonical q is injected into feedback, never physical q.
This diagnoses sensitivity; it does not identify real calibration or ownership.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from onnx.reference import ReferenceEvaluator

from audit_motion_deployment_csv import simulate, write
from replay_motion31_shadow_csv import read_csv, sha

LEGS=[0,1,3,4]


def derive_bias(x,t,window,canonical):
    x=np.asarray(x);t=np.asarray(t);q=np.asarray(canonical,dtype=float)
    if x.ndim!=2 or x.shape!=(len(t),31) or q.shape!=(4,) or not np.isfinite(x).all() or not np.isfinite(q).all():
        raise ValueError('finite chronological obs31 and four canonical leg angles required')
    if len(t)<2 or not np.isfinite(t).all() or (np.diff(t)<=0).any():raise ValueError('strictly increasing log timestamps required')
    lo,hi=map(float,window)
    if not np.isfinite([lo,hi]).all() or lo>=hi:raise ValueError('finite increasing source window required')
    mask=(t>=lo)&(t<=hi)
    if mask.sum()<20:raise ValueError('at least20 source frames required')
    measured=x[mask,6:10].astype(float)*1.5
    return dict(requested_window_s=[lo,hi],actual_window_s=[float(t[mask][0]),float(t[mask][-1])],
                frames=int(mask.sum()),canonical_q_rad=q.tolist(),reported_mean_q_rad=measured.mean(0).tolist(),
                reported_std_q_rad=measured.std(0).tolist(),bias_rad=(measured.mean(0)-q).tolist(),
                bias_deg=np.rad2deg(measured.mean(0)-q).tolist(),
                observed_height_command_range_m=[float(x[mask,24].min()*.5),float(x[mask,24].max()*.5)],
                observed_vx_command_range_mps=[float(x[mask,22].min()*4),float(x[mask,22].max()*4)],
                actual_height='UNKNOWN: no measured height in CSV',controller_owner='UNKNOWN: no owner in CSV',
                interpretation='q_reported=q_true+b ONLY IF the logged real pose equals the canonical simulator reset; not measured encoder offsets')


def scenarios(bias,seconds):
    b=np.asarray(bias,dtype=float)
    if b.shape!=(4,) or not np.isfinite(b).all():raise ValueError('four finite offsets required')
    settings=[('baseline',0,0),('actor_only',1,0),('pd_only',0,1),('actor_and_pd',1,1),
              ('half_actor_and_pd',.5,.5),('double_actor_and_pd',2,2),('opposite_actor_and_pd',-1,-1)]
    return [(name,None,'zero','bridge',seconds,{} if not(a or p) else
             dict(actor_leg_zero_bias_rad=(a*b).tolist(),pd_leg_zero_bias_rad=(p*b).tolist()))
            for name,a,p in settings]


def trajectory_metrics(arrays):
    t=np.asarray(arrays['time']);q=np.asarray(arrays['requested'])[:,LEGS]
    dq=np.asarray(arrays['dq'])[:,LEGS];state=np.asarray(arrays['state'])
    delta=np.diff(q,axis=0);events={}
    for label,hit in [('leg_dq_over_point5',np.max(abs(dq),axis=1)>.5),
                      ('leg_dq_over2',np.max(abs(dq),axis=1)>2),
                      ('leg_dq_over5',np.max(abs(dq),axis=1)>5),
                      ('tilt_over_point1',np.max(abs(state[:,3:5]),axis=1)>.1)]:
        ids=np.flatnonzero(hit);events[label]=float(t[ids[0]]) if len(ids) else None
    return dict(samples=len(t),leg_target_delta_rms_rad=np.sqrt(np.mean(delta*delta,axis=0)).tolist(),
                leg_velocity_abs_p95_rad_s=np.percentile(abs(dq),95,axis=0).tolist(),
                mean_saturation_per_channel=(abs(arrays['mean'])>=.99).mean(0).tolist(),
                motor_saturation_per_channel=(abs(arrays['torque'])>=np.array([40,40,5.5,40,40,5.5])*.99).mean(0).tolist(),
                events_s=events,physical_sampling_hz=100,
                scope='actual sampled simulated q/dq/torque, not MCU timing or real-hardware identification')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--plant-adapter',type=Path,required=True)
    p.add_argument('--canonical-result',type=Path,required=True,help='existing same-Plant normal_zero.json, read-only')
    p.add_argument('--window',type=float,nargs=2,default=[14.13,15.13])
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--wall-seconds',type=float,default=180.)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    a=p.parse_args();start=time.monotonic()
    if not np.isfinite([a.seconds,a.wall_seconds]).all() or not 2<=a.seconds<=15 or not 0<a.wall_seconds<=300:
        raise ValueError('2..15 physics seconds per case,0..300 wall seconds required')
    _,x,y,t,columns=read_csv(a.csv)
    if abs(x).max()>100 or abs(y).max()>1.0001:raise ValueError('invalid recorded data; no clipping or synthetic repair')
    manifest=json.loads((a.package/'manifest.json').read_text())
    expected='5b67781bdaa86b666683e315f685a0f19c45fee918da70439ea3bb43d0fbb03a'
    if sha(a.package/'actor.onnx')!=expected or manifest['actor_onnx_sha256']!=expected:raise ValueError('this diagnostic binds the requested Motion500 deployment model')
    canonical=json.loads(a.canonical_result.read_text())
    if canonical['case']!='normal_zero' or canonical['command_mode']!='zero':raise ValueError('same-Plant canonical zero case required')
    ref=np.asarray(canonical['readback']['reset_leg_q_rad'])
    measured=derive_bias(x,t,a.window,ref);cases=scenarios(measured['bias_rad'],a.seconds)
    plan=dict(version='motion31_log_inferred_zero_offset_v1',status='PLAN_ONLY',
              cases=cases,offset_estimate=measured,maximum_cases=7,maximum_physics_s=7*a.seconds,
              wall_limit_s=a.wall_seconds,storage_limit_bytes=128*2**20,online_PPO=0,optimizer_steps=0,
              true_initial_height_m=.34,vx_command_mps=0,yaw_command_rad_s=0,height_command_m=.34,
              sensor_bias_semantics='only active-leg positions q_reported=q_true+b; dq/IMU/physical initial pose unchanged',
              previous_semantics='actual PD-adopted target in reported coordinates; reset bridge=q_true_initial+b_PD; physical-equivalent target=adopted-b_PD',
              fixed_physics='existing Motion500 current-Plant;1/5/10ms;PD40/2/.2;40/5.5Nm;no new springs or compensation',
              not_hardware_reproduction=True,certificate_eligible=False,
              missing_task_flags_assumed={str(i):float(x[0,i]) for i in range(25,31) if i not in columns['observation']})
    if not a.run:print(json.dumps(plan,indent=2,ensure_ascii=False));return
    if a.output_dir.exists():raise FileExistsError('refuse to overwrite diagnostic output')
    dependencies=[a.csv,a.package/'actor.onnx',a.package/'manifest.json',a.canonical_result,
                  a.plant_adapter,a.plant_adapter.parent/'request.json',Path(__file__),
                  Path('scripts/audit_motion_deployment_csv.py'),Path('scripts/compare_motion_timing_profiles.py'),
                  Path('src/native_stand/students/mcu_interface.py'),Path('src/native_stand/students/real2sim_delay.py')]
    inputs={str(path.resolve()):sha(path) for path in dependencies}
    plant_request=json.loads((a.plant_adapter.parent/'request.json').read_text())
    inputs[str(Path(plant_request['xml']).resolve())]=sha(plant_request['xml'])
    a.output_dir.mkdir(parents=True,exist_ok=False)
    write(a.output_dir/'request.json',{**plan,'status':'AUTHORIZED_BOUNDED_DIAGNOSTIC',
         'authorization':'User requests simulating encoder-zero mismatch inferred from the .34m pre-oscillation log; bounded CPU MuJoCo only, no training or hardware',
         'inputs_sha256':inputs})
    try:
        runtime=ReferenceEvaluator(str(a.package/'actor.onnx'))
        infer=lambda obs:np.concatenate([runtime.run(None,{'observation':z[None].astype(np.float32)})[0] for z in obs])
        results=simulate(a.plant_adapter,a.package,a.output_dir,x,t,np.ones(len(t),bool),infer,manifest,
                         a.wall_seconds-(time.monotonic()-start),scenarios=cases,record_full_state=True)
        for result in results:
            with np.load(a.output_dir/(result['case']+'.npz'),allow_pickle=False) as arrays:
                result['zero_offset_diagnostics']=trajectory_metrics(arrays)
                # The physical starting pose is the canonical pose in every case.
                np.testing.assert_allclose(result['first']['q'][0:2]+result['first']['q'][3:5],ref,atol=1e-8,rtol=0)
                desired=ref+np.asarray(result['readback']['overrides'].get('actor_leg_zero_bias_rad',[0]*4))
                np.testing.assert_allclose(np.asarray(result['first']['input'])[6:10]*1.5,desired,atol=1e-7,rtol=0)
                if result['case']=='baseline':
                    # Regression against the previously executed same-Plant trajectory.
                    with np.load(a.canonical_result.with_suffix('.npz'),allow_pickle=False) as previous:
                        n=min(len(arrays['time']),len(previous['time']))
                        result['zero_offset_diagnostics']['old_baseline_max_abs_error']={k:float(abs(arrays[k][:n]-previous[k][:n]).max()) for k in ('actor_input','mean','adopted','q','dq','torque','state')}
            write(a.output_dir/(result['case']+'.json'),result)
        write(a.output_dir/'comparison.json',results)
        clips=[r['case'] for r in results]
        write(a.output_dir/'manifest.json',dict(version='motion31_deployment_log_visual_replay_v1',
             xml=plant_request['xml'],xml_sha256=sha(plant_request['xml']),clips=clips,
             clip_sha256={name:sha(a.output_dir/(name+'.npz')) for name in clips},
             CSV_sha256=sha(a.csv),actor_sha256=expected,
             semantics='stored autonomous ONNX closed loops with explicit sensor calibration hypotheses; viewer only; not a real-hardware state replay'))
        unchanged=all(sha(path)==digest for path,digest in inputs.items())
        if not unchanged:raise ValueError('input source identity changed during diagnostic')
        size=sum(path.stat().st_size for path in a.output_dir.rglob('*') if path.is_file())
        if size>plan['storage_limit_bytes']:raise RuntimeError('diagnostic output budget exceeded')
        write(a.output_dir/'terminal.json',dict(status='COMPLETED',wall_s=time.monotonic()-start,
              physical_cases=len(results),physics_s=sum(r['seconds'] for r in results),output_bytes=size,
              online_PPO=0,optimizer_steps=0,inputs_unchanged=unchanged,hardware_identified=False))
        print(json.dumps([dict(case=r['case'],seconds=r['seconds'],failure=r['failure'],
              pitch_peak=r['pitch_abs_peak'],roll_peak=r['roll_abs_peak'],vx_mean=r['vx_mean'],
              leg_target_delta_rms_rad=r['zero_offset_diagnostics']['leg_target_delta_rms_rad'],
              leg_dq_p95=r['leg_dq_abs_P95_rad_s']) for r in results],indent=2))
    except Exception as exc:
        write(a.output_dir/'terminal.json',dict(status='FAILED',error=repr(exc),wall_s=time.monotonic()-start,
              online_PPO=0,optimizer_steps=0,automatic_retry=False));raise


if __name__=='__main__':main()
