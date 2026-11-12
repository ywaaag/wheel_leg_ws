"""Bounded coarse/local MuJoCo handover hypothesis search, no parameter certification.

Reuses log-constrained initialization and the existing execution diagnostic.
Default plan writes nothing. --run requires user authorization for MuJoCo physics.
"""
import argparse
import itertools
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from diagnose_motion_handover import (
    DEFAULT_CP, DEFAULT_SPEC, ROOT, decode_log, initialize_from_log,
    model_plan, new_experiment_name, read_csv, run_case, sha,
    validate_new_experiment_name, LEG, SCALE,
)
from native_stand.students.real2sim_handover import signal_metrics


def configuration(target, feedback, rise, dq=0, wheel='transport11_7'):
    """Times are integer milliseconds and hypothetical, not identified constants."""
    if not (0 <= target <= 50 and 0 <= feedback <= 25 and 0 <= rise <= 15 and 0 <= dq <= 15):
        raise ValueError('search domain exceeded')
    if wheel not in ('transport11_7', 'transport7_11', 'rise11_7'):
        raise ValueError('wheel hypothesis unknown')
    delay = [target, target, 0, target, target, 0]
    tau = [rise, rise, 0, rise, rise, 0]
    if wheel == 'rise11_7': tau[2], tau[5] = 11, 7
    elif wheel == 'transport11_7': delay[2], delay[5] = 11, 7
    else: delay[2], delay[5] = 7, 11
    return dict(name=f'leg{target}_fb{feedback}_rise{rise}_dq{dq}_{wheel}',
                target_delay_ms=delay, pd_feedback_delay_ms=feedback,
                torque_tau_ms=tau, dq_delay_ms=dq, imu_delay_ms=0,
                search_values=dict(target=target, feedback=feedback, rise=rise, dq=dq, wheel=wheel))


def coarse_candidates():
    return [(configuration(t, f, r), cache)
            for t, f, r, cache in itertools.product((0, 10, 20, 40), (0, 5, 10, 20), (0, 5),
                ('bridge_actual', 'logged_seed', 'logged_as_executed'))]


def key(cfg, cache):
    return json.dumps([cfg['search_values'], cache], sort_keys=True)


def local_candidates(leaders):
    rows = {}
    for row in leaders:
        v = row['configuration']['search_values']; cache = row['cache_hypothesis']
        proposals = []
        for field, low, high in [('target', 0, 50), ('feedback', 0, 25), ('rise', 0, 15), ('dq', 0, 15)]:
            for delta in (-5, 5):
                w = {**v, field: max(low, min(high, v[field]+delta))}
                proposals.append(w)
        for wheel in ('transport7_11', 'rise11_7'): proposals.append({**v, 'wheel':wheel})
        for w in proposals:
            c = configuration(**w); rows[key(c, cache)] = (c, cache)
    return list(rows.values())


def compare_window(real_t, real_obs, real_action, arrays, start, stop):
    """Fixed windows, no time-warp/phase fitting or target-dependent window shrinking."""
    rm = (real_t >= start) & (real_t < stop)
    sm = (arrays['time'] >= start) & (arrays['time'] < stop)
    rt = real_t[rm]
    if rm.sum() < 8: raise ValueError('real comparison window missing')
    coverage = float(sm.sum()) / max(1, round((stop-start)*100))
    if sm.sum() < 8:
        return dict(score=100., coverage=coverage, missing_reason='simulation terminated before fixed window')
    rg = real_obs[rm,3:6]
    sg = arrays['gravity'][sm]
    real = signal_metrics(rt, real_action[rm]*SCALE, real_obs[rm,10:16]*20., rg)
    sim = signal_metrics(arrays['time'][sm], arrays['target'][sm], arrays['dq_pre'][sm], sg)
    errors = {}
    for field, floor in [('dominant_leg_target_hz', 2.), ('leg_target_p95_minus_p5_rad', .05),
                         ('leg_dq_p95_rad_s', .5), ('leg_target_step_max_rad', .05)]:
        a=np.asarray(real[field]);b=np.asarray(sim[field])
        errors[field]=float(np.mean(abs(a-b)/np.maximum(abs(a),floor)))
    # Signed waveform errors prevent a forward fall from matching a backward fall.
    rp=np.column_stack((np.arctan2(rg[:,2],-rg[:,1]),np.arctan2(rg[:,0],-rg[:,1])))
    sp=np.column_stack((np.arctan2(sg[:,2],-sg[:,1]),np.arctan2(sg[:,0],-sg[:,1])))
    # Missing tail is padded by the last state and penalized separately; never hidden.
    pred=np.column_stack([np.interp(rt,arrays['time'][sm],sp[:,i]) for i in range(2)])
    errors['signed_pitch_roll_rmse']=float(np.mean(np.sqrt(np.mean((pred-rp)**2,axis=0))/
                                                  np.maximum(np.sqrt(np.mean(rp**2,axis=0)),.02)))
    # Compare target/velocity phase coupling and wheel response, not only spectra.
    def correlations(target,velocity):
        a=target[:,LEG]-target[:,LEG].mean(0);b=velocity[:,LEG]-velocity[:,LEG].mean(0)
        return (a*b).mean(0)/np.maximum(np.sqrt((a*a).mean(0)*(b*b).mean(0)),1e-9)
    rc=correlations(real_action[rm]*SCALE,real_obs[rm,10:16]*20.)
    sc=correlations(arrays['target'][sm],arrays['dq_pre'][sm])
    errors['leg_target_dq_phase_correlation']=float(abs(rc-sc).mean()/2.)
    wheel_pred=np.column_stack([np.interp(rt,arrays['time'][sm],arrays['dq_pre'][sm,j]) for j in (2,5)])
    wheel_real=real_obs[rm,10:16][:,[2,5]]*20.
    errors['wheel_dq_signed_rmse']=float(np.mean(np.sqrt(((wheel_pred-wheel_real)**2).mean(0))/
                                                np.maximum(np.sqrt((wheel_real**2).mean(0)),1.)))
    score=float(np.mean(list(errors.values()))+2*max(0.,1-coverage))
    return dict(score=score, coverage=coverage, per_observable=errors, real=real, simulation=sim,
                signed_posture_note='no absolute-angle substitution, phase shift or time warp; last-state padding explicitly penalized')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--spec',type=Path,default=DEFAULT_SPEC);p.add_argument('--checkpoint',type=Path,default=DEFAULT_CP)
    p.add_argument('--xml',type=Path,default=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml'))
    p.add_argument('--wall-seconds',type=float,default=600.)
    p.add_argument('--output-dir',type=Path);p.add_argument('--run',action='store_true');a=p.parse_args()
    if not 30<=a.wall_seconds<=1800:raise ValueError('bounded wall30..1800 required')
    coarse=coarse_candidates()
    plan=dict(version='motion31_handover_coarse_local_search_v1', coarse_cases=len(coarse),
              local_top=4, local_maximum=40, validation_top=6, validation_contexts=3,
              maximum_cases=250, physics_seconds_per_case=2.5, wall_limit_s=a.wall_seconds,
              storage_max_bytes=64*2**20, trace_retention='top6 search candidates plus18 validation trajectories only',
              fit_window_s=[0.,.8], temporal_holdout_s=[.8,1.6],
              fit_start_time_assumption_s=17.06, fit_root_vx_assumption_mps=-.3,
              validation='same log/overlapping initial times, NOT independent hardware validation; no selection on holdout',
              score='equal-weight frequency/amplitude/dq/target-step/signed-posture/target-dq-correlation/wheel-dq errors plus missing-coverage penalty',
              known_wheel_data='7/11ms empty-load combined onset; transport OR rise with unresolved side mapping',
              unmeasured_parameters='all leg timing/feedback/rise values; root velocity; actual previous/execution source',
              unchanged=['mass','inertia','armature','PD','friction','reward','Actor','USD','external MJCF'],
              configuration_grid=[c for c, cache in coarse if cache=='bridge_actual'],
              online_PPO=0, training_export_ready=False)
    if not a.run:print(json.dumps(plan,indent=2));return
    torch.set_num_threads(1)
    _,obs,actions,t,_=read_csv(a.csv)
    spec=json.loads(a.spec.read_text())
    model,refs,actor,identity=model_plan(SimpleNamespace(spec=a.spec,checkpoint=a.checkpoint,xml=a.xml,
        wall_seconds=a.wall_seconds,commands=[0.],yaw_commands=[],viewer=False),spec)
    out=a.output_dir or ROOT/'sim2sim'/new_experiment_name('motion3250_handover_search')
    validate_new_experiment_name(out.name);out.mkdir(parents=True,exist_ok=False)
    def write(name,data):
        with (out/name).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
    inputs={**identity['input_sha256'],str(a.csv.resolve()):sha(a.csv)}
    for path in [Path(__file__),ROOT/'scripts/diagnose_motion_handover.py',
                 ROOT/'src/native_stand/students/real2sim_handover.py',ROOT/'src/native_stand/students/real2sim_delay.py']:
        inputs[str(path.resolve())]=sha(path)
    write('manifest.json',dict(**plan,input_sha256=inputs,checkpoint_sha256=sha(a.checkpoint),
          authorization='user approved constrained coarse/local MuJoCo bug reproduction search',
          initial_torque='UNKNOWN assumed0', contact_matching='UNKNOWN real contact not logged',
          fall_matching='qualitative~2s user observation only; no numerical ground-truth timestamp',
          identity=identity))
    from native_stand.command_v6.completion import write as atomic
    started=time.monotonic();status='FAILED';results=[];validation=[];keepers={};contexts={}
    weights={k:v.clone() for k,v in actor.state_dict().items()}
    budget=dict(version='mujoco_handover_search_budget_v1',attempt_id=uuid.uuid4().hex,status='RUNNING',
                maximum_cases=250,maximum_physics_steps=625000,wall_limit_s=a.wall_seconds,
                storage_max_bytes=64*2**20,online_PPO=0,attempts=[])
    atomic(out/'budget.json',budget)
    def context(event,vx):
        if (event,vx) in contexts:return contexts[event,vx]
        ix=int(np.searchsorted(t,event,side='right')-1)
        if ix<20 or t[-1]-t[ix]<2.5:raise ValueError('log window/prehistory unavailable')
        initial,fit=initialize_from_log(model,refs,obs[ix],vx)
        history=[]
        for j in range(ix-15,ix):
            s=decode_log(obs[j]);history.append((int(np.floor((t[j]-t[ix])*1000)),np.r_[s['gyro'],s['gravity'],s['q'],s['dq']]))
        result=(ix,initial,fit,history);contexts[event,vx]=result;return result
    def evaluate(cfg,cache,phase,event=17.06,vx=-.3):
        if len(budget['attempts'])>=250:raise RuntimeError('case budget exhausted')
        ix,initial,fit,history=context(event,vx)
        summary,arrays=run_case(model,refs,actor,initial,history,obs[ix],cfg=cfg,cache_mode=cache,
            trace_time=t,observations=obs,sample_time=float(t[ix]),duration=2.5,compare_seconds=.8,
            deadline=started+a.wall_seconds)
        name=f'case_{len(budget["attempts"]):03d}'
        row=dict(**summary,name=name,phase=phase,event_sample_time_s=float(t[ix]),assumed_root_vx_mps=vx,
                 initial_fit=fit)
        if arrays:
            row['fit']=compare_window(t-t[ix],obs,actions,arrays,0.,.8)
            row['holdout']=compare_window(t-t[ix],obs,actions,arrays,.8,1.6)
        else:row.update(fit=dict(score=100.,coverage=0.),holdout=dict(score=100.,coverage=0.))
        write(name+'.json',row)
        budget['attempts'].append(dict(name=name,phase=phase,physics_steps=summary['physics_steps'],termination=summary['termination']))
        atomic(out/'budget.json',budget)
        if summary['termination']=='WALL_BUDGET':raise TimeoutError('search wall bound')
        if phase!='validation':
            results.append(row);keepers[name]=arrays
            best=sorted(results,key=lambda r:r['fit']['score'])[:6]
            for k in list(keepers):
                if k not in {r['name'] for r in best}:del keepers[k]
        else:
            validation.append(row)
            np.savez_compressed(out/(name+'.npz'),**arrays)
        if sum(q.stat().st_size for q in out.iterdir() if q.is_file())>budget['storage_max_bytes']:
            raise RuntimeError('storage budget exhausted')
        return row
    try:
        with torch.inference_mode():
            visited=set()
            for cfg,cache in coarse:
                evaluate(cfg,cache,'coarse');visited.add(key(cfg,cache))
            print('coarse completed96; local search uses fit window only',flush=True)
            leaders=sorted(results,key=lambda r:r['fit']['score'])[:4]
            for cfg,cache in local_candidates(leaders):
                if key(cfg,cache) not in visited:
                    evaluate(cfg,cache,'local');visited.add(key(cfg,cache))
            top=sorted(results,key=lambda r:r['fit']['score'])[:6]
            write('frozen_shortlist.json',dict(selection='fit score only; holdout NOT used',candidates=top))
            for row in top:
                np.savez_compressed(out/(row['name']+'.npz'),**keepers[row['name']])
            print(f'search completed{len(results)}; shortlist frozen before validation',flush=True)
            for row in top:
                for event,vx in [(17.06,0.),(17.,-.3),(17.,0.)]:
                    v=evaluate(row['configuration'],row['cache_hypothesis'],'validation',event,vx)
                    v['frozen_candidate_name']=row['name']
            status='COMPLETED'
    finally:
        unchanged=all(torch.equal(v,actor.state_dict()[k]) for k,v in weights.items())
        budget.update(status=status,wall_used_s=time.monotonic()-started,
                      physics_steps_used=sum(r['physics_steps'] for r in budget['attempts']))
        atomic(out/'budget.json',budget)
        write('terminal_report.json',dict(status=status,search=results,validation=validation,model_unchanged=unchanged,
              online_PPO=0,hardware_identified=False,training_export_ready=False))
        if not unchanged:raise RuntimeError('Actor changed')
        for path,digest in inputs.items():
            if sha(path)!=digest:raise RuntimeError('input changed '+path)
    print('OUTPUT='+str(out.resolve()),flush=True)


if __name__=='__main__':main()
