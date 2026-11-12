"""Offline q/dq consistency profiles; NOT absolute latency identification.

Fit chronological intervals only. With reported dq approximated by a delayed
first-order observation of dq_true, integrate its inverse over a q interval:
  delta_q = gain * (integral(v_observed(t+relative_age))
                   + tau * delta(v_observed(t+relative_age))).
Positive relative_age means dq is older than q. Offline alignment can inspect
both signs; these aligned values are NEVER sent into a policy. Common absolute
age, true motor response and encoder zero are unidentifiable by this equation.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from replay_motion31_shadow_csv import read_csv,sha

AGES=list(range(-10,11))
TAUS=[0.,1.,2.,3.,5.,8.,10.,15.,20.]
LEGS=[0,1,3,4]


def integral_piecewise(t,v,s):
    t=np.asarray(t,float);v=np.asarray(v,float);s=np.asarray(s,float)
    if t.ndim!=1 or v.shape!=t.shape or len(t)<2 or not np.isfinite(np.r_[t,v,s]).all() or np.any(np.diff(t)<=0):
        raise ValueError('finite chronological scalar signal required')
    if (s<t[0]-1e-10).any() or (s>t[-1]+1e-10).any():raise ValueError('query outside recorded partition')
    k=np.clip(np.searchsorted(t,s,side='right')-1,0,len(t)-2);dt=s-t[k]
    slope=np.diff(v)/np.diff(t)
    cumulative=np.r_[0,np.cumsum((v[:-1]+v[1:])*.5*np.diff(t))]
    return cumulative[k]+v[k]*dt+.5*slope[k]*dt**2


def intervals(t,q,v,begin,end,age,tau):
    t,q,v=map(lambda z:np.asarray(z,float),(t,q,v))
    groups=[]
    for stride in (1,2,4):
        left=t[:-stride];right=t[stride:];shift=age*.001
        # Neither the q endpoints nor the aligned dq support may cross a
        # chronological fit/validation boundary. No padding or fake samples.
        margin=float(np.max(np.diff(t)))+.010+1e-6
        keep=(left>=begin+margin)&(right<=end-margin)&(left+shift>=begin)&(right+shift<=end)
        left=left[keep];right=right[keep]
        if len(left)<8:continue
        durations=right-left
        raw=integral_piecewise(t,v,right+shift)-integral_piecewise(t,v,left+shift)
        delta=np.interp(right+shift,t,v)-np.interp(left+shift,t,v)
        prediction=(raw+tau*.001*delta)/durations
        measured=(q[stride:]-q[:-stride])[keep]/durations
        groups.append((measured,prediction,durations,stride))
    if len(groups)!=3:raise ValueError('insufficient chronological intervals')
    return groups


def error(groups,gain):
    values=np.concatenate([gain*b-a for a,b,_,_ in groups])
    physical=np.concatenate([(gain*b-a)*dt for a,b,dt,_ in groups])
    return dict(interval_velocity_RMSE_rad_s=float(np.sqrt(np.mean(values**2))),
        interval_velocity_MAE_rad_s=float(np.mean(abs(values))),
        interval_position_RMSE_rad=float(np.sqrt(np.mean(physical**2))),
        interval_position_P95_rad=float(np.percentile(abs(physical),95)),intervals=len(values))


def profile(t,q,v,fit_window,validation_window):
    if not np.isfinite(np.r_[t,q,v]).all():raise ValueError('nonfinite q/dq log data')
    if np.any(np.diff(t)<=0) or np.max(np.diff(t))>.015:raise ValueError('chronological contiguous ~100Hz input required')
    if np.std(v)<1e-7:
        return dict(identifiable=False,reason='dq constant or disabled; do not fit an arbitrary observer')
    candidates=[]
    for age in AGES:
        for tau in TAUS:
            train=intervals(t,q,v,*fit_window,age,tau)
            num=sum(np.dot(a,b) for a,b,_,_ in train)
            den=sum(np.dot(b,b) for _,b,_,_ in train)
            gain=float(np.clip(num/max(den,1e-12),.8,1.2))
            validation=intervals(t,q,v,*validation_window,age,tau)
            candidates.append(dict(relative_age_ms=age,observer_tau_ms=tau,gain=gain,
                fit=error(train,gain),validation=error(validation,gain)))
    selected=min(candidates,key=lambda r:r['fit']['interval_velocity_RMSE_rad_s'])
    threshold=selected['fit']['interval_velocity_RMSE_rad_s']*1.05+1e-6
    near=[r for r in candidates if r['fit']['interval_velocity_RMSE_rad_s']<=threshold]
    baseline=next(r for r in candidates if r['relative_age_ms']==0 and r['observer_tau_ms']==0)
    return dict(identifiable=True,selected_by='fit partition only, validation never used to choose parameters',
        selected=selected,baseline_zero_relative_age=baseline,
        unmodified_baseline=dict(fit=error(intervals(t,q,v,*fit_window,0,0),1.),
            validation=error(intervals(t,q,v,*validation_window,0,0),1.)),near_fit_count=len(near),
        near_fit_relative_age_range_ms=[min(r['relative_age_ms'] for r in near),max(r['relative_age_ms'] for r in near)],
        near_fit_observer_tau_range_ms=[min(r['observer_tau_ms'] for r in near),max(r['observer_tau_ms'] for r in near)],
        candidates=candidates)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--logs',type=Path,nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);results=[];table=[]
    for path in a.logs:
        rows,x,_,t,_=read_csv(path)
        names=[k for k in rows[0] if '.rl_leg_torque_nm[' in k]
        if len(names)!=4:raise ValueError('explicit four logged RL torque channels required')
        torques=np.array([[float(row[k]) for k in names] for row in rows]);ids=np.flatnonzero(abs(torques).max(1)>1e-5)
        if len(ids)<180 or not np.all(np.diff(ids)==1):raise ValueError('one sufficient continuous RL debug interval required')
        tt=t[ids]-t[ids[0]];length=float(tt[-1]);split=length*.5
        fit_window=[.15,split];val_window=[split,length-.02]
        result=dict(path=str(path.resolve()),sha256=sha(path),activity_rows=[int(ids[0]),int(ids[-1])],
            time_s=[float(t[ids[0]]),float(t[ids[-1]])],fit_window_s=fit_window,validation_window_s=val_window,
            actual_build='UNKNOWN; logs may have different builds/models; fitted separately',channels=[])
        for index,ch in enumerate(LEGS):
            r=profile(tt,x[ids,6+index]*1.5,x[ids,10+ch]*20,fit_window,val_window)
            result['channels'].append(dict(model_channel=ch,**r))
            if r['identifiable']:
                for c in r['candidates']:table.append(dict(log=path.name,model_channel=ch,relative_age_ms=c['relative_age_ms'],
                    observer_tau_ms=c['observer_tau_ms'],gain=c['gain'],fit_RMSE=c['fit']['interval_velocity_RMSE_rad_s'],
                    validation_RMSE=c['validation']['interval_velocity_RMSE_rad_s']))
        results.append(result)
    output=dict(version='motion_relative_q_dq_profiles_v1',logs=results,physics_steps=0,optimizer_steps=0,
        search=dict(relative_age_ms=AGES,observer_tau_ms=TAUS,gain_bounds=[.8,1.2],interval_sample_strides=[1,2,4]),
        limitations=['relative q/dq timing, not absolute sensor/PD/transport delay',
            '100Hz aliasing/packet phase and different acquisition clocks remain unresolved',
            'encoder zero cancels in q increments and cannot be inferred',
            'residuals are not independent sensor-noise standard deviations',
            'model inversion is an offline consistency hypothesis, not a runtime observer or recommended filter'])
    (a.output_dir/'result.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (a.output_dir/'profiles.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['log','model_channel','relative_age_ms','observer_tau_ms','gain','fit_RMSE','validation_RMSE'])
        writer.writeheader();writer.writerows(table)
    print(json.dumps([dict(log=Path(r['path']).name,selected=[dict(channel=c['model_channel'],
        relative_age_ms=c.get('selected',{}).get('relative_age_ms'),tau_ms=c.get('selected',{}).get('observer_tau_ms'),
        gain=c.get('selected',{}).get('gain')) for c in r['channels']]) for r in results],ensure_ascii=False))


if __name__=='__main__':main()
