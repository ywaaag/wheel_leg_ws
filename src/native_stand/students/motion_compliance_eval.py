"""Same-condition comparison and course gates for compliance PPO."""
import itertools
import numpy as np
from .motion_compliance_training import PROFILES,command_set


def cases(stage=None,*,prefix=0.):
    cmds=command_set(3 if stage is None else stage)
    return [dict(profile=p['name'],command=c+[.34],yaw=y,prefix_s=prefix,episode_s=30.,training=False)
            for p,c,y in itertools.product(PROFILES,cmds,(0.,.015,-.015))]


def longest(mask,dt=.01):
    run=peak=0
    for flag in mask:
        run=run+1 if flag else 0;peak=max(peak,run)
    return peak*dt


def summarize(rows,terminal,case):
    valid=[r for r in rows if r is not None and r['controlled']]
    steady=[r for r in valid if 2<r['t']<=30.00001]
    complete=bool(terminal.get('timeout') and len(valid)==3000)
    def a(key):return np.asarray([r[key] for r in steady])
    if not steady:
        return dict(case=case,accepted=False,survival30=False,failed_checks=['incomplete_steady_window'],
            controlled_samples=len(valid),terminal_reason=terminal.get('reason'))
    velocity=a('velocity');command=a('command');height=a('height');angles=a('angles');contacts=a('contact')
    length=a('lengths');encoder=a('encoder_dq')[:,[0,1,3,4]];load=a('load_dq')[:,[0,1,3,4]]
    target=np.asarray(case['command']);zero=not target[:2].any();axis=2 if target[1] else 0
    expected=command[:,1] if target[1] else command[:,0];actual=velocity[:,axis]
    # yaw is a separate measured angular quantity, not linear z velocity.
    if target[1]:actual=a('yaw')
    error=np.abs(actual-expected)
    metrics=dict(vx_abs_mean=float(np.abs(velocity[:,0]).mean()),horizontal_p95=float(np.quantile(np.linalg.norm(velocity[:,:2],axis=-1),.95)),
        yaw_abs_mean=float(np.abs(a('yaw')).mean()),height_MAE=float(np.abs(height-.34).mean()),height_p95=float(np.quantile(np.abs(height-.34),.95)),
        pitch_abs_mean=float(np.abs(angles[:,0]).mean()),roll_abs_mean=float(np.abs(angles[:,1]).mean()),
        pitch_abs_p95=float(np.quantile(np.abs(angles[:,0]),.95)),roll_abs_p95=float(np.quantile(np.abs(angles[:,1]),.95)),
        leg_difference_MAE=float(np.abs(length[:,0]-length[:,1]).mean()),swing_difference_MAE=float(np.abs(a('swing')).mean()),
        swing_difference_peak=float(np.abs(a('swing')).max()),wheel_offset_MAE=float(np.abs(a('wheel_offset')).mean()),
        contact_fraction=(contacts>2).mean(0).tolist(),longest_contact_loss_s=[longest(contacts[:,i]<=2) for i in range(2)],
        saturation_fraction=a('saturated').mean(0).tolist(),encoder_dq_std=encoder.std(0).tolist(),load_dq_std=load.std(0).tolist(),
        slip_mean_mps=a('slip').mean(0).tolist(),tracking_MAE=float(error.mean()),response_mean=float(actual.mean()),
        direction_fraction=float((actual[np.abs(expected)>.05]*np.sign(expected[np.abs(expected)>.05])>0).mean()) if (np.abs(expected)>.05).any() else None)
    checks=dict(survival30=complete,height=metrics['height_MAE']<=.01 and metrics['height_p95']<=.02,
        posture=max(metrics['pitch_abs_mean'],metrics['roll_abs_mean'])<=.05 and max(metrics['pitch_abs_p95'],metrics['roll_abs_p95'])<=.10,
        leg_symmetry=metrics['leg_difference_MAE']<=.008,swing=metrics['swing_difference_peak']<.05,
        contact=min(metrics['contact_fraction'])>=.99 and max(metrics['longest_contact_loss_s'])<=.05,
        motor_limits=max(metrics['saturation_fraction'])<=.05,encoder_oscillation=max(metrics['encoder_dq_std'])<=.5)
    if zero:checks.update(zero_velocity=metrics['vx_abs_mean']<=.1 and metrics['horizontal_p95']<=.15,zero_yaw=metrics['yaw_abs_mean']<=.1)
    else:checks.update(tracking=metrics['tracking_MAE']<=.1,direction=(metrics['direction_fraction'] or 0)>=.95)
    fine=terminal.get('fine_tail')
    if fine is not None and len(fine)>=100:
        v=np.asarray(fine)[:,7:13][:,[0,1,3,4]];v=v-v.mean(0)
        power=np.abs(np.fft.rfft(v*np.hanning(len(v))[:,None],axis=0))**2;f=np.fft.rfftfreq(len(v),.001);mask=(f>=2)&(f<=200)
        metrics['encoder_tail_peak_hz']=f[mask][power[mask].argmax(0)].tolist()
        metrics['spectrum_scope']='last at most1s,1kHz encoder sample; no whole-window modal claim'
    return dict(case=case,accepted=all(checks.values()),survival30=complete,checks=checks,
        failed_checks=[k for k,v in checks.items() if not v],metrics=metrics,controlled_samples=len(valid),terminal_reason=terminal.get('reason'))


def evaluate(env,actor,rows,budget_step):
    """Independent worlds supplied by owner; no training RNG/cache reset."""
    import torch
    out=[]
    for offset in range(0,len(rows),env.n):
        subset=rows[offset:offset+env.n];padded=subset+[subset[-1]]*(env.n-len(subset))
        obs=env.reset(padded);history=[[] for _ in padded];done=[False]*env.n
        for _ in range(3202):
            budget_step(env.n*10)
            with torch.inference_mode():action=actor(obs['student_actor']).numpy()
            obs,transitions=env.step(action)
            for i,tr in enumerate(transitions):
                if done[i]:continue
                history[i].append(tr.get('metrics'))
                if tr['terminated'] or tr['timeout']:
                    if i<len(subset):out.append(summarize(history[i],tr,subset[i]))
                    done[i]=True
            if all(done):break
        if not all(done):raise RuntimeError('evaluation did not produce a terminal within declared horizon')
    return out
