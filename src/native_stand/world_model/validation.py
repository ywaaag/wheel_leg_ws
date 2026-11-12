"""Named physical prediction errors, analytic derivatives, and REAL selection."""
from dataclasses import replace
import numpy as np
import torch
from .geometry import rotation_error
from .learner import macro, frozen
from .execution import observation,measurements


def errors(predicted, actual, geometry):
    p,a=geometry(predicted.q,predicted.v),geometry(actual.q,actual.v)
    pq=predicted.q[:,3:7].double();aq=actual.q[:,3:7].double()
    pq=pq/pq.norm(dim=-1,keepdim=True);aq=aq/aq.norm(dim=-1,keepdim=True)
    dot=(pq*aq).sum(-1).abs().clamp(0,1)
    contact_pred=predicted.contact>2;contact_true=actual.contact>2
    return dict(orientation_rad=(2*torch.acos(dot)),
        q_rad=(predicted.q[:,7:]-actual.q[:,7:]).abs().mean(-1),
        active_q_rad=(predicted.q[:,geometry.active_q]-actual.q[:,geometry.active_q]).abs().mean(-1),
        dq_rad_s=(predicted.v[:,6:]-actual.v[:,6:]).abs().mean(-1),
        linear_velocity_mps=(p['velocity']-a['velocity']).abs().mean(-1),
        angular_velocity_rad_s=(p['sensors'][:,:3]-a['sensors'][:,:3]).abs().mean(-1),
        height_m=(p['height']-a['height']).abs(),
        closure_m=p['closure'].norm(dim=-1).amax(-1),
        contact_force_N=(predicted.contact-actual.contact).abs().mean(-1),
        root_contact_force_N=(predicted.body_contact-actual.body_contact).abs().mean(-1),
        contact_event_error=(contact_pred!=contact_true).float().mean(-1),
        energy_error_J=(p['energy']-a['energy']).abs(),
        energy_growth_error_J=(p['energy']-a['energy']),
        torque_Nm=(predicted.motor-actual.motor).abs().mean(-1),
        queue_Nm=(predicted.queue-actual.queue).abs().flatten(1).amax(-1),
        previous_target=(predicted.adopted-actual.adopted).abs().amax(-1),
        nonphysical=(~geometry.valid(predicted.q,predicted.v,p)).float())


def summarize_errors(items):
    if not items: return dict(count=0,status='UNKNOWN_NO_DATA')
    return dict(count=sum(len(next(iter(x.values()))) for x in items),
        metrics={k:dict(mean=float(torch.cat([r[k].detach().cpu().reshape(-1) for r in items]).mean()),
            p95=float(torch.quantile(torch.cat([r[k].detach().cpu().reshape(-1) for r in items]),.95)),
            max=float(torch.cat([r[k].detach().cpu().reshape(-1) for r in items]).max())) for k in items[0]})


def regime(before, after, geometry):
    m=geometry(before.q,before.v)
    if bool((before.owner==0).any()):return 'handover_prefix'
    if bool(((before.contact<=2).any(-1)&(after.contact>2).all(-1)).any()):return 'landing_nearby'
    if bool((before.contact<=2).any()):return 'lost_contact'
    if bool((before.v[:,geometry.active_v].abs().amax(-1)>2).any()):return 'oscillation'
    return 'stable_support'


def predict_report(replay, wm, geometry, reward, budget, *, samples=48, horizons=(8,16,32), seed=21):
    if replay.split not in ('validation','test'):
        raise ValueError('prediction acceptance uses held-out episodes, not training frames')
    rng=np.random.default_rng(seed);device=next(wm.parameters()).device
    geometry=geometry.to(device)
    groups={name:{str(h):[] for h in horizons} for name in
            ('stable_support','handover','oscillation','lost_contact','landing_nearby')}
    one=[]
    # Retain the original fixed random starts, then explicitly cover each held-out
    # continuous handover. The original48 draws happened to miss this regime.
    selected_starts=[]
    for _ in range(samples):
        _,_,_,selected=replay.batch(1,rng,horizon=max(horizons))
        selected_starts.extend(selected)
    random_starts=len(selected_starts)
    for ep,case in enumerate(replay.cases):
        handover=case.get('handover_ms',0)
        if not handover:continue
        last=len(replay.data[ep]['reward'])-max(horizons)
        for tick in (handover-200,handover,handover+200):
            start=max(0,tick//10)
            if start<=last and (ep,start) not in selected_starts:selected_starts.append((ep,start))
    with torch.no_grad():
        for ep,start in selected_starts:
            budget.check()
            b=replay.state(ep,slice(start,start+1),device=device)
            # Independent1ms prediction from an actual physical state.
            z=replay.data[ep];i=start*10
            micro=replace(b,q=torch.as_tensor(z['micro_q'][i:i+1],dtype=torch.float32,device=device),
                v=torch.as_tensor(z['micro_v'][i:i+1],dtype=torch.float32,device=device),
                warm=torch.as_tensor(z['micro_warm'][i:i+1],dtype=torch.float32,device=device))
            applied=torch.as_tensor(z['micro_applied'][i:i+1],dtype=torch.float32,device=device)
            actual=replace(micro,q=torch.as_tensor(z['micro_next_q'][i:i+1],dtype=torch.float32,device=device),
                v=torch.as_tensor(z['micro_next_v'][i:i+1],dtype=torch.float32,device=device),
                contact=torch.as_tensor(z['micro_contact'][i:i+1],dtype=torch.float32,device=device),
                body_contact=torch.as_tensor(z['micro_body_contact'][i:i+1],dtype=torch.float32,device=device))
            one.append(errors(wm.predict(micro,applied,0),actual,geometry))
            # Free mechanics/execution rollout; recorded actions do not replace predicted states.
            for member in range(len(wm.members)):
                s=b
                all_valid=torch.ones(len(s.q),dtype=torch.bool,device=device)
                invalid_fraction=s.q.new_zeros(len(s.q))
                for k in range(max(horizons)):
                    action=torch.as_tensor(z['action'][start+k:start+k+1],device=device)
                    s,terms,_,ok,_=macro(s,action,wm,geometry,reward,member=member)
                    all_valid&=ok
                    invalid_fraction+=terms['invalid_substep_fraction']
                    if not torch.isfinite(s.q).all() or not torch.isfinite(s.v).all():
                        raise FloatingPointError('nonfinite free prediction rollout')
                    if k+1 in horizons:
                        a=replay.state(ep,slice(start+k,start+k+1),'after',device)
                        name=regime(b,a,geometry)
                        case=replay.cases[ep];handover=case.get('handover_ms',0)
                        if name=='handover_prefix' or (handover and handover-500<=int(b.tick[0])<=handover+2000):name='handover'
                        metric=errors(s,a,geometry);metric['nonphysical']=(~all_valid).float()
                        metric['nonphysical_state_fraction']=invalid_fraction/(k+1)
                        groups[name][str(k+1)].append(metric)
    summary={name:{h:summarize_errors(v) for h,v in by.items()} for name,by in groups.items()}
    combined={str(h):summarize_errors([x for g in groups.values() for x in g[str(h)]]) for h in horizons}
    h8=combined.get('8',{})
    checks={}
    if h8.get('count'):
        m=h8['metrics']
        checks=dict(finite_nonphysical_fraction=m['nonphysical']['mean']==0,
            orientation_80ms=m['orientation_rad']['p95']<=.05,
            active_q_80ms=m['active_q_rad']['p95']<=.03,
            velocity_80ms=m['linear_velocity_mps']['p95']<=.15,
            height_80ms=m['height_m']['p95']<=.01,
            closure_80ms=m['closure_m']['max']<=.005,
            causal_queue_80ms=m['queue_Nm']['max']<=1e-5,
            adopted_previous_80ms=m['previous_target']['max']<=1e-5)
        # Open-loop queue identity also depends on predicted PD q; compare causal
        # execution separately in same-state oracle tests. Do not demand exact torque
        # identity under DIFFERENT predicted feedback states.
        checks.pop('causal_queue_80ms');checks.pop('adopted_previous_80ms')
    return dict(single_step_1ms=summarize_errors(one),free_rollout=combined,by_regime=summary,
        checks=checks,passed=bool(checks) and all(checks.values()),split=replay.split,
        horizons_seconds={str(h):h*.01 for h in horizons},
        sampling=dict(seed=seed,random_starts=random_starts,extra_handover_starts=len(selected_starts)-random_starts,
            split=replay.split,random_protocol='same fixed episode/start draws as prior window8 report'),
        missing_regimes=[k for k,v in summary.items() if all(x.get('count',0)==0 for x in v.values())],
        threshold_scope='frozen engineering pilot thresholds, not convergence or task certification')


def imagined_objective(s, action, actor, wm, critic, geometry, reward, horizon=8, member=0):
    total=s.q.new_zeros(len(s.q));alive=torch.ones(len(s.q),dtype=torch.bool,device=s.q.device)
    reliable=alive.clone();contact_events=[]
    for k in range(horizon):
        a=action if k==0 else actor.mean(observation(s,measurements(s,geometry)))
        s,terms,done,ok,_=macro(s,a,wm,geometry,reward,member=member)
        total=total+(.99**k)*terms['total']*alive
        alive=alive&~done;reliable&=ok;contact_events.append((s.contact>2).detach())
    total=total+(.99**horizon)*critic(s)*alive
    return total,reliable,torch.stack(contact_events)


def action_gradient(s, actor, wm, critic, geometry, reward, *, horizon=8, epsilon=.0005):
    with frozen(actor,wm,critic):
        action=actor.mean(observation(s,measurements(s,geometry))).detach().requires_grad_(True)
        objective,ok,events=imagined_objective(s,action,actor,wm,critic,geometry,reward,horizon)
        gradient=torch.autograd.grad(objective.sum(),action)[0]
        finite=[];switches=[]
        with torch.no_grad():
            for ch in range(6):
                delta=torch.zeros_like(action);delta[:,ch]=epsilon
                plus,_,pe=imagined_objective(s,action+delta,actor,wm,critic,geometry,reward,horizon)
                minus,_,me=imagined_objective(s,action-delta,actor,wm,critic,geometry,reward,horizon)
                finite.append((plus-minus)/(2*epsilon))
                switches.append((pe!=me).any(dim=(0,2)))
    fd=torch.stack(finite,-1);switch=torch.stack(switches,-1)
    relative=(gradient-fd).abs()/(gradient.abs()+fd.abs()).clamp_min(1e-4)
    smooth=~switch & ok[:,None]
    return dict(autograd=gradient.detach().cpu().tolist(),finite_difference=fd.cpu().tolist(),
        smooth_mask=smooth.cpu().tolist(),contact_switch=switch.cpu().tolist(),
        smooth_relative_error_mean=float(relative[smooth].mean()) if bool(smooth.any()) else None,
        norm=float(gradient.norm()),finite=bool(torch.isfinite(gradient).all()),
        horizon=horizon,horizon_s=.01*horizon,epsilon=epsilon),gradient.detach()


def directional_comparison(model_derivative, physical_derivative, smooth):
    a=np.asarray(model_derivative,dtype=float);b=np.asarray(physical_derivative,dtype=float)
    mask=np.asarray(smooth,dtype=bool)
    if not mask.any():return dict(passed=False,samples=0,status='UNKNOWN_CONTACT_SWITCH_ONLY')
    a=a[mask];b=b[mask];den=float(np.linalg.norm(a)*np.linalg.norm(b))
    cosine=float(np.dot(a,b)/den) if den>1e-12 else None
    useful=(np.abs(a)>1e-5)&(np.abs(b)>1e-5)
    agreement=float((a[useful]*b[useful]>0).mean()) if useful.any() else None
    return dict(samples=int(mask.sum()),cosine=cosine,nonzero_samples=int(useful.sum()),
        sign_agreement=agreement,passed=cosine is not None and cosine>0 and agreement is not None and agreement>=.6
        and int(useful.sum())>=6,interpretation='paired physical derivative agreement, not calibrated failure probability')


def choose_physical(source, candidate):
    """All old commands retained, meaningful zero improvement, same tuples/engine."""
    if set(source)!=set(candidate) or not source:
        raise ValueError('complete same-condition physical matrices required')
    checks={};improvement=[]
    for key,base in source.items():
        new=candidate[key]
        if base['condition_identity']!=new['condition_identity']:
            raise ValueError('physical evaluation conditions differ')
        for result in (base,new):
            if result.get('imagined') is not False: raise ValueError('imagined reward cannot select Actor')
        b,c=base['summary'],new['summary']
        checks[key+'_survival']=c['physical_seconds']>=min(30.,b['physical_seconds'])-1e-5
        checks[key+'_tracking']=c['tracking_MAE']<=b['tracking_MAE']+.05
        checks[key+'_contact']=c['contact_fraction']>=b['contact_fraction']-.02
        checks[key+'_posture']=c['roll_peak']<=max(.05,b['roll_peak'])+.005 and c['pitch_peak']<=max(.1,b['pitch_peak'])+.01
        for field,tolerance in (('height_MAE',.002),('leg_difference_MAE',.001),('leg_dq_p95',.1),('torque_limit_fraction',.01)):
            if field in b or field in c:
                if field not in b or field not in c:raise ValueError('complete physical quality metrics required')
                checks[key+'_'+field]=c[field]<=b[field]+tolerance
        if 'closure_peak' in c:checks[key+'_closure']=c['closure_peak']<=.005
        if 'swing_peak' in c:checks[key+'_swing']=c['swing_peak']<=max(.05,b['swing_peak'])+.005
        if 'slip_speed_mean' in c:checks[key+'_slip']=c['slip_speed_mean']<=b['slip_speed_mean']+.02
        if base['command']==[0.,0.] and b['physical_seconds']>=30:
            improvement.append(b['vx_abs_mean']-c['vx_abs_mean'])
    checks['zero_improves']=bool(improvement) and float(np.mean(improvement))>=.01
    return dict(selected=all(checks.values()),checks=checks,zero_improvement_mps=improvement,
                selection='real complete validation matrix only; final test excluded')
