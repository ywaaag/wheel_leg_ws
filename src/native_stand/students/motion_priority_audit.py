"""Offline, explicit-data view: standing/high-speed goals, low speed diagnostic.

No legacy evidence is rewritten. Some Native checks are not reconstructible
from100Hz MuJoCo traces; the output is never an equivalent Native certificate.
"""
import numpy as np
import torch

VERSION='motion31_highspeed_standing_offline_v1'
UNMAPPED=('Native collider clearance and fall-reset geometry',
          '5ms PD feedback samples and1ms actuator/contact histories',
          'eight-env stochastic initial conditions and PhysX solver behavior')


def category(row):
    vx,yaw,_=row['command']
    if vx==0 and yaw==0:return 'standing'
    if max(abs(vx),abs(yaw))>=2.:return 'highspeed'
    return 'low_speed_diagnostic'


def _controlled(arrays,prefix):
    mask=(arrays['time']>prefix+1e-8)&(arrays['time']<=prefix+30.+1e-8)
    return {k:np.asarray(v)[mask][:3000] for k,v in arrays.items()}


def first_quality_failure(a,row):
    """Reuse the real finite-state termination math, but never truncate history.

    This is the first violation on the recorded open continuation. A simulator
    that actually terminates there would have a different subsequent episode.
    """
    from .motion_quality import QualityTermination,configuration,CAUSES
    q=QualityTermination(1,'cpu',configuration())
    target=torch.tensor([row['command'][:2]],dtype=torch.float32)
    for i in range(len(a['time'])):
        posture=torch.zeros(1,10)
        posture[0,:2]=torch.tensor(a['native_posture'][i],dtype=torch.float32)
        posture[0,2:4]=torch.tensor(a['lengths'][i],dtype=torch.float32)
        posture[0,4]=float(a['q'][i,0]+a['q'][i,3])
        posture[0,5]=float(a['q'][i,1]-a['q'][i,4])
        out=q.observe(posture=posture,height=torch.tensor([a['state'][i,3]],dtype=torch.float32),
            wheel_offset=torch.tensor([a['heading_offset'][i]],dtype=torch.float32),
            contacts=torch.tensor(a['contact_resultant'][i:i+1],dtype=torch.float32),
            target=target,controlled=torch.tensor([True]),valid=torch.tensor([True]))
        if bool(out['terminated'][0]):
            return dict(observed=True,controlled_time_s=(i+1)*.01,
                        causes=[c for c,flag in zip(CAUSES,out['causes'][0].tolist()) if flag])
    return dict(observed=False,controlled_time_s=None,causes=[])


def review(arrays,row,previous_assessment,*,prefix=0.):
    import copy
    result=copy.deepcopy(previous_assessment)
    result['version']=VERSION
    result['category']=category(row)
    result['native_equivalent_certified']=False
    result['unmapped_native_checks']=list(UNMAPPED)
    a=_controlled(arrays,prefix)
    checks=dict(result['checks'])
    # Dropped/absent fields remain unknown; no name/shape-based substitute.
    if not len(a['time']):
        result.update(mapped_gate_passed=False,new_failed_checks=['complete30'])
        return result
    early=first_quality_failure(a,row)
    checks['native_quality_termination_absent']=not early['observed']
    metrics=result['metrics']
    if row['command'][1] and previous_assessment['full_window_completed']:
        checks['spin_center']=metrics['horizontal_hold_mean_mps']<=.1
    if previous_assessment['full_window_completed']:
        q=a['q'][1000:1800]
        mirror=np.stack((q[:,0]+q[:,3],q[:,1]-q[:,4]),-1)
        metrics['joint_mirror_hold_abs_mean_rad']=np.abs(mirror).mean(0).tolist()
        checks['joint_mirror_quality']=max(metrics['joint_mirror_hold_abs_mean_rad'])<=.1
        posture=a['native_posture']
        tilt=np.arctan(np.sqrt(np.tan(posture[:,0])**2+np.tan(posture[:,1])**2))
        tilt=np.where(np.cos(posture[:,0])>=0,tilt,np.pi-tilt)
        metrics['tilt_full_max_rad']=float(tilt.max())
        checks['tilt_full']=metrics['tilt_full_max_rad']<=.5
        delta=np.diff(a['requested_target'][1000:1800],axis=0)
        metrics.update(antiosc_wheel_delta_rms_rad_s=float(np.sqrt(np.mean(delta[:,[2,5]]**2))),
            antiosc_leg_delta_rms_rad=float(np.sqrt(np.mean(delta[:,[0,1,3,4]]**2))),
            antiosc_wheel_request_p95_Nm=float(np.quantile(np.abs(a['pd_request'][1000:1800,[2,5]]),.95)))
        checks.update(antiosc_wheel_delta=metrics['antiosc_wheel_delta_rms_rad_s']<=3.,
            antiosc_leg_delta=metrics['antiosc_leg_delta_rms_rad']<=.03,
            antiosc_request=metrics['antiosc_wheel_request_p95_Nm']<=5.5)
    if result['category']=='standing' and previous_assessment['full_window_completed']:
        xy=np.linalg.norm(a['root_velocity_w'][200:3000,:2],axis=-1)
        yaw=np.abs(a['state'][200:3000,2])
        metrics.update(standing_horizontal_mean_mps=float(xy.mean()),
            standing_horizontal_p95_mps=float(np.quantile(xy,.95)),standing_yaw_abs_mean_rad_s=float(yaw.mean()))
        # New design goals are SEPARATE from the frozen migration gates.
        goal=dict(horizontal_mean=metrics['standing_horizontal_mean_mps']<=.10,
                  horizontal_p95=metrics['standing_horizontal_p95_mps']<=.15,
                  yaw_mean=metrics['standing_yaw_abs_mean_rad_s']<=.10,
                  height_MAE=metrics['height_MAE_m']<=.01,height_p95=metrics['height_p95_m']<=.02)
        result['standing_design_goal_checks']=goal
        result['standing_design_goal_passed']=all(goal.values()) and all(checks.values())
    result.update(native_quality_termination_replay=early,mapped_checks={k:bool(v) for k,v in checks.items()},
        mapped_gate_passed=all(checks.values()),new_failed_checks=[k for k,v in checks.items() if not v],
        historical_pass_preserved=previous_assessment['accepted'],
        semantics='read-only stricter mapped-gate view; no physical re-evaluation or certificate')
    return result


def summarize(records,models):
    output=[]
    for name in models:
        rows=[r for r in records if r['model']==name]
        groups={g:[r for r in rows if r['review']['category']==g]
                for g in ('standing','highspeed','low_speed_diagnostic')}
        summary=dict(model=name,matrix_complete=len(rows)==60,
            groups={g:dict(conditions=len(rs),full30=sum(r['review']['full_window_completed'] for r in rs),
                original_passed=sum(r['review']['historical_pass_preserved'] for r in rs),
                mapped_passed=sum(r['review']['mapped_gate_passed'] for r in rs),
                falls=sum(r['termination']=='FALL' for r in rs)) for g,rs in groups.items()})
        summary['standing_design_goals_passed']=sum(r['review'].get('standing_design_goal_passed',False)
            for r in groups['standing'])
        required=groups['standing']+groups['highspeed']
        summary['all_required_mapped_passed']=bool(required) and len(rows)==60 and all(r['review']['mapped_gate_passed'] for r in required)
        summary['new_training_authorized']=False
        output.append(summary)
    return output
