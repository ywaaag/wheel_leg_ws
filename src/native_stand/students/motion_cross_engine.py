"""Explicit-data MuJoCo comparison matrix and gates; no simulator or file IO."""
import copy

import numpy as np

VERSION = 'motion31_height_pd_mujoco_comparison_v1'


def cases(spec):
    from .motion_height_pd import cases as native_cases, initial_state
    state = initial_state()
    state.update(stage=4, pd_level=2, height_level=2)
    rows = native_cases(copy.deepcopy(spec), 4, state)
    for i, row in enumerate(rows):
        row['group'] = 'nominal_old' if i < 22 else 'trained_height_pd_stress'
        row['_template_source'] = {k:row.pop(k) for k in ('checkpoint','sha256') if k in row}
    for height in (.30, .32, .36, .38, .40):
        for vx, yaw in ((.5, 0.), (-.5, 0.), (0., .5), (0., -.5)):
            rows.append(dict(case_id=f'height_motion_{height:g}_{vx:g}_{yaw:g}',
                command=[vx,yaw,height], _mcu_variant=4,
                _height=dict(initial_m=height,target_m=height,moving=False), group='height_motion_probe'))
    for vx, yaw in ((4.,0.),(-4.,0.),(0.,-4.)):
        rows.append(dict(case_id=f'endpoint_probe_{vx:g}_{yaw:g}', command=[vx,yaw,.34],
                         _mcu_variant=0, group='endpoint_probe'))
    if len(rows) != 60 or len({r['case_id'] for r in rows}) != 60:
        raise ValueError('explicit60-case matrix differs')
    return rows


def free_velocity(world_velocity, world_from_body):
    """MuJoCo free qvel: world linear velocity, BODY angular velocity."""
    v=np.asarray(world_velocity,dtype=float);r=np.asarray(world_from_body,dtype=float)
    if v.shape!=(6,) or r.shape!=(3,3) or not np.isfinite(v).all() or not np.isfinite(r).all():
        raise ValueError('finite world velocity6 and body rotation3x3 required')
    return np.r_[v[:3],r.T@v[3:]]


def scenario(row):
    from .motion_execution_uncertainty import nominal, validate
    value = nominal(); variant = row['_mcu_variant']
    if variant == 2:
        value.update(PD_age_ms=12, motor_delay_ms=[v+2 for v in value['motor_delay_ms']], spring_scale=1.1)
    if variant == 3:
        value['PD_age_ms'] = row['_execution_uncertainty']['PD_age_ms']
    value.update(name=row['case_id'], family=f'variant{variant}')
    # The Native wrapper writes WORLD [vx,vy,vz,wx,wy,wz], not body-forward vx.
    velocity = [.05,0.,0.,0.,.10,0.] if variant in (1,2,3) else [0.]*6
    prefix = row.get('_execution_uncertainty', {}).get('prefix_intervals',0)*.01
    return validate(value), velocity, prefix


def _longest(mask):
    padded = np.r_[False, np.asarray(mask,dtype=bool), False].astype(int)
    changes = np.diff(padded)
    return int(np.max(np.flatnonzero(changes == -1)-np.flatnonzero(changes == 1), initial=0))*.01


def score(arrays, row, *, prefix=0., termination='WINDOW_COMPLETE'):
    """Same physical fields/windows as Native; single-env migration gates only.

    Contact is the resultant of actual MuJoCo wheel/ground forces. Pitch/roll
    use native gravity-derived angles, not the unrelated GUI Euler convention.
    """
    mask = (arrays['time'] > prefix+1e-8) & (arrays['time'] <= prefix+30.+1e-8)
    a = {k: np.asarray(v)[mask][:3000] for k,v in arrays.items()}
    count = len(a['time'])
    complete = (count == 3000 and termination == 'WINDOW_COMPLETE' and
                np.allclose(a['time']-prefix, np.arange(1,3001)*.01, atol=1e-8,rtol=0))
    finite = all(np.isfinite(v).all() for v in a.values())
    checks = dict(complete30=bool(complete), finite=bool(finite))
    metrics = dict(controlled_samples=count, controlled_seconds=count*.01,
        windows_s=dict(hold=[10,18],zero=[20,30],standing=[2,30]),
        native_clearance_gate='NOT_MAPPED; body-ground contacts reported separately',
        contact_semantics='resultant sum of actual wheel/ground contact forces; threshold2N',
        posture_semantics='atan2(gravity_z,-gravity_y), atan2(gravity_x,-gravity_y)')
    if count:
        metrics.update(closure_max_m=float(a['state'][:,6].max()),
            swing_full_peak_rad=float(a['swing_difference'].max()),
            longest_loss_contact_s=[_longest(a['contact_resultant'][:,i]<=2.) for i in range(2)],
            undesired_ground_contact_fraction=float(a['undesired_ground_contact'].mean()))
    if not complete or not finite:
        return dict(version=VERSION, accepted=False, checks=checks,
                    failed_checks=[k for k,v in checks.items() if not v], metrics=metrics,
                    full_window_completed=False, capability_certified=False)
    h = slice(1000,1800); target=row['command']; nonzero=bool(target[0] or target[1])
    active = h if nonzero else slice(200,3000)
    speed = a['state'][h,2 if target[1] else 0]; t=target[1] if target[1] else target[0]
    horizontal = np.linalg.norm(a['root_velocity_w'][:,:2],axis=-1)
    contacts = a['contact_resultant'] > 2.
    height_error = np.abs(a['state'][:,3]-target[2])
    leg = np.abs(a['lengths'][h,0]-a['lengths'][h,1])
    offset = np.abs(a['heading_offset'][h])
    saturation = np.abs(a['motor'][active]) >= a['driver_cap'][active]-1e-4
    metrics.update(mean_signed_speed=float(speed.mean()), MAE=float(np.abs(speed-t).mean()),
        direction=float((speed*t>0).mean()) if t else None, gain=float(speed.mean()/t) if t else None,
        height_mean_m=float(a['state'][h,3].mean()), height_MAE_m=float(height_error[200:].mean()),
        height_p95_m=float(np.quantile(height_error[200:],.95)),
        pitch_peak_rad=float(np.abs(a['native_posture'][active,0]).max()),
        roll_peak_rad=float(np.abs(a['native_posture'][active,1]).max()),
        both_contact_fraction=float(contacts[h].all(-1).mean()),
        contact_fraction_by_wheel=contacts[h].mean(0).tolist(),
        leg_length_MAE_m=float(leg.mean()),leg_length_p95_m=float(np.quantile(leg,.95)),
        wheel_heading_offset_MAE_m=float(offset.mean()),wheel_heading_offset_p95_m=float(np.quantile(offset,.95)),
        motor_limit_fraction=float(saturation.mean()),motor_limit_fraction_by_channel=saturation.mean(0).tolist(),
        driver_clamp_fraction_by_channel=(np.abs(a['driver']) >= a['driver_cap']-1e-4).mean(0).tolist(),
        zero_horizontal_p95_mps=float(np.quantile(horizontal[2000:],.95)),
        zero_yaw_abs_mean_rad_s=float(np.abs(a['state'][2000:,2]).mean()),
        horizontal_hold_mean_mps=float(horizontal[h].mean()),
        leg_target_step_RMS_rad=np.sqrt(np.mean(np.diff(a['requested_target'][h][:,[0,1,3,4]],axis=0)**2,axis=0)).tolist(),
        wheel_target_step_RMS_rad_s=np.sqrt(np.mean(np.diff(a['requested_target'][h][:,[2,5]],axis=0)**2,axis=0)).tolist(),
        wheel_dq_hold_mean_rad_s=a['dq'][h][:,[2,5]].mean(0).tolist(),
        wheel_target_hold_mean_rad_s=a['adopted_target'][h][:,[2,5]].mean(0).tolist(),
        slip_geometric_RMS_mps=float(np.sqrt(np.mean(a['support_velocity_xy'][h]**2))))
    checks.update(contact=metrics['both_contact_fraction']>=.95, closure=metrics['closure_max_m']<=.005,
        pitch=metrics['pitch_peak_rad']<=.1,roll=metrics['roll_peak_rad']<=.05,
        leg_length_MAE=metrics['leg_length_MAE_m']<=.01,leg_length_p95=metrics['leg_length_p95_m']<=.02,
        wheel_heading_MAE=metrics['wheel_heading_offset_MAE_m']<=.01,
        wheel_heading_p95=metrics['wheel_heading_offset_p95_m']<=.02,
        swing_full=metrics['swing_full_peak_rad']<.05,motor_limit=metrics['motor_limit_fraction']<=.05,
        rolling_geometry=bool(a['rolling_geometry_valid'].all()))
    if target[2] == .34:
        checks.update(height=float(height_error[active].mean())<=.02,
                      height_hold_band=.32<=metrics['height_mean_m']<=.36)
    else:
        checks.update(height=metrics['height_MAE_m']<=.01,height_p95=metrics['height_p95_m']<=.02,
                      height_hold=float(height_error[h].mean())<=.01)
    if t:
        signed=speed*np.sign(t)
        metrics['fraction_above_70pct']=float((signed>=abs(t)*.7).mean())
        checks.update(tracking=metrics['MAE']<=.3,direction=metrics['direction']>=.95,
                      gain=.8<=metrics['gain']<=1.2,sustained_fraction=metrics['fraction_above_70pct']>=.8)
        if abs(t)==4:
            sliding=np.convolve(signed,np.ones(100)/100,mode='valid')
            metrics['sustained_sliding_mean_seconds']=_longest(sliding>=3.8)
            checks.update(final_mean=3.6<=float(signed.mean())<=4.4,
                          final_sustained=metrics['sustained_sliding_mean_seconds']>=3.)
    else:
        checks.update(zero_horizontal=metrics['zero_horizontal_p95_mps']<=.3,
                      zero_yaw=metrics['zero_yaw_abs_mean_rad_s']<=.3)
    if target[1]:
        r=a['rolling'][h];common=r.mean(-1);difference=(r[:,0]-r[:,1])*.5
        error=np.abs(difference-a['differential_target'][h])
        metrics.update(rolling_hold_mean_mps=r.mean(0).tolist(),spin_common_abs_mps=float(np.abs(common).mean()),
            differential_error_MAE_mps=float(error.mean()),opposite_fraction=float((r[:,0]*r[:,1]<0).mean()),
            relative_imbalance=float((np.abs(r.sum(-1))/np.maximum(np.abs(r).sum(-1),.02)).mean()))
        checks.update(in_place_translation=metrics['horizontal_hold_mean_mps']<=.3,
            spin_common=metrics['spin_common_abs_mps']<=.05,differential_tracking=metrics['differential_error_MAE_mps']<=.05,
            opposite_rolling=metrics['opposite_fraction']>=.95,rolling_equality=metrics['relative_imbalance']<=.20)
    return dict(version=VERSION,accepted=all(bool(v) for v in checks.values()),checks={k:bool(v) for k,v in checks.items()},
        failed_checks=[k for k,v in checks.items() if not v],metrics=metrics,
        full_window_completed=True,capability_certified=False,
        scope='single-env MuJoCo migration gates; not a Native eight-env or hardware certificate')


def ranking(results, names):
    rows=[]
    for name in names:
        r=[r for r in results if r['model']==name]
        old=[v for v in r if v['case']['group']=='nominal_old']
        complete_matrix=len(r)==60 and all(v['status']=='COMPLETED' for v in r)
        rows.append(dict(model=name,matrix_complete=complete_matrix,cases=len(r),
            falls=sum(v['termination']=='FALL' for v in r),
            nominal_old_passed=sum(v['assessment']['accepted'] for v in old),
            passed=sum(v['assessment']['accepted'] for v in r)))
    rows.sort(key=lambda r:(not r['matrix_complete'],r['falls'],-r['nominal_old_passed'],-r['passed'],names.index(r['model'])))
    return rows
