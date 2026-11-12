"""Explicit Recovery-only reward candidate; pure tensors, no phase action table.

Not a registered training recipe. Runtime enablement requires a new declaration
and the named reward logging/privileged-Critic wiring; legacy recipes unchanged.
"""
import torch

VERSION='student_recovery_broad_narrow_reward_v1'
STOP_VERSION='student_recovery_stop_reward_v2'
FIELDS=('lift_trend','supported_level','height_precision','mirror_precision',
        'posture_huber','pre_entry_rotation','post_entry_rotation','post_entry_velocity',
        'mirror_huber','effort','target_change')


def configuration(version=VERSION):
    if version not in (VERSION,STOP_VERSION):raise ValueError('unknown Recovery reward version')
    c={'version':version,'dt':.01,'height_target_m':.34,'minimum_height_m':.30,
        'posture_broad_rad':[.50,.35],'posture_narrow_rad':[.10,.05],
        'mirror_broad_rad':[.30,.30],'mirror_narrow_rad':[.06,.06],
        'height_broad_m':.10,'height_narrow_m':.025,'broad_mix':.40,'narrow_mix':.60,
        'posture_huber_scale_rad':[.50,.35],'mirror_huber_scale_rad':.20,
        'pre_entry_rotation_deadband_rad_s':2.,'pre_entry_rotation_scale_rad_s':2.,
        'post_entry_velocity_scale':[.50,.20,.30],'forward_deadband_mps':.30,
        'contact_min_n':2.,'effort_limits_nm':[40.,40.,5.5,40.,40.,5.5],
        'target_delta_scales':[.10,.10,10.,.10,.10,10.],
        'weights':dict(zip(FIELDS,[2.,3.,1.,1.,-.15,-.10,-.10,-.50,-.10,-.02,-.002])),
        'entry_once':1.,'early_entry_once':1.,'entry_deadline_s':2.5,
        'success_once':10.,'failure_once':0.,'BC_coefficient':0.,
        'state':'post_physics_pre_reset','kernel':'exp(-mean((error/width)^2))',
        'mirror_coordinates':['left_hip+right_hip','left_active-right_active'],
        'precision_gate':'H*dual_contact_positive_clearance*posture_broad',
        'schedule':'fixed simultaneous broad/narrow mix; no update/time-based switch',
        'event_source':'Certificate one-shot pulses; window-complete alone is not success'}
    if version==STOP_VERSION:c.update(post_entry_velocity_scale=[.20,.20,.30],forward_deadband_mps=0.)
    return c


def huber(x):
    x=x.abs()
    return torch.where(x<=1,.5*x.square(),x-.5)


def kernel(error,width):
    return torch.exp(-(error/error.new_tensor(width)).square().mean(-1))


def reward(*,height,h0,angles,contacts,clearance,leg_q,angular_pitch_roll,
           velocity_vx_vy_yaw,applied_torque,previous_targets,executed_targets,
           entered,controlled,valid,fault,entry_event,success_event,failure_event,age,
           reward_configuration=None):
    c=configuration() if reward_configuration is None else reward_configuration
    if c!=configuration(c.get('version')):raise ValueError('Recovery broad/narrow configuration differs')
    n=len(height)
    values={'height':(height,(n,)),'h0':(h0,(n,)),'angles':(angles,(n,2)),
        'contacts':(contacts,(n,2)),'clearance':(clearance,(n,)),'leg_q':(leg_q,(n,4)),
        'angular_pitch_roll':(angular_pitch_roll,(n,2)),
        'velocity_vx_vy_yaw':(velocity_vx_vy_yaw,(n,3)),
        'applied_torque':(applied_torque,(n,6)),'previous_targets':(previous_targets,(n,6)),
        'executed_targets':(executed_targets,(n,6)),'age':(age,(n,))}
    for name,(x,shape) in values.items():
        if x.shape!=shape or x.device!=height.device or x.dtype!=height.dtype or not torch.isfinite(x).all():
            raise ValueError('Recovery reward shape/device/dtype/finite differs: '+name)
    for x in (entered,controlled,valid,fault,entry_event,success_event,failure_event):
        if x.shape!=(n,) or x.dtype!=torch.bool or x.device!=height.device:
            raise ValueError('Recovery reward mask differs')
    if ((entry_event|success_event)&~entered).any() or (success_event&failure_event).any():
        raise ValueError('inconsistent Certificate event pulses')
    if (entry_event&((age<0)|(age>c['entry_deadline_s']+1e-5))).any():
        raise ValueError('entry event outside deadline')
    H=((height-h0)/torch.maximum(.30-h0,torch.full_like(h0,.05))).clamp(0,1)
    C=((contacts>c['contact_min_n']).all(-1)&(clearance>0)).to(height.dtype)
    pb=kernel(angles,c['posture_broad_rad']);pn=kernel(angles,c['posture_narrow_rad'])
    mirror=torch.stack((leg_q[:,0]+leg_q[:,2],leg_q[:,1]-leg_q[:,3]),-1)
    mb=kernel(mirror,c['mirror_broad_rad']);mn=kernel(mirror,c['mirror_narrow_rad'])
    hb=kernel((height-.34)[:,None],[c['height_broad_m']])
    hn=kernel((height-.34)[:,None],[c['height_narrow_m']])
    mix=lambda b,f:c['broad_mix']*b+c['narrow_mix']*f
    G=H*C*pb;I=entered.to(height.dtype)
    v=velocity_vx_vy_yaw.clone()
    v[:,0]=torch.relu(v[:,0].abs()-c['forward_deadband_mps'])
    raw=torch.stack((H*pb,H*C*mix(pb,pn),G*mix(hb,hn),G*mix(mb,mn),
        huber(angles/height.new_tensor(c['posture_huber_scale_rad'])).mean(-1),
        (1-I)*(torch.relu(angular_pitch_roll.abs()-2.)/2.).square().mean(-1),
        I*angular_pitch_roll.square().mean(-1),
        I*huber(v/height.new_tensor(c['post_entry_velocity_scale'])).sum(-1),
        G*huber(mirror/c['mirror_huber_scale_rad']).mean(-1),
        (applied_torque/height.new_tensor(c['effort_limits_nm'])).square().mean(-1),
        ((executed_targets-previous_targets)/height.new_tensor(c['target_delta_scales'])).square().mean(-1)), -1)
    eligible=controlled&valid&~fault
    raw=raw*eligible[:,None]
    density=raw*height.new_tensor([c['weights'][k] for k in FIELDS]);dt=density*c['dt']
    events=torch.stack((entry_event*(c['entry_once']+c['early_entry_once']*(1-age/2.5).clamp(0,1)),
        success_event.to(height.dtype)*c['success_once'],failure_event.to(height.dtype)*c['failure_once']),-1)*eligible[:,None]
    total=dt.sum(-1)+events.sum(-1)
    if not torch.isfinite(total).all():raise ValueError('nonfinite Recovery reward')
    return {'version':VERSION,'fields':FIELDS,'raw':raw,'weighted_density':density,
        'continuous_dt':dt,'events':events,'total':total,'eligible':eligible,
        'diagnostics':{'height_progress':H,'posture_broad':pb,'posture_narrow':pn,
            'mirror_broad':mb,'mirror_narrow':mn,'height_broad':hb,'height_narrow':hn,
            'precision_gate':G,'mirror_error_rad':mirror}}
