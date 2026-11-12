"""Post-physics source reward, with explicit motor/load measurement separation."""
from types import SimpleNamespace as S
import torch
from .motion_tracking_reward import reward,configuration,COMBO_BASE
from .motion_quality import apply_reward
from .motion_yaw_quality import apply as spin_apply
from .motion_differential import target_from_geometry,terms
from .motion_sensor_transfer import apply_costs as transfer
from .motion_antioscillation import apply_costs as anti
from .motion_posture_design import apply_runtime_reward


def compute(source,*,command,target,velocity,angular,height,angles,contacts,q,dq,
            motor,request,previous_motor,adopted,previous_target,older_target,
            lengths,wheel_offset,swing_difference,spin,controlled,failed,history_valid,
            linear_mix=0.,symmetry_mix=0.,vertical_velocity=None):
    """Every input is explicit. Continuous densities receive dt exactly once."""
    n=len(height);yes=torch.ones(n,dtype=torch.bool);no=torch.zeros(n,dtype=torch.bool)
    controlled=torch.as_tensor(controlled,dtype=torch.bool).expand(n)
    failed=torch.as_tensor(failed,dtype=torch.bool).expand(n)
    cmix=torch.as_tensor(linear_mix,dtype=height.dtype).expand(n)
    result=reward(command=command,target=target,velocity=velocity,yaw=angular[:,1],height=height,
        angles=angles,contacts=contacts,leg_q=q[:,[0,1,3,4]],wheel_offset=wheel_offset,
        leg_lengths=lengths,applied_torque=motor,previous_targets=previous_target,
        executed_targets=adopted,controlled=controlled,valid=yes,physical_failure=no,fault=no,
        config=configuration(COMBO_BASE),linear_narrow_mix=cmix)
    posture=torch.cat((angles,lengths,height.new_zeros(n,6)),-1)
    result=apply_reward(result,posture=posture,wheel_offset=wheel_offset,leg_q=q[:,[0,1,3,4]],
        controlled=controlled,valid=yes,quality_failed=failed,physical_failed=no,fault=no,
        config=source['reward']['quality_termination'],reward_config=source['reward'])
    result=spin_apply(result,target=target,command=command,velocity=velocity,contacts=contacts,
        previous_targets=previous_target,executed_targets=adopted,measurement=spin,
        config=source['reward']['spin_quality'])
    # Same pure v8/v9 equations; the Isaac runtime wrapper also builds privileged
    # fields, which are deliberately not imported by this CPU environment.
    from .motion_tracking_reward import huber
    mix=torch.as_tensor(symmetry_mix,dtype=height.dtype).expand(n)
    gate=result['spin_diagnostics']['activation'];e=swing_difference.abs()
    swing=-.5*huber(torch.relu(e-.02)/.03)+.25*((1-mix)*torch.exp(-e.square()/.0064)+mix*torch.exp(-e.square()/.0009)-1)
    def replace(name,raw,density):
        i=result['fields'].index(name);old=result['dt'][:,i].clone()
        result['raw'][:,i]=raw;result['density'][:,i]=density;result['dt'][:,i]=density*.01
        result['total']+=result['dt'][:,i]-old
    common=spin['rolling_speed_forward_mps'].mean(-1)
    cd=-.5*((1-mix)*huber(common/.15)+mix*huber(common/.08))*gate
    td=-.5*((1-mix)*huber(velocity[:,:2]/.2).sum(-1)+mix*huber(velocity[:,:2]/.1).sum(-1))*gate
    replace('spin_common_rolling',cd/-.5,cd);replace('spin_translation',td/-.5,td)
    def append(name,raw,density):
        result.update(fields=(*result['fields'],name),raw=torch.cat((result['raw'],raw[:,None]),-1),
            density=torch.cat((result['density'],density[:,None]),-1),
            dt=torch.cat((result['dt'],density[:,None]*.01),-1),total=result['total']+density*.01)
    swing*=result['eligible'];append('leg_swing_symmetry',-swing,swing)
    desired=target_from_geometry(spin['center_world'],spin['forward_world'],command[:,1])
    differential=terms(spin['rolling_speed_forward_mps'],desired,mix)
    append('spin_differential_target',differential['raw']*gate,differential['density']*gate)
    cfg=source['motion_timing']['configuration']
    result=transfer(result,targets=adopted,previous=previous_target,dq=dq,
        vertical_velocity=velocity[:,2] if vertical_velocity is None else vertical_velocity,
        angular_body=angular,torque=motor,config=cfg['reward_extension'])
    result=anti(result,request,motor,previous_motor,adopted,previous_target,older_target,
        contacts,history_valid,cfg)
    # Existing MCU request-excess uses actual wheel capacity, unlike the
    # reference40/5.5 scale retained by the source effort/torque-change terms.
    from .motion_mcu_execution import WHEEL_CAP
    excess=torch.relu(request[:,[2,5]].abs()/WHEEL_CAP-.8)
    val=((1-cfg['smooth_mix'])*huber(excess/2)+cfg['smooth_mix']*huber(excess)).mean(-1)*result['eligible']
    replace('wheel_request_excess',val,-.25*val)
    apply_runtime_reward(result,posture)
    result['measurement_semantics']='effort=motor; q/dq/geometry=load; target=PD adopted; no merged force effort'
    return result


def reward_snapshot(source):
    """Small source tables and true formula curves; not an optimizer claim."""
    from .motion_tracking_reward import FIELDS
    rows=[]
    for name,weight in zip(FIELDS,source['reward']['weights']):
        rows.append(dict(name=name,weight=weight,measurement='motor' if name=='effort' else 'adopted target' if name=='target_change' else 'load state',dt=.01))
    curves=[]
    for name,scale,weight in [('roll_excess',.05,-.5),('leg_balance',.008,-.15),('zero_velocity',.3,-1.)]:
        for error in (0.,.001,.01,.05,.1,.5,1.):
            x=torch.tensor(error,dtype=torch.float64,requires_grad=True)
            z=x/scale;value=weight*torch.where(z.abs()<=1,.5*z.square(),z.abs()-.5)
            curves.append(dict(field=name,error=error,density=float(value.detach()),derivative=float(torch.autograd.grad(value,x)[0]),dt_contribution=float(value.detach())*.01))
    return dict(version='motion_compliance_reward_measurement_v1',base=rows,
        source_reward=source['reward'],extension=source['motion_timing']['configuration']['reward_extension'],
        antioscillation=source['motion_timing']['configuration']['shaping'],posture_scales=dict(roll=.05,leg=.008),
        measurement_changes=['motor effort replaces summed motor+transmission+spring','load dq for physical smoothness; encoder dq only Actor/diagnostics','adopted targets only'],
        curves=curves,terminal_event=-1.,total_clip=None,convergence_guaranteed=False)
