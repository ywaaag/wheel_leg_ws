"""Explicit next-run coarse-speed scoring over existing v9 safety evidence.

No environment/IO. Old evaluator behavior stays unchanged unless this exact
protocol is supplied; this does not register a new training recipe.
"""
import copy,math
import torch
VERSION='motion31_parallel_highspeed_assessment_v1'
BASE=((0.,0.),(.5,0.),(1.,0.),(1.5,0.),(2.,0.),(0.,.5),(0.,-.5))
REQUIRED=('survival30','height','contact','saturation','closure','clearance','tilt',
 'quality_complete_window','absolute_posture_complete_window','pitch_absolute_quality',
 'roll_absolute_quality','roll_quality','leg_balance_quality','wheel_alignment_quality',
 'joint_mirror_quality','swing_hold','swing_full')

def safety_protocol(spec):
 """The same effective v8/v9 safety declaration used by runtime and CPU plans."""
 if not spec.get('motion_symmetry') or not spec.get('motion_assessment'):raise ValueError('symmetry safety declaration required')
 out={**copy.deepcopy(spec['motion_assessment']),'version':'motion31_symmetry85_assessment_v1',
 'symmetry':copy.deepcopy(spec['motion_symmetry']['assessment'])}
 if spec.get('motion_differential'):
  out.update(version='motion31_differential85_assessment_v1',differential=copy.deepcopy(spec['reward']['differential']['assessment']),
   wheel_scope='actual geometric rolling; signed left/right heading projection, not raw motor polarity')
 return out

def protocol(base):
 if base.get('version')!='motion31_differential85_assessment_v1':raise ValueError('explicit v9 safety protocol required')
 out=copy.deepcopy(base);out.update(version=VERSION,base_protocol=copy.deepcopy(base),
  new_tracking_MAE={'absolute_floor':.3,'relative_fraction':.15,'axes':{'vx':'m/s','yaw':'rad/s'}},
  holding_window_s=[10.,18.],sample_rate_hz=100.,controlled_seconds=30.,
  legacy_retention_commands=[list(x) for x in BASE],parking_required=False,
  final_endpoint_semantics='correct signed speed >=3.8 continuously for3 seconds; signed hold mean3.6..4.4',
  final_multi_seed_certified=False)
 out.pop('tracking_MAE_max',None)
 out['tracking_threshold_scope']='legacy_retention_commands use0.3; all other commands use new_tracking_MAE rule, respective axis units'
 return out

def validate(config):
 if not isinstance(config,dict) or config!=protocol(config.get('base_protocol',{})):raise ValueError('parallel effective protocol differs')
 return config

def assess(result,state,controlled,target,config):
 validate(config)
 if len(target) not in (2,3) or any(type(x) not in (int,float) or not math.isfinite(x) for x in target):raise ValueError('finite explicit command required')
 vx,yaw=target[:2]
 if max(abs(vx),abs(yaw))>4 or (vx!=0 and yaw!=0) or (len(target)==3 and target[2]!=.34):raise ValueError('straight OR in-place yaw within4 only')
 if state.ndim!=2 or state.shape[1]<30 or controlled.shape!=(len(state),) or controlled.dtype!=torch.bool:raise ValueError('named state layout/control mask differs')
 r=copy.deepcopy(result);checks=r['checks'];missing=[k for k in REQUIRED if type(checks.get(k)) is not bool]
 if yaw:missing += [k for k in ('spin_center','spin_common','difference_complete','difference_tracking','opposite_rolling','rolling_equality') if type(checks.get(k)) is not bool]
 valid=controlled&state[:,29].bool();s=state[valid][:3000];complete=len(s)==3000 and bool(torch.isfinite(s).all())
 axis=9 if yaw else 7;T=yaw if yaw else vx;unit='rad/s' if yaw else 'm/s';old=(vx,yaw) in BASE
 checks['parallel_complete_window']=complete;checks['parallel_required_safety_evidence']=not missing
 details={'axis':'yaw' if yaw else 'vx','unit':unit,'sample_rate_hz':100.,'window_s':[10.,18.],
 'required_hold_samples':800,'controlled_samples':len(s),'scope':'source retention uses original tracking threshold; new commands coarse-speed protocol',
 'missing_safety_evidence':missing,'original_command':old,'MAE_limit':.3 if old else max(.3,.15*abs(T))}
 if T:
  if complete:
   speed=s[1000:1800,axis];signed=speed*(1 if T>0 else -1);mean=speed.mean();mae=(speed-T).abs().mean();gain=mean/T
   direction=(speed*T>0).to(speed.dtype).mean();fraction=(signed>=abs(T)*.7).to(speed.dtype).mean()
   details.update(mean_signed_speed=float(mean),MAE=float(mae),gain=float(gain),direction=float(direction),fraction_above_70pct=float(fraction))
   checks.update(tracking=float(mae)<=details['MAE_limit'],direction=float(direction)>=.95,gain=.8<=float(gain)<=1.2,sustained_fraction=float(fraction)>=.8)
   r.setdefault('tracking',{}).update({k:details[k] for k in ('mean_signed_speed','MAE','gain','direction','fraction_above_70pct')})
   if abs(T)==4:
    above=signed>=3.8;idx=torch.arange(len(above),device=above.device);last_false=torch.where(~above,idx,idx.new_full(idx.shape,-1)).cummax(0).values
    longest=int(torch.where(above,idx-last_false,0).max());details['continuous_above_3p8_s']=longest*.01
    checks.update(final_mean=3.6<=float(signed.mean())<=4.4,final_sustained=longest>=300)
  else:
   checks.update(tracking=False,direction=False,gain=False,sustained_fraction=False)
   if abs(T)==4:checks.update(final_mean=False,final_sustained=False)
 r.update(version=VERSION,parallel_tracking=details,accepted=all(v is True for v in checks.values()),
  failed_checks=[k for k,v in checks.items() if v is not True],standing_quality_certified=False,
  capability_scope='parallel straight/backward/in-place yaw coarse-speed assessment; not multiseed or hardware certification')
 return r
