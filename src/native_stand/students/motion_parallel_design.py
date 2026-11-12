"""Pure parallel-course design/signals. Runtime activation requires separate explicit course/reward versions."""
import copy, math
import torch
VERSION='motion31_parallel_bidirectional_design_v1'
BASE=((.5,0.),(1.,0.),(1.5,0.),(2.,0.),(0.,.5),(0.,-.5))
LEVELS={'forward':((2.5,),(3.,),(3.5,),(4.,)),
        'backward':((-.5,-1.),(-1.5,),(-2.,),(-3.,),(-4.,)),
        'yaw':((1.,-1.),(1.5,-1.5),(2.,-2.),(3.,-3.),(4.,-4.))}

def configuration():
 return {'version':VERSION,'execution':'DESIGN_ONLY_NOT_REGISTERED','envs':4096,
 'slots':{'zero':204,'retention':1640,'forward':820,'backward':612,'yaw':820},
 'levels':copy.deepcopy(LEVELS),'source_checkpoint':'u3250; exact SHA bound by caller',
 'reward_base':'student_motion31_differential_quality_v9; geometry/control weights unchanged',
 'actor_critic':'Actor31 512/256/128; Critic85 512/512/256 unchanged',
 'new_active_denominator':'max(1.0,(0.5*abs(effective_command))**2)',
 'refinement':'new command only; two complete accepted fixed evaluations; mix +.25 capped .75, two regressions -.25; unknown freezes',
 'old_commands':'original v9 widths/refinement restored by target identity, not slot index',
 'zero':'original absolute error scales .3, no progress bonus; precision parking diagnostic only',
 'normalization':'existing global advantage normalization unchanged',
 'PPO':'LR3e-5/std/Adam/gamma/lambda/rollout128 unchanged; warmup counts toward approved budget',
 'authorization':'CPU_ONLY; no online budget granted; no runnable spec or ledger',
 'gate':{'envs':8,'hold_s':8.,'controlled_s':30.,'direction_min':.95,'sustained_fraction_min':.8,
 'tracking_MAE':'max(.3,.15*abs(target)) in axis physical units',
 'gain_range':[.8,1.2],'pitch_hold_peak_max_rad':.1,'roll_hold_peak_max_rad':.05,
 'swing_full_peak_strict_max_rad':.05,'contact_min':.95,'saturation_max':.05,
 'yaw_center_mean_max_mps':.1,'yaw_common_abs_mean_max_mps':.05,'opposite_fraction_min':.95,'imbalance_mean_max':.2,
 'final_4_endpoint':'mean abs response>=3.6, at least3 continuous seconds abs response>=3.8; correct sign, all safety gates',
 'scope':'new coarse high-speed protocol; never reinterpret historical accepted as this protocol'}}

def initial_state():
 return {k:{'level':0,'streak':0,'last_update':-1,'accepted':[]} for k in LEVELS}

def endpoints(lane,level):
 if lane not in LEVELS or type(level) is not int or level not in range(len(LEVELS[lane])):raise ValueError('invalid lane/level')
 return [(0.,v) if lane=='yaw' else (v,0.) for v in LEVELS[lane][level]]

def allocation(state):
 """4096 fixed slots. Equal signs; retain all mastered endpoints; no RNG."""
 groups={'zero':[(0.,0.)],'retention':list(BASE)}
 for lane in LEVELS:
  level=state[lane]['level'];groups[lane]=endpoints(lane,level)
  for old in range(level):groups['retention']+=endpoints(lane,old)
 groups['retention']=list(dict.fromkeys(groups['retention']))
 rows=[]
 for group,values in groups.items():
  # Share remainder deterministically, rather than rounding independent probabilities.
  total=configuration()['slots'][group];q,r=divmod(total,len(values))
  for i,target in enumerate(values):rows.append({'group':group,'target':list(target),'slots':q+(i<r)})
 assert sum(r['slots'] for r in rows)==4096
 return rows

def signals(command,velocity,yaw,*,new_task,mix,episode_target=None,check_values=True):
 """Tracking-only candidate density, with explicit Huber tails; NOT full v9 reward."""
 from .motion_tracking_reward import huber
 if command.ndim!=2 or command.shape[1]!=2 or velocity.shape!=command.shape or yaw.shape!=command.shape[:1] or new_task.shape!=yaw.shape or new_task.dtype!=torch.bool or mix.shape!=yaw.shape:raise ValueError('explicit batch shapes required')
 if check_values and (not all(torch.isfinite(t).all() for t in (command,velocity,yaw,mix)) or (mix<0).any() or (mix>.75).any()):raise ValueError('nonfinite or invalid mixture')
 target=command if episode_target is None else episode_target
 if target.shape!=command.shape or (check_values and not torch.isfinite(target).all()):raise ValueError('episode target shape/finite differs')
 lin=command[:,0];rot=command[:,1]
 if check_values and ((lin!=0)&(rot!=0)).any():raise ValueError('combined turning is not in this proposal')
 e=velocity[:,0]-lin;ey=yaw-rot;lat=velocity[:,1];zero_lin=target[:,0]==0;zero_yaw=target[:,1]==0
 dl=torch.where(zero_lin,.09,torch.where(new_task,torch.maximum(torch.ones_like(lin),(.5*lin).square()),torch.ones_like(lin)))
 dy=torch.where(zero_yaw,.09,torch.where(new_task,torch.maximum(torch.ones_like(rot),(.5*rot).square()),torch.full_like(rot,.25)))
 ml=mix*(~zero_lin);my=mix*(~zero_yaw)*new_task
 linear=(1-ml)*torch.exp(-e.square()/dl)+ml*torch.exp(-e.square()/.04)
 angular=(1-my)*torch.exp(-ey.square()/dy)+my*torch.exp(-ey.square()/.25)
 spin=zero_lin&~zero_yaw;gate=torch.exp(-velocity.square().sum(-1))
 angular=angular*torch.where(spin,gate,torch.ones_like(gate))
 fields={'linear_tracking':4*linear,'yaw_tracking':2*angular,'lateral_tracking':torch.exp(-lat.square()/.09),
 'linear_error':-huber(e/torch.where(zero_lin,.3,1.)),
 'yaw_error':-.5*huber(ey/torch.where(zero_yaw,.3,1.)),
 'lateral_error':-.5*huber(lat/.3)}
 return {'fields':fields,'density':sum(fields.values()),'linear_denominator':dl,'yaw_denominator':dy,
 'scope':'tracking six terms only; v9 support/height/posture/symmetry/contact/slip/effort/differential still required'}

def advance(state,lane,evidence,*,protocol_sha,source_sha):
 """Independent lanes; all established capabilities must remain valid."""
 if lane not in LEVELS or len(protocol_sha)!=64 or len(source_sha)!=64:raise ValueError('explicit lane/protocol/source identity required')
 prev=state[lane];out=copy.deepcopy(state);update=evidence.get('update')
 if type(update) is not int or update<=prev['last_update']:raise ValueError('need a distinct real update')
 if evidence.get('protocol_sha256')!=protocol_sha or evidence.get('source_sha256')!=source_sha:raise ValueError('protocol/source conflict')
 cp=evidence.get('checkpoint_sha256')
 if not isinstance(cp,str) or len(cp)!=64:raise ValueError('candidate identity missing')
 if cp==prev.get('last_checkpoint_sha256'):raise ValueError('repeated checkpoint cannot confirm learning')
 required={(0.,0.),*BASE}
 for key in LEVELS:
  for level in range(state[key]['level']):required.update(endpoints(key,level))
  for accepted in state[key].get('accepted',[]):required.update(endpoints(key,accepted['level']))
 required.update(endpoints(lane,prev['level']))
 records=evidence.get('records',[]);seen={};unknown=False;failed=False
 for record in records:
  target=tuple(record.get('target',[]))
  if target in seen:raise ValueError('duplicate command evidence')
  seen[target]=record
 for target in required:
  r=seen.get(target)
  if not r:unknown=True;continue
  if r.get('checkpoint_sha256')!=cp or r.get('protocol_sha256')!=protocol_sha:raise ValueError('mixed candidate/protocol evidence')
  envs=r.get('envs',[])
  if len(envs)!=8 or {x.get('env') for x in envs}!=set(range(8)):unknown=True;continue
  if any(x.get('accepted') is False for x in envs):failed=True
  if any(type(x.get('accepted')) is not bool or not isinstance(x.get('controlled_s'),(int,float)) or not math.isfinite(x['controlled_s']) or x['controlled_s']<30 for x in envs):unknown=True
 out[lane]['last_update']=update;out[lane]['last_checkpoint_sha256']=cp
 out[lane]['streak']=0 if failed or unknown else prev['streak']+1
 passed=out[lane]['streak']>=2
 if passed:
  out[lane]['accepted'].append({'level':prev['level'],'update':update,'checkpoint_sha256':cp,'protocol_sha256':protocol_sha})
  if prev['level']<len(LEVELS[lane])-1:out[lane].update(level=prev['level']+1,streak=0)
 return out,{'passed':passed,'missing_evidence':unknown,'known_failure':failed,'other_lanes_unchanged':True,'budget_reset':False,'simulation':0}
