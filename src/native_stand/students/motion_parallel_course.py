"""Independent forward/backward/yaw lanes under the existing Motion runner."""
import copy,hashlib,json
from . import motion_parallel_design as design
VERSION='motion31_parallel_bidirectional_course_v1'
REWARD='student_motion31_parallel_quality_v10'
TARGETS=tuple(range(12))
def may_stop_for_behavior(*,regression=False,all_stages_passed=False):
 # Explicit current user choice: even success/regression stays observational.
 return False

def initial_state():return {'stage':0,'stage_started_update':0,'last_evaluation_update':-1,'consecutive_passes':0,'accepted_by_stage':{},'lanes':design.initial_state()}
def configuration():
 return json.loads(json.dumps({**design.configuration(),'version':VERSION,'execution':'explicit shared Motion runner','formal_updates_max':4995,'smoke_updates_max':5,'stage_updates_max':4995,'evaluation_every':250,'confirmations':2,'budget_status':'PROPOSAL_ONLY_REQUIRES_NEW_AUTHORIZATION','stop_policy':{'source_skill_failure':'record_only','regression':'record_only','all_stages_passed':'record_only','budget_storage_runtime_fault':'stop','user_stop':'stop'}}))
def commands(stage=0,state=None):return [r['target'] for r in design.allocation((state or initial_state())['lanes'])]
def stage_config(stage=0,envs=4096,state=None):
 if type(stage) is not int or stage not in TARGETS or envs not in (64,4096):raise ValueError('parallel revision/env count differs')
 rows=design.allocation((state or initial_state())['lanes']);values=[r['target'] for r in rows];counts=[r['slots'] for r in rows]
 if envs==64:
  # Evaluation commands are overwritten by the existing explicit group target matrix.
  weights=[n/4096 for n in counts];counts=[envs]+[0]*(len(rows)-1)
 else:weights=[n/envs for n in counts]
 return {'id':stage,'name':f'parallel_revision_{stage}','commands':values,'weights':weights,'sampling_mode':'fixed_slots','sampling_slots':[i for i,n in enumerate(counts) for _ in range(n)],'slot_counts':counts,'enforce_whitelist':True}
def cases(spec,stage=0,state=None):
 template=spec['student_evaluation']['cases'][0]
 return [{**template,'case_id':f'parallel_{stage}_{i}','command':v+[.34]} for i,v in enumerate(commands(stage,state))]
def effective_spec(spec,stage,state=None):
 s=copy.deepcopy(spec);state=state or initial_state();s['motion_parallel']['state']=copy.deepcopy(state)
 s['student_training'].update(stage=stage,course_stage=stage,parallel_state=copy.deepcopy(state));s['motion_training']['stage']=stage
 s['curriculum']['stages']=[stage_config(stage,s['ppo']['envs'],state)];s['student_evaluation']['cases']=cases(s,stage,state)
 return s
def protocol():
 from .motion_large_combo import protocol as old
 return old()
def extend_mixture(previous,values):
 from .motion_course import extend_mixture as base
 return base(previous,values)
def advance(state,candidate):
 out=copy.deepcopy(state);lanes=copy.deepcopy(state['lanes']);records=candidate.get('evaluations',[])
 if candidate.get('stage')!=state['stage'] or candidate['update']<=state['last_evaluation_update']:raise ValueError('parallel evaluation revision/update differs')
 protocols=[r.get('protocol') for r in records]
 if not protocols or any(p!=protocols[0] for p in protocols):raise ValueError('parallel evaluation protocols absent/conflicting')
 from .motion_parallel_assessment import validate
 validate(protocols[0]);psha=hashlib.sha256(json.dumps(protocols[0],sort_keys=True).encode()).hexdigest()
 evidence={'update':candidate['update'],'protocol_sha256':psha,'source_sha256':candidate['source_sha256'],'checkpoint_sha256':candidate['sha256'],'records':[{'target':r['case']['command'][:2],'protocol_sha256':psha,'checkpoint_sha256':r['case']['sha256'],'envs':[{'env':e.get('env'),'accepted':e.get('accepted'),'controlled_s':e.get('controlled_seconds')} for e in r.get('per_env',[])]} for r in records]}
 decisions={}
 for lane in design.LEVELS:
  last=len(design.LEVELS[lane])-1
  if any(r['level']==last for r in lanes[lane]['accepted']):continue
  updated,decision=design.advance(lanes,lane,evidence,protocol_sha=psha,source_sha=candidate['source_sha256']);out['lanes'][lane]=updated[lane];decisions[lane]=decision
 advanced=any(out['lanes'][k]['level']!=lanes[k]['level'] for k in lanes)
 out.update(stage=state['stage']+int(advanced),last_evaluation_update=candidate['update'])
 if advanced:out['stage_started_update']=candidate['update']
 complete=all(r.get('per_env') and len(r['per_env'])==8 and all(e.get('accepted') is True for e in r['per_env']) for r in records)
 if complete and any(r['passed'] for r in decisions.values()):out['accepted_by_stage']={**state['accepted_by_stage'],str(state['stage']):{'checkpoint':candidate['checkpoint'],'sha256':candidate['sha256'],'update':candidate['update']}}
 finished=all(any(r['level']==len(design.LEVELS[k])-1 for r in out['lanes'][k]['accepted']) for k in lanes)
 return out,{'stage_passed':any(r['passed'] for r in decisions.values()),'advanced':advanced,'all_stages_passed':finished,'complete_matrix_passed':complete,'lanes':decisions}
def choose(candidates,accepted_by_stage):
 new=[c for c in candidates if c.get('update',0)>0 and c.get('complete_matrix') and c.get('per_env')]
 rank=lambda c:(sum(e.get('accepted',False) for e in c['per_env'])/len(c['per_env']),-max((e.get('tracking',{}).get('MAE',999) for e in c['per_env']),default=999))
 # Different revisions test different matrices; never rank their passing counts together.
 revision=max((c['stage'] for c in new),default=0);same=[c for c in new if c['stage']==revision];best=max(same,key=rank) if same else None
 accepted=next((c for c in reversed(same) if all(e.get('accepted') is True for e in c['per_env'])),None)
 streak=[]
 for item in same:
  if len(item.get('evaluations',[]))==27 and all(e.get('accepted') is True for e in item['per_env']):streak.append(item)
  else:streak=[]
 final=streak[-1] if len(streak)>=2 and streak[-1]['sha256']!=streak[-2]['sha256'] and streak[-1]['update']-streak[-2]['update']>=250 else None
 return {'source':candidates[0],'source_baseline':candidates[0],'fallback_checkpoint':candidates[0],'latest_checkpoint':candidates[-1],'best_new_evaluated_candidate':best,'best_new':best,'best_overall':best or candidates[0],'accepted_candidate':final,'stage_candidate':accepted,'accepted_by_stage':accepted_by_stage,'current_stage':revision,'final_multi_seed_certified':False,'standing_quality_certified':False}
def apply_tracking(result,env):
 import torch
 from .motion_tracking_reward import huber
 c=env.motion_action_command[:,:2];v=torch.stack((env.vx,env.vy),-1);targets=env.target_xy
 old=targets.new_tensor([(0.,0.),*design.BASE]);new=~(targets[:,None,:]==old[None,:,:]).all(-1).any(-1)
 mix=env.motion_cohort_mix[env.command_index];proposed=design.signals(c,v,env.yaw,new_task=new,mix=mix,episode_target=targets,check_values=False)
 out=dict(result);density=result['density'].clone();raw=result['raw'].clone();mask=result['eligible'].to(density.dtype)
 weights={'linear_tracking':4.,'yaw_tracking':2.,'lateral_tracking':1.,'linear_error':-1.,'yaw_error':-.5,'lateral_error':-.5}
 for name,value in proposed['fields'].items():
  index=result['fields'].index(name);density[:,index]=value*mask;raw[:,index]=value*mask/weights[name]
 out.update(raw=raw,density=density,dt=density*.01,total=result['total']+(density.sum(-1)-result['density'].sum(-1))*.01,version=REWARD)
 return out

def update_mixture(state,records,policy):
 """Two accepted complete events narrow, two known regressions broaden; unknown freezes."""
 out=copy.deepcopy(state);decisions=[];by={tuple(r['case']['command'][:2]):r for r in records}
 for row in out:
  target=tuple(row['target']);r=by.get(target);envs=r.get('per_env',[]) if r else []
  if len(envs)!=8 or {e.get('env') for e in envs}!=set(range(8)) or any(type(e.get('accepted')) is not bool for e in envs):continue
  if target in ((0.,0.),*design.BASE):continue # Preserve source mixture by identity.
  success=all(e['accepted'] for e in envs)
  row['consecutive_eligible']=row.get('consecutive_eligible',0)+1 if success else 0
  row['regressions']=0 if success else row.get('regressions',0)+1
  if row['consecutive_eligible']>=2:row.update(mix=min(.75,row['mix']+.25),consecutive_eligible=0)
  if row['regressions']>=2:row.update(mix=max(0.,row['mix']-.25),regressions=0)
  decisions.append({'target':row['target'],'eligible':success,'next_mix':row['mix']})
 return out,decisions

def retained_regression(baseline,records):
 by={tuple(r['case']['command']):r for r in records}
 for old in baseline:
  current=by.get(tuple(old['case']['command']))
  for e in old.get('per_env',[]):
   if not e.get('accepted'):continue
   matches=[r for r in current.get('per_env',[]) if r.get('env')==e.get('env')] if current else []
   if len(matches)!=1 or matches[0].get('accepted') is not True:return True
 return False

def reward_audit():
 import torch
 rows=[]
 for axis in (0,1):
  for target in (-4.,-2.,-1.,1.,2.,4.):
   for error in (-4.,-1.,-.2,0.,.2,1.,4.):
    for mix in (0.,.75):
     x=torch.tensor([target+error],dtype=torch.float64,requires_grad=True);cmd=x.new_zeros(1,2);cmd[:,axis]=target
     vel=torch.stack((x if axis==0 else x*0,x*0),-1);yaw=x if axis else x*0
     result=design.signals(cmd,vel,yaw,new_task=torch.ones(1,dtype=torch.bool),mix=x.new_full((1,),mix));value=result['density'].sum()
     rows.append({'axis':'vx_mps' if axis==0 else 'yaw_rad_s','target':target,'error':error,'mix':mix,'density':float(value.detach()),'dt_contribution':float(value.detach())*.01,'state_error_derivative':float(torch.autograd.grad(value,x)[0])})
 return rows

def checkpoint_diagnostics_configuration():
 return {'version':'motion31_lr_checkpoint_diagnostics_v1','every_formal_updates':100,
 'fields':['hold_tracking','cohort_critic_EV','PPO_KL','clip_fraction','source_retention_KL','actual_LR'],
 'scope':'actual training rollout/update; no extra simulation, optimizer or reconstructed gradients',
 'EV_scope':'within-cohort rollout return/value before PPO update; zero variance => null',
 'output':'lr_diagnostics/update_NNNN.json; checkpoint SHA bound'}

def checkpoint_diagnostics(metric,candidate):
 import math
 local=metric.get('update');offset=metric.get('parent_formal_offset',0)
 update=metric.get('formal_update_including_parent',local)
 if type(local) is not int or local<=0 or type(offset) is not int or offset<0 or type(update) is not int or update!=local+offset:
  raise ValueError('diagnostics local/cumulative update identity differs')
 if type(update) is not int or update<=0 or update%100:raise ValueError('diagnostics require actual formal100 boundary')
 if candidate.get('update')!=local or not isinstance(candidate.get('sha256'),str) or len(candidate['sha256'])!=64:raise ValueError('diagnostics checkpoint/update identity differs')
 scope=metric.get('actor_update_scope',{});retention=copy.deepcopy(scope.get('source_policy_retention'))
 report={'version':'motion31_lr_checkpoint_diagnostics_v1','update':update,'checkpoint':candidate['checkpoint'],'checkpoint_sha256':candidate['sha256'],
 'actual_LR':metric['actual_LR'],'hold_tracking':copy.deepcopy(metric.get('cohort_statistics',{}).get('hold',{})),
 'cohort_critic_EV':copy.deepcopy(metric.get('cohort_value_diagnostics',[])),
 'PPO_KL':scope.get('kl_valid_mean'),'clip_fraction':scope.get('clip_fraction_valid_mean'),
 'source_retention_KL':retention,'actor_gradient_norm_mean':scope.get('actor_gradient_norm_mean'),
 'critic_gradient_norm_mean':scope.get('critic_gradient_norm_mean'),'loss':copy.deepcopy(metric.get('loss')),
 'course':copy.deepcopy(metric.get('course')),'validity':{'tracking':'sample_count in each cohort; zero samples => null',
 'critic_EV':'null when sample/return variance insufficient','PPO':'null when Actor update skipped; never zero-filled',
 'retention':'reported actual minibatch/eligible counts; no frozen source-state replay'},
 'scope':'training stochastic samples; not deterministic capability evaluation','optimizer_or_simulation_added':0}
 if 'armature_value_diagnostics' in metric:report['armature_critic_EV']=copy.deepcopy(metric['armature_value_diagnostics'])
 if offset:report.update(local_update=local,parent_formal_offset=offset,formal_update_including_parent=update)
 def finite(value):
  if isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite LR diagnostic statistic')
  if isinstance(value,dict):
   for v in value.values():finite(v)
  elif isinstance(value,list):
   for v in value:finite(v)
 finite(report)
 return report
