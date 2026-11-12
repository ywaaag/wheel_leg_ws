"""Explicit R1/R2 Recovery DR contract and independent counter streams (no Isaac)."""
import copy
import numpy as np
VERSION='recovery_friction_pd_dr_v1'
SOURCE='2c671251be48cbef9e538be95d9c5d8cc895bc366f22cbe0feb210041b72cb41'
ASSET='32a56efc5ee677a7ae3b5e59c3848a141276ec4957757acde239b774b0c901b7'
LEVELS=[('friction',.25),('friction',.5),('friction',1.),('pd_gain',.25),('pd_gain',.5),('pd_gain',1.)]

def configuration():
 return {'version':VERSION,'seed':20261018,'source_sha256':SOURCE,'asset_sha256':ASSET,
  'friction':{'combine':'average','ground_static':.5,'ground_dynamic':.5,'wheel_nominal':.5,'wheel_full':[.3,.7],'full_range_buckets':64,'pair_full':[.4,.6],'restitution':0.,'left_right':'common'},
  'pd_gain':{'nominal':[60.,2.,1.],'full_scale':[.95,1.05],'left_right':'common','components':'independent','torque_limits':[40.,5.5]},
  'levels':[[f,s] for f,s in LEVELS],'slots':[256,256,512],
  'evaluation':{'every_updates':250,'nominal_envs':32,'current_envs':24,'previous_envs':24,'confirmations':2,'nominal_pass':32,'dr_pass':23,'nominal_mean_abs_vx_max':.15,'nominal_mean_entry_s_max':.70,'stall_checks':3,'severe_checks':2},
  'initialization':'source Actor/std and same-semantic Critic; fresh PPO Adam/RNG;5 Critic warmup independently per phase',
  'resample':'reset_only','privileged_parameters':False,'enabled_families':['friction','pd_gain']}

def validate(spec):
 c=spec.get('recovery_domain_randomization')
 if c!=configuration():raise ValueError('Recovery DR declaration unknown/missing/different fields')
 if spec.get('source_sha256')!=SOURCE or spec.get('asset_sha256')!=ASSET:raise ValueError('Recovery DR source/asset differs')
 if spec.get('student_training',{}).get('version')!='student_recovery_contact_quality_ppo_v1':raise ValueError('DR requires explicit contact64 recipe')
 if spec.get('student_gpu_resume') or spec.get('recovery_course') or spec.get('diagnostic_gas_spring'):raise ValueError('DR cannot inherit migration/teacher course/spring')
 return c

def initial_state():
 return {'version':VERSION,'level':0,'confirmations':0,'severe':0,'stalled':0,'best_current':None,'last_update':0,'finished':False,'stop_reason':None,'certified_levels':[]}

def strengths(level):
 if type(level) is not int or not 0<=level<len(LEVELS):raise ValueError('invalid DR level')
 family,s=LEVELS[level]
 return [s,0.] if family=='friction' else [1.,s]

def exposure(level,group):
 if group==0:return [0.,0.]
 if group==1:return strengths(level-1) if level else [0.,0.]
 if group==2:return strengths(level)
 raise ValueError('unknown DR group')

def pair_friction(wheel,ground=.5):
 return (np.asarray(wheel)+ground)*.5

def _uniform(seed,env,episode,family,channel):
 # SplitMix64 keyed independently by stable env/episode/family/channel. No global RNG.
 with np.errstate(over='ignore'):
  x=(np.asarray(env,dtype=np.uint64)*np.uint64(0x9E3779B97F4A7C15)+np.asarray(episode,dtype=np.uint64)*np.uint64(0xD1B54A32D192ED03)+np.uint64(seed)+np.uint64(family*1009+channel*9176))
  x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
  x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
  x=x^(x>>np.uint64(31))
 return (x>>np.uint64(11)).astype(np.float64)*(1./9007199254740992.)

def sample(seed,env_ids,episodes,strength):
 ids=np.asarray(env_ids,dtype=np.int64);ep=np.asarray(episodes,dtype=np.int64);s=np.asarray(strength,dtype=float)
 if s.shape==(2,):s=np.broadcast_to(s,(len(ids),2))
 if ep.shape!=ids.shape or s.shape!=(len(ids),2) or not np.isfinite(s).all() or ((s<0)|(s>1)).any():raise ValueError('invalid explicit DR sample')
 out=np.tile([.5,60.,2.,1.],(len(ids),1))
 for family,columns in ((0,[0]),(1,[1,2,3])):
  mask=s[:,family]>0
  if not mask.any():continue  # zero strength makes no random draw, including counter hash.
  for channel,col in enumerate(columns):
   u=_uniform(seed,ids[mask],ep[mask],family+1,channel)
   nominal=[.5,60.,2.,1.][col]
   full=.3+.4*np.minimum((u*64).astype(np.int64),63)/63. if col==0 else nominal*(.95+.1*u)
   out[mask,col]=nominal+s[mask,family]*(full-nominal)
 return out.astype(np.float32)

def advance(state,nominal,current,previous,update):
 """Independent fixed-window evidence only; a different DR draw stream each check."""
 s=copy.deepcopy(state)
 if s['finished'] or update<=s['last_update']:raise ValueError('DR assessment duplicate/out of order')
 s['last_update']=update
 def metrics(rows):
  passed=sum(bool(r['accepted']) for r in rows);entries=sum(r.get('entry_index') is not None for r in rows)
  prefix=float(np.median([r.get('valid_hold_prefix_s',0.) for r in rows]))
  return (passed,entries,prefix)
 n,c,p=metrics(nominal),metrics(current),metrics(previous)
 vx=[r.get('checks',{}).get('mean_abs_vx') for r in nominal]
 # protocol's reported keys are consumed by assessment_summary instead; missing is not success.
 quality=[r.get('dr_nominal_quality',{}) for r in nominal]
 means=[r.get('mean_abs_vx') for r in quality];times=[r.get('entry_s') for r in quality]
 complete=len(nominal)==32 and len(current)==24 and len(previous)==24
 known=all(x is not None and np.isfinite(x) for x in means+times)
 avg_v=float(np.mean(means)) if known else None;avg_t=float(np.mean(times)) if known else None
 passed=complete and known and n[0]==32 and c[0]>=23 and p[0]>=23 and avg_v<=.15 and avg_t<=.70
 s['confirmations']=s['confirmations']+1 if passed else 0
 severe=complete and (n[0]<28 or avg_v is not None and avg_v>.25)
 s['severe']=s['severe']+1 if severe else 0
 best=s['best_current'];improved=best is None or c[0]>=best[0]+2 or c[1]>=best[1]+2 or c[2]>=best[2]+.5
 if improved:s['best_current']=list(c);s['stalled']=0
 elif not passed:s['stalled']+=1
 if s['confirmations']>=2:
  s['certified_levels'].append(s['level']);s['confirmations']=s['severe']=s['stalled']=0;s['best_current']=None
  if s['level']==len(LEVELS)-1:s['finished']=True;s['stop_reason']='R1_R2_COMPLETE'
  else:s['level']+=1
 elif s['severe']>=2:s['finished']=True;s['stop_reason']='NOMINAL_REGRESSION_STOP_KEEP_SOURCE'
 elif s['stalled']>=3:s['finished']=True;s['stop_reason']='DR_STALLED_NEEDS_DESIGN_REVIEW'
 return s,{'nominal':list(n),'current':list(c),'previous':list(p),'nominal_mean_abs_vx':avg_v,'nominal_mean_entry_s':avg_t,'quality_known':known,'gate_passed':passed}
