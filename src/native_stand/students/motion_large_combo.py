"""Explicit combined large Motion experiment; old course defaults stay untouched."""
import copy
from . import motion_yaw_retention_course as base

VERSION='motion31_large_combo4096_v1'
NETWORK_VERSION='motion31_large_actor_critic_v1'
TARGETS=base.TARGETS
ASSESSMENT=base.ASSESSMENT
commands=base.commands
cases=base.cases
advance=base.advance
choose=base.choose
extend_mixture=base.extend_mixture
protocol=base.protocol


def network():
 return {'version':NETWORK_VERSION,'actor_hidden_dims':[512,256,128],
         'critic_hidden_dims':[512,512,256],'actor_input':31,'critic_input':49,
         'hidden_activation':'ELU(alpha=1)','mean_output':'Tanh'}


def configuration():
 c=base.configuration();c.update(version=VERSION,envs=4096,zero_slots=1024,old_slots=1536,yaw_slots=1536,
  scope='combined Actor/Critic widening, yaw translation gate and4096env; not single-variable causality')
 return c


def stage_config(stage,envs=4096):
 if stage!=0 or envs not in (64,4096):raise ValueError('large combo only explicit64 evaluation or4096 training')
 s=copy.deepcopy(base.stage_config(0,64));unit=envs//64;counts=[16*unit]+[6*unit]*4+[12*unit]*2
 s.update(weights=[n/envs for n in counts],slot_counts=counts,sampling_slots=[i for i,n in enumerate(counts) for _ in range(n)])
 return s


def effective_spec(spec,stage):
 s=copy.deepcopy(spec);s['curriculum']['stages']=[stage_config(stage,s['ppo']['envs'])]
 s['student_evaluation']['cases']=cases(s,stage)
 return s


def verify(spec):
 """Compare against exact parent after projecting only declared changes."""
 if spec.get('motion_large_continuation'):
  from .motion_large_continuation import verify as verify_continuation
  return verify_continuation(spec)
 import json
 from pathlib import Path
 from .data import sha
 from .motion_yaw_retention_requests import verify as verify_parent
 from .motion_tracking_reward import configuration as reward,COMBO_VERSION
 from .training_recipe import validate_instance
 c=spec['motion_large_combo'];parent=Path(c['parent_spec'])
 if set(c)!={'version','parent_spec','parent_spec_sha256'} or c['version']!=VERSION or sha(parent)!=c['parent_spec_sha256']:
  raise ValueError('large combo parent identity differs')
 p=json.loads(parent.read_text());verify_parent(p)
 if spec['network_architecture']!=network() or spec['student_training'].get('network_architecture')!=network():raise ValueError('large network declaration differs')
 if spec['reward']!=reward(COMBO_VERSION) or spec['student_training']['reward']!=spec['reward']:raise ValueError('large combo gate reward differs')
 if spec['motion_course']['configuration']!=configuration() or spec['student_training']['course_version']!=VERSION:raise ValueError('large combo course differs')
 validate_instance(spec['student_training'],spec['source_sha256'])
 if spec['curriculum']['stages']!=[stage_config(0)]:raise ValueError('large4096 integer quota differs')
 if spec['ppo']!={**p['ppo'],'envs':4096,'hidden_dims':[512,256,128]}:raise ValueError('large combo PPO differs beyond width/envs')
 expected_budget={**p['budget'],'sample_limit':1005*4096*128,'formal_samples':1000*4096*128}
 if spec['budget']!=expected_budget:raise ValueError('large combo first-block budget differs')
 from .performance import parent_motion_performance_matches
 if not parent_motion_performance_matches(spec.get('performance'),p.get('performance')):raise ValueError('large Motion unsupported performance drift')
 allowed={'experiment','artifacts','authorization','motion_training','motion_course','student_training','reward','curriculum','ppo','budget','network_architecture','motion_large_combo','checkpoint_quick_evaluation','student_dependencies','performance'}
 if set(spec)-set(p)-allowed or any(spec.get(k)!=v for k,v in p.items() if k not in allowed):raise ValueError('large combo changed unrelated parent field')
 for key in ('motion_training','motion_course','student_training','curriculum','artifacts'):
  view=copy.deepcopy(spec[key]);before=copy.deepcopy(p[key])
  if key=='motion_training':view['spec_path']=before['spec_path']
  elif key=='motion_course':view['configuration']=before['configuration']
  elif key=='student_training':
   view.pop('network_architecture');view['course_version']=before['course_version'];view['reward']=before['reward']
  elif key=='curriculum':view['stages']=before['stages']
  elif key=='artifacts':view['run_name']=before['run_name']
  if view!=before:raise ValueError('large combo extra variable:'+key)
 return True


def expanded_source(source):
 """Map exact original1600 full state into large network parameterization."""
 from .actor_widening import declaration,critic_declaration,expand_actor,expand_critic,expand_optimizer
 if source.get('network_architecture') is not None:
  if source['network_architecture']!=network():raise ValueError('large full resume architecture differs')
  return source
 actor_names=['log_std']+[f'mlp.{2*i}.{suffix}' for i in range(4) for suffix in ('weight','bias')]
 critic_names=[f'mlp.{2*i}.{suffix}' for i in range(4) for suffix in ('weight','bias')]
 ids=[pid for group in source['optimizer']['param_groups'] for pid in group['params']]
 if len(ids)!=17:raise ValueError('source Adam parameter order declaration differs')
 a_ids=dict(zip(actor_names,ids[:9]));c_ids=dict(zip(critic_names,ids[9:]))
 for state,mapping in ((source['actor'],a_ids),(source['critic'],c_ids)):
  for name,pid in mapping.items():
   for field in ('exp_avg','exp_avg_sq'):
    if pid in source['optimizer']['state'] and source['optimizer']['state'][pid][field].shape!=state[name].shape:
     raise ValueError('source Adam parameter identity differs')
 opt=expand_optimizer(source['optimizer'],source['actor'],a_ids,declaration())
 opt=expand_optimizer(opt,source['critic'],c_ids,critic_declaration(),role='critic')
 return {**source,'actor':expand_actor(source['actor'],declaration()),'critic':expand_critic(source['critic'],critic_declaration()),
         'optimizer':opt,'network_architecture':network()}


def value_diagnostics(storage,*,commands_override=None,cohort_key='motion_diagnostic_cohort'):
 """One host copy per update, exact episode-cohort metadata; no gating."""
 import torch
 from native_stand.recovery.controlled_ppo import KEY
 valid=storage.observations[KEY].reshape(-1).bool()
 ids=storage.observations[cohort_key].reshape(-1).long()
 ret=storage.returns.reshape(-1);val=storage.values.reshape(-1);raw=ret-val;norm=storage.advantages.reshape(-1)
 data=torch.stack((ret,ret.square(),raw,raw.square(),norm,((raw>0)!=(norm>0)).float()),-1)
 values=commands(0) if commands_override is None else commands_override
 counts=torch.zeros(len(values),device=ret.device).scatter_add_(0,ids,valid.float())
 sums=torch.zeros(len(values),6,device=ret.device).scatter_add_(0,ids[:,None].expand(-1,6),torch.where(valid[:,None],data,0.))
 means=sums/counts.clamp_min(1)[:,None]
 rv=(means[:,1]-means[:,0].square()).clamp_min(0);ev=1-(means[:,3]-means[:,2].square()).clamp_min(0)/rv.clamp_min(1e-12)
 rows=torch.cat((counts[:,None],means,rv[:,None],ev[:,None]),-1).cpu().tolist()
 return [{'command':c,'samples':int(r[0]),'return_mean':r[1],'value_error_MSE':r[4],
          'raw_advantage_mean':r[3],'normalized_advantage_mean':r[5],'sign_flip_fraction':r[6],
          'return_variance':r[7],'explained_variance':r[8] if r[0] and r[7]>1e-8 else None} for c,r in zip(values,rows)]
