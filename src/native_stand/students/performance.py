"""Opt-in evidence transport optimizations; no actions, RNG or optimizer changes."""
import torch

VERSION='student_performance_v1'
VERSION2='student_performance_v2'
def settings(spec):
 defaults={'version':VERSION,'evaluation_gpu_buffer':False,'evaluation_reuse_environment':False,
  'training_gpu_statistics':False,'failure_tail_on_device':False,'trace_buffer_max_bytes':256*2**20}
 cfg=spec.get('performance')
 if cfg is None:return defaults
 if isinstance(cfg,dict) and cfg.get('version')==VERSION2:
  defaults.update(version=VERSION2,combined_step_checks=False,fixed_shape_statistics=False,evaluation_group_commands=False,evaluation_group_max_commands=4)
  defaults.update(motion_hotpath=False,contact_history_disabled=False,motion_runtime_reduction=False)
 if not isinstance(cfg,dict) or set(cfg)-set(defaults):raise ValueError('unknown student performance fields')
 cfg={**defaults,**cfg}
 if (cfg.get('motion_hotpath') or cfg.get('contact_history_disabled')) and spec.get('student_training',{}).get('version')!='student_motion31_training_v1':raise ValueError('Motion hotpath requires explicit Motion training')
 if cfg.get('motion_hotpath') and not cfg.get('combined_step_checks'):raise ValueError('Motion hotpath requires combined checks')
 if cfg.get('motion_runtime_reduction') and not cfg.get('motion_hotpath'):raise ValueError('Motion runtime reduction requires explicit hotpath')
 if cfg['version'] not in (VERSION,VERSION2) or any(type(cfg[k]) is not bool for k in defaults if k not in ('version','trace_buffer_max_bytes','evaluation_group_max_commands')):raise ValueError('invalid student performance version/flags')
 if 'evaluation_group_max_commands' in cfg and (type(cfg['evaluation_group_max_commands']) is not int or not 1<=cfg['evaluation_group_max_commands']<=8):raise ValueError('maximum grouped commands must be1..8')
 if type(cfg['trace_buffer_max_bytes']) is not int or cfg['trace_buffer_max_bytes']<=0:raise ValueError('positive trace buffer byte limit required')
 if cfg['evaluation_reuse_environment']:
  if spec.get('recovery_course') or any(r['kind']!='student' or r['family']!='motion_native_normal' for r in spec.get('student_evaluation',{}).get('cases',[])):
   raise ValueError('environment reuse only for independent normal student cases')
 if any(cfg.get(k,False) for k in ('combined_step_checks','fixed_shape_statistics','evaluation_group_commands')) and spec.get('student_training',{}).get('version')!='student_motion31_training_v1':
  raise ValueError('performance v2 runtime flags require explicit Motion training')
 if cfg.get('evaluation_group_commands'):
  if spec.get('student_training',{}).get('version')!='student_motion31_training_v1' or not cfg['evaluation_gpu_buffer']:
   raise ValueError('grouped evaluation requires Motion training and GPU trace buffers')
  if not cfg['evaluation_reuse_environment']:raise ValueError('grouped evaluation requires environment reuse')
  from .grouped_evaluation import plan
  groups=plan(spec.get('student_evaluation',{}).get('cases',[]),maximum_commands=cfg.get('evaluation_group_max_commands',4))
  import math
  peak=max((math.ceil(max(r['seconds'] for r in group)/.01)*8*len(group)*640*4 for group in groups),default=0)
  if peak>cfg['trace_buffer_max_bytes']:raise ValueError('grouped evaluation trace buffer below conservative peak')
 return cfg

def new_motion_defaults(previous=None,*,grouped_evaluation=False):
 """Freeze next-run defaults in the spec; never reinterpret an existing v1 spec."""
 cfg=dict(previous or {})
 cfg.update(version=VERSION2,evaluation_gpu_buffer=True,evaluation_reuse_environment=True,
  training_gpu_statistics=True,failure_tail_on_device=True,combined_step_checks=True,
  fixed_shape_statistics=True,evaluation_group_commands=grouped_evaluation,
  motion_hotpath=True,motion_runtime_reduction=True)
 cfg.setdefault('trace_buffer_max_bytes',256*2**20)
 return cfg

class DeviceTrace:
 """Preallocated per-field device copies; export only at episode/fault boundary."""
 def __init__(self,steps,max_bytes):
  self.steps=steps;self.max_bytes=max_bytes;self.buffers={};self.counts={};self.allocated_bytes=0;self.host_transfers=0
 def add(self,name,value):
  value=value.detach()
  if name not in self.buffers:
   size=self.steps*value.numel()*value.element_size()
   if self.allocated_bytes+size>self.max_bytes:raise ValueError('evaluation GPU trace byte limit exceeded')
   self.buffers[name]=torch.empty((self.steps,*value.shape),dtype=value.dtype,device=value.device)
   self.counts[name]=0;self.allocated_bytes+=size
  buffer=self.buffers[name];index=self.counts[name]
  if value.shape!=buffer.shape[1:] or value.dtype!=buffer.dtype or value.device!=buffer.device or index>=self.steps:raise ValueError('trace field shape/dtype/device/count differs')
  buffer[index].copy_(value);self.counts[name]+=1
 def export(self,*,completed=None,pre_only=False):
  pre_fields={'actor_input','critic_input','root_pre','joint_q_pre','joint_dq_pre','previous_targets','raw_action','mean','actor_valid'}
  result={}
  for name,value in self.buffers.items():
   n=self.counts[name]
   if pre_only and name not in pre_fields:continue
   if completed is not None:
    if n<completed:continue
    n=completed
   if n:
    result[name]=value[:n].cpu().numpy().copy();self.host_transfers+=1
  return result

class RolloutCounts:
 """Integer counts and exact episode duration samples; one export per rollout."""
 def __init__(self,steps,envs,device):
  self.counts=torch.zeros(8,dtype=torch.long,device=device)
  self.duration=torch.zeros(steps,envs,device=device);self.done=torch.zeros(steps,envs,dtype=torch.bool,device=device);self.step=0
 def observe(self,valid,rewards,dones,state,events,posture,handoffs=None,*,episode_duration_s=None):
  self.counts+=torch.stack((valid.sum(),(rewards!=0).sum(),dones.sum(),events[:,0].sum(),events[:,1].sum(),events[:,2].sum(),
   ((posture[:,2:4]<=.34).all(-1)&valid).sum(),handoffs.sum() if handoffs is not None else torch.zeros((),dtype=torch.long,device=valid.device))).long()
  duration=state[:,0] if episode_duration_s is None else episode_duration_s
  if duration.shape!=valid.shape:raise ValueError('episode duration shape differs')
  self.duration[self.step].copy_(duration);self.done[self.step].copy_(dones);self.step+=1
 def export(self):
  counts=self.counts.cpu().tolist();duration=self.duration[:self.step].cpu().numpy();done=self.done[:self.step].cpu().numpy()
  return counts,duration[done].tolist()

def tail_arrays(tail):
 import numpy as np
 return {k:torch.stack([r[k] for r in tail]).cpu().numpy() if isinstance(tail[0][k],torch.Tensor)
  else np.stack([r[k] for r in tail]) for k in tail[0]}

def reuse_key(row):
 return (row['kind'],row['role'],row['family'])

class EvaluationEnvironment:
 """Owner of a single compatible environment; reset happens in every rollout."""
 def __init__(self,enabled,factory):self.enabled=enabled;self.factory=factory;self.env=None;self.key=None;self.creations=0;self.reuses=0
 def acquire(self,row):
  key=reuse_key(row)
  if self.env is not None and self.enabled and key==self.key:self.reuses+=1;return self.env
  self.close();self.env=self.factory(row);self.key=key;self.creations+=1;return self.env
 def close(self):
  env,self.env=self.env,None
  if env is not None:env.close()


def parent_motion_performance_matches(current, previous):
 """Only the canonical declared performance upgrade may differ from parent."""
 if current==previous:return True
 expected=new_motion_defaults(previous,grouped_evaluation=False)
 # Preserve historical explicit profiles and allow the documented fallback.
 for key in ('motion_hotpath','motion_runtime_reduction'):
  if key not in current:expected.pop(key,None)
  elif type(current[key]) is bool:expected[key]=current[key]
 return current==expected


def new_recovery_defaults(previous=None):
 """Recovery64-safe subset; no Motion observation or grouping shortcuts."""
 cfg=dict(previous or {})
 cfg.update(version=VERSION2,evaluation_gpu_buffer=True,evaluation_reuse_environment=False,
  training_gpu_statistics=True,failure_tail_on_device=True,combined_step_checks=False,
  fixed_shape_statistics=False,evaluation_group_commands=False,evaluation_group_max_commands=4)
 cfg.setdefault('trace_buffer_max_bytes',256*2**20)
 return cfg


def recovery_training_v2(spec):
 return (settings(spec)['version']==VERSION2 and
  spec.get('student_training',{}).get('version') in ('student_recovery_broad_narrow_ppo_v1','student_recovery_stop_ppo_v2','student_recovery_large_stop_ppo_v1','student_recovery_contact_quality_ppo_v1'))


def recovery_step_flags(state,rewards,valid,owner,truncated):
 """All original trainer predicates, one host transfer at caller; no RNG draws."""
 return torch.stack(((~truncated).all(),(~state[:,4].bool()).all(),
  (~state[:,30].bool()).all(),torch.isfinite(rewards).all(),(valid==owner).all()))


def fixed_masked_counts(ids,mask,bins):
 """Fixed-size count; invalid samples contribute zero without dynamic indexing."""
 return torch.zeros(bins,device=ids.device,dtype=torch.long).scatter_add_(0,ids.long(),mask.long())
