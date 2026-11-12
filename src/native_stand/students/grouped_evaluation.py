"""Explicit Motion-only evaluation batches; independent8-env command results."""
import copy,math,json

VERSION='motion_command_groups_v1'

def plan(cases,*,capacity=64,scored_envs=8,maximum_commands=4):
 if type(maximum_commands) is not int or not 1<=maximum_commands<=8:raise ValueError('maximum grouped commands must be1..8')
 if capacity!=64 or scored_envs!=8:raise ValueError('grouped Motion evaluation requires64 capacity/8 scored')
 groups=[];pending=[];key=None
 for row in cases:
  if (row.get('kind')!='student' or row.get('role')!='MotionZeroStudent' or row.get('family')!='motion_native_normal'
      or row.get('yaw_rad')!=0. or len(row.get('command',[]))!=3 or (row['command'][2]!=.34 and not row.get('_height'))):
   raise ValueError('unsupported grouped evaluation case; use explicit serial mode')
  if any(type(v) not in (int,float) or not math.isfinite(v) for v in row['command']) or type(row.get('seconds')) not in (int,float) or not math.isfinite(row['seconds']) or row['seconds']<=0:raise ValueError('nonfinite command/duration')
  current=tuple(row[k] for k in ('kind','role','family','seed','yaw_rad','seconds','checkpoint','sha256'))
  if '_height' in row:current+=(json.dumps(row['_height'],sort_keys=True),)
  if '_motion_dr' in row:current+=(json.dumps(row['_motion_dr'],sort_keys=True),)
  if '_mcu_variant' in row:current+=(row['_mcu_variant'],)
  if pending and (current!=key or len(pending)==min(maximum_commands,capacity//scored_envs)):groups.append(pending);pending=[]
  key=current;pending.append(row)
 if pending:groups.append(pending)
 if len({r['case_id'] for r in cases})!=len(cases):raise ValueError('duplicate grouped case identity')
 return groups

def describe_plan(cases,*,capacity=64,scored_envs=8,maximum_commands=4):
 """Read-only exact batching/capacity report; no environment creation."""
 groups=plan(cases,capacity=capacity,scored_envs=scored_envs,maximum_commands=maximum_commands)
 batches=[]
 for index,group in enumerate(groups):
  seconds=max(r['seconds'] for r in group)
  batches.append({'batch':index,'case_ids':[r['case_id'] for r in group],
   'commands':[r['command'] for r in group],'actual_envs':capacity,'scored_envs':len(group)*scored_envs,
   'scored_envs_per_command':scored_envs,'unused_scoring_envs':capacity-len(group)*scored_envs,
   'required_policy_seconds':seconds,'policy_steps':math.ceil(seconds/.01),
   'physical_env_steps':math.ceil(seconds/.01)*capacity,
   'estimated_device_trace_bytes':math.ceil(seconds/.01)*scored_envs*len(group)*640*4})
 return {'version':VERSION,'case_count':len(cases),'batch_count':len(groups),'actual_env_capacity':capacity,
 'total_scored_episodes':len(cases)*scored_envs,'maximum_commands_per_batch':maximum_commands,
 'batches':batches,'unique_policy_steps':sum(r['policy_steps'] for r in batches),
 'all_simulated_env_steps':sum(r['physical_env_steps'] for r in batches),
 'estimated_peak_device_trace_bytes':max((r['estimated_device_trace_bytes'] for r in batches),default=0),
 'wall_seconds':'UNKNOWN: policy steps are not wall time; no simulator executed',
 'score_scope':'each command has its own eight first episodes; never pool commands',
 'simulation':0,'files_written':0}

def split_arrays(arrays,start,count=8):
 # policy_time_s has no env axis, all other collector fields do.
 result={k:v if k=='policy_time_s' else v[:,start:start+count].copy() for k,v in arrays.items()}
 if any(v.shape[1]!=count for k,v in result.items() if k!='policy_time_s'):raise ValueError('grouped trace env count differs')
 return result

def rollout(env,actor,cases,*,stop=None,maximum_commands=4):
 from .evaluator import rollout as collect,score_arrays
 if len(plan(cases,maximum_commands=maximum_commands))!=1 or not 1<=len(cases)<=maximum_commands or env.num_envs!=64:raise ValueError('incompatible grouped rollout')
 row=copy.deepcopy(cases[0]);row['_group_cases']=copy.deepcopy(cases)
 if hasattr(env,'configure_mcu_cases'):env.configure_mcu_cases(cases)
 arrays=collect(env,actor,row,stop=stop,scored_envs=8*len(cases),collect_only=True)
 results={}
 for i,case in enumerate(cases):
  part=split_arrays(arrays,i*8)
  payload,_=score_arrays(env,case,part,scored_envs=8)
  # A grouped execution layout is recorded, never passed off as a serial trajectory.
  payload['execution_layout']={'version':VERSION,'case_id':case['case_id'],'scored_env_ids':list(range(i*8,i*8+8)),
   'local_scored_env_ids':list(range(8)),'seed':case['seed'],'capacity':64,'command_group_size':len(cases),'maximum_commands_per_batch':maximum_commands,
   'initialization':'per-group reset with identical declared seed; runtime initial state retained in trace',
   'batch_id':'|'.join(c['case_id'] for c in cases),'physical_step_accounting_owner':i==0,
   'all_simulated_env_steps_attributed':len(arrays['first_episode_valid'])*64 if i==0 else 0,
   'equivalence':'same scoring; physical serial-vs-grouped parity requires separate validation'}
  results[case['case_id']]=(payload,part)
 return results


class CommandGroups:
 """One collector/cache owner shared by training milestones and PPO0 evaluation."""
 def __init__(self,cases,*,maximum_commands=4):
  self.maximum_commands=maximum_commands
  self.pending={};self.groups={row['case_id']:group for group in plan(cases,maximum_commands=maximum_commands) for row in group}
 def collecting_cases(self,row):
  return [] if row['case_id'] in self.pending else self.groups[row['case_id']]
 def acquire(self,env,actor,row,*,stop=None):
  group=self.collecting_cases(row)
  if group:self.pending.update(rollout(env,actor,group,stop=stop,maximum_commands=self.maximum_commands))
  return self.pending.pop(row['case_id'])
