"""File-boundary validator for trained Motion31 three-evaluation-seed PPO0."""
import json
from pathlib import Path
from .data import sha
VERSION='motion31_source_three_seed_confirmation_v1'
SEEDS=[20261014,20261015,20261016]
LIMIT_VERSION='motion31_forward_limit_probe_v1'
LIMIT_COMMANDS=[[2.5,0.],[3.,0.],[3.5,0.],[4.,0.]]
PARALLEL_VERSION='motion31_parallel_full_checkpoint_ppo0_v1'

def parallel_commands():
 from .motion_parallel_design import BASE,LEVELS,endpoints
 values=[(0.,0.),*BASE]
 for lane,levels in LEVELS.items():
  for level in range(len(levels)):values.extend(endpoints(lane,level))
 return [list(v)+[.34] for v in dict.fromkeys(values)]

def verify_parallel(spec):
 import torch
 c=spec['motion_confirmation'];parent=Path(c['parent_spec']);p=json.loads(parent.read_text())
 if set(c)!={'version','parent_spec','parent_spec_sha256','checkpoint_sha256','seeds'} or c['version']!=PARALLEL_VERSION or c['seeds']!=[20261014]:raise ValueError('parallel confirmation schema/seed differs')
 if sha(parent)!=c['parent_spec_sha256'] or sha(spec['source_checkpoint'])!=c['checkpoint_sha256'] or spec['source_sha256']!=c['checkpoint_sha256']:raise ValueError('parallel confirmation source identity differs')
 cp=torch.load(spec['source_checkpoint'],map_location='cpu',weights_only=False)
 if cp.get('spec_sha256')!=sha(parent) or cp.get('asset_sha256')!=p['asset_sha256'] or cp.get('critic_privileged_version')!='motion_symmetry_privileged85_v1':raise ValueError('parallel checkpoint internal identity differs')
 if not spec.get('evaluation_only') or any(spec['budget'][k] for k in ('source_eval_updates','smoke_updates','formal_updates','sample_limit','formal_samples')):raise ValueError('parallel confirmation PPO0 only')
 recipe=dict(spec['student_training']);recipe['storage_max_bytes']=p['student_training']['storage_max_bytes']
 if recipe!=p['student_training']:raise ValueError('parallel confirmation training recipe differs')
 for k in ('reward','motion_course','motion_parallel','motion_parallel_assessment','motion_assessment','action','control','ground','observation','history','ppo','student_interface','network_architecture','performance','device','asset_path','asset_sha256'):
  if spec[k]!=p[k]:raise ValueError('parallel confirmation frozen contract differs: '+k)
 rows=spec['student_evaluation']['cases'];expected=parallel_commands()
 if [r['command'] for r in rows]!=expected or len(rows)!=27:raise ValueError('parallel confirmation requires full27 commands')
 if any(r['seed']!=20261014 or r['checkpoint']!=spec['source_checkpoint'] or r['sha256']!=spec['source_sha256'] or r['role']!='MotionZeroStudent' or r['kind']!='student' or r['family']!='motion_native_normal' or r['yaw_rad']!=0. or r['seconds']!=31.52 for r in rows):raise ValueError('parallel confirmation case identity/window differs')
 if spec['evaluation']!=p['evaluation'] or spec['student_evaluation']['version']!=p['student_evaluation']['version']:raise ValueError('parallel confirmation scoring differs')
 return True

def verify(spec):
 if spec.get('motion_confirmation',{}).get('version')==PARALLEL_VERSION:return verify_parallel(spec)
 if spec.get('motion_confirmation',{}).get('version')=='motion31_trained_spin_three_seed_v1':
  return verify_trained_spin(spec)
 from .motion_guard import protocol
 from .motion_training import commands
 from .motion_tracking_reward import QUALITY_VERSION
 c=spec.get('motion_confirmation',{})
 limit=c.get('version')==LIMIT_VERSION
 expected_seeds=[20261015] if limit else SEEDS
 if set(c)!={'version','parent_spec','parent_spec_sha256','checkpoint_sha256','seeds'} or c['version'] not in (VERSION,LIMIT_VERSION) or c['seeds']!=expected_seeds:raise ValueError('confirmation schema/seeds differ')
 parent=Path(c['parent_spec']);p=json.loads(parent.read_text())
 if sha(parent)!=c['parent_spec_sha256'] or sha(spec['source_checkpoint'])!=c['checkpoint_sha256'] or spec['source_sha256']!=c['checkpoint_sha256']:raise ValueError('confirmation parent/checkpoint SHA differs')
 if not spec.get('evaluation_only') or any(spec['budget'][k] for k in ('smoke_updates','formal_updates','sample_limit','formal_samples')):raise ValueError('confirmation PPO0 only')
 if spec['reward']!=p['reward'] or spec['reward']['version']!=QUALITY_VERSION or spec['student_training']!=p['student_training']:raise ValueError('confirmation preserves original v3 reward/recipe')
 if spec['motion_assessment']!=protocol() or spec['student_evaluation']['version']!=protocol()['version'] or spec['evaluation']['protocol_identity']!=protocol()['version']:raise ValueError('confirmation effective protocol differs')
 expected=[(seed,v+[.34]) for seed in expected_seeds for v in (LIMIT_COMMANDS if limit else commands(0))]
 rows=spec['student_evaluation']['cases']
 if [(r['seed'],r['command']) for r in rows]!=expected or any(r['checkpoint']!=spec['source_checkpoint'] or r['sha256']!=spec['source_sha256'] or r['family']!='motion_native_normal' or r['role']!='MotionZeroStudent' or r['seconds']!=31.52 for r in rows):raise ValueError('confirmation full15 case matrix differs')
 if spec['evaluation']['scored_envs']!=8 or spec['evaluation']['seconds']!=31.52:raise ValueError('confirmation duration/count differs')
 import torch
 cp=torch.load(spec['source_checkpoint'],map_location='cpu',weights_only=False)
 if cp.get('critic_privileged_version')!='motion_privileged49_v1' or cp['training_recipe']['reward']!=spec['reward'] or cp.get('recipe_version')!='student_motion31_training_v1':raise ValueError('trained Motion49 source contract differs')
 for k in ('action','control','ground','observation','history','ppo','student_interface'):
  if spec[k]!=p[k]:raise ValueError('confirmation frozen contract differs: '+k)
 return True


def verify_trained_spin(spec):
 """Same trained spin recipe and scoring; independent PPO0 matrix only."""
 import torch
 c=spec['motion_confirmation'];parent=Path(c['parent_spec'])
 if set(c)!={'version','parent_spec','parent_spec_sha256','checkpoint_sha256','seeds'} or c['seeds']!=SEEDS:raise ValueError('spin confirmation schema/seeds differ')
 if sha(parent)!=c['parent_spec_sha256'] or sha(spec['source_checkpoint'])!=c['checkpoint_sha256'] or spec['source_sha256']!=c['checkpoint_sha256']:raise ValueError('spin confirmation source SHA differs')
 p=json.loads(parent.read_text());cp=torch.load(spec['source_checkpoint'],map_location='cpu',weights_only=False)
 if cp.get('spec_sha256')!=sha(parent) or cp.get('critic_privileged_version')!='motion_spin_privileged61_v1':raise ValueError('spin confirmation internal checkpoint identity differs')
 if not spec.get('evaluation_only') or any(spec['budget'][k] for k in ('smoke_updates','formal_updates','sample_limit','formal_samples')):raise ValueError('spin confirmation is PPO0 only')
 for key in ('reward','student_training','motion_assessment','action','control','ground','observation','history','ppo','student_interface','network_architecture','motion_spin_quality','motion_course'):
  if spec[key]!=p[key]:raise ValueError('spin confirmation parent contract differs:'+key)
 commands=[r['command'] for r in p['student_evaluation']['cases']]
 if commands!=[[0.,0.,.34],[.5,0.,.34],[1.,0.,.34],[1.5,0.,.34],[2.,0.,.34],[0.,.5,.34],[0.,-.5,.34]]:raise ValueError('spin confirmation parent seven commands differ')
 rows=spec['student_evaluation']['cases']
 if [(r['seed'],r['command']) for r in rows]!=[(seed,cmd) for seed in SEEDS for cmd in commands]:raise ValueError('spin confirmation requires21 cases')
 if any(r['checkpoint']!=spec['source_checkpoint'] or r['sha256']!=spec['source_sha256'] or r['role']!='MotionZeroStudent' or r['family']!='motion_native_normal' or r['seconds']!=31.52 for r in rows):raise ValueError('spin confirmation case identity/window differs')
 if spec['evaluation']['scored_envs']!=8 or spec['evaluation']['envs']!=64 or spec['evaluation']['seconds']!=31.52 or spec['student_evaluation']['version']!=p['student_evaluation']['version']:raise ValueError('spin confirmation scoring/count differs')
 return True


def finalize_parallel(output,spec,status):
 """Owner-only terminal summary, preserving all original gate results."""
 from native_stand.command_v6.completion import write
 output=Path(output);terminal=json.loads((output/'terminal_report.json').read_text());rows=[]
 for condition in terminal['conditions']:
  if condition['status']!='COMPLETED':continue
  file=output/condition['result'];d=json.loads(file.read_text());envs=d['per_env']
  if d['checkpoint_sha256']!=spec['source_sha256'] or len(envs)!=8:raise ValueError('full report identity/count differs')
  import collections
  rows.append({'command':d['case']['command'],'passed':sum(e['accepted'] for e in envs),'scored_envs':8,
   'response_mean':sum(e['tracking']['mean_signed_speed'] for e in envs)/8,
   'MAE_mean':sum(e['tracking']['MAE'] for e in envs)/8,
   'pitch_peak':max(e['tracking']['pitch_hold_abs_max_rad'] for e in envs),
   'roll_peak':max(e['tracking']['roll_hold_abs_max_rad'] for e in envs),
   'failures':dict(collections.Counter(k for e in envs for k in e['failed_checks'])),
   'result':str(file),'sha256':sha(file)})
 complete=status=='COMPLETED' and len(rows)==27
 write(output/'full_assessment_summary.json',{'status':status,'checkpoint':spec['source_checkpoint'],'checkpoint_sha256':spec['source_sha256'],
  'commands_completed':len(rows),'expected_commands':27,'expected_episodes':216,'full_matrix_complete':complete,
  'all_commands_passed':complete and all(r['passed']==8 for r in rows),'multi_seed_certified':False,'PPO':0,'rows':rows})
 lines=['# Motion4995完整性能评估','',f'状态：{status}；完成{len(rows)}/27命令，每命令8环境；PPO0。',
  '沿用原native完整窗口与评分；包含未训练±4探测，不授多seed或实机认证。','',
  '|命令|通过|响应均值|MAE均值|pitch峰值rad|roll峰值rad|失败原因|','|---|---|---:|---:|---:|---:|---|']
 for r in rows:lines.append(f"|{r['command'][:2]}|{r['passed']}/8|{r['response_mean']:.5f}|{r['MAE_mean']:.5f}|{r['pitch_peak']:.5f}|{r['roll_peak']:.5f}|{r['failures']}|")
 (output/'completion_report.md').write_text('\n'.join(lines)+'\n')
 write(output/'report_ready.json',{'status':'report_ready','report_sha256':sha(output/'completion_report.md')})
