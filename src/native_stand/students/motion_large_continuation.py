"""Explicit one-shot same-recipe large Motion continuation, never an auto retry."""
import copy,json
from pathlib import Path
from .data import sha
VERSION='motion31_large_same_recipe_continuation_v1'

def latest(folder):
 """Use full, readable atomic checkpoints, ordered by numeric update."""
 import torch
 rows=[]
 for p in Path(folder).glob('model_update_*.pt'):
  n=int(p.stem.rsplit('_',1)[1])
  rows.append((n,p))
 if not rows:raise ValueError('parent has no saved checkpoint')
 n,p=max(rows)
 data=torch.load(p,map_location='cpu',weights_only=False)
 if data.get('update')!=n or not all(k in data for k in ('actor','critic','optimizer','training_rng','reward_refinement_state')):raise ValueError('latest checkpoint incomplete; do not silently fall back')
 return p.resolve(),data

def verify(spec):
 from .motion_large_combo import verify as verify_parent,network
 from native_stand.command_v6.wiring import validate_persisted_certificate
 c=spec['motion_large_continuation'];parent=Path(c['parent_spec']);run=Path(c['parent_run'])
 if c['version']!=VERSION or sha(parent)!=c['parent_spec_sha256']:raise ValueError('continuation parent identity differs')
 old=json.loads(parent.read_text());verify_parent(old)
 auth_path=Path(c['authorization_path']);auth=json.loads(auth_path.read_text())
 if sha(auth_path)!=c['authorization_sha256'] or auth['parent']['spec_sha256']!=sha(parent) or auth['outputs']['spec_path']!=spec['motion_training']['spec_path'] or auth['recipe']['formal_updates']!=2000:raise ValueError('continuation authorization differs')
 for name in ('chain_terminal_report.json','report_ready.json'):
  if sha(run/name)!=c['evidence_sha256'][str(run/name)]:raise ValueError('continuation terminal evidence differs')
 terminal=json.loads((run/'chain_terminal_report.json').read_text());ready=json.loads((run/'report_ready.json').read_text());book=json.loads(parent.with_name('budget.json').read_text())
 if terminal['status'] not in ('COMPLETED','STOPPED_BOUNDED') or terminal.get('failure') or ready['report_sha256']!=sha(run/'completion_report.md') or any(a['status']=='RUNNING' or a.get('exit_code')!=0 for a in book['attempts'].values()):raise ValueError('continuation parent not normally settled')
 if validate_persisted_certificate(run/'smoke',old)['status']!='PASSED':raise ValueError('parent wiring certificate invalid')
 p,data=latest(run/'formal')
 if str(p)!=spec['source_checkpoint'] or sha(p)!=spec['source_sha256'] or data.get('spec_sha256')!=sha(run/'formal/source_snapshot/spec.json') or data.get('asset_sha256')!=old['asset_sha256'] or data.get('network_architecture')!=network():raise ValueError('latest full-state source identity differs')
 if {g['lr'] for g in data['optimizer']['param_groups']}!={old['ppo']['learning_rate']}:raise ValueError('continuation Adam LR differs')
 expected=copy.deepcopy(old)
 expected['experiment']=spec['experiment'];expected['artifacts']['run_name']=spec['experiment']
 expected['source_checkpoint']=str(p);expected['source_sha256']=sha(p)
 expected['motion_training']['spec_path']=spec['motion_training']['spec_path']
 expected['motion_training']['resume']={'parent_spec':str(parent),'parent_spec_sha256':sha(parent),'formal_dir':str(run/'formal'),'parent_recipe_updates':data['recipe_updates']}
 expected['motion_course']['source_recipe_updates']=data['recipe_updates']
 expected['motion_course']['configuration'].update(formal_updates_max=2000,smoke_updates_max=0,stage_updates_max=2000)
 expected.pop('motion_startup',None)
 expected['student_training']['source_actor_sha256']=sha(p)
 expected['student_training']['continuation_teacher']={'checkpoint':old['source_checkpoint'],'sha256':old['source_sha256']}
 expected['student_training']['milestones']=list(range(250,2001,250))
 for row in expected['student_evaluation']['cases']:row.update(checkpoint=str(p),sha256=sha(p))
 expected['budget'].update(formal_updates=2000,smoke_updates=0,sample_limit=2000*4096*128,formal_samples=2000*4096*128,formal_wall_s=63000,chain_wall_s=64800)
 if c.get('execution_policy')=='motion31_unlimited_wall_authorized_retry_v1':
  if auth['budget'].get('chain_wall_limit_s','absent') is not None or auth['budget'].get('formal_wall_limit_s','absent') is not None:
   raise ValueError('unlimited wall requires explicit current authorization')
  recovery=c['source_evidence_retry'];failed=Path(recovery['failed_run'])
  if sha(failed/'chain_terminal_report.json')!=recovery['terminal_sha256']:raise ValueError('source retry terminal SHA differs')
  failed_terminal=json.loads((failed/'chain_terminal_report.json').read_text())
  failed_book=json.loads(Path(recovery['failed_spec']).with_name('budget.json').read_text())
  if failed_terminal['status']!='FAILED' or failed_book['used']['formal'] or failed_book['used']['smoke'] or any(a['status']=='RUNNING' for a in failed_book['attempts'].values()):raise ValueError('retry requires settled PPO0 failure')
  if abs(recovery['historical_wall_s']-failed_terminal['chain_wall_s'])>1e-6:raise ValueError('retry historical wall differs')
  expected['budget'].update(formal_wall_s=None,chain_wall_s=None,source_eval_wall_s=7200)
  expected['motion_startup']=spec['motion_startup']
  from .motion_startup import resolve
  records,missing=resolve(spec)
  if len(records)!=5 or [r['command'][:2] for r in missing]!=[[0.,.5],[0.,-.5]]:raise ValueError('retry must reuse five complete810 cases and fill both yaw')
 from .performance import parent_motion_performance_matches
 if not parent_motion_performance_matches(spec.get('performance'),old.get('performance')):raise ValueError('continuation unsupported performance drift')
 expected['performance']=spec.get('performance')
 expected['authorization']=spec['authorization'];expected['student_dependencies']=spec['student_dependencies']
 expected['motion_large_continuation']=c
 if spec!=expected:raise ValueError('same-recipe continuation has undeclared differences')
 if spec['authorization']['status']!='AUTHORIZED' or spec['authorization']['hapi_session']!=auth['authorization']['owner_session']:raise ValueError('continuation owner authorization differs')
 return True
