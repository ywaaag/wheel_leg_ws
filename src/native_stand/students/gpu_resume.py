"""Authorized same-recipe Recovery device migration; no quota reset or PhysX replay."""
import copy,json
from pathlib import Path
import torch
from .data import sha
VERSION='recovery_large_gpu_migration_v1'

def verify(spec):
 c=spec['student_gpu_resume']
 if set(c)!={'version','parent_spec','parent_run','inputs','parent_checkpoint_update','historical_online_updates','original_online_limit','unused_parent_formal_updates','uncheckpointed_updates'} or c['version']!=VERSION:raise ValueError('GPU resume schema differs')
 for path,digest in c['inputs'].items():
  if sha(path)!=digest:raise ValueError('GPU resume evidence SHA differs: '+path)
 parent=Path(c['parent_spec']);run=Path(c['parent_run']);old=json.loads(parent.read_text());book=json.loads(parent.with_name('budget.json').read_text())
 terminal=json.loads((run/'chain_terminal_report.json').read_text())
 if terminal['status']!='STOPPED_BOUNDED' or terminal.get('failure') or any(a['status']=='RUNNING' for a in book['attempts'].values()):raise ValueError('GPU migration parent not normally stopped/settled')
 from native_stand.command_v6.wiring import validate_persisted_certificate
 if validate_persisted_certificate(run/'smoke',old)['status']!='PASSED':raise ValueError('parent original smoke evidence invalid')
 from .model import load
 _,data=load(spec['source_checkpoint'],'RecoveryStudent')
 n=c['parent_checkpoint_update'];used=sum(book['used'][k] for k in ('smoke','formal'));limit=sum(book['limits'][k] for k in ('smoke','formal'))
 if n!=data['update'] or sha(spec['source_checkpoint'])!=spec['source_sha256'] or data['spec_sha256']!=sha(run/'formal/source_snapshot/spec.json'):raise ValueError('migration full checkpoint identity differs')
 if used!=c['historical_online_updates'] or limit!=c['original_online_limit'] or c['unused_parent_formal_updates']!=limit-used or c['uncheckpointed_updates']!=book['used']['formal']-n:raise ValueError('migration historical accounting differs')
 if spec['budget']['smoke_updates']!=0 or spec['budget']['formal_updates']!=limit-used:raise ValueError('migration remaining quota differs')
 for k in ('ppo','reward','action','control','ground','termination','student_interface','network_architecture','performance','recovery','curriculum','seed'):
  if spec[k]!=old[k]:raise ValueError('migration changes frozen behavior: '+k)
 new_recipe=copy.deepcopy(spec['student_training']);old_recipe=copy.deepcopy(old['student_training'])
 for k in ('source_actor_sha256','milestones'):
  new_recipe.pop(k);old_recipe.pop(k)
 if new_recipe!=old_recipe or spec['student_training']['source_actor_sha256']!=spec['source_sha256']:raise ValueError('migration reward/PPO recipe differs')
 if spec['device']['physical_index']!=1 or spec['device']['logical']!='cuda:0':raise ValueError('explicit GPU1/logical0 required')
 if spec['artifacts']['storage_max_bytes']!=old['artifacts']['storage_max_bytes']:raise ValueError('migration storage ceiling differs')
 if not all(k in data for k in ('actor','critic','optimizer','training_rng')):raise ValueError('incomplete training state')
 return {'verified':True,'checkpoint_update':n,'historical_online_updates':used,'remaining_updates':limit-used,'discarded_but_charged_updates':c['uncheckpointed_updates'],'warmup_repeated':False,'physical_state':'new real episode reset; PhysX state not restored'}

def restore(ppo,data,spec):
 report=verify(spec)
 ppo.actor.load_state_dict(data['actor']);ppo.critic.load_state_dict(data['critic']);ppo.optimizer.load_state_dict(data['optimizer'])
 if any(g['lr']!=spec['ppo']['learning_rate'] for g in ppo.optimizer.param_groups):raise ValueError('resume optimizer LR differs')
 for param,state in ppo.optimizer.state.items():
  for k,v in state.items():
   if torch.is_tensor(v) and v.numel()>1 and (v.shape!=param.shape or not torch.isfinite(v).all()):raise ValueError('resume Adam tensor invalid')
 from .resume import restore_rng
 restore_rng(data['training_rng'])
 return {**report,'source_actor_loaded':True,'source_critic_loaded':True,'source_Adam_loaded':True,'source_std_loaded':True,'source_RNG_loaded':True,'Adam_state_count':len(ppo.optimizer.state)}
