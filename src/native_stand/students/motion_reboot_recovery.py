"""Explicit one-shot boot-loss recovery; historical ledger stays byte-for-byte intact."""
import copy,json,os,time
from pathlib import Path
from .data import sha
VERSION='motion_symmetry85_boot_loss_recovery_v1'

def verify(spec):
 c=spec['motion_training']['wiring_reuse']
 if c['version']!=VERSION:raise ValueError('boot recovery explicit version required')
 parent=Path(c['parent_spec']);run=Path(c['parent_run']);old=json.loads(parent.read_text())
 for p,h in c['inputs'].items():
  if sha(p)!=h:raise ValueError('boot loss evidence changed:'+p)
 receipt=json.loads(Path(c['boot_loss_receipt']).read_text())
 if receipt['parent_spec_sha256']!=sha(parent) or receipt['destination']!=spec['experiment'] or receipt['last_formal_updates']!=8 or receipt['conservative_uncommitted_update_reserve']!=1:raise ValueError('boot recovery receipt identity differs')
 book=json.loads(parent.with_name('budget.json').read_text())
 if book['used']['smoke']!=5 or book['used']['formal']!=8 or list((run/'formal').glob('model_update_*.pt')):raise ValueError('recovery is limited to8 uncheckpointed formal updates')
 attempt=book['attempts'][receipt['parent_attempt_id']]
 if attempt['status']!='RUNNING' or attempt['updates']!=8 or attempt['started_unix_s']>=receipt['current_boot_started_unix_s']:raise ValueError('parent not from preceding boot')
 for pid in receipt['old_owner_pids']:
  path=Path(f'/proc/{pid}/cmdline')
  if path.exists() and str(run).encode() in path.read_bytes():raise ValueError('old owner still active')
 claim=Path(c['claim'])
 if not claim.exists() or json.loads(claim.read_text())['destination']!=spec['experiment']:raise ValueError('single-use boot recovery claim differs')
 from native_stand.command_v6.wiring import validate_persisted_certificate
 if validate_persisted_certificate(run/'smoke',old)['status']!='PASSED':raise ValueError('parent wiring certificate invalid')
 if json.loads((run/'source_eval/symmetry_readback.json').read_text())['status']!='PASSED':raise ValueError('source85 readback invalid')
 expected=copy.deepcopy(old);expected['experiment']=spec['experiment'];expected['artifacts']['run_name']=spec['experiment']
 expected['motion_training']['spec_path']=spec['motion_training']['spec_path'];expected['motion_training']['wiring_reuse']=c
 expected['student_training']['milestones']=list(range(250,2987,250))
 expected['motion_course']['configuration'].update(smoke_updates_max=0,formal_updates_max=2986,stage_updates_max=2986)
 expected['budget'].update(smoke_updates=0,formal_updates=2986,formal_samples=2986*4096*128,sample_limit=2986*4096*128)
 expected['authorization']=spec['authorization']
 if spec!=expected:raise ValueError('boot recovery changes behavior or remaining budget')
 return {'status':'VERIFIED','historical_smoke_updates':5,'historical_formal_updates':8,'conservative_uncommitted_update_reserve':1,'remaining_formal_updates':2986,'recovery_source':'original750;8 formal updates lost; smoke checkpoint not formal baseline','old_ledger_rewritten':False,'capability_accepted':False}

def prepare(parent,*,root,session):
 from native_stand.command_v6.domain.experiment_naming import new_experiment_name
 from native_stand.command_v6.tools.experiment_preparation import publish
 from .motion_reward_audit import audit
 root=Path(root);parent=Path(parent).resolve();old=json.loads(parent.read_text());run=root/'runs'/old['experiment']
 name=new_experiment_name('motion750_symmetry85_reboot_resume');folder=root/'configs/experiments'/name
 receipt_dir=root/'artifacts'/(name+'__recovery');receipt_dir.mkdir(exist_ok=False)
 book=json.loads(parent.with_name('budget.json').read_text());attempt=next((k for k,v in book['attempts'].items() if v['phase']=='formal'))
 manifest=json.loads((run/'formal/manifest.json').read_text())
 boot_started=time.time()-float(Path('/proc/uptime').read_text().split()[0])
 receipt={'version':VERSION,'destination':name,'parent_spec_sha256':sha(parent),'parent_attempt_id':attempt,'last_formal_updates':8,
 'historical_smoke_updates':5,'conservative_uncommitted_update_reserve':1,'current_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 'current_boot_started_unix_s':boot_started,'old_owner_pids':[3981098,manifest['trainer_pid']],
 'user_scope':'User explicitly requests restarting Motion long training after host shutdown; original3000 inclusive budget retained; no LR change',
 'parent_ledger_status':'RUNNING stale after host power loss; derived interrupted classification only, no historical mutation'}
 rp=receipt_dir/'boot_loss_receipt.json';rp.write_text(json.dumps(receipt,indent=2))
 claims=root/'configs/experiments/.motion_boot_recovery_claims';claims.mkdir(exist_ok=True)
 claim=claims/(sha(parent)+'.json')
 with claim.open('x') as f:json.dump({'destination':name,'parent_spec_sha256':sha(parent),'receipt_sha256':sha(rp)},f,indent=2)
 inputs=[parent,parent.with_name('budget.json'),run/'formal/progress.json',run/'formal/metrics.jsonl',run/'formal/manifest.json',run/'smoke/wiring_certificate.json',run/'source_eval/symmetry_readback.json',run/'source_eval/spin_measurement_readback.json',run/'source_eval/candidate_registry.json',rp,claim]
 s=copy.deepcopy(old);s['experiment']=name;s['artifacts']['run_name']=name;s['motion_training']['spec_path']=str(folder/'spec.json')
 s['motion_training']['wiring_reuse']={'version':VERSION,'parent_spec':str(parent),'parent_run':str(run),'inputs':{str(p):sha(p) for p in inputs},
 'boot_loss_receipt':str(rp),'claim':str(claim),'historical_smoke_updates':5,'historical_formal_updates':8,'conservative_uncommitted_update_reserve':1,
 'historical_wall_s':json.loads((run/'formal/progress.json').read_text())['latest_metric']['wall_s']+sum(float(v.get('wall_s') or 0.) for v in book['attempts'].values() if v['phase']!='formal'),
 'original_online_limit':3000}
 s['student_training']['milestones']=list(range(250,2987,250));s['motion_course']['configuration'].update(smoke_updates_max=0,formal_updates_max=2986,stage_updates_max=2986)
 s['budget'].update(smoke_updates=0,formal_updates=2986,formal_samples=2986*4096*128,sample_limit=2986*4096*128)
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':session,'user_scope':receipt['user_scope']}
 verify(s)
 phases=('source_eval','smoke','formal','evaluate','play')
 ledger={'authorization':'UNAUTHORIZED','experiment':name,'hapi_session':session,'limits':dict(zip(phases,(0,0,2986,0,0))),'used':{k:0 for k in phases},'attempts':{},'total_online_PPO_limit':2986,
 'prior_consumption':{'source_spec':str(parent),'source_budget_sha256':sha(parent.with_name('budget.json')),'smoke':5,'formal':8,'uncommitted_reserve':1,'cumulative_limit':3000}}
 publish(folder,{'spec.json':s,'budget.json':ledger,'reward_snapshot.json':audit(s['reward']),'recovery_plan.json':receipt},s['student_dependencies'],completion_marker=True)
 return {'spec':str(folder/'spec.json'),'run':str(root/'runs'/name),'receipt':str(rp),'remaining_formal':2986,'historical_charged':13,'conservative_reserve':1,'source':s['source_checkpoint']}
