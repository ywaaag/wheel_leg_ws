"""Fresh5000-online geometric spin experiment from settled symmetry85/u1000."""
import copy,json
from pathlib import Path
from .data import sha
SOURCE='570934e44541b102498f50303ce50b552f9b9e49463c3723dffb243b08423ca5'
VERSION='motion31_differential85_5000_v1'
RETRY='motion_differential_preppo_save_retry_v1'

def retry_evidence(parent):
 """Only the settled zero-PPO compact-field failure; no repeated retry chain."""
 p=Path(parent).resolve();old=json.loads(p.read_text())
 if old.get('motion_differential_retry'):raise ValueError('retry cannot recursively authorize another retry')
 # Project root is configs/experiments/<name>/spec.json -> parents[3].
 run=p.parents[3]/'runs'/old['experiment']
 book=json.loads(p.with_name('budget.json').read_text());chain=json.loads((run/'chain_terminal_report.json').read_text());term=json.loads((run/'source_eval/terminal_report.json').read_text())
 if chain['status']!='FAILED' or term['failure']['exception']!="ValueError('differential measurement missing')" or term['actual_updates']!=0:raise ValueError('retry requires exact pre-PPO save fault')
 if book['used']['smoke'] or book['used']['formal'] or any(a['status']=='RUNNING' for a in book['attempts'].values()):raise ValueError('retry parent must be settled zero-PPO')
 if (run/'formal').exists() or (run/'smoke').exists():raise ValueError('unexpected parent online phase')
 required=[p,p.with_name('budget.json'),run/'chain_terminal_report.json',run/'source_eval/terminal_report.json']
 for i in range(7):
  raw=run/f'source_eval/evaluation_{i:02}.json';d=json.loads(raw.read_text());trace=raw.parent/d['trace']['path']
  if sha(trace)!=d['trace']['sha256']:raise ValueError('parent trace changed')
  required.extend((raw,trace))
 return {'version':RETRY,'parent_spec':str(p),'parent_run':str(run),'inputs':{str(x):sha(x) for x in required},'historical_online_updates':0,'historical_wall_s':chain['chain_wall_s'],'original_online_limit':5000,'scope':'new source readback required; missing differential arrays never reused as certificate'}

def build(parent,checkpoint,*,experiment,spec_path):
 import torch
 from .motion_differential import configuration
 from .motion_guard import behavior_mode
 p=Path(parent).resolve();cp=Path(checkpoint).resolve();s=json.loads(p.read_text());saved=torch.load(cp,map_location='cpu',weights_only=False)
 if sha(cp)!=SOURCE or saved['spec_sha256']!=sha(p) or saved['critic_privileged_version']!='motion_symmetry_privileged85_v1' or saved['update']!=1000:raise ValueError('exact settled symmetry85/u1000 required')
 run=cp.parent.parent;term=json.loads((run/'chain_terminal_report.json').read_text());book=json.loads(p.with_name('budget.json').read_text())
 if term['status']!='STOPPED_BOUNDED' or term['formal_updates']!=1000 or book['used']['formal']!=1000 or any(a['status']=='RUNNING' for a in book['attempts'].values()):raise ValueError('parent not settled1000')
 if {g['lr'] for g in saved['optimizer']['param_groups']}!={3e-5}:raise ValueError('source Adam LR differs')
 s=copy.deepcopy(s);s.update(experiment=experiment,source_checkpoint=str(cp),source_sha256=SOURCE,reward=configuration(),motion_behavior_guard=behavior_mode())
 s['artifacts'].update(run_name=experiment,storage_max_bytes=4*2**30)
 # New reward, identical observation/action/Actor31/Critic85/PPO and old frozen750 teacher.
 r=s['student_training'];r.update(source_actor_sha256=SOURCE,reward=configuration(),milestones=list(range(250,4996,250)))
 r['critic']='source61_to85_zero_new_columns';r['optimizer']='source_Adam_pad_new_critic_columns_zero'
 s['motion_training'].update(spec_path=str(Path(spec_path).resolve()),resume={'parent_spec':str(p),'parent_spec_sha256':sha(p),'formal_dir':str(cp.parent),'parent_recipe_updates':saved['recipe_updates']})
 s['motion_training'].pop('wiring_reuse',None)
 s['motion_course']['source_recipe_updates']=saved['recipe_updates'];s['motion_course']['configuration'].update(formal_updates_max=4995,stage_updates_max=4995,smoke_updates_max=5)
 s['motion_symmetry']['budget']='new5000 inclusive=smoke5+formal4995; no PPO/chain wall ceiling; no transfer of old1986'
 s['motion_symmetry']['stage_scope']='S0 original7 only; no speed/yaw/backward unlock; difference target plus preserved costs'
 s['motion_differential']={'version':VERSION,'parent_spec':str(p),'parent_spec_sha256':sha(p),'source_checkpoint_sha256':SOURCE,'online_limit':5000,
 'restore':'source1000 Actor/Critic85/Adam/std/RNG; new environment reset; five critic warmup formal updates counted','teacher':'original750 only, old five cohorts KL.01, yaw excluded','behavior_regression':'observe only; acceptance and physical faults unchanged'}
 s['motion_startup']['references']=[];s['motion_startup']['missing_commands']=[x['command'] for x in s['student_evaluation']['cases']]
 s['budget'].update(smoke_updates=5,formal_updates=4995,formal_samples=4995*4096*128,sample_limit=5000*4096*128,formal_wall_s=None,chain_wall_s=None,source_eval_wall_s=3600,smoke_wall_s=1800,evaluation_wall_s=1800)
 for row in s['student_evaluation']['cases']:row.update(checkpoint=str(cp),sha256=SOURCE)
 s['student_dependencies'].update({str(p):sha(p),str(cp):SOURCE,str(run/'chain_terminal_report.json'):sha(run/'chain_terminal_report.json'),str(p.with_name('budget.json')):sha(p.with_name('budget.json'))})
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'User approves new5000 inclusive unlimited training wall, geometric differential target, preserved costs, observe-only behavior guard, GPU0 shared'}
 s['resources']={'mode':'shared','sharing_authorized':True,'authorization_note':'User approved next Motion long5000; inherited GPU0 shared; no preemption'}
 return s

def verify(spec):
 c=spec['motion_differential']
 if c['version']!=VERSION or sha(c['parent_spec'])!=c['parent_spec_sha256']:raise ValueError('differential declaration identity differs')
 expected=build(c['parent_spec'],spec['source_checkpoint'],experiment=spec['experiment'],spec_path=spec['motion_training']['spec_path']);expected['authorization']=spec['authorization']
 if spec.get('motion_differential_retry'):
  evidence=retry_evidence(spec['motion_differential_retry']['parent_spec'])
  if evidence!=spec['motion_differential_retry']:raise ValueError('retry frozen evidence differs')
  expected['motion_differential_retry']=evidence
  expected['student_dependencies'].update(evidence['inputs'])
 if spec!=expected:raise ValueError('undeclared differential configuration drift')
 from .training_recipe import validate_instance
 validate_instance(spec['student_training'],spec['source_sha256'])
 return True
