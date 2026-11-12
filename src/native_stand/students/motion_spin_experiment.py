"""CPU-only explicit spin-quality experiment construction and validation; no ledger."""
import copy,json
from pathlib import Path
VERSION='motion31_spin_quality61_v1'
SOURCE_SHA='475bc778376d8e426ffa38dfb745e16d5a0c82aea8bfff812d52ff38b6efc0f2'

def build(parent_spec,checkpoint,*,experiment,spec_path,formal_updates=1000,unlimited_wall=False):
 from .data import sha
 from .model import load
 from .motion_tracking_reward import configuration,SPIN_VERSION
 from .motion_spin_privileged import contract,network,VERSION as CRITIC
 from .motion_startup import smoke_profile
 from .performance import new_motion_defaults
 if type(formal_updates) is not int or not 1<=formal_updates<=2000:raise ValueError('proposed formal limit must be1..2000; not authorization')
 parent=Path(parent_spec).resolve();source=Path(checkpoint).resolve();p=json.loads(parent.read_text());_,data=load(source,'MotionZeroStudent')
 if sha(source)!=SOURCE_SHA or data['spec_sha256']!=sha(parent) or data['update']!=810:raise ValueError('spin quality source must be original large810 with its actual spec')
 s=copy.deepcopy(p);s.update(experiment=experiment,source_checkpoint=str(source),source_sha256=SOURCE_SHA,
  network_architecture=network(),student_interface=contract(),observation={'frame_dim':22,'actor_dim':31,'critic_dim':61},reward=configuration(SPIN_VERSION))
 s['artifacts']['run_name']=experiment
 s['motion_training'].update(spec_path=str(Path(spec_path).resolve()),resume={'parent_spec':str(parent),'parent_spec_sha256':sha(parent),'formal_dir':str(source.parent),'parent_recipe_updates':data['recipe_updates']})
 s['motion_course']['source_recipe_updates']=data['recipe_updates'];s['motion_course']['configuration'].update(formal_updates_max=formal_updates,stage_updates_max=formal_updates)
 s['student_training'].update(source_actor_sha256=SOURCE_SHA,reward=s['reward'],critic_privileged_version=CRITIC,critic='source49_to61_zero_new_columns',optimizer='source_Adam_pad_new_critic_columns_zero',network_architecture=network(),milestones=list(range(250,formal_updates+1,250)),smoke_wiring=smoke_profile(),
  continuation_teacher={'checkpoint':p['source_checkpoint'],'sha256':p['source_sha256']})
 s['motion_startup']={'version':'motion31_minimal_startup_v1','source_policy':'reuse_exact_source_evidence_then_evaluate_only_missing_endpoints','smoke_policy':'motion31_wiring5_no_capability_eval_v2','references':[],'missing_commands':[row['command'] for row in s['student_evaluation']['cases']],'source_evaluation_is_not_wiring':True,'no_capability_prerequisite_for_smoke':True}
 for row in s['student_evaluation']['cases']:row.update(checkpoint=str(source),sha256=SOURCE_SHA)
 s['budget'].update(formal_updates=formal_updates,formal_samples=formal_updates*4096*128,sample_limit=(formal_updates+5)*4096*128,source_eval_wall_s=7200,chain_wall_s=64800,formal_wall_s=54000)
 if unlimited_wall:s['budget'].update(chain_wall_s=None,formal_wall_s=None)
 s['student_dependencies'][str(parent)]=sha(parent);s['student_dependencies'][str(source)]=SOURCE_SHA
 s['performance']=new_motion_defaults(p.get('performance'))
 s['performance'].update(evaluation_group_commands=True,evaluation_group_max_commands=7,trace_buffer_max_bytes=512*2**20)
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'CPU preparation only; new explicit budget/GPU/session authorization required'}
 s['motion_spin_quality']={'version':VERSION,'parent_spec':str(parent),'parent_spec_sha256':sha(parent),'formal_updates_proposed':formal_updates,'unlimited_wall':bool(unlimited_wall),
  'physical_calibration_required':True,'calibration_evidence':None,'scope':'combined reward/contact measurement/Critic61; not single-variable causality'}
 return s

def verify(spec):
 if spec.get('motion_timing'):
  from .motion_timing_experiment import verify as verify_timing
  return verify_timing(spec)
 if spec.get('motion_confirmation',{}).get('version')=='motion31_parallel_full_checkpoint_ppo0_v1':
  from .motion_confirmation import verify as verify_confirmation
  return verify_confirmation(spec)
 if spec.get('motion_parallel'):
  from .motion_parallel_experiment import verify as verify_parallel
  return verify_parallel(spec)
 if spec.get("motion_symmetry"):
  from .motion_symmetry_experiment import verify as verify_symmetry
  return verify_symmetry(spec)
 if spec.get('motion_confirmation',{}).get('version')=='motion31_trained_spin_three_seed_v1':
  from .motion_confirmation import verify_trained_spin
  return verify_trained_spin(spec)
 if spec.get('motion_training',{}).get('wiring_reuse',{}).get('version')==RECOVERY:return verify_recovery(spec)
 from .data import sha
 from .motion_large_combo import verify as verify_parent
 c=spec['motion_spin_quality']
 if c.get('version')!=VERSION:raise ValueError('unknown spin experiment')
 p=Path(c['parent_spec'])
 if sha(p)!=c['parent_spec_sha256']:raise ValueError('spin parent SHA differs')
 verify_parent(json.loads(p.read_text()))
 expected=build(p,spec['source_checkpoint'],experiment=spec['experiment'],spec_path=spec['motion_training']['spec_path'],formal_updates=c['formal_updates_proposed'],unlimited_wall=c['unlimited_wall'])
 # Only authorization metadata may be filled after explicit user approval. Physical
 # measurement calibration remains mandatory before a submission can reserve PPO.
 expected['authorization']=spec['authorization']
 if spec!=expected:raise ValueError('spin recipe has undeclared differences')
 from .training_recipe import validate_instance
 validate_instance(spec['student_training'],spec['source_sha256'])
 return True

RECOVERY='motion31_spin_preformal_dispatch_recovery_v1'

def recover_spec(parent_spec,*,experiment,spec_path):
 from .data import sha
 parent=Path(parent_spec).resolve();old=json.loads(parent.read_text());s=copy.deepcopy(old)
 run=parent.parents[3]/'runs'/old['experiment']
 s['experiment']=experiment;s['artifacts']['run_name']=experiment;s['motion_training']['spec_path']=str(Path(spec_path).resolve())
 s['budget'].update(smoke_updates=0,sample_limit=1995*4096*128)
 s['motion_course']['configuration']['smoke_updates_max']=0
 inputs={str(p):sha(p) for p in (parent,parent.with_name('budget.json'),run/'chain_terminal_report.json',run/'source_eval/spin_measurement_readback.json',run/'smoke/wiring_certificate.json')}
 s['motion_training']['wiring_reuse']={'version':RECOVERY,'parent_spec':str(parent),'parent_run':str(run),'inputs':inputs,
  'historical_smoke_updates':5,'historical_wall_s':json.loads((run/'chain_terminal_report.json').read_text())['chain_wall_s'],'original_online_limit':2000}
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'new explicit user retry authorization required; remaining1995 only'}
 return s

def verify_recovery(spec):
 from .data import sha
 from native_stand.command_v6.wiring import validate_persisted_certificate
 c=spec['motion_training']['wiring_reuse'];parent=Path(c['parent_spec']);run=Path(c['parent_run']);old=json.loads(parent.read_text());verify(old)
 for p,h in c['inputs'].items():
  if sha(p)!=h:raise ValueError('spin recovery evidence changed')
 terminal=json.loads((run/'chain_terminal_report.json').read_text());book=json.loads(parent.with_name('budget.json').read_text())
 if terminal['status']!='FAILED' or terminal['failure']!="KeyError('task_manifest_sha256')" or book['used']['smoke']!=5 or book['used']['formal']!=0 or any(x['status']=='RUNNING' for x in book['attempts'].values()):raise ValueError('spin retry requires settled exact preformal failure')
 if validate_persisted_certificate(run/'smoke',old)['status']!='PASSED' or json.loads((run/'source_eval/spin_measurement_readback.json').read_text())['status']!='PASSED':raise ValueError('spin source/wiring not valid')
 expected=recover_spec(parent,experiment=spec['experiment'],spec_path=spec['motion_training']['spec_path']);expected['authorization']=spec['authorization']
 if spec!=expected:raise ValueError('spin recovery undeclared behavior/budget change')
 claims=parent.parents[1]/'.motion_spin_recovery_claims';claim=claims/(sha(parent)+'.json')
 if claim.exists() and json.loads(claim.read_text())['destination']!=spec['experiment']:raise ValueError('spin original attempts already claimed')
 return {'status':'VERIFIED','historical_smoke_updates':5,'remaining_formal_updates':1995,'historical_wall_s':c['historical_wall_s'],'capability_accepted':False}
