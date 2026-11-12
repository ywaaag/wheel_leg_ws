"""Motion block adapter under the existing public trainer/monitor, no autonomous chain."""
import json,time,sys,traceback
from collections import deque
from pathlib import Path
import torch
from .data import sha
from .artifacts import checkpoint as atomic_checkpoint,secondary,storage
from .training import observations,evaluate_cases
from .evaluator import environment
from .model import load
from .ppo import assemble
from .motion_training import choose
from native_stand.command_v6.completion import write

def motion_observations(env,raw,retention=None):
 """Keep optional episode metadata aligned at initial/reset/post-eval entry."""
 from .motion_source_retention import add_metadata
 obs=add_metadata(observations(env,raw),env.target_xy,retention,
                  variants=getattr(getattr(env,'motion_timing',None),'variant',None))
 if getattr(env,'motion_config',{}).get('course_version') in ('motion31_large_combo4096_v1','motion31_parallel_bidirectional_course_v1','motion31_standing_timing_course_v1','motion31_standing_fixed_timing_v2','motion31_standing_sensor_transfer_v3','motion31_standing_antiosc_armature_v4','motion31_posture_reacquisition_v1','motion31_alternating_standing_robustness_v1','motion31_decoupled_height_course_v1','motion31_fixed_height_motion_v2','motion31_persistent_frontiers_v1'):
  obs['motion_diagnostic_cohort']=env.command_index[:,None].float()
 if getattr(env,'mcu_execution',None):obs['motion_diagnostic_cohort']=env.command_index[:,None].float()
 if getattr(env,'critic_privileged_version',None) in ('motion_antiosc_privileged140_v1','motion_height_privileged144_v1','motion_height_robust_privileged184_v1'):
  obs['motion_inertia_cohort']=env.motion_timing.armature_levels[:,None].float()
 if hasattr(env,'motion_dr'):
  obs['motion_command_dr_cohort']=(env.command_index*3+env.motion_dr.group_ids)[:,None].float()
 return obs

def checkpoint_plan(recipe,phase,update,limit,*,offset=0):
 """Periodic saves do not imply evaluation or an environment reset."""
 evaluate=update in recipe['milestones'] or update==limit
 interval=recipe.get('checkpoint_every_updates')
 periodic=phase=='formal' and interval is not None and (update+offset)%interval==0
 return evaluate or periodic,evaluate

def save_unevaluated(output,spec,ppo,update,progress,candidates,select):
 candidate=save(output,spec,ppo,update)
 candidate.update(per_env=[],complete_matrix=False,evaluations=[],evaluated=False)
 candidates.append(candidate);progress['latest_checkpoint']=candidate;progress['checkpoint_roundtrip_verified']=True
 write(output/'candidate_registry.json',candidates);write(output/'candidate_selection.json',select(candidates));write(output/'progress.json',progress)
 return candidate

def _record_optimizer_completion(progress,update,unit,mask_summary):
 """Count real optimizer completion before fallible diagnostics/serialization."""
 progress.update(actual_updates=update,samples=update*unit,
  effective_actor_samples=progress['effective_actor_samples']+mask_summary['effective_actor_update_samples'])

def effective_course_spec(spec,state):
 """Refresh the active course using its declared stateful/stateless contract."""
 from .motion_course import implementation
 course=implementation(spec['student_training']['course_version'])
 if spec.get('motion_parallel') or spec.get('motion_robustness') or spec.get('motion_height_dr'):
  return course.effective_spec(spec,state['stage'],state)
 return course.effective_spec(spec,state['stage'])

def advance_course_boundary(spec,ppo,candidate):
 """Bind the owner and saved state to the matrix of the resulting revision."""
 from .motion_course import implementation
 course=implementation(spec['student_training']['course_version'])
 state,decision=course.advance(ppo.motion_course_state,candidate)
 if spec.get('motion_decoupled'):
  # effective_spec deliberately copies its inputs. Keep the returned state,
  # including regenerated assessment keys, before writing progress/checkpoints.
  spec=effective_course_spec(spec,state)
  state=spec['motion_decoupled']['state']
 if spec.get('motion_height_dr'):
  spec=effective_course_spec(spec,state)
  state=spec['motion_height_dr']['state']
 ppo.motion_course_state=state
 return spec,state,decision

_SOURCE_METADATA={}
_SOURCE_IDENTITY={}

def _source_metadata(spec):
 """来源模型只读取一次；后续保存复用已读出的元数据。"""
 key=(str(spec['source_checkpoint']),spec['source_sha256'])
 if key not in _SOURCE_METADATA:
  source=torch.load(spec['source_checkpoint'],map_location='cpu',weights_only=False)
  _SOURCE_METADATA[key]={'dataset_sha256':source.get('dataset_sha256'),'manifest_sha256':source.get('manifest_sha256')}
  del source
 return _SOURCE_METADATA[key]


def _source_code_sha256(spec):
 """源码身份按run记录一次，不随每次保存重扫全树。"""
 key=str(spec.get('execution_source_root'))
 if key not in _SOURCE_IDENTITY:
  from native_stand.command_v6.tools.execution_identity import identity
  _SOURCE_IDENTITY[key]=identity(spec)['location_independent_modules_sha256']
 return _SOURCE_IDENTITY[key]


def _state_digest(state_dict):
 """序列化前后比较用的稳定摘要；不重建网络。"""
 import hashlib
 digest=hashlib.sha256()
 for name in sorted(state_dict):
  value=state_dict[name].detach().cpu().contiguous()
  digest.update(name.encode());digest.update(str(tuple(value.shape)).encode())
  digest.update(value.numpy().tobytes())
 return digest.hexdigest()


def save(output,spec,ppo,update):
 from .resume import rng_state
 cfg=spec['motion_training'];base=spec['motion_course']['source_recipe_updates'] if spec.get('motion_course') else (cfg.get('successor') or cfg['resume']).get('parent_recipe_updates',0)
 source=_source_metadata(spec)
 data={'interface_version':'multiskill_mcu_v2','role':'MotionZeroStudent','actor':ppo.actor.state_dict(),'critic':ppo.critic.state_dict(),'optimizer':ppo.optimizer.state_dict(),
  'recipe_version':spec['student_training']['version'],'training_recipe':spec['student_training'],'source_sha256':spec['source_sha256'],
  'dataset_sha256':source['dataset_sha256'],'manifest_sha256':source['manifest_sha256'],'asset_sha256':spec['asset_sha256'],'spec_sha256':sha(output/'source_snapshot/spec.json'),
  'source_code_sha256':_source_code_sha256(spec),
  'fixed_std':True,'BC_coefficient':0.,'update':update,'recipe_updates':base+update,'critic_privileged_version':spec['student_training']['critic_privileged_version'],'training_rng':rng_state(),'motion_block':cfg}
 if hasattr(ppo,'reward_refinement_state'):data['reward_refinement_state']=ppo.reward_refinement_state
 if hasattr(ppo,'motion_dr_adapter'):
  ppo.motion_course_state['DR_episode_counters']=ppo.motion_dr_adapter.episodes.detach().cpu().tolist()
 if hasattr(ppo,'motion_course_state'):data['motion_course_state']=ppo.motion_course_state
 if hasattr(ppo,'symmetry_mix_state'):data['symmetry_mix_state']=ppo.symmetry_mix_state
 if hasattr(ppo,'source_retention'):data['source_policy_retention']=ppo.source_retention.cfg
 if spec.get('network_architecture'):data['network_architecture']=spec['network_architecture']
 path=output/f'model_update_{update}.pt';expected=_state_digest(data['actor']);atomic_checkpoint(path,data)
 # 只核对写盘后的序列化往返，不重建网络、不重读来源模型。
 written=torch.load(path,map_location='cpu',weights_only=False)
 if _state_digest(written['actor'])!=expected or written['update']!=update:
  raise ValueError('Motion checkpoint序列化往返不一致')
 del written
 return {'checkpoint':str(path.resolve()),'sha256':sha(path),'update':update,'stage':cfg['stage']}

def execute(args,spec):
 from native_stand.command_v6.ledger import Ledger
 from native_stand.command_v6.train import record_run_configuration,install_stop_handlers,send_handshake
 envs=spec['ppo']['envs'];steps=spec['ppo']['steps_per_env'];unit=envs*steps
 out=args.output_dir;phase=args.phase;book=json.loads(args.spec.with_name('budget.json').read_text());attempt=book['attempts'][args.attempt_id]
 if attempt['status']!='RUNNING' or attempt['output_dir']!=str(out.resolve()):raise ValueError('Motion exact reservation required')
 from native_stand.command_v6.domain.execution_limits import ExecutionLimits
 started=time.monotonic();deadline=ExecutionLimits.resolve(spec,phase).deadline(attempt['started_unix_s']);env=app=None;failure=None;status='FAILED';sent=False;ppo=None;cleanup=[];tail=deque(maxlen=200)
 tb_writer=None
 progress={'status':'STARTING','phase':phase,'attempt_id':args.attempt_id,'output_dir':str(out.resolve()),'actual_updates':0,'samples':0,'effective_actor_samples':0,'actor_parameters_changed':False,'checkpoint_roundtrip_verified':False,'evaluations':[]}
 def stopped():
  from native_stand.command_v6 import train
  return train.STOP or time.time()>deadline-90
 def guard():
  limit=storage(spec);size=sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file() and not p.is_symlink())
  recovery=spec.get('motion_training',{}).get('wiring_reuse',{})
  if spec.get('motion_timing',{}).get('fixed_parent'):
   size+=sum(p.stat().st_size for p in Path(spec['motion_timing']['fixed_parent']['parent_run']).rglob('*') if p.is_file() and not p.is_symlink())
  if recovery.get('version')=='motion_timing_posteval_wiring_resume_v1':
   size+=sum(p.stat().st_size for p in Path(recovery['parent_run']).rglob('*') if p.is_file() and not p.is_symlink())
  if spec.get('motion_height_dr') and spec['motion_height_dr'].get('count_original_storage',True):
   original=json.loads(Path(spec['motion_height_dr']['original_spec']).read_text())
   old_run=Path(spec['motion_height_dr']['original_spec']).parents[3]/'runs'/original['experiment']
   size+=sum(p.stat().st_size for p in old_run.rglob('*') if p.is_file() and not p.is_symlink())
  if size>limit['chain_max_bytes']-limit['terminal_reserve_bytes']:raise TimeoutError('Motion storage reserve reached')
  if spec.get('motion_decoupled') or spec.get('mcu_posture') or spec.get('motion_execution_robustness'):
   import shutil
   if shutil.disk_usage(out).free<limit['terminal_reserve_bytes']:raise TimeoutError('Motion filesystem terminal reserve reached')
 try:
  out.mkdir(parents=True,exist_ok=False);record_run_configuration(out,args.spec)
  if phase=='formal' and (spec.get('motion_height_dr') or spec.get('mcu_posture') or spec.get('motion_spin_quality') or spec.get('motion_large_continuation',{}).get('execution_policy')=='motion31_unlimited_wall_authorized_retry_v1'):
   from torch.utils.tensorboard import SummaryWriter
   tb_writer=SummaryWriter(str(out/'tensorboard'),flush_secs=30)
  from .motion_reward_audit import audit as reward_audit
  from . import motion_tracking_reward as reward_source
  snapshot=reward_audit(spec['reward']);snapshot.update(spec_sha256=sha(args.spec),reward_source_sha256=sha(Path(reward_source.__file__)),source_sha256=spec['source_sha256'])
  if spec['student_training'].get('source_policy_retention') is not None:
   snapshot['optimizer_auxiliary_losses']={'source_policy_retention':spec['student_training']['source_policy_retention']}
  if spec.get('motion_timing',{}).get('version')=='motion31_standing_sensor_transfer_v3':
   from .motion_sensor_transfer import reward_snapshot
   snapshot['runtime_reward_extension']=reward_snapshot()
  if spec.get('motion_timing',{}).get('version')=='motion31_standing_antiosc_armature_v4':
   from .motion_antioscillation import reward_snapshot
   snapshot['runtime_reward_extension']=reward_snapshot(spec['motion_timing']['configuration'])
  if spec.get('mcu_execution'):
   from .motion_mcu_course import execution_reward_snapshot
   snapshot['mcu_execution']=execution_reward_snapshot()
  if spec.get('mcu_posture') or spec.get('motion_execution_robustness') or spec.get('motion_standing_delay'):
   from .motion_posture_preparation import runtime_reward_snapshot
   snapshot['posture_reward']=runtime_reward_snapshot()
  if spec.get('motion_height_dr'):
   from .motion_height import reward_snapshot as height_snapshot
   snapshot['fixed_height_reward']=height_snapshot()
   from .motion_height_pd import learning_signal_audit
   if spec.get('motion_highspeed_standing'):
    from .motion_highspeed_standing import learning_signal_audit
   if spec.get('motion_standing_delay'):
    from .motion_standing_delay import learning_signal_audit
   snapshot['learning_signal_audit']=learning_signal_audit(spec)
  if spec.get('motion_decoupled'):
   from .motion_height import reward_snapshot as height_snapshot
   if spec['motion_decoupled'].get('fixed_episode_height'):
    from .motion_fixed_height import reward_snapshot as height_snapshot
   snapshot['height_task']=height_snapshot()
   if spec['student_training'].get('course_version')=='motion31_persistent_frontiers_v1':
    from .motion_persistent import height_reward_snapshot
    snapshot['height_frontier_reward']=height_reward_snapshot()
  with (out/'reward_snapshot.json').open('x') as stream:json.dump(snapshot,stream,indent=2,allow_nan=False)
  from native_stand.command_v6.tools.health_check import own_process_start
  import os
  write(out/'manifest.json',{'attempt_id':args.attempt_id,'phase':phase,'pid':os.getpid(),'trainer_pid':os.getpid(),'process_started_unix_s':own_process_start(os.getpid()),'source_sha256':spec['source_sha256'],'spec_sha256':sha(args.spec),'asset_sha256':spec['asset_sha256'],'hapi_session':args.hapi_session,'interface_version':'multiskill_mcu_v2','recipe_version':spec['student_training']['version'],'reward_snapshot_sha256':sha(out/'reward_snapshot.json')})
  write(out/'progress.json',progress);sys.path.insert(0,spec['asset_module_root'])
  if phase=='source_eval' and spec.get('motion_standing_delay'):
   from .motion_standing_delay import startup_evidence,choose as source_choose
   records,candidates=startup_evidence(spec)
   progress.update(status='COMPLETED',engineering_valid=True,evaluations=[],
    source_reuse={'new_physics_steps':0,'optimizer_updates':0,
     'scope':'complete saved3896 state verified; ability under new20/5/20ms plant UNKNOWN; no ability prerequisite'})
   write(out/'source_reuse.json',progress['source_reuse']);write(out/'candidate_registry.json',candidates)
   write(out/'candidate_selection.json',source_choose(candidates,{}));status='COMPLETED'
   return 0
  if phase=='source_eval' and spec.get('motion_highspeed_standing'):
   from .motion_highspeed_standing import startup_evidence,choose as source_choose
   records,candidates=startup_evidence(spec)
   progress.update(status='COMPLETED',engineering_valid=True,evaluations=records,
    source_reuse={'new_physics_steps':0,'optimizer_updates':0,
     'scope':'original9995 Native evidence; new primary/standing gates UNKNOWN; no ability prerequisite'})
   write(out/'source_reuse.json',progress['source_reuse']);write(out/'candidate_registry.json',candidates)
   write(out/'candidate_selection.json',source_choose(candidates,{}));status='COMPLETED'
   return 0
  height_recovery=spec.get('motion_decoupled',{}).get('engineering_recovery',{})
  if phase=='source_eval' and height_recovery.get('version')=='motion_height_preppo_storage_recovery_v1':
   from .motion_decoupled import reused_source
   records,candidates=reused_source(spec)
   progress.update(status='COMPLETED',engineering_valid=True,evaluations=records,
    source_assessment_scope={'engineering_only':True,'covered_commands':[[0.,0.,.34]],'other_commands':'UNKNOWN; no full source capability claim'},
    source_reuse={'source_run':height_recovery['source_evidence_run'],'new_physics_steps':0,'new_optimizer_updates':0})
   write(out/'source_reuse.json',height_recovery)
   write(out/'candidate_registry.json',candidates);write(out/'candidate_selection.json',choose(candidates))
   status='COMPLETED'
   return 0
  if phase=='source_eval' and (spec.get('mcu_measured') or spec.get('mcu_posture')):
   if spec.get('mcu_posture'):
    from .motion_posture_preparation import reused_source
    reuse_section=spec['mcu_posture']
   else:
    from .motion_mcu_measured_io import reused_source
    reuse_section=spec['mcu_measured']
   records,candidates=reused_source(spec)
   progress.update(status='COMPLETED',engineering_valid=True,evaluations=records,
    source_reuse={'assessment':reuse_section['assessment'],'new_physics_steps':0,'optimizer_updates':0,
     'scope':'same Actor31 and execution/scoring; Critic312 requires new wiring smoke'})
   write(out/'source_reuse.json',progress['source_reuse']);write(out/'candidate_registry.json',candidates)
   write(out/'candidate_selection.json',choose(candidates));status='COMPLETED'
   return 0
  if phase=='source_eval' and spec.get('motion_height_dr',{}).get('startup_scope')=='reuse_qualified034_only_new_tasks_UNKNOWN':
   from .motion_height_pd import startup_evidence,choose as choose_partial_source
   records,missing=startup_evidence(spec)
   scope={'status':'QUALIFIED034_REUSED_NEW_TASKS_UNKNOWN','reused_conditions':len(records),
    'new_task_case_ids_not_evaluated':[r['case_id'] for r in missing],'new_physics_steps':0,
    'reason':'known source capability and prior Native wiring readback suffice for learning; no new-task ability prerequisite'}
   candidate={'checkpoint':spec['source_checkpoint'],'sha256':spec['source_sha256'],'update':0,
    'stage':spec['motion_training']['stage'],'per_env':[e for r in records for e in r['per_env']],
    'evaluations':records,'complete_matrix':False,'source_assessment_scope':scope}
   progress.update(status='COMPLETED',engineering_valid=True,evaluations=records,source_assessment_scope=scope)
   write(out/'source_reuse.json',scope);write(out/'candidate_registry.json',[candidate])
   write(out/'candidate_selection.json',choose_partial_source([candidate],{}));status='COMPLETED'
   return 0
  from isaaclab.app import AppLauncher
  app=AppLauncher({'headless':True,'device':'cuda:0','multi_gpu':False}).app;install_stop_handlers();torch.manual_seed(spec['seed'])
  actor,source=load(spec['source_checkpoint'],'MotionZeroStudent');actor=actor.to('cuda:0');source_actor_weights={k:v.clone() for k,v in actor.state_dict().items()};cases=spec['student_evaluation']['cases']
  if phase=='formal' and spec.get('motion_decoupled',{}).get('formal_engineering_probe'):
   from .motion_height_runtime import engineering_probe
   engineering_probe(spec,actor,out,stopped)
  course=bool(spec.get('motion_course'));course_state={'stage':0,'stage_started_update':0,'last_evaluation_update':-1,'consecutive_passes':0,'accepted_by_stage':{}}
  if spec.get('motion_height_dr'):
   course_state=__import__('copy').deepcopy(spec['motion_height_dr']['state'])
  if spec.get('mcu_measured'):course_state['stage']=spec['motion_training']['stage']
  if spec.get('motion_parallel'):
   from .motion_parallel_course import initial_state
   course_state=initial_state()
  if spec.get('motion_robustness'):
   import copy
   course_state=copy.deepcopy(spec['motion_robustness']['state'])
  minimal_startup=bool(spec.get('motion_startup'))
  if spec.get('motion_timing') and phase=='source_eval':minimal_startup=False
  if phase=='source_eval':
   if spec.get('motion_robustness'):
    if spec.get('motion_decoupled'):
     from .motion_height_runtime import engineering_probe
    else:
     from .motion_robustness import engineering_probe
    engineering_probe(spec,actor,out,stopped)
    cases=[r for r in cases if r['command']==[0.,0.,.34] and r['_motion_dr']['group']==0]
   if spec.get('motion_timing_recovery'):
    import copy
    from .motion_timing_experiment import recovery_evidence
    certificate=recovery_evidence(spec['motion_timing_recovery']['parent_spec'])
    if certificate!=spec['motion_timing_recovery']:raise ValueError('timing source recovery evidence changed')
    folder=Path(certificate['parent_run'])/'source_eval'
    records=json.loads((folder/'terminal_report.json').read_text())['evaluations']
    records=copy.deepcopy(records)
    for rec in records:
     rec['path']=str(folder/rec['path']);rec['reused']=True
    write(out/'source_reuse.json',certificate)
   elif spec.get('motion_height_dr'):
    from .motion_height_pd import startup_evidence
    records,missing=startup_evidence(spec)
    write(out/'source_reuse.json',{'version':spec['motion_height_dr']['version'],
     'reused':[r['reuse_reference'] for r in records],'new_conditions':len(missing),
     'scope':'immutable4500 raw034 evidence; new PD-age/height physically evaluated'})
    records+=evaluate_cases(spec,out,actor,missing,offset=len(records),stop=stopped,checkpoint=spec['source_checkpoint'])
    expected=[r['case_id'] for r in cases]
    if [r['case']['case_id'] for r in records]!=expected:raise ValueError('parallel source matrix ordering differs')
   elif spec.get('motion_execution_robustness'):
    records=evaluate_cases(spec,out,actor,cases,stop=stopped)
   elif minimal_startup:
    from .motion_startup import resolve,merge
    reused_records,missing_cases=resolve(spec)
    new_records=evaluate_cases(spec,out,actor,missing_cases,stop=stopped) if missing_cases else []
    records=merge(spec,reused_records,new_records)
    write(out/'source_reuse.json',{'version':spec['motion_startup']['version'],'reused':[r['reuse_reference'] for r in reused_records.values()],'new_evaluations':[{'path':r['path'],'sha256':r['sha256']} for r in new_records],'no_new_simulation_for_reused_commands':True})
   else:records=evaluate_cases(spec,out,actor,cases,stop=stopped)
   if spec.get('motion_spin_quality') and not spec.get('motion_timing'):
    from .motion_spin_calibration import verify_records
    progress['spin_measurement_readback']=verify_records(records,out,allow_incomplete_new=True) if spec.get('motion_parallel') else verify_records(records,out)
    if spec.get('motion_symmetry'):
     from .motion_symmetry import verify_source_features
     progress['symmetry_readback']=verify_source_features(records,out,allow_incomplete_new=True) if spec.get('motion_parallel') else verify_source_features(records,out)
    if spec.get('motion_differential'):
     from .motion_differential import verify_records
     progress['differential_readback']=verify_records(records,out,spec)
   progress.update(status='COMPLETED',engineering_valid=True,evaluations=records)
   candidate={'checkpoint':spec['source_checkpoint'],'sha256':spec['source_sha256'],'update':0,'per_env':[e for r in records for e in r['per_env']],'complete_matrix':True,'stage':spec['motion_training']['stage'],'evaluations':records}
   if spec.get('motion_robustness'):
    candidate['complete_matrix']=False
    progress['source_assessment_scope']={'engineering_only':True,'covered_commands':[[0.,0.,.34]],'other_commands':'UNKNOWN; no full source capability claim'}
   write(out/'candidate_registry.json',[candidate]);write(out/'candidate_selection.json',choose([candidate]));status='COMPLETED'
  else:
   source_retention=spec['student_training'].get('source_policy_retention')
   env=environment(spec,'MotionZeroStudent',envs=envs);env.curriculum_training=True
   if spec.get('motion_timing') and phase=='smoke':env.motion_timing.configure(0 if spec['motion_timing']['version']=='motion31_standing_antiosc_armature_v4' else 3)
   elif spec.get('motion_timing',{}).get('version')=='motion31_standing_antiosc_armature_v4':env.motion_timing.configure(None)
   env.select_initial_family('motion_native_normal','train');obs,_=env.reset();obs=motion_observations(env,obs,source_retention)
   if spec['student_training'].get('course_version')=='motion31_persistent_frontiers_v1':
    from .motion_persistent import frontiers
    stage=env.stage_spec
    observed=torch.bincount(env.command_index,minlength=len(stage['commands']))
    expected=torch.tensor(stage['slot_counts'],device=env.device)
    if not torch.equal(observed,expected):raise RuntimeError('persistent runtime cohort quota differs')
    if not torch.equal(env.motion_dr.level_ids,torch.tensor(stage['DR_level_slots'],device=env.device)):raise RuntimeError('persistent DR level wiring differs')
    targets=torch.tensor(stage['commands'],device=env.device)[env.command_index]
    if not torch.allclose(env.target_xy,targets):raise RuntimeError('persistent target command wiring differs')
    if not torch.allclose(env.height_adapter.target,env.height_adapter.initial):raise RuntimeError('persistent fixed height differs')
    write(out/'persistent_curriculum_readback.json',{'status':'PASSED','frontiers':frontiers(course_state),
     'cohort_axes':stage['cohort_axes'],'cohort_purposes':stage['cohort_purposes'],'slot_counts':observed.cpu().tolist(),
     'DR_levels':sorted(set(stage['DR_level_slots'])),'height_frontier_reward':spec['student_training']['height_frontier_reward'],
     'actual31_184':list(obs['student_actor'].shape)+list(obs['student_critic'].shape),'PPO_updates':0,'scope':'initial runtime quotas/commands/heights/DR; not ability proof'})
   if spec.get('performance',{}).get('motion_hotpath') or spec.get('performance',{}).get('contact_history_disabled'):
    from .motion_hotpath import readback as hotpath_readback
    write(out/'motion_hotpath_readback.json',hotpath_readback(env))
   if hasattr(env,'motion_timing'):
    write(out/'timing_runtime_readback.json',env.motion_timing.summary())
   if minimal_startup and phase=='smoke':
    from native_stand.command_v6.runtime.control_readback import verify,verify_ground
    from native_stand.command_v6.runtime.configure import verify_asset_import
    write(out/'wiring_runtime_readback.json',{'asset':verify_asset_import(spec),'control':verify(env),'ground':verify_ground(env,spec),'scope':'actual smoke environment before online sampling'})
   wiring_command_valid=torch.ones((),device=env.device,dtype=torch.bool);wiring_command_samples=0
   if spec.get('motion_highspeed_standing'):
    from .motion_highspeed_standing import runtime_readback
    write(out/'highspeed2048_runtime_readback.json',runtime_readback(env,obs))
   if spec.get('motion_standing_delay'):
    from .motion_standing_delay_runtime import runtime_readback
    write(out/'standing_delay_runtime_readback.json',runtime_readback(env,obs))
   ppo=assemble(spec['source_checkpoint'],obs,steps_per_env=steps,recipe=spec['student_training']);offset=0
   if hasattr(env,'motion_dr'):ppo.motion_dr_adapter=env.motion_dr
   if course:
    from .motion_course_requests import restore as restore_course
    progress['course_restore']=restore_course(ppo,source,spec,phase)
    env.motion_cohort_mix.copy_(torch.tensor([r['mix'] for r in ppo.reward_refinement_state],device=env.device))
    ppo.motion_course_state=course_state
   if spec.get('motion_symmetry'):
    from .motion_parallel_course import extend_mixture as extend_parallel_mixture
    ppo.symmetry_mix_state=extend_parallel_mixture(source.get('symmetry_mix_state',[]),env.stage_spec['commands']) if spec.get('motion_parallel') else [{'target':v,'mix':0.,'confirmations':0,'regressions':0} for v in env.stage_spec['commands']]
    if spec.get('mcu_execution'):
     from .motion_mcu_course import extend_mixture
     ppo.symmetry_mix_state=extend_mixture(source.get('symmetry_mix_state',[]),env.stage_spec['commands'])
    if spec.get('motion_decoupled'):
     from .motion_decoupled import extend_mixture
     ppo.symmetry_mix_state=extend_mixture(source.get('symmetry_mix_state',[]),env.stage_spec['cohort_keys'])
    elif spec.get('motion_robustness'):
     from .motion_robustness import extend_mixture
     ppo.symmetry_mix_state=extend_mixture(source.get('symmetry_mix_state',[]),env.stage_spec['commands'])
    env.symmetry_mix=torch.zeros(env.num_envs,device=env.device)
   refinement=spec['reward'].get('refinement')
   if refinement:
    from .motion_refinement import initial as initial_refinement,advance
    if not course:ppo.reward_refinement_state=initial_refinement(env.stage_spec['commands'])
   if spec['motion_training'].get('successor'):
    from .motion_successor import restore
    progress['successor_restore']=restore(ppo,source,spec,phase)
    env.motion_cohort_mix.copy_(torch.tensor([r['mix'] for r in ppo.reward_refinement_state],device=env.device))
   if spec['motion_training']['resume'] and not spec.get('motion_large_continuation') and not spec.get('motion_spin_quality'):
    from .resume import restore_rng
    ppo.actor.load_state_dict(source['actor']);ppo.critic.load_state_dict(source['critic']);ppo.optimizer.load_state_dict(source['optimizer']);restore_rng(source['training_rng']);offset=source['recipe_updates']
   timing_resume=spec.get('motion_training',{}).get('wiring_reuse',{})
   if timing_resume.get('version')=='motion_mcu_measured_posteval_resume_v1':
    import copy
    boundary=timing_resume['boundary'];course_state=copy.deepcopy(boundary['state'])
    ppo.motion_course_state=course_state;offset=timing_resume['checkpoint_update']
    ppo.symmetry_mix_state=copy.deepcopy(boundary['symmetry_mix'])
    ppo.reward_refinement_state=copy.deepcopy(boundary['reward_mix'])
    env.motion_cohort_mix.copy_(torch.tensor([r['mix'] for r in ppo.reward_refinement_state],device=env.device))
    progress['posteval_recovery']={'checkpoint_update':offset,'historical_online_updates':timing_resume['historical_online_updates'],
     'warmup_repeated':False,'matrix_repeated':False,'physical_state_restored':False,'decision':boundary['decision']}
    write(out/'course_progress.json',{'state':course_state,'decision':boundary['decision'],'reconstructed_from':timing_resume['inputs']})
   if timing_resume.get('version')=='motion_fixed_height_persistent_resume_v1':
    offset=timing_resume['formal_update_offset']
    progress['persistent_course_resume']={'checkpoint_update':timing_resume['checkpoint_update'],'formal_update_offset':offset,
     'current_block_used_updates':timing_resume['current_block_used_updates'],'remaining_formal_updates':timing_resume['remaining_formal_updates'],
     'warmup_repeated':False,'physical_state_restored':False}
   if timing_resume.get('version')=='motion_fixed_height_assessment_keys_resume_v1':
    offset=timing_resume['checkpoint_update']
    progress['course_key_repair']={'checkpoint_update':offset,'matrix_update':timing_resume['matrix_update'],
     'historical_online_updates':timing_resume['historical_online_updates'],'new_authorized_online_updates':10000,
     'warmup_repeated':False,'physical_state_restored':False,'replay_decision':timing_resume['replay_decision']}
   if timing_resume.get('version') in ('motion_robustness_material_resume_v1','motion_robustness_diagnostic_resume_v1','motion_robustness_eval_material_resume_v1','motion_robustness_group_baseline_resume_v1'):
    offset=timing_resume.get('formal_update_offset',timing_resume['checkpoint_update'])
    progress['material_recovery']={'checkpoint_update':timing_resume['checkpoint_update'],'formal_update_offset':offset,'historical_online_updates':timing_resume['historical_online_updates'],
     'warmup_repeated':False,'physical_state_restored':False,'source_capability':'unassessed restored checkpoint; ancestral source evidence only'}
   if timing_resume.get('version')=='motion_timing_posteval_wiring_resume_v1':
    from .motion_timing_experiment import restored_posteval_state
    course_state,ppo.symmetry_mix_state,ppo.reward_refinement_state,pending_decision=restored_posteval_state(source,timing_resume,spec['reward'])
    ppo.motion_course_state=course_state;offset=source['update']
    env.motion_cohort_mix.copy_(torch.tensor([r['mix'] for r in ppo.reward_refinement_state],device=env.device))
    progress['posteval_recovery']={'checkpoint_update':offset,'pending_decision':pending_decision,
     'historical_online_updates':timing_resume['historical_online_updates'],'warmup_repeated':False,'physical_state_restored':False}
    write(out/'course_progress.json',{'state':course_state,'decision':pending_decision,'reconstructed_from':timing_resume['inputs']})
   progress['optimizer_initial']={'source_critic_loaded':bool(offset or spec['motion_training'].get('successor') or course),'source_Adam_loaded':bool(offset or spec['motion_training'].get('successor') or course),'source_RNG_loaded':bool(offset or spec['motion_training'].get('successor') or course),'state_count':len(ppo.optimizer.state),'BC_coefficient':0.}
   if source_retention is not None:progress['optimizer_initial']['source_policy_retention']=source_retention
   reused=spec['motion_training'].get('wiring_reuse') or spec['motion_training'].get('partial_smoke_recovery')
   source_folder=Path(reused.get('source_evidence_run',reused['parent_run']))/'source_eval' if reused else out.parent/'source_eval'
   fixed=ppo.actor.log_std.detach().clone();candidates=json.loads((source_folder/'candidate_registry.json').read_text());index=0;limit=spec['budget'][phase+'_updates']
   if timing_resume.get('version') in ('motion_robustness_material_resume_v1','motion_robustness_diagnostic_resume_v1','motion_robustness_eval_material_resume_v1','motion_robustness_group_baseline_resume_v1','motion_fixed_height_critic_restart_v1','motion_fixed_height_assessment_keys_resume_v1','motion_fixed_height_persistent_resume_v1','motion_mcu_measured_posteval_resume_v1'):
    progress['ancestral_source_evidence']={'registry':str(source_folder/'candidate_registry.json'),'sha256':sha(source_folder/'candidate_registry.json')}
    candidates=[{'checkpoint':spec['source_checkpoint'],'sha256':spec['source_sha256'],'update':0,'stage':spec['motion_training']['stage'],
      'per_env':[],'evaluations':[],'complete_matrix':False,'evaluated':False}]
   from .performance import RolloutCounts,settings
   from .motion_statistics import MotionStatistics
   from .motion_tracking_reward import reward_fields,QUALITY_VERSIONS
   from .motion_tracking_reward import effective_reward_fields
   names=effective_reward_fields(spec)
   quality_enabled=spec['reward']['version'] in QUALITY_VERSIONS
   perf=settings(spec)
   tolerant_guard=None if spec.get('motion_robustness') or spec.get('mcu_execution') else spec['motion_training'].get('early_stop_policy');guard_state={}
   retention_baseline=candidates[0]['evaluations']
   if spec.get('motion_symmetry'):
    import copy
    retention_baseline=copy.deepcopy(retention_baseline)
    for rec in retention_baseline:
     for e in rec['per_env']:
      e['accepted']=all(v for k,v in e['checks'].items() if k not in ('swing_hold','swing_full','spin_center','spin_common'))
    progress['retention_baseline_scope']='derived source original capability gates only; new symmetry failures do not erase old tracking protection; raw source acceptance unchanged'
   if course:
    from .motion_course import implementation
    choose_course=implementation(spec['student_training']['course_version']).choose
    selection_fn=(lambda values:choose_course(values,course_state)) if spec.get('motion_robustness') else (lambda values:choose_course(values,course_state['accepted_by_stage']))
   else:selection_fn=choose
   if timing_resume.get('version') in ('motion_robustness_eval_material_resume_v1','motion_robustness_group_baseline_resume_v1') and phase=='formal':
    # The parent saved/charged the boundary checkpoint, then failed during its
    # assessment. Complete that assessment before allowing any further PPO.
    env.close();env=None
    records=evaluate_cases(spec,out,ppo.actor,cases,stop=stopped,checkpoint=spec['source_checkpoint']);index=len(records)
    from .motion_robustness import complete_pending_assessment,compact_historical_records
    spec,course_state,candidate,decision=complete_pending_assessment(spec,ppo,records)
    cases=spec['student_evaluation']['cases'];candidates=[candidate]
    progress.update(evaluations=records,latest_checkpoint={k:candidate[k] for k in ('checkpoint','sha256','update','stage','evaluated')},
     pending_assessment={'status':'COMPLETED','formal_update_including_parent':offset,'new_optimizer_updates':0,'decision':decision})
    write(out/'candidate_registry.json',candidates);write(out/'candidate_selection.json',selection_fn(candidates))
    write(out/'course_progress.json',{'state':course_state,'decision':decision,'checkpoint':candidate['checkpoint'],'checkpoint_sha256':candidate['sha256']})
    write(out/'progress.json',progress);compact_historical_records(records)
    env=environment(spec,'MotionZeroStudent',envs=envs);env.curriculum_training=True;ppo.motion_dr_adapter=env.motion_dr
    if refinement:env.motion_cohort_mix.copy_(torch.tensor([r['mix'] for r in ppo.reward_refinement_state],device=env.device))
    env.select_initial_family('motion_native_normal','train');obs,_=env.reset();obs=motion_observations(env,obs,source_retention)
   if spec['student_training'].get('course_version')=='motion31_persistent_frontiers_v1':
    from .motion_persistent import frontiers
    stage=env.stage_spec
    observed=torch.bincount(env.command_index,minlength=len(stage['commands']))
    expected=torch.tensor(stage['slot_counts'],device=env.device)
    if not torch.equal(observed,expected):raise RuntimeError('persistent runtime cohort quota differs')
    if not torch.equal(env.motion_dr.level_ids,torch.tensor(stage['DR_level_slots'],device=env.device)):raise RuntimeError('persistent DR level wiring differs')
    targets=torch.tensor(stage['commands'],device=env.device)[env.command_index]
    if not torch.allclose(env.target_xy,targets):raise RuntimeError('persistent target command wiring differs')
    if not torch.allclose(env.height_adapter.target,env.height_adapter.initial):raise RuntimeError('persistent fixed height differs')
    write(out/'persistent_curriculum_readback.json',{'status':'PASSED','frontiers':frontiers(course_state),
     'cohort_axes':stage['cohort_axes'],'cohort_purposes':stage['cohort_purposes'],'slot_counts':observed.cpu().tolist(),
     'DR_levels':sorted(set(stage['DR_level_slots'])),'height_frontier_reward':spec['student_training']['height_frontier_reward'],
     'actual31_184':list(obs['student_actor'].shape)+list(obs['student_critic'].shape),'PPO_updates':0,'scope':'initial runtime quotas/commands/heights/DR; not ability proof'})
   for update in range(1,limit+1):
    if stopped():status='STOPPED_BOUNDED';break
    if spec.get('mcu_execution'):env.motion_timing.stats_reset()
    if spec.get('motion_execution_robustness'):env.robust_control_counts.zero_()
    guard();start_roll=time.perf_counter();counts=RolloutCounts(steps,envs,env.device);fields=torch.zeros(len(names),device=env.device);dt_fields=torch.zeros_like(fields);slots=torch.zeros(len(env.stage_spec['commands']),device=env.device,dtype=torch.long);hold=torch.zeros_like(slots);episodes=torch.zeros_like(slots);timeouts=torch.zeros((),device=env.device,dtype=torch.long)
    event_sum=torch.zeros(3,device=env.device);quality_counts=torch.zeros(7,device=env.device,dtype=torch.long)
    cohort=MotionStatistics(env.stage_spec['commands'],env.device,schedule_s=spec['student_training']['train_schedule_s'],parking_s=spec['student_training']['parking_s'],fixed_shape=perf.get('fixed_shape_statistics',False))
    spin_statistics=None
    if spec['reward'].get('version') in ('student_motion31_yaw_contact_quality_v7','student_motion31_symmetry_quality_v8','student_motion31_differential_quality_v9','student_motion31_parallel_quality_v10'):
     from .motion_yaw_quality import SpinStatistics
     spin_statistics=SpinStatistics(len(env.stage_spec['commands']),env.device)
    raw_squared=torch.zeros_like(fields);bounded_tail=torch.zeros(3,device=env.device);eligible_count=torch.zeros((),device=env.device)
    before={k:v.clone() for k,v in ppo.actor.state_dict().items()};critic_before={k:v.clone() for k,v in ppo.critic.state_dict().items()}
    with torch.inference_mode():
     for step in range(steps):
      if stopped():raise TimeoutError('Motion user/wall stop during rollout')
      valid=obs['student_actor_valid'].squeeze(-1) if 'student_actor_valid' in obs else obs[__import__('native_stand.recovery.controlled_ppo',fromlist=['KEY']).KEY].squeeze(-1)
      age=env.rise_hold.episode.steps.float()*.01;command_index=env.command_index.clone()
      if minimal_startup and phase=='smoke':
       actor_obs=obs['student_actor'];expected_command=env.command/actor_obs.new_tensor([4.,4.,.5])
       wiring_command_valid&=(actor_obs[:,22:25]-expected_command).abs().amax()<=1e-6
       wiring_command_valid&=(actor_obs[:,25]==1).all()&(actor_obs[:,26:28]==0).all();wiring_command_samples+=envs
      if spec.get('motion_symmetry'):
       env.symmetry_mix=torch.tensor([x['mix'] for x in ppo.symmetry_mix_state],device=env.device)[env.command_index]
      env.motion_defer_post_checks=perf.get('motion_hotpath',False)
      raw=ppo.act(obs);newobs,reward,terminated,truncated,extras=env.step(raw);done=terminated|truncated
      if perf['failure_tail_on_device']:
       tail.append({'state':env.recovery_snapshot.detach().clone(),'posture':env.standing_snapshot.detach().clone(),
        'geometry':env.wheel_geometry_snapshot.detach().clone(),'control':env.recovery_control_snapshot.detach().clone()})
       if quality_enabled:tail[-1]['quality_termination_causes']=env.motion_quality_snapshot.detach().clone()
      if perf.get('combined_step_checks'):
       from .step_checks import StepChecks
       checks=StepChecks()
       checks.require(~env.recovery_snapshot[:,4].bool(),'runtime fault')
       checks.require(~env.recovery_snapshot[:,30].bool(),'initialization rejected')
       checks.require(torch.isfinite(reward),'nonfinite reward')
       checks.require(valid==env.student_actor_valid,'Motion actor scope mismatch')
       if perf.get('motion_hotpath'):
        pending=getattr(env,'motion_post_physics_checks',None)
        if pending is not None:
         checks.flags.extend(pending.flags);checks.names.extend(pending.names);env.motion_post_physics_checks=None
        from .motion_hotpath import runner_flags
        timeout_present=runner_flags(checks,truncated)
       else:checks.verify(RuntimeError)
      else:
       if env.recovery_snapshot[:,4].bool().any() or env.recovery_snapshot[:,30].bool().any() or not torch.isfinite(reward).all():raise RuntimeError('Motion runtime/initialization/nonfinite fault')
       if not torch.equal(valid,env.student_actor_valid):raise RuntimeError('Motion actor scope mismatch')
      storage_reward=reward
      if timeout_present if perf.get('motion_hotpath') else truncated.any():
       from .standing_survival_task import bootstrap_reward
       if env.student_terminal_critic is None:raise RuntimeError('Motion terminal Critic snapshot missing')
       storage_reward=bootstrap_reward(reward,terminated,truncated,ppo.critic(env.student_terminal_critic).squeeze(-1),gamma=ppo.gamma)
      newobs=motion_observations(env,newobs,source_retention)
      ppo.process_env_step(newobs,storage_reward,done,{k:v for k,v in extras.items() if k!='time_outs'});obs=newobs
      s=env.recovery_snapshot;counts.observe(valid,reward,done,s,env.student_event_snapshot,env.standing_snapshot,episode_duration_s=age+.01)
      cohort.observe(command_index,valid,age,env.motion_action_command,s,env.standing_snapshot,env.wheel_geometry_snapshot)
      if spin_statistics is not None:
       start,end=cohort.windows[0]
       spin_statistics.observe(command_index,valid,(age>=start)&(age<end),env.student_reward_result['spin_diagnostics'],s[:,10:12],env.motion_spin_previous_targets,env.motion_spin_executed_targets)
      fields+=env.student_reward_result['raw'].sum(0);dt_fields+=env.student_reward_result['dt'].sum(0);timeouts+=truncated.sum()
      raw_squared+=env.student_reward_result['raw'].square().sum(0)
      event_sum+=env.student_reward_result['events'].sum(0)
      if quality_enabled:quality_counts+=env.motion_quality_snapshot.sum(0)
      eligible=env.student_reward_result['eligible'];eligible_count+=eligible.sum()
      bounded_tail+=((env.student_reward_result['raw'][:,9:12]>=.95)&eligible[:,None]).sum(0)
      if perf.get('fixed_shape_statistics'):
       start,end=cohort.windows[0];holdmask=valid&(age>=start)&(age<end)
       slots.scatter_add_(0,command_index,valid.long())
       hold.scatter_add_(0,command_index,holdmask.long())
       episodes.scatter_add_(0,command_index,(valid&done).long())
      else:
       slots+=torch.bincount(command_index[valid],minlength=len(slots));start,end=cohort.windows[0];holdmask=valid&(age>=start)&(age<end)
       hold+=torch.bincount(command_index[holdmask],minlength=len(slots))
       episodes+=torch.bincount(command_index[valid&done],minlength=len(slots))

     ppo.compute_returns(obs)
    diagnostic_values=None;inertia_diagnostics=None
    if spec.get('motion_large_combo'):
     from .motion_large_combo import value_diagnostics
     diagnostic_values=value_diagnostics(ppo.storage,commands_override=env.stage_spec['commands'] if spec.get('motion_parallel') or spec.get('motion_timing') else None)
    if spec['student_training']['critic_privileged_version'] in ('motion_antiosc_privileged140_v1','motion_height_privileged144_v1','motion_height_robust_privileged184_v1'):
     from .motion_large_combo import value_diagnostics
     from .motion_antioscillation import ARMATURES
     inertia_diagnostics=value_diagnostics(ppo.storage,commands_override=list(ARMATURES),cohort_key='motion_inertia_cohort')
     for row in inertia_diagnostics:row['armature_kg_m2']=row.pop('command')
    robust_diagnostics=None
    if spec.get('motion_robustness'):
     from .motion_large_combo import value_diagnostics
     labels=[{'command':v,'DR_group':g} for v in env.stage_spec['commands'] for g in ('nominal','previous','current')]
     robust_diagnostics=value_diagnostics(ppo.storage,commands_override=labels,cohort_key='motion_command_dr_cohort')
    if spec['student_training'].get('course_version')=='motion31_persistent_frontiers_v1' and update==1:
     rr=env.student_reward_result;hi=rr['fields'].index('height');bi=rr['fields'].index('height_frontier_bonus')
     mask=env.height_adapter.height_frontier[env.command_index].to(rr['raw'].dtype)
     if tuple(rr['fields'])!=tuple(names) or not torch.allclose(rr['density'][:,bi],rr['raw'][:,hi]*mask*2.):raise RuntimeError('height frontier bonus runtime weight differs')
     if not torch.allclose(rr['dt'][:,bi],rr['density'][:,bi]*.01):raise RuntimeError('height frontier bonus dt differs')
     write(out/'persistent_reward_readback.json',{'status':'PASSED','before_first_optimizer':True,'height_base_weight':2.,'height_frontier_total_weight':4.,
      'other_cohort_bonus_zero':True,'height_bonus_nonzero_envs':int((rr['raw'][:,bi]>0).sum()),'fields':list(rr['fields']),'dt_s':.01,'scope':'actual rollout reward wiring, not height ability'})
    from .motion_training import phase_warmup
    cnt,durations=counts.export();roll_time=time.perf_counter()-start_roll;ppo.actor_update_enabled=offset+update>phase_warmup(spec['student_training'],phase);start_ppo=time.perf_counter();loss=ppo.update();ppo_time=time.perf_counter()-start_ppo
    _record_optimizer_completion(progress,update,unit,ppo.mask_summary)
    Ledger(args.spec.with_name('budget.json')).charge(args.attempt_id,out)
    changed=any(not torch.equal(v,ppo.actor.state_dict()[k]) for k,v in before.items());critic_changed=any(not torch.equal(v,ppo.critic.state_dict()[k]) for k,v in critic_before.items())
    if not torch.equal(fixed,ppo.actor.log_std) or not all(torch.isfinite(p).all() for m in (ppo.actor,ppo.critic) for p in m.parameters()):raise RuntimeError('Motion std/nonfinite optimizer fault')
    metric={'update':update,'parent_formal_offset':offset,'formal_update_including_parent':offset+update,'controlled_samples':cnt[0],'nonzero_rewards':cnt[1],'episode_completed':cnt[2],'episode_duration_mean_s':sum(durations)/len(durations) if durations else None,
     'episode_duration_max_s':max(durations,default=None),'normal_timeout_count':int(timeouts),'actor_update_scope':ppo.mask_summary,'actor_changed_this_update':changed,'critic_changed_this_update':critic_changed,
     'BC_loss':0.,'BC_gradient_norm':None,'BC_updates':0,'std_unchanged':True,'reward_fields':list(names),
     'reward_raw_mean':(fields/unit).cpu().tolist(),'reward_raw_std':(raw_squared/unit-(fields/unit).square()).clamp_min(0).sqrt().cpu().tolist(),
     'reward_bounded_tail_fraction':dict(zip(('mirror','wheel_alignment','leg_balance'),(bounded_tail/eligible_count.clamp_min(1)).cpu().tolist())),
     'reward_eligible_samples':int(eligible_count),'reward_dt_mean':(dt_fields/unit).cpu().tolist(),'reward_dt_s':spec['reward']['dt'],'cohort_statistics':cohort.export(),'episode_duration_scope':'episode policy clock; pre-reset, includes transition',
     'command_sample_counts':slots.cpu().tolist(),'hold_sample_counts':hold.cpu().tolist(),'command_episode_counts':episodes.cpu().tolist(),
     'loss':loss,'actual_LR':[g['lr'] for g in ppo.optimizer.param_groups],'wall_s':time.monotonic()-started,'timing':{'rollout_s':roll_time,'PPO_s':ppo_time}}
    if spec.get('motion_execution_robustness'):metric['execution_robustness']=env.robustness_summary()
    if spin_statistics is not None:metric['spin_statistics']=spin_statistics.export()
    if course:
     if spec.get('motion_decoupled'):
      metric['task_levels']=course_state['levels'];metric['active_focus']=course_state['focus']
     metric['course']={'stage':course_state['stage'],'stage_name':spec['curriculum']['stages'][0]['name'],'stage_updates':update-course_state['stage_started_update'],'formal_updates_global':offset+update,'formal_updates_local':update,'source_recipe_updates':spec['motion_course']['source_recipe_updates'],'accepted_stage_count':len(course_state['accepted_by_stage'])}
    if diagnostic_values is not None:metric['cohort_value_diagnostics']=diagnostic_values
    if spec.get('mcu_execution'):
     from .motion_mcu_course import value_diagnostics
     metric['cohort_value_diagnostics']=value_diagnostics(ppo.storage,env.stage_spec['commands'])
     if spec.get('motion_height_dr'):
      stage=env.stage_spec
      metric['height_PD_course']=(dict(focus='fixed034_standing_delay20',PD_feedback_age_ms=0,
        Actor_sensor_ms=20,target_ms=5,motor_transport_ms=20,motor_tau_ms=2)
        if spec.get('motion_standing_delay') else
        {k:course_state[k] for k in ('pd_level','height_level','qualified_pd_levels','qualified_height_levels')})
      for i,row in enumerate(metric['cohort_value_diagnostics']):
       row.update(height_target_m=stage['height_targets_m'][i],mcu_variant=stage['mcu_variants'][i],
                  purpose=stage['cohort_purposes'][i],PD_age_max_ms=stage.get('PD_feedback_max_ms',0))
      for scope in metric['cohort_statistics'].values():
       for i in range(len(stage['commands'])):
        scope[f'cohort_{i}'].update(height_target_m=stage['height_targets_m'][i],mcu_variant=stage['mcu_variants'][i],purpose=stage['cohort_purposes'][i])
     if spec.get('mcu_posture'):
      metric['posture_reward_version']=spec['student_training']['posture_reward']['version']
      for row,variant in zip(metric['cohort_value_diagnostics'],env.stage_spec['mcu_variants']):row['mcu_variant']=variant
      for scope in metric['cohort_statistics'].values():
       for i,variant in enumerate(env.stage_spec['mcu_variants']):scope[f'cohort_{i}']['mcu_variant']=variant
     t=env.motion_timing;caps=t.driver.new_tensor(spec['mcu_execution']['profile']['driver_limits_nm'])
     metric['mcu_execution']={'driver_saturation_fraction':float((t.driver.abs()>=caps-1e-4).float().mean()),
       'motor_abs_peak_Nm':t.motor_actual.abs().amax(0).cpu().tolist(),'net_joint_abs_peak_Nm':t.actual.abs().amax(0).cpu().tolist(),
       'scope':'last policy sample; deterministic full-window assessment separately recorded'}
     numbers=t.statistics.cpu().tolist();den=max(1,numbers[0])
     metric['mcu_execution']['rollout']={'eligible_samples':int(numbers[0]),'single_wheel_lost_fraction':numbers[1]/den,
       'both_wheels_lost_fraction':numbers[2]/den,'any_driver_saturation_fraction':numbers[3]/den,
       'longest_wheel_loss_s':(t.contact_max*.01).cpu().tolist(),'geometry_invalid_samples':int(numbers[4])}
    if inertia_diagnostics is not None:metric['armature_value_diagnostics']=inertia_diagnostics
    if spec.get('motion_symmetry'):metric['symmetry_mix_state']=ppo.symmetry_mix_state
    if refinement:metric['reward_refinement_state']=ppo.reward_refinement_state
    metric['reward_event_sum']=dict(zip(('success','task_failure','runtime_fault'),event_sum.cpu().tolist()))
    metric['reward_total_per_step']=float((dt_fields.sum()+event_sum.sum())/unit)
    if quality_enabled:
     from .motion_quality import CAUSES
     metric['quality_termination_counts']=dict(zip(CAUSES,quality_counts.cpu().tolist()))
     metric['geometry_high_error_fraction']=metric.pop('reward_bounded_tail_fraction')
    if hasattr(env,'motion_timing'):
     metric['timing_randomization']=env.motion_timing.summary()
    if hasattr(env,'motion_dr'):metric['domain_randomization']=env.motion_dr.summary()
    if robust_diagnostics is not None:metric['command_DR_value_diagnostics']=robust_diagnostics
    log_metric=metric
    if (spec.get('motion_decoupled') or spec.get('motion_highspeed_standing') or spec.get('motion_standing_delay')) and update%100 and update!=limit:
     # Full latest progress stays available; persist detailed cohort matrices at
     # the requested100-update boundaries instead of10000 duplicate copies.
     log_metric={k:v for k,v in metric.items() if k not in ('command_DR_value_diagnostics','cohort_value_diagnostics','armature_value_diagnostics','cohort_statistics','spin_statistics','symmetry_mix_state','reward_refinement_state','domain_randomization','timing_randomization')}
     log_metric['detail_scope']='compact per-update; full cohort/tracking/value detail every100'
    with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(log_metric,allow_nan=False)+'\n')
    if tb_writer is not None:
     from native_stand.command_v6.tools.metrics_tensorboard import row_scalars
     for tag,value in row_scalars(metric):
      if not spec.get('motion_robustness') or update%10==0 or tag.startswith(('loss/','reward/','timing/','actual_LR/','reward_total_per_step')):tb_writer.add_scalar(tag,value,update)
     for row in inertia_diagnostics or []:
      for key in ('explained_variance','value_error_MSE','return_variance','samples'):
       if row[key] is not None:tb_writer.add_scalar(f'CriticArmature/{row["armature_kg_m2"]:g}/{key}',row[key],update)
    progress.update(status='RUNNING',actor_parameters_changed=any(not torch.equal(v,ppo.actor.state_dict()[k]) for k,v in source_actor_weights.items()),latest_metric=metric)
    write(out/'progress.json',progress)
    if not sent and args.handshake_fd is not None:
     send_handshake(args.handshake_fd,{'status':'RUNNING','attempt_id':args.attempt_id,'trainer_pid':__import__('os').getpid(),'actual_updates':update,'output_dir':str(out.resolve())});sent=True
    save_due,evaluate_due=checkpoint_plan(spec['student_training'],phase,update,limit,offset=offset if spec.get('motion_timing') else 0)
    quick_candidate=None
    if save_due and not evaluate_due:
     quick_candidate=save_unevaluated(out,spec,ppo,update,progress,candidates,selection_fn)
    if spec.get('mcu_execution') and phase=='formal' and (update%100==0 or update==limit):
     from .motion_mcu_course import save_execution_tail
     save_execution_tail(out,env,update)
    if (spec.get('motion_parallel') or spec.get('motion_timing')) and phase=='formal' and (update+(offset if spec.get('motion_timing') else 0))%100==0:
     from .motion_parallel_course import checkpoint_diagnostics
     if quick_candidate is None:quick_candidate=save(out,spec,ppo,update)
     destination=out/'lr_diagnostics';destination.mkdir(exist_ok=True)
     write(destination/f'update_{offset+update:04}.json',checkpoint_diagnostics(metric,quick_candidate))
    from .motion_quick_evaluation import due as quick_due,run as quick_evaluate
    from .motion_quick_evaluation import validate as quick_config,WARM_VERSION
    quick_settings=quick_config(spec);warm_quick=bool(quick_settings and quick_settings['version']==WARM_VERSION)
    screen_due=quick_due(spec,phase,update)
    if screen_due and warm_quick and not evaluate_due:
     # PPO storage/returns/update are complete. No GAE sequence crosses this reset.
     from .motion_evaluation_review import warm_screen,screen_training_boundary
     if quick_candidate is None:quick_candidate=save(out,spec,ppo,update)
     def rebuild_after_screen():
      fresh=environment(spec,'MotionZeroStudent',envs=envs);fresh.curriculum_training=True
      if refinement:fresh.motion_cohort_mix.copy_(torch.tensor([r['mix'] for r in ppo.reward_refinement_state],device=fresh.device))
      fresh.select_initial_family('motion_native_normal','train');newobs,_=fresh.reset()
      return fresh,motion_observations(fresh,newobs,source_retention)
     oldenv,env=env,None
     env,obs,result=screen_training_boundary(oldenv,lambda:warm_screen(spec,out,ppo.actor,quick_candidate['checkpoint'],update,stop=stopped),rebuild_after_screen)
     progress.setdefault('checkpoint_quick_evaluations',[]).append(result)
     write(out/'progress.json',progress)
    elif screen_due and not warm_quick:
     # Independent child leaves this live training environment and RNG untouched.
     if quick_candidate is None:quick_candidate=save(out,spec,ppo,update)
     result=quick_evaluate(out,spec,quick_candidate['checkpoint'],update=update,parent_spec=args.spec,attempt_id=args.attempt_id,deadline=deadline)
     progress.setdefault('checkpoint_quick_evaluations',[]).append(result)
     write(out/'progress.json',progress)
    if evaluate_due:
     candidate=quick_candidate if quick_candidate is not None else save(out,spec,ppo,update);progress['latest_checkpoint']={**candidate,'evaluated':False};progress['checkpoint_roundtrip_verified']=True;env.close();env=None
     if minimal_startup and phase=='smoke':
      progress['wiring_command_readback']={'valid':bool(wiring_command_valid),'sample_count':wiring_command_samples,'scope':'all real pre-action Actor command/task observations'}
      progress['independent_capability_evaluation']={'status':'NOT_RUN','reason':'explicit minimal startup; full assessment reserved for formal milestones'}
      candidate.update(per_env=[],complete_matrix=False,evaluations=[])
      progress['latest_checkpoint']={**candidate,'evaluated':False};candidates.append(candidate)
      write(out/'candidate_registry.json',candidates);write(out/'candidate_selection.json',selection_fn(candidates));write(out/'progress.json',progress)
      continue
     records=evaluate_cases(spec,out,ppo.actor,cases,offset=index,stop=stopped,checkpoint=candidate['checkpoint'],source_records=candidates[0]['evaluations']);index+=len(records)
     if screen_due and warm_quick:
      from .motion_evaluation_review import warm_screen
      result=warm_screen(spec,out,ppo.actor,candidate['checkpoint'],update,records=records)
      progress.setdefault('checkpoint_quick_evaluations',[]).append(result)
     candidate.update(per_env=[e for r in records for e in r['per_env']],complete_matrix=len(records)==len(cases),evaluations=records)
     progress['latest_checkpoint']={k:candidate[k] for k in ('checkpoint','sha256','update','stage')};progress['latest_checkpoint']['evaluated']=True
     if spec.get('motion_parallel'):candidate['source_sha256']=spec['source_sha256']
     candidates.append(candidate);selection=selection_fn(candidates);write(out/'candidate_registry.json',candidates);write(out/'candidate_selection.json',selection);progress['evaluations']+=records;write(out/'progress.json',progress)
     # Only genuinely passed source capabilities are retention prerequisites.
     retained=[i for i,e in enumerate(candidates[0]['per_env']) if e['accepted']]
     if spec.get('motion_robustness'):
      # A changing command x DR matrix cannot use legacy flat-env indices.
      # advance_course below owns qualified-command regression and rollback.
      regression=False
     elif spec.get('motion_height_dr'):
      from .training import course_baseline
      regression=False
      for record in records:
       reference=course_baseline(spec,record['case'],candidates[0]['evaluations'])
       if reference and all(e.get('accepted') is True for e in reference['per_env']):
        regression |= any(e.get('accepted') is not True for e in record['per_env'])
     elif spec.get('motion_parallel'):
      from .motion_parallel_course import retained_regression
      regression=retained_regression(candidates[0]['evaluations'],records)
     else:regression=bool(retained) and any(not candidate['per_env'][i]['accepted'] for i in retained)
     progress['retention_failures']=progress.get('retention_failures',0)+1 if regression else 0
     retention_stop=progress['retention_failures']>=2
     if tolerant_guard and phase=='formal':
      from .motion_guard import advance as advance_guard
      if course:
       old_commands={tuple(r['case']['command']) for r in retention_baseline if all(e['accepted'] for e in r['per_env'])}
       old_records=[r for r in records if tuple(r['case']['command']) in old_commands]
       old_baseline=[next(b for b in retention_baseline if b['case']['command']==r['case']['command']) for r in old_records]
       guard_update=update-course_state['stage_started_update']
      else:old_records=records;old_baseline=candidates[0]['evaluations'];guard_update=update
      guard_state,decision=advance_guard(guard_state,old_records,old_baseline,guard_update,tolerant_guard)
      from .motion_guard import apply_behavior_mode,validate_behavior_mode
      decision=apply_behavior_mode(decision,validate_behavior_mode(spec))
      progress['retention_guard']=decision;progress['retention_guard_state']=guard_state;retention_stop=decision['stop']
      write(out/'progress.json',progress)
     if spec.get('motion_symmetry') and phase=='formal' and not spec.get('motion_decoupled'):
      from .motion_symmetry import update_mix
      if spec.get('motion_differential'):
       from .motion_differential import update_mix
      mix_records=records
      if spec.get('motion_timing'):
       from .motion_course import implementation
       refinement_records=implementation(spec['student_training']['course_version']).refinement_records
       mix_records=refinement_records(records,course_state['stage'])
      ppo.symmetry_mix_state=update_mix(ppo.symmetry_mix_state,mix_records)
      write(out/f'symmetry_refinement_after_{update}.json',{'update':update,'state':ppo.symmetry_mix_state})
     course_decision=None
     if course and phase=='formal':
      from .motion_course import implementation
      course_module=implementation(spec['student_training']['course_version'])
      advance_course,effective_spec,extend_mixture=course_module.advance,course_module.effective_spec,course_module.extend_mixture
      if spec.get('motion_robustness'):
       from .motion_robustness import archive_mixtures
       archive_mixtures(course_state,ppo)
      if spec.get('motion_decoupled'):
       from .motion_decoupled import refine_mixtures
       refine_mixtures(ppo,spec,records)
      spec,course_state,course_decision=advance_course_boundary(spec,ppo,candidate)
      write(out/'course_progress.json',{'state':course_state,'decision':course_decision,'checkpoint':candidate['checkpoint'],'checkpoint_sha256':candidate['sha256']})
      selection=selection_fn(candidates);write(out/'candidate_selection.json',selection)
     final_confirmed=course_decision['all_stages_passed'] if course_decision else bool(selection['accepted_candidate'])
     if spec.get('motion_parallel'):
      from .motion_parallel_course import may_stop_for_behavior
      progress['behavior_stop_observe_only']={'regression':retention_stop,'all_stages_passed':final_confirmed,'stop':False}
      final_confirmed=retention_stop=may_stop_for_behavior(regression=retention_stop,all_stages_passed=final_confirmed)
     if spec.get('motion_timing'):
      progress['behavior_stop_observe_only']={'regression':retention_stop,'confirmed':final_confirmed,'stop':False}
      final_confirmed=retention_stop=False
     if phase=='formal' and (final_confirmed or retention_stop):
      status='STOPPED_BOUNDED';progress['stop_reason']='stage_confirmed' if final_confirmed else 'repeated_certified_source_regression';break
     if refinement and phase=='formal' and update<limit and not spec.get('motion_decoupled'):
      if spec.get('motion_parallel'):
       from .motion_parallel_course import update_mixture
       ppo.reward_refinement_state,decisions=update_mixture(ppo.reward_refinement_state,records,refinement)
      else:
       refinement_input=records
       if spec.get('motion_timing'):
        from .motion_course import implementation
        refinement_records=implementation(spec['student_training']['course_version']).refinement_records
        refinement_input=refinement_records(records,candidate['stage'])
       ppo.reward_refinement_state,decisions=advance(ppo.reward_refinement_state,refinement_input,refinement)
      progress['reward_refinement_state']=ppo.reward_refinement_state
      write(out/f'reward_refinement_after_{update}.json',{'update':update,'policy':refinement,'next_state':ppo.reward_refinement_state,'decisions':decisions,'evaluation_inputs':[{'path':r['path'],'sha256':r['sha256']} for r in records]})
     if course_decision and course_decision['advanced'] and update<limit:
      retention_baseline=records;guard_state={}
      spec=effective_course_spec(spec,course_state);cases=spec['student_evaluation']['cases']
      values=spec['curriculum']['stages'][0]['cohort_keys'] if spec.get('motion_decoupled') else spec['curriculum']['stages'][0]['commands']
      ppo.reward_refinement_state=extend_mixture(ppo.reward_refinement_state,values)
      if spec.get('motion_parallel') or spec.get('motion_reacquisition') or spec.get('motion_robustness') or spec.get('mcu_execution'):
       ppo.symmetry_mix_state=extend_mixture(ppo.symmetry_mix_state,values)
      if spec.get('motion_robustness') and not spec.get('motion_decoupled'):
       from .motion_robustness import restored_mixture
       values=spec['curriculum']['stages'][0]['commands']
       ppo.reward_refinement_state=restored_mixture(course_state,'mixture_archive',ppo.reward_refinement_state,values)
       ppo.symmetry_mix_state=restored_mixture(course_state,'symmetry_archive',ppo.symmetry_mix_state,values)
     if course and phase=='formal' and update-course_state['stage_started_update']>=spec['motion_course']['configuration']['stage_updates_max']:
      status='STOPPED_BOUNDED';progress['stop_reason']='stage_budget_exhausted';break
     if spec.get('motion_timing',{}).get('version')=='motion31_standing_antiosc_armature_v4':
      spec=effective_course_spec(spec,course_state);cases=spec['student_evaluation']['cases']
     if spec.get('motion_robustness'):
      from .motion_robustness import compact_historical_records
      compact_historical_records(records)
     if update<limit:
      env=environment(spec,'MotionZeroStudent',envs=envs);env.curriculum_training=True
      if hasattr(env,'motion_dr'):ppo.motion_dr_adapter=env.motion_dr
      if spec.get('motion_timing',{}).get('version')=='motion31_standing_antiosc_armature_v4':env.motion_timing.configure(None)
      if refinement:env.motion_cohort_mix.copy_(torch.tensor([r['mix'] for r in ppo.reward_refinement_state],device=env.device))
      env.select_initial_family('motion_native_normal','train');obs,_=env.reset();obs=motion_observations(env,obs,source_retention)
   if spec['student_training'].get('course_version')=='motion31_persistent_frontiers_v1':
    from .motion_persistent import frontiers
    stage=env.stage_spec
    observed=torch.bincount(env.command_index,minlength=len(stage['commands']))
    expected=torch.tensor(stage['slot_counts'],device=env.device)
    if not torch.equal(observed,expected):raise RuntimeError('persistent runtime cohort quota differs')
    if not torch.equal(env.motion_dr.level_ids,torch.tensor(stage['DR_level_slots'],device=env.device)):raise RuntimeError('persistent DR level wiring differs')
    targets=torch.tensor(stage['commands'],device=env.device)[env.command_index]
    if not torch.allclose(env.target_xy,targets):raise RuntimeError('persistent target command wiring differs')
    if not torch.allclose(env.height_adapter.target,env.height_adapter.initial):raise RuntimeError('persistent fixed height differs')
    write(out/'persistent_curriculum_readback.json',{'status':'PASSED','frontiers':frontiers(course_state),
     'cohort_axes':stage['cohort_axes'],'cohort_purposes':stage['cohort_purposes'],'slot_counts':observed.cpu().tolist(),
     'DR_levels':sorted(set(stage['DR_level_slots'])),'height_frontier_reward':spec['student_training']['height_frontier_reward'],
     'actual31_184':list(obs['student_actor'].shape)+list(obs['student_critic'].shape),'PPO_updates':0,'scope':'initial runtime quotas/commands/heights/DR; not ability proof'})
    if warm_quick and phase=='formal' and update%100==0:
     from .motion_evaluation_review import write_review
     progress['human_evaluation_review']=write_review(out)
     write(out/'progress.json',progress)
   else:status='COMPLETED'
 except BaseException as exc:
  if tail:
   from .artifacts import npz
   def preserve_failure_tail():
    maximum=spec.get('motion_robustness',{}).get('failure_tail_saved_envs')
    if maximum is None:arrays={k:torch.stack([r[k] for r in tail]).cpu().numpy() for k in tail[0]}
    else:
     from .motion_robustness import failure_tail_arrays
     material=getattr(getattr(env,'motion_dr',None),'material_failure',None)
     arrays=failure_tail_arrays(tail,maximum,material['env_ids'] if material else ())
     write(out/'failure_tail_storage.json',{'saved_envs':len(arrays['env_ids']),'env_ids':arrays['env_ids'].tolist(),
      'all_env_final_state_retained':True,'full_all_env_histories_retained':False,'material_readback_failure':material})
    npz(out/'failure_tail.npz',arrays)
   secondary(exc,'Motion failure tail',preserve_failure_tail)
  failure={'exception':repr(exc),'traceback':traceback.format_exc()};status='STOPPED_BOUNDED' if isinstance(exc,TimeoutError) else 'FAILED'
 finally:
  if tb_writer is not None:
   try:tb_writer.close()
   except Exception as exc:cleanup.append('tensorboard.close: '+repr(exc))
  if ppo is not None and progress['actual_updates'] and not (out/f"model_update_{progress['actual_updates']}.pt").exists():
   latest=secondary(RuntimeError('terminal preservation'),'latest checkpoint',lambda:save(out,spec,ppo,progress['actual_updates']))
   if latest:progress['latest_checkpoint']={**latest,'evaluated':False}
  if progress.get('latest_checkpoint'):
   selection_path=out/'candidate_selection.json'
   if selection_path.exists():
    selected=json.loads(selection_path.read_text());selected['latest_checkpoint']=progress['latest_checkpoint'];write(selection_path,selected)
  stop_kind='user_signal' if __import__('native_stand.command_v6.train',fromlist=['STOP']).STOP else progress.get('stop_reason','budget_or_completion')
  write(out/'progress.json',{**progress,'status':status});write(out/'terminal_report.json',{'status':status,'attempt_id':args.attempt_id,'phase':phase,'actual_updates':progress['actual_updates'],'samples':progress['samples'],'effective_actor_samples':progress['effective_actor_samples'],'evaluations':progress['evaluations'],'failure':failure if status=='FAILED' else None,'budget_interruption':failure if status=='STOPPED_BOUNDED' else None,'stop_kind':stop_kind,'wall_s_before_cleanup':time.monotonic()-started})
  if env is not None:
   try:env.close()
   except BaseException as exc:cleanup.append('env.close: '+repr(exc))
  write(out/'cleanup.json',{'errors':cleanup,'app_close':'pending'})
  if app is not None:
   try:app.close(exit_code=0 if status in ('COMPLETED','STOPPED_BOUNDED') and not cleanup else 1)
   except BaseException as exc:cleanup.append('app.close: '+repr(exc))
  write(out/'cleanup.json',{'errors':cleanup,'app_close':'returned'})
 return 0 if status in ('COMPLETED','STOPPED_BOUNDED') and not cleanup else 1
