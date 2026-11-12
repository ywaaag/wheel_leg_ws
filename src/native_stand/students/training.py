"""Student recipe adapter invoked by public train.py; shared PPO/storage/owner."""
import copy,json,os,time,traceback,sys
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from .data import sha
from .evaluator import environment,rollout,teacher_models
from .model import load
from .ppo import assemble
from .artifacts import npz as atomic_npz,checkpoint as atomic_checkpoint,storage,secondary
from native_stand.recovery.controlled_ppo import KEY,pre_action_valid
from native_stand.recovery.domain import Mode
from native_stand.command_v6.completion import write
from native_stand.recovery.handoff_evaluator import atomic_result

def compact_motion_evidence(spec,arrays,phase):
 """Same compact field owner as evaluator CLI; retain source calibration proof."""
 from .evaluator import _compact_motion_evidence
 result=_compact_motion_evidence(arrays,wheel_scale=spec['action']['wheel_scale_rad_s'])
 if spec.get('mcu_execution'):result['mcu_execution']=arrays['mcu_execution']
 if phase=='source_eval':
  if spec.get('motion_spin_quality'):
   result.update({k:v for k,v in arrays.items() if k.startswith('spin_')})
  if spec.get('motion_symmetry'):
   result.update({k:arrays[k] for k in ('actor_input','critic_input','control') if k in arrays})
 if spec.get('motion_timing',{}).get('version')=='motion31_standing_antiosc_armature_v4':
  from native_stand.command_v6.domain.trace_schema import CONTROL
  result.update(antiosc_request_torque_Nm=arrays['control'][...,CONTROL['requested_torque']],antiosc_applied_torque_Nm=arrays['control'][...,CONTROL['applied_torque']],antiosc_leg_target_rad=arrays['control'][...,CONTROL['leg_target_after_limit']])
 if spec.get('motion_differential'):
  required={'differential_target','differential_difference','differential_error','differential_rolling'}
  if not required<=set(result):raise ValueError('differential compact evidence missing before save')
 return result

def observations(env,obs):
 result=TensorDict(obs,batch_size=[env.num_envs]).clone()
 result[KEY]=(env.actor_valid_pre() if hasattr(env,'actor_valid_pre') else pre_action_valid(env.fsm.mode==int(Mode.INITIALIZING),env.initialization_checked))[:,None]
 return result

def rank(row):
 if row.get('standing_survival_quality'):
  from .standing_survival_task import rank as quality_rank
  return quality_rank(row)
 if row.get('stop_quality_selection'):
  from .recovery_large_training import selection_rank
  return selection_rank(row)
 return (sum(r['accepted'] for r in row['per_env']),sum(r['entry_index'] is not None for r in row['per_env']),sum(r.get('valid_hold_prefix_s',0.) for r in row['per_env']),-row.get('update',0))

def choose(candidates):
 if not candidates:raise ValueError('candidate comparisons absent')
 best=max(candidates,key=rank)
 repeated=True
 if best.get('standing_alignment_repeat_required') or best.get('standing_survival_quality'):
  passing=[c for c in candidates if c.get('update',0)>0 and len(c.get('per_env',[]))==24 and all(r['accepted'] for r in c['per_env'])]
  repeated=len(passing)>=2 and max(c['update'] for c in passing)-min(c['update'] for c in passing)>=100
 accepted=best if repeated and all(r['accepted'] for r in best['per_env']) and not best.get('teacher_prefix_intervals',0) else None
 if best.get('standing_survival_quality'):
  accepted=max((r for r in candidates if r.get('update',0)>0 and len(r['per_env'])==24 and all(e['accepted'] for e in r['per_env'])),key=rank,default=None) if repeated else None
 return {'source':next((r for r in candidates if r.get('update')==0),None),
  'latest':max(candidates,key=lambda r:r.get('update',0)),'best_new':max((r for r in candidates if r.get('update',0)>0),key=rank,default=None),
  'best_overall':best,'best_observed_candidate':best,'accepted_candidate':accepted,
  'accepted_suffix_candidate':best if best.get('teacher_prefix_intervals',0)>0 and all(r['accepted'] for r in best['per_env']) else None,
  'capability_scope':best.get('capability_scope','Swing autonomous recovery'),
  'selection':'same declared primary cases: accepted count, entry count, first-window valid prefix, earlier checkpoint'}

def course_baseline(spec,row,records):
 """Match an evaluated command in its physical/scoring context, never by label suffix alone."""
 matches=[r for r in records if r['case']['command']==row['command']]
 if spec.get('mcu_execution'):
  matches=[r for r in matches if r['case'].get('_mcu_variant')==row.get('_mcu_variant')]
  if spec.get('motion_execution_robustness'):
   fields=('kind','role','family','seed','yaw_rad','seconds','_timing_level','_mcu_revision','_execution_uncertainty','_height')
   if spec.get('motion_height_dr'):fields=tuple(k for k in fields if k!='_mcu_revision')
   matches=[r for r in matches if all(r['case'].get(k)==row.get(k) for k in fields)]
 elif spec.get('motion_robustness'):
  context=row.get('_motion_dr',{})
  if any(k not in context for k in ('level','group')):raise ValueError('Motion DR comparison scope missing')
  matches=[r for r in matches if all(r['case'].get('_motion_dr',{}).get(k)==context.get(k) for k in ('level','group','seed'))
   and all(r['case'].get(k)==row.get(k) for k in ('kind','role','family','seed','yaw_rad','seconds','_timing_level'))]
 elif spec.get('motion_timing'):
  matches=[r for r in matches if r['case']['case_id'].rsplit('_',1)[-1]==row['case_id'].rsplit('_',1)[-1]]
 if len(matches)>1:raise ValueError('duplicate course baseline command')
 return matches[0] if matches else None


def evaluate_cases(spec,output,actor,cases,*,offset=0,stop=None,teachers=None,checkpoint=None,source_records=None,screen_single_env=False):
 records=[]
 baselines=[course_baseline(spec,row,source_records or []) for row in cases] if spec.get('motion_training') and spec.get('motion_course') else None
 from .performance import settings,EvaluationEnvironment
 perf=settings(spec)
 groups=None
 if perf.get('evaluation_group_commands') and not screen_single_env:
  from .grouped_evaluation import CommandGroups
  groups=CommandGroups(cases,maximum_commands=perf.get('evaluation_group_max_commands',4))
 pool=EvaluationEnvironment(perf['evaluation_reuse_environment'],lambda row:environment(spec,row['role'],envs=1 if screen_single_env else 64,teacher=row['kind'] not in ('student','student_suffix'),initial_family=row['family'],autonomous_preview=screen_single_env,single_condition_screen=screen_single_env))
 condition_states=[{'case_id':r['case_id'],'status':'NOT_RUN'} for r in cases]
 write(output/f'evaluation_matrix_{offset:02}.json',condition_states)
 for i,row in enumerate(cases):
  env=None;cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
  try:
   condition_states[i]['status']='INCOMPLETE';write(output/f'evaluation_matrix_{offset:02}.json',condition_states)
   setup_started=time.perf_counter();teacher=row['kind'] not in ('student','student_suffix');env=pool.acquire(row);setup_wall=time.perf_counter()-setup_started
   from native_stand.command_v6.runtime.configure import verify_asset_import
   from native_stand.command_v6.runtime.control_readback import verify,verify_ground
   write(output/f'evaluation_readback_{offset+i:02}.json',{'asset':verify_asset_import(spec),'control':verify(env),'ground':verify_ground(env,spec),'active_indices':env.ids,'joint_names':env.robot.joint_names,'reference':env.reference[:8].detach().cpu().tolist()})
   if not teacher and checkpoint:
    row={**row,'checkpoint':str(Path(checkpoint).resolve()),'sha256':sha(checkpoint)}
   if hasattr(env,'recovery_dr'):env.recovery_dr.configure(override=row.get('_dr_override'))
   if hasattr(env,'motion_timing'):env.motion_timing.configure(row['_timing_level'])
   if hasattr(env,'configure_mcu_cases'):env.configure_mcu_cases([row])
   if hasattr(env,'motion_dr'):env.motion_dr.configure(row['_motion_dr'])
   models=teachers if teacher else [actor]
   before=[{k:v.clone() for k,v in model.state_dict().items()} for model in models]
   rollout_started=time.perf_counter()
   if groups is not None:
    for case in groups.collecting_cases(row):
     condition_states[next(j for j,c in enumerate(cases) if c['case_id']==case['case_id'])]['status']='INCOMPLETE'
    write(output/f'evaluation_matrix_{offset:02}.json',condition_states)
    payload,arrays=groups.acquire(env,actor,row,stop=stop)
    payload['case']=row
   else:payload,arrays=rollout(env,actor,row,teacher=teachers if teacher else None,stop=stop,scored_envs=1 if screen_single_env else 8)
   rollout_wall=time.perf_counter()-rollout_started
   if not all(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()) for model,state in zip(models,before)):raise RuntimeError('evaluation modified fixed model')
   payload['model_unchanged']=True
   payload.update(checkpoint_sha256=row['sha256'],asset_sha256=spec['asset_sha256'],spec_sha256=sha(output/'source_snapshot/spec.json'))
   if spec.get('motion_training'):
    from .motion_training import compare_source
    matched_baseline=True
    if spec.get('motion_course'):
     baseline=baselines[i]
     matched_baseline=baseline is not None
     if baseline is None:baseline={'per_env':copy.deepcopy(payload['per_env'])}
    else:
     if source_records is not None and (len(source_records)!=len(cases) or source_records[i]['case']['command']!=row['command']):raise ValueError('Motion source comparison command matrix differs')
     baseline=source_records[i] if source_records is not None else {'per_env':copy.deepcopy(payload['per_env'])}
    if (spec.get('motion_robustness') or spec.get('motion_height_dr')) and not matched_baseline:
     for result in payload['per_env']:result['source_posture_delta_rad']={'pitch':None,'roll':None}
    elif spec.get('motion_training',{}).get('early_stop_policy'):
     for result,reference in zip(payload['per_env'],baseline['per_env']):
      result['source_posture_delta_rad']={axis:result['tracking'].get(axis+'_hold_abs_mean_rad',0)-reference['tracking'].get(axis+'_hold_abs_mean_rad',0) for axis in ('pitch','roll')}
    else:compare_source({'per_env':payload['per_env']},baseline)
    if spec['reward']['version']=='student_motion31_hold_level_reward_v4':
     from .motion_successor import apply_assessment
     for result in payload['per_env']:apply_assessment(result)
    if spec.get('motion_timing',{}).get('version')=='motion31_standing_antiosc_armature_v4':
     from .motion_antioscillation import assess
     for eid,result in enumerate(payload['per_env']):
      assess(result,arrays.get('control')[:,eid] if arrays.get('control') is not None else None,arrays['first_episode_valid'][:,eid]&(arrays['controller'][:,eid,0]==3),arrays['state'][:,eid])
    if spec.get('mcu_execution'):
     from .motion_mcu_course import assess as mcu_assess,protocol as mcu_protocol
     if spec.get('motion_height_dr'):
      from .motion_height_pd import assess as mcu_assess,protocol as mcu_protocol
     if spec.get('motion_highspeed_standing'):
      from .motion_highspeed_standing import assess as mcu_assess,protocol as mcu_protocol
     if spec.get('motion_standing_delay'):
      from .motion_standing_delay import assess as mcu_assess,protocol as mcu_protocol
     for eid,result in enumerate(payload['per_env']):mcu_assess(result,arrays,eid,row)
     payload['protocol']=mcu_protocol()
    elif spec.get('motion_decoupled'):
     from .motion_decoupled import assess as height_assess,protocol as height_protocol
     if spec['motion_decoupled'].get('fixed_episode_height'):
      from .motion_fixed_height import assess as height_assess,protocol as height_protocol
     for eid,result in enumerate(payload['per_env']):height_assess(result,arrays,eid,row)
     payload['protocol']=height_protocol()
    elif spec.get('motion_reacquisition') or spec.get('motion_robustness'):
     from .motion_reacquisition import assess as assess_reacquisition,scoring_protocol
     for eid,result in enumerate(payload['per_env']):assess_reacquisition(result,arrays,eid)
     payload['protocol']=scoring_protocol(spec)
    payload['passed_envs']=sum(e['accepted'] for e in payload['per_env']);payload['accepted']=all(e['accepted'] for e in payload['per_env'])
    payload['source_comparison_scope']='same-command/env baseline' if source_records is not None else 'source compared to itself'
    if spec.get('motion_robustness'):
     payload['source_comparison_scope']='same command/DR level/group/seed/window baseline; absolute acceptance unchanged' if matched_baseline else 'NOT_AVAILABLE: no matching command/DR/seed/window baseline; absolute acceptance unchanged'
   from .protocol import score
   # Report first fixed window's valid prefix, never reselection.
   for e,r in enumerate(payload['per_env']):
    entry=r['entry_index'];prefix=0
    if entry is not None:
     s=arrays['state'][:,e];p=arrays['posture'][:,e]
     for k in range(entry+1,min(entry+1001,len(s))):
      if not arrays['first_episode_valid'][k,e] or s[k,6]<.30 or (abs(p[k,:2])>.15).any() or s[k,15]>.18 or (s[k,10:12]<=2).any() or s[k,13]<=0 or s[k,12]>.005:break
      prefix+=1
    r['valid_hold_prefix_s']=prefix*.01
    if spec.get('recovery_domain_randomization'):
     r['dr_nominal_quality']={'mean_abs_vx':float(r['velocity_means'][0]) if r.get('velocity_means') is not None else None,'entry_s':float(arrays['state'][entry,e,0]) if entry is not None else None}
    if r.get('hold_index_half_open'):
     lo,hi=r['hold_index_half_open'];s=arrays['state'][lo:hi,e];c=arrays['control'][lo:hi,e]
     from native_stand.command_v6.domain.trace_schema import STATE,CONTROL
     wheel=c[:,CONTROL['clipped_action']][:,[2,5]]*100.
     r['limit_fraction']=float(s[:,STATE['saturated']].mean())
     r['wheel_delta_rms']=np.sqrt(np.mean(np.diff(wheel,axis=0)**2,axis=0)).tolist()
     torque=c[:,CONTROL['physical_clipped_torque']]
     r['channel_limit_fraction']=(abs(torque)>=np.array([40,40,5.5,40,40,5.5])-1e-4).mean(0).tolist()
     r['wheel_torque_sign_changes']=(torque[1:,[2,5]]*torque[:-1,[2,5]]<0).sum(0).tolist()
   if hasattr(env,'recovery_dr'):payload['domain_randomization']=env.recovery_dr.summary()
   if hasattr(env,'motion_timing'):payload['motion_timing']=env.motion_timing.summary()
   if spec.get('motion_execution_robustness'):
    payload['execution_robustness']={'scope':'hypothetical continuous frozen4500 handover; not LQR or real hardware',
      'PD_feedback_ms_scored_initial':arrays['mcu_PD_age_ms_pre'][0,:,0].tolist(),
      'teacher_prefix_intervals':row.get('_execution_uncertainty',{}).get('prefix_intervals',0),
      'handover_events':[{ 'env':int(e),'policy_index':int(k),
          'physical_state':arrays['handoff_state'][k,e].tolist()}
          for k,e in np.argwhere(arrays['handoff_event'][...,0].astype(bool))],
      'no_state_cache_queue_reset_at_handover':True}
   if hasattr(env,'motion_dr'):
    payload['motion_domain_randomization']=env.motion_dr.summary()
    payload['motion_domain_randomization']['scored_initial_parameters']=arrays['motion_dr_parameters_pre'][0].tolist()
    payload['motion_domain_randomization']['parameter_scope']='initial first episode for these8 scored envs; fixed until reset; runtime summary separately reports latest state'
   other_bytes=sum(p.stat().st_size for p in output.parent.rglob('*') if p.is_file() and not p.is_symlink() and not p.is_relative_to(output))
   limits=storage(spec)
   remaining=limits['chain_max_bytes']-other_bytes-limits['terminal_reserve_bytes']
   if remaining<=0:raise (RuntimeError if spec.get('recovery_course') else TimeoutError)('evaluation output budget exhausted')
   if spec.get('motion_training'):
    # Every checkpoint retains complete scoring inputs; full action/critic
    # traces are limited to source, wiring and terminal evidence.
    full=output.name in ('source_eval','smoke') or (checkpoint and Path(checkpoint).stem==f"model_update_{spec['budget']['formal_updates']}")
    from .motion_tracking_reward import QUALITY_VERSIONS
    quality=spec['reward']['version'] in QUALITY_VERSIONS
    if quality and output.name in ('source_eval','smoke'):full=row['command'][:2]==[0.,0.]
    payload['storage_profile']='full_key_evidence' if full else 'compact_motion_scoring_proof'
    if spec.get('motion_course'):full=False;payload['storage_profile']='compact_course_scoring_proof'
    if not full:
     original=arrays;arrays=compact_motion_evidence(spec,original,output.name)
     if quality and not spec.get('motion_course') and output.name in ('source_eval','smoke'):
      arrays.update({k:original[k] for k in ('control','actor_input') if k in original})
     payload['storage_profile']='compact_motion_scoring_with_wheel_readback_v2'
   if spec.get('recovery_course'):
    c=spec['recovery_course'];phase=output.name
    full=bool(spec.get('student_temporal_regularization')) or phase=='smoke' or (phase=='source_eval' and c['first_recipe_block']) or c['state']['stagnant']>=2 or any(r['accepted'] for r in payload['per_env'])
    payload['storage_profile']='full_key_evidence' if full else 'compact_scoring_proof'
    if spec.get('motion_course'):full=False;payload['storage_profile']='compact_course_scoring_proof'
    if not full:
     keep={'state','posture','controller','first_episode_valid','policy_time_s','student_task_events'}
     if spec['student_training']['version']=='student_recovery_ppo_standing_mirror_huber_v11':keep.update(('control','actor_input','previous_targets','mean','raw_action'))
     arrays={k:v for k,v in arrays.items() if k in keep}
   if spec.get('recovery_domain_randomization') and output.name=='formal':
    keep={'state','posture','control','controller','first_episode_valid','policy_time_s','student_task_events','dr_parameters_pre','dr_episode_pre'}
    arrays={k:v for k,v in arrays.items() if k in keep}
    payload['storage_profile']='DR compact full-scoring proof; policy-sampled torque'
   payload['performance']={**perf,'environment_creations':pool.creations,'environment_reuses':pool.reuses}
   payload['timing']={'environment_acquire_wall_s':setup_wall,'rollout_scoring_transfer_wall_s':rollout_wall}
   if (spec.get('motion_highspeed_standing') or spec.get('motion_standing_delay')) and output.name=='formal':
    # All8 environments were scored above. Retain one complete diagnostic
    # trajectory and every per-env gate/metric; do not claim8 raw trajectories.
    arrays={k:v[:,:1].copy() if v.ndim>=2 and v.shape[1]==8 else v for k,v in arrays.items()}
    payload['storage_profile']='eight_env_scores_one_env_diagnostics_v1'
    payload['diagnostic_env_ids']=[0];payload['scored_env_count']=8
   if spec.get('motion_robustness') and output.name=='formal':
    from .motion_robustness import diagnostic_storage
    if spec.get('motion_decoupled'):
     from .motion_decoupled import diagnostic_storage
     if spec['motion_decoupled'].get('fixed_episode_height'):
      from .motion_fixed_height import diagnostic_storage
    arrays=diagnostic_storage(payload,arrays)
    payload['storage_profile']='focus_full_retention_tail_v1' if spec.get('motion_decoupled') else 'eight_env_metrics_one_env_trace_v1'
    if spec.get('motion_decoupled',{}).get('fixed_episode_height'):payload['storage_profile']='fixed_height_bounded_diagnostics_v1'
   path=atomic_result(output,offset+i,payload,arrays,remaining,artifact_cap_bytes=limits['evaluation_max_bytes'])
   record={'case':row,'path':path.name,'sha256':sha(path),'per_env':payload['per_env']}
   if spec.get('motion_parallel'):record['protocol']=payload['protocol']
   records.append(record)
   condition_states[i].update(status='COMPLETED',result=path.name);write(output/f'evaluation_matrix_{offset:02}.json',condition_states)
  except BaseException as error:
   partial=getattr(error,'student_partial_arrays',None)
   pre=getattr(error,'student_pre_action_arrays',None)
   arrays=partial if partial is not None else pre
   if arrays:
    path=output/f'evaluation_fault_{offset+i:02}.npz'
    saved_fault=secondary(error,'fault NPZ save',lambda:atomic_npz(path,arrays))
    condition_states[i].update(status='INCOMPLETE',fault_trace=path.name if saved_fault is not None else None,pre_action_only=partial is None)
    secondary(error,'fault matrix save',lambda:write(output/f'evaluation_matrix_{offset:02}.json',condition_states))
   raise
  finally:
   primary=sys.exception()
   try:
    if env is not None and (primary is not None or not pool.enabled or i==len(cases)-1):
     if primary is not None:secondary(primary,'evaluation env.close',pool.close)
     else:pool.close()
   finally:
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
 return records

def save_checkpoint(output,spec,actor,critic,ppo,source,update,course_state=None,dr_context=None):
 path=output/f'model_update_{update}.pt'
 data={'interface_version':spec['interface_version'],'role':'RecoveryStudent','actor':actor.state_dict(),'critic':critic.state_dict(),
  'optimizer':ppo.optimizer.state_dict(),'update':update,'recipe_version':spec['student_training']['version'],'training_recipe':spec['student_training'],
  'source_sha256':spec['source_sha256'],'dataset_sha256':source['dataset_sha256'],'manifest_sha256':source['manifest_sha256'],
  'asset_sha256':spec['asset_sha256'],'spec_sha256':sha(output/'source_snapshot/spec.json'),
  'source_code_sha256':__import__('native_stand.command_v6.tools.execution_identity',fromlist=['identity']).identity(spec)['location_independent_modules_sha256'],
  'offline_bc_steps':source.get('offline_bc_steps',0),'online_updates_this_stage':update,'BC_coefficient':0.,'fixed_std':True}
 if dr_context is not None:data['recovery_domain_randomization']={'contract':spec['recovery_domain_randomization'],'context':copy.deepcopy(dr_context)}
 if spec.get('student_gpu_resume'):
  data['gpu_migration']=spec['student_gpu_resume']
  data['checkpoint_formal_update_including_parent']=spec['student_gpu_resume']['parent_checkpoint_update']+update
  data['online_updates_including_parent']=spec['student_gpu_resume']['historical_online_updates']+update
 if spec['student_training'].get('network_architecture'):data['network_architecture']=spec['student_training']['network_architecture']
 if spec['student_training'].get('critic_privileged_version'):data['critic_privileged_version']=spec['student_training']['critic_privileged_version']
 if spec['student_training']['version'] in ('student_standing_survival_quality_v14','student_recovery_broad_narrow_ppo_v1','student_recovery_stop_ppo_v2','student_recovery_large_stop_ppo_v1','student_recovery_contact_quality_ppo_v1'):
  from .resume import rng_state
  data['training_rng']=rng_state()
  data['recipe_updates']=spec.get('student_gpu_resume',{}).get('parent_checkpoint_update',spec.get('student_storage_resume',{}).get('parent_formal_updates',0))+update
 if spec.get('student_temporal_regularization'):data['temporal_regularization']=spec['student_temporal_regularization']
 if spec.get('recovery_course'):
  from .resume import rng_state
  c=spec['recovery_course'];base=c['state']['cumulative_updates']
  if output.name=='formal':base+=json.loads((Path('/home/kellen/wheel_leg_native_train/configs/experiments')/spec['experiment']/'budget.json').read_text())['used']['smoke']
  data.update(course_resume_version='student_full_training_resume_v1',training_rng=rng_state(),cumulative_update=base+update,
    recipe_updates=(source.get('recipe_updates',0) if c['resume'] or spec.get('student_temporal_regularization') else 0)+update,course_state=course_state or c['state'],certified_capabilities=(course_state or c['state'])['certified'])
 atomic_checkpoint(path,data)
 reloaded,data=load(path,'RecoveryStudent')
 assert all(torch.equal(v.detach().cpu(),reloaded.state_dict()[k]) for k,v in actor.state_dict().items())
 candidate={'checkpoint':str(path.resolve()),'sha256':sha(path),'update':update}
 if spec.get('artifacts',{}).get('checkpoint_retention'):
  inventory_path=output/'checkpoint_inventory.json'
  inventory=json.loads(inventory_path.read_text()) if inventory_path.exists() else []
  inventory=[row for row in inventory if row['path']!=candidate['checkpoint']]
  inventory.append({'path':candidate['checkpoint'],'sha256':candidate['sha256'],'update':update})
  write(inventory_path,inventory)
 return candidate

def execute(args,spec,plan):
 from native_stand.command_v6.train import record_run_configuration,send_handshake,install_stop_handlers
 from native_stand.command_v6.ledger import Ledger
 from native_stand.command_v6.tools.health_check import own_process_start
 output=args.output_dir;phase=args.phase;attempt=args.attempt_id
 from native_stand.command_v6.domain.execution_plan import samples_per_update
 training_envs=64 if phase=='source_eval' else spec['ppo']['envs']
 batch_samples=0 if phase=='source_eval' else samples_per_update(spec)
 large_stop=spec['student_training']['version'] in ('student_recovery_large_stop_ppo_v1','student_recovery_contact_quality_ppo_v1')
 book=json.loads(args.spec.with_name('budget.json').read_text());reservation=book['attempts'].get(attempt)
 if reservation is None or reservation['status']!='RUNNING' or reservation['output_dir']!=str(output.resolve()):raise ValueError('public student exact reservation required')
 from native_stand.command_v6.domain.execution_limits import ExecutionLimits
 started=time.monotonic();deadline=ExecutionLimits.resolve(spec,phase).deadline(reservation['started_unix_s']);app=env=None;failure=None;status='FAILED'
 progress={'status':'STARTING','phase':phase,'stage':'manifest','attempt_id':attempt,'pid':os.getpid(),'output_dir':str(output.resolve()),'actual_updates':0,'samples':0,'evaluations':[],'actor_parameters_changed':False,'checkpoint_roundtrip_verified':False,'effective_actor_samples':0}
 sent=False
 initial_cpu_rng=torch.get_rng_state().clone()
 initial_cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
 tail=[]
 def stopped():
  from native_stand.command_v6 import train
  task_stop=spec.get('recovery_course',{}).get('task_manifest')
  return train.STOP or time.time()>=deadline-60 or bool(task_stop and Path(task_stop).with_name('STOP').exists())
 def storage_guard():
  size=sum(p.stat().st_size for p in output.parent.rglob('*') if p.is_file() and not p.is_symlink())
  if size>storage(spec)['chain_max_bytes']-storage(spec)['terminal_reserve_bytes']:raise (RuntimeError if spec.get('recovery_course') else TimeoutError)('chain output budget reserved tail reached; stop without deleting history')
  if spec.get('recovery_course'):
   import shutil
   if shutil.disk_usage(output).free<storage(spec)['terminal_reserve_bytes']:raise RuntimeError('course filesystem reserve exhausted; save and stop, no cleanup or retry')
 def handshake():
  nonlocal sent
  if not sent and args.handshake_fd is not None:
   send_handshake(args.handshake_fd,{'status':'RUNNING','attempt_id':attempt,'trainer_pid':os.getpid(),'actual_updates':progress['actual_updates'],'output_dir':str(output.resolve())});sent=True
 try:
  output.mkdir(parents=True,exist_ok=False);snapshot=record_run_configuration(output,args.spec)
  write(output/'manifest.json',{'attempt_id':attempt,'phase':phase,'pid':os.getpid(),'process_started_unix_s':own_process_start(os.getpid()),'spec_sha256':sha(args.spec),'source_sha256':spec['source_sha256'],'asset_sha256':spec['asset_sha256'],
   'interface_version':spec['interface_version'],'recipe_version':spec['student_training']['version'],'source_snapshot_sha256':snapshot,'execution_identity':__import__('native_stand.command_v6.tools.execution_identity',fromlist=['reference']).reference(output),'simulated_envs':64 if phase=='source_eval' else training_envs,'scored_envs':8,'hapi_session':args.hapi_session})
  write(output/'progress.json',progress)
  import sys
  sys.path.insert(0,spec['asset_module_root'])
  from isaaclab.app import AppLauncher
  app=AppLauncher({'headless':True,'device':'cuda:0','multi_gpu':False}).app
  install_stop_handlers();torch.manual_seed(spec['seed'])
  actor,source=load(spec['source_checkpoint'],'RecoveryStudent');actor=actor.to('cuda:0')
  source_actor={k:v.detach().clone() for k,v in actor.state_dict().items()};fixed_std=actor.log_std.detach().clone()
  cases=spec['student_evaluation']['cases'];students=[r for r in cases if r['kind'] in ('student','student_suffix')]
  dr_context=None
  if spec.get('recovery_domain_randomization'):
   from .recovery_dr import initial_state
   dr_context={'state':initial_state(),'stream':None}
   if phase=='smoke':dr_context['state']['level']=3  # wiring includes friction + PD, formal starts fresh R1
  if phase=='source_eval':
   progress['stage']='source_evaluation';write(output/'progress.json',progress)
   teachers=teacher_models(spec,'cuda:0') if spec['student_training'].get('source_teacher_baseline',True) else None
   records=evaluate_cases(spec,output,actor,cases,stop=stopped,teachers=teachers,checkpoint=spec['source_checkpoint'])
   progress['evaluations']=records
   if spec['student_training']['version']=='student_recovery_contact_quality_ppo_v1':
    from .contact_readback import verify_records
    readback=verify_records(output,records);write(output/'contact_runtime_readback.json',readback)
    progress['contact_readback_status']=readback['status']
   if dr_context is not None:
    from .recovery_dr_training import verify_records as verify_dr_records
    write(output/'dr_source_readback.json',verify_dr_records(output,records))
   standing=spec['student_training']['training_family']=='motion_native_normal'
   teacher_records=[r for r in records if r['case']['kind']==('teacher_normal' if standing else 'teacher_combination')]
   student_records=[r for r in records if (r['case']['case_id'].startswith('current_') if spec.get('recovery_course') else r['case']['kind']=='student' and r['case']['family']==spec['student_training']['training_family'])]
   if spec['student_training'].get('source_teacher_baseline',True):
    if len(teacher_records)!=1 or not all(r['accepted'] for r in teacher_records[0]['per_env']):raise ValueError('teacher shared-environment baseline failed; no PPO')
   elif not student_records or any(not e.get('initial_valid') or e['termination_causes'][0] or e['termination_causes'][1] or e['termination_causes'][5] for r in student_records for e in r['per_env']):raise ValueError('v14 source physical/initialization evidence invalid')
   candidate={'checkpoint':spec['source_checkpoint'],'sha256':spec['source_sha256'],'update':0,'per_env':[e for r in student_records for e in r['per_env']],'evaluations':student_records,'capability_scope':'normal standing only; Swing not certified' if standing else 'Swing autonomous recovery'}
   if spec['student_training']['version']=='student_standing_survival_quality_v14':candidate['standing_survival_quality']=True
   if spec.get('student_temporal_regularization'):candidate['evaluations']=records
   if spec.get('recovery_course'):
    c=spec['recovery_course'];candidate['teacher_prefix_intervals']=c['state']['prefix'] if c['state']['stage']!='A' else 0
    if candidate['teacher_prefix_intervals']>0:candidate['capability_scope']='auxiliary dynamic suffix only; not independent Swing recovery'
   if large_stop:candidate['stop_quality_selection']=True
   write(output/'candidate_registry.json',[candidate]);write(output/'candidate_selection.json',choose([candidate]))
   progress['engineering_valid']=True;progress['source_behavior_passed']=all(r['accepted'] for r in candidate['per_env'])
  else:
   progress['stage']='training_environment_init';write(output/'progress.json',progress)
   from .recovery_dr_training import training_environment
   env,obs=training_environment(spec,training_envs,dr_context)
   from native_stand.command_v6.runtime.configure import verify_asset_import
   from native_stand.command_v6.runtime.control_readback import verify,verify_ground
   write(output/'runtime_readback.json',{'asset':verify_asset_import(spec),'control':verify(env),'ground':verify_ground(env,spec),'active_indices':env.ids,'joint_names':env.robot.joint_names,'interface':spec['student_interface']})
   ppo=assemble(spec['source_checkpoint'],obs,recipe=spec['student_training']);actor=ppo.actor;critic=ppo.critic;ppo.record_gradient_norms=True
   assert not ppo.optimizer.state
   if dr_context is not None:
    progress['DR_source_critic_loaded_fresh_Adam']=True
    write(output/'dr_runtime_readback.json',env.recovery_dr.summary())
   course=copy.deepcopy(spec.get('recovery_course'));course_state=copy.deepcopy(course['state']) if course else None
   recipe_offset=0
   if spec.get('student_gpu_resume'):
    from .gpu_resume import restore
    progress['resume']=restore(ppo,source,spec);recipe_offset=spec['student_gpu_resume']['parent_checkpoint_update']
    progress['physical_resets']=[{'reason':'GPU migration; real environment reset; full optimizer/RNG resumed','PhysX_state_restored':False}]
   if spec.get('student_storage_resume'):
    from .storage_resume import restore
    progress['resume']=restore(ppo,source,spec)
    recipe_offset=spec['student_storage_resume']['parent_formal_updates']
   if course:
    ppo.critic_control_only=True
    if spec.get('student_temporal_regularization'):
     from .temporal_ab_io import verify_source
     verify_source(spec)
    if course['resume'] or spec.get('student_temporal_regularization'):
     from .resume import restore
     progress['resume']=restore(ppo,source,spec['student_training']);recipe_offset=source['recipe_updates']
     obs,_=env.reset();obs=observations(env,obs)
     progress.setdefault('physical_resets',[]).append({'reason':'process_restart_full_optimizer_resume','PhysX_state_restored':False})
   if spec.get('student_temporal_regularization'):
    from .temporal import Regularizer
    ppo.temporal_regularizer=Regularizer(spec['student_temporal_regularization'],env.device)
   restored_full=bool(spec.get('student_gpu_resume') or spec.get('student_storage_resume') or course and (course['resume'] or spec.get('student_temporal_regularization')))
   progress['optimizer_initial']={'source_critic_loaded':restored_full or dr_context is not None,'source_Adam_loaded':restored_full,'source_RNG_loaded':restored_full and bool(source.get('training_rng')),'state_count':len(ppo.optimizer.state),'BC_coefficient':0.}
   source_evaluation=Path(spec.get('student_storage_resume',{}).get('source_eval_dir',output.parent/'source_eval'))
   candidates=json.loads((source_evaluation/'candidate_registry.json').read_text())
   if course:
    initial_checkpoint=save_checkpoint(output,spec,actor,critic,ppo,source,0,course_state)
    candidates[0]={**candidates[0],**initial_checkpoint,'behavior_evidence_source_checkpoint':spec['source_checkpoint'],'actor_identical_to_source':True}
    write(output/'candidate_registry.json',candidates)
   index=0
   for update in range(1,spec['budget'][phase+'_updates']+1):
    if stopped():status='STOPPED_BOUNDED';break
    v14=spec['student_training']['version']=='student_standing_survival_quality_v14'
    recovery64=spec['student_training']['version'] in ('student_recovery_broad_narrow_ppo_v1','student_recovery_stop_ppo_v2','student_recovery_large_stop_ppo_v1','student_recovery_contact_quality_ppo_v1')
    from .performance import settings,RolloutCounts
    perf=settings(spec);rollout_started=time.perf_counter()
    from .performance import recovery_training_v2,recovery_step_flags,fixed_masked_counts
    recovery_fast=recovery64 and recovery_training_v2(spec)
    gpu_counts=RolloutCounts(128,env.num_envs,env.device) if perf['training_gpu_statistics'] else None
    storage_guard();progress['stage']='rollout';reward_sum=torch.zeros(16 if spec['student_training']['version']=='student_recovery_contact_quality_ppo_v1' else 13 if large_stop else 11 if recovery64 else 10 if v14 else 6,device=env.device);reward_weighted=torch.zeros_like(reward_sum);reward_dt=torch.zeros_like(reward_sum);events=torch.zeros(3,device=env.device)
    pre_entry_cost_sum=torch.zeros(3,device=env.device)
    alignment_sum=torch.zeros((3,4),device=env.device)
    motion_mirror_sum=torch.zeros((3,2),device=env.device);mirror_abs_sum=torch.zeros(2,device=env.device)
    timeout_count=torch.zeros((),device=env.device,dtype=torch.long)
    controlled_count=nonzero=episodes=0;episode_durations=[];causes=torch.zeros(6,device=env.device);entry_count=complete_count=success_count=0;short_count=0;target_change=torch.zeros(6,device=env.device);clips=torch.zeros(6,device=env.device);limits=torch.zeros(6,device=env.device);captured={}
    phase_counts=torch.zeros(4,device=env.device)
    owner_counts=torch.zeros(4,device=env.device,dtype=torch.long);group_counts=torch.zeros(4,device=env.device,dtype=torch.long);smooth_sum=torch.zeros(3,device=env.device);handoff_count=0
    contact_cost_sum=torch.zeros(3,device=env.device)
    temporal_rollout={k:[] for k in ('observations','student','entered','episode','time_s','done')}
    family_counts=torch.zeros(2,device=env.device,dtype=torch.long)
    from .course_statistics import WheelStatistics
    wheel_stats=WheelStatistics(env.num_envs,env.device)
    contact_metrics=None
    if spec['student_training']['version']=='student_recovery_contact_quality_ppo_v1':
     from .standing_phase_metrics import PhaseMetrics
     contact_metrics=PhaseMetrics(env.device)
    selected=update in spec['student_training']['selected_rollouts_'+phase]
    actor_before={k:v.clone() for k,v in actor.state_dict().items()};critic_before={k:v.clone() for k,v in critic.state_dict().items()}
    with torch.inference_mode():
     for step in range(128):
      if stopped():raise TimeoutError('stage wall/user budget during rollout')
      obs=observations(env,obs);valid=obs[KEY].squeeze(-1).clone();preobs=obs['student_actor'].clone();raw=ppo.act(obs);mean=actor.output_mean.clone();logp=actor.get_output_log_prob(raw).clone()
      if ppo.temporal_regularizer is not None:
       for k,v in {'observations':preobs,'student':valid,'entered':env.rise_hold.entry>=0,'episode':env.reset_count,'time_s':env.episode_length_buf.to(torch.float64)*.01}.items():temporal_rollout[k].append(v.clone())
      normal_pre=env.standing_rehearsal_pose_mask[env.initial_pose_index].clone()
      family_counts+=torch.stack(((valid&~normal_pre).sum(),(valid&normal_pre).sum()))
      executing=env.compose_training_action(raw) if hasattr(env,'compose_training_action') else raw.clone()
      if recovery64:executing[~valid]=0. # Passive initialization has no student target request.
      group_pre=env.recovery_dr.group_ids.clone() if dr_context is not None else env.course_group_index.clone() if hasattr(env,'course_group_index') else torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
      nextobs,rewards,terminated,truncated,extras=env.step(executing)
      owner=env.course_owner_snapshot if hasattr(env,'course_owner_snapshot') else torch.where(valid,3,0)
      owner_counts+=torch.bincount(owner,minlength=4)
      if hasattr(env,'course_handoff_snapshot') and gpu_counts is None:handoff_count+=int(env.course_handoff_snapshot.sum())
      group_counts+=(fixed_masked_counts(group_pre,valid,4) if recovery_fast else torch.bincount(group_pre[valid],minlength=4))
      timeout_count+=truncated.sum()
      s=env.recovery_snapshot
      # Recovery v2: transfer five original predicates once, before storage/PPO.
      step_flags=recovery_step_flags(s,rewards,valid,env.student_actor_valid,truncated).detach().cpu().tolist() if recovery_fast else None
      if (not step_flags[0] if recovery_fast else bool(truncated.any())) and not v14:raise RuntimeError('unexpected engineering timeout; no time-out bootstrap or task success')
      def tail_copy(t):return t[:8].detach().clone() if perf['failure_tail_on_device'] else t[:8].detach().cpu().numpy()
      tail.append({'actor_input':tail_copy(preobs),'raw_action':tail_copy(raw),'old_log_probability':tail_copy(logp),'state':tail_copy(s),'posture':tail_copy(env.standing_snapshot),'control':tail_copy(env.recovery_control_snapshot),'reward':tail_copy(rewards)})
      if len(tail)>32:tail.pop(0)
      if (not all(step_flags[1:4]) if recovery_fast else (s[:,4].bool().any() or s[:,30].bool().any() or not torch.isfinite(rewards).all())):
       from .fault_evidence import capture,describe
       original=RuntimeError('runtime fault/initialization rejected; see fault_all_environments.json; no automatic retry')
       def save_fault():
        fault_data=capture(s,rewards,getattr(env,'student_fault_components',{}),getattr(env,'student_fault_joint_data',None))
        if fault_data is not None:
         atomic_npz(output/'fault_all_environments.npz',{k:v.detach().cpu().numpy() for k,v in fault_data.items()})
         masks={k:fault_data[k].detach().cpu().tolist() for k in ('runtime_fault','initial_rejected','nonfinite_reward')}
         write(output/'fault_all_environments.json',{**describe(fault_data),'phase':phase,'completed_updates':progress['actual_updates'],'next_update':update,'rollout_step':step,'total_environments':env.num_envs,'env_ids':fault_data['env_ids'].tolist(),'trigger_masks':masks,'schema':'recovery_policy_trace_v1','evidence':'fault_all_environments.npz','joint_names':env.robot.joint_names,'joint_limit_epsilon_rad':.001,'safety_checks_unchanged':True})
       secondary(original,'full environment fault save',save_fault)
       raise original
      if (not step_flags[4] if recovery_fast else not torch.equal(valid,env.student_actor_valid)):raise RuntimeError('pre-action owner mask differs')
      nextobs=observations(env,nextobs);dones=terminated|truncated;extras={k:v for k,v in extras.items() if k!='time_outs'}
      if ppo.temporal_regularizer is not None:temporal_rollout['done'].append(dones.clone())
      storage_rewards=rewards
      if v14 and truncated.any():
       from .standing_survival_task import bootstrap_reward
       if env.student_terminal_critic is None:raise RuntimeError('v14 reset-before-terminal Critic evidence missing')
       terminal_values=critic(env.student_terminal_critic).squeeze(-1)
       storage_rewards=bootstrap_reward(rewards,terminated,truncated,terminal_values,gamma=ppo.gamma)
      ppo.process_env_step(nextobs,storage_rewards,dones,extras);obs=nextobs
      phase_id=torch.where(s[:,1]<=0,0,torch.where(s[:,1]<2.,1,torch.where(s[:,1]<5.,2,3)))
      phase_counts+=(fixed_masked_counts(phase_id,valid,4) if recovery_fast else torch.bincount(phase_id[valid],minlength=4))
      rr=env.student_reward_result;reward_sum+=rr['raw'].sum(0);reward_weighted+=rr['density' if v14 else 'weighted_density'].sum(0);reward_dt+=rr['dt' if v14 else 'continuous_dt'].sum(0);events+=rr['events'].sum(0)
      if not v14 and not recovery64:
       pre_entry_cost_sum+=torch.stack([rr[k].sum() for k in ('pre_entry_angular_raw','pre_entry_angular_density','pre_entry_angular_dt')])
       smooth_sum+=torch.stack([rr[k].sum() for k in ('hold_wheel_smooth_raw','hold_wheel_smooth_density','hold_wheel_smooth_dt')])
       contact_cost_sum+=torch.stack([rr[k].sum() for k in ('hold_contact_deficit_raw','hold_contact_deficit_density','hold_contact_deficit_dt')])
       if 'alignment_raw' in rr:alignment_sum+=torch.stack([rr[k].sum(0) for k in ('alignment_raw','alignment_density','alignment_dt')])
       motion_mirror_sum+=torch.stack([rr[k].sum(0) for k in ('motion_mirror_raw','motion_mirror_weighted_density','motion_mirror_dt')])
      q=env.recovery_control_snapshot[:,:6]
      measured_mirror=torch.stack((q[:,0]+q[:,3],q[:,1]-q[:,4]),-1)
      mirror_abs_sum+=(measured_mirror.abs()*valid[:,None]).sum(0)
      causes+=env.recovery_cause_snapshot[dones].sum(0);ev=env.student_event_snapshot
      if gpu_counts is not None:
       gpu_counts.observe(valid,rewards,dones,s,ev,env.standing_snapshot,getattr(env,'course_handoff_snapshot',None))
      else:
       controlled_count+=int(valid.sum());nonzero+=int((rewards!=0).sum());episodes+=int(dones.sum());episode_durations+=s[dones,0].tolist()
       entry_count+=int(ev[:,0].sum());complete_count+=int(ev[:,1].sum());success_count+=int(ev[:,2].sum())
       short_count+=int(((env.standing_snapshot[:,2:4]<=.34).all(-1)&valid).sum())
      actual_targets=torch.zeros_like(env.student_previous_physical);actual_targets[:,[0,1,3,4]]=env.recovery_control_snapshot[:,28:32];actual_targets[:,[2,5]]=env.recovery_control_snapshot[:,[20,23]]*100
      target_change+=((actual_targets-env.student_previous_physical).abs()*valid[:,None]).sum(0)
      from native_stand.command_v6.domain.trace_schema import CONTROL
      control=env.recovery_control_snapshot
      wheel_stats.observe(actual_targets[:,[2,5]],env.student_previous_physical[:,[2,5]],control[:,CONTROL['requested_torque']][:,[2,5]],control[:,CONTROL['physical_clipped_torque']][:,[2,5]],control[:,CONTROL['applied_torque']][:,[2,5]],valid,dones)
      if contact_metrics is not None:
       contact_metrics.observe(elapsed=env.contact_elapsed_post,entered=env.contact_entered_post,controlled=valid,contacts=s[:,10:12],angles=env.standing_snapshot[:,:2],velocity=s[:,7:10],previous=env.student_previous_physical,executed=actual_targets,requested_torque=control[:,CONTROL['requested_torque']],clipped_torque=control[:,CONTROL['physical_clipped_torque']],applied_torque=control[:,CONTROL['applied_torque']],measurement=env.contact_measurement_post)
      clips+=(raw.abs()>1).float().sum(0);limits+=(env.effort.abs()>=env.effort.new_tensor([40,40,5.5,40,40,5.5])-1e-4).float().sum(0)
      if selected:
       for k in ('hold_contact_deficit_raw','hold_contact_deficit_density','hold_contact_deficit_dt') if not recovery64 else ():captured.setdefault(k,[]).append(rr[k].detach().cpu().numpy())
       for k in ('hold_wheel_smooth_raw','hold_wheel_smooth_density','hold_wheel_smooth_dt') if not recovery64 else ():captured.setdefault(k,[]).append(rr[k].detach().cpu().numpy())
       if hasattr(env,'course_owner_snapshot'):
        for k,v in {'controller':env.course_owner_snapshot,'absolute_control_intervals':env.course_absolute_snapshot,'teacher_intervals':env.course_teacher_snapshot,'handoff_event':env.course_handoff_snapshot,'handoff_state':env.course_handoff_state_snapshot}.items():captured.setdefault(k,[]).append(v.detach().cpu().numpy())
       for k in ('motion_mirror_raw','motion_mirror_weighted_density','motion_mirror_dt','mirror_error_rad') if not recovery64 else ():captured.setdefault(k,[]).append(rr[k].detach().cpu().numpy())
       for k in ('pre_entry_angular_raw','pre_entry_angular_density','pre_entry_angular_dt') if not recovery64 else ():captured.setdefault(k,[]).append(rr[k].detach().cpu().numpy())
       for k,v in {'actor_input':preobs,'critic_input':ppo.storage.observations['student_critic'][step].clone(),'actor_valid':valid,'raw_action':raw,'mean':mean,'old_log_probability':logp,'reward':rewards,'reward_raw':rr['raw'],'reward_weighted_density':rr['density' if v14 else 'weighted_density'],'reward_dt':rr['dt' if v14 else 'continuous_dt'],'reward_events':rr['events'],'state':s,'posture':env.standing_snapshot,'control':env.recovery_control_snapshot,'events':ev,'done':dones}.items():captured.setdefault(k,[]).append(v.detach().cpu().numpy())
     ppo.compute_returns(obs)
    if gpu_counts is not None:
     counts,episode_durations=gpu_counts.export()
     controlled_count,nonzero,episodes,entry_count,complete_count,success_count,short_count,handoff_count=counts
    rollout_wall_s=time.perf_counter()-rollout_started
    if ppo.temporal_regularizer is not None:
     from .temporal import pairs
     previous,current=pairs(**{k:torch.stack(v) for k,v in temporal_rollout.items()})
     ppo.temporal_regularizer.set_pairs(previous,current)
    ret=ppo.storage.returns;val=ppo.storage.values;adv=ppo.storage.advantages
    if selected:
     captured['storage_return']=[ret.detach().cpu().numpy()]
     captured['storage_value']=[val.detach().cpu().numpy()]
     captured['storage_advantage']=[adv.detach().cpu().numpy()]
    distributions={name:{'mean':float(t.mean()),'std':float(t.std()),'min':float(t.min()),'max':float(t.max())} for name,t in [('returns',ret),('values',val),('advantages',adv)]}
    ppo.actor_update_enabled=bool(course and course['resume']) or recipe_offset+update>spec['student_training']['warmup_updates_per_phase']
    from native_stand.command_v6.tools.optimizer_capture import capture as capture_optimizer
    capture_optimizer(output,spec,phase,update,actor,critic,ppo.optimizer)
    progress['stage']='optimizer';ppo_started=time.perf_counter();loss=ppo.update();ppo_wall_s=time.perf_counter()-ppo_started
    actor_changed=any(not torch.equal(v,actor.state_dict()[k]) for k,v in actor_before.items());critic_changed=any(not torch.equal(v,critic.state_dict()[k]) for k,v in critic_before.items())
    if not ppo.actor_update_enabled and actor_changed:raise RuntimeError('warmup Actor changed')
    if not torch.equal(fixed_std,actor.log_std):raise RuntimeError('fixed std changed')
    progress['actual_updates']=update;progress['samples']=update*batch_samples;progress['effective_actor_samples']+=ppo.mask_summary['effective_actor_update_samples']
    Ledger=__import__('native_stand.command_v6.ledger',fromlist=['Ledger']).Ledger
    Ledger(args.spec.with_name('budget.json')).charge(attempt,output)
    if not all(torch.isfinite(p).all() for model in (actor,critic) for p in model.parameters()) or not all(np.isfinite(v) for v in loss.values()):raise RuntimeError('nonfinite PPO update')
    metric={'update':update,'raw_samples':batch_samples,'controlled_samples':controlled_count,'nonzero_rewards':nonzero,'episode_completed':episodes,'episode_duration_mean_s':sum(episode_durations)/len(episode_durations) if episode_durations else None,'episode_duration_max_s':max(episode_durations,default=None),'termination_counts':causes.tolist(),'dual_short_sample_fraction':short_count/max(controlled_count,1),'entry_events':entry_count,'hold_window_complete_events':complete_count,'certificate_success_events':success_count,'reward_raw_mean':(reward_sum/batch_samples).tolist(),'reward_weighted_density_mean':(reward_weighted/batch_samples).tolist(),'reward_dt_mean':(reward_dt/batch_samples).tolist(),'reward_events_sum':events.tolist(),'loss':loss,'distributions':distributions,'actor_update_scope':ppo.mask_summary,'actor_changed_this_update':actor_changed,'critic_changed_this_update':critic_changed,'std_unchanged':True,'BC_loss':0.,'BC_gradient_norm':None,'BC_updates':0,'initialization_fraction':1-controlled_count/batch_samples,'controlled_family':'swing_candidate_from_historical_r1','phase_samples_pre_entry_hold0_2_hold2_5_hold5_10':phase_counts.tolist(),'max_hold_prefix_s':float(env.student_hold_prefix_snapshot.max())*.01,'action_clip_per_channel':(clips/batch_samples).tolist(),'actual_target_abs_delta_per_channel_rad_or_rad_s':(target_change/max(controlled_count,1)).tolist(),'torque_limit_per_channel':(limits/batch_samples).tolist(),'wall_s':time.monotonic()-started}
    if dr_context is not None:
     from .recovery_dr_training import capture as capture_dr
     capture_dr(env,dr_context)
     metric['domain_randomization']=env.recovery_dr.summary()
     metric['DR_controlled_samples_by_group_nominal_previous_current']=group_counts[:3].tolist()
     metric['DR_effective_actor_samples_by_group']=(group_counts[:3] if ppo.actor_update_enabled else group_counts[:3]*0).tolist()
    metric['wheel_execution_statistics']=wheel_stats.result()
    if contact_metrics is not None:metric['standing_phase_metrics']=contact_metrics.result()
    metric['timing']={'rollout_and_statistics_wall_s':rollout_wall_s,'ppo_update_wall_s':ppo_wall_s,'scope':'host elapsed incl natural synchronization; not CUDA kernel exclusive timing'}
    metric['performance']=perf
    metric['course_control_counts_initial_B50_B250_student']=owner_counts.tolist()
    metric['course_real_handoffs']=handoff_count
    metric['course_student_samples_by_group']=dict(zip(getattr(env,'group_names',['normal']),group_counts.tolist()))
    metric['hold_wheel_smooth_raw_density_dt_mean']=(smooth_sum/batch_samples).tolist()
    metric['hold_contact_deficit_raw_density_dt_mean']=(contact_cost_sum/batch_samples).tolist()
    metric['pre_entry_angular_cost_raw_density_dt_mean']=(pre_entry_cost_sum/batch_samples).tolist()
    metric['motion_mirror_raw_density_dt_mean']=(motion_mirror_sum/batch_samples).tolist()
    if recovery64:
     metric['reward_fields']=list(env.student_reward_result['fields'])
     metric['critic_privileged_version']=spec['student_training']['critic_privileged_version']
     metric['optimizer_learning_rate']=[g['lr'] for g in ppo.optimizer.param_groups]
    if v14:
     from .standing_survival_reward import FIELDS
     metric['reward_fields']=list(FIELDS)
     metric['controlled_survival_prefix_s']=metric['max_hold_prefix_s']
     metric['max_hold_prefix_s']=None
     metric['timeout_bootstrap']='gamma times pre-reset terminal Critic value; raw reward summaries exclude bootstrap'
     metric['normal_timeout_count']=int(timeout_count)
     for key in ('hold_wheel_smooth_raw_density_dt_mean','hold_contact_deficit_raw_density_dt_mean','pre_entry_angular_cost_raw_density_dt_mean','motion_mirror_raw_density_dt_mean'):metric[key]=None
    metric['actual_q_mirror_abs_mean_rad']=(mirror_abs_sum/max(controlled_count,1)).tolist()
    if 'alignment_raw' in env.student_reward_result:metric['alignment_raw_density_dt_mean']=(alignment_sum/batch_samples).detach().cpu().tolist()
    if spec.get('student_gpu_resume'):
     metric['checkpoint_formal_update_including_parent']=recipe_offset+update
     metric['online_updates_including_parent']=spec['student_gpu_resume']['historical_online_updates']+update
    metric['controlled_family']=spec['student_training']['training_family']
    if spec.get('student_storage_resume'):
     metric['formal_updates_including_parent']=recipe_offset+update
     metric['online_updates_including_parent_smoke']=spec['student_storage_resume']['historical_online_updates']+update
    metric['controlled_samples_by_family']={'Swing':int(family_counts[0]),'normal_standing':int(family_counts[1])}
    metric['effective_actor_samples_by_family']={k:v if ppo.actor_update_enabled else 0 for k,v in metric['controlled_samples_by_family'].items()}
    if spec['student_training'].get('standing_rehearsal'):metric['controlled_family']='mixed_fixed_32_Swing_32_normal'
    with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric,allow_nan=False)+'\n')
    if selected:atomic_npz(output/f'rollout_update_{update}.npz',{k:np.stack(v)[:,:8] if large_stop and np.stack(v).ndim>1 and np.stack(v).shape[1]==training_envs else np.stack(v) for k,v in captured.items()})
    progress.update(status='RUNNING',actor_parameters_changed=any(not torch.equal(v,actor.state_dict()[k]) for k,v in source_actor.items()),latest_metric=metric)
    write(output/'progress.json',progress);handshake()
    checkpoint_every=spec['student_training'].get('checkpoint_every_updates')
    if phase=='formal' and checkpoint_every and update%checkpoint_every==0 and update not in spec['student_training']['milestones']:
     progress['latest_checkpoint']=save_checkpoint(output,spec,actor,critic,ppo,source,update,course_state,dr_context)
     progress['checkpoint_roundtrip_verified']=True
    milestone=phase=='formal' and (update in spec['student_training']['milestones'] or bool(course and update==spec['budget']['formal_updates'])) or phase=='smoke' and update==spec['budget']['smoke_updates']
    from .recovery_large_training import quick_due
    if quick_due(spec,phase,update,full_assessment=milestone):
     quick_candidate=progress['latest_checkpoint']
     if dr_context is not None:
      from .recovery_dr_training import capture as capture_dr
      capture_dr(env,dr_context)
     env.close();env=None
     quick_records=evaluate_cases(spec,output,actor,students[:1],offset=index,stop=stopped,checkpoint=quick_candidate['checkpoint'],screen_single_env=True);index+=len(quick_records)
     write(output/f'quick_evaluation_update_{update}.json',{'version':'recovery_single_native_screen_v1','checkpoint':quick_candidate,'cases':quick_records,'simulated_envs':1,'scored_envs':1,'condition_count':1,'selection_eligible':False,'accepted_certificate':False,'other_conditions':'UNKNOWN'})
     progress.setdefault('quick_evaluations',[]).append({'update':update,'path':f'quick_evaluation_update_{update}.json'})
     env,obs=training_environment(spec,training_envs,dr_context);progress.setdefault('evaluation_training_resets',[]).append({'after_update':update,'reason':'single-native quick screen','GAE_storage_clear':True})
    if milestone:
     candidate=save_checkpoint(output,spec,actor,critic,ppo,source,update,course_state,dr_context);progress['checkpoint_roundtrip_verified']=True
     if dr_context is not None:
      from .recovery_dr_training import capture as capture_dr
      capture_dr(env,dr_context)
     env.close();env=None
     evaluated_cases=([r for r in students if r['case_id'].startswith('current_') or update==spec['budget']['formal_updates'] or spec.get('student_temporal_regularization')] if course and phase=='formal' else students if phase=='formal' else students[:1])
     if course and phase=='formal' and not spec.get('student_temporal_regularization'):
      evaluated_cases=[{**r,'seed':r['seed']+course['block_index']*100000+update} for r in evaluated_cases]
     records=evaluate_cases(spec,output,actor,evaluated_cases,offset=index,stop=stopped,checkpoint=candidate['checkpoint']);index+=len(records)
     if dr_context is not None and phase=='formal':
      from .recovery_dr_training import assess as assess_dr,observe as observe_dr
      extras=assess_dr(spec,output,actor,students,dr_context,update,index,candidate['checkpoint'],stopped);index+=len(extras)
      report=observe_dr(dr_context,records,extras,update)
      write(output/f'dr_assessment_update_{update}.json',report)
      progress['DR_state']=copy.deepcopy(dr_context['state'])
      progress['evaluations']+=extras
      # Never rewrite an evaluated checkpoint: evaluation binds its exact SHA.
      atomic_checkpoint(output/f'dr_resume_context_update_{update}.pt',{'checkpoint':candidate,'context':copy.deepcopy(dr_context)})
     if large_stop:
      candidate['stop_quality_selection']=True
      if phase=='formal':
       write(output/f'quick_evaluation_update_{update}.json',{'version':'recovery_single_native_screen_v1','source':'full_matrix_includes_native','checkpoint':candidate,'native_result':records[0],'selection_eligible':False,'superseded_by_full_assessment':True,'simulated_envs':64,'scored_envs':8})
     primary_family=spec['student_training']['training_family']
     candidate.update(per_env=[e for r in records if (r['case']['case_id'].startswith('current_') if course else r['case']['family']==primary_family) for e in r['per_env']],evaluations=records,standing_diagnostics=[r for r in records if r['case']['family']=='motion_native_normal'],capability_scope='normal standing only; Swing not certified' if primary_family=='motion_native_normal' else 'Swing autonomous recovery')
     if spec['student_training']['version']=='student_recovery_ppo_standing_mirror_huber_v11':
      from .standing import training_phase_metrics
      assessment=training_phase_metrics(candidate['per_env'],phase)
      if assessment is not None:candidate['standing_metrics']=assessment
      normal=[e for r in records if r['case']['family']=='motion_native_normal' and r['case']['kind']=='student' for e in r['per_env']]
      candidate['standing_certification32_pass']=len(normal)==32 and all(r['accepted'] for r in normal)
     if course:
      candidate['teacher_prefix_intervals']=course_state['prefix'] if course_state['stage']!='A' else 0
      if course_state['stage']=='B' and course_state['prefix']>0:candidate['capability_scope']='auxiliary dynamic suffix only; not independent Swing recovery'
     if course and phase=='formal':
      from .course import summary,observe_confirmation,standing_rank,rank as course_rank
      cm=summary(candidate['per_env']);candidate['course_metrics']=cm
      course_state=observe_confirmation(course_state,cm,course_state['cumulative_updates']+update)
      if phase=='formal':
       candidates.append(candidate)
       best=max(candidates,key=lambda x:(standing_rank if course_state['stage']=='A' else course_rank)(x.get('course_metrics',summary(x['per_env']))))
       write(output/'course_result.json',{'state':course_state,'best':best,'latest':candidate,'metrics':best.get('course_metrics',summary(best['per_env'])),'current_metrics':cm,'cumulative_update':course_state['cumulative_updates']+update})
       write(output/'candidate_selection.json',choose(candidates))
     elif phase=='formal':candidates.append(candidate);write(output/'candidate_selection.json',choose(candidates))
     progress['evaluations']+=records;progress['latest_checkpoint']=candidate;write(output/'candidate_registry.json',candidates);write(output/'progress.json',progress)
     from native_stand.command_v6.tools.checkpoint_lifecycle import prune_owned
     prune_owned(output,spec,attempt_id=attempt)
     if course and phase=='formal' and course['early_end_allowed'] and course_state['confirmation_streak']>=2:
      status='COMPLETED';break
     if dr_context is not None and phase=='formal' and dr_context['state']['finished']:
      progress['DR_stop_reason']=dr_context['state']['stop_reason'];status='COMPLETED' if progress['DR_stop_reason']=='R1_R2_COMPLETE' else 'STOPPED_BOUNDED';break
     if update<spec['budget'][phase+'_updates']:
      # Explicit physical reset/recreation after full-boundary evaluation; no stale obs/GAE.
      env,obs=training_environment(spec,training_envs,dr_context);progress.setdefault('evaluation_training_resets',[]).append({'after_update':update,'method':'shared env close/recreate/reset; PPO storage clear at complete rollout'})
   else:status='COMPLETED'
   if course and phase=='formal' and progress['actual_updates']:
    final_update=progress['actual_updates']
    if not (output/f'model_update_{final_update}.pt').exists():
     latest=save_checkpoint(output,spec,actor,critic,ppo,source,final_update,course_state,dr_context)
     from .course import summary
     result_path=output/'course_result.json'
     result=json.loads(result_path.read_text()) if result_path.exists() else {'state':course_state,'best':candidates[0],'metrics':summary(candidates[0]['per_env']),'current_metrics':summary(candidates[0]['per_env'])}
     result.update(latest={**latest,'behavior_assessed':False},cumulative_update=course_state['cumulative_updates']+final_update)
     write(result_path,result)
   if phase=='formal' and status=='COMPLETED' and not stopped() and not course:
    selection=choose(candidates)
    best=(selection['accepted_candidate'] or selection['best_new'] or selection['best_overall']) if spec['student_training']['version']=='student_standing_survival_quality_v14' else selection['best_overall'];selected_actor,_=load(best['checkpoint'],'RecoveryStudent');selected_actor=selected_actor.to('cuda:0')
    from .training_recipe import final_assessment_cases
    normal=spec['student_training']['training_family']=='motion_native_normal'
    final_cases=final_assessment_cases(spec['student_training'],students)
    if env is not None:env.close();env=None
    final_records=evaluate_cases(spec,output,selected_actor,final_cases,offset=index,stop=stopped,checkpoint=best['checkpoint'])
    swing=[e for r in final_records if r['case']['family']=='swing_candidate_from_historical_r1' for e in r['per_env']]
    write(output/'final_recheck.json',{'checkpoint':best,'conditions':final_records,'accepted':all(e['accepted'] for r in final_records if not normal or r['case']['family']=='motion_native_normal' for e in r['per_env']),'capability_scope':'normal standing only; Swing regression separate' if normal else 'Swing autonomous recovery','Swing_regression_accepted':all(e['accepted'] for e in swing) if normal and swing else None})
   elif phase=='formal':write(output/'final_recheck.json',{'status':'NOT_RUN','reason':'wall/user budget','conditions':[]})
  if phase=='source_eval':status='COMPLETED'
 except BaseException as exc:
  original_traceback=traceback.format_exc()
  if tail and output.exists():
   from .performance import tail_arrays
   secondary(exc,'failure tail save',lambda:atomic_npz(output/'failure_tail.npz',tail_arrays(tail)))
  if isinstance(exc,TimeoutError) and spec.get('recovery_course') and phase=='formal' and progress['actual_updates'] and 'ppo' in locals():
   def preserve_last():
    u=progress['actual_updates'];path=output/f'model_update_{u}.pt'
    latest={'checkpoint':str(path.resolve()),'sha256':sha(path),'update':u} if path.exists() else save_checkpoint(output,spec,actor,critic,ppo,source,u,course_state)
    from .course import summary
    rp=output/'course_result.json';r=json.loads(rp.read_text()) if rp.exists() else {'state':course_state,'best':candidates[0],'metrics':summary(candidates[0]['per_env']),'current_metrics':summary(candidates[0]['per_env'])}
    r['latest']=latest;write(rp,r)
   secondary(exc,'bounded block checkpoint',preserve_last)
  failure={'exception':repr(exc),'traceback':original_traceback,'stage':progress.get('stage'),'secondary_errors':list(getattr(exc,'__notes__',[]))};status='STOPPED_BOUNDED' if isinstance(exc,TimeoutError) else 'FAILED'
 finally:
  cleanup=[]
  def finish(name,operation):
   try:operation()
   except BaseException as error:
    cleanup.append(name+': '+repr(error))
    if failure is not None:failure.setdefault('secondary_errors',[]).append(cleanup[-1])
  try:
   progress['status']=status
   if output.exists():
    finish('progress save',lambda:write(output/'progress.json',progress))
    finish('terminal save',lambda:write(output/'terminal_report.json',{'status':status,'attempt_id':attempt,'phase':phase,'actual_updates':progress['actual_updates'],'samples':progress['samples'],'effective_actor_samples':progress['effective_actor_samples'],'evaluations':progress['evaluations'],'failure':failure if status=='FAILED' else None,'budget_interruption':failure if status=='STOPPED_BOUNDED' else None,'stop_kind':('user_signal' if __import__('native_stand.command_v6.train',fromlist=['STOP']).STOP or bool(spec.get('recovery_course') and Path(spec['recovery_course']['task_manifest']).with_name('STOP').exists()) else 'budget_or_completion') if status!='FAILED' else 'exception','wall_s_before_cleanup':time.monotonic()-started}))

   if env is not None:
    try:env.close()
    except BaseException as exc:cleanup.append(repr(exc))
   if output.exists():finish('cleanup report',lambda:write(output/'cleanup.json',{'errors':cleanup,'app_close':'pending','wall_s_before_app_close':time.monotonic()-started}))
   # Kit may exit Python inside close; restore caller RNG before that boundary too.
   torch.set_rng_state(initial_cpu_rng)
   if initial_cuda_rng is not None:torch.cuda.set_rng_state_all(initial_cuda_rng)
   if app is not None:
    try:app.close(exit_code=1 if status=='FAILED' or cleanup else 0)
    except BaseException as exc:cleanup.append(repr(exc))
   if output.exists():finish('cleanup report',lambda:write(output/'cleanup.json',{'errors':cleanup,'app_close':'returned','wall_s_including_cleanup':time.monotonic()-started}))
   # Existing monitor owns exact-ID finalization after child exit, including cleanup.
  finally:
   torch.set_rng_state(initial_cpu_rng)
   if initial_cuda_rng is not None:torch.cuda.set_rng_state_all(initial_cuda_rng)
 if args.handshake_fd is not None:
  try:os.close(args.handshake_fd)
  except OSError:pass
 return 0 if status in ('COMPLETED','STOPPED_BOUNDED') and not cleanup else 1
