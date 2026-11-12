"""Explicit standing-first timing curriculum; hypotheses, not motor identification."""
import copy
import torch
VERSION='motion31_standing_timing_course_v1'
CRITIC='motion_timing_privileged110_v1'
TARGETS=(0.,.25,.5,1.)

def network():
 from .motion_symmetry import network as base
 n=base();n.update(version='motion31_timing_large_actor_critic_v1',critic_input=110);return n

def contract():
 from .motion_symmetry import contract as base
 c=base();c.update(critic_dim=110,critic_privileged_version=CRITIC)
 c['critic_privileged']+=['target_transport_ms:6/20','PD_feedback_ms:1/10','torque_tau_ms:6/10',
   'actual_torque:6/effort_limits','adopted_minus_requested_target:6/action_scales']
 return c

def configuration():
 return {'version':VERSION,'strengths':list(TARGETS),'seed':20261021,
  'standing_command':[0.,0.,.34],'episode_parameter_sampling':'reset_only_independent_counter',
  'nominal_fraction':.25,'previous_fraction':.25,'current_fraction':.5,
  'leg_target_delay_ms':[0,10],'PD_feedback_ms':[0,10],'leg_torque_tau_ms':[0,5],
  'wheel_combined_onset_ms_ch2_ch5':[7,11],
  'wheel_interpretations':['transport','first_order_response'],
  'physical_parameters':'4750 nominal physics/PD/friction/armature unchanged',
  'actor':31,'critic':CRITIC,'confirmations':2,'evaluation_every':250,
  'source_failure':'does_not_block_learning','behavior_stop':'observe_only',
  'wall':'unlimited_training; bounded source/smoke/evaluation',
  'parameter_provenance':'wheel sides confirmed; all leg times hypotheses; combined wheel onset not split or double-counted'}

def commands(stage=0):return [[0.,0.]]
def stage_config(stage=0,envs=4096):
 if type(stage) is not int or stage not in range(4) or envs not in (1,64,4096):raise ValueError('timing stage/env differs')
 return {'id':stage,'name':f'standing_timing_{stage}','commands':commands(), 'weights':[1.],
  'sampling_mode':'fixed_slots','sampling_slots':[0]*envs,'slot_counts':[envs],'enforce_whitelist':True}
def protocol():
 from .motion_yaw_course import protocol as p
 return p()
def cases(spec,stage=0):
 t=spec['student_evaluation']['cases'][0]
 return [{**t,'case_id':f'timing_{stage}_{kind}','command':[0.,0.,.34],
          '_timing_level':level} for kind,level in [('nominal',-1),('previous',max(-1,stage-1)),('current',stage)]]
def effective_spec(spec,stage):
 s=copy.deepcopy(spec);s['motion_training']['stage']=stage
 s['student_training'].update(stage=stage,course_stage=stage)
 s['motion_timing']['stage']=stage;s['curriculum']['stages']=[stage_config(stage,s['ppo']['envs'])]
 s['student_evaluation']['cases']=cases(s,stage);return s
def extend_mixture(previous,values):
 from .motion_course import extend_mixture as base
 return base(previous,values)
def choose(candidates,accepted_by_stage):
 from .motion_course import choose as base
 return base(candidates,accepted_by_stage)
def refinement_records(records,stage):
 """One command slot per evaluation: all timing conditions must agree, never three confirmations."""
 expected=[(f'timing_{stage}_{kind}',level) for kind,level in
           [('nominal',-1),('previous',max(-1,stage-1)),('current',stage)]]
 by_id={r.get('case',{}).get('case_id'):r for r in records}
 complete=len(records)==3 and len(by_id)==3
 for key,level in expected:
  r=by_id.get(key,{})
  complete=complete and r.get('case',{}).get('command')==[0.,0.,.34] and r.get('case',{}).get('_timing_level')==level
  es=r.get('per_env',[])
  complete=complete and len(es)==8 and {e.get('env') for e in es}==set(range(8))
 if not complete:return [{'case':{'command':[0.,0.,.34]},'per_env':[]}]
 result=[]
 for eid in range(8):
  rows=[next(e for e in by_id[key]['per_env'] if e['env']==eid) for key,_ in expected]
  row=copy.deepcopy(rows[0]);row['accepted']=all(e.get('accepted') is True for e in rows)
  metric_keys=set.intersection(*(set(e.get('symmetry_metrics',{})) for e in rows))
  row['symmetry_metrics']={k:max(e['symmetry_metrics'][k] for e in rows) for k in metric_keys}
  result.append(row)
 return [{'case':{'command':[0.,0.,.34]},'per_env':result}]

def advance(state,candidate):
 s=copy.deepcopy(state);rows=candidate.get('evaluations',[])
 if candidate.get('stage')!=s['stage'] or candidate['update']<=s.get('last_evaluation_update',-1):raise ValueError('timing stage/update evidence differs')
 normalized=refinement_records(rows,s['stage'])[0]['per_env']
 passed=candidate.get('complete_matrix') is True and len(normalized)==8 and all(e.get('accepted') is True for e in normalized)
 s['consecutive_passes']=s.get('consecutive_passes',0)+1 if passed else 0
 s['last_evaluation_update']=candidate['update'];advanced=False
 if s['consecutive_passes']>=2:
  s['accepted_by_stage'][str(s['stage'])]={'checkpoint':candidate['checkpoint'],'sha256':candidate['sha256'],'update':candidate['update']}
  if s['stage']<3:
   s['stage']+=1;s['stage_started_update']=candidate['update'];s['consecutive_passes']=0;advanced=True
 return s,{'stage_passed':passed,'advanced':advanced,'all_stages_passed':False,
           'final_level_confirmed':s['stage']==3 and s['consecutive_passes']>=2,'stop':False}

def expand_source(source):
 from .motion_symmetry import network as old
 if source.get('network_architecture')==network() and source.get('critic_privileged_version')==CRITIC:return source
 if source.get('network_architecture')!=old() or source.get('critic_privileged_version')!='motion_symmetry_privileged85_v1':raise ValueError('timing requires explicit4750 Motion85 source')
 out=copy.deepcopy(source);w=out['critic']['mlp.0.weight']
 def pad(v):return torch.cat([v,torch.zeros(v.shape[0],25,dtype=v.dtype,device=v.device)],1)
 out['critic']['mlp.0.weight']=pad(w)
 ids=[p for g in out['optimizer']['param_groups'] for p in g['params']]
 pid=ids[len(out['actor'])+list(out['critic']).index('mlp.0.weight')]
 for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
  if name in out['optimizer']['state'].get(pid,{}):out['optimizer']['state'][pid][name]=pad(out['optimizer']['state'][pid][name])
 out.update(network_architecture=network(),critic_privileged_version=CRITIC);return out

def validate_recipe(recipe,source_sha):
 from .training_recipe import validate_instance
 r=copy.deepcopy(recipe)
 if r.get('course_version')=='motion31_standing_antiosc_armature_v4':
  from .motion_antioscillation import validate_configuration
  validate_configuration(r['timing_configuration'])
  if r.get('critic_privileged_version')=='motion_antiosc_privileged140_v1':
   from .motion_antiosc_privileged import network as anti_network
   matched=r['timing_configuration'].get('wheel_execution') is not None
   expected_restore=('full_motion140_wheel_match','source_Adam_full_state') if matched else ('source110_zero_pad_to140','source_Adam_zero_new_columns')
   if r['network_architecture']!=anti_network() or (r['critic'],r['optimizer'])!=expected_restore:raise ValueError('Critic140 recipe declaration differs')
   r.update(critic_privileged_version=CRITIC,network_architecture=network(),critic='full_motion110',optimizer='source_Adam_full_state')
  if r['stage'] not in range(5):raise ValueError('inertia stage invalid')
  r.update(course_version='motion31_standing_sensor_transfer_v3',timing_configuration=__import__('native_stand.students.motion_sensor_transfer',fromlist=['configuration']).configuration(),stage=0,course_stage=0)
  validate_recipe(r,source_sha);return recipe
 from .motion_fixed_timing import VERSION as FIXED,configuration as fixed_configuration
 from .motion_sensor_transfer import VERSION as SENSOR,configuration as sensor_configuration
 if r.get('course_version')==SENSOR:
  if r.get('timing_configuration')!=sensor_configuration():raise ValueError('sensor transfer contract differs')
  r.update(course_version=FIXED,timing_configuration=fixed_configuration())
  validate_recipe(r,source_sha);return recipe
 if r.get('course_version')==FIXED:
  if r.get('timing_configuration')!=fixed_configuration() or r.get('critic')!='full_motion110' or r.get('optimizer')!='source_Adam_full_state':raise ValueError('fixed timing configuration/restore differs')
  r.update(course_version=VERSION,timing_configuration=configuration(),critic='source85_zero_pad_to110',optimizer='source_Adam_zero_new_columns')
  validate_recipe(r,source_sha);return recipe
 if r.get('critic_privileged_version')!=CRITIC or r.get('network_architecture')!=network():raise ValueError('timing Critic/network differs')
 if r.get('course_version')!=VERSION or r['stage'] not in range(4):raise ValueError('timing course differs')
 expected=r.pop('timing_configuration',None)
 if expected!=configuration():raise ValueError('timing contract differs')
 from .motion_symmetry import network as old
 r.update(critic_privileged_version='motion_symmetry_privileged85_v1',network_architecture=old(),
          course_version='motion31_parallel_bidirectional_course_v1',parallel_state={'stage':r['stage'],'lanes':{},'stage_started_update':0})
 # Validate complete recipe against the existing versioned reward/algorithm baseline,
 # while replacing only the separately declared course/privileged dimensions.
 from .motion_training import declaration
 base=declaration(source_sha,r['stage'],reward_version=r['reward']['version'])
 allowed={'milestones','selected_rollouts_smoke','selected_rollouts_formal','final_seeds','storage_max_bytes',
          'checkpoint_every_updates','network_architecture','course_version','course_stage','smoke_wiring','selection_scope',
          'timing_configuration','critic_privileged_version','critic','optimizer'}
 for k,v in base.items():
  if k not in allowed and recipe.get(k)!=v:raise ValueError('timing changed baseline recipe '+k)
 if set(recipe)-set(base)-allowed:raise ValueError('unknown timing recipe fields')
 if recipe.get('source_policy_retention') is not None:raise ValueError('timing does not add undeclared BC')
 if recipe.get('critic')!='source85_zero_pad_to110' or recipe.get('optimizer')!='source_Adam_zero_new_columns':raise ValueError('timing initialization differs')
 if recipe['checkpoint_every_updates']!=100 or recipe['milestones']!=list(range(250,10000,250)):raise ValueError('timing save/evaluation cadence differs')
 return recipe
