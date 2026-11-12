"""Explicit wheel-inertia/Kv adaptation; no change to the31/140D schemas."""
import copy,json
from pathlib import Path
import torch
from .data import sha

VERSION='motion31_wheel_pd_match_v1'

def profile():
 return {'version':VERSION,'wheel_kv_Nm_per_rad_s':.2,
  'wheel_total_J_candidate_kg_m2_ch2_ch5':[.004204,.0042075],
  'wheel_body_axial_J_reference_kg_m2':.00066759,
  'wheel_extra_armature_kg_m2_ch2_ch5':[.00353641,.00353991],
  'wheel_peak_Nm':5.5,'PD_period_ms':5,'PD_feedback_ms':10,
  'mechanical_drag':'unchanged native zero drive damping; no new B/C compensation',
  'provenance':'reported fixed-parent output-shaft J candidate, not hardware calibration; Kv selected by bounded J/feedback sweep'}

def validate_profile(p):
 if p!=profile():raise ValueError('explicit wheel matching profile differs')
 return p

def configuration():
 from .motion_antioscillation import configuration as base
 c=base();c.update(wheel_kv=.2,wheel_execution=profile())
 c['actor_sensor']['critic']='Actor31/Critic140 unchanged; delayed Actor prefix and current physical privileged columns'
 return c

def setup(env,p):
 """Set only the two wheel added inertias; backend readback once at setup."""
 validate_profile(p)
 from native_stand.control_readback import tensor
 ids=torch.tensor([env.ids[i] for i in (2,5)],device=env.device,dtype=torch.int32)
 expected=torch.tensor(p['wheel_extra_armature_kg_m2_ch2_ch5'],device=env.device)[None].expand(env.num_envs,-1).contiguous()
 env.robot.write_joint_armature_to_sim_index(armature=expected,joint_ids=ids)
 actual=torch.as_tensor(tensor(env.robot.root_view.get_dof_armatures()),device=env.device)[:,ids.long()]
 if not torch.allclose(actual,expected,atol=1e-7,rtol=0):raise RuntimeError('wheel armature backend readback differs')
 dampings=torch.as_tensor(tensor(env.robot.root_view.get_dof_dampings()),device=env.device)[:,ids.long()]
 if not torch.allclose(dampings,torch.zeros_like(dampings),atol=1e-7,rtol=0):raise RuntimeError('wheel built-in damping must remain zero with explicit velocity P')
 return {'version':VERSION,'channels':[2,5],'geometric_sides':['right','left'],
  'actual_extra_armature_kg_m2_ch2_ch5':actual[0].cpu().tolist(),
  'backend_wheel_damping':dampings[0].cpu().tolist(),'wheel_kv':env.cfg.wheel_kv,
  'expected_total_J_candidate':p['wheel_total_J_candidate_kg_m2_ch2_ch5'],
  'scope':'armature backend verified; total fixed-parent J uses frozen asset reference, not independent hardware identification'}

def build(parent,checkpoint,*,experiment,spec_path):
 from .motion_antiosc_privileged import VERSION as CRITIC,network
 from .motion_antioscillation import VERSION as COURSE,cases,stage_config
 p=Path(parent).resolve();cp=Path(checkpoint).resolve();old=json.loads(p.read_text());root=p.parents[3];run=root/'runs'/old['experiment']
 book=json.loads(p.with_name('budget.json').read_text());terminal=json.loads((run/'formal/terminal_report.json').read_text());chain=json.loads((run/'chain_terminal_report.json').read_text())
 if (old.get('motion_timing',{}).get('version')!=COURSE or old['student_training'].get('critic_privileged_version')!=CRITIC
     or old['motion_timing'].get('wheel_match_restart') or terminal.get('stop_kind')!='user_signal'
     or terminal['status']!='STOPPED_BOUNDED' or chain['status']!='STOPPED_BOUNDED'
     or any(a['status']=='RUNNING' for a in book['attempts'].values())):raise ValueError('settled user-stopped antiosc140 parent required')
 if cp!=run/'formal'/f"model_update_{terminal['actual_updates']}.pt" or terminal['actual_updates']!=book['used']['formal']:raise ValueError('exact latest full checkpoint from stopped parent required')
 data=torch.load(cp,map_location='cpu',weights_only=False)
 if (data['spec_sha256']!=sha(p) or data['update']!=terminal['actual_updates'] or data.get('critic_privileged_version')!=CRITIC
     or data.get('network_architecture')!=network() or not all(k in data for k in ('actor','critic','optimizer','training_rng','fixed_std'))):raise ValueError('source140 full-state identity differs')
 if {g['lr'] for g in data['optimizer']['param_groups']}!={3e-5}:raise ValueError('wheel match keeps original LR')
 inputs={str(q):sha(q) for q in (p,cp,p.with_name('budget.json'),run/'formal/terminal_report.json',run/'chain_terminal_report.json')}
 prior=old['motion_timing'].get('critic_upgrade',{}).get('historical_online_updates',0)
 history={'version':VERSION,'parent_spec':str(p),'parent_run':str(run),'inputs':inputs,
  'parent_used':book['used'],'previous_authorization_online_used':prior+book['used']['smoke']+book['used']['formal'],
  'new_authorization_online_limit':10000,'old_unused_budget_transferred':False,
  'source_checkpoint_update':data['update'],'physical_state_restored':False,
  'course_state':'reset stage0/confirmations/mix for changed plant; Actor/Critic/Adam/std/RNG restored'}
 s=copy.deepcopy(old);s.update(experiment=experiment,source_checkpoint=str(cp),source_sha256=sha(cp))
 s['control']['wheel_kv']=.2
 s['motion_timing']={'version':COURSE,'configuration':configuration(),'stage':0,'parent_spec':str(p),
  'parent_spec_sha256':sha(p),'source_checkpoint_sha256':sha(cp),'wheel_match_restart':history}
 s['motion_training'].update(stage=0,spec_path=str(Path(spec_path).resolve()),resume=None)
 for key in ('wiring_reuse','partial_smoke_recovery','prior_consumption'):s['motion_training'].pop(key,None)
 s['student_training'].update(source_actor_sha256=sha(cp),timing_configuration=configuration(),
  critic='full_motion140_wheel_match',optimizer='source_Adam_full_state',stage=0,course_stage=0,
  selection_scope='standing034 anti-oscillation with candidate wheel J and Kv.2; evidence-gated leg armature, no speed certification')
 s['motion_course']['source_recipe_updates']=data['recipe_updates']
 s['motion_course']['configuration'].update(formal_updates_max=9995,smoke_updates_max=5,stage_updates_max=9995)
 s['budget'].update(smoke_updates=5,formal_updates=9995,formal_samples=9995*4096*128,
  sample_limit=10000*4096*128,formal_wall_s=None,chain_wall_s=None,source_eval_wall_s=3600,smoke_wall_s=1800,evaluation_wall_s=7200)
 s['artifacts']['run_name']=experiment;s['curriculum']['stages']=[stage_config()];s['curriculum']['active_stage']=0
 s['student_evaluation']['cases']=cases(s)
 for r in s['student_evaluation']['cases']:r.update(checkpoint=str(cp),sha256=sha(cp))
 s['motion_startup'].update(references=[],missing_commands=[[0.,0.,.34]])
 s['student_dependencies'].update(inputs)
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':None,
  'user_scope':'User2026-10-22 explicitly approves configuration/PD changes and new10000 inclusive online updates, no training wall limit; wheel J candidate/Kv.2, Actor31/Critic140/reward/leg course unchanged; GPU0 shared; preserve stopped parent actual usage, no old balance transfer'}
 s['notifications']={'enabled':True};return s

def verify(spec):
 from .motion_timing import validate_recipe
 parent=spec['motion_timing']['parent_spec']
 expected=build(parent,spec['source_checkpoint'],experiment=spec['experiment'],spec_path=spec['motion_training']['spec_path'])
 for k in ('authorization','resources','notifications'):expected[k]=spec[k]
 if expected!=spec:raise ValueError('wheel matching restart has undeclared drift')
 claim=Path(parent).parent.parent/'.motion_wheel_match_claims'/(sha(parent)+'.json')
 if claim.exists() and json.loads(claim.read_text())['destination']!=spec['experiment']:raise ValueError('wheel matching parent already claimed')
 validate_recipe(spec['student_training'],spec['source_sha256']);return True
