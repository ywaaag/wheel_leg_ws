"""Bounded first symmetry block; no speed promotion before geometry certification."""
import copy,json
from pathlib import Path
from .data import sha
from .motion_symmetry import VERSION,CRITIC,configuration,contract,network
SOURCE='87c00b82741b9b587d5373b4e8fa983537c245aa93a88f6494ab7da4f32ecff6'

def build(parent,checkpoint,*,experiment,spec_path):
 parent=Path(parent).resolve();cp=Path(checkpoint).resolve();p=json.loads(parent.read_text())
 import torch
 saved=torch.load(cp,map_location='cpu',weights_only=False)
 if sha(cp)!=SOURCE or saved['spec_sha256']!=sha(parent) or saved['critic_privileged_version']!='motion_spin_privileged61_v1':raise ValueError('exact750 source required')
 s=copy.deepcopy(p);s.update(experiment=experiment,source_checkpoint=str(cp),source_sha256=SOURCE,evaluation_only=False,
  reward=configuration(),student_interface=contract(),network_architecture=network(),observation={'frame_dim':22,'actor_dim':31,'critic_dim':85})
 s['artifacts'].update(run_name=experiment,storage_max_bytes=4*2**30)
 from .motion_training import declaration
 from .motion_startup import smoke_profile
 from .motion_source_retention import declaration as retention
 r=declaration(SOURCE,0,reward_version=VERSION)
 r.update(course_version='motion31_large_combo4096_v1',course_stage=0,network_architecture=network(),
  checkpoint_every_updates=100,smoke_wiring=smoke_profile(),milestones=list(range(250,2996,250)),
  storage_max_bytes=4*2**30,source_policy_retention=retention(SOURCE),continuation_teacher={'checkpoint':str(cp),'sha256':SOURCE})
 s['student_training']=r
 s['motion_training'].update(spec_path=str(Path(spec_path).resolve()),resume={'parent_spec':str(parent),'parent_spec_sha256':sha(parent),'formal_dir':str(cp.parent),'parent_recipe_updates':saved['recipe_updates']})
 s['motion_training'].pop('wiring_reuse',None)
 s['motion_course']['source_recipe_updates']=saved['recipe_updates']
 s['motion_course']['configuration'].update(formal_updates_max=2995,stage_updates_max=2995,smoke_updates_max=5)
 s.pop('motion_startup',None)
 # Minimal wiring policy without reusing source protocol or bypassing new readback.
 s['motion_startup']={'version':'motion31_minimal_startup_v1','source_policy':'reuse_exact_source_evidence_then_evaluate_only_missing_endpoints','smoke_policy':'motion31_wiring5_no_capability_eval_v2','references':[],
 'missing_commands':[x['command'] for x in s['student_evaluation']['cases']],'source_evaluation_is_not_wiring':True,'no_capability_prerequisite_for_smoke':True}
 s['motion_symmetry']={'version':'motion_symmetry85_s0_v1','parent_spec':str(parent),'parent_spec_sha256':sha(parent),
 'assessment':{'swing_strict_max_rad':.05,'full_swing_required_for_final':True,'yaw_translation_mean_max_mps':.1,'common_rolling_abs_mean_max_mps':.05},
 'refinement':{'version':'symmetry_evidence_mix_v1','confirmations':2,'increment':.25,'maximum':.75,'regression_confirmations':2,'swing_peak_before_refine_rad':.045},
 'stage_scope':'S0 original7 commands only; no backward/speed unlock in this3000 update block',
 'budget':'3000 inclusive=smoke5+formal2995; no PPO/chain wall ceiling; evaluation child bounded',
 'teacher_scope':'frozen750; old five zero/forward cohorts KL.01; yaw remains PPO-only to avoid anchoring its known swing violation'}
 # Keep yaw unanchored in first symmetry block: source yaw has known new geometry violation.
 s['budget'].update(smoke_updates=5,formal_updates=2995,formal_samples=2995*4096*128,sample_limit=3000*4096*128,
  source_eval_updates=0,source_eval_wall_s=3600,smoke_wall_s=1800,formal_wall_s=None,chain_wall_s=None,evaluation_wall_s=1800)
 for row in s['student_evaluation']['cases']:row.update(checkpoint=str(cp),sha256=SOURCE)
 s['student_dependencies'].update({str(parent):sha(parent),str(cp):SOURCE})
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'CPU preparation; user approved3000 inclusive no training wall ceiling'}
 s['resources']={'mode':'shared','sharing_authorized':True,'authorization_note':'User approves Motion750 symmetry3000 online, inherited GPU0 shared, no preemption'}
 return s

def verify(spec):
 if spec.get("motion_differential"):
  from .motion_differential_experiment import verify as verify_differential
  return verify_differential(spec)
 if 'motion_behavior_guard' in spec:
  from .motion_guard import validate_behavior_mode
  validate_behavior_mode(spec)
  base=copy.deepcopy(spec);base.pop('motion_behavior_guard')
  return verify(base)
 if spec.get('motion_training',{}).get('wiring_reuse',{}).get('version')=='motion_symmetry85_boot_loss_recovery_v1':
  from .motion_reboot_recovery import verify as verify_reboot
  return verify_reboot(spec)
 c=spec['motion_symmetry'];p=Path(c['parent_spec'])
 if sha(p)!=c['parent_spec_sha256']:raise ValueError('symmetry parent changed')
 expected=build(p,spec['source_checkpoint'],experiment=spec['experiment'],spec_path=spec['motion_training']['spec_path'])
 expected['authorization']=spec['authorization']
 if expected!=spec:raise ValueError('symmetry undeclared configuration drift')
 from .training_recipe import validate_instance
 validate_instance(spec['student_training'],spec['source_sha256'])
 return True
