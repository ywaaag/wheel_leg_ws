"""Boundary verification for the explicitly authorized yaw progression with forward retention."""
import json
from pathlib import Path
from .data import sha
from . import motion_yaw_course as course

def verify(spec):
 from .motion_tracking_reward import configuration,QUALITY_VERSION
 from .motion_guard import configuration as guard,evidence_configuration
 from .training_recipe import validate_instance
 from .motion_startup import resolve,smoke_profile
 c=spec['motion_course'];parent=Path(c['source_parent_spec'])
 if sha(parent)!=c['source_parent_spec_sha256'] or c['configuration']!=course.configuration():raise ValueError('yaw parent/config differs')
 p=json.loads(parent.read_text())
 # Parent is the exact settled previous course, binding source identity and contract.
 from .motion_forward_requests import verify as verify_parent
 verify_parent(p)
 if spec['source_checkpoint']!=p['source_checkpoint'] or spec['source_sha256']!=p['source_sha256'] or sha(spec['source_checkpoint'])!=spec['source_sha256']:raise ValueError('yaw source differs')
 for k in ('action','control','ground','observation','history','ppo','student_interface','reward'):
  if spec[k]!=p[k]:raise ValueError('yaw contract drift:'+k)
 if spec['reward']!=configuration(QUALITY_VERSION):raise ValueError('yaw preserves v3 reward')
 r=validate_instance(spec['student_training'],spec['source_sha256'])
 if r.get('course_version')!=course.VERSION or r.get('course_stage')!=0 or r['stage']!=0 or r['smoke_wiring']!=smoke_profile():raise ValueError('yaw recipe differs')
 if spec['motion_assessment']!=course.protocol() or spec['motion_training']['early_stop_policy'] not in (guard(),evidence_configuration()):raise ValueError('yaw gate differs')
 if spec['student_evaluation']['version']!=course.ASSESSMENT or [r['command'] for r in spec['student_evaluation']['cases']]!=[v+[.34] for v in course.commands(0)]:raise ValueError('yaw initial matrix differs')
 if spec['curriculum']['stages']!=[course.stage_config(0)] or spec['curriculum']['schedule_s']!=r['train_schedule_s']:raise ValueError('yaw sampling/schedule differs')
 b=spec['budget']
 if b['smoke_updates']!=5 or b['formal_updates']!=1995 or b['sample_limit']!=2000*65536 or b['chain_wall_s']!=64800 or b['formal_wall_s']!=57600:raise ValueError('yaw2000 budget differs')
 if r['milestones']!=list(range(250,1996,250)):raise ValueError('yaw cadence differs')
 if 'checkpoint_every_updates' in r and spec['evaluation'].get('checkpoint_every_updates')!=r['checkpoint_every_updates']:raise ValueError('yaw checkpoint recipe/report cadence differs')
 reused,_=resolve(spec)
 retention=r.get('source_policy_retention')
 if retention is not None:
  from .motion_source_retention import verify_source_records
  verify_source_records(list(reused.values()),retention)
 import torch
 cp=torch.load(spec['source_checkpoint'],map_location='cpu',weights_only=False)
 if cp['recipe_updates']!=1600 or cp['critic_privileged_version']!='motion_privileged49_v1' or {g['lr'] for g in cp['optimizer']['param_groups']}!={3e-5}:raise ValueError('yaw full source differs')
 return True
