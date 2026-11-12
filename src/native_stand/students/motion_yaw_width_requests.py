"""Strict frozen boundary for the authorized same-source yaw-kernel comparison."""
import json
from pathlib import Path
from .data import sha
from . import motion_yaw_width_course as course

def verify(spec):
 from .motion_yaw_requests import verify as verify_parent
 from .training_recipe import validate_instance
 from .motion_tracking_reward import configuration,QUALITY_VERSION,YAW_NARROW_VERSION
 from .motion_guard import evidence_configuration
 from .motion_startup import resolve,smoke_profile
 c=spec['motion_course'];parent=Path(c['source_parent_spec'])
 if sha(parent)!=c['source_parent_spec_sha256'] or c['configuration']!=course.configuration():raise ValueError('fixed yaw parent/config differs')
 p=json.loads(parent.read_text());verify_parent(p)
 for key in ('source_checkpoint','source_sha256','action','control','ground','observation','history','ppo','student_interface','performance'):
  if key=='performance':
   from .performance import parent_motion_performance_matches
   if not parent_motion_performance_matches(spec.get(key),p.get(key)):raise ValueError('unsupported performance drift')
   continue
  if spec.get(key)!=p.get(key):raise ValueError('fixed yaw contract drift:'+key)
 reward=spec['reward']
 if reward.get('version') not in (QUALITY_VERSION,YAW_NARROW_VERSION) or reward!=configuration(reward['version']):raise ValueError('fixed yaw width declaration differs')
 r=validate_instance(spec['student_training'],spec['source_sha256'])
 if r['reward']!=reward or r.get('course_version')!=course.VERSION or r.get('course_stage')!=0 or r['stage']!=0 or r.get('smoke_wiring')!=smoke_profile() or r.get('source_policy_retention') is not None:raise ValueError('fixed yaw recipe/retention differs')
 if spec['motion_assessment']!=course.protocol() or spec['motion_training']['early_stop_policy']!=evidence_configuration():raise ValueError('fixed yaw guard/protocol differs')
 if spec['curriculum']['stages']!=[course.stage_config(0)] or spec['curriculum']['schedule_s']!=r['train_schedule_s']:raise ValueError('fixed yaw sampling/schedule differs')
 if spec['student_evaluation']['version']!=course.ASSESSMENT or [x['command'] for x in spec['student_evaluation']['cases']]!=[v+[.34] for v in course.commands(0)]:raise ValueError('fixed yaw matrix differs')
 b=spec['budget']
 if any(b.get(k)!=v for k,v in {'smoke_updates':5,'formal_updates':1000,'sample_limit':1005*65536,'chain_wall_s':32400,'formal_wall_s':28800}.items()):raise ValueError('fixed yaw1005 inclusive budget differs')
 if r['milestones']!=[250,500,750,1000] or r.get('checkpoint_every_updates')!=100 or spec['evaluation']['checkpoint_every_updates']!=100 or spec['evaluation']['full_every_updates']!=250:raise ValueError('fixed yaw cadence differs')
 resolve(spec)
 if sha(spec['source_checkpoint'])!=spec['source_sha256']:raise ValueError('fixed yaw source changed')
 return True
