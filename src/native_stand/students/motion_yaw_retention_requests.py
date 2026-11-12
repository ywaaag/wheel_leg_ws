"""Bind the only new training variable to a settled narrow-yaw control."""
import copy,json
from pathlib import Path
from .data import sha
from . import motion_yaw_retention_course as course

def verify(spec):
 from .motion_yaw_width_requests import verify as verify_control
 from .motion_yaw_width_course import configuration as control_configuration,VERSION as CONTROL
 from .motion_source_retention import declaration,verify_source_records
 from .motion_startup import resolve
 from .training_recipe import validate_instance
 c=spec['motion_course'];p=Path(c['retention_parent_spec'])
 if sha(p)!=c['retention_parent_spec_sha256']:raise ValueError('retention control SHA differs')
 control=json.loads(p.read_text());verify_control(control)
 if control['reward']['yaw_denominator']!=.25:raise ValueError('retention control must be narrow B')
 if c['configuration']!=course.configuration():raise ValueError('retention fixed course differs')
 recipe=validate_instance(spec['student_training'],spec['source_sha256'])
 if recipe.get('source_policy_retention')!=declaration(spec['source_sha256'],.01):raise ValueError('retention coefficient/scope differs')
 if recipe['course_version']!=course.VERSION:raise ValueError('retention course not explicitly enabled')
 # Reuse the frozen baseline validator only after removing the explicit new loss.
 view=copy.deepcopy(spec);view['student_training'].pop('source_policy_retention')
 view['student_training']['course_version']=CONTROL
 view['motion_course']['configuration']=control_configuration()
 view['motion_course'].pop('retention_parent_spec');view['motion_course'].pop('retention_parent_spec_sha256')
 verify_control(view)
 # Compare the actual behavior contract, not only the generic baseline schema.
 for key in ('source_checkpoint','source_sha256','reward','action','control','ground','observation','history','ppo','student_interface','performance','curriculum','motion_assessment','evaluation','budget'):
  if key=='performance':
   from .performance import parent_motion_performance_matches
   if not parent_motion_performance_matches(spec.get(key),control[key]):raise ValueError('unsupported performance drift')
   continue
  if spec[key]!=control[key]:raise ValueError('retention changed another variable:'+key)
 before=copy.deepcopy(control['student_training']);after=copy.deepcopy(recipe)
 after.pop('source_policy_retention');after['course_version']=CONTROL
 if before!=after:raise ValueError('retention changed PPO recipe beyond auxiliary loss')
 reused,_=resolve(spec);verify_source_records(list(reused.values()),recipe['source_policy_retention'])
 return True
