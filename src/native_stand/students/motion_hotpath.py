"""Explicit state epochs and immutable geometry; no simulator reads in pure helpers."""
import torch
VERSION='motion31_hotpath_v1'

def cache_key(physics_step,reset_epoch):return (int(physics_step),int(reset_epoch))


def body_consumers(body_names,allowed,restricted):
 """Audit only: never remove contact bodies based on naming alone."""
 wheels={'left_wheel','right_wheel'};forbidden={n for n in body_names if n not in wheels and not n.startswith('guide_wheel_')}
 required=wheels|forbidden|set(allowed)|set(restricted)
 missing=required-set(body_names)
 if missing:raise ValueError('required contact bodies missing: '+str(sorted(missing)))
 return {'required':sorted(required),'removable_if_no_other_consumer':sorted(set(body_names)-required),'scope':'audit, no sensor subset enabled'}


def runner_flags(checks,truncated):
 """One transfer for safety plus timeout branch; all original safety checks remain."""
 flags=torch.stack(checks.flags+[truncated.any()]).detach().cpu().tolist()
 if not all(flags[:-1]):raise RuntimeError('step safety: '+', '.join(n for n,ok in zip(checks.names,flags[:-1]) if not ok))
 return flags[-1]


def readback(env):
 requested=env.cfg.scene.contact_forces.history_length
 actual=env.sensor.cfg.history_length
 if actual!=requested:raise RuntimeError('ContactSensor history effective configuration differs')
 return {'version':VERSION,'motion_cache_enabled':getattr(env.cfg,'motion_hotpath',False),'runtime_reduction_enabled':getattr(env.cfg,'motion_runtime_reduction',False),'settled_diagnostic_scope':'initializing rows only, controlled rows marked false/not evaluated' if getattr(env.cfg,'motion_runtime_reduction',False) else 'all rows','cache_clock':'simulator physics substep counter + reset epoch; subclass reset invalidates again','contact_history_length_requested':requested,'contact_history_length_effective':actual,'sensor_bodies':list(env.sensor.body_names),'wheel_sensor_ids':list(env.wheel_sensor_ids),'forbidden_sensor_ids':list(env.forbidden_sensor_ids),'allowed_sensor_ids':list(getattr(env,'allowed_sensor',[])),'restricted_sensor_ids':list(getattr(env,'restricted_sensor',[])),'body_subset_changed':False,'physical_equivalence':'NOT_VERIFIED'}
