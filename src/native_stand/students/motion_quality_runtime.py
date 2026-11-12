"""Dormant explicit v3 quality wrapper, imported after AppLauncher only."""
import copy
import torch
from .motion_runtime import MotionStudentEnv
from .motion_quality import QualityTermination,apply_reward,configuration,CAUSES
from .motion_tracking_reward import quality_base_configuration

class QualityMotionStudentEnv(MotionStudentEnv):
 def __init__(self,*args,motion_training_config,**kwargs):
  self.motion_quality_ready=False
  self.motion_quality_reward_config=motion_training_config['reward']
  self.quality_config=motion_training_config['reward']['quality_termination']
  if self.quality_config!=configuration():raise ValueError('quality runtime declaration differs')
  base=copy.deepcopy(motion_training_config);base['reward']=quality_base_configuration(self.motion_quality_reward_config['version'])
  super().__init__(*args,motion_training_config=base,**kwargs)
  self.quality_monitor=QualityTermination(self.num_envs,self.device,self.quality_config)
  self.motion_quality_snapshot=torch.zeros(self.num_envs,len(CAUSES),device=self.device,dtype=torch.bool)
  self.motion_quality_ready=True
 def _reset_idx(self,ids):
  super()._reset_idx(ids)
  if getattr(self,'motion_quality_ready',False):
   ids=torch.arange(self.num_envs,device=self.device) if ids is None else ids
   self.quality_monitor.reset(ids)
  # Pre-reset cause snapshot survives reset for evaluator and rollout summaries.
 def _get_dones(self):
  terminated,truncated=super()._get_dones()
  if getattr(self,'motion_manual_command',None) is not None:
   # Manual preview keeps physical termination only: the recipe quality gate is not applied to
   # out-of-distribution manual commands, so preview smoothness is never capability evidence.
   return terminated,truncated
  if not getattr(self,'motion_quality_ready',False):return terminated,truncated
  s=self.recovery_snapshot;p=self.standing_snapshot;fault=s[:,4].bool();valid=s[:,29].bool()
  decision=self.quality_monitor.observe(posture=p,height=s[:,6],wheel_offset=self.rise_hold.offset,contacts=s[:,10:12],
   target=self.target_xy,controlled=self.student_actor_valid,valid=valid&~fault)
  quality_failed=decision['terminated'];self.motion_quality_snapshot.copy_(decision['causes'])
  self.student_reward_result=apply_reward(self.student_reward_result,posture=p,wheel_offset=self.rise_hold.offset,
   leg_q=self.robot.data.joint_pos.torch[:,self.ids][:,[0,1,3,4]],controlled=self.student_actor_valid,valid=valid,
   quality_failed=quality_failed,physical_failed=terminated&~fault,fault=fault,config=self.quality_config,reward_config=self.motion_quality_reward_config)
  if self.motion_quality_reward_config['version'] in ('student_motion31_yaw_contact_quality_v7','student_motion31_symmetry_quality_v8','student_motion31_differential_quality_v9','student_motion31_parallel_quality_v10'):
   from .motion_yaw_quality import apply as apply_spin
   self.student_reward_result=apply_spin(self.student_reward_result,target=self.target_xy,command=self.motion_action_command,
    velocity=torch.stack((self.vx,self.vy,s[:,8]*0),-1),contacts=s[:,10:12],
    previous_targets=self.motion_spin_previous_targets,executed_targets=self.motion_spin_executed_targets,
    measurement=self.motion_spin_measurement_post,config=self.motion_quality_reward_config['spin_quality'])
  if self.motion_quality_reward_config['version'] in ('student_motion31_symmetry_quality_v8','student_motion31_differential_quality_v9','student_motion31_parallel_quality_v10'):
   from .motion_symmetry import apply as apply_symmetry
   self.student_reward_result=apply_symmetry(self.student_reward_result,self)
  if self.motion_quality_reward_config['version'] in ('student_motion31_differential_quality_v9','student_motion31_parallel_quality_v10'):
   from .motion_differential import apply as apply_differential
   self.student_reward_result=apply_differential(self.student_reward_result,self)
  if self.motion_quality_reward_config['version']=='student_motion31_parallel_quality_v10':
   from .motion_parallel_course import apply_tracking
   self.student_reward_result=apply_tracking(self.student_reward_result,self)
  self.student_reward_result['version']=self.motion_quality_reward_config['version']
  terminated=terminated|quality_failed;truncated=truncated&~terminated
  s[:,3]=s[:,3].bool()|quality_failed
  self.rise_hold.task_failure|=quality_failed;self.rise_hold.normal_timeout.copy_(truncated)
  self.student_event_snapshot[:,3]|=quality_failed
  self.episode_result_snapshot[:,0]=terminated;self.episode_result_snapshot[:,1]=truncated
  self.episode_result_snapshot[:,4]=terminated|truncated
  self.recovery_snapshot_valid.copy_(terminated|truncated)
  from native_stand.recovery.domain import Mode
  self.fsm.mode[quality_failed]=int(Mode.RECOVERY_FAILED);s[:,18]=self.fsm.mode.float()
  return terminated,truncated
