"""Explicit quality adapter; shares StudentEnv, physics/PD and certificate."""
from .runtime import TrainingStudentEnv

class ContactStudentEnv(TrainingStudentEnv):
 def _get_observations(self):
  # Base64 uses existing geometric, height, clock production; replace only v2 slots.
  obs=super()._get_observations()
  if getattr(self,'task_ready',False) and getattr(self,'student_ready',False):
   from .contact_quality import runtime_measurement,critic_input
   self.contact_measurement_pre=runtime_measurement(self)
   obs={**obs,'student_critic':critic_input(obs['student_critic'],self.contact_measurement_pre,
    self.robot.data.applied_torque.torch[:,self.ids][:,[2,5]])}
   self.student_obs=obs
  return obs

 def _get_dones(self):
  result=super()._get_dones()
  if getattr(self,'task_ready',False):
   self.contact_entered_post=(self.rise_hold.entry>=0).clone()
   self.contact_elapsed_post=self.rise_hold.count.to(self.student_h0.dtype).clone()*.01
  return result
