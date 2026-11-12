"""Compact controlled pre-entry/catch/steady statistics, no scoring or RNG changes."""
import torch
NAMES=('pre_entry','entry_0_1s','entry_1_2s','entry_2_10s')
class PhaseMetrics:
 def __init__(self,device):
  self.count=torch.zeros(4,device=device);self.sums=torch.zeros(4,35,device=device)
 def observe(self,*,elapsed,entered,controlled,contacts,angles,velocity,previous,executed,requested_torque,clipped_torque,applied_torque,measurement=None):
  phase=torch.where((~entered)|(elapsed<=0),0,torch.where(elapsed<=1.+1e-6,1,torch.where(elapsed<=2.+1e-6,2,3)))
  controlled=controlled&((~entered)|(elapsed<=10.+1e-6))
  # Four legs & two wheels are separate, in physical units; deltas don't bridge resets.
  delta=executed-previous
  per=torch.cat((angles.abs(),velocity.abs(),(contacts>2).float(),contacts,
   delta[:,[0,1,3,4]].square(),delta[:,[2,5]].square(),
   requested_torque[:,[2,5]].square(),clipped_torque[:,[2,5]].square(),applied_torque[:,[2,5]].square(),
   (clipped_torque.abs()>=clipped_torque.new_tensor([40,40,5.5,40,40,5.5])-1e-4).float(),
   torch.zeros(len(elapsed),8,device=elapsed.device)), -1)
  # 2+3+2+2+4+2+2+2+2+6+8=35
  if measurement is not None:
   touching=(contacts>2)&measurement['geometry_valid']
   per[:,-8:-4]=(measurement['tangent_velocity_w'].square()*touching[...,None]).reshape(-1,4)
   per[:,-4:-2]=touching.float();per[:,-2:]=measurement['geometry_valid'].float()
  for i in range(4):
   mask=controlled&(phase==i);self.count[i]+=mask.sum();self.sums[i]+=(per*mask[:,None]).sum(0)
 def result(self):
  rows={}
  for name,n,s in zip(NAMES,self.count.cpu().tolist(),self.sums.cpu()):
   if n==0:rows[name]={'controlled_samples':0,'status':'NO_SAMPLES'};continue
   m=s/n
   counts=s[-4:-2];slip=(s[-8:-4].reshape(2,2)/counts.clamp_min(1)[:,None]).sqrt()
   rows[name]={'controlled_samples':int(n),'pitch_roll_abs_mean_rad':m[:2].tolist(),
    'velocity_abs_mean_vx_vy_yaw':m[2:5].tolist(),'left_right_contact_fraction':m[5:7].tolist(),
    'left_right_contact_mean_n':m[7:9].tolist(),'leg_target_delta_rms_rad':m[9:13].sqrt().tolist(),
    'wheel_target_delta_rms_rad_s':m[13:15].sqrt().tolist(),
    'wheel_requested_clipped_applied_torque_rms_nm':m[15:21].reshape(3,2).sqrt().tolist(),
    'channel_limit_fraction':m[21:27].tolist(),'contact_point_velocity_xy_rms_mps':slip.tolist() if counts.sum()>0 else None,
    'slip_contact_samples':counts.tolist(),'slip_geometry_valid_samples':s[-2:].tolist()}
  return {'version':'recovery_standing_phase_metrics_v1','windows':'first confirmed entry; fixed clock, never restart; no capability gate change','sampling':'100Hz policy held torque','phases':rows}
