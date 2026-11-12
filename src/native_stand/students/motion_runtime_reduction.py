"""Explicit Motion runtime reduction: unchanged physical gates, fewer tensor launches."""
import torch
VERSION='motion31_runtime_reduction_v1'


def settled_initial_subset(*,initializing,velocity,angular,joint_velocity,settlement_joint_ids,up,height,chassis,root_bounds,chassis_bounds,fault,restricted,config):
 """Fixed-shape GPU predicate; no dynamic indices or host decision per step.

 Only initializing rows affect acceptance; controlled rows remain false.
 """
 v=velocity;w=angular;dq=joint_velocity[:,settlement_joint_ids]
 rb=root_bounds;cb=chassis_bounds
 good=((v.norm(dim=-1)<=config['settled_linear_speed_max_mps'])&
       (w.norm(dim=-1)<=config['settled_angular_speed_max_rad_s'])&
       (dq.abs().amax(-1)<=config['settled_joint_speed_max_rad_s'])&
       (torch.acos(up.clamp(-1,1))<=config['tilt_max_rad'])&
       (height>=rb[:,0])&(height<=rb[:,1])&
       (chassis>=cb[:,0])&(chassis<=cb[:,1])&~fault&~restricted)
 return good&initializing


class GeometryGaps:
 """Batch all authored collider types, including every severe-penetration check."""
 def __init__(self,geometry,body_ids,device,dtype):
  self.ids=torch.tensor([body for _,body,_ in geometry],device=device)
  self.rotation=torch.stack([matrix[:3,:3] for _,_,matrix in geometry])
  self.translation=torch.stack([matrix[:3,3] for _,_,matrix in geometry])
  self.kind=torch.tensor([{'Cube':0,'Sphere':1,'Capsule':2,'Cylinder':3}[shape['type']] for shape,_,_ in geometry],device=device)
  self.radius=torch.tensor([shape.get('radius',0.) for shape,_,_ in geometry],device=device,dtype=dtype)
  self.size=torch.tensor([shape.get('size',0.)/2 for shape,_,_ in geometry],device=device,dtype=dtype)
  self.height=torch.tensor([shape.get('height',0.)/2 for shape,_,_ in geometry],device=device,dtype=dtype)
  self.axis=torch.tensor(['XYZ'.index(shape.get('axis','Z')) for shape,_,_ in geometry],device=device)
  self.chassis=torch.tensor([i for i,(shape,_,_) in enumerate(geometry) if shape['body']=='robot'],device=device)
  if not len(self.chassis):raise ValueError('no chassis colliders')
 def measure(self,positions,quaternions,quat_inverse,ground):
  q=quaternions[:,self.ids];z=q.new_tensor([0.,0.,1.]).expand(*q.shape[:-1],3)
  n=quat_inverse(q,z);normal=torch.einsum('nsi,sij->nsj',n,self.rotation)
  center=(n*self.translation).sum(-1)+positions[:,self.ids,2]
  axis=normal.gather(-1,self.axis[None,:,None].expand(len(q),-1,1)).squeeze(-1)
  radial=normal.masked_fill(torch.nn.functional.one_hot(self.axis,3).bool()[None],0.).norm(dim=-1)
  extent=torch.where(self.kind==0,normal.abs().sum(-1)*self.size,
    torch.where(self.kind==1,normal.norm(dim=-1)*self.radius,
     axis.abs()*self.height+torch.where(self.kind==2,normal.norm(dim=-1),radial)*self.radius))
  gaps=center-extent-ground
  return gaps,gaps[:,self.chassis].amin(-1)


def diagnostic_snapshot(env,values,slot='_motion_fault_buffers'):
 """Reuse allocations; keep full dynamic pre-reset evidence and immutable limits view."""
 buffers=getattr(env,slot,None)
 if buffers is None:buffers={};setattr(env,slot,buffers)
 for key,value in values.items():
  if key=='limits':
   if key not in buffers:buffers[key]=value # read-only expanded immutable joint limits
  elif key not in buffers:buffers[key]=value.clone()
  else:buffers[key].copy_(value)
 return buffers
