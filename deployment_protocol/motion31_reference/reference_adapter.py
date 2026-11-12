"""Motion31 NumPy reference: no history, no hardware or inference dependencies."""
import numpy as np
LEG=np.array([0,1,3,4]);WHEEL=np.array([2,5])

def vector(x,n,name):
 a=np.asarray(x,dtype=np.float32)
 if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError(name+' shape/nonfinite')
 return a

def encode(angular_body,gravity_body,q,dq,previous_legs,previous_wheels,command,reference=None):
 gyro=vector(angular_body,3,'gyro');gravity=vector(gravity_body,3,'gravity')
 if not np.isclose(np.linalg.norm(gravity),1.,atol=2e-4):raise ValueError('gravity must be body unit vector')
 q=vector(q,6,'q');dq=vector(dq,6,'dq');ref=np.zeros(6,np.float32) if reference is None else vector(reference,6,'reference')
 pl=vector(previous_legs,4,'actual previous leg target rad');pw=vector(previous_wheels,2,'actual previous wheel target rad/s');cmd=vector(command,3,'command')
 return np.concatenate((gyro*.25,gravity,(q[LEG]-ref[LEG])/1.5,dq*.05,(pl-ref[LEG])/1.5,pw/100.,cmd/np.array([4.,4.,.5],np.float32),np.array([1,0,0,1,0,0],np.float32))).astype(np.float32)

def decode(mean,leg_limits_rad,reference=None):
 m=vector(mean,6,'tanh deterministic mean');ref=np.zeros(6,np.float32) if reference is None else vector(reference,6,'reference')
 limits=np.asarray(leg_limits_rad,dtype=np.float32)
 if limits.shape!=(4,2) or not np.isfinite(limits).all() or (limits[:,0]>limits[:,1]).any():raise ValueError('explicit four joint mechanical limits required')
 a=np.clip(m,-1.,1.);requested=ref[LEG]+1.5*a[LEG];legs=np.clip(requested,limits[:,0],limits[:,1]);wheels=100.*a[WHEEL]
 return {'clipped_action':a,'leg_requested_rad':requested,'leg_target_rad':legs,'wheel_target_rad_s':wheels,
  'previous_legs_next':legs.copy(),'previous_wheels_next':wheels.copy()}

def pd(q,dq,leg_target_rad,wheel_target_rad_s):
 q=vector(q,6,'q');dq=vector(dq,6,'dq');lt=vector(leg_target_rad,4,'leg target');wt=vector(wheel_target_rad_s,2,'wheel target')
 requested=np.zeros(6,np.float32);requested[LEG]=60.*(lt-q[LEG])-2.*dq[LEG];requested[WHEEL]=wt-dq[WHEEL]
 limit=np.array([40,40,5.5,40,40,5.5],np.float32)
 return requested,np.clip(requested,-limit,limit)
