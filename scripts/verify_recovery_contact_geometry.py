"""CPU USD geometry/read-only runtime-API audit, not physical slip certification."""
import argparse,json
from pathlib import Path
import numpy as np
from native_stand.students.data import sha
ASSET_SHA='32a56efc5ee677a7ae3b5e59c3848a141276ec4957757acde239b774b0c901b7'

def verify(asset):
 from pxr import Usd,UsdGeom,UsdPhysics
 asset=Path(asset)
 if sha(asset)!=ASSET_SHA:raise ValueError('contact geometry asset identity differs')
 stage=Usd.Stage.Open(str(asset));cache=UsdGeom.XformCache();rows=[]
 for side in ('left','right'):
  body=stage.GetPrimAtPath('/rm_train_ground/'+side+'_wheel');cylinders=[]
  for prim in Usd.PrimRange(body):
   if prim.IsA(UsdGeom.Cylinder) and prim.HasAPI(UsdPhysics.CollisionAPI):cylinders.append(prim)
  if len(cylinders)!=1:raise ValueError('expected exactly one main wheel cylinder collider')
  prim=cylinders[0];g=UsdGeom.Cylinder(prim);local=np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse())
  axis=local[2,:3];center=local[3,:3]
  if g.GetAxisAttr().Get()!='Z' or abs(g.GetRadiusAttr().Get()-.06)>1e-9 or abs(g.GetHeightAttr().Get()-.033)>1e-9 or np.linalg.norm(center)>1e-6 or abs(abs(axis[0])-1)>1e-6:raise ValueError('wheel cylinder frame/radius/center differs')
  joint=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/rm_train_ground/joints/'+side+'_wheel_spin'))
  if joint.GetAxisAttr().Get()!='X':raise ValueError('wheel revolute axis differs')
  rows.append({'side':side,'collider':str(prim.GetPath()),'radius_m':.06,'half_width_m':.0165,'center_in_link_m':center.tolist(),'cylinder_axis_in_link':axis.tolist(),'revolute_positive_axis':'X','COM_in_link_m':list(UsdPhysics.MassAPI(body).GetCenterOfMassAttr().Get())})
 return {'asset':str(asset.resolve()),'asset_sha256':ASSET_SHA,'geometry_verified':True,'wheels':rows,
  'velocity_contract':'body_com_pos_w + body_com_lin_vel_w + body_com_ang_vel_w, world SI; xyzw body_link_quat_w; wheel center body_link_pos_w',
  'scope':'CPU authored geometry only; exact rigid support point approximation, not PhysX contact patch or hardware slip certification',
  'physical_signed_readback':'NOT_RUN','slip_reward':'DISABLED'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError(out)
 result=verify(a.asset);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
