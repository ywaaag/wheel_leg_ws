"""Read-only fit of user-identified planar hole rims, no physics or force model."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Gf

def main():
 p=argparse.ArgumentParser();p.add_argument('--usd',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 stage=Usd.Stage.Open(str(a.usd));cache=UsdGeom.XformCache();rng=np.random.default_rng(18);rows=[]
 for marker,body in [('spring_A','left_lower_leg'),('spring_B','left_upper_leg')]:
  m=stage.GetPrimAtPath('/rm_train_ground/'+marker);b=stage.GetPrimAtPath('/rm_train_ground/'+body)
  if not m or not b.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('marker/body absent')
  bt=cache.GetLocalToWorldTransform(b);world=cache.GetLocalToWorldTransform(m).ExtractTranslation();local=np.array(bt.GetInverse().Transform(world))
  mesh=next(x for x in Usd.PrimRange(b) if x.IsA(UsdGeom.Mesh));mat=cache.GetLocalToWorldTransform(mesh)*bt.GetInverse()
  pts=np.array([mat.Transform(Gf.Vec3d(*v)) for v in UsdGeom.Mesh(mesh).GetPointsAttr().Get()])
  u=np.unique(np.round(pts[:,1:],8),axis=0);u=u[np.linalg.norm(u-local[1:],axis=1)<.008];best=None
  for _ in range(4000):
   q=u[rng.choice(len(u),3,replace=False)];M=2*(q[1:]-q[0])
   if abs(np.linalg.det(M))<1e-12:continue
   center=np.linalg.solve(M,(q[1:]**2).sum(1)-(q[0]**2).sum());radius=np.linalg.norm(q[0]-center)
   if not .001<radius<.006 or np.linalg.norm(center-local[1:])>.004:continue
   mask=np.abs(np.linalg.norm(u-center,axis=1)-radius)<2e-6
   if best is None or mask.sum()>best[0]:best=(int(mask.sum()),mask)
  if best is None:raise ValueError('no nearby circle')
  v=u[best[1]];coef=np.linalg.lstsq(np.c_[2*v,np.ones(len(v))],(v*v).sum(1),rcond=None)[0];center=coef[:2];radius=np.sqrt(coef[2]+center@center)
  ang=np.sort(np.arctan2(v[:,1]-center[1],v[:,0]-center[0]));coverage=2*np.pi-np.max(np.diff(np.r_[ang,ang[0]+2*np.pi]))
  radial=np.abs(np.linalg.norm(pts[:,1:]-center,axis=1)-radius)<2e-6
  nearx=np.unique(np.round(pts[radial,0],8)).tolist()
  approximate=local.copy();approximate[1:]=center
  rows.append(dict(marker=str(m.GetPath()),declared_body=str(b.GetPath()),mesh=str(mesh.GetPath()),rigid_body_ownership_source='user supplied; marker itself is root-level, not parented to body',marker_world_m=list(world),marker_body_local_m=local.tolist(),hole_axis_body=[1,0,0],fitted_center_body_yz_m=center.tolist(),radius_m=float(radius),unique_rim_points=len(v),angular_coverage_deg=float(np.rad2deg(coverage)),max_radial_residual_m=float(np.abs(np.linalg.norm(v-center,axis=1)-radius).max()),marker_planar_offset_m=float(np.linalg.norm(center-local[1:])),rim_body_x_layers_m=nearx,provisional_point_keep_marker_x_body_m=approximate.tolist(),provisional_point_world_m=list(bt.Transform(Gf.Vec3d(*approximate))),axial_attachment_center_status='UNDETERMINED: mesh rim faces are not necessarily pin/ball joint center'))
 result=dict(version='gas_spring_user_marker_hole_fit_v1',usd=str(a.usd.resolve()),usd_sha256=hashlib.sha256(a.usd.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),rows=rows,marker_distance_m=float(np.linalg.norm(np.array(rows[0]['marker_world_m'])-rows[1]['marker_world_m'])),provisional_distance_keep_marker_x_m=float(np.linalg.norm(np.array(rows[0]['provisional_point_world_m'])-rows[1]['provisional_point_world_m'])),simulation_steps=0,force_model_enabled=False,limitations='YZ-plane circle fit on user identified holes; mesh precision is not manufacturing accuracy; X connection center unresolved; no right-side symmetry assumed')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
