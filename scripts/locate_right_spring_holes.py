"""Mirror is only a search prior; validate right-side holes against actual mesh."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Gf

def fit_circle_near(points,hint,seed=18):
    # Plate thickness is local X; only search within 8mm in the YZ plane.
    u=np.unique(np.round(points[:,1:],8),axis=0)
    u=u[np.linalg.norm(u-hint[1:],axis=1)<.008]
    if len(u)<12:raise ValueError('insufficient nearby vertices')
    rng=np.random.default_rng(seed);best=None
    for _ in range(5000):
        q=u[rng.choice(len(u),3,replace=False)];M=2*(q[1:]-q[0])
        if abs(np.linalg.det(M))<1e-12:continue
        c=np.linalg.solve(M,(q[1:]**2).sum(1)-(q[0]**2).sum());r=np.linalg.norm(q[0]-c)
        if abs(r-.003)>.0002 or np.linalg.norm(c-hint[1:])>.004:continue
        mask=np.abs(np.linalg.norm(u-c,axis=1)-r)<2e-6
        if best is None or mask.sum()>best.sum():best=mask
    if best is None:raise ValueError('no expected 6mm hole near mirrored hint')
    v=u[best];coeff=np.linalg.lstsq(np.c_[2*v,np.ones(len(v))],(v*v).sum(1),rcond=None)[0];c=coeff[:2];r=np.sqrt(coeff[2]+c@c)
    angles=np.sort(np.arctan2(v[:,1]-c[1],v[:,0]-c[0]));coverage=2*np.pi-np.diff(np.r_[angles,angles[0]+2*np.pi]).max()
    residual=np.abs(np.linalg.norm(v-c,axis=1)-r).max()
    if len(v)<24 or coverage<np.deg2rad(330) or residual>2e-6:raise ValueError('hole coverage/residual insufficient')
    mask=np.abs(np.linalg.norm(points[:,1:]-c,axis=1)-r)<2e-6
    return dict(center_yz_m=c.tolist(),radius_m=float(r),unique_points=len(v),coverage_deg=float(np.rad2deg(coverage)),max_radial_error_m=float(residual),rim_x_layers_m=np.unique(np.round(points[mask,0],8)).tolist())

def main():
    p=argparse.ArgumentParser();p.add_argument('--left-fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    d=json.loads(a.left_fit.read_text());usd=Path(d['usd'])
    if hashlib.sha256(usd.read_bytes()).hexdigest()!=d['usd_sha256']:raise ValueError('marked USD changed')
    s=Usd.Stage.Open(str(usd));cache=UsdGeom.XformCache();root=cache.GetLocalToWorldTransform(s.GetPrimAtPath('/rm_train_ground/robot'));rows=[]
    for lr in d['rows']:
        name=lr['declared_body'].rsplit('/',1)[-1].replace('left_','right_');bp=s.GetPrimAtPath('/rm_train_ground/'+name)
        if not bp.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError('right RigidBody absent')
        bt=cache.GetLocalToWorldTransform(bp)
        # Use fitted left YZ and plate midpoint, not inaccurate marker depth.
        lb=cache.GetLocalToWorldTransform(s.GetPrimAtPath(lr['declared_body']))
        lc=[float(np.mean(lr['rim_body_x_layers_m'])),*lr['fitted_center_body_yz_m']]
        rootpt=np.array(root.GetInverse().Transform(lb.Transform(Gf.Vec3d(*lc))));rootpt[0]*=-1
        mirrored_world=root.Transform(Gf.Vec3d(*rootpt));hint=np.array(bt.GetInverse().Transform(mirrored_world))
        mesh=next(x for x in Usd.PrimRange(bp) if x.IsA(UsdGeom.Mesh));mat=cache.GetLocalToWorldTransform(mesh)*bt.GetInverse()
        pts=np.array([mat.Transform(Gf.Vec3d(*v)) for v in UsdGeom.Mesh(mesh).GetPointsAttr().Get()]);fit=fit_circle_near(pts,hint)
        center=[float(np.mean(fit['rim_x_layers_m'])),*fit['center_yz_m']];world=bt.Transform(Gf.Vec3d(*center))
        rows.append(dict(left_marker=lr['marker'],body=str(bp.GetPath()),mesh=str(mesh.GetPath()),mirrored_hint_local_m=hint.tolist(),fit=fit,plate_midpoint_local_m=center,plate_midpoint_world_m=list(world),planar_mirror_error_m=float(np.linalg.norm(np.array(fit['center_yz_m'])-hint[1:])),plate_midpoint_mirror_error_m=float(np.linalg.norm(np.array(world)-np.array(mirrored_world)))))
    # Same planar approximation as left: common side plane; fixed local anchors.
    cad=[np.array(root.GetInverse().Transform(Gf.Vec3d(*r['plate_midpoint_world_m']))) for r in rows];plane=float(np.mean([q[0] for q in cad]))
    for r,q in zip(rows,cad):
        q[0]=plane;world=root.Transform(Gf.Vec3d(*q));bt=cache.GetLocalToWorldTransform(s.GetPrimAtPath(r['body']))
        r['planar_anchor_local_m']=list(bt.GetInverse().Transform(world));r['planar_anchor_reference_world_m']=list(world)
    result=dict(version='gas_spring_right_mesh_verified_v1',usd=str(usd),usd_sha256=d['usd_sha256'],left_fit_sha256=hashlib.sha256(a.left_fit.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),mirror_plane='root CAD X=0 lateral symmetry, transformed to world; not world Z plane',search_radius_yz_m=.008,rows=rows,right_common_plane_root_cad_x_m=plane,reference_length_m=float(np.linalg.norm(np.array(rows[0]['planar_anchor_reference_world_m'])-rows[1]['planar_anchor_reference_world_m'])),hardware={'count_left':1,'count_right':1,'manufacturer':'扬州市宏翔气弹簧','marking':'93 * 60 * 350N','extended_center_distance_m':[.20,.21],'compressed_center_distance_m_approx':.16,'source':'user manual measurement and marking, 2026-10-18','stroke_inferred_range_m':[.04,.05],'marking_60_meaning':'UNCONFIRMED; differs from manually inferred 40-50mm stroke','force_curve':'UNMEASURED; 350N marked nominal, position/temperature unspecified'},runtime_enabled=False,simulation_steps=0,limitations='actual right mesh fitted; real hardware side/depth/rod-end offsets unmeasured; not gas-spring force/stroke certification')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
