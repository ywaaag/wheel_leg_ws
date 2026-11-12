"""Read-only USD/MJCF/readback comparison. Compiles models but never steps physics."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdPhysics,UsdGeom
import mujoco

def align_joint_damping(xml,readback):
    """Change only named hinge damping to saved explicit actuator coefficients."""
    import xml.etree.ElementTree as ET
    names=readback['joint_names'];wanted={}
    for group,c in readback['control']['actuators'].items():
        for index,kd in zip(c['joint_ids'],c['damping']):
            name=names[index]
            if name in wanted:raise ValueError('duplicate damping joint')
            wanted[name]=(group,float(kd))
    tree=ET.fromstring(xml);changes=[];seen=set()
    for joint in tree.findall('.//joint'):
        name=joint.get('name')
        if name not in wanted:continue
        group,kd=wanted[name];seen.add(name)
        changes.append(dict(joint=name,group=group,reference_attribute=joint.get('damping'),aligned_damping=kd))
        joint.set('damping',format(kd,'.16g'))
    if seen!=set(wanted):raise ValueError('damping readback joints absent from MJCF')
    return ET.tostring(tree,encoding='unicode'),changes

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inertia(diag,quat_wxyz):
    q=np.asarray(quat_wxyz);r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    return r@np.diag(diag)@r.T

def audit(spec_path,readback_path,xml_path):
    from validate_recovery31_mujoco import xml_with_native_limits
    spec=json.loads(spec_path.read_text());rb=json.loads(readback_path.read_text())
    usd=Path(spec['asset_path']);assert digest(usd)==spec['asset_sha256']
    xml,changes=xml_with_native_limits(xml_path,usd)
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    stage=Usd.Stage.Open(str(usd));cache=UsdGeom.XformCache()
    prims={p.GetName():p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    bodies=[]
    for i in range(1,model.nbody):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i)
        if name not in prims:continue
        p=prims[name];mass=UsdPhysics.MassAPI(p);diag=mass.GetDiagonalInertiaAttr().Get();axes=mass.GetPrincipalAxesAttr().Get()
        if diag is None or axes is None:continue
        ui=inertia(diag,[axes.GetReal(),*axes.GetImaginary()]);mi=inertia(model.body_inertia[i],model.body_iquat[i])
        # Both tensors expressed in each body's authored link frame, not principal-value order.
        com=np.array(mass.GetCenterOfMassAttr().Get())
        transform=cache.GetLocalToWorldTransform(p)
        rotation=np.array(transform.ExtractRotationMatrix()).T
        bodies.append(dict(body=name,usd_mass_kg=float(mass.GetMassAttr().Get()),mj_mass_kg=float(model.body_mass[i]),com_error_m=float(np.linalg.norm(com-model.body_ipos[i])),usd_com=com.tolist(),mj_com=model.body_ipos[i].tolist(),usd_inertia_link=ui.tolist(),mj_inertia_link=mi.tolist(),inertia_relative_frobenius=float(np.linalg.norm(ui-mi)/np.linalg.norm(ui)),zero_pose_world_position_error_m=float(np.linalg.norm(np.array(transform.ExtractTranslation())-data.xpos[i])),zero_pose_world_rotation_error=float(np.linalg.norm(rotation-data.xmat[i].reshape(3,3)))))
    groups={}
    for name,g in rb['control']['actuators'].items():
        for idx,kd in zip(g['joint_ids'],g['damping']):groups[rb['joint_names'][idx]]=(name,kd)
    usdj={p.GetName():p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    joints=[]
    for j in range(model.njnt):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j)
        if name not in usdj:continue
        p=usdj[name];dof=model.jnt_dofadr[j];group,kd=groups.get(name,('UNKNOWN',None));a=p.GetAttribute('physxJoint:armature').Get()
        joints.append(dict(joint=name,group=group,native_explicit_actuator_damping=kd,mj_mechanical_damping=float(model.dof_damping[dof]),usd_authored_armature=a,mj_armature=float(model.dof_armature[dof]),mj_frictionloss=float(model.dof_frictionloss[dof]),usd_joint_friction=p.GetAttribute('physxJoint:jointFriction').Get()))
    collisions=[]
    for p in stage.Traverse():
        if not p.HasAPI(UsdPhysics.CollisionAPI):continue
        collisions.append(dict(path=str(p.GetPath()),type=p.GetTypeName(),attributes={a.GetName():str(a.Get()) for a in p.GetAttributes() if any(k in a.GetName() for k in ['radius','height','size','axis','contactOffset','restOffset','friction','approximation'])}))
    mjgeoms=[dict(name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i),body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i])),type=int(model.geom_type[i]),size=model.geom_size[i].tolist(),friction=model.geom_friction[i].tolist(),condim=int(model.geom_condim[i]),margin=float(model.geom_margin[i]),solref=model.geom_solref[i].tolist(),solimp=model.geom_solimp[i].tolist(),contype=int(model.geom_contype[i]),conaffinity=int(model.geom_conaffinity[i])) for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    paths=[spec_path,readback_path,xml_path,usd,Path(__file__),Path(__file__).with_name('validate_recovery31_mujoco.py'),Path(__file__).with_name('validate_native_mujoco.py')]
    for k in ['module_path','yaml_path']:
        p=Path(rb['asset'][k]);expected=rb['asset'][k.replace('_path','_sha256')]
        assert digest(p)==expected,'asset module/YAML drift';paths.append(p)
    return dict(version='recovery_usd_mjcf_static_parity_v1',simulation_steps=0,optimizer_steps=0,sources_sha256={str(p.resolve()):digest(p) for p in paths},scope='USD authored properties and saved runtime actuator/ground readback vs compiled MJCF; no live PhysX inertia/armature/collider readback',bodies=bodies,joints=joints,usd_collisions=collisions,mj_collisions=mjgeoms,native_ground_readback=rb['ground'],native_control=spec['control'],limit_changes=changes,total_compared_mass_kg={'usd':sum(b['usd_mass_kg'] for b in bodies),'mj':sum(b['mj_mass_kg'] for b in bodies)},mj_solver=dict(timestep=model.opt.timestep,iterations=model.opt.iterations,integrator=int(model.opt.integrator),solver=int(model.opt.solver)),unknown=['PhysX cooked/effective inertia and mass not recorded','PhysX effective per-joint armature/friction/velocity caps not recorded','PhysX effective collider offsets/material-combine not fully recorded'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--readback',type=Path,required=True);p.add_argument('--xml',type=Path,default=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=audit(a.spec,a.readback,a.xml);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(bodies=len(result['bodies']),joints=len(result['joints']),mass=result['total_compared_mass_kg'],top_inertia_errors=sorted([(b['body'],b['inertia_relative_frobenius']) for b in result['bodies']],key=lambda x:-x[1])[:8]),indent=2))

if __name__=='__main__':main()
