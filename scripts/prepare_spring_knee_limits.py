"""Offline spring-distance to knee-angle limits on the zero-connected branch."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import brentq
from pxr import Usd,UsdPhysics
from audit_recovery_same_state import joint_tree,transform

def main():
 p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--usd',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 original_sha=hashlib.sha256(a.usd.read_bytes()).hexdigest();g=json.loads(a.geometry.read_text());stage=Usd.Stage.Open(str(a.usd));tree=joint_tree(stage);rows=[]
 for spring in g['springs']:
  side=spring['side'];name=side+'_knee_pitch';_,parent,child,l0,l1,axis=next(j for j in tree if j[0]==name)
  if [x['body'].rsplit('/',1)[-1] for x in spring['anchors']]!=[child,parent]:raise ValueError('anchors not across direct knee joint')
  A,B=[np.array(x['local_m']) for x in spring['anchors']]
  def length(q):
   vec=np.zeros(3);vec[axis]=q;t=l0@transform([0,0,0],Rotation.from_rotvec(vec).as_matrix())@np.linalg.inv(l1)
   return float(np.linalg.norm(B-(t[:3,:3]@A+t[:3,3])))
  prim=next(p for p in stage.Traverse() if p.GetName()==name);joint=UsdPhysics.RevoluteJoint(prim);old=np.deg2rad([joint.GetLowerLimitAttr().Get(),joint.GetUpperLimitAttr().Get()]);q=np.linspace(*old,4001)
  valid=np.array([.155<=length(v)<=.215 for v in q]);zero=np.argmin(abs(q));assert valid[zero]
  low=zero;high=zero
  while low>0 and valid[low-1]:low-=1
  while high<len(q)-1 and valid[high+1]:high+=1
  limits=[float(old[0]),float(old[1])]
  if low>0:limits[0]=brentq(lambda v:length(v)-(.155 if length(q[low-1])<.155 else .215),q[low-1],q[low])
  if high<len(q)-1:limits[1]=brentq(lambda v:length(v)-(.155 if length(q[high+1])<.155 else .215),q[high],q[high+1])
  check=np.array([length(v) for v in np.linspace(*limits,2001)])
  assert check.min()>=.155-1e-9 and check.max()<=.215+1e-9
  rows.append(dict(joint=name,parent=parent,child=child,prim=str(prim.GetPath()),old_limits_rad=old.tolist(),candidate_limits_rad=limits,candidate_limits_deg=np.rad2deg(limits).tolist(),length_range_m=[float(check.min()),float(check.max())],branch='continuous feasible component containing q=0; nonmonotonic remote component excluded'))
 a.output.mkdir(parents=True)
 candidate=a.output/'robot_spring_knee_limits_candidate.usd';stage.Export(str(candidate));dst=Usd.Stage.Open(str(candidate))
 for row in rows:
  j=UsdPhysics.RevoluteJoint(dst.GetPrimAtPath(row['prim']));j.GetLowerLimitAttr().Set(row['candidate_limits_deg'][0]);j.GetUpperLimitAttr().Set(row['candidate_limits_deg'][1])
 dst.GetRootLayer().Save()
 for row in rows:
  j=UsdPhysics.RevoluteJoint(dst.GetPrimAtPath(row['prim']));row['saved_limits_rad']=np.deg2rad([j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get()]).tolist()
 assert hashlib.sha256(a.usd.read_bytes()).hexdigest()==original_sha
 payload=dict(version='gas_spring_knee_solver_limit_candidate_v1',source_usd=str(a.usd),source_usd_sha256=original_sha,geometry=str(a.geometry),geometry_sha256=hashlib.sha256(a.geometry.read_bytes()).hexdigest(),candidate_usd=str(candidate),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),rows=rows,status='OFFLINE_CANDIDATE_NOT_PHYSICS_VALIDATED',gas_force='350N constant inside stroke, zero above extension; solver knees enforce branch stops independently',limitations='2D fixed anchor model; fixed upper/lower knee topology assumed; preserves old extension-side limit; no new training spec or runtime gas hook enabled',simulation_steps=0)
 (a.output/'manifest.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
