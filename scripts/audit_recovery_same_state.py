"""Offline same-state encoder/Actor/PD and independent USD joint-frame FK audit."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdPhysics

def rot(q):
 return Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix()
def transform(p,r):
 t=np.eye(4);t[:3,:3]=r;t[:3,3]=p;return t
def joint_tree(stage):
 rows=[]
 for p in stage.Traverse():
  if not p.IsA(UsdPhysics.RevoluteJoint):continue
  j=UsdPhysics.RevoluteJoint(p);a=j.GetBody0Rel().GetTargets();b=j.GetBody1Rel().GetTargets()
  if len(a)!=1 or len(b)!=1:raise ValueError('joint body mapping absent')
  rows.append((p.GetName(),str(a[0]).rsplit('/',1)[-1],str(b[0]).rsplit('/',1)[-1],transform(j.GetLocalPos0Attr().Get(),rot(j.GetLocalRot0Attr().Get())),transform(j.GetLocalPos1Attr().Get(),rot(j.GetLocalRot1Attr().Get())),{'X':0,'Y':1,'Z':2}[j.GetAxisAttr().Get()]))
 return rows
def usd_fk(tree,q,root):
 poses={'robot':root};pending=list(tree)
 while pending:
  remaining=[]
  for name,parent,child,l0,l1,axis in pending:
   if parent not in poses:remaining.append((name,parent,child,l0,l1,axis));continue
   v=np.zeros(3);v[axis]=q[name]
   poses[child]=poses[parent]@l0@transform([0,0,0],Rotation.from_rotvec(v).as_matrix())@np.linalg.inv(l1)
  if len(remaining)==len(pending):raise ValueError('USD joint tree disconnected')
  pending=remaining
 return poses

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 torch.set_num_threads(1)
 from validate_native_mujoco import binding,state
 from validate_recovery31_mujoco import observation
 from native_stand.students.interface import frame,targets
 from native_stand.students.mcu_interface import actor_input,task
 from native_stand.students.model import load
 from native_stand.command_v6.domain.contract import effort
 from native_stand.contract import physical_pd
 root=Path(__file__).resolve().parents[1]
 cp=root/'runs/2026_10_17-17_59__recovery31_standing_contact_quality/formal/model_update_1500.pt'
 actor,_=load(cp,'RecoveryStudent');actor.eval();actor.requires_grad_(False)
 directory=root/'sim2sim/2026_10_18-09_59__recovery1500_mujoco'
 model=mujoco.MjModel.from_xml_path(str(directory/'diagnostic_model.xml'));refs=binding(model)
 stage=Usd.Stage.Open(str(root/'assets/diagnostic/rmuc_robot_closed_loop_knee_stop_0233_v2.usd'));tree=joint_tree(stage)
 records=[];files=[cp,directory/'trajectory.npz',directory/'diagnostic_model.xml',root/'assets/diagnostic/rmuc_robot_closed_loop_knee_stop_0233_v2.usd',Path(__file__)]
 z=np.load(directory/'trajectory.npz')
 # Use actual pre-action physical states, including all passive joint coordinates.
 for k in np.linspace(0,len(z['actor_input'])-1,128,dtype=int):
  data=mujoco.MjData(model);data.qpos[:]=z['joint_q_pre'][k];data.qvel[:]=z['joint_dq_pre'][k];mujoco.mj_forward(model,data)
  prev=torch.tensor(z['previous_targets'][k],dtype=torch.float32)[None]
  mx,_=observation(model,data,refs,prev[:,:4],prev[:,4:])
  r=Rotation.from_quat(data.qpos[[4,5,6,3]]).as_matrix()
  # Independent native-style angular production from generalized free-joint body omega.
  angular=torch.tensor(data.qvel[3:6],dtype=torch.float32)[None]
  gravity=torch.tensor(r.T@np.array([0.,0.,-1.]),dtype=torch.float32)[None]
  q=torch.tensor(data.qpos[refs['qpos']],dtype=torch.float32)[None];dq=torch.tensor(data.qvel[refs['dofs']],dtype=torch.float32)[None]
  nx=actor_input(frame(angular,gravity,q,dq,prev[:,:4],prev[:,4:],torch.zeros(1,6)),task('Recovery',torch.tensor([[0.,0.,.34]])))
  with torch.inference_mode():na=actor.mean(nx);ma=actor.mean(mx)
  limits=torch.tensor(model.jnt_range[refs['joints']],dtype=torch.float32)[None]
  for col,j in enumerate(refs['joints']):
   if not model.jnt_limited[j]:limits[:,col]=torch.tensor([-float('inf'),float('inf')])
  nt=targets(na,torch.zeros(1,6),limits);mt=targets(ma,torch.zeros(1,6),limits)
  nr,nc=physical_pd(q,dq,nt['leg_executed'],nt['wheel_executed'],kp=60,kd=2,kv=1)
  mr,mc=effort(q,dq,mt['leg_executed'],mt['wheel_executed'],leg_kp=60,leg_kd=2,wheel_kv=1)
  qs={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j):float(data.qpos[model.jnt_qposadr[j]]) for j in range(1,model.njnt)}
  fk=usd_fk(tree,qs,transform(data.qpos[:3],r));poserr=[];roterr=[]
  for name,t in fk.items():
   bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
   poserr.append(np.linalg.norm(t[:3,3]-data.xpos[bid]));roterr.append(np.linalg.norm(t[:3,:3]-data.xmat[bid].reshape(3,3)))
  records.append(dict(index=int(k),input_max_abs=float((nx-mx).abs().max()),saved_input_max_abs=float((mx-torch.tensor(z['actor_input'][k])[None]).abs().max()),actor_max_abs=float((na-ma).abs().max()),saved_actor_max_abs=float((ma-torch.tensor(z['mean'][k])[None]).abs().max()),leg_target_max_abs_rad=float((nt['leg_executed']-mt['leg_executed']).abs().max()),wheel_target_max_abs_rad_s=float((nt['wheel_executed']-mt['wheel_executed']).abs().max()),requested_pd_max_abs_nm=float((nr-mr).abs().max()),clipped_pd_max_abs_nm=float((nc-mc).abs().max()),fk_position_max_m=float(max(poserr)),fk_rotation_max_fro=float(max(roterr))))
 # Native trace reconstruction checks that exported rows really encode pre-action state.
 native=root/'runs/2026_10_17-17_59__recovery31_standing_contact_quality/formal'
 ev=None
 for f in sorted(native.glob('evaluation_*.json')):
  try:d=json.loads(f.read_text())
  except Exception:continue
  if isinstance(d,dict) and Path(d.get('case',{}).get('checkpoint','')).name==cp.name and d.get('per_env'):
   ev=d;ep=f;break
 if ev is None:raise ValueError('matching native evaluation absent')
 tp=native/ev['trace']['path']
 if hashlib.sha256(tp.read_bytes()).hexdigest()!=ev['trace']['sha256']:raise ValueError('native trace SHA differs')
 files.extend([ep,tp]);nz=np.load(tp);ids=ev['field_schema']['active_indices'];native_rows=[]
 valid=nz['actor_valid'][...,0].astype(bool)&nz['first_episode_valid'].astype(bool);indices=np.argwhere(valid)
 for ix in np.linspace(0,len(indices)-1,128,dtype=int):
  t,e=indices[ix];rp=nz['root_pre'][t,e];rr=Rotation.from_quat(rp[3:7]).as_matrix();av=torch.tensor(rr.T@rp[10:13],dtype=torch.float32)[None];gv=torch.tensor(rr.T@[0.,0.,-1.],dtype=torch.float32)[None]
  q=torch.tensor(nz['joint_q_pre'][t,e,ids])[None];dq=torch.tensor(nz['joint_dq_pre'][t,e,ids])[None];prev=torch.tensor(nz['previous_targets'][t,e])[None]
  x=actor_input(frame(av,gv,q,dq,prev[:,:4],prev[:,4:],torch.zeros(1,6)),task('Recovery',torch.tensor([[0.,0.,.34]])))
  with torch.inference_mode():mean=actor.mean(x)
  native_rows.append(dict(step=int(t),env=int(e),input_error=float((x-torch.tensor(nz['actor_input'][t,e])[None]).abs().max()),actor_error=float((mean-torch.tensor(nz['mean'][t,e])[None]).abs().max())))
 result=dict(version='recovery_same_state_fk_audit_v1',simulation_steps=0,optimizer_steps=0,mujoco_samples=records,native_samples=native_rows,sources_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},limitations=['USD authored joint-frame forward kinematics, not a fresh PhysX dynamics comparison','native pipeline uses source pure frame/task functions and saved root state; no live env constructed','identical-state tests do not establish identical physical transition dynamics'])
 result['maxima']={key:max(v[key] for v in records) for key in records[0] if key!='index'};result['native_maxima']={key:max(v[key] for v in native_rows) for key in ['input_error','actor_error']}
 result['checks']={'same_state_input_actor_pd':all(result['maxima'][key]<=1e-6 for key in ['input_max_abs','actor_max_abs','leg_target_max_abs_rad','wheel_target_max_abs_rad_s','requested_pd_max_abs_nm','clipped_pd_max_abs_nm']),
                   'native_saved_pre_action_replay':max(result['native_maxima'].values())<=1e-5,
                   'nonzero_pose_authored_fk':result['maxima']['fk_position_max_m']<=1e-5 and result['maxima']['fk_rotation_max_fro']<=1e-5}
 result['status']='PASSED' if all(result['checks'].values()) else 'FAILED'
 files.extend([root/'src/native_stand/students/interface.py',root/'src/native_stand/students/mcu_interface.py',root/'src/native_stand/students/model.py',root/'src/native_stand/contract.py',root/'src/native_stand/command_v6/domain/contract.py',Path(__file__).with_name('validate_native_mujoco.py'),Path(__file__).with_name('validate_recovery31_mujoco.py')])
 result['sources_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'maxima':result['maxima'],'native_maxima':result['native_maxima']},indent=2))

if __name__=='__main__':main()
