"""CPU-only authored-geometry equivalence and backend dependency audit; no Isaac."""
import ast,json,hashlib,time
from pathlib import Path
import torch
from types import SimpleNamespace as NS
from native_stand.students.motion_runtime_reduction import GeometryGaps
from native_stand.command_v6.domain.experiment_naming import new_experiment_name


def inverse(q,v):
 # Synthetic unit quaternions WXYZ, same operation used for original/batched math.
 xyz=q[...,1:];w=q[...,:1];return v-2*w*torch.cross(xyz,v,dim=-1)+2*torch.cross(xyz,torch.cross(xyz,v,dim=-1),dim=-1)


def audit(spec):
 torch.set_num_threads(4);torch.manual_seed(20261020);s=json.loads(Path(spec).read_text());library=json.loads(Path(s['recovery']['pose_library']).read_text());shapes=library['collision_geometry'];names=list(dict.fromkeys(shape['body'] for shape in shapes));ids={n:i for i,n in enumerate(names)}
 geometry=[(shape,ids[shape['body']],torch.tensor(shape['matrix'],dtype=torch.float32)) for shape in shapes]
 count=128;q=torch.randn(count,len(names),4);q=q/q.norm(dim=-1,keepdim=True);p=torch.randn(count,len(names),3)
 env=NS(cfg=NS(motion_runtime_reduction=False,ground_height_m=0.),geometry=geometry,device='cpu',num_envs=count,robot=NS(data=NS(body_pos_w=NS(torch=p),body_quat_w=NS(torch=q))))
 path=Path('src/native_stand/recovery/runtime.py');tree=ast.parse(path.read_text());cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and any(isinstance(f,ast.FunctionDef) and f.name=='_geometry_gaps' for f in c.body));method=next(f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=='_geometry_gaps');ns={'torch':torch,'quat_apply_inverse':inverse};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'actual_geometry','exec'),ns)
 original=lambda:ns['_geometry_gaps'](env);batched=GeometryGaps(geometry,ids,'cpu',p.dtype);changed=lambda:batched.measure(p,q,inverse,0.)
 a,b=original(),changed();errors=[float((x-y).abs().max()) for x,y in zip(a,b)]
 for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
 timing={}
 for label,fn in [('original',original),('batched',changed)]:
  for _ in range(5):fn()
  start=time.perf_counter()
  for _ in range(30):fn()
  timing[label]=(time.perf_counter()-start)/30
 import importlib.util
 backend=Path(next(iter(importlib.util.find_spec('isaaclab_physx').submodule_search_locations)))
 files=[backend/'assets/articulation/articulation.py',backend/'assets/articulation/articulation_data.py']
 return {'status':'CPU_EQUIVALENCE_PASS','spec':str(spec),'spec_SHA256':hashlib.sha256(Path(spec).read_bytes()).hexdigest(),'colliders':len(shapes),'bodies':len(names),'synthetic_states':count,'quaternion':'synthetic WXYZ; not a captured PhysX trajectory','gap_max_abs_error_m':errors,'CPU_seconds_per_call':timing,'GPU_speedup':'NOT_MEASURED','backend_readonly_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'reward_fusion_review':{'implementation':'existing ordered apply chain unchanged','opportunity':'avoid intermediate cat/reduction/config construction; preserve raw/density/dt/events/masks/gradients','decision':'defer fused runtime until independent numerical/gradient equivalence; do not claim five layers are independent timings'},'backend_execution_review':{'implicit_actuator':'not identity metadata: compute/update_targets/update_actuator_state_model update torque/gear/velocity-limit buffers consumed by reward/Critic; active PD zero does not apply to all passive groups','joint_acc':'ArticulationData.update eagerly computes finite difference; no direct project consumer found, backend/actuator dependency still needs runtime proof','decision':'no installed source edits, no skips of write_data_to_sim/scene.update; explicit local adapter requires separate equivalence experiment'},'simulation':0,'PPO':0,'backend_modified':False}


def main():
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output-dir',type=Path);p.add_argument('--write',action='store_true');a=p.parse_args()
 if not a.write:print(json.dumps({'status':'PLAN_ONLY','files_written':0,'simulation':0,'optimizer':0,'spec':str(a.spec)},indent=2));return
 out=a.output_dir or Path('artifacts')/new_experiment_name('motion_runtime_reduction_cpu');out.mkdir(parents=True,exist_ok=False);r=audit(a.spec);(out/'audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps({'output':str(out),**r},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
