"""Export exact small Recovery4100 Actor, using its own verified saved inputs."""
import argparse,copy,json,zipfile
from pathlib import Path
import numpy as np
import torch
from native_stand.students.data import sha
from native_stand.students.model import load
from native_stand.students.export import DeterministicActor
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'runs/2026_10_15-22_57__recovery31_swing_broad_narrow_10000/formal/model_update_4100.pt'
EXPECTED='66243a253e7f1d595d2950f5686ec6f715fd6d79edd481c52bb34abcb36a4d3a'
SPEC=CHECKPOINT.parent/'source_snapshot/spec.json'
EVIDENCE=ROOT/'runs/2026_10_17-11_05__recovery31_saved_checkpoints_ppo0'

def export(output):
 import onnx
 from onnx.reference import ReferenceEvaluator
 output=Path(output).resolve()
 if output.exists():raise FileExistsError(output)
 if sha(CHECKPOINT)!=EXPECTED:raise ValueError('original small4100 SHA differs')
 actor,cp=load(CHECKPOINT,'RecoveryStudent');spec=json.loads(SPEC.read_text())
 if cp['spec_sha256']!=sha(SPEC) or cp['asset_sha256']!=spec['asset_sha256']:raise ValueError('checkpoint frozen identity differs')
 if [m.out_features for m in actor.mlp if isinstance(m,torch.nn.Linear)]!=[128,64,32,6]:raise ValueError('small network required')
 xs=[];categories=[];sources={str(SPEC):sha(SPEC)}
 for i in range(4,8):
  ep=EVIDENCE/f'evaluation_{i:02}.json';d=json.loads(ep.read_text());tp=EVIDENCE/d['trace']['path']
  if d['checkpoint_sha256']!=EXPECTED or sha(tp)!=d['trace']['sha256'] or not all(e['accepted'] for e in d['per_env']):raise ValueError('source4100 trace identity/capability differs')
  sources.update({str(ep):sha(ep),str(tp):sha(tp)})
  with np.load(tp) as a:
   controlled=a['actor_valid'][...,0].astype(bool)&a['first_episode_valid'].astype(bool)
   pts=np.argwhere(controlled);chosen=pts[np.linspace(0,len(pts)-1,64).astype(int)]
   for t,e in chosen:xs.append(a['actor_input'][t,e]);categories.append('actual4100_full_recovery')
   for e,row in enumerate(d['per_env']):
    for t in [0,max(0,row['entry_index']-1),row['entry_index'],min(len(a['state'])-1,row['entry_index']+1)]:
     xs.append(a['actor_input'][t,e]);categories.append('actual_reset_or_entry_boundary')
 # Additional real failed independent-Recovery states, when fields and role match.
 failed=ROOT/'runs/2026_10_17-13_34__recovery31_large_stop_gpu1_resume/formal/evaluation_00.json'
 if failed.exists():
  d=json.loads(failed.read_text());tp=failed.parent/d['trace']['path']
  if sha(tp)!=d['trace']['sha256'] or d['case']['role']!='RecoveryStudent':raise ValueError('failure diagnostic provenance differs')
  with np.load(tp) as a:
   pts=np.argwhere(a['actor_valid'][...,0].astype(bool)&a['first_episode_valid'].astype(bool))
   for t,e in pts[np.linspace(0,len(pts)-1,64).astype(int)]:xs.append(a['actor_input'][t,e]);categories.append('actual_other_recovery_failure_input_not4100_trajectory')
  sources.update({str(failed):sha(failed),str(tp):sha(tp)})
 from native_stand.students.mcu_interface import encode
 g=torch.Generator().manual_seed(20261017);n=32;gravity=torch.randn(n,3,generator=g);gravity/=gravity.norm(dim=-1,keepdim=True)
 q=(torch.rand(n,6,generator=g)*2-1)*1.5;dq=(torch.rand(n,6,generator=g)*2-1)*100
 boundary=encode(torch.randn(n,3,generator=g)*8,gravity,q,dq,q[:,[0,1,3,4]],(torch.rand(n,2,generator=g)*2-1)*100,torch.zeros(n,6),'Recovery',torch.tensor([[0.,0.,.34]]).expand(n,-1)).numpy()
 xs.extend(boundary);categories+=['synthetic_encoding_boundary_not_capability']*n
 x=np.asarray(xs,np.float32)
 if not np.isfinite(x).all() or not np.all(x[:,25:31]==[0,1,0,1,0,0]) or not np.allclose(x[:,22:25],[0,0,.68]):raise ValueError('Recovery command/task inputs differ')
 torch.set_num_threads(4);model=DeterministicActor(actor).eval();output.mkdir(parents=True,exist_ok=False)
 with torch.inference_mode():
  torch.onnx.export(model,torch.zeros(1,31),str(output/'actor.onnx'),input_names=['observation'],output_names=['mean_action'],opset_version=17,dynamo=False)
  expected=model(torch.from_numpy(x)).numpy()
 graph=onnx.load(output/'actor.onnx');onnx.checker.check_model(graph,full_check=True)
 for field,shape in [(graph.graph.input[0],[1,31]),(graph.graph.output[0],[1,6])]:
  if [d.dim_value for d in field.type.tensor_type.shape.dim]!=shape:raise ValueError('fixed ONNX shape differs')
 evaluator=ReferenceEvaluator(graph);actual=np.concatenate([evaluator.run(None,{'observation':v[None]})[0] for v in x]);error=np.abs(expected-actual)
 parameters=sum(p.numel() for p in model.parameters());mac=sum(m.in_features*m.out_features for m in model.mlp if isinstance(m,torch.nn.Linear))
 if parameters!=14630 or mac!=14400 or any('log_std' in k for k in model.state_dict()):raise ValueError('unexpected deployed actor content')
 np.savez_compressed(output/'golden_vectors.npz',observation=x,mean_action=expected,onnx_mean=actual,category=np.array(categories))
 golden={'input_name':'observation','output_name':'mean_action','vectors':[{'category':c,'observation':v.tolist(),'mean_action':a.tolist()} for c,v,a in zip(categories,x,expected)]}
 (output/'golden_vectors.json').write_text(json.dumps(golden,indent=2)+'\n')
 # Same coordinate/target/PD adapter as Motion31; only explicit skill constant changes.
 adapter=(ROOT/'deployment_protocol/motion31_reference/reference_adapter.py').read_text().replace('Motion31 NumPy reference','Recovery31/4100 NumPy reference').replace('np.array([1,0,0,1,0,0],np.float32)','np.array([0,1,0,1,0,0],np.float32)')
 adapter=adapter.replace("cmd=vector(command,3,'command')", "cmd=vector(command,3,'command')\n if not np.allclose(cmd,[0.,0.,.34],atol=1e-6,rtol=0):raise ValueError('Recovery4100 requires zero vx/yaw and height .34m')")
 (output/'reference_adapter.py').write_text(adapter)
 errors={'normalized_max_abs':float(error.max()),'per_channel_max_abs':error.max(0).tolist(),'leg_target_max_abs_rad':float(error[:,[0,1,3,4]].max()*1.5),'wheel_target_max_abs_rad_s':float(error[:,[2,5]].max()*100)}
 report={'version':'recovery31_small4100_onnx_v1','checkpoint':str(CHECKPOINT),'checkpoint_sha256':EXPECTED,'spec':str(SPEC),'spec_sha256':sha(SPEC),'asset_sha256':spec['asset_sha256'],'input_name':'observation','output_name':'mean_action','input_shape':[1,31],'output_shape':[1,6],'dtype':'float32','network':[31,128,64,32,6],'hidden_activation':'ELU(alpha=1)','output':'deterministic tanh mean, included; do not tanh again','parameters':parameters,'FP32_parameter_bytes':parameters*4,'linear_MAC':mac,'onnx_bytes':(output/'actor.onnx').stat().st_size,'operators':sorted(set(n.op_type for n in graph.graph.node)),'onnx_sha256':sha(output/'actor.onnx'),'numerical_backend':'onnx.reference.ReferenceEvaluator float32 CPU','numerical_pass':bool(error.max()<=1e-5),'tolerance':1e-5,'errors':errors,'golden_count':len(x),'vector_sources':sources,'fixed_command':[0,0,.34],'skill_jump_onehot':[0,1,0,1,0,0],'action_scale':{'leg_rad':1.5,'wheel_rad_s':100},'hardware_status':'HARDWARE_UNVERIFIED','excluded':['Critic','optimizer','Gaussian std','RNG','history','reward'],'source_unchanged':sha(CHECKPOINT)==EXPECTED}
 (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 if not report['numerical_pass']:raise ValueError('ONNX numerical tolerance failed; preserved report')
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args();out=a.output or ROOT/'deployment_protocol'/new_experiment_name('recovery31_small4100_onnx');print(json.dumps(export(out),indent=2))
