"""Export trained31D Motion actor to a fresh electrical handoff package, CPU only."""
import argparse,copy,ctypes,importlib.util,json,shutil,subprocess,tempfile,zipfile
from pathlib import Path
import numpy as np
import torch
from native_stand.students.data import sha
from native_stand.students.model import load
from native_stand.students.export import DeterministicActor
ROOT=Path(__file__).resolve().parents[1]

def export_accepted(checkpoint,spec_path,evaluation_dir,diagnostic_result,output):
 """Actor-only export from a complete accepted matrix; no BC/PPO or hardware.

 Uses the existing deterministic wrapper and ONNX reference backend. Network
 width is read from the strictly loaded checkpoint, never inferred from input
 length. The historical49D handoff path below retains its original contract.
 """
 import onnx
 from onnx.reference import ReferenceEvaluator
 checkpoint,spec_path,evaluation_dir,diagnostic_result,output=[Path(p).resolve() for p in
     (checkpoint,spec_path,evaluation_dir,diagnostic_result,output)]
 if output.exists():raise FileExistsError(output)
 actor,saved=load(checkpoint,'MotionZeroStudent');actor.eval().requires_grad_(False)
 spec=json.loads(spec_path.read_text());digest=sha(checkpoint)
 if saved['interface_version']!='multiskill_mcu_v2' or saved['spec_sha256']!=sha(spec_path) or saved['asset_sha256']!=spec['asset_sha256']:
  raise ValueError('exact accepted Motion31 source/spec identity required')
 if spec['action']['leg_scale_rad']!=1.5 or spec['action']['wheel_scale_rad_s']!=100:
  raise ValueError('unsupported deployment action scale')
 candidates=json.loads((evaluation_dir/'candidate_registry.json').read_text())
 accepted=next((r for r in candidates if r.get('sha256')==digest and r.get('checkpoint')==str(checkpoint)),None)
 if not accepted or accepted.get('complete_matrix') is not True or not accepted.get('per_env') or not all(e.get('accepted') is True for e in accepted['per_env']):
  raise ValueError('same-checkpoint complete accepted matrix required')
 evidence={str(checkpoint):digest,str(spec_path):sha(spec_path)};commands=[]
 for r in accepted['evaluations']:
  p=evaluation_dir/r['path']
  if sha(p)!=r['sha256']:raise ValueError('accepted evaluation SHA differs')
  raw=json.loads(p.read_text());trace=p.parent/raw['trace']['path']
  if (sha(trace)!=raw['trace']['sha256'] or raw['checkpoint_sha256']!=digest or
      raw['asset_sha256']!=spec['asset_sha256'] or raw['spec_sha256']!=sha(spec_path) or
      len(raw['per_env'])!=8 or not all(e['accepted'] is True for e in raw['per_env'])):
   raise ValueError('accepted raw trace/source/scoring identity differs')
  evidence.update({str(p):sha(p),str(trace):sha(trace)});commands.append(r['case']['command'])
 diagnostic=json.loads(diagnostic_result.read_text())
 if diagnostic.get('status')!='COMPLETED' or diagnostic.get('model_unchanged') is not True:
  raise ValueError('complete unchanged-model diagnostic required for golden inputs')
 manifest_path=diagnostic_result.with_name('manifest.json');dm=json.loads(manifest_path.read_text())
 model_names=[k for k,v in dm['models'].items() if v['sha256']==digest]
 if len(model_names)!=1:raise ValueError('diagnostic does not bind this source checkpoint')
 batches=[];categories=[]
 evidence.update({str(diagnostic_result):sha(diagnostic_result),str(manifest_path):sha(manifest_path)})
 for r in diagnostic['records']:
  if r['controller']!=model_names[0]:continue
  p=diagnostic_result.parent/r['trace']
  if sha(p)!=r['trace_sha256']:raise ValueError('diagnostic input trace SHA differs')
  with np.load(p,allow_pickle=False) as z:
   points=np.arange(len(z['actor_input']))
   if r.get('predecessor'):points=points[z['owner']==1]
   points=points[np.linspace(0,len(points)-1,min(24,len(points))).astype(int)]
   batches.extend(z['actor_input'][points]);categories.extend(['actual_'+r['case_id']]*len(points))
  evidence[str(p)]=sha(p)
 # Fixed complete former deployment vectors extend numerical coverage; they
 # provide input fixtures only, not source ability or successful4500 trajectories.
 old_vectors=ROOT/'deployment_protocol/2026_10_23-19_53__motion500_currentplant_small31_onnx/golden_vectors.npz'
 with np.load(old_vectors,allow_pickle=False) as z:
  x=z['observation'];points=np.linspace(0,len(x)-1,96).astype(int)
  batches.extend(x[points]);categories.extend(['previous_deployment_complete_input_fixture']*len(points))
 evidence[str(old_vectors)]=sha(old_vectors)
 x=np.asarray(batches,dtype=np.float32)
 if x.ndim!=2 or x.shape[1]!=31 or not np.isfinite(x).all():raise ValueError('invalid golden observation matrix')
 output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
 model=DeterministicActor(actor).eval();path=output/'actor.onnx'
 with torch.no_grad():
  torch.onnx.export(model,torch.zeros(1,31),str(path),input_names=['observation'],output_names=['mean_action'],opset_version=17,dynamo=False)
  expected=model(torch.from_numpy(x)).numpy()
 graph=onnx.load(path);onnx.checker.check_model(graph,full_check=True)
 shapes=[[d.dim_value for d in v.type.tensor_type.shape.dim] for v in [graph.graph.input[0],graph.graph.output[0]]]
 if shapes!=[[1,31],[1,6]]:raise ValueError('fixed31D/6D export shape differs')
 backend=ReferenceEvaluator(graph)
 actual=np.concatenate([backend.run(None,{'observation':v[None]})[0] for v in x])
 error=np.abs(actual-expected);maximum=float(error.max())
 np.savez_compressed(output/'golden_vectors.npz',observation=x,pytorch_mean_action=expected,
                     onnx_mean_action=actual,category=np.asarray(categories))
 golden={'schema':'motion31_model_golden_v1','input':'float32[31] already encoded',
   'output':'float32[6] mean with tanh included','vectors':[
     {'id':i,'category':c,'observation':v.tolist(),'mean_action':a.tolist()}
       for i,(v,a,c) in enumerate(zip(x,expected,categories))]}
 (output/'golden_vectors.json').write_text(json.dumps(golden,ensure_ascii=False,indent=2)+'\n')
 linears=[m for m in model.mlp if isinstance(m,torch.nn.Linear)]
 dims=[linears[0].in_features]+[m.out_features for m in linears]
 parameters=sum(p.numel() for p in model.parameters());mac=sum(m.in_features*m.out_features for m in linears)
 count=sum(int(np.prod(t.dims)) for t in graph.graph.initializer)
 if count!=parameters:raise ValueError('export contains non-Actor parameters or misses weights')
 from native_stand.students.mcu_interface import contract
 interface=contract();interface.update(actor_hidden_dims=dims[1:-1],height_command_m=.34,critic_dim=None,
     critic_hidden_dims=None,critic_deployed=False)
 report={'version':'motion31_accepted_actor_electrical_candidate_v1','interface_version':'multiskill_mcu_v2',
     'source_checkpoint':str(checkpoint),'source_sha256':digest,'parent_spec':str(spec_path),'spec_sha256':sha(spec_path),
     'asset_sha256':spec['asset_sha256'],'actor_onnx_sha256':sha(path),'architecture':dims,
     'parameters':parameters,'fp32_parameter_bytes':parameters*4,'linear_MAC_per_inference':mac,
     'onnx_file_bytes':path.stat().st_size,'graph_input_name':'observation','graph_output_name':'mean_action',
     'input_shape':[1,31],'output_shape':[1,6],'dtype':'float32','operators':sorted(set(n.op_type for n in graph.graph.node)),
     'interface':interface,'mean_tanh_included':True,'history':False,'critic_included':False,'exploration_included':False,
     'verification':{'backend':'onnx.reference.ReferenceEvaluator CPU float32','vectors':len(x),'tolerance':1e-5,
         'normalized_max_abs_error':maximum,'leg_target_max_abs_error_rad':float(error[:,[0,1,3,4]].max()*1.5),
         'wheel_target_max_abs_error_rad_s':float(error[:,[2,5]].max()*100),'pass':maximum<=1e-5},
     'capability':{'accepted_conditions':len(accepted['evaluations']),'scored_envs':len(accepted['per_env']),
         'passed_envs':sum(e['accepted'] is True for e in accepted['per_env']),'commands':commands,
         'scope':'same-source Native MCU-hypothesis matrix only; not hardware certification',
         'MuJoCo_nominal_zero_vx_mps':next(r['windows']['steady_2_to_30']['vx_mean_mps'] for r in diagnostic['records'] if r['case_id']==model_names[0]+'__nominal'),
         'hardware_verified':False},
     'deployment_status':'CANDIDATE_FOR_ELECTRICAL_RESOURCE_AND_SHADOW_CHECK; not flashed or enabled',
     'hardware_analysis_tool':next((shutil.which(t) for t in ('stedgeai','stm32ai') if shutil.which(t)),None),
     'hardware_inference_time_ms':None,'firmware_memory_fit':None,
     'input_evidence_sha256':evidence,'golden_sha256':sha(output/'golden_vectors.npz')}
 (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 (output/'verification.json').write_text(json.dumps(report['verification'],indent=2)+'\n')
 if not report['verification']['pass']:raise ValueError('export numerical tolerance failed')
 return report

def export(checkpoint,spec_path,capture_run,output):
 import onnx
 from onnx.reference import ReferenceEvaluator
 checkpoint=Path(checkpoint).resolve();spec_path=Path(spec_path).resolve();capture_run=Path(capture_run).resolve();output=Path(output).resolve()
 if output.exists():raise FileExistsError(output)
 actor,saved=load(checkpoint,'MotionZeroStudent');actor.eval();spec=json.loads(spec_path.read_text())
 if saved['interface_version']!='multiskill_mcu_v2' or saved['spec_sha256']!=sha(spec_path) or saved['asset_sha256']!=spec['asset_sha256'] or saved['critic_privileged_version']!='motion_privileged49_v1':raise ValueError('Exact Motion31 checkpoint/spec identity required')
 if spec['action']['leg_scale_rad']!=1.5 or spec['action']['wheel_scale_rad_s']!=100:raise ValueError('Unexpected physical scale')
 meta=json.loads((capture_run/'probe_result.json').read_text());capture=capture_run/meta['capture']['path']
 if meta['source_sha256']!=sha(checkpoint) or sha(capture)!=meta['capture']['sha256']:raise ValueError('Actual input provenance differs')
 with np.load(capture,allow_pickle=False) as a:
  batches=[];categories=[]
  for cohort in range(7):
   ids=np.argwhere(a['command_index']==cohort);chosen=ids[np.linspace(0,len(ids)-1,16).astype(int)]
   batches.extend(a['actor_input'][t,e].copy() for t,e in chosen);categories.extend([f'actual_native_cohort_{cohort}']*len(chosen))
 rng=np.random.default_rng(20261015);n=32;gravity=rng.normal(size=(n,3));gravity/=np.linalg.norm(gravity,axis=-1,keepdims=True)
 from native_stand.students.mcu_interface import encode
 q=torch.tensor(rng.uniform(-1.2,1.2,(n,6)),dtype=torch.float32);dq=torch.tensor(rng.uniform(-30,30,(n,6)),dtype=torch.float32)
 synthetic=encode(torch.tensor(rng.normal(0,2,(n,3)),dtype=torch.float32),torch.tensor(gravity,dtype=torch.float32),q,dq,q[:,[0,1,3,4]],torch.tensor(rng.uniform(-50,50,(n,2)),dtype=torch.float32),torch.zeros(n,6),'Motion',torch.tensor([[0.,0.,.34]]).repeat(n,1)).numpy()
 batches.extend(synthetic);categories.extend(['synthetic_encoding_envelope_not_capability']*n)
 for v,yaw in [(0,0),(.5,0),(1,0),(1.5,0),(2,0),(0,.5),(0,-.5),(0,1),(0,-1)]:
  x=np.zeros(31,np.float32);x[4]=-1;x[22:25]=[v/4,yaw/4,.68];x[25]=1;x[28]=1;batches.append(x);categories.append('zero_pose_command_fixture_not_capability')
 x=np.asarray(batches,np.float32);torch.set_num_threads(1);model=DeterministicActor(actor).eval()
 output.mkdir(parents=True,exist_ok=False)
 with torch.no_grad():
  torch.onnx.export(model,torch.zeros(1,31),str(output/'actor.onnx'),input_names=['observation'],output_names=['mean_action'],opset_version=17,dynamo=False)
  expected=model(torch.from_numpy(x)).numpy();ts=torch.jit.trace(model,torch.zeros(1,31));ts.save(str(output/'actor_torchscript.pt'))
  ts_result=torch.jit.load(str(output/'actor_torchscript.pt'))(torch.from_numpy(x)).numpy()
 torch.save(model.state_dict(),output/'actor_weights_fp32.pt')
 graph=onnx.load(output/'actor.onnx');onnx.checker.check_model(graph,full_check=True)
 if [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]!=[1,31]:raise ValueError('ONNX shape differs')
 actual=np.concatenate([ReferenceEvaluator(graph).run(None,{'observation':row[None]})[0] for row in x])
 template=ROOT/'deployment_protocol/motion31_reference'
 for name in ['reference_adapter.py','actor_runtime.c']:shutil.copyfile(template/name,output/name)
 lines=['/* FP32 row-major Linear weights: output_channel,input_channel. */','#ifndef MOTION31_WEIGHTS_H','#define MOTION31_WEIGHTS_H']
 linears=[layer for layer in model.mlp if isinstance(layer,torch.nn.Linear)]
 for i,layer in enumerate(linears):
  for prefix,param in [('W',layer.weight),('B',layer.bias)]:
   data=param.detach().numpy().reshape(-1);floats=[format(float(v),'.9e')+'f' for v in data]
   lines.append(f'static const float {prefix}{i}[{len(data)}]={{'+','.join(floats)+'};')
 lines.append('#endif');(output/'actor_weights.h').write_text('\n'.join(lines)+'\n')
 compiler=shutil.which('cc')
 if not compiler:raise ValueError('C compiler required for handoff numerical validation')
 with tempfile.TemporaryDirectory(prefix='motion31-c-') as tmp:
  lib=Path(tmp)/'actor.so';subprocess.run([compiler,'-std=c99','-O2','-shared','-fPIC',str(output/'actor_runtime.c'),'-lm','-o',str(lib)],check=True,capture_output=True)
  runtime=ctypes.CDLL(str(lib));ptr=ctypes.POINTER(ctypes.c_float);runtime.motion31_actor.argtypes=[ptr,ptr];runtime.motion31_actor.restype=ctypes.c_int
  c_result=np.zeros((len(x),6),np.float32)
  for i,row in enumerate(x):
   if runtime.motion31_actor(row.ctypes.data_as(ptr),c_result[i].ctypes.data_as(ptr)):raise ValueError('C reference returned nonfinite')
 errors={'ONNX_vs_PyTorch':float(np.abs(actual-expected).max()),'TorchScript_vs_PyTorch':float(np.abs(ts_result-expected).max()),'C_float32_vs_PyTorch':float(np.abs(c_result-expected).max())}
 if max(errors.values())>1e-5:raise ValueError('Export numerical mismatch '+str(errors))
 vectors={'schema':'motion31_model_golden_v1','input':'31D normalized float32','output':'tanh deterministic mean6; no extra tanh/std/noise','vectors':[
  {'id':i,'category':category,'observation':row.tolist(),'mean_action':out.tolist()} for i,(row,out,category) in enumerate(zip(x,expected,categories))]}
 (output/'model_golden_vectors.json').write_text(json.dumps(vectors,indent=2)+'\n')
 np.savez_compressed(output/'model_golden_vectors.npz',observation=x,mean_action=expected,onnx_mean=actual,c_mean=c_result,category=np.array(categories))
 mapping=json.loads((ROOT/'deployment_protocol/motion_legacy150d_v1/HARDWARE_MAPPING_TEMPLATE.json').read_text())
 mapping.update(protocol='motion31_u1600_electrical_v1',initialization='normal direct_zero_reset in simulation; hardware calibration and safe enable not validated',height_command_m=.34,
  root_height_measurement='simulation USD root-link origin height above plane; hardware matching CAD point must be calibrated',joint_mechanical_limits_rad=None,policy_deadline_ms=None)
 (output/'HARDWARE_MAPPING_TEMPLATE.json').write_text(json.dumps(mapping,indent=2)+'\n')
 from native_stand.students.mcu_interface import contract
 contract=contract();contract.update(height_command_m=.34,critic_dim=None,critic_deployed=False)
 report={'protocol_version':'motion31_u1600_electrical_v1','source_checkpoint':str(checkpoint),'source_sha256':sha(checkpoint),'spec_path':str(spec_path),'spec_sha256':sha(spec_path),'asset_sha256':spec['asset_sha256'],
  'interface':contract,'input_name':'observation','output_name':'mean_action','input_shape':[1,31],'output_shape':[1,6],'dtype':'float32',
  'network':[31,128,64,32,6],'hidden_activation':'ELU(alpha=1)','output_activation':'tanh included in graph; do not apply again',
  'parameters':sum(p.numel() for p in model.parameters()),'linear_MAC_per_inference':sum(l.in_features*l.out_features for l in linears),
  'fp32_parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'input_output_bytes':4*(31+6),
  'reference_C_activation_stack_bytes':4*(128+64+32+6),'stack_scope':'activation arrays only; actual target firmware/math stack unmeasured',
  'operators':sorted(set(n.op_type for n in graph.graph.node)),'verification':errors,'golden_count':len(x),'numerical_pass':True,
  'backend':'ONNX ReferenceEvaluator, TorchScript and host C99; onnxruntime not installed',
  'vector_sources':{str(capture):sha(capture)},'hardware_validated':False,'yaw_capability':'NOT_PASSED; MuJoCo yaw commands wrong direction/low gain',
  'ability_report':str(ROOT/'docs/MOTION31_U1600_MUJOCO_20261015.md'),
  'deployment_excludes':['Critic49','Adam','Gaussian std','RNG','history','simulator','training reward'],
  'files_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file()}}
 (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');(output/'verification.json').write_text(json.dumps({'errors':errors,'pass':True,'samples':len(x),'tolerance':1e-5,'physical_leg_max_error_rad':max(errors.values())*1.5,'physical_wheel_max_error_rad_s':max(errors.values())*100},indent=2)+'\n')
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
 group=p.add_mutually_exclusive_group(required=True);group.add_argument('--capture-run',type=Path);group.add_argument('--evaluation-dir',type=Path,help='Complete accepted checkpoint registry/raw matrix; Actor-only export without historical49D assumptions')
 p.add_argument('--diagnostic-result',type=Path,help='Completed same-model closed-loop result supplies actual golden inputs')
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.evaluation_dir and not a.diagnostic_result:p.error('--evaluation-dir requires --diagnostic-result')
 print(json.dumps(export_accepted(a.checkpoint,a.spec,a.evaluation_dir,a.diagnostic_result,a.output) if a.evaluation_dir else export(a.checkpoint,a.spec,a.capture_run,a.output),indent=2))
