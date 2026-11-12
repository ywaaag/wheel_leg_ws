"""Recovery large→small data adapter; reuse shared Student/distill.train, no simulator."""
import json
from pathlib import Path
import numpy as np
import torch
from .data import sha
from .model import load
from .recovery_dr import SOURCE,ASSET
VERSION='recovery1500_same31_small_distillation_v1'
STAGES=['early_preparation_0_0p10s','turn_lift_0p10_0p25s','dynamic_until_entry','first_fixed_10s_hold']


def validate_small_lineage(checkpoint,dataset,parent_spec):
 """Explicit BC/aggregation lineage, separate from legacy PPO checkpoint identity."""
 actor,cp=load(checkpoint,'RecoveryStudent');dataset=Path(dataset);parent_spec=Path(parent_spec)
 m=json.loads(dataset.with_name('manifest.json').read_text());s=json.loads(parent_spec.read_text())
 if m['version']!=VERSION or m['teacher_parent_spec_sha256']!=sha(parent_spec) or m['asset_sha256']!=ASSET or s['asset_sha256']!=ASSET or sha(s['asset_path'])!=ASSET:raise ValueError('Recovery dataset parent/asset differs')
 if cp['dataset_sha256']!=sha(dataset) or cp['manifest_sha256']!=sha(dataset.with_name('manifest.json')):raise ValueError('Recovery checkpoint dataset SHA differs')
 teacher=m['teachers']['Recovery1500']
 if teacher['sha256']!=SOURCE or sha(teacher['path'])!=SOURCE or cp['teacher_sha256']!={teacher['path']:SOURCE}:raise ValueError('Recovery Teacher lineage differs')
 if [actor.mlp[0].in_features]+[v.out_features for v in actor.mlp if hasattr(v,'out_features')]!=[31,128,64,32,6]:raise ValueError('explicit small Recovery architecture required')
 return actor,cp,m


def phase_ids(controlled_count,entry_local):
 if controlled_count<=entry_local or entry_local<25:raise ValueError('entry too early for declared four-phase dataset')
 k=np.arange(controlled_count)
 return np.where(k<10,0,np.where(k<25,1,np.where(k<=entry_local,2,3)))


def extract_episode(a,per_env,e,reference):
 """Pre-action actual31D -> this interval's executed physical targets; no shifted state."""
 from native_stand.command_v6.domain.trace_schema import CONTROL
 if not per_env['accepted'] or not per_env['hold_complete']:raise ValueError('only successful complete first episodes are expert data')
 end=per_env['hold_index_half_open'][1];entry=per_env['entry_index']
 valid=a['first_episode_valid'][:end,e].astype(bool)&a['actor_valid'][:end,e,0].astype(bool)&(a['controller'][:end,e,0]==3)
 if not valid.any():raise ValueError('no student-controlled transition')
 ix=np.flatnonzero(valid)
 if np.any(np.diff(ix)!=1):raise ValueError('missing/noncontinuous controlled interval')
 if len(np.unique(a['state'][ix,e,31]))!=1 or not np.all(a['state'][ix,e,29]>0):raise ValueError('episode reset/invalid physics inside teaching interval')
 clock=a['policy_time_s'][ix]
 if not np.allclose(np.diff(clock),.01,atol=1e-6,rtol=0):raise ValueError('invalid policy clock')
 x=a['actor_input'][ix,e].astype(np.float32).copy()
 if x.shape[1]!=31 or not np.isfinite(x).all():raise ValueError('actual31D input missing/nonfinite')
 if not np.allclose(x[:,22:25],[0,0,.68],atol=1e-6) or not np.all(x[:,25:31]==[0,1,0,1,0,0]):raise ValueError('Recovery/zero/.34/NONE task differs')
 active=a['active_indices']
 q=a['joint_q_pre'][ix,e][:,active];dq=a['joint_dq_pre'][ix,e][:,active]
 expected_q=(q[:,[0,1,3,4]]-reference[[0,1,3,4]])/1.5
 if not np.allclose(x[:,6:10],expected_q,atol=1e-6,rtol=0) or not np.allclose(x[:,10:16],dq*.05,atol=1e-6,rtol=0):raise ValueError('pre-action q/dq encoding alignment differs')
 prev=a['previous_targets'][ix,e]
 expected_prev=np.c_[(prev[:,:4]-reference[[0,1,3,4]])/1.5,prev[:,4:]/100.]
 if not np.allclose(x[:,16:22],expected_prev,atol=1e-6,rtol=0):raise ValueError('previous executed target encoding differs')
 ctrl=a['control'][ix,e]
 physical=np.zeros((len(ix),6),np.float32)
 physical[:,[0,1,3,4]]=ctrl[:,CONTROL['leg_target_after_limit']]
 physical[:,[2,5]]=ctrl[:,CONTROL['clipped_action']][:,[2,5]]*100.
 label=physical.copy();label[:,[0,1,3,4]]=(label[:,[0,1,3,4]]-reference[[0,1,3,4]])/1.5;label[:,[2,5]]/=100.
 if not np.isfinite(label).all() or (abs(label)>1+1e-6).any():raise ValueError('teacher target not representable; do not silently clip')
 entry_local=int(np.searchsorted(ix,entry))
 if entry_local>=len(ix) or ix[entry_local]!=entry:raise ValueError('first entry not in controlled first episode')
 return {'x':x,'y':label,'stage':phase_ids(len(ix),entry_local),'sample':ix,'physical':physical}


def prepare(checkpoint,parent_spec,frozen_manifest,output):
 checkpoint,parent_spec,frozen_manifest,output=map(lambda p:Path(p).resolve(),(checkpoint,parent_spec,frozen_manifest,output))
 if output.exists():raise FileExistsError(output)
 if sha(checkpoint)!=SOURCE:raise ValueError('fixed frozen Recovery1500 teacher differs')
 actor,cp=load(checkpoint,'RecoveryStudent');spec=json.loads(parent_spec.read_text());frozen=json.loads(frozen_manifest.read_text())
 if cp['spec_sha256']!=sha(parent_spec) or cp['asset_sha256']!=ASSET or sha(spec['asset_path'])!=ASSET or frozen['checkpoint_sha256']!=SOURCE:raise ValueError('teacher spec/asset/freeze identity differs')
 if cp['network_architecture']['actor_hidden_dims']!=[512,256,128] or cp['interface_version']!='multiskill_mcu_v2':raise ValueError('explicit31D large teacher required')
 root=parent_spec.parents[3];source_run=checkpoint.parent
 pieces=[];episodes=[];sources={str(parent_spec):sha(parent_spec),str(frozen_manifest):sha(frozen_manifest),str(checkpoint):SOURCE}
 max_mean_error=0.;teacher_before={k:v.clone() for k,v in actor.state_dict().items()}
 torch.set_num_threads(4);actor.eval().requires_grad_(False)
 for evidence_index,evidence in enumerate(frozen['evidence']):
  path=root/evidence['path']
  if sha(path)!=evidence['sha256']:raise ValueError('frozen evaluation identity differs')
  result=json.loads(path.read_text());trace=path.parent/result['trace']['path']
  if sha(trace)!=result['trace']['sha256'] or result['checkpoint_sha256']!=SOURCE or result['asset_sha256']!=ASSET or result['spec_sha256']!=sha(parent_spec):raise ValueError('trace/model/spec/asset provenance differs')
  if result['field_schema']['state']!='recovery_policy_trace_v1' or result['field_schema']['control']!='recovery_policy_trace_v1':raise ValueError('trace named schema differs')
  with np.load(trace,allow_pickle=False) as saved:a={k:saved[k] for k in saved.files}
  required={'actor_input','actor_valid','controller','first_episode_valid','state','control','joint_q_pre','joint_dq_pre','previous_targets','policy_time_s','mean'}
  if not required<=set(a):raise ValueError('missing pre-action teaching fields')
  a['active_indices']=result['field_schema']['active_indices']
  readback=json.loads(path.with_name(path.name.replace('evaluation_','evaluation_readback_')).read_text())
  reference=np.asarray(readback['reference'][0],dtype=np.float32)
  sources.update({str(path):sha(path),str(trace):sha(trace),str(path.with_name(path.name.replace('evaluation_','evaluation_readback_'))):sha(path.with_name(path.name.replace('evaluation_','evaluation_readback_')))})
  split=0 if evidence_index<5 else 1 if evidence_index==5 else 2
  for e,row in enumerate(result['per_env']):
   item=extract_episode(a,row,e,reference)
   with torch.inference_mode():mean=torch.cat([actor.mean(t) for t in torch.from_numpy(item['x']).split(1024)]).numpy()
   ix=item['sample'];err=float(abs(mean-a['mean'][ix,e]).max());max_mean_error=max(max_mean_error,err)
   if err>1e-5:raise ValueError('teacher replay does not match recorded pre-action mean')
   # In these valid source traces targets use identical scales and no leg clipping.
   executed_difference=float(abs(mean-item['y']).max())
   if executed_difference>1e-5:raise ValueError('executed label differs from deterministic teacher; explicit clipping adapter required')
   eid=len(episodes);item.update(split=np.full(len(ix),split),episode=np.full(len(ix),eid));pieces.append(item)
   episodes.append({'id':eid,'evaluation':str(path),'case_id':result['case']['case_id'],'env':e,'split':['train','validation','test'][split],'seed':result['case']['seed'],'yaw_rad':result['case']['yaw_rad'],'entry_index':row['entry_index'],'hold_window':row['hold_index_half_open'],'unique_phase_samples':[int((item['stage']==j).sum()) for j in range(4)],'teacher_mean_vs_executed_max_abs':executed_difference})
 if len(episodes)!=64 or len({e['split'] for e in episodes})!=3:raise ValueError('expected original64 successful episodes with train/validation/test isolation')
 data={k:torch.from_numpy(np.concatenate([p[k] for p in pieces])).float() if k in ('x','y') else torch.from_numpy(np.concatenate([p[k] for p in pieces])).long() for k in ('x','y','stage','split','episode','sample')}
 data['stage_names']=STAGES
 output.mkdir(parents=True,exist_ok=False);dataset=output/'RecoveryStudent.pt';torch.save(data,dataset)
 manifest={'version':VERSION,'interface_version':'multiskill_mcu_v2','sampling_version':VERSION,'teachers':{'Recovery1500':{'path':str(checkpoint),'sha256':SOURCE}},'teacher_parent_spec':str(parent_spec),'teacher_parent_spec_sha256':sha(parent_spec),'asset_sha256':ASSET,'source_inputs_sha256':sources,'episodes':episodes,'split_policy':'40train(original32+finalnative8),8validation(finalyaw20261013),16test(finalyaw20261014/15); entire episodes; no temporal frame split','phase_policy':'controlled first.10s, .10-.25s, .25s through first confirmed entry, next fixed10s; sampling strata only, not Actor input or runtime schedule','label_semantics':'pre-action actual31D -> same-interval post-limit physical target -> student coordinates1.5rad/100rad_s; deterministic Teacher replay agrees','teacher_replay_max_abs':max_mean_error,'student_architecture':[31,128,64,32,6],'RecoveryStudent':{'file':dataset.name,'sha256':sha(dataset),'unique_samples':len(data['x']),'split_stage_counts':{name:{stage:int(((data['split']==i)&(data['stage']==j)).sum()) for j,stage in enumerate(STAGES)} for i,name in enumerate(('train','validation','test'))}},'limitations':'64 nearby successful evaluations, not64 independent recovery conditions; no terrain/real-world distribution coverage; Student closed loop NOT_RUN','simulation':0,'optimizer_steps':0}
 assert all(torch.equal(v,actor.state_dict()[k]) for k,v in teacher_before.items()) and sha(checkpoint)==SOURCE
 (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 return {'dataset':str(dataset),'manifest':str(output/'manifest.json'),'episodes':len(episodes),'samples':len(data['x']),'split_stage_counts':manifest['RecoveryStudent']['split_stage_counts'],'teacher_replay_max_abs':max_mean_error,'simulation':0,'optimizer_steps':0}


def export(student,dataset,output):
 """Use existing Actor-only wrapper and ONNX ReferenceEvaluator; no physics."""
 import onnx
 from onnx.reference import ReferenceEvaluator
 from .export import DeterministicActor
 student,dataset,output=map(lambda p:Path(p).resolve(),(student,dataset,output))
 if output.exists():raise FileExistsError(output)
 m=json.loads(dataset.with_name('manifest.json').read_text());actor,cp=load(student,'RecoveryStudent')
 if m['version']!=VERSION or cp['dataset_sha256']!=sha(dataset) or cp['manifest_sha256']!=sha(dataset.with_name('manifest.json')) or cp['teacher_sha256']!={m['teachers']['Recovery1500']['path']:SOURCE}:raise ValueError('Recovery Student dataset/teacher provenance differs')
 model=DeterministicActor(actor).eval();data=torch.load(dataset,map_location='cpu',weights_only=False);groups=[]
 for split in (1,2):
  for phase in range(4):
   ids=torch.where((data['split']==split)&(data['stage']==phase))[0]
   if not len(ids):raise ValueError('heldout phase missing')
   groups.append(data['x'][ids[torch.linspace(0,len(ids)-1,min(32,len(ids))).long()]])
 x=torch.cat(groups);torch.set_num_threads(4)
 output.mkdir(parents=True,exist_ok=False)
 with torch.inference_mode():
  torch.onnx.export(model,torch.zeros(1,31),str(output/'actor.onnx'),input_names=['observation'],output_names=['mean_action'],opset_version=17,dynamo=False)
  expected=model(x).numpy()
 graph=onnx.load(output/'actor.onnx');onnx.checker.check_model(graph,full_check=True)
 evaluator=ReferenceEvaluator(graph);actual=np.concatenate([evaluator.run(None,{'observation':row[None]})[0] for row in x.numpy()])
 error=abs(expected-actual)
 np.savez_compressed(output/'golden_vectors.npz',observation=x.numpy(),pytorch_mean=expected,onnx_mean=actual)
 report={'version':VERSION,'role':'RecoveryStudent','checkpoint':str(student),'checkpoint_sha256':sha(student),'teacher_checkpoint':m['teachers']['Recovery1500']['path'],'teacher_sha256':SOURCE,'parent_spec':m['teacher_parent_spec'],'parent_spec_sha256':m['teacher_parent_spec_sha256'],'dataset_sha256':sha(dataset),'asset_sha256':ASSET,'network':[31,128,64,32,6],'parameters':sum(p.numel() for p in model.parameters()),'FP32_parameter_bytes':sum(p.numel() for p in model.parameters())*4,'linear_MAC':sum(l.in_features*l.out_features for l in model.mlp if isinstance(l,torch.nn.Linear)),'operators':sorted(set(n.op_type for n in graph.graph.node)),'input':{'name':'observation','shape':[1,31],'dtype':'float32'},'output':{'name':'mean_action','shape':[1,6],'dtype':'float32','Tanh_included':True},'command':[0,0,.34],'skill_jump':[0,1,0,1,0,0],'action_scales':[1.5,1.5,100,1.5,1.5,100],'backend':'onnx.reference.ReferenceEvaluator CPU','golden_count':len(x),'max_abs_error':float(error.max()),'physical_leg_max_error_rad':float(error[:,[0,1,3,4]].max()*1.5),'physical_wheel_max_error_rad_s':float(error[:,[2,5]].max()*100),'numerical_pass':bool(np.isfinite(actual).all() and error.max()<=1e-5),'closed_loop':'NOT_RUN','hardware':'HARDWARE_UNVERIFIED','critic_exported':False,'optimizer_exported':False,'std_exported':False,'onnx_sha256':sha(output/'actor.onnx'),'golden_sha256':sha(output/'golden_vectors.npz')}
 assert report['parameters']==14630 and report['linear_MAC']==14400
 for value,target in ((graph.graph.input[0],[1,31]),(graph.graph.output[0],[1,6])):
  if [d.dim_value for d in value.type.tensor_type.shape.dim]!=target:raise ValueError('exported fixed shape differs')
 (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 (output/'README.md').write_text('# Recovery1500蒸馏小模型：离线实验候选\n\n31→128→64→32→6，ELU/Tanh，输入observation float32[1,31]，输出mean_action float32[1,6]。无历史，不再tanh、不采样噪声；仅Actor导出。\n\n沿用multiskill_mcu_v2本体字段/关节零位/方向和上一实际执行目标语义；命令[0,0,.34]，Motion/Recovery/Jump=[0,1,0]，NONE/SMALL/BIG=[1,0,0]，全程学生控制、不执行教师前缀或定时切换。四腿目标按reference+1.5a，轮目标100a，机械保护后实际目标回写previous。原PD60/2/1、40/5.5Nm、100Hz策略/200Hz PD保持。\n\n来源为冻结大Recovery1500；本学生只完成CPU监督拟合和ONNX数值验证，独立Swing起身、保持、sim2sim及实机均NOT_RUN，不能沿用教师64/64证书。不要直接替换已部署模型并认定能力相同。manifest绑定源模型、数据、学生与数值验证；golden_vectors供同输入对照。\n')
 if not report['numerical_pass']:raise ValueError('ONNX numerical tolerance failed; report preserved')
 return report
