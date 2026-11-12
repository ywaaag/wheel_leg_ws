"""Finite Recovery state aggregation, reusing shared BC and MuJoCo execution."""
import copy,json,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from .data import sha
from .recovery_distillation import VERSION as DATA_VERSION,STAGES,validate_small_lineage
from .model import load
VERSION='recovery31_state_aggregation_cycle_v1'


def eligible(x,state,posture):
 """Conservative measurement gate; does not certify Teacher labels as correct."""
 return (np.isfinite(x).all(-1)&np.isfinite(state).all(-1)&np.isfinite(posture).all(-1)
         &(state[:,6]>=.15)&(state[:,6]<=.43)&(state[:,12]<=.005)&(state[:,13]>0)
         &(np.abs(posture[:,:2]).max(-1)<=.30)&(state[:,15]<=.35)&(state[:,29]>0))


def query(x,teacher,ranges):
 from .mcu_interface import targets,labels
 with torch.inference_mode():
  means=torch.cat([teacher.mean(chunk) for chunk in x.split(1024)])
  physical=targets(means,torch.zeros_like(means),ranges)
  y=labels(physical['leg_executed'],physical['wheel_executed'],torch.zeros_like(means))
 return y


def replay_native(run):
 """Only training initial groups; do not aggregate heldout yaw11/yaw12."""
 parts=[];provenance={}
 for index in (0,1):
  p=Path(run)/f'evaluation_{index:02}.json';d=json.loads(p.read_text());trace=p.parent/d['trace']['path']
  if d['evaluated_model']!='student' or d['case']['seed']!=20261010 or sha(trace)!=d['trace']['sha256']:raise ValueError('training-only native replay identity differs')
  a=dict(np.load(trace));valid=a['first_episode_valid'][:,0].astype(bool)&a['actor_valid'][:,0,0].astype(bool)&(a['controller'][:,0,0]==3)
  ix=np.flatnonzero(valid)
  if not len(ix) or not np.allclose(np.diff(a['policy_time_s'][ix]),.01,atol=1e-6,rtol=0):raise ValueError('native replay clock differs')
  if len(np.unique(a['state'][ix,0,31]))!=1:raise ValueError('replay crosses reset')
  parts.append((a['actor_input'][ix,0],a['state'][ix,0],a['posture'][ix,0],d['per_env'][0].get('entry_index'),ix,index,'native'))
  provenance[str(p.resolve())]=sha(p);provenance[str(trace.resolve())]=sha(trace)
 return parts,provenance


def replay_mujoco(run,episode):
 p=Path(run);m=json.loads((p/'manifest.json').read_text());terminal=json.loads((p/'terminal_report.json').read_text())
 if terminal['status']!='COMPLETED' or not terminal['model_unchanged']:raise ValueError('MuJoCo replay engineering failure')
 f=p/'trajectory.npz';a=dict(np.load(f));clock=a['policy_time_s']
 if not len(clock) or not np.allclose(np.diff(clock),.01,atol=1e-6,rtol=0):raise ValueError('MuJoCo replay clock differs')
 r=terminal['result'].get('score',{})
 return [(a['actor_input'],a['state'],a['posture'],r.get('entry_index'),np.arange(len(clock)),episode,'mujoco')],{str(f.resolve()):sha(f),str((p/'manifest.json').resolve()):sha(p/'manifest.json')}


def append_replay(data,parts,teacher,ranges,first_episode):
 added=[];stats=[]
 for offset,(x,state,posture,entry,indices,_,domain) in enumerate(parts):
  if x.shape[1]!=31 or not np.allclose(x[:,22:25],[0,0,.68],atol=1e-6) or not np.all(x[:,25:31]==[0,1,0,1,0,0]):raise ValueError('actual Recovery31 input differs')
  mask=eligible(x,state,posture);local=np.arange(len(x));stage=np.where(local<10,0,np.where(local<25,1,np.where(indices<=(entry if entry is not None else indices[-1]),2,3)))
  xx=torch.tensor(x[mask],dtype=torch.float32)
  if not len(xx):raise ValueError('no eligible Teacher query states')
  added.append(dict(x=xx,y=query(xx,teacher,ranges),stage=torch.tensor(stage[mask]),episode=torch.full((len(xx),),first_episode+offset,dtype=torch.long),split=torch.zeros(len(xx),dtype=torch.long)))
  stats.append({'domain':domain,'available':len(x),'included':len(xx),'excluded':int((~mask).sum()),'unique_phase_samples':np.bincount(stage[mask],minlength=4).tolist(),'teacher_label_reliability':'UNCERTAIN on off-Teacher states; measurement gate is not a capability certificate'})
 for key in ('x','y','stage','episode','split'):data[key]=torch.cat([data[key]]+[d[key] for d in added])
 return stats


def validate(r):
 required={'version','authorization','session','teacher','parent_spec','dataset','initial_student','native_replay_run','output','max_rounds','steps_per_round','total_optimizer_steps','wall_limit_s','output_max_bytes'}
 if set(r)!=required or r['version']!=VERSION or r['authorization']!='AUTHORIZED':raise ValueError('Recovery cycle explicit request differs')
 import os
 if not r['session'] or r['session']!=os.environ.get('HAPI_SESSION_ID'):raise ValueError('current session differs')
 if r['max_rounds']!=3 or r['steps_per_round']!=5000 or r['total_optimizer_steps']!=15000 or r['wall_limit_s']!=3600 or r['output_max_bytes']!=2**30:raise ValueError('finite Recovery cycle limits differ')
 actor,cp,m=validate_small_lineage(r['initial_student'],r['dataset'],r['parent_spec'])
 if sha(r['teacher'])!=m['teachers']['Recovery1500']['sha256']:raise ValueError('cycle Teacher differs')
 from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
 validate_new_experiment_name(Path(r['output']).name)
 return {'status':'PLAN','files_written':0,'online_PPO':0,'max_rounds':3,'max_BC_steps':15000,'MuJoCo_episodes_max':5,'native_cases':'separate final public supervisor request','input_sha256':{str(Path(r[k]).resolve()):sha(r[k]) for k in ('teacher','parent_spec','dataset','initial_student')}}


def run(r,extra_replays=()):
 from .distill import train
 from native_stand.command_v6.completion import write
 from native_stand.command_v6.domain.experiment_naming import new_experiment_name
 plan=validate(r);out=Path(r['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();results=[];used=0;status='FAILED';error=None
 write(out/'request.json',r);write(out/'input_identity.json',plan)
 def budget():
  if time.monotonic()-start>=r['wall_limit_s']-60:raise RuntimeError('cycle wall reserve reached')
  if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>r['output_max_bytes']-32*2**20:raise RuntimeError('cycle storage reserve reached')
 def sim(checkpoint,dataset,purpose):
  budget();dest=out/new_experiment_name(purpose)
  cmd=[sys.executable,'scripts/validate_native_mujoco.py','--spec',r['parent_spec'],'--checkpoint',str(checkpoint),'--output-dir',str(dest),'--commands','0','--wall-seconds','180','--run']
  if dataset:cmd+=['--student-dataset',str(dataset)]
  with (out/(purpose+'.log')).open('x') as log:
   result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=210)
  if result.returncode:raise RuntimeError('MuJoCo execution failed: '+str(dest))
  term=json.loads((dest/'terminal_report.json').read_text())
  if term['status']!='COMPLETED':raise RuntimeError('MuJoCo engineering failure: '+str(dest))
  return dest,term['result']
 try:
  torch.set_num_threads(4)
  teacher,_=load(r['teacher'],'RecoveryStudent');teacher.eval().requires_grad_(False)
  original=torch.load(r['dataset'],map_location='cpu',weights_only=False);data=copy.deepcopy(original)
  manifest=json.loads(Path(r['dataset']).with_name('manifest.json').read_text())
  from pxr import Usd,UsdPhysics
  spec=json.loads(Path(r['parent_spec']).read_text());stage=Usd.Stage.Open(spec['asset_path']);limits={}
  for prim in stage.Traverse():
   if prim.IsA(UsdPhysics.RevoluteJoint):
    j=UsdPhysics.RevoluteJoint(prim);limits[prim.GetName()]=np.deg2rad([j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get()])
  from .interface import ORDER
  joint_names=[n+('_spin' if 'wheel' in n else '_pitch') for n in ORDER]
  ranges=torch.tensor(np.array([limits[n] for n in joint_names]),dtype=torch.float32)[None]
  teacher_run,teacher_result=sim(r['teacher'],None,'teacher_mujoco_reference')
  baseline_run,baseline_result=sim(r['initial_student'],r['dataset'],'student_before_aggregation')
  write(out/'baselines.json',{'teacher':teacher_result,'student':baseline_result,'same_domain_only':True,'teacher_run':str(teacher_run),'student_run':str(baseline_run)})
  parts,inputs=replay_native(r['native_replay_run'])
  previous_inputs=manifest.get('aggregation',{}).get('all_replay_inputs_sha256',{})
  if inputs and all(previous_inputs.get(k)==v for k,v in inputs.items()):parts=[]
  for key,value in previous_inputs.items():
   if key in inputs and inputs[key]!=value:raise ValueError('historical replay identity changed')
  inputs={**previous_inputs,**inputs}
  mjparts,mjinputs=replay_mujoco(baseline_run,1000);parts+=mjparts;inputs.update(mjinputs)
  for replay in extra_replays:
   more,hashes=replay_mujoco(replay,1001)
   if not all(inputs.get(k)==v for k,v in hashes.items()):parts+=more;inputs.update(hashes)
  source=Path(r['initial_student']);previous_sim=baseline_run
  for round_id in range(1,r['max_rounds']+1):
   budget();stats=append_replay(data,parts,teacher,ranges,int(data['episode'].max())+1)
   folder=out/f'round_{round_id:02}';folder.mkdir();ds=folder/'RecoveryStudent.pt';torch.save(data,ds)
   m=copy.deepcopy(manifest);m.update(sampling_version=DATA_VERSION,source_checkpoint={'path':str(source.resolve()),'sha256':sha(source)},aggregation={'version':VERSION,'round':round_id,'all_replay_inputs_sha256':inputs,'new_states':stats,'replay_split':'train only; original heldout untouched','muJoCo_scope':'migration-domain labels are uncertain; no teacher control'},RecoveryStudent={'sha256':sha(ds)})
   write(folder/'manifest.json',m)
   for split in (1,2):
    for key in ('x','y','stage','episode','split'):
     if not torch.equal(data[key][data['split']==split],original[key][original['split']==split]):raise RuntimeError('heldout contamination')
   report=train(ds,folder/'fit','RecoveryStudent',steps=r['steps_per_round'],wall_s=min(1200,int(r['wall_limit_s']-(time.monotonic()-start)-240)),source_checkpoint=source)
   used+=report['offline_optimizer_steps'];candidate=folder/'fit/best_validation.pt'
   candidate_run,behavior=sim(candidate,ds,f'round{round_id}_student_mujoco')
   entry=behavior.get('score',{}).get('entry_index')
   item={'round':round_id,'student':str(candidate),'sha256':sha(candidate),'dataset':str(ds),'BC_steps':report['offline_optimizer_steps'],'new_states':stats,'MuJoCo':behavior,'MuJoCo_run':str(candidate_run),'accepted_native':False}
   results.append(item);write(out/'rounds.json',results)
   # Keep all rounds for native selection: domain transfer alone cannot select native best.
   source=candidate;previous_sim=candidate_run
   if round_id<r['max_rounds']:
    parts,more=replay_mujoco(previous_sim,1000+round_id);inputs.update(more)
  status='COMPLETED'
 except BaseException as exc:error=repr(exc);write(out/'failure.json',{'exception':error,'traceback':traceback.format_exc()})
 finally:
  summary={'status':status,'version':VERSION,'rounds':results,'offline_optimizer_steps':used,'online_PPO':0,'wall_s':time.monotonic()-start,'teacher_unchanged':sha(r['teacher'])==plan['input_sha256'][str(Path(r['teacher']).resolve())],'latest_candidate':results[-1]['student'] if results else None,'selection':'native independent validation required; no accepted replacement selected','error':error}
  write(out/'summary.json',summary);write(out/'budget.json',{'status':status,'BC_used':used,'BC_max':15000,'PPO':0,'wall_s':summary['wall_s'],'wall_max_s':3600,'bytes_used':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),'bytes_max':2**30})
 return summary


TASK_VERSION='recovery31_continuous_fidelity_distillation_v1'
CPU_TASK_VERSION='recovery31_cpu_mujoco_fidelity_distillation_v2'
CPU_CRITERIA={'mean_abs_vx_gap_mps':.10,'vx_rmse_mps':.15,'yaw_rmse_rad_s':.05,'posture_rmse_rad':.05,'height_rmse_m':.02,'entry_time_gap_s':.10,'contact_disagreement_max':.01,'extra_saturation_max':.02}


def compare_mujoco(teacher_run,student_run,criteria):
 """Fixed Teacher-defined support mask, identical policy clock; not a task certificate."""
 t=Path(teacher_run);s=Path(student_run)
 tm=json.loads((t/'manifest.json').read_text());sm=json.loads((s/'manifest.json').read_text())
 if tm['model_xml_sha256']!=sm['model_xml_sha256'] or tm['protocol']!=sm['protocol'] or tm.get('initial_yaw_rad',0)!=sm.get('initial_yaw_rad',0):raise ValueError('MuJoCo physical/protocol/initialization mismatch')
 tt=json.loads((t/'terminal_report.json').read_text());st=json.loads((s/'terminal_report.json').read_text())
 if any(z['status']!='COMPLETED' or not z['model_unchanged'] for z in [tt,st]):raise ValueError('MuJoCo engineering result incomplete')
 ta=dict(np.load(t/'trajectory.npz'));sa=dict(np.load(s/'trajectory.npz'));n=min(len(ta['state']),len(sa['state']))
 if not np.allclose(ta['policy_time_s'][:n],sa['policy_time_s'][:n],atol=1e-6,rtol=0):raise ValueError('MuJoCo policy clock mismatch')
 ts=ta['state'][:n];ss=sa['state'][:n];tp=ta['posture'][:n,:2];sp=sa['posture'][:n,:2]
 # Define usable comparison intervals solely from Teacher physical support, never from Student performance.
 mask=(ts[:,6]>=.30)&(abs(tp).max(-1)<=.15)&(ts[:,15]<=.18)&(ts[:,10:12]>2).all(-1)&(ts[:,12]<=.005)&(ts[:,13]>0)
 if mask.sum()<20:raise ValueError('Teacher lacks valid supported comparison intervals')
 rmse=lambda a:float(np.sqrt(np.mean(a*a)))
 tr=tt['result'];sr=st['result'];te=tr.get('score',{}).get('settle_s');se=sr.get('score',{}).get('settle_s')
 entry_gap=None if te is None or se is None else abs(te-se)
 values={'mean_abs_vx_gap_mps':float(abs(abs(ss[mask,7]).mean()-abs(ts[mask,7]).mean())),'vx_rmse_mps':rmse(ss[mask,7]-ts[mask,7]),'yaw_rmse_rad_s':rmse(ss[mask,9]-ts[mask,9]),'posture_rmse_rad':max(rmse(sp[mask,i]-tp[mask,i]) for i in [0,1]),'height_rmse_m':rmse(ss[mask,6]-ts[mask,6]),'entry_time_gap_s':entry_gap,'contact_disagreement_max':float(np.mean(~(ss[mask,10:12]>2).all(-1))),'extra_saturation_max':float(max(0,ss[mask,17].mean()-ts[mask,17].mean()))}
 complete=sr['controlled_s']>=tr['controlled_s']-.02
 passed=complete and entry_gap is not None and all(values[k] is not None and values[k]<=v for k,v in criteria.items())
 return {'passed':bool(passed),'metrics':values,'criteria':criteria,'student_matches_teacher_duration':complete,'teacher_controlled_s':tr['controlled_s'],'student_controlled_s':sr['controlled_s'],'teacher_accepted':tr['accepted'],'student_accepted':sr['accepted'],'teacher_comparison_samples':int(mask.sum()),'teacher_mask_sha256':__import__('hashlib').sha256(mask.tobytes()).hexdigest(),'teacher_run':str(t),'student_run':str(s),'scope':'same-MuJoCo Teacher fidelity only; Teacher defects need not be repaired; native/hardware NOT_CERTIFIED'}


def cpu_matrix(r,block_result,base,out,sequence,teacher_cache):
 """Only CPU child processes; no Isaac/GPU/public-native dispatch."""
 import random
 from native_stand.command_v6.domain.experiment_naming import new_experiment_name
 latest=block_result['rounds'][-1];current_native=latest['MuJoCo_run']
 teacher_native=json.loads((Path(block_result['rounds'][0]['student']).parents[2]/'baselines.json').read_text())['teacher_run']
 def execute(checkpoint,dataset,purpose,yaw):
  dest=out/new_experiment_name(purpose)
  cmd=[sys.executable,'scripts/validate_native_mujoco.py','--spec',base['parent_spec'],'--checkpoint',str(checkpoint),'--output-dir',str(dest),'--commands','0','--initial-yaw-rad',str(yaw),'--wall-seconds','180','--run']
  if dataset:cmd+=['--student-dataset',str(dataset)]
  with (out/(purpose+'.log')).open('x') as log:
   done=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=210)
  if done.returncode:raise RuntimeError('CPU MuJoCo diagnostic failed: '+str(dest))
  return str(dest)
 rows=[compare_mujoco(teacher_native,current_native,r['criteria'])]
 for seed in [20261010,20261011,20261012]:
  yaw=random.Random(seed).uniform(-.01,.01);key=str(seed)
  if key not in teacher_cache:teacher_cache[key]=execute(base['teacher'],None,f'teacher_yaw{seed}',yaw)
  student=execute(latest['student'],latest['dataset'],f'student_block{sequence:04}_yaw{seed}',yaw)
  row=compare_mujoco(teacher_cache[key],student,r['criteria']);row.update(seed=seed,initial_yaw_rad=yaw)
  rows.append(row)
 return {'passed':all(row['passed'] for row in rows),'rows':rows,'engine':'CPU MuJoCo only','criteria':r['criteria'],'scope':'four deterministic yaw settings, not Isaac or hardware certification'}

def fidelity(student_run,teacher_frozen_manifest,criteria):
 """Teacher-relative native behavior; no demand to exceed Teacher or zero drift."""
 root=Path.cwd();m=json.loads(Path(teacher_frozen_manifest).read_text());run=Path(student_run)
 summary=json.loads((run/'summary.json').read_text())
 if summary['status']!='COMPLETED' or summary['scored_episodes']!=4:raise ValueError('complete four-condition Student result required')
 rows=[]
 for evidence,row in zip(m['evidence'][:4],summary['results']):
  path=root/evidence['path']
  if sha(path)!=evidence['sha256'] or evidence['case_id']!=row['case_id']:raise ValueError('Teacher reference identity/order differs')
  t=json.loads(path.read_text());p=run/row['path']
  if sha(p)!=row['sha256']:raise ValueError('Student evidence SHA differs')
  s=json.loads(p.read_text());teacher=t['per_env'];student=s['per_env'][0]
  if student['version']!=teacher[0]['version']:raise ValueError('Teacher/Student scoring protocol differs')
  velocities=np.mean([e['velocity_means'] for e in teacher],axis=0)
  current=student.get('velocity_means');gap=abs(np.array(current)-velocities).tolist() if current else None
  settle=abs(student.get('settle_s',float('inf'))-np.mean([e['settle_s'] for e in teacher]))
  passed=student['accepted'] and all(e['accepted'] for e in teacher) and gap is not None and all(v<=limit for v,limit in zip(gap,criteria['velocity_gap_abs'])) and settle<=criteria['entry_time_gap_s']
  rows.append({'case_id':row['case_id'],'teacher_mean_abs_vx_vy_yaw':velocities.tolist(),'student_mean_abs_vx_vy_yaw':current,'velocity_gap':gap,'entry_time_gap_s':float(settle) if np.isfinite(settle) else None,'student_accepted':student['accepted'],'teacher_accepted':all(e['accepted'] for e in teacher),'fidelity_pass':bool(passed),'coverage':'Student1env versus historicalTeacher8env; not broad generalization certification'})
 return {'passed':all(r['fidelity_pass'] for r in rows),'rows':rows,'criteria':criteria,'goal':'copy Teacher, including tolerated residual drift; no superiority claim'}


def validate_task(r):
 import os
 required={'version','authorization','session','initial_block','teacher_frozen_manifest','native_request_template','output','cumulative_optimizer_limit','cumulative_round_limit','per_block_rounds','criteria','gpu_mode','minimum_free_disk_bytes'}
 cpu=r.get('version')==CPU_TASK_VERSION
 if cpu:required.add('route_change_receipt')
 if set(r)!=required or r['version'] not in (TASK_VERSION,CPU_TASK_VERSION) or r['authorization']!='AUTHORIZED' or r['session']!=os.environ.get('HAPI_SESSION_ID'):raise ValueError('continuous Recovery authorization/session differs')
 if r['cumulative_optimizer_limit'] is not None or r['cumulative_round_limit'] is not None or r['per_block_rounds']!=3 or r['gpu_mode']!=('cpu_only_no_isaac' if cpu else 'physical0_shared') or r['minimum_free_disk_bytes']!=2*2**30:raise ValueError('unlimited task/finite block/resource scope differs')
 if r['criteria']!=(CPU_CRITERIA if cpu else {'velocity_gap_abs':[.05,.02,.03],'entry_time_gap_s':.10}):raise ValueError('explicit Teacher fidelity criteria differ')
 p=Path(r['initial_block']);s=json.loads((p/'summary.json').read_text())
 if s['status']!='COMPLETED' or not s['teacher_unchanged']:raise ValueError('completed initial aggregation block required')
 old=json.loads((p/'request.json').read_text());latest=s['rounds'][-1]
 validate_small_lineage(latest['student'],latest['dataset'],old['parent_spec'])
 m=json.loads(Path(r['teacher_frozen_manifest']).read_text())
 if m['checkpoint_sha256']!=sha(old['teacher']):raise ValueError('Teacher reference differs')
 history=s['offline_optimizer_steps']
 if cpu:
  receipt=json.loads(Path(r['route_change_receipt']).read_text())
  if receipt['reason']!='USER_ROUTE_CHANGE_CPU_MUJOCO_ONLY' or Path(receipt['completed_CPU_block'])!=p:raise ValueError('CPU migration receipt source differs')
  history=receipt['historical_BC_steps']
 return {'status':'PLAN','files_written':0,'task_optimizer_limit':None,'task_round_limit':None,'block_rounds':3,'block_BC_steps_max':15000,'block_CPU_wall_max_s':3600,'block_native_wall_max_s':0 if cpu else 1800,'online_PPO':0,'initial_BC_actual':history,'identity_sha256':{str(Path(r[k]).resolve()):sha(r[k]) for k in ('teacher_frozen_manifest','native_request_template')}|{str((p/'summary.json').resolve()):sha(p/'summary.json')}}


def run_task(r):
 """The existing finite cycle owns its children; no polling external logs/processes."""
 import os,shutil
 from native_stand.command_v6.completion import write
 from native_stand.command_v6.domain.experiment_naming import new_experiment_name
 plan=validate_task(r);out=Path(r['output']);out.mkdir(parents=True,exist_ok=False)
 write(out/'request.json',r);write(out/'identity.json',plan)
 block=Path(r['initial_block']);s=json.loads((block/'summary.json').read_text());base=json.loads((block/'request.json').read_text());template=json.loads(Path(r['native_request_template']).read_text());latest=s['rounds'][-1]
 state={'status':'RUNNING','task_version':r['version'],'cumulative_optimizer_limit':None,'cumulative_round_limit':None,'historical_BC_steps':plan['initial_BC_actual'],'new_BC_steps':0,'total_BC_steps':plan['initial_BC_actual'],'completed_blocks':[],'online_PPO':0,'student':latest['student'],'accepted_fidelity':False,'stop_file':str(out/'STOP'),'first_stage':'CPU_AGGREGATION'}
 teacher_cache={}
 extra_replays=[]
 started=time.monotonic();sequence=1
 try:
  while True:
   if (out/'STOP').exists():state.update(status='STOPPED_USER');break
   if shutil.disk_usage(out).free<r['minimum_free_disk_bytes']:raise RuntimeError('free disk reserve reached; preserve all existing runs')
   current=dict(base);name=new_experiment_name(f'recovery1500_small31_fidelity_block{sequence:04}')
   current.update(dataset=latest['dataset'],initial_student=latest['student'],output=str(out/name))
   state.update(stage='CPU_AGGREGATION',next_block=current['output']);write(out/'task_state.json',state)
   result=run(current,extra_replays=extra_replays)
   state['new_BC_steps']+=result['offline_optimizer_steps'];state['total_BC_steps']=state['historical_BC_steps']+state['new_BC_steps']
   if result['status']!='COMPLETED':raise RuntimeError('aggregation block engineering failure: '+str(result['error']))
   latest=result['rounds'][-1]
   if (out/'STOP').exists():state.update(status='STOPPED_USER');break
   if r['version']==CPU_TASK_VERSION:
    state.update(stage='CPU_MUJOCO_INDEPENDENT_FIDELITY',student=latest['student']);write(out/'task_state.json',state)
    comparison=cpu_matrix(r,result,base,out,sequence,teacher_cache);write(out/f'mujoco_fidelity_{sequence:04}.json',comparison)
    state['completed_blocks'].append({'block':current['output'],'BC_steps':result['offline_optimizer_steps'],'candidate':latest,'fidelity':comparison,'Isaac_episodes':0,'GPU_used':False})
    state['accepted_fidelity']=comparison['passed'];write(out/'task_state.json',state)
    if comparison['passed']:state.update(status='COMPLETED_MUJOCO_FIDELITY');break
    extra_replays=[row['student_run'] for row in comparison['rows'] if row.get('seed')==20261010]
    sequence+=1;continue
   request=copy.deepcopy(template);name=new_experiment_name(f'recovery1500_small31_fidelity_native{sequence:04}')
   request.update(version='recovery31_distilled_native_student_only_v2',session=r['session'],checkpoints=[{'checkpoint':latest['student'],'sha256':sha(latest['student'])}],dataset=latest['dataset'],dataset_sha256=sha(latest['dataset']),dataset_manifest_sha256=sha(Path(latest['dataset']).with_name('manifest.json')),output=str(Path.cwd()/'runs'/name))
   from .recovery_distilled_native import validate as validate_native
   validate_native(request);path=out/f'native_request_{sequence:04}.json';write(path,request)
   cmd=[sys.executable,'-m','native_stand.command_v6.supervisor','--spec',request['parent_spec'],'--output-dir',request['output'],'--single-screen-request',str(path),'--resource-mode','shared','--sharing-authorization','User authorized continuous Recovery fidelity distillation; GPU0 shared, no preemption; each native block4 episodes/1800s/256MiB, PPO0/optimizer0']
   state.update(stage='NATIVE_INDEPENDENT_EVALUATION',native_run=request['output'],student=latest['student']);write(out/'task_state.json',state)
   with (out/f'native_{sequence:04}.log').open('x') as log:
    # Wait only on this owner's bounded child, never monitor another training job.
    job=subprocess.run(cmd+['--launch'],stdout=log,stderr=subprocess.STDOUT,timeout=1860)
   if job.returncode:raise RuntimeError('native child failed; no automatic physical retry')
   terminal=json.loads((Path(request['output'])/'terminal_report.json').read_text())
   if terminal['status']!='COMPLETED':raise RuntimeError('native child evidence incomplete')
   comparison=fidelity(request['output'],r['teacher_frozen_manifest'],r['criteria']);write(out/f'fidelity_{sequence:04}.json',comparison)
   state['completed_blocks'].append({'block':current['output'],'BC_steps':result['offline_optimizer_steps'],'native_run':request['output'],'native_wall_s':terminal['wall_s'],'native_episodes':4,'candidate':latest,'fidelity':comparison})
   state['accepted_fidelity']=comparison['passed'];write(out/'task_state.json',state)
   if comparison['passed']:state.update(status='COMPLETED_FIDELITY');break
   # This completed Student-native trajectory becomes the next block's train-only replay.
   base['native_replay_run']=request['output'];sequence+=1
 except BaseException as exc:state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
 finally:
  state.update(wall_s=time.monotonic()-started,student=latest['student']);write(out/'task_state.json',state);write(out/'terminal_report.json',state)
 return state
