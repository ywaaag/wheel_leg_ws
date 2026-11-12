"""Same31D Motion compression pipeline. Default plan never writes/optimizes/simulates."""
import argparse,json,subprocess
from pathlib import Path
from native_stand.students import motion_distillation_pipeline as pipeline
from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('stage',choices=['plan','prepare','relabel','fit','sim2sim','replay','cycle','export'])
 p.add_argument('--teacher',type=Path,required=True);p.add_argument('--parent-spec',type=Path,required=True)
 p.add_argument('--capture-run',type=Path);p.add_argument('--dataset',type=Path);p.add_argument('--student',type=Path)
 p.add_argument('--replay-data',type=Path);p.add_argument('--rollout-run',type=Path)
 p.add_argument('--cycle-request',type=Path,help='Finite replay/refit/MuJoCo request; native/PPO/export never automatic')
 p.add_argument('--output-dir',type=Path);p.add_argument('--steps',type=int,default=5000);p.add_argument('--wall-seconds',type=int,default=1200)
 p.add_argument('--execute',action='store_true');p.add_argument('--authorization',help='Concrete current user-approved scope; flag is a record, not self-granted approval')
 a=p.parse_args()
 if a.stage=='cycle':
  if not a.cycle_request:p.error('--cycle-request required')
  from native_stand.students import motion_distillation_workflow as workflow
  from native_stand.students.data import sha
  request=json.loads(a.cycle_request.read_text())
  if request.get('version') in ('motion31_mcu_distillation_cycle_v2','motion31_mcu_distillation_cycle_v3'):
   from motion_mcu_distillation import validate,run
   from native_stand.students.data import sha
   if sha(a.teacher)!=request['teacher_sha256'] or sha(a.parent_spec)!=request['parent_spec_sha256']:raise ValueError('MCU distillation CLI/source differs')
   print(json.dumps(run(request,a.authorization) if a.execute else validate(request),indent=2,ensure_ascii=False));return
  if request.get('version') in ('recovery31_continuous_fidelity_distillation_v1','recovery31_cpu_mujoco_fidelity_distillation_v2'):
   from native_stand.students import recovery_distillation_cycle as recovery_workflow
   base=json.loads((Path(request['initial_block'])/'request.json').read_text())
   if sha(a.teacher)!=sha(base['teacher']) or sha(a.parent_spec)!=sha(base['parent_spec']):raise ValueError('continuous Recovery CLI identity differs')
   if a.execute and not a.authorization:p.error('explicit continuous Recovery authorization required')
   result=recovery_workflow.run_task(request) if a.execute else recovery_workflow.validate_task(request)
   print(json.dumps(result,indent=2,ensure_ascii=False))
   if result.get('status')=='FAILED':raise SystemExit(1)
   return
  if request.get('version')=='recovery31_state_aggregation_cycle_v1':
   from native_stand.students import recovery_distillation_cycle as workflow
  if sha(a.teacher)!=sha(request['teacher']) or sha(a.parent_spec)!=sha(request['parent_spec']):raise ValueError('cycle CLI/request Teacher or parent differs')
  if request.get('version')=='recovery31_state_aggregation_cycle_v1':
   if a.execute and not a.authorization:p.error('explicit current Recovery aggregation authorization required')
   result=workflow.run(request) if a.execute else workflow.validate(request)
  else:result=workflow.run(request,a.authorization) if a.execute else workflow.validate(request)
  print(json.dumps(result,indent=2,ensure_ascii=False))
  if result.get('status')=='FAILED':raise SystemExit(1)
  return
 pipeline.validate_teacher(a.teacher,a.parent_spec)
 required={'plan':['capture_run'],'prepare':['capture_run','output_dir'],'relabel':['dataset','output_dir'],'fit':['dataset','output_dir'],'sim2sim':['student','output_dir'],'replay':['rollout_run','output_dir'],'export':['student','dataset','output_dir']}[a.stage]
 for name in required:
  if getattr(a,name) is None:p.error('--'+name.replace('_','-')+' required for '+a.stage)
 if a.output_dir:
  if a.output_dir.exists():raise FileExistsError(a.output_dir)
  validate_new_experiment_name(a.output_dir.name)
 if a.stage=='relabel':
  if a.execute and (not a.authorization or not a.authorization.strip()):p.error('execution requires explicit authorized scope')
  print(json.dumps(pipeline.relabel(a.teacher,a.parent_spec,a.dataset,a.output_dir,execute=a.execute),indent=2,ensure_ascii=False));return
 if a.stage=='fit' and a.dataset and a.dataset.exists():
  from native_stand.students.data import sha
  m=json.loads(a.dataset.with_name('manifest.json').read_text())
  if m['teacher_sha256']!=sha(a.teacher) or m['parent_spec_sha256']!=sha(a.parent_spec):raise ValueError('CLI teacher/spec differs from dataset')
 if a.stage=='sim2sim' and a.student:
  import torch
  from native_stand.students.data import sha
  d=torch.load(a.student,map_location='cpu',weights_only=False)
  if d.get('distillation',{}).get('teacher_sha256')!=sha(a.teacher):raise ValueError('CLI teacher differs from Student lineage')
 if not a.execute or a.stage=='plan':
  if a.stage in ('plan','prepare'):result=pipeline.plan(a.teacher,a.parent_spec,a.capture_run)
  else:result={'version':pipeline.VERSION,'stage':a.stage,'files_written':0,'simulation':0,'optimizer_updates':0,'status':'PLAN_ONLY','execution_requires':'--execute plus explicitly approved --authorization'}
  if a.stage=='fit':result.update(offline_step_limit=a.steps,wall_limit_s=a.wall_seconds,output=str(a.output_dir))
  if a.stage=='sim2sim':
   # No --run on emitted plan command, to avoid copy/paste silently executing.
   result['command']=pipeline.mujoco_command(a.student,a.parent_spec,a.output_dir,min(a.wall_seconds,1800))[:-1]
  print(json.dumps(result,indent=2,ensure_ascii=False));return
 if not a.authorization or not a.authorization.strip():p.error('execution requires explicit authorized scope, never infer approval from old budgets')
 if a.stage=='prepare':result=pipeline.prepare(a.teacher,a.parent_spec,a.capture_run,a.output_dir)
 elif a.stage=='fit':result=pipeline.fit(a.dataset,a.output_dir,steps=a.steps,wall_s=a.wall_seconds,initial_student=a.student,replay=a.replay_data)
 elif a.stage=='export':result=pipeline.export_onnx(a.student,a.parent_spec,a.dataset,a.output_dir)
 elif a.stage=='replay':result=pipeline.collect_replay(a.rollout_run,a.teacher,a.parent_spec,a.output_dir)
 else:
  if not 1<=a.wall_seconds<=1800:raise ValueError('sim2sim wall bound must be1..1800s')
  command=pipeline.mujoco_command(a.student,a.parent_spec,a.output_dir,a.wall_seconds)
  # Own one bounded child; no log polling or detached watcher.
  child=subprocess.run(command,timeout=a.wall_seconds+30)
  if child.returncode:raise RuntimeError('sim2sim child failed:'+str(child.returncode))
  terminal=json.loads((a.output_dir/'terminal_report.json').read_text())
  result={'status':terminal['status'],'old_gate_all_passed':terminal['all_passed'],'capability_certified':False,
          'all_passed_with_swing':len(terminal['results'])==7 and all(r.get('passed') and r.get('supplemental_swing_constraint',{}).get('full_passed') for r in terminal['results'])}
 with (a.output_dir/'pipeline_execution.json').open('x') as f:json.dump({'stage':a.stage,'authorization':a.authorization,'version':pipeline.VERSION,'result':result},f,indent=2,ensure_ascii=False)
 print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
