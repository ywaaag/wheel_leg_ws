"""Fixed-checkpoint PPO0 backfill under public supervisor; no checkpoint/log polling."""
import json,os,subprocess,sys,time,uuid
from pathlib import Path
from .data import sha
from .motion_quick_evaluation import configuration,validate as validate_quick
from native_stand.command_v6.completion import read,write
VERSION='motion31_saved_checkpoint_backfill_v1'
MATRIX_VERSION='motion31_saved_checkpoint_matrix_screen_v2'
GROUPED_VERSION='motion31_saved_checkpoint_grouped_screen_v3'

def validate(request,*,current_session=None,verify_inputs=True):
 required={'version','experiment','parent_spec','parent_spec_sha256','parent_output','output','session','physical_gpu','gpu_uuid','authorization','configuration','checkpoints','wall_limit_s','output_max_bytes','online_PPO'}
 if set(request)!=required or request['version'] not in (VERSION,MATRIX_VERSION,GROUPED_VERSION) or request['online_PPO']!=0:raise ValueError('quick backfill request schema differs')
 if request['authorization']!='AUTHORIZED' or not request['session'] or (current_session is not None and request['session']!=current_session):raise ValueError('quick backfill needs current explicit authorization/session')
 matrix=request['version'] in (MATRIX_VERSION,GROUPED_VERSION)
 if request['configuration']!=configuration() or (not matrix and (request['wall_limit_s']!=1800 or request['output_max_bytes']!=384*2**20)) or (matrix and (not 600<=request['wall_limit_s']<=21600 or not 128*2**20<=request['output_max_bytes']<=1024*2**20)):raise ValueError('backfill wall/storage/config differs')
 rows=request['checkpoints']
 if any(set(r)!={'update','checkpoint','sha256'} for r in rows):raise ValueError('checkpoint row schema differs')
 if not matrix and (len(rows)!=2 or [r.get('update') for r in rows]!=[100,200]):raise ValueError('backfill only explicitly requested100/200')
 if matrix and (not 1<=len(rows)<=8 or any(type(r['update']) is not int or r['update']<=0 for r in rows) or len({r['update'] for r in rows})!=len(rows)):raise ValueError('matrix screen needs unique positive checkpoint updates')
 if request['physical_gpu']!=0 or not request['gpu_uuid'].startswith('GPU-'):raise ValueError('backfill requires explicit inherited GPU0')
 if verify_inputs:
  p=Path(request['parent_spec']);s=read(p)
  if sha(p)!=request['parent_spec_sha256']:raise ValueError('backfill parent spec SHA differs')
  from .requests import validate as validate_spec
  errors=validate_spec(s)
  if errors:raise ValueError(errors)
  local=dict(s,checkpoint_quick_evaluation=request['configuration']);validate_quick(local)
  if s['device']['physical_index']!=request['physical_gpu'] or s['device']['uuid']!=request['gpu_uuid']:raise ValueError('backfill device differs from parent')
  from .model import load
  for row in rows:
   cp=Path(row['checkpoint'])
   if cp.parent!=Path(request['parent_output']) or cp.name!=f"model_update_{row['update']}.pt" or sha(cp)!=row['sha256']:raise ValueError('backfill checkpoint identity/path differs')
   _,data=load(cp,'MotionZeroStudent')
   if data.get('update')!=row['update'] or data.get('spec_sha256')!=request['parent_spec_sha256']:raise ValueError('backfill checkpoint internal identity differs')
 return request

def launch(args):
 from native_stand.command_v6.domain.resource_policy import resolve_resources
 from native_stand.command_v6.tools.health_check import gpu_snapshot,own_process_start
 req_path=Path(args.quick_screening_request).resolve();req=validate(read(req_path),current_session=os.environ.get('HAPI_SESSION_ID'))
 if Path(args.spec).resolve()!=Path(req['parent_spec']) or Path(args.output_dir).resolve()!=Path(req['output']):raise ValueError('backfill CLI parent/output differs')
 if args.hapi_session!=req['session'] or args.physical_gpu!=req['physical_gpu']:raise ValueError('backfill CLI session/GPU differs')
 if args.formal_only or args.next_spec or args.evaluation_only or args.wait:raise ValueError('backfill cannot combine other submission modes')
 policy=resolve_resources(read(args.spec),mode=args.resource_mode,sharing_authorization=args.sharing_authorization)
 output=Path(req['output']);plan={'status':'PASS','kind':req['version'],'checkpoints':req['checkpoints'],'online_PPO':0,'scored_episodes':len(req['checkpoints'])*(7 if req['version']==MATRIX_VERSION else 56),'wall_limit_s':req['wall_limit_s'],'output_max_bytes':req['output_max_bytes'],'files_written':0,'simulation':0,'parent_run_modified':False,'resource_policy':policy}
 if output.exists():raise FileExistsError(output)
 if not args.launch and not args.check_submission:print(json.dumps(plan,indent=2));return
 gpu=gpu_snapshot()
 devices=[d for d in gpu.get('devices',[]) if d['index']==req['physical_gpu']]
 if gpu.get('status')!='OK' or len(devices)!=1 or devices[0]['uuid']!=req['gpu_uuid']:raise ValueError('backfill GPU identity unavailable')
 if policy['mode']!='shared' or not policy['sharing_authorized']:raise ValueError('backfill requires explicit shared authorization')
 if float(devices[0]['memory_total_mib'])-float(devices[0]['memory_used_mib'])<6144:raise ValueError('backfill requires6GiB free GPU memory; no preemption')
 import shutil
 if shutil.disk_usage(output.parent).free<req['output_max_bytes']+256*2**20:raise ValueError('backfill filesystem reserve insufficient')
 plan['resource_verified']=True
 if not args.launch:print(json.dumps(plan,indent=2));return
 # Exclusive directory claim is the single submission/attempt owner, no parent reservation.
 output.mkdir(exist_ok=False);attempt=uuid.uuid4().hex
 from native_stand.command_v6.tools.source_identity import record_configuration
 record_configuration(output,Path(req['parent_spec']))
 (output/'request.json').write_text(json.dumps(req,indent=2))
 write(output/'budget.json',{'version':req['version'],'attempt_id':attempt,'status':'RUNNING','limits':{'checkpoints':len(req['checkpoints']),'online_PPO':0,'wall_s':req['wall_limit_s'],'bytes':req['output_max_bytes']},'used':{'checkpoints':0,'online_PPO':0},'parent_budget_modified':False})
 write(output/'manifest.json',{'attempt_id':attempt,'request_sha256':sha(output/'request.json'),'submitted_unix_s':time.time(),'gpu_snapshot':gpu,'resource_policy':policy,'hapi_session':req['session'],'notification_route':{'owner_session':req['session'],'target_session':req['session']},'kind':req['version']})
 command=[sys.executable,'-m',__name__,'--worker','--request',str(output/'request.json'),'--attempt-id',attempt]
 with (output/'owner.log').open('x') as log:
  child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':read(args.spec)['execution_source_root']})
 started=own_process_start(child.pid)
 # One bounded submission receipt check, never training log polling.
 alive=Path(f'/proc/{child.pid}/cmdline').read_bytes().split(b'\0')
 verified=b'native_stand.students.motion_quick_backfill' in alive and str(output/'request.json').encode() in alive and attempt.encode() in alive
 receipt={'status':'SUBMITTED' if verified else 'UNKNOWN','submitted':True,'submission_verified':verified,'owner_pid':child.pid,'attempt_id':attempt,'command':command,'process_started_unix_s':started,'output':str(output),'online_PPO':0,'training_restarted':False,'results_ready':False}
 write(output/'startup_handshake.json',receipt);print(json.dumps(receipt,indent=2))

def verify_child_request(request):
 path=Path(request['standalone_request']);r=validate(read(path))
 if sha(path)!=request['standalone_request_sha256']:raise ValueError('standalone request SHA differs')
 if request['parent_spec']!=r['parent_spec'] or request['parent_spec_sha256']!=r['parent_spec_sha256'] or request['parent_output']!=r['parent_output']:raise ValueError('backfill child parent mismatch')
 row=next((x for x in r['checkpoints'] if x['update']==request['update']),None)
 if not row or row['checkpoint']!=request['checkpoint'] or row['sha256']!=request['checkpoint_sha256']:raise ValueError('backfill child checkpoint not authorized')
 ledger=read(Path(r['output'])/'budget.json');manifest=read(Path(r['output'])/'manifest.json')
 if ledger['attempt_id']!=request['attempt_id'] or ledger['status']!='RUNNING' or manifest['attempt_id']!=request['attempt_id'] or ledger['used']['checkpoints']>=len(r['checkpoints']):raise ValueError('backfill exact-ID/remaining budget differs')
 return r

def worker(path,attempt):
 path=Path(path).resolve();r=validate(read(path));out=Path(r['output']);manifest=read(out/'manifest.json');deadline=manifest['submitted_unix_s']+r['wall_limit_s'];rows=[];failure=None;status='FAILED'
 try:
  for row in r['checkpoints']:
   remaining=min(7200 if r['version'] in (MATRIX_VERSION,GROUPED_VERSION) else 900,deadline-time.time()-45)
   if remaining<60:raise TimeoutError('backfill cumulative wall reserve reached')
   size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
   if size>r['output_max_bytes']-128*2**20:raise TimeoutError('backfill storage reserve reached')
   folder=out/f"update_{row['update']}";folder.mkdir(exist_ok=False)
   child_request={'standalone_request':str(path),'standalone_request_sha256':sha(path),'parent_spec':r['parent_spec'],'parent_spec_sha256':r['parent_spec_sha256'],'parent_output':r['parent_output'],'attempt_id':attempt,'checkpoint':row['checkpoint'],'checkpoint_sha256':row['sha256'],'update':row['update'],'deadline_unix_s':time.time()+remaining,'output':str(folder)}
   write(folder/'request.json',child_request)
   command=['timeout','--signal=TERM','--kill-after=15s',f'{remaining-20}s',sys.executable,'-m','native_stand.students.motion_quick_evaluation','--request',str(folder/'request.json')]
   start=time.monotonic()
   with (folder/'runtime.log').open('x') as stream:
    code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=remaining).returncode
   summary=read(folder/'summary.json',{})
   valid=code==0 and summary.get('checkpoint_sha256')==row['sha256'] and summary.get('update')==row['update'] and len(summary.get('commands',[]))==7 and all(len(c.get('per_env',[]))==(1 if r['version']==MATRIX_VERSION else 8) for c in summary.get('commands',[]))
   result={'update':row['update'],'status':'COMPLETED' if valid else 'FAILED','exit_code':code,'wall_s':time.monotonic()-start,'summary':str(folder/'summary.json'),'summary_sha256':sha(folder/'summary.json') if (folder/'summary.json').exists() else None}
   rows.append(result);book=read(out/'budget.json');book['used']['checkpoints']+=1;write(out/'budget.json',book)
   if not valid:raise RuntimeError(f"update{row['update']} incomplete/failed; no retries")
  status='COMPLETED'
 except BaseException as exc:failure=repr(exc)
 finally:
  book=read(out/'budget.json');book.update(status=status,failure=failure,wall_s=time.time()-manifest['submitted_unix_s']);write(out/'budget.json',book)
  write(out/'terminal_report.json',{'status':status,'attempt_id':attempt,'failure':failure,'results':rows,'online_PPO':0,'parent_run_modified':False,'wall_s':book['wall_s']})
  lines=['# Motion已保存checkpoint全命令快速补评',f'状态：{status}；PPO0；失败：{failure}。',f"每模型七命令×{1 if r['version']==MATRIX_VERSION else 8}评分，原完整窗口；诊断布局，不授accepted，不替代原250验收。"]
  for row in rows:
   lines.append(f"## update{row['update']}：{row['status']}，{row['wall_s']:.1f}秒")
   summary=read(row['summary'],{})
   for c in summary.get('commands',[]):lines.append(f"命令{c['command']}：{c['passed']}/{c['scored_envs']}；逐环境见{row['summary']}")
  if r['version']==GROUPED_VERSION:
   try:
    ranking=baseline_ranking(r,out)
    write(out/'baseline_ranking.json',ranking)
    lines.extend(['## 下一次训练基线筛选',json.dumps(ranking,ensure_ascii=False,indent=2)])
   except Exception as error:
    write(out/'ranking_error.json',{'error':repr(error)});lines.append('基线排名失败：'+repr(error))
  (out/'completion_report.md').write_text('\n\n'.join(lines)+'\n')
  write(out/'analysis.json',{'status':status,'actual_updates':0,'online_PPO':0,'accepted':False})
  write(out/'candidate_selection.json',{'best_observed_candidate':None,'accepted_candidate':None,'scope':'diagnostic screening only'})
  write(out/'report_ready.json',{'event_id':attempt[:24],'status':'report_ready','report_sha256':sha(out/'completion_report.md')})
  try:
   from native_stand.command_v6.completion import notify
   notify(out)
  except Exception as error:write(out/'notification_error.json',{'error':repr(error),'training_failure':False})
 return 0 if status=='COMPLETED' else 1

def baseline_ranking(request,out):
 """Read-only complete-matrix comparison; source remains a distinct fallback."""
 import collections
 parent=Path(request['parent_output']);candidates=[]
 def add(cp,update,digest,cases,origin):
  if len(cases)!=7 or {tuple(c['command'][:2]) for c in cases}!={(0.,0.),(.5,0.),(1.,0.),(1.5,0.),(2.,0.),(0.,.5),(0.,-.5)}:return
  if sha(cp)!=digest:raise ValueError('ranking checkpoint SHA differs')
  rows=[]
  for c in cases:
   es=c['per_env'];ts=[e.get('tracking',{}) for e in es]
   if not es or any(not e.get('hold_complete') or not e.get('survival_passed') for e in es):complete=False
   else:complete=True
   rows.append({'command':c['command'],'scored_envs':len(es),'passed':sum(bool(e.get('accepted')) for e in es),'all_passed':all(e.get('accepted') for e in es),
    'survival_complete':complete,'MAE_mean':sum(t['MAE'] for t in ts)/len(ts),'pitch_max_rad':max(t['pitch_hold_abs_max_rad'] for t in ts),
    'failed_checks':dict(collections.Counter(k for e in es for k in e.get('failed_checks',[])))})
  old=sum(x['all_passed'] for x in rows if not x['command'][1]);total=sum(x['all_passed'] for x in rows)
  candidates.append({'checkpoint':cp,'sha256':digest,'update':update,'origin':origin,'old_commands_passed':old,'commands_passed':total,
   'yaw_MAE_mean':sum(x['MAE_mean'] for x in rows if x['command'][1])/2,'pitch_peak_rad':max(x['pitch_max_rad'] for x in rows),'cases':rows})
 for c in read(parent/'candidate_registry.json',[]):
  if not c.get('complete_matrix'):continue
  cases=[]
  for ev in c['evaluations']:
   p=Path(ev['path'])
   options=[p] if p.is_absolute() else [parent/p,parent.parent/'source_eval'/p]
   matches=[q for q in options if q.is_file() and sha(q)==ev['sha256']]
   if len(matches)!=1:raise ValueError('ranking raw evaluation SHA missing/ambiguous')
   p=matches[0]
   d=read(p);cases.append({'command':d['case']['command'],'per_env':d['per_env']})
  add(c['checkpoint'],c.get('update',0),c['sha256'],cases,'existing_full_matrix')
 # Completed earlier independent screen is reusable, but never merged across models.
 for p in parent.parent.parent.glob('*motion600_700_800_all_commands_ppo0/update_600/summary.json'):
  d=read(p)
  if d.get('status')=='COMPLETED' and Path(d['checkpoint']).parent==parent:add(d['checkpoint'],600,d['checkpoint_sha256'],d['commands'],str(p))
 for row in request['checkpoints']:
  p=Path(out)/f"update_{row['update']}"/'summary.json'
  d=read(p,{})
  if d.get('status')=='COMPLETED':add(row['checkpoint'],row['update'],row['sha256'],d['commands'],str(p))
 key=lambda c:(-c['old_commands_passed'],-c['commands_passed'],c['yaw_MAE_mean'],c['pitch_peak_rad'],c['update'])
 new=sorted([c for c in candidates if Path(c['checkpoint']).parent==parent],key=key)
 fallback=[c for c in candidates if Path(c['checkpoint']).parent!=parent]
 return {'selection_rule':'old-command full pass count, complete command pass count, yaw MAE, pitch peak, earlier update; per-command all scored envs required',
  'best_new_baseline_candidate':new[0] if new else None,'source_fallback':fallback,'ranked_new_candidates':new,
  'accepted':False,'scope':'single-seed screening plus reused full matrices; different execution layouts recorded; baseline recommendation not final capability certification; no automatic training'}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--request',type=Path,required=True);p.add_argument('--attempt-id',required=True);a=p.parse_args();raise SystemExit(worker(a.request,a.attempt_id))
