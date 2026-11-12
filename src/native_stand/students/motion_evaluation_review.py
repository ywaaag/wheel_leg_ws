"""Human-readable Motion evidence, and opt-in warm-App deterministic screening."""
import copy,json,math,time
from collections import deque
from pathlib import Path
from .data import sha


def representative_cases(cases):
 """Only declared cases, deterministic representatives; never unlock a command."""
 if not cases or len({r['case_id'] for r in cases})!=len(cases):raise ValueError('distinct declared Motion cases required')
 for row in cases:
  if row.get('kind')!='student' or row.get('family')!='motion_native_normal' or len(row.get('command',[]))!=3:raise ValueError('warm screening requires independent normal Motion cases')
  if not all(math.isfinite(v) for v in row['command']):raise ValueError('nonfinite declared command')
 selected=[]
 def add(rows,key=None):
  if rows:
   row=max(rows,key=key) if key else rows[0]
   if row['case_id'] not in {r['case_id'] for r in selected}:selected.append(copy.deepcopy(row))
 add([r for r in cases if r['command'][:2]==[0,0]])
 for axis,sign in [(0,1),(0,-1),(1,1),(1,-1)]:
  add([r for r in cases if sign*r['command'][axis]>0 and r['command'][1-axis]==0],lambda r:abs(r['command'][axis]))
 add([r for r in cases if r['command'][:2]==[.5,0]])
 if not selected:raise ValueError('no representative axis/zero conditions')
 return selected


def warm_screen(spec,output,actor,checkpoint,update,*,stop=None,records=None,evaluate=None):
 """Caller closes training env first; this function never starts Isaac or trains."""
 from native_stand.command_v6.completion import write
 from .motion_quick_evaluation import validate,WARM_VERSION
 cfg=validate(spec)
 if cfg is None or cfg['version']!=WARM_VERSION:raise ValueError('explicit warm-App v2 required')
 folder=Path(output)/f'quick_update_{update}';folder.mkdir(exist_ok=False)
 started=time.monotonic();deadline=started+cfg['wall_limit_s'];selected=representative_cases(spec['student_evaluation']['cases'])
 write(folder/'requested.json',{'version':WARM_VERSION,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'update':update,'cases':selected,'selection_eligible':False,'full_window':'parent protocol unchanged','execution':cfg['execution']})
 import torch
 from .resume import rng_state,restore_rng
 rng=rng_state();mode=actor.training;weights={k:v.detach().clone() for k,v in actor.state_dict().items()}
 try:
  if records is None:
   if evaluate is None:
    from .training import evaluate_cases
    evaluate=evaluate_cases
   local=copy.deepcopy(spec);local['performance'].update(evaluation_group_commands=False,evaluation_reuse_environment=True,evaluation_gpu_buffer=True)
   local['student_evaluation']['cases']=selected;local['student_evaluation']['max_output_bytes']=cfg['output_max_bytes']
   # evaluate_cases binds raw results to the frozen parent declaration, not to
   # the runtime-local case/layout overrides. Copy only this small JSON file.
   parent_spec=Path(output)/'source_snapshot/spec.json'
   frozen=parent_spec.read_bytes()
   snapshot=folder/'source_snapshot';snapshot.mkdir()
   with (snapshot/'spec.json').open('xb') as stream:stream.write(frozen)
   write(folder/'effective_layout.json',{'parent_spec_path':str(parent_spec.resolve()),
    'parent_spec_sha256':sha(parent_spec),'performance':local['performance'],
    'cases':selected,'physical_envs':1,'scored_envs_per_command':1,'selection_eligible':False})
   records=evaluate(local,folder,actor,selected,checkpoint=checkpoint,screen_single_env=True,stop=lambda:time.monotonic()>=deadline or bool(stop and stop()))
   source='warm_AppLauncher_independent_single_env';base=folder
  else:source='derived_from_complete_evaluation_no_new_simulation';base=Path(output)
  by_id={r['case']['case_id']:r for r in records};commands=[];intervals=0;intervals_known=True
  for case in selected:
   rec=by_id.get(case['case_id'])
   if rec is None or not rec.get('per_env') or rec['case'].get('sha256')!=sha(checkpoint):raise ValueError('quick evidence incomplete or checkpoint mismatch')
   f=Path(rec['path']);f=f if f.is_absolute() else base/f
   if sha(f)!=rec['sha256']:raise ValueError('quick raw result SHA differs')
   if not source.startswith('derived'):
    payload=json.loads(f.read_text());trace=payload.get('trace')
    if trace and 'path' in trace:
     import numpy as np
     tf=f.parent/trace['path']
     if trace.get('sha256') and sha(tf)!=trace['sha256']:raise ValueError('quick trace SHA differs')
     with np.load(tf) as a:intervals+=int(a['state'].shape[0]*a['state'].shape[1])
    else:intervals_known=False
   commands.append({'case_id':case['case_id'],'command':case['command'],'passed':sum(e['accepted'] for e in rec['per_env']),'scored_envs':len(rec['per_env']),'per_env':rec['per_env'],'raw':str(f.resolve()),'sha256':rec['sha256']})
  if not all(torch.equal(v,actor.state_dict()[k]) for k,v in weights.items()):raise RuntimeError('screening modified Actor weights')
  summary={'status':'COMPLETED','version':WARM_VERSION,'checkpoint_sha256':sha(checkpoint),'update':update,'selection_eligible':False,'accepted_candidate':None,'condition_count':len(commands),'commands':commands,'execution':source,'new_scored_episodes':0 if source.startswith('derived') else len(commands),'physical_serial_equivalence':'NOT_VERIFIED','wall_s':time.monotonic()-started,'training_resume':'explicit rebuild/reset by caller; complete rollout/returns/update boundary'}
  summary.update(new_policy_env_intervals=intervals if intervals_known else None,new_physics_env_steps=intervals*10 if intervals_known else None,optimizer_updates=0,accounting='evaluation sampling separate from PPO rollout samples; physics/policy1ms/10ms')
  write(folder/'summary.json',summary);write(folder/'owner_result.json',{k:v for k,v in summary.items() if k!='commands'})
  return summary
 except BaseException as exc:
  write(folder/'worker_failure.json',{'status':'FAILED','exception':repr(exc),'update':update,'checkpoint_sha256':sha(checkpoint),'training_resume':'forbidden on failure'})
  raise
 finally:
  actor.train(mode);restore_rng(rng)


def training_window(rows):
 groups={}
 for row in rows:
  for item in row.get('cohort_statistics',{}).get('hold',{}).values():
   mean=item.get('mean');n=item.get('samples',0)
   if not mean or n<=0:continue
   key=(item['target_vx'],item['target_yaw']);g=groups.setdefault(key,{'samples':0,'sum':{}});g['samples']+=n
   for name,value in mean.items():
    if isinstance(value,(int,float)) and math.isfinite(value):g['sum'][name]=g['sum'].get(name,0.)+value*n
 return {str(k):{'command':list(k),'samples':v['samples'],'mean':{name:value/v['samples'] for name,value in v['sum'].items()}} for k,v in groups.items()}


def screen_training_boundary(env,screen,rebuild):
 """Called after PPO update only. Failed screening never resumes stale obs."""
 from .resume import rng_state,restore_rng
 rng=rng_state()
 try:
  env.close();result=screen()
 finally:restore_rng(rng)
 fresh,obs=rebuild()
 result['training_resume']={'explicit_reset':True,'reason':'warm screening at completed update boundary','next_observations':'fresh environment reset; physical state not restored','optimizer_updates_during_screen':0,'RNG_restored_before_rebuild':True}
 return fresh,obs,result


def report(run,window=100):
 """One read of existing artifacts. No simulator, watcher, checkpoint load or GPU."""
 run=Path(run);metrics=run/'metrics.jsonl';tail=deque(maxlen=window*2);partial=False
 with metrics.open() as stream:
  for line in stream:
   try:row=json.loads(line)
   except json.JSONDecodeError:partial=True;continue
   tail.append(row)
 if not tail:raise ValueError('no complete training updates')
 rows=list(tail);latest=rows[-1];stage=latest.get('course',{}).get('stage')
 # Never rank different curriculum revisions as the same condition distribution.
 rows=[r for r in rows if r.get('course',{}).get('stage')==stage]
 current=training_window(rows[-window:]);previous=training_window(rows[-2*window:-window]);trends={}
 for key,value in current.items():
  old=previous.get(key)
  if old:trends[key]={name:value['mean'][name]-old['mean'][name] for name in ['vx_mae_mps','yaw_mae_rad_s','pitch_abs_rad','roll_abs_rad','controller_effort_limit_hit_any','both_wheel_contact'] if name in value['mean'] and name in old['mean']}
 quick=[]
 for p in run.glob('quick_update_*/summary.json'):
  try:d=json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):continue
  if d.get('status')=='COMPLETED':quick.append((d['update'],p,d))
 quick.sort(key=lambda x:x[0]);q=None
 if quick:
  _,p,d=quick[-1];q={'path':str(p),'update':d['update'],'checkpoint_sha256':d.get('checkpoint_sha256'),'selection_eligible':False,'commands':d['commands']}
 course=run/'course_progress.json';c=json.loads(course.read_text()) if course.exists() else None
 return {'version':'motion31_evaluation_review_v1','run':str(run.resolve()),'latest_update':latest['update'],'course':latest.get('course'),'training_window_updates':[rows[-min(window,len(rows))]['update'],latest['update']],'previous_window_updates':len(rows[-2*window:-window]),'training_scope':'stochastic on-policy hold samples, sample-weighted; not deterministic certification','training_hold':list(current.values()),'hold_mean_delta_current_minus_previous':trends,'latest_deterministic_screen':q,'latest_complete_course_evidence':c,'partial_jsonl_rows_ignored':partial,'stop_recommendation':'No automatic stop from training reward or single screening; inspect same-stage trend plus complete course evidence','new_simulation':0,'optimizer_steps':0}


def markdown(data):
 lines=['# Motion训练与评估成绩单','',f"最新更新：{data['latest_update']}；最近训练窗口：{data['training_window_updates']}。",'', '训练表来自带探索噪声的保持段，按有效样本加权；不是确定性能力证书。不同课程版本不直接比较。','', '| vx/yaw命令 | 有效样本 | 实际vx | vx MAE | yaw MAE | pitch/roll平均绝对值 | 双轮接触 | 触限 |','|---|---:|---:|---:|---:|---|---:|---:|']
 def f(m,k):
  v=m.get(k);return 'UNKNOWN' if v is None else f'{v:.4f}'
 for row in data['training_hold']:
  m=row['mean'];lines.append(f"|{row['command']}|{row['samples']}|{f(m,'vx_mps')}|{f(m,'vx_mae_mps')}|{f(m,'yaw_mae_rad_s')}|{f(m,'pitch_abs_rad')} / {f(m,'roll_abs_rad')}|{f(m,'both_wheel_contact')}|{f(m,'controller_effort_limit_hit_any')}|")
 lines+=['','## 最近确定性快评','']
 q=data['latest_deterministic_screen']
 if q:
  lines+=[f"更新{q['update']}；checkpoint SHA：{q['checkpoint_sha256']}。快评不用于晋级。",'', '| 命令 | 通过/评分环境 | 首个失败项 |','|---|---:|---|']
  for row in q['commands']:
   failures=sorted({x for e in row.get('per_env',[]) for x in e.get('failed_checks',[])})
   lines.append(f"|{row['command']}|{row['passed']}/{row['scored_envs']}|{', '.join(failures) or '未列失败项，详见原始结果'}|")
 else:lines+=['UNKNOWN：没有完整快评结果。']
 c=data['latest_complete_course_evidence'];lines+=['','## 晋级证据','']
 if c:
  decision=c.get('decision',{});lines+=[f"最近完整课程检查更新：{c.get('update','UNKNOWN')}；本次晋级：{decision.get('advanced','UNKNOWN')}；全部阶段通过：{decision.get('all_stages_passed','UNKNOWN')}。",'', '|课程分支|本次通过|已知失败|缺证据|','|---|---|---|---|']
  for lane,value in decision.get('lanes',{}).items():lines.append(f"|{lane}|{value.get('passed','UNKNOWN')}|{value.get('known_failure','UNKNOWN')}|{value.get('missing_evidence','UNKNOWN')}|")
 else:lines+=['UNKNOWN：尚未找到course_progress.json。']
 lines+=['','训练趋势只作观察；完整评估仍决定晋级。旧快评不能解释成当前未评checkpoint已通过。']
 lines+=['','## 最近训练趋势（当前窗口减前窗口）','', '误差/姿态/触限的负值表示下降；接触率正值表示上升。缺前窗口不推断趋势。','', '|命令|vx MAE变化|yaw MAE变化|pitch变化|roll变化|接触率变化|触限变化|','|---|---:|---:|---:|---:|---:|---:|']
 for key,row in data['hold_mean_delta_current_minus_previous'].items():
  lines.append('|'+key+'|'+'|'.join(f(row,k) for k in ['vx_mae_mps','yaw_mae_rad_s','pitch_abs_rad','roll_abs_rad','both_wheel_contact','controller_effort_limit_hit_any'])+'|')
 lines+=['','## 双腿对称（训练保持段均值）','', '|命令|髋镜像误差rad|主动连杆镜像误差rad|左右腿长差m|轮心前后差m|','|---|---:|---:|---:|---:|']
 for row in data['training_hold']:lines.append('|'+str(row['command'])+'|'+'|'.join(f(row['mean'],k) for k in ['hip_mirror_abs_rad','active_mirror_abs_rad','leg_length_difference_m','wheel_offset_abs_m'])+'|')
 return '\n'.join(lines)+'\n'


def write_review(run):
 """Finite synchronous summary at update boundaries, never a background watcher."""
 from native_stand.command_v6.completion import write
 run=Path(run);data=report(run)
 write(run/'evaluation_review.json',data)
 from native_stand.command_v6.tools.artifact_io import atomic
 atomic(run/'evaluation_review.md',lambda stream:stream.write(markdown(data).encode()))
 return {'latest_update':data['latest_update'],'path':'evaluation_review.md','simulation':0}
