"""One explicitly authorized, SHA-bound successor arm; not a retry/continuation loop."""
import json,os,subprocess,sys,time
from pathlib import Path
from .completion import read,sha,write

def validate(first_path,next_path,next_output,session):
 from native_stand.students.motion_yaw_width_requests import verify
 a,b=read(first_path),read(next_path);verify(a);verify(b)
 if a['reward']['yaw_denominator']!=1. or b['reward']['yaw_denominator']!=.25:raise ValueError('queue is A1.0 then B.25 only')
 from native_stand.command_v6.tools.experiment_preparation import differences
 diff=[r['field'] for r in differences(a,b)]
 allowed=('experiment','artifacts.run_name','reward.version','reward.yaw_denominator','student_training.reward.version','student_training.reward.yaw_denominator','motion_training.spec_path')
 if set(diff)-set(allowed):raise ValueError('A/B has additional variable: '+str(set(diff)-set(allowed)))
 for p in (Path(first_path),Path(next_path)):
  s=read(p);book=read(p.with_name('budget.json'));receipt=read(p.with_name('authorization_receipt.json'))
  if s['authorization']['hapi_session']!=session or book['authorization']!='AUTHORIZED' or receipt['spec_sha256']!=sha(p):raise ValueError('queue needs exact authorized specs/current session')
 if a['authorization']['approval_id']!=b['authorization']['approval_id']:raise ValueError('queue separate approval scope')
 if sum(read(Path(p).with_name('budget.json'))['limits'][k] for p in (first_path,next_path) for k in ('smoke','formal'))!=2010:raise ValueError('queue inclusive2010 budget differs')
 if Path(next_output).exists() or Path(next_path).with_name('chain_submission.json').exists():raise ValueError('next arm already submitted')
 return {'version':'authorized_yaw_width_queue_v1','spec':str(Path(next_path).resolve()),'spec_sha256':sha(next_path),'output_dir':str(Path(next_output).resolve()),'session':session,'wall_limit_s':64800,'retries':0}

def launch_successor(chain,status,failure):
 manifest=read(chain/'manifest.json');q=manifest.get('authorized_next_arm')
 if not q:return
 event=chain/'next_arm_submission.json'
 # Exactly one invocation, after current owner's final reconciliation, no polling.
 payload={'status':'NOT_SUBMITTED','parent_status':status,'parent_failure':failure,'request':q}
 with event.open('x') as stream:json.dump(payload,stream)
 if failure or status not in ('COMPLETED','STOPPED_BOUNDED'):
  payload['reason']='runtime/infrastructure failure; no automatic retry or arm launch';write(event,payload);return
 next_path=Path(q['spec'])
 if sha(next_path)!=q['spec_sha256']:raise ValueError('queued spec changed')
 current=read(manifest['spec_path']);verify=validate(Path(manifest['spec_path']),next_path,Path(q['output_dir']),q['session'])
 if verify!=q:raise ValueError('queued authorization changed')
 deadline=manifest['submitted_unix_s']+q['wall_limit_s']
 if deadline-time.time()<600:
  payload['reason']='all-arm wall budget exhausted';write(event,payload);return
 command=[sys.executable,'-m','native_stand.command_v6.supervisor','--spec',str(next_path),'--output-dir',q['output_dir'],
  '--physical-gpu',str(current['device']['physical_index']),'--hapi-session',q['session'],'--resource-mode','shared',
  '--sharing-authorization',current['resources']['authorization_note'],'--deadline-unix-s',str(deadline),'--comparison-parent',str(chain),'--launch']
 with (chain/'next_arm_submission.log').open('x') as stream:
  code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':current['execution_source_root']}).returncode
 payload.update(status='SUBMISSION_RETURNED' if code==0 else 'SUBMISSION_FAILED',exit_code=code,command=command)
 write(event,payload)

def finish_comparison(chain,status,failure):
 """Compact same-protocol arm/candidate/per-env comparison, references raw files."""
 manifest=read(chain/'manifest.json');parent=manifest.get('comparison_parent')
 if not parent:return
 from collections import Counter
 a=Path(parent['chain'])
 if sha(a/'chain_terminal_report.json')!=parent['terminal_sha256']:raise ValueError('A terminal identity changed')
 rows={};total=0
 for label,folder in [('A',a),('B',chain)]:
  spec=read(read(folder/'manifest.json')['spec_path']);book=read(Path(read(folder/'manifest.json')['spec_path']).with_name('budget.json'))
  total+=book['used']['smoke']+book['used']['formal'];candidates=[]
  for candidate in read(folder/'formal/candidate_registry.json',[]):
   if not candidate.get('complete_matrix'):continue
   commands=[]
   for r in candidate.get('evaluations',[]):
    envs=r.get('per_env',[])
    commands.append({'command':r['case']['command'],'passed':sum(e.get('accepted') is True for e in envs),'scored_envs':len(envs),
     'failed_gates':dict(Counter(g for e in envs for g in e.get('failed_checks',[]))),
     'per_env':[{'env':e.get('env'),'accepted':e.get('accepted'),'survival_passed':e.get('survival_passed'),'tracking':e.get('tracking'),'metrics':e.get('metrics'),'failed_checks':e.get('failed_checks')} for e in envs],
     'raw_path':str((folder/'formal'/r['path']).resolve()) if not Path(r['path']).is_absolute() else r['path'],'raw_sha256':r['sha256']})
   candidates.append({'update':candidate['update'],'checkpoint':candidate['checkpoint'],'sha256':candidate['sha256'],'commands':commands})
  rows[label]={'run':str(folder),'yaw_denominator':spec['reward']['yaw_denominator'],'used':book['used'],'candidates':candidates,'selection_reference':str(folder/'formal/candidate_selection.json')}
 result={'version':'motion31_yaw_width_comparison_v1','status':status,'failure':failure,'arms':rows,'online_updates':total,
  'source_sha256':read(manifest['spec_path'])['source_sha256'],'comparison':'same protocol, compare matched update numbers; source is not a new trained candidate','causality_limit':'one training seed; time/resource contention may affect throughput, not a multiseed causal certification'}
 with (chain/'comparison_report.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
 lines=['# Yaw核宽单变量A/B终态比较',f'状态：{status}；累计online更新：{total}/2010。','来源原1600，A=1.0，B=0.25；只比较共同评估更新，缺证据不判通过。']
 for label,data in rows.items():
  lines.append(f"## {label}：{data['run']}，已用{data['used']}")
  for c in data['candidates']:
   summary=', '.join(f"{r['command'][:2]}={r['passed']}/{r['scored_envs']}({r['failed_gates']})" for r in c['commands'])
   lines.append(f"update{c['update']} SHA{c['sha256']}：{summary}")
 lines+=['逐环境速度、方向率、姿态、接触、饱和、停车及raw SHA见comparison_report.json；无新仿真。','单seed结果不证明唯一根因，也不授权继续训练。']
 with (chain/'comparison_report.md').open('x') as stream:stream.write('\n\n'.join(lines)+'\n')
