"""Bounded replay/refit/MuJoCo workflow; never authorizes native/PPO/deployment."""
import json,math,subprocess,time,traceback
from pathlib import Path
import torch
from . import motion_distillation_pipeline as p
from .data import sha
VERSION='motion31_distillation_cycle_v1'

def aggregate_replays(paths,output,teacher_sha):
 output=Path(output);parts=[];inputs={}
 if not paths or len(set(map(str,paths)))!=len(paths):raise ValueError('distinct replay paths required')
 for path in paths:
  path=Path(path).resolve();mp=path.with_name('replay_manifest.json');m=json.loads(mp.read_text())
  if m.get('version')!=p.VERSION or m['teacher_sha256']!=teacher_sha or sha(path)!=m['replay_sha256']:raise ValueError('replay Teacher/SHA differs')
  d=torch.load(path,map_location='cpu',weights_only=False);n=len(d['x'])
  if not n or d['x'].shape!=(n,31) or d['y'].shape!=(n,6) or any(d[k].shape!=(n,) for k in ('cohort','phase','split')):raise ValueError('replay shape differs')
  if not torch.isfinite(d['x']).all() or not torch.isfinite(d['y']).all() or d['split'].any() or not ((d['cohort']>=0)&(d['cohort']<7)).all() or not ((d['phase']>=0)&(d['phase']<5)).all():raise ValueError('invalid replay tensors; replay is train only')
  parts.append(d);inputs[str(path)]=sha(path);inputs[str(mp)]=sha(mp)
 output.mkdir(exist_ok=False,parents=True);torch.save({k:torch.cat([d[k] for d in parts]) for k in ('x','y','cohort','phase','split')},output/'replay.pt')
 p.write_json(output/'replay_manifest.json',{'version':p.VERSION,'teacher_sha256':teacher_sha,'replay_sha256':sha(output/'replay.pt'),'source_inputs_sha256':inputs,'samples':sum(len(d['x']) for d in parts),'simulation':0,'teacher_control':False,'scope':'aggregated Student states, train only; Teacher domain reliability unknown'})
 return output/'replay.pt'

def read_sim(run,checkpoint_sha):
 run=Path(run);terminal=json.loads((run/'terminal_report.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
 if terminal['status']!='COMPLETED' or not terminal['model_unchanged'] or manifest['checkpoint_sha256']!=checkpoint_sha:raise ValueError('simulation not settled or model differs')
 rows=terminal['results']
 if [r['target'] for r in rows]!=[c[:2] for c in p.COMMANDS]:raise ValueError('simulation command matrix differs')
 return manifest,rows

def compare(teacher_run,student_run,teacher_sha,student_sha,criteria):
 tm,teacher=read_sim(teacher_run,teacher_sha);sm,student=read_sim(student_run,student_sha)
 if tm['protocol']!=sm['protocol']:raise ValueError('Teacher/Student MuJoCo protocols differ')
 # Compare physical inputs, not historical code locations; record code separately.
 physical=lambda m:{k:v for k,v in m['input_sha256'].items() if Path(k).suffix in ('.xml','.stl','.usd')}
 if physical(tm)!=physical(sm):raise ValueError('Teacher/Student physical assets differ')
 for key in ('solver_profile','plant_profile','actuator_configuration'):
  if tm.get(key)!=sm.get(key):raise ValueError('Teacher/Student '+key+' differs; matching MuJoCo baseline required')
 rows=[]
 for t,s in zip(teacher,student):
  axis='yaw_mean' if t['target'][1] else 'vx_mean';scale=criteria['yaw_gap_rad_s'] if t['target'][1] else criteria['linear_gap_mps']
  if t['continuous_s']<30 or t.get('termination') is not None:raise ValueError('Teacher baseline incomplete; cannot certify fidelity')
  for key in (axis,'pitch_abs_max','roll_abs_max'):
   if type(t.get(key)) not in (int,float) or not math.isfinite(t[key]):raise ValueError('missing/nonfinite Teacher comparison metric')
  survived=s['continuous_s']>=30 and s.get('termination') is None
  for key in (axis,'pitch_abs_max','roll_abs_max'):
   value=s.get(key)
   if (survived or value is not None) and (type(value) not in (int,float) or not math.isfinite(value)):raise ValueError('missing/nonfinite comparison metric')
  gap=abs(s[axis]-t[axis]) if s.get(axis) is not None else None
  posture=max(0,s['pitch_abs_max']-t['pitch_abs_max'],s['roll_abs_max']-t['roll_abs_max']) if all(s.get(k) is not None for k in ('pitch_abs_max','roll_abs_max')) else None
  rows.append({'command':t['target'],'axis':axis,'teacher_response':t[axis],'student_response':s.get(axis),'response_abs_gap':gap,'normalized_gap':gap/scale if gap is not None else None,'posture_excess_rad':posture,'continuous_s':s['continuous_s'],'termination':s.get('termination'),'survived30':survived,'fidelity_passed':survived and gap is not None and posture is not None and gap<=scale and posture<=criteria['posture_excess_rad'],'student_task_passed':s['passed'],'student_full_swing_passed':s.get('supplemental_swing_constraint',{}).get('full_passed',False),'comparison_scope':'full_window' if survived else 'incomplete_window; missing metrics remain null'})
 return {'rows':rows,'score':sum(r['normalized_gap'] for r in rows)/7 if all(r['normalized_gap'] is not None for r in rows) else None,'fidelity_passed':all(r['fidelity_passed'] for r in rows),'all_survived':all(r['survived30'] for r in rows),'capability_certified':False,'native_confirmation':'REQUIRED; same frozen candidate, pure Student; no automatic native submission'}


def validate(request):
 required={'version','authorization','teacher','parent_spec','dataset','initial_student','initial_student_mujoco','teacher_mujoco','prior_replays','output','limits','criteria'}
 if set(request)!=required or request['version']!=VERSION or request['authorization'] not in ('UNAUTHORIZED','AUTHORIZED'):raise ValueError('cycle schema differs')
 lim=request['limits'];keys={'max_rounds','max_offline_updates','updates_per_round','fit_wall_s','sim_wall_s','total_wall_s','storage_max_bytes'}
 if set(lim)!=keys or any(type(v) is not int or v<=0 for v in lim.values()):raise ValueError('explicit positive finite integer limits required')
 if lim['max_rounds']>5 or lim['updates_per_round']>5000 or lim['max_offline_updates']>25000 or lim['fit_wall_s']>1200 or lim['sim_wall_s']>1800 or lim['total_wall_s']>14400 or lim['storage_max_bytes']>4*2**30:raise ValueError('cycle upper bounds exceeded')
 crit=request['criteria']
 if set(crit)!={'linear_gap_mps','yaw_gap_rad_s','posture_excess_rad','minimum_score_improvement','per_command_regression_margin'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in crit.values()) or crit['linear_gap_mps']==0 or crit['yaw_gap_rad_s']==0:raise ValueError('explicit fidelity criteria required')
 from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
 out=Path(request['output']);validate_new_experiment_name(out.name)
 if out.exists():raise FileExistsError(out)
 _,_,_=p.validate_teacher(request['teacher'],request['parent_spec']);saved=torch.load(request['initial_student'],map_location='cpu',weights_only=False);p.validate_student(saved,request['parent_spec'])
 if saved['distillation']['teacher_sha256']!=sha(request['teacher']):raise ValueError('cycle Student Teacher differs')
 dataset=Path(request['dataset']);m=json.loads(dataset.with_name('manifest.json').read_text())
 if sha(dataset)!=m['dataset_sha256'] or m['teacher_sha256']!=sha(request['teacher']) or m['parent_spec_sha256']!=sha(request['parent_spec']):raise ValueError('cycle dataset differs')
 if not isinstance(request['prior_replays'],list) or len(set(request['prior_replays']))!=len(request['prior_replays']):raise ValueError('prior replay paths must be a distinct list')
 for path in request['prior_replays']:
  path=Path(path);meta=json.loads(path.with_name('replay_manifest.json').read_text())
  if meta['teacher_sha256']!=sha(request['teacher']) or meta['replay_sha256']!=sha(path):raise ValueError('prior replay identity differs')
 initial=compare(request['teacher_mujoco'],request['initial_student_mujoco'],sha(request['teacher']),sha(request['initial_student']),crit)
 return {'status':'PLAN_ONLY','files_written':0,'simulation':0,'optimizer_updates':0,'initial_fidelity':initial,'limits':lim,'native_confirmation':'separate supervisor request; never implicit','output':str(out)}

def run(request,authorization):
 plan=validate(request)
 if request['authorization']!='AUTHORIZED' or not authorization or not authorization.strip():raise ValueError('current explicit authorization required for BC and MuJoCo')
 out=Path(request['output']);out.mkdir(parents=True,exist_ok=False);p.write_json(out/'request.json',request)
 inputs={str(Path(x).resolve()):sha(x) for x in [request['teacher'],request['parent_spec'],request['dataset'],request['initial_student'],*request['prior_replays']]}
 for run_path in (request['teacher_mujoco'],request['initial_student_mujoco']):
  for name in ('manifest.json','terminal_report.json'):
   f=Path(run_path)/name;inputs[str(f.resolve())]=sha(f)
 p.write_json(out/'input_manifest.json',inputs)
 started=time.monotonic();lim=request['limits'];used=0;history=[];replays=list(request['prior_replays']);best=request['initial_student'];sim=request['initial_student_mujoco'];best_result=plan['initial_fidelity'];status='BOUNDED_COMPLETE';failure=None
 def budget():
  if time.monotonic()-started>=lim['total_wall_s']:raise TimeoutError('cycle wall budget exhausted')
  if sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>=lim['storage_max_bytes']:raise TimeoutError('cycle storage budget exhausted')
 try:
  if best_result['fidelity_passed']:status='NATIVE_CONFIRMATION_REQUIRED'
  else:
   for round_id in range(1,lim['max_rounds']+1):
    budget();steps=min(lim['updates_per_round'],lim['max_offline_updates']-used)
    remaining=lim['total_wall_s']-(time.monotonic()-started)
    if steps<=0 or remaining<lim['fit_wall_s']+lim['sim_wall_s']+30:status='BUDGET_COMPLETE';break
    folder=out/f'round_{round_id:02}';folder.mkdir();p.collect_replay(sim,request['teacher'],request['parent_spec'],folder/'replay');replays.append(str(folder/'replay/replay.pt'))
    rp=aggregate_replays(replays,folder/'aggregate',sha(request['teacher']));budget()
    try:fit=p.fit(request['dataset'],folder/'fit',steps=steps,wall_s=lim['fit_wall_s'],initial_student=best,replay=rp)
    finally:
     bp=folder/'fit/budget.json'
     if bp.exists():used+=json.loads(bp.read_text())['used']
    budget();candidate=folder/'fit/best_validation.pt'
    from native_stand.command_v6.domain.experiment_naming import new_experiment_name
    simulation_output=folder/new_experiment_name(f'mujoco_round_{round_id:02}')
    command=p.mujoco_command(candidate,request['parent_spec'],simulation_output,min(lim['sim_wall_s'],int(lim['total_wall_s']-(time.monotonic()-started)-30)))
    with (folder/'mujoco_owner.log').open('x') as log:child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=lim['sim_wall_s']+30)
    if child.returncode:raise RuntimeError('MuJoCo child failed; no infrastructure retry')
    result=compare(request['teacher_mujoco'],simulation_output,sha(request['teacher']),sha(candidate),request['criteria']);budget()
    # Keep prior candidate unless ALL command responses avoid material regression.
    nonregression=all(b['normalized_gap'] is None or (a['normalized_gap'] is not None and a['normalized_gap']<=b['normalized_gap']+request['criteria']['per_command_regression_margin']) for a,b in zip(result['rows'],best_result['rows']))
    posture_nonregression=all(b['posture_excess_rad'] is None or (a['posture_excess_rad'] is not None and a['posture_excess_rad']<=b['posture_excess_rad']+request['criteria']['posture_excess_rad']) for a,b in zip(result['rows'],best_result['rows']))
    improved_score=result['score'] is not None and (best_result['score'] is None or best_result['score']-result['score']>=request['criteria']['minimum_score_improvement'])
    # An early fall is behavior evidence, not missing infrastructure. Bootstrap on
    # per-command survival until full-window tracking becomes observable.
    bootstrap=(not best_result['all_survived'] and all(a['continuous_s']>=b['continuous_s'] for a,b in zip(result['rows'],best_result['rows'])) and any(a['continuous_s']>b['continuous_s'] for a,b in zip(result['rows'],best_result['rows'])))
    promote=(result['all_survived'] and nonregression and posture_nonregression and improved_score) or bootstrap
    history.append({'round':round_id,'offline_updates':fit['offline_optimizer_updates'],'candidate':str(candidate),'candidate_sha256':sha(candidate),'comparison':result,'replaces_previous_candidate':promote})
    p.write_json(folder/'comparison.json',history[-1])
    if promote:best=str(candidate);sim=str(simulation_output);best_result=result
    if best_result['fidelity_passed']:status='NATIVE_CONFIRMATION_REQUIRED';break
    if not promote:status='NO_IMPROVEMENT_RETAIN_PREVIOUS';break
 except BaseException as error:
  status='FAILED';failure={'exception':repr(error),'traceback':traceback.format_exc()}
 finally:
  summary={'status':status,'failure':failure,'offline_updates_used':used,'wall_used_s':time.monotonic()-started,'limits':lim,'history':history,'retained_candidate':str(best),'retained_candidate_sha256':sha(best),'comparison':best_result,'online_PPO':0,'native_certified':False,'deployment_certified':False,'authorization':authorization}
  p.write_json(out/'summary.json',summary);p.write_json(out/'budget.json',{'status':status,'offline_updates_used':used,'wall_used_s':summary['wall_used_s'],'limits':lim,'online_PPO':0})
 return summary
