"""Bounded MuJoCo two-policy factorial screen; frozen inputs, no PPO or Isaac."""
import argparse,itertools,json,time
from collections import deque
import copy
from pathlib import Path
import numpy as np
import torch
import mujoco
from native_stand.students.data import sha
from native_stand.students.model import load
from native_stand.students.mcu_interface import encode
from native_stand.students.interface import targets
from native_stand.students.real2sim_delay import ObservationHistory,TargetTransport,torque_response
from native_stand.students.real2sim_handover import decode_log,initialize_from_log
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from validate_native_mujoco import binding,state
from diagnose_motion_real2sim import sensors
from replay_motion31_shadow_csv import read_csv
LEGS=[0,1,3,4];WHEELS=[2,5]
LIMIT=np.array([40,40,5.5,40,40,5.5])
PROFILES=[('old',.01,60.,0),('new',.35,40.,10),('old_only_armature',.35,60.,0),
          ('old_only_kp',.01,40.,0),('old_only_sensor',.01,60.,10)]


def summarize(rows,failed,duration,initial):
 a={k:np.stack([r[k] for r in rows]) for k in rows[0]}
 contacts=a['contact']>2;legdelta=np.diff(a['target'][:,LEGS],axis=0);wdelta=np.diff(a['target'][:,WHEELS],axis=0)
 def rms(x):return float(np.sqrt(np.mean(x*x))) if x.size else None
 def longest(mask):
  best=run=0
  for v in mask:run=run+1 if v else 0;best=max(best,run)
  return best*.01
 def frequency(x):
  if len(x)<50:return None
  y=x-x.mean(0);power=(abs(np.fft.rfft(y,axis=0))**2).mean(1);freq=np.fft.rfftfreq(len(x),.01);mask=(freq>=2)&(freq<=45)
  return float(freq[mask][np.argmax(power[mask])]) if power[mask].max()>1e-12 else None
 # Scalar aggregates and per-channel values remain separate: cancellation must not hide vibration.
 return {'termination':failed or 'WINDOW_COMPLETED','fall_time_s':rows[-1]['time'] if failed=='UPRIGHT' else None,
  'simulated_seconds':rows[-1]['time'],'requested_seconds':duration,'initialization':initial,
  'both_contact_fraction':float(contacts.all(1).mean()),'single_lost_fraction':float((contacts.sum(1)==1).mean()),
  'both_lost_fraction':float((~contacts.any(1)).mean()),'longest_both_lost_s':longest(~contacts.any(1)),
  'saturation_any_fraction':float((abs(a['torque'])>=LIMIT-1e-4).any(1).mean()),
  'saturation_per_channel':(abs(a['torque'])>=LIMIT-1e-4).mean(0).tolist(),
  'pitch_abs_peak_rad':float(abs(a['state'][:,0]).max()),'roll_abs_peak_rad':float(abs(a['state'][:,1]).max()),
  'height_mean_m':float(a['state'][:,2].mean()),'vx_abs_mean_mps':float(abs(a['state'][:,3]).mean()),
  'leg_target_delta_rms_rad':rms(legdelta),'wheel_target_delta_rms_rad_s':rms(wdelta),
  'leg_target_peak_hz':frequency(a['target'][:,LEGS]),'wheel_target_peak_hz':frequency(a['target'][:,WHEELS]),
  'frequency_scope':'100Hz sampled target FFT2..45Hz, failed-window length differs; not motor or1kHz torque certification',
  'leg_dq_rms_rad_s':rms(a['dq'][:,LEGS]),'wheel_dq_rms_rad_s':rms(a['dq'][:,WHEELS])},a


def rollout(model,refs,actor,initial,history_rows,packet_seed,bridge,command_times,command_obs,event,profile,duration,deadline):
 name,arm,kp,sensor=profile;model.dof_armature[np.asarray(refs['dofs'])[LEGS]]=arm
 d=mujoco.MjData(model);d.qpos[:]=initial.qpos;d.qvel[:]=initial.qvel;mujoco.mj_forward(model,d)
 bounds=torch.tensor(model.jnt_range[refs['joints']],dtype=torch.float32)[None]
 for i,j in enumerate(refs['joints']):
  if not model.jnt_limited[j]:bounds[0,i]=torch.tensor([-float('inf'),float('inf')])
 adopted=bridge.copy();requested=bridge.copy();actual=np.zeros(6);demand=actual.copy();preclip=actual.copy()
 transport=TargetTransport([10,10,7,10,10,11],adopted)
 h=ObservationHistory(history_rows[0][1],initial_tick=history_rows[0][0])
 for stamp,row in history_rows:h.append(stamp,row)
 packet=packet_seed.copy();rows=[];failure=None;ref=torch.zeros(1,6)
 with torch.inference_mode():
  for tick in range(round(duration*1000)):
   if tick%100==0 and time.monotonic()>deadline:raise TimeoutError('bounded comparison wall exceeded')
   current=sensors(model,d,refs);h.append(tick,current)
   if tick%10==0:
    sensed=current if sensor==0 else packet;packet=current.copy()
    i=max(0,int(np.searchsorted(command_times,event+tick*.001,side='right')-1));cmd=torch.tensor(command_obs[i,22:25]*[4.,4.,.5],dtype=torch.float32)[None]
    obs=encode(torch.tensor(sensed[:3],dtype=torch.float32)[None],torch.tensor(sensed[3:6],dtype=torch.float32)[None],
     torch.tensor(sensed[6:12],dtype=torch.float32)[None],torch.tensor(sensed[12:18],dtype=torch.float32)[None],
     torch.tensor(adopted[LEGS],dtype=torch.float32)[None],torch.tensor(adopted[WHEELS],dtype=torch.float32)[None],ref,'Motion',cmd)
    out=targets(actor.mean(obs),ref,bounds);requested[LEGS]=out['leg_executed'][0].numpy();requested[WHEELS]=out['wheel_executed'][0].numpy();transport.submit(tick,requested)
   delivered=transport.delivered(tick)
   if tick%5==0:
    adopted=delivered;f=h.at(tick-10)
    demand[LEGS]=kp*(adopted[LEGS]-f[6:12][LEGS])-2*f[12:18][LEGS];demand[WHEELS]=adopted[WHEELS]-f[12:18][WHEELS];preclip=demand.copy();demand=np.clip(demand,-LIMIT,LIMIT)
   actual=torque_response(actual,demand,[5,5,0,5,5,0]);d.ctrl[refs['actuators']]=actual/refs['gears'];mujoco.mj_step(model,d)
   if tick%10==9:
    _,_,s=state(model,d,refs)
    rows.append({'time':(tick+1)*.001,'target':requested.copy(),'torque':actual.copy(),'dq':d.qvel[refs['dofs']].copy(),
     'adopted':adopted.copy(),'torque_preclip':preclip.copy(),'q':sensors(model,d,refs)[6:12].copy(),'gyro':sensors(model,d,refs)[:3].copy(),
     'contact':np.array(s['contact_force_n']),'state':np.array([s['pitch'],s['roll'],s['height'],s['vx'],s['up'],s['closure_m']])})
    if not np.isfinite(d.qpos).all():failure='NONFINITE'
    elif s['closure_m']>.005:failure='CLOSURE'
    elif s['up']<.5:failure='UPRIGHT'
    if failure:break
 return rows,failure


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--armature-011-five',action='store_true',help='five actors,armature.11, old/new PD-sensor settings')
 p.add_argument('--wheel-match-compare',action='store_true',help='source/candidate low-inertia pairing plus deployed ONNX Kv1/.2 comparison; no PPO')
 p.add_argument('--motion-command-screen',action='store_true',help='one frozen candidate, fixed leg.012/Kp40 and candidate wheel profile, scheduled Motion commands')
 p.add_argument('--command-target',type=float,nargs=2,action='append',metavar=('VX','YAW'),help='motion-command-screen only; repeat for pure straight or in-place targets, default nine conditions')
 p.add_argument('--deployed-reference-only',action='store_true',help='with wheel-match-compare: four ONNX cases at original Kp60/leg-armature.01/Actor0ms; only Kv varies')
 p.add_argument('--source-spec',type=Path);p.add_argument('--source-checkpoint',type=Path)
 p.add_argument('--deployed-onnx',type=Path,help='exact deployed actor.onnx with adjacent manifest and golden_vectors.npz')
 p.add_argument('--actuator-audit',action='store_true',help='bounded wheel dynamics, handover ablations and 1kHz tail diagnostics; CPU only')
 p.add_argument('--audit-confirm',action='store_true',help='with actuator-audit: nine paired log-state baseline/wheel-J/leg-armature cases')
 p.add_argument('--actuator-rig-only',action='store_true',help='with actuator-audit: isolate wheel total J x feedback delay, without Actor or robot rollout')
 p.add_argument('--rig-wheel-kv',type=float,nargs='+',default=[1.],help='rig-only: explicit wheel feedback gains, at most6 values in .05..1')
 p.add_argument('--rig-mechanical-b',type=float,default=.01,help='rig-only: mechanical viscous coefficient0..02 Nm s/rad')
 p.add_argument('--candidate-spec',type=Path);p.add_argument('--candidate-checkpoint',type=Path);p.add_argument('--armature-values',type=float,nargs='+',help='scan refinement within .01..11, at most12 values');p.add_argument('--armature-scan',action='store_true',help='old3250 teacher/student, six armatures, fixed Kp60 and Actor0ms');p.add_argument('--output-dir',type=Path);p.add_argument('--run',action='store_true');p.add_argument('--seconds',type=float,default=10.);p.add_argument('--wall-seconds',type=float,default=300.)
 a=p.parse_args()
 if a.motion_command_screen:
  if any((a.wheel_match_compare,a.deployed_reference_only,a.actuator_audit,a.audit_confirm,a.actuator_rig_only,a.armature_scan,a.armature_011_five,a.armature_values,a.source_spec,a.source_checkpoint,a.deployed_onnx)) or a.rig_wheel_kv!=[1.] or a.rig_mechanical_b!=.01:raise ValueError('Motion command screen is a separate frozen-candidate diagnostic')
  if not a.candidate_spec or not a.candidate_checkpoint:raise ValueError('explicit candidate spec/checkpoint required')
  return motion_command_screen(a)
 if a.command_target:raise ValueError('command-target requires motion-command-screen')
 if a.wheel_match_compare:
  if a.actuator_audit or a.audit_confirm or a.actuator_rig_only or a.armature_scan or a.armature_011_five or a.armature_values or a.rig_wheel_kv!=[1.] or a.rig_mechanical_b!=.01:raise ValueError('wheel match comparison is a separate bounded matrix')
  if not all((a.source_spec,a.source_checkpoint,a.candidate_spec,a.candidate_checkpoint,a.deployed_onnx)):raise ValueError('explicit source/candidate spec/checkpoint and deployed ONNX required')
  return wheel_match_compare(a)
 if a.source_spec or a.source_checkpoint or a.deployed_onnx or a.deployed_reference_only:raise ValueError('source/deployed flags require wheel-match-compare')
 if a.audit_confirm and not a.actuator_audit:raise ValueError('audit-confirm requires actuator-audit')
 if a.actuator_rig_only and (not a.actuator_audit or a.audit_confirm):raise ValueError('rig-only requires actuator-audit and excludes audit-confirm')
 if (a.rig_wheel_kv!=[1.] or a.rig_mechanical_b!=.01) and not a.actuator_rig_only:raise ValueError('rig gains only apply to rig-only')
 if not 1<=len(a.rig_wheel_kv)<=6 or len(set(a.rig_wheel_kv))!=len(a.rig_wheel_kv) or any(not .05<=k<=1. for k in a.rig_wheel_kv) or not 0<=a.rig_mechanical_b<=.02:raise ValueError('rig gain/drag scan outside bounds')
 if a.actuator_audit:
  if a.candidate_spec or a.candidate_checkpoint or a.armature_scan or a.armature_011_five or a.armature_values:raise ValueError('actuator audit has its own paired model matrix')
  return actuator_audit(a)
 if bool(a.candidate_spec)!=bool(a.candidate_checkpoint):raise ValueError('candidate spec/checkpoint required together')
 if a.candidate_spec and (a.armature_scan or a.armature_011_five):raise ValueError('candidate mode is separate')
 if a.armature_scan and a.armature_011_five:raise ValueError('choose one comparison mode')
 if a.armature_values is not None and (not (a.armature_scan or a.candidate_spec) or not 1<=len(a.armature_values)<=12 or len(set(a.armature_values))!=len(a.armature_values) or any(not .01<=v<=(.35 if a.candidate_spec else .11) for v in a.armature_values)):raise ValueError('unique armature values within .01..11 require scan mode')
 if not 2.5<=a.seconds<=30 or not 30<=a.wall_seconds<=600:raise ValueError('bounded duration/wall invalid')
 root=Path(__file__).resolve().parents[1];search=root/'sim2sim/2026_10_21-09_35__motion3250_handover_search'
 m=json.loads((search/'manifest.json').read_text());case=json.loads((search/'case_039.json').read_text());csv=next(Path(k) for k in m['input_sha256'] if k.endswith('.csv'))
 if sha(csv)!=m['input_sha256'][str(csv)]:raise ValueError('log identity differs')
 xml=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml');model=mujoco.MjModel.from_xml_path(str(xml));refs=binding(model)
 _,x,_,t,_=read_csv(csv);event=case['event_sample_time_s'];ix=int(np.argmin(abs(t-event)))
 initial,fit=initialize_from_log(model,refs,x[ix],case['assumed_root_vx_mps']);hist=[]
 for i in range(ix-15,ix):
  v=decode_log(x[i]);hist.append((int(round((t[i]-event)*1000)),np.r_[v['gyro'],v['gravity'],v['q'],v['dq']]))
 v=decode_log(x[ix]);bridge=v['q'].copy();bridge[WHEELS]=v['dq'][WHEELS]
 standard=mujoco.MjData(model);mujoco.mj_forward(model,standard);ss=sensors(model,standard,refs)
 inits={'log':(initial,hist,hist[-1][1],bridge),'normal':(standard,[(-10,ss)],ss,np.zeros(6))}
 policies=[('teacher3250',root/'runs/2026_10_18-23_10__motion1000_differential85_5000_retry/formal/model_update_3250.pt',root/'configs/experiments/2026_10_18-23_10__motion1000_differential85_5000_retry/spec.json'),
 ('new250',root/'runs/2026_10_21-18_58__motion526_sensor_schema_formal9183/formal/model_update_250.pt',root/'configs/experiments/2026_10_21-18_58__motion526_sensor_schema_formal9183/spec.json')]
 profiles=PROFILES
 if a.armature_011_five:
  profiles=[('arm011_kp60_sensor0',.11,60.,0),('arm011_kp40_sensor10',.11,40.,10)]
  policies += [('before526',root/'runs/2026_10_21-16_04__motion_fixed_timing_armature035_kp40/formal/model_update_526.pt',root/'configs/experiments/2026_10_21-16_04__motion_fixed_timing_armature035_kp40/spec.json'),
   ('original4750',root/'runs/2026_10_19-18_38__motion3250_parallel4096_5000/formal/model_update_4750.pt',root/'configs/experiments/2026_10_19-18_38__motion3250_parallel4096_5000/spec.json'),
   ('small3250',root/'isolated/2026_10_19-19_55__motion3250_small31_dagger_round2/round_01/fit/best_validation.pt',root/'configs/experiments/2026_10_18-23_10__motion1000_differential85_5000_retry/spec.json')]
 if a.armature_scan:
  policies=[policies[0],('small3250',root/'isolated/2026_10_19-19_55__motion3250_small31_dagger_round2/round_01/fit/best_validation.pt',root/'configs/experiments/2026_10_18-23_10__motion1000_differential85_5000_retry/spec.json')]
  profiles=[(f'arm{arm:g}_kp60_sensor0',arm,60.,0) for arm in (a.armature_values or (.01,.02,.04,.06,.08,.11))]
 if a.candidate_spec:
  policies=[('new_final',a.candidate_checkpoint.resolve(),a.candidate_spec.resolve())]
  profiles=[(f'arm{arm:g}_kp40_sensor10',arm,40.,10) for arm in (a.armature_values or (.012,.02,.04,.11,.35))]
 count=len(policies)*len(inits)*len(profiles)
 inputs={str(q):sha(q) for q in [xml,csv,search/'manifest.json',search/'case_039.json',Path(__file__)]}
 actors={};torch.set_num_threads(1)
 for name,cp,sp in policies:
  actor,data=load(cp,'MotionZeroStudent');s=json.loads(sp.read_text())
  if data.get('distillation'):
   from native_stand.students.motion_distillation_pipeline import validate_student
   validate_student(data,sp)
  elif data['spec_sha256']!=sha(sp) or data['asset_sha256']!=s['asset_sha256']:raise ValueError('exact model/spec identity differs')
  actors[name]=actor.eval();inputs.update({str(cp):sha(cp),str(sp):sha(sp)})
 out=a.output_dir or root/'sim2sim'/new_experiment_name('motion3250_armature_scan' if a.armature_scan else 'motion_timing_factorial20')
 manifest={'version':'motion_timing_factorial_screen_v1','input_sha256':inputs,'profiles':profiles,'conditions':count,
  'seconds_per_condition':a.seconds,'wall_limit_s':a.wall_seconds,'storage_max_bytes':32*2**20,'PPO':0,'fixed_command_source':'same logged command sequence for both initial states',
  'log_initial_fit':fit,'contact_scope':'MuJoCo normal wheel-ground forces, sampled100Hz, threshold2N; not Isaac net force parity',
  'old_configuration':'target transport[10,10,7,10,10,11], PDfeedback10, legtau5; armature/kp/Actor delay by profile',
  'interpretation':'deterministic single-state screen, censored at first failure or window; no training effect inference across different source lineages'}
 print(json.dumps({'output':str(out),'conditions':count,'simulation':int(a.run),'PPO':0,'profiles':profiles}),flush=True)
 if not a.run:return
 out.mkdir(parents=True,exist_ok=False)
 def write(name,d):
  with (out/name).open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
 write('manifest.json',manifest);started=time.monotonic();results=[];status='FAILED'
 try:
  for (policy,actor),(init_name,(d,h,seed,b)),profile in itertools.product(actors.items(),inits.items(),profiles):
   rows,failure=rollout(model,refs,actor,d,h,seed,b,t,x,event,profile,a.seconds,started+a.wall_seconds)
   summary,arrays=summarize(rows,failure,a.seconds,init_name);key=f'{policy}__{init_name}__{profile[0]}'
   summary.update(policy=policy,profile=profile[0],trace=key+'.npz');np.savez_compressed(out/(key+'.npz'),**arrays);write(key+'.json',summary);results.append(summary)
   print(key,summary['termination'],summary['simulated_seconds'],flush=True)
   if sum(f.stat().st_size for f in out.iterdir())>31*2**20:raise RuntimeError('comparison storage cap')
  status='COMPLETED'
 finally:
  write('summary.json',{'status':status,'results':results,'wall_seconds':time.monotonic()-started,'PPO_updates':0,'independent_seeds':1})
  lines=['# MuJoCo同延迟配置对照','', '|模型|初态|配置|终态/时间s|双轮接触|单轮失触|双轮失触|触限|腿目标Δ RMS rad|轮目标Δ RMS rad/s|','|---|---|---|---|---:|---:|---:|---:|---:|---:|']
  for r in results:lines.append(f"|{r['policy']}|{r['initialization']}|{r['profile']}|{r['termination']} / {r['simulated_seconds']:.2f}|{r['both_contact_fraction']:.1%}|{r['single_lost_fraction']:.1%}|{r['both_lost_fraction']:.1%}|{r['saturation_any_fraction']:.1%}|{r['leg_target_delta_rms_rad']:.4f}|{r['wheel_target_delta_rms_rad_s']:.3f}|")
  lines+=['','各指标只统计失败前有效窗口；不同窗口不能直接比较振荡/触限均值。log工况包含未测root速度/高度/previous桥接假设；不是实机辨识，也非多seed能力认证。完整配置、输入SHA与逐条件trace见manifest/JSON/NPZ。']
  (out/'REPORT.md').write_text('\n'.join(lines)+'\n')

def actuator_profiles():
 """Diagnostic hypotheses only. Every row starts from the same immutable base."""
 base={'name':'baseline','leg_armature':.35,'kp':40.,'kd':2.,'kv':1.,'wheel_peak':5.5,
       'wheel_J':None,'wheel_B':None,'wheel_C':None,'friction_epsilon':.1,
       'transport_ms':[10,10,7,10,10,11],'feedback_ms':10,'tau_ms':[5,5,0,5,5,0],
       'actor_ms':10,'actor_dq_ms':None,'cache':'adopted','root_vx':-.3,'event':'main',
       'command':'zero','scope':'one changed mechanism from baseline'}
 rows=[]
 def add(name,**changes):rows.append({**copy.deepcopy(base),'name':name,**changes})
 add('baseline')
 add('peak489',wheel_peak=4.893896)
 add('peak2447',wheel_peak=2.446948)
 # Model wheel order ch2/ch5 = geometric right/left. J is TOTAL fixed-parent J.
 add('wheel_J_reported',wheel_J=[.004204,.0042075])
 add('wheel_friction_reported',wheel_B=[.00066957,.0025485],wheel_C=[.0528843,.054699],
     scope='replace wheel B/C together as reported drag profile; not independent B-vs-C attribution')
 return base,rows


def actuator_cases(confirm=False):
 base,profiles=actuator_profiles();rows=[]
 if confirm:
  selected=[profiles[0],next(p for p in profiles if p['name']=='wheel_J_reported'),
            {**copy.deepcopy(base),'name':'leg_arm012_only','leg_armature':.012}]
  return [{'policy':policy,'initial':'log','configuration':copy.deepcopy(cfg),'group':'paired_followup'}
          for policy,cfg in itertools.product(('teacher3250','small3250','final500'),selected)]
 for policy,initial,cfg in itertools.product(('teacher3250','small3250','final500'),('normal','log'),profiles):
  rows.append({'policy':policy,'initial':initial,'configuration':copy.deepcopy(cfg),'group':'wheel_physics'})
 variants=[('previous_logged_seed',{'cache':'logged_seed'}),('previous_zero_seed',{'cache':'zero_seed'}),
  ('previous_requested_recursive',{'cache':'requested_recursive'}),('previous_measured_wheels',{'cache':'measured_wheels'}),
  ('PDfeedback0',{'feedback_ms':0}),('PDfeedback20',{'feedback_ms':20}),
  ('Actor0',{'actor_ms':0}),('Actor20',{'actor_ms':20}),('Actor_dq20',{'actor_dq_ms':20}),
  ('leg_transport0',{'transport_ms':[0,0,7,0,0,11]}),
  ('wheel_transport0',{'transport_ms':[10,10,0,10,10,0]}),('leg_response0',{'tau_ms':[0]*6}),
  ('root_vx0',{'root_vx':0.}),('earlier_log_state',{'event':'earlier'}),
  ('logged_command',{'command':'logged_hold_last'})]
 for policy,(name,changes) in itertools.product(('teacher3250','small3250','final500'),variants):
  rows.append({'policy':policy,'initial':'log','configuration':{**copy.deepcopy(base),'name':name,**changes},'group':'handover'})
 for policy,initial in itertools.product(('teacher3250','small3250','final500'),('normal','log')):
  cfg={**copy.deepcopy(base),'name':'historical_stress','leg_armature':.012,'kp':60.,'actor_ms':0,
       'scope':'historical combined stress; armature, Kp and Actor delay change together, not a single-factor result'}
  rows.append({'policy':policy,'initial':initial,'configuration':cfg,'group':'historical_control'})
 return rows


def wheel_match_cases(reference_only=False):
 """Freeze paired hypotheses; deployed pairs differ ONLY in Kv, not the plant."""
 base,_=actuator_profiles()
 if reference_only:
  return [{'policy':'deployed_onnx','initial':initial,'configuration':{**copy.deepcopy(base),
   'name':f'deployment_reference_Kv{kv:g}','leg_armature':.01,'kp':60.,'actor_ms':0,'kv':kv,
   'scope':'original nominal leg settings and previous timing hypotheses; only Kv differs within pair'},
   'group':'deployed_Kv_only','pair_id':f'deployment_reference_{initial}','initial_scenario':initial}
   for initial,kv in itertools.product(('normal','log'),(1.,.2))]
 matched={**base,'kv':.2,'wheel_J':[.004204,.0042075],'wheel_B':[0.,0.],
  'scope':'new candidate wheel J/Kv, native zero wheel drive drag; MJCF contacts/leg drag unchanged'}
 scenarios=[('normal','normal',{}),('log_bridge','log',{}),
  ('log_logged_seed','log',{'cache':'logged_seed'}),('log_feedback20','log',{'feedback_ms':20})]
 rows=[]
 for arm,(name,initial,changes),policy in itertools.product((.35,.02,.012),scenarios,('source387','candidate')):
  cfg={**copy.deepcopy(matched),'leg_armature':arm,**changes,'name':f'arm{arm:g}_{name}'}
  rows.append({'policy':policy,'initial':initial,'configuration':cfg,'group':'source_candidate',
   'pair_id':cfg['name'],'initial_scenario':name})
 profiles=[('prior_audit',base),
  ('historical_stress',{**base,'leg_armature':.012,'kp':60.,'actor_ms':0,
   'scope':'fixed historical fall-prone combined plant; only Kv varies within each paired condition'}),
  ('new_candidate_plant',matched)]
 for (profile,cfg),initial,kv in itertools.product(profiles,('normal','log'),(1.,.2)):
  c={**copy.deepcopy(cfg),'name':f'{profile}_Kv{kv:g}','kv':kv}
  rows.append({'policy':'deployed_onnx','initial':initial,'configuration':c,'group':'deployed_Kv_only',
   'pair_id':f'{profile}_{initial}','initial_scenario':initial})
 return rows


class DeployedONNX:
 """Execute the actual exported graph on CPU, not a substitute PyTorch model."""
 def __init__(self,path,expected_manifest):
  from onnx.reference import ReferenceEvaluator
  if sha(path)!=expected_manifest['files_sha256']['actor.onnx']:raise ValueError('deployed ONNX SHA differs')
  self.engine=ReferenceEvaluator(str(path));self.input=expected_manifest['input']['name']
  self.output=expected_manifest['output']['name']
 def mean(self,obs):
  x=obs.detach().cpu().numpy()
  if x.ndim!=2 or x.shape[1]!=31 or x.dtype!=np.float32:raise ValueError('deployed input must be float32[N,31]')
  result=self.engine.run([self.output],{self.input:x})[0]
  if result.shape!=(len(x),6) or not np.isfinite(result).all():raise ValueError('deployed output shape/finite differs')
  return torch.from_numpy(result.copy())


def wheel_match_diagnostics(summary,low):
 """Same declared posture targets; preserve censoring, whole/hold windows apart."""
 t=low.get('policy_time',low['time']);hold=(t>=10)&(t<18);complete=summary['termination']=='WINDOW_COMPLETED' and summary['simulated_seconds']>=30.-1e-8
 result={'full_window_complete':complete,'holding_window_s':[10.,18.],
  'hold_complete':bool(hold.sum()==800),'diagnostic_only':True}
 if result['hold_complete']:
  a=low['state'][hold];g=low['geometry'][hold];target=low['target'][hold]
  result.update(pitch_hold_peak_rad=float(abs(a[:,0]).max()),roll_hold_peak_rad=float(abs(a[:,1]).max()),
   height_hold_mean_m=float(a[:,2].mean()),wheel_offset_hold_MAE_m=float(abs(g[:,0]).mean()),
   leg_length_hold_MAE_m=float(abs(g[:,1]).mean()),swing_hold_peak_rad=float(g[:,2].max()),
   wheel_delta_hold_RMS_rad_s=float(np.sqrt(np.mean(np.diff(target[:,WHEELS],axis=0)**2))),
   leg_delta_hold_RMS_rad=float(np.sqrt(np.mean(np.diff(target[:,LEGS],axis=0)**2))),
   wheel_request_hold_p95_Nm=float(np.percentile(abs(low['preclip'][hold][:,WHEELS]),95)))
 checks={'complete30':complete,'contact_ge_0p95':summary['both_contact_fraction']>=.95,
  'saturation_le_0p05':summary['saturation_any_fraction']<=.05,
  'swing_full_strict_lt_0p05':summary['swing_difference_peak_rad']<.05,
  'pitch_hold_le_0p1':result.get('pitch_hold_peak_rad',float('inf'))<=.1,
  'roll_hold_le_0p05':result.get('roll_hold_peak_rad',float('inf'))<=.05,
  'wheel_delta_hold_le_3':result.get('wheel_delta_hold_RMS_rad_s',float('inf'))<=3.,
  'leg_delta_hold_le_0p03':result.get('leg_delta_hold_RMS_rad',float('inf'))<=.03,
  'wheel_request_hold_le_5p5':result.get('wheel_request_hold_p95_Nm',float('inf'))<=5.5}
 result.update(checks=checks,failed_checks=[k for k,v in checks.items() if not v],
  diagnostic_screen_passed=all(checks.values()))
 return result


def motion_command_cases(targets=None):
 targets=targets or [[0.,0.],[.5,0.],[1.,0.],[1.5,0.],[2.,0.],[-.5,0.],[-1.,0.],[0.,.5],[0.,-.5]]
 if not 1<=len(targets)<=13 or len({tuple(t) for t in targets})!=len(targets) or any(len(t)!=2 or not np.isfinite(t).all() or max(abs(float(x)) for x in t)>4 or (t[0]!=0 and t[1]!=0) for t in targets):raise ValueError('distinct finite pure vx/yaw targets within4 required')
 base,_=actuator_profiles();base.update(leg_armature=.012,kp=40.,kd=2.,kv=.2,
  wheel_J=[.004204,.0042075],wheel_B=[0.,0.],command='scheduled_motion',schedule_s=[2.,8.,8.,8.],
  scope='candidate wheel profile, four legs fixed.012/Kp40/Kd2, same fixed sensing/target/PD delays')
 rows=[]
 for target in targets:
  initializations=('normal','log') if target==[0.,0.] else ('normal',)
  for initial in initializations:
   cfg={**copy.deepcopy(base),'command_target':list(target),'name':f'vx{target[0]:g}_yaw{target[1]:g}_{initial}'}
   rows.append({'policy':'candidate','initial':initial,'configuration':cfg,'group':'motion_command_screen'})
 return rows


def motion_command(config,age):
 from native_stand.command_v6.domain.command_schedule import effective
 entry,ramp,hold,exit=config['schedule_s']
 return effective(torch.tensor([config['command_target']],dtype=torch.float32),torch.tensor([age],dtype=torch.float32),
  height=.34,entry=entry,ramp=ramp,hold=hold,exit=exit)[0].numpy()


def motion_command_metrics(summary,low,requested):
 result=wheel_match_diagnostics(summary,low);t=low['policy_time'];mask=(t>=10)&(t<18);complete=result['full_window_complete'] and result['hold_complete']
 axis=2 if requested[1] else 0;target=float(requested[1] if requested[1] else requested[0]);zero=not any(requested)
 result.update(target=list(requested),response_axis='yaw_rad_s' if requested[1] else 'vx_mps',target_value=target,
  command_readback_complete=False,tracking_passed=False,physical_quality_passed=False,diagnostic_passed=False)
 if not result['hold_complete']:return result
 actual=low['velocity'][mask];command=low['command'][mask]
 expected=np.r_[requested,.34];readback=float(abs(command-expected).max());result['command_readback_max_error']=readback;result['command_readback_complete']=readback<1e-6
 signed=float(actual[:,axis].mean());horizontal=np.linalg.norm(actual[:,:2],axis=1)
 result.update(measured_mean=signed,tracking_MAE=float(abs(actual[:,axis]-target).mean()),
  direction_fraction=float((actual[:,axis]*target>0).mean()) if target else None,gain=signed/target if target else None,
  horizontal_hold_mean_mps=float(horizontal.mean()),horizontal_hold_p95_mps=float(np.percentile(horizontal,95)),
  vx_hold_mean_mps=float(actual[:,0].mean()),vx_hold_abs_mean_mps=float(abs(actual[:,0]).mean()),yaw_hold_mean_rad_s=float(actual[:,2].mean()),
  yaw_hold_abs_mean_rad_s=float(abs(actual[:,2]).mean()),height_hold_MAE_m=float(abs(low['state'][mask,2]-.34).mean()),
  wheel_actual_dq_hold_mean_rad_s=low['dq'][mask][:,WHEELS].mean(0).tolist(),wheel_target_hold_mean_rad_s=low['target'][mask][:,WHEELS].mean(0).tolist())
 park=(t>=26)&(t<30)
 if park.sum()==400:
  result['parking_horizontal_mean_mps']=float(np.linalg.norm(low['velocity'][park,:2],axis=1).mean())
  result['parking_yaw_abs_mean_rad_s']=float(abs(low['velocity'][park,2]).mean())
 tracking={'complete30':complete,'command_readback':result['command_readback_complete']}
 if zero:tracking.update(zero_horizontal_p95_le_0p3=result['horizontal_hold_p95_mps']<=.3,zero_yaw_mean_le_0p3=result['yaw_hold_abs_mean_rad_s']<=.3)
 else:
  tracking.update(MAE_le_0p3=result['tracking_MAE']<=.3,direction_ge_0p95=result['direction_fraction']>=.95,gain_0p8_to_1p2=.8<=result['gain']<=1.2)
  if requested[1]:tracking['in_place_translation_le_0p1']=result['horizontal_hold_mean_mps']<=.1
 physical={**result['checks'],'height_hold_band_0p32_to_0p36':.32<=result['height_hold_mean_m']<=.36}
 result.update(tracking_checks=tracking,physical_checks=physical,tracking_failed_checks=[k for k,v in tracking.items() if not v],
  physical_failed_checks=[k for k,v in physical.items() if not v],tracking_passed=all(tracking.values()),
  physical_quality_passed=all(physical.values()),diagnostic_passed=all(tracking.values()) and all(physical.values()))
 return result


def wheel_axis_inertia(model,joint):
 """Rigid-body inertia about its hinge, including the COM offset, no armature."""
 b=int(model.jnt_bodyid[joint]);r=model.body_ipos[b]-model.jnt_pos[joint]
 rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,model.body_iquat[b])
 axis=rotation.reshape(3,3).T@model.jnt_axis[joint]
 return float((model.body_inertia[b]*axis**2).sum()+model.body_mass[b]*(r@r-(r@model.jnt_axis[joint])**2))


def apply_actuator_profile(model,refs,cfg,original):
 """Restore before every case; mechanical drag remains outside motor clipping."""
 for key,value in original.items():getattr(model,key)[:]=value
 dofs=np.asarray(refs['dofs']);model.dof_armature[dofs[LEGS]]=cfg['leg_armature']
 for i,ch in enumerate(WHEELS):
  if cfg['wheel_J'] is not None:
   extra=cfg['wheel_J'][i]-wheel_axis_inertia(model,refs['joints'][ch])
   if extra<0:raise ValueError('reported total wheel J is below its rigid-body inertia')
   model.dof_armature[dofs[ch]]=extra
  if cfg['wheel_B'] is not None:model.dof_damping[dofs[ch]]=cfg['wheel_B'][i]
  if cfg['wheel_C'] is not None:model.dof_frictionloss[dofs[ch]]=0.
 return {'wheel_total_J_kg_m2':[wheel_axis_inertia(model,refs['joints'][ch])+float(model.dof_armature[dofs[ch]]) for ch in WHEELS],
  'wheel_damping_Nm_s_rad':model.dof_damping[dofs[WHEELS]].tolist(),
  'wheel_C_Nm':cfg['wheel_C'] or [0.,0.], 'wheel_peak_Nm':cfg['wheel_peak'],
  'leg_armature_kg_m2':model.dof_armature[dofs[LEGS]].tolist()}


class AuditExecution:
 """Same explicit PD/FIFO/hold/tau equations as timing runtime, in CPU diagnostics."""
 def __init__(self,cfg,bridge,history):
  self.cfg=cfg;self.adopted=bridge.copy();self.actual=np.zeros(6);self.pre=np.zeros(6);self.demand=np.zeros(6)
  self.transport=TargetTransport(cfg['transport_ms'],bridge)
  self.history=ObservationHistory(history[0][1][6:18],initial_tick=history[0][0])
  for tick,sample in history:self.history.append(tick,sample[6:18])
  self.limit=np.array([40,40,cfg['wheel_peak'],40,40,cfg['wheel_peak']])
 def step(self,tick,requested,q,dq):
  self.history.append(tick,np.r_[q,dq])
  if tick%10==0:self.transport.submit(tick,requested)
  self.delivered=self.transport.delivered(tick)
  if tick%5==0:
   self.adopted=self.delivered.copy();f=self.history.at(tick-self.cfg['feedback_ms'])
   self.pre[LEGS]=self.cfg['kp']*(self.adopted[LEGS]-f[:6][LEGS])-self.cfg['kd']*f[6:][LEGS]
   self.pre[WHEELS]=self.cfg['kv']*(self.adopted[WHEELS]-f[6:][WHEELS])
   self.demand=np.clip(self.pre,-self.limit,self.limit)
  self.actual=torque_response(self.actual,self.demand,self.cfg['tau_ms'])
  return self.actual


def normal_forces(model,data,refs):
 force=np.zeros(2)
 for i in range(data.ncon):
  c=data.contact[i]
  if refs['ground'] not in (c.geom1,c.geom2):continue
  other=c.geom2 if c.geom1==refs['ground'] else c.geom1;body=int(model.geom_bodyid[other])
  if body in refs['wheels']:
   f=np.zeros(6);mujoco.mj_contactForce(model,data,i,f);force[refs['wheels'].index(body)]+=max(0.,f[0])
 return force


def audit_spectrum(x,dt):
 x=np.asarray(x)
 if x.ndim==1:x=x[:,None]
 if len(x)<50:return {'peak_hz':None,'rms':None}
 x=x-x.mean(0);power=(abs(np.fft.rfft(x*np.hanning(len(x))[:,None],axis=0))**2).sum(1)
 freq=np.fft.rfftfreq(len(x),dt);mask=(freq>=2)&(freq<.45/dt)
 return {'peak_hz':float(freq[mask][power[mask].argmax()]) if power[mask].max()>1e-12 else None,
         'rms':float(np.sqrt(np.mean(x*x)))}


def audit_tail(rows):
 a={k:np.stack([r[k] for r in rows]) for k in rows[0]};contact=a['contact']>2
 # Describe event order, never infer causality from a single correlation.
 events={}
 for name,mask in [('PD_request_above_limit',(abs(a['preclip'])>a['limits']).any(1)),
  ('single_or_dual_contact_loss',~contact.all(1)),('dual_contact_loss',~contact.any(1))]:
  idx=np.flatnonzero(mask);events[name]=float(a['time'][idx[0]]) if len(idx) else None
 spectra={name:audit_spectrum(a[field][:,cols],.001) for name,field,cols in
  [('wheel_applied_torque','torque',WHEELS),('wheel_dq','dq',WHEELS),('leg_dq','dq',LEGS),('leg_q','q',LEGS)]}
 duration=len(rows)*.001
 flips=(np.sign(a['torque'][1:,WHEELS])*np.sign(a['torque'][:-1,WHEELS])<0).sum(0)/duration
 return a,{'interval_s':[float(a['time'][0]),float(a['time'][-1])],'sample_rate_hz':1000,
  'first_observed_in_tail_s':events,'wheel_torque_sign_flips_per_s':flips.tolist(),'spectra':spectra,
  'both_contact_fraction':float(contact.all(1).mean()),'single_lost_fraction':float((contact.sum(1)==1).mean()),
  'both_lost_fraction':float((contact.sum(1)==0).mean()),
  'interpretation':'tail only; an earlier event may be outside retained window; no contact force or PWM hardware parity claim'}


def audit_rollout(model,refs,actor,initial,history,bridge,logged,case,seconds,deadline):
 cfg=case['configuration'];d=mujoco.MjData(model);d.qpos[:]=initial.qpos;d.qvel[:]=initial.qvel;mujoco.mj_forward(model,d)
 h=ObservationHistory(history[0][1],initial_tick=history[0][0])
 for tick,sample in history:h.append(tick,sample)
 ex=AuditExecution(cfg,bridge,history);requested=bridge.copy();last_prediction=bridge.copy()
 bounds=torch.tensor(model.jnt_range[refs['joints']],dtype=torch.float32)[None]
 for i,j in enumerate(refs['joints']):
  if not model.jnt_limited[j]:bounds[0,i]=torch.tensor([-float('inf'),float('inf')])
 rows=[];tail=deque(maxlen=2000);failed=None;first_obs=None;first_target=None;last_obs=np.zeros(31)
 first_events={k:None for k in ('PD_request_above_limit','single_or_dual_contact_loss','dual_contact_loss')}
 for tick in range(round(seconds*1000)):
  if tick%100==0 and time.monotonic()>deadline:raise TimeoutError('actuator audit wall bound reached')
  current=sensors(model,d,refs);h.append(tick,current)
  if tick%10==0:
   sensed=h.at(tick-cfg['actor_ms']);dq_ms=cfg['actor_dq_ms']
   if dq_ms is not None:sensed[12:18]=h.at(tick-dq_ms)[12:18]
   cache=ex.adopted.copy()
   if cfg['cache']=='logged_seed' and tick==0:cache=logged['row']['previous'].copy()
   elif cfg['cache']=='zero_seed' and tick==0:cache[:]=0
   elif cfg['cache']=='requested_recursive':cache=last_prediction.copy()
   elif cfg['cache']=='measured_wheels':cache[WHEELS]=sensed[12:18][WHEELS]
   cmd=np.array([0.,0.,.34])
   if cfg['command']=='logged_hold_last':
    ix=max(0,int(np.searchsorted(logged['times'],logged['event']+tick*.001,side='right')-1));cmd=logged['observations'][ix,22:25]*[4.,4.,.5]
   elif cfg['command']=='scheduled_motion':cmd=motion_command(cfg,tick*.001)
   obs=encode(torch.tensor(sensed[:3],dtype=torch.float32)[None],torch.tensor(sensed[3:6],dtype=torch.float32)[None],
    torch.tensor(sensed[6:12],dtype=torch.float32)[None],torch.tensor(sensed[12:18],dtype=torch.float32)[None],
    torch.tensor(cache[LEGS],dtype=torch.float32)[None],torch.tensor(cache[WHEELS],dtype=torch.float32)[None],
    torch.zeros(1,6),'Motion',torch.tensor(cmd,dtype=torch.float32)[None])
   mean=actor.mean(obs)
   if not torch.isfinite(mean).all():raise RuntimeError('nonfinite Actor output')
   out=targets(mean,torch.zeros(1,6),bounds);requested[LEGS]=out['leg_executed'][0].numpy();requested[WHEELS]=out['wheel_executed'][0].numpy()
   last_prediction=requested.copy();last_obs=obs[0].numpy().copy()
   if tick==0:first_obs=last_obs.copy();first_target=requested.copy()
  actual=ex.step(tick,requested,current[6:12],current[12:18]);d.qfrc_applied[:]=0.
  if cfg['wheel_C'] is not None:
   d.qfrc_applied[np.asarray(refs['dofs'])[WHEELS]]=-np.asarray(cfg['wheel_C'])*np.tanh(current[12:18][WHEELS]/cfg['friction_epsilon'])
  d.ctrl[refs['actuators']]=actual/refs['gears'];mujoco.mj_step(model,d)
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError('nonfinite MuJoCo state')
  # Align post-step geometry/contacts with controls held over the preceding 1ms interval.
  mujoco.mj_forward(model,d);rotation=d.xmat[refs['root']].reshape(3,3);gravity=rotation.T@np.array([0.,0.,-1.])
  force=normal_forces(model,d,refs);now=(tick+1)*.001
  tail.append({'time':now,'q':d.qpos[refs['qpos']].copy(),'dq':d.qvel[refs['dofs']].copy(),
   'target':requested.copy(),'adopted':ex.adopted.copy(),'preclip':ex.pre.copy(),'torque':actual.copy(),
   'limits':ex.limit.copy(),'contact':force,'gravity':gravity.copy()})
  for key,flag in [('PD_request_above_limit',bool((abs(ex.pre)>ex.limit).any())),
   ('single_or_dual_contact_loss',not bool((force>2).all())),('dual_contact_loss',not bool((force>2).any()))]:
   if flag and first_events[key] is None:first_events[key]=now
  if tick%10==9:
   _,_,s=state(model,d,refs);geom=s['posture_geometry'];v=(d.xpos[refs['wheels']]-d.xanchor[np.asarray(refs['joints'])[[0,3]]])@rotation
   swing=np.arctan2(-v[:,2],-v[:,1]);diff=float(abs(np.arctan2(np.sin(swing[0]-swing[1]),np.cos(swing[0]-swing[1]))))
   rows.append({'time':now,'target':requested.copy(),'torque':actual.copy(),'preclip':ex.pre.copy(),
    'dq':d.qvel[refs['dofs']].copy(),'contact':force.copy(),'state':np.array([s['pitch'],s['roll'],s['height'],s['vx'],s['up'],s['closure_m']]),
    'geometry':np.array([geom['wheel_heading_offset_m'],geom['leg_length_difference_m'],diff]),'actor_input':last_obs.copy()})
   if cfg['command']=='scheduled_motion':rows[-1].update(policy_time=(tick//10)*.01,
    command=last_obs[22:25].copy()*[4.,4.,.5],velocity=np.array([s['vx'],s['vy'],s['yaw_rate']]))
   if s['closure_m']>.005:failed='CLOSURE'
   elif s['up']<.5:failed='UPRIGHT'
   if failed:break
 summary,a=summarize(rows,failed,seconds,case['initial']);hf,hf_summary=audit_tail(tail)
 summary['saturation_per_channel']=(abs(a['torque'])>=ex.limit-1e-4).mean(0).tolist()
 summary['saturation_any_fraction']=float((abs(a['torque'])>=ex.limit-1e-4).any(1).mean())
 summary.update(wheel_preclip_abs_p95_Nm=np.percentile(abs(a['preclip'][:,WHEELS]),95,axis=0).tolist(),
  wheel_heading_offset_peak_m=float(abs(a['geometry'][:,0]).max()),leg_length_difference_peak_m=float(abs(a['geometry'][:,1]).max()),
  swing_difference_peak_rad=float(a['geometry'][:,2].max()),initial_actor_input=first_obs.tolist(),
  initial_target=first_target.tolist(),initial_leg_target_jump_rad=(first_target[LEGS]-initial.qpos[np.asarray(refs['qpos'])[LEGS]]).tolist(),
  first_observed_events_s=first_events,high_frequency_tail=hf_summary,
  posture_diagnostic_checks={'pitch_max_0p1':float(abs(a['state'][:,0]).max())<=.1,'roll_max_0p05':float(abs(a['state'][:,1]).max())<=.05,
   'swing_strict_0p05':float(a['geometry'][:,2].max())<.05},accepted='NOT_ASSESSED_DIAGNOSTIC_SCREEN')
 return summary,a,hf


def actuator_rig_checks(model,refs,out,deadline,*,sensitivity=False,kvs=(1.,),mechanical_b=.01):
 """One-axis MuJoCo plants exercise the same FIFO/PD/response used above."""
 base,_=actuator_profiles();checks=[]
 matrix=[(kind,wave,None,10,1.) for kind,wave in itertools.product(('wheel','leg'),('step','sine','brake'))]
 if sensitivity:matrix=[('wheel',wave,J,feedback,kv) for wave,J,feedback,kv in itertools.product(('step','sine','brake'),(.01066759,.004204),(0,10,20),kvs)]
 for kind,wave,J,feedback,kv in matrix:
  ch=2 if kind=='wheel' else 0;arm=(J-.00066759) if J is not None else .01 if kind=='wheel' else .35
  xml=f'<mujoco><option timestep=".001" gravity="0 0 0" integrator="implicitfast"/><worldbody><body><joint axis="1 0 0" armature="{arm}" damping="{mechanical_b}"/><inertial pos="0 0 0" mass=".41275" diaginertia=".00066759 .00035004 .00035004"/></body></worldbody><actuator><motor joint=""/></actuator></mujoco>'
  xml=xml.replace('<joint axis','<joint name="joint" axis').replace('joint=""','joint="joint"')
  rig=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(rig)
  cfg={**base,'feedback_ms':feedback,'kv':kv};initial=np.zeros(18);ex=AuditExecution(cfg,np.zeros(6),[(-20,initial)])
  rows=[];first_adopt=None;max_motor_error=0.
  for tick in range(3000):
   if tick%100==0 and time.monotonic()>deadline:raise TimeoutError('actuator rig wall exceeded')
   q=np.zeros(6);dq=q.copy();q[ch]=d.qpos[0];dq[ch]=d.qvel[0]
   t=(tick//10)*.01;requested=np.zeros(6)
   amplitude=(1. if sensitivity else 20.) if kind=='wheel' else .1
   if wave=='step':value=amplitude if t>=.2 else 0.
   elif wave=='brake':value=amplitude if .2<=t<1.5 else 0.
   else:value=((.5 if sensitivity else 4.) if kind=='wheel' else .05)*np.sin(2*np.pi*2*t) if t>=.2 else 0.
   requested[ch]=value;actual=ex.step(tick,requested,q,dq);d.ctrl[0]=actual[ch]
   mujoco.mj_step(rig,d);max_motor_error=max(max_motor_error,abs(float(d.qfrc_actuator[0])-actual[ch]))
   if first_adopt is None and abs(ex.adopted[ch])>1e-10:first_adopt=tick
   rows.append([tick*.001,value,ex.adopted[ch],ex.pre[ch],ex.demand[ch],actual[ch],d.qpos[0],d.qvel[0]])
  a=np.array(rows);name=kind+'_'+wave+(f'_J{J:g}_fb{feedback}_Kv{kv:g}' if sensitivity else '');np.savez_compressed(out/(name+'.npz'),trace=a)
  expected_first=210 # wheel FIFO7ms rounded by5ms PD; leg FIFO10ms.
  passed=bool(max_motor_error<1e-10 and abs(a[:,5]).max()<=ex.limit[ch]+1e-10 and (wave!='step' or first_adopt==expected_first))
  checks.append({'name':name,'passed':passed,'first_nonzero_adopt_ms':first_adopt,'motor_force_error_max_Nm':max_motor_error,
    'peak_actual_Nm':float(abs(a[:,5]).max()),'peak_preclip_Nm':float(abs(a[:,3]).max()),
    'final_q_rad':float(a[-1,6]),'final_dq_rad_s':float(a[-1,7]),'trace':name+'.npz',
    'total_J_kg_m2':arm+.00066759,'PD_feedback_ms':feedback,'Kv_Nm_per_rad_s':kv,'mechanical_B_Nm_s_rad':mechanical_b,
    'tail_speed_error_rms_rad_s':float(np.sqrt(np.mean((a[-1000:,7]-a[-1000:,1])**2))),
    'tail_saturation_fraction':float((abs(a[-1000:,5])>=ex.limit[ch]-1e-4).mean()),
    'tail_velocity_spectrum':audit_spectrum(a[-1000:,7],.001),
    'trace_columns':['time_pre_s','request','adopted','PD_preclip_Nm','clipped_demand_Nm','applied_Nm','q_post_rad','dq_post_rad_s'],
    'scope':'isolated one-axis simulation consistency; not full robot or measured hardware response'})
 return checks


def wheel_match_compare(args):
 """Authorized CPU-only source/new-policy test and exact deployed-graph Kv ablation."""
 if args.seconds!=30 or not 30<=args.wall_seconds<=600:raise ValueError('wheel comparison requires30s windows and bounded wall<=600s')
 root=Path(__file__).resolve().parents[1];search=root/'sim2sim/2026_10_21-09_35__motion3250_handover_search'
 meta=json.loads((search/'manifest.json').read_text());saved_case=json.loads((search/'case_039.json').read_text())
 csv=next(Path(p) for p in meta['input_sha256'] if p.endswith('.csv'))
 if sha(csv)!=meta['input_sha256'][str(csv)]:raise ValueError('real log SHA differs')
 xml=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml')
 model=mujoco.MjModel.from_xml_path(str(xml));refs=binding(model)
 original={k:getattr(model,k).copy() for k in ('dof_armature','dof_damping','dof_frictionloss')}
 inputs={str(p.resolve()):sha(p) for p in (xml,csv,search/'manifest.json',search/'case_039.json',Path(__file__))}
 torch.set_num_threads(1);actors={};checkpoint_metadata={}
 for name,cp,sp in [('source387',args.source_checkpoint,args.source_spec),('candidate',args.candidate_checkpoint,args.candidate_spec)]:
  cp=cp.resolve();sp=sp.resolve();s=json.loads(sp.read_text());actor,data=load(cp,'MotionZeroStudent')
  if data['spec_sha256']!=sha(sp) or data['asset_sha256']!=s['asset_sha256'] or sha(s['asset_path'])!=s['asset_sha256']:raise ValueError('paired checkpoint/spec/USD identity differs')
  actors[name]=actor.eval();inputs.update({str(cp):sha(cp),str(sp):sha(sp),str(Path(s['asset_path']).resolve()):s['asset_sha256']})
  checkpoint_metadata[name]={'checkpoint':str(cp),'sha256':sha(cp),'spec':str(sp),'update':data['update']}
  if name=='candidate':
   if s['source_checkpoint']!=str(args.source_checkpoint.resolve()) or s['source_sha256']!=sha(args.source_checkpoint):raise ValueError('candidate must descend from this exact source')
   cfg=s.get('motion_timing',{}).get('configuration',{});p=cfg.get('wheel_execution',{})
   if cfg.get('wheel_kv')!=.2 or p.get('wheel_total_J_candidate_kg_m2_ch2_ch5')!=[.004204,.0042075]:raise ValueError('candidate wheel profile differs from diagnostic')
 deployed=args.deployed_onnx.resolve();manifest_path=deployed.with_name('manifest.json');gold_path=deployed.with_name('golden_vectors.npz')
 package=json.loads(manifest_path.read_text());onnx_actor=DeployedONNX(deployed,package)
 if sha(gold_path)!=package['files_sha256']['golden_vectors.npz']:raise ValueError('deployed golden SHA differs')
 with np.load(gold_path) as golden:
  actual=onnx_actor.mean(torch.from_numpy(golden['observation'].copy())).numpy()
  parity_error=float(abs(actual-golden['pytorch_mean_action']).max());count=len(actual)
 if parity_error>1e-5:raise ValueError('actual ONNX/golden parity differs')
 actors['deployed_onnx']=onnx_actor
 checkpoint_metadata['deployed_onnx']={'checkpoint':str(deployed),'sha256':sha(deployed),'backend':'onnx.reference.ReferenceEvaluator',
  'source_sha256':package['source_sha256'],'golden_count':count,'golden_max_abs_error':parity_error}
 for p in (deployed,manifest_path,gold_path,root/'src/native_stand/students/real2sim_delay.py',root/'src/native_stand/students/real2sim_handover.py',root/'src/native_stand/students/mcu_interface.py',root/'src/native_stand/students/interface.py',Path(__file__).with_name('validate_native_mujoco.py'),Path(__file__).with_name('diagnose_motion_real2sim.py')):inputs[str(p.resolve())]=sha(p)
 _,observations,_,times,_=read_csv(csv);cases=wheel_match_cases(args.deployed_reference_only)
 out=args.output_dir or root/'sim2sim'/new_experiment_name('motion387_1750_deployed_kv_compare')
 from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
 validate_new_experiment_name(out.name)
 if out.exists():raise FileExistsError(out)
 manifest={'version':'motion_wheel_match_paired_diagnostic_v1','input_sha256':inputs,'models':checkpoint_metadata,
  'cases':cases,'max_cases':len(cases),'seconds_per_case':30.,'wall_limit_s':args.wall_seconds,'storage_max_bytes':128*2**20,
  'device':'cpu','PPO_updates':0,'independent_seeds':1,'tail_hz':1000,'tail_max_s':2.,
  'authorization':'User approves source/candidate .02/.012 log handover pairing plus deployed small ONNX changing only wheel Kv to.2',
  'scope':'finite diagnostic, no training mutation or new formal acceptance; current MJCF contact/leg damping remain and may differ from Native',
  'missing_real_state':['actual LQR executed targets','root velocity/height','contact/true takeover timestamp','feedback age','residual spring torque'],
  'previous':'bridge uses measured q/dq as initial target hypothesis; logged_seed changes only first Actor previous; subsequent cache PD-adopted',
  'root_velocity':'assumed -.3m/s at root body origin from existing initializer; COM readback reported separately, not measured hardware velocity'}
 print(json.dumps({'output':str(out),'cases':len(cases),'models':checkpoint_metadata,'wall_limit_s':args.wall_seconds,'simulation':bool(args.run),'PPO':0}),flush=True)
 if not args.run:return
 import shutil
 if shutil.disk_usage(root).free<384*2**20:raise RuntimeError('diagnostic storage reserve unavailable')
 out.mkdir(parents=True,exist_ok=False)
 def write(name,value):
  with (out/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
 write('manifest.json',manifest);start=time.monotonic();results=[];initializations={};status='FAILED';failure=None
 write('budget.json',{'status':'RUNNING','wall_limit_s':args.wall_seconds,'storage_max_bytes':128*2**20,'max_cases':len(cases),'PPO_updates':0})
 try:
  with torch.inference_mode():
   for index,case in enumerate(cases):
    cfg=case['configuration'];readback=apply_actuator_profile(model,refs,cfg,original)
    key=(case['initial'],cfg['event'],cfg['root_vx'])
    if key not in initializations:
     event=saved_case['event_sample_time_s'];ix=int(np.searchsorted(times,event,side='right')-1)
     row=decode_log(observations[ix]);history=[]
     for j in range(ix-15,ix):
      v=decode_log(observations[j]);history.append((int(round((times[j]-times[ix])*1000)),np.r_[v['gyro'],v['gravity'],v['q'],v['dq']]))
     logged={'row':row,'times':times,'observations':observations,'event':float(times[ix])}
     if case['initial']=='log':
      initial,fit=initialize_from_log(model,refs,observations[ix],cfg['root_vx']);bridge=row['q'].copy();bridge[WHEELS]=row['dq'][WHEELS]
     else:
      initial=mujoco.MjData(model);mujoco.mj_forward(model,initial);ss=sensors(model,initial,refs)
      history=[(-20,ss),(-10,ss)];bridge=np.zeros(6);fit={'source':'MJCF normal zero pose'}
     fit['initial_COM_vx_mps']=float(state(model,initial,refs)[2]['vx'])
     initializations[key]=(initial,history,bridge,logged,fit)
    initial,history,bridge,logged,fit=initializations[key]
    summary,low,high=audit_rollout(model,refs,actors[case['policy']],initial,history,bridge,logged,case,30.,start+args.wall_seconds)
    name=f"case_{index:03d}__{case['policy']}__{cfg['name']}__{case['initial']}"
    summary.update(case_id=name,policy=case['policy'],profile=cfg['name'],group=case['group'],pair_id=case['pair_id'],
     initial_scenario=case['initial_scenario'],configuration=cfg,readback=readback,initial_fit=fit,
     diagnostic=wheel_match_diagnostics(summary,low),trace=name+'.npz',tail_trace=name+'__tail.npz')
    np.savez_compressed(out/summary['trace'],**low);np.savez_compressed(out/summary['tail_trace'],**high);write(name+'.json',summary);results.append(summary)
    print(json.dumps({'case':index+1,'total':len(cases),'policy':case['policy'],'profile':cfg['name'],'initial':case['initial'],
     'termination':summary['termination'],'seconds':summary['simulated_seconds'],'screen':summary['diagnostic']['diagnostic_screen_passed']}),flush=True)
    if sum(p.stat().st_size for p in out.iterdir())>126*2**20:raise RuntimeError('diagnostic storage cap')
  for path,digest in inputs.items():
   if sha(path)!=digest:raise RuntimeError('input changed during diagnostic: '+path)
  status='COMPLETED'
 except BaseException as exc:
  failure={'type':type(exc).__name__,'message':str(exc)};raise
 finally:
  wall=time.monotonic()-start
  write('summary.json',{'status':status,'failure':failure,'wall_seconds':wall,'results':results,'PPO_updates':0})
  write('terminal_report.json',{'status':status,'failure':failure,'wall_seconds':wall,'completed_cases':len(results),'planned_cases':len(cases),'PPO_updates':0})
  (out/'budget.json').write_text(json.dumps({'status':status,'wall_limit_s':args.wall_seconds,'wall_used_s':wall,'storage_max_bytes':128*2**20,'completed_cases':len(results),'PPO_updates':0},indent=2)+'\n')
  lines=['# 来源／新候选与电控ONNX轮Kv对照','',f'状态：{status}；{len(results)}/{len(cases)}条件；PPO0；每case要求30秒。',
   '实际电控ONNX直接CPU推理，未用同名PyTorch模型替代。deployed_Kv_only同一pair仅Kv变化；source_candidate使用同一新执行模型。',
   '', '|模型|配对条件|Kv|实际时长s|双轮接触|近限|全程pitch峰值rad|摆角差峰值rad|轮目标Δ RMS rad/s|诊断全过|',
   '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
  for r in results:lines.append(f"|{r['policy']}|{r['pair_id']}|{r['configuration']['kv']:g}|{r['simulated_seconds']:.2f}|{r['both_contact_fraction']:.2%}|{r['saturation_any_fraction']:.2%}|{r['pitch_abs_peak_rad']:.5f}|{r['swing_difference_peak_rad']:.5f}|{r['wheel_target_delta_rms_rad_s']:.3f}|{r['diagnostic']['diagnostic_screen_passed']}|")
  lines+=['','诊断筛选不授Native/实机能力，存活与姿态/对称/平滑分别判断；失败早停不把短窗口当30秒通过。',
   '相同pair初态、previous、接触参数、时延和轮J均相同。真实LQR执行目标/反馈年龄等仍未知，日志previous首帧仅是假设。',
   '诊断全过只覆盖列出的窗口/姿态/摆角/接触/限矩/平滑检查，不含高度精度或零速漂移；两者必须另看物理指标，不能称完整站立认证。',
   '高频仅保留末2秒1kHz，不重复保存整树源码/模型，不修改活动训练。']
  (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def motion_command_screen(args):
 """Fixed selected candidate; command, observed axes and scoring windows explicit."""
 if args.seconds!=30 or not 30<=args.wall_seconds<=600:raise ValueError('Motion screen requires30s per case and wall<=600s')
 root=Path(__file__).resolve().parents[1];sp=args.candidate_spec.resolve();cp=args.candidate_checkpoint.resolve()
 spec=json.loads(sp.read_text());actor,data=load(cp,'MotionZeroStudent');torch.set_num_threads(1);actor=actor.eval()
 if data['spec_sha256']!=sha(sp) or data['asset_sha256']!=spec['asset_sha256'] or sha(spec['asset_path'])!=spec['asset_sha256']:raise ValueError('candidate/spec/USD identity differs')
 cfg=spec.get('motion_timing',{}).get('configuration',{})
 if cfg.get('wheel_kv')!=.2 or cfg.get('leg_kp')!=40 or cfg.get('leg_kd')!=2 or cfg.get('wheel_execution',{}).get('wheel_total_J_candidate_kg_m2_ch2_ch5')!=[.004204,.0042075]:raise ValueError('explicit wheel-match/Kp40 candidate required')
 xml=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml');model=mujoco.MjModel.from_xml_path(str(xml));refs=binding(model)
 original={k:getattr(model,k).copy() for k in ('dof_armature','dof_damping','dof_frictionloss')}
 search=root/'sim2sim/2026_10_21-09_35__motion3250_handover_search';meta=json.loads((search/'manifest.json').read_text());saved=json.loads((search/'case_039.json').read_text())
 csv=next(Path(p) for p in meta['input_sha256'] if p.endswith('.csv'))
 if sha(csv)!=meta['input_sha256'][str(csv)]:raise ValueError('log input differs')
 _,observations,_,times,_=read_csv(csv);cases=motion_command_cases(args.command_target)
 paths=[sp,cp,Path(spec['asset_path']),xml,csv,search/'manifest.json',search/'case_039.json',Path(__file__),
  root/'src/native_stand/students/real2sim_delay.py',root/'src/native_stand/students/real2sim_handover.py',
  root/'src/native_stand/students/mcu_interface.py',root/'src/native_stand/students/interface.py',
  root/'src/native_stand/command_v6/domain/command_schedule.py',Path(__file__).with_name('validate_native_mujoco.py'),Path(__file__).with_name('diagnose_motion_real2sim.py')]
 inputs={str(p.resolve()):sha(p) for p in paths}
 out=args.output_dir or root/'sim2sim'/new_experiment_name(f'motion_u{data["update"]}_arm012_kp40_commands')
 from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
 validate_new_experiment_name(out.name)
 if out.exists():raise FileExistsError(out)
 manifest={'version':'motion_arm012_command_diagnostic_v1','checkpoint':str(cp),'checkpoint_sha256':sha(cp),'checkpoint_update':data['update'],
  'spec':str(sp),'input_sha256':inputs,'cases':cases,'max_cases':len(cases),'seconds_per_case':30.,'wall_limit_s':args.wall_seconds,
  'storage_max_bytes':64*2**20,'device':'cpu','PPO_updates':0,'independent_seeds':1,
  'authorization':'User explicitly asks new update at legarmature.012 Kp40 for Motion and smooth standing; current training unchanged',
  'protocol':{'schedule_s':[2.,8.,8.,8.],'hold_s':[10.,18.],'parking_s':[26.,30.],
   'clock':'pre-policy time selects input command; state is post-policy interval, explicit policy_time and physical time both saved',
   'target_height_m':.34,'height_band_m':[.32,.36],'tracking_MAE_max':.3,'gain_range':[.8,1.2],'direction_min':.95,
   'zero_horizontal_p95_max_mps':.3,'in_place_horizontal_mean_max_mps':.1,'pitch_hold_peak_max_rad':.1,'roll_hold_peak_max_rad':.05,
   'swing_full_peak_strict_max_rad':.05,'contact_min':.95,'saturation_max':.05},
  'trace_schema':{'velocity':['native_vx_mps','native_vy_mps','native_yaw_rate_rad_s'],'command':['requested_vx_mps','requested_yaw_rad_s','height_m'],
   'geometry':['wheel_heading_offset_m','leg_length_difference_m','swing_difference_rad'],'state':['pitch_rad','roll_rad','root_height_m','native_vx_mps','up_cos','closure_m']},
  'tail':{'hz':1000,'max_s':2},'scope':'MuJoCo single-state command retention screen, not new Native/real acceptance; no optimizer/teacher',
  'known_differences':'MJCF contacts/leg mechanical drag retained, not PhysX parity; log velocity and previous remain explicit hypotheses'}
 print(json.dumps({'output':str(out),'checkpoint_update':data['update'],'checkpoint_sha256':sha(cp),'commands':[c['configuration']['command_target'] for c in cases],
  'conditions':len(cases),'simulation':bool(args.run),'PPO':0}),flush=True)
 if not args.run:return
 import shutil
 if shutil.disk_usage(root).free<256*2**20:raise RuntimeError('diagnostic storage reserve unavailable')
 out.mkdir(parents=True,exist_ok=False)
 def write(name,value):
  with (out/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
 write('manifest.json',manifest);start=time.monotonic();results=[];initials={};status='FAILED';failure=None
 write('budget.json',{'status':'RUNNING','wall_limit_s':args.wall_seconds,'storage_max_bytes':64*2**20,'max_cases':len(cases),'PPO_updates':0})
 try:
  with torch.inference_mode():
   for index,case in enumerate(cases):
    c=case['configuration'];readback=apply_actuator_profile(model,refs,c,original)
    if case['initial'] not in initials:
     ix=int(np.searchsorted(times,saved['event_sample_time_s'],side='right')-1);row=decode_log(observations[ix]);history=[]
     for j in range(ix-15,ix):
      v=decode_log(observations[j]);history.append((int(round((times[j]-times[ix])*1000)),np.r_[v['gyro'],v['gravity'],v['q'],v['dq']]))
     logged={'row':row,'times':times,'observations':observations,'event':float(times[ix])}
     if case['initial']=='log':
      initial,fit=initialize_from_log(model,refs,observations[ix],-.3);bridge=row['q'].copy();bridge[WHEELS]=row['dq'][WHEELS]
     else:
      initial=mujoco.MjData(model);mujoco.mj_forward(model,initial);v=sensors(model,initial,refs)
      history=[(-20,v),(-10,v)];bridge=np.zeros(6);fit={'source':'MJCF normal zero pose'}
     fit['COM_vx_mps']=float(state(model,initial,refs)[2]['vx']);initials[case['initial']]=(initial,history,bridge,logged,fit)
    initial,history,bridge,logged,fit=initials[case['initial']]
    summary,low,high=audit_rollout(model,refs,actor,initial,history,bridge,logged,case,30.,start+args.wall_seconds)
    name=f'case_{index:02d}__{c["name"]}'
    summary.update(case_id=name,command=c['command_target'],configuration=c,readback=readback,initial_fit=fit,
     motion=motion_command_metrics(summary,low,c['command_target']),trace=name+'.npz',tail_trace=name+'__tail.npz')
    np.savez_compressed(out/summary['trace'],**low);np.savez_compressed(out/summary['tail_trace'],**high);write(name+'.json',summary);results.append(summary)
    print(json.dumps({'case':index+1,'conditions':len(cases),'command':c['command_target'],'initial':case['initial'],'termination':summary['termination'],
     'seconds':summary['simulated_seconds'],'tracking':summary['motion']['tracking_passed'],'physical':summary['motion']['physical_quality_passed']}),flush=True)
    if sum(p.stat().st_size for p in out.iterdir())>62*2**20:raise RuntimeError('Motion screen storage cap')
  for p,digest in inputs.items():
   if sha(p)!=digest:raise RuntimeError('diagnostic input changed: '+p)
  status='COMPLETED'
 except BaseException as exc:
  failure={'type':type(exc).__name__,'message':str(exc)};raise
 finally:
  wall=time.monotonic()-start
  write('summary.json',{'status':status,'failure':failure,'wall_seconds':wall,'results':results,'PPO_updates':0,'capability_certified':False})
  write('terminal_report.json',{'status':status,'failure':failure,'wall_seconds':wall,'completed_cases':len(results),'planned_cases':len(cases),'PPO_updates':0})
  (out/'budget.json').write_text(json.dumps({'status':status,'wall_used_s':wall,'wall_limit_s':args.wall_seconds,'storage_max_bytes':64*2**20,'completed_cases':len(results),'PPO_updates':0},indent=2)+'\n')
  def fmt(value):return '缺完整保持窗' if value is None else f'{value:.4f}'
  lines=[f'# update{data["update"]}：腿armature.012/Kp40运动与平滑站立','',
   f'状态{status}，{len(results)}/{len(cases)}条件，0PPO；每条件30秒上限，保持段10..18秒；同一固定checkpoint。',
   '', '|命令vx/yaw|初态|实际时长s|实际响应均值|MAE|gain|保持水平速度m/s|双接触|限矩|跟踪通过|物理/姿态/平滑通过|',
   '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
  for r in results:
   m=r['motion'];lines.append(f"|{r['command']}|{r['initialization']}|{r['simulated_seconds']:.2f}|{fmt(m.get('measured_mean'))}|{fmt(m.get('tracking_MAE'))}|{fmt(m.get('gain'))}|{fmt(m.get('horizontal_hold_mean_mps'))}|{r['both_contact_fraction']:.2%}|{r['saturation_any_fraction']:.2%}|{m['tracking_passed']}|{m['physical_quality_passed']}|")
  lines+=['','非零vx用m/s，yaw用rad/s，yaw评分绝不读取vx当响应。每个case保存真实Actor命令readback与逐轴速度，missing hold不估算通过。',
   '高度目标.34m，诊断带[.32,.36]m；高度MAE另报，不将带内视作毫米精度。每项失败原因/姿态/摆角/轮目标平滑/轮矩尾窗见逐case JSON与NPZ。',
   '当前训练/奖励/课程/checkpoint/账本未改。MuJoCo单环境诊断，不构成Native多环境或实机能力认证。']
  (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def actuator_audit(args):
 """Run one bounded diagnostic matrix without touching trainer state or external assets."""
 if not 2.5<=args.seconds<=30 or not 30<=args.wall_seconds<=600:raise ValueError('audit duration/wall outside bounds')
 root=Path(__file__).resolve().parents[1];search=root/'sim2sim/2026_10_21-09_35__motion3250_handover_search'
 metadata=json.loads((search/'manifest.json').read_text());saved_case=json.loads((search/'case_039.json').read_text())
 csv=next(Path(p) for p in metadata['input_sha256'] if p.endswith('.csv'))
 if sha(csv)!=metadata['input_sha256'][str(csv)]:raise ValueError('real log changed')
 xml=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml');model=mujoco.MjModel.from_xml_path(str(xml));refs=binding(model)
 original={k:getattr(model,k).copy() for k in ('dof_armature','dof_damping','dof_frictionloss')}
 _,x,_,times,_=read_csv(csv);rig_only=getattr(args,'actuator_rig_only',False)
 cases=[] if rig_only else actuator_cases(getattr(args,'audit_confirm',False));policies={
  'teacher3250':('runs/2026_10_18-23_10__motion1000_differential85_5000_retry/formal/model_update_3250.pt','configs/experiments/2026_10_18-23_10__motion1000_differential85_5000_retry/spec.json'),
  'small3250':('isolated/2026_10_19-19_55__motion3250_small31_dagger_round2/round_01/fit/best_validation.pt','configs/experiments/2026_10_18-23_10__motion1000_differential85_5000_retry/spec.json'),
  'final500':('runs/2026_10_21-18_58__motion526_sensor_schema_formal9183/formal/model_update_500.pt','configs/experiments/2026_10_21-18_58__motion526_sensor_schema_formal9183/spec.json')}
 actors={};inputs={str(p.resolve()):sha(p) for p in (xml,csv,search/'manifest.json',search/'case_039.json',Path(__file__))}
 torch.set_num_threads(1)
 for name,(cp,sp) in ([] if rig_only else policies.items()):
  cp=root/cp;sp=root/sp;s=json.loads(sp.read_text());actor,data=load(cp,'MotionZeroStudent')
  if data.get('distillation'):
   from native_stand.students.motion_distillation_pipeline import validate_student
   lineage=validate_student(data,sp)
   if lineage['teacher_sha256']!=sha(root/policies['teacher3250'][0]):raise ValueError('Student/Teacher pairing differs')
  elif data['spec_sha256']!=sha(sp) or data['asset_sha256']!=s['asset_sha256']:raise ValueError('Actor/spec identity differs')
  if sha(s['asset_path'])!=s['asset_sha256']:raise ValueError('USD identity changed')
  actors[name]=actor.eval();inputs.update({str(cp):sha(cp),str(sp):sha(sp),str(Path(s['asset_path']).resolve()):s['asset_sha256']})
 for p in (root/'src/native_stand/students/real2sim_delay.py',root/'src/native_stand/students/real2sim_handover.py',root/'src/native_stand/students/mcu_interface.py',Path(__file__).with_name('validate_native_mujoco.py')):inputs[str(p.resolve())]=sha(p)
 out=args.output_dir or root/'sim2sim'/new_experiment_name('motion_actuator_paired_audit')
 from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
 validate_new_experiment_name(out.name)
 if out.exists():raise FileExistsError(out)
 manifest={'version':'motion_actuator_paired_audit_v1','input_sha256':inputs,'cases':cases,'max_cases':len(cases),
  'seconds_per_case':args.seconds,'wall_limit_s':args.wall_seconds,'storage_max_bytes':128*2**20,'device':'cpu','PPO_updates':0,
  'authorization':'user explicitly approved simulator-side actuator, paired dynamics, Teacher/Student, handover and high-frequency diagnostics',
  'tail':{'rate_hz':1000,'maximum_seconds':2.,'whole_case_metrics_rate_hz':100},
  'missing_real_state':['true control-owner/event time','executed LQR targets','root velocity/height','real contact','spring residual'],
  'scope':'single-state deterministic sensitivity screen; hypotheses not hardware identification or capability certification'}
 print(json.dumps({'output':str(out),'cases':len(cases),'rig_cases':18*len(args.rig_wheel_kv) if rig_only else 6,'seconds_per_case':args.seconds,'simulation':bool(args.run),'PPO':0}),flush=True)
 if not args.run:return
 import shutil
 if shutil.disk_usage(root).free<384*2**20:raise RuntimeError('diagnostic storage reserve unavailable')
 out.mkdir(parents=True,exist_ok=False)
 def write(name,payload):
  with (out/name).open('x') as f:json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False)
 write('manifest.json',manifest);start=time.monotonic();results=[];rig=[];failure=None;status='FAILED'
 write('budget.json',{'status':'RUNNING','wall_limit_s':args.wall_seconds,'storage_max_bytes':128*2**20,'max_cases':len(cases),'PPO_updates':0})
 try:
  rig=actuator_rig_checks(model,refs,out,start+args.wall_seconds,sensitivity=rig_only,kvs=args.rig_wheel_kv,mechanical_b=args.rig_mechanical_b);write('actuator_rig.json',rig)
  if not all(r['passed'] for r in rig):raise RuntimeError('isolated actuator consistency check failed')
  initializations={}
  with torch.inference_mode():
   for index,case in enumerate(cases):
    cfg=case['configuration'];key=(case['initial'],cfg['event'],cfg['root_vx'])
    readback=apply_actuator_profile(model,refs,cfg,original)
    if key not in initializations:
     event=saved_case['event_sample_time_s'] if cfg['event']=='main' else 17.
     ix=int(np.searchsorted(times,event,side='right')-1);row=decode_log(x[ix]);history=[]
     for j in range(ix-15,ix):
      v=decode_log(x[j]);history.append((int(round((times[j]-times[ix])*1000)),np.r_[v['gyro'],v['gravity'],v['q'],v['dq']]))
     logged={'row':row,'times':times,'observations':x,'event':float(times[ix])}
     if case['initial']=='log':
      initial,fit=initialize_from_log(model,refs,x[ix],cfg['root_vx']);bridge=row['q'].copy();bridge[WHEELS]=row['dq'][WHEELS]
     else:
      initial=mujoco.MjData(model);mujoco.mj_forward(model,initial);ss=sensors(model,initial,refs)
      history=[(-20,ss),(-10,ss)];bridge=np.zeros(6);fit={'source':'MJCF normal zero pose'}
     initializations[key]=(initial,history,bridge,logged,fit)
    initial,history,bridge,logged,fit=initializations[key]
    summary,low,high=audit_rollout(model,refs,actors[case['policy']],initial,history,bridge,logged,case,args.seconds,start+args.wall_seconds)
    name=f"case_{index:03d}__{case['policy']}__{case['initial']}__{cfg['name']}"
    summary.update(case_id=name,policy=case['policy'],profile=cfg['name'],group=case['group'],configuration=cfg,readback=readback,initial_fit=fit,trace=name+'.npz',tail_trace=name+'__tail.npz')
    np.savez_compressed(out/(name+'.npz'),**low);np.savez_compressed(out/(name+'__tail.npz'),**high);write(name+'.json',summary);results.append(summary)
    print(json.dumps({'case':index+1,'total':len(cases),'policy':case['policy'],'profile':cfg['name'],'initial':case['initial'],'termination':summary['termination'],'seconds':summary['simulated_seconds']}),flush=True)
    if sum(p.stat().st_size for p in out.iterdir())>126*2**20:raise RuntimeError('actuator audit storage cap reached')
  for p,digest in inputs.items():
   if sha(p)!=digest:raise RuntimeError('diagnostic input changed during execution: '+p)
  status='COMPLETED'
 except BaseException as exc:
  failure={'type':type(exc).__name__,'message':str(exc)};raise
 finally:
  write('summary.json',{'status':status,'failure':failure,'wall_seconds':time.monotonic()-start,'results':results,'rig_checks':rig,'PPO_updates':0})
  write('terminal_report.json',{'status':status,'failure':failure,'wall_seconds':time.monotonic()-start,'completed_cases':len(results),'planned_cases':len(cases),'PPO_updates':0})
  # Replace only this task's own mutable diagnostic budget, never any training ledger.
  (out/'budget.json').write_text(json.dumps({'status':status,'wall_limit_s':args.wall_seconds,'wall_used_s':time.monotonic()-start,'storage_max_bytes':128*2**20,'completed_cases':len(results),'PPO_updates':0},indent=2)+'\n')
  lines=['# MuJoCo执行器与接管配对诊断','',f'状态：{status}；PPO更新0；每case请求{args.seconds:g}秒。',
   '', '|模型|初态|配置|终态/秒|双轮接触|触限|pitch峰值rad|roll峰值rad|摆角差rad|轮目标Δ RMS|',
   '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
  for r in results:lines.append(f"|{r['policy']}|{r['initialization']}|{r['profile']}|{r['termination']}/{r['simulated_seconds']:.2f}|{r['both_contact_fraction']:.1%}|{r['saturation_any_fraction']:.1%}|{r['pitch_abs_peak_rad']:.5f}|{r['roll_abs_peak_rad']:.5f}|{r['swing_difference_peak_rad']:.5f}|{r['wheel_target_delta_rms_rad_s']:.3f}|")
  lines+=['','同一profile内Teacher/Student可配对；final500是不同训练来源，不能当网络容量单变量。主矩阵命令为zero，logged_command另行回放日志命令且超出日志后保持最后值。',
   '各case于首次故障或窗口结束停止，失败短窗统计不能冒充完整窗口通过。tail为末2秒1kHz；命令100Hz与PD200Hz仍保持，未提高策略帧率。',
   '轮峰值/J/B/C是诊断假设，armature仅加总J减刚体惯量，库仑tanh过渡未辨识；previous错误对照不用于部署。原始spec/模型/资产只读，未改当前训练。']
  (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
