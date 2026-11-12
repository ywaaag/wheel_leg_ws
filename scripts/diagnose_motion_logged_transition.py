"""Bounded ONNX/log-state MuJoCo contrasts; no training or hardware execution.

The event time is an explicit hypothesis, not inferred controller ownership.
Uses the existing current-Plant and execution adapter; never teleports after reset.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from onnx.reference import ReferenceEvaluator
from replay_motion31_shadow_csv import read_csv,sha
from audit_motion_deployment_csv import simulate,write


def scenarios(index,seconds):
 return [('normal_zero',None,'zero','bridge',seconds,{}),
  ('log_nominal_bridge',index,'logged','bridge',seconds,{}),
  ('log_fixed_command',index,'fixed','bridge',seconds,{}),
  ('log_nominal_logged_previous',index,'logged','logged',seconds,{}),
  ('log_shadow_previous',index,'logged','shadow',seconds,{}),
  ('log_legacy_gains',index,'logged','logged',seconds,{'kp':60.,'kv':1.}),
  ('log_zero_latency',index,'logged','logged',seconds,{'transport_ms':[0]*6,'feedback_ms':0,'tau_ms':[0]*6,'actor_age_ms':0}),
  ('log_root_vx_minus_point3',index,'logged','logged',seconds,{'root_vx_mps':-.3}),
  ('recorded_actions_open_loop',index,'logged','logged',1.33,{'action_source':'logged'})]


def firmware_scenarios(index,seconds,library,timing_screen=False):
 """Declared computer-only hypotheses, never firmware or hardware measurements."""
 if timing_screen:
  # Same zero command for every state/profile; no replay beyond the source log.
  families=[('normal',None,'zero'),('log',index,'zero')]
  profiles=[('age0_pd0',0,0),('age10_pd0',10,0),('age0_pd10',0,10),
            ('age10_pd10',10,10),('age20_pd20',20,20)]
 else:
  families=[('normal',None,'zero'),('log',index,'logged')]
  profiles=[('age10_pd10',10,10)]
 cases=[('reference_'+name,ix,cmd,'bridge',seconds,{}) for name,ix,cmd in families]
 for mode in ('motor_only','ideal_cancellation'):
  selected=profiles if mode=='ideal_cancellation' else [p for p in profiles if p[1:]==(10,10)]
  for label,actor_age,pd_age in selected:
   for family,ix,cmd in families:
    name=f'firmware_{mode}_{family}'+('_'+label if timing_screen else '')
    cases.append((name,ix,cmd,'owner_adopted',seconds,
                  dict(firmware_library=str(library.resolve()),spring_mode=mode,
                       actor_age_ms=actor_age,feedback_ms=pd_age)))
 return cases


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--plant-adapter',type=Path,required=True);p.add_argument('--event-time',type=float,required=True);p.add_argument('--seconds',type=float,default=6.);p.add_argument('--wall-seconds',type=float,default=180.);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--prepare-replay',action='store_true',help='two-case recorded-action/ONNX physical comparison with full qpos for existing GUI replay');p.add_argument('--firmware-library',type=Path);p.add_argument('--firmware-timing-screen',action='store_true',help='with firmware-library: fixed14-case zero-command computer-only timing hypotheses, up to30s each');p.add_argument('--firmware-motor-delay-ms',type=int,nargs=6,help='explicit torque-command transport AFTER local C PD adoption, model order,0..100ms');p.add_argument('--firmware-motor-tau-ms',type=float,nargs=6,help='explicit motor torque rise time after transport,0..100ms; not feedback age');p.add_argument('--run',action='store_true');args=p.parse_args()
 if args.firmware_timing_screen and (not args.firmware_library or args.prepare_replay):raise ValueError('firmware timing screen requires its own C-library diagnostic')
 if (args.firmware_motor_delay_ms is not None or args.firmware_motor_tau_ms is not None) and not args.firmware_library:raise ValueError('post-PD motor timing requires firmware-library')
 for name in ('firmware_motor_delay_ms','firmware_motor_tau_ms'):
  values=getattr(args,name)
  if values is not None and (not np.isfinite(values).all() or any(v<0 or v>100 for v in values)):raise ValueError(name+' needs six finite values0..100ms')
 maximum=30 if args.firmware_timing_screen else 10
 if not 2<=args.seconds<=maximum or not 0<args.wall_seconds<=600:raise ValueError(f'finite seconds2..{maximum}/wall0..600 required')
 rows,x,y,t,columns=read_csv(args.csv);index=int(np.searchsorted(t,args.event_time))
 if index>=len(t) or (not args.firmware_timing_screen and t[index]+args.seconds>t[-1]):raise ValueError('event interval outside recorded timeline')
 if (abs(x)>100).any() or (abs(y)>1.0001).any():raise ValueError('impossible logged values: do not silently reconstruct')
 manifest=json.loads((args.package/'manifest.json').read_text())
 if sha(args.package/'actor.onnx')!=manifest['actor_onnx_sha256']:raise ValueError('ONNX SHA differs')
 cases=scenarios(index,args.seconds)
 if args.prepare_replay and args.firmware_library:raise ValueError('separate diagnostic identities; do not mix legacy replay and firmware contrast')
 if args.prepare_replay:cases=[c for c in cases if c[0] in ('recorded_actions_open_loop','log_nominal_logged_previous')]
 if args.firmware_library:
  validation=args.firmware_library.parent/'validation.json'
  v=json.loads(validation.read_text())
  if v['status']!='CPU_PASS' or sha(args.firmware_library)!=v['library_sha256']:raise ValueError('validated firmware library required')
  if any(sha(p)!=digest for p,digest in v['source_sha256'].items()):raise ValueError('firmware source changed since CPU validation')
  cases=firmware_scenarios(index,args.seconds,args.firmware_library,args.firmware_timing_screen)
  for case in cases:
   if not case[5].get('firmware_library'):continue
   if args.firmware_motor_delay_ms is not None:case[5]['firmware_motor_delay_ms']=args.firmware_motor_delay_ms
   if args.firmware_motor_tau_ms is not None:case[5]['firmware_motor_tau_ms']=args.firmware_motor_tau_ms
 plan={'status':'PLAN_ONLY','cases':cases,'event_time_s':float(t[index]),'actor_task_flags_assumed':[1,0,0,1,0,0],'wall_limit_s':args.wall_seconds,'storage_limit_bytes':128*2**20,'PPO':0,'optimizer':0,'newton_or_Isaac':False,'simulation':0,'firmware_timing_screen':args.firmware_timing_screen,'hardware_parameters_identified':False}
 if not args.run:print(json.dumps(plan,indent=2));return
 args.output_dir.mkdir(parents=True,exist_ok=False);start=time.monotonic();inputs={str(p):sha(p) for p in [args.csv,args.package/'actor.onnx',args.package/'manifest.json',args.plant_adapter,Path(__file__),Path('scripts/audit_motion_deployment_csv.py')] }
 if args.firmware_library:
  extra=[args.firmware_library,args.firmware_library.parent/'validation.json',Path('src/native_stand/students/firmware_control.py')]
  inputs.update({str(path):sha(path) for path in extra})
 write(args.output_dir/'request.json',{**plan,'status':'AUTHORIZED_BOUNDED_DIAGNOSTIC','authorization':'User authorizes execution alignment and computer-only tests/simulation while hardware is unavailable; finite declared hypotheses, no PPO/hardware','inputs_sha256':inputs})
 try:
  if args.firmware_library:
   write(args.output_dir/'execution_manifest.json',dict(version='motion_firmware_timing_screen_v1' if args.firmware_timing_screen else 'motion_firmware_contrast_v1',library_validation=v,storage_limit_bytes=128*2**20,maximum_cases=len(cases),maximum_physics_seconds=sum(c[4] for c in cases),spring_identified=False,hardware_record='NOT_AVAILABLE',PPO=0))
  runtime=ReferenceEvaluator(str(args.package/'actor.onnx'))
  infer=lambda z:np.concatenate([runtime.run(None,{'observation':a[None].astype(np.float32)})[0] for a in z])
  pred=infer(x);error=abs(pred-y).max(1);matching=error<=1e-5
  edges=np.r_[0,np.flatnonzero(np.diff(matching.astype(int))!=0)+1,len(t)]
  segments=[{'from_s':float(t[l]),'to_s':float(t[r-1]),'rows':int(r-l),'matches_ONNX':bool(matching[l])} for l,r in zip(edges[:-1],edges[1:])]
  mask=(t>=10.13)&(t<=16.4604)
  data={'CSV':str(args.csv),'CSV_sha256':sha(args.csv),'samples':len(t),'time_s':[float(t[0]),float(t[-1])],'dt_quantiles_s':np.percentile(np.diff(t),[0,50,100]).tolist(),'output_match_segments':segments,'matched_rows':int(matching.sum()),'matched_error_max':float(abs(pred[matching]-y[matching]).max()),'command_physical_range':[np.min(x[mask,22:25]*[4,4,.5],axis=0).tolist(),np.max(x[mask,22:25]*[4,4,.5],axis=0).tolist()],'switch_time':'UNKNOWN: no controller owner, enable or actual torque fields','missing':['controller_owner','actual PD gains','requested/clipped/applied torque','gas compensation and source','target adoption and feedback timestamps','root linear velocity, actual height and contact','obs25:30 task flags']}
  pair=matching[1:]&matching[:-1]&(np.diff(t)<.015);pd=x[1:,16:22]-y[:-1][:,[0,1,3,4,2,5]]
  data['previous_vs_last_prediction']={'fraction_exact_1e5':float((abs(pd[pair]).max(1)<1e-5).mean()),'scope':'not actual PD-adopted evidence'}
  windows=[]
  for name,lo,hi in [('LQR_late',14.13,15.13),('oscillation',15.1499,16.4604)]:
   sel=(t>=lo)&(t<hi);adj=sel[1:]&sel[:-1]&(np.diff(t)<.015);leg=y[:,[0,1,3,4]].astype(float)*1.5;wheel=y[:,[2,5]].astype(float)*100
   windows.append({'name':name,'time_s':[lo,hi],'rows':int(sel.sum()),'leg_target_step_RMS_rad':np.sqrt(np.mean(np.diff(leg,axis=0)[adj]**2,axis=0)).tolist(),'wheel_target_step_RMS_rad_s':np.sqrt(np.mean(np.diff(wheel,axis=0)[adj]**2,axis=0)).tolist(),'leg_dq_P95_rad_s':np.percentile(abs(x[sel,10:16][:,[0,1,3,4]]*20),95,axis=0).tolist(),'tilt_peak_rad':float(np.arccos(np.clip(-x[sel,4],-1,1)).max())})
  data['windows']=windows;write(args.output_dir/'log_summary.json',data)
  with (args.output_dir/'replayed.csv').open('x') as f:
   import csv
   w=csv.writer(f);w.writerow(['Time','matches_same_ONNX']+[f'logged[{j}]' for j in range(6)]+[f'onnx[{j}]' for j in range(6)])
   w.writerows(np.c_[t,matching,y,pred].tolist())
  results=simulate(args.plant_adapter,args.package,args.output_dir,x,t,matching,infer,manifest,args.wall_seconds-(time.monotonic()-start),scenarios=cases,logged_actions=y,record_full_state=args.prepare_replay)
  if args.prepare_replay:
   request=json.loads((args.plant_adapter.parent/'request.json').read_text());clips=['recorded_actions_open_loop','log_nominal_logged_previous']
   write(args.output_dir/'manifest.json',{'version':'motion31_deployment_log_visual_replay_v1','xml':request['xml'],'xml_sha256':sha(request['xml']),'clips':clips,'clip_sha256':{name:sha(args.output_dir/(name+'.npz')) for name in clips},'CSV_sha256':sha(args.csv),'actor_sha256':sha(args.package/'actor.onnx'),'event_time_s':float(t[index]),'semantics':'stored full qpos from real MuJoCo PD rollouts; first clip recorded-actions OPEN LOOP, second autonomous ONNX CLOSED LOOP; viewer only mj_forward, not a hardware trajectory reconstruction','configuration':manifest['control']})
  write(args.output_dir/'terminal.json',{'status':'COMPLETED','wall_s':time.monotonic()-start,'simulated_seconds':sum(r['seconds'] for r in results),'cases':len(results),'optimizer':0,'PPO':0,'inputs_unchanged':all(sha(p)==digest for p,digest in inputs.items())})
  print(json.dumps([{k:r[k] for k in ['case','seconds','failure','pitch_abs_peak','roll_abs_peak','contact_fraction','leg_target_step_RMS_rad']} for r in results],indent=2))
 except Exception as exc:
  write(args.output_dir/'terminal.json',{'status':'FAILED','error':repr(exc),'wall_s':time.monotonic()-start,'automatic_retry':False,'PPO':0});raise


if __name__=='__main__':main()
