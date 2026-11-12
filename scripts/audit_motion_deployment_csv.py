"""Named Ozone input/ONNX replay; optional bounded existing-Plant MuJoCo diagnostic.

Logged observations are used without normalization/phase/polarity corrections.
Missing task flags and unknown physical handover quantities remain assumptions.
"""
import argparse,csv,importlib.util,json,time
from pathlib import Path
from contextlib import ExitStack
import numpy as np
from onnx.reference import ReferenceEvaluator
from replay_motion31_shadow_csv import read_csv,sha,stats

LEGS=[0,1,3,4];WHEELS=[2,5];ORDER=[0,1,3,4,2,5]
SCALE=np.array([1.5,1.5,100.,1.5,1.5,100.])


def leg_zero_bias(value):
 """Diagnostic encoder offset, q_reported=q_true+b, in four-leg order."""
 b=np.asarray(value,dtype=float)
 if b.shape!=(4,) or not np.isfinite(b).all():raise ValueError('four finite leg zero offsets in rad required')
 return b.copy()


def biased_sensors(sensors,bias):
 """Change only reported active-leg positions, never the physical state/dq."""
 s=np.asarray(sensors,dtype=float)
 if s.shape!=(18,) or not np.isfinite(s).all():raise ValueError('finite native18 sensors required')
 out=s.copy();out[6+np.asarray(LEGS)]+=leg_zero_bias(bias)
 return out


def target_in_reported_coordinates(target,bias):
 """Reset bridge for a miscalibrated position feedback coordinate system."""
 a=np.asarray(target,dtype=float)
 if a.shape!=(6,) or not np.isfinite(a).all():raise ValueError('finite six-channel target required')
 out=a.copy();out[LEGS]+=leg_zero_bias(bias)
 return out


def write(path,value):
 temp=path.with_name(path.name+'.tmp')
 temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def diagnostic_windows(arrays,motor_limits):
 """Fixed causal windows, reporting only. No entry selection or certificate.

 End-of-interval records in (start,end] at100Hz must all be present. Short or
 missing windows retain their denominator and cannot be marked complete.
 """
 t=np.asarray(arrays['time'],dtype=float);limits=np.asarray(motor_limits,dtype=float)
 if limits.shape!=(6,) or not np.isfinite(limits).all() or (limits<=0).any():raise ValueError('six finite motor command limits required')
 if t.ndim!=1 or not len(t) or not np.isfinite(t).all() or (np.diff(t)<=0).any():raise ValueError('strictly increasing finite time required')
 out=[]
 for start,end in ((0.,2.),(2.,10.),(10.,20.),(20.,30.)):
  keep=(t>start+1e-6)&(t<=end+1e-6);stamp=t[keep];need=round((end-start)*100)
  complete=len(stamp)==need and np.allclose(stamp,start+np.arange(1,need+1)*.01,atol=1e-6,rtol=0)
  r={'requested_window_s':[start,end],'expected_samples':need,'samples':len(stamp),
     'status':'COMPLETE' if complete else 'PARTIAL' if len(stamp) else 'NOT_RUN',
     'certificate_eligible':False,'sampled_hz':100,'motor_command_limits_nm':limits.tolist()}
  if len(stamp):
   state=arrays['state'][keep];torque=arrays.get('firmware_motor_nm',arrays['torque'])[keep]
   r.update(actual_first_last_sample_s=[float(stamp[0]),float(stamp[-1])],
            vx_mean_mps=float(state[:,0].mean()),vx_abs_mean_mps=float(abs(state[:,0]).mean()),
            vx_abs_p95_mps=float(np.percentile(abs(state[:,0]),95)),
            pitch_abs_peak_rad=float(abs(state[:,3]).max()),roll_abs_peak_rad=float(abs(state[:,4]).max()),
            height_mean_m=float(state[:,2].mean()),
            both_wheel_contact_fraction=float((arrays['contact'][keep]>2).all(1).mean()),
            motor_command_saturation_per_channel=(abs(torque)>=limits*.99).mean(0).tolist(),
            nominal_5p5_wheel_torque_fraction=(abs(torque[:,WHEELS])>=5.5*.99).mean(0).tolist(),
            motor_command_rms_nm=np.sqrt(np.mean(torque*torque,axis=0)).tolist(),
            motor_command_sign_changes=(np.diff(np.sign(torque),axis=0)!=0).sum(0).tolist())
   r['target_delta_rms']=np.sqrt(np.mean(np.diff(arrays['adopted'][keep],axis=0)**2,axis=0)).tolist() if len(stamp)>1 else None
  out.append(r)
 return out


def replay(source,package,out):
 manifest=json.loads((package/'manifest.json').read_text())
 for name,digest in json.loads((package/'files_sha256.json').read_text()).items():
  if sha(package/name)!=digest:raise ValueError('package SHA differs: '+name)
 if sha(package/'actor.onnx')!=manifest['actor_onnx_sha256']:raise ValueError('ONNX identity differs')
 rows,x,logged,t,cols=read_csv(source);runtime=ReferenceEvaluator(str(package/'actor.onnx'))
 infer=lambda z:np.concatenate([runtime.run(None,{'observation':v[None].astype(np.float32)})[0] for v in z])
 pred=infer(x);active=np.array([float(r['(g_rl_shadow_debug).rl_active']) for r in rows])>0
 if len(t)<3 or np.count_nonzero(active[1:]&active[:-1])<2:raise ValueError('at least three samples with two consecutive active pairs required')
 delta=abs(pred-logged);previous_delta=x[1:,16:22]-logged[:-1][:,ORDER]
 with (out/'replayed.csv').open('x',newline='') as stream:
  w=csv.writer(stream);w.writerow(['Time','rl_active']+[f'obs[{i}]' for i in range(31)]+[f'logged[{i}]' for i in range(6)]+[f'onnx[{i}]' for i in range(6)]+[f'abs_error[{i}]' for i in range(6)]+['onnx_ch2_rad_s','onnx_ch5_rad_s'])
  w.writerows(np.c_[t,active,x,logged,pred,delta,pred[:,WHEELS]*100].tolist())
 windows=[]
 for label,sel in [('active',active),('inactive',~active),('first15s',t<15),('last_active_1s',active&(t>=t[active][-1]-1))]:
  if not sel.any():continue
  a=x[sel];p=pred[sel];y=logged[sel]
  windows.append({'name':label,'samples':int(sel.sum()),'time_s':[float(t[sel][0]),float(t[sel][-1])],
   'onnx_abs_max_error':abs(p-y).max(0).tolist(),'fraction_all_channels_within_1e_5':float((abs(p-y).max(1)<=1e-5).mean()),
   'q_legs_rad':stats(a[:,6:10]*1.5),'wheel_feedback_rad_s':stats(a[:,[12,15]]*20),
   'wheel_requested_rad_s':stats(y[:,WHEELS]*100),'command_physical':stats(a[:,22:25]*[4,4,.5]),
   'leg_request_minus_feedback_rad':stats(y[:,LEGS]*1.5-a[:,6:10]*1.5),
   'tilt_peak_rad':float(np.arccos(np.clip(-a[:,4],-1,1)).max())})
 validpairs=active[1:]&active[:-1]
 q=x[:,6:10]*1.5;dq=x[:,10:16]*20;dt=np.diff(t)
 qrate=np.diff(q,axis=0)/dt[:,None];correlations=[]
 for i in range(4):
  a,b=qrate[validpairs,i],dq[1:,LEGS[i]][validpairs]
  correlations.append(float(np.corrcoef(a,b)[0,1]) if min(a.std(),b.std())>1e-10 else None)
 segments=[]
 for lo,hi in [(0,15),(15,23.5),(23.5,24.32),(24.32,33)]:
  sel=(t>=lo)&(t<hi)
  if not sel.any():continue
  segments.append({'time_s':[lo,hi],'q_leg_peak_to_peak_rad':np.ptp(q[sel],axis=0).tolist(),'gravity_peak_to_peak':np.ptp(x[sel,3:6],axis=0).tolist()})
 report={'CSV':str(source),'CSV_sha256':sha(source),'package':str(package),'ONNX_sha256':sha(package/'actor.onnx'),
  'rows':len(rows),'time_s':[float(t[0]),float(t[-1])],'dt_s':stats(dt),'missing_task_flags_assumed':[1,0,0,1,0,0],
  'task_flag_warning':'obs25:30 not captured; successful replay supports this assumption but does not measure firmware flags',
  'active_edges':[{'time_s':float(t[i]),'active':bool(active[i])} for i in np.r_[0,np.flatnonzero(active[1:]!=active[:-1])+1]],
  'windows':windows,'segments':segments,'gravity_norm':stats(np.linalg.norm(x[:,3:6],axis=1)),
  'previous_vs_last_prediction_active':{'max_abs':abs(previous_delta[validpairs]).max(0).tolist(),'match_fraction':float((abs(previous_delta[validpairs]).max(1)<1e-5).mean()),'meaning':'matches requested prediction, not evidence of PD adoption'},
  'leg_q_derivative_vs_dq_correlation':correlations,'actor_gyro_rad_s':stats(x[:,:3]*4),
  'hardware_unobserved':['actual control owner beyond rl_active flag','PD-adopted physical targets','requested/clipped/applied motor torque','motor Kp/Kd/Kv and compensation','encoder calibration and motor polarity','actual root height/velocity/contact','IMU/encoder/PD timestamps'],
  'optimizer_steps':0,'replay_is_closed_loop':False}
 np.savez_compressed(out/'replay.npz',time=t,active=active,actor_input=x,logged=logged,replayed=pred)
 write(out/'replay_summary.json',report)
 return report,(x,t,active),infer,manifest


def simulate(adapter,package,out,x,t,active,infer,manifest,wall_s,*,scenarios=None,logged_actions=None,record_full_state=False):
 with ExitStack() as stack:
  return _simulate(adapter,package,out,x,t,active,infer,manifest,wall_s,scenarios=scenarios,
                   logged_actions=logged_actions,record_full_state=record_full_state,stack=stack)


def _simulate(adapter,package,out,x,t,active,infer,manifest,wall_s,*,scenarios,logged_actions,record_full_state,stack):
 # Reuse the exact existing current-plant adapter, not legacy native MJCF defaults.
 spec=importlib.util.spec_from_file_location('deployment_diagnostic_plant',adapter);task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task)
 request=json.loads((adapter.parent/'request.json').read_text());_,_,config=task.inputs(request)
 for key,source in [('student_sha256',manifest['student']),('teacher_sha256',request['teacher']),('spec_sha256',request['parent_spec'])]:
  if sha(source)!=manifest[key]:raise ValueError('frozen Plant/model lineage differs: '+key)
 plant=task.Plant(request,config);task.torch.set_num_threads(4)
 from native_stand.students.real2sim_handover import initialize_from_log,decode_log
 from native_stand.students.real2sim_delay import ObservationHistory
 from compare_motion_timing_profiles import AuditExecution
 case={'target':[0.,0.,.34],'dr_group':0,'env_id':2};deadline=time.monotonic()+wall_s;results=[]
 cases=scenarios or [(name,ix,cmd,cache,seconds,{}) for name,ix,cmd,cache,seconds in [('normal_zero',None,'zero','bridge',30.),('normal_logged_commands',None,'logged','bridge',24.32),('log_first_bridge',0,'logged','bridge',10.),('log_first_previous',0,'logged','logged',10.),('log_pre_exit_bridge',int(np.searchsorted(t,24.0)),'zero','bridge',5.)]]
 for name,ix,command_mode,cache_mode,seconds,overrides in cases:
  actor_bias=leg_zero_bias(overrides.get('actor_leg_zero_bias_rad',[0]*4))
  pd_bias=leg_zero_bias(overrides.get('pd_leg_zero_bias_rad',[0]*4))
  actor_biased=bool(actor_bias.any());pd_biased=bool(pd_bias.any())
  if (actor_biased or pd_biased) and overrides.get('firmware_library'):
   raise ValueError('zero-offset diagnostic uses the declared reference executor, not an implicit firmware calibration change')
  d,ex,history,params,readback=plant.reset(case,request['seeds'][1]);fit=None
  cfg=dict(readback['configuration']);cfg.update({k:v for k,v in overrides.items() if k in ('kp','kd','kv','transport_ms','feedback_ms','tau_ms')})
  params=params.copy()
  if 'actor_age_ms' in overrides:params[0]=overrides['actor_age_ms']
  readback={**readback,'effective_diagnostic_configuration':cfg,'effective_actor_age_ms':int(params[0]),'overrides':overrides}
  if ix is not None:
   d,fit=initialize_from_log(plant.model,plant.refs,x[ix],overrides.get('root_vx_mps',0.)) # unknown real vx, explicit hypothesis
   initial=task.sensors(plant.model,d,plant.refs);bridge=initial[6:12].copy();bridge[WHEELS]=initial[12:18][WHEELS]
   if cache_mode in ('logged','shadow'):bridge=decode_log(x[ix])['previous']
   ex=AuditExecution(cfg,bridge,[(-32,initial),(0,initial)]);history=ObservationHistory(initial,initial_tick=-32)
  elif overrides:
   initial=task.sensors(plant.model,d,plant.refs);ex=AuditExecution(cfg,ex.adopted.copy(),[(-32,initial),(0,initial)])
  if pd_biased:
   initial=task.sensors(plant.model,d,plant.refs)
   # Bridge targets come from actual initialization in the same reported
   # coordinates. Logged previous, if requested, is already reported data.
   bridge=target_in_reported_coordinates(ex.adopted,pd_bias) if cache_mode=='bridge' else ex.adopted.copy()
   feedback=biased_sensors(initial,pd_bias)
   ex=AuditExecution(cfg,bridge,[(-32,feedback),(0,feedback)])
  if overrides.get('firmware_library'):
   from native_stand.students.firmware_control import FirmwareExecution
   ex=FirmwareExecution(overrides['firmware_library'],ex.adopted,overrides['spring_mode'],
                        motor_delay_ms=overrides.get('firmware_motor_delay_ms'),motor_tau_ms=overrides.get('firmware_motor_tau_ms'))
   stack.callback(ex.close)
   readback['firmware_effective_execution']={
    'library':str(overrides['firmware_library']),'library_sha256':sha(overrides['firmware_library']),
    'policy_period_ms':10,'pd_period_ms':5,'pd_feedback_age_ms':int(cfg['feedback_ms']),
    'actor_age_ms':int(params[0]),'local_target_transport_ms':0,
    'motor_torque_transport_ms_ch0_ch5':ex.motor_delay_ms.tolist(),'motor_response_tau_ms_ch0_ch5':ex.motor_tau_ms.tolist(),
    'scope':'local host-ABI timing plus explicit post-PD torque transport/response hypotheses; original target transport/tau are NOT applied in this branch',
    'spring_hypothesis':overrides['spring_mode']}
  requested=ex.adopted.copy();arrays={};failure=None;first=None;encoder_error=0.
  if record_full_state:initial_qpos=d.qpos.copy();initial_qvel=d.qvel.copy()
  for tick in range(round(seconds*1000)):
   if tick%100==0 and time.monotonic()>deadline:raise TimeoutError('bounded MuJoCo audit deadline')
   current=task.sensors(plant.model,d,plant.refs);history.append(tick,current)
   if tick%10==0:
    sensed=history.at(tick-int(params[0]));cache=requested.copy() if cache_mode=='shadow' else ex.adopted.copy();query=(float(t[ix]) if ix is not None else 0)+tick*.001
    if actor_biased:sensed=biased_sensors(sensed,actor_bias)
    li=min(len(t)-1,max(0,int(np.searchsorted(t,query,side='right')-1)))
    command=np.array([0.,0.,.34]) if command_mode=='zero' else x[ix if command_mode=='fixed' else li,22:25]*[4,4,.5]
    obs=task.encode(*[task.torch.tensor(z,dtype=task.torch.float32)[None] for z in (sensed[:3],sensed[3:6],sensed[6:12],sensed[12:18],cache[LEGS],cache[WHEELS])],plant.reference,'Motion',task.torch.tensor(command,dtype=task.torch.float32)[None])
    if overrides.get('firmware_library'):
     c_obs=task.torch.tensor(ex.observe(tick,tick-int(params[0]),sensed,command))[None]
     encoder_error=max(encoder_error,float((obs-c_obs).abs().max()))
     if encoder_error>1e-6:raise ValueError('same-state C/Python Actor encoder differs')
     obs=c_obs
    if overrides.get('action_source')=='logged':
     if logged_actions is None:raise ValueError('recorded action sequence required')
     mean=logged_actions[li].copy()
    else:mean=infer(obs.numpy())[0]
    if overrides.get('firmware_library'):ex.publish(tick,mean)
    target=task.targets(task.torch.tensor(mean)[None],plant.reference,plant.bounds)
    requested[LEGS]=target['leg_executed'][0].numpy();requested[WHEELS]=target['wheel_executed'][0].numpy()
    if first is None:first={'input':obs[0].tolist(),'mean':mean.tolist(),'q':current[6:12].tolist(),'target':requested.tolist()}
   if overrides.get('firmware_library'):
    pd_age=int(cfg['feedback_ms']);pd_sensors=history.at(tick-pd_age)
    torque=ex.step(tick,tick-pd_age,pd_sensors,command)
   else:
    feedback=biased_sensors(current,pd_bias) if pd_biased else current
    torque=ex.step(tick,requested,feedback[6:12],feedback[12:18])
   d.ctrl[plant.refs['actuators']]=torque/plant.refs['gears'];task.mujoco.mj_step(plant.model,d)
   if tick%10==9:
    task.mujoco.mj_forward(plant.model,d);_,_,state=task.state(plant.model,d,plant.refs)
    record={'time':np.array((tick+1)*.001),'actor_input':obs[0].numpy(),'mean':mean,'requested':requested.copy(),'adopted':ex.adopted.copy(),'q':task.sensors(plant.model,d,plant.refs)[6:12],'dq':d.qvel[plant.refs['dofs']].copy(),'torque':torque.copy(),'preclip':ex.pre.copy(),'state':np.array([state['vx'],state['yaw_rate'],state['height'],state['pitch'],state['roll'],state['closure_m']]),'contact':np.array(state['contact_force_n'])}
    if actor_biased or pd_biased:
     pd_used=ex.history.at(tick-tick%5-int(cfg['feedback_ms']))
     record.update(actor_feedback_q_rad=sensed[6:12],pd_feedback_q_rad=pd_used[:6],
                   pd_feedback_dq_rad_s=pd_used[6:],physical_equivalent_adopted=target_in_reported_coordinates(ex.adopted,-pd_bias))
    if overrides.get('firmware_library'):
     record.update(firmware_motor_nm=ex.last['motor_model_nm'],
                   spring_compensation_model_nm=ex.last['spring_compensation_model_nm'],
                   pd_limited_nm=ex.last['pd_limited_nm'],firmware_metadata=ex.last['metadata'],
                   pd_feedback_native16=ex.pd_feedback_native16,
                   simulated_motor_response_nm=ex.motor_actual if ex.motor_physics_enabled else ex.last['motor_model_nm'])
    if record_full_state:record.update(full_qpos=d.qpos.copy(),full_qvel=d.qvel.copy())
    for k,v in record.items():arrays.setdefault(k,[]).append(np.array(v,copy=True))
    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError('NONFINITE')
    if state['closure_m']>.005:failure='CLOSURE'
    elif state['up']<.5:failure='UPRIGHT'
    if failure:break
  a={k:np.stack(v) for k,v in arrays.items()}
  if record_full_state:a.update(replay_time=np.r_[0.,a['time']],replay_qpos=np.vstack([initial_qpos,a['full_qpos']]),replay_qvel=np.vstack([initial_qvel,a['full_qvel']]))
  trace=out/(name+'.npz');temp=trace.with_name(trace.name+'.tmp')
  with temp.open('xb') as stream:np.savez_compressed(stream,**a)
  temp.replace(trace)
  hold=a['time']>=min(2.,a['time'][-1]/2)
  result={'case':name,'seconds':float(a['time'][-1]),'failure':failure,'initial_log_time_s':None if ix is None else float(t[ix]),'root_vx_assumed_mps':None if ix is None else overrides.get('root_vx_mps',0.),'action_source':overrides.get('action_source','ONNX_closed_loop'),'previous_mode':cache_mode,'command_mode':command_mode,'initial_fit':fit,'readback':readback,'first':first,'vx_mean':float(a['state'][hold,0].mean()),'pitch_abs_peak':float(abs(a['state'][:,3]).max()),'roll_abs_peak':float(abs(a['state'][:,4]).max()),'leg_target_step_RMS_rad':np.sqrt(np.mean(np.diff(a['requested'][:,LEGS],axis=0)**2,axis=0)).tolist(),'leg_dq_abs_P95_rad_s':np.percentile(abs(a['dq'][:,LEGS]),95,axis=0).tolist(),'wheel_target_rad_s':stats(a['requested'][hold][:,WHEELS]),'contact_fraction':float((a['contact'][hold]>2).all(1).mean()),'torque_limit_any_fraction':float((abs(a['torque'][hold])>=np.array([40,40,5.5,40,40,5.5])*.99).any(1).mean()),'scope':'same exported model lineage and existing simulated plant; any action replay is OPEN LOOP, not ONNX feedback; log initial velocity/contact/passive state and overrides are hypotheses, not reproduced hardware state'}
  if overrides.get('firmware_library'):
   result['firmware_execution_scope']='S1 C control core 200Hz with explicit receipt-age/transport/torque-response hypotheses; no LQR physically simulated'
   result['spring_hypothesis']=overrides['spring_mode']
   result['not_identified_hardware']=True
   result['effective_wheel_cap_nominal_nm']=8192/(19/.00036/(268/17))
   result['same_state_C_Python_encoder_max_error']=encoder_error
   # Recompute against the exact pre-PD q/dq, not the action's post-physics state.
   feedback=a['pd_feedback_native16'];expected=np.zeros_like(a['preclip'])
   expected[:,LEGS]=40*(a['adopted'][:,LEGS]-feedback[:,6:10])-2*feedback[:,10:16][:,LEGS]
   expected[:,WHEELS]=.2*(a['adopted'][:,WHEELS]-feedback[:,10:16][:,WHEELS])
   result['same_state_C_Python_PD_request_max_error_nm']=np.max(abs(a['preclip']-expected),axis=0).tolist()
  limits=np.array([40,40,5.5,40,40,5.5],dtype=float)
  if overrides.get('firmware_library'):limits[WHEELS]=result['effective_wheel_cap_nominal_nm']
  result['fixed_diagnostic_windows']=diagnostic_windows(a,limits)
  result['window_scope']='fixed100Hz sampled commands/state; not200Hz PD spectrum or policy acceptance, missing windows stay incomplete'
  write(out/(name+'.json'),result);results.append(result)
  if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>127*2**20:raise RuntimeError('bounded diagnostic storage reserve reached')
 write(out/'mujoco_summary.json',results);return results


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--plant-adapter',type=Path);p.add_argument('--run-mujoco',action='store_true');p.add_argument('--wall-seconds',type=float,default=300.)
 a=p.parse_args()
 if not 0<a.wall_seconds<=600:raise ValueError('wall0..600 required')
 if a.run_mujoco and not a.plant_adapter:raise ValueError('explicit frozen Plant adapter required')
 a.output_dir.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 report,data,infer,manifest=replay(a.csv,a.package,a.output_dir)
 results=simulate(a.plant_adapter,a.package,a.output_dir,*data,infer,manifest,a.wall_seconds) if a.run_mujoco else []
 write(a.output_dir/'terminal.json',{'status':'COMPLETED','wall_s':time.monotonic()-start,'physical_cases':len(results),'online_PPO':0,'optimizer_steps':0,'CSV_SHA_unchanged':sha(a.csv)==report['CSV_sha256'],'ONNX_SHA_unchanged':sha(a.package/'actor.onnx')==report['ONNX_sha256']})
 print(json.dumps({'output':str(a.output_dir),'active_replay':report['windows'][0],'simulation':[{k:r[k] for k in ['case','seconds','failure','vx_mean','pitch_abs_peak','contact_fraction']} for r in results]},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
