"""Frozen Motion actors, same fixed sensor/execution delays, MuJoCo GUI."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import mujoco
from native_stand.students.data import sha
from native_stand.students.model import load
from native_stand.students.mcu_interface import encode
from native_stand.students.interface import targets
from native_stand.students.real2sim_delay import ObservationHistory,TargetTransport,torque_response
from validate_native_mujoco import binding,state
from diagnose_motion_real2sim import sensors


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for role in ('before','after'):
  p.add_argument('--'+role+'-spec',type=Path,required=True);p.add_argument('--'+role+'-checkpoint',type=Path,required=True)
 p.add_argument('--original-spec',type=Path,help='optional original4750 source spec')
 p.add_argument('--original-checkpoint',type=Path,help='optional original4750 checkpoint')
 for role in ('legacy-small','legacy-teacher'):
  p.add_argument('--'+role+'-spec',type=Path)
  p.add_argument('--'+role+'-checkpoint',type=Path)
 p.add_argument('--leg-armature',type=float,help='explicit diagnostic override .01..35 kg m2 on four active leg joints');p.add_argument('--execution-profile',choices=('current','historical-fall','arm011-current','arm011-old'),default='current',help='historical-fall uses leg armature.01/Kp60 and no Actor sensor delay')
 p.add_argument('--handover-search',type=Path,help='existing search manifest supplies real CSV and case initialization')
 p.add_argument('--handover-case',default='case_039')
 p.add_argument('--xml',type=Path,default=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml'))
 p.add_argument('--wall-seconds',type=float,default=1800);p.add_argument('--run',action='store_true');a=p.parse_args()
 if not 0<a.wall_seconds<=1800:raise ValueError('wall bound0..1800 required')
 if bool(a.original_spec)!=bool(a.original_checkpoint):p.error('original-spec and original-checkpoint must be provided together')
 for role in ('legacy_small','legacy_teacher'):
  if bool(getattr(a,role+'_spec'))!=bool(getattr(a,role+'_checkpoint')):p.error(role+' spec/checkpoint must be provided together')
 if a.legacy_small_checkpoint and not a.original_checkpoint:p.error('legacy-small key4 requires original key3')
 if a.legacy_teacher_checkpoint and not a.legacy_small_checkpoint:p.error('legacy-teacher key5 requires legacy-small key4')
 roles=['before','after']+(['original'] if a.original_checkpoint else [])+(['legacy_small'] if a.legacy_small_checkpoint else [])+(['legacy_teacher'] if a.legacy_teacher_checkpoint else [])
 labels=['1 BEFORE526','2 AFTER250']+(['3 ORIGINAL4750'] if a.original_checkpoint else [])+(['4 OLD3250_SMALL'] if a.legacy_small_checkpoint else [])+(['5 OLD3250_TEACHER'] if a.legacy_teacher_checkpoint else [])
 torch.set_num_threads(1);actors=[];asset_sha=None
 for role in roles:
  path=getattr(a,role+'_spec');cp=getattr(a,role+'_checkpoint');spec=json.loads(path.read_text());actor,saved=load(cp,'MotionZeroStudent')
  if saved.get('distillation'):
   from native_stand.students.motion_distillation_pipeline import validate_student
   lineage=validate_student(saved,path)
   if role!='legacy_small':raise ValueError('distilled Actor must be explicitly legacy-small')
   if a.legacy_teacher_checkpoint and lineage['teacher_sha256']!=sha(a.legacy_teacher_checkpoint):raise ValueError('small/large3250 teacher identity differs')
  elif saved['spec_sha256']!=sha(path) or saved['asset_sha256']!=spec['asset_sha256'] or saved['critic_privileged_version']!=spec['student_training']['critic_privileged_version']:raise ValueError('checkpoint/spec identity differs')
  if spec['student_training']['critic_privileged_version'] not in ('motion_timing_privileged110_v1','motion_symmetry_privileged85_v1'):raise ValueError('explicit timing110/symmetry85 source required')
  if asset_sha is not None and asset_sha!=spec['asset_sha256']:raise ValueError('comparison source assets differ')
  asset_sha=spec['asset_sha256']
  actor.eval();actors.append(actor)
 model=mujoco.MjModel.from_xml_path(str(a.xml));refs=binding(model);legs=[0,1,3,4];wheels=[2,5]
 if not np.isclose(model.opt.timestep,.001):raise ValueError('1ms physics required')
 historical=a.execution_profile in ('historical-fall','arm011-old')
 armature=.11 if a.execution_profile.startswith('arm011-') else .01 if historical else .35
 if a.leg_armature is not None:
  if not .01<=a.leg_armature<=.35:raise ValueError('diagnostic leg armature must be within .01..35')
  armature=a.leg_armature
 kp=60. if historical else 40.;sensor_delay=0 if historical else 10
 if historical and a.handover_search is None:p.error('historical-fall requires --handover-search')
 model.dof_armature[np.asarray(refs['dofs'])[legs]]=armature
 log_initial=None;log_history=[];packet_seed=None;bridge=None;log_times=log_obs=None;event_time=None;fit=None
 if a.handover_search:
  from replay_motion31_shadow_csv import read_csv
  from native_stand.students.real2sim_handover import initialize_from_log,decode_log
  meta=json.loads((a.handover_search/'manifest.json').read_text());case=json.loads((a.handover_search/(a.handover_case+'.json')).read_text())
  sources=[Path(p) for p in meta['input_sha256'] if p.endswith('.csv')]
  if len(sources)!=1 or sha(sources[0])!=meta['input_sha256'][str(sources[0])]:raise ValueError('source log identity differs')
  _,log_obs,_,log_times,_=read_csv(sources[0]);event_time=case['event_sample_time_s'];ix=int(np.argmin(abs(log_times-event_time)))
  if ix<15 or abs(log_times[ix]-event_time)>1e-6:raise ValueError('event frame/prehistory absent')
  log_initial,fit=initialize_from_log(model,refs,log_obs[ix],case['assumed_root_vx_mps'])
  for j in range(ix-15,ix):
   v=decode_log(log_obs[j]);log_history.append((int(round((log_times[j]-event_time)*1000)),np.r_[v['gyro'],v['gravity'],v['q'],v['dq']]))
  v=decode_log(log_obs[ix]);bridge=v['q'].copy();bridge[wheels]=v['dq'][wheels]
  packet_seed=log_history[-1][1].copy()
  print(json.dumps({'initialization':'real log constrained handover','event_time_s':event_time,'fit':fit,
   'previous':'assumed bridge: measured leg q and wheel dq; later PD adopted targets; not reconstructed LQR target',
   'command':'logged command playback only; future sensors never replayed'},ensure_ascii=False),flush=True)
 print(json.dumps({'actors':[str(getattr(a,role+'_checkpoint')) for role in roles],'execution_profile':a.execution_profile,'leg_armature':armature,'kp':kp,'kd':2,
  'Actor_packet_ms':sensor_delay,'PD_feedback_ms':10,'target_ms':[10,10,7,10,10,11],'tau_ms':[5,5,0,5,5,0],
  'keys':';'.join(labels)+';R reset;Space pause','simulation':int(a.run)},ensure_ascii=False),flush=True)
 if not a.run:return
 from mujoco import viewer as mjviewer
 import threading
 lock=threading.Lock();control={'which':0,'reset':True,'paused':False}
 def key(k):
  with lock:
   if 49<=k<49+len(actors):control.update(which=k-49,reset=True)
   elif k in (82,114):control['reset']=True
   elif k==32:control['paused']=not control['paused']
 data=mujoco.MjData(model);limit=np.array([40,40,5.5,40,40,5.5]);bounds=torch.tensor(model.jnt_range[refs['joints']],dtype=torch.float32)[None]
 for i,j in enumerate(refs['joints']):
  if not model.jnt_limited[j]:bounds[0,i]=torch.tensor([-float('inf'),float('inf')])
 reference=torch.zeros(1,6);command=torch.tensor([[0.,0.,.34]]);tick=0;started=time.monotonic()
 with mjviewer.launch_passive(model,data,key_callback=key) as viewer,torch.inference_mode():
  while viewer.is_running() and time.monotonic()-started<a.wall_seconds:
   frame=time.monotonic()
   with lock:which=control['which'];reset=control['reset'];control['reset']=False;paused=control['paused']
   if reset:
    mujoco.mj_resetData(model,data)
    if log_initial is not None:data.qpos[:]=log_initial.qpos;data.qvel[:]=log_initial.qvel
    mujoco.mj_forward(model,data);tick=0
    adopted=np.zeros(6) if bridge is None else bridge.copy();requested=adopted.copy();actual=np.zeros(6);demand=actual.copy()
    if log_history:
     history=ObservationHistory(log_history[0][1],initial_tick=log_history[0][0])
     for stamp,sample in log_history:history.append(stamp,sample)
    else:history=ObservationHistory(sensors(model,data,refs))
    transport=TargetTransport([10,10,7,10,10,11],adopted)
    packet=None if packet_seed is None else packet_seed.copy();contact_count=0;sample_count=0;print('Showing '+labels[which],flush=True)
   if not paused:
    for sub in range(10):
     current=sensors(model,data,refs);history.append(tick,current)
     if tick%10==0:
      sensed=current if sensor_delay==0 or packet is None else packet;packet=current.copy()
      if log_times is not None:
       idx=max(0,int(np.searchsorted(log_times,event_time+tick*.001,side='right')-1))
       command=torch.tensor(log_obs[idx,22:25]*[4.,4.,.5],dtype=torch.float32)[None]
      obs=encode(torch.tensor(sensed[:3],dtype=torch.float32)[None],torch.tensor(sensed[3:6],dtype=torch.float32)[None],
       torch.tensor(sensed[6:12],dtype=torch.float32)[None],torch.tensor(sensed[12:18],dtype=torch.float32)[None],
       torch.tensor(adopted[legs],dtype=torch.float32)[None],torch.tensor(adopted[wheels],dtype=torch.float32)[None],reference,'Motion',command)
      out=targets(actors[which].mean(obs),reference,bounds);requested[legs]=out['leg_executed'][0].numpy();requested[wheels]=out['wheel_executed'][0].numpy()
      transport.submit(tick,requested)
     delivered=transport.delivered(tick)
     if tick%5==0:
      adopted=delivered;feedback=history.at(tick-10)
      demand[legs]=kp*(adopted[legs]-feedback[6:12][legs])-2*feedback[12:18][legs]
      demand[wheels]=adopted[wheels]-feedback[12:18][wheels];demand=np.clip(demand,-limit,limit)
     actual=torque_response(actual,demand,[5,5,0,5,5,0]);data.ctrl[refs['actuators']]=actual/refs['gears']
     mujoco.mj_step(model,data);tick+=1
    _,_,measured=state(model,data,refs);contact=np.asarray(measured['contact_force_n'])>2
    contact_count+=int(contact.all());sample_count+=1
    if tick%1000==0:
     print(json.dumps({'model':labels[which],'time_s':tick*.001,'height_m':measured['height'],
       'pitch_rad':measured['pitch'],'roll_rad':measured['roll'],'wheel_contact_N':measured['contact_force_n'],
       'both_contact_fraction':contact_count/sample_count,'contact_state':'BOTH' if contact.all() else 'SINGLE_LOST' if contact.any() else 'BOTH_LOST'},ensure_ascii=False),flush=True)
    if not np.isfinite(data.qpos).all():raise RuntimeError('nonfinite MuJoCo state')
   with viewer.lock():viewer.cam.lookat[:]=data.xpos[refs['root']];viewer.cam.distance=1.8;viewer.cam.elevation=-18
   viewer.sync();time.sleep(max(0,.01-(time.monotonic()-frame)))

if __name__=='__main__':main()
