"""Offline Ozone torque/ONNX handover audit; no physical control or simulation.

Torque arrays are inferred numerically, not declared as verified motor wiring.
Corrupt rows remain exported and explicitly excluded from finite-range statistics.
"""
import argparse,csv,itertools,json,time
from pathlib import Path
import numpy as np
from onnx.reference import ReferenceEvaluator
from replay_motion31_shadow_csv import read_csv,sha

LEGS=[0,1,3,4];WHEELS=[2,5]


def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def torque_arrays(rows):
 return {name:np.array([[float(r[f'(g_rl_shadow_debug).{name}[{j}]']) for j in range(n)] for r in rows])
  for name,n in [('lqr_leg_torque_nm',4),('lqr_wheel_torque_nm',2),('rl_leg_torque_nm',4),('rl_wheel_torque_nm',2)]}


def gain_candidates(target,q,dq,logged,mask,*,gains,damping,limit):
 """Search column/sign/gain on recorded same-frame arrays, with explicit residuals."""
 candidates=[]
 for output in range(logged.shape[1]):
  ranked=[]
  for source,kp,kd,sign in itertools.product(range(q.shape[1]),gains,damping,[-1,1]):
   predicted=np.clip(sign*(kp*(target[mask,source]-q[mask,source])-kd*dq[mask,source]),-limit,limit)
   error=predicted-logged[mask,output]
   ranked.append({'source_column':source,'gain':kp,'damping':kd,'sign':sign,'RMSE_Nm':float(np.sqrt(np.mean(error**2))),'max_abs_error_Nm':float(abs(error).max())})
  ranked.sort(key=lambda x:x['RMSE_Nm']);candidates.append({'logged_column':output,'best':ranked[0],'runner_up':ranked[1]})
 return candidates


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 a.output_dir.mkdir(parents=True,exist_ok=False);started=time.monotonic();source_sha=sha(a.csv)
 manifest=json.loads((a.package/'manifest.json').read_text());onnx_path=a.package/'actor.onnx'
 for name,digest in json.loads((a.package/'files_sha256.json').read_text()).items():
  if sha(a.package/name)!=digest:raise ValueError('deployment identity differs: '+name)
 if sha(onnx_path)!=manifest['actor_onnx_sha256']:raise ValueError('ONNX SHA differs')
 rows,x,y,t,columns=read_csv(a.csv);torques=torque_arrays(rows);rl=torques['rl_leg_torque_nm'];rw=torques['rl_wheel_torque_nm']
 # Preserve every row, distinguish physically impossible logged values from live faults.
 bad=(abs(x).max(1)>100)|(abs(y).max(1)>1.0001)|(~np.isfinite(np.c_[x,y]).all(1))
 inferred=(abs(y).max(1)>1e-9);ids=np.flatnonzero(inferred)
 if not len(ids):raise ValueError('no recorded nonzero policy outputs')
 begin,end=int(ids[0]),int(ids[-1]);span=(np.arange(len(t))>=begin)&(np.arange(len(t))<=end)
 # Exclude stale unchanged output after torque computations cease, without calling this control ownership.
 computing=(abs(rl).max(1)>1e-6)|(abs(rw).max(1)>1e-6)
 inference_mask=span&computing&~bad
 runtime=ReferenceEvaluator(str(onnx_path));pred=np.full_like(y,np.nan)
 for i in np.flatnonzero(inference_mask):pred[i]=runtime.run(None,{'observation':x[i:i+1]})[0][0]
 delta=abs(pred[inference_mask]-y[inference_mask])
 q=x[:,6:10]*1.5;dq=x[:,10:16][:,LEGS]*20;target=y[:,LEGS]*1.5
 fitmask=(t>30.5)&(t<35.2)&inference_mask
 if fitmask.sum()<20:raise ValueError('specified pre-disturbance comparison interval lacks valid samples')
 legfit=gain_candidates(target,q,dq,rl,fitmask,gains=[40.,60.],damping=[1.,2.],limit=40.)
 wheelfit=gain_candidates(y[:,WHEELS]*100,x[:,[12,15]]*20,np.zeros((len(t),2)),rw,fitmask,gains=[.2,1.],damping=[0.],limit=5.5)
 invalid=[]
 for i in np.flatnonzero(bad):
  invalid.append({'row':int(i),'Index':rows[i]['Index'],'time_s':float(t[i]),'observations':{str(j):float(x[i,j]) for j in np.flatnonzero(abs(x[i])>100)},'outputs':{str(j):float(y[i,j]) for j in np.flatnonzero(abs(y[i])>1.0001)}})
 # Validate inferred gain mapping on all clean computing rows, not only the fitted window.
 mapping_checks=[]
 for name,fits,targ,pos,vel,logged,limit in [('leg',legfit,target,q,dq,rl,40.),('wheel',wheelfit,y[:,WHEELS]*100,x[:,[12,15]]*20,np.zeros((len(t),2)),rw,5.5)]:
  for fit in fits:
   k=fit['logged_column'];b=fit['best'];j=b['source_column'];test=np.clip(b['sign']*(b['gain']*(targ[inference_mask,j]-pos[inference_mask,j])-b['damping']*vel[inference_mask,j]),-limit,limit);e=test-logged[inference_mask,k]
   mapping_checks.append({'kind':name,'logged_column':k,'max_abs_error_Nm':float(abs(e).max()),'RMSE_Nm':float(np.sqrt(np.mean(e**2)))})
 windows=[]
 for name,lo,hi in [('pre_response',34.3,35.3453),('response_first100ms',35.3453,35.4453),('oscillation',35.355,36.8752)]:
  mask=(t>=lo)&(t<hi)&~bad;pair=mask[1:]&mask[:-1]&(np.diff(t)<.015);di=np.diff(target.astype(float),axis=0)[pair];dv=np.diff(y[:,WHEELS].astype(float)*100,axis=0)[pair]
  windows.append({'name':name,'time_s':[lo,hi],'valid_samples':int(mask.sum()),'excluded_bad_rows':int(((t>=lo)&(t<hi)&bad).sum()),'leg_target_step_RMS_rad':np.sqrt(np.mean(di**2,axis=0)).tolist(),'wheel_target_step_RMS_rad_s':np.sqrt(np.mean(dv**2,axis=0)).tolist(),
   'leg_feedback_speed_abs_P95_rad_s':np.percentile(abs(dq[mask]),95,axis=0).tolist(),'RL_leg_limit_fraction':(abs(rl[mask])>=39.99).mean(0).tolist(),'RL_wheel_limit_fraction':(abs(rw[mask])>=5.49).mean(0).tolist(),
   'RL_leg_mean_Nm':rl[mask].mean(0).tolist(),'LQR_leg_mean_Nm':torques['lqr_leg_torque_nm'][mask].mean(0).tolist(),'command_mean_physical':(x[mask,22:25]*[4,4,.5]).mean(0).tolist()})
 previous=x[1:,16:22]-y[:-1][:,[0,1,3,4,2,5]];pair=inference_mask[1:]&inference_mask[:-1]&(np.diff(t)<.015)
 events={}
 tilt=np.arccos(np.clip(-x[:,4],-1,1));pitch=np.arctan2(x[:,5],-x[:,4]);roll=np.arctan2(x[:,3],-x[:,4])
 for name,mask in [('first_RL_torque_nonzero',computing),('first_leg_speed_over_0p5',((abs(dq)>.5).any(1))&(t>34)),('first_leg_speed_over_2',((abs(dq)>2).any(1))&(t>34)),('first_tilt_over_0p1',(tilt>.1)&(t>34)),('first_tilt_over_0p3',(tilt>.3)&(t>34))]:
  ix=np.flatnonzero(mask&~bad);events[name]=float(t[ix[0]]) if len(ix) else None
 near=int(np.searchsorted(t,35.345));examples={}
 for sec in [30.344,35.324,35.345,35.355,35.415]:
  i=int(np.searchsorted(t,sec));examples[str(t[i])]={'row':i,'bad':bool(bad[i]),'q_rad':q[i].tolist(),'target_rad':target[i].tolist(),'leg_target_minus_q_rad':(target[i]-q[i]).tolist(),'dq_leg_rad_s':dq[i].tolist(),'RL_leg_logged_Nm':rl[i].tolist(),'LQR_leg_logged_Nm':torques['lqr_leg_torque_nm'][i].tolist(),'wheel_target_rad_s':(y[i,WHEELS]*100).tolist(),'RL_wheel_logged_Nm':rw[i].tolist()}
 result={'CSV':str(a.csv),'CSV_sha256':source_sha,'ONNX':str(onnx_path),'ONNX_sha256':sha(onnx_path),'model_identity':'same 19_53 candidate assumed, supported by clean computing rows exact replay','missing_task_bits_assumed':[1,0,0,1,0,0],
  'samples':len(t),'time_s':[float(t[0]),float(t[-1])],'median_dt_s':float(np.median(np.diff(t))),'gaps_over15ms':[{'from':float(t[i]),'to':float(t[i+1]),'dt':float(t[i+1]-t[i])} for i in np.flatnonzero(np.diff(t)>.015)],
  'corrupt_rows':invalid,'clean_computing_rows':int(inference_mask.sum()),'clean_replay_max_abs_error':delta.max(0).tolist(),'clean_replay_match_fraction':float((delta.max(1)<=1e-5).mean()),
  'declared_control':manifest['control'],'same_frame_gain_search_interval_s':[30.5,35.2],'leg_gain_search':legfit,'wheel_gain_search':wheelfit,'all_clean_rows_gain_validation':mapping_checks,
  'mapping_warning':'numeric column/sign correspondence only; torque variables may be pre-driver diagnostics; motor wiring/compensation/actual adoption not measured',
  'previous_vs_last_mean':{'matching_fraction_1e5':float((abs(previous[pair]).max(1)<=1e-5).mean()),'max_abs':abs(previous[pair]).max(0).tolist(),'warning':'not necessarily actual adopted targets'},
  'events':events,'windows':windows,'examples':examples,'handover_time':'UNKNOWN; first marked motion response near35.35s, no owner/enable flag captured',
  'unresolved':['origin of three corrupt snapshots versus live actor inputs','actual driver gain/torque application and compensation','PD adoption/feedback timestamps','precise controller handover marker'],
  'simulation':0,'PPO':0,'optimizer_steps':0}
 write(a.output_dir/'analysis.json',result)
 np.savez_compressed(a.output_dir/'analysis.npz',time=t,observation=x,logged=y,replayed=pred,invalid=bad,computing=computing,**torques)
 with (a.output_dir/'replayed.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['Index','Time','invalid_row','torque_computing','ONNX_replayed']+[f'logged[{j}]' for j in range(6)]+[f'onnx[{j}]' for j in range(6)])
  for i in range(len(t)):w.writerow([rows[i]['Index'],t[i],int(bad[i]),int(computing[i]),int(inference_mask[i]),*y[i],*[float(v) if np.isfinite(v) else '' for v in pred[i]]])
 write(a.output_dir/'terminal.json',{'status':'COMPLETED','wall_s':time.monotonic()-started,'PPO':0,'simulation':0,'source_unchanged':source_sha==sha(a.csv)})
 print(json.dumps({k:result[k] for k in ['clean_computing_rows','clean_replay_max_abs_error','corrupt_rows','leg_gain_search','wheel_gain_search','events']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
