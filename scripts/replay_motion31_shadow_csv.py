"""CPU-only named-index replay of Ozone shadow observations, without altering measurements."""
import argparse,csv,ctypes,hashlib,json,re,subprocess,tempfile
from pathlib import Path
import numpy as np
import torch

PATTERN=re.compile(r'\.(observation|mean_action)\[(\d+)\]$')
TASK=np.array([1,0,0,1,0,0],dtype=np.float32)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_csv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as stream:
  reader=csv.DictReader(stream);rows=list(reader);fields=reader.fieldnames
 if not rows:raise ValueError('empty CSV')
 columns={'observation':{},'mean_action':{}}
 for name in fields:
  match=PATTERN.search(name)
  if match:
   group,index=match[1],int(match[2])
   if index in columns[group]:raise ValueError('duplicate numeric field index')
   columns[group][index]=name
 if not set(range(25)).issubset(columns['observation']) or set(columns['mean_action'])!=set(range(6)):raise ValueError('missing required observation0:24/action0:5')
 if set(columns['observation'])-set(range(31)):raise ValueError('unknown observation index')
 x=np.zeros((len(rows),31),np.float32);x[:,25:31]=TASK
 for index,name in columns['observation'].items():x[:,index]=[float(r[name]) for r in rows]
 if not np.all(x[:,25:31]==TASK):raise ValueError('this replay requires Motion/NONE task flags')
 logged=np.array([[float(row[columns['mean_action'][i]]) for i in range(6)] for row in rows],np.float32)
 time=np.array([float(row['Time']) for row in rows])
 if not np.isfinite(x).all() or not np.isfinite(logged).all() or not np.isfinite(time).all():raise ValueError('nonfinite CSV value')
 if len(time)>1 and (np.diff(time)<=0).any():raise ValueError('non-increasing timestamps; do not infer missing data')
 return rows,x,logged,time,columns

def stats(values):
 a=np.asarray(values,dtype=float)
 return {'mean':a.mean(0).tolist(),'abs_mean':abs(a).mean(0).tolist(),'min':a.min(0).tolist(),'max':a.max(0).tolist(),'abs_p95':np.percentile(abs(a),95,axis=0).tolist()}

def replay(source,package,output,mujoco):
 source,package,output=map(lambda p:Path(p).resolve(),(source,package,output))
 if output.exists() or output.with_suffix('.analysis.json').exists():raise FileExistsError('refuse to overwrite existing replay')
 manifest=json.loads((package/'manifest.json').read_text())
 if manifest['input_shape']!=[1,31] or manifest['output_shape']!=[1,6] or manifest['source_sha256']!='83ee2371d37c48cefcf61c7de98a1cfb5eb1e01c19071efbb2e6949a36f9a820':raise ValueError('explicit Motion u1600 interface/identity required')
 consumed=['actor_torchscript.pt','actor_runtime.c','actor_weights.h','reference_adapter.py']
 for name in consumed:
  if sha(package/name)!=manifest['files_sha256'][name]:raise ValueError('deployment package SHA differs: '+name)
 rows,x,logged,time,columns=read_csv(source)
 torch.set_num_threads(4);model=torch.jit.load(str(package/'actor_torchscript.pt'),map_location='cpu').eval()
 # Keep the actual exported [1,31] inference boundary, rather than retraining/re-encoding inputs.
 with torch.inference_mode():pred=np.concatenate([model(torch.from_numpy(row[None])).numpy() for row in x])
 output.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='motion31_host_replay_',dir=output.parent) as temporary:
  library=Path(temporary)/'actor.so'
  command=['cc','-shared','-fPIC','-O2',str(package/'actor_runtime.c'),'-o',str(library),'-lm']
  subprocess.run(command,check=True,capture_output=True)
  lib=ctypes.CDLL(str(library));ptr=np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS')
  lib.motion31_actor.argtypes=[ptr,ptr];lib.motion31_actor.restype=ctypes.c_int
  c_pred=np.zeros_like(pred)
  for i in range(len(x)):
   if lib.motion31_actor(x[i],c_pred[i])!=0:raise ValueError('C reference rejected input/output')
 delta=pred-logged
 names=['Index','Time']+[f'input_used[{i}]' for i in range(31)]+[f'logged_mean_action[{i}]' for i in range(6)]+[f'replayed_mean_action[{i}]' for i in range(6)]+[f'C_reference_mean_action[{i}]' for i in range(6)]+[f'replayed_minus_logged[{i}]' for i in range(6)]+['replayed_left_wheel_target_rad_s','replayed_right_wheel_target_rad_s','logged_left_wheel_target_rad_s','logged_right_wheel_target_rad_s']
 with output.open('x',newline='',encoding='utf-8-sig') as stream:
  writer=csv.writer(stream);writer.writerow(names)
  for i,row in enumerate(rows):writer.writerow([row['Index'],row['Time']]+[format(float(v),'.9g') for v in np.r_[x[i],logged[i],pred[i],c_pred[i],delta[i],pred[i,[2,5]]*100.,logged[i,[2,5]]*100.]])
 previous_delta=x[1:,16:22]-logged[:-1][:,[0,1,3,4,2,5]]
 report={'source_csv':str(source),'source_csv_sha256':sha(source),'output_csv':str(output),'output_sha256':sha(output),'model_checkpoint_sha256':manifest['source_sha256'],'package':str(package),'package_manifest_sha256':sha(package/'manifest.json'),'consumed_model_files_sha256':{n:sha(package/n) for n in consumed},'rows':len(x),'time_s':[float(time[0]),float(time[-1])],'policy_timestamp_delta_s':stats(np.diff(time)),
  'input_note':'observation fields are ALREADY normalized; mapped by bracket index, not CSV column order; no auto wrapping/zero/scale/polarity fixes',
  'missing_observations_assumed':{str(i):float(TASK[i-25]) for i in range(25,31) if i not in columns['observation']},
  'assumption_note':'25:31 absent from this log; filled Motion/Recovery/Jump=1/0/0, NONE/SMALL/BIG=1/0/0; this is a declared assumption, not a captured measurement',
  'named_index_reordering':{'observation22_csv_column':list(rows[0]).index(columns['observation'][22]),'observation23_csv_column':list(rows[0]).index(columns['observation'][23])},
  'logged_vs_torchscript_max_abs':abs(delta).max(0).tolist(),'logged_vs_torchscript_MAE':abs(delta).mean(0).tolist(),'C_vs_torchscript_max_abs':abs(c_pred-pred).max(0).tolist(),
  'output_action_order':['left_hip','left_active_link','left_wheel','right_hip','right_active_link','right_wheel'],
  'logged_wheel_targets_rad_s':stats(logged[:,[2,5]]*100),'replayed_wheel_targets_rad_s':stats(pred[:,[2,5]]*100),
  'first_row':{'input':x[0].tolist(),'logged_action':logged[0].tolist(),'replayed_action':pred[0].tolist(),'q_leg_rad':(x[0,6:10]*1.5).tolist(),'command_vx_yaw_height':(x[0,22:25]*[4,4,.5]).tolist()},
  'input_leg_q_rad':stats(x[:,6:10]*1.5),'leg_abs_q_gt_3p1_fraction':(abs(x[:,6:10]*1.5)>3.1).mean(0).tolist(),'leg_large_jumps_gt_3rad':(abs(np.diff(x[:,6:10]*1.5,axis=0))>3.).sum(0).tolist(),
  'command_physical':stats(x[:,22:25]*[4,4,.5]),'nonzero_vx_command_fraction':float((abs(x[:,22])>1e-6).mean()),
  'previous_vs_last_logged_action':{'order':'four legs then two wheels; compare previous[t] to action[t-1] reordered0,1,3,4,2,5','max_abs':abs(previous_delta).max(0).tolist(),'all_channels_within_1e_5_fraction':float((abs(previous_delta).max(1)<1e-5).mean()),'limitation':'LQR actual controller output not logged; cannot establish whether these RL targets actually controlled hardware'},'simulation_started':False,'optimizer_steps':0}
 if mujoco:
  path=Path(mujoco).resolve();meta=json.loads((path.parent/'manifest.json').read_text())
  if meta['checkpoint_sha256']!=manifest['source_sha256']:raise ValueError('MuJoCo model differs from deployed model')
  with np.load(path) as a:
   hold=(a['time']>=10)&(a['time']<18)
   if not hold.any() or not np.allclose(a['command'][hold,:2],0):raise ValueError('expected MuJoCo zero-command10:18s window')
   report['mujoco_comparison']={'trace':str(path),'trace_sha256':sha(path),'manifest_sha256':sha(path.parent/'manifest.json'),'checkpoint_sha256':meta['checkpoint_sha256'],'full_wheel_targets_rad_s':stats(a['wheel_target']),'hold_10_18s_wheel_targets_rad_s':stats(a['wheel_target'][hold]),'hold_leg_q_rad':stats(a['actor_input'][hold,6:10]*1.5),'first_wheel_targets_rad_s':a['wheel_target'][0].tolist(),'original_summary':json.loads((path.parent/'command_00.json').read_text()),'scope':'existing single-condition reference trace, not a new simulation or perfect zero-motion certificate'}
 # Sensitivity examples are OFFLINE, first-row only: do not export as corrected hardware actions.
 counter=[]
 for label,wrap,zero,prev in [('as_logged',False,False,False),('zero_command_only',False,True,False),('wrapped_q_only',True,False,False),('wrapped_q_zero_command',True,True,False),('wrapped_q_zero_command_zero_previous',True,True,True)]:
  v=x[:1].copy()
  if wrap:v[:,6:10]=((v[:,6:10]*1.5+np.pi)%(2*np.pi)-np.pi)/1.5
  if zero:v[:,22:24]=0.
  if prev:v[:,16:22]=0.
  with torch.inference_mode():a=model(torch.from_numpy(v)).numpy()[0]
  counter.append({'change':label,'wheel_targets_rad_s':(a[[2,5]]*100).tolist()})
 report['first_row_sensitivity_hypotheses_only']=counter
 report['sensitivity_warning']='Wrapping and zero previous are numerical counterfactuals, NOT verified encoder calibration, LQR target conversion, safe controller initialization or closed-loop evidence.'
 output.with_suffix('.analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mujoco-trace',type=Path);a=p.parse_args()
 r=replay(a.csv,a.package,a.output,a.mujoco_trace);print(json.dumps({k:r[k] for k in ('output_csv','rows','logged_vs_torchscript_max_abs','C_vs_torchscript_max_abs','missing_observations_assumed')},indent=2))
if __name__=='__main__':main()
