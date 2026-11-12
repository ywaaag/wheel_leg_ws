"""One-shot identity-checked graceful stop; bounded exit wait, no watcher/retry."""
import argparse,json,os,select,signal,time
from pathlib import Path
from native_stand.students.data import sha

def stop(run,output,*,execute=False):
 run=Path(run).resolve();output=Path(output).resolve()
 import torch
 cps=sorted(run.joinpath('formal').glob('model_update_*.pt'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
 if not cps:raise ValueError('No complete saved checkpoint; do not stop')
 cp=cps[-1];d=torch.load(cp,map_location='cpu',weights_only=False)
 if not all(k in d for k in ('actor','critic','optimizer','training_rng','training_recipe')):raise ValueError('incomplete checkpoint')
 manifest=json.loads((run/'formal/manifest.json').read_text());pid=manifest['pid']
 from native_stand.command_v6.tools.health_check import process_snapshot
 proc=process_snapshot(pid)
 if (str(run/'formal') not in proc['command'] or '--phase formal' not in proc['command'] or
     abs(proc['started_unix_s']-manifest['process_started_unix_s'])>.01):raise ValueError('exact trainer process identity differs')
 progress=json.loads((run/'formal/progress.json').read_text())
 report={'run':str(run),'trainer_pid':pid,'checkpoint':str(cp),'sha256':sha(cp),'checkpoint_update':d['update'],
  'last_completed_updates_before_signal':progress['actual_updates'],'execute':execute,
  'budget_rule':'all parent executed updates remain charged, including unsaved work; no refund',
  'process_state':'PhysX reset on successor; full optimizer/RNG resume', 'signalled':False}
 if execute:
  if output.exists():raise FileExistsError(output)
  output.parent.mkdir(parents=True,exist_ok=True)
  import ctypes
  libc=ctypes.CDLL(None,use_errno=True)
  pidfd=libc.syscall(434,pid,0)
  if pidfd<0:raise OSError(ctypes.get_errno(),"pidfd_open")
  try:
   if libc.syscall(424,pidfd,signal.SIGTERM,0,0)<0:raise OSError(ctypes.get_errno(),"pidfd_send_signal")
   report['signalled']=True
   poll=select.poll();poll.register(pidfd,select.POLLIN)
   report['trainer_exit_observed']=bool(poll.poll(60000))
  finally:os.close(pidfd)
  report['time_unix_s']=time.time();output.write_text(json.dumps(report,indent=2))
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();print(json.dumps(stop(a.run,a.output,execute=a.execute),indent=2))
