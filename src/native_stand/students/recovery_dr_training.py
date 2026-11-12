"""DR hooks for the public student training loop; no launcher or second trainer."""
import copy
from pathlib import Path
from . import recovery_dr as dr

def training_environment(spec,count,context):
 from .evaluator import environment
 from .training import observations
 env=environment(spec,'RecoveryStudent',envs=count);env.curriculum_training=True
 env.select_initial_family(spec['recovery']['active_family'],'train');env.student_yaw_amplitude=.01
 if context is not None:
  adapter=env.recovery_dr;adapter.configure(state=context['state'])
  if context.get('stream'):
   import numpy as np
   adapter.episodes=np.asarray(context['stream']['episodes'],dtype=np.int64)
   adapter.reset_count=context['stream']['reset_count']
 obs,_=env.reset()
 return env,observations(env,obs)

def capture(env,context):
 if context is not None:
  context['stream']=env.recovery_dr.snapshot()
  context['state']=copy.deepcopy(env.recovery_dr.state)

def assess(spec,output,actor,cases,context,update,offset,checkpoint,stop):
 from .training import evaluate_cases
 from native_stand.command_v6.completion import write
 records=[];level=context['state']['level']
 for group in (2,1):
  rows=[]
  for k,row in enumerate(cases[:3]):
   seed=20270000+update*100+group*10+k
   rows.append({**row,'case_id':f"dr_level{level}_{'current' if group==2 else 'previous'}_{k}_u{update}",
    '_dr_override':{'strength':dr.exposure(level,group),'seed':seed}})
  result=evaluate_cases(spec,output,actor,rows,offset=offset+len(records),stop=stop,checkpoint=checkpoint)
  records+=result
 return records

def observe(context,nominal,extra,update):
 nominal=[e for r in nominal for e in r['per_env']]
 current=[e for r in extra[:3] for e in r['per_env']]
 previous=[e for r in extra[3:] for e in r['per_env']]
 state,report=dr.advance(context['state'],nominal,current,previous,update)
 context['state']=state
 return {'version':dr.VERSION,'state':state,'assessment':report,'records':extra,'scope':'fixed model/asset R1-R2 finite-time DR results; not sim2real certification'}


def verify_records(folder,records):
 import json,numpy as np
 from .data import sha
 rows=[]
 for record in records:
  path=Path(folder)/record['path'];payload=json.loads(path.read_text())
  if sha(path)!=record['sha256']:raise ValueError('DR readback evaluation SHA differs')
  m=payload.get('domain_randomization',{})
  if m.get('endpoint_probe',{}).get('status')!='PASSED' or not m.get('executed_PD_readback') or not m.get('last_readback',{}).get('readback_matches_requested'):raise ValueError('DR material/PD runtime readback missing')
  trace=Path(folder)/payload['trace']['path']
  if sha(trace)!=payload['trace']['sha256']:raise ValueError('DR source trace SHA differs')
  with np.load(trace) as a:
   for key in ('dr_parameters_pre','dr_episode_pre'):
    if key not in a:raise ValueError('DR trace missing '+key)
   valid=a['first_episode_valid'].astype(bool)
   if not np.isfinite(a['dr_parameters_pre'][valid]).all():raise ValueError('nonfinite DR parameters')
   for e in range(valid.shape[1]):
    values=a['dr_parameters_pre'][valid[:,e],e]
    if len(values) and not np.all(values==values[0]):raise ValueError('DR parameters changed inside first episode')
  rows.append({'case_id':record['case']['case_id'],'backend_endpoints':m['endpoint_probe'],'PD':m['executed_PD_readback'],'readback':m['last_readback']})
 return {'version':dr.VERSION,'status':'PASSED','rows':rows,'material_pair':'average(.5 ground, backend wheel); explicit mode on both','force_identification':'NOT_MEASURED','teacher_control':False}
