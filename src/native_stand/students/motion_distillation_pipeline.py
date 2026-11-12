"""Same31D large→small offline BC and bounded MuJoCo orchestration.

Planning is read-only. Every mutating/optimizer/simulation operation has an explicit
execution flag at CLI. Offline fit never grants native/sim2sim acceptance.
"""
import copy,json,time,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from .data import sha
from .model import Student,load
from .distill import loss,physical_errors
VERSION='motion31_same_input_distillation_v1'
COMMANDS=[[0.,0.,.34],[.5,0.,.34],[1.,0.,.34],[1.5,0.,.34],[2.,0.,.34],[0.,.5,.34],[0.,-.5,.34]]


def dataset_cohorts(data,manifest):
 """Old seven-command datasets stay strict; new sets require named declarations."""
 if manifest.get('command_set_version') is None:
  return [{'case_id':str(i),'target':c,'variant':0} for i,c in enumerate(COMMANDS)]
 if manifest['command_set_version']!='motion31_explicit_commands_v2':raise ValueError('unknown distillation command set')
 rows=data.get('cohorts')
 if not isinstance(rows,list) or not rows or rows!=manifest.get('cohorts'):raise ValueError('explicit dataset/manifest cohorts differ')
 if len({r.get('case_id') for r in rows})!=len(rows):raise ValueError('duplicate distillation case ID')
 for row in rows:
  target=row.get('target',[])
  if len(target)!=3 or not np.isfinite(target).all() or row.get('variant') not in (0,1,2):raise ValueError('invalid explicit distillation cohort')
 return rows


def validate_teacher(checkpoint,spec_path):
 checkpoint=Path(checkpoint).resolve();spec_path=Path(spec_path).resolve()
 actor,data=load(checkpoint,'MotionZeroStudent');spec=json.loads(spec_path.read_text())
 if data.get('spec_sha256')!=sha(spec_path) or data.get('asset_sha256')!=spec['asset_sha256'] or sha(spec['asset_path'])!=spec['asset_sha256']:raise ValueError('teacher exact spec/asset binding differs')
 arch=data.get('network_architecture',{})
 if data['interface_version']!='multiskill_mcu_v2' or arch.get('actor_input')!=31 or arch.get('actor_hidden_dims')!=[512,256,128]:raise ValueError('pipeline requires explicit same31D large Motion teacher')
 if spec['action']['leg_scale_rad']!=1.5 or spec['action']['wheel_scale_rad_s']!=100.:raise ValueError('physical action contract differs')
 return actor.eval().requires_grad_(False),data,spec


def trace_inputs(run,checkpoint):
 run=Path(run).resolve();digest=sha(checkpoint);rows=[];inputs={}
 for ep in sorted(run.glob('evaluation_*.json')):
  d=json.loads(ep.read_text())
  if not isinstance(d,dict) or 'case' not in d:continue
  if d.get('checkpoint_sha256')!=digest:raise ValueError('capture checkpoint SHA differs:'+str(ep))
  command=d['case'].get('command')
  if command not in COMMANDS:continue
  path=(ep.parent/d['trace']['path']).resolve()
  if not path.is_relative_to(run):raise ValueError('capture escapes declared run')
  if d['trace'].get('sha256') and sha(path)!=d['trace']['sha256']:raise ValueError('capture trace SHA differs')
  with np.load(path,allow_pickle=False) as a:
   required={'actor_input','controller','first_episode_valid','state','effective_command_pre'}
   if not required<=set(a.files):raise ValueError('actual native capture fields missing:'+str(path))
   x=a['actor_input']
   if x.ndim!=3 or x.shape[-1]!=31 or not np.isfinite(x).all():raise ValueError('capture Actor input invalid')
   if not np.allclose(x[...,22:25],a['effective_command_pre']/np.array([4.,4.,.5]),atol=1e-6):raise ValueError('command/obs readback differs')
   if not ((x[...,25]==1)&(x[...,26:28]==0).all(-1)&(x[...,28]==1)&(x[...,29:31]==0).all(-1)).all():raise ValueError('Motion/NONE task encoding differs')
   valid=a['first_episode_valid'].astype(bool)&(a['controller'][...,0]==3)&a['state'][...,29].astype(bool)
   if x.shape[1]<2:raise ValueError('need independent env episodes for train/validation split')
   for e in range(x.shape[1]):
    ix=np.flatnonzero(valid[:,e]);age=np.arange(len(ix))*.01
    phase=np.select([age<2,age<10,age<18,age<26],[0,1,2,3],default=4)
    rows.append({'x':x[ix,e].copy(),'phase':phase,'command':COMMANDS.index(command),'episode':f'{d["case"]["seed"]}:{command}:{e}',
      'validation':e>=x.shape[1]-2,'source':str(ep),'teacher_case_accepted':bool(d['per_env'][e].get('accepted'))})
  inputs[str(ep)]=sha(ep);inputs[str(path)]=sha(path)
 if {r['command'] for r in rows}!=set(range(7)):raise ValueError('teacher capture must cover all seven commands')
 return rows,inputs


def plan(checkpoint,spec_path,capture_run):
 _,_,_=validate_teacher(checkpoint,spec_path);rows,inputs=trace_inputs(capture_run,checkpoint)
 return {'version':VERSION,'status':'CPU_PLAN','files_written':0,'simulation':0,'optimizer_updates':0,
  'teacher_sha256':sha(checkpoint),'teacher_spec_sha256':sha(spec_path),'student_network':[31,128,64,32,6],
  'interface':'same normalized31D, Motion/NONE, deterministic6D Tanh; no history/Critic deployed',
  'samples_available':sum(len(r['x']) for r in rows),'episodes':len(rows),'validation_split':'last2env per case, entire episodes, never random rows',
  'commands':COMMANDS,'capture_inputs_sha256':inputs,'pipeline':['prepare actual native inputs + teacher labels','bounded command/phase balanced offline BC','optional bounded Student-driven MuJoCo collection + frozen Teacher labels','offline aggregate refit','pure Student MuJoCo seven-command30s assessment'],
  'authorization':'PLAN_ONLY; fitting and simulation require their own explicit execute+authorization text',
  'teacher_limitations_scope':'Read this exact Teacher native and MuJoCo baseline; historical750 swing/drift findings do not automatically apply to a different Teacher. Distillation cannot certify or repair missing teacher skills.',
  'native_diagnostic':'explicit motion31_distilled_native_screen_v1 through supervisor --single-screen-request; seven commands x one env, not certification',
  'not_implemented':['native Isaac multi-env Student certification','automatic native DAgger supervisor','PPO finetune; no inherited Teacher Adam/Critic','hardware realtime verification'],
  'acceptance':'offline metrics diagnostic only; native/student sim2sim/hardware are separate; supplemental swing gate cannot be ignored'}


def write_json(path,data):
 with Path(path).open('x') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)


def prepare(checkpoint,spec_path,capture_run,output,max_samples_per_episode=1500):
 if not 100<=max_samples_per_episode<=3000:raise ValueError('capture bound100..3000 per episode')
 output=Path(output).resolve()
 if output.exists():raise FileExistsError(output)
 actor,_,_=validate_teacher(checkpoint,spec_path);rows,inputs=trace_inputs(capture_run,checkpoint)
 xs=[];cohorts=[];phases=[];splits=[];episodes=[]
 for eid,row in enumerate(rows):
  # Fixed stratified temporal coverage, keep transition and parking phases.
  ids=[]
  for phase in range(5):
   eligible=np.flatnonzero(row['phase']==phase)
   if len(eligible):ids.extend(eligible[np.linspace(0,len(eligible)-1,min(len(eligible),max_samples_per_episode//5)).astype(int)])
  ids=np.array(sorted(set(ids)),dtype=int)
  xs.append(torch.from_numpy(row['x'][ids]).float());cohorts.extend([row['command']]*len(ids));phases.extend(row['phase'][ids]);splits.extend([int(row['validation'])]*len(ids));episodes.extend([eid]*len(ids))
 x=torch.cat(xs)
 with torch.inference_mode():y=torch.cat([actor.mean(b) for b in x.split(1024)])
 data={'x':x,'y':y,'cohort':torch.tensor(cohorts),'phase':torch.tensor(phases),'split':torch.tensor(splits),'episode':torch.tensor(episodes),
 'episode_keys':[r['episode'] for r in rows],'provenance':'actual native teacher-policy states; deterministic Teacher query; not Student closed loop',
 'teacher_checkpoint':str(Path(checkpoint).resolve()),'teacher_sha256':sha(checkpoint),'parent_spec':str(Path(spec_path).resolve()),'parent_spec_sha256':sha(spec_path)}
 output.mkdir(parents=True,exist_ok=False);torch.save(data,output/'dataset.pt')
 write_json(output/'manifest.json',{'version':VERSION,'teacher_checkpoint':data['teacher_checkpoint'],'teacher_sha256':data['teacher_sha256'],'parent_spec':data['parent_spec'],'parent_spec_sha256':data['parent_spec_sha256'],
  'dataset_sha256':sha(output/'dataset.pt'),'source_inputs_sha256':inputs,'episodes':len(rows),'samples':len(x),'source_case_accepted':[r['teacher_case_accepted'] for r in rows],
  'capability_certified':False,'optimizer_updates':0,'simulation':0,'student_state_replay':'NOT_COLLECTED'})
 return {'dataset':str(output/'dataset.pt'),'samples':len(x),'optimizer_updates':0,'simulation':0}


def relabel(checkpoint,spec_path,source_dataset,output,*,execute=False):
 """Query a new same-contract Teacher on existing states; never claim new rollouts."""
 source_dataset=Path(source_dataset).resolve();output=Path(output).resolve()
 if output.exists():raise FileExistsError(output)
 manifest_path=source_dataset.with_name('manifest.json');m=json.loads(manifest_path.read_text())
 if sha(source_dataset)!=m['dataset_sha256']:raise ValueError('source dataset SHA differs')
 old_actor,_,old_spec=validate_teacher(m['teacher_checkpoint'],m['parent_spec'])
 actor,source,spec=validate_teacher(checkpoint,spec_path)
 if any(old_spec.get(k)!=spec.get(k) for k in ('control','action','asset_sha256')):raise ValueError('relabel physical contract differs')
 data=torch.load(source_dataset,map_location='cpu',weights_only=False)
 if any(data.get(k)!=m.get(k) for k in ('teacher_checkpoint','teacher_sha256','parent_spec','parent_spec_sha256')):raise ValueError('source dataset embedded provenance differs')
 if sha(m['teacher_checkpoint'])!=m['teacher_sha256'] or sha(m['parent_spec'])!=m['parent_spec_sha256']:raise ValueError('source Teacher/spec SHA differs')
 x=data['x'];n=len(x)
 if x.shape!=(n,31) or data['y'].shape!=(n,6) or not torch.isfinite(x).all() or not torch.isfinite(data['y']).all():raise ValueError('source dataset tensors invalid')
 if any(data[k].shape!=(n,) for k in ('cohort','phase','split','episode')):raise ValueError('source metadata shape differs')
 if not ((data['split']==0)|(data['split']==1)).all() or not (data['split']==1).any():raise ValueError('source episode split invalid')
 if set(data['episode'][data['split']==0].tolist())&set(data['episode'][data['split']==1].tolist()):raise ValueError('source episode split overlaps')
 if not ((x[:,25]==1)&(x[:,26:28]==0).all(-1)&(x[:,28]==1)&(x[:,29:31]==0).all(-1)).all():raise ValueError('source Motion task encoding differs')
 if set(data['cohort'].tolist())!=set(range(7)) or not ((data['phase']>=0)&(data['phase']<5)).all():raise ValueError('source command/phase coverage invalid')
 provenance={str(source_dataset):sha(source_dataset),str(manifest_path):sha(manifest_path),str(Path(m['teacher_checkpoint']).resolve()):m['teacher_sha256'],str(Path(m['parent_spec']).resolve()):m['parent_spec_sha256']}
 result={'version':VERSION,'status':'CPU_PLAN','files_written':0,'simulation':0,'optimizer_updates':0,'samples':n,'teacher_sha256':sha(checkpoint),'source_state_teacher_sha256':m['teacher_sha256'],'source_inputs_sha256':provenance,'scope':'existing source-Teacher states, newly queried actions; not new-Teacher rollouts; heldout measures old-state fit only'}
 if not execute:return result
 with torch.inference_mode():labels=torch.cat([actor.mean(b) for b in x.split(1024)])
 if not torch.isfinite(labels).all():raise ValueError('new Teacher labels nonfinite')
 data={**data,'y':labels,'teacher_checkpoint':str(Path(checkpoint).resolve()),'teacher_sha256':sha(checkpoint),'parent_spec':str(Path(spec_path).resolve()),'parent_spec_sha256':sha(spec_path),'provenance':result['scope'],'state_source':provenance}
 output.mkdir(parents=True,exist_ok=False);torch.save(data,output/'dataset.pt')
 write_json(output/'manifest.json',{'version':VERSION,**{k:data[k] for k in ('teacher_checkpoint','teacher_sha256','parent_spec','parent_spec_sha256')},'dataset_sha256':sha(output/'dataset.pt'),'source_inputs_sha256':provenance,'state_source_teacher_sha256':m['teacher_sha256'],'episodes':len(data['episode_keys']),'samples':n,'capability_certified':False,'optimizer_updates':0,'simulation':0,'student_state_replay':'NOT_COLLECTED','scope':result['scope']})
 return {**result,'status':'COMPLETED','files_written':2,'dataset':str(output/'dataset.pt')}


def validate_student(data,spec_path):
 d=data.get('distillation',{})
 if d.get('version')!=VERSION or data.get('interface_version')!='multiskill_mcu_v2' or data.get('role')!='MotionZeroStudent':raise ValueError('explicit distilled Motion31 identity required')
 if data.get('network_architecture') is not None or data.get('recipe_version') is not None:raise ValueError('distilled small Actor cannot pretend to be large PPO continuation')
 if d.get('parent_spec_sha256')!=sha(spec_path) or d.get('parent_spec')!=str(Path(spec_path).resolve()):raise ValueError('distilled parent spec differs')
 teacher,cp,_=validate_teacher(d['teacher_checkpoint'],spec_path)
 if sha(d['teacher_checkpoint'])!=d['teacher_sha256'] or data.get('asset_sha256')!=cp['asset_sha256']:raise ValueError('distilled Teacher/asset differs')
 if not (data.get('actor',{}).get('mlp.0.weight',torch.empty(0)).shape==(128,31)):raise ValueError('explicit small128/64/32 shape required')
 return d


def fit(dataset,output,*,steps=5000,wall_s=1200,batch_size=512,seed=20261018,initial_student=None,replay=None,validation_every=100,patience=None):
 if not 1<=steps<=50000 or not 1<=wall_s<=14400 or not 32<=batch_size<=4096:raise ValueError('bounded BC limits exceeded')
 output=Path(output).resolve()
 if output.exists():raise FileExistsError(output)
 dataset=Path(dataset).resolve();m=json.loads(dataset.with_name('manifest.json').read_text())
 if sha(dataset)!=m['dataset_sha256']:raise ValueError('dataset SHA differs')
 data=torch.load(dataset,map_location='cpu',weights_only=False);teacher,source,_=validate_teacher(m['teacher_checkpoint'],m['parent_spec'])
 if any(data.get(k)!=m.get(k) for k in ('teacher_checkpoint','teacher_sha256','parent_spec','parent_spec_sha256')):raise ValueError('dataset embedded provenance differs')
 if sha(m['parent_spec'])!=m['parent_spec_sha256']:raise ValueError('dataset parent spec changed')
 if data['x'].ndim!=2 or data['x'].shape[1]!=31 or data['y'].shape!=(len(data['x']),6) or not torch.isfinite(data['x']).all() or not torch.isfinite(data['y']).all():raise ValueError('dataset tensors invalid')
 if m['teacher_sha256']!=sha(m['teacher_checkpoint']):raise ValueError('Teacher changed')
 cohorts=dataset_cohorts(data,m);count=len(cohorts)
 if not 1<=validation_every<=1000 or (patience is not None and not 1<=patience<=20):raise ValueError('invalid validation schedule')
 for key in ('cohort','phase','split','episode'):
  if data[key].shape!=(len(data['x']),):raise ValueError('dataset metadata shape differs')
 if set(data['cohort'].tolist())!=set(range(count)) or not ((data['split']==0)|(data['split']==1)|(data['split']==2)).all():raise ValueError('declared cohort/split coverage differs')
 for split_id in (1,2):
  if set(data['episode'][data['split']==0].tolist())&set(data['episode'][data['split']==split_id].tolist()):raise ValueError('whole episode split overlaps')
 torch.set_num_threads(4)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed);student=Student(True,version='multiskill_mcu_v2')
 student.log_std.requires_grad_(False)
 if initial_student:
  old=torch.load(initial_student,map_location='cpu',weights_only=False);validate_student(old,m['parent_spec']);student.load_state_dict(old['actor'])
 if replay:
  rp=Path(replay);rm=json.loads(rp.with_name('replay_manifest.json').read_text())
  if sha(rp)!=rm['replay_sha256'] or rm['teacher_sha256']!=m['teacher_sha256']:raise ValueError('DAgger replay identity differs')
  rd=torch.load(rp,map_location='cpu',weights_only=False)
  rd_episodes=rd.get('episode',rd['cohort'])+int(data['episode'].max())+1
  data['episode']=torch.cat((data['episode'],rd_episodes))
  for key in ('x','y','cohort','phase','split'):data[key]=torch.cat((data[key],rd[key]))
 x,y=data['x'],data['y'];split=data['split'];groups=[]
 for c in range(count):
  for phase in range(5):
   ids=torch.where((split==0)&(data['cohort']==c)&(data['phase']==phase))[0]
   if len(ids):groups.append(ids)
 if {int(c) for c in data['cohort'][split==0].unique()}!=set(range(count)) or {int(c) for c in data['cohort'][split==1].unique()}!=set(range(count)):raise ValueError('complete declared training/heldout cohorts required')
 # New MCU datasets balance command/phase, then episode, without per-item RNG calls.
 episode_groups=[[ids[data['episode'][ids]==ep] for ep in data['episode'][ids].unique()] for ids in groups]
 group_weights=None
 if m.get('cohort_sampling_weights') is not None:
  weights=np.asarray(m['cohort_sampling_weights'],dtype=float)
  if weights.shape!=(count,) or not np.isfinite(weights).all() or (weights<=0).any() or not np.isclose(weights.sum(),1.):raise ValueError('explicit positive cohort sampling weights must sum to one')
  group_cohorts=[int(data['cohort'][ids[0]]) for ids in groups]
  counts=np.bincount(group_cohorts,minlength=count)
  group_weights=np.array([weights[c]/counts[c] for c in group_cohorts])
 optimizer=torch.optim.Adam(student.mlp.parameters(),lr=1e-3);g=torch.Generator().manual_seed(seed+1)
 output.mkdir(parents=True,exist_ok=False);started=time.monotonic();best=float('inf');actual=0;history=[];bad=0;reason='step_limit';visits=torch.zeros(len(x),dtype=torch.long)
 torch.save({'actor':copy.deepcopy(student.state_dict()),'seed':seed,'interface_version':'multiskill_mcu_v2','source':str(initial_student) if initial_student else None},output/('inherited_actor_initialization.pt' if initial_student else 'random_initialization.pt'))
 write_json(output/'budget.json',{'status':'RUNNING','online_PPO':0,'offline_step_limit':steps,'wall_limit_s':wall_s,'used':0})
 def evaluate():
  result={};student.eval()
  with torch.inference_mode():
   for c in range(count):
    ids=torch.where((split==1)&(data['cohort']==c))[0]
    if not len(ids):continue
    pred=torch.cat([student.mean(x[ix]) for ix in ids.split(1024)])
    result[str(c)]={**physical_errors(pred,y[ids]),'loss':float(loss(pred,y[ids])),'samples':len(ids)}
  student.train();return result
 def checkpoint(step,metrics):
  return {'interface_version':'multiskill_mcu_v2','role':'MotionZeroStudent','actor':copy.deepcopy(student.state_dict()),'asset_sha256':source['asset_sha256'],
   'offline_bc_steps':step,'distillation':{'version':VERSION,'teacher_checkpoint':m['teacher_checkpoint'],'teacher_sha256':m['teacher_sha256'],'parent_spec':m['parent_spec'],'parent_spec_sha256':m['parent_spec_sha256'],
    'dataset_sha256':sha(dataset),'replay_sha256':sha(replay) if replay else None,'cohorts':cohorts,'student_architecture':[31,128,64,32,6],'method':'command-phase/episode balanced physical-scaled action BC; std frozen; no inherited PPO optimizer','validation_metrics':metrics,
    **{k:m[k] for k in ('solver_profile','plant_profile','objective') if k in m}},'capability_certified':False}
 try:
  for step in range(1,steps+1):
   if time.monotonic()-started>=wall_s:reason='wall_limit';break
   # Equal draw quota per populated command/phase; exact batch length bounded.
   selected=[]
   quotas=None
   if group_weights is not None:
    exact=group_weights*batch_size;quotas=np.floor(exact).astype(int)
    # Rotate ties while retaining the declared command quota and phase balance.
    order=sorted(range(len(groups)),key=lambda i:(-(exact[i]-quotas[i]),(i-step+1)%len(groups)))
    quotas[order[:batch_size-int(quotas.sum())]]+=1
   for gi,episodes in enumerate(episode_groups):
    quota=int(quotas[gi]) if quotas is not None else batch_size//len(groups)+int((gi-(step-1))%len(groups)<batch_size%len(groups))
    chosen=torch.randint(len(episodes),(quota,),generator=g)
    for ep in chosen.unique():
     pool=episodes[int(ep)];n=int((chosen==ep).sum());selected.append(pool[torch.randint(len(pool),(n,),generator=g)])
   ids=torch.cat(selected);visits.index_add_(0,ids,torch.ones_like(ids));student.train();optimizer.zero_grad(set_to_none=True);value=loss(student.mean(x[ids]),y[ids]);value.backward()
   if not torch.isfinite(value):raise ValueError('nonfinite BC loss')
   torch.nn.utils.clip_grad_norm_(student.mlp.parameters(),1.);optimizer.step();actual=step
   if step==1 or step%validation_every==0 or step==steps:
    metrics=evaluate();score=sum(v['loss'] for v in metrics.values())/len(metrics);history.append({'step':step,'validation_loss':score,'per_cohort':metrics})
    if score<best-1e-8:best=score;bad=0;torch.save(checkpoint(step,metrics),output/'best_validation.pt')
    else:bad+=1
    if patience is not None and bad>=patience:reason='validation_patience';break
  metrics=evaluate();torch.save(checkpoint(actual,metrics),output/'latest.pt')
  summary={'status':'COMPLETED','offline_optimizer_updates':actual,'online_PPO':0,'wall_s':time.monotonic()-started,'best_validation_loss':best,
   'best_checkpoint':str(output/'best_validation.pt'),'history':history,'stop_reason':reason,'batch_size':batch_size,'seed':seed,'threads':4,'learning_rate':.001,
   'cohort_sampling_weights':m.get('cohort_sampling_weights'),
   'sampling':{r['case_id']:{'unique_available':int(((data['cohort']==c)&(split==0)).sum()),'draws':int(visits[data['cohort']==c].sum()),'unique_drawn':int((visits[data['cohort']==c]>0).sum()),'repeat_draws':int(visits[data['cohort']==c].sum()-(visits[data['cohort']==c]>0).sum())} for c,r in enumerate(cohorts)},
   'closed_loop':'NOT_RUN','capability_certified':False,'teacher_unchanged':sha(m['teacher_checkpoint'])==m['teacher_sha256']}
  write_json(output/'summary.json',summary);return summary
 finally:
  b={'status':'COMPLETED' if (output/'summary.json').exists() else 'FAILED','online_PPO':0,'used':actual,'offline_step_limit':steps,'wall_limit_s':wall_s,'wall_s':time.monotonic()-started}
  (output/'budget.json').write_text(json.dumps(b,indent=2))


def mujoco_command(student,parent_spec,output,wall_s=600,solver_profile='legacy_xml_v1'):
 # Historical seven-command workflow v1 pins its original plant. New current-MCU
 # precision/fidelity requests use explicit cycle_v3, not this old protocol.
 data=torch.load(student,map_location='cpu',weights_only=False);validate_student(data,parent_spec)
 return [sys.executable,str(Path(__file__).resolve().parents[3]/'scripts/validate_native_mujoco.py'),
 '--spec',str(Path(parent_spec).resolve()),'--checkpoint',str(Path(student).resolve()),'--output-dir',str(Path(output).resolve()),
 '--commands','0','.5','1','1.5','2','--yaw-commands','.5','-.5','--wall-seconds',str(wall_s),
 '--solver-profile',solver_profile,'--plant-profile','source_xml_v1','--run']


def collect_replay(run,teacher_checkpoint,parent_spec,output):
 """Frozen Teacher labels on actual Student-driven MuJoCo states. No teacher control."""
 run=Path(run).resolve();output=Path(output).resolve()
 if output.exists():raise FileExistsError(output)
 terminal=json.loads((run/'terminal_report.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
 if terminal['status']!='COMPLETED' or not terminal['model_unchanged']:raise ValueError('bounded Student rollout not settled')
 if not manifest.get('distilled_student'):raise ValueError('replay requires declared pure Student rollout')
 if not manifest.get('input_sha256'):raise ValueError('Student rollout input identities missing')
 code_drift=[]
 for path,digest in manifest['input_sha256'].items():
  current=sha(path) if Path(path).exists() else None
  if current!=digest:
   if Path(path).suffix=='.py':code_drift.append({'path':path,'recorded_sha256':digest,'current_sha256':current,'scope':'historical producer code; replay consumes saved inputs, does not reexecute it'})
   else:raise ValueError('Student rollout input changed:'+path)
 teacher,_,_=validate_teacher(teacher_checkpoint,parent_spec)
 if manifest['distilled_student']['teacher_sha256']!=sha(teacher_checkpoint):raise ValueError('replay Teacher mismatch')
 xs=[];cs=[];ps=[];provenance={}
 for i,c in enumerate(COMMANDS):
  ep=run/f'command_{i:02}.json';tp=run/f'command_{i:02}.npz'
  if not ep.exists() or not tp.exists():continue
  d=json.loads(ep.read_text())
  if d['target']!=c[:2]:raise ValueError('replay command order differs')
  with np.load(tp) as a:
   x=a['actor_input'];t=a['time'];keep=np.isfinite(x).all(-1)
   x=x[keep];t=t[keep]
   if not len(x):continue
   idx=np.linspace(0,len(x)-1,min(1500,len(x))).astype(int);xs.append(torch.tensor(x[idx],dtype=torch.float32));cs.extend([i]*len(idx));ps.extend(np.select([t[idx]<2,t[idx]<10,t[idx]<18,t[idx]<26],[0,1,2,3],default=4))
  provenance[str(ep)]=sha(ep);provenance[str(tp)]=sha(tp)
 if not xs:raise ValueError('no finite Student states')
 x=torch.cat(xs)
 with torch.inference_mode():y=torch.cat([teacher.mean(b) for b in x.split(1024)])
 output.mkdir(parents=True,exist_ok=False);torch.save({'x':x,'y':y,'cohort':torch.tensor(cs),'phase':torch.tensor(ps),'split':torch.zeros(len(x),dtype=torch.long)},output/'replay.pt')
 write_json(output/'replay_manifest.json',{'version':VERSION,'teacher_sha256':sha(teacher_checkpoint),'replay_sha256':sha(output/'replay.pt'),'source_inputs_sha256':provenance,
 'historical_source_code_drift':code_drift,'producer_input_sha256':manifest['input_sha256'],'scope':'MuJoCo Student states queried with native-trained Teacher; labels not guaranteed correct out of Teacher domain','teacher_control':False,'capability_certified':False,'simulation':0})
 return {'samples':len(x),'replay':str(output/'replay.pt'),'teacher_control':False}


def export_onnx(student_path,spec_path,dataset,output):
 """CPU deterministic Actor export; numerical equivalence is not hardware acceptance."""
 import onnx
 from onnx.reference import ReferenceEvaluator
 from .export import DeterministicActor
 saved=torch.load(student_path,map_location='cpu',weights_only=False);identity=validate_student(saved,spec_path)
 dataset=Path(dataset);manifest_path=dataset.with_name('manifest.json');manifest=json.loads(manifest_path.read_text())
 if sha(dataset)!=manifest['dataset_sha256'] or sha(dataset)!=identity['dataset_sha256'] or manifest['teacher_sha256']!=identity['teacher_sha256']:raise ValueError('export dataset lineage differs')
 data=torch.load(dataset,map_location='cpu',weights_only=False)
 # Use actual whole-episode heldout states across every cohort and phase.
 ids=[]
 cohorts=dataset_cohorts(data,manifest)
 for cohort in range(len(cohorts)):
  for phase in range(5):
   group=torch.nonzero((data['cohort']==cohort)&(data['phase']==phase)&(data['split']==1)).flatten()
   if not len(group):
    if manifest.get('command_set_version')=='motion31_explicit_commands_v2':continue
    raise ValueError('heldout cohort/phase missing')
   ids.append(group[torch.linspace(0,len(group)-1,min(16,len(group))).long()])
 x=data['x'][torch.cat(ids)].float();actor=Student(True,version='multiskill_mcu_v2');actor.load_state_dict(saved['actor'],strict=True);model=DeterministicActor(actor).eval();output=Path(output)
 if output.exists():raise FileExistsError(output)
 output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
 with torch.inference_mode():
  expected=model(x).numpy()
  torch.onnx.export(model,torch.zeros(1,31),str(output/'actor.onnx'),input_names=['observation'],output_names=['mean_action'],opset_version=17,dynamo=False)
 graph=onnx.load(output/'actor.onnx');onnx.checker.check_model(graph,full_check=True)
 runtime=ReferenceEvaluator(graph);actual=np.concatenate([runtime.run(None,{'observation':row[None]})[0] for row in x.numpy()]);error=float(np.abs(actual-expected).max())
 if not np.isfinite(actual).all() or error>1e-5:raise ValueError('ONNX numerical mismatch')
 np.savez_compressed(output/'golden_vectors.npz',observation=x.numpy(),pytorch_mean_action=expected,onnx_mean_action=actual)
 result={'version':'motion31_distilled_onnx_v1','source_checkpoint':str(Path(student_path).resolve()),'source_sha256':sha(student_path),'teacher_sha256':identity['teacher_sha256'],'parent_spec':str(Path(spec_path).resolve()),'parent_spec_sha256':sha(spec_path),'dataset_sha256':sha(dataset),'input':{'name':'observation','shape':[1,31],'dtype':'float32','semantics':'unchanged multiskill_mcu_v2 normalized 31D, Motion task'},'output':{'name':'mean_action','shape':[1,6],'dtype':'float32','semantics':'deterministic normalized action, Tanh included exactly once; left hip/active/wheel then right hip/active/wheel'},'architecture':[31,128,64,32,6],'parameters':sum(p.numel() for p in model.parameters()),'opset':17,'verification':{'backend':'ONNX ReferenceEvaluator','golden_count':len(x),'max_abs_error':error,'tolerance':1e-5,'passed':True},'control':{'physics_ms':1,'PD_ms':5,'policy_ms':10,'leg_scale_rad':1.5,'wheel_scale_rad_s':100,'leg_Kp':60,'leg_Kd':2,'wheel_Kv':1,'effort_limits_Nm':[40,40,5.5,40,40,5.5]},'hardware_validated':False,'critic_exported':False,'simulation':0,'PPO_updates':0}
 result['files_sha256']={f.name:sha(f) for f in output.iterdir() if f.is_file()}
 spec=json.loads(Path(spec_path).read_text());control=spec.get('control',{})
 if control:
  result['control']={'physics_ms':control['physics_dt_s']*1000,'PD_ms':control['pd_dt_s']*1000,'policy_ms':control['policy_dt_s']*1000,
   'leg_scale_rad':spec['action']['leg_scale_rad'],'wheel_scale_rad_s':spec['action']['wheel_scale_rad_s'],
   'leg_Kp':control['leg_kp'],'leg_Kd':control['leg_kd'],'wheel_Kv':control['wheel_kv'],
   'effort_limits_Nm':control.get('effort_limits_nm',[40,40,5.5,40,40,5.5])}
 result['mcu_execution']=spec.get('mcu_execution');result['cohorts']=cohorts
 result.update(actor_onnx_sha256=sha(output/'actor.onnx'),spec_sha256=sha(spec_path),
  teacher=identity['teacher_checkpoint'],student=str(Path(student_path).resolve()),student_sha256=sha(student_path))
 result['FP32_weight_bytes']=result['parameters']*4;result['linear_MACs']=14400
 result.update({k:identity[k] for k in ('solver_profile','plant_profile','objective') if k in identity})
 result['verification']['max_physical_leg_error_rad']=float(np.abs(actual[:,[0,1,3,4]]-expected[:,[0,1,3,4]]).max()*1.5)
 result['verification']['max_physical_wheel_error_rad_s']=float(np.abs(actual[:,[2,5]]-expected[:,[2,5]]).max()*100)
 result['operators']=sorted({node.op_type for node in graph.graph.node});result['onnx_bytes']=(output/'actor.onnx').stat().st_size
 write_json(output/'manifest.json',result)
 (output/'README.md').write_text('# Motion31 distilled Student ONNX\n\n来源与逐项数值校验见manifest.json。输入float32 [1,31] observation，输出float32 [1,6] mean_action，确定性Tanh已包含，禁止再次Tanh或加入探索噪声。\n\n输入字段/坐标/关节零位必须沿用multiskill_mcu_v2与现有电控协议；本包不重新定义电机正负方向。腿输出乘1.5rad并按原参考位生成位置目标；轮输出乘100rad/s，经原Kv及限矩执行。31D上一执行目标缓存须保存实际执行值，不能改存原始动作。100Hz policy/200Hz PD不得按推理速度任意提高。\n\n仅导出Actor，无Critic/Adam/std。golden_vectors.npz用于下位机同输入对照；数值等价不代表实机或全部姿态约束通过。该Student的闭环结果与几何指标须查同模型评估记录。\n')
 return result
