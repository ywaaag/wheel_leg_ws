"""Explicit standing anti-oscillation shaping and evidence-gated inertia course."""
import copy,json
from pathlib import Path
import torch
from .data import sha
from . import motion_sensor_transfer as sensor
VERSION='motion31_standing_antiosc_armature_v4'
ARMATURES=(.35,.11,.04,.02,.012)
FIELDS=('wheel_request_excess','wheel_torque_change','leg_target_curvature','wheel_target_curvature','support_force_deficit')
WEIGHTS=(-.25,-.10,-.015,-.03,-.5)

def configuration():
 c=sensor.configuration();c.update(version=VERSION,armature_levels_kg_m2=list(ARMATURES),smooth_mix=0.,
  sampling='fixed_env_quota_nominal25_previous25_current50; armature set once at environment setup',
  shaping={'version':'motion_antiosc_huber_v1','fields':list(FIELDS),'weights':list(WEIGHTS),
   'request_inner_fraction':.8,'request_wide_scale':2.,'request_narrow_scale':1.,
   'maximum_narrow_mix':.75,'leg_curvature_scale_rad':.1,'wheel_curvature_scale_rad_s':10.,
   'contact_force_floor_N':2.,'dt_s':.01,'sampling':'pre-reset100Hz; previous control snapshots at policy boundaries; no physical-step CPU sync'})
 return c

def validate_configuration(c):
 if c.get('standing_delay_execution'):
  from .motion_standing_delay import validate_timing
  return validate_timing(c)
 expected=configuration();mix=c.get('smooth_mix');expected['smooth_mix']=mix
 if c.get('wheel_execution') is not None:
  from .motion_wheel_matching import validate_profile
  expected['wheel_execution']=validate_profile(c['wheel_execution']);expected['wheel_kv']=expected['wheel_execution']['wheel_kv_Nm_per_rad_s']
  expected['actor_sensor']['critic']='Actor31/Critic140 unchanged; delayed Actor prefix and current physical privileged columns'
 if c!=expected or type(mix) not in (int,float) or not 0<=mix<=.75:raise ValueError('anti-oscillation configuration differs')
 return c

def huber(x):
 x=x.abs();return torch.where(x<=1,.5*x.square(),x-.5)

def apply_costs(result,request,torque,previous_torque,targets,previous_targets,older_targets,contacts,history_valid,config):
 validate_configuration(config);c=config['shaping'];mix=config['smooth_mix']
 excess=torch.relu(request[:,[2,5]].abs()/5.5-c['request_inner_fraction'])
 request_cost=((1-mix)*huber(excess/c['request_wide_scale'])+mix*huber(excess/c['request_narrow_scale'])).mean(-1)
 delta=(torque[:,[2,5]]-previous_torque[:,[2,5]])/5.5
 curvature=targets-2*previous_targets+older_targets
 raw=torch.stack((request_cost,huber(delta).mean(-1)*history_valid,
  huber(curvature[:,[0,1,3,4]]/.1).mean(-1)*history_valid,
  huber(curvature[:,[2,5]]/10.).mean(-1)*history_valid,
  huber(torch.relu(1-contacts/2.)).mean(-1)),-1)
 raw*=result['eligible'][:,None];density=raw*raw.new_tensor(WEIGHTS);dt=density*.01
 return {**result,'fields':(*result['fields'],*FIELDS),'raw':torch.cat((result['raw'],raw),-1),
  'density':torch.cat((result['density'],density),-1),'dt':torch.cat((result['dt'],dt),-1),'total':result['total']+dt.sum(-1)}

commands=sensor.commands
protocol=sensor.protocol
extend_mixture=sensor.extend_mixture

def stage_config(stage=0,envs=4096):
 if stage not in range(5) or envs not in (1,64,4096):raise ValueError('inertia course stage/env invalid')
 r=sensor.stage_config(0,envs);r.update(id=stage,name=f'standing_armature_{ARMATURES[stage]:g}');return r

def cases(spec,stage=0):
 levels=sorted({0,max(0,stage-1),stage});template=spec['student_evaluation']['cases'][0]
 return [{**template,'case_id':f'armature_{stage}_level_{level}','_timing_level':level,'command':[0.,0.,.34]} for level in levels]

def effective_spec(spec,stage,state=None):
 s=copy.deepcopy(spec);s['motion_training']['stage']=stage;s['student_training'].update(stage=stage,course_stage=stage)
 s['motion_timing']['stage']=stage;s['curriculum']['stages']=[stage_config(stage,s['ppo']['envs'])]
 s['student_evaluation']['cases']=cases(s,stage)
 s['motion_timing']['configuration']['smooth_mix']=(state or {}).get('smooth_mix',0.)
 s['student_training']['timing_configuration']=copy.deepcopy(s['motion_timing']['configuration'])
 return s

def refinement_records(records,stage):
 expected={f'armature_{stage}_level_{level}':level for level in sorted({0,max(0,stage-1),stage})}
 if len(records)!=len(expected) or {r.get('case',{}).get('case_id') for r in records}!=set(expected):return [{'case':{'command':[0.,0.,.34]},'per_env':[]}]
 for r in records:
  if r['case'].get('_timing_level')!=expected[r['case']['case_id']] or r['case'].get('command')!=[0.,0.,.34] or len(r.get('per_env',[]))!=8 or {e.get('env') for e in r['per_env']}!=set(range(8)):return [{'case':{'command':[0.,0.,.34]},'per_env':[]}]
 rows=[]
 for eid in range(8):
  es=[next(e for e in r['per_env'] if e['env']==eid) for r in records];row=copy.deepcopy(es[0]);row['accepted']=all(e.get('accepted') is True for e in es)
  keys=set.intersection(*(set(e.get('symmetry_metrics',{})) for e in es));row['symmetry_metrics']={k:max(e['symmetry_metrics'][k] for e in es) for k in keys};rows.append(row)
 return [{'case':{'command':[0.,0.,.34]},'per_env':rows}]

def advance(state,candidate):
 s=copy.deepcopy(state);stage=s['stage']
 if candidate.get('stage')!=stage or candidate['update']<=s.get('last_evaluation_update',-1):raise ValueError('inertia evidence stage/update differs')
 rows=refinement_records(candidate.get('evaluations',[]),stage)[0]['per_env']
 passed=candidate.get('complete_matrix') is True and len(rows)==8 and all(e.get('accepted') is True for e in rows)
 s['last_evaluation_update']=candidate['update'];s['consecutive_passes']=s.get('consecutive_passes',0)+1 if passed else 0
 s['smooth_mix']=min(.75,s.get('smooth_mix',0.)+.25) if passed else max(0.,s.get('smooth_mix',0.)-.25)
 advanced=False
 if s['consecutive_passes']>=2:
  s['accepted_by_stage'][str(stage)]={k:candidate[k] for k in ('checkpoint','sha256','update')}
  if stage<4:s.update(stage=stage+1,stage_started_update=candidate['update'],consecutive_passes=0,smooth_mix=0.);advanced=True
 return s,{'stage_passed':passed,'advanced':advanced,'all_stages_passed':stage==4 and s['consecutive_passes']>=2,'stop':False,'smooth_mix':s['smooth_mix'],'progression':'full acceptance AND anti-oscillation checks, two separate checkpoints; no update-count promotion'}

def choose(candidates,accepted_by_stage):
 from .motion_course import choose as base
 result=base(candidates,accepted_by_stage)
 # Generic course has four stages; this explicitly versioned course has five.
 final=accepted_by_stage.get('4');result['accepted_candidate']=next((c for c in candidates if c.get('stage')==4 and final and c['sha256']==final['sha256']),None)
 return result

def assess(result,control,valid,state):
 """Native full-window physical gates remain; add hold-window smoothness proof."""
 import numpy as np
 from native_stand.command_v6.domain.trace_schema import CONTROL
 lo,hi=1000,1800 # controlled evaluation clock, shared protocol hold10..18s.
 mask=np.asarray(valid).astype(bool)&(state[:,29]>0)
 selected=control[mask] if control is not None else None
 ok=selected is not None and len(selected)>=3000
 metrics={'complete':ok,'window_s':[10.,18.],'sample_rate_hz':100.}
 if ok:
  c=selected[lo:hi];torque=c[:,CONTROL['requested_torque']];target=np.empty((len(c),6))
  target[:,[0,1,3,4]]=c[:,CONTROL['leg_target_after_limit']];target[:,[2,5]]=c[:,CONTROL['clipped_action']][:,[2,5]]*100
  metrics.update(wheel_target_delta_rms_rad_s=float(np.sqrt(np.mean(np.diff(target[:,[2,5]],axis=0)**2))),
   leg_target_delta_rms_rad=float(np.sqrt(np.mean(np.diff(target[:,[0,1,3,4]],axis=0)**2))),
   wheel_request_abs_p95_Nm=float(np.quantile(abs(torque[:,[2,5]]),.95)))
 checks={'antiosc_complete':ok,'antiosc_wheel_delta':ok and metrics['wheel_target_delta_rms_rad_s']<=3.,
  'antiosc_leg_delta':ok and metrics['leg_target_delta_rms_rad']<=.03,
  'antiosc_request':ok and metrics['wheel_request_abs_p95_Nm']<=5.5}
 result['antioscillation']=metrics;result['checks'].update(checks)
 result['failed_checks']+= [k for k,v in checks.items() if not v];result['accepted']=bool(result['accepted'] and all(checks.values()))
 return result

def reward_snapshot(config=None):
 actual=configuration() if config is None else validate_configuration(config)
 rows=[]
 for mix in (0.,.25,.5,.75):
  for request in (0.,4.4,5.5,11.,28.,90.):
   x=torch.tensor(request,requires_grad=True);e=torch.relu(x/5.5-.8);cost=-.25*((1-mix)*huber(e/2)+mix*huber(e))
   rows.append({'request_abs_Nm':request,'narrow_mix':mix,'density':float(cost.detach()),'derivative_per_Nm':float(torch.autograd.grad(cost,x)[0]),'dt_contribution':float(cost.detach()*.01)})
 return {'configuration':actual,'formula_source_sha256':sha(__file__),'base_extension':sensor.reward_snapshot(),'curves':rows,
  'interpretation':'non-saturated request tail; not direct policy gradient; unbounded linear costs can dominate; no guarantee of convergence'}

def build(parent,checkpoint,*,experiment,spec_path):
 p=Path(parent).resolve();cp=Path(checkpoint).resolve();old=json.loads(p.read_text());root=p.parents[3];run=root/'runs'/old['experiment']
 terminal=json.loads((run/'formal/terminal_report.json').read_text());book=json.loads(p.with_name('budget.json').read_text());data=torch.load(cp,map_location='cpu',weights_only=False)
 if old.get('motion_timing',{}).get('version')!=sensor.VERSION or terminal.get('stop_kind')!='user_signal' or terminal['status']!='STOPPED_BOUNDED' or any(a['status']=='RUNNING' for a in book['attempts'].values()):raise ValueError('settled sensor transfer parent required')
 if cp!=run/'formal'/f"model_update_{terminal['actual_updates']}.pt" or data['update']!=500 or data['spec_sha256']!=sha(p) or data['critic_privileged_version']!='motion_timing_privileged110_v1':raise ValueError('exact final500 full-state source required')
 s=copy.deepcopy(old);s.update(experiment=experiment,source_checkpoint=str(cp),source_sha256=sha(cp))
 s['motion_training'].update(stage=0,spec_path=str(Path(spec_path).resolve()),resume=None);s['motion_training'].pop('wiring_reuse',None)
 s['motion_timing']={'version':VERSION,'configuration':configuration(),'stage':0,'parent_spec':str(p),'parent_spec_sha256':sha(p),'source_checkpoint_sha256':sha(cp)}
 s['student_training'].update(course_version=VERSION,timing_configuration=configuration(),source_actor_sha256=sha(cp),stage=0,course_stage=0,selection_scope='standing034 anti-oscillation and delay robustness; evidence-gated armature .35 to .012; no speed certification')
 s['motion_course']['source_recipe_updates']=data['recipe_updates'];s['motion_course']['configuration'].update(version=VERSION,formal_updates_max=9995,smoke_updates_max=5,stage_updates_max=9995)
 s['budget'].update(smoke_updates=5,formal_updates=9995,formal_samples=9995*4096*128,sample_limit=10000*4096*128,source_eval_wall_s=3600,evaluation_wall_s=7200,formal_wall_s=None,chain_wall_s=None)
 s['artifacts']['run_name']=experiment;s['curriculum']['stages']=[stage_config()]
 s['student_evaluation']['cases']=cases(s)
 for row in s['student_evaluation']['cases']:row.update(checkpoint=str(cp),sha256=sha(cp))
 s['motion_startup'].update(references=[],missing_commands=[[0.,0.,.34]])
 s['student_dependencies']={str(p):sha(p),str(cp):sha(cp),s['asset_path']:s['asset_sha256']}
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'User approves new10000 inclusive online updates from final500, no training wall limit, anti-oscillation shaping and full-evidence inertia curriculum .35/.11/.04/.02/.012, GPU0 shared; old1317 remains historical and old8683 not transferred'}
 s['notifications']={'enabled':True}
 return s

def verify(spec):
 if spec['motion_timing'].get('wheel_match_restart'):
  from .motion_wheel_matching import verify as verify_wheel
  return verify_wheel(spec)
 if spec['motion_timing'].get('critic_upgrade'):
  from .motion_antiosc_privileged import verify as verify_upgrade
  return verify_upgrade(spec)
 t=spec['motion_timing'];expected=build(t['parent_spec'],spec['source_checkpoint'],experiment=spec['experiment'],spec_path=spec['motion_training']['spec_path'])
 for k in ('authorization','resources','notifications'):expected[k]=spec[k]
 if expected!=spec:raise ValueError('anti-oscillation undeclared spec drift')
 from .motion_timing import validate_recipe
 validate_recipe(spec['student_training'],spec['source_sha256']);return True
