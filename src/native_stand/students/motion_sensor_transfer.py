"""Explicit standing transfer experiment: delayed sensing and measured-scale costs."""
import copy,json
from pathlib import Path
import torch
from . import motion_fixed_timing as fixed
from .data import sha
VERSION='motion31_standing_sensor_transfer_v3'
FIELDS=('transfer_leg_target_rate','transfer_wheel_target_rate','transfer_leg_velocity',
        'transfer_vertical_velocity','transfer_body_rate','transfer_effort')
WEIGHTS=(-.05,-.02,-.005,-.5,-.05,-.08)

def configuration():
 c=fixed.configuration();c.update(version=VERSION,
  actor_sensor={'version':'motion31_sensor_packet_delay_v1','sample_period_ms':10,'imu_delay_ms':10,'joint_delay_ms':10,
   'jitter_ms':0,'noise':False,'sampling':'100Hz synchronous packets; fixed unmeasured hypothesis',
   'previous':'PD adopted target, not delayed as sensor','critic':'110D unchanged; delayed Actor prefix plus current privileged state'},
  reward_extension={'version':'motion31_transfer_smoothness_cost_v1','fields':list(FIELDS),'weights':list(WEIGHTS),
   'leg_target_scale_rad':.1,'wheel_target_scale_rad_s':10.,'dt_s':.01,'formula':'squared physical costs; dt once; base effort -.02 plus extension -.08 = -.10',
   'source':'V14 leg velocity/vertical velocity/body rate and separate action smoothness idea; adapted means/scales, not identical weights'})
 return c
commands=fixed.commands
stage_config=fixed.stage_config
protocol=fixed.protocol
cases=fixed.cases
effective_spec=fixed.effective_spec
extend_mixture=fixed.extend_mixture
choose=fixed.choose
refinement_records=fixed.refinement_records
advance=fixed.advance

class SensorPackets:
 """One-step100Hz observation delay; repeated getters are idempotent; per-env flush."""
 def __init__(self,n,device,*,delay_ms=10):
  if type(delay_ms) is not int or delay_ms not in (10,20):raise ValueError('explicit10/20ms sensor delay required')
  self.delay_ms=delay_ms;self.delay_packets=delay_ms//10
  self.values=torch.zeros(self.delay_packets+2,n,16,device=device);self.ready=torch.zeros(n,dtype=torch.bool,device=device)
  self.birth_tick=torch.zeros(n,device=device,dtype=torch.long)
  self.age=torch.zeros(n,device=device);self.last_tick=None;self.result=torch.zeros(n,16,device=device)
 def reset(self,ids):self.ready[ids]=False;self.age[ids]=0
 def read(self,current,tick):
  if current.shape!=self.result.shape:raise ValueError('sensor packet16 shape differs')
  # Lazy fill uses actual initialized measurements, not zero gravity or prior episode.
  fresh=~self.ready
  self.birth_tick.copy_(torch.where(fresh,self.birth_tick.new_tensor(tick),self.birth_tick))
  self.values.copy_(torch.where(fresh[None,:,None],current[None],self.values))
  self.result.copy_(torch.where(fresh[:,None],current,self.result));self.ready.fill_(True)
  if tick!=self.last_tick:
   if self.last_tick is not None and tick-self.last_tick!=10:raise ValueError('sensor packet policy clock differs from10ms')
   slot=(tick//10)%len(self.values);self.values[slot]=current
   self.result.copy_(self.values[(slot-self.delay_packets)%len(self.values)])
   self.age.copy_((tick-self.birth_tick).clamp(0,self.delay_ms).float())
   self.last_tick=tick
  return self.result

 def pending(self):
  """Already sampled packets waiting to reach Actor; newest first, no new sample."""
  if self.last_tick is None:raise ValueError('sensor packets have not been sampled')
  ages=torch.arange(self.delay_packets,device=self.values.device)
  slots=(self.last_tick//10-ages)%len(self.values)
  return self.values[slots].permute(1,0,2)

def apply_costs(result,*,targets,previous,dq,vertical_velocity,angular_body,torque,config):
 if config!=configuration()['reward_extension']:raise ValueError('transfer reward configuration differs')
 raw=torch.stack((((targets[:,[0,1,3,4]]-previous[:,[0,1,3,4]])/.1).square().mean(-1),
  ((targets[:,[2,5]]-previous[:,[2,5]])/10).square().mean(-1),dq[:,[0,1,3,4]].square().mean(-1),
  vertical_velocity.square(),angular_body[:,[0,2]].square().sum(-1),
  (torque/torque.new_tensor([40,40,5.5,40,40,5.5])).square().mean(-1)),-1)
 raw=raw*result['eligible'][:,None];density=raw*raw.new_tensor(WEIGHTS);dt=density*.01
 return {**result,'fields':(*result['fields'],*FIELDS),'raw':torch.cat((result['raw'],raw),-1),
  'density':torch.cat((result['density'],density),-1),'dt':torch.cat((result['dt'],dt),-1),'total':result['total']+dt.sum(-1)}

def reward_snapshot():
 from . import motion_sensor_transfer as module
 rows=[]
 for j,name in enumerate(FIELDS):
  for error in (0.,.01,.05,.1,.5,1.,2.):
   x=torch.tensor(error,dtype=torch.float64,requires_grad=True);value=WEIGHTS[j]*x.square()
   rows.append({'field':name,'normalized_error':error,'weighted_density':float(value.detach()),
    'derivative':float(torch.autograd.grad(value,x)[0]),'dt_contribution':float(value.detach()*.01)})
 return {'configuration':configuration()['reward_extension'],'formula_source_sha256':sha(module.__file__),
         'curves':rows,'note':'physical scaling specified per field; state-error derivative is not PPO convergence proof; square costs are unbounded'}

def build(parent,checkpoint,*,experiment,spec_path):
 p=Path(parent).resolve();cp=Path(checkpoint).resolve();old=json.loads(p.read_text());run=p.parents[3]/'runs'/old['experiment']
 terminal=json.loads((run/'formal/terminal_report.json').read_text());chain=json.loads((run/'chain_terminal_report.json').read_text());book=json.loads(p.with_name('budget.json').read_text())
 data=torch.load(cp,map_location='cpu',weights_only=False)
 if old.get('motion_timing',{}).get('version')!=fixed.VERSION or chain['status']!='STOPPED_BOUNDED' or terminal.get('stop_kind')!='user_signal' or any(a['status']=='RUNNING' for a in book['attempts'].values()):raise ValueError('settled manually stopped fixed parent required')
 n=book['used']['formal'];used=old['motion_timing']['fixed_parent']['historical_online_updates']+book['used']['smoke']+n
 if (cp!=run/'formal'/f'model_update_{n}.pt' or data['update']!=n or data['spec_sha256']!=sha(p)
     or data['critic_privileged_version']!='motion_timing_privileged110_v1' or n!=terminal['actual_updates']):raise ValueError('exact latest complete stopped checkpoint required')
 remaining=10000-used
 if remaining<=5:raise ValueError('remaining inclusive budget insufficient')
 s=copy.deepcopy(old);s.update(experiment=experiment,source_checkpoint=str(cp),source_sha256=sha(cp))
 s['motion_timing'].pop('fixed_retry',None)
 paths=[p,p.with_name('budget.json'),run/'formal/terminal_report.json',run/'chain_terminal_report.json',cp]
 s['motion_timing'].update(version=VERSION,configuration=configuration(),parent_spec=str(p),parent_spec_sha256=sha(p),source_checkpoint_sha256=sha(cp),
  fixed_parent={'parent_spec':str(p),'parent_run':str(run),'inputs':{str(q):sha(q) for q in paths},'historical_online_updates':used,
   'original_online_limit':10000,'remaining_online_updates':remaining,'physical_state_restored':False,'curriculum_confirmations':'new sensor/reward recipe resets capability confirmation'})
 s['student_training'].update(course_version=VERSION,timing_configuration=configuration(),source_actor_sha256=sha(cp),selection_scope='standing034 under10ms Actor packet delay; smoothness and applied-effort costs; no new speed')
 s['motion_training'].update(spec_path=str(Path(spec_path).resolve()),resume=None)
 s['motion_course']['source_recipe_updates']=data['recipe_updates']
 s['motion_course']['configuration'].update(version=VERSION,formal_updates_max=remaining-5,smoke_updates_max=5,stage_updates_max=remaining-5)
 for row in s['student_evaluation']['cases']:row.update(checkpoint=str(cp),sha256=sha(cp))
 s['budget'].update(smoke_updates=5,formal_updates=remaining-5,formal_samples=(remaining-5)*4096*128,sample_limit=remaining*4096*128)
 s['artifacts']['run_name']=experiment;s['student_dependencies'].update({str(p):sha(p),str(cp):sha(cp)})
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'User requests stopping old chain and next training from actual final checkpoint; add Actor sensing gap and adapted V14 smoothness/effort reward within remaining original10000 budget; GPU0 shared unlimited wall'}
 return s

def verify(spec):
 if spec.get('motion_training',{}).get('wiring_reuse',{}).get('version')=='motion_sensor_reward_schema_recovery_v1':
  return verify_recovery(spec,require_certificate=False)
 p=Path(spec['motion_timing']['parent_spec']);claim=p.parent.parent/'.motion_sensor_transfer_claims'/(sha(p)+'.json')
 if claim.exists() and json.loads(claim.read_text())['destination']!=spec['experiment']:raise ValueError('sensor transfer parent already claimed')
 expected=build(p,spec['source_checkpoint'],experiment=spec['experiment'],spec_path=spec['motion_training']['spec_path'])
 for k in ('authorization','resources','notifications'):expected[k]=spec[k]
 if spec!=expected:raise ValueError('sensor transfer undeclared drift')
 from .motion_timing import validate_recipe
 validate_recipe(spec['student_training'],spec['source_sha256']);return True

RECOVERY_VERSION='motion_sensor_reward_schema_recovery_v1'

def recovery_evidence(parent):
 """Only the settled certificate-consumer mismatch; five real updates stay charged."""
 from native_stand.command_v6.wiring import smoke_certificate
 p=Path(parent).resolve();old=json.loads(p.read_text());run=p.parents[3]/'runs'/old['experiment']
 book=json.loads(p.with_name('budget.json').read_text());chain=json.loads((run/'chain_terminal_report.json').read_text())
 original=json.loads((run/'smoke/wiring_certificate.json').read_text())
 if (old.get('motion_timing',{}).get('version')!=VERSION or old['motion_training'].get('wiring_reuse')
     or chain['status']!='FAILED' or original.get('status')!='FAILED'
     or original.get('errors')!=['Motion reward readback missing'] or book['used']['smoke']!=5 or book['used']['formal']!=0
     or any(a['status']=='RUNNING' or a['phase']=='formal' for a in book['attempts'].values()) or (run/'formal').exists()):
  raise ValueError('only settled five-update reward-schema preformal failure can recover')
 source=json.loads((run/'source_eval/terminal_report.json').read_text())
 if source['status']!='COMPLETED' or source.get('failure') or len(source.get('evaluations',[]))!=len(old['student_evaluation']['cases']):raise ValueError('source phase must be complete')
 corrected=smoke_certificate(run/'smoke',old)
 if corrected['status']!='PASSED':raise ValueError(corrected['errors'])
 if corrected['inputs']!=original['inputs']:raise ValueError('historical smoke evidence changed since original certificate')
 paths=[p,p.with_name('budget.json'),run/'chain_terminal_report.json',run/'smoke/wiring_certificate.json',
        run/'source_eval/terminal_report.json',run/'source_eval/candidate_registry.json']
 paths += [run/'smoke'/k for k in corrected['inputs']]
 for expected_case,record in zip(old['student_evaluation']['cases'],source['evaluations']):
  path=run/'source_eval'/record['path'];raw=json.loads(path.read_text());trace=path.parent/raw['trace']['path']
  if sha(path)!=record['sha256'] or sha(trace)!=raw['trace']['sha256'] or raw['checkpoint_sha256']!=old['source_sha256'] or raw.get('case')!=expected_case or len(raw.get('per_env',[]))!=8 or raw.get('model_unchanged') is not True:raise ValueError('source evidence identity differs')
  paths.extend([path,trace])
 used=old['motion_timing']['fixed_parent']['historical_online_updates']+5
 return {'version':RECOVERY_VERSION,'parent_spec':str(p),'parent_run':str(run),'inputs':{str(q):sha(q) for q in paths},
  'historical_smoke_updates':5,'historical_formal_updates':0,'historical_online_updates':used,'original_online_limit':10000,
  'remaining_formal_updates':10000-used,'historical_wall_s':chain['chain_wall_s'],'corrected_certificate':corrected,
  'capability_accepted':False,'resume_source':'original formal526 full checkpoint; smoke model is not formal initialization'}

def build_recovery(parent,checkpoint,*,experiment,spec_path):
 import hashlib
 p=Path(parent).resolve();old=json.loads(p.read_text());e=recovery_evidence(p);cp=Path(checkpoint).resolve()
 if str(cp)!=old['source_checkpoint'] or sha(cp)!=old['source_sha256']:raise ValueError('schema recovery must retain original formal526 source')
 cert_path=Path(spec_path).resolve().with_name('corrected_wiring_certificate.json')
 content=(json.dumps(e['corrected_certificate'],indent=2,allow_nan=False)+'\n').encode()
 e.update(corrected_certificate_path=str(cert_path),corrected_certificate_sha256=hashlib.sha256(content).hexdigest())
 s=copy.deepcopy(old);s.update(experiment=experiment)
 s['motion_training'].update(spec_path=str(Path(spec_path).resolve()),wiring_reuse=e,resume=None)
 s['motion_timing'].update(parent_spec=str(p),parent_spec_sha256=sha(p),
  fixed_parent={**old['motion_timing']['fixed_parent'],'parent_spec':str(p),'parent_run':e['parent_run'],
                'historical_online_updates':e['historical_online_updates'],'remaining_online_updates':e['remaining_formal_updates'],
                'inputs':e['inputs']})
 s['motion_course']['configuration']['smoke_updates_max']=0
 s['budget'].update(smoke_updates=0,sample_limit=e['remaining_formal_updates']*4096*128)
 s['artifacts']['run_name']=experiment
 s['student_dependencies'].update({str(p):sha(p)})
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'Existing user authorization for automatic same-recipe wiring repair; preserve real5 smoke updates and total817/10000; formal9183 no repeated smoke; GPU0 shared, unlimited training wall'}
 return s

def verify_recovery(spec,*,require_certificate=True):
 c=spec['motion_training']['wiring_reuse'];parent=Path(c['parent_spec'])
 claim=parent.parent.parent/'.motion_sensor_schema_recovery_claims'/(sha(parent)+'.json')
 if claim.exists() and json.loads(claim.read_text())['destination']!=spec['experiment']:raise ValueError('schema recovery already claimed')
 expected=build_recovery(parent,spec['source_checkpoint'],experiment=spec['experiment'],spec_path=spec['motion_training']['spec_path'])
 for k in ('authorization','resources','notifications'):expected[k]=spec[k]
 if expected!=spec:raise ValueError('schema recovery changed frozen learning configuration')
 if require_certificate:
  from native_stand.command_v6.wiring import validate_persisted_certificate
  cert=Path(c['corrected_certificate_path'])
  if sha(cert)!=c['corrected_certificate_sha256']:raise ValueError('corrected certificate SHA differs')
  old=json.loads(parent.read_text())
  if validate_persisted_certificate(Path(c['parent_run'])/'smoke',old,certificate_path=cert)['status']!='PASSED':raise ValueError('corrected smoke certificate invalid')
 return {'status':'VERIFIED','historical_online_updates':c['historical_online_updates'],
  'remaining_formal_updates':c['remaining_formal_updates'],'capability_accepted':False}
