"""Explicit fixed-delay standing recipe; selected parameters are not identification."""
import copy,json
from pathlib import Path
from . import motion_timing as base
from .data import sha
VERSION='motion31_standing_fixed_timing_v2'

def configuration():
 return {'version':VERSION,'seed':20261021,'target_delay_ms_ch0_ch5':[10,10,7,10,10,11],
  'PD_feedback_ms':10,'torque_tau_ms_ch0_ch5':[5,5,0,5,5,0],
  'leg_armature_kg_m2':.35,'leg_kp':40.,'leg_kd':2.,'wheel_kv':1.,
  'sampling':'fixed_all_envs_no_nominal_or_previous_cohorts',
  'provenance':'user-selected leg dynamics; leg timings hypotheses; wheel combined onset approximated as transport only, not identified'}

def commands(stage=0):return [[0.,0.]]
def stage_config(stage=0,envs=4096):
 if stage!=0:raise ValueError('fixed timing has one stage')
 return base.stage_config(0,envs)
protocol=base.protocol
choose=base.choose
extend_mixture=base.extend_mixture

def cases(spec,stage=0):
 if stage!=0:raise ValueError('fixed timing has one stage')
 return [{**spec['student_evaluation']['cases'][0],'case_id':'fixed_timing_target','command':[0.,0.,.34],'_timing_level':0}]

def effective_spec(spec,stage):
 if stage!=0:raise ValueError('fixed timing cannot promote')
 return copy.deepcopy(spec)

def refinement_records(records,stage):
 if stage!=0 or len(records)!=1:return [{'case':{'command':[0.,0.,.34]},'per_env':[]}]
 r=records[0];es=r.get('per_env',[])
 if (r.get('case',{}).get('case_id')!='fixed_timing_target' or r['case'].get('command')!=[0.,0.,.34]
     or r['case'].get('_timing_level')!=0 or len(es)!=8 or {e.get('env') for e in es}!=set(range(8))):
  return [{'case':{'command':[0.,0.,.34]},'per_env':[]}]
 return records

def advance(state,candidate):
 s=copy.deepcopy(state)
 if candidate.get('stage')!=0 or candidate['update']<=s.get('last_evaluation_update',-1):raise ValueError('fixed timing stage/update differs')
 rows=refinement_records(candidate.get('evaluations',[]),0)[0]['per_env']
 passed=candidate.get('complete_matrix') is True and len(rows)==8 and all(e.get('accepted') is True for e in rows)
 s['last_evaluation_update']=candidate['update'];s['consecutive_passes']=s.get('consecutive_passes',0)+1 if passed else 0
 if s['consecutive_passes']>=2:s['accepted_by_stage']['0']={k:candidate[k] for k in ('checkpoint','sha256','update')}
 return s,{'stage_passed':passed,'advanced':False,'all_stages_passed':False,'target_condition_confirmed':s['consecutive_passes']>=2,'stop':False}

def apply_actuator(cfg,spec):
 """Copy only four leg actuator settings; never mutate external asset defaults."""
 c=spec.get('motion_timing',{}).get('configuration',{})
 wheel=c.get('wheel_execution')
 from .motion_sensor_transfer import VERSION as SENSOR,configuration as sensor_configuration
 if c.get('version')=='motion31_standing_antiosc_armature_v4':
  from .motion_antioscillation import validate_configuration
  validate_configuration(c);c=configuration()
 if c.get('version') not in (VERSION,SENSOR):return
 expected=sensor_configuration() if c['version']==SENSOR else configuration()
 kv=wheel['wheel_kv_Nm_per_rad_s'] if wheel is not None else 1.
 if c!=expected or [spec['control'].get(k) for k in ('leg_kp','leg_kd','wheel_kv')]!=[40.,2.,kv]:raise ValueError('fixed timing actuator declaration differs')
 from native_stand.contract import JOINT_NAMES,LEG_COLUMNS
 robot=copy.deepcopy(cfg.robot_cfg);legs=robot.actuators['leg_motors']
 if set(legs.joint_names_expr)!={JOINT_NAMES[i] for i in LEG_COLUMNS}:raise ValueError('armature must cover exactly four active leg joints')
 legs.armature=.35;cfg.robot_cfg=robot

def verify_armature(env):
 """Read PhysX after environment creation, once; config values are not readback."""
 import torch
 from native_stand.control_readback import tensor
 values=torch.as_tensor(tensor(env.robot.root_view.get_dof_armatures()))
 ids=[env.ids[i] for i in (0,1,3,4)]
 if values.ndim!=2 or not torch.allclose(values[:,ids],torch.full_like(values[:,ids],.35),atol=1e-6,rtol=0):raise RuntimeError('PhysX leg armature differs from .35')
 return {'joint_names':[env.robot.joint_names[i] for i in ids],'leg_armature_kg_m2':values[0,ids].tolist(),
         'leg_kp':env.cfg.leg_kp,'leg_kd':env.cfg.leg_kd,'scope':'four actuated hip/active joints; wheel and passive configuration preserved'}

def build(parent,checkpoint,*,experiment,spec_path):
 import torch
 p=Path(parent).resolve();cp=Path(checkpoint).resolve();old=json.loads(p.read_text());run=p.parents[3]/'runs'/old['experiment']
 book=json.loads(p.with_name('budget.json').read_text());terminal=json.loads((run/'formal/terminal_report.json').read_text())
 chain=json.loads((run/'chain_terminal_report.json').read_text());data=torch.load(cp,map_location='cpu',weights_only=False)
 if (old.get('motion_timing',{}).get('version')!=base.VERSION or chain['status']!='STOPPED_BOUNDED'
     or terminal.get('stop_kind')!='user_signal' or any(a['status']=='RUNNING' for a in book['attempts'].values())):raise ValueError('settled user-stopped timing parent required')
 n=book['used']['formal'];prior=old['motion_training']['wiring_reuse']['historical_online_updates'];used=prior+book['used']['smoke']+n
 if data['update']!=n or terminal['actual_updates']!=n or data['spec_sha256']!=sha(p) or data.get('critic_privileged_version')!=base.CRITIC or cp!=run/'formal'/f'model_update_{n}.pt':raise ValueError('exact final full-state checkpoint required')
 if used!=281:raise ValueError('this explicit transition requires charged281 updates')
 s=copy.deepcopy(old);s.pop('motion_timing_recovery',None)
 s.update(experiment=experiment,source_checkpoint=str(cp),source_sha256=sha(cp))
 s['control'].update(leg_kp=40.,leg_kd=2.)
 s['motion_training'].update(spec_path=str(Path(spec_path).resolve()),resume=None)
 s['motion_training'].pop('wiring_reuse',None)
 paths=[p,p.with_name('budget.json'),run/'chain_terminal_report.json',run/'formal/terminal_report.json',cp]
 history={'parent_spec':str(p),'parent_run':str(run),'inputs':{str(q):sha(q) for q in paths},
  'historical_online_updates':used,'original_online_limit':10000,'remaining_online_updates':10000-used,
  'physical_state_restored':False,'curriculum_confirmations':'reset for changed dynamics; old nominal passing is not fixed-delay passing'}
 s['motion_timing'].update(version=VERSION,configuration=configuration(),stage=0,parent_spec=str(p),parent_spec_sha256=sha(p),source_checkpoint_sha256=sha(cp),fixed_parent=history)
 r=s['student_training'];r.update(course_version=VERSION,timing_configuration=configuration(),source_actor_sha256=sha(cp),critic='full_motion110',optimizer='source_Adam_full_state',selection_scope='fixed target timing standing034; new PD/armature; no new speed')
 s['motion_course']['source_recipe_updates']=data['recipe_updates'];s['motion_course']['configuration'].update(version=VERSION,formal_updates_max=9714,smoke_updates_max=5,stage_updates_max=9714)
 s['student_evaluation']['cases']=cases(s)
 for row in s['student_evaluation']['cases']:row.update(checkpoint=str(cp),sha256=sha(cp))
 s['motion_startup'].update(references=[],missing_commands=[[0.,0.,.34]])
 s['budget'].update(smoke_updates=5,formal_updates=9714,formal_samples=9714*4096*128,sample_limit=9719*4096*128)
 s['artifacts']['run_name']=experiment
 s['student_dependencies'].update({str(p):sha(p),str(cp):sha(cp)})
 s['authorization']={'status':'UNAUTHORIZED','hapi_session':None,'user_scope':'User requests next fixed-delay standing training, four leg armature0.35 Kp40 Kd2, original remaining9719 inclusive updates, GPU0 shared, unlimited training wall'}
 s['notifications']={'enabled':True}
 return s

def verify(spec):
 retry=spec['motion_timing'].get('fixed_retry')
 if retry and retry!=retry_evidence(retry['parent_spec']):raise ValueError('fixed retry evidence differs')
 parent=Path(retry['parent_spec'] if retry else spec['motion_timing']['parent_spec'])
 claim=parent.parent.parent/'.motion_fixed_timing_claims'/(sha(parent)+'.json')
 if claim.exists() and json.loads(claim.read_text())['destination']!=spec['experiment']:raise ValueError('stopped parent already claimed by another fixed timing experiment')
 expected=build(spec['motion_timing']['parent_spec'],spec['source_checkpoint'],experiment=spec['experiment'],spec_path=spec['motion_training']['spec_path'])
 for k in ('authorization','resources','notifications'):expected[k]=spec[k]
 if retry:expected['motion_timing']['fixed_retry']=retry
 if expected!=spec:raise ValueError('fixed timing experiment has undeclared drift')
 base.validate_recipe(spec['student_training'],spec['source_sha256'])
 return True


def retry_evidence(parent):
 p=Path(parent).resolve();old=json.loads(p.read_text());run=p.parents[3]/'runs'/old['experiment']
 book=json.loads(p.with_name('budget.json').read_text());terminal=json.loads((run/'source_eval/terminal_report.json').read_text())
 chain=json.loads((run/'chain_terminal_report.json').read_text())
 if (old.get('motion_timing',{}).get('version')!=VERSION or chain['status']!='FAILED' or terminal['actual_updates']!=0
     or terminal.get('failure',{}).get('exception')!="ValueError('absolute posture protocol mismatch')"
     or any(book['used'].get(k,0) for k in ('smoke','formal')) or any(a['status']=='RUNNING' for a in book['attempts'].values())):
  raise ValueError('fixed retry only permits settled pre-PPO protocol dispatch failure')
 paths=[p,p.with_name('budget.json'),run/'chain_terminal_report.json',run/'source_eval/terminal_report.json']
 return {'version':'motion_fixed_timing_dispatch_retry_v1','parent_spec':str(p),'parent_run':str(run),
         'inputs':{str(q):sha(q) for q in paths},'additional_online_updates':0,'historical_failed_wall_s':chain['chain_wall_s']}
