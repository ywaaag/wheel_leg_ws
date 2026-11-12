FROZEN_TEST_SCOPE='PORTABLE'
import pytest
import torch
from native_stand.students.motion_sensor_transfer import SensorPackets,configuration,apply_costs,FIELDS,WEIGHTS,reward_snapshot

def test_sensor_exact_past_packet_idempotence_and_tail_untouched():
 b=SensorPackets(2,'cpu');first=torch.arange(32,dtype=torch.float32).reshape(2,16)
 rng=torch.get_rng_state().clone()
 assert torch.equal(b.read(first,0),first)
 assert torch.equal(b.read(first+1,10),first)
 assert torch.equal(b.read(first+2,10),first) # repeated getter does not enqueue a second packet.
 assert torch.equal(b.read(first+3,20),first+1)
 assert torch.equal(rng,torch.get_rng_state())
 with pytest.raises(ValueError):b.read(first,25)


def test_reset_one_env_never_replays_prior_episode_or_zero_gravity():
 b=SensorPackets(2,'cpu');x=torch.ones(2,16)
 b.read(x,0);b.read(x*2,10)
 b.reset(torch.tensor([0]));current=x*5
 out=b.read(current,10).clone()
 assert torch.equal(out[0],current[0]) and torch.equal(out[1],x[1])
 out=b.read(current*2,20)
 assert torch.equal(out[0],current[0]) and torch.equal(out[1],x[1]*2)


def test_transfer_cost_direction_scale_dt_mask_and_gradient():
 n=2;raw=torch.zeros(n,1);base={'fields':('base',),'raw':raw,'density':raw.clone(),'dt':raw.clone(),
 'total':torch.zeros(n),'eligible':torch.tensor([True,False])}
 targets=torch.zeros(n,6,requires_grad=True);previous=torch.zeros(n,6)
 with torch.no_grad():targets[:,[0,1,3,4]]=.1;targets[:,[2,5]]=10
 args=dict(targets=targets,previous=previous,dq=torch.ones(n,6),vertical_velocity=torch.ones(n),
 angular_body=torch.ones(n,3),torque=torch.tensor([40,40,5.5,40,40,5.5]).expand(n,-1),config=configuration()['reward_extension'])
 out=apply_costs(base,**args)
 assert out['fields']==('base',*FIELDS)
 expected=torch.tensor([1.,1.,1.,1.,2.,1.])
 assert torch.allclose(out['raw'][0,1:],expected) and (out['raw'][1]==0).all()
 assert torch.allclose(out['total'][0],(expected*torch.tensor(WEIGHTS)).sum()*.01)
 g=torch.autograd.grad(out['total'].sum(),targets)[0]
 assert (g[0]<0).all() and (g[1]==0).all()
 bad=dict(args);bad['config']={**args['config'],'dt_s':1}
 with pytest.raises(ValueError):apply_costs(base,**bad)
 snap=reward_snapshot();assert snap['formula_source_sha256'] and len(snap['curves'])==42
 assert all(row['derivative']<0 for row in snap['curves'] if row['normalized_error']>0)


def test_new_course_dispatch_and_recipe_legacy_defaults_unchanged():
 from native_stand.students.motion_course import implementation
 from native_stand.students.motion_timing import configuration as legacy
 from native_stand.students.motion_fixed_timing import configuration as fixed
 from native_stand.students.motion_sensor_transfer import VERSION
 assert implementation(VERSION).configuration()==configuration()
 assert legacy()['version']=='motion31_standing_timing_course_v1'
 assert 'actor_sensor' not in fixed() and 'actor_sensor' not in legacy()


def test_real_timing_wrapper_delays_only_sensors_and_preserves_base_cache():
 from types import SimpleNamespace as S
 from test_motion_timing import fake_env
 from native_stand.students.motion_timing_runtime import subclass
 class Base:
  def __init__(self):
   self.__dict__.update(fake_env(2).__dict__);self.cfg=S(leg_kp=40.,leg_kd=2.,wheel_kv=1.)
   self.critic_privileged_version='motion_timing_privileged110_v1'
   self.robot.joint_names=list('abcdef')
   self.robot.root_view=S(get_dof_armatures=lambda:torch.tensor([[.35,.35,.01,.35,.35,.01]]).expand(2,-1))
   self.cache={'student_actor':torch.zeros(2,31),'student_critic':torch.zeros(2,85)}
  def _get_observations(self):return self.cache
  def _reset_idx(self,ids):pass
 e=subclass(Base,configuration(),0)();e.cache['student_actor'][:,3:6]=torch.tensor([0.,-1.,0.])
 first=e._get_observations();assert first['student_critic'].shape==(2,110)
 e.motion_timing.tick=10;e.cache['student_actor'][:,:3]=1.;e.cache['student_actor'][:,16:]=2.
 out=e._get_observations()
 assert (out['student_actor'][:,:3]==0).all() and (out['student_actor'][:,16:]==2).all()
 assert (e.cache['student_actor'][:,:3]==1).all() # no delayed feedback into next packet or safety truth.
 assert torch.equal(out['student_critic'][:,:31],out['student_actor'])
 e.motion_timing.tick=20;e.cache['student_actor'][:,:3]=3.
 assert (e._get_observations()['student_actor'][:,:3]==1).all()


def test_certificate_consumes_exact_runtime_reward_schema_and_preserves_old_failure(tmp_path):
 import json
 from native_stand.students.training_evidence import smoke_certificate
 from native_stand.students.motion_tracking_reward import effective_reward_fields,reward_fields,PARALLEL_VERSION
 from native_stand.students.motion_training import declaration
 from native_stand.students.motion_startup import smoke_profile
 from native_stand.students.data import sha
 from native_stand.command_v6.wiring import validate_persisted_certificate
 from native_stand.students.motion_sensor_transfer import VERSION
 recipe=declaration(reward_version=PARALLEL_VERSION);recipe['smoke_wiring']=smoke_profile()
 spec={'source_sha256':'a'*64,'student_training':recipe,'reward':recipe['reward'],'motion_startup':{'version':'motion31_minimal_startup_v1'},
  'motion_training':{'stage':0},'motion_timing':{'version':VERSION,'configuration':configuration()},'budget':{'smoke_updates':5},
  'ppo':{'envs':512,'steps_per_env':128},'curriculum':{'stages':[{'commands':[[0,0]]}]}}
 def write(name,d):
  p=tmp_path/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(d))
 write('source_snapshot/spec.json',spec)
 write('manifest.json',{'spec_sha256':sha(tmp_path/'source_snapshot/spec.json'),'source_sha256':'a'*64})
 write('terminal_report.json',{'status':'COMPLETED','actual_updates':5,'failure':None})
 write('progress.json',{'actual_updates':5,'actor_parameters_changed':True,'checkpoint_roundtrip_verified':True,
  'wiring_command_readback':{'valid':True,'sample_count':327680},'independent_capability_evaluation':{'status':'NOT_RUN'}})
 write('wiring_runtime_readback.json',{'asset':{'ok':True},'control':{'ok':True},'ground':{'ok':True}})
 rows=[{'update':u,'controlled_samples':65536,'actor_update_scope':{'valid_actor_samples':65536,'invalid_actor_samples':0,'actor_optimizer_steps':16 if u==5 else 0},
 'actor_changed_this_update':u==5,'critic_changed_this_update':True,'nonzero_rewards':65536,'episode_completed':0,
 'BC_loss':0,'BC_gradient_norm':None,'BC_updates':0,'std_unchanged':True,'reward_fields':list(effective_reward_fields(spec)),
 'hold_sample_counts':[1],'command_episode_counts':[0]} for u in range(1,6)]
 def metrics(): (tmp_path/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 metrics();cert=smoke_certificate(tmp_path,spec);assert cert['status']=='PASSED',cert['errors']
 original={**cert,'status':'FAILED','errors':['Motion reward readback missing']};write('wiring_certificate.json',original)
 digest=sha(tmp_path/'wiring_certificate.json');corrected=tmp_path/'corrected.json';corrected.write_text(json.dumps(cert))
 assert validate_persisted_certificate(tmp_path,spec,certificate_path=corrected)['status']=='PASSED'
 assert sha(tmp_path/'wiring_certificate.json')==digest
 rows[-1]['reward_fields']=list(reward_fields(spec['reward']));metrics()
 assert smoke_certificate(tmp_path,spec)['status']=='FAILED' # missing extension is not permitted.
 rows[-1]['reward_fields']=list(effective_reward_fields(spec));rows[-1]['reward_fields'][-1]='unknown_extra';metrics()
 assert smoke_certificate(tmp_path,spec)['status']=='FAILED'
 spec['motion_timing']['configuration']['reward_extension']['weights'][0]=-.5
 with pytest.raises(ValueError):effective_reward_fields(spec)
