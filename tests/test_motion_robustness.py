"""CPU contracts for the explicit alternating robustness successor."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as S,ModuleType
import numpy as np
import pytest
import torch
from native_stand.students import motion_robustness as d


def candidate(s,update,fail=None,unknown=None):
 rows=[]
 for group in range(3):
  for i,v in enumerate(d.commands(s['stage'])):
   rows.append(dict(case=dict(command=v+[.34],_timing_level=4,_motion_dr=dict(level=s['DR_level'],group=group,mode=s['mode'],skill_stage=s['stage'])),
     per_env=[dict(env=j,accepted=None if unknown==(group,i,j) else fail!=(group,i,j),survival_passed=True,tracking={'MAE':.01}) for j in range(8)]))
 return dict(stage=s['stage'],update=update,checkpoint=f'/model{update}.pt',sha256=f'{update:064x}',complete_matrix=True,
  evaluations=rows,per_env=[x for row in rows for x in row['per_env']])


def test_single_confirmation_alternates_and_completes_all_DR():
 s=d.initial_state(0);seen=[]
 for update in range(250,2751,250):
  before=copy.deepcopy(s);s,a=d.advance(s,candidate(s,update));seen.append((s['stage'],s['DR_level'],s['mode']))
  assert s['DR_level']>=before['DR_level'] and s['certified_DR_level']>=before['certified_DR_level']
  assert not (s['stage']>before['stage'] and s['DR_level']>before['DR_level'])
  assert not a['stop']
 assert s['stage']==5 and s['DR_level']==5 and s['mode']=='FINAL_CONSOLIDATE'
 assert seen[:4]==[(1,0,'COMMAND_ACQUIRE'),(1,1,'DR_ACQUIRE'),(2,1,'COMMAND_ACQUIRE'),(2,2,'DR_ACQUIRE')]


def test_old_regression_rolls_commands_back_then_repairs_without_DR_retreat():
 s,_=d.advance(d.initial_state(1),candidate(d.initial_state(1),250))
 assert s['stage']==2 and s['DR_level']==0
 before=copy.deepcopy(s)
 bad=candidate(s,500,fail=(2,1,0));s,a=d.advance(s,bad)
 assert a['old_command_regression'] and s['stage']==1 and s['mode']=='COMMAND_REPAIR'
 assert s['pending_skill_stage']==2 and s['DR_level']==before['DR_level']
 assert d.advance(s,bad)[0]==s
 s,a=d.advance(s,candidate(s,750));assert s['stage']==2 and s['DR_level']==0 and s['mode']=='COMMAND_ACQUIRE'
 s,a=d.advance(s,candidate(s,1000));assert s['stage']==2 and s['DR_level']==1


@pytest.mark.parametrize('kind',['new','unknown','missing'])
def test_unlearned_or_unknown_does_not_become_old_regression(kind):
 s,_=d.advance(d.initial_state(1),candidate(d.initial_state(1),250))
 if kind=='new':c=candidate(s,500,fail=(2,len(d.commands(s['stage']))-1,0))
 elif kind=='unknown':c=candidate(s,500,unknown=(2,0,0))
 else:c=candidate(s,500);c['evaluations'].pop()
 result,action=d.advance(s,c)
 assert result['stage']==s['stage'] and result['DR_level']==s['DR_level'] and result['mode']==s['mode']
 assert not action['stage_passed'] and not action['old_command_regression']
 if kind in ('unknown','missing'):assert result['smooth_mix']==s['smooth_mix']


def test_DR_failure_does_not_rollback_or_stop():
 s=d.initial_state(2);s['qualified_by_DR']['0']=d.commands(2)
 result,a=d.advance(s,candidate(s,250,fail=(2,0,0)))
 assert result['stage']==2 and result['DR_level']==0 and not a['stop'] and not a['advanced']


@pytest.mark.parametrize('stage',range(6))
def test_exact4096_cross_quota_and_sign_balance(stage):
 counts=d.stage_config(stage)['slot_counts'];table=d.cross_quotas(counts)
 assert table.sum(0).tolist()==[1024,1024,2048] and table.sum(1).tolist()==counts
 for i in range(1,len(counts),2):assert torch.equal(table[i],table[i+1])


def test_parameters_have_no_policy_RNG_and_reset_subset_is_stable():
 ids=torch.arange(4096);ep=ids%17;strength=torch.ones(4096,6);rng=torch.get_rng_state().clone()
 p=d.parameters(ids,ep,strength)
 assert torch.equal(rng,torch.get_rng_state())
 reverse=d.parameters(ids.flip(0),ep.flip(0),strength).flip(0)
 assert torch.equal(p,reverse)
 assert torch.equal(p[[17]],d.parameters(ids[[17]],ep[[17]],strength[[17]]))
 assert p[:,0].min()==8 and p[:,0].max()==12
 assert p[:,1].min()==-2 and p[:,1].max()==2
 assert p[:,2].min()==8 and p[:,2].max()==12
 assert p[:,3].min()>=.9 and p[:,3].max()<=1.1
 assert p[:,4].min()>=.4 and p[:,4].max()<=.6
 nominal=d.parameters(ids,ep,torch.zeros_like(strength))
 assert torch.equal(nominal,torch.tensor([10,0,10,1,.5,0,0]).expand(4096,-1))


def test_mass_friction_push_are_standing_only():
 ids=torch.arange(8);p=d.parameters(ids,ids,torch.ones(8,6),standing=torch.tensor([1,1,0,0,0,0,0,0]),eval_corners=True)
 assert torch.allclose(p[0,3:6],torch.tensor([.9,.4,.1]))
 assert torch.allclose(p[1,3:6],torch.tensor([1.1,.6,.1]))
 assert torch.equal(p[2:,3:],torch.tensor([1,.5,0,0]).expand(6,-1))
 assert torch.allclose((p[:2,4]+.5)/2,torch.tensor([.45,.55]))


def test_material_lattice_keeps_bounds_nominal_and_independent_sampling():
 ids=torch.arange(4096);ep=ids%13;strength=torch.ones(4096,6)
 before=torch.get_rng_state().clone();continuous=d.parameters(ids,ep,strength)
 bucketed=d.parameters(ids,ep,strength,material_backend=d.MATERIAL_BACKEND)
 assert torch.equal(torch.get_rng_state(),before)
 assert len(torch.unique(bucketed[:,4]))<=65
 assert bucketed[:,4].min()>=.4 and bucketed[:,4].max()<=.6
 assert (bucketed[:,4]-continuous[:,4]).abs().max()<=1/640+1e-7
 torch.testing.assert_close(bucketed[:,:4],continuous[:,:4]);torch.testing.assert_close(bucketed[:,5:],continuous[:,5:])
 nominal=d.parameters(ids,ep,torch.zeros_like(strength),material_backend=d.MATERIAL_BACKEND)
 assert torch.equal(nominal[:,4],torch.full((4096,),.5))
 half=d.parameters(ids,ep,strength*.5,material_backend=d.MATERIAL_BACKEND)
 assert half[:,4].min()>=.45 and half[:,4].max()<=.55
 with pytest.raises(ValueError):d.parameters(ids,ep,strength,material_backend='arbitrary')


def test_failure_tail_is_bounded_but_preserves_fault_env_and_all_final_states():
 tail=[{'state':torch.zeros(4096,39),'geometry':torch.full((4096,13),float(i)),'control':torch.zeros(4096,42)} for i in range(3)]
 tail[-1]['state'][101,4]=1
 arrays=d.failure_tail_arrays(tail,8,[666])
 assert arrays['env_ids'].tolist()[:2]==[666,101]
 assert arrays['geometry'].shape==(3,8,13) and arrays['all_env_final_state'].shape==(4096,39)
 assert arrays['all_env_final_state'][101,4]==1 and np.all(arrays['geometry'][-1]==2)


def test_storage_retains_all8_scores_and_100Hz_worst_environment_geometry():
 payload={'scored_envs':8,'per_env':[dict(env=i,accepted=i!=3,tracking={'MAE':float(i)}) for i in range(8)]}
 original=copy.deepcopy(payload)
 arrays={'state':np.arange(3000*8*39,dtype=np.float32).reshape(3000,8,39),
         'geometry':np.zeros((3000,8,13),np.float32),'policy_time_s':np.arange(3000)*.01}
 result=d.diagnostic_storage(payload,arrays)
 assert payload['per_env']==original['per_env'] and payload['scored_envs']==8
 assert np.array_equal(result['state'],arrays['state'][:,3:4])
 assert result['geometry'].shape==(3000,1,13) and len(result['policy_time_s'])==3000
 assert not payload['diagnostic_storage']['full8_trajectory_replay_available']


def test_mixture_archive_keeps_temporarily_excluded_higher_endpoints():
 s=d.initial_state();ppo=S(reward_refinement_state=[dict(target=[2.,0.],mix=.75)],symmetry_mix_state=[])
 d.archive_mixtures(s,ppo)
 assert d.restored_mixture(s,'mixture_archive',[],[[2.,0.]])[0]['mix']==.75


def fake_env(n=8):
 from native_stand.students.motion_robustness_runtime import Adapter
 mass=torch.ones(n,2)*5;inertia=torch.eye(3).reshape(1,1,9).repeat(n,2,1)
 quat=torch.zeros(n,4);quat[:,3]=1
 data=S(root_quat_w=S(torch=quat),root_ang_vel_w=S(torch=torch.zeros(n,3)),root_vel_w=S(torch=torch.zeros(n,6)))
 class View:
  count=n;max_shapes=1
  def __init__(self):self.m=torch.tensor([.5,.5,0.]).reshape(1,1,3).repeat(n,1,1)
  def get_material_properties(self):return self.m
  def set_material_properties(self,values,indices):self.m[indices.long()]=values[indices.long()]
 views={'left':View(),'right':View()}
 robot=S(data=data,body_names=['base_link','extra'],root_view=S(get_masses=lambda:mass,get_inertias=lambda:inertia),
  _physics_sim_view=S(create_rigid_body_view=lambda path:views['left' if 'left_wheel' in path else 'right']))
 def set_mass(*,masses,body_ids,env_ids):mass[env_ids.long()[:,None],body_ids.long()]=masses
 def set_inertia(*,inertias,body_ids,env_ids):inertia[env_ids.long()[:,None],body_ids.long()]=inertias
 def set_velocity(*,root_velocity,env_ids):data.root_vel_w.torch[env_ids.long()]=root_velocity
 robot.set_masses_index=set_mass;robot.set_inertias_index=set_inertia;robot.write_root_velocity_to_sim_index=set_velocity
 t=S(tick=0,delays=torch.zeros(n,6,dtype=torch.long),feedback=torch.zeros(n,dtype=torch.long),
     sensors=torch.zeros(32,n,12),channels=torch.arange(6))
 e=S(num_envs=n,device='cpu',robot=robot,motion_timing=t,curriculum_training=False,
     target_xy=torch.cat((torch.zeros(n//2,2),torch.ones(n-n//2,2))),reference=torch.zeros(n,6),rise_hold=S(episode=S(steps=torch.zeros(n,dtype=torch.long))))
 cfg=dict(state=d.initial_state(),configuration=d.configuration())
 a=Adapter(e,cfg);e.motion_dr=a
 return e,a,mass,inertia,views


def test_backend_mass_inertia_and_friction_roundtrip_and_nominal_restoration():
 e,a,mass,inertia,views=fake_env()
 a.configure(dict(level=5,group=2,seed=20261023));a.reset(a.ids)
 torch.testing.assert_close(mass[:,0],5*a.params[:,3])
 torch.testing.assert_close(inertia[:,0],torch.eye(3).reshape(1,9)*a.params[:,3,None])
 torch.testing.assert_close(views['left'].m[:,0,0],a.params[:,4])
 assert torch.equal(mass[:,1],torch.full((8,),5.))
 e.target_xy[:]=1;a.reset(a.ids)
 torch.testing.assert_close(mass[:,0],torch.full((8,),5.))
 torch.testing.assert_close(views['left'].m[:,0,0],torch.full((8,),.5))


@pytest.mark.parametrize('corrupt',[False,True])
def test_articulation_writer_preserves_other_shapes_and_nonreset_rows(corrupt):
 e,a,_,_,views=fake_env();root=e.robot.root_view
 materials=torch.tensor([.7,.6,.1]).reshape(1,1,3).repeat(8,4,1)
 materials[:,[1,3]]=torch.tensor([.5,.5,0.])
 before=materials.clone();calls=[]
 root.get_material_properties=lambda:materials
 def write(values,ids):
  calls.append(ids.clone());materials[ids.long()]=values[ids.long()]
  if corrupt:materials[0,0,0]=.9
 root.set_material_properties=write
 for side,column in [('left',1),('right',3)]:
  views[side].get_material_properties=lambda col=column:materials[:,col:col+1]
 a.material_shape_columns={'left':1,'right':3}
 a.config['material_writer']=d.MATERIAL_WRITER
 a.configure(dict(level=5,group=2,seed=1));ids=torch.tensor([0,2,6])
 if corrupt:
  with pytest.raises(RuntimeError,match='articulation material readback differs'):a.reset(ids)
  return
 a.reset(ids)
 assert len(calls)==1 and torch.equal(calls[0],ids.int())
 assert torch.equal(materials[:,[0,2]],before[:,[0,2]])
 assert torch.equal(materials[[1,3,4,5,7]],before[[1,3,4,5,7]])
 for col in (1,3):torch.testing.assert_close(materials[ids,col,:2],a.params[ids,4,None].expand(-1,2))
 assert a.last_readback['material_writer']==d.MATERIAL_WRITER


def test_push_only_zero_target_and_each_episode_has_two_bounded_events():
 e,a,*_=fake_env();a.configure(dict(level=5,group=2,seed=1));a.reset(a.ids)
 e.rise_hold.episode.steps[:]=1200;a.maybe_push();before=e.robot.data.root_vel_w.torch.clone()
 torch.testing.assert_close(before[:4,:2].norm(dim=-1),torch.full((4,),.1))
 assert not before[4:].any();assert a.push_count==4
 a.maybe_push();assert torch.equal(before,e.robot.data.root_vel_w.torch) and a.push_count==4
 e.rise_hold.episode.steps[:]=2200;a.maybe_push()
 torch.testing.assert_close(e.robot.data.root_vel_w.torch,torch.zeros(8,6),atol=2e-8,rtol=0)
 assert a.push_count==8


def test_timestamped_sensor_reads_actual_history_not_fractional_interpolation(monkeypatch):
 from native_stand.students.motion_robustness_runtime import DelayedPackets
 module=ModuleType('isaaclab.utils.math');module.quat_apply_inverse=lambda q,v:v
 monkeypatch.setitem(sys.modules,'isaaclab.utils.math',module)
 e,a,*_=fake_env();e.actor_sensor_packets=DelayedPackets(a)
 for tick in range(21):
  e.motion_timing.tick=tick;a.capture();e.motion_timing.sensors[tick%32,:,6:]=float(tick)
 e.motion_timing.tick=20;a.params[:,0]=torch.arange(8,16).clamp_max(12)
 current=torch.zeros(8,16);current[:,5]=-1
 out=e.actor_sensor_packets.read(current,20)
 expected=(20-a.params[:,0])*.05
 torch.testing.assert_close(out[:,10],expected)
 assert torch.equal(out[:,5],torch.full((8,),-1.))
 a.reset(torch.tensor([0]));out=e.actor_sensor_packets.read(current,20)
 assert torch.equal(out[0],current[0])


def test_actual_PD_adoption_age_is_not_transport_arrival():
 e,a,*_=fake_env();e.motion_timing.delays[:]=torch.tensor([10,10,7,10,10,11])
 for tick in range(21):
  e.motion_timing.tick=tick;a.capture();e.motion_timing.tick+=1;a.adoption_readback()
 assert a.first_adoption_ages[0].tolist()==[10,10,10,10,10,15]


def test_PD_adoption_buffer_survives_inference_steps_then_regular_reset():
 e,a,*_=fake_env();buffer=a.first_adoption_ages
 with torch.inference_mode():
  for tick in range(21):
   e.motion_timing.tick=tick;a.capture();e.motion_timing.tick+=1;a.adoption_readback()
 assert a.first_adoption_ages is buffer
 assert not torch.is_inference(a.first_adoption_ages)
 a.reset(torch.tensor([0,3]))
 assert torch.equal(a.first_adoption_ages[[0,3]],torch.full((2,6),-1))
 assert a.first_adoption_ages[1].tolist()==[10,10,10,10,10,15]


def recipe_fixture():
 from native_stand.students.motion_training import declaration
 from native_stand.students.motion_tracking_reward import PARALLEL_VERSION
 from native_stand.students.motion_antiosc_privileged import VERSION,network
 from native_stand.students.motion_wheel_matching import configuration
 from native_stand.students.motion_startup import smoke_profile
 r=declaration('a'*64,0,reward_version=PARALLEL_VERSION)
 r.update(course_version=d.VERSION,course_stage=0,timing_configuration=configuration(),
          critic_privileged_version=VERSION,network_architecture=network(),
          critic='full_motion140_wheel_match',optimizer='source_Adam_full_state',
          checkpoint_every_updates=100,milestones=list(range(250,10000,250)),smoke_wiring=smoke_profile())
 return r


def test_new_recipe_returns_own_version_and_preserves_original_PPO():
 from native_stand.students.training_recipe import validate_instance
 r=recipe_fixture()
 r['course_version']=d.VERSION
 original=copy.deepcopy(r)
 assert validate_instance(r,r['source_actor_sha256']) is r
 assert r==original
 bad=copy.deepcopy(r);bad['BC_coefficient']=.1
 with pytest.raises(ValueError):validate_instance(bad,bad['source_actor_sha256'])


def test_evaluation_grouping_keeps_DR_groups_separate():
 from native_stand.students.grouped_evaluation import plan
 template=dict(kind='student',role='MotionZeroStudent',family='motion_native_normal',seed=7,yaw_rad=0.,seconds=31.52,checkpoint='/fixture.pt',sha256='a'*64)
 s={'motion_robustness':{'state':d.initial_state(0)},'student_evaluation':{'cases':[template]}}
 rows=d.cases(s)
 batches=plan(rows)
 assert len(batches)==6
 for group in batches:assert len({r['_motion_dr']['group'] for r in group})==1


def test_one_env_trace_storage_bound_fits_independent8GiB(tmp_path):
 from native_stand.command_v6.tools.storage_budget import estimate
 source=tmp_path/'source.pt';source.write_bytes(b'x'*7821978)
 s=dict(source_checkpoint=str(source),budget=dict(formal_updates=9995),student_training=recipe_fixture(),
        student_evaluation={'cases':[{'seconds':31.52}]},ppo={'envs':4096},artifacts=dict(storage_max_bytes=8*2**30,terminal_reserve_bytes=512*2**20))
 s['motion_robustness']={'version':d.VERSION}
 result=estimate(s,'formal')
 required=sum(result[k] for k in ('estimated_artifact_bytes','atomic_temp_peak_bytes','terminal_reserve_bytes'))
 assert result['schema_version']=='motion_robustness_storage_v1'
 assert required<8*2**30


def test_snapshot_fields_do_not_hide_evaluation_failure():
 records=candidate(d.initial_state(),250,fail=(2,1,0))['evaluations']
 before=[e['accepted'] for r in records for e in r['per_env']]
 d.compact_historical_records(records)
 assert [e['accepted'] for r in records for e in r['per_env']]==before
 assert all('tracking' in e for r in records for e in r['per_env'])


@pytest.fixture
def built_recipe(tmp_path):
 from native_stand.students.data import sha
 r=recipe_fixture();r['course_version']='motion31_posture_reacquisition_v1'
 old=dict(experiment='fixture_parent',student_training=r,reward=copy.deepcopy(r['reward']),
          ppo={'envs':4096,'steps_per_env':128},motion_timing=dict(configuration=copy.deepcopy(r['timing_configuration']),stage=4),
          motion_training={'stage':0},motion_course={},budget={},artifacts={},student_dependencies={},
          motion_startup={},student_evaluation=dict(cases=[dict(kind='student',role='MotionZeroStudent',family='motion_native_normal',seed=7,yaw_rad=0.,seconds=31.52)],max_output_bytes=128*2**20),
          curriculum={'stages':[]},resources={'mode':'shared','sharing_authorized':True})
 folder=tmp_path/'configs/experiments/fixture_parent';folder.mkdir(parents=True)
 p=folder/'spec.json';p.write_text(json.dumps(old))
 run=tmp_path/'runs/fixture_parent';formal=run/'formal';formal.mkdir(parents=True)
 terminal=dict(status='STOPPED_BOUNDED',actual_updates=2001,failure=None)
 (formal/'terminal_report.json').write_text(json.dumps(terminal));(run/'chain_terminal_report.json').write_text(json.dumps(dict(status='STOPPED_BOUNDED',failure=None)))
 book=dict(used={'formal':2001,'smoke':5},attempts={});(folder/'budget.json').write_text(json.dumps(book))
 cp=formal/'model_update_2001.pt'
 source=dict(update=2001,spec_sha256=sha(p),critic_privileged_version='motion_antiosc_privileged140_v1',
  recipe_updates=4501,training_recipe=copy.deepcopy(old['student_training']))
 torch.save(source,cp)
 proof=dict(status='CONFIRMED',parent_spec_sha256=sha(p),certified_skill_stage=1,inputs={})
 s=d.build(p,cp,experiment='future',spec_path=tmp_path/'configs/experiments/future/spec.json',confirmation=proof,owner_session='fixture_session')
 return s,old,p,cp,proof


def test_build_latest_full_checkpoint_and_bind_declared10000(built_recipe):
 s,old,p,cp,proof=built_recipe
 assert s['motion_training']['stage']==1 and s['motion_robustness']['state']['mode']=='DR_ACQUIRE'
 assert s['budget']['smoke_updates']+s['budget']['formal_updates']==10000
 assert s['budget']['formal_wall_s'] is None and s['budget']['chain_wall_s'] is None
 assert s['notifications']['enabled'] and s['authorization']['hapi_session']=='fixture_session'
 assert s['motion_robustness']['source_latest_unassessed']
 assert s['reward']==old['reward'] and s['ppo']==old['ppo']
 assert d.verify(s)
 bad=copy.deepcopy(s);bad['reward']['weights'][0]*=2
 with pytest.raises(ValueError):d.verify(bad)


def failed_robustness(built_recipe):
 s,_,p,_,_=built_recipe
 folder=p.parent.parent/'future';folder.mkdir()
 spec=folder/'spec.json';spec.write_text(json.dumps(s))
 book=dict(used=dict(source_eval=1,smoke=0,formal=0,evaluate=0,play=0),
           attempts={'attempt':dict(phase='source_eval',updates=0,status='FAILED')})
 (folder/'budget.json').write_text(json.dumps(book))
 run=p.parents[3]/'runs/future';(run/'source_eval').mkdir(parents=True)
 chain=dict(status='FAILED',smoke_updates=0,formal_updates=0,chain_wall_s=9.5)
 terminal=dict(status='FAILED',actual_updates=0,failure=dict(
  exception='RuntimeError: Inplace update to inference tensor outside InferenceMode',
  traceback='motion_robustness_runtime.py: first_adoption_ages'))
 (run/'chain_terminal_report.json').write_text(json.dumps(chain))
 (run/'source_eval/terminal_report.json').write_text(json.dumps(terminal))
 return spec,book,run,terminal


def test_zero_PPO_buffer_recovery_preserves_source_and_original_budget(built_recipe):
 s,_,p,cp,proof=built_recipe;failed,_,_,_=failed_robustness(built_recipe)
 recovery=d.build(p,cp,experiment='retry',spec_path=p.parent.parent/'retry/spec.json',
  confirmation=proof,owner_session='fixture_session',recovery_parent=failed)
 assert d.verify(recovery)
 assert recovery['source_sha256']==s['source_sha256'] and recovery['reward']==s['reward']
 prior=recovery['motion_training']['prior_consumption']
 assert prior['historical_online_updates']==0 and prior['historical_wall_s']==9.5
 assert recovery['budget']==s['budget'] and prior['original_online_limit']==10000
 assert str(failed) in recovery['student_dependencies']


@pytest.mark.parametrize('invalid',['active','charged','other_error','human_stop'])
def test_buffer_recovery_rejects_ambiguous_or_out_of_scope_failure(built_recipe,invalid):
 failed,book,run,terminal=failed_robustness(built_recipe)
 if invalid=='active':book['attempts']['attempt']['status']='RUNNING'
 if invalid=='charged':book['used']['smoke']=1
 if invalid=='other_error':terminal['failure']['exception']='nonfinite physical state'
 if invalid=='human_stop':terminal['status']='STOPPED_BOUNDED'
 failed.with_name('budget.json').write_text(json.dumps(book))
 (run/'source_eval/terminal_report.json').write_text(json.dumps(terminal))
 with pytest.raises(ValueError):d.preppo_recovery_evidence(failed)


@pytest.fixture
def formal_material_failure(built_recipe,monkeypatch):
 from native_stand.command_v6 import wiring
 monkeypatch.setattr(wiring,'validate_persisted_certificate',lambda *a,**kw:dict(status='PASSED'))
 s,_,p,_,_=built_recipe;folder=p.parent.parent/'future';folder.mkdir();spec=folder/'spec.json';spec.write_text(json.dumps(s))
 run=p.parents[3]/'runs/future'
 for name in ('source_eval','smoke','formal'):(run/name).mkdir(parents=True)
 def write(path,value):path.write_text(json.dumps(value))
 write(folder/'budget.json',dict(used=dict(source_eval=1,smoke=5,formal=46),attempts={'f':dict(status='FAILED',phase='formal',updates=46)}))
 write(run/'chain_terminal_report.json',dict(status='FAILED',chain_wall_s=100.,smoke_updates=5,formal_updates=46))
 error=dict(exception="RuntimeError('Motion DR friction backend readback differs left')",traceback='motion_robustness_runtime.py')
 write(run/'formal/terminal_report.json',dict(status='FAILED',actual_updates=46,failure=error))
 write(run/'formal/progress.json',dict(actual_updates=46))
 (run/'formal/metrics.jsonl').write_text(''.join(json.dumps(dict(update=i))+'\n' for i in range(1,47)))
 source=dict(update=46,spec_sha256=d.sha(spec),critic_privileged_version='motion_antiosc_privileged140_v1',training_recipe=s['student_training'],
  actor={},critic={},optimizer={},training_rng={},reward_refinement_state=[],symmetry_mix_state=[],motion_course_state=copy.deepcopy(s['motion_robustness']['state']),recipe_updates=4500)
 source['motion_course_state']['DR_episode_counters']=[4]*4096
 cp=run/'formal/model_update_46.pt';torch.save(source,cp)
 for name in ('wiring_certificate.json','terminal_report.json','progress.json','metrics.jsonl'):write(run/'smoke'/name,{})
 write(run/'source_eval/terminal_report.json',dict(status='COMPLETED',failure=None,evaluations=[]))
 write(run/'source_eval/robustness_engineering_readback.json',dict(status='PASSED'))
 write(run/'source_eval/candidate_registry.json',[])
 probe=p.parents[3]/'material_probe.json'
 write(probe,dict(status='COMPLETED',variant=d.MATERIAL_BACKEND,envs=4096,PPO_updates=0,checks=[dict(passed=True,**{side:dict(all_env_max_diff=0.,nonreset_max_change=0.) for side in ('left','right')}) for _ in range(31)]))
 return spec,cp,probe,s,run


def test_material_resume_preserves_51_charged_updates_and_remaining_cadence(formal_material_failure):
 parent,cp,probe,old,run=formal_material_failure
 s=d.build_resume(parent,cp,experiment='resume',spec_path=parent.parent.parent/'resume/spec.json',material_probe=probe,owner_session='fixture')
 assert d.verify(s)['status']=='VERIFIED'
 assert s['budget']['smoke_updates']==0 and s['budget']['formal_updates']==9949
 assert s['motion_training']['prior_consumption']['historical_online_updates']==51
 assert s['student_training']['milestones']==[v-46 for v in old['student_training']['milestones']]
 assert s['motion_robustness']['state']['stage_started_update']==-46
 assert s['motion_robustness']['state']['DR_episode_counters']==[4]*4096
 assert s['reward']==old['reward'] and s['ppo']==old['ppo'] and s['motion_timing']['configuration']==old['motion_timing']['configuration']
 from native_stand.students.motion_runner import checkpoint_plan
 assert checkpoint_plan(s['student_training'],'formal',54,9949,offset=46)==(True,False)
 assert checkpoint_plan(s['student_training'],'formal',204,9949,offset=46)==(True,True)
 changed=copy.deepcopy(s);changed['budget']['formal_updates']+=1
 with pytest.raises(ValueError):d.verify(changed)


@pytest.mark.parametrize('kind',['active','wrong_checkpoint','unproved_backend','unknown_failure','charge_conflict'])
def test_material_resume_rejects_unsafe_or_unaccounted_recovery(formal_material_failure,kind):
 parent,cp,probe,_,run=formal_material_failure
 if kind=='active':
  b=json.loads(parent.with_name('budget.json').read_text());b['attempts']['f']['status']='RUNNING';parent.with_name('budget.json').write_text(json.dumps(b))
 if kind=='wrong_checkpoint':
  x=torch.load(cp,weights_only=False);x['update']=45;torch.save(x,cp)
 if kind=='unproved_backend':
  x=json.loads(probe.read_text());x['checks'][3]['left']['all_env_max_diff']=.01;probe.write_text(json.dumps(x))
 if kind=='unknown_failure':
  f=run/'formal/terminal_report.json';x=json.loads(f.read_text());x['failure']['exception']='RuntimeError(nonfinite)';f.write_text(json.dumps(x))
 if kind=='charge_conflict':
  b=json.loads(parent.with_name('budget.json').read_text());b['used']['formal']=47;parent.with_name('budget.json').write_text(json.dumps(b))
 with pytest.raises(ValueError):d.resume_evidence(parent,cp,probe)


def test_second_resume_retains_105_updates_and_finds_ancestral_source(formal_material_failure):
 parent,cp,probe,old,run=formal_material_failure
 snapshot=run/'source_eval/source_snapshot';snapshot.mkdir();(snapshot/'spec.json').write_bytes(parent.read_bytes())
 first_path=parent.parent.parent/'resume_one/spec.json';first_path.parent.mkdir()
 first=d.build_resume(parent,cp,experiment='resume_one',spec_path=first_path,material_probe=probe,owner_session='fixture')
 first_path.write_text(json.dumps(first))
 second_run=run.parent/'resume_one';formal=second_run/'formal';formal.mkdir(parents=True)
 first_path.with_name('budget.json').write_text(json.dumps(dict(used=dict(source_eval=0,smoke=0,formal=54),attempts={'f':dict(status='FAILED',phase='formal',updates=54)})))
 (second_run/'chain_terminal_report.json').write_text(json.dumps(dict(status='FAILED',chain_wall_s=40.,smoke_updates=0,formal_updates=54)))
 error=dict(exception="ValueError('diagnostics require actual formal100 boundary')",traceback='motion_parallel_course.py checkpoint_diagnostics')
 (formal/'terminal_report.json').write_text(json.dumps(dict(status='FAILED',actual_updates=54,failure=error)))
 (formal/'progress.json').write_text(json.dumps(dict(actual_updates=54)))
 (formal/'metrics.jsonl').write_text(''.join(json.dumps(dict(update=i,parent_formal_offset=46,formal_update_including_parent=46+i,actual_LR=[3e-5]))+'\n' for i in range(1,55)))
 data=torch.load(cp,weights_only=False);data.update(update=54,spec_sha256=d.sha(first_path),training_recipe=first['student_training'],motion_course_state=copy.deepcopy(first['motion_robustness']['state']))
 latest=formal/'model_update_54.pt';torch.save(data,latest)
 future=parent.parent.parent/'resume_two/spec.json'
 second=d.build_resume(first_path,latest,experiment='resume_two',spec_path=future,material_probe=probe,owner_session='fixture')
 assert d.verify(second)['status']=='VERIFIED'
 c=second['motion_training']['wiring_reuse']
 assert c['version']==d.DIAGNOSTIC_RESUME_VERSION and c['source_evidence_run']==str(run)
 assert c['checkpoint_update']==54 and c['formal_update_offset']==100
 assert c['historical_online_updates']==105 and c['historical_smoke_updates']==5 and c['historical_formal_updates']==100
 assert c['historical_wall_s']==140. and second['budget']['formal_updates']==9895 and second['budget']['smoke_updates']==0
 assert second['student_training']['formal_resume_offset']==100 and second['student_training']['milestones'][:2]==[150,400]
 assert second['motion_robustness']['state']['stage_started_update']==-100
 from native_stand.students.motion_runner import checkpoint_plan
 assert checkpoint_plan(second['student_training'],'formal',100,9895,offset=100)==(True,False)
 assert checkpoint_plan(second['student_training'],'formal',150,9895,offset=100)==(True,True)
 mutated=copy.deepcopy(second);mutated['student_training']['formal_resume_offset']=54
 with pytest.raises(ValueError):d.verify(mutated)


def test_forward_default_path_is_unchanged_without_explicit_robust_version():
 from native_stand.students.motion_course import implementation
 from native_stand.students import motion_reacquisition
 assert implementation(motion_reacquisition.VERSION) is motion_reacquisition
 assert implementation(d.VERSION) is d


@pytest.fixture
def evaluation_material_failure(formal_material_failure):
 parent,cp,probe,_,run=formal_material_failure
 snapshot=run/'source_eval/source_snapshot';snapshot.mkdir();(snapshot/'spec.json').write_bytes(parent.read_bytes())
 def settle(config,name,updates,error):
  path=parent.parent.parent/name/'spec.json';path.parent.mkdir();path.write_text(json.dumps(config))
  folder=run.parent/name/'formal';folder.mkdir(parents=True)
  path.with_name('budget.json').write_text(json.dumps(dict(used=dict(source_eval=0,smoke=0,formal=updates),attempts={'f':dict(status='FAILED',phase='formal',updates=updates)})))
  (folder.parent/'chain_terminal_report.json').write_text(json.dumps(dict(status='FAILED',chain_wall_s=40.,smoke_updates=0,formal_updates=updates)))
  (folder/'terminal_report.json').write_text(json.dumps(dict(status='FAILED',actual_updates=updates,failure=error)))
  (folder/'progress.json').write_text(json.dumps(dict(actual_updates=updates)))
  offset=config['student_training']['formal_resume_offset']
  (folder/'metrics.jsonl').write_text(''.join(json.dumps(dict(update=i,parent_formal_offset=offset,formal_update_including_parent=offset+i,actual_LR=[3e-5]))+'\n' for i in range(1,updates+1)))
  saved=torch.load(cp,weights_only=False);saved.update(update=updates,spec_sha256=d.sha(path),training_recipe=config['student_training'],motion_course_state=copy.deepcopy(config['motion_robustness']['state']))
  latest=folder/f'model_update_{updates}.pt';torch.save(saved,latest)
  return path,latest
 first_path=parent.parent.parent/'boundary_one/spec.json'
 first=d.build_resume(parent,cp,experiment='boundary_one',spec_path=first_path,material_probe=probe,owner_session='fixture')
 first_path,first_cp=settle(first,'boundary_one',54,dict(exception="ValueError('diagnostics require actual formal100 boundary')",traceback='motion_parallel_course.py'))
 second_path=parent.parent.parent/'boundary_two/spec.json'
 second=d.build_resume(first_path,first_cp,experiment='boundary_two',spec_path=second_path,material_probe=probe,owner_session='fixture')
 second_path,latest=settle(second,'boundary_two',150,dict(exception='RuntimeError("Motion DR friction backend readback differs left {\\\'env_ids\\\': [46]}")',traceback='evaluate_cases motion_robustness_runtime.py'))
 evaluation=probe.with_name('eval_probe.json');evaluation.write_text(json.dumps(dict(status='COMPLETED',envs=64,reset_calls=486,physics_steps=31520,material_writer=d.MATERIAL_WRITER)))
 proof=json.loads(probe.read_text());proof.update(material_writer=d.MATERIAL_WRITER,
  evaluation_probe={**json.loads(evaluation.read_text()),'path':str(evaluation),'sha256':d.sha(evaluation)},training_probe=dict(path=str(probe),sha256=d.sha(probe)))
 both=probe.with_name('both_probes.json');both.write_text(json.dumps(proof))
 return second_path,latest,both


@pytest.mark.parametrize('passed',[True,False])
def test_evaluation_material_resume_keeps_255_used_and_replays_boundary_before_PPO(evaluation_material_failure,passed):
 parent,cp,proof=evaluation_material_failure
 s=d.build_resume(parent,cp,experiment='eval_resume',spec_path=parent.parent.parent/'eval_resume/spec.json',material_probe=proof,owner_session='fixture')
 assert d.verify(s)['status']=='VERIFIED'
 c=s['motion_training']['wiring_reuse']
 assert c['version']==d.EVALUATION_RESUME_VERSION and c['historical_online_updates']==255
 assert c['checkpoint_update']==150 and c['formal_update_offset']==c['pending_assessment_formal_update']==250
 assert s['budget']['smoke_updates']==0 and s['budget']['formal_updates']==9745
 assert s['student_training']['milestones'][0]==250
 assert s['motion_robustness']['material_writer']==d.MATERIAL_WRITER
 state=copy.deepcopy(s['motion_robustness']['state']);values=d.commands(state['stage'])
 from native_stand.students.motion_refinement import initial
 ppo=S(motion_course_state=state,reward_refinement_state=initial(values),symmetry_mix_state=d.extend_mixture([],values),optimizer='untouched')
 rows=candidate(state,0,fail=None if passed else (2,0,0))['evaluations']
 new,state,cand,decision=d.complete_pending_assessment(s,ppo,rows)
 assert decision['stage_passed']==decision['advanced']==passed and not decision['stop']
 assert state['stage']==cand['stage']+int(passed)
 assert cand['update']==0 and cand['formal_update_including_parent']==250 and cand['checkpoint']==str(cp)
 assert ppo.optimizer=='untouched' and new['budget']==s['budget']
 assert len(ppo.reward_refinement_state)==len(d.commands(state['stage']))
 from native_stand.command_v6.tools.storage_budget import robustness_estimate
 bare=copy.deepcopy(s);bare['motion_training']['wiring_reuse'].pop('pending_assessment_formal_update')
 trace=131072+round(max(row['seconds'] for row in s['student_evaluation']['cases'])/.01)*114*4
 assert robustness_estimate(s,'formal')['evaluation_bytes']-robustness_estimate(bare,'formal')['evaluation_bytes']==len(s['student_evaluation']['cases'])*trace
 with pytest.raises(ValueError,match='pending first full'):d.complete_pending_assessment(new,ppo,cand['evaluations'])
 bad=copy.deepcopy(s);bad['motion_robustness']['material_writer']='unverified'
 with pytest.raises(ValueError):d.verify(bad)


def test_evaluation_material_resume_requires_actual_full_window_probe(evaluation_material_failure):
 parent,cp,proof=evaluation_material_failure
 data=json.loads(proof.read_text());data['evaluation_probe']['physics_steps']=500;proof.write_text(json.dumps(data))
 with pytest.raises(ValueError,match='actual64'):d.resume_evidence(parent,cp,proof)


@pytest.fixture
def group_baseline_failure(evaluation_material_failure):
 parent,cp,proof=evaluation_material_failure
 path=parent.parent.parent/'after_pending/spec.json'
 config=d.build_resume(parent,cp,experiment='after_pending',spec_path=path,material_probe=proof,owner_session='fixture')
 path.parent.mkdir();path.write_text(json.dumps(config))
 run=parent.parents[3]/'runs/after_pending';formal=run/'formal';formal.mkdir(parents=True)
 origin=torch.load(cp,weights_only=False);values=d.commands(config['motion_training']['stage'])
 ppo=S(motion_course_state=copy.deepcopy(config['motion_robustness']['state']),
  reward_refinement_state=d.extend_mixture(origin['reward_refinement_state'],values),
  symmetry_mix_state=d.extend_mixture(origin['symmetry_mix_state'],values))
 records=candidate(ppo.motion_course_state,0)['evaluations']
 active,state,boundary,decision=d.complete_pending_assessment(config,ppo,records)
 for i,record in enumerate(records):
  trace=formal/f'trace_{i}.npz';trace.write_bytes(b'CPU fixture, not physical evidence')
  raw=formal/f'evaluation_{i}.json'
  raw.write_text(json.dumps(dict(case=record['case'],per_env=record['per_env'],checkpoint_sha256=config['source_sha256'],spec_sha256=d.sha(path),trace=dict(path=trace.name,sha256=d.sha(trace)))))
  record.update(path=raw.name,sha256=d.sha(raw))
 d.compact_historical_records(records)
 (formal/'candidate_registry.json').write_text(json.dumps([boundary]))
 (formal/'course_progress.json').write_text(json.dumps(dict(state=state,decision=decision)))
 path.with_name('budget.json').write_text(json.dumps(dict(used=dict(source_eval=0,smoke=0,formal=250),attempts={'f':dict(status='FAILED',phase='formal',updates=250)})))
 (run/'chain_terminal_report.json').write_text(json.dumps(dict(status='FAILED',chain_wall_s=80.)))
 (formal/'terminal_report.json').write_text(json.dumps(dict(status='FAILED',actual_updates=250,failure=dict(exception="ValueError('duplicate course baseline command')",traceback='training.py evaluate_cases'))))
 (formal/'progress.json').write_text(json.dumps(dict(actual_updates=250,pending_assessment={'status':'COMPLETED'})))
 (formal/'metrics.jsonl').write_text(''.join(json.dumps(dict(update=i,parent_formal_offset=250,formal_update_including_parent=250+i))+'\n' for i in range(1,251)))
 latest=formal/'model_update_250.pt'
 origin.update(update=250,spec_sha256=d.sha(path),training_recipe=active['student_training'],motion_course_state=state,
  reward_refinement_state=ppo.reward_refinement_state,symmetry_mix_state=ppo.symmetry_mix_state,recipe_updates=5000)
 torch.save(origin,latest)
 return path,latest,proof,formal


def test_group_baseline_recovery_preserves_completed_promotion_and_505_charged(group_baseline_failure):
 parent,cp,proof,_=group_baseline_failure
 s=d.build_resume(parent,cp,experiment='baseline_resume',spec_path=parent.parent.parent/'baseline_resume/spec.json',material_probe=proof,owner_session='fixture')
 assert d.verify(s)['status']=='VERIFIED'
 c=s['motion_training']['wiring_reuse'];state=s['motion_robustness']['state']
 assert c['version']==d.BASELINE_RESUME_VERSION and c['historical_online_updates']==505
 assert c['formal_update_offset']==c['pending_assessment_formal_update']==500 and c['checkpoint_update']==250
 assert s['budget']['formal_updates']==9495 and s['budget']['smoke_updates']==0
 assert state['stage']==s['motion_training']['stage']==s['student_training']['stage']==2
 assert state['stage_started_update']==state['last_evaluation_update']==-250
 assert next(iter(state['accepted_by_stage'].values()))['update']==-250
 assert s['student_training']['milestones'][0]==250
 ppo=S(motion_course_state=copy.deepcopy(state),reward_refinement_state=d.extend_mixture([],d.commands(2)),symmetry_mix_state=d.extend_mixture([],d.commands(2)))
 _,state,_,decision=d.complete_pending_assessment(s,ppo,candidate(state,0,fail=(2,1,0))['evaluations'])
 assert decision['old_command_regression'] and state['stage']==1 and state['DR_level']==0 and not decision['stop']


@pytest.mark.parametrize('what',['raw','course_state','refinement'])
def test_group_baseline_resume_rejects_unproven_promotion(group_baseline_failure,what):
 parent,cp,proof,formal=group_baseline_failure
 if what=='raw':
  raw=formal/'evaluation_0.json';value=json.loads(raw.read_text());value['per_env'][0]['accepted']=False;raw.write_text(json.dumps(value))
 else:
  value=torch.load(cp,weights_only=False)
  if what=='course_state':value['motion_course_state']['certified_DR_level']=-1
  else:value['reward_refinement_state'][0]['mix']=.75
  torch.save(value,cp)
 with pytest.raises(ValueError):d.resume_evidence(parent,cp,proof)


def test_engineering_probe_is_bounded_PPO0_and_does_not_require_ability(monkeypatch,tmp_path):
 from native_stand.students import evaluator
 class ProbeDR:
  last_push=None
  def configure(self,override):self.override=override
  def summary(self):return {'level':self.override['level'],'backend':'MOCK_ONLY'}
  def impulse(self,ids,pulse):self.last_push={'delta_world_XY_mps':[[.1,0.]]}
 class ProbeEnv:
  device='cpu';motion_dr=ProbeDR();recovery_snapshot=torch.zeros(64,39);steps=0;closed=False
  def select_initial_family(self,*a):pass
  def reset(self):return {'student_actor':torch.zeros(64,31),'student_critic':torch.zeros(64,140)},{}
  def step(self,action):
   self.steps+=1;obs,_=self.reset()
   return obs,torch.ones(64),torch.zeros(64,dtype=torch.bool),torch.zeros(64,dtype=torch.bool),{}
  def close(self):self.closed=True
 e=ProbeEnv();monkeypatch.setattr(evaluator,'environment',lambda *a,**k:e)
 d.engineering_probe({},lambda o:torch.zeros(64,6),tmp_path,lambda:False)
 result=json.loads((tmp_path/'robustness_engineering_readback.json').read_text())
 assert e.steps==120 and e.closed and result['PPO_updates']==0 and result['physics_steps']==1200
 assert len(result['levels'])==6


def controller():
 path=Path.cwd()/'scripts/transition_motion_to_robust_dr.py'
 spec=importlib.util.spec_from_file_location('robust_transition_test',path)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m


def test_watcher_ignores_already_seen_parent_promotion(tmp_path):
 m=controller();run=tmp_path/'run';formal=run/'formal';formal.mkdir(parents=True)
 parent=tmp_path/'spec.json';parent.write_text('{}')
 (formal/'course_progress.json').write_text(json.dumps(dict(state={'accepted_by_stage':{'0':{'update':1750}}})))
 plan=dict(parent_run=str(run),parent_spec=str(parent),parent_spec_sha256=m.sha(parent),watch_stage=1,after_evaluation_update=1750)
 assert m.promotion(plan) is None


def test_latest_source_is_actual_last_full_save_not_best_or_promoted(tmp_path):
 m=controller();run=tmp_path/'run';formal=run/'formal';formal.mkdir(parents=True)
 (formal/'terminal_report.json').write_text(json.dumps(dict(actual_updates=2263)))
 (formal/'candidate_registry.json').write_text(json.dumps([dict(sha256='old',update=2250,evaluated=True,evaluations=[{}])]))
 cp=formal/'model_update_2263.pt'
 torch.save(dict(update=2263,spec_sha256='b'*64,critic_privileged_version='motion_antiosc_privileged140_v1',
  actor={},critic={},optimizer={},training_rng={},reward_refinement_state=[]),cp)
 result=m.latest_stopped(dict(parent_run=str(run),parent_spec_sha256='b'*64))
 assert result['update']==2263 and result['checkpoint']==str(cp) and not result['evaluated']


def test_selection_reports_active_rollback_stage_not_historical_maximum():
 s=d.initial_state(2);s['mode']='COMMAND_ACQUIRE';s['qualified_by_DR']['0']=d.commands(1)
 before=candidate(s,250)
 source={**candidate(s,0), 'complete_matrix':False}
 s,decision=d.advance(s,candidate(s,500,fail=(2,1,0)))
 result=d.choose([source,before],s)
 assert s['stage']==1 and result['current_stage']==1 and result['highest_command_stage_seen']==2
 assert result['best_new_evaluated_candidate'] is None and result['accepted_candidate'] is None
