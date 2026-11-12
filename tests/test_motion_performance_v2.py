FROZEN_TEST_SCOPE='PORTABLE'
import copy
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from native_stand.students.step_checks import StepChecks
from native_stand.students.performance import settings
from native_stand.students.motion_statistics import MotionStatistics
from native_stand.students.grouped_evaluation import plan,split_arrays


def cases(n=7):
 return [dict(case_id=str(i),kind='student',role='MotionZeroStudent',family='motion_native_normal',seed=17,
  yaw_rad=0.,seconds=31.52,checkpoint='/fake/source.pt',sha256='a'*64,command=[0.,float(i)/2,.34]) for i in range(n)]


def test_course_baseline_matches_DR_scope_and_rejects_only_actual_duplicates(tmp_path,monkeypatch):
 from native_stand.students.training import course_baseline,evaluate_cases
 from native_stand.students import training
 rows=[{**cases(1)[0],'case_id':f'robust_0_{g}_0','_timing_level':4,
        '_motion_dr':dict(level=0,group=g,mode='DR_ACQUIRE',skill_stage=1)} for g in range(3)]
 records=[dict(case=row,per_env=[{'env':0}]) for row in rows]
 spec=dict(motion_course={'version':'fixture'},motion_training={'stage':2},motion_robustness={'version':'fixture'},motion_timing={'stage':4})
 for i,row in enumerate(rows):
  # Command stage/mode changes do not change the physical comparison context.
  new={**row,'case_id':f'robust_0_{i}_new_index','_motion_dr':{**row['_motion_dr'],'skill_stage':2,'mode':'COMMAND_ACQUIRE'}}
  assert course_baseline(spec,new,records) is records[i]
  for key,value in [('seed',18),('seconds',30.),('family','other')]:
   assert course_baseline(spec,{**new,key:value},records) is None
  assert course_baseline(spec,{**new,'_motion_dr':{**new['_motion_dr'],'level':1}},records) is None
 with pytest.raises(ValueError,match='duplicate course baseline'):
  course_baseline(spec,rows[0],records+[records[0]])
 monkeypatch.setattr(training,'environment',lambda *a,**kw:pytest.fail('must reject duplicate before simulator creation'))
 with pytest.raises(ValueError,match='duplicate course baseline'):
  evaluate_cases(spec,tmp_path,torch.nn.Linear(1,1),rows,source_records=records+[records[0]])


@pytest.mark.parametrize('with_baseline',[True,False])
def test_actual_grouped_evaluator_reports_DR_baseline_or_unavailable(tmp_path,monkeypatch,with_baseline):
 import json,sys
 from native_stand.students import training,grouped_evaluation,motion_reacquisition
 out=tmp_path/'chain/formal';(out/'source_snapshot').mkdir(parents=True)
 rows=[{**cases(1)[0],'case_id':f'robust_0_{g}_0','_timing_level':4,
        '_motion_dr':dict(level=0,group=g,mode='COMMAND_ACQUIRE',skill_stage=2)} for g in range(3)]
 spec={'performance':{'version':'student_performance_v2','evaluation_group_commands':True,'evaluation_reuse_environment':True,'evaluation_gpu_buffer':True},
  'student_evaluation':{'cases':rows,'max_output_bytes':10**6},'student_training':{'version':'student_motion31_training_v1','storage_max_bytes':10**7},
  'artifacts':{'terminal_reserve_bytes':10000},'asset_sha256':'b'*64,'motion_training':{'stage':2},'motion_course':{'version':'fixture'},
  'motion_robustness':{'version':'fixture'},'motion_timing':{'stage':4},'reward':{'version':'fixture'}}
 (out/'source_snapshot/spec.json').write_text(json.dumps(spec))
 env=SimpleNamespace(ids=list(range(6)),reference=torch.zeros(8,6),robot=SimpleNamespace(joint_names=['joint']),close=lambda:None)
 monkeypatch.setattr(training,'environment',lambda *a,**k:env)
 monkeypatch.setitem(sys.modules,'native_stand.command_v6.runtime.configure',SimpleNamespace(verify_asset_import=lambda s:{}))
 monkeypatch.setitem(sys.modules,'native_stand.command_v6.runtime.control_readback',SimpleNamespace(verify=lambda e:{},verify_ground=lambda e,s:{}))
 monkeypatch.setattr(training,'compact_motion_evidence',lambda s,a,p:a)
 monkeypatch.setattr(motion_reacquisition,'assess',lambda *a:None)
 monkeypatch.setattr(motion_reacquisition,'scoring_protocol',lambda s:{})
 def per_env(pitch):return [dict(env=j,entry_index=None,accepted=True,checks={'physical':True},tracking=dict(pitch_hold_abs_mean_rad=pitch,roll_hold_abs_mean_rad=0.)) for j in range(8)]
 def grouped(e,actor,group,**kw):
  return {row['case_id']:({'case':row,'per_env':per_env(row['_motion_dr']['group']*.01)},
   {'state':np.zeros((1,8,39),np.float32),'first_episode_valid':np.ones((1,8),bool)}) for row in group}
 monkeypatch.setattr(grouped_evaluation,'rollout',grouped)
 baseline=[dict(case=row,per_env=per_env(row['_motion_dr']['group']*.01)) for row in rows] if with_baseline else []
 result=training.evaluate_cases(spec,out,torch.nn.Linear(1,1),rows,source_records=baseline)
 assert len(result)==3
 for rec in result:
  raw=json.loads((out/rec['path']).read_text());delta=raw['per_env'][0]['source_posture_delta_rad']['pitch']
  assert delta==(0. if with_baseline else None)
  assert raw['accepted'] and raw['passed_envs']==8
  assert ('same command/DR' if with_baseline else 'NOT_AVAILABLE') in raw['source_comparison_scope']

def test_v2_is_opt_in_and_motion_only():
 assert 'combined_step_checks' not in settings({})
 with pytest.raises(ValueError):settings({'performance':{'combined_step_checks':True}})
 spec={'student_training':{'version':'student_motion31_training_v1'},'student_evaluation':{'cases':cases()},
 'performance':{'version':'student_performance_v2','combined_step_checks':True,'fixed_shape_statistics':True,
 'evaluation_group_commands':True,'evaluation_reuse_environment':True,'evaluation_gpu_buffer':True}}
 assert settings(spec)['evaluation_group_commands']
 bad=copy.deepcopy(spec);bad['performance']['evaluation_gpu_buffer']=False
 with pytest.raises(ValueError,match='trace'):settings(bad)
 bad=copy.deepcopy(spec);bad['performance']['trace_buffer_max_bytes']=2**20
 with pytest.raises(ValueError,match='conservative peak'):settings(bad)
 bad=copy.deepcopy(spec);bad['student_training']['version']='recovery'
 with pytest.raises(ValueError,match='Motion'):settings(bad)


def test_combined_encoder_and_clock_match_strict_path_and_faults():
 from native_stand.students.interface import frame
 from native_stand.students.mcu_interface import task,actor_input,PolicyClock
 from native_stand.students.motion_privileged import critic_input
 torch.manual_seed(43);n=8
 q=torch.randn(n,6);dq=torch.randn(n,6);gyro=torch.randn(n,3);gravity=torch.zeros(n,3);gravity[:,1]=-1
 legs=torch.randn(n,4);wheels=torch.randn(n,2);ref=torch.zeros(n,6);cmd=torch.zeros(n,3);cmd[:,2]=.34
 def encode(checks):
  f=frame(gyro,gravity,q,dq,legs,wheels,ref,checks=checks)
  actor=actor_input(f,task('Motion',cmd,checks=checks),checks=checks)
  critic=critic_input(actor,torch.ones(n)*.34,torch.zeros(n,3),torch.ones(n,2)*20,cmd,
   torch.zeros(n),torch.ones(n),torch.ones(n,dtype=torch.bool),torch.zeros(n,6),checks=checks)
  return actor,critic
 rng=torch.get_rng_state().clone()
 a,c=encode(None);checks=StepChecks();b,d=encode(checks);checks.verify()
 torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(c,d,rtol=0,atol=0)
 assert torch.equal(rng,torch.get_rng_state())
 clock=PolicyClock(n);fused_clock=PolicyClock(n)
 for time in (0.,.01,.01,.02):
  t=torch.ones(n)*time;clock.check(t);checks=StepChecks();fused_clock.check(t,checks=checks);checks.verify()
  torch.testing.assert_close(clock.last,fused_clock.last)
 checks=StepChecks();fused_clock.check(torch.ones(n)*.05,checks=checks)
 with pytest.raises(ValueError,match='policy time'):checks.verify()
 gyro[0,0]=float('nan');checks=StepChecks();encode(checks)
 with pytest.raises(ValueError,match='student frame'):checks.verify()
 with pytest.raises(ValueError):encode(None)
 checks=StepChecks();checks.require(torch.tensor([True,False]),'runtime fault')
 with pytest.raises(RuntimeError,match='runtime fault'):checks.verify(RuntimeError)


@pytest.mark.parametrize('n',[64,512])
def test_fixed_shape_statistics_match_dynamic_reference_including_empty_and_nan(n):
 torch.manual_seed(9)
 a=MotionStatistics([[0,0],[1,0],[0,-.5]],'cpu',schedule_s=[.5,1,25.5,1],parking_s=2)
 b=MotionStatistics(a.commands,'cpu',schedule_s=[.5,1,25.5,1],parking_s=2,fixed_shape=True)
 for step in range(7):
  index=torch.randint(0,3,(n,));valid=torch.rand(n)>.25
  age=torch.rand(n)*31;cmd=torch.randn(n,3);s=torch.randn(n,39);p=torch.randn(n,10);g=torch.randn(n,13)
  # NaN in excluded environments must not contaminate a cohort (0*nan is not sufficient).
  s[~valid,7]=float('nan')
  for stats in (a,b):stats.observe(index,valid,age,cmd,s,p,g)
 torch.testing.assert_close(a.count,b.count,rtol=0,atol=0)
 torch.testing.assert_close(a.sum,b.sum,rtol=1e-6,atol=1e-5)
 torch.testing.assert_close(a.direction,b.direction,rtol=0,atol=0)
 torch.testing.assert_close(a.yaw_direction,b.yaw_direction,rtol=0,atol=0)
 assert b.export()['hold']['cohort_0']['samples']==a.export()['hold']['cohort_0']['samples']


def test_batch_planner_keeps_seed_checkpoint_and_case_order_and_bounds_memory():
 rows=cases(11);groups=plan(rows)
 assert list(map(len,groups))==[4,4,3] and [r for group in groups for r in group]==rows
 rows=cases(3);rows[1]['seed']=18
 assert len(plan(rows))==3
 bad=cases();bad[1]['role']='RecoveryStudent'
 with pytest.raises(ValueError,match='unsupported'):plan(bad)
 with pytest.raises(ValueError):plan(cases(),scored_envs=7)
 # Conservative640 float fields, max4 groups,31.52s fits256MiB device trace cap.
 assert 640*4*3152*(8*4)<256*2**20


def test_split_preserves_individual_failure_windows_and_does_not_alias():
 valid=np.ones((3000,16),dtype=bool);valid[50:,2]=False
 data={'state':np.zeros((3000,16,39),np.float32),'first_episode_valid':valid,'policy_time_s':np.arange(3000)*.01}
 first=split_arrays(data,0);second=split_arrays(data,8)
 assert not first['first_episode_valid'][50,2] and second['first_episode_valid'].all()
 assert first['state'].shape==(3000,8,39) and len(first['policy_time_s'])==3000
 first['state'][0,0,0]=9;assert data['state'][0,0,0]==0


def test_grouped_real_scoring_matches_same_arrays_serial_scoring(monkeypatch):
 from native_stand.students import evaluator,grouped_evaluation
 from native_stand.students.motion_guard import protocol
 rows=cases(2);steps=3000;envs=16
 state=np.zeros((steps,envs,39),np.float32);state[:,:,6]=.34;state[:,:,10:12]=20
 state[:,:,13]=.1;state[:,:,29]=1
 posture=np.zeros((steps,envs,10),np.float32);posture[:,:,2:4]=.3
 geometry=np.zeros((steps,envs,13),np.float32);geometry[:,:,7]=-1
 state[:,8:,9]=.5
 arrays={'state':state,'posture':posture,'geometry':geometry,'root_post':np.zeros((steps,envs,13),np.float32),
 'controller':np.full((steps,envs,1),3),'first_episode_valid':np.ones((steps,envs),bool),
 'policy_time_s':np.arange(1,steps+1)*.01,'termination_causes':np.zeros((steps,envs,6),bool)}
 env=SimpleNamespace(num_envs=64,interface_version='multiskill_mcu_v2',motion_training_enabled=True,motion_absolute_protocol=protocol(),ids=list(range(6)))
 expected=[evaluator.score_arrays(env,row,split_arrays(arrays,i*8))[0] for i,row in enumerate(rows)]
 seen=[]
 def collect(e,a,row,**kw):seen.append((row,kw));return arrays
 monkeypatch.setattr(evaluator,'rollout',collect)
 actual=grouped_evaluation.rollout(env,torch.nn.Linear(1,1),rows)
 assert len(seen)==1 and seen[0][1]['collect_only'] is True and seen[0][1]['scored_envs']==16
 for row,old in zip(rows,expected):
  new=actual[row['case_id']][0]
  assert new['per_env']==old['per_env'] and new['protocol']==old['protocol']
  assert new['scored_envs']==8 and new['case']==row
  assert len(new['execution_layout']['scored_env_ids'])==8


def test_group_fault_restores_rng_modes_and_temporary_command_matrix():
 from test_student_performance import FakeEnv
 from native_stand.students.evaluator import rollout
 class Actor(torch.nn.Module):
  def forward(self,obs,stochastic_output=False):return torch.zeros(len(obs['student_actor']),6)
 env=FakeEnv(buffered=True,fault_at=3);actor=Actor();actor.train()
 rng=torch.get_rng_state().clone();original=env.forced_target=('old',);env.motion_group_targets=torch.ones(2,2)
 saved=env.motion_group_targets
 with pytest.raises(RuntimeError,match='fixture'):
  rollout(env,actor,{'kind':'student','family':'normal','seconds':.04,'seed':7,'yaw_rad':0.},scored_envs=2)
 assert actor.training and torch.equal(rng,torch.get_rng_state())
 assert env.forced_target==original and env.motion_group_targets is saved


def test_fixed_statistics_hot_path_has_no_boolean_selection():
 # Structural check complements numerical reference tests; ensures the optimized branch does not regress to dynamic shapes.
 tree=ast.parse(Path('src/native_stand/students/motion_statistics.py').read_text())
 branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.fixed_shape')
 calls=[ast.unparse(n.func) for n in ast.walk(branch) if isinstance(n,ast.Call)]
 assert not any(x.endswith(('nonzero','bincount','tolist','cpu','item')) for x in calls)


def test_group_commands_survive_subset_reset_without_affecting_training_path():
 # Execute the actual new runtime methods over a tiny reset owner; no Isaac imports.
 tree=ast.parse(Path('src/native_stand/students/motion_runtime.py').read_text())
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MotionStudentEnv')
 methods=[copy.deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_reset_idx','set_grouped_evaluation_targets')]
 class Parent:
  def _reset_idx(self,ids):self.target_xy[ids]=0
 class_definition=ast.ClassDef(name='Runtime',bases=[ast.Name(id='Parent',ctx=ast.Load())],keywords=[],body=methods,decorator_list=[])
 ns={'torch':torch,'Parent':Parent}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[class_definition],type_ignores=[])),'<runtime methods>','exec'),ns)
 env=ns['Runtime']();env.num_envs=64;env.device='cpu';env.curriculum_training=False;env.target_xy=torch.zeros(64,2)
 values=torch.zeros(64,2);values[:8,0]=1;values[8:16,1]=-.5
 env.set_grouped_evaluation_targets(values)
 env._reset_idx(torch.tensor([1,9,20]))
 torch.testing.assert_close(env.target_xy,values,rtol=0,atol=0)
 env.curriculum_training=True
 with pytest.raises(ValueError):env.set_grouped_evaluation_targets(values)


def test_grouped_collect_once_and_fault_before_bad_command_action():
 from test_student_performance import FakeEnv
 from native_stand.students.evaluator import rollout
 class Actor(torch.nn.Module):
  def forward(self,obs,stochastic_output=False):return torch.zeros(len(obs['student_actor']),6)
 class MotionFake(FakeEnv):
  def __init__(self):
   super().__init__(buffered=True);self.num_envs=64
   self.motion_training_enabled=True;self.steps=0;self.command=torch.zeros(64,3)
   self.target_xy=torch.zeros(64,2);self.rise_hold=SimpleNamespace(episode=SimpleNamespace(steps=torch.zeros(64)))
  def select_initial_family(self,*a):pass
  def set_grouped_evaluation_targets(self,values):self.motion_group_targets=values;self.target_xy.copy_(values)
  def _reset_idx(self,ids):pass
  def _get_observations(self):
   from native_stand.students.motion_training import effective
   command=effective(self.target_xy,self.rise_hold.episode.steps*.01)
   actor=torch.zeros(64,31);actor[:,22:25]=command/torch.tensor([4.,4.,.5]);actor[:,25]=1
   if getattr(self,'corrupt',False):actor[0,22]=9
   return {'student_actor':actor,'student_critic':torch.zeros(64,49)}
  def reset(self):
   self.fsm.mode=torch.ones(64,dtype=torch.long)*2;self.initialization_checked=torch.ones(64,dtype=torch.bool)
   self.executed_leg=torch.zeros(64,4);self.executed_wheel=torch.zeros(64,2)
   self.robot=SimpleNamespace(data=SimpleNamespace(root_state_w=SimpleNamespace(torch=torch.zeros(64,13)),joint_pos=SimpleNamespace(torch=torch.zeros(64,6)),joint_vel=SimpleNamespace(torch=torch.zeros(64,6))))
   self.recovery_snapshot=torch.zeros(64,39);self.recovery_snapshot[:,29]=1
   self.standing_snapshot=torch.zeros(64,10);self.recovery_control_snapshot=torch.zeros(64,54);self.wheel_geometry_snapshot=torch.zeros(64,13)
   self.student_root_post=torch.zeros(64,13);self.recovery_cause_snapshot=torch.zeros(64,6,dtype=torch.bool)
   return self._get_observations(),{}
  def step(self,action):
   self.steps+=1;self.rise_hold.episode.steps+=1
   done=torch.zeros(64,dtype=torch.bool);done[0]=self.steps==2
   return self._get_observations(),torch.zeros(64),done,torch.zeros(64,dtype=torch.bool),{}
 env=MotionFake();env.student_performance.update(combined_step_checks=True)
 group=cases(2)
 for row in group:row['seconds']=.04
 row={**group[0],'_group_cases':group};actor=Actor();actor.train();rng=torch.get_rng_state().clone()
 arrays=rollout(env,actor,row,scored_envs=16,collect_only=True)
 assert env.steps==4 and arrays['state'].shape==(4,16,39)
 assert arrays['first_episode_valid'][:,0].tolist()==[True,True,False,False]
 assert actor.training and torch.equal(rng,torch.get_rng_state()) and not hasattr(env,'motion_group_targets')
 env=MotionFake();env.student_performance.update(combined_step_checks=True);env.corrupt=True
 with pytest.raises(RuntimeError,match='readback'):rollout(env,actor,row,scored_envs=16,collect_only=True)
 assert env.steps==0 # Fusing checks must never postpone this failure until after an action.


@pytest.mark.parametrize('fault',[False,True])
def test_training_evaluate_cases_uses_groups_and_keeps_individual_evidence(tmp_path,monkeypatch,fault):
 import json,sys
 from native_stand.students import training,grouped_evaluation
 from native_stand.students.data import sha
 output=tmp_path/'chain'/'formal';output.mkdir(parents=True);(output/'source_snapshot').mkdir()
 rows=cases(7)
 spec={'performance':{'version':'student_performance_v2','evaluation_group_commands':True,'evaluation_gpu_buffer':True,'evaluation_reuse_environment':True},
 'student_evaluation':{'cases':rows,'max_output_bytes':10**6},'student_training':{'version':'student_motion31_training_v1','storage_max_bytes':10**7},
 'artifacts':{'terminal_reserve_bytes':10000},'asset_sha256':'b'*64}
 (output/'source_snapshot/spec.json').write_text(json.dumps(spec))
 class Env:
  ids=list(range(6));reference=torch.zeros(8,6);robot=SimpleNamespace(joint_names=['joint'])
  def close(self):self.closed=getattr(self,'closed',0)+1
 env=Env();monkeypatch.setattr(training,'environment',lambda *a,**k:env)
 monkeypatch.setitem(sys.modules,'native_stand.command_v6.runtime.configure',SimpleNamespace(verify_asset_import=lambda s:{}))
 monkeypatch.setitem(sys.modules,'native_stand.command_v6.runtime.control_readback',SimpleNamespace(verify=lambda e:{},verify_ground=lambda e,s:{}))
 calls=[]
 def grouped(e,actor,group,**kw):
  calls.append([row['case_id'] for row in group])
  if fault:raise RuntimeError('whole group physics fault')
  return {row['case_id']:({'case':row,'execution_layout':{'scored_env_ids':list(range(i*8,i*8+8))},
    'per_env':[{'env':j,'entry_index':None,'accepted':False} for j in range(8)]},
    {'state':np.zeros((1,8,39),np.float32),'first_episode_valid':np.ones((1,8),bool)}) for i,row in enumerate(group)}
 monkeypatch.setattr(grouped_evaluation,'rollout',grouped)
 actor=torch.nn.Linear(1,1);rng=torch.get_rng_state().clone()
 if fault:
  with pytest.raises(RuntimeError,match='whole group'):training.evaluate_cases(spec,output,actor,rows)
  assert [r['status'] for r in json.loads((output/'evaluation_matrix_00.json').read_text())]==['INCOMPLETE']*4+['NOT_RUN']*3
  assert not list(output.glob('evaluation_0[0-9].json'))
 else:
  result=training.evaluate_cases(spec,output,actor,rows)
  assert calls==[['0','1','2','3'],['4','5','6']]
  assert [row['case']['case_id'] for row in result]==[row['case_id'] for row in rows]
  for record in result:
   path=output/record['path'];raw=json.loads(path.read_text())
   assert sha(path)==record['sha256'] and raw['case']==record['case'] and len(raw['per_env'])==8
 assert env.closed==1 and torch.equal(rng,torch.get_rng_state())


def test_actual_motion_observation_methods_match_and_skip_unused_legacy_forward(monkeypatch):
 import sys
 from native_stand.students.interface import History,frame,task,critic_input,targets
 from native_stand.students.mcu_interface import PolicyClock
 from native_stand.students.motion_training import effective
 math_module=SimpleNamespace(quat_apply_inverse=lambda quat,vec:vec)
 monkeypatch.setitem(sys.modules,'isaaclab.utils.math',math_module)
 class Parent:
  def _get_observations(self):self.legacy_calls+=1;self.command.fill_(42);return {'legacy':True}
 def method_class(path,name,base,ns):
  tree=ast.parse(Path(path).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
  method=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_observations'))
  definition=ast.ClassDef(name=name,bases=[ast.Name(id=base,ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
  exec(compile(ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[])),'<real observation method>','exec'),ns)
 ns={'__name__':'native_stand.students.runtime','__package__':'native_stand.students','torch':torch,'Parent':Parent,'frame':frame,'task':task,'critic_input':critic_input,'effective':effective,'Mode':SimpleNamespace(INITIALIZING=0)}
 method_class('src/native_stand/students/runtime.py','StudentEnv','Parent',ns)
 method_class('src/native_stand/students/motion_runtime.py','MotionStudentEnv','StudentEnv',ns)
 def env(fast):
  e=ns['MotionStudentEnv']();n=8;e.num_envs=n;e.device='cpu';e.legacy_calls=0
  e.student_ready=True;e.motion_training_enabled=True;e.motion_manual_command=None;e.motion_config={}
  e.student_performance={'combined_step_checks':fast};e.curriculum_training=False;e.critic_privileged_version='motion_privileged49_v1'
  e.target_xy=torch.tensor([[0.,.5]]).expand(n,-1).clone();e.command=torch.zeros(n,3)
  e.rise_hold=SimpleNamespace(episode=SimpleNamespace(steps=torch.ones(n)*1200),age=torch.zeros(n),dwell=torch.zeros(n),count=torch.zeros(n))
  e.robot=SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.zeros(n,6)),joint_vel=SimpleNamespace(torch=torch.zeros(n,6)),
   root_quat_w=SimpleNamespace(torch=torch.zeros(n,4)),root_lin_vel_w=SimpleNamespace(torch=torch.zeros(n,3)),applied_torque=SimpleNamespace(torch=torch.zeros(n,6))))
  e.fsm=SimpleNamespace(mode=torch.ones(n,dtype=torch.long));e.interface_version='multiskill_mcu_v2';e.student_role='MotionZeroStudent'
  e.student_command_height=.34;e.root_height=torch.ones(n)*.34;e.contact_force=torch.ones(n,2)*20;e.yaw=torch.zeros(n);e.ids=list(range(6))
  e.executed_leg=torch.zeros(n,4);e.executed_wheel=torch.zeros(n,2);e.reference=torch.zeros(n,6)
  e.student_history=History(n);e.student_obs=None;e.mcu_clock=PolicyClock(n);e.episode_length_buf=torch.zeros(n);e.step_dt=.01
  e._motion_state=lambda :(torch.zeros(n,3),torch.tensor([[0.,-1.,0.]]).expand(n,-1))
  return e
 old=env(False);new=env(True);a=old._get_observations();b=new._get_observations()
 assert old.legacy_calls==1 and new.legacy_calls==0
 assert a.keys()==b.keys()=={'student_actor','student_critic'}
 for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
 assert b['student_critic'].shape==(8,49) and b['student_actor'].shape==(8,31)
 assert new.motion_observation_checks is None


def test_new_motion_defaults_enable_safe_optimizations_without_reinterpreting_v1():
 from native_stand.students.performance import new_motion_defaults
 old={'version':'student_performance_v1','evaluation_gpu_buffer':False}
 before=copy.deepcopy(old);cfg=new_motion_defaults(old)
 assert old==before and settings({'performance':old})['version']=='student_performance_v1'
 assert cfg['version']=='student_performance_v2'
 assert all(cfg[k] for k in ('combined_step_checks','fixed_shape_statistics','evaluation_gpu_buffer','evaluation_reuse_environment','training_gpu_statistics','failure_tail_on_device'))
 assert cfg['evaluation_group_commands'] is False
 assert cfg['motion_hotpath'] and cfg['motion_runtime_reduction']
 # Loading a pre-existing v2 spec never acquires the new preparation defaults.
 assert not settings({'performance':{'version':'student_performance_v2'}})['motion_runtime_reduction']
 assert new_motion_defaults(old,grouped_evaluation=True)['evaluation_group_commands'] is True
 for filename in ('scripts/prepare_motion31_yaw.py','scripts/prepare_motion31_confirmation.py'):
  tree=ast.parse(Path(filename).read_text());prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
  defaults={arg.arg:ast.literal_eval(value) for arg,value in zip(prepare.args.kwonlyargs,prepare.args.kw_defaults)}
  assert defaults['performance_v2'] is True and defaults['grouped_evaluation'] is False
