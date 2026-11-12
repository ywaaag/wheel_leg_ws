FROZEN_TEST_SCOPE='PORTABLE'
"""Behavioural CPU tests for the explicit post-PD MCU playback chain.

The chain is the only place that may advance motor transport/response, so these
tests drive 1ms ticks and check the declared timings instead of reading comments.
"""
import ast,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'scripts') not in sys.path:sys.path.insert(0,str(ROOT/'scripts'))
MCU_SPEC=ROOT/'configs/experiments/2026_10_25-17_15__motion500_mcu312_interface_resume/spec.json'
LEGACY_SPEC=ROOT/'configs/experiments/2026_10_23-15_16__motion_fixed_height184_9953/spec.json'
Q=np.array([.1,.1,0.,.1,.1,0.]);DQ=np.array([0.,0.,1.,0.,0.,-1.])
ZERO=np.zeros(6)


def chain(delay=None,tau=None):
 from native_stand.students.motion_mcu_execution import profile
 p=profile()
 if delay is not None:p['motor_delay_ms']=list(delay)
 if tau is not None:p['motor_tau_ms']=list(tau)
 return {'version':p['version'],'profile':p}


def spec_with(delay=None):
 from native_stand.students.motion_mcu_execution import profile
 p=profile()
 if delay is not None:p['motor_delay_ms']=list(delay)
 return {'mcu_execution':{'profile':p,'pose':{'height_m':.34,'root_quaternion_xyzw':[.7071,0,0,.7071],
  'joint_pos':{'left_hip_pitch':0.,'left_active_link_pitch':0.,'left_wheel_spin':0.,
   'right_hip_pitch':0.,'right_active_link_pitch':0.,'right_wheel_spin':0.}}}}


def playback():
 from motion_mcu_playback import MCUPlayback
 return MCUPlayback(chain(),Q,ZERO,tick=0)


def test_declared_dispatch_is_explicit_version_only():
 from motion_mcu_playback import VERSION,declared
 assert declared({}) is None and declared({'motion_timing':{}}) is None
 block=declared(spec_with())
 assert block['version']==VERSION and block['profile']['motor_delay_ms']==[10,10,7,10,10,11]
 with pytest.raises(ValueError,match='frozen contract'):declared(spec_with(delay=[0]*6))
 unknown=spec_with();unknown['mcu_execution']['profile']['version']='motion_unknown_v9'
 with pytest.raises(ValueError,match='unknown MCU execution version'):declared(unknown)
 with pytest.raises(ValueError,match='pose'):declared({'mcu_execution':{'profile':spec_with()['mcu_execution']['profile']}})


def test_new_spec_playback_consumes_its_own_execution_profile():
 pytest.importorskip('mujoco')
 if not MCU_SPEC.exists():pytest.skip('MCU spec fixture absent')
 from validate_motion31_mujoco import actuator_cfg
 spec=json.loads(MCU_SPEC.read_text())
 cfg=actuator_cfg(SimpleNamespace(leg_armature_m=.012,zero_latency=False),spec)
 assert cfg['mcu'] and cfg['mcu_version']=='motion_mcu_post_pd_execution_v1'
 assert cfg['transport_ms']==[10,10,7,10,10,11] and cfg['tau_ms']==[5,5,0,5,5,0]
 assert cfg['transport_placement']=='post_PD_motor_torque_command' and cfg['local_target_transport_ms']==0
 assert cfg['feedback_ms']==10 and cfg['actor_sensor_ms']==10
 assert cfg['pd_limits_nm']==[40,40,5.5,40,40,5.5] and cfg['solver_limits_nm'][0]==67
 assert abs(cfg['driver_limits_nm'][2]-2.44694786377709)<1e-9 and cfg['driver_limits_nm'][0]==40
 assert cfg['execution_source']=='mcu_execution.profile' and cfg['leg_armature']==.012
 with pytest.raises(ValueError,match='training-fixed'):actuator_cfg(SimpleNamespace(leg_armature_m=.35),spec)
 zero=actuator_cfg(SimpleNamespace(leg_armature_m=.012,zero_latency=True),spec)
 assert zero['zero_latency'] and zero['transport_ms']==[0]*6 and zero['tau_ms']==[0]*6 and zero['feedback_ms']==0
 assert zero['profile']['motor_delay_ms']==[0]*6 and zero['declared']['motor_delay_ms']==[10,10,7,10,10,11]
 assert (zero['kp'],zero['kd'],zero['kv'],zero['driver_limits_nm'][2])==(40.,2.,.2,cfg['driver_limits_nm'][2])


def test_forced_mcu_chain_keeps_the_recipe_start_and_armature():
 pytest.importorskip('mujoco')
 if not LEGACY_SPEC.exists():pytest.skip('legacy spec fixture absent')
 from validate_motion31_mujoco import actuator_cfg
 spec=json.loads(LEGACY_SPEC.read_text())
 plain=actuator_cfg(SimpleNamespace(leg_armature_m=.35,mcu_execution=False),spec)
 forced=actuator_cfg(SimpleNamespace(leg_armature_m=.35,mcu_execution=True),spec)
 assert not plain.get('mcu') and forced['mcu']
 assert forced['mcu_version']=='motion_mcu_post_pd_execution_v1' and forced['forced_by_cli']
 assert forced['mcu_pose'] is None  # no declared MCU pose: keep the recipe initial state
 assert forced['leg_armature']==.35 and forced['armature_source']=='--leg-armature-m explicit (forced MCU chain)'
 assert forced['transport_placement']=='post_PD_motor_torque_command'
 assert forced['transport_ms']==[10,10,7,10,10,11] and forced['tau_ms']==[5,5,0,5,5,0]
 assert 'user-explicit' in forced['execution_source']
 with pytest.raises(ValueError,match='declared level'):actuator_cfg(SimpleNamespace(leg_armature_m=.021),spec)


def test_legacy_spec_keeps_its_frozen_hypothesis():
 pytest.importorskip('mujoco')
 if not LEGACY_SPEC.exists():pytest.skip('legacy spec fixture absent')
 from validate_motion31_mujoco import actuator_cfg
 spec=json.loads(LEGACY_SPEC.read_text())
 cfg=actuator_cfg(SimpleNamespace(leg_armature_m=.35,zero_latency=False),spec)
 assert not cfg.get('mcu') and cfg['leg_armature']==.35
 assert cfg['transport_ms']==[10,10,7,10,10,11] and cfg['tau_ms']==[5,5,0,5,5,0]
 assert actuator_cfg(SimpleNamespace(leg_armature_m=None),spec) is None


def test_pd_is_every_5ms_while_response_advances_every_1ms():
 p=playback();previous_pre=None;previous_motor=None;pre_changes=[];motor_changes=0
 for tick in range(40):
  p.step(tick,ZERO,Q,DQ)
  pre=p.pre.numpy();motor=p.motor_actual.numpy()
  if previous_pre is not None and not np.allclose(pre,previous_pre):pre_changes.append(tick)
  if previous_motor is not None and not np.allclose(motor,previous_motor):motor_changes+=1
  previous_pre=pre;previous_motor=motor
 assert pre_changes and all(tick%5==0 for tick in pre_changes)
 assert motor_changes>len(pre_changes)
 assert motor_changes>=20  # the relaxed response updates on almost every 1ms tick


def test_each_wheel_arrives_at_its_declared_transport():
 from native_stand.students.motion_mcu_execution import profile
 declared=profile();p=playback()
 first_driver={i:None for i in range(6)};first_motor={i:None for i in range(6)}
 for tick in range(80):
  applied=p.step(tick,ZERO,Q,DQ).numpy();driver=p.driver.numpy()
  for channel in range(6):
   if first_driver[channel] is None and abs(driver[channel])>1e-9:first_driver[channel]=tick
   if first_motor[channel] is None and abs(applied[channel])>1e-9:first_motor[channel]=tick
 for channel in range(6):
  assert first_driver[channel] is not None and first_motor[channel] is not None
  assert first_motor[channel]-first_driver[channel]==declared['motor_delay_ms'][channel]
 assert declared['motor_delay_ms'][2]==7 and declared['motor_delay_ms'][5]==11


def test_previous_is_the_adopted_target_and_ignores_off_cycle_requests():
 p=playback();p.step(0,np.full(6,.2),Q,DQ)
 assert np.allclose(p.adopted.numpy(),.2) and np.allclose(p.previous.numpy(),.2)
 for tick in range(1,10):
  p.step(tick,np.full(6,.9),Q,DQ)  # off-cycle requests are not adopted
 assert np.allclose(p.adopted.numpy(),.2) and np.allclose(p.previous.numpy(),.2)
 p.step(10,np.full(6,.4),Q,DQ)
 assert np.allclose(p.adopted.numpy(),.4) and np.allclose(p.previous.numpy(),.4)
 for tick in range(11,20):p.step(tick,np.full(6,.3),Q,DQ)
 target=np.full(6,.7);p.step(20,target,Q,DQ)
 assert np.allclose(p.adopted.numpy(),.7);target[:]=.0
 assert np.allclose(p.adopted.numpy(),.7)


def test_reset_clears_queue_previous_and_restores_declared_initial_motor():
 from motion_mcu_playback import MCUPlayback
 from native_stand.students.motion_mcu_execution import spring_compensation
 p=playback()
 for tick in range(30):p.step(tick,np.full(6,.3),Q,DQ)
 q2=np.array([.2,.15,0.,.2,.15,0.]);target2=np.array([.2,.15,0.,.2,.15,0.])
 p.reset(100,q2,target2)
 compensation=spring_compensation(torch.tensor(q2[[0,1,3,4]],dtype=torch.float32)[None])[0][0]
 np.testing.assert_allclose(p.driver.numpy(),compensation.numpy(),atol=1e-6)
 np.testing.assert_allclose(p.motor_actual.numpy(),compensation.numpy(),atol=1e-6)
 np.testing.assert_allclose(p.spring_actual.numpy(),-compensation.numpy(),atol=1e-6)
 np.testing.assert_allclose(p.actual.numpy(),np.zeros(6),atol=0)
 np.testing.assert_allclose(p.previous.numpy(),target2)
 np.testing.assert_allclose(p.adopted.numpy(),target2)
 assert torch.allclose(p.ring.history[:,0],compensation.expand(64,6),atol=1e-6)
 p.step(100,target2,Q,DQ);p.step(101,target2,Q,DQ)
 with pytest.raises(ValueError,match='exactly1ms'):p.step(105,target2,Q,DQ)


def test_spring_reaction_uses_current_geometry_and_sums_with_motor():
 from native_stand.students.motion_mcu_execution import spring_compensation
 p=playback()
 for tick in range(20):p.step(tick,ZERO,Q,DQ)
 aged=p.compensation.numpy().copy()
 moved=np.array([.25,.2,0.,.25,.2,0.])
 p.step(20,ZERO,moved,DQ)
 assert np.allclose(p.compensation.numpy(),aged)  # compensation follows the aged PD feedback
 current=spring_compensation(torch.tensor(moved[[0,1,3,4]],dtype=torch.float32)[None])[0][0].numpy()
 np.testing.assert_allclose(p.spring_actual.numpy(),-current,atol=1e-6)
 np.testing.assert_allclose(p.actual.numpy(),p.motor_actual.numpy()+p.spring_actual.numpy(),atol=0)
 assert p.spring_actual.numpy()[[2,5]].tolist()==[0.,0.]
 for tick in range(21,31):p.step(tick,ZERO,moved,DQ)
 assert np.allclose(p.compensation.numpy(),current)  # a later PD tick re-reads the aged sample


def test_playback_pd_matches_the_training_pure_function():
 from native_stand.students.motion_mcu_execution import pd_and_motor
 p=playback()
 for tick in range(25):p.step(tick,np.full(6,.15),Q,DQ)
 feedback=p.history.at(25-int(p.profile['PD_age_ms']))
 request,limited,compensation,motor,valid=pd_and_motor(
  torch.tensor(feedback[:6],dtype=torch.float32)[None],torch.tensor(feedback[6:],dtype=torch.float32)[None],
  p.adopted[None])
 np.testing.assert_allclose(p.pre.numpy(),request[0].numpy(),atol=1e-6)
 np.testing.assert_allclose(p.pd_limited.numpy(),limited[0].numpy(),atol=1e-6)
 np.testing.assert_allclose(p.driver.numpy(),motor[0].numpy(),atol=1e-6)
 assert bool(valid.all())==p.geometry_valid


def test_playback_starts_at_the_training_standing_height():
 from motion_mcu_playback import DEFAULT_START_HEIGHT_M,start_pose
 assert DEFAULT_START_HEIGHT_M==.34
 if not LEGACY_SPEC.exists():pytest.skip('legacy spec fixture absent')
 legacy=json.loads(LEGACY_SPEC.read_text())
 start=start_pose(legacy,None)
 assert start is not None and start['height_m']==.34 and len(start['joint_pos'])>10
 assert start['source'].startswith('recorded playback start pose')
 assert start['source_spec'] and start['source_spec_sha256']
 assert start_pose(legacy,0) is None    # explicit opt out keeps the recipe start
 assert start_pose(legacy,.4) is None   # no verified pose at another height: never scaled
 if MCU_SPEC.exists():
  declared=start_pose(json.loads(MCU_SPEC.read_text()),.34)
  assert declared['source']=='spec mcu_execution.pose' and declared['height_m']==.34


def test_loop_starts_the_legacy_chain_at_034_with_a_closed_chain():
 mujoco=pytest.importorskip('mujoco')
 import time
 from validate_native_mujoco import binding
 from validate_motion31_mujoco import actuator_cfg,case
 from compare_motion_timing_profiles import apply_actuator_profile
 xml=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml')
 if not (LEGACY_SPEC.exists() and xml.exists()):pytest.skip('legacy playback fixture absent')
 spec=json.loads(LEGACY_SPEC.read_text())
 cfg=actuator_cfg(SimpleNamespace(leg_armature_m=.35,mcu_execution=False),spec)
 model=mujoco.MjModel.from_xml_path(str(xml));refs=binding(model)
 original={k:getattr(model,k).copy() for k in ('dof_armature','dof_damping','dof_frictionloss')}
 apply_actuator_profile(model,refs,cfg,original)
 class ZeroActor:
  def mean(self,obs,name=None):return torch.zeros(1,6)
  def names(self):return ['zero']
 summary,arrays=case(spec,model,refs,ZeroActor(),[0.,0.],time.monotonic()+1.2,cfg=cfg)
 assert summary['controlled_samples']>0
 height=arrays['state'][:,3]
 assert abs(height[0]-.34)<2e-3,height[0]
 assert arrays['state'][0,7]<5e-3  # closed chain still holds on the first recorded step


def test_case_never_hides_the_execution_step_inside_the_5ms_branch():
 source=(ROOT/'scripts/validate_motion31_mujoco.py').read_text()
 function=next(node for node in ast.walk(ast.parse(source))
  if isinstance(node,ast.FunctionDef) and node.name=='case')
 hidden=[]
 for node in ast.walk(function):
  if isinstance(node,ast.If) and ast.unparse(node.test).replace(' ','')=='step%5==0':
   hidden+=[ast.unparse(call) for call in ast.walk(node)
    if isinstance(call,ast.Call) and ast.unparse(call.func).endswith('execution.step')]
 assert not hidden
 compact=ast.unparse(function).replace(' ','')
 # The frozen V4 path explicitly uses20ms while the old MCU path uses cfg.
 # Both branches must feed the same history consumer; do not pin obsolete text.
 sensed=next(node.value for node in ast.walk(function) if isinstance(node,ast.Assign)
  and any(isinstance(t,ast.Name) and t.id=='sensed' for t in node.targets)
  and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='history.at'
  and isinstance(node.value.args[0],ast.BinOp))
 delay=sensed.args[0]
 assert isinstance(delay.op,ast.Sub) and ast.unparse(delay.left)=='step'
 assert isinstance(delay.right,ast.IfExp)
 assert ast.literal_eval(delay.right.body)==20
 assert ast.unparse(delay.right.orelse)=="cfg['actor_sensor_ms']"
 assert 'previous=torch.as_tensor(np.asarray(execution.adopted)' in compact


def test_loop_runs_1ms_execution_with_the_declared_initial_pose():
 mujoco=pytest.importorskip('mujoco')
 xml=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml')
 if not (MCU_SPEC.exists() and xml.exists()):pytest.skip('MCU playback fixture absent')
 import time
 from validate_native_mujoco import binding
 from validate_motion31_mujoco import actuator_cfg,case
 from compare_motion_timing_profiles import apply_actuator_profile
 import sys
 sys.path.insert(0,str(ROOT/'scripts'))
 spec=json.loads(MCU_SPEC.read_text())
 cfg=actuator_cfg(SimpleNamespace(leg_armature_m=.012,zero_latency=False),spec)
 model=mujoco.MjModel.from_xml_path(str(xml));refs=binding(model)
 original={k:getattr(model,k).copy() for k in ('dof_armature','dof_damping','dof_frictionloss')}
 apply_actuator_profile(model,refs,cfg,original)
 class ZeroActor:
  def mean(self,obs,name=None):return torch.zeros(1,6)
  def names(self):return ['zero']
 summary,arrays=case(spec,model,refs,ZeroActor(),[0.,0.],time.monotonic()+1.5,cfg=cfg)
 assert summary['controlled_samples']>0
 assert arrays['mcu_motor_nm'].shape==(summary['controlled_samples'],6)
 assert arrays['mcu_driver_nm'].shape==(summary['controlled_samples'],6)
 assert np.isfinite(arrays['mcu_spring_nm']).all()
 assert np.allclose(arrays['applied_torque'],arrays['mcu_motor_nm']+arrays['mcu_spring_nm'],atol=1e-6)
