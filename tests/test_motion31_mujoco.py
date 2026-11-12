FROZEN_TEST_SCOPE='PORTABLE'
import ast,importlib.util,sys
from pathlib import Path
import numpy as np
import torch


def test_mcu_encoder_mean_and_physical_target_semantics():
 from native_stand.students.mcu_interface import encode
 from native_stand.students.interface import targets
 q=torch.zeros(1,6);legs=torch.tensor([[.1,.2,.3,.4]]);wheels=torch.tensor([[5.,-6.]])
 x=encode(torch.zeros(1,3),torch.tensor([[0.,-1.,0.]]),q,q,legs,wheels,q,'Motion',torch.tensor([[0.,1.,.34]]))
 assert x.shape==(1,31) and torch.allclose(x[:,22:25],torch.tensor([[0.,.25,.68]]))
 assert x[0,25]==1 and x[0,26]==0 and torch.allclose(x[0,20:22],torch.tensor([.05,-.06]))
 limits=torch.tensor([[[-1.,1.]]*6]);t=targets(torch.ones(1,6),q,limits)
 assert t['leg_executed'].eq(1.).all() and t['wheel_executed'].eq(100.).all()


def test_public_entry_explicit31d_dispatch_before_legacy_plan():
 root=Path(__file__).resolve().parents[1]
 source=(root/'scripts/validate_native_mujoco.py').read_text()
 assert source.index("return execute(a)")<source.index('    task_started=time.monotonic()')
 ast.parse((root/'scripts/validate_motion31_mujoco.py').read_text())


def test_normal_direct_zero_reset_not_passive_settle():
 source=(Path(__file__).resolve().parents[1]/'scripts/validate_motion31_mujoco.py').read_text()
 assert "pose.get('initialization_control')=='direct_zero_reset':control_start=0." in source

def test_trained_spin61_plan_keeps_actor31_and_never_loads_critic():
 import json
 from types import SimpleNamespace
 import pytest
 root=Path(__file__).resolve().parents[1]
 parent=root/'configs/experiments/2026_10_17-18_18__motion810_spin61_formal_recovery/spec.json'
 cp=root/'runs/2026_10_17-18_18__motion810_spin61_formal_recovery/formal/model_update_750.pt'
 xml=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml')
 if not all(p.exists() for p in (parent,cp,xml)):pytest.skip('local Motion750 fixture absent')
 sys.path.insert(0,str(root/'scripts'))
 from validate_motion31_mujoco import plan
 args=SimpleNamespace(spec=parent,checkpoint=cp,xml=xml,wall_seconds=600,commands=[0,.5,1,1.5,2],yaw_commands=[.5,-.5])
 _,_,_,m=plan(args,json.loads(parent.read_text()))
 assert m['critic_not_loaded'] and m['checkpoint_critic_version']=='motion_spin_privileged61_v1'
 assert len(m['initial_actor_input'])==31 and len(m['protocol']['commands'])==7

def test_zero_latency_profile_keeps_spec_pd_and_armature_only_when_asked():
 import sys,json,pytest
 from pathlib import Path
 from types import SimpleNamespace
 root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
 spec_path=root/'configs/experiments/2026_10_23-15_16__motion_fixed_height184_9953/spec.json'
 if not spec_path.exists():pytest.skip('local spec fixture absent')
 from validate_motion31_mujoco import actuator_cfg
 spec=json.loads(spec_path.read_text())
 declared=actuator_cfg(SimpleNamespace(leg_armature_m=.35,zero_latency=False),spec)
 assert declared['transport_ms']==[10,10,7,10,10,11] and declared['feedback_ms']==10
 assert declared['tau_ms']==[5,5,0,5,5,0] and declared['actor_sensor_ms']==10
 zero=actuator_cfg(SimpleNamespace(leg_armature_m=.35,zero_latency=True),spec)
 assert zero['zero_latency'] and zero['transport_ms']==[0]*6 and zero['feedback_ms']==0
 assert zero['tau_ms']==[0]*6 and zero['actor_sensor_ms']==0
 assert (zero['leg_armature'],zero['kp'],zero['kd'],zero['kv'])==(.35,40.,2.,.2)
 assert zero['declared']['transport_ms']==declared['transport_ms']


def test_onnx_preview_actor_matches_exported_pytorch_golden():
 import sys,json,pytest
 from pathlib import Path
 root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
 pkg=root/'deployment_protocol/2026_10_23-19_53__motion500_currentplant_small31_onnx'
 spec=root/'configs/experiments/2026_10_23-15_16__motion_fixed_height184_9953/spec.json'
 if not (pkg/'actor.onnx').exists() or not spec.exists():pytest.skip('ONNX export fixture absent')
 from validate_motion31_mujoco import OnnxActor
 actor=OnnxActor(pkg/'actor.onnx',pkg/'manifest.json',spec)
 golden=np.load(pkg/'golden_vectors.npz')
 obs=torch.tensor(golden['observation'][:64],dtype=torch.float32)
 out=actor.mean(obs).numpy()
 assert out.shape==(64,6)
 assert np.max(np.abs(out-golden['pytorch_mean_action'][:64]))<1e-5
 with pytest.raises(ValueError,match='different spec'):
  other=root/'configs/experiments/2026_10_23-06_57__motion500_group_baseline_resume/spec.json'
  if other.exists():OnnxActor(pkg/'actor.onnx',pkg/'manifest.json',other)


def test_play_wrapper_derives_spec_and_armature_and_stays_read_only():
 import os,subprocess,pytest
 from pathlib import Path
 root=Path(__file__).resolve().parents[1];script=root/'scripts/play_motion31_mujoco.sh'
 cp=root/'runs/2026_10_23-15_16__motion_fixed_height184_9953/formal/model_update_500.pt'
 if not (script.is_file() and cp.is_file()):pytest.skip('play wrapper fixture absent')
 env={**os.environ,'PLAY_DRY_RUN':'1'}
 result=subprocess.run(['bash',str(script),'--checkpoint',str(cp)],capture_output=True,text=True,env=env,timeout=60)
 assert result.returncode==0,result.stderr
 assert 'configs/experiments/2026_10_23-15_16__motion_fixed_height184_9953/spec.json' in result.stderr
 assert '--leg-armature-m 0.35' in result.stderr
 assert 'scripts/validate_native_mujoco.py --run' in result.stderr
 assert '--output-dir' not in result.stderr
 # Browser panel is the default control surface: window, LAN panel and session caps are added.
 assert '--viewer' in result.stderr and '--panel' in result.stderr
 if '--panel-host 192.168.250.1' not in result.stderr:
  assert '--panel-host 0.0.0.0' in result.stderr
  assert 'panel host 192.168.250.1 is not local; binding 0.0.0.0 instead' in result.stderr
 assert '--panel-port' in result.stderr
 assert 'play: web panel http://' in result.stderr  # the effective URL is printed, port may advance
 assert '--max-speed .5' in result.stderr and '--max-yaw .5' in result.stderr
 keyboard=subprocess.run(['bash',str(script),'--checkpoint',str(cp),'--keyboard'],
  capture_output=True,text=True,env=env,timeout=60)
 assert keyboard.returncode==0 and '--keyboard' in keyboard.stderr and '--panel' not in keyboard.stderr
 missing=subprocess.run(['bash',str(script),'--viewer'],capture_output=True,text=True,env=env,timeout=60)
 assert missing.returncode==2 and '--checkpoint or --models is required' in missing.stderr


def test_play_wrapper_keeps_web_control_when_the_panel_port_is_taken():
 # Web control is the default; an earlier preview holding the port must not leave the new
 # playback with a window but no panel. The effective URL is printed before playback.
 import os,re,socket,subprocess,pytest
 from pathlib import Path
 root=Path(__file__).resolve().parents[1];script=root/'scripts/play_motion31_mujoco.sh'
 cp=root/'runs/2026_10_23-23_11__motion_course_keys184_10000/formal/model_update_1500.pt'
 if not (script.is_file() and cp.is_file()):pytest.skip('play wrapper fixture absent')
 env={**os.environ,'PLAY_DRY_RUN':'1','PLAY_PANEL_HOST':'127.0.0.1'}
 probe=socket.socket();probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
 try:probe.bind(('127.0.0.1',8766));probe.listen(1)
 except OSError:pytest.skip('port 8766 unavailable for the busy-port fixture')
 try:
  busy=subprocess.run(['bash',str(script),'--checkpoint',str(cp)],capture_output=True,text=True,env=env,timeout=60)
 finally:probe.close()
 assert busy.returncode==0,busy.stderr
 ports=[int(v) for v in re.findall(r'--panel-port (\d+)',busy.stderr)]
 assert ports and all(port!=8766 for port in ports)
 assert 'http://127.0.0.1:8766/' not in busy.stderr and 'play: web panel http://127.0.0.1:' in busy.stderr
 off=subprocess.run(['bash',str(script),'--checkpoint',str(cp)],capture_output=True,text=True,
  env={**env,'PLAY_PANEL':'0'},timeout=60)
 assert off.returncode==0 and '--panel' not in off.stderr


def test_saved_swing_geometry_matches_native_root_convention():
 # Signed angle, wrap and projection are independent of wheel joint encoder sign.
 vectors=np.array([[0.,-1.,-.06],[0.,-1.,0.]])
 theta=np.arctan2(-vectors[:,2],-vectors[:,1])
 difference=abs(np.arctan2(np.sin(theta[0]-theta[1]),np.cos(theta[0]-theta[1])))
 assert difference>.05 and np.isclose(theta[0],np.arctan(.06))
