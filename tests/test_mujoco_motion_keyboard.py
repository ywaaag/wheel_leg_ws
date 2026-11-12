FROZEN_TEST_SCOPE='PORTABLE'
import pytest
from native_stand.students.mujoco_keyboard import MotionKeyboard

def test_keyboard_commands_clamp_zero_reset_and_no_repeat_reset():
 k=MotionKeyboard()
 for _ in range(30):k.key(ord('W'));k.key(ord('A'))
 assert k.read()==(2.,.5,False)
 k.key(32);assert k.read()==(0.,0.,False)
 k.key(ord('S'));k.key(ord('D'));assert k.read()==(-.25,-.25,False)
 k.key(ord('R'));assert k.read()==(0.,0.,True) and k.read()==(0.,0.,False)
 assert MotionKeyboard(max_speed=6.,max_yaw=8.).max_speed==6.
 with pytest.raises(ValueError):MotionKeyboard(max_speed=float('inf'))

@pytest.mark.parametrize('panel',[False,True])
def test_exact_u3250_cpu_plan_accepts85_and_keyboard_not_scoring(panel):
 import sys,json
 from pathlib import Path
 from types import SimpleNamespace
 root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
 from validate_motion31_mujoco import plan
 spec=root/'configs/experiments/2026_10_18-23_10__motion1000_differential85_5000_retry/spec.json';cp=root/'runs/2026_10_18-23_10__motion1000_differential85_5000_retry/formal/model_update_3250.pt';xml=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml')
 if not all(p.exists() for p in (spec,cp,xml)):pytest.skip('local model fixture absent')
 args=SimpleNamespace(spec=spec,checkpoint=cp,xml=xml,wall_seconds=600,commands=[0.],yaw_commands=None,keyboard=not panel,panel=panel,panel_port=8766,viewer=True,max_speed=6. if panel else 2.,max_yaw=8. if panel else .5)
 _,_,_,m=plan(args,json.loads(spec.read_text()));assert m['checkpoint_critic_version']=='motion_symmetry_privileged85_v1' and m['critic_not_loaded']
 if panel:assert m['panel']['backend']=='MuJoCo' and m['panel']['url']=='http://127.0.0.1:8766/'
 assert m['mode']==('manual_browser_preview' if panel else 'manual_keyboard_preview') and m['protocol']['scoring'].startswith('NOT_RUN') and len(m['initial_actor_input'])==31


def test_browser_and_viewer_share_one_command_and_reset():
 from native_stand.students.motion_panel import Plan
 from native_stand.students.mujoco_keyboard import MotionPanelControls
 p=Plan(2.,.5,backend='MuJoCo');c=MotionPanelControls(p,True)
 p.apply({'vx':1.,'yaw':-.5});c.key(ord('W'))
 assert c.read()==(1.25,-.5,False)
 p.apply({'max_speed':.5});c.key(ord('W'))
 assert c.read()==(.5,-.5,False)
 p.apply({'reset':True});assert c.read()==(0.,0.,True)
 assert c.read()==(0.,0.,False)
 p.apply({'vx':.25});disabled=MotionPanelControls(p)
 disabled.key(ord('W'));assert disabled.read()==(.25,0.,False)


def test_panel_edits_height_and_31d_skill_jump_inputs():
 import sys
 from pathlib import Path
 from native_stand.students.motion_panel import HTML_PATH, Plan
 from native_stand.students.mujoco_keyboard import MotionPanelControls
 p=Plan(2.,.5,backend='MuJoCo');c=MotionPanelControls(p)
 assert c.read_full()=={'vx':0.,'yaw':0.,'reset':False,'height':.34,'skill':'Motion','jump':'NONE','model':'main'}
 p.apply({'height':.40,'skill':'Jump','jump':'SMALL'})
 row=c.read_full();assert row['height']==.40 and row['skill']=='Jump' and row['jump']=='SMALL'
 snap=p.snapshot();assert snap['skills']==['Motion','Recovery','Jump'] and snap['jumps']==['NONE','SMALL','BIG']
 for payload in ({'height':.05},{'height':float('nan')},{'skill':'Stand'},{'jump':'HUGE'}):
  with pytest.raises(ValueError):p.apply(payload)
 with pytest.raises(ValueError,match='fixed zero'):Plan(1.,0.,mode='Recovery').apply({'height':.30})
 page=HTML_PATH.read_text()
 assert all(tag in page for tag in ('height-target','skill-select','jump-select','m-tail'))
 root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
 from validate_motion31_mujoco import manual_command
 assert manual_command(None)==(0.,0.,False,.34,'Motion','NONE','main')
 assert manual_command(None,'big')==(0.,0.,False,.34,'Motion','NONE','big')
 class Plain:
  def read(self):return .5,.25,False
 assert manual_command(Plain())==(.5,.25,False,.34,'Motion','NONE','main')


def test_panel_reports_an_occupied_port_instead_of_a_raw_oserror():
 import pytest
 from native_stand.students.motion_panel import Plan,start,stop
 first,_=start(Plan(1.,.5,backend='MuJoCo'),0,'127.0.0.1')
 try:
  port=first.server_address[1]
  with pytest.raises(RuntimeError,match='not available'):
   start(Plan(1.,.5,backend='MuJoCo'),port,'127.0.0.1')
 finally:stop(first)


def test_panel_telemetry_serializes_numpy_scalars_and_arrays():
 """Replay producers publish numpy values; /telemetry must stay valid JSON."""
 import json,urllib.request
 import numpy as np
 from native_stand.students.motion_panel import Plan,start,stop
 plan=Plan(1.,.5,backend='MuJoCo')
 plan.publish({'steps':np.int64(7),'vx':np.float32(.2),'flag':np.bool_(True),
  'actor_input':np.arange(3,dtype=np.float32),'nested':{'resets':np.int32(2)}})
 server,_=start(plan,0,'127.0.0.1')
 try:
  port=server.server_address[1]
  with urllib.request.urlopen(f'http://127.0.0.1:{port}/telemetry',timeout=5) as response:
   payload=json.loads(response.read())
 finally:stop(server)
 telemetry=payload['telemetry']
 assert telemetry['steps']==7 and telemetry['flag'] is True and telemetry['nested']['resets']==2
 assert abs(telemetry['vx']-.2)<1e-6 and telemetry['actor_input']==[0.,1.,2.]


def test_panel_switches_named_preview_models_in_one_plant():
 from native_stand.students.motion_panel import Plan
 from native_stand.students.mujoco_keyboard import MotionPanelControls
 p=Plan(2.,.5,backend='MuJoCo',models=('big','small'),model='small');c=MotionPanelControls(p)
 assert c.read_full()['model']=='small'
 assert p.snapshot()['models']==['big','small'] and p.snapshot()['model']=='small'
 p.apply({'model':'big'});assert c.read_full()['model']=='big'
 with pytest.raises(ValueError):p.apply({'model':'other'})
 with pytest.raises(ValueError):Plan(2.,.5,models=('big','big'))
 with pytest.raises(ValueError):Plan(2.,.5,models=('big',),model='small')


def test_mujoco_browser_http_command_telemetry_and_shutdown():
 import json
 from urllib.request import urlopen,Request
 from native_stand.students.motion_panel import Plan,start,stop
 from native_stand.students.mujoco_keyboard import MotionPanelControls
 p=Plan(2.,.5,backend='MuJoCo');c=MotionPanelControls(p)
 server,thread=start(p,0);url=f'http://127.0.0.1:{server.server_port}'
 try:
  with urlopen(url) as r:assert b'MuJoCo' in r.read()
  with urlopen(Request(url+'/command',data=b'{"max_speed":6,"max_yaw_speed":8,"vx":5,"yaw":-7}',headers={'Content-Type':'application/json'})) as r:assert r.status==200
  assert c.read()==(5.,-7.,False)
  c.publish({'vx':.9,'yaw':.45,'pitch':.02,'roll':.01,'wheel_qdot_l':2.,'wheel_tau_r':1.})
  with urlopen(url+'/telemetry') as r:row=json.load(r)
  assert row['backend']=='MuJoCo' and row['telemetry']['yaw']==.45
 finally:stop(server);thread.join(timeout=2)
 assert not thread.is_alive()


def test_mujoco_caps_can_increase_and_shrink_while_isaac_remains_bounded():
 from native_stand.students.motion_panel import Plan
 from native_stand.students.mujoco_keyboard import MotionPanelControls
 p=Plan(2.,.5,backend='MuJoCo');c=MotionPanelControls(p,True)
 assert p.snapshot()['editable_limits']
 p.apply({'max_speed':6.,'max_yaw_speed':8.,'vx':5.,'yaw':-7.})
 assert c.read()==(5.,-7.,False)
 c.key(ord('W'));assert c.read()==(5.25,-7.,False)
 p.apply({'max_speed':1.,'max_yaw_speed':.25})
 assert c.read()==(1.,-.25,False)
 for payload in ({'max_speed':0.},{'max_speed':-1.},{'max_speed':float('inf')},{'max_yaw_speed':float('nan')}):
  with pytest.raises(ValueError):p.apply(payload)
 p.apply({'max_yaw_speed':0.});assert c.read()==(1.,0.,False)
 legacy=Plan(2.,.5)
 assert not legacy.snapshot()['editable_limits']
 with pytest.raises(ValueError):legacy.apply({'max_speed':6.})


def test_distilled_small_actor_from_motion85_uses_student_lineage_not_parent_shape():
 import sys,json
 from pathlib import Path
 from types import SimpleNamespace
 root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
 from validate_motion31_mujoco import plan
 spec=root/'configs/experiments/2026_10_18-23_10__motion1000_differential85_5000_retry/spec.json'
 cp=root/'isolated/2026_10_19-19_47__motion3250_small31_bc/best_validation.pt'
 xml=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml')
 if not all(p.exists() for p in (spec,cp,xml)):pytest.skip('local model fixture absent')
 args=SimpleNamespace(spec=spec,checkpoint=cp,xml=xml,wall_seconds=600,commands=[0.],yaw_commands=None,keyboard=False,panel=False,viewer=False)
 _,_,actor,m=plan(args,json.loads(spec.read_text()))
 assert actor.mlp[0].weight.shape==(128,31) and m['distilled_student'] and m['critic_not_loaded']
