FROZEN_TEST_SCOPE='PORTABLE'
import importlib.util
from pathlib import Path
import sys
import pytest

p=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(p))
spec=importlib.util.spec_from_file_location('handover_visual',p/'play_motion_handover.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_geometric_mapping_changes_only_wheels_and_preserves_source():
    c=dict(name='old',target_delay_ms=[10,10,11,10,10,7],torque_tau_ms=[5,5,0,5,5,0],
           pd_feedback_delay_ms=10,search_values={'wheel':'transport11_7'})
    d=m.geometric_wheel_mapping(c)
    assert d['target_delay_ms']==[10,10,7,10,10,11]
    assert d['torque_tau_ms']==c['torque_tau_ms'] and d['pd_feedback_delay_ms']==10
    assert c['target_delay_ms'][2]==11 and c['search_values']['wheel']=='transport11_7'
    assert d['search_values']['wheel']=='transport7_11'
    c['torque_tau_ms'][2]=11
    with pytest.raises(ValueError):m.geometric_wheel_mapping(c)


def test_response_mapping_keeps_transport_zero():
    c=dict(name='rise',target_delay_ms=[0]*6,torque_tau_ms=[0,0,11,0,0,7])
    d=m.geometric_wheel_mapping(c)
    assert d['target_delay_ms']==[0]*6 and d['torque_tau_ms']==[0,0,7,0,0,11]


def test_deployment_log_replay_validates_full_state_and_identity(tmp_path):
    import json
    import numpy as np
    xml=tmp_path/'robot.xml'
    xml.write_text('<mujoco><worldbody><body name="robot"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    clip=tmp_path/'recorded_actions_open_loop.npz'
    q=np.zeros((2,7));q[:,3]=1
    np.savez_compressed(clip,replay_time=[0,.01],replay_qpos=q,replay_qvel=np.zeros((2,6)))
    meta=dict(version='motion31_deployment_log_visual_replay_v1',xml=str(xml),xml_sha256=m.sha(xml),
              clips=['recorded_actions_open_loop'],clip_sha256={'recorded_actions_open_loop':m.sha(clip)})
    (tmp_path/'manifest.json').write_text(json.dumps(meta))
    (tmp_path/'terminal.json').write_text(json.dumps(dict(status='COMPLETED')))
    _,_,names,clips=m.load_replay(tmp_path,'recorded_actions_open_loop')
    assert names==['recorded_actions_open_loop'] and clips[0]['q'].shape==(2,7)
    with pytest.raises(ValueError,match='unknown clip'):m.load_replay(tmp_path,'missing')
    np.savez_compressed(clip,replay_time=[0,0],replay_qpos=q,replay_qvel=np.zeros((2,6)))
    with pytest.raises(ValueError,match='SHA differs'):m.load_replay(tmp_path)
    meta['clip_sha256']['recorded_actions_open_loop']=m.sha(clip)
    (tmp_path/'manifest.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='invalid full-state'):m.load_replay(tmp_path)


def test_motion_wrapper_dispatches_replay_without_model_or_physics(tmp_path):
    import os
    import subprocess
    result=subprocess.run(['bash',str(p/'play_motion31_mujoco.sh'),'--replay-dir',str(tmp_path),
        '--clip','recorded_actions_open_loop','--start-paused','--playback-speed','.25'],
        env={**os.environ,'PLAY_DRY_RUN':'1','PLAY_DISPLAY':':2'},text=True,capture_output=True,check=True)
    assert 'play_motion_handover.py --viewer' in result.stderr
    assert 'validate_native_mujoco.py' not in result.stderr
    assert '--panel --panel-host' in result.stderr


def test_replay_browser_controls_and_http_boundary():
    import json
    from urllib.request import urlopen,Request
    from native_stand.students.replay_panel import ReplayPlan
    from native_stand.students.motion_panel import start,stop
    plan=ReplayPlan(['logged','closed'],'logged',.25,True)
    plan.apply({'model':'closed','playback_speed':.5,'paused':False})
    assert plan.take()==dict(model='closed',playback_speed=.5,paused=False,restart=True)
    assert not plan.take()['restart']
    for bad in ({'vx':1},{'playback_speed':float('nan')},{'paused':1},{'model':'missing'}):
        with pytest.raises(ValueError):plan.apply(bad)
    server,_=start(plan,0)
    try:
        base='http://127.0.0.1:'+str(server.server_port)
        from native_stand.students.motion_panel import HTML_PATH
        assert urlopen(base).read().decode()==HTML_PATH.read_text()
        request=Request(base+'/command',data=json.dumps(dict(paused=True,reset=True)).encode(),headers={'Content-Type':'application/json'})
        result=json.load(urlopen(request))
        assert result['paused'] and plan.take()['restart']
        plan.publish(dict(clip_time_s=.5,wall_time_s=2,realtime_factor=.25))
        assert json.load(urlopen(base+'/telemetry'))['telemetry']['realtime_factor']==.25
    finally:stop(server)


def test_replay_telemetry_preserves_pre_post_alignment_and_model_side_order():
    import numpy as np
    from native_stand.students.replay_panel import frame_telemetry,ReplayTiming
    arrays={k:np.arange(2*n,dtype=float).reshape(2,n) for k,n in
            [('actor_input',31),('mean',6),('state',6),('dq',6),('torque',6),('contact',2),('adopted',6)]}
    clip=dict(name='closed',t=np.array([0,.01,.02]),telemetry=arrays)
    assert 'actor_input' not in frame_telemetry(clip,0)
    first,last=frame_telemetry(clip,1),frame_telemetry(clip,2)
    assert first['actor_input']==arrays['actor_input'][0].tolist()
    assert last['actor_input']==arrays['actor_input'][1].tolist()
    assert first['wheel_tau_l']==2 and first['wheel_tau_r']==5
    assert first['leg_tau_2']==3 and first['contact_r']==1
    assert first['cmd_vx']==22*4 and first['cmd_height']==24*.5
    clock=ReplayTiming(0)
    clock.sample('closed',0,0)
    assert clock.sample('closed',.125,.5)['realtime_factor']==.25
    assert clock.sample('closed',.125,1)['realtime_factor']==0
    assert clock.sample('other',0,1.5)['realtime_factor']==0
    assert clock.sample('other',.1,2)['realtime_factor']==pytest.approx(.2)
