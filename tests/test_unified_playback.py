"""Unified loop and real C/V4 CPU execution, zero MuJoCo/Isaac integration."""
import builtins
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from native_stand.artifacts.io import file_sha256
from native_stand.cli import main
from native_stand.simulation import playback
from native_stand.simulation.control_core import build,ControlCore
from native_stand.simulation.playback_execution import PlaybackExecution
from native_stand.tasks.motion.delayed_execution import configuration
from test_motion_production_integration import make_job

FROZEN_TEST_SCOPE='PORTABLE'


def test_play_dry_run_no_model_deserialize_build_engine_or_ports(tmp_path,monkeypatch,capsys):
    from dataclasses import replace
    job,contract=make_job(tmp_path,retention=False)
    usd=tmp_path/'asset.usd';usd.write_bytes(b'CPU-boundary-USD-fixture')
    template=json.loads(Path(job.assets['runtime_template']).read_text())
    template.update(asset_sha256=file_sha256(usd),asset_path=str(usd))
    Path(job.assets['runtime_template']).write_text(json.dumps(template))
    assets=dict(job.assets,runtime_template_sha256=file_sha256(job.assets['runtime_template']),delayed_execution=configuration())
    job=replace(job,assets=assets)
    refs={}
    for name in ('description','initial_states','V4'):
        p=tmp_path/(name+'.json');p.write_text('{}')
        refs[name]=dict(path=str(p),sha256=file_sha256(p))
    manifest=tmp_path/'assets.json';manifest.write_text(json.dumps(dict(asset_sha256=file_sha256(usd),bindings=refs)))
    monkeypatch.setattr(playback,'ASSET_BINDINGS',manifest)
    run=tmp_path/'run';run.mkdir();(run/'run.json').write_text(json.dumps({'config':job.as_run_record()}))
    checkpoint=run/'explicit.pt';checkpoint.write_bytes(b'not-deserialized-in-dry-run')
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith(('mujoco','isaaclab.app')):pytest.fail('dry-run imported physical engine')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    monkeypatch.setattr('subprocess.run',lambda *a,**k:pytest.fail('dry-run compiled or spawned'))
    monkeypatch.setattr('socket.socket',lambda *a,**k:pytest.fail('dry-run probed a port'))
    assert main(['play','--run',str(run),'--checkpoint','explicit.pt','--headless','--steps','12','--dry-run','--json'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['checkpoint']==str(checkpoint)
    assert result['physics_steps']==0 and result['timing']['wheel_transport_nominal_ms']==[7,11]
    assert set(p.name for p in run.iterdir())=={'run.json','explicit.pt'}


def test_headless_and_GUI_share_same_steps_and_pause_keeps_tick():
    class Plant:
        physics_dt=0.
        def __init__(self):self.execution=NS(tick=0);self.events=[]
        def step(self,c):self.events.append(('step',self.execution.tick,c.teacher));self.execution.tick+=1
        def reset(self):self.events.append(('reset',self.execution.tick));self.execution.tick=0
    first=Plant();playback.run_loop(first,playback.Controls(),steps=3)
    second=Plant();controls=playback.Controls(paused=True)
    class Viewer:
        def is_running(self):return True
        def sync(self):
            if controls.paused:assert second.execution.tick==0
            controls.paused=False
    playback.run_loop(second,controls,steps=3,viewer=Viewer())
    assert first.events==second.events
    third=Plant();controls=playback.Controls(teacher=True)
    playback.run_loop(third,controls,steps=2)
    assert third.events==[('step',0,True),('step',1,True)]


def test_actual_C_V4_target_transport_and_wheel_arrivals(tmp_path):
    job,contract=make_job(tmp_path,retention=False)
    core=ControlCore(build(tmp_path/'core.so'),contract)
    names=[f'p{i}' for i in range(13)]
    values=[1.,.1,.2,4.,6.,1.,-1.,2.,3.,.4,0.,0.,0.]
    row=dict(parameters=dict(zip(names,values,strict=True)),effective_observer={
        'transport_ms':8,'q_age_ms':2,'v_age_ms':4,'q_reference':[0.,0.],'velocity_tau_ms':3.})
    V4=dict(parameter_names=names,sides={'L':row,'R':row})
    e=PlaybackExecution(core,V4,configuration(),job.control,[-1.5,-1.5,-80,-1.5,-1.5,-80],[1.5,1.5,80,1.5,1.5,80])
    q=np.zeros(6);dq=q.copy();e.reset(q,dq)
    e.submit([0,0,80,0,0,80]);arrivals={}
    for tick in range(30):
        if tick%10==0:
            observed=e.observation([0,0,0],[0,-1,0],[0,0,.34])
            if tick==10:np.testing.assert_allclose(observed[20:22],[.8,.8],atol=1e-7)
        motor,NET=e.step(q,dq)
        assert not NET[[2,5]].any()
        if tick<5:assert not core.adopted.any()
        for channel in (2,5):
            if motor[channel] and channel not in arrivals:arrivals[channel]=tick
    assert arrivals=={2:12,5:16}
    assert e.tick==30 and abs(e.motor[2])==pytest.approx(2.44694786377709)
