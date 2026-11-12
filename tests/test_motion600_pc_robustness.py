"""Diagnostic boundary checks around the reused real MuJoCo step and recipe."""
FROZEN_TEST_SCOPE='PROJECT_DATA'
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
loader=importlib.util.spec_from_file_location('motion600_pc_sweep',ROOT/'isolated/2026_11_01__motion600_pc_followup/sweep.py')
tool=importlib.util.module_from_spec(loader);loader.loader.exec_module(tool)


def test_pulse_is_one_increment_at_declared_time_without_hidden_physics(monkeypatch):
    calls=[]
    def original(m,d):calls.append(d.qvel.copy())
    monkeypatch.setattr(tool.mujoco,'mj_step',original)
    model=object();data=SimpleNamespace(xmat=np.array([[0,0,-1,-1,0,0,0,1,0]],dtype=float),qvel=np.array([0.,0.,0.,1.,2.,3.]))
    event=tool.StepEvents(model,{'root':0},.001,dict(student_time_s=.001,delta_vx_m_s=.1))
    with event:
        for i in range(5):tool.mujoco.mj_step(model,data)
    assert tool.mujoco.mj_step is original and len(calls)==event.steps==5
    np.testing.assert_allclose([x[0] for x in calls],[0,0,.1,.1,.1])
    np.testing.assert_array_equal(data.qvel[1:],[0,0,1,2,3])
    assert len(event.events)==1 and event.events[0]['student_tick_ms']==1


def test_step_adapter_restored_on_exception(monkeypatch):
    original=lambda m,d:None;monkeypatch.setattr(tool.mujoco,'mj_step',original)
    model=object()
    with pytest.raises(RuntimeError,match='intentional'):
        with tool.StepEvents(model,{'root':0},0,None):raise RuntimeError('intentional')
    assert tool.mujoco.mj_step is original


def test_finite_matrix_preserves_physical_feedback_and_real_adopted_cache():
    geometry=dict(candidates=[dict(bound_degrees=2,queried_root_height_m=h,bias_rad=[.01]*4,
        support_compatible_within_1mm=True) for h in (None,.35)])
    cases=tool.matrix(geometry)
    assert len(cases)==24
    assert sum(12000+round(c['settings']['pre_teacher_s']*1000) for c in cases)==291000
    assert all(not c['settings']['zero_actor_dq'] and c['settings']['controller']=='student' and
        c['settings']['command']=='zero' for c in cases)
    assert all(c['settings']['previous']=='hold' for c in cases if c['settings']['pre_teacher_s'])
    pulses=[c['pulse'] for c in cases if c['pulse']]
    assert len(pulses)==4 and {p['delta_vx_m_s'] for p in pulses}=={-.05,.05,-.1,.1}


def test_real_V4_response_and_Actor_history_use_the_declared_past_samples():
    contract=tool.core.read(tool.core.CONTRACT)
    zero=np.zeros((2,2));leg=tool.core.plant.base.LegResponse(contract,zero,zero,zero)
    leg.qa[:]=20;leg.va[:]=10
    observer=tool.core.LegObserver(np.zeros(4),np.zeros(4))
    for tick in range(31):
        q=np.full((2,2),tick*.0001);v=np.full((2,2),tick*.01)
        leg.step(q,v,zero)
        if tick==30:Actor_q,Actor_v=observer.before(tick,10,10)
        observer.after(tick,np.full(4,tick*.0001),np.full(4,tick*.01))
    qstep=12.566/65535;vstep=90/4095
    np.testing.assert_allclose(leg.heldq,np.round(.001/qstep)*qstep,atol=1e-12) # PD q tick10
    np.testing.assert_allclose(leg.heldv,np.round(.2/vstep)*vstep,atol=1e-12) # PD dq tick20
    np.testing.assert_allclose(Actor_q,np.round(.0015/qstep)*qstep,atol=1e-12) # Actor history tick19, held tick15
    np.testing.assert_allclose(Actor_v,np.round(.15/vstep)*vstep,atol=1e-12)
