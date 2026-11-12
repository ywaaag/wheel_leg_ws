"""Real C reference versus production tensor decoder and input contracts."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from native_stand.simulation.control_core import ControlCore,build,verify_source
from native_stand.tasks.motion.action_contract import ActionDecoder
from native_stand.tasks.motion.encoding import encode_frame,encode_task,encode_actor_input

FROZEN_TEST_SCOPE='PORTABLE'
FIXTURES=Path(__file__).parent/'fixtures'


@pytest.fixture(scope='module')
def binding(tmp_path_factory):return build(tmp_path_factory.mktemp('core')/'control.so')


@pytest.fixture
def core(binding):
    return ControlCore(binding,json.loads((FIXTURES/'motion_C0_control_contract.json').read_text()))


def test_actual_C_matches_production_decoder_all_heights_and_raw_clipping(core):
    rng=np.random.default_rng(198)
    raw=rng.uniform(-2,2,(41,6)).astype(np.float32)
    heights=np.linspace(.30,.40,41,dtype=np.float32)
    c=np.stack([core.decode(r,float(h)) for r,h in zip(raw,heights,strict=True)])
    expected=ActionDecoder.from_contract(core.contract).decode(torch.from_numpy(raw),torch.from_numpy(heights))['requested_target'].numpy()
    np.testing.assert_allclose(c,expected,atol=3e-7,rtol=0)
    assert np.count_nonzero(np.abs(c[:,[2,5]])==80)>0


def test_actual_C_observation_previous_and_shadow_independence(core):
    raw=np.array([0,0,1,0,0,-1],np.float32)
    request=core.decode(raw,.34)
    np.testing.assert_array_equal(core.adopted,np.zeros(6))
    core.adopt(request,[-1.5,-1.5,-80,-1.5,-1.5,-80],[1.5,1.5,80,1.5,1.5,80])
    gyro=np.array([1,2,3],np.float32);gravity=np.array([0,-1,0],np.float32)
    q=np.array([.1,.2,-.1,.2],np.float32);dq=np.arange(6,dtype=np.float32)
    command=np.array([1,2,.34],np.float32)
    encoded=core.encode(gyro,gravity,q,dq,command)
    fullq=np.zeros(6,np.float32);fullq[[0,1,3,4]]=q
    frame=encode_frame(*(torch.from_numpy(x[None]) for x in (gyro,gravity,fullq,dq,core.adopted[[0,1,3,4]],core.adopted[[2,5]],np.zeros(6,np.float32))))
    expected=encode_actor_input(frame,encode_task(torch.from_numpy(command[None])))[0].numpy()
    np.testing.assert_allclose(encoded,expected,atol=1e-7,rtol=0)
    np.testing.assert_allclose(encoded[20:22],[.8,-.8],atol=1e-7)


def test_actual_C_PD_intermediate_driver_caps_and_original_compensation(core):
    q=np.array([.2,-.3,-.2,-.3],np.float32);dq=np.array([1,2,10,-1,2,-10],np.float32)
    target=np.array([-.8,-.5,80,.8,-.5,-80],np.float32)
    raw,limited=core.pd(q,dq,target=target)
    independent=np.array([40*(target[i]-q[i if i<2 else i-1])-2*dq[i] if i not in (2,5) else .2*(target[i]-dq[i]) for i in range(6)],np.float32)
    np.testing.assert_allclose(raw,independent,atol=2e-6)
    np.testing.assert_allclose(limited,np.clip(independent,[-40,-40,-5.5,-40,-40,-5.5],[40,40,5.5,40,40,5.5]),atol=1e-7)
    assert np.abs(limited[[2,5]]).tolist()==[5.5,5.5]
    actual_driver=np.clip(limited,[-40,-40,-2.44694786377709,-40,-40,-2.44694786377709],[40,40,2.44694786377709,40,40,2.44694786377709])
    assert np.abs(actual_driver[2])==pytest.approx(2.44694786377709)
    # Software compensation is one explicit branch, never a fabricated V4 NET.
    out=core.compensate([0,0,0,0],[.28,.28],[.4,.4])
    spring=np.clip((-83.121*.28+40.242)*.8,16.5,27)*np.sin(.4)
    np.testing.assert_allclose(out,[-spring,spring,spring,-spring],atol=2e-6)


def test_piecewise_and_continuous_center_remain_distinct(core,binding):
    continuous=ControlCore(binding,{'action_version':'motion_multiheight_support_center_poly5_015_80_v1'})
    assert not np.allclose(core.decode(np.zeros(6),.35),continuous.decode(np.zeros(6),.35),atol=1e-5)
    with pytest.raises(ValueError):core.decode(np.zeros(6),.41)
    assert verify_source()['physical_NET_in_core'] is False


def test_C_PD_and_device_V4_match_same_reset_and_sensor_timeline(core):
    from native_stand.simulation.v4_response import Response
    parameters=torch.tensor([[1.,.1,.2,4.,6.,1.,-1.,2.,3.,.4,0.,0.,0.]]).expand(2,-1).clone()
    profile={'transport_ms':8,'q_age_ms':2,'v_age_ms':4,'q_reference':[0.,0.],'velocity_tau_ms':3.}
    def c_pd(q,dq,target):
        rows=[]
        for i in range(len(q)):
            physical_q=np.zeros(4,np.float32);physical_q[:2]=q[i].numpy()
            physical_dq=np.zeros(6,np.float32);physical_dq[:2]=dq[i].numpy()
            physical_target=np.zeros(6,np.float32);physical_target[:2]=target[i].numpy()
            raw,_=core.pd(physical_q,physical_dq,target=physical_target)
            rows.append(raw[:2])
        return torch.tensor(np.stack(rows))
    device=Response(parameters.clone(),[profile]*2)
    c=Response(parameters.clone(),[profile]*2,pd=c_pd)
    q=torch.zeros(2,2);dq=q.clone();target=q.clone()
    device.reset(q,dq,target);c.reset(q,dq,target)
    for tick in range(260):
        q=torch.full((2,2),np.sin(tick*.02)*.1)
        dq=torch.full((2,2),np.cos(tick*.02)*2)
        target=torch.full((2,2),.1 if tick>=65 else 0.)
        m,n=device.step(q,dq,target);cm,cn=c.step(q,dq,target)
        torch.testing.assert_close(m,cm,atol=3e-6,rtol=0)
        torch.testing.assert_close(n,cn,atol=0,rtol=0)
        if tick==129:
            ids=torch.tensor([0]);device.reset(q[ids],dq[ids],target[ids],ids=ids);c.reset(q[ids],dq[ids],target[ids],ids=ids)


def test_recovery_physical_mean_matches_actual_shared_C(binding):
    from native_stand.learning.ppo.recovery_cubic import PhysicalActor
    contract={'action_version':'recovery_cubic_physical_Gaussian_v2',
        'center_rad':[-.130974,-.151177,.130974,-.151177],
        'leg_lower_rad':[-1.5]*4,'leg_upper_rad':[1.5]*4}
    actor=PhysicalActor(contract,hidden_dims=(8,))
    core=ControlCore(binding,contract)
    observation=torch.randn(32,31)
    with torch.no_grad():
        mean=actor.mean(observation).numpy()
        physical=actor.physical_mean(observation).numpy()
    actual=np.stack([core.decode(a,.34) for a in mean])
    np.testing.assert_allclose(actual,physical,atol=3e-6,rtol=0)
