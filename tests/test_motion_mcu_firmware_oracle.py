"""Compare new vectorized GPU algebra against actual firmware C, CPU only."""
FROZEN_TEST_SCOPE='PROJECT_DATA'
from pathlib import Path
import importlib.util
import numpy as np
import torch


def test_actual_c_oracle_for_compensation_and_both_limit_layers(tmp_path):
    from native_stand.students.firmware_control import FirmwareControl
    from native_stand.students.motion_mcu_execution import spring_compensation,pd_and_motor
    root=Path(__file__).resolve().parents[1]
    loader=importlib.util.spec_from_file_location('validator',root/'scripts/validate_motion_firmware.py')
    m=importlib.util.module_from_spec(loader);loader.loader.exec_module(m)
    out=tmp_path/'cpu';m.build(Path('/home/kellen/combat-dm-balance-main'),out)
    rng=np.random.default_rng(73)
    for _ in range(64):
        q=np.array([.21648856,.21663799,0,-.21648856,.21663799,0],np.float32)
        q+=rng.normal(0,.08,6).astype(np.float32);dq=rng.normal(0,1,6).astype(np.float32)
        native=np.r_[np.zeros(3),[0,-1,0],q[[0,1,3,4]],dq].astype(np.float32)
        target=q+rng.normal(0,.5,6).astype(np.float32);target[[2,5]]*=50
        target=np.clip(target,[-1.5,-1.5,-100,-1.5,-1.5,-100],[1.5,1.5,100,1.5,1.5,100]).astype(np.float32)
        tq=torch.tensor(q[None]);tdq=torch.tensor(dq[None]);tt=torch.tensor(target[None])
        request,limited,comp,motor,valid=pd_and_motor(tq,tdq,tt)
        assert valid.all()
        with FirmwareControl(out/'motion_firmware.so') as c:
            lengths,spread=c.geometry(q[[0,1,3,4]])
            tc,tl,ts,_=spring_compensation(tq[:,[0,1,3,4]])
            np.testing.assert_allclose(tl.numpy()[0],lengths,atol=2e-6)
            np.testing.assert_allclose(ts.numpy()[0],spread,atol=1e-6)
            c.observe(0,-10,native,[0,0,.34]);c.publish(0,target/np.array([1.5,1.5,100,1.5,1.5,100],np.float32))
            result=c.pd(0,-10,native,[0,0,.34],lengths,spread)
            for field,x in [('pd_request_nm',request),('pd_limited_nm',limited),('spring_compensation_model_nm',comp),('motor_model_nm',motor)]:
                np.testing.assert_allclose(x.numpy()[0],result[field],atol=2e-5,rtol=2e-6)
