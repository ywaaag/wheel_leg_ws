FROZEN_TEST_SCOPE='PORTABLE'
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch


def adapter():
 p=Path(__file__).resolve().parents[1]/'deployment_protocol/motion31_reference/reference_adapter.py'
 s=importlib.util.spec_from_file_location('motion31_adapter',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_numpy_adapter_equals_training_encoder_and_target_cache():
 from native_stand.students.mcu_interface import encode
 m=adapter();q=np.array([.1,.2,0,-.1,.2,0],np.float32);dq=np.arange(6,dtype=np.float32);pl=np.array([.2,.3,-.2,.3],np.float32);pw=np.array([4.,-5.],np.float32)
 args=[np.array([.1,.2,.3],np.float32),np.array([0,-1,0],np.float32),q,dq,pl,pw,np.array([1.,.5,.34],np.float32)]
 got=m.encode(*args)
 tensor=lambda x:torch.tensor(x)[None]
 expected=encode(*[tensor(x) for x in args[:6]],torch.zeros(1,6),'Motion',tensor(args[6]))[0].numpy()
 assert np.allclose(got,expected,atol=1e-7) and len(got)==31
 d=m.decode(np.ones(6),np.array([[-.3,.4]]*4));assert np.allclose(d['leg_target_rad'],.4) and np.allclose(d['wheel_target_rad_s'],100)
 new=m.encode(*args[:4],d['previous_legs_next'],d['previous_wheels_next'],args[6])
 assert np.allclose(new[16:20],.4/1.5) and np.allclose(new[20:22],1)
 request,applied=m.pd(np.zeros(6),np.zeros(6),d['leg_target_rad'],d['wheel_target_rad_s'])
 assert np.allclose(request,[24,24,100,24,24,100]) and np.allclose(applied,[24,24,5.5,24,24,5.5])


def test_adapter_rejects_bad_sensor_or_unknown_mechanical_limits():
 m=adapter()
 with pytest.raises(ValueError):m.decode(np.zeros(6),None)
 with pytest.raises(ValueError):m.encode(np.zeros(3),np.zeros(3),np.zeros(6),np.zeros(6),np.zeros(4),np.zeros(2),[0,0,.34])
