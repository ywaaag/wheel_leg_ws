import torch
import pytest
from native_stand.students.physics_readback import read_view,validate

def test_backend_readback_does_not_substitute_missing_config_values():
 class View:
  def get_dof_limits(self):return torch.tensor([[[-1.,2.]]])
 r=read_view(View(),['get_dof_limits','get_rest_offsets'])
 assert r['get_dof_limits']=={'status':'READ','values':[[[-1.,2.]]]}
 assert r['get_rest_offsets']['status']=='UNAVAILABLE'
 assert 'values' not in r['get_rest_offsets']

def test_readback_request_rejects_unknown_fields_before_any_runtime():
 with pytest.raises(ValueError,match='unknown/missing'):validate({'version':'recovery_physics_readback_v1','unknown':True})
