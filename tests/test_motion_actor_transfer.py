FROZEN_TEST_SCOPE='PORTABLE'
import torch,pytest
from native_stand.students.model import Student,load
from native_stand.students.motion_transfer import transfer_actor,compare
from native_stand.students.mcu_interface import task,actor_input

def test_motion_to_recovery_transfer_keeps_function_std_rng_and_independent_storage():
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(11);source=Student(version='multiskill_mcu_v2')
  x=actor_input(torch.randn(128,22),task('Motion',torch.tensor([[0.,0.,.34]]).repeat(128,1)))
  before={k:v.clone() for k,v in source.state_dict().items()};rng=torch.get_rng_state().clone()
  new=transfer_actor(source,'Motion','Recovery')
  assert torch.equal(rng,torch.get_rng_state())
  y=x.clone();y[:,25]=0;y[:,26]=1
  with torch.no_grad():torch.testing.assert_close(source(x),new(y),atol=1e-6,rtol=0)
  assert all(torch.equal(v,before[k]) for k,v in source.state_dict().items())
  assert torch.equal(new.log_std,source.log_std)
  assert new.mlp[0].weight.data_ptr()!=source.mlp[0].weight.data_ptr()
  for col in range(31):
   if col!=26:assert torch.equal(new.mlp[0].weight[:,col],source.mlp[0].weight[:,col])
  with pytest.raises(ValueError):transfer_actor(source,'Motion','Motion')

def test_transfer_preserves_output_and_source_and_std(tmp_path):
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(7);source=Student(version='multiskill_mcu_v2');before={k:v.clone() for k,v in source.state_dict().items()}
  command=torch.randn(128,3);command[:,2]=.34
  x=actor_input(torch.randn(128,22),task('Recovery',command));new=transfer_actor(source);proof=compare(source,new,x)
 assert proof['mapped_action_max']<1e-6 and proof['std_equal']
 assert all(torch.equal(v,before[k]) for k,v in source.state_dict().items())
 assert all(torch.equal(v,before[k]) for k,v in new.state_dict().items() if k!='mlp.0.weight')
 p=tmp_path/'actor.pt';torch.save({'role':'MotionZeroStudent','interface_version':'multiskill_mcu_v2','actor':new.state_dict()},p)
 reloaded,_=load(p,'MotionZeroStudent');assert compare(source,reloaded,x)['mapped_action_max']<1e-6
 with pytest.raises(ValueError,match='role'):load(p,'RecoveryStudent')

def test_transfer_rejects_wrong_interface_or_mode():
 with pytest.raises(ValueError,match='MCU'):transfer_actor(Student())
 source=Student(version='multiskill_mcu_v2');new=transfer_actor(source)
 with pytest.raises(ValueError,match='Recovery'):compare(source,new,torch.zeros(3,31))
 with pytest.raises(ValueError,match='finite'):compare(source,new,torch.ones(2,25))
