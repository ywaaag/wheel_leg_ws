import copy
from pathlib import Path
import pytest
import torch
from native_stand.students.motion_symmetry import network,expand_source,CRITIC,update_mix
from native_stand.students.model import Student
ROOT=Path(__file__).resolve().parents[1]

def test85_migration_preserves_actor_value_and_old_adam():
 p=ROOT/'runs/2026_10_17-18_18__motion810_spin61_formal_recovery/formal/model_update_750.pt'
 if not p.exists():pytest.skip('local750 required')
 old=torch.load(p,map_location='cpu',weights_only=False);new=expand_source(old)
 assert all(torch.equal(v,new['actor'][k]) for k,v in old['actor'].items())
 from native_stand.students.motion_spin_privileged import network as n61
 a=Student(False,version='multiskill_mcu_v2',critic_privileged_version='motion_spin_privileged61_v1',network_architecture=n61());a.load_state_dict(old['critic'])
 b=Student(False,version='multiskill_mcu_v2',critic_privileged_version=CRITIC,network_architecture=network());b.load_state_dict(new['critic'])
 torch.manual_seed(2);x=torch.randn(17,61)
 assert torch.allclose(a(x),b(torch.cat((x,torch.randn(17,24)),1)),atol=1e-5)
 assert torch.count_nonzero(b.mlp[0].weight[:,61:])==0
 ids=[p for g in new['optimizer']['param_groups'] for p in g['params']];pid=ids[len(old['actor'])]
 for k in ('exp_avg','exp_avg_sq'):
  assert torch.equal(old['optimizer']['state'][pid][k],new['optimizer']['state'][pid][k][:,:61])
  assert not new['optimizer']['state'][pid][k][:,61:].count_nonzero()

def test_evidence_mix_requires_two_and_regresses():
 state=[{'target':[0,0],'mix':0.,'confirmations':0,'regressions':0}]
 e={'accepted':True,'symmetry_metrics':{'swing_hold_peak_rad':.03,'horizontal_hold_mean_mps':.02,'common_abs_hold_mean_mps':.01}}
 records=[{'case':{'command':[0,0,.34]},'per_env':[copy.deepcopy(e) for _ in range(8)]}]
 one=update_mix(state,records);assert one[0]['mix']==0
 two=update_mix(one,records);assert two[0]['mix']==.25
 for e in records[0]['per_env']:e['accepted']=False
 two=update_mix(two,records);two=update_mix(two,records);assert two[0]['mix']==0
 assert update_mix(state,[{'case':{'command':[0,0,.34]},'per_env':[]}])==state

def test_actual85_recipe_assembly_cpu():
 from native_stand.students.motion_symmetry_experiment import build,verify
 parent=ROOT/'configs/experiments/2026_10_17-18_18__motion810_spin61_formal_recovery/spec.json';cp=ROOT/'runs/2026_10_17-18_18__motion810_spin61_formal_recovery/formal/model_update_750.pt'
 if not cp.exists():pytest.skip('local source')
 s=build(parent,cp,experiment='test',spec_path='/tmp/test');assert verify(s)
 from native_stand.students.requests import validate
 assert validate(s)==[]
 from native_stand.students.ppo import assemble
 from native_stand.recovery.controlled_ppo import KEY
 from tensordict import TensorDict
 from native_stand.students.motion_source_retention import KEY as RKEY
 obs=TensorDict({'student_actor':torch.zeros(8,31),'student_critic':torch.zeros(8,85),KEY:torch.ones(8,1,dtype=torch.bool),RKEY:torch.zeros(8,1,dtype=torch.long)},batch_size=[8])
 ppo=assemble(cp,obs,recipe=s['student_training']);assert ppo.critic.mlp[0].in_features==85

def test_live_density_replaces_without_double_counting(monkeypatch):
 from types import SimpleNamespace
 from native_stand.students import motion_symmetry as m
 from native_stand.students.motion_tracking_reward import reward,configuration,SPIN_VERSION,reward_fields
 z=torch.zeros(2);yes=torch.ones(2,dtype=torch.bool)
 measurement={'tangent_velocity_w':torch.zeros(2,2,2),'rolling_speed_forward_mps':torch.full((2,2),.1),'geometry_valid':torch.ones(2,2,dtype=torch.bool)}
 kw=dict(command=torch.tensor([[0.,.5,.34]]*2),target=torch.tensor([[0.,.5]]*2),velocity=torch.zeros(2,3),yaw=z+.5,height=z+.34,angles=torch.zeros(2,2),contacts=torch.full((2,2),20.),leg_q=torch.zeros(2,4),wheel_offset=z,leg_lengths=torch.full((2,2),.3),applied_torque=torch.zeros(2,6),previous_targets=torch.zeros(2,6),executed_targets=torch.zeros(2,6),controlled=yes,valid=yes,physical_failure=~yes,fault=~yes)
 base=reward(**kw,config=configuration(SPIN_VERSION),spin_measurement=measurement)
 fields={'leg_swing_signed_difference_rad':torch.full((2,1),.065),'rolling_common_differential_mps':torch.tensor([[.1,.1]]*2),'root_heading_horizontal_mps':torch.tensor([[.1,0.]]*2)}
 monkeypatch.setattr(m,'runtime_fields',lambda env,measurement:fields)
 env=SimpleNamespace(motion_spin_measurement_post=measurement,symmetry_mix=torch.tensor([0.,.75]))
 out=m.apply(base,env)
 assert len(out['fields'])==len(reward_fields(configuration(m.VERSION)))
 # No terminal events in this fixture: total exactly density*dt.
 torch.testing.assert_close(out['total'],out['density'].sum(-1)*.01)
 assert torch.isfinite(out['total']).all() and (out['density'][:,-1]<0).all()
