"""PORTABLE CPU batching/scoring; no simulator, optimizer or ledger."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy,ast
from pathlib import Path
import pytest
import torch
from native_stand.students.grouped_evaluation import describe_plan,CommandGroups
from native_stand.students import motion_parallel_assessment as a

def cases(n=12):
 return [dict(case_id=str(i),kind='student',role='MotionZeroStudent',family='motion_native_normal',seed=17,yaw_rad=0.,seconds=31.52,checkpoint='/fake/a.pt',sha256='a'*64,command=[0.,i/4,.34]) for i in range(n)]

def base_protocol():
 return {'version':'motion31_differential85_assessment_v1','controlled_seconds':30.,'symmetry':{'swing_strict_max_rad':.05},'sample':'CPU fixture; not physical certification'}

def inputs(target,response,n=3000):
 s=torch.zeros(n,39,dtype=torch.float64);s[:,29]=1;s[:,9 if target[1] else 7]=response
 checks={k:True for k in a.REQUIRED}
 if target[1]:checks.update({k:True for k in ('spin_center','spin_common','difference_complete','difference_tracking','opposite_rolling','rolling_equality')})
 checks.update(tracking=False,direction=False,gain=False,sustained_fraction=False)
 return {'checks':checks,'tracking':{},'accepted':False},s,torch.ones(n,dtype=torch.bool)

def test_twelve_commands_use_three_batches_not_ninety_six_simultaneously():
 p=describe_plan(cases());assert p['case_count']==12 and p['batch_count']==3
 assert p['total_scored_episodes']==96 and p['actual_env_capacity']==64
 assert [b['scored_envs'] for b in p['batches']]==[32,32,32]
 assert p['unique_policy_steps']==3*3152 and p['all_simulated_env_steps']==3*3152*64
 p=describe_plan(cases(),maximum_commands=8);assert [b['scored_envs'] for b in p['batches']]==[64,32]
 assert p['wall_seconds'].startswith('UNKNOWN')

def test_group_settings_are_not_ignored_in_ppo0():
 tree=ast.parse(Path('src/native_stand/students/evaluator.py').read_text())
 calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='CommandGroups']
 assert len(calls)==1 and any(k.arg=='maximum_commands' for k in calls[0].keywords)
 g=CommandGroups(cases(),maximum_commands=2);assert all(len(group)==2 for group in g.groups.values())

def test_invalid_batch_numbers_and_seed_separation():
 x=cases();x[0]['command'][0]=float('nan')
 with pytest.raises(ValueError):describe_plan(x)
 x=cases();x[1]['case_id']='0'
 with pytest.raises(ValueError):describe_plan(x)
 x=cases(2);x[1]['seed']=18;assert describe_plan(x)['batch_count']==2

@pytest.mark.parametrize('target,response', [([4.,0.,.34],3.8),([-4.,0.,.34],-3.8),([0.,4.,.34],3.8),([0.,-4.,.34],-3.8)])
def test_four_endpoints_sign_axis_units_and_full_window(target,response):
 r,s,c=inputs(target,response);r0=copy.deepcopy(r)
 out=a.assess(r,s,c,target,a.protocol(base_protocol()))
 assert out['accepted'] and out['parallel_tracking']['MAE']==pytest.approx(.2)
 assert out['parallel_tracking']['continuous_above_3p8_s']==8.
 assert out['parallel_tracking']['axis']==('yaw' if target[1] else 'vx')
 assert out['parallel_tracking']['unit']==('rad/s' if target[1] else 'm/s') and r==r0

@pytest.mark.parametrize('response',[-4.,0.,5.])
def test_wrong_sign_still_overshoot_not_accepted(response):
 r,s,c=inputs([4.,0.],response);assert not a.assess(r,s,c,[4.,0.],a.protocol(base_protocol()))['accepted']

def test_coarse_mae_applies_only_new_endpoint():
 r,s,c=inputs([3.,0.],2.65);new=a.assess(r,s,c,[3.,0.],a.protocol(base_protocol()))
 assert new['accepted'] and new['parallel_tracking']['MAE_limit']==pytest.approx(.45)
 r,s,c=inputs([1.,0.],1.1);old=a.assess(r,s,c,[1.,0.],a.protocol(base_protocol()))
 assert old['parallel_tracking']['MAE_limit']==.3 and old['parallel_tracking']['original_command']

def test_incomplete_and_missing_safety_never_pass():
 r,s,c=inputs([4.,0.],4.,2999);o=a.assess(r,s,c,[4.,0.],a.protocol(base_protocol()))
 assert not o['accepted'] and not o['checks']['parallel_complete_window']
 r,s,c=inputs([4.,0.],4.);r['checks'].pop('swing_full');o=a.assess(r,s,c,[4.,0.],a.protocol(base_protocol()))
 assert not o['accepted'] and o['parallel_tracking']['missing_safety_evidence']==['swing_full']
 r,s,c=inputs([4.,0.],4.);r['checks']['contact']=False;assert not a.assess(r,s,c,[4.,0.],a.protocol(base_protocol()))['accepted']


def test_continuous_three_seconds_cannot_be_spliced():
 r,s,c=inputs([4.,0.],3.7)
 s[1000:1250,7]=3.9;s[1350:1600,7]=3.9
 o=a.assess(r,s,c,[4.,0.],a.protocol(base_protocol()))
 assert o['checks']['final_mean'] and not o['checks']['final_sustained'] and o['parallel_tracking']['continuous_above_3p8_s']==2.5


def test_unknown_protocol_and_nonfinite_rejected():
 r,s,c=inputs([4.,0.],4.);config=a.protocol(base_protocol());config['sample_rate_hz']=50
 with pytest.raises(ValueError):a.assess(r,s,c,[4.,0.],config)
 with pytest.raises(ValueError):a.assess(r,s,c,[1.,1.],a.protocol(base_protocol()))
 s[1500,7]=float('nan');assert not a.assess(r,s,c,[4.,0.],a.protocol(base_protocol()))['accepted']

def test_shared_evaluator_rejects_wrong_task_or_safety_protocol():
 from types import SimpleNamespace
 from native_stand.students.evaluator import score_arrays
 env=SimpleNamespace(motion_parallel_protocol=a.protocol(base_protocol()),motion_training_enabled=False)
 with pytest.raises(ValueError,match='Motion85'):score_arrays(env,{}, {},scored_envs=0)
 env.motion_training_enabled=True;env.critic_privileged_version='motion_symmetry_privileged85_v1';env.motion_symmetry_protocol={}
 with pytest.raises(ValueError,match='base protocol'):score_arrays(env,{}, {},scored_envs=0)


def test_explicit_new_protocol_uses_same_scorer_for_serial_and_grouped(monkeypatch):
 import numpy as np
 from types import SimpleNamespace
 from native_stand.students import evaluator,grouped_evaluation,motion_training,motion_symmetry,motion_differential
 base=base_protocol();env=SimpleNamespace(num_envs=64,interface_version='multiskill_mcu_v2',ids=list(range(6)),
 motion_training_enabled=True,critic_privileged_version='motion_symmetry_privileged85_v1',motion_symmetry_protocol=base,
 motion_quality_reward_config={'version':'student_motion31_differential_quality_v9'},motion_parallel_protocol=a.protocol(base))
 n=3000;arrays={'state':np.zeros((n,16,39),np.float32),'posture':np.zeros((n,16,10),np.float32),
 'geometry':np.zeros((n,16,13),np.float32),'root_post':np.zeros((n,16,13),np.float32),
 'spin_rolling_speed_forward_mps':np.zeros((n,16,2),np.float32),
 'controller':np.full((n,16,1),3),'first_episode_valid':np.ones((n,16),bool),
 'policy_time_s':np.arange(n)*.01,'termination_causes':np.zeros((n,16,6),bool)}
 arrays['state'][:,:,29]=1;arrays['state'][:,:8,7]=3.8;arrays['state'][:,8:,9]=-3.8
 def safety_result(*args):
  return {'checks':{k:True for k in (*a.REQUIRED,'spin_center','spin_common','difference_complete','difference_tracking','opposite_rolling','rolling_equality')},'tracking':{},'symmetry_metrics':{},'accepted':True}
 monkeypatch.setattr(motion_training,'score',safety_result)
 monkeypatch.setattr(motion_symmetry,'assess',lambda result,*args:result)
 monkeypatch.setattr(motion_differential,'assess',lambda result,*args:result)
 rows=cases(2);rows[0]['command']=[4.,0.,.34];rows[1]['command']=[0.,-4.,.34]
 expected=[evaluator.score_arrays(env,row,grouped_evaluation.split_arrays(arrays,i*8))[0] for i,row in enumerate(rows)]
 monkeypatch.setattr(evaluator,'rollout',lambda *args,**kwargs:arrays)
 actual=grouped_evaluation.rollout(env,None,rows)
 for row,old in zip(rows,expected):
  new=actual[row['case_id']][0];assert new['protocol']==a.protocol(base) and new['per_env']==old['per_env'] and new['passed_envs']==8
 assert actual['0'][0]['per_env'][0]['parallel_tracking']['axis']=='vx'
 assert actual['1'][0]['per_env'][0]['parallel_tracking']['axis']=='yaw'

def test_local_frozen_v9_effective_protocol_is_preserved():
 import json
 root=Path(__file__).resolve().parents[1]
 spec=root/'configs/experiments/2026_10_18-23_10__motion1000_differential85_5000_retry/spec.json'
 raw=root/'runs/2026_10_18-23_10__motion1000_differential85_5000_retry/formal/evaluation_84.json'
 if not spec.exists() or not raw.exists():pytest.skip('local frozen source protocol fixture unavailable')
 assert a.safety_protocol(json.loads(spec.read_text()))==json.loads(raw.read_text())['protocol']
