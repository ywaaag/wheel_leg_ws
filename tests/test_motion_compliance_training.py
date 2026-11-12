"""Pure CPU semantic preparation; no simulator or optimizer steps."""
FROZEN_TEST_SCOPE='PORTABLE'
import pytest
import torch
from native_stand.students.motion_compliance_training import (
    CRITIC,PROFILES,assignments,command_set,critic_input,declaration)


def fields(n=4):
    return dict(height=torch.full((n,1),.34),velocity_native=torch.zeros(n,3),
        contacts=torch.full((n,2),50.),controlled_s=torch.full((n,1),15.),
        load_q=torch.full((n,4),.2),load_dq=torch.ones(n,4),
        encoder_q=torch.full((n,4),.21),encoder_dq=torch.full((n,4),1.2),
        K=torch.tensor([r['K'] for r in PROFILES])[:,None].expand(-1,4).clone(),
        B=torch.tensor([r['B'] for r in PROFILES])[:,None].expand(-1,4).clone(),
        rotor_J=torch.tensor([r['rotor_J'] for r in PROFILES])[:,None],
        load_armature=torch.tensor([r['load_armature'] for r in PROFILES])[:,None],version=CRITIC)


def test_critic64_preserves_observed_prefix_and_has_explicit_physical_scaling():
    actor=torch.arange(124,dtype=torch.float32).reshape(4,31)
    result=critic_input(actor,**fields())
    assert result.shape==(4,64) and torch.equal(result[:,:31],actor)
    assert torch.allclose(result[:,31],torch.full((4,),.68))
    assert torch.allclose(result[:,35:37],torch.full((4,2),.5))
    assert (result[0,46:63]==0).all() and result[0,63]==1
    assert torch.allclose(result[1:,46:50],torch.full((3,4),.1))
    assert torch.allclose(result[1:,50:54],torch.full((3,4),.2))
    assert torch.allclose(result[1:,62],torch.full((3,),.75))
    assert torch.allclose(result[1:,63],torch.full((3,),.25))


def test_unrelated64d_semantics_and_nonfinite_cannot_be_silently_accepted():
    actor=torch.zeros(4,31);kw=fields();kw['version']='recovery_privileged64_v1'
    with pytest.raises(ValueError,match='explicit'):critic_input(actor,**kw)
    kw=fields();kw['encoder_dq'][1,0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):critic_input(actor,**kw)
    kw=fields();kw['load_q']=torch.zeros(4,6)
    with pytest.raises(ValueError,match='shape'):critic_input(actor,**kw)


@pytest.mark.parametrize('stage',range(4))
def test_profiles_prefix_and_commands_are_crossed_over_reset_cycles(stage):
    seen={p['name']:set() for p in PROFILES}
    rng=torch.get_rng_state().clone()
    for cycle in range(32):
        rows=assignments(stage,cycle)
        assert len(rows)==8
        assert sum(r['command'][:2]==[0.,0.] for r in rows)==(8 if stage==0 else 4)
        for profile in seen:
            pair=[r for r in rows if r['profile']==profile]
            assert len(pair)==2 and {r['prefix_s'] for r in pair}=={0.,1.}
            for r in pair:seen[profile].add((tuple(r['command'][:2]),r['prefix_s']))
    required={(tuple(c),prefix) for c in command_set(stage) for prefix in (0.,1.)}
    assert all(v==required for v in seen.values())
    assert torch.equal(rng,torch.get_rng_state())


def test_budget_and_smoke_do_not_claim_impossible_timeout_coverage():
    d=declaration();b=d['budget_proposal']
    assert b['smoke']+b['formal']==b['online_updates']==300
    assert 300*b['envs']*b['steps_per_env']==b['raw_rollout_samples']==307200
    assert d['smoke_reset_gap']['per_env_policy_seconds']<30
    assert not d['smoke_reset_gap']['timeout_coverage_guaranteed']
    assert d['runtime_recipe_ready'] is False and d['remaining_wiring']
    assert all(p['rotor_J']+p['load_armature']==pytest.approx(.012) for p in PROFILES)
