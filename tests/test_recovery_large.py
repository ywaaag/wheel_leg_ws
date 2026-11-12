"""Recovery64 widening and real shared model-loading boundaries, CPU only."""
FROZEN_TEST_SCOPE='PORTABLE'
import copy
import pytest
import torch
from native_stand.students.model import Student,load
from native_stand.students.recovery_large import network,expand,VERSION,SOURCE_SHA


def candidate(a,c):
    return {'initialization_kind':VERSION,'interface_version':'multiskill_mcu_v2',
        'role':'RecoveryStudent','network_architecture':network(),
        'critic_privileged_version':'recovery_privileged64_v1','source_sha256':SOURCE_SHA,
        **{k:'a'*64 for k in ('asset_sha256','source_spec_sha256','dataset_sha256','manifest_sha256')},
        'accepted':False,'online_updates_this_stage':0,'actor':a,'critic':c}


def models():
    return (Student(version='multiskill_mcu_v2'),
            Student(False,version='multiskill_mcu_v2',critic_privileged_version='recovery_privileged64_v1'),
            Student(version='multiskill_mcu_v2',network_architecture=network()),
            Student(False,version='multiskill_mcu_v2',critic_privileged_version='recovery_privileged64_v1',network_architecture=network()))


def test_actor_and_critic_functions_jacobian_std_and_no_mutation():
    a,c,wide,wc=[m.double() for m in models()]
    original=copy.deepcopy(a.state_dict());cv=copy.deepcopy(c.state_dict())
    x=torch.randn(13,31,dtype=torch.float64,requires_grad=True)
    v=torch.randn(13,64,dtype=torch.float64,requires_grad=True)
    rng=torch.get_rng_state().clone();sa,sc=expand(a.state_dict(),c.state_dict())
    assert torch.equal(rng,torch.get_rng_state())
    wide.load_state_dict(sa);wc.load_state_dict(sc)
    for small,big,obs in [(a,wide,x),(c,wc,v)]:
        assert torch.allclose(small(obs),big(obs),atol=1e-12,rtol=1e-12)
        js=torch.autograd.grad(small(obs).sum(),obs)[0]
        jl=torch.autograd.grad(big(obs).sum(),obs)[0]
        assert torch.allclose(js,jl,atol=1e-12,rtol=1e-12)
    assert all(torch.equal(v,a.state_dict()[k]) for k,v in original.items())
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in cv.items())
    assert torch.equal(a.log_std,wide.log_std) and a.log_std.data_ptr()!=wide.log_std.data_ptr()


def test_gaussian_coordinates_and_same_noise_samples():
    a,c,wide,_=models();sa,_=expand(a.state_dict(),c.state_dict());wide.load_state_dict(sa)
    x=torch.randn(17,31)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31);left=a(x,True);torch.manual_seed(31);right=wide(x,True)
    assert torch.allclose(left,right,atol=1e-6,rtol=1e-5)
    assert torch.allclose(a.log_prob(x,left),wide.log_prob(x,left),atol=1e-4,rtol=1e-5)


def test_shared_loading_roundtrip_and_no_implicit_ppo_recipe(tmp_path):
    a,c,_,_=models();sa,sc=expand(a.state_dict(),c.state_dict())
    p=tmp_path/'initial.pt';torch.save(candidate(sa,sc),p)
    rng=torch.get_rng_state().clone();model,data=load(p,'RecoveryStudent')
    assert torch.equal(rng,torch.get_rng_state()) and model.mlp[0].out_features==512
    from native_stand.students.ppo import assemble
    with pytest.raises(ValueError,match='new explicit training recipe'):assemble(p,None)
    from native_stand.students.model import ppo_initialization
    with pytest.raises(ValueError,match='explicit training recipe'):ppo_initialization(p)
    data['accepted']=True;torch.save(data,p)
    with pytest.raises(ValueError,match='no new physical capability'):load(p,'RecoveryStudent')


def test_explicit_critic_semantics_not_inferred_from_shape():
    with pytest.raises(ValueError,match='Recovery64'):
        Student(False,version='multiskill_mcu_v2',critic_privileged_version='motion_privileged49_v1',network_architecture=network())
    altered=network();altered['critic_input']=49
    with pytest.raises(ValueError,match='declaration'):
        Student(version='multiskill_mcu_v2',network_architecture=altered)
    a=Student(version='multiskill_mcu_v2');motion=Student(False,version='multiskill_mcu_v2',critic_privileged_version='motion_privileged49_v1')
    with pytest.raises(ValueError,match='shape'):expand(a.state_dict(),motion.state_dict())


def test_evaluator_reports_each_loaded_architecture_with_recovery64():
    from types import SimpleNamespace
    from native_stand.students.evaluator import model_contract
    a,c,wide,_=models()
    env=SimpleNamespace(interface_version='multiskill_mcu_v2',critic_privileged_version='recovery_privileged64_v1')
    assert model_contract(env,a)['actor_hidden_dims']==[128,64,32]
    contract=model_contract(env,wide)
    assert contract['actor_hidden_dims']==[512,256,128]
    assert contract['critic_dim']==64 and contract['critic_privileged_version']=='recovery_privileged64_v1'
