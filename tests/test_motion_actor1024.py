FROZEN_TEST_SCOPE = 'PORTABLE'
import copy

import pytest
import torch

from native_stand.students.actor_widening import (wide_declaration,wide_network,expand_actor,
    expand_optimizer,expand_mcu_checkpoint,validate_wide_candidate)
from native_stand.students.motion_mcu_privileged import network,VERSION
from native_stand.students.model import Student,load


def source():
    actor=Student(True,version='multiskill_mcu_v2',network_architecture=network())
    critic=Student(False,version='multiskill_mcu_v2',critic_privileged_version=VERSION,network_architecture=network())
    optimizer=torch.optim.Adam([*actor.parameters(),*critic.parameters()],lr=3e-5)
    x=torch.linspace(-.5,.5,31).expand(3,-1)
    loss=actor.mean(x).square().mean()+critic(torch.zeros(3,312)).square().mean()
    loss.backward();optimizer.step()
    return dict(actor=actor.state_dict(),critic=critic.state_dict(),optimizer=optimizer.state_dict(),
        network_architecture=network(),critic_privileged_version=VERSION,interface_version='multiskill_mcu_v2',
        role='MotionZeroStudent',spec_sha256='b'*64,asset_sha256='c'*64,
        training_recipe={'network_architecture':network()},recipe_version='source-only',
        RNG=torch.get_rng_state(),fixed_std=True)


@pytest.mark.parametrize('width',[1024,2048])
def test_wide_mean_jacobian_and_std_preserved_without_rng_or_source_mutation(width):
    a=Student(True,version='multiskill_mcu_v2',network_architecture=network()).double()
    old=copy.deepcopy(a.state_dict());rng=torch.get_rng_state().clone()
    wide=Student(True,version='multiskill_mcu_v2',network_architecture=wide_network(width)).double()
    # Network construction uses RNG; expansion itself must not.
    rng=torch.get_rng_state().clone();wide.load_state_dict(expand_actor(old,wide_declaration(width)))
    assert torch.equal(rng,torch.get_rng_state())
    x=torch.linspace(-1,1,31).expand(11,-1).clone().requires_grad_()
    y0,y1=a.mean(x.double()),wide.mean(x.double())
    assert torch.allclose(y0,y1,atol=1e-12,rtol=1e-12)
    j0=torch.autograd.grad(y0.sum(),x,retain_graph=True)[0]
    j1=torch.autograd.grad(y1.sum(),x)[0]
    assert torch.allclose(j0,j1,atol=1e-7,rtol=1e-6)
    assert all(torch.equal(v,a.state_dict()[k]) for k,v in old.items())
    assert torch.equal(a.log_std,wide.log_std)


@pytest.mark.parametrize('width',[1024,2048])
def test_adam_critic_steps_groups_rng_and_save_readback_retained(tmp_path,width):
    s=source();before=copy.deepcopy(s);out=expand_mcu_checkpoint(s,source_checkpoint_sha256='a'*64,width=width)
    assert out['runtime_recipe_ready'] is False and out['online_updates_after_expansion']==0
    assert out['optimizer']['param_groups']==s['optimizer']['param_groups']
    assert torch.equal(out['RNG'],s['RNG'])
    ids=out['optimizer']['param_groups'][0]['params']
    for pid in ids[len(s['actor']):]:
        for k,v in s['optimizer']['state'][pid].items():assert torch.equal(out['optimizer']['state'][pid][k],v)
    for k,v in s['critic'].items():assert torch.equal(out['critic'][k],v)
    for k,v in s['actor'].items():assert torch.equal(before['actor'][k],v)
    for pid,state in out['optimizer']['state'].items():
        assert torch.equal(state['step'],s['optimizer']['state'][pid]['step'])
        assert all(torch.isfinite(v).all() for v in state.values())
    path=tmp_path/'initial.pt';torch.save(out,path);a,data=load(path,'MotionZeroStudent')
    assert data['initialization_kind'] and a.mlp[0].weight.shape==(width,31)
    x=torch.zeros(17,31)
    old=Student(True,version='multiskill_mcu_v2',network_architecture=network());old.load_state_dict(s['actor'])
    assert torch.allclose(a.mean(x),old.mean(x),atol=2e-6,rtol=2e-6)
    # Confirm a real backward/optimizer step is possible on a disposable copy.
    critic=Student(False,version='multiskill_mcu_v2',critic_privileged_version=VERSION,network_architecture=wide_network(width))
    critic.load_state_dict(out['critic']);opt=torch.optim.Adam([*a.parameters(),*critic.parameters()]);opt.load_state_dict(out['optimizer'])
    opt.zero_grad();loss=a.mean(x).square().mean()+critic(torch.zeros(17,312)).square().mean();loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in [*a.parameters(),*critic.parameters()])
    opt.step()


def test_unknown_topology_or_wrong_critic_rejected():
    s=source();s['critic_privileged_version']='standing_privileged49_v1'
    with pytest.raises(ValueError,match='Motion31'):expand_mcu_checkpoint(s,source_checkpoint_sha256='a'*64)
    with pytest.raises(ValueError,match='declaration'):expand_actor(s['actor'],{**wide_declaration(),'target_dims':[31,2048,1024,512,6]})
    with pytest.raises(ValueError,match='required'):Student(False,version='multiskill_mcu_v2',network_architecture=wide_network())


def test_candidate_cannot_claim_trained_ready():
    out=expand_mcu_checkpoint(source(),source_checkpoint_sha256='a'*64);out['runtime_recipe_ready']=True
    with pytest.raises(ValueError,match='untrained'):validate_wide_candidate(out)


def test_larger_topology_requires_explicit_new_version_and_declared_width():
    assert wide_network(2048)['actor_hidden_dims']==[2048,1024,512]
    out=expand_mcu_checkpoint(source(),source_checkpoint_sha256='a'*64,width=2048)
    assert out['initialization_kind']=='motion31_mcu_actor2048_initial_v1'
    out['initialization_kind']='motion31_mcu_actor1024_initial_v1'
    with pytest.raises(ValueError,match='untrained'):validate_wide_candidate(out)
    with pytest.raises(ValueError,match='width'):wide_network(4096)
