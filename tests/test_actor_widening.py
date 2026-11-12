FROZEN_TEST_SCOPE = 'PORTABLE'
import copy

import pytest
import torch
from torch import nn

from native_stand.students.actor_widening import (
    SOURCE_DIMS, TARGET_DIMS, declaration, expand_actor, expand_optimizer,
)
from native_stand.students.model import Student


def mean(state, x):
    for i in range(4):
        x = nn.functional.linear(x, state[f'mlp.{2*i}.weight'], state[f'mlp.{2*i}.bias'])
        x = nn.functional.elu(x) if i < 3 else x.tanh()
    return x


def test_widened_function_and_local_input_jacobian_match_without_rng_or_mutation():
    actor = Student(True, version='multiskill_mcu_v2').double()
    original = copy.deepcopy(actor.state_dict())
    x = torch.randn(37, 31, dtype=torch.float64, requires_grad=True)
    rng = torch.get_rng_state().clone()
    wide = expand_actor(original, declaration())
    small_y, large_y = mean(original, x), mean(wide, x)
    assert torch.allclose(small_y, large_y, atol=1e-12, rtol=1e-12)
    js = torch.autograd.grad(small_y.sum(), x)[0]
    jl = torch.autograd.grad(large_y.sum(), x)[0]
    assert torch.allclose(js, jl, atol=1e-12, rtol=1e-12)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, actor.state_dict()[k]) for k, v in original.items())


def test_nonuniform_split_replicas_have_distinct_gradients():
    source = Student(True, version='multiskill_mcu_v2').state_dict()
    wide = {k:v.detach().requires_grad_(k!='log_std') for k,v in expand_actor(source,declaration()).items()}
    loss = mean(wide,torch.randn(19,31)).square().sum()
    g = torch.autograd.grad(loss,wide['mlp.0.weight'])[0]
    assert g[:128].norm()>0
    assert torch.allclose(g[128:256],2*g[:128],atol=2e-6,rtol=2e-4)
    assert not torch.equal(g[:128],g[128:256])


def test_explicit_adam_mapping_preserves_critic_steps_groups_and_input():
    actor=Student(True,version='multiskill_mcu_v2');source=actor.state_dict()
    names=list(dict(actor.named_parameters()));ids=dict(zip(names,range(len(names))))
    state={i:{'step':torch.tensor(1600.),'exp_avg':torch.ones_like(source[k])*.2,
              'exp_avg_sq':torch.ones_like(source[k])*.4} for k,i in ids.items() if k!='log_std'}
    state[99]={'step':torch.tensor(1600.),'exp_avg':torch.ones(3,4),'exp_avg_sq':torch.ones(3,4)*2}
    opt={'state':state,'param_groups':[{'params':list(ids.values())+[99],'lr':3e-5}]}
    before=copy.deepcopy(opt);new=expand_optimizer(opt,source,ids,declaration())
    assert new['param_groups']==opt['param_groups']
    assert torch.equal(new['state'][99]['exp_avg'],before['state'][99]['exp_avg'])
    pid=ids['mlp.0.weight'];m=new['state'][pid]
    assert m['exp_avg'].shape==(512,31) and torch.equal(m['step'],state[pid]['step'])
    assert torch.allclose(m['exp_avg'][:128],torch.ones(128,31)*.02)
    assert torch.allclose(m['exp_avg_sq'][:128],torch.ones(128,31)*.004)
    assert torch.equal(opt['state'][pid]['exp_avg'],before['state'][pid]['exp_avg'])
    with pytest.raises(ValueError,match='map required'):expand_optimizer(opt,source,{},declaration())


def test_declaration_and_source_shapes_are_explicit():
    actor=Student(True,version='multiskill_mcu_v2').state_dict()
    cfg=declaration();cfg['target_dims']=[31,1024,512,256,6]
    with pytest.raises(ValueError,match='declaration'):expand_actor(actor,cfg)
    actor['mlp.0.weight']=torch.zeros(128,30)
    with pytest.raises(ValueError,match='shape'):expand_actor(actor,declaration())


def test_critic_expansion_keeps_linear_value_and_explicit_49d_semantics():
    from native_stand.students.actor_widening import expand_critic,critic_declaration
    critic=Student(False,version='multiskill_mcu_v2',critic_privileged_version='motion_privileged49_v1').double()
    original=critic.state_dict();wide=expand_critic(original,critic_declaration())
    x=torch.randn(17,49,dtype=torch.float64)
    def value(s,x):
        for i in range(4):
            x=nn.functional.linear(x,s[f'mlp.{i*2}.weight'],s[f'mlp.{i*2}.bias'])
            if i<3:x=nn.functional.elu(x)
        return x
    assert torch.allclose(value(original,x),value(wide,x),atol=1e-12,rtol=1e-12)
    ids=dict(zip(original,range(len(original))))
    opt={'param_groups':[{'params':list(ids.values()),'lr':3e-5}],
         'state':{i:{'step':torch.tensor(1600.),'exp_avg':torch.ones_like(original[k]),'exp_avg_sq':torch.ones_like(original[k])} for k,i in ids.items()}}
    new=expand_optimizer(opt,original,ids,critic_declaration(),role='critic')
    assert new['state'][ids['mlp.0.weight']]['exp_avg'].shape==(512,49)
    assert new['state'][ids['mlp.2.weight']]['exp_avg'].shape==(512,512)
    assert new['state'][ids['mlp.4.weight']]['exp_avg'].shape==(256,512)
