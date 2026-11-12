"""CPU/tensor-only Motion Actor expansion; no runtime dispatch or file IO."""
import copy

import torch

VERSION = 'motion31_actor512_function_preserving_v1'
SOURCE_DIMS = (31, 128, 64, 32, 6)
TARGET_DIMS = (31, 512, 256, 128, 6)
CRITIC_SOURCE_DIMS = (49, 256, 128, 64, 1)
CRITIC_TARGET_DIMS = (49, 512, 512, 256, 1)
WIDE_SOURCE_DIMS = TARGET_DIMS
WIDE_TARGET_DIMS = (31, 1024, 512, 256, 6)
WIDE_VERSION = 'motion31_actor1024_function_preserving_v1'
WIDE_INITIALIZATION = 'motion31_mcu_actor1024_initial_v1'
LARGER_TARGET_DIMS = (31, 2048, 1024, 512, 6)
LARGER_VERSION = 'motion31_actor2048_function_preserving_v1'
LARGER_INITIALIZATION = 'motion31_mcu_actor2048_initial_v1'


def wide_declaration(width=1024):
    if type(width) is not int or width not in (1024,2048):
        raise ValueError('explicit Actor width1024 or2048 required')
    return {**declaration(), 'version': WIDE_VERSION if width==1024 else LARGER_VERSION,
            'source_dims': list(WIDE_SOURCE_DIMS),
            'target_dims': list(WIDE_TARGET_DIMS if width==1024 else LARGER_TARGET_DIMS),
            'split': 'positive nonuniform replica fractions 1:2' if width==1024 else
                     'positive nonuniform replica fractions 1:2:3:4'}


def wide_network(width=1024):
    from .motion_mcu_privileged import network
    dims=wide_declaration(width)['target_dims']
    return {**network(), 'version': f'motion31_mcu_actor{width}_critic312_v1',
            'actor_hidden_dims': dims[1:-1]}


def _actor_dims(config):
    if config == declaration():
        return SOURCE_DIMS, TARGET_DIMS
    if config == wide_declaration():
        return WIDE_SOURCE_DIMS, WIDE_TARGET_DIMS
    if config == wide_declaration(2048):
        return WIDE_SOURCE_DIMS, LARGER_TARGET_DIMS
    raise ValueError('explicit Actor widening declaration differs')


def declaration():
    return {'version': VERSION, 'source_dims': list(SOURCE_DIMS),
            'target_dims': list(TARGET_DIMS), 'activation': 'ELU(alpha=1)',
            'mean_output': 'Tanh', 'split': 'positive nonuniform replica fractions 1:2:3:4',
            'optimizer': 'preserve steps; replicate first/second moments using local gradient factors; critic unchanged',
            'scope': 'function at initialization only; not equal Adam trajectory or capacity/survival guarantee'}


def critic_declaration():
    return {**declaration(), 'version':'motion49_critic512_function_preserving_v1',
            'source_dims':list(CRITIC_SOURCE_DIMS), 'target_dims':list(CRITIC_TARGET_DIMS),
            'mean_output':'linear scalar', 'split':'positive nonuniform replica fractions per expanded layer',
            'optimizer':'preserve steps; map Critic moments using local gradient factors; Actor unchanged'}


def _layout(layer, reference, source_dims=SOURCE_DIMS, target_dims=TARGET_DIMS):
    old_in, old_out = source_dims[layer:layer+2]
    new_in, new_out = target_dims[layer:layer+2]
    rows = torch.arange(new_out, device=reference.device) % old_out
    cols = torch.arange(new_in, device=reference.device) % old_in
    # A replica's activation is identical at initialization. Fractions sum to
    # one, preserving downstream preactivation while breaking gradient symmetry.
    def fractions(old, new):
        replicas = new // old
        group = torch.arange(new, device=reference.device) // old
        return (group.to(reference.dtype)+1)/(replicas*(replicas+1)/2)
    incoming = fractions(old_in, new_in)
    downstream = fractions(old_out, new_out)
    return rows, cols, incoming, downstream


def _validate(actor, source_dims=SOURCE_DIMS):
    keys = {'log_std'} | {f'mlp.{2*i}.{suffix}' for i in range(4) for suffix in ('weight', 'bias')}
    if set(actor) != keys:
        raise ValueError('explicit small Motion Actor parameter names required')
    for i, (n_in, n_out) in enumerate(zip(source_dims, source_dims[1:])):
        w, b = actor[f'mlp.{2*i}.weight'], actor[f'mlp.{2*i}.bias']
        if w.shape != (n_out, n_in) or b.shape != (n_out,):
            raise ValueError('explicit small Motion Actor shape differs')
    if actor['log_std'].shape != (6,) or any(not v.is_floating_point() or not torch.isfinite(v).all() for v in actor.values()):
        raise ValueError('finite floating Actor/std required')


def expand_actor(actor, config):
    """Replicate hidden neurons and split each outgoing weight; no RNG use."""
    source_dims, target_dims = _actor_dims(config)
    _validate(actor, source_dims)
    result = {'log_std': actor['log_std'].clone()}
    for i in range(4):
        w, b = actor[f'mlp.{2*i}.weight'], actor[f'mlp.{2*i}.bias']
        rows, cols, split, _ = _layout(i, w, source_dims, target_dims)
        result[f'mlp.{2*i}.weight'] = w[rows[:, None], cols[None, :]] * split[None, :]
        result[f'mlp.{2*i}.bias'] = b[rows].clone()
    return result


def _validate_critic(critic, source_dims=CRITIC_SOURCE_DIMS):
    keys={f'mlp.{2*i}.{suffix}' for i in range(4) for suffix in ('weight','bias')}
    if set(critic)!=keys:
        raise ValueError('explicit Motion Critic parameter names required')
    for i,(n_in,n_out) in enumerate(zip(source_dims,source_dims[1:])):
        if critic[f'mlp.{2*i}.weight'].shape!=(n_out,n_in) or critic[f'mlp.{2*i}.bias'].shape!=(n_out,):
            raise ValueError('explicit Motion Critic shape differs')
    if any(not v.is_floating_point() or not torch.isfinite(v).all() for v in critic.values()):
        raise ValueError('finite floating Critic required')


def expand_critic(critic, config):
    if config!=critic_declaration():
        raise ValueError('explicit Critic widening declaration differs')
    return expand_value_network(critic,CRITIC_SOURCE_DIMS,CRITIC_TARGET_DIMS)


def expand_value_network(critic, source_dims, target_dims):
    """Same replication algebra for explicitly declared Motion49/Recovery64."""
    allowed=((CRITIC_SOURCE_DIMS,CRITIC_TARGET_DIMS),
             ((64,256,128,64,1),(64,512,512,256,1)))
    if (source_dims,target_dims) not in allowed:
        raise ValueError('unsupported explicit Critic expansion topology')
    _validate_critic(critic,source_dims)
    result={}
    for i in range(4):
        w,b=critic[f'mlp.{2*i}.weight'],critic[f'mlp.{2*i}.bias']
        rows,cols,split,_=_layout(i,w,source_dims,target_dims)
        result[f'mlp.{2*i}.weight']=w[rows[:,None],cols[None,:]]*split[None,:]
        result[f'mlp.{2*i}.bias']=b[rows].clone()
    return result


def expand_optimizer(optimizer, actor, parameter_ids, config, *, role='actor'):
    """Explicit name→optimizer-ID mapping; no shape guessing or Adam reset.

    At the replicated function, each hidden replica receives its downstream
    split of the original local gradient. This maps moments, not finite Adam
    updates; epsilon, clipping and the wider parameterization change learning.
    """
    if role not in ('actor','critic') or (role=='critic' and config != critic_declaration()):
        raise ValueError('explicit Actor widening declaration differs')
    source_dims, target_dims = (_actor_dims(config) if role=='actor' else
                                (CRITIC_SOURCE_DIMS, CRITIC_TARGET_DIMS))
    (_validate if role=='actor' else _validate_critic)(actor, source_dims)
    if set(parameter_ids) != set(actor) or len(set(parameter_ids.values())) != len(actor):
        raise ValueError('complete unique Actor optimizer parameter-ID map required')
    all_ids = [pid for group in optimizer['param_groups'] for pid in group['params']]
    if len(set(all_ids)) != len(all_ids) or any(pid not in all_ids for pid in parameter_ids.values()):
        raise ValueError('Actor optimizer mapping not in parameter groups')
    result = copy.deepcopy(optimizer)
    for name, pid in parameter_ids.items():
        state = result['state'].get(pid)
        if not state:
            continue  # Frozen std usually has no Adam state.
        if set(state) - {'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'}:
            raise ValueError('unsupported Adam state fields')
        for field in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if field not in state:
                continue
            value = state[field]
            if value.shape != actor[name].shape or not torch.isfinite(value).all():
                raise ValueError('Actor Adam moment identity/shape differs')
            if name == 'log_std':
                continue
            layer = int(name.split('.')[1])//2
            rows, cols, _, factor = _layout(layer, value,
                source_dims, target_dims)
            expanded = value[rows[:, None], cols[None, :]] if name.endswith('weight') else value[rows]
            factor = factor[:, None] if name.endswith('weight') else factor
            state[field] = expanded * (factor if field == 'exp_avg' else factor.square())
    return result


def expand_mcu_checkpoint(source, *, source_checkpoint_sha256, width=1024):
    """A new initialization artifact, never a completed/submit-ready PPO recipe."""
    from .motion_mcu_privileged import network, VERSION as critic_version
    if (source.get('network_architecture') != network() or
            source.get('critic_privileged_version') != critic_version or
            source.get('interface_version') != 'multiskill_mcu_v2' or
            source.get('role') != 'MotionZeroStudent'):
        raise ValueError('explicit trained Motion31/MCU312 source required')
    actor = source['actor']; _validate(actor, WIDE_SOURCE_DIMS)
    names = list(actor)
    # Match the declared registration order, including frozen log_std.
    expected = ['log_std'] + [f'mlp.{2*i}.{s}' for i in range(4) for s in ('weight','bias')]
    ids = [pid for group in source['optimizer']['param_groups'] for pid in group['params']]
    if names != expected or len(ids) != len(actor) + len(source['critic']):
        raise ValueError('explicit source Adam parameter order differs')
    out = copy.deepcopy(source)
    config=wide_declaration(width)
    out['optimizer'] = expand_optimizer(source['optimizer'], actor,
        dict(zip(names, ids[:len(names)])), config)
    out['actor'] = expand_actor(actor, config)
    out['source_training_recipe'] = out.pop('training_recipe', None)
    out['source_recipe_version'] = out.pop('recipe_version', None)
    out.update(initialization_kind=WIDE_INITIALIZATION if width==1024 else LARGER_INITIALIZATION,
               network_architecture=wide_network(width),
               actor_widening=config, source_checkpoint_sha256=source_checkpoint_sha256,
               online_updates_after_expansion=0, runtime_recipe_ready=False,
               capability_certified=False)
    validate_wide_candidate(out)
    return out


def validate_wide_candidate(data):
    from .motion_mcu_privileged import VERSION as critic_version
    kind=data.get('initialization_kind')
    if kind not in (WIDE_INITIALIZATION,LARGER_INITIALIZATION):
        raise ValueError('explicit untrained wide Motion initialization required')
    width=1024 if kind==WIDE_INITIALIZATION else 2048
    if (data.get('network_architecture') != wide_network(width) or
            data.get('actor_widening') != wide_declaration(width) or
            data.get('critic_privileged_version') != critic_version or
            data.get('runtime_recipe_ready') is not False or
            data.get('online_updates_after_expansion') != 0):
        raise ValueError('explicit untrained wide Motion initialization required')
    for key in ('source_checkpoint_sha256', 'spec_sha256', 'asset_sha256'):
        value = data.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('wide candidate source identity missing: ' + key)
    _validate(data['actor'], tuple(wide_declaration(width)['target_dims']))
    _validate_critic(data['critic'], (312,512,512,256,1))
    return True
