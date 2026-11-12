"""Explicit Recovery31/64 function-preserving initialization; no runtime IO."""
from .actor_widening import expand_actor, declaration, expand_value_network

VERSION='recovery31_function_preserving_widening_v1'
NETWORK_VERSION='recovery31_large_actor_critic_v1'
SOURCE_SHA='66243a253e7f1d595d2950f5686ec6f715fd6d79edd481c52bb34abcb36a4d3a'


def network():
    return {'version':NETWORK_VERSION,'actor_hidden_dims':[512,256,128],
            'critic_hidden_dims':[512,512,256],'actor_input':31,'critic_input':64,
            'hidden_activation':'ELU(alpha=1)','mean_output':'Tanh'}


def expand(actor,critic):
    # Actor coordinates match the existing explicit MCU network; Critic64
    # retains Recovery semantics, never Motion49 or a guessed input layout.
    return (expand_actor(actor,declaration()),
            expand_value_network(critic,(64,256,128,64,1),(64,512,512,256,1)))


def validate_candidate(data):
    if (data.get('initialization_kind')!=VERSION or
        data.get('interface_version')!='multiskill_mcu_v2' or
        data.get('role')!='RecoveryStudent' or data.get('network_architecture')!=network() or
        data.get('critic_privileged_version')!='recovery_privileged64_v1' or
        data.get('source_sha256')!=SOURCE_SHA):
        raise ValueError('Recovery widened initialization identity differs')
    for key in ('asset_sha256','source_spec_sha256','dataset_sha256','manifest_sha256'):
        if not isinstance(data.get(key),str) or len(data[key])!=64:
            raise ValueError('Recovery initialization missing '+key)
    if any(k in data for k in ('recipe_version','training_recipe','optimizer','training_rng')):
        raise ValueError('initialization must not impersonate trained large checkpoint/resume')
    if data.get('accepted') is not False or data.get('online_updates_this_stage')!=0:
        raise ValueError('widened initialization has no new physical capability certificate')
    return data
