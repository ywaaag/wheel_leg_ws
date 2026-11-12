"""One shared mechanics ensemble; task-conditioned value is a separate network."""
from dataclasses import replace
import torch
from torch import nn
from .geometry import integrate, quat_matrix


class Standardizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))
        self.register_buffer('fitted', torch.tensor(False))

    @torch.no_grad()
    def fit(self, train_only):
        if len(train_only) < 2 or not torch.isfinite(train_only).all():
            raise ValueError('finite train-episode statistics required')
        self.mean.copy_(train_only.mean(0))
        self.scale.copy_(train_only.std(0).clamp_min(.01))
        self.fitted.fill_(True)

    def forward(self, x):
        return (x-self.mean)/self.scale


class ResidualMLP(nn.Module):
    """Four hidden residual layers in two blocks; LayerNorm controls rollout scales."""
    def __init__(self, inputs, outputs, width=1024, blocks=2):
        super().__init__()
        self.input = nn.Linear(inputs, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.ELU(),
            nn.Linear(width, width), nn.ELU(), nn.Linear(width, width)) for _ in range(blocks)])
        self.output = nn.Linear(width, outputs)

    def forward(self, x):
        x = self.input(x)
        for block in self.blocks:
            x = x+block(x)*.5
        return self.output(torch.nn.functional.elu(x))


CONTACT_FORCE_VERSION = 'contact_force_v1'
CONTACT_FORCE_DISABLED = 'none'
# Frozen from the audited train label statistics and measured gradient magnitudes.
# lambda: the force term equals20% of the base gradient norm at the MEDIAN of8
# independent80ms train batches (median ||g_base||/||g_force,lambda=1|| =0.5475).
# scale: 10N ~ one tenth of the steady support force (train median97N), so a few-N
# residual stays quadratic while the measured85-110N false-support regime is linear.
# Evidence: artifacts/2026_10_28__world_model_contact_force_supervision/candidate_recipe.json
CONTACT_FORCE_FROZEN = dict(version=CONTACT_FORCE_VERSION, scale_n=10.0, lambda_force=.11,
                            huber_delta=1.0, force_dims=2)


def contact_force_config(declaration):
    """Explicit, dimension-free dispatch of the optional physical-force loss.

    Absent/None/'none' keeps the historical supervision bit-for-bit. A declared
    version must match this module; unknown versions or invalid scales are errors,
    never silently downgraded.
    """
    if declaration is None:
        return None
    if not isinstance(declaration, dict):
        raise ValueError('contact force supervision declaration must be a mapping')
    version = declaration.get('version')
    if version == CONTACT_FORCE_DISABLED:
        return None
    if version != CONTACT_FORCE_VERSION:
        raise ValueError('unknown contact force supervision version: '+repr(version))
    scale = declaration.get('scale_n', CONTACT_FORCE_FROZEN['scale_n'])
    weight = declaration.get('lambda_force', CONTACT_FORCE_FROZEN['lambda_force'])
    delta = declaration.get('huber_delta', CONTACT_FORCE_FROZEN['huber_delta'])
    dims = declaration.get('force_dims', CONTACT_FORCE_FROZEN['force_dims'])
    if type(scale) not in (int, float) or not (scale >= 1. and scale < 1e6):
        raise ValueError('contact force scale must be a finite positive newton scale >=1N')
    if type(weight) not in (int, float) or not (0. <= weight < 1e3):
        raise ValueError('contact force weight must be a finite scalar in [0,1e3)')
    if weight == 0.:
        # Explicit `lambda_force=0` is the documented way to prove the term is inert:
        # it returns the historical objective bit-for-bit instead of a scaled copy.
        return None
    if type(delta) not in (int, float) or not (0. < delta <= 1e3):
        raise ValueError('contact force huber delta must be positive')
    if dims != 2:
        raise ValueError('first version supervises only the two wheel contact forces')
    return dict(version=CONTACT_FORCE_VERSION, scale_n=float(scale), lambda_force=float(weight),
                huber_delta=float(delta), force_dims=2)


def wheel_force(values, nv):
    """Physical wheel normal force (newtons) implied by the two wheel head rows."""
    return torch.nn.functional.softplus(values[..., 2*nv:2*nv+2])


def force_supervision(values, real, nv, config):
    """Huber in physical force space; returns (loss, diagnostics) with detached scalars."""
    force = wheel_force(values, nv)
    normalized = (force-real)/config['scale_n']
    loss = torch.nn.functional.smooth_l1_loss(normalized, torch.zeros_like(normalized),
                                              beta=config['huber_delta'])
    with torch.no_grad():
        error = (force-real).abs()
        support = force > 2
        truth = real > 2
        # Detached tensors only: the caller averages over substeps and syncs once.
        detail = dict(contact_force_huber=loss.detach(),
                      contact_force_mae_N=error.mean(),
                      contact_force_false_support=(support & ~truth).float().mean(),
                      contact_force_missed_support=(~support & truth).float().mean(),
                      contact_force_real_support_fraction=truth.float().mean())
    return loss, detail


def mechanical_features(q, v, warm, applied, context):
    # Root XY translation excluded; SO(3) matrix removes quaternion double cover.
    return torch.cat((q[:, 2:3], quat_matrix(q[:, 3:7]).flatten(1), q[:, 7:],
                      v, warm, applied, context), -1)


def mechanics_targets(next_v,v,next_warm,contacts,body_contact=None):
    # Supervise the PRE-softplus head, avoiding an artificial0.693N support floor.
    if body_contact is None:body_contact=contacts.new_zeros(len(contacts),1)
    force=torch.cat((contacts,body_contact),-1).clamp_min(1e-6)
    inverse_softplus=force+torch.log(-torch.expm1(-force))
    return torch.cat((next_v-v,next_warm,inverse_softplus),-1)


class Ensemble(nn.Module):
    def __init__(self, nq, nv, context_dim, *, members=3, width=1024, blocks=2):
        super().__init__()
        self.config = dict(nq=nq, nv=nv, context_dim=context_dim, members=members, width=width, blocks=blocks)
        inputs = nq-7+10+nv*2+6+context_dim
        self.normalizer = Standardizer(inputs)
        # Predict velocity increment, next solver warmstart, and wheel contact force.
        self.target = Standardizer(2*nv+3)
        self.members = nn.ModuleList([ResidualMLP(inputs, 2*nv+3, width, blocks) for _ in range(members)])
        self.nv = nv

    def features(self, s, applied):
        return mechanical_features(s.q, s.v, s.warm, applied, s.context)

    def raw(self, features, member):
        return self.members[member](self.normalizer(features))

    def predict(self, s, applied, member=0):
        raw = self.raw(self.features(s, applied), member)
        values = raw*self.target.scale+self.target.mean
        dv = values[:, :self.nv]
        warm = values[:, self.nv:2*self.nv]
        # Nonnegative support forces. Supervision uses normalized inverse-softplus
        # targets; a separate loss on physical force magnitude is not implemented.
        contacts = torch.nn.functional.softplus(values[:, 2*self.nv:])
        v = s.v+dv
        return replace(s, q=integrate(s.q, v), v=v, solver_q=s.q,solver_v=s.v,
                       warm=warm, contact=contacts[:,:2],body_contact=contacts[:,2:])

    def disagreement(self, s, applied):
        outputs = torch.stack([self.raw(self.features(s, applied), k)[:, :self.nv]
                               for k in range(len(self.members))])
        # Dimensionless disagreement; NOT a calibrated probability.
        return outputs.var(0, unbiased=False).mean(-1)


def value_features(s):
    # Includes every execution bank, clock, numerical state and reward/terminal memory.
    # Owner is deliberately omitted: it does not change dynamics or target return.
    names = ('v', 'solver_q', 'solver_v', 'warm', 'contact', 'body_contact', 'sensors', 'queue', 'pending', 'adopted', 'driver', 'motor',
             'pd_request', 'pd_limited', 'compensation', 'spring', 'applied', 'generalized', 'delay', 'tau',
             'pd_age', 'actor_age', 'spring_scale', 'actor_bias', 'pd_bias', 'driver_cap',
             'last_target', 'older_target', 'last_torque', 'history_valid', 'episode_step',
             'quality_ticks', 'quality_age', 'quality_sent', 'context', 'task', 'target', 'jump_kind')
    pieces = [s.q[:, 2:3], quat_matrix(s.q[:, 3:7]).flatten(1), s.q[:, 7:],
              (s.tick % 5).to(s.q.dtype)[:, None], (s.tick % 10).to(s.q.dtype)[:, None]]
    pieces += [getattr(s, k).to(s.q.dtype).reshape(len(s.q), -1) for k in names]
    return torch.cat(pieces, -1)


class Critic(nn.Module):
    def __init__(self, inputs, width=512):
        super().__init__()
        self.config = dict(inputs=inputs, width=width)
        self.normalizer = Standardizer(inputs)
        self.mlp = nn.Sequential(nn.Linear(inputs, width), nn.LayerNorm(width), nn.ELU(),
            nn.Linear(width, width), nn.ELU(), nn.Linear(width, 256 if width >= 256 else width),
            nn.ELU(), nn.Linear(256 if width >= 256 else width, 1))

    def forward(self, state):
        return self.mlp(self.normalizer(value_features(state))).squeeze(-1)


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())
