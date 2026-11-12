"""Actual analytical policy gradients, real replay TD, and supervised mechanics.

No PPO rollout, old log probability, GAE or clipped PPO objective exists here.
"""
from contextlib import contextmanager
from dataclasses import replace
import copy
import torch
from . import execution


@contextmanager
def frozen(*modules):
    saved = [(p, p.requires_grad) for m in modules for p in m.parameters()]
    try:
        for p, _ in saved:
            p.requires_grad_(False); p.grad = None
        yield
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)


def bootstrap_target(reward, terminated, truncated, next_value, gamma=.99):
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError('explicit terminated/truncated masks required')
    # True termination takes precedence; timeout uses PRE-RESET next state.
    return reward+gamma*(~terminated).to(reward.dtype)*next_value


def macro(s, action, wm, geometry, reward, *, member=0):
    before = s
    requested = execution.decode(action, geometry, s)
    disagreements, validities = [], []
    for _ in range(10):
        m = execution.measurements(s,geometry)
        e, ok = execution.step(s, requested, m['sensors'])
        e=replace(e,generalized=geometry.effective_torque(e.applied))
        disagreements.append(wm.disagreement(s, e.generalized))
        s = wm.predict(e, e.generalized, member)
        m = geometry(s.q, s.v)
        validities.append(ok & geometry.valid(s.q, s.v, m) & geometry.contact_valid(s))
    m=execution.measurements(s,geometry)
    s, terms, terminated = reward(before, s, m)
    terms['invalid_substep_fraction']=(~torch.stack(validities)).to(s.q.dtype).mean(0)
    return s, terms, terminated, torch.stack(validities).all(0), torch.stack(disagreements).mean(0)


def critic_step(critic, target, optimizer, before, after, reward, terminated, truncated, *, gamma=.99,before_optimizer=None):
    # REAL replay only. Both terminal masks and post-physics/pre-reset state mandatory.
    with torch.no_grad():
        y = bootstrap_target(reward, terminated, truncated, target(after), gamma)
    prediction = critic(before)
    loss = torch.nn.functional.smooth_l1_loss(prediction, y)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite real TD critic loss')
    optimizer.zero_grad(); loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 10., error_if_nonfinite=True)
    if before_optimizer is not None:before_optimizer()
    optimizer.step()
    with torch.no_grad():
        for a, b in zip(target.parameters(), critic.parameters()):
            a.lerp_(b, .01)
    return dict(loss=float(loss.detach()), grad_norm=float(norm), real_transitions=len(before.q),
                terminal_count=int(terminated.sum()), timeout_count=int(truncated.sum()))


def actor_step(actor, teacher, wm, critic, geometry, reward, starts, optimizer, *,
               horizon=8, gamma=.99, member=0, retention=.01, uncertainty=.01,
               max_normalized_action_change=.1, before_optimizer=None, retention_observations=None):
    if horizon not in (8, 16, 32):
        raise ValueError('explicit 100Hz horizon must be8/16/32')
    starts = starts.detach()
    initial_obs = execution.observation(starts, execution.measurements(starts,geometry))
    with torch.no_grad():
        old_mean = actor.mean(initial_obs).clone()
        source_mean = teacher.mean(initial_obs)
    saved_parameters = copy.deepcopy(actor.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad()
    with frozen(wm, critic, teacher):
        s = starts
        objective = starts.q.new_zeros(len(starts.q))
        alive = torch.ones(len(starts.q), device=starts.q.device, dtype=torch.bool)
        valid = alive.clone()
        for k in range(horizon):
            obs = execution.observation(s, execution.measurements(s,geometry))
            action = actor.mean(obs)             # graph is deliberately NOT detached
            s, terms, done, ok, disagreement = macro(s, action, wm, geometry, reward, member=member)
            objective = objective+(gamma**k)*(terms['total']-uncertainty*disagreement)*alive
            valid &= ok
            alive = alive & ~done
        objective = objective+(gamma**horizon)*critic(s)*alive
        # Nominal source actions only; PD-age/spring uncertainty excluded.
        nominal = (starts.pd_age == 10) & (starts.spring_scale[:, 0] == 1)
        delta = (actor.mean(initial_obs)-source_mean)/teacher.log_std.detach().exp()
        kl = .5*delta.square().mean(-1)
        retention_loss = (kl*nominal).sum()/nominal.sum().clamp_min(1)
        if retention_observations is not None:
            # Original nominal command replay protects high-speed skills not yet
            # sampled in the low-speed mechanics pilot. Teacher only labels.
            replay_obs=retention_observations.to(starts.q.device)
            replay_source=teacher.mean(replay_obs).detach()
            replay_kl=.5*((actor.mean(replay_obs)-replay_source)/teacher.log_std.detach().exp()).square().mean()
            retention_loss=(retention_loss+replay_kl)*.5
        loss = -objective.mean()+retention*retention_loss
        if not bool(valid.all()):
            return dict(updated=False, reason='invalid imagined mechanical/closure/contact state',
                        invalid_fraction=float((~valid).float().mean()), actor_optimizer_steps=0)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite BPTT objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],
                                             1., error_if_nonfinite=True)
        if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in actor.parameters()):
            raise RuntimeError('BPTT graph disconnected from Actor')
        if before_optimizer is not None: before_optimizer()
        optimizer.step()
    with torch.no_grad():
        change = ((actor.mean(initial_obs)-old_mean)/teacher.log_std.exp()).square().mean().sqrt()
    # A bounded update; optimizer state is also rolled back. The attempted call is charged.
    accepted = float(change) <= max_normalized_action_change
    if not accepted:
        actor.load_state_dict(saved_parameters); optimizer.load_state_dict(saved_optimizer)
    return dict(updated=accepted, reason=None if accepted else 'source-scale policy step exceeded bound',
        loss=float(loss.detach()), imagined_return=float(objective.detach().mean()),
        retention_kl=float(retention_loss.detach()), grad_norm=float(norm),
        source_scaled_action_change=float(change), horizon=horizon, horizon_s=horizon*.01,
        actor_optimizer_steps=1, accepted_optimizer_steps=int(accepted),
        invalid_fraction=float((~valid).float().mean()), gradient_path='Actor -> execution -> mechanics -> reward/terminal Critic -> Actor')
