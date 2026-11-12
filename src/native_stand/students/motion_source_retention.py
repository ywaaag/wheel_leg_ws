"""Opt-in old-command policy anchoring; no simulator or reference-data IO."""
import copy
import math

import torch

VERSION = 'motion31_source_policy_retention_v1'
NOMINAL_VERSION = 'motion31_mcu_nominal_source_retention_v2'
KEY = 'motion_source_retention_cohort'
OLD_COMMANDS = [[0., 0.], [.5, 0.], [1., 0.], [1.5, 0.], [2., 0.]]


def declaration(source_sha256, coefficient=.01):
    result = {'version': VERSION, 'source_sha256': source_sha256,
              'coefficient': coefficient, 'commands': copy.deepcopy(OLD_COMMANDS),
              'loss': 'cohort-balanced mean channel KL(teacher||student), equal frozen std',
              'inputs': 'current on-policy old-command states; not frozen reference replay',
              'cohort_identity': 'episode target, never instantaneous ramp command',
              'teacher': 'frozen source Actor; no Critic, optimizer, or runtime control',
              'warmup': 'skip when Actor updates are disabled',
              'scope': 'old commands only; new yaw and initialization excluded'}
    validate(result, source_sha256)
    return result


def validate(config, source_sha256):
    if isinstance(config,dict) and config.get('version')=='motion31_mcu9995_nominal_retention_v5':
        from .motion_highspeed_standing import retention
        if config!=retention(source_sha256):raise ValueError('9995 nominal retention differs')
        return config
    if isinstance(config,dict) and config.get('version')=='motion31_mcu4500_height_scoped_retention_v4':
        from .motion_height_pd import retention
        if config!=retention(source_sha256):raise ValueError('height-scoped4500 retention differs')
        return config
    if isinstance(config,dict) and config.get('version')=='motion31_mcu4500_nominal_source_retention_v3':
        from .motion_handover_robustness import retention
        if config!=retention(source_sha256):raise ValueError('4500 nominal retention differs')
        return config
    if isinstance(config,dict) and config.get('version')==NOMINAL_VERSION:
        expected=nominal_declaration(source_sha256)
        if config!=expected:raise ValueError('explicit MCU nominal retention declaration differs')
        return config
    if not isinstance(config, dict) or config.get('version') != VERSION:
        raise ValueError('explicit Motion source retention version required')
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in source_sha256)
            or config.get('source_sha256') != source_sha256):
        raise ValueError('source retention checkpoint identity differs')
    coefficient = config.get('coefficient')
    if (type(coefficient) not in (float, int) or not math.isfinite(coefficient)
            or not 0 < coefficient <= 1.):
        raise ValueError('source retention coefficient must be finite in (0,1]')
    expected = {'version', 'source_sha256', 'coefficient', 'commands', 'loss',
                'inputs', 'cohort_identity', 'teacher', 'warmup', 'scope'}
    if set(config) != expected or config['commands'] != OLD_COMMANDS:
        raise ValueError('source retention fields/old commands differ')
    # Semantic strings are part of the versioned contract, not free-form knobs.
    canonical = {
        'loss': 'cohort-balanced mean channel KL(teacher||student), equal frozen std',
        'inputs': 'current on-policy old-command states; not frozen reference replay',
        'cohort_identity': 'episode target, never instantaneous ramp command',
        'teacher': 'frozen source Actor; no Critic, optimizer, or runtime control',
        'warmup': 'skip when Actor updates are disabled',
        'scope': 'old commands only; new yaw and initialization excluded'}
    if any(config[k] != v for k, v in canonical.items()):
        raise ValueError('source retention semantics differ')
    return config


def nominal_declaration(source_sha256):
    from .motion_posture_design import SOURCE_SHA,commands
    if source_sha256!=SOURCE_SHA:raise ValueError('posture teacher must be original1500')
    result=declaration(source_sha256)
    result.update(version=NOMINAL_VERSION,commands=commands(),eligible_mcu_variant=0,
                  scope='source-qualified nominal zero/forward/backward/yaw only; stress/takeover excluded')
    return result


def cohort_ids(targets, valid, config):
    """Metadata stays with each observation through storage/minibatch shuffle."""
    if targets.ndim != 2 or targets.shape[1] != 2 or valid.shape != targets.shape[:1] or valid.dtype != torch.bool:
        raise ValueError('source retention episode-target/mask shape differs')
    result = torch.full_like(valid, -1, dtype=torch.long)
    for i, command in enumerate(config['commands']):
        result = torch.where(valid & (targets == targets.new_tensor(command)).all(-1), i, result)
    return result[:, None]


def verify_source_records(records, config):
    """Caller verifies raw/spec/SHA; this pure boundary checks certified scope."""
    available = {tuple(r['case']['command']): r for r in records}
    for command in config['commands']:
        rows = available.get(tuple(command + [.34]), {}).get('per_env', [])
        ids = {r.get('env', r.get('env_id')) for r in rows}
        if len(rows) != 8 or ids != set(range(8)) or not all(r.get('accepted') is True for r in rows):
            raise ValueError('source retention requires existing8/8 proof for each old command')
    return True


def add_metadata(observations, targets, config, *, variants=None):
    if config is None:
        return observations
    from native_stand.recovery.controlled_ppo import KEY as VALID
    valid=observations[VALID].squeeze(-1)
    if config.get('version') in (NOMINAL_VERSION,'motion31_mcu4500_nominal_source_retention_v3','motion31_mcu4500_height_scoped_retention_v4','motion31_mcu9995_nominal_retention_v5'):
        if variants is None or variants.shape!=valid.shape or variants.dtype!=torch.long:
            raise ValueError('nominal retention requires actual pre-action MCU variant')
        valid=valid & (variants==config['eligible_mcu_variant'])
    if config.get('version') in ('motion31_mcu4500_height_scoped_retention_v4','motion31_mcu9995_nominal_retention_v5'):
        valid=valid & torch.isclose(observations['student_actor'][:,24],valid.new_full(valid.shape,.68,dtype=torch.float32),atol=1e-6,rtol=0)
    observations[KEY] = cohort_ids(targets, valid, config)
    return observations


class SourcePolicyRetention:
    def __init__(self, actor, config):
        validate(config, config.get('source_sha256'))
        if actor.interface_version != 'multiskill_mcu_v2' or not actor.is_actor or actor.log_std.requires_grad:
            raise ValueError('source retention requires MCU Actor with frozen std')
        self.cfg = copy.deepcopy(config)
        self.teacher = copy.deepcopy(actor).eval().requires_grad_(False)
        self.std = actor.log_std.detach().exp().clone()
        self.reset()

    def reset(self):
        device = self.std.device
        self.uses = torch.zeros(len(self.cfg['commands']), device=device, dtype=torch.long)
        self.channel_kl = torch.zeros(6, device=device)
        self.physical_error = torch.zeros(6, device=device)
        self.loss_sum = torch.zeros((), device=device)
        self.calls = 0

    def loss(self, actor, observations, valid, *, student_mean=None):
        if KEY not in observations:
            raise ValueError('source retention pre-action cohort metadata missing')
        x = observations['student_actor']
        ids = observations[KEY].squeeze(-1)
        if x.shape != (len(ids), 31) or ids.dtype != torch.long or valid.shape != ids.shape:
            raise ValueError('source retention storage metadata differs')
        count=len(self.cfg['commands'])
        selected = valid & (ids >= 0) & (ids < count)
        # No stochastic forward: preserve PPO distribution cache and sampling RNG.
        mean = actor.mean(x) if student_mean is None else student_mean
        if mean.shape != (len(ids), 6):
            raise ValueError('source retention student mean shape differs')
        with torch.no_grad():
            reference = self.teacher.mean(x)
        kl = .5 * ((mean-reference)/self.std).square()
        values = torch.where(selected, kl.mean(-1), 0.)
        safe_ids = ids.clamp(0, count-1)
        counts = torch.zeros(count, device=x.device, dtype=torch.long).scatter_add_(0, safe_ids, selected.long())
        sums = x.new_zeros(count).scatter_add_(0, safe_ids, values)
        result = (sums / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)
        with torch.no_grad():
            self.uses += counts
            self.channel_kl += torch.where(selected[:, None], kl, 0.).sum(0)
            scales = x.new_tensor([1.5, 1.5, 100., 1.5, 1.5, 100.])
            self.physical_error += torch.where(selected[:, None], (mean-reference).abs()*scales, 0.).sum(0)
            self.loss_sum += result.detach()
            self.calls += 1
        return self.cfg['coefficient'] * result

    def summary(self, eligible):
        uses = self.uses.sum()
        denominator = uses.clamp_min(1)
        return {'version': self.cfg['version'], 'coefficient': self.cfg['coefficient'],
                'source_sha256': self.cfg['source_sha256'],
                'eligible_rollout_transitions': eligible,
                'selected_sample_uses_over_epochs': int(uses),
                'selected_by_cohort_over_epochs': self.uses.cpu().tolist(),
                'loss_mean': float(self.loss_sum / max(self.calls, 1)),
                'weighted_loss_mean': float(self.loss_sum / max(self.calls, 1)) * self.cfg['coefficient'],
                'channel_KL_mean': (self.channel_kl/denominator).cpu().tolist(),
                'physical_channel_MAE_rad_or_rad_s': (self.physical_error/denominator).cpu().tolist(),
                'actor_minibatches': self.calls,
                'status': 'COLLECTED' if self.calls else 'SKIPPED_ACTOR_UPDATE',
                'state_coverage': 'current old-command states; source-state replay NOT_COLLECTED'}


def ppo_class():
    """Import installed PPO only at assembly; old default class is untouched."""
    from native_stand.recovery.controlled_ppo import ControlledPPO, KEY as VALID

    class RetainedMotionPPO(ControlledPPO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if KEY not in self.storage.observations:
                raise ValueError('source retention storage metadata missing')
            # Installed RolloutStorage initially allocates float for all extras.
            # Preserve the versioned categorical metadata, as VALID preserves bool.
            self.storage.observations[KEY] = torch.full_like(self.storage.observations[KEY], -1, dtype=torch.long)

        def actor_auxiliary_loss(self, batch, valid):
            return self.source_retention.loss(self.actor, batch.observations, valid, student_mean=self.actor.output_mean)

        def update(self):
            self.source_retention.reset()
            valid = self.storage.observations[VALID].squeeze(-1)
            ids = self.storage.observations[KEY].squeeze(-1)
            eligible = int((valid & (ids >= 0) & (ids < len(self.source_retention.cfg['commands']))).sum())
            result = super().update()
            self.mask_summary['source_policy_retention'] = self.source_retention.summary(eligible)
            return result

    return RetainedMotionPPO
