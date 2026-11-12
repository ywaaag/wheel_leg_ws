"""CPU planning and snapshot checks for a future independent Motion evaluator.

No simulator, subprocess, optimizer, filesystem watcher or training mutation.
The active synchronous runner does not import this module.
"""
import copy
import hashlib
import json
import math

VERSION = 'motion_snapshot_evaluation_design_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')


def plan(cases, *, capacity=256, scored_envs=8, float_equivalents=128,
         trace_limit_bytes=512 * 2**20):
    """Keep every case/window, estimate a compact one-batch scoring buffer.

    float_equivalents is an explicit planning allowance, not a measured schema.
    Physical readback, per-case random streams and scorer parity remain required.
    """
    from .grouped_evaluation import plan as legacy_plan
    if not cases:
        raise ValueError('empty evaluation matrix')
    for name, value in (('capacity', capacity), ('float_equivalents', float_equivalents),
                        ('trace_limit_bytes', trace_limit_bytes)):
        _positive_integer(value, name)
    if scored_envs != 8 or type(scored_envs) is not int:
        raise ValueError('this design preserves eight scored environments per case')
    if capacity < scored_envs or capacity > 256:
        raise ValueError('capacity must be 8..256')
    old = legacy_plan(cases, maximum_commands=8)
    # Legacy validation and compatibility keys are preserved. Different seeds,
    # episode lengths or construction settings cannot silently share a reset.
    groups = []
    for row in cases:
        compatible = bool(groups) and len(legacy_plan([groups[-1][-1], row],
                                                      maximum_commands=8)) == 1
        if not compatible or (len(groups[-1]) + 1) * scored_envs > capacity:
            groups.append([])
        groups[-1].append(copy.deepcopy(row))
    batches = []
    for group in groups:
        samples = math.ceil(max(r['seconds'] for r in group) / .01)
        scored = len(group) * scored_envs
        estimated_bytes = samples * scored * float_equivalents * 4
        if estimated_bytes > trace_limit_bytes:
            raise ValueError('planned score buffer exceeds byte limit')
        batches.append(dict(case_ids=[r['case_id'] for r in group],
                            actual_envs=capacity, scored_envs=scored,
                            policy_steps=samples,
                            env_physics_steps=capacity * samples * 10,
                            estimated_score_buffer_bytes=estimated_bytes))
    return dict(version=VERSION, runtime_recipe_ready=False, simulation=0,
                online_PPO=0, case_count=len(cases), scored_episodes=len(cases)*8,
                legacy_batch_count=len(old), proposed_batch_count=len(groups),
                batches=batches, float_equivalents_per_sample=float_equivalents,
                trace_limit_bytes=trace_limit_bytes,
                wall_time='UNKNOWN; batching does not eliminate physical work',
                missing_runtime_validation=[
                    'explicit version dispatch and real evaluator worker',
                    'compact capture schema measured before allocation',
                    'per-case seed/local-env random streams and reset parity',
                    'same checkpoint old/new full scorer and physical readback',
                    'concurrent GPU memory and end-to-end throughput',
                    'public owner/ledger submission and result delivery'])


def snapshot(cases, *, checkpoint, checkpoint_sha256, spec_sha256,
             course_revision, update):
    """Bind a saved model and its evaluation matrix without touching live PPO."""
    _positive_integer(update, 'checkpoint update')
    if type(course_revision) is not int or course_revision < 0:
        raise ValueError('invalid course revision')
    if not isinstance(checkpoint, str) or not checkpoint.startswith('/'):
        raise ValueError('absolute saved checkpoint path required')
    for value in (checkpoint_sha256, spec_sha256):
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('full lowercase SHA256 required')
    plan(cases)
    keys = [copy.deepcopy(r['_identity_key']) for r in cases]
    if len({digest(k) for k in keys}) != len(keys):
        raise ValueError('duplicate assessment identity')
    result = dict(version=VERSION, checkpoint=checkpoint,
                  checkpoint_sha256=checkpoint_sha256, spec_sha256=spec_sha256,
                  course_revision=course_revision, update=update,
                  case_ids=[r['case_id'] for r in cases], assessment_keys=keys)
    return {**result, 'snapshot_sha256':digest(result)}


def result_decision(request, result, *, current_revision, current_update,
                    max_lag_updates=100):
    """Late evidence remains attributable to its model; never certify latest.

    This only validates an envelope. Existing course advancement must still
    consume actual per-environment gates; this function never advances a course.
    """
    _positive_integer(max_lag_updates, 'maximum result lag')
    expected = {k:v for k,v in request.items() if k != 'snapshot_sha256'}
    errors = []
    if digest(expected) != request.get('snapshot_sha256'):
        errors.append('request_snapshot_changed')
    for key in ('snapshot_sha256', 'checkpoint_sha256', 'spec_sha256', 'course_revision', 'update'):
        if result.get(key) != request.get(key):
            errors.append('result_identity_' + key)
    if result.get('complete_matrix') is not True:
        errors.append('incomplete_matrix')
    for key in ('online_PPO', 'offline_optimizer'):
        if type(result.get(key)) is not int or result[key] != 0:
            errors.append('evaluation_must_have_zero_' + key)
    records = result.get('records', [])
    if not isinstance(records, list):
        records = []
    if [r.get('case', {}).get('case_id') for r in records] != request['case_ids']:
        errors.append('case_order_or_coverage_differs')
    if [r.get('case', {}).get('_identity_key') for r in records] != request['assessment_keys']:
        errors.append('assessment_keys_differ')
    for record in records:
        case = record.get('case', {})
        if (case.get('checkpoint') != request['checkpoint']
                or case.get('sha256') != request['checkpoint_sha256']):
            errors.append('case_checkpoint_differs')
            break
        envs = record.get('per_env', [])
        if (len(envs) != 8 or {r.get('env') for r in envs} != set(range(8))
                or any(type(r.get('env')) is not int for r in envs)
                or any(type(r.get('accepted')) is not bool for r in envs)):
            errors.append('missing_eight_independent_env_results')
            break
    lag = current_update - request['update']
    stale = current_revision != request['course_revision'] or not 0 <= lag <= max_lag_updates
    return dict(evidence_valid=not errors, errors=errors, stale_for_curriculum=stale,
                eligible_for_existing_course_checks=not errors and not stale,
                checkpoint_sha256=request['checkpoint_sha256'],
                checkpoint_update=request['update'], result_lag_updates=lag,
                certifies_current_weights=False, course_advanced=False)
