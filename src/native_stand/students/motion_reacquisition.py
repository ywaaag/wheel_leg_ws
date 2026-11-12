"""Explicit posture-preserving Motion curriculum after confirmed low-inertia standing.

New recipe only. The actor/control/reward interfaces and existing recipes stay
unchanged. File checks belong to build/verify; course and scoring take data.
"""
import copy
import json
import math
from pathlib import Path

import torch

from .data import sha

VERSION = 'motion31_posture_reacquisition_v1'
ASSESSMENT = 'motion31_posture_reacquisition_assessment_v1'
TARGETS = (.5, 1., 1.5, 2., 3., 4.)
SOURCE_COURSE = 'motion31_standing_antiosc_armature_v4'
STORAGE_BYTES = 13 * 2**30


def configuration():
    return {'version': VERSION, 'targets': list(TARGETS), 'envs': 4096,
            'zero_fraction': .25, 'old_fraction_after_first_stage': .25,
            'current_fraction_after_first_stage': .5,
            'command_modes': ['forward', 'backward', 'in_place_yaw_positive', 'in_place_yaw_negative'],
            'leg_armature_kg_m2': .012, 'fixed_timing_level': 4,
            'formal_updates_max': 9995, 'smoke_updates_max': 5, 'stage_updates_max': 9995,
            'evaluation_every': 250, 'checkpoint_every': 100, 'confirmations': 2,
            'preserve': 'all unlocked endpoints and zero in every complete evaluation',
            'source_capability': 'failure recorded; never a prerequisite for learning',
            'stop_policy': 'behavior regression/success record only; runtime/storage/budget/user stop remain',
            'scope': 'flat straight OR in-place yaw, no combined turn/terrain/recovery',
            'budget': 'one independent inclusive10000; no parent balance transfer, no stage counter reset',
            'storage_max_bytes': STORAGE_BYTES}


def commands(stage=0):
    if type(stage) is not int or stage not in range(len(TARGETS)):
        raise ValueError('invalid posture Motion course stage')
    return [[0., 0.]] + [v for t in TARGETS[:stage + 1]
                        for v in ([t, 0.], [-t, 0.], [0., t], [0., -t])]


def stage_config(stage=0, envs=4096):
    if envs not in (1, 64, 4096):
        raise ValueError('posture Motion course supports explicit1/64/4096 environments')
    values = commands(stage)
    counts = [0] * len(values)
    if envs == 1:
        counts[0] = 1  # Explicit evaluation/preview overwrites the target.
    else:
        counts[0] = envs // 4
        new_total = envs - counts[0] if stage == 0 else envs // 2
        if stage:
            # Allocate whole +/- pairs, so integer rounding cannot favor a sign.
            pair_total = (envs // 4) // 2
            q, r = divmod(pair_total, 2 * stage)
            for pair in range(2 * stage):
                counts[1 + pair * 2] = counts[2 + pair * 2] = q + (pair < r)
        counts[-4:] = [new_total // 4] * 4
    if sum(counts) != envs:
        raise ValueError('posture Motion fixed quota sum differs')
    return {'id': stage, 'name': f'posture_motion_{TARGETS[stage]:g}', 'commands': values,
            'weights': [n / envs for n in counts], 'sampling_mode': 'fixed_slots',
            'sampling_slots': [i for i, n in enumerate(counts) for _ in range(n)],
            'slot_counts': counts, 'enforce_whitelist': True}


def protocol():
    from .motion_yaw_course import protocol as base
    p = base()
    p.update(version=ASSESSMENT, height_mean_band_m=[.32, .36],
             reacquisition={'leg_length_mae_max_m': .01, 'leg_length_p95_max_m': .02,
                            'wheel_offset_mae_max_m': .01, 'wheel_offset_p95_max_m': .02,
                            'swing_full_strict_max_rad': .05, 'pitch_hold_peak_max_rad': .1,
                            'roll_hold_peak_max_rad': .05,
                            'height_mean_band_m': [.32, .36]},
             course_scope='retain zero and every unlocked +/-vx and +/-in-place yaw; physical/symmetry first')
    return p


def cases(spec, stage=0):
    template = spec['student_evaluation']['cases'][0]
    return [{**template, 'case_id': f'reacquire_{stage}_{i}', 'command': v + [.34],
             '_timing_level': 4} for i, v in enumerate(commands(stage))]


def effective_spec(spec, stage, state=None):
    s = copy.deepcopy(spec)
    s['motion_training']['stage'] = stage
    s['student_training'].update(stage=stage, course_stage=stage)
    # Command stage and physical inertia level are different concepts.
    s['motion_timing']['stage'] = 4
    s['curriculum']['stages'] = [stage_config(stage, s['ppo']['envs'])]
    s['student_evaluation']['cases'] = cases(s, stage)
    s['motion_timing']['configuration']['smooth_mix'] = (state or {}).get(
        'smooth_mix', s['motion_timing']['configuration']['smooth_mix'])
    s['student_training']['timing_configuration'] = copy.deepcopy(s['motion_timing']['configuration'])
    return s


def refinement_records(records, stage):
    expected = commands(stage)
    if (len(records) != len(expected) or any(
            r.get('case', {}).get('command') != target + [.34]
            or r['case'].get('_timing_level') != 4
            or len(r.get('per_env', [])) != 8
            or {e.get('env') for e in r['per_env']} != set(range(8))
            for r, target in zip(records, expected))):
        return [{'case': {'command': v + [.34]}, 'per_env': []} for v in expected]
    return records


def advance(state, candidate):
    s = copy.deepcopy(state)
    stage, update = s['stage'], candidate['update']
    if candidate.get('stage') != stage or update <= s.get('last_evaluation_update', -1):
        raise ValueError('posture Motion stage/update differs')
    records = refinement_records(candidate.get('evaluations', []), stage)
    complete = (candidate.get('complete_matrix') is True
                and all(len(r['per_env']) == 8 and all(e.get('accepted') is True for e in r['per_env'])
                        for r in records))
    # A duplicate parameter set or a skipped milestone cannot count twice.
    consecutive = (s.get('last_evaluation_update') == update - 250
                   and s.get('last_checkpoint_sha256') != candidate.get('sha256'))
    s['consecutive_passes'] = (s.get('consecutive_passes', 0) + 1 if consecutive else 1) if complete else 0
    s.update(last_evaluation_update=update, last_checkpoint_sha256=candidate['sha256'])
    s['smooth_mix'] = min(.75, s.get('smooth_mix', 0.) + .25) if complete else max(0., s.get('smooth_mix', 0.) - .25)
    passed = s['consecutive_passes'] >= 2
    if passed:
        s['accepted_by_stage'][str(stage)] = {k: candidate[k] for k in ('checkpoint', 'sha256', 'update')}
    advanced = passed and stage < len(TARGETS) - 1
    if advanced:
        s.update(stage=stage + 1, stage_started_update=update, consecutive_passes=0)
    return s, {'stage_passed': passed, 'advanced': advanced,
               'all_stages_passed': passed and not advanced,
               'complete_matrix_passed': complete, 'stop': False,
               'progression': 'two consecutive complete8-env matrices, all unlocked commands, same checkpoint per matrix'}


def choose(candidates, accepted_by_stage):
    from .motion_course import choose as base
    result = base(candidates, {})
    stage = result['current_stage']
    final = accepted_by_stage.get(str(len(TARGETS) - 1))
    result.update(accepted_by_stage=accepted_by_stage,
                  accepted_candidate=next((c for c in candidates if final and c.get('stage') == len(TARGETS) - 1
                                           and c['sha256'] == final['sha256']), None),
                  capability_scope=f'posture Motion stage{stage}; all unlocked commands, not multiseed/hardware certified')
    return result


from .motion_course import extend_mixture


def assess(result, arrays, env_id):
    """Extra posture/geometry gates; never shorten the required30s window."""
    import numpy as np
    from native_stand.recovery.standing import longitudinal_geometry
    mask = arrays['first_episode_valid'][:, env_id].astype(bool) & (arrays['controller'][:, env_id, 0] == 3)
    state = arrays['state'][mask, env_id][:3000]
    p = arrays['posture'][mask, env_id][:3000]
    g = arrays['geometry'][mask, env_id][:3000]
    complete = len(state) == 3000 and all(np.isfinite(x).all() for x in (state, p, g))
    names = ('reacquisition_height', 'reacquisition_leg_mae', 'reacquisition_leg_p95',
             'reacquisition_wheel_mae', 'reacquisition_wheel_p95')
    checks = dict.fromkeys(names, False)
    metrics = {'complete_30s': complete, 'holding_window_s': [10., 18.],
               'required_sample_rate_hz': 100., 'scope': 'same first controlled episode, no reselected window'}
    if complete:
        h = slice(1000, 1800)
        legs = np.abs(p[h, 2] - p[h, 3])
        wheel = longitudinal_geometry(torch.as_tensor(g[h, :6]).reshape(-1, 2, 3),
                                      torch.as_tensor(g[h, 6:9])).abs().numpy()
        if not np.isfinite(wheel).all():
            result['reacquisition_geometry'] = {**metrics, 'geometry_valid': False}
            result['checks'].update(checks)
            result.update(version=ASSESSMENT, accepted=False,
                          failed_checks=[k for k, v in result['checks'].items() if v is not True])
            return result
        metrics.update(height_mean_m=float(state[h, 6].mean()),
                       leg_length_mae_m=float(legs.mean()), leg_length_p95_m=float(np.quantile(legs, .95)),
                       wheel_offset_mae_m=float(wheel.mean()), wheel_offset_p95_m=float(np.quantile(wheel, .95)))
        checks.update(reacquisition_height=.32 <= metrics['height_mean_m'] <= .36,
                      reacquisition_leg_mae=metrics['leg_length_mae_m'] <= .01,
                      reacquisition_leg_p95=metrics['leg_length_p95_m'] <= .02,
                      reacquisition_wheel_mae=metrics['wheel_offset_mae_m'] <= .01,
                      reacquisition_wheel_p95=metrics['wheel_offset_p95_m'] <= .02)
    result['reacquisition_geometry'] = metrics
    result['checks'].update(checks)
    result.update(version=ASSESSMENT, accepted=all(v is True for v in result['checks'].values()),
                  failed_checks=[k for k, v in result['checks'].items() if v is not True])
    return result


def scoring_protocol(spec):
    from .motion_parallel_assessment import safety_protocol
    p = safety_protocol(spec)
    p.update(version=ASSESSMENT, reacquisition=protocol()['reacquisition'],
             anti_oscillation={'wheel_delta_rms_max_rad_s': 3., 'leg_delta_rms_max_rad': .03,
                               'wheel_request_p95_max_Nm': 5.5, 'holding_window_s': [10., 18.]})
    return p


def validate_recipe(recipe, source_sha):
    if (recipe.get('course_version') != VERSION or recipe.get('source_actor_sha256') != source_sha
            or type(recipe.get('stage')) is not int or recipe['stage'] not in range(len(TARGETS))
            or recipe['course_stage'] != recipe['stage']):
        raise ValueError('posture Motion explicit recipe/source/stage differs')
    from .motion_timing import validate_recipe as validate_parent
    r = copy.deepcopy(recipe)
    r.update(course_version=SOURCE_COURSE, stage=0, course_stage=0)
    validate_parent(r, source_sha)
    return recipe


def _declaration(old, saved, parent, checkpoint, experiment, spec_path, confirmation, owner_session):
    from .motion_wheel_matching import configuration as plant
    s = copy.deepcopy(old)
    source_sha = sha(checkpoint)
    s.update(experiment=experiment, source_checkpoint=str(checkpoint), source_sha256=source_sha)
    c = plant()
    c['smooth_mix'] = saved['training_recipe']['timing_configuration']['smooth_mix']
    s['motion_timing'].update(stage=4, configuration=c, parent_spec=str(parent),
                              parent_spec_sha256=sha(parent), source_checkpoint_sha256=source_sha)
    s['motion_reacquisition'] = {'version': VERSION, 'parent_spec': str(parent),
                                'parent_spec_sha256': sha(parent), 'configuration': configuration(),
                                'confirmation': copy.deepcopy(confirmation), 'physical_level': 4}
    s['student_training'].update(source_actor_sha256=source_sha, course_version=VERSION,
                                course_stage=0, stage=0, timing_configuration=copy.deepcopy(c),
                                critic='full_motion140_wheel_match', optimizer='source_Adam_full_state',
                                milestones=list(range(250, 10000, 250)), checkpoint_every_updates=100,
                                storage_max_bytes=STORAGE_BYTES,
                                selection_scope='horizontal/symmetric Motion under fixed .012 plant; all unlocked straight/yaw commands')
    s['motion_training'].update(stage=0, spec_path=str(spec_path), resume=None)
    for k in ('wiring_reuse', 'partial_smoke_recovery', 'prior_consumption', 'successor'):
        s['motion_training'].pop(k, None)
    s['motion_course'] = {'source_recipe_updates': saved['recipe_updates'], 'configuration': configuration()}
    s['motion_assessment'] = protocol()
    s['student_evaluation']['version'] = ASSESSMENT
    template = s['student_evaluation']['cases'][0]
    s['student_evaluation']['cases'] = [{**template, 'checkpoint': str(checkpoint), 'sha256': source_sha}]
    s['evaluation'].update(protocol_identity=ASSESSMENT, full_every_updates=250,
                           checkpoint_every_updates=100, run_final_durability=False)
    s['motion_startup'].update(references=[], missing_commands=[v + [.34] for v in commands(0)])
    s['performance'].update(evaluation_group_commands=True, evaluation_group_max_commands=4,
                            evaluation_reuse_environment=True, evaluation_gpu_buffer=True,
                            training_gpu_statistics=True, motion_hotpath=True, motion_runtime_reduction=True,
                            contact_history_disabled=True, combined_step_checks=True, fixed_shape_statistics=True)
    s['motion_symmetry'].update(stage_scope='flat straight/in-place yaw; strict swing<.05 and posture/geometry retained',
                                teacher_scope='no teacher/BC; same140 full state', budget='new inclusive10000')
    s['motion_differential'].update(source_checkpoint_sha256=source_sha, online_limit=10000,
                                    restore='same140 Actor/Critic/Adam/std/RNG; new physical reset and command course',
                                    teacher='none; BC0')
    s['motion_spin_quality'].update(formal_updates_proposed=9995, scope='same actuator and v10 reward; new command course only')
    s['budget'].update(smoke_updates=5, formal_updates=9995, formal_samples=9995 * 4096 * 128,
                       sample_limit=10000 * 4096 * 128, formal_wall_s=None, chain_wall_s=None,
                       source_eval_wall_s=7200, smoke_wall_s=1800, evaluation_wall_s=7200)
    s['artifacts'].update(run_name=experiment, storage_max_bytes=STORAGE_BYTES, terminal_reserve_bytes=512 * 2**20)
    s['authorization'] = {'status': 'UNAUTHORIZED', 'hapi_session': owner_session,
                          'user_scope': 'User2026-10-22 explicitly authorizes one progressive Motion10000 inclusive updates after final .012 standing confirmation; unlimited training wall; horizontal posture, leg symmetry, no scissor, aligned wheel centers; GPU0 shared'}
    s['resources'] = {'mode': 'shared', 'sharing_authorized': True,
                      'authorization_note': 'Same explicitly authorized physical GPU0 shared; no preemption or automatic migration'}
    s['notifications'] = {'enabled': True}
    s = effective_spec(s, 0)
    return s


def build(parent, checkpoint, *, experiment, spec_path, confirmation, owner_session):
    p, cp = Path(parent).resolve(), Path(checkpoint).resolve()
    old = json.loads(p.read_text())
    run = p.parents[3] / 'runs' / old['experiment']
    term = json.loads((run / 'formal/terminal_report.json').read_text())
    chain = json.loads((run / 'chain_terminal_report.json').read_text())
    book = json.loads(p.with_name('budget.json').read_text())
    if (old.get('motion_timing', {}).get('version') != SOURCE_COURSE
            or not old['motion_timing'].get('wheel_match_restart')
            or term['status'] not in ('STOPPED_BOUNDED', 'COMPLETED') or term.get('failure')
            or chain['status'] not in ('STOPPED_BOUNDED', 'COMPLETED') or chain.get('failure')
            or book['used']['formal'] != term['actual_updates']
            or any(a['status'] == 'RUNNING' for a in book['attempts'].values())):
        raise ValueError('normally settled wheel-matched standing parent required')
    if confirmation.get('status') != 'CONFIRMED' or confirmation['parent_spec_sha256'] != sha(p):
        raise ValueError('final .012 confirmation receipt required')
    final = confirmation['checkpoints'][-1]
    if len(confirmation['checkpoints']) != 2 or str(cp) != final['checkpoint'] or sha(cp) != final['sha256']:
        raise ValueError('must restore the final fully evaluated .012 checkpoint')
    for path, digest in confirmation['inputs'].items():
        if sha(path) != digest:
            raise ValueError('standing confirmation raw evidence changed')
    source = torch.load(cp, map_location='cpu', weights_only=False)
    from .motion_antiosc_privileged import VERSION as CRITIC, network
    if (source.get('network_architecture') != network() or source.get('critic_privileged_version') != CRITIC
            or source['spec_sha256'] != sha(p) or source['update'] != final['update']
            or source['update'] > term['actual_updates']
            or not all(k in source for k in ('actor', 'critic', 'optimizer', 'training_rng', 'reward_refinement_state'))
            or not source.get('fixed_std')
            or {g['lr'] for g in source['optimizer']['param_groups']} != {3e-5}):
        raise ValueError('full same140 source state/Adam/fixed std differs')
    s = _declaration(old, source, p, cp, experiment, Path(spec_path).resolve(), confirmation, owner_session)
    paths = [p, p.with_name('budget.json'), run / 'formal/terminal_report.json', run / 'chain_terminal_report.json', cp]
    s['student_dependencies'].update({str(q): sha(q) for q in paths})
    s['student_dependencies'].update(confirmation['inputs'])
    return s


def verify(spec):
    cfg = spec['motion_reacquisition']
    if cfg.get('version') != VERSION or cfg.get('configuration') != configuration():
        raise ValueError('explicit posture Motion configuration differs')
    expected = build(cfg['parent_spec'], spec['source_checkpoint'], experiment=spec['experiment'],
                     spec_path=spec['motion_training']['spec_path'], confirmation=cfg['confirmation'],
                     owner_session=spec['authorization']['hapi_session'])
    for key in ('authorization', 'resources', 'notifications'):
        expected[key] = spec[key]
    if spec != expected:
        raise ValueError('posture Motion has undeclared configuration/physics/budget drift')
    validate_recipe(spec['student_training'], spec['source_sha256'])
    return True


def reward_snapshot(spec):
    from .motion_reward_audit import audit
    from .motion_antioscillation import reward_snapshot as extension
    from . import motion_tracking_reward as tracking
    snapshot = audit(spec['reward'])
    snapshot.update(source_sha256=spec['source_sha256'], reward_source_sha256=sha(tracking.__file__),
                    runtime_reward_extension=extension(spec['motion_timing']['configuration']),
                    course=configuration(), scoring=scoring_protocol(spec),
                    note='same v10 reward and non-saturated Huber geometry/anti-oscillation costs; only command curriculum/physical fixed-level selection changes')
    return snapshot
