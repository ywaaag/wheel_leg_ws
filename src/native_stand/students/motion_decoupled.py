"""One Actor/owner, independent command/height/DR progress and bounded repair."""
import copy
import json
from pathlib import Path
import torch
from .data import sha
from . import motion_robustness as dr
from . import motion_height as height

VERSION = 'motion31_decoupled_height_course_v1'
ASSESSMENT = 'motion31_decoupled_height_assessment_v1'
AXES = ('forward', 'backward', 'yaw_positive', 'yaw_negative')
FOCI = ('height', 'forward', 'backward', 'yaw_positive', 'yaw_negative')
TARGETS = dr.TARGETS
STORAGE_BYTES = 6 * 2**30
STORAGE_PREFLIGHT = 'motion_stage_capacity_v1'
SMOKE_RECOVERY = 'motion_height_h0_smoke_coverage_recovery_v1'


def configuration():
    return dict(version=VERSION, axes=list(AXES) + ['height', 'DR'],
                targets=list(TARGETS), envs=4096, focus_fraction=.5,
                rehearsal_fraction=.35, nominal_zero_fraction=.15,
                confirmations=1, regression_confirmations=2, evaluation_every=250,
                checkpoint_every=100, stage_updates_max=10000,
                formal_updates_max=9995, smoke_updates_max=5,
                storage_profile='focus_full_retention_tail_v1', storage_max_bytes=STORAGE_BYTES,
                DR_retreat=False, simultaneous_difficulty_increases=False,
                unlock='only the active focus; other tasks cannot veto its promotion',
                regression='two consecutive complete known failures lower only that task ceiling; no model rollback',
                evaluation='retained tasks at certified DR; DR focus adds current full motion commands and nominal zero; no repeated nominal matrix',
                height=height.configuration(), behavior_chain_stop=False)


def protocol():
    p = dr.protocol(); p.update(version=ASSESSMENT, confirmations_required=1,
                                height=height.configuration(), course_controller=configuration())
    return p


def axis_command(axis, level):
    value = TARGETS[level]
    return {'forward': [value, 0.], 'backward': [-value, 0.],
            'yaw_positive': [0., value], 'yaw_negative': [0., -value]}[axis]


def initial_state(source_state):
    qualified = source_state.get('qualified_by_DR', {}).get(str(source_state['certified_DR_level']), [])
    levels = {}
    for axis in AXES:
        levels[axis] = max((i for i in range(6) if axis_command(axis, i) in qualified), default=-1)
    levels.update(height=-1, DR=source_state['certified_DR_level'])
    return dict(version=VERSION, levels=levels, focus='height', focus_level=0, focus_cursor=0,
                stage=0, stage_started_update=0, last_evaluation_update=-1,
                DR_level=max(0, levels['DR']), certified_DR_level=levels['DR'],
                mode='FOCUS_ACQUIRE', accepted_by_stage={}, regression_streaks={}, repair_queue=[],
                last_checkpoint_sha256=None, smooth_mix=source_state.get('smooth_mix', .75))


def skill_commands(state):
    values = [[0., 0.]]
    for axis in AXES:
        values += [axis_command(axis, i) for i in range(state['levels'][axis] + 1)]
    return values


def height_scenarios(level, state, *, evaluation=False):
    if level == 0: return [(0., 0., .34, .34, False)]
    if level == 1: return [(0., 0., .34, h, False) for h in (.32, .36)]
    motions = [[0., 0.]] + [axis_command(axis, 0) for axis in AXES if state['levels'][axis] >= 0]
    pairs = [(a, b) for a, b in ((.34, .32), (.34, .36))]
    if level >= 3:
        pairs = [(a, b) for a in height.HEIGHTS for b in height.HEIGHTS if a != b]
        if evaluation:
            pairs = [(.32, .36), (.36, .32), (.33, .35), (.35, .33)]
    return [(v[0], v[1], a, b, level == 4) for a, b in pairs for v in motions]


def focus_scenarios(state, *, evaluation=False):
    axis, level = state['focus'], state['focus_level']
    if axis == 'height': return height_scenarios(level, state, evaluation=evaluation)
    if axis == 'DR': return [(v[0], v[1], .34, .34, False) for v in skill_commands(state)]
    return [(*axis_command(axis, level), .34, .34, False)]


def retained_scenarios(state):
    result = [(v[0], v[1], .34, .34, False) for v in skill_commands(state)]
    if state['levels']['height'] >= 1:
        result += height_scenarios(state['levels']['height'], state, evaluation=True)
    return result


def _allocate(total, n):
    q, r = divmod(total, n)
    return [q + (i < r) for i in range(n)]


def stage_config(stage=0, envs=4096, *, state=None, policy=None):
    if state is None or stage != state['stage'] or envs not in (1, 64, 4096):
        raise ValueError('decoupled stage requires explicit state and1/64/4096 envs')
    # Cohorts remain separate even when XY commands are identical: height and purpose own identity.
    old = (policy.retained_scenarios if policy else retained_scenarios)(state)
    focus = (policy.focus_scenarios if policy else focus_scenarios)(state)
    cohorts = [(0., 0., .34, .34, False)] + old + focus
    zero = round(envs * .15); new = envs // 2; rehearsal = envs - zero - new
    if envs == 1: zero, rehearsal, new = 1, 0, 0
    counts = [zero] + _allocate(rehearsal, len(old)) + _allocate(new, len(focus))
    slots = [i for i, count in enumerate(counts) for _ in range(count)]
    groups = [0] * zero + [2] * (rehearsal + new)
    values = [list(c[:2]) for c in cohorts]
    return dict(id=stage, name=f"{state['focus']}_level_{state['focus_level']}", commands=values,
                weights=[n / envs for n in counts], sampling_mode='fixed_slots', sampling_slots=slots,
                slot_counts=counts, enforce_whitelist=True, cohort_keys=[list(c[:4]) for c in cohorts],
                height_scenarios=[list(c) for c in cohorts], group_slots=groups,
                cohort_purposes=['nominal_zero'] + ['rehearsal'] * len(old) + ['focus'] * len(focus))


def commands(stage=0, state=None):
    return stage_config(stage, state=state)['commands']


def holding_coverage(stage, rows):
    """Report actual slot coverage, with a bounded legacy H0 alias correction."""
    count = len(stage['commands']); values = [0] * count
    for row in rows:
        counts = row.get('hold_sample_counts', [])
        if len(counts) != count or any(type(n) is not int or n < 0 for n in counts):
            return dict(version='height_cohort_coverage_v1', status='FAILED', reason='malformed holding counts')
        values = [a + b for a, b in zip(values, counts)]
    missing = [i for i, n in enumerate(stage['slot_counts']) if n and not values[i]]
    keys = [tuple(v) for v in stage['height_scenarios']]
    # Only the historical H0 smoke has physically identical height scenarios.
    # Later transitions and their moving flag can never borrow other heights.
    h0 = stage.get('name') == 'height_level_0' and all(k[2:] == (.34, .34, False) for k in keys)
    unresolved = [i for i in missing if not h0 or not any(values[j] and keys[i] == k for j, k in enumerate(keys))]
    return dict(version='height_cohort_coverage_v1', status='FAILED' if unresolved else 'PASSED',
                scope='per_slot' if not missing else 'legacy_H0_identical_task_keys_only',
                actual_slot_holding_samples=values, unobserved_slot_labels=missing, unresolved_slots=unresolved,
                separate_alias_DR_coverage_verified=not missing, full_episode_verified=False,
                note='H0 equivalent keys do not prove separate slot/DR histories; no skill success prerequisite')


def cases(spec, stage=0, state=None, *, policy=None):
    state = state or spec['motion_decoupled']['state']
    template = spec['student_evaluation']['cases'][0]
    level = state['DR_level']; result = []; seen = set()
    def add(scenario, task, purpose, group=2):
        vx, yaw, initial, final, moving = scenario
        key = (vx, yaw, initial, final, moving, group)
        required = purpose == 'focus' and (state['focus'] != 'DR' or group == 2)
        if key in seen:
            next(row for row in result if row['_identity_key'] == list(key))['_focus_required'] |= required
            return
        seen.add(key)
        row = {k: copy.deepcopy(v) for k, v in template.items() if not k.startswith('_')}
        row.update(case_id=f'decoupled_{stage}_{len(result)}', command=[vx, yaw, final],
                   _timing_level=4, _height=dict(initial_m=initial, target_m=final, moving=moving),
                   _motion_dr=dict(level=level, group=group, seed=20261023 + stage),
                   _task_axis=task, _purpose=purpose, _focus_required=required,
                   _identity_key=list(key), _course_revision=stage)
        result.append(row)
    add((0., 0., .34, .34, False), 'zero', 'retention', 0)
    for axis in AXES:
        for i in range(state['levels'][axis] + 1):
            add((*axis_command(axis, i), .34, .34, False), axis, 'retention', 1 if state['focus'] == 'DR' else 2)
    if state['levels']['height'] >= (0 if policy else 1):
        retained = policy.height_retention_scenarios(state) if policy else height_scenarios(state['levels']['height'], state, evaluation=True)
        for scenario in retained:
            add(scenario, 'height', 'retention', 1 if state['focus'] == 'DR' else 2)
    for scenario in (policy.focus_scenarios if policy else focus_scenarios)(state, evaluation=True):
        for group in ((1, 2) if state['focus'] == 'DR' else (2,)):
            add(scenario, state['focus'], 'focus', group)
    return result


def effective_spec(spec, stage, state=None, *, policy=None):
    s = copy.deepcopy(spec); state = copy.deepcopy(state or s['motion_decoupled']['state'])
    if state['stage'] != stage: raise ValueError('decoupled revision differs')
    rows = cases(s, stage, state, policy=policy)
    state['assessment_keys'] = [r['_identity_key'] for r in rows]
    s['motion_decoupled']['state'] = state
    s['motion_robustness']['state'] = state
    s['motion_training']['stage'] = stage
    s['student_training'].update(stage=stage, course_stage=stage, decoupled_state=state)
    s['curriculum']['stages'] = [stage_config(stage, s['ppo']['envs'], state=state, policy=policy)]
    s['motion_timing']['stage'] = 4
    s['student_evaluation']['cases'] = rows
    return s


def extend_mixture(previous, keys):
    by_key = {tuple(row['target']): row for row in previous}
    result = []
    for key in keys:
        old = by_key.get(tuple(key))
        # Only the nominal .34/.34 cohort may inherit legacy XY refinement.
        if old is None and len(key) == 4 and key[2:] == [.34, .34]: old = by_key.get(tuple(key[:2]))
        row = copy.deepcopy(old or dict(mix=0., consecutive_eligible=0, confirmations=0, regressions=0))
        row['target'] = list(key); row.setdefault('consecutive_eligible', 0)
        result.append(row)
    return result


def refinement_records(records, stage):
    # The caller uses refine_mixtures, which matches full height-aware cohort keys.
    return records


def refine_mixtures(ppo, spec, records):
    from .motion_refinement import advance as refine
    from .motion_differential import update_mix
    keys = spec['curriculum']['stages'][0]['cohort_keys']
    grouped = []
    for key in keys:
        parts = [r for r in records if r['case']['command'][:2] == key[:2]
                 and [r['case']['_height'][k] for k in ('initial_m', 'target_m')] == key[2:]]
        rows = []
        if parts and all(len(r.get('per_env', [])) == 8 and all(type(e.get('accepted')) is bool for e in r['per_env']) for r in parts):
            for eid in range(8):
                es = [next(e for e in r['per_env'] if e['env'] == eid) for r in parts]
                e = copy.deepcopy(max(es, key=lambda e: (e.get('accepted') is not True, e.get('tracking', {}).get('MAE', 999))))
                e['accepted'] = all(x.get('accepted') is True for x in es)
                rows.append(e)
        grouped.append(dict(case=dict(command=key[:2] + [key[3]]), per_env=rows))
    old = copy.deepcopy(ppo.reward_refinement_state)
    nominal = [i for i, key in enumerate(keys) if key[2:] == [.34, .34]]
    probe = [{**old[i], 'target': keys[i][:2]} for i in nominal]
    narrowed, decisions = refine(probe, [grouped[i] for i in nominal], spec['reward']['refinement'])
    for i, row in zip(nominal, narrowed): old[i] = {**row, 'target': keys[i]}
    ppo.reward_refinement_state = old
    sym = [{**row, 'target': key[:2]} for row, key in zip(ppo.symmetry_mix_state, keys)]
    sym = update_mix(sym, grouped)
    ppo.symmetry_mix_state = [{**row, 'target': key} for row, key in zip(sym, keys)]
    return decisions


def _complete(record):
    es = record.get('per_env', [])
    return len(es) == 8 and {e.get('env') for e in es} == set(range(8)) and all(type(e.get('accepted')) is bool for e in es)


def _next_focus(state, previous):
    if state['repair_queue']:
        axis, target = state['repair_queue'].pop(0); return axis, target
    if previous != 'DR' and state['levels']['DR'] < len(dr.LEVELS) - 1:
        return 'DR', state['levels']['DR'] + 1
    for distance in range(1, len(FOCI) + 1):
        cursor = (state['focus_cursor'] + distance) % len(FOCI); axis = FOCI[cursor]
        maximum = 4 if axis == 'height' else 5
        if state['levels'][axis] < maximum:
            state['focus_cursor'] = cursor
            return axis, state['levels'][axis] + 1
    if state['levels']['DR'] < len(dr.LEVELS) - 1: return 'DR', state['levels']['DR'] + 1
    return 'height', 4


def advance(state, candidate):
    s = copy.deepcopy(state); records = candidate.get('evaluations', [])
    if candidate.get('stage') != s['stage'] or candidate['update'] <= s['last_evaluation_update']:
        raise ValueError('decoupled checkpoint/revision differs')
    if any(r['case'].get('_course_revision') != s['stage'] for r in records): raise ValueError('mixed course revision')
    valid = (candidate.get('complete_matrix') is True and bool(records) and all(_complete(r) for r in records)
             and [r['case'].get('_identity_key') for r in records] == s.get('assessment_keys'))
    focus_rows = [r for r in records if r['case'].get('_focus_required')]
    passed = valid and bool(focus_rows) and all(e['accepted'] for r in focus_rows for e in r['per_env'])
    regressions = []
    if valid:
        for axis in (*AXES, 'height'):
            # A new, not-yet-certified DR distribution cannot prove loss of an old skill.
            eligible = [r for r in records if r['case']['_task_axis'] == axis
                        and r['case']['_purpose'] == 'retention'
                        and (s['focus'] != 'DR' or r['case']['_motion_dr']['group'] == 1)]
            if not eligible: continue
            failed = any(e['accepted'] is False for r in eligible for e in r['per_env'])
            streak_key = f"{axis}@DR{s['levels']['DR']}"
            s['regression_streaks'][streak_key] = s['regression_streaks'].get(streak_key, 0) + 1 if failed else 0
            if s['regression_streaks'][streak_key] >= 2 and s['levels'][axis] >= 0:
                previous = s['levels'][axis]; s['levels'][axis] -= 1
                s['regression_streaks'][streak_key] = 0
                if not any(a == axis for a, _ in s['repair_queue']): s['repair_queue'].append([axis, previous])
                regressions.append(axis)
    s.update(last_evaluation_update=candidate['update'], last_checkpoint_sha256=candidate['sha256'])
    changed = bool(regressions)
    if passed:
        axis = s['focus']; s['levels'][axis] = max(s['levels'][axis], s['focus_level'])
        s['accepted_by_stage'][str(s['stage'])] = dict(checkpoint=candidate['checkpoint'], sha256=candidate['sha256'],
            update=candidate['update'], focus=axis, focus_level=s['focus_level'])
        s['focus'], s['focus_level'] = _next_focus(s, axis); changed = True
    elif valid:
        # Keep the unsuccessful task at its own level, but let other independent
        # tasks have their next250-update focus. UNKNOWN does not rotate/promote.
        previous = (s['focus'], s['focus_level'])
        s['focus'], s['focus_level'] = _next_focus(s, s['focus'])
        changed |= previous != (s['focus'], s['focus_level'])
    if changed:
        s['stage'] += 1; s['stage_started_update'] = candidate['update']
    s['certified_DR_level'] = s['levels']['DR']
    s['DR_level'] = s['focus_level'] if s['focus'] == 'DR' else max(0, s['levels']['DR'])
    all_done = all(s['levels'][axis] >= 5 for axis in AXES) and s['levels']['height'] >= 4 and s['levels']['DR'] >= 5
    return s, dict(advanced=changed, stage_passed=passed, evidence_valid=valid,
                   all_stages_passed=all_done, stop=False, focus=state['focus'],
                   lowered_tasks=regressions, next_focus=s['focus'], levels=s['levels'],
                   progression='one focus; one complete matrix pass; two explicit failures per mastered task; DR never retreats')


def assess(result, arrays, env_id, case):
    import numpy as np
    from .motion_reacquisition import assess as base_assess
    base_assess(result, arrays, env_id)
    m = arrays['first_episode_valid'][:, env_id].astype(bool) & (arrays['controller'][:, env_id, 0] == 3)
    states = arrays['state'][m, env_id][:3000]
    # Gate consumers require literal Python booleans (``is True``), and the
    # same result is persisted by the strict JSON writer. NumPy's scalar bool
    # passes neither contract even though its truth value is correct.
    complete = bool(len(states) == 3000 and np.isfinite(states).all())
    c = case['_height']; window = [18., 26.] if c['moving'] else [10., 18.]
    checks = dict(height_transition_complete=complete, height_transition_tracking=False,
                  height_stable_tracking=False)
    metrics = dict(version=height.VERSION, initial_m=c['initial_m'], target_m=c['target_m'],
                   moving=c['moving'], stable_window_s=window, complete_30s=complete)
    if complete:
        age = torch.arange(3000, dtype=torch.float64) * .01
        ref, rate = height.reference(age, torch.full_like(age, c['initial_m']),
                                    torch.full_like(age, c['target_m']), torch.full_like(age, c['moving'], dtype=torch.bool))
        error = abs(states[:, 6] - ref.numpy())
        a, b = (int(t * 100) for t in window)
        start = 1200 if c['moving'] else 200; end = start + 400
        metrics.update(height_stable_mae_m=float(error[a:b].mean()),
                       transition_error_p95_m=float(np.quantile(error[start:end], .95)),
                       initial_observed_height_m=float(states[0, 6]),
                       stable_height_mean_m=float(states[a:b, 6].mean()))
        checks.update(height_transition_tracking=metrics['transition_error_p95_m'] <= .04,
                      height_stable_tracking=metrics['height_stable_mae_m'] <= .02)
        result['checks']['height'] = checks['height_stable_tracking']
        result['checks']['reacquisition_height'] = checks['height_stable_tracking']
    result['height_task'] = metrics
    result['checks'].update(checks)
    result.update(version=ASSESSMENT, accepted=all(v is True for v in result['checks'].values()),
                  failed_checks=[k for k,v in result['checks'].items() if v is not True])
    return result


def choose(candidates, state):
    evaluated = [c for c in candidates if c.get('update', 0) > 0 and c.get('complete_matrix')]
    best = max(evaluated, key=lambda c: (sum(e.get('accepted') is True for e in c['per_env']) / max(1, len(c['per_env'])), c['update']), default=None)
    return dict(source=candidates[0], source_baseline=candidates[0], latest=candidates[-1], latest_checkpoint=candidates[-1],
                best_new=best, best_new_evaluated_candidate=best, best_overall=best or candidates[0],
                accepted_candidate=None, accepted_by_stage=state['accepted_by_stage'], current_stage=state['stage'],
                task_levels=state['levels'], focus=state['focus'], standing_quality_certified=False,
                final_multi_seed_certified=False, capability_scope='one checkpoint exact evaluated task/height/DR matrix; no success-point stitching')


def validate_recipe(recipe, source_sha):
    from .motion_antioscillation import validate_configuration
    from .motion_training import declaration
    if recipe.get('course_version') != VERSION or recipe.get('source_actor_sha256') != source_sha:
        raise ValueError('decoupled recipe/source version differs')
    state = recipe['decoupled_state']
    if recipe['stage'] != state['stage'] or recipe['course_stage'] != state['stage'] or state['version'] != VERSION:
        raise ValueError('decoupled explicit state differs')
    if recipe['critic_privileged_version'] != height.CRITIC or recipe['network_architecture'] != height.network():
        raise ValueError('height144 declaration differs')
    validate_configuration(recipe['timing_configuration'])
    baseline = declaration(source_sha, 0, reward_version=recipe['reward']['version'])
    for key in ('version','reward','training_family','training_yaw_rad','target_height_m','train_schedule_s',
                'parking_s','environment_seconds','warmup_updates_per_phase','BC_coefficient',
                'fixed_log_std','physical_std_leg_rad','physical_std_wheel_rad_s'):
        if recipe[key] != baseline[key]: raise ValueError('decoupled changed frozen baseline: ' + key)
    if recipe['milestones'] != list(range(250, 10000, 250)) or recipe['checkpoint_every_updates'] != 100:
        raise ValueError('decoupled checkpoint/evaluation cadence differs')
    return recipe


def preppo_recovery_evidence(parent_spec):
    """Known height-result/phase-capacity wiring faults before any PPO."""
    p = Path(parent_spec).resolve(); old = json.loads(p.read_text())
    run = p.parents[3] / 'runs' / old['experiment']
    book = json.loads(p.with_name('budget.json').read_text())
    chain = json.loads((run / 'chain_terminal_report.json').read_text())
    if chain.get('failure') == "RuntimeError(['Motion command holding coverage missing'])":
        return smoke_recovery_evidence(p)
    terminal = json.loads((run / 'source_eval/terminal_report.json').read_text())
    log = run / 'source_eval.log'
    attempts = list(book['attempts'].values())
    if (old.get('motion_decoupled', {}).get('version') != VERSION
            or chain['status'] != 'FAILED'
            or terminal['actual_updates'] != 0 or not attempts
            or any(a['phase'] != 'source_eval' or a['status'] not in ('FAILED', 'COMPLETED') or a.get('updates') != 0 for a in attempts)
            or any(book['used'].get(k, 0) for k in ('smoke', 'formal', 'evaluate', 'play'))
            or chain['smoke_updates'] != 0 or chain['formal_updates'] != 0
            or (run / 'smoke').exists() or (run / 'formal').exists()):
        raise ValueError('settled zero-PPO height source failure required')
    previous = old['motion_decoupled'].get('engineering_recovery', {})
    bool_fault = (not previous and terminal['status'] == 'FAILED'
                  and all(a['status'] == 'FAILED' for a in attempts)
                  and 'TypeError: Object of type bool is not JSON serializable' in log.read_text()
                  and 'motion_runner.py' in log.read_text())
    storage_fault = False
    if terminal['status'] == 'COMPLETED' and all(a['status'] == 'COMPLETED' for a in attempts):
        import ast
        failure = chain.get('failure') or ''
        try:
            call = ast.parse(failure, mode='eval').body
            text = ast.literal_eval(call.args[0])
            report = ast.literal_eval(text.removeprefix('smoke preflight: '))
            storage_fault = (text.startswith('smoke preflight: ')
                and report.get('errors') == ['student output filesystem reserve insufficient']
                and previous.get('version') in (None, 'motion_height_preppo_bool_recovery_v1'))
        except (ValueError, SyntaxError, AttributeError, IndexError, TypeError):
            pass
    if not (bool_fault or storage_fault):
        raise ValueError('unsupported height failure; diagnose before recovery')
    verify(old, require_current_claim=False)
    readback = run / 'source_eval/height_engineering_readback.json'
    probe = json.loads(readback.read_text())
    if (probe.get('status') != 'PASSED' or probe.get('PPO_updates') != 0
            or len(probe.get('results', [])) != len(height.HEIGHTS)):
        raise ValueError('completed height engineering readback required')
    paths = [p, p.with_name('budget.json'), run / 'chain_terminal_report.json',
             run / 'source_eval/terminal_report.json', log, readback]
    original = old['motion_training'].get('prior_consumption', {})
    result = dict(version='motion_height_preppo_storage_recovery_v1' if storage_fault else 'motion_height_preppo_bool_recovery_v1', parent_spec=str(p), parent_run=str(run),
                inputs={**original.get('inputs', {}), **{str(q): sha(q) for q in paths}}, historical_online_updates=0, original_online_limit=10000,
                historical_wall_s=original.get('historical_wall_s', 0.) + chain['chain_wall_s'],
                historical_source_eval_attempts=original.get('historical_source_eval_attempts', 0) + book['used']['source_eval'],
                historical_artifact_bytes=original.get('historical_artifact_bytes', 0) + sum(q.stat().st_size for q in run.rglob('*') if q.is_file() and not q.is_symlink()),
                source_sha256=old['source_sha256'], source_parent_spec=old['motion_decoupled']['parent_spec'],
                reason=('phase-capacity estimate and terminal reserve unified; reuse completed source with strict identity and raw trace verification' if storage_fault
                        else 'strict bool producer repaired; repeat bounded source readback, retain original failed evidence and10000 inclusive budget'))
    if storage_fault:
        import numpy as np
        folder = run / 'source_eval'
        paths = [folder / name for name in ('progress.json', 'manifest.json', 'candidate_registry.json')]
        progress = json.loads(paths[0].read_text()); manifest = json.loads(paths[1].read_text())
        if (progress.get('engineering_valid') is not True or progress.get('status') != 'COMPLETED'
                or progress.get('actual_updates') != 0 or progress.get('evaluations') != terminal.get('evaluations')
                or manifest.get('spec_sha256') != sha(p) or manifest.get('source_sha256') != old['source_sha256']):
            raise ValueError('completed source identity/progress differs')
        expected_cases = [r for r in old['student_evaluation']['cases'] if r['command'] == [0., 0., .34] and r['_motion_dr']['group'] == 0]
        records = terminal['evaluations']
        if len(records) != 1 or [r['case'] for r in records] != expected_cases:
            raise ValueError('exact nominal source case required')
        for record in records:
            raw_path = folder / record['path']; raw = json.loads(raw_path.read_text())
            trace = folder / raw['trace']['path']
            if (sha(raw_path) != record['sha256'] or sha(trace) != raw['trace']['sha256']
                    or raw['case'] != record['case'] or raw['per_env'] != record['per_env']
                    or raw.get('model_unchanged') is not True or raw.get('spec_sha256') != sha(p)
                    or raw['checkpoint_sha256'] != old['source_sha256'] or raw['asset_sha256'] != old['asset_sha256']
                    or raw['protocol'] != protocol() or len(raw['per_env']) != 8
                    or {e['env'] for e in raw['per_env']} != set(range(8))):
                raise ValueError('source raw/trace/protocol identity differs')
            with np.load(trace, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            for eid, row in enumerate(raw['per_env']):
                rescored = assess(copy.deepcopy(row), arrays, eid, raw['case'])
                if rescored != row or row['height_task'].get('complete_30s') is not True:
                    raise ValueError('source full-window scoring differs')
            paths.extend([raw_path, trace])
        result['inputs'].update({str(q): sha(q) for q in paths})
        result['source_evidence_run'] = str(run)
    return result


def smoke_recovery_evidence(parent_spec):
    """Reuse completed H0 short-smoke engineering events, never a skill claim."""
    from .training_evidence import smoke_certificate
    p = Path(parent_spec).resolve(); old = json.loads(p.read_text())
    run = p.parents[3] / 'runs' / old['experiment']; folder = run / 'smoke'
    book = json.loads(p.with_name('budget.json').read_text())
    chain = json.loads((run / 'chain_terminal_report.json').read_text())
    terminal = json.loads((folder / 'terminal_report.json').read_text())
    previous = old['motion_decoupled'].get('engineering_recovery', {})
    if (previous.get('version') != 'motion_height_preppo_storage_recovery_v1'
            or chain.get('status') != 'FAILED'
            or chain.get('failure') != "RuntimeError(['Motion command holding coverage missing'])"
            or book['used']['smoke'] != 5 or book['used']['formal'] != 0
            or chain['smoke_updates'] != 5 or chain['formal_updates'] != 0
            or terminal['status'] != 'COMPLETED' or terminal.get('failure') or terminal['actual_updates'] != 5
            or (run / 'formal').exists() or any(a['status'] == 'RUNNING' for a in book['attempts'].values())):
        raise ValueError('settled completed H0 smoke with exact coverage-only fault required')
    verify(old, require_current_claim=False)
    original_certificate = json.loads((folder / 'wiring_certificate.json').read_text())
    if original_certificate.get('status') != 'FAILED' or original_certificate.get('errors') != ['Motion command holding coverage missing']:
        raise ValueError('original coverage-only certificate required')
    derived = smoke_certificate(folder, old)
    if (derived['status'] != 'PASSED' or derived['command_coverage'].get('scope') != 'legacy_H0_identical_task_keys_only'
            or derived['command_coverage'].get('unresolved_slots')):
        raise ValueError('H0 equivalent-key engineering coverage not proven')
    rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    if [row['update'] for row in rows] != list(range(1, 6)):
        raise ValueError('exact charged five real smoke updates required')
    stage = old['curriculum']['stages'][0]; unit = old['ppo']['steps_per_env']
    aliases = [i for i, row in enumerate(stage['height_scenarios']) if row == [0., 0., .34, .34, False]]
    if aliases != [0, 1, len(stage['commands']) - 1]:
        raise ValueError('only the original H0 zero aliases may be recovered')
    for row in rows:
        expected = [n * unit for n in stage['slot_counts']]
        expected[0] = sum(expected[i] for i in aliases)
        for i in aliases[1:]: expected[i] = 0
        if row['command_sample_counts'] != expected:
            raise ValueError('actual command counts do not match exact H0 alias collapse')
    source = run / 'source_eval'; source_terminal = json.loads((source / 'terminal_report.json').read_text())
    if source_terminal['status'] != 'COMPLETED' or source_terminal['actual_updates'] != 0:
        raise ValueError('completed source reuse required')
    paths = [p, p.with_name('budget.json'), run / 'chain_terminal_report.json',
             source / 'terminal_report.json', source / 'candidate_registry.json', source / 'source_reuse.json',
             folder / 'wiring_certificate.json', folder / 'metrics.jsonl', folder / 'terminal_report.json',
             folder / 'model_update_5.pt'] + [folder / name for name in derived['inputs']]
    checkpoint = torch.load(folder / 'model_update_5.pt', map_location='cpu', weights_only=False)
    if (checkpoint['update'] != 5 or checkpoint['spec_sha256'] != sha(p)
            or checkpoint['source_sha256'] != old['source_sha256']
            or checkpoint['critic_privileged_version'] != height.CRITIC):
        raise ValueError('charged smoke checkpoint identity differs')
    prior = old['motion_training']['prior_consumption']
    return dict(version=SMOKE_RECOVERY, parent_spec=str(p), parent_run=str(run), source_evidence_run=str(run),
                inputs={**prior['inputs'], **{str(q): sha(q) for q in paths}},
                historical_online_updates=5, historical_smoke_updates=5, original_online_limit=10000,
                historical_wall_s=prior['historical_wall_s'] + chain['chain_wall_s'],
                historical_source_eval_attempts=prior['historical_source_eval_attempts'] + book['used']['source_eval'],
                historical_artifact_bytes=prior['historical_artifact_bytes'] + sum(q.stat().st_size for q in run.rglob('*') if q.is_file() and not q.is_symlink()),
                source_sha256=old['source_sha256'], source_parent_spec=old['motion_decoupled']['parent_spec'],
                derived_engineering_certificate=derived, no_new_smoke=True,
                reason='preserve sampled cohort identity; H0 equivalent task-key smoke accepted by derived engineering evidence only; old FAILED certificate unchanged')


def build(parent, checkpoint, *, experiment, spec_path, poses, owner_session, recovery_parent=None, storage_preflight_version=None):
    p = Path(parent).resolve(); cp = Path(checkpoint).resolve(); old = json.loads(p.read_text())
    run = p.parents[3] / 'runs' / old['experiment']; book = json.loads(p.with_name('budget.json').read_text())
    terminal = json.loads((run / 'formal/terminal_report.json').read_text())
    chain = json.loads((run / 'chain_terminal_report.json').read_text())
    if chain['status'] != 'STOPPED_BOUNDED' or chain.get('failure') or terminal.get('stop_kind') != 'user_signal' or any(a['status'] == 'RUNNING' for a in book['attempts'].values()):
        raise ValueError('settled user-stopped Motion parent required')
    source = torch.load(cp, map_location='cpu', weights_only=False)
    if cp != run / 'formal' / f"model_update_{terminal['actual_updates']}.pt" or source['spec_sha256'] != sha(p):
        raise ValueError('exact latest complete stopped checkpoint required')
    height.expand_source(source)
    state = initial_state(source['motion_course_state']); s = copy.deepcopy(old)
    for key in ('motion_large_continuation','checkpoint_quick_evaluation','motion_confirmation','motion_reacquisition'):
        s.pop(key, None)
    s.update(experiment=experiment, source_checkpoint=str(cp), source_sha256=sha(cp),
             network_architecture=height.network(), student_interface=height.contract(),
             observation={**old['observation'], 'critic_dim':144}, motion_assessment=protocol())
    s['motion_decoupled'] = dict(version=VERSION, configuration=configuration(), state=state,
        height_poses=poses, parent_spec=str(p), parent_spec_sha256=sha(p),
        previous_consumption=chain.get('online_updates_including_parent'), latest_source_unassessed=True)
    if storage_preflight_version is not None:
        if storage_preflight_version != STORAGE_PREFLIGHT: raise ValueError('unknown Motion storage preflight version')
        s['motion_decoupled']['storage_preflight_version'] = storage_preflight_version
    s['motion_robustness'] = dict(version=dr.VERSION, configuration=dr.configuration(), state=state,
        material_backend=dr.MATERIAL_BACKEND, material_writer=dr.MATERIAL_WRITER, failure_tail_saved_envs=8,
        storage_profile='eight_env_metrics_one_env_trace_v1')
    s['student_training'].pop('formal_resume_offset', None)
    s['student_training'].update(course_version=VERSION, decoupled_state=state, source_actor_sha256=sha(cp),
        stage=0, course_stage=0, critic_privileged_version=height.CRITIC, network_architecture=height.network(),
        critic='source140_zero_pad_height_context144', optimizer='source_Adam_zero_new_columns',
        milestones=list(range(250,10000,250)), storage_max_bytes=STORAGE_BYTES,
        selection_scope='independent forward/backward/yaw signs/height/DR; exact same-checkpoint matrices')
    s['motion_training'].update(stage=0, spec_path=str(Path(spec_path).resolve()), resume=None)
    for key in ('wiring_reuse','partial_smoke_recovery','prior_consumption','successor'):
        s['motion_training'].pop(key, None)
    s['motion_course'] = dict(source_recipe_updates=source['recipe_updates'], configuration=configuration())
    s['motion_timing'].update(parent_spec=str(p), parent_spec_sha256=sha(p), source_checkpoint_sha256=sha(cp))
    for key in ('fixed_parent','critic_upgrade','wheel_match_restart'): s['motion_timing'].pop(key, None)
    s['student_evaluation']['version'] = ASSESSMENT
    s['student_evaluation']['cases'][0].update(checkpoint=str(cp), sha256=sha(cp))
    s['evaluation'].update(protocol_identity=ASSESSMENT, full_every_updates=250)
    s['curriculum']['minimum_zero_fraction'] = .15
    s['budget'].update(smoke_updates=5, formal_updates=9995, formal_samples=9995*4096*128,
        sample_limit=10000*4096*128, formal_wall_s=None, chain_wall_s=None,
        source_eval_wall_s=7200, smoke_wall_s=3600, evaluation_wall_s=21600)
    s['artifacts'].update(run_name=experiment, storage_max_bytes=STORAGE_BYTES, terminal_reserve_bytes=512*2**20)
    s['authorization'] = dict(status='UNAUTHORIZED', hapi_session=owner_session,
        user_scope='User2026-10-23 approves normal stop/latest checkpoint and new decoupled forward/backward/yaw signs/height/DR10000 inclusive PPO, unlimited training wall, GPU0 shared; retain existing fault protections and owner notification')
    s['resources'].update(mode='shared', sharing_authorized=True,
        authorization_note='User explicitly authorizes new10000 decoupled Motion/height course on inherited GPU0 shared')
    s['notifications'] = dict(enabled=True)
    s['motion_startup'].update(references=[], missing_commands=[[0.,0.,.34]], no_capability_prerequisite_for_smoke=True)
    paths = [p, p.with_name('budget.json'), run/'chain_terminal_report.json', run/'formal/terminal_report.json', cp]
    s['student_dependencies'] = {str(q):sha(q) for q in paths}
    s['student_dependencies'][s['asset_path']] = s['asset_sha256']
    s['student_dependencies'][s['recovery']['pose_library']] = sha(s['recovery']['pose_library'])
    if recovery_parent is not None:
        evidence = preppo_recovery_evidence(recovery_parent)
        if evidence['source_sha256'] != s['source_sha256'] or evidence['source_parent_spec'] != str(p):
            raise ValueError('height recovery must retain the original source and course parent')
        s['motion_decoupled']['engineering_recovery'] = evidence
        s['motion_training']['prior_consumption'] = evidence
        if evidence['version'] == SMOKE_RECOVERY:
            s['motion_training']['wiring_reuse'] = evidence
            s['budget'].update(smoke_updates=0, sample_limit=9995*4096*128)
        s['authorization']['user_scope'] += '; existing authorization covers the identified engineering repair/resume only; original10000 and all charged updates/wall/storage consumption retained, no new allowance'
        maximum = STORAGE_BYTES - evidence['historical_artifact_bytes']
        if maximum <= s['artifacts']['terminal_reserve_bytes']:
            raise ValueError('failed height parent exhausted the cumulative storage ceiling')
        s['student_training']['storage_max_bytes'] = maximum
        s['artifacts']['storage_max_bytes'] = maximum
        s['student_dependencies'].update(evidence['inputs'])
    return effective_spec(s, 0, state)


def verify(spec, *, require_current_claim=True):
    cfg = spec['motion_decoupled']
    if cfg['version'] != VERSION or cfg['configuration'] != configuration(): raise ValueError('decoupled configuration differs')
    expected = build(cfg['parent_spec'], spec['source_checkpoint'], experiment=spec['experiment'],
                     spec_path=spec['motion_training']['spec_path'], poses=cfg['height_poses'], owner_session=spec['authorization']['hapi_session'],
                     recovery_parent=cfg.get('engineering_recovery', {}).get('parent_spec'),
                     storage_preflight_version=cfg.get('storage_preflight_version'))
    for key in ('authorization','resources','notifications'): expected[key] = spec[key]
    if spec != expected: raise ValueError('decoupled spec has undeclared drift')
    for path, digest in spec['student_dependencies'].items():
        if sha(path) != digest: raise ValueError('decoupled dependency SHA differs: ' + path)
    if len(cfg['height_poses']) != len(height.HEIGHTS) or any(p['offline_valid'] is not True or p['closure_max_m'] > 1e-4 for p in cfg['height_poses']):
        raise ValueError('height closure-solved initialization evidence missing')
    validate_recipe(spec['student_training'], spec['source_sha256'])
    claim = Path(spec['motion_training']['spec_path']).parents[1]/'.motion_decoupled_claims'/(sha(cfg['parent_spec'])+'.json')
    if require_current_claim and Path(spec['motion_training']['spec_path']).exists() and claim.exists() and json.loads(claim.read_text())['destination'] != spec['experiment']:
        raise ValueError('unused declaration superseded by a newer explicit submission')
    return True


def reused_source(spec):
    """Reference immutable completed source results; no physics, no optimizer."""
    evidence = spec['motion_decoupled']['engineering_recovery']
    if evidence['version'] != 'motion_height_preppo_storage_recovery_v1':
        raise ValueError('height source reuse requires the identified post-source storage failure')
    if preppo_recovery_evidence(evidence['parent_spec']) != evidence:
        raise ValueError('height source reuse evidence changed')
    folder = Path(evidence['source_evidence_run']) / 'source_eval'
    records = json.loads((folder / 'terminal_report.json').read_text())['evaluations']
    for record in records:
        record['path'] = str((folder / record['path']).resolve()); record['reused'] = True
    candidates = json.loads((folder / 'candidate_registry.json').read_text())
    if len(candidates) != 1 or candidates[0]['sha256'] != spec['source_sha256'] or candidates[0]['update'] != 0:
        raise ValueError('source candidate registry differs')
    candidates[0]['evaluations'] = records
    return records, candidates


def verify_wiring_recovery(spec):
    proof = spec['motion_training']['wiring_reuse']
    if proof['version'] != SMOKE_RECOVERY or smoke_recovery_evidence(proof['parent_spec']) != proof:
        raise ValueError('height H0 wiring recovery evidence changed')
    if spec['budget']['smoke_updates'] != 0 or spec['budget']['formal_updates'] + proof['historical_online_updates'] != 10000:
        raise ValueError('height wiring recovery cannot repeat smoke or enlarge budget')
    verify(spec)
    return dict(status='VERIFIED', historical_smoke_updates=5, remaining_formal_updates=9995,
                capability_accepted=False, scope='H0 equivalent-key short engineering smoke only')


def diagnostic_storage(payload, arrays):
    from .motion_robustness import diagnostic_storage as select_env
    result = select_env(payload, arrays)
    if not payload['case'].get('_focus_required'):
        count = len(result['policy_time_s'])
        result = {k:(v[-200:].copy() if getattr(v,'ndim',0) and len(v)==count else v) for k,v in result.items()}
        payload['diagnostic_storage'].update(version='focus_full_retention_tail_v1',
            retained_policy_samples=min(200,count), full_episode_trace_retained=False,
            all8_per_env_results_retained=True)
    else:
        payload['diagnostic_storage'].update(version='focus_full_retention_tail_v1', full_episode_trace_retained=True)
    return result
