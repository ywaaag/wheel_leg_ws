"""Bounded 4500 continuation: actual handover states and PD-age uncertainty.

Actor31/Critic312, reward, PD, motor capability and exploration stay unchanged.
Only new explicit declarations select this branch. No simulator imports here.
"""
import copy
import json
from pathlib import Path

from .data import sha
from .motion_posture_design import commands, apply_runtime_reward
from .motion_mcu_course import protocol, extend_mixture

VERSION = 'motion31_mcu_handover_pd_age_v1'
SOURCE_SHA = '0be977b308cfe493fdddee54823e21c713508fa582ce1618ab34e57b13fbc619'
SOURCE_RELATIVE = 'runs/2026_10_25-23_52__motion1500_posture_retention312_10000/formal/model_update_4500.pt'
RETENTION = 'motion31_mcu4500_nominal_source_retention_v3'
ONLINE = 1005
FORMAL = 1000
STORAGE = 2 * 2**30


def retention(source_sha):
    from .motion_source_retention import declaration
    if source_sha != SOURCE_SHA:
        raise ValueError('4500 retention source differs')
    return dict(declaration(source_sha), version=RETENTION, commands=commands(), eligible_mcu_variant=0,
                scope='source-qualified nominal zero/forward/backward/yaw only; stress/takeover excluded')


def recipe(source_sha):
    from .motion_posture_preparation import recipe as old_recipe
    from .motion_posture_design import SOURCE_SHA as OLD_SHA
    from .motion_startup import smoke_profile
    if source_sha != SOURCE_SHA:
        raise ValueError('full-passed4500 source required')
    result = old_recipe(OLD_SHA)
    result.update(source_actor_sha256=source_sha, course_version=VERSION,
        critic='student_control_only', optimizer='source_same312_Adam',
        milestones=[500, 1000], storage_max_bytes=STORAGE,
        environment_seconds=33.52, warmup_updates_per_phase=5,
        smoke_wiring=smoke_profile(), source_policy_retention=retention(source_sha),
        selection_scope='same4500 fixed034 nominal motion plus real frozen-policy handover and PD-age hypotheses; not real hardware alignment')
    return result


def validate_recipe(value, source_sha):
    if value != recipe(source_sha):
        raise ValueError('handover robustness recipe differs')
    return value


def configuration():
    return dict(version=VERSION, formal_updates_max=FORMAL, smoke_updates_max=5,
                stage_updates_max=FORMAL, evaluation_every=500, progression=False,
                height_m=.34, PD_feedback_range_ms=[8, 20],
                uncertainty_sampling='reset-only per-env counter stream; never policy RNG',
                handover_prefix_policy_intervals=100, handover_controller='frozen same4500, not LQR',
                handover_state='continuous physics, local adopted previous and complete motor queue; no reset at switch',
                allocation={'uncertain_handover_zero': .25, 'nominal_zero': .25, 'old_motion': .5},
                single_changed_execution_parameter='PD feedback age only; Actor age/transport/response/capability/spring fixed',
                behavior_stop=False, independent_task_online_limit=ONLINE)


def stage_config(stage=0, envs=4096):
    from .motion_posture_design import stage_config as base
    if stage != 0:
        raise ValueError('fixed bounded uncertainty branch has no automatic new stage')
    out = base(stage, envs)
    out['name'] = '4500_handover_PD_age'
    out['mcu_variants'][0] = 3
    out['cohort_keys'][0][-1] = 3
    return out


def cases(spec, stage=0):
    from .motion_posture_design import cases as base
    out = base(spec, stage)
    for r in out:
        r['case_id'] = 'robust_' + r['case_id']
    template = out[0]
    # New conditions are distinct identities even when vx/yaw are identical.
    for age in (8, 14, 20):
        for prefix in (0, 100):
            out.append(dict(template, case_id=f'robust_PD{age}_prefix{prefix}',
                _mcu_variant=3, _execution_uncertainty={'PD_age_ms': age, 'prefix_intervals': prefix},
                seconds=31.52 + prefix * .01))
    return out


def effective_spec(spec, stage):
    s = copy.deepcopy(spec)
    s['motion_training']['stage'] = stage
    s['student_training'].update(stage=stage, course_stage=stage)
    s['curriculum']['stages'] = [stage_config(stage, s['ppo']['envs'])]
    s['student_evaluation']['cases'] = cases(s, stage)
    return s


def refinement_records(records, stage):
    # Nominal old capability controls narrowing; an unseen uncertainty is not
    # silently reclassified as nominal or source-certified.
    from .motion_posture_design import refinement_records as base
    rows = [r for r in records if '_execution_uncertainty' not in r['case']]
    return base(rows, stage)


def advance(state, candidate):
    from .motion_mcu_measured import complete
    if candidate.get('stage') != 0 or candidate['update'] <= state.get('last_evaluation_update', -1):
        raise ValueError('stale handover evaluation')
    records = candidate.get('evaluations', [])
    identities = [r['case']['case_id'] for r in records]
    valid = (candidate.get('complete_matrix') is True and len(records) == 30
             and len(set(identities)) == 30 and all(complete(r) for r in records))
    passed = bool(valid and all(all(e['accepted'] for e in r['per_env']) for r in records))
    out = copy.deepcopy(state)
    out.update(last_evaluation_update=candidate['update'], consecutive_passes=(state.get('consecutive_passes', 0) + 1 if passed else 0))
    if passed:
        out['accepted_by_stage']['0'] = {k: candidate[k] for k in ('checkpoint', 'sha256', 'update')}
    return out, dict(evidence_valid=valid, stage_passed=passed, advanced=False,
                     final_confirmed=passed, all_stages_passed=False, stop=False)


def choose(candidates, accepted_by_stage):
    from .motion_posture_design import choose as base
    out = base(candidates, accepted_by_stage)
    out.update(capability_scope='fixed034 MCU PD-age/handover hypotheses and retained nominal motion; not hardware certification')
    return out


def source_records(parent_spec, checkpoint):
    """Read immutable4500 raw evaluations, never bind the mutable registry SHA."""
    p, cp = Path(parent_spec).resolve(), Path(checkpoint).resolve()
    if cp.name != 'model_update_4500.pt' or sha(cp) != SOURCE_SHA:
        raise ValueError('4500 checkpoint identity differs')
    old = json.loads(p.read_text())
    registry = json.loads((cp.parent / 'candidate_registry.json').read_text())
    candidate = next(r for r in registry if r.get('update') == 4500)
    if (candidate.get('sha256') != SOURCE_SHA or candidate.get('complete_matrix') is not True
            or len(candidate.get('per_env', [])) != 192 or not all(e['accepted'] is True for e in candidate['per_env'])):
        raise ValueError('same4500 complete24-condition192/192 evidence required')
    records, inputs = [], {str(cp): SOURCE_SHA, str(p): sha(p)}
    for rec in candidate['evaluations']:
        path = cp.parent / rec['path']
        if sha(path) != rec['sha256']:
            raise ValueError('4500 raw JSON SHA differs')
        raw = json.loads(path.read_text())
        trace = path.parent / raw['trace']['path']
        if sha(trace) != raw['trace']['sha256'] or raw['checkpoint_sha256'] != SOURCE_SHA:
            raise ValueError('4500 trace/model identity differs')
        inputs.update({str(path): sha(path), str(trace): sha(trace)})
        records.append(dict(rec, path=str(path), per_env=raw['per_env']))
    if len(records) != 24:
        raise ValueError('4500 complete24 conditions required')
    return old, records, inputs


def build(parent, checkpoint, diagnostic, *, experiment, spec_path, session):
    import torch
    p, cp, diagnostic = Path(parent).resolve(), Path(checkpoint).resolve(), Path(diagnostic).resolve()
    old, records, inputs = source_records(p, cp)
    result = json.loads(diagnostic.read_text())
    if (result.get('status') != 'COMPLETED' or result.get('actual_cases') != 54
            or result.get('PPO_updates') != 0 or result.get('model_unchanged') is not True):
        raise ValueError('complete fixed-model closed-loop matrix required')
    pd_case = next(r for r in result['records'] if r['case_id'] == 'baseline4500__PD_age_20')
    if pd_case['termination'] != 'FALL':
        raise ValueError('this PD-age branch requires the recorded sensitivity evidence')
    source = torch.load(cp, map_location='cpu', weights_only=False)
    if source['spec_sha256'] != sha(p) or not all(k in source for k in ('actor', 'critic', 'optimizer', 'training_rng')):
        raise ValueError('4500 full training state/spec binding required')
    s = copy.deepcopy(old)
    s.pop('mcu_posture', None)
    s.update(experiment=experiment, source_checkpoint=str(cp), source_sha256=SOURCE_SHA)
    s['student_training'] = recipe(SOURCE_SHA)
    s['motion_training'].update(spec_path=str(Path(spec_path).resolve()), stage=0, resume=None)
    s['motion_course'] = dict(source_recipe_updates=source['recipe_updates'], configuration=configuration())
    s['motion_execution_robustness'] = dict(version=VERSION, parent_spec=str(p), parent_spec_sha256=sha(p),
        source_checkpoint=str(cp), source_sha256=SOURCE_SHA, diagnostic=str(diagnostic), diagnostic_sha256=sha(diagnostic),
        configuration=configuration(), old_training_not_stopped_or_changed=True,
        source_native_conditions=24, source_native_passed=192,
        source_reused_only_for_unchanged_nominal_conditions=False)
    s['budget'].update(source_eval_wall_s=1800, smoke_updates=5, formal_updates=FORMAL,
        formal_samples=FORMAL * 4096 * 64, sample_limit=ONLINE * 4096 * 64,
        smoke_wall_s=1800, formal_wall_s=14400, chain_wall_s=18000, evaluation_wall_s=1800)
    s['artifacts'].update(run_name=experiment, storage_max_bytes=STORAGE, terminal_reserve_bytes=256 * 2**20)
    s['evaluation'].update(full_every_updates=500, full_trace_updates=[])
    s['student_evaluation']['max_output_bytes'] = 128 * 2**20
    s['student_dependencies'] = dict(inputs, **{str(diagnostic): sha(diagnostic), s['asset_path']: s['asset_sha256']})
    s['motion_startup']['references'] = []
    s['motion_startup']['missing_commands'] = []
    s['authorization'] = dict(status='UNAUTHORIZED', hapi_session=session,
        user_scope='User2026-10-26 approves complete closed-loop sensitivity/handover and targeted PPO; explicitly selects complete-passed4500. Independent bounded smoke5+formal1000, GPU0 shared, 18000s total,2GiB; no old budget reuse, no hardware, no retry or automatic successor.')
    s['resources'] = dict(mode='shared', sharing_authorized=True,
        authorization_note='User-approved GPU0 sharing for4500 execution robustness; no preemption or GPU switch')
    s['notifications'] = dict(enabled=False)
    s = effective_spec(s, 0)
    # Existing minimal startup keys omit variant/uncertainty. Keep its validated
    # empty-reference shape, while this version's source branch evaluates ALL
    # declared cases directly rather than deduplicating same-command conditions.
    key=lambda row:json.dumps({k:row[k] for k in ('command','seed','family','yaw_rad','seconds')},sort_keys=True)
    s['motion_startup']['missing_commands'] = [r['command'] for r in {key(r):r for r in s['student_evaluation']['cases']}.values()]
    return s


def verify(spec):
    c = spec['motion_execution_robustness']
    hold=Path(spec['motion_training']['spec_path']).with_name('submission_hold.json')
    if hold.is_file():
        value=json.loads(hold.read_text())
        if value.get('status')=='USER_HOLD_BEFORE_SUBMISSION':
            raise ValueError('user requested no training submission; submission_hold.json remains active')
    expected = build(c['parent_spec'], spec['source_checkpoint'], c['diagnostic'], experiment=spec['experiment'],
                     spec_path=spec['motion_training']['spec_path'], session=spec['authorization'].get('hapi_session'))
    for key in ('authorization', 'resources', 'notifications'):
        expected[key] = spec[key]
    if spec != expected:
        raise ValueError('4500 handover robustness declaration drift')
    return True
