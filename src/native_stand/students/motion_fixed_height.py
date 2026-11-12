"""Fixed episode height locomotion and causal plant-state privileged Critic.

Explicit opt-in successor of the common-height course. No height transitions,
future actions, teacher controls or changes to the deployable Actor interface.
"""
import copy
import json
import sys
from pathlib import Path
import torch
from . import motion_decoupled as base, motion_height as height
from .data import sha

VERSION = 'motion31_fixed_height_motion_v2'
CRITIC = 'motion_height_robust_privileged184_v1'
NETWORK = 'motion31_fixed_height_critic184_v1'
ASSESSMENT = 'motion31_fixed_height_motion_assessment_v2'
RESTART = 'motion_fixed_height_critic_restart_v1'
HEIGHTS = tuple(round(.30 + i * .01, 2) for i in range(11))
STORAGE_BYTES = base.STORAGE_BYTES
STORAGE_PREFLIGHT = base.STORAGE_PREFLIGHT


def configuration():
    c = base.configuration()
    c.update(version=VERSION, smoke_updates_max=0, formal_updates_max=9953,
             storage_profile='fixed_height_bounded_diagnostics_v1',
             height=dict(version='fixed_episode_height_v1', heights_m=list(HEIGHTS),
                         levels=['fixed032_036_locomotion', 'fixed030_038_locomotion',
                                 'fixed040_locomotion', 'eleven_fixed_heights_low_speed',
                                 'eleven_fixed_heights_mastered_motion_matrix'],
                         initial_equals_target=True, moving_height_transition=False,
                         H0_training=False, engineering_readback_required=True,
                         stable_error_max_m=.02, height_error_scale_m=.04))
    return c


def protocol():
    p = base.protocol()
    p.update(version=ASSESSMENT, height=configuration()['height'], course_controller=configuration())
    p.pop('height_mean_band_m', None)
    p['reacquisition'].pop('height_mean_band_m', None)
    p['height_tracking'] = dict(reference='fixed per-case command root-link-origin world Z', stable_mae_max_m=.02,
                                stable_window_s=[10.,18.], full_controlled_window_s=30.)
    return p


def network():
    n = height.network(); n.update(version=NETWORK, critic_input=184)
    return n


def contract():
    c = height.contract(); c.update(critic_dim=184, critic_privileged_version=CRITIC)
    c['critic_privileged'] += [
        'queued_requested_physical_targets_at_policy_ages0_10_20ms/scales:18',
        'actual_adopted_target_publication_age_ms/20:6',
        'root_body_mass_inertia_scale:1', 'wheel_shape_friction_coefficient:1',
        'actual_Actor_packet_age_ms/20:1',
        'last_PD_feedback_leg_q/1.5:4', 'last_PD_feedback_joint_dq*.05:6',
        'most_recent_applied_worldXY_push_delta_v/.1:2', 'age_since_applied_push_s/30:1']
    c['privileged_scope'] = 'causal simulation state only; queue samples are past requests, never future policy predictions'
    return c


def expand_source(source):
    if source.get('network_architecture') == network() and source.get('critic_privileged_version') == CRITIC:
        return source
    if source.get('network_architecture') != height.network() or source.get('critic_privileged_version') != height.CRITIC:
        raise ValueError('fixed-height184 requires explicitly declared Motion height144 source')
    result = copy.deepcopy(source)
    w = result['critic']['mlp.0.weight']
    if w.shape != (512, 144): raise ValueError('fixed-height source144 weight differs')
    def pad(x): return torch.cat((x, x.new_zeros(512, 40)), -1)
    result['critic']['mlp.0.weight'] = pad(w)
    ids = [pid for g in result['optimizer']['param_groups'] for pid in g['params']]
    if len(ids) != len(result['actor']) + len(result['critic']): raise ValueError('fixed-height Adam ordering differs')
    state = result['optimizer']['state'].get(ids[len(result['actor']) + list(result['critic']).index('mlp.0.weight')], {})
    for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        if name in state: state[name] = pad(state[name])
    result.update(network_architecture=network(), critic_privileged_version=CRITIC)
    return result


def critic_features(queued_targets, adopted_age_ms, mass_scale, friction, actor_age_ms,
                    feedback_leg_q, feedback_dq, applied_push_xy, push_age_s):
    """Explicit tensors; no file, environment, RNG or optimizer access."""
    scales = queued_targets.new_tensor([1.5, 1.5, 100., 1.5, 1.5, 100.])
    x = torch.cat(((queued_targets / scales).flatten(1), adopted_age_ms / 20.,
                   mass_scale[:, None], friction[:, None], actor_age_ms[:, None] / 20.,
                   feedback_leg_q / 1.5, feedback_dq * .05,
                   applied_push_xy / .1, push_age_s[:, None] / 30.), -1)
    if x.shape[-1] != 40: raise ValueError('fixed-height privileged fields must total40')
    return x


def command(target_xy, age, initial, target, moving, training):
    from .motion_training import effective
    result = effective(target_xy, age, training=training)
    result[:, 2] = target
    return result, torch.zeros_like(age)


def learned_heights(state):
    level = state['levels']['height']
    values = [.34]
    if level >= 0: values += [.32, .36]
    if level >= 1: values += [.30, .38]
    if level >= 2: values += [.40]
    if level >= 3: values = list(HEIGHTS)
    return sorted(set(values))


def height_scenarios(level, state, *, evaluation=False):
    levels = {0: (.32, .36), 1: (.30, .38), 2: (.40,), 3: HEIGHTS, 4: HEIGHTS}
    motions = base.skill_commands(state) if level == 4 else [[0., 0.]] + [base.axis_command(a, 0) for a in base.AXES if state['levels'][a] >= 0]
    return [(v[0], v[1], h, h, False) for h in levels[level] for v in motions]


def height_retention_scenarios(state):
    hs = learned_heights(state)
    anchors = sorted({hs[0], .34, hs[-1]})
    motions = [[0., 0.]] + [base.axis_command(a, 0) for a in base.AXES if state['levels'][a] >= 0]
    return [(v[0], v[1], h, h, False) for h in anchors for v in motions]


def retained_scenarios(state):
    return [(v[0], v[1], .34, .34, False) for v in base.skill_commands(state)] + (
        height_retention_scenarios(state) if state['levels']['height'] >= 0 else [])


def focus_scenarios(state, *, evaluation=False):
    if state['focus'] == 'height': return height_scenarios(state['focus_level'], state, evaluation=evaluation)
    if state['focus'] == 'DR': return retained_scenarios(state)
    vx, yaw = base.axis_command(state['focus'], state['focus_level'])
    return [(vx, yaw, h, h, False) for h in learned_heights(state)]


def stage_config(stage=0, envs=4096, *, state=None):
    return base.stage_config(stage, envs, state=state, policy=sys.modules[__name__])


def cases(spec, stage=0, state=None):
    rows = base.cases(spec, stage, state, policy=sys.modules[__name__])
    for row in [r for r in rows if r['_focus_required']][:3]: row['_full_diagnostic'] = True
    return rows


def effective_spec(spec, stage, state=None):
    s = base.effective_spec(spec, stage, state, policy=sys.modules[__name__])
    for row in [r for r in s['student_evaluation']['cases'] if r['_focus_required']][:3]: row['_full_diagnostic'] = True
    return s


def assess(result, arrays, env_id, case):
    result = base.assess(result, arrays, env_id, case)
    result['version'] = ASSESSMENT
    result['height_task'].update(version='fixed_episode_height_v1', height_transition_required=False)
    return result


def commands(stage=0, state=None):
    return stage_config(stage, state=state)['commands']


advance = base.advance
choose = base.choose
extend_mixture = base.extend_mixture
refine_mixtures = base.refine_mixtures
refinement_records = base.refinement_records


def validate_recipe(recipe, source_sha):
    from .motion_antioscillation import validate_configuration
    if recipe.get('course_version') != VERSION or recipe.get('source_actor_sha256') != source_sha:
        raise ValueError('fixed-height recipe/source differs')
    state = recipe['decoupled_state']
    if state['version'] != VERSION or recipe['stage'] != state['stage'] or recipe['course_stage'] != state['stage']:
        raise ValueError('fixed-height course revision differs')
    if recipe['critic_privileged_version'] != CRITIC or recipe['network_architecture'] != network():
        raise ValueError('fixed-height184 explicit contract differs')
    validate_configuration(recipe['timing_configuration'])
    from .motion_training import declaration
    baseline = declaration(source_sha, 0, reward_version=recipe['reward']['version'])
    for key in ('version','reward','training_family','training_yaw_rad','target_height_m','train_schedule_s',
                'parking_s','environment_seconds','warmup_updates_per_phase','BC_coefficient',
                'fixed_log_std','physical_std_leg_rad','physical_std_wheel_rad_s'):
        if recipe[key] != baseline[key]: raise ValueError('fixed-height changed unrelated baseline: '+key)
    if recipe['checkpoint_every_updates'] != 100:
        raise ValueError('fixed-height save/sampling differs')
    return recipe


def build(parent, checkpoint, *, experiment, spec_path, poses, owner_session, **unused):
    p = Path(parent).resolve(); cp = Path(checkpoint).resolve(); old = json.loads(p.read_text())
    base.verify(old, require_current_claim=False)
    run = p.parents[3] / 'runs' / old['experiment']
    book = json.loads(p.with_name('budget.json').read_text())
    terminal = json.loads((run/'chain_terminal_report.json').read_text())
    formal = json.loads((run/'formal/terminal_report.json').read_text())
    source = torch.load(cp, map_location='cpu', weights_only=False)
    if (terminal['status'] != 'STOPPED_BOUNDED' or terminal.get('failure') or formal.get('stop_kind') != 'user_signal'
            or any(a['status'] == 'RUNNING' for a in book['attempts'].values())
            or cp != run/'formal'/f"model_update_{formal['actual_updates']}.pt"
            or source['spec_sha256'] != sha(p) or book['used']['formal'] != formal['actual_updates']):
        raise ValueError('fixed-height restart requires exact settled latest complete parent')
    expand_source(source)
    original = old['motion_training']['prior_consumption']
    used = original['historical_online_updates'] + book['used']['smoke'] + book['used']['formal']
    if used >= 10000: raise ValueError('original10000 PPO budget exhausted')
    evidence_paths = [p, p.with_name('budget.json'), run/'chain_terminal_report.json', run/'formal/terminal_report.json', cp]
    proof = dict(version=RESTART, parent_spec=str(p), parent_run=str(run),
                 inputs={str(q): sha(q) for q in evidence_paths},
                 historical_online_updates=used, historical_smoke_updates=original['historical_smoke_updates'] + book['used']['smoke'],
                 historical_formal_updates=used-original['historical_smoke_updates']-book['used']['smoke'],
                 historical_wall_s=original['historical_wall_s'] + terminal['chain_wall_s'],
                 historical_artifact_bytes=original['historical_artifact_bytes'] + sum(q.stat().st_size for q in run.rglob('*') if q.is_file() and not q.is_symlink()),
                 original_online_limit=10000, parent_wiring_proof=old['motion_training']['wiring_reuse'],
                 source_evidence_run=old['motion_training']['wiring_reuse']['source_evidence_run'],
                 runtime_probe_required=True, capability_accepted=False)
    state = copy.deepcopy(source['motion_course_state'])
    state.update(version=VERSION, focus='height', focus_level=0, focus_cursor=0, stage=0,
                 stage_started_update=0, last_evaluation_update=-1, accepted_by_stage={}, regression_streaks={}, repair_queue=[])
    state['levels']['height'] = -1
    state['DR_level'] = state['levels']['DR']
    state['certified_DR_level'] = state['levels']['DR']
    state.pop('assessment_keys', None)
    s = copy.deepcopy(old)
    s.update(experiment=experiment, source_checkpoint=str(cp), source_sha256=sha(cp), network_architecture=network(),
             student_interface=contract(), observation={**old['observation'], 'critic_dim':184}, motion_assessment=protocol())
    maximum = STORAGE_BYTES - proof['historical_artifact_bytes']
    s['motion_decoupled'] = dict(version=VERSION, configuration=configuration(), state=state, height_poses=poses,
        parent_spec=str(p), parent_spec_sha256=sha(p), previous_consumption=proof,
        storage_preflight_version=STORAGE_PREFLIGHT, engineering_recovery=proof,
        fixed_episode_height=True, formal_engineering_probe=True)
    s['student_training'].pop('formal_resume_offset', None)
    s['student_training'].update(course_version=VERSION, stage=0, course_stage=0, decoupled_state=state,
        critic_privileged_version=CRITIC, network_architecture=network(), source_actor_sha256=sha(cp),
        critic='source144_zero_pad40_causal_plant_state', optimizer='source_Adam_zero_new_columns',
        milestones=list(range(250,10000-used,250)), storage_max_bytes=maximum)
    s['motion_training'].update(stage=0, spec_path=str(Path(spec_path).resolve()), resume=None, wiring_reuse=proof, prior_consumption=proof)
    s['motion_course'] = dict(source_recipe_updates=source['recipe_updates'], configuration=configuration())
    s['motion_robustness']['state'] = state
    s['motion_timing'].update(parent_spec=str(p), parent_spec_sha256=sha(p), source_checkpoint_sha256=sha(cp))
    s['student_evaluation'].update(version=ASSESSMENT)
    s['student_evaluation']['cases'][0].update(checkpoint=str(cp), sha256=sha(cp))
    s['evaluation'].update(protocol_identity=ASSESSMENT, full_every_updates=250)
    s['budget'].update(smoke_updates=0, formal_updates=10000-used, formal_samples=(10000-used)*4096*128,
                       sample_limit=(10000-used)*4096*128)
    s['artifacts'].update(run_name=experiment, storage_max_bytes=maximum)
    s['authorization'] = dict(status='UNAUTHORIZED', hapi_session=owner_session,
        user_scope='User2026-10-23 explicitly approves Critic privileges184 and fixed episode heights.30-.40m locomotion, no H0 training or moving height switching; normal latest restart within remaining original10000, GPU0 shared, unlimited training wall, existing owner notification')
    s['student_dependencies'] = {**old['student_dependencies'], **proof['inputs']}
    return effective_spec(s, 0, state)


def verify(spec, *, require_current_claim=True):
    cfg = spec['motion_decoupled']
    if cfg.get('assessment_keys_resume'):
        from .motion_course_keys_resume import verify as verify_resume
        return verify_resume(spec, require_current_claim=require_current_claim)
    expected = build(cfg['parent_spec'], spec['source_checkpoint'], experiment=spec['experiment'],
                     spec_path=spec['motion_training']['spec_path'], poses=cfg['height_poses'],
                     owner_session=spec['authorization']['hapi_session'])
    for key in ('authorization', 'resources', 'notifications'): expected[key] = spec[key]
    if spec != expected: raise ValueError('fixed-height declaration has undeclared drift')
    if [p['height_m'] for p in cfg['height_poses']] != list(HEIGHTS) or any(p['offline_valid'] is not True or p['closure_max_m'] > 1e-4 for p in cfg['height_poses']):
        raise ValueError('eleven fixed-height closure branches required')
    for path, digest in spec['student_dependencies'].items():
        if sha(path) != digest: raise ValueError('fixed-height dependency changed: '+path)
    validate_recipe(spec['student_training'], spec['source_sha256'])
    claim = Path(spec['motion_training']['spec_path']).parents[1]/'.motion_fixed_height_claims'/(sha(cfg['parent_spec'])+'.json')
    if require_current_claim and claim.exists() and json.loads(claim.read_text())['destination'] != spec['experiment']:
        raise ValueError('fixed-height restart already claimed')
    return True


def verify_wiring_recovery(spec):
    verify(spec)
    return dict(status='VERIFIED', scope='historical real five-update wiring; new184 physical readback before first optimizer step',
                capability_accepted=False, remaining_formal_updates=spec['budget']['formal_updates'])


def reward_snapshot():
    value = height.reward_snapshot()
    value.update(configuration=configuration()['height'], formula_source_sha256=sha(__file__),
                 reference_rate_mps=0., height_formula_source_sha256=sha(height.__file__))
    return value


def diagnostic_storage(payload, arrays):
    from .motion_robustness import diagnostic_storage as select_env
    result = select_env(payload, arrays)
    # All eight scored results remain; keep full raw traces for at most three
    # representative focus cases per matrix, others a finite2s geometry tail.
    case = payload['case']
    full = case.get('_focus_required') and case.get('_full_diagnostic', False)
    count = len(result['policy_time_s'])
    if not full:
        result = {k:(v[-200:].copy() if getattr(v,'ndim',0) and len(v)==count else v) for k,v in result.items()}
    payload['diagnostic_storage'].update(version='fixed_height_bounded_diagnostics_v1',
        retained_policy_samples=count if full else min(200,count), full_episode_trace_retained=bool(full), all8_per_env_results_retained=True)
    return result
