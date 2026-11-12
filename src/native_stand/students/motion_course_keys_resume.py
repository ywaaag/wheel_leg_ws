"""Explicit fixed-height course-key repair and freshly authorized bounded resume.

Historical spec/ledger/checkpoint/evaluations remain read-only. The latest
completed matrix is checked against the reconstructed revision, then applied
exactly once to a derived state; it is not evidence for the next focus.
"""
import copy
import json
from pathlib import Path
import torch
from .data import sha

VERSION = 'motion_fixed_height_assessment_keys_resume_v1'


def repaired_state(spec, source, candidate):
    from . import motion_fixed_height as course
    state = copy.deepcopy(source['motion_course_state'])
    if state['stage'] != candidate['stage'] or candidate['update'] > source['update']:
        raise ValueError('repair matrix is not in the checkpoint active revision')
    effective = course.effective_spec(spec, state['stage'], state)
    if (state.get('assessment_keys') == effective['motion_decoupled']['state']['assessment_keys']
            or state['last_evaluation_update'] != candidate['update']):
        raise ValueError('repair requires the known stale-key last completed boundary')
    actual = candidate['evaluations']; expected = effective['student_evaluation']['cases']
    if len(actual) != len(expected): raise ValueError('repair matrix coverage differs')
    for record, case in zip(actual, expected):
        observed = record['case']
        for key in ('command','_identity_key','_course_revision','_height','_motion_dr','_focus_required','_purpose','_task_axis'):
            if observed.get(key) != case.get(key): raise ValueError('repair case identity differs: '+key)
    state = effective['motion_decoupled']['state']
    state['last_evaluation_update'] = candidate['update'] - 1
    state, decision = course.advance(state, candidate)
    if decision['evidence_valid'] is not True: raise ValueError('repair did not produce valid complete evidence')
    # New run local time starts at zero. Accepted history retains the original
    # checkpoint/update. No old raw result or checkpoint is rewritten.
    state.update(stage_started_update=0, last_evaluation_update=-1)
    effective = course.effective_spec(spec, state['stage'], state)
    return effective['motion_decoupled']['state'], decision


def evidence(parent, checkpoint):
    from . import motion_fixed_height as course
    parent = Path(parent).resolve(); checkpoint = Path(checkpoint).resolve()
    old = json.loads(parent.read_text()); course.verify(old, require_current_claim=False)
    run = parent.parents[3]/'runs'/old['experiment']; formal = run/'formal'
    book = json.loads(parent.with_name('budget.json').read_text())
    terminal = json.loads((run/'chain_terminal_report.json').read_text())
    end = json.loads((formal/'terminal_report.json').read_text())
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (terminal['status'] != 'STOPPED_BOUNDED' or terminal.get('failure') or end['stop_kind'] != 'user_signal'
            or any(a['status'] == 'RUNNING' for a in book['attempts'].values())
            or checkpoint != formal/f"model_update_{end['actual_updates']}.pt"
            or source['update'] != end['actual_updates'] or source['spec_sha256'] != sha(parent)
            or book['used']['formal'] != end['actual_updates'] or source['network_architecture'] != course.network()
            or source['critic_privileged_version'] != course.CRITIC):
        raise ValueError('repair requires settled exact latest full184 parent')
    if any(k not in source for k in ('actor','critic','optimizer','training_rng','reward_refinement_state','symmetry_mix_state')):
        raise ValueError('full PPO/refinement checkpoint required')
    registry = json.loads((formal/'candidate_registry.json').read_text())
    complete = [c for c in registry if c.get('complete_matrix') is True and c.get('stage') == source['motion_course_state']['stage']
                and 0 < c['update'] <= source['update']]
    if not complete: raise ValueError('no latest-revision complete matrix')
    candidate = max(complete, key=lambda c:c['update'])
    if sha(candidate['checkpoint']) != candidate['sha256']: raise ValueError('matrix checkpoint changed')
    paths = [parent,parent.with_name('budget.json'),run/'chain_terminal_report.json',formal/'terminal_report.json',
             formal/'manifest.json',formal/'candidate_registry.json',formal/'course_progress.json',checkpoint,Path(candidate['checkpoint'])]
    for record in candidate['evaluations']:
        path = (formal/record['path']).resolve()
        if not path.is_relative_to(formal) or sha(path) != record['sha256']: raise ValueError('matrix raw identity differs')
        raw = json.loads(path.read_text()); trace = (formal/raw['trace']['path']).resolve()
        if (not trace.is_relative_to(formal) or sha(trace) != raw['trace']['sha256']
                or raw['case'] != record['case']
                or len(record['per_env']) != 8
                or any(any(full.get(k) != v for k,v in compact.items()) for full,compact in zip(raw['per_env'],record['per_env']))
                or raw['checkpoint_sha256'] != candidate['sha256'] or raw['spec_sha256'] != sha(parent)
                or raw['asset_sha256'] != old['asset_sha256'] or raw['protocol'] != old['motion_assessment']
                or raw['model_unchanged'] is not True or len(raw['per_env']) != 8
                or {e['env'] for e in raw['per_env']} != set(range(8))
                or any(type(e.get('accepted')) is not bool or e.get('controlled_seconds') != 30. for e in raw['per_env'])):
            raise ValueError('complete original eight-env raw evidence differs')
        paths += [path, trace]
    state, decision = repaired_state(old, source, candidate)
    previous = old['motion_decoupled']['previous_consumption']
    proof = dict(version=VERSION, parent_spec=str(parent), parent_run=str(run),
        inputs={str(p):sha(p) for p in paths}, checkpoint_update=source['update'], matrix_update=candidate['update'],
        historical_online_updates=previous['historical_online_updates']+book['used']['smoke']+book['used']['formal'],
        historical_wall_s=previous['historical_wall_s']+terminal['chain_wall_s'],
        historical_artifact_bytes=previous['historical_artifact_bytes']+sum(p.stat().st_size for p in run.rglob('*') if p.is_file() and not p.is_symlink()),
        historical_smoke_updates=previous['historical_smoke_updates']+book['used']['smoke'],
        historical_formal_updates=previous['historical_formal_updates']+book['used']['formal'],
        original_online_limit=10000, new_authorized_online_updates=10000,
        parent_wiring_proof=old['motion_training']['wiring_reuse'],
        source_evidence_run=old['motion_training']['wiring_reuse']['source_evidence_run'],
        reconstructed_state=state, replay_decision=decision, capability_accepted=False,
        matrix_scope='one derived course transition from the latest complete matrix; not next-focus or latest-source certification',
        physical_state_restored=False, warmup_repeated=False)
    return old, source, proof


def build(parent,checkpoint,*,experiment,spec_path,owner_session,**unused):
    from . import motion_fixed_height as course
    old,source,proof = evidence(parent,checkpoint); s = copy.deepcopy(old)
    state = copy.deepcopy(proof['reconstructed_state']); cp = Path(checkpoint).resolve(); parent = Path(parent).resolve()
    s.update(experiment=experiment,source_checkpoint=str(cp),source_sha256=sha(cp))
    cfg = s['motion_decoupled']; cfg.update(parent_spec=str(parent),parent_spec_sha256=sha(parent),state=state,
        previous_consumption=proof,engineering_recovery=proof,formal_engineering_probe=True,
        assessment_keys_resume=proof,storage_preflight_version=course.STORAGE_PREFLIGHT)
    cfg['configuration'].update(formal_updates_max=10000,smoke_updates_max=0)
    s['motion_course'].update(source_recipe_updates=source['recipe_updates'],configuration=copy.deepcopy(cfg['configuration']))
    s['motion_training'].update(spec_path=str(Path(spec_path).resolve()),resume=None,wiring_reuse=proof,prior_consumption=proof)
    recipe = s['student_training']
    recipe.update(source_actor_sha256=sha(cp),critic='source_same_semantic184_full_state',optimizer='source_full_Adam',
        formal_resume_offset=source['update'],milestones=list(range(250,10001,250)),storage_max_bytes=course.STORAGE_BYTES)
    s['student_evaluation']['cases'][0].update(checkpoint=str(cp),sha256=sha(cp))
    s['motion_timing'].update(parent_spec=str(parent),parent_spec_sha256=sha(parent),source_checkpoint_sha256=sha(cp))
    s['budget'].update(source_eval_updates=0,smoke_updates=0,formal_updates=10000,
        formal_samples=10000*s['ppo']['envs']*s['ppo']['steps_per_env'],sample_limit=10000*s['ppo']['envs']*s['ppo']['steps_per_env'],
        formal_wall_s=None,chain_wall_s=None)
    s['artifacts'].update(run_name=experiment,storage_max_bytes=course.STORAGE_BYTES)
    s['authorization']=dict(status='UNAUTHORIZED',hapi_session=owner_session,
        user_scope='User2026-10-23 explicitly authorizes course promotion key repair, controlled latest full-state restart and fresh10000 online updates without training wall limit; keep historical consumption, GPU0 shared and existing notification/TensorBoard; unchanged physics/reward/PPO/interface')
    s['student_dependencies'].update(proof['inputs'])
    s = course.effective_spec(s,state['stage'],state)
    return s


def verify(spec,*,require_current_claim=True):
    proof = spec['motion_decoupled']['assessment_keys_resume']
    expected = build(proof['parent_spec'],spec['source_checkpoint'],experiment=spec['experiment'],
        spec_path=spec['motion_training']['spec_path'],owner_session=spec['authorization']['hapi_session'])
    for key in ('authorization','resources','notifications'): expected[key]=spec[key]
    if spec != expected: raise ValueError('course-key repair declaration drift')
    if require_current_claim:
        claim = Path(spec['motion_training']['spec_path']).parents[1]/'.motion_course_key_claims'/(sha(proof['parent_spec'])+'.json')
        if claim.exists() and json.loads(claim.read_text())['destination'] != spec['experiment']:
            raise ValueError('course-key recovery already claimed')
    return True


def verify_wiring(spec):
    verify(spec)
    return dict(status='VERIFIED',scope='same184D full-state execution wiring; derived course-key repair only',
        capability_accepted=False,remaining_formal_updates=10000,prior_online_updates=spec['motion_decoupled']['previous_consumption']['historical_online_updates'])
