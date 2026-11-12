"""CPU plan/publish for the explicitly authorized decoupled Motion/height course."""
import argparse
import json
import os
from pathlib import Path

from native_stand.students.data import sha
from native_stand.students.motion_decoupled import build, verify, STORAGE_BYTES, STORAGE_PREFLIGHT
from native_stand.students.motion_height import prepare_poses, reward_snapshot
from native_stand.students.motion_reward_audit import audit
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-spec', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--recover-from', type=Path, help='Settled known height wiring failure spec; retains all original charged PPO')
    p.add_argument('--fixed-height-motion', action='store_true', help='Explicit184D Critic, fixed episode heights.30-.40; restart within remaining original10000')
    p.add_argument('--repair-assessment-keys', action='store_true', help='Explicit user-authorized fixed184 course-key repair and fresh10000; settled latest full-state parent required')
    p.add_argument('--persistent-frontiers', action='store_true', help='Persistent independent Motion/height/DR frontiers; normal settled parent, remaining current10000 only')
    p.add_argument('--write', action='store_true')
    p.add_argument('--authorize', action='store_true')
    a = p.parse_args()
    if a.authorize and not a.write: p.error('--authorize requires --write and current user authorization')
    if a.fixed_height_motion and a.recover_from: p.error('fixed-height design restart is separate from engineering failure recovery')
    if a.repair_assessment_keys and (a.fixed_height_motion or a.recover_from): p.error('course-key repair must be a separate explicit restart')
    if a.persistent_frontiers and (a.repair_assessment_keys or a.fixed_height_motion or a.recover_from): p.error('persistent curriculum is a separate explicit transition')
    build_fn, verify_fn, snapshot_fn = build, verify, reward_snapshot
    heights = None
    if a.fixed_height_motion:
        from native_stand.students.motion_fixed_height import build as build_fn, verify as verify_fn, reward_snapshot as snapshot_fn, HEIGHTS
        heights = HEIGHTS
    if a.repair_assessment_keys:
        from native_stand.students.motion_course_keys_resume import build as build_fn, verify as verify_fn
        from native_stand.students.motion_fixed_height import reward_snapshot as snapshot_fn, HEIGHTS
        heights = HEIGHTS
    if a.persistent_frontiers:
        from native_stand.students.motion_persistent_resume import build as build_fn, verify as verify_fn
        from native_stand.students.motion_fixed_height import reward_snapshot as snapshot_fn, HEIGHTS
        heights = HEIGHTS
    root = Path(__file__).resolve().parents[1]
    name = new_experiment_name('motion_persistent184_remaining' if a.persistent_frontiers else 'motion_course_keys184_10000' if a.repair_assessment_keys else 'motion_fixed_height184_9953' if a.fixed_height_motion else 'motion_height144_engineering_recovery' if a.recover_from else 'motion_decoupled_height144_10000')
    folder = root / 'configs/experiments' / name
    parent = json.loads(a.parent_spec.read_text())
    poses = prepare_poses(parent, heights) if heights else prepare_poses(parent)
    session = os.environ['HAPI_SESSION_ID']
    s = build_fn(a.parent_spec, a.checkpoint, experiment=name, spec_path=folder/'spec.json',
              poses=poses, owner_session=session, recovery_parent=a.recover_from, storage_preflight_version=STORAGE_PREFLIGHT)
    verify_fn(s)
    phases = ('source_eval', 'smoke', 'formal', 'evaluate', 'play')
    book = dict(authorization='UNAUTHORIZED', experiment=name, hapi_session=session,
                limits=dict(zip(phases, [1,s['budget']['smoke_updates'],s['budget']['formal_updates'],0,0])), used={k:0 for k in phases},
                attempts={}, source_eval_wall_used_s=0., total_online_PPO_limit=10000,
                previous_experiment_consumption=s['motion_decoupled']['previous_consumption'])
    if a.recover_from:
        book['engineering_recovery'] = s['motion_decoupled']['engineering_recovery']
    result = dict(spec=str(folder/'spec.json'), run=str(root/'runs'/name), source_sha256=s['source_sha256'],
                  online_limit=10000, smoke=s['budget']['smoke_updates'], formal=s['budget']['formal_updates'], training_wall_s=None, storage_max_bytes=s['artifacts']['storage_max_bytes'],
                  levels=s['motion_decoupled']['state']['levels'], focus=s['motion_decoupled']['state']['focus'],
                  actor_dim=31, critic_dim=s['observation']['critic_dim'], performance=s['performance'], simulation=0, files_written=0)
    if a.write:
        claim = folder.parent / ('.motion_persistent_claims' if a.persistent_frontiers else '.motion_course_key_claims' if a.repair_assessment_keys else '.motion_fixed_height_claims' if a.fixed_height_motion else '.motion_decoupled_claims') / (sha(a.parent_spec) + '.json')
        previous = None
        if claim.exists():
            previous = json.loads(claim.read_text()); prior = Path(previous['spec'])
            prior_book = json.loads(prior.with_name('budget.json').read_text())
            consumed = bool(prior_book['attempts'] or any(prior_book['used'].values()) or prior.with_name('chain_submission.json').exists())
            if consumed and (not a.recover_from or prior.resolve() != a.recover_from.resolve()):
                raise ValueError('parent already consumed by submitted successor; do not duplicate')
        snapshot = audit(s['reward']); snapshot['height_task'] = snapshot_fn()
        from native_stand.students.motion_antioscillation import reward_snapshot as extension
        snapshot['runtime_reward_extension'] = extension(s['motion_timing']['configuration'])
        if a.persistent_frontiers:
            from native_stand.students.motion_persistent import height_reward_snapshot
            snapshot['height_frontier_reward'] = height_reward_snapshot()
        publish(folder, {'spec.json':s, 'budget.json':book, 'preparation.json':result,
                         'reward_snapshot.json':snapshot}, s['student_dependencies'], completion_marker=True)
        result['files_written'] = 5
        claim.parent.mkdir(exist_ok=True)
        from native_stand.command_v6.completion import write
        write(claim, dict(destination=name, spec=str(folder/'spec.json'), supersedes=previous,
                         recovery_parent=str(a.recover_from.resolve()) if a.recover_from else None))
        if a.authorize:
            from native_stand.command_v6.tools.register_authorization import register
            result['registration'] = register(dict(approval_id=('user_motion_persistent_remaining_20261024_' if a.persistent_frontiers else 'user_motion_decoupled10000_20261023_')+name,
                user_scope=s['authorization']['user_scope'], session=session,
                physical_gpu=s['device']['physical_index'], gpu_uuid=s['device']['uuid'],
                max_online_updates=s['budget']['smoke_updates']+s['budget']['formal_updates'], max_phase_wall_s=None,
                arms=[dict(spec=str(folder/'spec.json'), spec_sha256=sha(folder/'spec.json'),
                           budget_sha256=sha(folder/'budget.json'))]), current_session=session, write=True)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
