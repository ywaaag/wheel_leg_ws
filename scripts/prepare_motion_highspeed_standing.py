"""Prepare/authorize the explicit9995 Actor2048 continuation; never launches."""
import argparse
import fcntl
import json
import os
from pathlib import Path

from native_stand.command_v6.completion import write
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish
from native_stand.command_v6.tools.storage_budget import check
from native_stand.students import motion_highspeed_standing as course
from native_stand.students.data import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-spec',type=Path,required=True)
    parser.add_argument('--original-checkpoint',type=Path,required=True)
    parser.add_argument('--initialization',type=Path,required=True)
    parser.add_argument('--recover-zero-attempt',type=Path,
        help='exact settled prelaunch missing obsolete dependency failure,0attempt/0PPO; no additional budget')
    parser.add_argument('--write',action='store_true');parser.add_argument('--authorize',action='store_true')
    args=parser.parse_args()
    if args.authorize and not args.write:parser.error('--authorize requires --write')
    root=Path(__file__).resolve().parents[1];session=os.environ['HAPI_SESSION_ID']
    name=new_experiment_name('motion9995_actor2048_highspeed_standing10000')
    folder=root/'configs/experiments'/name
    spec=course.build(args.parent_spec,args.original_checkpoint,args.initialization,
        experiment=name,spec_path=folder/'spec.json',session=session,recover_zero_attempt=args.recover_zero_attempt)
    course.verify(spec)
    capacities={phase:check(spec,root/'runs'/name/phase,phase) for phase in ('source_eval','smoke','formal')}
    result=dict(spec=str(folder/'spec.json'),run=str(root/'runs'/name),source=spec['source_checkpoint'],
        source_sha256=spec['source_sha256'],original9995_sha256=course.SOURCE_SHA,
        online_limit=10000,smoke=5,formal=9995,wall_seconds=None,capacity=capacities,
        scored_conditions=len(spec['student_evaluation']['cases']),simulation=0,PPO=0,files_written=0)
    if args.write:
        if any(c['status']!='PASS' for c in capacities.values()):raise ValueError(capacities)
        claims=root/'configs/experiments/.motion_highspeed_standing_claims';claims.mkdir(exist_ok=True)
        claim=claims/course.SOURCE_SHA
        with (claims/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            previous=json.loads(claim.read_text()) if claim.exists() else None
            if previous and (not args.recover_zero_attempt or previous['spec']!=str(args.recover_zero_attempt.resolve())):
                raise ValueError('9995 Actor2048 authorization already prepared: '+str(claim))
            from native_stand.students.motion_reward_audit import audit
            from native_stand.students.motion_posture_preparation import runtime_reward_snapshot
            from native_stand.students.motion_antioscillation import reward_snapshot
            from native_stand.students.motion_height import reward_snapshot as height_snapshot
            from native_stand.students.motion_mcu_course import execution_reward_snapshot
            snapshot=audit(spec['reward']);signals=course.learning_signal_audit(spec)
            snapshot.update(posture_reward=runtime_reward_snapshot(),
                runtime_reward_extension=reward_snapshot(spec['motion_timing']['configuration']),
                fixed_height_reward=height_snapshot(),mcu_execution=execution_reward_snapshot(),
                optimizer_auxiliary_losses=spec['student_training']['source_policy_retention'],learning_signal_audit=signals)
            phases=('source_eval','smoke','formal','evaluate','play')
            book=dict(authorization='UNAUTHORIZED',experiment=name,hapi_session=session,
                limits=dict(zip(phases,(1,5,9995,0,0))),used={k:0 for k in phases},attempts={},
                source_eval_wall_used_s=0.,total_online_PPO_limit=10000,
                prior_task_consumption=dict(parent_online_used=10000,separate_history_not_refunded=True))
            if args.recover_zero_attempt:
                book['prelaunch_failure']=spec['motion_highspeed_standing']['prelaunch_recovery']
            publish(folder,{'spec.json':spec,'budget.json':book,'preparation.json':result,
                'reward_snapshot.json':snapshot,'learning_signal_audit.json':signals},
                spec['student_dependencies'],completion_marker=True)
            data=dict(spec=str(folder/'spec.json'),source9995=course.SOURCE_SHA,online_limit=10000)
            if previous:
                data['replaces_zero_attempt']=previous;write(claim,data)
            else:
                with claim.open('x') as stream:json.dump(data,stream)
        if args.authorize:
            from native_stand.command_v6.tools.register_authorization import register
            result['registration']=register(dict(approval_id='user_highspeed_standing2048_20261028_'+name,
                user_scope=spec['authorization']['user_scope'],session=session,physical_gpu=0,gpu_uuid=spec['device']['uuid'],
                max_online_updates=10000,max_phase_wall_s=None,
                arms=[dict(spec=str(folder/'spec.json'),spec_sha256=sha(folder/'spec.json'),budget_sha256=sha(folder/'budget.json'))]),
                current_session=session,write=True)
        result['files_written']=6
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
