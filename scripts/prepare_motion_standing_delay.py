"""Prepare the explicit Actor2048 standing20/5/20ms recipe; never launches."""
import argparse
import fcntl
import json
import os
from pathlib import Path

from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish
from native_stand.command_v6.tools.storage_budget import check
from native_stand.students import motion_standing_delay as course
from native_stand.students.data import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-spec',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--replace-unlaunched',type=Path,
        help='replace exact zero-attempt declaration after user changes motor response tau; never reuse its authorization')
    parser.add_argument('--recover-preppo',type=Path,
        help='exact settled zero-PPO protocol-dispatch failure; preserve history and10000 total ceiling')
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--authorize',action='store_true')
    args=parser.parse_args()
    if args.authorize and not args.write:parser.error('--authorize requires --write')
    root=Path(__file__).resolve().parents[1];session=os.environ['HAPI_SESSION_ID']
    name=new_experiment_name('motion3896_standing_sensor20_target5_motor20_10000')
    folder=root/'configs/experiments'/name
    spec=course.build(args.parent_spec,args.checkpoint,experiment=name,
        spec_path=folder/'spec.json',session=session,replaces_unlaunched=args.replace_unlaunched,
        recover_preppo=args.recover_preppo)
    course.verify(spec)
    capacity={p:check(spec,root/'runs'/name/p,p) for p in ('source_eval','smoke','formal')}
    result=dict(spec=str(folder/'spec.json'),run=str(root/'runs'/name),source=spec['source_checkpoint'],
        source_sha256=spec['source_sha256'],online_limit=course.ONLINE,smoke=5,formal=course.FORMAL,
        wall_seconds=None,capacity=capacity,scored_conditions=len(spec['student_evaluation']['cases']),
        critic_dim=411,simulation=0,PPO=0,files_written=0)
    if args.write:
        if any(v['status']!='PASS' for v in capacity.values()):raise ValueError(capacity)
        claims=root/'configs/experiments/.motion_standing_delay_claims';claims.mkdir(exist_ok=True)
        claim=claims/spec['source_sha256']
        with (claims/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            previous=json.loads(claim.read_text()) if claim.exists() else None
            replacing=args.replace_unlaunched or args.recover_preppo
            if previous and (not replacing or previous['spec']!=str(replacing.resolve())):
                raise ValueError('standing20/5/20 authorization already prepared: '+str(claim))
            from native_stand.students.motion_reward_audit import audit
            from native_stand.students.motion_posture_preparation import runtime_reward_snapshot
            from native_stand.students.motion_antioscillation import reward_snapshot
            from native_stand.students.motion_mcu_course import execution_reward_snapshot
            signals=course.learning_signal_audit(spec)
            snapshot=audit(spec['reward'])
            snapshot.update(spec_source_sha256=spec['source_sha256'],
                posture_reward=runtime_reward_snapshot(),
                runtime_reward_extension=reward_snapshot(spec['motion_timing']['configuration']),
                mcu_execution=execution_reward_snapshot(),execution_profile=course.execution_profile(),
                learning_signal_audit=signals,critic_contract=course.contract(),
                formula_source_sha256={str(p):sha(p) for p in (
                    Path(course.__file__),root/'src/native_stand/students/motion_standing_delay_runtime.py')})
            phases=('source_eval','smoke','formal','evaluate','play')
            book=dict(authorization='UNAUTHORIZED',experiment=name,hapi_session=session,
                limits=dict(zip(phases,(1,5,course.FORMAL,0,0))),used={p:0 for p in phases},attempts={},
                source_eval_wall_used_s=0.,total_online_PPO_limit=course.ONLINE,
                prior_task_consumption=dict(parent_online_used=spec['motion_standing_delay']['historical_online_updates'],
                    stopped_parent_remaining_not_transferred=True,separate_history_not_refunded=True))
            if args.recover_preppo:book['preppo_recovery']=spec['motion_standing_delay']['preppo_recovery']
            publish(folder,{'spec.json':spec,'budget.json':book,'preparation.json':result,
                'reward_snapshot.json':snapshot,'learning_signal_audit.json':signals},
                spec['student_dependencies'],completion_marker=True)
            data=dict(spec=str(folder/'spec.json'),online_limit=course.ONLINE)
            if previous:
                from native_stand.command_v6.completion import write
                data['replaces_unlaunched']=previous
                write(claim,data)
            else:
                with claim.open('x') as stream:json.dump(data,stream)
        if args.authorize:
            from native_stand.command_v6.tools.register_authorization import register
            result['registration']=register(dict(approval_id='user_standing20_target5_motor20_20261029_'+name,
                user_scope=spec['authorization']['user_scope'],session=session,physical_gpu=0,gpu_uuid=spec['device']['uuid'],
                max_online_updates=course.ONLINE,max_phase_wall_s=None,
                arms=[dict(spec=str(folder/'spec.json'),spec_sha256=sha(folder/'spec.json'),budget_sha256=sha(folder/'budget.json'))]),
                current_session=session,write=True)
        result['files_written']=6
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
