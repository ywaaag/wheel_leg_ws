"""Prepare4500 height/PD course with explicit remaining or newly authorized budget; never launch."""
import argparse
import fcntl
import json
import os
from pathlib import Path

from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish,verify_publication
from native_stand.command_v6.completion import write
from native_stand.students import motion_height_pd as course
from native_stand.students.data import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-spec',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--diagnostic-result',type=Path,required=True)
    p.add_argument('--online-limit',type=int,choices=(course.ONLINE,10000),default=course.ONLINE)
    replacement=p.add_mutually_exclusive_group()
    replacement.add_argument('--replace-settled',type=Path,help='new10000 authorization replaces a settled zero-online predecessor')
    replacement.add_argument('--supersede-unsubmitted',type=Path,
        help='replace the exclusive claim only if its old declaration is unauthorized and has no attempt/run')
    p.add_argument('--write',action='store_true');p.add_argument('--authorize',action='store_true')
    a=p.parse_args()
    if a.authorize and not a.write:p.error('--authorize requires --write')
    root=Path(__file__).resolve().parents[1];session=os.environ['HAPI_SESSION_ID']
    name=new_experiment_name('motion4500_parallel_height_pd'+str(a.online_limit));folder=root/'configs/experiments'/name
    spec=course.build(a.parent_spec,a.checkpoint,a.diagnostic_result,
        experiment=name,spec_path=folder/'spec.json',session=session,online=a.online_limit,replace_settled=a.replace_settled)
    course.verify(spec)
    from native_stand.command_v6.tools.storage_budget import check
    capacity=check(spec,root/'runs'/name/'formal','formal')
    result=dict(spec=str(folder/'spec.json'),run=str(root/'runs'/name),source_sha256=spec['source_sha256'],
        historical_used=4870,inflight_reserve=1,online_limit=a.online_limit,
        budget_scope=spec['motion_height_dr'].get('budget_scope','remaining_original10000'),
        smoke=5,formal=a.online_limit-5,wall_seconds=None,storage_preflight=capacity,
        initial_conditions=len(spec['student_evaluation']['cases']),simulation=0,PPO=0,files_written=0)
    if a.write:
        if capacity['status']!='PASS':raise ValueError(capacity)
        claim=root/'configs/experiments/.motion_height_pd_claims'/sha(a.parent_spec.resolve())
        previous=None
        if claim.exists():
            previous=json.loads(claim.read_text())
            oldpath=Path(previous['spec']).resolve()
            replacing=a.replace_settled or a.supersede_unsubmitted
            if not replacing or replacing.resolve()!=oldpath:
                raise ValueError('this interrupted-task remainder has already been prepared')
            verify_publication(oldpath,authorized=bool(a.replace_settled))
            oldspec=json.loads(oldpath.read_text());oldbook=json.loads(oldpath.with_name('budget.json').read_text())
            if a.supersede_unsubmitted and (oldbook['authorization']!='UNAUTHORIZED' or oldbook['attempts'] or any(oldbook['used'].values())
                    or oldpath.with_name('authorization_receipt.json').exists()
                    or (root/'runs'/oldspec['experiment']).exists()):
                raise ValueError('only a zero-attempt unauthorized draft can be superseded')
            label='replaced_settled' if a.replace_settled else 'superseded_unsubmitted'
            result[label]=dict(spec=str(oldpath),sha256=sha(oldpath),
                reason='new10000 explicit user authorization' if a.replace_settled else 'source evaluation needs finite wall allowance',
                prior_used=oldbook['used'],prior_source_wall_s=oldbook.get('source_eval_wall_used_s',0.))
        elif a.replace_settled:raise ValueError('settled predecessor must own the existing exclusive claim')
        from native_stand.students.motion_reward_audit import audit
        from native_stand.students.motion_posture_preparation import runtime_reward_snapshot
        from native_stand.students.motion_antioscillation import reward_snapshot
        from native_stand.students.motion_height import reward_snapshot as height_snapshot
        snapshot=audit(spec['reward']);snapshot.update(posture_reward=runtime_reward_snapshot(),
            runtime_reward_extension=reward_snapshot(spec['motion_timing']['configuration']),
            fixed_height_reward=height_snapshot(),optimizer_auxiliary_losses=spec['student_training']['source_policy_retention'])
        signals=course.learning_signal_audit(spec)
        snapshot['learning_signal_audit']=signals
        phases=('source_eval','smoke','formal','evaluate','play')
        book=dict(authorization='UNAUTHORIZED',experiment=name,hapi_session=session,
            limits=dict(zip(phases,(1,5,a.online_limit-5,0,0))),used={k:0 for k in phases},attempts={},
            source_eval_wall_used_s=0.,total_online_PPO_limit=a.online_limit,
            prior_task_consumption=dict(original_online_limit=10000,historical_smoke=5,
                historical_formal=4865,conservative_inflight_reserve=1,
                charged_updates_after_selected_checkpoint=365,uncheckpointed_charged=65,
                source_checkpoint_update=4500,parent_budget=spec['motion_height_dr']['original_budget'],
                parent_budget_sha256=spec['motion_height_dr']['original_budget_sha256']))
        if a.replace_settled:book['prior_parallel_attempt']=result['replaced_settled']
        publish(folder,{'spec.json':spec,'budget.json':book,'preparation.json':result,'reward_snapshot.json':snapshot,
            'learning_signal_audit.json':signals},
            spec['student_dependencies'],completion_marker=True)
        claim.parent.mkdir(exist_ok=True)
        claim_data=dict(spec=str(folder/'spec.json'),online=a.online_limit,
            parent_budget_sha256=spec['motion_height_dr']['original_budget_sha256'])
        with (claim.parent/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if previous is None:
                with claim.open('x') as f:json.dump(claim_data,f)
            else:
                if json.loads(claim.read_text())!=previous:raise ValueError('remainder claim changed')
                claim_data[label]=result[label]
                write(claim,claim_data)
        if a.authorize:
            from native_stand.command_v6.tools.register_authorization import register
            result['registration']=register(dict(approval_id='user_parallel_height_PD_20261026_'+name,
                user_scope=spec['authorization']['user_scope'],session=session,physical_gpu=0,gpu_uuid=spec['device']['uuid'],
                max_online_updates=a.online_limit,max_phase_wall_s=None,
                arms=[dict(spec=str(folder/'spec.json'),spec_sha256=sha(folder/'spec.json'),budget_sha256=sha(folder/'budget.json'))]),
                current_session=session,write=True)
        result['files_written']=6
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
