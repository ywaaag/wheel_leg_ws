"""CPU-only plan/publication for the user-authorized5000-update MCU adaptation."""
import argparse
import json
import os
from pathlib import Path
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish
from native_stand.students.motion_mcu_course import build,verify,STORAGE,execution_reward_snapshot
from native_stand.students.motion_mcu_execution import profile
from native_stand.students.motion_reward_audit import audit
from native_stand.students.motion_antioscillation import reward_snapshot
from native_stand.students.data import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-spec',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--write',action='store_true');p.add_argument('--authorize',action='store_true')
    p.add_argument('--baseline-probe',action='store_true',help='Independent bounded original1500 PPO0 through +/-4, after settled stop')
    p.add_argument('--measured-assessment',type=Path,help='Completed same-profile source assessment; creates Critic312 measured-start remaining-budget training')
    p.add_argument('--posture-plan',action='store_true',help='CPU-only fixed034 posture/rehearsal design; never creates a launchable spec or authorization')
    p.add_argument('--posture-training',action='store_true',help='New authorized10000 fixed034 posture/nominal-retention recipe; default zero-write')
    p.add_argument('--design-output-dir',type=Path,help='New artifact directory for posture CPU design and audit, with --write')
    p.add_argument('--recover-measured',action='store_true',help='Exact settled update500 posteval interface recovery, original5000 remaining budget only')
    p.add_argument('--preformal-storage-failure',type=Path,help='Settled zero-attempt storage rejection; same remainder, bounded1500 block, no storage increase')
    p.add_argument('--complete-probe',type=Path,help='Only complete remaining8 after exact settled18-case storage failure')
    p.add_argument('--replace-unsubmitted',action='store_true',help='Replace only the unused zero-attempt preparation; old files retained and blocked by the active claim')
    a=p.parse_args()
    if a.authorize and not a.write:p.error('--authorize requires --write')
    if a.posture_plan:
        if a.authorize or a.posture_training or a.baseline_probe or a.recover_measured or a.complete_probe or a.preformal_storage_failure or a.replace_unsubmitted:
            p.error('--posture-plan is CPU design only; cannot authorize/launch/recover or replace an existing declaration')
        if not a.measured_assessment:p.error('--posture-plan requires --measured-assessment')
        if a.write != bool(a.design_output_dir):p.error('write posture design with both --write and --design-output-dir')
        from native_stand.students.motion_posture_preparation import prepare
        print(json.dumps(prepare(a.parent_spec,a.checkpoint,a.measured_assessment,
            a.design_output_dir if a.write else None),indent=2,ensure_ascii=False))
        return
    if a.design_output_dir:p.error('--design-output-dir requires --posture-plan')
    if a.preformal_storage_failure and not a.recover_measured:p.error('--preformal-storage-failure requires --recover-measured')
    root=Path(__file__).resolve().parents[1];session=os.environ['HAPI_SESSION_ID']
    if a.posture_training and (not a.measured_assessment or a.baseline_probe or a.recover_measured or a.replace_unsubmitted or a.complete_probe or a.preformal_storage_failure):p.error('posture training requires assessment and a fresh separate declaration')
    if not a.posture_training and sum(bool(v) for v in (a.baseline_probe,a.measured_assessment,a.recover_measured))>1:p.error('probe, measured training and recovery are separate declarations')
    name=new_experiment_name('motion1500_posture_retention312_10000' if a.posture_training else 'motion500_mcu312_interface_resume' if a.recover_measured else 'motion1500_mcu_measured312' if a.measured_assessment else 'motion1500_mcu_capability_probe' if a.baseline_probe else 'motion1500_mcu_standing5000');folder=root/'configs/experiments'/name
    builder=build
    if a.baseline_probe:
        from native_stand.students.motion_mcu_probe import build as builder
    kwargs=dict(recover_from=a.complete_probe) if a.baseline_probe else {}
    if a.complete_probe and not a.baseline_probe:p.error('--complete-probe requires --baseline-probe')
    if a.posture_training:
        from native_stand.students.motion_posture_preparation import build_training
        s=build_training(a.parent_spec,a.checkpoint,a.measured_assessment,experiment=name,spec_path=folder/'spec.json',owner_session=session)
    elif a.recover_measured:
        from native_stand.students.motion_mcu_measured_io import build_recovery
        s=build_recovery(a.parent_spec,a.checkpoint,experiment=name,spec_path=folder/'spec.json',owner_session=session,
            preformal_failure=a.preformal_storage_failure)
    elif a.measured_assessment:
        from native_stand.students.motion_mcu_measured_io import build as measured_builder
        s=measured_builder(a.parent_spec,a.measured_assessment,experiment=name,spec_path=folder/'spec.json',owner_session=session)
        if str(a.checkpoint.resolve())!=s['source_checkpoint']:raise ValueError('measured source must remain original1500')
    else:s=builder(a.parent_spec,a.checkpoint,experiment=name,spec_path=folder/'spec.json',owner_session=session,**kwargs)
    verify(s)
    phases=['source_eval','smoke','formal','evaluate','play']
    book=dict(authorization='UNAUTHORIZED',experiment=name,hapi_session=session,
        limits=dict(zip(phases,[0,0,0,1,0] if a.baseline_probe else [1,5,4995,0,0])),used={k:0 for k in phases},attempts={},
        source_eval_wall_used_s=0.,total_online_PPO_limit=0 if a.baseline_probe else 5000,
        previous_experiment_consumption=s['mcu_execution']['old_consumption_separate'])
    snapshot=audit(s['reward']);snapshot['runtime_reward_extension']=reward_snapshot(s['motion_timing']['configuration'])
    snapshot['mcu_execution']=execution_reward_snapshot()
    if a.posture_training:
        from native_stand.students.motion_posture_preparation import runtime_reward_snapshot
        snapshot['posture_reward']=runtime_reward_snapshot()
        snapshot['optimizer_auxiliary_losses']={'source_policy_retention':s['student_training']['source_policy_retention']}
    result=dict(spec=str(folder/'spec.json'),run=str(root/'runs'/name),online=5000,smoke=5,formal=4995,
        formal_critic_warmup=20,envs=4096,steps_per_env=64,training_wall_s=None,
        source_sha256=s['source_sha256'],storage_max_bytes=STORAGE,simulation=0,files_written=0)
    if a.baseline_probe:
        result.update(online=0,smoke=0,formal=0,formal_critic_warmup=0,envs=64,
            conditions=len(s['student_evaluation']['cases']),scored_envs_per_condition=8,physical_envs=64,wall_limit_s=3600,
            storage_max_bytes=s['artifacts']['storage_max_bytes'],
            remaining_training_online=s['mcu_baseline_probe']['remaining_online'])
    if a.posture_training:
        book['limits'].update(smoke=5,formal=9995);book['total_online_PPO_limit']=10000
        book['prior_task_consumption_separate']={'used':1296,'original_limit':5000,'unspent_not_transferred':3704}
        result.update(online=10000,smoke=5,formal=9995,critic_dim=312,storage_max_bytes=s['artifacts']['storage_max_bytes'])
    elif a.measured_assessment or a.recover_measured:
        formal=s['budget']['formal_updates'];smoke=s['budget']['smoke_updates']
        book['limits'].update(source_eval=0 if a.recover_measured else 1,smoke=smoke,formal=formal);book['total_online_PPO_limit']=formal+smoke
        book['historical_online_updates']=s['mcu_measured']['historical_online_used'];book['original_online_limit']=5000
        result.update(online=formal+smoke,smoke=smoke,formal=formal,formal_critic_warmup=s['student_training']['warmup_updates_per_phase'],storage_max_bytes=s['artifacts']['storage_max_bytes'],
            initial_capability=s['mcu_measured']['initial_capability'],critic_dim=312)
        if a.preformal_storage_failure:
            book['unallocated_online_updates']=2870
            result.update(remaining_authorized_online_updates=4370,unallocated_online_updates=2870)
    from native_stand.command_v6.tools.storage_budget import check as storage_check
    result['storage_preflight']=storage_check(s,root/'runs'/name/('evaluate' if a.baseline_probe else 'formal'),'evaluate' if a.baseline_probe else 'formal')
    if a.write and result['storage_preflight']['status']!='PASS':raise ValueError(result['storage_preflight'])
    if a.write:
        claim=root/'configs/experiments'/('.motion_mcu_probe_claims' if a.baseline_probe else '.motion_mcu_claims')/(s['mcu_baseline_probe']['parent_spec_sha256'] if a.baseline_probe else s['source_sha256'])
        if a.complete_probe:claim=claim.with_name(claim.name+'_storage_completion')
        if a.measured_assessment and not a.posture_training:claim=root/'configs/experiments/.motion_mcu_measured_claims'/s['mcu_measured']['assessment_sha256']
        if a.posture_training:claim=root/'configs/experiments/.motion_mcu_posture_claims'/s['mcu_posture']['parent_spec_sha256']
        if a.recover_measured:claim=root/'configs/experiments/.motion_mcu_measured_recovery_claims'/sha(a.parent_spec.resolve())
        previous=None
        if claim.exists():
            previous=json.loads(claim.read_text());old_path=Path(previous['spec']);book_before=json.loads(old_path.with_name('budget.json').read_text())
            safe_storage_replacement=(a.preformal_storage_failure and old_path==a.preformal_storage_failure.resolve())
            if not safe_storage_replacement and (not a.replace_unsubmitted or book_before['attempts'] or any(book_before['used'].values())
                or old_path.with_name('chain_submission.json').exists() or (root/'runs'/previous['experiment']).exists()):
                raise ValueError('MCU authorization already prepared/submitted; never duplicate a consumed attempt')
        publish(folder,{'spec.json':s,'budget.json':book,'preparation.json':result,'reward_snapshot.json':snapshot},
                s['student_dependencies'],completion_marker=True)
        claim.parent.mkdir(exist_ok=True)
        value={'spec':str(folder/'spec.json'),'experiment':name,'online_limit':0 if a.baseline_probe else s['budget']['smoke_updates']+s['budget']['formal_updates'],'supersedes':previous}
        if previous is None:
            with claim.open('x') as f:json.dump(value,f)
        else:
            from native_stand.command_v6.completion import write
            write(claim,value)
        if a.authorize:
            from native_stand.command_v6.tools.register_authorization import register
            result['registration']=register(dict(approval_id='user_motion_mcu5000_20261025_'+name,
                user_scope=s['authorization']['user_scope'],session=session,physical_gpu=0,gpu_uuid=s['device']['uuid'],
                max_online_updates=0 if a.baseline_probe else s['budget']['smoke_updates']+s['budget']['formal_updates'],max_phase_wall_s=3600 if a.baseline_probe else None,
                arms=[dict(spec=str(folder/'spec.json'),spec_sha256=sha(folder/'spec.json'),budget_sha256=sha(folder/'budget.json'))]),
                current_session=session,write=True)
        result['files_written']=5
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
