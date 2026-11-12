"""旧多轮提交流程已于2026-11-05退役；本模块只保留submission_readiness函数。

公开入口统一为 `wheel-leg train/evaluate/...`（`native_stand.cli`）。
旧 `--check-submission` / `--launch` / `--next-spec` 多轮流程不再提供。
"""
import json
import os
from pathlib import Path


def submission_readiness(args, spec, preflight, health=None):
    from .domain.submission_contract import metadata
    from .tools.health_check import submission_health
    output = args.output_dir / 'formal' if args.formal_only else args.output_dir
    result = metadata(spec, target_session=args.hapi_session,
        current_session=os.environ.get('HAPI_SESSION_ID'), physical_gpu=args.physical_gpu,
        output_available=not output.exists(), duplicate_submission=(
            not args.formal_only and not args.evaluation_only and
            args.spec.with_name('chain_submission.json').exists()))
    result.update(configuration_status='VALID' if preflight['status']=='PASS' else 'NOT_READY',
                  resource_verified=False, submit_ready=False)
    if preflight['status']!='PASS':
        result['errors'].append('preflight prerequisite: '+str(preflight.get('errors',preflight['status'])))
        result['reason_code']=preflight.get('reason_code','MISSING_EVIDENCE')
    if not result['errors'] and health is not None:
        try:
            result['resource_review']=submission_health(health,
                current_session=os.environ.get('HAPI_SESSION_ID'),target_session=args.hapi_session,
                phase='evaluate' if args.evaluation_only else 'formal' if args.formal_only or spec.get('motion_training',{}).get('wiring_reuse') else 'smoke' if spec.get('motion_training',{}).get('partial_smoke_recovery') else 'source_eval',
                physical_gpu=args.physical_gpu,resource_policy=args.resource_policy)
            expected_uuid=spec.get('device',{}).get('uuid')
            devices=(health.get('gpu') or {}).get('devices',[])
            if expected_uuid and not any(row['index']==args.physical_gpu and
                                        row['uuid']==expected_uuid for row in devices):
                raise ValueError('physical GPU UUID differs from spec')
            large_recovery=spec.get('student_training',{}).get('version') in ('student_recovery_large_stop_ppo_v1','student_recovery_contact_quality_ppo_v1')
            if spec.get('recovery_course') or large_recovery:
                selected=[d for d in devices if d['index']==args.physical_gpu]
                if len(selected)!=1:raise ValueError('Recovery course GPU inventory missing')
                free=float(selected[0]['memory_total_mib'])-float(selected[0]['memory_used_mib'])
                minimum=6144 if large_recovery else 4096
                if free<minimum:raise ValueError(f'Recovery requires{minimum}MiB available GPU memory; no preemption/wait/switch')
                result['resource_review']['course_available_memory_mib']=free
                result['resource_review']['course_minimum_available_memory_mib']=minimum
            result['resource_verified']=True
            result['submit_ready']=True
        except ValueError as exc:
            result['errors'].append(str(exc))
            result['reason_code']='IDENTITY_ERROR' if 'UUID differs' in str(exc) else 'MISSING_EVIDENCE'
    result.update(config_valid=preflight.get('config_valid',preflight.get('cpu_valid',preflight['status']=='PASS')),
        authorized=preflight.get('authorized',preflight['status']=='PASS'),
        prerequisites_ready=preflight.get('prerequisites_ready',preflight['status']=='PASS'),
        resource_ready=result['resource_verified'] if health is not None else None)
    result['status']=('NOT_READY' if result['errors'] else
                      'READY' if result['submit_ready'] else 'READY_METADATA')
    if result['status']=='READY_METADATA':
        result['reason_code']='MISSING_EVIDENCE'
        result['action']='use --check-submission for one read-only resource check'
    return result
