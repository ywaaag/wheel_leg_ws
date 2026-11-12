"""Bounded, independent MCU-baseline capability probe through the existing owner."""
import copy
import json
from pathlib import Path
from .data import sha

VERSION='motion31_mcu_capability_probe_v1'
SPEEDS=(.5,1.,1.5,2.,3.,4.)


def cases(template):
    rows=[]
    # Compatible nominal commands run in three groups of8, not24 serial worlds.
    for axis in (0,1):
        for speed in SPEEDS:
            for sign in (1,-1):
                cmd=[0.,0.,.34];cmd[axis]=sign*speed
                rows.append(dict(template,case_id=f'mcu_probe_{len(rows):02d}',command=cmd,
                    _mcu_variant=0,_mcu_revision=3,_timing_level=4))
    for variant in (1,2):
        rows.append(dict(template,case_id=f'mcu_probe_{len(rows):02d}',command=[0.,0.,.34],
            _mcu_variant=variant,_mcu_revision=3,_timing_level=4))
    return rows


def build(parent,checkpoint,*,experiment,spec_path,owner_session,recover_from=None):
    from .motion_mcu_course import verify as verify_parent,effective_spec
    p=Path(parent).resolve();old=json.loads(p.read_text());verify_parent(old)
    root=p.parents[3];run=root/'runs'/old['experiment']
    terminal=json.loads((run/'chain_terminal_report.json').read_text())
    book=json.loads(p.with_name('budget.json').read_text())
    if terminal['status']!='STOPPED_BOUNDED' or terminal.get('failure') or any(r['status']=='RUNNING' for r in book['attempts'].values()):
        raise ValueError('probe requires settled user-stopped MCU parent')
    cp=Path(checkpoint).resolve()
    if str(cp)!=old['source_checkpoint'] or sha(cp)!=old['source_sha256']:
        raise ValueError('probe uses original evaluated1500, not latest training checkpoint')
    raw=run/'source_eval/evaluation_00.json';a=json.loads(raw.read_text());trace=raw.parent/a['trace']['path']
    if (a['checkpoint_sha256']!=old['source_sha256'] or a['case']['command']!=[0.,0.,.34]
        or len(a['per_env'])!=8 or not all(e.get('accepted') is True and e.get('controlled_seconds',0)>=30 for e in a['per_env'])
        or sha(trace)!=a['trace']['sha256']):raise ValueError('normal standing reuse incomplete or conflicting')
    source_registry=json.loads((raw.parent/'candidate_registry.json').read_text())
    if source_registry[0]['evaluations'][0]['sha256']!=sha(raw):raise ValueError('source evaluation SHA differs')
    s=effective_spec(old,3)
    s.update(experiment=experiment,evaluation_only=True)
    s['motion_training']['spec_path']=str(Path(spec_path).resolve())
    template=s['student_evaluation']['cases'][0]
    s['student_evaluation']['cases']=cases(template)
    s['student_evaluation']['max_output_bytes']=512*2**20
    s['performance'].update(evaluation_group_commands=True,evaluation_group_max_commands=8)
    s['budget'].update(source_eval_updates=0,smoke_updates=0,formal_updates=0,formal_samples=0,sample_limit=0,
        source_eval_wall_s=0,smoke_wall_s=0,formal_wall_s=0,evaluation_wall_s=3600,chain_wall_s=3600)
    s['artifacts'].update(run_name=experiment,storage_max_bytes=1024*2**20,terminal_reserve_bytes=128*2**20)
    s['student_training']['storage_max_bytes']=s['artifacts']['storage_max_bytes']
    inputs=[p,p.with_name('budget.json'),run/'chain_terminal_report.json',raw,trace]
    s['mcu_baseline_probe']=dict(version=VERSION,parent_spec=str(p),parent_spec_sha256=sha(p),
        inputs={str(q):sha(q) for q in inputs},normal_zero_source=str(raw),
        historical_online_used=book['used']['smoke']+book['used']['formal'],
        original_online_limit=5000,remaining_online=5000-book['used']['smoke']-book['used']['formal'],
        task='24 signed linear/yaw commands through4, plus two zero initial/execution conditions; PPO0',
        new_training_authorized_by_user=True,probe_itself_runs_training=False)
    if recover_from is not None:
        previous=Path(recover_from).resolve();before=json.loads(previous.read_text())
        if before.get('mcu_baseline_probe',{}).get('recovery'):
            raise ValueError('probe storage recovery is one explicit continuation, not recursive retry')
        verify(before);failed_run=root/'runs'/before['experiment']
        term=json.loads((failed_run/'terminal_report.json').read_text())
        failed_book=json.loads(previous.with_name('budget.json').read_text())
        if (term['status']!='FAILED' or term.get('failure',{}).get('message')!='Handoff cumulative storage cap exceeded'
            or failed_book['used']['formal'] or failed_book['used']['smoke']
            or any(r['status']=='RUNNING' for r in failed_book['attempts'].values())):
            raise ValueError('only settled zero-PPO probe storage failure can resume')
        completed=[];complete_ids=set()
        for row in term['conditions']:
            if row['status']!='COMPLETED':continue
            raw=failed_run/row['result'];payload=json.loads(raw.read_text());trace=raw.parent/payload['trace']['path']
            if (payload['case']!=row['condition'] or payload['checkpoint_sha256']!=s['source_sha256']
                or payload['spec_sha256']!=sha(previous) or payload.get('model_unchanged') is not True
                or len(payload['per_env'])!=8 or sha(trace)!=payload['trace']['sha256']):
                raise ValueError('completed probe evidence cannot be reused')
            completed.append(dict(path=str(raw),sha256=sha(raw),trace=str(trace),trace_sha256=sha(trace)))
            complete_ids.add(payload['case']['case_id'])
        if len(completed)!=18:raise ValueError('storage recovery expects exactly18 immutable completed cases')
        s['student_evaluation']['cases']=[r for r in s['student_evaluation']['cases'] if r['case_id'] not in complete_ids]
        s['mcu_baseline_probe']['recovery']=dict(version='motion_mcu_probe_storage_completion_v1',
            parent_spec=str(previous),parent_spec_sha256=sha(previous),parent_run=str(failed_run),
            reused_results=completed,remaining_conditions=8,
            historical_wall_s=sum(float(r.get('wall_s') or 0) for r in failed_book['attempts'].values()),
            historical_online=0,orphan_trace_not_used='trace18 has no complete identity-bound JSON; resampled')
        s['mcu_baseline_probe']['storage_profile']='compact_mcu_scoring_control_and_motor_v1'
        s['student_dependencies'].update({str(previous):sha(previous),str(failed_run/'terminal_report.json'):sha(failed_run/'terminal_report.json')})
    s['student_dependencies'].update({str(q):sha(q) for q in inputs})
    s['authorization']={'status':'UNAUTHORIZED','hapi_session':owner_session,
        'user_scope':'User2026-10-25 explicitly authorizes stopping current chain, original1500 MCU capability probe up to +/-4m/s and +/-4rad/s,8scored per26 new conditions,PPO0,1h/1GiB,GPU0 shared,current owner notification; subsequent measured curriculum may train within remaining original5000.'}
    s['notifications']={'enabled':True,'hapi_session':owner_session}
    return s


def verify(spec):
    p=spec['mcu_baseline_probe']
    if p.get('version')!=VERSION or spec.get('evaluation_only') is not True:
        raise ValueError('MCU probe cannot authorize online updates')
    expected=build(p['parent_spec'],spec['source_checkpoint'],experiment=spec['experiment'],
        spec_path=spec['motion_training']['spec_path'],owner_session=spec['authorization'].get('hapi_session'),
        recover_from=p.get('recovery',{}).get('parent_spec'))
    for key in ('authorization','resources','notifications'):expected[key]=spec[key]
    if spec!=expected:raise ValueError('MCU capability probe undeclared drift')
    return True
