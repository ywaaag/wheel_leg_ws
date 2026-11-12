"""File-boundary measured MCU course preparation and exact source-evidence reuse."""
import copy,json
from pathlib import Path
from .data import sha
from . import motion_mcu_measured as course
RESUME_VERSION='motion_mcu_measured_posteval_resume_v1'


def analyze_probe(spec_path,run,output):
    """Apply the same training-side MCU/anti-oscillation scorer to frozen PPO0 arrays.

    The generic standalone evaluator emits its base student score. Keep those raw
    files untouched and persist this explicit derived scoring layer separately.
    """
    import numpy as np
    from .motion_antioscillation import assess as anti_assess
    from .motion_mcu_course import assess as mcu_assess,protocol
    p=Path(spec_path).resolve();s=json.loads(p.read_text());run=Path(run).resolve();out=Path(output).resolve()
    terminal=json.loads((run/'terminal_report.json').read_text())
    book=json.loads(p.with_name('budget.json').read_text())
    if terminal['status']!='COMPLETED' or terminal.get('failure') or any(a['status']=='RUNNING' for a in book['attempts'].values()):
        raise ValueError('capability probe must settle before source inference')
    if len(terminal['conditions'])!=len(s['student_evaluation']['cases']) or any(r['status']!='COMPLETED' for r in terminal['conditions']):
        raise ValueError('complete declared capability conditions required')
    records=[];inputs={str(p):sha(p),str(run/'terminal_report.json'):sha(run/'terminal_report.json')}
    source=Path(s['mcu_baseline_probe']['normal_zero_source']);a=json.loads(source.read_text())
    if a['protocol']!=protocol():raise ValueError('normal standing protocol differs')
    records.append(dict(path=str(source),sha256=sha(source),case=a['case'],per_env=a['per_env'],protocol=a['protocol']))
    inputs[str(source)]=sha(source)
    out.mkdir(parents=True,exist_ok=False)
    raw_rows=[(run/c['result'],expected,sha(p)) for c,expected in zip(terminal['conditions'],s['student_evaluation']['cases'])]
    recovery=s['mcu_baseline_probe'].get('recovery')
    if recovery:
        for previous in recovery['reused_results']:
            raw=Path(previous['path']);payload=json.loads(raw.read_text())
            if sha(raw)!=previous['sha256']:raise ValueError('reused raw changed')
            raw_rows.append((raw,payload['case'],recovery['parent_spec_sha256']))
    raw_rows.sort(key=lambda row:int(row[1]['case_id'].rsplit('_',1)[1]))
    if len(raw_rows)!=26:raise ValueError('combined capability matrix must contain26 new cases')
    for i,(raw,expected,expected_spec_sha) in enumerate(raw_rows):
        a=json.loads(raw.read_text());trace=raw.parent/a['trace']['path']
        if (a['case']!=expected or a['checkpoint_sha256']!=s['source_sha256'] or a['spec_sha256']!=expected_spec_sha
            or a.get('model_unchanged') is not True or len(a['per_env'])!=8 or sha(trace)!=a['trace']['sha256']):
            raise ValueError('probe raw/trace/Actor identity differs')
        derived=copy.deepcopy(a)
        with np.load(trace) as arrays:
            for eid,e in enumerate(derived['per_env']):
                anti_assess(e,arrays['control'][:,eid],
                    arrays['first_episode_valid'][:,eid]&(arrays['controller'][:,eid,0]==3),arrays['state'][:,eid])
                mcu_assess(e,arrays,eid,a['case'])
        derived.update(protocol=protocol(),accepted=all(e['accepted'] is True for e in derived['per_env']),
            passed_envs=sum(e['accepted'] is True for e in derived['per_env']),
            derived_from=dict(raw=str(raw),raw_sha256=sha(raw),trace=str(trace),trace_sha256=sha(trace),
                reason='same training-side MCU/anti-oscillation scoring missing from generic standalone output; no new simulation'))
        dest=out/f'evaluation_{i:02d}.json';dest.write_text(json.dumps(derived,indent=2,allow_nan=False)+'\n')
        records.append(dict(path=str(dest),sha256=sha(dest),case=derived['case'],per_env=derived['per_env'],protocol=derived['protocol']))
        inputs.update({str(raw):sha(raw),str(trace):sha(trace),str(dest):sha(dest)})
    initial=course.infer_stage(records)
    report=dict(version='motion_mcu_capability_derived_v1',probe_spec=str(p),probe_run=str(run),
        source_sha256=s['source_sha256'],inputs=inputs,records=records,initial_stage=initial,
        initial_state=course.decode(initial),new_simulation=0,new_PPO=0,
        passed_conditions=sum(course.passed(r) for r in records),total_conditions=27)
    dest=out/'assessment.json';dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def evidence(assessment):
    path=Path(assessment).resolve();a=json.loads(path.read_text())
    if a.get('version')!='motion_mcu_capability_derived_v1' or a['source_sha256']!='9de0a63d46548bd100882929ef19a4dd16897d91a44d3a250b833fd4350559a7':
        raise ValueError('explicit MCU1500 source assessment required')
    for p,h in a['inputs'].items():
        if sha(p)!=h:raise ValueError('source assessment input SHA differs: '+p)
    if len(a['records'])!=27 or course.infer_stage(a['records'])!=a['initial_stage']:
        raise ValueError('source capability matrix differs')
    return a


def recipe(source_sha,stage,formal,storage,timing,*,recovery=False):
    from .motion_training import declaration
    from .motion_mcu_privileged import network,VERSION
    r=declaration(source_sha,stage,reward_version='student_motion31_parallel_quality_v10')
    r.update(critic_privileged_version=VERSION,network_architecture=network(),course_version=course.VERSION,
        course_stage=stage,timing_configuration=copy.deepcopy(timing),critic='source184_zero_new_motor_columns312',
        optimizer='source_Adam_zero_new_motor_columns312',milestones=list(range(500,formal+1,500)),
        checkpoint_every_updates=100,storage_max_bytes=storage,smoke_wiring=course.smoke_profile(),
        warmup_updates_per_phase=20,
        selection_scope='measured MCU capability start; independent signed straight/yaw frontier through4, fixed034')
    if recovery:
        r.update(critic='source_same_semantic312_full_state',optimizer='source_full_Adam',
            warmup_updates_per_phase=0,formal_resume_offset=500,
            engineering_resume=dict(version=RESUME_VERSION,remaining_formal_updates=formal))
    return r


def validate_recipe(r,source_sha):
    from .motion_wheel_matching import configuration
    c=configuration();c['smooth_mix']=.75
    course.decode(r['stage'])
    recovery=r.get('engineering_resume')
    if recovery and recovery not in [dict(version=RESUME_VERSION,remaining_formal_updates=n) for n in (1500,4370)]:
        raise ValueError('undeclared measured posteval recovery budget')
    if ((not recovery and source_sha!='9de0a63d46548bd100882929ef19a4dd16897d91a44d3a250b833fd4350559a7')
        or c!=r['timing_configuration']):
        raise ValueError('measured MCU source/timing differs')
    if not 0<r['storage_max_bytes']<=4*2**30:raise ValueError('measured storage exceeds original4GiB')
    expected=recipe(source_sha,r['stage'],recovery['remaining_formal_updates'] if recovery else 4870,r['storage_max_bytes'],c,recovery=bool(recovery))
    if expected!=r:raise ValueError('measured MCU recipe undeclared drift')
    return r


def build(parent,assessment,*,experiment,spec_path,owner_session):
    from .motion_mcu_course import verify as verify_parent
    from .motion_mcu_privileged import network,contract
    p=Path(parent).resolve();old=json.loads(p.read_text());verify_parent(old)
    a=evidence(assessment);probe=json.loads(Path(a['probe_spec']).read_text())
    if probe['mcu_baseline_probe']['parent_spec']!=str(p):raise ValueError('assessment parent differs')
    used=probe['mcu_baseline_probe']['historical_online_used'];remaining=5000-used
    if used!=125 or remaining!=4875:raise ValueError('this measured continuation preserves actual125/5000 charge')
    root=p.parents[3];run=root/'runs'/old['experiment']
    probe_runs=[Path(a['probe_run'])]
    if probe['mcu_baseline_probe'].get('recovery'):probe_runs.append(Path(probe['mcu_baseline_probe']['recovery']['parent_run']))
    history_bytes=sum(q.stat().st_size for folder in (run,*probe_runs) for q in folder.rglob('*') if q.is_file() and not q.is_symlink())
    storage=4*2**30-history_bytes
    if storage<1024*2**20:raise ValueError('remaining cumulative output space insufficient')
    s=copy.deepcopy(old);s.update(experiment=experiment,evaluation_only=False,network_architecture=network(),
        student_interface=contract(),observation=dict(frame_dim=22,actor_dim=31,critic_dim=312))
    stage=a['initial_stage'];r=recipe(s['source_sha256'],stage,remaining-5,storage,old['motion_timing']['configuration'])
    s['student_training']=r;s['motion_training'].update(stage=stage,spec_path=str(Path(spec_path).resolve()),resume=None)
    s['motion_course']['configuration']={**course.configuration(),'formal_updates_max':remaining-5,'smoke_updates_max':5}
    s['mcu_measured']=dict(version=course.VERSION,parent_spec=str(p),parent_spec_sha256=sha(p),
        assessment=str(Path(assessment).resolve()),assessment_sha256=sha(assessment),
        initial_stage=stage,initial_capability=course.decode(stage),historical_online_used=used,
        remaining_online=remaining,original_online_limit=5000,historical_output_bytes=history_bytes,
        original_output_limit=4*2**30,
        trace_scope='all8env scoring/Actor readback and motor state; Critic312 training-only, not duplicated in evaluation trace')
    s['budget'].update(smoke_updates=5,formal_updates=remaining-5,formal_samples=(remaining-5)*4096*64,
        sample_limit=remaining*4096*64,formal_wall_s=None,chain_wall_s=None,
        source_eval_wall_s=600,smoke_wall_s=3600,evaluation_wall_s=7200)
    s['artifacts'].update(run_name=experiment,storage_max_bytes=storage)
    s['performance'].update(evaluation_group_commands=True,evaluation_group_max_commands=8)
    s['student_dependencies'].update({str(p):sha(p),str(Path(assessment).resolve()):sha(assessment)})
    s['authorization']={'status':'UNAUTHORIZED','hapi_session':owner_session,
        'user_scope':'User2026-10-25 authorizes original1500 measured-start MCU motion through signed4, same Actor31 and explicit Critic312, remaining4875 online(smoke5+formal4870 including20critic), original5000 cumulative/no refund, GPU0shared/unlimited training wall/current owner notification/TensorBoard6006.'}
    s['notifications']={'enabled':True,'hapi_session':owner_session}
    return course.effective_spec(s,stage)


def verify(spec):
    if spec['mcu_measured'].get('engineering_resume'):
        return verify_recovery(spec)
    c=spec['mcu_measured']
    expected=build(c['parent_spec'],c['assessment'],experiment=spec['experiment'],
        spec_path=spec['motion_training']['spec_path'],owner_session=spec['authorization'].get('hapi_session'))
    for k in ('authorization','resources','notifications'):expected[k]=spec[k]
    if expected!=spec:raise ValueError('measured MCU declaration differs')
    validate_recipe(spec['student_training'],spec['source_sha256']);return True


def reconstructed_boundary(old,source,candidate,persisted,symmetry,refinement):
    """Replay only CPU post-evaluation bookkeeping; never an optimizer step."""
    from .motion_refinement import advance
    from .motion_symmetry import update_mix
    if old.get('motion_differential'):
        from .motion_differential import update_mix
    state,decision=course.advance(source['motion_course_state'],candidate)
    if (decision['evidence_valid'] is not True or persisted['state']!=state
        or persisted['decision']!=decision):
        raise ValueError('completed MCU matrix does not reproduce persisted course boundary')
    rows=course.refinement_records(candidate['evaluations'],candidate['stage'])
    actual_symmetry=update_mix(source['symmetry_mix_state'],rows)
    actual_refinement,_=advance(source['reward_refinement_state'],rows,old['reward']['refinement'])
    if actual_symmetry!=symmetry or actual_refinement!=refinement:
        raise ValueError('completed MCU posteval mixture differs')
    # The new run has local update numbers. Preserve accepted history verbatim.
    state=copy.deepcopy(state);state['stage_started_update']-=source['update'];state['last_evaluation_update']=0
    values=course.commands(state['stage'])
    return dict(state=state,decision=decision,
        symmetry_mix=course.extend_mixture(symmetry,values),reward_mix=course.extend_mixture(refinement,values))


def recovery_evidence(parent,checkpoint):
    import torch
    from native_stand.command_v6.wiring import validate_persisted_certificate
    p=Path(parent).resolve();cp=Path(checkpoint).resolve();old=json.loads(p.read_text())
    if old.get('mcu_measured',{}).get('engineering_resume'):
        raise ValueError('this recovery is for the first measured500 interface failure only')
    verify(old);run=p.parents[3]/'runs'/old['experiment'];formal=run/'formal'
    book=json.loads(p.with_name('budget.json').read_text())
    chain=json.loads((run/'chain_terminal_report.json').read_text());end=json.loads((formal/'terminal_report.json').read_text())
    failure=end.get('failure') or {}
    if (chain['status']!='FAILED' or end['status']!='FAILED'
        or failure.get('exception')!="TypeError('effective_spec() takes 2 positional arguments but 3 were given')"
        or 'motion_runner.py' not in failure.get('traceback','')
        or book['used']['smoke']!=5 or book['used']['formal']!=500 or end['actual_updates']!=500
        or any(a['status']=='RUNNING' for a in book['attempts'].values()) or cp!=formal/'model_update_500.pt'):
        raise ValueError('settled exact500 effective_spec interface failure required')
    if validate_persisted_certificate(run/'smoke',old)['status']!='PASSED':
        raise ValueError('completed original312 wiring smoke required')
    source=torch.load(cp,map_location='cpu',weights_only=False)
    if (source['spec_sha256']!=sha(p) or source['update']!=500
        or source['critic_privileged_version']!=old['student_training']['critic_privileged_version']
        or source['network_architecture']!=old['network_architecture']
        or any(k not in source for k in ('actor','critic','optimizer','training_rng','reward_refinement_state','symmetry_mix_state'))):
        raise ValueError('exact full312 PPO checkpoint required')
    registry=json.loads((formal/'candidate_registry.json').read_text())
    candidates=[c for c in registry if c.get('update')==500 and c.get('complete_matrix') is True]
    if len(candidates)!=1:raise ValueError('one complete update500 matrix required')
    candidate=copy.deepcopy(candidates[0])
    if candidate['checkpoint']!=str(cp) or candidate['sha256']!=sha(cp):raise ValueError('matrix checkpoint identity differs')
    expected=course.cases(old,source['motion_course_state']['stage'])
    for row in expected:row.update(checkpoint=str(cp),sha256=candidate['sha256'])
    if len(candidate['evaluations'])!=len(expected):raise ValueError('complete declared MCU matrix required')
    paths=[p,p.with_name('budget.json'),cp,run/'chain_terminal_report.json',formal/'terminal_report.json',
        formal/'manifest.json',formal/'candidate_registry.json',formal/'course_progress.json',
        formal/'symmetry_refinement_after_500.json',formal/'reward_refinement_after_500.json',
        run/'smoke/wiring_certificate.json',run/'smoke/terminal_report.json',
        run/'source_eval/candidate_registry.json']
    for record,case in zip(candidate['evaluations'],expected):
        raw_path=(formal/record['path']).resolve();raw=json.loads(raw_path.read_text())
        trace=(formal/raw['trace']['path']).resolve()
        if (not raw_path.is_relative_to(formal) or not trace.is_relative_to(formal)
            or raw['case']!=case or record['case']!=case or sha(raw_path)!=record['sha256']
            or sha(trace)!=raw['trace']['sha256'] or raw['checkpoint_sha256']!=candidate['sha256']
            or raw['spec_sha256']!=sha(p) or raw['asset_sha256']!=old['asset_sha256']
            or raw['protocol']!=old['motion_assessment'] or raw.get('model_unchanged') is not True
            or not course.complete(raw) or len(record['per_env'])!=8
            or any(any(full.get(k)!=v for k,v in compact.items()) for full,compact in zip(raw['per_env'],record['per_env']))):
            raise ValueError('completed MCU raw/trace/eight-env identity differs')
        paths += [raw_path,trace]
        record['path']=str(raw_path)
    persisted=json.loads((formal/'course_progress.json').read_text())
    symmetry=json.loads((formal/'symmetry_refinement_after_500.json').read_text())['state']
    refinement=json.loads((formal/'reward_refinement_after_500.json').read_text())['next_state']
    boundary=reconstructed_boundary(old,source,candidate,persisted,symmetry,refinement)
    historical=old['mcu_measured']['historical_online_used']+5+500
    history_bytes=old['mcu_measured']['historical_output_bytes']+sum(q.stat().st_size for q in run.rglob('*') if q.is_file() and not q.is_symlink())
    if historical!=630 or 5000-historical!=4370:raise ValueError('original inclusive5000 consumption differs')
    first=Path(old['mcu_measured']['parent_spec']);first_run=first.parents[3]/'runs'/json.loads(first.read_text())['experiment']
    first_end=first_run/'chain_terminal_report.json';paths.append(first_end)
    historical_wall=chain['chain_wall_s']+json.loads(first_end.read_text())['chain_wall_s']
    return old,source,dict(version=RESUME_VERSION,parent_spec=str(p),parent_run=str(run),
        inputs={str(q):sha(q) for q in paths},checkpoint=str(cp),checkpoint_update=500,
        historical_smoke_updates=10,historical_formal_updates=620,historical_online_updates=historical,
        historical_wall_s=historical_wall,historical_output_bytes=history_bytes,
        original_online_limit=5000,remaining_formal_updates=4370,original_output_limit=4*2**30,
        boundary=boundary,source_evidence_scope='ancestral source phase only; completed500 matrix is bound separately, new endpoints unassessed',
        physical_state_restored=False,warmup_repeated=False,capability_accepted=False)


def build_recovery(parent,checkpoint,*,experiment,spec_path,owner_session,preformal_failure=None):
    old,source,proof=recovery_evidence(parent,checkpoint);s=copy.deepcopy(old)
    if preformal_failure:
        p=Path(preformal_failure).resolve();prior=json.loads(p.read_text())
        expected=build_recovery(parent,checkpoint,experiment=prior['experiment'],spec_path=p,
            owner_session=prior['authorization'].get('hapi_session'))
        for k in ('authorization','resources','notifications'):expected[k]=prior[k]
        run=p.parents[3]/'runs'/prior['experiment'];book=json.loads(p.with_name('budget.json').read_text())
        end=json.loads((run/'chain_terminal_report.json').read_text())
        if (prior!=expected or end['status']!='FAILED'
            or end.get('failure')!="ValueError('insufficient output storage before reserve')"
            or book['attempts'] or any(book['used'].values()) or (run/'formal').exists()):
            raise ValueError('only exact settled zero-attempt preformal storage rejection can be superseded')
        proof['preformal_storage_failure']=dict(spec=str(p),spec_sha256=sha(p),run=str(run),
            actual_online_updates=0,block_formal_updates=1500,unallocated_online_updates=2870)
        proof['inputs'].update({str(q):sha(q) for q in (p,p.with_name('budget.json'),run/'chain_terminal_report.json')})
        proof['historical_wall_s']+=end['chain_wall_s']
        proof['historical_output_bytes']+=sum(q.stat().st_size for q in run.rglob('*') if q.is_file() and not q.is_symlink())
    storage=proof['original_output_limit']-proof['historical_output_bytes'];formal=proof['remaining_formal_updates']
    if preformal_failure:formal=1500
    if storage<1024*2**20:raise ValueError('remaining cumulative MCU output space insufficient')
    cp=proof['checkpoint'];digest=sha(cp);stage=proof['boundary']['state']['stage']
    s.update(experiment=experiment,source_checkpoint=cp,source_sha256=digest)
    s['student_training']=recipe(digest,stage,formal,storage,old['motion_timing']['configuration'],recovery=True)
    s['motion_training'].update(spec_path=str(Path(spec_path).resolve()),resume=None,wiring_reuse=proof)
    s['motion_course'].update(source_recipe_updates=source['recipe_updates'])
    s['motion_course']['configuration'].update(smoke_updates_max=0,formal_updates_max=formal)
    s['mcu_measured'].update(parent_spec=proof['parent_spec'],parent_spec_sha256=sha(parent),
        initial_stage=stage,initial_capability=course.decode(stage),historical_online_used=630,
        remaining_online=formal,historical_output_bytes=proof['historical_output_bytes'],engineering_resume=proof)
    s['motion_timing']['source_checkpoint_sha256']=digest
    s['student_evaluation']['cases'][0].update(checkpoint=cp,sha256=digest)
    s['budget'].update(source_eval_updates=0,smoke_updates=0,formal_updates=formal,
        formal_samples=formal*4096*64,sample_limit=formal*4096*64)
    s['artifacts'].update(run_name=experiment,storage_max_bytes=storage)
    s['student_dependencies'].update(proof['inputs'])
    s['authorization']=dict(status='UNAUTHORIZED',hapi_session=owner_session,
        user_scope='Existing user authorization: repair ordinary Motion interface failure and resume full update500 within original5000;630 used,4370 formal left, no repeated smoke/warmup/matrix; same GPU0shared/unlimited training wall/physics/reward/Actor31/Critic312/notifications/TensorBoard6006.')
    if preformal_failure:
        s['authorization']['user_scope']+=' Original4GiB cumulative storage unchanged; schedule1500 formal now, leave2870 unallocated, preserve settled zero-attempt storage rejection.'
    s['notifications']=dict(enabled=True,hapi_session=owner_session)
    return course.effective_spec(s,stage)


def verify_recovery(spec):
    c=spec['mcu_measured']['engineering_resume']
    expected=build_recovery(c['parent_spec'],spec['source_checkpoint'],experiment=spec['experiment'],
        spec_path=spec['motion_training']['spec_path'],owner_session=spec['authorization'].get('hapi_session'),
        preformal_failure=c.get('preformal_storage_failure',{}).get('spec'))
    for k in ('authorization','resources','notifications'):expected[k]=spec[k]
    if expected!=spec:raise ValueError('MCU posteval recovery declaration drift')
    claim=Path(spec['motion_training']['spec_path']).parents[1]/'.motion_mcu_measured_recovery_claims'/sha(c['parent_spec'])
    allowed={spec['motion_training']['spec_path']}
    if c.get('preformal_storage_failure'):allowed.add(c['preformal_storage_failure']['spec'])
    if claim.exists() and json.loads(claim.read_text())['spec'] not in allowed:
        raise ValueError('MCU parent recovery already claimed')
    validate_recipe(spec['student_training'],spec['source_sha256'])
    return dict(status='VERIFIED',historical_online_updates=630,remaining_formal_updates=4370,
        scheduled_formal_updates=spec['budget']['formal_updates'],capability_accepted=False)


def reused_source(spec):
    a=evidence(spec['mcu_measured']['assessment'])
    by={course.key(r['case']):r for r in a['records']}
    records=[copy.deepcopy(by[course.key(row)]) for row in spec['student_evaluation']['cases']]
    candidate=dict(checkpoint=spec['source_checkpoint'],sha256=spec['source_sha256'],update=0,
        stage=spec['motion_training']['stage'],per_env=[e for r in records for e in r['per_env']],
        complete_matrix=True,evaluations=records,
        scope='same Actor31/execution/scoring evidence; Critic312 engineering is verified separately')
    return records,[candidate]
