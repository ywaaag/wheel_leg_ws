"""Explicit same31D Actor2048 continuation: high-speed motion and standing.

Course/selection helpers use explicit data. build/verify own file identities.
Old height/PD recipes and their gates remain unchanged.
"""
import copy
import json
from pathlib import Path

from .data import sha

VERSION = 'motion31_mcu_highspeed_standing2048_v1'
ASSESSMENT = 'motion31_highspeed_standing_assessment_v1'
RETENTION = 'motion31_mcu9995_nominal_retention_v5'
SOURCE_SHA = 'd1da9580d064014a74dae79cb7b8fc98c4806c7b6038d019b65aee57645596b9'
ONLINE, FORMAL, STORAGE = 10000, 9995, 10 * 2**30
HEIGHTS = (.30, .32, .34, .36, .38, .40)
DIRECTIONS = ('forward', 'backward', 'yaw_positive', 'yaw_negative')


def command(direction, magnitude):
    if direction not in DIRECTIONS:raise ValueError('unknown high-speed direction')
    return {'forward':[float(magnitude),0.], 'backward':[-float(magnitude),0.],
            'yaw_positive':[0.,float(magnitude)], 'yaw_negative':[0.,-float(magnitude)]}[direction]


def initial_state():
    return dict(stage=0, stage_started_update=0, last_evaluation_update=-1,
        consecutive_passes=0, accepted_by_stage={}, pd_level=2, height_level=2,
        qualified_pd_levels=[], qualified_height_levels=[],
        current_direction_passed={d: False for d in DIRECTIONS}, standing_passed=False)


def configuration():
    return dict(version=VERSION, formal_updates_max=FORMAL, smoke_updates_max=5,
        stage_updates_max=FORMAL, progression=False, evaluation_every=500,
        evaluation_updates=[*range(500, FORMAL, 500), FORMAL], confirmations=1,
        behavior_stop=False, height_m=.34, moving_height_transition=False,
        allocation=dict(highspeed=.50, nominal_standing=.20, stress_standing=.20, low_transition=.10),
        independent_directions=list(DIRECTIONS), speed_targets=[2.,3.,4.],
        low_speed_precision='diagnostic only; safety and direction still reported',
        handover_prefix_policy_intervals=100, PD_feedback_range_ms=[8,20],
        handover_controller='frozen original9995 Actor; no reset at boundary; not LQR',
        promotion='no new endpoints beyond4; each direction recorded independently; all budget remains available')


def stage_config(stage=0, envs=4096, *, state=None):
    if stage != 0 or envs not in (1,64,4096):
        raise ValueError('explicit fixed high-speed/standing stage and1/64/4096 envs required')
    rows=[dict(target=[0.,0.],height=.34,variant=0,purpose='nominal_standing',prefix=0)]
    rows += [dict(target=[0.,0.],height=h,variant=4,purpose='height_standing',prefix=0)
             for h in HEIGHTS if h != .34]
    rows += [dict(target=[0.,0.],height=.34,variant=2,purpose='stress_standing',prefix=0),
             dict(target=[0.,0.],height=.34,variant=3,purpose='PD_standing',prefix=100)]
    rows += [dict(target=command(d,m),height=.34,variant=0,purpose='highspeed_'+d,prefix=0)
             for d in DIRECTIONS for m in (2,3,4)]
    rows += [dict(target=command(d,m),height=.34,variant=0,purpose='low_transition',prefix=0)
             for d in DIRECTIONS for m in (.5,1.)]
    if envs == 4096:
        counts=[659,*([32]*5),409,410,*([128,128,256]*4),52,52,*([51]*6)]
    elif envs == 64:
        counts=[7,*([1]*5),6,6,*([2,2,4]*4),*([1]*8)]
    else:
        counts=[1]+[0]*(len(rows)-1)
    if len(counts)!=len(rows) or sum(counts)!=envs:
        raise ValueError('high-speed fixed-slot allocation differs')
    return dict(id=0,name='highspeed_and_standing2048',commands=[r['target'] for r in rows],
        slot_counts=counts,weights=[n/envs for n in counts],sampling_mode='fixed_slots',
        sampling_slots=[i for i,n in enumerate(counts) for _ in range(n)],enforce_whitelist=True,
        cohort_keys=[[*r['target'],r['height'],r['variant']] for r in rows],
        mcu_variants=[r['variant'] for r in rows],height_targets_m=[r['height'] for r in rows],
        cohort_purposes=[r['purpose'] for r in rows],prefix_intervals=[r['prefix'] for r in rows],
        PD_feedback_max_ms=20)


def retained_commands():
    # Source9995 passed all these nominal034 Native commands. Positive yaw4
    # failed spin_common, and +/-linear4 and negative yaw4 were not qualified.
    return [[0.,0.]]+[command(d,m) for d in DIRECTIONS for m in (.5,1.,1.5,2.,3.)]


def retention(source_sha):
    from .motion_source_retention import declaration
    if source_sha != SOURCE_SHA:raise ValueError('retention teacher must be original9995')
    value=declaration(source_sha)
    value.update(version=RETENTION,commands=retained_commands(),eligible_mcu_variant=0,
        eligible_height_m=.34,scope='source-qualified nominal034 only; new4/stress/height excluded')
    return value


def recipe(initialization_sha):
    from .motion_height_pd import recipe as base
    from .motion_handover_robustness import SOURCE_SHA as OLD_SOURCE
    from .actor_widening import wide_network
    value=base(OLD_SOURCE,ONLINE)
    from .motion_mcu_measured import smoke_profile
    value.update(source_actor_sha256=initialization_sha,course_version=VERSION,
        network_architecture=wide_network(2048),height_dr_state=initial_state(),
        source_policy_retention=retention(SOURCE_SHA),
        continuation_teacher=dict(checkpoint='',sha256=SOURCE_SHA),
        critic='source_same_semantic312',optimizer='source_mapped_Adam2048',
        milestones=configuration()['evaluation_updates'],storage_max_bytes=STORAGE,
        warmup_updates_per_phase=5,
        smoke_wiring=smoke_profile(),
        selection_scope='highspeed and standing primary; low precision diagnostic; same checkpoint full evidence')
    return value


def runtime_readback(env,obs):
    import torch
    stage=env.stage_spec;device=env.device
    counts=torch.bincount(env.command_index,minlength=len(stage['commands']))
    if not torch.equal(counts,torch.tensor(stage['slot_counts'],device=device)):
        raise RuntimeError('Actor2048 runtime fixed-slot counts differ')
    variants=torch.tensor(stage['mcu_variants'],device=device)[env.command_index]
    height=torch.tensor(stage['height_targets_m'],device=device)[env.command_index]
    if not torch.equal(variants,env.motion_timing.variant) or not torch.allclose(height,env.height_adapter.target):
        raise RuntimeError('Actor2048 variant/height runtime wiring differs')
    if obs['student_actor'].shape!=(env.num_envs,31) or obs['student_critic'].shape!=(env.num_envs,312):
        raise RuntimeError('Actor2048 runtime observation shape differs')
    if not torch.allclose(obs['student_actor'][:,24],height/.5,atol=1e-6,rtol=0):
        raise RuntimeError('Actor2048 command-height observation differs')
    if not torch.allclose(obs['student_critic'][:,140:143],(height/.5)[:,None].expand(-1,3),atol=1e-6,rtol=0):
        raise RuntimeError('Actor2048 actual-height Critic context differs')
    from .motion_source_retention import KEY
    ids=obs[KEY].squeeze(-1)
    eligible=(variants==0)&torch.isclose(height,height.new_tensor(.34))
    if (ids[~eligible]>=0).any():raise RuntimeError('Teacher leaked into stress/height cohort')
    if (env.motion_timing.actual_armature-.012).abs().max()>1e-6:
        raise RuntimeError('Actor2048 actual leg armature differs')
    return dict(status='PASSED',before_first_optimizer=True,actor_shape=list(obs['student_actor'].shape),
        critic_shape=list(obs['student_critic'].shape),slot_counts=counts.cpu().tolist(),
        variants=sorted(set(stage['mcu_variants'])),height_targets_m=sorted(set(stage['height_targets_m'])),
        PD_age_range_ms=[int(env.motion_timing.feedback.min()),int(env.motion_timing.feedback.max())],
        actual_leg_armature=float(env.motion_timing.actual_armature[0]),
        source_retention_scored_samples=int((ids>=0).sum()),scope='actual engineering wiring; not skill certification')


def validate_recipe(value,source_sha):
    expected=recipe(source_sha)
    expected['continuation_teacher']['checkpoint']=value.get('continuation_teacher',{}).get('checkpoint','')
    for name in ('height_dr_state','stage','course_stage'):
        expected[name]=value.get(name,expected.get(name))
    if value != expected:raise ValueError('Actor2048 high-speed/standing recipe drift')
    return value


def expanded_source(source):
    """Already widened source, never reinterpret a512 Actor by tensor shape."""
    from .actor_widening import wide_network,validate_wide_candidate,LARGER_INITIALIZATION
    from .motion_mcu_privileged import VERSION as CRITIC
    if source.get('network_architecture')!=wide_network(2048) or source.get('critic_privileged_version')!=CRITIC:
        raise ValueError('explicit Actor2048/MCU312 source required')
    if source.get('initialization_kind')==LARGER_INITIALIZATION:
        validate_wide_candidate(source)
    elif source.get('training_recipe',{}).get('course_version')!=VERSION:
        raise ValueError('source has no explicit2048 initialization/training identity')
    return source


def protocol():
    from .motion_height_pd import protocol as base
    return dict(base(),version=ASSESSMENT,
        primary_scope='standing and absolute command magnitude2..4; low-speed precision diagnostic',
        standing_horizontal_mean_max_mps=.10,standing_horizontal_p95_max_mps=.15,
        standing_yaw_abs_mean_max_rad_s=.10,standing_height_MAE_max_m=.01,
        standing_height_p95_max_m=.02,
        task_termination='unchanged quality termination; no behavior-based chain stopping')


def cases(spec,stage=0,state=None):
    if stage != 0:raise ValueError('high-speed stage differs')
    original=spec['student_evaluation']['cases'][0]
    template={k:original[k] for k in ('kind','role','family','seed','yaw_rad','seconds')}
    template.update(checkpoint=spec['source_checkpoint'],sha256=spec['source_sha256'],
                    _timing_level=4,_mcu_revision=0)
    rows=[]
    def add(cmd,height=.34,variant=0,**extra):
        c=dict(template,case_id=f'hs_{len(rows):02}',command=list(cmd)+[height],_mcu_variant=variant,**extra)
        if height != .34:c['_height']=dict(initial_m=height,target_m=height,moving=False)
        c['_primary_required']=cmd==[0.,0.] or max(map(abs,cmd))>=2
        rows.append(c)
    for h in HEIGHTS:add([0.,0.],h,0 if h==.34 else 4)
    for variant in (1,2):add([0.,0.],variant=variant)
    for age in (8,20):
        for prefix in (0,100):
            add([0.,0.],variant=3,_execution_uncertainty=dict(PD_age_ms=age,prefix_intervals=prefix),
                seconds=31.52+prefix*.01)
    for d in DIRECTIONS:
        for m in (2.,3.,4.):add(command(d,m))
    for d in DIRECTIONS:
        for m in (.5,1.):add(command(d,m))
    return rows


def identity(row):
    return [row['command'],row.get('_mcu_variant'),row.get('_execution_uncertainty'),
            row.get('_height'),row.get('_mcu_revision'),row.get('_primary_required')]


def effective_spec(spec,stage,state=None):
    s=copy.deepcopy(spec);current=copy.deepcopy(state or s['motion_height_dr']['state'])
    if current['stage']!=stage or stage!=0:raise ValueError('high-speed state revision differs')
    s['motion_training']['stage']=stage
    s['curriculum']['stages']=[stage_config(stage,s['ppo']['envs'],state=current)]
    s['student_evaluation']['cases']=cases(s,stage,current)
    current['assessment_keys']=[identity(r) for r in s['student_evaluation']['cases']]
    s['motion_height_dr']['state']=current;s['motion_course']['state']=current
    s['student_training'].update(stage=stage,course_stage=stage,height_dr_state=current)
    return s


def advance(state,candidate):
    from .motion_mcu_measured import complete
    rows=candidate.get('evaluations',[])
    if candidate.get('stage')!=0 or candidate['update']<=state['last_evaluation_update']:
        raise ValueError('stale high-speed assessment')
    valid=bool(candidate.get('complete_matrix') is True and rows and
        [identity(r['case']) for r in rows]==state['assessment_keys'] and all(complete(r) for r in rows))
    good=lambda group:bool(valid and group and all(e.get('accepted') is True for r in group for e in r['per_env']))
    out=copy.deepcopy(state);out['last_evaluation_update']=candidate['update']
    out['standing_passed']=good([r for r in rows if r['case']['command'][:2]==[0.,0.]])
    for direction in DIRECTIONS:
        out['current_direction_passed'][direction]=good([r for r in rows if r['case']['command'][:2] in
            [command(direction,m) for m in (2.,3.,4.)]])
    required=[r for r in rows if r['case']['_primary_required']]
    passed=good(required) and all(low_safety_passed(r) for r in rows if not r['case']['_primary_required'])
    out['consecutive_passes']=state['consecutive_passes']+1 if passed else 0
    if passed:out['accepted_by_stage']['0']={k:candidate[k] for k in ('checkpoint','sha256','update')}
    return out,dict(evidence_valid=valid,stage_passed=passed,standing_passed=out['standing_passed'],
        direction_passed=out['current_direction_passed'],advanced=False,all_stages_passed=False,
        final_confirmed=passed,stop=False,low_precision_blocks_primary=False)


def choose(candidates,accepted_by_stage):
    def required(c):return [r for r in c.get('evaluations',[]) if r['case'].get('_primary_required')]
    def rank(c):
        rows=required(c);envs=[e for r in rows for e in r['per_env']]
        return (sum(e.get('survival_passed') is True for e in envs),
                sum(e.get('accepted') is True for e in envs),-c.get('update',0))
    evaluated=[c for c in candidates if c.get('update',0)>0 and c.get('complete_matrix')]
    best=max(evaluated,key=rank,default=None)
    accepted=[c for c in evaluated if required(c) and
              all(e.get('accepted') is True for r in required(c) for e in r['per_env']) and
              all(low_safety_passed(r) for r in c['evaluations'] if not r['case']['_primary_required'])]
    return dict(source=candidates[0],source_baseline=candidates[0],fallback_checkpoint=candidates[0],
        latest=candidates[-1],latest_checkpoint=candidates[-1],best_new=best,
        best_new_evaluated_candidate=best,best_overall=accepted[-1] if accepted else candidates[0],
        best_observed_candidate=best or candidates[0],accepted_candidate=accepted[-1] if accepted else None,
        accepted_by_stage=accepted_by_stage,current_stage=0,final_multi_seed_certified=False,
        standing_quality_certified=False,selection='primary full survival then accepted envs; low precision excluded',
        capability_scope='same-model Native standing/highspeed; not hardware or sim2sim certification')


def extend_mixture(previous,values):
    from .motion_mcu_course import extend_mixture as base
    return base(previous,values)


def low_safety_passed(record):
    """Only low precision is optional; direction/survival/physics still count."""
    ignored={'tracking','gain','sustained_fraction'}
    return bool(record.get('per_env') and all(e.get('survival_passed') is True and
        e.get('checks') and all(v is True for k,v in e['checks'].items() if k not in ignored)
        for e in record['per_env']))


def refinement_records(records,stage):
    cfg=stage_config(stage);out=[]
    for cmd,h,v in zip(cfg['commands'],cfg['height_targets_m'],cfg['mcu_variants']):
        matches=[r for r in records if r['case']['command']==cmd+[h] and r['case']['_mcu_variant']==v]
        out.append(copy.deepcopy(matches[0]) if len(matches)==1 else dict(case=dict(command=cmd+[h]),per_env=[]))
    return out


def assess(result,arrays,eid,row):
    import numpy as np
    from .motion_height_pd import assess as base
    base(result,arrays,eid,row)
    checks=result['checks'];cmd=row['command'][:2]
    if cmd==[0.,0.]:
        mask=arrays['first_episode_valid'][:,eid] & (arrays['controller'][:,eid,0]==3)
        state=arrays['state'][mask,eid][:3000];world=arrays['root_post'][mask,eid][:3000]
        full=len(state)==3000
        metrics=dict(complete30=full,window_s=[2,30])
        if full:
            speed=np.linalg.norm(world[200:3000,7:9],axis=-1)
            error=abs(state[200:3000,6]-row['command'][2])
            metrics.update(horizontal_mean_mps=float(speed.mean()),horizontal_p95_mps=float(np.quantile(speed,.95)),
                yaw_abs_mean_rad_s=float(abs(state[200:3000,9]).mean()),
                height_MAE_m=float(error.mean()),height_p95_m=float(np.quantile(error,.95)))
            checks.update(standing_drift=metrics['horizontal_mean_mps']<=.1 and metrics['horizontal_p95_mps']<=.15,
                standing_yaw=metrics['yaw_abs_mean_rad_s']<=.1,
                standing_height=metrics['height_MAE_m']<=.01 and metrics['height_p95_m']<=.02)
        else:checks.update(standing_drift=False,standing_yaw=False,standing_height=False)
        result['standing_target']=metrics
    result.update(version=ASSESSMENT,primary_required=row.get('_primary_required',False),
        accepted=all(v is True for v in checks.values()),failed_checks=[k for k,v in checks.items() if v is not True])
    return result


def source_records(parent,checkpoint):
    """Read immutable raw9995 records; no source capability prerequisite."""
    parent,checkpoint=Path(parent).resolve(),Path(checkpoint).resolve()
    if sha(checkpoint)!=SOURCE_SHA:raise ValueError('original9995 source identity differs')
    old=json.loads(parent.read_text())
    registry=json.loads((checkpoint.parent/'candidate_registry.json').read_text())
    candidate=next(c for c in registry if c.get('sha256')==SOURCE_SHA and c.get('update')==9995)
    if candidate.get('complete_matrix') is not True:raise ValueError('source9995 complete evidence required')
    records=[];inputs={str(parent):sha(parent),str(checkpoint):SOURCE_SHA}
    for record in candidate['evaluations']:
        p=checkpoint.parent/record['path'];raw=json.loads(p.read_text());trace=p.parent/raw['trace']['path']
        if sha(p)!=record['sha256'] or raw['checkpoint_sha256']!=SOURCE_SHA:
            raise ValueError('source9995 raw/model identity differs')
        trace_present=trace.is_file()
        if trace_present and sha(trace)!=raw['trace']['sha256']:
            raise ValueError('source9995 trace identity differs')
        if len(raw['per_env'])!=8:raise ValueError('source9995 per-environment evidence incomplete')
        inputs[str(p)]=sha(p)
        if trace_present:inputs[str(trace)]=raw['trace']['sha256']
        records.append(dict(record,path=str(p),per_env=raw['per_env'],
            source_trace_status='SHA_VERIFIED' if trace_present else 'PRUNED_OR_MISSING_NOT_REPLAYABLE',
            source_trace_reference=dict(path=str(trace),sha256=raw['trace']['sha256'])))
    from .motion_source_retention import verify_source_records
    nominal=[r for r in records if r['case'].get('_mcu_variant')==0]
    verify_source_records(nominal,retention(SOURCE_SHA))
    return old,records,inputs


def build(parent,original,initialization,*,experiment,spec_path,session,recover_zero_attempt=None):
    import torch
    from .actor_widening import validate_wide_candidate,wide_network
    from .motion_mcu_privileged import contract
    parent,original,initialization=map(lambda p:Path(p).resolve(),(parent,original,initialization))
    old,records,inputs=source_records(parent,original)
    source=torch.load(initialization,map_location='cpu',weights_only=False)
    validate_wide_candidate(source)
    if source['source_checkpoint_sha256']!=SOURCE_SHA or source['network_architecture']!=wide_network(2048) or source['spec_sha256']!=sha(parent):
        raise ValueError('2048 initialization belongs to another source/spec')
    s=copy.deepcopy(old);init_sha=sha(initialization)
    # Autonomous Motion passes a ForbiddenTeacher sentinel to the shared base;
    # legacy STAND/Recovery teacher files are not read by this execution path.
    s['student_teachers']=[]
    for key in ('stand_checkpoint','stand_sha256','stand_spec','stand_spec_sha256'):
        s['recovery'].pop(key,None)
    for key in ('mcu_posture','mcu_measured','checkpoint_quick_evaluation','motion_large_continuation'):
        s.pop(key,None)
    s.update(experiment=experiment,source_checkpoint=str(initialization),source_sha256=init_sha,
             network_architecture=wide_network(2048),student_interface=contract())
    s['student_training']=recipe(init_sha)
    s['student_training']['continuation_teacher']['checkpoint']=str(original)
    s['motion_training']={k:v for k,v in old['motion_training'].items() if k in ('version','block_index','early_stop_policy')}
    s['motion_training'].update(stage=0,resume=None,spec_path=str(Path(spec_path).resolve()))
    s['motion_course']=dict(source_recipe_updates=source['recipe_updates'],configuration=configuration())
    s['motion_highspeed_standing']=dict(version=VERSION,parent_spec=str(parent),parent_spec_sha256=sha(parent),
        original_checkpoint=str(original),original_sha256=SOURCE_SHA,initialization_sha256=init_sha,
        total_online_limit=ONLINE,storage_profile='eight_env_scores_one_env_diagnostics_v1',
        source_evidence_conditions=len(records),historical_budget_not_reused=True)
    s['motion_height_dr']=dict(version=VERSION,state=initial_state(),height_poses=old['motion_height_dr']['height_poses'],
        startup_scope='highspeed9995_CPU_reuse_only',count_original_storage=False)
    s['motion_execution_robustness']=dict(version=VERSION,source_checkpoint=str(original),source_sha256=SOURCE_SHA,
        configuration=configuration())
    s['motion_assessment']=protocol();s['student_evaluation'].update(version=ASSESSMENT,max_output_bytes=128*2**20)
    s['budget'].update(source_eval_wall_s=1800,smoke_wall_s=None,formal_wall_s=None,chain_wall_s=None,
        evaluation_wall_s=14400,formal_updates=FORMAL,smoke_updates=5,
        formal_samples=FORMAL*4096*64,sample_limit=ONLINE*4096*64)
    s['artifacts'].update(run_name=experiment,storage_max_bytes=STORAGE,terminal_reserve_bytes=512*2**20)
    s['evaluation'].update(full_every_updates=500,full_trace_updates=[])
    s['performance'].update(evaluation_group_commands=True,evaluation_group_max_commands=8)
    s['student_dependencies']={**inputs,str(initialization):init_sha,s['asset_path']:s['asset_sha256']}
    # Actual current plant dependencies remain explicit; obsolete model/data
    # names in historical declarations are not new runtime dependencies.
    for key in ('asset_module_root',):
        if not Path(s[key]).is_dir():raise ValueError('asset module root missing')
    s['motion_startup'].update(references=[],missing_commands=[])
    s['authorization']=dict(status='UNAUTHORIZED',hapi_session=session,
        user_scope='User2026-10-28 approves NEW10000 inclusive online updates(smoke5+formal9995), unlimited training wall; original9995 widened31D Actor2048/1024/512 with312 Critic and mappedAdam/std/RNG;50highspeed20nominal-standing20stress-standing10low-transition, old physical/reward/LR/std;GPU0shared,100save/500full8env,TensorBoard6006 and owner notification; no reuse of exhausted prior10000.')
    s['notifications']=dict(enabled=True,hapi_session=session)
    s['resources']=dict(mode='shared',sharing_authorized=True,authorization_note='User authorized same GPU0 shared Motion continuation; no preemption')
    if recover_zero_attempt:
        prior=Path(recover_zero_attempt).resolve();oldspec=json.loads(prior.read_text())
        bookpath=prior.with_name('budget.json');book=json.loads(bookpath.read_text())
        terminalpath=prior.parents[3]/'runs'/oldspec['experiment']/'chain_terminal_report.json'
        terminal=json.loads(terminalpath.read_text())
        if (oldspec.get('motion_highspeed_standing',{}).get('version')!=VERSION
                or oldspec['source_sha256']!=init_sha or book['attempts'] or any(book['used'].values())
                or terminal['status']!='FAILED' or 'FileNotFoundError' not in str(terminal.get('failure'))):
            raise ValueError('only exact zero-attempt missing obsolete binding failure may recover')
        recovery=dict(version='motion2048_unused_legacy_dependency_recovery_v1',
            parent_spec=str(prior),parent_spec_sha256=sha(prior),parent_run=str(terminalpath.parent),
            historical_online_updates=0,historical_wall_s=terminal['chain_wall_s'],
            cumulative_online_limit=ONLINE,reason='unused legacy STAND file was still bound by old inherited metadata')
        s['motion_highspeed_standing']['prelaunch_recovery']=recovery
        for path in (prior,bookpath,terminalpath):s['student_dependencies'][str(path)]=sha(path)
    return effective_spec(s,0,initial_state())


def verify(spec):
    c=spec['motion_highspeed_standing']
    expected=build(c['parent_spec'],c['original_checkpoint'],spec['source_checkpoint'],experiment=spec['experiment'],
        spec_path=spec['motion_training']['spec_path'],session=spec['authorization'].get('hapi_session'),
        recover_zero_attempt=c.get('prelaunch_recovery',{}).get('parent_spec'))
    for field in ('authorization','resources','notifications'):expected[field]=spec[field]
    if expected!=spec:raise ValueError('high-speed2048 declaration differs')
    validate_recipe(spec['student_training'],spec['source_sha256'])
    return True


def startup_evidence(spec):
    """Reference source results with original gates; new gates stay UNKNOWN."""
    c=spec['motion_highspeed_standing']
    _,records,_=source_records(c['parent_spec'],c['original_checkpoint'])
    rows=[]
    for record in records:
        row=copy.deepcopy(record)
        row.update(reused=True,reuse_reference=dict(path=record['path'],sha256=record['sha256'],
            original_checkpoint=c['original_checkpoint'],original_checkpoint_sha256=SOURCE_SHA,
            scope='original9995 Native protocol;2048 function equivalence verified on CPU; new primary protocol not certified'))
        rows.append(row)
    candidate=dict(checkpoint=spec['source_checkpoint'],sha256=spec['source_sha256'],update=0,stage=0,
        per_env=[e for r in rows for e in r['per_env']],evaluations=rows,complete_matrix=False,
        source_scope='ancestral9995 evidence; new standing/primary gates UNKNOWN; no ability prerequisite')
    return rows,[candidate]


def learning_signal_audit(spec):
    from .motion_height_pd import learning_signal_audit as base
    result=base(spec);result.update(version='motion31_highspeed_standing2048_signal_audit_v1',
        teacher_eligible_cohort_indices=[i for i,(v,h,cmd) in enumerate(zip(
            stage_config()['mcu_variants'],stage_config()['height_targets_m'],stage_config()['commands']))
            if v==0 and h==.34 and cmd in retained_commands()])
    import torch
    from .motion_parallel_design import signals
    for direction in DIRECTIONS:
        for magnitude in (2.,3.,4.):
            target=command(direction,magnitude);axis=int(bool(target[1]))
            for error in (-4.,-2.,-.5,.5,2.,4.):
                x=torch.tensor(error,dtype=torch.float64,requires_grad=True)
                cmd=x.new_tensor([target]);velocity=torch.stack((x+target[0] if axis==0 else x*0,x*0)).reshape(1,2)
                yaw=(x+target[1] if axis==1 else x*0).reshape(1)
                value=signals(cmd,velocity,yaw,new_task=torch.zeros(1,dtype=torch.bool),mix=x.new_zeros(1))['density'][0]
                slope=float(torch.autograd.grad(value,x)[0])
                if not torch.isfinite(value) or slope*error>=0 or abs(slope)<1e-6:
                    raise ValueError('high-speed tracking signal reversed/saturated')
                result['curves'].append(dict(term='highspeed_'+direction,target=target,error=error,
                    weighted_density=float(value.detach()),derivative_per_physical_error=slope,
                    dt_contribution=float(value.detach())*.01))
    return result
