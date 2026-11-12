"""4500 continuation: independent PD-age and fixed-height standing curricula.

Course mathematics takes explicit data. File identities are handled by build.
No change to the existing handover-only or fixed034 recipes.
"""
import copy
import json
from pathlib import Path

from .data import sha
from . import motion_handover_robustness as handover
from .motion_posture_design import commands

VERSION = 'motion31_mcu_height_pd_parallel_v1'
ASSESSMENT = 'motion31_mcu_height_pd_assessment_v1'
RETENTION = 'motion31_mcu4500_height_scoped_retention_v4'
PD_MAX = (12, 16, 20)
HEIGHT_LEVELS = ((.32, .36), (.30, .38), (.40,))
HEIGHTS = (.30, .32, .34, .36, .38, .40)
LOW = ((.5, 0.), (-.5, 0.), (0., .5), (0., -.5))
FORMAL = 5124
ONLINE = FORMAL + 5
STORAGE = 10 * 2**30


def initial_state():
    return dict(stage=0, stage_started_update=0, last_evaluation_update=-1,
                consecutive_passes=0, accepted_by_stage={}, pd_level=0,
                height_level=0, qualified_pd_levels=[], qualified_height_levels=[])


def configuration(online=ONLINE):
    if online not in (ONLINE,10000):raise ValueError('explicit5129 or new10000 online authorization required')
    formal=online-5
    return dict(version=VERSION, formal_updates_max=formal, smoke_updates_max=5,
        stage_updates_max=formal, height_m=.34, progression=True,
        PD_feedback_max_levels_ms=list(PD_MAX), PD_feedback_min_ms=8,
        fixed_height_levels_m=[list(v) for v in HEIGHT_LEVELS], moving_height_transition=False,
        allocation=dict(PD_age=.25, new_height_standing=.25, nominal_motion=.35, nominal_zero=.15),
        evaluation_updates=[500,*range(1000,formal,1000),formal], confirmations=1,
        handover_prefix_policy_intervals=100, behavior_stop=False,
        progression_scope='PD and height independently; valid complete evidence only; no capability prerequisite for learning',
        physical_DR='PD feedback age only at034; new heights nominal; no new mass/friction/transport/spring DR')


def learned_heights(state):
    return sorted({.34, *[h for level in range(state['height_level']+1) for h in HEIGHT_LEVELS[level]]})


def stage_config(stage=0, envs=4096, *, state=None):
    state = state or initial_state()
    if state['stage'] != stage or envs not in (1,64,4096):
        raise ValueError('parallel height/PD revision or environment count differs')
    # Five DR cohorts include low-speed motion; only zero has a frozen prefix.
    rows=[dict(target=[0.,0.],variant=3,height=.34,purpose='PD_zero',prefix=100)]
    rows += [dict(target=list(v),variant=3,height=.34,purpose='PD_motion',prefix=0) for v in LOW]
    rows += [dict(target=[0.,0.],variant=4,height=h,purpose='height',prefix=0)
             for h in learned_heights(state) if h != .34]
    rows += [dict(target=[0.,0.],variant=0,height=.34,purpose='nominal_zero',prefix=0)]
    rows += [dict(target=v,variant=0,height=.34,purpose='nominal_motion',prefix=0) for v in commands()[1:]]
    counts=[0]*len(rows)
    def assign(indices,total):
        q,r=divmod(total,len(indices))
        for k,i in enumerate(indices):counts[i]=q+(k<r)
    if envs==1:counts[next(i for i,r in enumerate(rows) if r['purpose']=='nominal_zero')]=1
    else:
        totals=(1024,1024,614,1434) if envs==4096 else (16,16,9,23)
        assign(list(range(5)),totals[0])
        assign([i for i,r in enumerate(rows) if r['purpose']=='height'],totals[1])
        assign([i for i,r in enumerate(rows) if r['purpose']=='nominal_zero'],totals[2])
        assign([i for i,r in enumerate(rows) if r['purpose']=='nominal_motion'],totals[3])
    assert sum(counts)==envs
    return dict(id=stage,name='parallel_PD_age_and_fixed_height',commands=[r['target'] for r in rows],
        slot_counts=counts,weights=[v/envs for v in counts],sampling_mode='fixed_slots',
        sampling_slots=[i for i,n in enumerate(counts) for _ in range(n)],enforce_whitelist=True,
        cohort_keys=[[*r['target'],r['height'],r['variant']] for r in rows],
        mcu_variants=[r['variant'] for r in rows],height_targets_m=[r['height'] for r in rows],
        cohort_purposes=[r['purpose'] for r in rows],prefix_intervals=[r['prefix'] for r in rows],
        PD_feedback_max_ms=PD_MAX[state['pd_level']])


def protocol():
    p=copy.deepcopy(handover.protocol())
    p.update(version=ASSESSMENT,height_reference='fixed per-episode commanded root-link world Z',
        height_target_MAE_max_m=.01,height_target_p95_max_m=.02,
        course_scope='independent fixed-height standing and034 PD-age motion; no moving height switch')
    return p


def cases(spec,stage=0,state=None):
    state=state or initial_state()
    if state['stage']!=stage:raise ValueError('height/PD assessment revision differs')
    rows=handover.cases(spec,0)[:24]
    for i,r in enumerate(rows):
        r.update(case_id=f'hpd_nominal_{i}',_mcu_revision=stage,
            checkpoint=spec.get('source_checkpoint',r['checkpoint']),sha256=spec.get('source_sha256',r['sha256']))
    template=rows[0]
    for age in sorted({8,PD_MAX[state['pd_level']]}):
        for prefix in (0,100):
            rows.append(dict(template,case_id=f'hpd_PD{age}_prefix{prefix}',_mcu_variant=3,
                _execution_uncertainty=dict(PD_age_ms=age,prefix_intervals=prefix),
                _pd_focus=True,seconds=31.52+prefix*.01))
    for i,cmd in enumerate(LOW):
        rows.append(dict(template,case_id=f'hpd_PDmotion_{i}',command=list(cmd)+[.34],
            _mcu_variant=3,_execution_uncertainty=dict(PD_age_ms=PD_MAX[state['pd_level']],prefix_intervals=0),
            _pd_focus=True))
    for h in learned_heights(state):
        if h==.34:continue
        rows.append(dict(template,case_id=f'hpd_height_{h:.2f}',command=[0.,0.,h],_mcu_variant=4,
            _height=dict(initial_m=h,target_m=h,moving=False),_height_focus=h in HEIGHT_LEVELS[state['height_level']]))
    for row in rows:row.update(_pd_level=state['pd_level'],_height_level=state['height_level'])
    return rows


def identity(row):
    return [row['command'],row.get('_mcu_variant'),row.get('_execution_uncertainty'),row.get('_height'),row.get('_mcu_revision')]


def effective_spec(spec,stage,state=None):
    s=copy.deepcopy(spec);state=copy.deepcopy(state or s['motion_height_dr']['state'])
    if state['stage']!=stage:raise ValueError('parallel height/PD state revision differs')
    s['motion_height_dr']['state']=state
    s['motion_training']['stage']=stage
    s['student_training'].update(stage=stage,course_stage=stage,height_dr_state=state)
    s['motion_course']['state']=state
    s['curriculum']['stages']=[stage_config(stage,s['ppo']['envs'],state=state)]
    s['student_evaluation']['cases']=cases(s,stage,state)
    state['assessment_keys']=[identity(r) for r in s['student_evaluation']['cases']]
    return s


def advance(state,candidate):
    from .motion_mcu_measured import complete
    if candidate.get('stage')!=state['stage'] or candidate['update']<=state['last_evaluation_update']:
        raise ValueError('stale parallel height/PD evidence')
    rows=candidate.get('evaluations',[])
    valid=bool(candidate.get('complete_matrix') is True and rows
        and [identity(r['case']) for r in rows]==state['assessment_keys'] and all(complete(r) for r in rows))
    out=copy.deepcopy(state);out['last_evaluation_update']=candidate['update']
    passed=lambda group: bool(valid and group and all(e.get('accepted') is True for r in group for e in r['per_env']))
    pd_ok=passed([r for r in rows if r['case'].get('_pd_focus')])
    height_ok=passed([r for r in rows if r['case'].get('_height_focus')])
    full=passed(rows);out['consecutive_passes']=state.get('consecutive_passes',0)+1 if full else 0
    if full:out['accepted_by_stage'][str(state['stage'])]={k:candidate[k] for k in ('checkpoint','sha256','update')}
    for axis,ok,maximum in (('pd',pd_ok,len(PD_MAX)-1),('height',height_ok,len(HEIGHT_LEVELS)-1)):
        if ok:
            field='qualified_'+axis+'_levels';level=state[axis+'_level']
            out[field]=sorted(set(out[field]+[level]))
            out[axis+'_level']=min(level+1,maximum)
    advanced=out['pd_level']!=state['pd_level'] or out['height_level']!=state['height_level']
    if advanced:out.update(stage=state['stage']+1,stage_started_update=candidate['update'],consecutive_passes=0)
    return out,dict(evidence_valid=valid,pd_passed=pd_ok,height_passed=height_ok,stage_passed=full,
        advanced=advanced,all_stages_passed=False,stop=False,
        final_confirmed=full and max(out['qualified_pd_levels'],default=-1)==2 and max(out['qualified_height_levels'],default=-1)==2)


def retention(source_sha):
    r=handover.retention(source_sha)
    r.update(version=RETENTION,eligible_height_m=.34,
        scope='source-qualified nominal034 only; new heights and uncertain PD groups excluded')
    return r


def recipe(source_sha,online=ONLINE):
    r=handover.recipe(source_sha)
    r.update(course_version=VERSION,milestones=configuration(online)['evaluation_updates'],
        environment_seconds=33.52,storage_max_bytes=STORAGE,
        source_policy_retention=retention(source_sha),height_dr_state=initial_state(),
        fixed_height_reference=dict(version=VERSION,heights_m=list(HEIGHTS),moving=False,
            reward='replace height reference only; original Cauchy scale.04 weight2 dt.01',
            critic='same312 columns; column31 effective-height error and140:144 actual height context'))
    if online!=ONLINE:r['authorized_online_limit']=online
    return r


def validate_recipe(value,source_sha):
    expected=recipe(source_sha,value.get('authorized_online_limit',ONLINE))
    for field in ('height_dr_state','stage','course_stage'):expected[field]=value.get(field,expected.get(field))
    if value!=expected:raise ValueError('parallel height/PD recipe drift')
    return value


def extend_mixture(previous,values):
    # Existing widths may carry over only to corresponding034 command cohorts.
    return handover.extend_mixture(previous,values)


def refinement_records(records,stage):
    if not records:return []
    first=records[0]['case']
    state=dict(initial_state(),stage=stage,pd_level=first['_pd_level'],height_level=first['_height_level'])
    cfg=stage_config(stage,state=state);out=[]
    for cmd,h,variant in zip(cfg['commands'],cfg['height_targets_m'],cfg['mcu_variants']):
        matches=[r for r in records if r['case']['command']==cmd+[h] and r['case']['_mcu_variant']==variant]
        row=copy.deepcopy(matches[0]) if len(matches)==1 else dict(case=dict(command=cmd+[h]),per_env=[])
        out.append(row)
    return out


def choose(candidates,accepted_by_stage):
    stage=max((r.get('stage',0) for r in candidates),default=0)
    rows=[r for r in candidates if r.get('stage')==stage and r.get('complete_matrix')]
    new=[r for r in rows if r.get('update',0)>0]
    def rank(r):return sum(e.get('accepted') is True for e in r.get('per_env',[]))
    best=max(new,key=rank,default=None)
    accepted=[r for r in new if r.get('per_env') and all(e.get('accepted') is True for e in r['per_env'])]
    return dict(source=candidates[0],source_baseline=candidates[0],fallback_checkpoint=candidates[0],
        latest=candidates[-1],latest_checkpoint=candidates[-1],best_new=best,
        best_new_evaluated_candidate=best,best_overall=best or candidates[0],
        best_observed_candidate=best or candidates[0],accepted_candidate=accepted[-1] if accepted else None,
        accepted_by_stage=accepted_by_stage,current_stage=stage,final_multi_seed_certified=False,
        standing_quality_certified=False,
        capability_scope='same checkpoint fixed-height/PD-age matrix; stage-specific, not hardware certification')


def assess(result,arrays,eid,row):
    import numpy as np
    from .motion_mcu_course import assess as base
    base(result,arrays,eid,row)
    target=row['command'][2]
    if target==.34:return result
    mask=arrays['first_episode_valid'][:,eid] & (arrays['controller'][:,eid,0]==3)
    values=arrays['state'][mask,eid][:3000,6]
    checks=result['checks'];complete=len(values)==3000
    fields={'complete30':complete,'target_height_m':target,'window_s':[2,30]}
    if complete:
        error=np.abs(values[200:3000]-target);holding=np.abs(values[1000:1800]-target)
        fields.update(MAE_m=float(error.mean()),p95_m=float(np.quantile(error,.95)))
        checks.update(height=fields['MAE_m']<=.01,reacquisition_height=float(holding.mean())<=.01,
                      mcu_height=fields['MAE_m']<=.01,height_target_p95=fields['p95_m']<=.02)
        result['mcu_execution']['height_MAE_m']=fields['MAE_m']
    else:checks.update(height=False,reacquisition_height=False,mcu_height=False,height_target_p95=False)
    result.update(version=ASSESSMENT,height_tracking=fields,accepted=all(v is True for v in checks.values()),
                  failed_checks=[k for k,v in checks.items() if v is not True])
    return result


def startup_evidence(spec):
    """Reuse4500's unchanged034 matrix; physically assess only new task conditions."""
    old,records,_=handover.source_records(spec['motion_height_dr']['original_spec'],spec['source_checkpoint'])
    for key in ('mcu_execution','action','control','ground','asset_sha256'):
        if old[key]!=spec[key]:raise ValueError('unchanged034 source evidence plant differs: '+key)
    pose=next(p for p in spec['motion_height_dr']['height_poses'] if p['height_m']==.34)
    if pose!=spec['mcu_execution']['pose']:raise ValueError('034 reset pose differs from source')
    index={(tuple(r['case']['command']),r['case']['_mcu_variant']):r for r in records}
    reused=[];missing=[]
    for case in spec['student_evaluation']['cases']:
        if case.get('_execution_uncertainty') or case.get('_height'):
            missing.append(case);continue
        source=index[(tuple(case['command']),case['_mcu_variant'])]
        if any(source['case'][k]!=case[k] for k in ('seed','family','yaw_rad','seconds')):
            raise ValueError('034 evaluation window/seed differs from source')
        derived=copy.deepcopy(source)
        derived.update(case=case,reused=True,reuse_reference=dict(path=source['path'],sha256=source['sha256'],
            original_case=source['case'],scope='same4500 Actor/034 reset/MCU plant/scoring; new case metadata only'))
        reused.append(derived)
    return reused,missing


def learning_signal_audit(spec):
    """Evaluate actual density formulas; no simulator, optimizer or policy-gradient claim."""
    import torch
    from .motion_height import correct_reward
    from .motion_posture_design import posture_terms
    from .motion_symmetry_design import density
    from .motion_parallel_design import signals
    rows=[]
    def record(name,error,value,x,**context):
        derivative=float(torch.autograd.grad(value,x)[0])
        if error and (not torch.isfinite(value) or derivative*error>=0 or abs(derivative)<1e-6):
            raise ValueError('non-discriminating or reversed reward signal: '+name)
        rows.append(dict(term=name,error=error,weighted_density=float(value.detach()),
            derivative_per_physical_error=derivative,dt_contribution=float(value.detach())*.01,**context))
    for target in HEIGHTS:
        for error in (-.10,-.04,-.02,-.008,0.,.008,.02,.04,.10):
            x=torch.tensor(error,dtype=torch.float64,requires_grad=True);z=x.new_zeros(1,2)
            empty=dict(fields=('height','transfer_vertical_velocity'),raw=z,density=z,dt=z,
                total=x.new_zeros(1),eligible=torch.ones(1,dtype=torch.bool))
            result=correct_reward(empty,(x+target).reshape(1),x.new_tensor([target]),x.new_zeros(1),x.new_zeros(1))
            record('height',error,result['density'][0,0],x,target_height_m=target)
    for name,errors in (('leg',(-.10,-.008,-.001,0.,.001,.008,.10)),
                        ('roll',(-.30,-.05,-.01,0.,.01,.05,.30)),
                        ('swing',(-.20,-.05,-.005,0.,.005,.05,.20))):
        for error in errors:
            x=torch.tensor(error,dtype=torch.float64,requires_grad=True)
            value=(density(x,x*0,torch.stack((x*0,x*0)))['swing'] if name=='swing'
                else posture_terms(x if name=='leg' else x*0,x if name=='roll' else x*0)['new_'+name+'_density'])
            record(name,error,value,x)
    for axis in (0,1):
        for target in (-.5,.5):
            for error in (-2.,-.5,-.05,0.,.05,.5,2.):
                x=torch.tensor(error,dtype=torch.float64,requires_grad=True);command=x.new_zeros(1,2);command[0,axis]=target
                velocity=torch.stack((x+target if axis==0 else x*0,x*0)).reshape(1,2)
                out=signals(command,velocity,(x+target if axis==1 else x*0).reshape(1),
                    new_task=torch.zeros(1,dtype=torch.bool),mix=x.new_zeros(1))
                record('linear_tracking' if axis==0 else 'yaw_tracking',error,out['density'][0],x,target=target)
    cfg=stage_config(0);masked=[i for i,(v,h) in enumerate(zip(cfg['mcu_variants'],cfg['height_targets_m'])) if v==0 and h==.34]
    return dict(version='motion31_height_pd_signal_audit_v1',status='PASSED',curves=rows,
        dt_s=.01,total_clip=None,teacher_eligible_cohort_indices=masked,
        frozen_signal='original broad height Cauchy and continuous roll/leg Huber; existing evidence-driven widths and smoothness unchanged',
        delay_learning='PD age is a plant condition observed by Critic; reward physical tracking/posture/contact/smoothness, not age reduction',
        limitations=['density-error slopes are not PPO parameter gradients or a convergence guarantee',
            'independent perturbations do not prove all coupled physical objectives are jointly feasible'],
        source_sha256=spec['source_sha256'],simulation=0,optimizer_updates=0)


def build(parent,checkpoint,diagnostic,*,experiment,spec_path,session,online=ONLINE,replace_settled=None):
    s=handover.build(parent,checkpoint,diagnostic,experiment=experiment,spec_path=spec_path,session=session)
    original=Path(parent).resolve();old=json.loads(original.read_text())
    oldbook=original.with_name('budget.json');book=json.loads(oldbook.read_text())
    if book['used']['smoke']!=5 or book['used']['formal']!=4865:
        raise ValueError('power-loss accounting differs; audit actual consumption first')
    pose_spec=Path(old['mcu_execution']['source_spec']);poses=json.loads(pose_spec.read_text())['motion_decoupled']['height_poses']
    selected=[p for p in poses if p['height_m'] in HEIGHTS]
    if len(selected)!=len(HEIGHTS) or not all(p['offline_valid'] is True for p in selected):
        raise ValueError('six legal fixed-height reset candidates required')
    s['student_training']=recipe(s['source_sha256'],online)
    s['motion_course']=dict(source_recipe_updates=s['motion_course']['source_recipe_updates'],configuration=configuration(online))
    s['motion_height_dr']=dict(version=VERSION,state=initial_state(),height_poses=selected,
        pose_source=str(pose_spec),pose_source_sha256=sha(pose_spec),original_spec=str(original),
        original_budget=str(oldbook),original_budget_sha256=sha(oldbook),
        historical_online_used=4870,conservative_inflight_reserve=1,remaining_online=online,
        original_online_limit=10000,checkpoint_model_update=4500,
        superseded_held_spec=str(original.parent.parent/'2026_10_26-16_32__motion4500_handover_pdage1000/spec.json'))
    s['motion_execution_robustness']['version']=VERSION
    s['motion_execution_robustness']['configuration']=configuration(online)
    s['motion_execution_robustness']['configuration']['handover_prefix_policy_intervals']=100
    s['motion_execution_robustness']['configuration']['PD_feedback_range_ms']=[8,20]
    s['motion_assessment']=protocol();s['student_evaluation']['version']=ASSESSMENT
    s['budget'].update(source_eval_wall_s=7200,smoke_wall_s=None,formal_wall_s=None,chain_wall_s=None,
        evaluation_wall_s=7200,formal_updates=online-5,smoke_updates=5,
        formal_samples=(online-5)*4096*64,sample_limit=online*4096*64)
    s['artifacts'].update(storage_max_bytes=STORAGE,terminal_reserve_bytes=512*2**20)
    s['evaluation']['full_every_updates']=1000
    s['student_dependencies'].update({str(pose_spec):sha(pose_spec),str(oldbook):sha(oldbook)})
    s['authorization']=dict(status='UNAUTHORIZED',hapi_session=session,
        user_scope='User2026-10-26 agrees execute parallel034 timingDR and032/036standing then staged030..040; source4500; original10000 inclusive budget retains4870 plus1inflight reserve, newsmoke5+formal5124, unlimited wall, GPU0shared, TensorBoard6006 and owner notification; held1005 declaration untouched.')
    s['notifications']=dict(enabled=True,hapi_session=session)
    if online==10000:
        if replace_settled is None:raise ValueError('new10000 requires explicit settled predecessor')
        prior=Path(replace_settled).resolve();prior_spec=json.loads(prior.read_text())
        prior_run=original.parents[3]/'runs'/prior_spec['experiment']
        terminal_path=prior_run/'chain_terminal_report.json';terminal=json.loads(terminal_path.read_text())
        prior_book_path=prior.with_name('budget.json');prior_book=json.loads(prior_book_path.read_text())
        if (prior_spec.get('motion_height_dr',{}).get('original_spec')!=str(original)
                or prior_spec['source_sha256']!=s['source_sha256'] or terminal['status']!='STOPPED_BOUNDED'
                or terminal['failure'] is not None or prior_book['used']['smoke'] or prior_book['used']['formal']
                or any(a['status']=='RUNNING' for a in prior_book['attempts'].values())):
            raise ValueError('budget-only replacement requires settled same-source zero-online predecessor')
        s['motion_height_dr'].update(budget_scope='new10000_independent_of_historical4870',
            replace_settled=str(prior),replace_settled_sha256=sha(prior),
            historical_parallel_wall_s=prior_book.get('source_eval_wall_used_s',0.),
            count_original_storage=False,
            startup_scope='reuse_qualified034_only_new_tasks_UNKNOWN',
            storage_scope='newrun_only10GiB; previous runs preserved separately')
        for path in (prior,prior_book_path,terminal_path):s['student_dependencies'][str(path)]=sha(path)
        s['authorization']['user_scope']='User2026-10-26 explicitly grants this parallel course a NEW10000 online updates (smoke5+formal9995), unlimited training wall; supersedes remaining5129 after settled zero-online predecessor. Historical4870 and all actual prior wall/storage retained separately; independent10GiB, GPU0shared, original recipe/physics, TensorBoard6006 and current owner notification. No other held/stopped declaration is revived.'
    s['motion_startup']['missing_commands']=[]
    s=effective_spec(s,0,initial_state())
    s['motion_startup']['missing_commands']=[r['command'] for r in s['student_evaluation']['cases']]
    return s


def verify(spec):
    c=spec['motion_height_dr'];expected=build(c['original_spec'],spec['source_checkpoint'],
        spec['motion_execution_robustness']['diagnostic'],experiment=spec['experiment'],
        spec_path=spec['motion_training']['spec_path'],session=spec['authorization'].get('hapi_session'),
        online=spec['student_training'].get('authorized_online_limit',ONLINE),replace_settled=c.get('replace_settled'))
    for k in ('authorization','resources','notifications'):expected[k]=spec[k]
    if expected!=spec:raise ValueError('parallel height/PD declaration differs')
    if c.get('budget_scope')=='new10000_independent_of_historical4870':
        if spec['budget']['formal_updates']+spec['budget']['smoke_updates']!=10000:raise ValueError('new10000 ledger differs')
    elif 4870+1+spec['budget']['formal_updates']+spec['budget']['smoke_updates']!=10000:
        raise ValueError('historical consumption must not be refunded')
    return True
