"""Measured-start four-lane MCU curriculum; explicit data only, no file IO."""
import copy,math
from .motion_mcu_course import protocol,extend_mixture
from .motion_mcu_probe import SPEEDS

VERSION='motion31_mcu_measured_frontiers_v2'
AXES=('forward','backward','yaw_positive','yaw_negative')


def encode(counts,readiness):
    if len(counts)!=4 or any(type(x) is not int or not 0<=x<=6 for x in counts) or type(readiness) is not int or not 0<=readiness<=3:
        raise ValueError('explicit four mastered counts and zero readiness required')
    return sum(x*7**i for i,x in enumerate([*counts,readiness]))


def decode(stage):
    if type(stage) is not int or not 0<=stage<7**5:raise ValueError('invalid measured revision')
    values=[stage//7**i%7 for i in range(5)]
    if values[4]>3:raise ValueError('invalid standing readiness')
    return dict(counts=values[:4],readiness=values[4])


def target(axis,index):
    if axis not in range(4) or index not in range(6):raise ValueError('invalid axis/endpoint')
    value=SPEEDS[index]*(1 if axis%2==0 else -1)
    return [value,0.] if axis<2 else [0.,value]


def complete(record):
    es=record.get('per_env',[]) if record else []
    return (len(es)==8 and {e.get('env') for e in es}==set(range(8))
            and all(type(e.get('accepted')) is bool and isinstance(e.get('controlled_seconds'),(int,float))
                and math.isfinite(e['controlled_seconds']) and e['controlled_seconds']>=0
                and (e['accepted'] is False or e['controlled_seconds']>=30) for e in es))


def passed(record):return complete(record) and all(e['accepted'] for e in record['per_env'])


def key(row):return (row.get('_mcu_variant',0),tuple(row['command']))


def infer_stage(records):
    by={key(r['case']):r for r in records}
    if len(by)!=len(records):raise ValueError('duplicate capability case')
    readiness=0
    for variant in (0,1,2):
        if not passed(by.get((variant,(0.,0.,.34)))):break
        readiness+=1
    counts=[]
    for axis in range(4):
        n=0
        for i in range(6):
            if not passed(by.get((0,tuple(target(axis,i)+[.34])))):break
            n+=1
        counts.append(n)
    return encode(counts,readiness)


def configuration():
    return dict(version=VERSION,axes=list(AXES),endpoints=list(SPEEDS),envs=4096,
        evaluation_every=500,confirmations=1,behavior_stop=False,stage_updates_max=4870,
        allocation=dict(focus=.50,rehearsal=.35,zero=.15),
        physical_height_m=.34,skill_limit=4.,source_failure_blocks_learning=False,
        baseline='same original1500/current MCU execution, complete8env actual results',
        progression='each axis advances only its own complete endpoint and lower endpoints; no unrelated-axis blocker',
        regression='record only; never stop the chain for behavior',
        standing='only train missing normal/takeover/stress readiness; already-passed source conditions are retained')


def smoke_profile():
    from .motion_startup import smoke_profile as base
    p=base();p.update(version='motion31_mcu312_sample_wiring5_v1',
        scope='five real updates; declared cohort sampling/readback, not successful hold or survival')
    p['mandatory']=[x if x!='all_cohort_hold_samples' else 'all_cohort_pre_action_samples' for x in p['mandatory']]
    return p


def commands(stage=0):
    state=decode(stage);values=[[0.,0.]]
    for axis,n in enumerate(state['counts']):
        values += [target(axis,i) for i in range(min(n+(state['readiness']==3),6))]
    return values


def stage_config(stage=0,envs=4096):
    if envs not in (1,64,4096):raise ValueError('explicit MCU env count required')
    values=commands(stage);s=decode(stage);counts=[0]*len(values)
    if s['readiness']<3:
        old=values[1:];rehearsal=round(envs*.35) if old else 0;counts[0]=envs-rehearsal
        if old:
            q,r=divmod(rehearsal,len(old))
            for i,_ in enumerate(old):counts[i+1]=q+(i<r)
    else:
        focus=[target(a,n) for a,n in enumerate(s['counts']) if n<6]
        if not focus:focus=[target(a,5) for a in range(4)]
        old=[v for v in values[1:] if v not in focus]
        nz=round(envs*.15);nf=envs//2;nr=envs-nz-nf;counts[0]=nz
        for group,total in ((focus,nf),(old,nr)):
            if not group:counts[0]+=total;continue
            q,r=divmod(total,len(group))
            for i,v in enumerate(group):counts[values.index(v)]+=q+(i<r)
    return dict(id=stage,name='measured_'+str(stage),commands=values,
        slot_counts=counts,weights=[n/envs for n in counts],sampling_mode='fixed_slots',
        sampling_slots=[i for i,n in enumerate(counts) for _ in range(n)],enforce_whitelist=True)


def cases(spec,stage=0):
    original=spec['student_evaluation']['cases'][0]
    template={k:original[k] for k in ('kind','role','checkpoint','sha256','family','seed','yaw_rad','seconds')}
    rows=[dict(template,command=cmd+[.34],_mcu_variant=0) for cmd in commands(stage)]
    rows += [dict(template,command=[0.,0.,.34],_mcu_variant=v) for v in (1,2)]
    return [dict(r,case_id=f'mcu_measured_{stage}_{i}',_mcu_revision=stage,_timing_level=4) for i,r in enumerate(rows)]


def effective_spec(spec,stage):
    decode(stage);s=copy.deepcopy(spec)
    s['motion_training']['stage']=stage;s['student_training'].update(stage=stage,course_stage=stage)
    s['motion_timing']['stage']=4;s['curriculum']['stages']=[stage_config(stage,s['ppo']['envs'])]
    s['student_evaluation']['cases']=cases(s,stage)
    return s


def refinement_records(records,stage):
    # Nominal motion is scored independently; zero includes all available variants.
    by={key(r['case']):r for r in records};out=[]
    for cmd in commands(stage):
        record=copy.deepcopy(by.get((0,tuple(cmd+[.34])),dict(case=dict(command=cmd+[.34]),per_env=[])))
        if cmd==[0.,0.]:
            zero=[by.get((v,(0.,0.,.34))) for v in (0,1,2)]
            if all(complete(r) for r in zero):
                for e in record['per_env']:
                    e['accepted']=all(next(x for x in r['per_env'] if x['env']==e['env'])['accepted'] for r in zero)
        out.append(record)
    return out


def advance(state,candidate):
    stage=state['stage'];s=decode(stage);out=copy.deepcopy(state)
    if candidate.get('stage')!=stage or candidate['update']<=state.get('last_evaluation_update',-1):
        raise ValueError('measured course stale revision/update')
    records=candidate.get('evaluations',[])
    expected=[(0,tuple(cmd+[.34])) for cmd in commands(stage)]+[(v,(0.,0.,.34)) for v in (1,2)]
    valid=(candidate.get('complete_matrix') is True and [key(r['case']) for r in records]==expected
        and all(r['case'].get('_mcu_revision')==stage and complete(r) for r in records))
    counts=list(s['counts']);ready=s['readiness'];by={key(r['case']):r for r in records};advanced_axes=[]
    if valid:
        for v in range(ready,3):
            if not passed(by.get((v,(0.,0.,.34)))):break
            ready=v+1
        if s['readiness']==3 and passed(by[(0,(0.,0.,.34))]):
            for axis,n in enumerate(counts):
                if n<6 and all(passed(by.get((0,tuple(target(axis,i)+[.34])))) for i in range(n+1)):
                    counts[axis]+=1;advanced_axes.append(AXES[axis])
    next_stage=encode(counts,ready);advanced=next_stage!=stage
    full=bool(valid and all(passed(r) for r in records))
    out.update(stage=next_stage,last_evaluation_update=candidate['update'])
    if advanced:out['stage_started_update']=candidate['update']
    if full:out['accepted_by_stage'][str(stage)]={k:candidate[k] for k in ('checkpoint','sha256','update')}
    return out,dict(evidence_valid=bool(valid),stage_passed=full,advanced=advanced,
        advanced_axes=advanced_axes,mastered_counts=dict(zip(AXES,counts)),standing_readiness=ready,
        all_stages_passed=False,final_confirmed=full and counts==[6]*4 and ready==3,stop=False)


def choose(candidates,accepted_by_stage):
    from .motion_course import choose as base
    out=base(candidates,accepted_by_stage)
    if not any(c.get('update',0)>0 for c in candidates):out['current_stage']=candidates[0].get('stage',0)
    out.update(standing_quality_certified=False,final_multi_seed_certified=False,
        capability_scope='measured MCU standing and independently advanced +/-4 straight/in-place yaw; hypothetical plant')
    current=decode(out['current_stage'])
    if current['readiness']==3 and current['counts']==[6]*4:
        out['accepted_candidate']=next((c for c in reversed(candidates) if c.get('stage')==out['current_stage']
            and c.get('update',0)>0 and c.get('complete_matrix') is True and c.get('per_env')
            and all(e.get('accepted') is True for e in c['per_env'])),None)
    return out
