"""Explicit-data posture rehearsal mathematics and fixed course; no IO or RNG."""
import copy
import torch

VERSION = 'motion31_mcu_posture_rehearsal_design_v1'
SOURCE_SHA = '9de0a63d46548bd100882929ef19a4dd16897d91a44d3a250b833fd4350559a7'
REWARD_VERSION = 'motion31_mcu_dense_posture_candidate_v1'
RETENTION_VERSION = 'motion31_mcu_nominal_retention_candidate_v2'
ENDPOINTS = (.5, 1., 1.5, 2., 3., 4.)
SOURCE_COUNTS = (5, 5, 6, 5)
COURSE_VERSION = 'motion31_mcu_posture_rehearsal_v1'
RUNTIME_REWARD_VERSION = 'motion31_mcu_dense_posture_reward_v1'


def commands(stage=0):
    if stage!=0:raise ValueError('fixed posture commands have no unlocked higher stage')
    values = []
    for axis, count in enumerate(SOURCE_COUNTS):
        for speed in ENDPOINTS[:count]:
            value = speed if axis % 2 == 0 else -speed
            values.append([value, 0.] if axis < 2 else [0., value])
    return [[0., 0.], *values]


def allocation(envs=4096):
    """Distinct variant keys prevent stress-zero from inheriting nominal BC."""
    if type(envs) is not int or envs != 4096:
        raise ValueError('this design freezes4096 environments')
    values = commands()
    rows = [dict(target=[0., 0.], variant=2, slots=1024, retention_id=-1),
            dict(target=[0., 0.], variant=0, slots=1024, retention_id=0)]
    q, remainder = divmod(2048, len(values) - 1)
    rows += [dict(target=target, variant=0, slots=q+(i < remainder), retention_id=i+1)
             for i, target in enumerate(values[1:])]
    return [dict(row, cohort_key=[*row['target'], .34, row['variant']]) for row in rows]


def retention_ids(targets, variants, valid):
    if targets.ndim != 2 or targets.shape[-1] != 2 or variants.shape != valid.shape or len(targets) != len(valid):
        raise ValueError('explicit episode target/variant/control masks required')
    if variants.dtype != torch.long or valid.dtype != torch.bool:
        raise ValueError('variant categorical/valid boolean required')
    result = torch.full_like(variants, -1)
    for i, command in enumerate(commands()):
        selected = valid & (variants == 0) & (targets == targets.new_tensor(command)).all(-1)
        result = torch.where(selected, i, result)
    return result


def balanced_policy_loss(student_mean, teacher_mean, fixed_std, ids, coefficient=.01):
    """Equal cohort/channel KL; frozen teacher detached, no distribution cache."""
    n = len(ids)
    if student_mean.shape != (n, 6) or teacher_mean.shape != (n, 6) or fixed_std.shape != (6,):
        raise ValueError('explicit31D Actor output/frozen6D std contract required')
    count = len(commands())
    selected = (ids >= 0) & (ids < count)
    kl = .5 * ((student_mean-teacher_mean.detach()) / fixed_std.detach()).square()
    counts = torch.zeros(count, device=ids.device, dtype=torch.long).scatter_add_(0, ids.clamp(0, count-1), selected.long())
    sums = student_mean.new_zeros(count).scatter_add_(0, ids.clamp(0, count-1),
                                                  torch.where(selected, kl.mean(-1), 0.))
    loss = coefficient * (sums/counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)
    return loss, kl, counts


def huber(value):
    absolute = value.abs()
    return torch.where(absolute <= 1, .5*value.square(), absolute-.5)


def posture_terms(leg_difference_m, roll_rad):
    """Existing terms are replaced, not duplicated; one dt scaling downstream."""
    old_leg = -.15 * huber(leg_difference_m/.02)
    new_leg = -.15 * huber(leg_difference_m/.008)
    old_roll = -.5 * huber(torch.relu(roll_rad.abs()-.05)/.10)
    new_roll = -.5 * huber(roll_rad/.05)
    return dict(old_leg_density=old_leg, new_leg_density=new_leg,
                old_roll_density=old_roll, new_roll_density=new_roll,
                density_delta=(new_leg-old_leg)+(new_roll-old_roll), dt_s=.01)


def declaration():
    return dict(version=VERSION, status='CPU_DESIGN_NOT_RUNTIME_RECIPE', source_sha256=SOURCE_SHA,
        height_m=.34, nominal_commands=commands(), cohorts=allocation(),
        progression=False, extra_height_or_speed=False,
        reward=dict(version=REWARD_VERSION, leg_scale_m=.008, leg_weight=-.15,
                    roll_scale_rad=.05, roll_weight=-.5, dt_s=.01,
                    replacement='leg_balance and roll_excess only; no duplicate cost or total clip',
                    unchanged='pitch/tracking/height/swing/wheel alignment/antioscillation/control'),
        retention=dict(version=RETENTION_VERSION, coefficient=.01,
                       mask='episode target + nominal variant0 + valid control, stress2 excluded',
                       commands=commands(), channel_reduction='mean6', cohort_reduction='equal observed cohort mean',
                       teacher='frozen original1500, no control/no optimizer',
                       caveat='current policy states only; source replay not implemented; coefficient benefit unproven'),
        safety=dict(physical_faults='retained', behavior_chain_stop=False,
                    source_ability_not_learning_prerequisite=True,
                    qualification='each accepted old nominal command requires existing same-MCU8/8 evidence'))


def configuration():
    return dict(version=COURSE_VERSION,formal_updates_max=9995,smoke_updates_max=5,
        stage_updates_max=9995,evaluation_every=500,confirmations=1,behavior_stop=False,
        progression=False,height_m=.34,allocation=dict(stress_zero=.25,nominal_zero=.25,old_motion=.5))


def stage_config(stage=0,envs=4096):
    if stage!=0 or envs not in (1,64,4096):raise ValueError('fixed posture stage/env count differs')
    rows=allocation();counts=[r['slots'] for r in rows]
    if envs!=4096:
        counts=[0]*len(rows)
        if envs==1:counts[1]=1
        else:
            counts[:2]=[16,16]
            q,r=divmod(32,21);counts[2:]=[q+(i<r) for i in range(21)]
    return dict(id=0,name='fixed034_posture_rehearsal',commands=[r['target'] for r in rows],
        slot_counts=counts,weights=[n/envs for n in counts],sampling_mode='fixed_slots',
        sampling_slots=[i for i,n in enumerate(counts) for _ in range(n)],enforce_whitelist=True,
        cohort_keys=[r['cohort_key'] for r in rows],mcu_variants=[r['variant'] for r in rows])


def cases(spec,stage=0):
    if stage!=0:raise ValueError('no automatic posture course promotion')
    original=spec['student_evaluation']['cases'][0]
    template={k:original[k] for k in ('kind','role','checkpoint','sha256','family','seed','yaw_rad','seconds')}
    return [dict(template,case_id=f'posture_{i}',command=cmd+[.34],_mcu_variant=v,
                 _mcu_revision=0,_timing_level=4)
            for i,(v,cmd) in enumerate([(0,c) for c in commands()]+[(1,[0.,0.]),(2,[0.,0.])])]


def effective_spec(spec,stage):
    s=copy.deepcopy(spec);s['motion_training']['stage']=stage
    s['student_training'].update(stage=stage,course_stage=stage);s['motion_timing']['stage']=4
    s['curriculum']['stages']=[stage_config(stage,s['ppo']['envs'])]
    s['student_evaluation']['cases']=cases(s,stage)
    return s


def extend_mixture(previous,values):
    from .motion_mcu_course import extend_mixture as base
    return base(previous,values)


def refinement_records(records,stage):
    from .motion_mcu_measured import key,complete
    by={key(r['case']):r for r in records};out=[]
    # Sampler has two distinct zero identities; keep both reward states aligned.
    for variant,cmd in [(2,[0.,0.])]+[(0,c) for c in commands()]:
        r=copy.deepcopy(by.get((variant,tuple(cmd+[.34])),dict(case=dict(command=cmd+[.34]),per_env=[])))
        if cmd==[0.,0.] and not complete(r):r['per_env']=[]
        out.append(r)
    return out


def advance(state,candidate):
    from .motion_mcu_measured import key,complete,passed
    if candidate.get('stage')!=0 or candidate['update']<=state.get('last_evaluation_update',-1):raise ValueError('posture matrix revision/update differs')
    rows=candidate.get('evaluations',[])
    expected=[(0,tuple(c+[.34])) for c in commands()]+[(v,(0.,0.,.34)) for v in (1,2)]
    valid=bool(candidate.get('complete_matrix') is True and [key(r['case']) for r in rows]==expected
               and all(r['case'].get('_mcu_revision')==0 and complete(r) for r in rows))
    full=bool(valid and all(passed(r) for r in rows));out=copy.deepcopy(state)
    out.update(last_evaluation_update=candidate['update'],consecutive_passes=state.get('consecutive_passes',0)+1 if full else 0)
    if full:out['accepted_by_stage']['0']={k:candidate[k] for k in ('checkpoint','sha256','update')}
    return out,dict(evidence_valid=valid,stage_passed=full,advanced=False,all_stages_passed=False,
                    final_confirmed=full,stop=False,high_speed_expansion=False)


def choose(candidates,accepted_by_stage):
    from .motion_course import choose as base
    out=base(candidates,accepted_by_stage)
    new=[c for c in candidates if c.get('update',0)>0 and c.get('complete_matrix') and c.get('per_env')]
    complete=[c for c in new if all(e.get('accepted') is True for e in c['per_env'])]
    def rank(c):return (sum(e.get('accepted') is True for e in c.get('per_env',[])),
                        sum(e.get('survival_passed') is True for e in c.get('per_env',[])))
    best=max([candidates[0],*new],key=rank)
    out.update(best_overall=best,best_observed_candidate=best)
    out['accepted_candidate']=complete[-1] if complete else None
    out.update(capability_scope='fixed034 nominal motion plus takeover/stress standing; hypothetical MCU plant',
               final_multi_seed_certified=False,standing_quality_certified=False)
    return out


def apply_runtime_reward(result,posture):
    """Replace existing raw/density/dt entries while preserving terminal events."""
    indices=[result['fields'].index(k) for k in ('leg_balance','roll_excess')]
    raw=torch.stack((huber((posture[:,2]-posture[:,3])/.008),huber(posture[:,1]/.05)),-1)
    raw=raw*result['eligible'][:,None];density=raw*raw.new_tensor([-.15,-.5]);dt=density*.01
    old=result['dt'][:,indices].clone()
    result['raw'][:,indices]=raw;result['density'][:,indices]=density;result['dt'][:,indices]=dt
    result['total']+=dt.sum(-1)-old.sum(-1);result['posture_extension_version']=RUNTIME_REWARD_VERSION
    return result
