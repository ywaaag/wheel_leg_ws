"""Explicit persistent Motion frontiers; unchanged fixed-height184 scoring/plant.

All unmastered axes retain samples. Height and DR are condition axes, not
simultaneous vx/yaw commands. This module never reads files or simulator state.
"""
import copy
from . import motion_fixed_height as fixed, motion_decoupled as base, motion_robustness as dr

VERSION = 'motion31_persistent_frontiers_v1'
AXES = base.AXES
network = fixed.network
contract = fixed.contract
assess = fixed.assess
protocol = fixed.protocol  # Scoring, units and thresholds are unchanged.
extend_mixture = fixed.extend_mixture
refine_mixtures = fixed.refine_mixtures
refinement_records = fixed.refinement_records
choose = fixed.choose
STORAGE_BYTES = fixed.STORAGE_BYTES
STORAGE_PREFLIGHT = fixed.STORAGE_PREFLIGHT


def configuration():
    c = fixed.configuration()
    c.update(version=VERSION, scheduler='persistent independent frontiers, no focus rotation',
             unlock='complete per-axis matrix at its declared height/DR; other axes do not veto',
             frontier_fraction=.5, rehearsal_fraction=.35, nominal_zero_fraction=.15,
             failed_frontier='retain target and nonzero integer quota at every update',
             new_height_DR='nominal until final all-height motion matrix at certified DR',
             new_motion_DR='certified DR only; new DR trains mastered tasks only',
             no_moving_height=True, no_combined_vx_yaw=True,
             height_frontier_reward=dict(version='motion_height_frontier_bonus_v1',
                 multiplier=2., base_weight=2., bonus_weight=2., scale_m=.04,
                 scope='training height-frontier cohorts only; old tasks/review unchanged'))
    return c


def initial_state(previous):
    s = copy.deepcopy(previous)
    s.update(version=VERSION, scheduler_version=VERSION, focus='persistent', focus_level=0,
             stage=s['stage'] + 1, stage_started_update=0, last_evaluation_update=-1,
             DR_level=s['levels']['DR'], certified_DR_level=s['levels']['DR'],
             mode='PERSISTENT_FRONTIERS', repair_queue=[], frontier_evidence={},
             regression_streaks={})
    return s


def frontiers(state):
    result = {a: state['levels'][a] + 1 for a in AXES if state['levels'][a] < 5}
    if state['levels']['height'] < 4: result['height'] = state['levels']['height'] + 1
    if state['levels']['DR'] < len(dr.LEVELS) - 1: result['DR'] = state['levels']['DR'] + 1
    return result


def _rows(state):
    """Scenario, axis, purpose, DR level/group; only past/certified conditions."""
    level = max(0, state['levels']['DR'])
    old = [(v[0], v[1], .34, .34, False) for v in base.skill_commands(state)]
    result = [(x, 'review', 'rehearsal', level, 2) for x in old]
    if state['levels']['height'] >= 0:
        group = 2 if state['levels']['height'] >= 4 else 0
        result += [(x, 'height', 'rehearsal', level, group)
                   for x in fixed.height_retention_scenarios(state)]
    new = {}
    for axis, target in frontiers(state).items():
        if axis in AXES:
            vx, yaw = base.axis_command(axis, target)
            # New speed at each already learned fixed height, certified timing.
            new[axis] = [((vx, yaw, h, h, False), axis, 'frontier', level, 2)
                         for h in fixed.learned_heights(state)]
        elif axis == 'height':
            group = 2 if target == 4 else 0
            new[axis] = [(x, axis, 'frontier', level, group)
                         for x in fixed.height_scenarios(target, state, evaluation=True)]
        else:
            # Do not combine a new command/height with unconfirmed DR.
            scenarios = old
            if state['levels']['height'] >= 4: scenarios += fixed.height_retention_scenarios(state)
            new[axis] = [(x, axis, 'frontier', target, 2) for x in scenarios]
    return result, new


def stage_config(stage=0, envs=4096, *, state=None):
    if state is None or stage != state['stage'] or envs not in (1, 64, 4096):
        raise ValueError('persistent course requires exact revision and1/64/4096 envs')
    review, new = _rows(state)
    zero = round(envs * .15); learning = envs // 2 if new else 0
    rehearsal = envs - zero - learning
    if envs == 1: zero, learning, rehearsal = 1, 0, 0
    rows = [((0., 0., .34, .34, False), 'zero', 'nominal_zero', max(0,state['levels']['DR']),0)]
    counts = [zero]
    rows += review; counts += base._allocate(rehearsal, len(review))
    for (axis, part), quota in zip(new.items(), base._allocate(learning, len(new)) if new else []):
        rows += part; counts += base._allocate(quota, len(part))
    slots = [i for i,n in enumerate(counts) for _ in range(n)]
    if len(slots) != envs: raise ValueError('persistent slot quota does not sum to envs')
    if envs == 4096 and any(not sum(n for row,n in zip(rows,counts) if row[1]==a and row[2]=='frontier') for a in new):
        raise ValueError('persistent frontier was starved')
    return dict(id=stage,name='persistent_frontiers',commands=[list(x[0][:2]) for x in rows],
        weights=[n/envs for n in counts],sampling_mode='fixed_slots',sampling_slots=slots,
        slot_counts=counts,enforce_whitelist=True,cohort_keys=[list(x[0][:4]) for x in rows],
        height_scenarios=[list(x[0]) for x in rows],cohort_purposes=[x[2] for x in rows],
        cohort_axes=[x[1] for x in rows],
        group_slots=[rows[i][4] for i in slots],DR_level_slots=[rows[i][3] for i in slots],
        frontiers=frontiers(state))


def commands(stage=0,state=None): return stage_config(stage,state=state)['commands']


def cases(spec, stage=0, state=None):
    state = state or spec['motion_decoupled']['state']
    if state['stage'] != stage: raise ValueError('persistent matrix revision differs')
    template = spec['student_evaluation']['cases'][0]
    old, new = _rows(state); result = []; by_key = {}
    def add(item, retain=(), required=()):
        scenario, axis, purpose, level, group = item
        vx,yaw,initial,target,moving = scenario
        key = (vx,yaw,initial,target,moving,level,group)
        if key not in by_key:
            row = {k:copy.deepcopy(v) for k,v in template.items() if not k.startswith('_')}
            row.update(case_id=f'persistent_{stage}_{len(result)}',command=[vx,yaw,target],
                _timing_level=4,_height=dict(initial_m=initial,target_m=target,moving=False),
                _motion_dr=dict(level=level,group=group,seed=20261024+stage),
                _task_axis=axis,_purpose=purpose,_focus_required=False,
                _identity_key=list(key),_course_revision=stage,
                _frontier_axes=[],_retention_axes=[])
            by_key[key]=row;result.append(row)
        row=by_key[key]
        row['_frontier_axes']=sorted(set(row['_frontier_axes'])|set(required))
        row['_retention_axes']=sorted(set(row['_retention_axes'])|set(retain))
        row['_focus_required']=bool(row['_frontier_axes'])
    add(((0.,0.,.34,.34,False),'zero','retention',max(0,state['levels']['DR']),0))
    for item in old:
        xy=item[0][:2]
        if item[1]=='height': retain=['height']
        elif xy[0]: retain=['forward' if xy[0]>0 else 'backward']
        elif xy[1]: retain=['yaw_positive' if xy[1]>0 else 'yaw_negative']
        else:retain=[]
        add(item,retain=retain)
    for axis, rows in new.items():
        for item in rows:add(item,required=[axis])
    # Only three representative cases retain full traces, all results score8.
    for row in [x for x in result if x['_focus_required']][:3]:row['_full_diagnostic']=True
    return result


def effective_spec(spec,stage,state=None):
    s=copy.deepcopy(spec);state=copy.deepcopy(state or s['motion_decoupled']['state'])
    if state.get('scheduler_version')!=VERSION or state['stage']!=stage:raise ValueError('persistent scheduler declaration differs')
    rows=cases(s,stage,state);state['assessment_keys']=[x['_identity_key'] for x in rows]
    state['assessment_roles']=[[x['_frontier_axes'],x['_retention_axes']] for x in rows]
    s['motion_decoupled']['state']=state;s['motion_robustness']['state']=state
    s['motion_training']['stage']=stage
    s['student_training'].update(stage=stage,course_stage=stage,decoupled_state=state)
    s['curriculum']['stages']=[stage_config(stage,s['ppo']['envs'],state=state)]
    s['motion_timing']['stage']=4;s['student_evaluation']['cases']=rows
    return s


def advance(state,candidate):
    if candidate.get('stage')!=state['stage'] or candidate['update']<=state['last_evaluation_update']:
        raise ValueError('persistent checkpoint/revision differs')
    records=candidate.get('evaluations',[]);s=copy.deepcopy(state)
    valid=(candidate.get('complete_matrix') is True and bool(records)
           and all(base._complete(r) for r in records)
           and [r['case'].get('_identity_key') for r in records]==s.get('assessment_keys')
           and all(r['case'].get('_course_revision')==s['stage'] for r in records))
    promoted=[];lowered=[];outcomes={};targets=frontiers(state)
    # Every record's role tags are part of the declared assessment contract.
    expected=s.get('assessment_roles')
    if expected is not None:
        valid=valid and expected==[[r['case'].get('_frontier_axes'),r['case'].get('_retention_axes')] for r in records]
    if valid:
        for axis in (*AXES,'height'):
            old=[r for r in records if axis in r['case'].get('_retention_axes',[])]
            if not old:continue
            failed=any(e['accepted'] is False for r in old for e in r['per_env'])
            s['regression_streaks'][axis]=s['regression_streaks'].get(axis,0)+1 if failed else 0
            if s['regression_streaks'][axis]>=2 and s['levels'][axis]>=0:
                s['levels'][axis]-=1;s['regression_streaks'][axis]=0;lowered.append(axis)
        for axis,target in targets.items():
            rows=[r for r in records if axis in r['case'].get('_frontier_axes',[])]
            passed=bool(rows) and all(e['accepted'] is True for r in rows for e in r['per_env'])
            outcomes[axis]=dict(target_level=target,passed=passed,conditions=len(rows),
                failed_checks=sorted({k for r in rows for e in r['per_env'] for k in e.get('failed_checks',[])}))
            if passed and axis not in lowered:
                s['levels'][axis]=target;promoted.append(axis)
                proof=dict(checkpoint=candidate['checkpoint'],sha256=candidate['sha256'],update=candidate['update'],
                           focus=axis,focus_level=target)
                s['accepted_by_stage'][f"{s['stage']}:{axis}"]=proof
                s['frontier_evidence'][axis]=copy.deepcopy(proof)
        s['stage']+=1;s['stage_started_update']=candidate['update']
    s.update(last_evaluation_update=candidate['update'],last_checkpoint_sha256=candidate['sha256'],
             DR_level=max(0,s['levels']['DR']),certified_DR_level=s['levels']['DR'])
    done=all(s['levels'][a]>=5 for a in AXES) and s['levels']['height']>=4 and s['levels']['DR']>=5
    return s,dict(advanced=valid,stage_passed=bool(promoted),evidence_valid=valid,all_stages_passed=done,
        stop=False,focus='persistent',next_focus='persistent',promoted_axes=promoted,lowered_tasks=lowered,
        frontier_results=outcomes,levels=s['levels'],remaining_frontiers=frontiers(s),
        progression='persistent nonzero frontier quotas; per-axis promotion; no failure rotation')


def validate_recipe(recipe,source_sha):
    if recipe.get('course_version')!=VERSION or recipe['decoupled_state'].get('scheduler_version')!=VERSION:
        raise ValueError('persistent course explicit opt-in missing')
    if recipe.get('height_frontier_reward')!=configuration()['height_frontier_reward']:
        raise ValueError('height-frontier bonus declaration differs')
    old=copy.deepcopy(recipe);old['course_version']=fixed.VERSION;old['decoupled_state']['version']=fixed.VERSION
    fixed.validate_recipe(old,source_sha)
    return recipe


def verify(spec,*,require_current_claim=True):
    from .motion_persistent_resume import verify as verify_resume
    return verify_resume(spec,require_current_claim=require_current_claim)


def apply_height_bonus(result,height_frontier):
    """Add only the second copy of the existing2/(1+(height_error/.04)^2)."""
    import torch
    index=result['fields'].index('height');out=dict(result)
    raw=result['raw'][:,index]*height_frontier.to(result['raw'].dtype)
    density=raw*configuration()['height_frontier_reward']['bonus_weight'];dt=density*.01
    out.update(fields=(*result['fields'],'height_frontier_bonus'),
        raw=torch.cat((result['raw'],raw[:,None]),-1),
        density=torch.cat((result['density'],density[:,None]),-1),
        dt=torch.cat((result['dt'],dt[:,None]),-1),total=result['total']+dt)
    return out


def height_reward_snapshot():
    import torch
    from .data import sha
    curves=[]
    for error in (-.10,-.06,-.04,-.02,-.01,0.,.01,.02,.04,.06,.10):
        e=torch.tensor(error,dtype=torch.float64,requires_grad=True)
        base=2./(1+(e/.04).square());total=2*base
        curves.append(dict(error_m=error,base_density=base.item(),height_frontier_density=total.item(),
            derivative_per_m=torch.autograd.grad(total,e)[0].item(),dt_contribution=total.item()*.01))
    return dict(configuration=configuration()['height_frontier_reward'],formula_source_sha256=sha(__file__),
        curves=curves,total_clip=None,dt_s=.01,convergence_guaranteed=False)
