"""Explicit standing continuation: sensor20, target5, post-PD torque20 ms.

Tensor helpers are causal; build/verify alone own filesystem provenance.
The old312 Critic prefix is preserved, with99 new execution-state columns.
"""
import copy
import json
from pathlib import Path

import torch

from .data import sha

VERSION = 'motion31_standing_sensor20_target5_motor20_v1'
EXECUTION = 'motion_mcu_sensor20_target5_motor20_v1'
CRITIC = 'motion_standing_delay_privileged411_v1'
NETWORK = 'motion31_actor2048_delaycritic411_v1'
ASSESSMENT = 'motion31_standing_delay20_assessment_v1'
ONLINE, FORMAL, STORAGE = 10000, 9995, 10 * 2**30


def execution_profile():
    from .motion_mcu_execution import profile
    p = profile()
    p.update(version=EXECUTION, actor_age_ms=20, PD_age_ms=0,
             local_target_delay_ms=5, motor_delay_ms=[20]*6,motor_tau_ms=[2]*6,
             source='user-declared fixed stress hypothesis; not identified hardware timing')
    return p


def network():
    from .actor_widening import wide_network
    p = wide_network(2048)
    p.update(version=NETWORK, critic_input=411)
    return p


def contract():
    from .motion_mcu_privileged import contract as base
    c = base()
    c.update(critic_dim=411, critic_privileged_version=CRITIC)
    c['critic_privileged'] += [
        'published_motor_commands_ages15_to20ms:36 (Nm/driver_limit)',
        'pending_physical_target_ages1_to5ms:30 (leg/1.5, wheel/100)',
        'queued_sensor_packets_newest_then_previous:32 (same16D physical encoding)',
        'local_target_transport_ms:1 (ms/20)',
    ]
    c['privileged_scope'] = 'past published motor/target queues and sampled sensor packets only; no future actions'
    return c


def expand_source(source):
    from .actor_widening import wide_network
    from .motion_mcu_privileged import VERSION as OLD
    if source.get('network_architecture') == network() and source.get('critic_privileged_version') == CRITIC:
        return source
    if source.get('network_architecture') != wide_network(2048) or source.get('critic_privileged_version') != OLD:
        raise ValueError('delay411 requires explicit2048 Actor and312 Critic source')
    out = copy.deepcopy(source)
    w = out['critic']['mlp.0.weight']
    if w.shape != (512, 312):raise ValueError('source Critic312 first layer differs')
    def pad(x):
        if x.shape != w.shape:raise ValueError('source Critic Adam tensor differs')
        return torch.cat((x, x.new_zeros(512, 99)), -1)
    out['critic']['mlp.0.weight'] = pad(w)
    ids = [i for g in out['optimizer']['param_groups'] for i in g['params']]
    if len(ids) != len(out['actor']) + len(out['critic']):raise ValueError('Adam parameter ordering differs')
    pid = ids[len(out['actor']) + list(out['critic']).index('mlp.0.weight')]
    for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        if key in out['optimizer']['state'].get(pid, {}):
            out['optimizer']['state'][pid][key] = pad(out['optimizer']['state'][pid][key])
    out.update(network_architecture=network(), critic_privileged_version=CRITIC)
    return out


def critic_input(base, motor_history, target_history, sensor_packets, tick, driver_limits):
    n = len(base)
    if base.shape != (n,312) or sensor_packets.shape != (n,2,16):
        raise ValueError('delay411 explicit base/queued sensor shape differs')
    if motor_history.shape[0] < 21 or target_history.shape[0] < 6 or type(tick) is not int:
        raise ValueError('delay411 causal history/clock missing')
    ages = torch.arange(15,21,device=base.device)
    motors = motor_history[(tick-ages)%len(motor_history)].permute(1,0,2)
    ages = torch.arange(1,6,device=base.device)
    targets = target_history[(tick-ages)%len(target_history)].permute(1,0,2)
    scales = base.new_tensor([1.5,1.5,100.,1.5,1.5,100.])
    extra = torch.cat(((motors/driver_limits[:,None]).flatten(1),
                       (targets/scales).flatten(1),sensor_packets.flatten(1),
                       base.new_full((n,1),5./20)), -1)
    if extra.shape != (n,99):raise ValueError('delay411 extension dimension differs')
    return torch.cat((base,extra),-1)


def timing_configuration(parent):
    c = copy.deepcopy(parent)
    c['standing_delay_execution'] = EXECUTION
    c['actor_sensor'].update(imu_delay_ms=20,joint_delay_ms=20,
        sampling='100Hz synchronous packets, fixed20ms; reset fills actual initial measurements')
    c['actor_sensor']['critic'] = 'delayed Actor31 prefix; current physical state and complete pending queues in411D Critic'
    return c


def validate_timing(c):
    from .motion_antioscillation import validate_configuration
    old = copy.deepcopy(c)
    if old.pop('standing_delay_execution',None) != EXECUTION:
        raise ValueError('explicit delay execution version required')
    from .motion_sensor_transfer import configuration
    old['actor_sensor'] = configuration()['actor_sensor']
    old['actor_sensor']['critic'] = 'Actor31/Critic140 unchanged; delayed Actor prefix and current physical privileged columns'
    validate_configuration(old)
    if c != timing_configuration(old):raise ValueError('standing delay sensor configuration drift')
    return c


def configuration():
    return dict(version=VERSION,formal_updates_max=FORMAL,smoke_updates_max=5,
        stage_updates_max=FORMAL,evaluation_every=500,
        evaluation_updates=[100,*range(500,FORMAL,500),FORMAL],
        progression=False,behavior_stop=False,height_m=.34,
        allocation=dict(normal_standing=.75,small_initial_velocity_standing=.25),
        sensor_delay_ms=20,target_delay_ms=5,PD_feedback_age_ms=0,motor_transport_ms=20,
        physical_DR=False,source_ability_not_learning_prerequisite=True,
        standing_only=True,motion_capability='not trained or certified by this recipe')


def initial_state():
    return dict(stage=0,stage_started_update=0,last_evaluation_update=-1,
        consecutive_passes=0,accepted_by_stage={},standing_passed=False)


def commands(stage=0):
    if stage != 0:raise ValueError('fixed standing stage only')
    return [[0.,0.],[0.,0.]]


def stage_config(stage=0,envs=4096,*,state=None):
    if stage != 0 or envs not in (1,64,4096):raise ValueError('standing delay stage/env differs')
    counts = [envs-envs//4,envs//4]
    return dict(id=0,name='standing_sensor20_target5_motor20',commands=commands(),
        slot_counts=counts,weights=[v/envs for v in counts],sampling_mode='fixed_slots',
        sampling_slots=[i for i,n in enumerate(counts) for _ in range(n)],enforce_whitelist=True,
        cohort_keys=[[0.,0.,.34,i] for i in (0,1)],mcu_variants=[0,1],
        height_targets_m=[.34,.34],cohort_purposes=['standing_normal','standing_initial_velocity'],
        prefix_intervals=[0,0])


def protocol():
    from .motion_highspeed_standing import protocol as base
    p = base()
    p.update(version=ASSESSMENT,primary_scope='fixed034 zero command standing at sensor20/target5/postPD20',
        motion_capability_certified=False,delays=execution_profile())
    return p


def cases(spec,stage=0,state=None):
    if stage != 0:raise ValueError('fixed standing assessment stage differs')
    old = spec['student_evaluation']['cases'][0]
    template = {k:old[k] for k in ('kind','role','family','seed','yaw_rad','seconds')}
    template.update(checkpoint=spec['source_checkpoint'],sha256=spec['source_sha256'],
        command=[0.,0.,.34],_timing_level=4,_mcu_revision=0,_primary_required=True,
        _standing_delay_execution=EXECUTION)
    return [dict(template,case_id=f'standing_delay20_variant{v}',_mcu_variant=v) for v in (0,1)]


def identity(row):
    return [row['command'],row['_mcu_variant'],row.get('_standing_delay_execution')]


def effective_spec(spec,stage,state=None):
    s = copy.deepcopy(spec)
    current = copy.deepcopy(state or s['motion_height_dr']['state'])
    if stage != 0 or current['stage'] != 0:raise ValueError('fixed standing revision differs')
    s['motion_training']['stage']=0
    s['curriculum']['stages']=[stage_config(0,s['ppo']['envs'])]
    s['student_evaluation']['cases']=cases(s)
    current['assessment_keys']=[identity(r) for r in s['student_evaluation']['cases']]
    s['motion_height_dr']['state']=current;s['motion_course']['state']=current
    s['student_training'].update(stage=0,course_stage=0,height_dr_state=current)
    return s


def advance(state,candidate):
    from .motion_mcu_measured import complete
    rows=candidate.get('evaluations',[])
    if candidate.get('stage') != 0 or candidate['update'] <= state['last_evaluation_update']:
        raise ValueError('standing delay stale assessment')
    valid=bool(candidate.get('complete_matrix') is True and rows and
        [identity(r['case']) for r in rows] == state['assessment_keys'] and all(complete(r) for r in rows))
    passed=bool(valid and all(e.get('accepted') is True for r in rows for e in r['per_env']))
    out=copy.deepcopy(state)
    out.update(last_evaluation_update=candidate['update'],standing_passed=passed,
        consecutive_passes=state['consecutive_passes']+1 if passed else 0)
    if passed:out['accepted_by_stage']['0']={k:candidate[k] for k in ('checkpoint','sha256','update')}
    return out,dict(evidence_valid=valid,stage_passed=passed,standing_passed=passed,
        advanced=False,all_stages_passed=False,final_confirmed=passed,stop=False)


def choose(candidates,accepted_by_stage):
    from .motion_highspeed_standing import choose as base
    out=base(candidates,accepted_by_stage)
    out.update(capability_scope='fixed034 standing under sensor20/target5/postPD20 hypothesis; not motion/hardware certification',
        selection='standing full-window acceptance and survival; all conditions from one checkpoint')
    return out


def extend_mixture(previous,values):
    from .motion_mcu_course import extend_mixture as base
    return base(previous,values)


def refinement_records(records,stage):
    return [copy.deepcopy(next((r for r in records if r['case'].get('_mcu_variant')==v),
        dict(case=dict(command=[0.,0.,.34]),per_env=[]))) for v in (0,1)]


def assess(result,arrays,eid,row):
    from .motion_highspeed_standing import assess as base
    base(result,arrays,eid,row)
    result['version']=ASSESSMENT
    return result


def recipe(source_sha,timing):
    from .motion_highspeed_standing import recipe as base
    r=base(source_sha)
    for k in ('source_policy_retention','continuation_teacher'):r.pop(k,None)
    r.update(course_version=VERSION,critic_privileged_version=CRITIC,network_architecture=network(),
        critic='source312_zero_extend_to411',optimizer='source_Adam_zero_new411_columns',
        timing_configuration=copy.deepcopy(timing),height_dr_state=initial_state(),
        milestones=configuration()['evaluation_updates'],warmup_updates_per_phase=20,
        selection_scope='stable standing034 at fixed sensor20/target5/torque20; no motion promotion')
    return r


def validate_recipe(r,source_sha):
    validate_timing(r['timing_configuration'])
    expected=recipe(source_sha,r['timing_configuration'])
    for key in ('height_dr_state','stage','course_stage'):expected[key]=r.get(key,expected.get(key))
    if r != expected:raise ValueError('standing delay411 recipe drift')
    return r


def learning_signal_audit(spec):
    from .motion_height_pd import learning_signal_audit as base
    out=base(spec)
    from .motion_reward_audit import audit
    curves=audit(spec['reward'])
    required=('linear_zero','yaw_zero','pitch','wheel_offset','hip_mirror')
    selected=[r for r in curves['curves'] if r['signal'] in required and
        r['error']>(.01 if r['signal']=='wheel_offset' else 0)]
    if not selected or any(r['state_error_derivative']>=0 for r in selected):
        raise ValueError('standing reward direction or tail discrimination differs')
    from .motion_antioscillation import reward_snapshot
    out.update(version=VERSION+'_signal_audit',teacher_eligible_cohort_indices=[],
        standing_zero_and_posture_curves=selected,
        retained_deadbands=dict(wheel_center_offset_m=.01,pitch_tail_rad=.10),
        antioscillation=reward_snapshot(spec['motion_timing']['configuration']),
        delay_learning='reward actual standing/posture/contact/effort, never reward a smaller declared delay',
        critic_observes='full20ms published motor queue,5ms target queue, queued sensor packets; no future actions')
    return out


def build(parent,checkpoint,*,experiment,spec_path,session,replaces_unlaunched=None,recover_preppo=None):
    p,cp=Path(parent).resolve(),Path(checkpoint).resolve()
    old=json.loads(p.read_text());run=p.parents[3]/'runs'/old['experiment']
    book=json.loads(p.with_name('budget.json').read_text())
    terminal=json.loads((run/'formal/terminal_report.json').read_text())
    chain=json.loads((run/'chain_terminal_report.json').read_text())
    from .model import load
    _,source=load(cp,'MotionZeroStudent')
    if (old.get('motion_highspeed_standing',{}).get('version')!='motion31_mcu_highspeed_standing2048_v1'
        or chain['status']!='STOPPED_BOUNDED' or terminal.get('failure')
        or terminal.get('stop_kind')!='user_signal' or any(a['status']=='RUNNING' for a in book['attempts'].values())):
        raise ValueError('settled user-stopped Actor2048 source required')
    if (cp != run/'formal'/f"model_update_{book['used']['formal']}.pt"
        or source['update']!=terminal['actual_updates'] or source['spec_sha256']!=sha(p)):
        raise ValueError('exact complete final checkpoint/spec required')
    expand_source(source)
    s=copy.deepcopy(old)
    for key in ('motion_highspeed_standing','motion_execution_robustness','mcu_posture','mcu_measured',
                'checkpoint_quick_evaluation','motion_large_continuation'):
        s.pop(key,None)
    digest=sha(cp)
    s.update(experiment=experiment,source_checkpoint=str(cp),source_sha256=digest,
        network_architecture=network(),student_interface=contract(),
        observation={**old['observation'],'critic_dim':411})
    timing=timing_configuration(old['motion_timing']['configuration'])
    s['motion_timing'].update(configuration=timing)
    s['student_training']=recipe(digest,timing)
    s['motion_training'].update(stage=0,resume=None,spec_path=str(Path(spec_path).resolve()))
    s['motion_course']=dict(source_recipe_updates=source['recipe_updates'],configuration=configuration())
    paths=(p,p.with_name('budget.json'),run/'formal/terminal_report.json',run/'chain_terminal_report.json',cp)
    s['motion_standing_delay']=dict(version=VERSION,parent_spec=str(p),parent_spec_sha256=sha(p),
        source_checkpoint_sha256=digest,historical_online_updates=book['used']['formal']+book['used']['smoke'],
        original_stopped_budget_not_reused=True,total_online_limit=ONLINE,
        source_capability_in_new_plant='UNKNOWN, not a prerequisite for learning',
        inputs={str(q):sha(q) for q in paths})
    if replaces_unlaunched:
        prior=Path(replaces_unlaunched).resolve()
        prior_spec=json.loads(prior.read_text());prior_book=json.loads(prior.with_name('budget.json').read_text())
        if (prior_spec.get('motion_standing_delay',{}).get('version')!=VERSION
            or prior_spec['source_sha256']!=digest or prior_book['attempts'] or any(prior_book['used'].values())):
            raise ValueError('tau update may replace only an exact unlaunched zero-consumption declaration')
        s['motion_standing_delay']['replaces_unlaunched']=str(prior)
        s['motion_standing_delay']['inputs'].update({str(prior):sha(prior),str(prior.with_name('budget.json')):sha(prior.with_name('budget.json'))})
    if recover_preppo:
        prior=Path(recover_preppo).resolve();prior_spec=json.loads(prior.read_text())
        prior_book=json.loads(prior.with_name('budget.json').read_text())
        prior_run=prior.parents[3]/'runs'/prior_spec['experiment']
        prior_terminal=json.loads((prior_run/'smoke/terminal_report.json').read_text())
        prior_chain=json.loads((prior_run/'chain_terminal_report.json').read_text())
        if (prior_spec.get('motion_standing_delay',{}).get('version')!=VERSION
            or prior_spec['source_sha256']!=digest or prior_chain['status']!='FAILED'
            or any(a['status']=='RUNNING' for a in prior_book['attempts'].values())
            or prior_book['used']['smoke'] or prior_book['used']['formal']
            or prior_terminal['actual_updates']!=0
            or prior_terminal.get('failure',{}).get('exception')!="ValueError('absolute posture protocol mismatch')"):
            raise ValueError('only exact settled zero-PPO absolute-protocol dispatch failure can recover')
        s['motion_standing_delay']['preppo_recovery']=dict(version='motion_standing_delay_protocol_preppo_recovery_v1',
            parent_spec=str(prior),parent_run=str(prior_run),historical_online_updates=0,
            historical_wall_s=prior_chain['chain_wall_s'],reason='new explicit course protocol runtime dispatch missing')
        for item in (prior,prior.with_name('budget.json'),prior_run/'smoke/terminal_report.json',prior_run/'chain_terminal_report.json'):
            s['motion_standing_delay']['inputs'][str(item)]=sha(item)
    s['motion_height_dr']=dict(version=VERSION,state=initial_state(),
        height_poses=[x for x in old['motion_height_dr']['height_poses'] if x['height_m']==.34],
        count_original_storage=False,startup_scope='source_full_state_only_new_delay_ability_UNKNOWN')
    s['mcu_execution']['profile']=execution_profile()
    s['motion_assessment']=protocol()
    s['student_evaluation'].update(version=ASSESSMENT,max_output_bytes=128*2**20)
    s['evaluation'].update(full_every_updates=500,checkpoint_every_updates=100,
        unlock_full_updates=[],protocol_identity=ASSESSMENT,full_trace_updates=[])
    s['budget'].update(source_eval_wall_s=1800,smoke_wall_s=None,formal_wall_s=None,chain_wall_s=None,
        evaluation_wall_s=7200,smoke_updates=5,formal_updates=FORMAL,
        formal_samples=FORMAL*4096*64,sample_limit=ONLINE*4096*64)
    s['artifacts'].update(run_name=experiment,storage_max_bytes=STORAGE,terminal_reserve_bytes=512*2**20)
    s['performance'].update(evaluation_group_commands=True,evaluation_group_max_commands=8)
    # Bind used inputs, not historical teacher/initialization files that this recipe never reads.
    s['student_dependencies']={**s['motion_standing_delay']['inputs'],s['asset_path']:s['asset_sha256']}
    s['motion_startup'].update(references=[],missing_commands=[])
    s['authorization']=dict(status='UNAUTHORIZED',hapi_session=session,
        user_scope='User2026-10-29 explicitly approves NEW10000 total PPO updates, unlimited training wall; resume stopped3896 Actor2048, zero-extend Critic312 to411 and Adam, fixed standing034, sensor20ms/Actor-to-PD target5ms/current-PD feedback/post-PD motor20ms/all motor response tau2ms; other physical gains/reward/LR/std unchanged; GPU0shared, smoke5/formal9995,100save,100first/500full,TensorBoard6006 and existing owner notification.')
    s['notifications']=dict(enabled=True,hapi_session=session)
    s['resources']=dict(mode='shared',sharing_authorized=True,authorization_note='User-authorized same GPU0 Motion standing delay continuation; no preemption')
    return effective_spec(s,0)


def verify(spec):
    info=spec['motion_standing_delay']
    expected=build(info['parent_spec'],spec['source_checkpoint'],experiment=spec['experiment'],
        spec_path=spec['motion_training']['spec_path'],session=spec['authorization'].get('hapi_session'),
        replaces_unlaunched=info.get('replaces_unlaunched'),
        recover_preppo=info.get('preppo_recovery',{}).get('parent_spec'))
    for field in ('authorization','notifications','resources'):expected[field]=spec[field]
    if expected != spec:raise ValueError('standing delay411 declaration drift')
    validate_recipe(spec['student_training'],spec['source_sha256'])
    claim=Path(spec['motion_training']['spec_path']).parents[1]/'.motion_standing_delay_claims'/spec['source_sha256']
    if Path(spec['motion_training']['spec_path']).is_file() and claim.is_file() and json.loads(claim.read_text())['spec']!=spec['motion_training']['spec_path']:
        raise ValueError('standing delay authorization already consumed by another spec')
    return True


def startup_evidence(spec):
    candidate=dict(checkpoint=spec['source_checkpoint'],sha256=spec['source_sha256'],update=0,stage=0,
        per_env=[],evaluations=[],complete_matrix=False,evaluated=False,
        source_scope='exact complete saved Actor/Critic/Adam/std/RNG; new delay capability UNKNOWN')
    return [],[candidate]
