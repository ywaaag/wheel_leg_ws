"""Read-only evidence boundary for a CPU-reviewed, non-submittable Motion plan."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from . import motion_posture_design as design
from .data import sha


def build_plan(parent, checkpoint, assessment):
    parent, checkpoint, assessment = map(lambda p: Path(p).resolve(), (parent, checkpoint, assessment))
    old = json.loads(parent.read_text()); book = json.loads(parent.with_name('budget.json').read_text())
    root = parent.parents[3]; run = root/'runs'/old['experiment']
    terminal = json.loads((run/'chain_terminal_report.json').read_text())
    if terminal['status'] != 'STOPPED_BOUNDED' or terminal['failure'] is not None or any(a['status']=='RUNNING' for a in book['attempts'].values()):
        raise ValueError('settled manually stopped Motion parent required')
    used = book['historical_online_updates'] + book['used']['formal'] + book['used']['smoke']
    if used != terminal['online_updates_including_parent'] or used != 1296:
        raise ValueError('actual charged parent consumption differs')
    if sha(checkpoint) != design.SOURCE_SHA:
        raise ValueError('this design starts at the original1500, not current500/final666')
    source = json.loads(assessment.read_text())
    if source['source_sha256'] != design.SOURCE_SHA:
        raise ValueError('assessment belongs to another Actor')
    records = {}
    verified = []
    for record in source['records']:
        case = record['case']; key = (case.get('_mcu_variant',0), tuple(case['command']))
        if key in records: raise ValueError('duplicate source command/variant evidence')
        raw_path = Path(record['path']); raw = json.loads(raw_path.read_text())
        if sha(raw_path) != record['sha256'] or raw['checkpoint_sha256'] != design.SOURCE_SHA or raw['case'] != case:
            raise ValueError('source raw JSON/Actor identity differs')
        origin = raw.get('derived_from', {})
        trace = Path(origin.get('trace') or str(raw_path.parent/raw['trace']['path']))
        digest = origin.get('trace_sha256') or raw['trace']['sha256']
        if sha(trace) != digest: raise ValueError('source trace identity differs')
        inputs = raw.get('protocol', {}).get('execution_profile')
        if inputs != old['mcu_execution']['profile']:
            raise ValueError('source MCU execution profile differs')
        envs = record['per_env']
        accepted = len(envs)==8 and {e.get('env') for e in envs}==set(range(8)) and all(e.get('accepted') is True for e in envs)
        records[key] = dict(record=record, raw=raw, trace=trace, passed=accepted)
        verified.append(dict(case_key=[*case['command'],key[0]], raw=str(raw_path), raw_sha256=sha(raw_path),
                             trace=str(trace), trace_sha256=digest, passed=accepted))
    for command in design.commands():
        if not records.get((0,tuple(command+[.34])),{}).get('passed'):
            raise ValueError('every nominal retained command must have same-profile8/8 evidence')
    if records.get((2,(0.,0.,.34)),{}).get('passed') is not False:
        raise ValueError('stress is a learning task, not source-certified retention')
    plan = design.declaration()
    plan.update(parent_spec=str(parent), parent_spec_sha256=sha(parent),
                source_checkpoint=str(checkpoint), assessment=str(assessment), assessment_sha256=sha(assessment),
                source_evidence=verified, execution=copy.deepcopy(old['mcu_execution']),
                ppo=copy.deepcopy(old['ppo']), network_architecture=copy.deepcopy(old['network_architecture']),
                original_reward=copy.deepcopy(old['reward']), timing_configuration=copy.deepcopy(old['motion_timing']['configuration']),
                resources=copy.deepcopy(old['device']), physics_scope='current Native MCU physical hypotheses; MuJoCo solver candidate is evaluation-only',
                budget=dict(original_limit=5000, historical_used=used, unspent=5000-used,
                            new_online_authorized=0, limits_not_assigned=True, no_automatic_reuse=True),
                cadence=dict(checkpoint_every=100, complete_matrix_every=500, scored_envs=8,
                             required_matrix_conditions=24, tensorboard='existing6006 on future launch',
                             matrix='22 nominal commands plus takeover/stress zero, no new speed/height'),
                initialization=dict(actor='original1500 full',critic='184->312 semantic zero-extension',
                    Adam='retain old columns, zero new columns',std='frozen source',RNG='source checkpoint',
                    physical_state='new declared reset, not memory resume',
                    critic_warmup='proposed20formal critic-only plus smoke4critic+1actor, charged within any future assigned budget'),
                implementation_status=dict(pure_math='CPU candidate', runtime_dispatch='NOT_CONNECTED',
                    sampler_variant_metadata='NOT_CONNECTED', retention_PPO_hook='NOT_CONNECTED',
                    reward_runtime='NOT_CONNECTED', public_schema='NOT_REGISTERED',
                    physical_readback='NOT_RUN', submit_ready=False, training_started=False),
                unresolved=['MuJoCo remaining zero drift; do not train toward a migration bias',
                            'current-state teacher anchoring has no source-state replay',
                            'retention .01 provisional; audit actual PPO-vs-anchor gradients and critic by cohort',
                            'PPO/runtime/schema/smoke consume the new explicit recipe only after integration',
                            'new run/spec/ledger and execution authorization required; old stopped chain stays stopped'])
    return plan, records


def cpu_audit(plan, records):
    """No physical sampling or optimizer: verify exact initialization and loss signal."""
    from .model import load, Student
    from .motion_mcu_privileged import expand_source, network
    teacher, saved = load(plan['source_checkpoint'], 'MotionZeroStudent'); teacher.eval().requires_grad_(False)
    expanded = expand_source(saved)
    actor_equal = all(torch.equal(v, expanded['actor'][k]) for k,v in saved['actor'].items())
    w0, w1 = saved['critic']['mlp.0.weight'], expanded['critic']['mlp.0.weight']
    with torch.random.fork_rng(devices=[]):
        critic = Student(False, version='multiskill_mcu_v2',
                         critic_privileged_version='motion_mcu_motorqueue_privileged312_v1',network_architecture=network())
    critic.load_state_dict(expanded['critic'])
    old_columns_preserved = torch.equal(w1[:,:184],w0) and torch.count_nonzero(w1[:,184:]).item()==0
    def equal(a, b):
        if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a,b)
        if isinstance(a, np.ndarray): return isinstance(b, np.ndarray) and np.array_equal(a,b)
        if isinstance(a, dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a, (tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    expected_optimizer = copy.deepcopy(saved['optimizer'])
    padded_states = 0
    for state in expected_optimizer['state'].values():
        for key,value in list(state.items()):
            if key in ('exp_avg','exp_avg_sq','max_exp_avg_sq') and isinstance(value,torch.Tensor) and value.shape==w0.shape:
                state[key]=torch.cat((value,value.new_zeros(512,128)),-1);padded_states+=1
    optimizer_preserved = padded_states>=2 and equal(expected_optimizer,expanded['optimizer'])
    rng_preserved = 'training_rng' in saved and equal(saved['training_rng'],expanded['training_rng'])
    std_preserved = torch.equal(saved['actor']['log_std'],expanded['actor']['log_std'])
    curves = []
    for term, errors in [('leg',[0,.004,.008,.015,.032,.05,.1]), ('roll',[0,.02,.034,.05,.073,.1,.2])]:
        for error in errors:
            value = torch.tensor(error,dtype=torch.float64,requires_grad=True)
            result = design.posture_terms(value if term=='leg' else value*0, value if term=='roll' else value*0)
            old, new = result[f'old_{term}_density'], result[f'new_{term}_density']
            curves.append(dict(term=term,error=error,old_density=float(old.detach()),new_density=float(new.detach()),
                derivative=float(torch.autograd.grad(new,value)[0]),dt_contribution=float(new.detach())*.01))
    # These are existing teacher states, labelled by the fixed episode command.
    # They do not constitute on-policy PPO advantages or a new rollout.
    xs, ids, used = [], [], []
    for cid, command in enumerate(design.commands()):
        r = records[(0,tuple(command+[.34]))]
        with np.load(r['trace'], allow_pickle=False) as trace:
            if 'actor_input' not in trace:
                used.append(dict(command=command,status='ACTOR_INPUT_NOT_SAVED'));continue
            times=trace['policy_time_s'];mask=(times>=10)&(times<18)
            if not trace['first_episode_valid'][mask,0].all():raise ValueError('retained source hold trace incomplete')
            obs=trace['actor_input'][mask,0][::25].copy()
        xs.append(torch.from_numpy(obs));ids.extend([cid]*len(obs));used.append(dict(command=command,samples=len(obs)))
    observations=torch.cat(xs);labels=torch.tensor(ids,dtype=torch.long)
    with torch.no_grad(): reference=teacher.mean(observations)
    same = reference.clone().requires_grad_(True)
    loss,_,_=design.balanced_policy_loss(same,reference,teacher.log_std.exp(),labels)
    grad=torch.autograd.grad(loss,same)[0]
    # Small, explicitly synthetic mean displacement checks which direction the
    # regularizer pushes. It is never passed to the simulator as an action.
    shifted=(reference+.001).requires_grad_(True)
    shifted_loss,_,counts=design.balanced_policy_loss(shifted,reference,teacher.log_std.exp(),labels)
    shifted_gradient=torch.autograd.grad(shifted_loss,shifted)[0]
    return dict(status='PASSED',simulation=0,PPO_updates=0,offline_optimizer_steps=0,
        source_actor_exact=actor_equal,critic_old184_columns_and_zero_new128=old_columns_preserved,
        source_Adam_old_state_preserved_new_columns_zero=optimizer_preserved,
        source_training_RNG_preserved=rng_preserved,source_std_preserved=std_preserved,
        initialization_scope='migration tensors checked; no optimizer step or physical state restored',
        reward_curves=curves,retention_data=used,retention_observations=len(observations),
        source_self_KL=float(loss.detach()),source_self_KL_gradient_norm=float(grad.norm()),
        shifted_mean_KL=float(shifted_loss.detach()),shifted_mean_gradient_norm=float(shifted_gradient.norm()),
        shifted_mean_gradient_toward_teacher=bool((shifted_gradient>0).all()),
        retained_observed_cohorts=int((counts>0).sum()),
        scope='pure loss/initialization CPU audit; not PPO credit assignment, physical wiring or convergence evidence',
        no_new_source_tree_or_checkpoint_copy=True,submit_ready=False)


def prepare(parent, checkpoint, assessment, output=None):
    plan, records = build_plan(parent,checkpoint,assessment)
    if output is None:
        return dict(status=plan['status'],cpu_design=plan['version'],submit_ready=False,files_written=0,
                    source_checkpoint=plan['source_checkpoint'],cohorts=plan['cohorts'],budget=plan['budget'])
    destination=Path(output).resolve();destination.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    audit=cpu_audit(plan,records)
    if not all(audit[k] for k in ['source_actor_exact','critic_old184_columns_and_zero_new128',
        'source_Adam_old_state_preserved_new_columns_zero','source_training_RNG_preserved',
        'source_std_preserved','shifted_mean_gradient_toward_teacher']):
        raise ValueError('CPU design audit failed')
    snapshot=dict(status='CANDIDATE_NOT_ACTIVE_RUNTIME_REWARD',version=design.REWARD_VERSION,
        source_checkpoint_sha256=plan['source_sha256'],parent_spec_sha256=plan['parent_spec_sha256'],
        assessment_sha256=plan['assessment_sha256'],formula_source=str(Path(design.__file__).resolve()),
        formula_source_sha256=sha(Path(design.__file__)),original_reward=plan['original_reward'],
        replacements=plan['reward'],curves=audit['reward_curves'],
        formula='leg_balance=-.15*Huber(delta_length_m/.008); roll_excess=-.5*Huber(roll_rad/.05)',
        dt='weighted continuous density*.01 once; terminal event semantics unchanged',
        signal_scope='derivative versus physical error; not a PPO parameter gradient or convergence proof',
        retention='separate Actor loss, not an environment reward; no dt multiplication')
    for name,data in [('recipe_design.json',plan),('cpu_audit.json',audit),('reward_snapshot.json',snapshot)]:
        with (destination/name).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
    return dict(status='CPU_DESIGN_READY_RUNTIME_INTEGRATION_PENDING',submit_ready=False,
                output=str(destination),files_written=3,simulation=0,PPO_updates=0,
                cohorts=len(plan['cohorts']),retained_commands=len(plan['nominal_commands']))


STORAGE = 10*2**30


def recipe(source_sha):
    from .motion_training import declaration
    from .motion_mcu_privileged import network,VERSION
    from .motion_mcu_measured import smoke_profile
    from .motion_source_retention import nominal_declaration
    from .motion_wheel_matching import configuration
    timing=configuration();timing['smooth_mix']=.75
    r=declaration(source_sha,0,reward_version='student_motion31_parallel_quality_v10')
    r.update(critic_privileged_version=VERSION,network_architecture=network(),
        course_version=design.COURSE_VERSION,course_stage=0,timing_configuration=timing,
        critic='source184_zero_new_motor_columns312',optimizer='source_Adam_zero_new_motor_columns312',
        milestones=list(range(500,9996,500)),checkpoint_every_updates=100,storage_max_bytes=STORAGE,
        smoke_wiring=smoke_profile(),warmup_updates_per_phase=20,
        posture_reward=dict(version=design.RUNTIME_REWARD_VERSION,leg_scale_m=.008,roll_scale_rad=.05,
                            leg_weight=-.15,roll_weight=-.5,dt_s=.01),
        source_policy_retention=nominal_declaration(source_sha),
        selection_scope='fixed034 stress standing/horizontal/symmetric and all source-qualified nominal motion; no promotion')
    return r


def validate_recipe(value,source_sha):
    if source_sha!=design.SOURCE_SHA or value!=recipe(source_sha):
        raise ValueError('posture training recipe undeclared drift')
    return value


def build_training(parent,checkpoint,assessment,*,experiment,spec_path,owner_session):
    from .motion_mcu_course import protocol
    from .motion_mcu_privileged import contract,network
    plan,_=build_plan(parent,checkpoint,assessment)
    p=Path(parent).resolve();s=json.loads(p.read_text())
    for key in ('mcu_measured','motion_large_continuation','checkpoint_quick_evaluation'):
        s.pop(key,None)
    s.update(experiment=experiment,evaluation_only=False,source_checkpoint=plan['source_checkpoint'],
             source_sha256=design.SOURCE_SHA,network_architecture=network(),student_interface=contract())
    s['student_training']=recipe(design.SOURCE_SHA)
    s['motion_training']={k:v for k,v in s['motion_training'].items() if k in ('version','block_index','early_stop_policy')}
    s['motion_training'].update(stage=0,spec_path=str(Path(spec_path).resolve()),resume=None)
    s['motion_timing'].update(parent_spec=str(p),parent_spec_sha256=sha(p),source_checkpoint_sha256=design.SOURCE_SHA,stage=4)
    s['mcu_posture']=dict(version=design.COURSE_VERSION,parent_spec=str(p),parent_spec_sha256=sha(p),
        assessment=plan['assessment'],assessment_sha256=plan['assessment_sha256'],source_checkpoint=plan['source_checkpoint'],
        historical_online_used_separate=1296,previous_unspent_not_transferred=3704,new_online_limit=10000,
        physical_hypotheses_unchanged=True,source_reward_not_replayed=True)
    s['motion_course']={'source_recipe_updates':s['motion_course']['source_recipe_updates'],'configuration':design.configuration()}
    s['motion_assessment']=protocol();s['student_evaluation']['version']=protocol()['version']
    s['student_evaluation']['cases']=[{**s['student_evaluation']['cases'][0],
        'checkpoint':s['source_checkpoint'],'sha256':design.SOURCE_SHA}]
    s['motion_symmetry']['stage_scope']='fixed034 nominal learned motion and stress standing; no new endpoints'
    s['budget'].update(smoke_updates=5,formal_updates=9995,formal_samples=9995*4096*64,
        sample_limit=10000*4096*64,formal_wall_s=None,chain_wall_s=None,
        source_eval_wall_s=600,smoke_wall_s=3600,evaluation_wall_s=7200)
    s['artifacts'].update(run_name=experiment,storage_max_bytes=STORAGE,terminal_reserve_bytes=512*2**20)
    s['evaluation'].update(full_every_updates=500,full_trace_updates=[])
    s['performance'].update(evaluation_group_commands=True,evaluation_group_max_commands=8)
    s['student_dependencies'].update({str(p):sha(p),s['source_checkpoint']:design.SOURCE_SHA,
                                    plan['assessment']:plan['assessment_sha256']})
    s['authorization']={'status':'UNAUTHORIZED','hapi_session':owner_session,
        'user_scope':'User2026-10-25 approves new10000 inclusive online updates(smoke5+formal9995 incl20critic), unlimited training wall, original1500/fixed034/25stress25nominal50old-motion/dense roll-leg costs/nominal teacherKL.01, GPU0 shared, owner notification and TensorBoard6006; old1296 separate, no old stopped-chain restart.'}
    s['notifications']={'enabled':True,'hapi_session':owner_session}
    s['resources']={'mode':'shared','sharing_authorized':True,'authorization_note':'User authorizes GPU0 shared for fixed034 posture retention10000; no preemption'}
    return design.effective_spec(s,0)


def verify(spec):
    c=spec['mcu_posture']
    expected=build_training(c['parent_spec'],c['source_checkpoint'],c['assessment'],experiment=spec['experiment'],
        spec_path=spec['motion_training']['spec_path'],owner_session=spec['authorization'].get('hapi_session'))
    for key in ('authorization','resources','notifications'):expected[key]=spec[key]
    if spec!=expected:raise ValueError('posture training declaration differs')
    validate_recipe(spec['student_training'],spec['source_sha256'])
    return True


def reused_source(spec):
    _,records=build_plan(spec['mcu_posture']['parent_spec'],spec['source_checkpoint'],spec['mcu_posture']['assessment'])
    rows=[copy.deepcopy(records[(r.get('_mcu_variant',0),tuple(r['command']))]['record']) for r in spec['student_evaluation']['cases']]
    candidate=dict(checkpoint=spec['source_checkpoint'],sha256=spec['source_sha256'],update=0,stage=0,
        per_env=[e for r in rows for e in r['per_env']],complete_matrix=True,evaluations=rows,
        scope='same frozen Actor/execution/scoring; new reward and312 Critic require new engineering smoke')
    return rows,[candidate]


def runtime_reward_snapshot():
    rows=[]
    for name,values in [('leg',[0,.004,.008,.015,.032,.05,.1]),('roll',[0,.02,.034,.05,.073,.1,.2])]:
        for error in values:
            x=torch.tensor(error,dtype=torch.float64,requires_grad=True)
            result=design.posture_terms(x if name=='leg' else x*0,x if name=='roll' else x*0)
            value=result[f'new_{name}_density']
            rows.append(dict(term=name,error=error,density=float(value.detach()),
                derivative=float(torch.autograd.grad(value,x)[0]),dt_contribution=float(value.detach())*.01))
    return dict(version=design.RUNTIME_REWARD_VERSION,formula_source_sha256=sha(design.__file__),
        weights=[-.15,-.5],scales=[.008,.05],dt_s=.01,curves=rows,
        replaces=['leg_balance','roll_excess'],total_clip=None,teacher_loss='separate optimizer auxiliary, no dt')
