"""Explicit new5000-update MCU standing/takeover/low-speed curriculum."""
import copy
import json
from pathlib import Path
import torch
from .data import sha
from .motion_mcu_execution import profile, WHEEL_CAP

VERSION='motion31_mcu_standing_adaptation_v1'
ASSESSMENT='motion31_mcu_standing_assessment_v1'
SOURCE_SHA='9de0a63d46548bd100882929ef19a4dd16897d91a44d3a250b833fd4350559a7'
TARGETS=(0,1,2,3)
STORAGE=4*2**30


def configuration():
    return dict(version=VERSION,formal_updates_max=4995,smoke_updates_max=5,
                stage_updates_max=4995,evaluation_every=500,confirmations=1,
                stages=['normal_standing','takeover_initial_hypothesis','small_execution_stress','low_speed_recovery'],
                behavior_stop=False,old_DR_proofs_imported=False)


def commands(stage=0):
    if stage not in TARGETS:raise ValueError('MCU course stage invalid')
    return [[0.,0.]] if stage<3 else [[0.,0.],[.5,0.],[-.5,0.],[0.,.5],[0.,-.5]]


def stage_config(stage=0,envs=4096):
    if envs not in (1,64,4096):raise ValueError('MCU course envs invalid')
    values=commands(stage);counts=[envs] if stage<3 else [envs//2]+[envs//8]*4
    if envs==1:counts=[1]+[0]*4 if stage==3 else [1]
    return dict(id=stage,name=configuration()['stages'][stage],commands=values,slot_counts=counts,
                weights=[n/envs for n in counts],sampling_mode='fixed_slots',
                sampling_slots=[i for i,n in enumerate(counts) for _ in range(n)],enforce_whitelist=True)


def protocol():
    from .motion_reacquisition import protocol as base
    p=base();p.update(version=ASSESSMENT,execution_profile=profile(),holding_window_s=[10.,18.],
        mcu_standing_window_s=[2.,30.],driver_saturation_max=.05,
        temporal_scope='first30 controlled seconds, all8 environments; no reselected stable window',
        confirmations_required=1)
    return p


def cases(spec,stage=0):
    original=spec['student_evaluation']['cases'][0]
    template={k:original[k] for k in ('kind','role','checkpoint','sha256','family','seed','yaw_rad','seconds')}
    variants=[0] if stage==0 else [0,1] if stage==1 else [0,1,2]
    return [{**template,'case_id':f'mcu_{stage}_{v}_{i}','command':cmd+[.34],
             '_timing_level':4,'_mcu_variant':v,'_mcu_revision':stage}
            for v in variants for i,cmd in enumerate(commands(stage))]


def effective_spec(spec,stage):
    s=copy.deepcopy(spec)
    s['motion_training']['stage']=stage;s['student_training'].update(stage=stage,course_stage=stage)
    s['motion_timing']['stage']=4
    s['curriculum']['stages']=[stage_config(stage,s['ppo']['envs'])]
    s['student_evaluation']['cases']=cases(s,stage)
    return s


def extend_mixture(previous,values):
    result=[]
    for value in values:
        matches=[r for r in previous if r['target'][:2]==value and (len(r['target'])==2 or r['target'][2:]==[.34,.34])]
        old=matches[0] if matches else {}
        result.append({**copy.deepcopy(old),'target':list(value),'mix':old.get('mix',0.),'consecutive_eligible':0})
    return result


def refinement_records(records,stage):
    # Every physical variant must agree, never count them as multiple confirmations.
    output=[]
    for cmd in commands(stage):
        rows=[r for r in records if r['case']['command']==cmd+[.34]]
        expected=1 if stage==0 else 2 if stage==1 else 3
        envs=[]
        if len(rows)==expected and all(len(r['per_env'])==8 for r in rows):
            for eid in range(8):
                es=[next(e for e in r['per_env'] if e['env']==eid) for r in rows]
                e=copy.deepcopy(es[0]);e['accepted']=all(e.get('accepted') is True for e in es)
                envs.append(e)
        output.append({'case':{'command':cmd+[.34]},'per_env':envs})
    return output


def advance(state,candidate):
    s=copy.deepcopy(state);stage=s['stage'];records=candidate.get('evaluations',[])
    if candidate['stage']!=stage or candidate['update']<=s['last_evaluation_update']:raise ValueError('MCU course stale revision/update')
    expected=[(v,i) for v in ([0] if stage==0 else [0,1] if stage==1 else [0,1,2]) for i in range(len(commands(stage)))]
    valid=candidate.get('complete_matrix') is True and len(records)==len(expected)
    for r,(v,i) in zip(records,expected):
        c=r['case'];es=r['per_env']
        valid=valid and c.get('_mcu_variant')==v and c.get('_mcu_revision')==stage and c.get('command')==commands(stage)[i]+[.34]
        valid=valid and len(es)==8 and {e.get('env') for e in es}==set(range(8))
    passed=bool(valid and all(e.get('accepted') is True for r in records for e in r['per_env']))
    s['last_evaluation_update']=candidate['update']
    if passed:
        s['accepted_by_stage'][str(stage)]={k:candidate[k] for k in ('checkpoint','sha256','update')}
        if stage<3:s.update(stage=stage+1,stage_started_update=candidate['update'])
    return s,dict(evidence_valid=bool(valid),stage_passed=passed,advanced=passed and stage<3,
                  all_stages_passed=False,final_confirmed=passed and stage==3,stop=False)


def choose(candidates,accepted_by_stage):
    from .motion_course import choose as base
    out=base(candidates,accepted_by_stage)
    out['standing_quality_certified']=False
    out['capability_scope']='new hypothetical MCU execution; per-stage standing/takeover/low-speed; not hardware certification'
    return out


def validate_recipe(recipe,source_sha):
    from .motion_training import declaration
    from .motion_fixed_height import CRITIC,network
    from .motion_startup import smoke_profile
    if source_sha!=SOURCE_SHA:raise ValueError('MCU source must be evaluated parent1500')
    base=declaration(source_sha,recipe['stage'],reward_version='student_motion31_parallel_quality_v10')
    base.update(critic_privileged_version=CRITIC,network_architecture=network(),course_version=VERSION,
        course_stage=recipe['stage'],timing_configuration=recipe['timing_configuration'],
        critic='source_same_semantic184_new_plant',optimizer='source_Adam_same_semantic184',
        milestones=list(range(500,4996,500)),checkpoint_every_updates=100,storage_max_bytes=STORAGE,
        smoke_wiring=smoke_profile(),warmup_updates_per_phase=20,
        selection_scope='MCU standing034, then declared takeover hypotheses and low-speed; no old DR certification')
    from .motion_wheel_matching import configuration as timing_config
    timing_expected=timing_config();timing_expected['smooth_mix']=.75
    if recipe['stage'] not in TARGETS or recipe['timing_configuration']!=timing_expected or recipe!=base:
        raise ValueError('MCU recipe has undeclared drift')
    return recipe


def build(parent,checkpoint,*,experiment,spec_path,owner_session):
    from .model import load
    from .motion_fixed_height import network,contract
    from .motion_training import declaration
    from .motion_startup import smoke_profile
    p=Path(parent).resolve();cp=Path(checkpoint).resolve();old=json.loads(p.read_text());_,source=load(cp,'MotionZeroStudent')
    if sha(cp)!=SOURCE_SHA or source['spec_sha256']!=sha(p) or source.get('update')!=1500:
        raise ValueError('exact evaluated1500 checkpoint/spec required')
    s=copy.deepcopy(old)
    for key in ('motion_decoupled','motion_robustness','checkpoint_quick_evaluation','motion_parallel_assessment'):
        s.pop(key,None)
    s.update(experiment=experiment,source_checkpoint=str(cp),source_sha256=SOURCE_SHA)
    s['network_architecture']=network();s['student_interface']=contract()
    recipe=declaration(SOURCE_SHA,0,reward_version=s['reward']['version'])
    recipe.update(critic_privileged_version='motion_height_robust_privileged184_v1',network_architecture=network(),
        course_version=VERSION,course_stage=0,timing_configuration=old['motion_timing']['configuration'],
        critic='source_same_semantic184_new_plant',optimizer='source_Adam_same_semantic184',
        milestones=list(range(500,4996,500)),checkpoint_every_updates=100,storage_max_bytes=STORAGE,
        smoke_wiring=smoke_profile(),warmup_updates_per_phase=20,
        selection_scope='MCU standing034, then declared takeover hypotheses and low-speed; no old DR certification')
    s['student_training']=recipe
    s['mcu_execution']={'profile':profile(),'pose':next(x for x in old['motion_decoupled']['height_poses'] if x['height_m']==.34),
        'source_spec':str(p),'source_spec_sha256':sha(p),'authority':'user2026-10-25 explicit5000 unlimited wall',
        'old_consumption_separate':{'latest10000_used':4848,'latest10000_unspent_not_transferred':5152,'earlier_history':869}}
    s['motion_timing']={'version':'motion31_standing_antiosc_armature_v4','configuration':copy.deepcopy(recipe['timing_configuration']),
        'stage':4,'parent_spec':str(p),'parent_spec_sha256':sha(p),'source_checkpoint_sha256':SOURCE_SHA}
    s['motion_training']={k:v for k,v in s['motion_training'].items() if k in ('version','block_index','early_stop_policy')}
    s['motion_training'].update(stage=0,spec_path=str(Path(spec_path).resolve()),resume=None)
    s['motion_course']={'configuration':configuration(),'source_recipe_updates':source['recipe_updates']}
    s['motion_assessment']=protocol();s['motion_symmetry']['stage_scope']='MCU standing and low-speed only'
    from .motion_guard import behavior_mode
    s['motion_behavior_guard']=behavior_mode()
    s['student_evaluation']['version']=ASSESSMENT
    s['student_evaluation']['cases']=[{**old['student_evaluation']['cases'][0],'checkpoint':str(cp),'sha256':SOURCE_SHA}]
    s['student_evaluation']['cases']=cases(s)
    s['motion_startup'].update(references=[],missing_commands=[[0.,0.,.34]])
    s['ppo'].update(envs=4096,steps_per_env=64)
    s['curriculum']['stages']=[stage_config()];s['curriculum']['active_stage']=0
    s['evaluation'].update(full_every_updates=500,protocol_identity=ASSESSMENT,full_trace_updates=[])
    s['budget'].update(smoke_updates=5,formal_updates=4995,formal_samples=4995*4096*64,
        sample_limit=5000*4096*64,formal_wall_s=None,chain_wall_s=None,
        source_eval_wall_s=3600,smoke_wall_s=3600,evaluation_wall_s=7200)
    s['artifacts'].update(run_name=experiment,storage_max_bytes=STORAGE,terminal_reserve_bytes=512*2**20)
    s['performance'].update(evaluation_group_commands=True,evaluation_group_max_commands=4)
    s['student_dependencies']={**old['student_dependencies'],str(p):sha(p),str(cp):SOURCE_SHA}
    s['authorization']={'status':'UNAUTHORIZED','hapi_session':owner_session,'user_scope':'User2026-10-25 approves new MCU execution standing/takeover/low-speed5000 inclusive online updates(smoke5+formal4995 incl20critic), unlimited training wall, GPU0shared, notification and TensorBoard6006; parameters hypotheses, no hardware'}
    s['notifications']={'enabled':True,'hapi_session':owner_session}
    s['resources']={'mode':'shared','sharing_authorized':True,
        'authorization_note':'User2026-10-25 approves MCU execution adaptation5000 inclusive, GPU0 shared; no preemption'}
    return s


def verify(spec):
    if spec.get('motion_highspeed_standing'):
        from .motion_highspeed_standing import verify as verify_highspeed
        return verify_highspeed(spec)
    if spec.get('motion_height_dr'):
        from .motion_height_pd import verify as verify_height_pd
        return verify_height_pd(spec)
    if spec.get('motion_execution_robustness'):
        from .motion_handover_robustness import verify as verify_handover
        return verify_handover(spec)
    if spec.get('mcu_posture'):
        from .motion_posture_preparation import verify as verify_posture
        return verify_posture(spec)
    if spec.get('mcu_measured'):
        from .motion_mcu_measured_io import verify as verify_measured
        return verify_measured(spec)
    if spec.get('mcu_baseline_probe'):
        from .motion_mcu_probe import verify as verify_probe
        return verify_probe(spec)
    c=spec['mcu_execution'];expected=build(c['source_spec'],spec['source_checkpoint'],experiment=spec['experiment'],
        spec_path=spec['motion_training']['spec_path'],owner_session=spec['authorization'].get('hapi_session'))
    for k in ('authorization','resources','notifications'):expected[k]=spec[k]
    if spec!=expected:raise ValueError('MCU experiment contains undeclared drift')
    path=Path(spec['motion_training']['spec_path'])
    claim=path.parent.parent/'.motion_mcu_claims'/spec['source_sha256']
    if path.is_file() and claim.is_file() and json.loads(claim.read_text())['spec']!=str(path):
        raise ValueError('MCU preparation superseded; do not submit twice')
    validate_recipe(spec['student_training'],spec['source_sha256'])
    return True


def assess(result,arrays,eid,row):
    import numpy as np
    from .motion_reacquisition import assess as geometry_assess
    geometry_assess(result,arrays,eid)
    mask=arrays['first_episode_valid'][:,eid] & (arrays['controller'][:,eid,0]==3)
    state=arrays['state'][mask,eid][:3000];posture=arrays['posture'][mask,eid][:3000]
    motor=arrays['mcu_execution'][mask,eid][:3000,:6]
    full=len(state)==3000
    metrics={'complete30':full,'window_s':[2,30] if row['command'][:2]==[0.,0.] else [10,18],
        'driver_limit_nm':profile()['driver_limits_nm'],'simulated_initial_state':'hypothesis, not executed LQR'}
    checks=result['checks'];checks['mcu_complete30']=full
    if full:
        lo,hi=(200,3000) if row['command'][:2]==[0.,0.] else (1000,1800)
        h=state[lo:hi];p=posture[lo:hi];m=motor[lo:hi]
        caps=np.array(profile()['driver_limits_nm'])
        metrics.update(pitch_peak_rad=float(abs(p[:,0]).max()),roll_peak_rad=float(abs(p[:,1]).max()),
            height_MAE_m=float(abs(h[:,6]-.34).mean()),driver_saturation_fraction=float((abs(m)>=caps-1e-4).mean()),
            horizontal_speed_mean_mps=float(np.linalg.norm(arrays['root_post'][mask,eid][lo:hi,7:9],axis=-1).mean()),
            vx_abs_mean_mps=float(abs(h[:,7]).mean()))
        checks.update(mcu_pitch=metrics['pitch_peak_rad']<=.1,mcu_roll=metrics['roll_peak_rad']<=.05,
                      mcu_height=metrics['height_MAE_m']<=.02,mcu_driver_saturation=metrics['driver_saturation_fraction']<=.05)
        metrics['legacy_net_joint_reference_saturation_passed']=checks.get('saturation')
        # The generalized motor+spring torque is not motor output authority.
        checks['saturation']=checks['mcu_driver_saturation']
    else:
        checks.update(mcu_pitch=False,mcu_roll=False,mcu_height=False,mcu_driver_saturation=False)
    result.update(version=ASSESSMENT,mcu_execution=metrics,accepted=all(checks.values()),
                  failed_checks=[k for k,v in checks.items() if v is not True])


def value_diagnostics(storage,commands):
    """Stable two-pass float64 centered variances; not a training gate."""
    from native_stand.recovery.controlled_ppo import KEY
    ids=storage.observations['motion_diagnostic_cohort'].reshape(-1).long()
    valid=storage.observations[KEY].reshape(-1).bool()
    ret=storage.returns.reshape(-1).double();val=storage.values.reshape(-1).double()
    err=ret-val;norm=storage.advantages.reshape(-1).double();n=len(commands)
    count=torch.zeros(n,device=ret.device,dtype=torch.float64).scatter_add_(0,ids,valid.double())
    def sums(x):return torch.zeros(n,device=ret.device,dtype=torch.float64).scatter_add_(0,ids,torch.where(valid,x,0.))
    mean=sums(ret)/count.clamp_min(1);em=sums(err)/count.clamp_min(1)
    variance=sums((ret-mean[ids]).square())/count.clamp_min(1)
    error_variance=sums((err-em[ids]).square())/count.clamp_min(1)
    values=torch.stack((count,mean,variance,error_variance,sums(err.square())/count.clamp_min(1),em,
        sums(norm)/count.clamp_min(1),sums(((err>0)!=(norm>0)).double())/count.clamp_min(1)),-1).cpu().tolist()
    return [dict(command=c,samples=int(x[0]),return_mean=x[1],return_variance=x[2],
        explained_variance=1-x[3]/x[2] if x[0]>1 and x[2]>1e-8 else None,
        value_error_MSE=x[4],raw_advantage_mean=x[5],normalized_advantage_mean=x[6],sign_flip_fraction=x[7],
        estimator='centered_two_pass_float64') for c,x in zip(commands,values)]


def execution_reward_snapshot():
    from .motion_antioscillation import huber
    from . import motion_mcu_runtime,motion_mcu_execution
    rows=[]
    for request in [0.,.5,1.,1.95,WHEEL_CAP,3.,5.5,11.,28.]:
        x=torch.tensor(request,dtype=torch.float64,requires_grad=True)
        error=torch.relu(x.abs()/WHEEL_CAP-.8)
        density=-.25*(.25*huber(error/2)+.75*huber(error))
        rows.append(dict(request_Nm=request,density=float(density.detach()),dt_contribution=float(density.detach())*.01,
                         derivative_per_Nm=float(torch.autograd.grad(density,x)[0])))
    return dict(profile=profile(),formula_sha256=sha(motion_mcu_execution.__file__),
        runtime_reward_sha256=sha(motion_mcu_runtime.__file__),curves=rows,
        request_cost='-.25*(.25*Huber(relu(abs(request)/driver_cap-.8)/2)+.75*Huber(relu(abs(request)/driver_cap-.8)));dt once',
        other_effort_scales='reference40/5.5 retained; driver limits distinct',
        caveat='density derivatives do not guarantee PPO gradient quality or convergence')


def save_execution_tail(output,env,update):
    from .artifacts import npz
    t=env.motion_timing;count=min(t.tick,1000)
    ids=torch.arange(t.tick-count,t.tick,device=env.device)%1000
    # One environment, finite1s1kHz window, episode counter disambiguates resets.
    data=t.tail[ids].detach().cpu().numpy()
    folder=Path(output)/'execution_diagnostics';folder.mkdir(exist_ok=True)
    npz(folder/f'update_{update}.npz',{'execution_1kHz_env0':data})
