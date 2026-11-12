"""Finite algorithm owner entered ONLY through command_v6.supervisor."""
import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import math
import numpy as np
import torch
from .experiment import Budget,verify,validate_request
from .real import ROOT,sha,load_plant,description,actors,TEACHER_SHA,SMALL_SHA
from .dataset import Replay,write_json,build_manifest
from .geometry import Kinematics
from .models import Ensemble,Critic
from .reward import MotionReward


def check_provenance(data,spec_path):
    if data.get('provenance',{}).get('spec_sha256')!=sha(spec_path):
        raise ValueError('new algorithm checkpoint spec/plant lineage differs')


def load_world(path,device,spec_path=None):
    d=torch.load(path,map_location=device,weights_only=False)
    if d['version']!='wheel_leg_shared_dynamics_bptt_v1':raise ValueError('World Model algorithm identity differs')
    if spec_path is not None:check_provenance(d,spec_path)
    m=Ensemble(**d['config']).to(device);m.load_state_dict(d['model']);return m


def load_critic(path,device,spec_path=None):
    d=torch.load(path,map_location=device,weights_only=False)
    if d['version']!='full_execution_real_TD_critic_v1':raise ValueError('Critic semantics differ')
    if spec_path is not None:check_provenance(d,spec_path)
    m=Critic(**d['config']).to(device);m.load_state_dict(d['model']);return m


def load_actor(path,teacher,spec_path=None):
    d=torch.load(path,map_location='cpu',weights_only=False)
    if d['version']!='motion31_bptt_actor_v1' or d['interface_version']!='multiskill_mcu_v2' or d['network_architecture']!=teacher.network_architecture:
        raise ValueError('explicit BPTT Actor interface/architecture required')
    if spec_path is not None:check_provenance(d,spec_path)
    actor=copy.deepcopy(teacher);actor.load_state_dict(d['actor']);return actor


def resources(spec,operation):
    if operation not in ('train-world','train-critic','train-actor','predict','gradient'):
        os.environ['CUDA_VISIBLE_DEVICES']='';return 'cpu'
    wanted=spec['resources']
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    rows=[r.split(',') for r in raw.splitlines()]
    selected=[r for r in rows if int(r[0])==wanted['physical_gpu']]
    if len(selected)!=1 or selected[0][1].strip()!=wanted['uuid']:
        raise ValueError('requested shared physical GPU UUID absent; no switch/preemption')
    if float(selected[0][2])<6144:raise RuntimeError('shared GPU requires at least6GiB free at launch')
    os.environ['CUDA_VISIBLE_DEVICES']=str(wanted['physical_gpu'])
    if not torch.cuda.is_available():raise RuntimeError('requested CUDA unavailable')
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.25,0)
    return 'cuda:0'


def identity_report(request):
    return {k:sha(request[k]) for k in ('dataset','world','critic','actor','prediction','gradient','selection') if request.get(k)}


def gradients(request,replay,wm,critic,actor,geometry,reward,budget,model,refs,cfg):
    from .validation import action_gradient,directional_comparison
    from .execution import observation,measurements
    from .real import paired_return
    # Finite difference/autograd in double precision reduces cancellation at small gradients.
    device=next(wm.parameters()).device
    # Model finite differences stay on the authorized GPU. Physics remains the
    # unchanged CPU engine, using separate FP32 copies of Actor/Critic/geometry.
    pa=copy.deepcopy(actor).cpu().float();pc=copy.deepcopy(critic).cpu().float();pg=copy.deepcopy(geometry).cpu().float()
    wm=wm.to(device).double();critic=critic.to(device).double();actor=actor.to(device).double();geometry=geometry.to(device).double()
    rng=np.random.default_rng(103);analytical=[];physics=[];smooth=[];detail=[];horizon_rows=[]
    for sample in range(8):
        budget.check()
        s,_,_,choice=replay.batch(1,rng,horizon=32,device=device)
        s=s.map(lambda x:x.double() if x.is_floating_point() else x)
        ep,i=choice[0]
        if sample==0:
            for h in (8,16,32):
                r,_=action_gradient(s,actor,wm,critic,geometry,reward,horizon=h)
                horizon_rows.append(r)
        report,g=action_gradient(s,actor,wm,critic,geometry,reward,horizon=8)
        with torch.no_grad():action=actor.mean(observation(s,measurements(s,geometry)))
        row=[]
        # The authoritative MCU path is FP32. Physical branch uses FP32 Actor and
        # exact stored integration blob, while analytic checks above use FP64.
        state=s.to('cpu').map(lambda x:x.float() if x.is_floating_point() else x)
        for ch in range(6):
            delta=torch.zeros_like(action);delta[0,ch]=.0005
            plus=paired_return(model,refs,cfg,state,replay.data[ep]['integration_state'][i],
                (action+delta).detach().cpu().float(),pa,pc,pg,reward,budget,8)
            minus=paired_return(model,refs,cfg,state,replay.data[ep]['integration_state'][i],
                (action-delta).detach().cpu().float(),pa,pc,pg,reward,budget,8)
            switched=plus['contact_events'].shape!=minus['contact_events'].shape or not np.array_equal(plus['contact_events'],minus['contact_events'])
            fd=(plus['return_with_value']-minus['return_with_value'])/.001
            analytical.append(float(g[0,ch]));physics.append(fd)
            smooth.append(not switched and report['smooth_mask'][0][ch])
            row.append(dict(channel=ch,model_derivative=float(g[0,ch]),physical_derivative=fd,
                physical_reward_only_derivative=(plus['reward_only']-minus['reward_only'])/.001,
                physical_contact_switch=switched))
        detail.append(dict(episode=replay.rows[ep]['episode_id'],start=i,model=report,physical=row))
    comparison=directional_comparison(analytical,physics,smooth)
    finite_ok=all(r['finite'] and r['norm']>1e-9 and r['norm']<1e6 for r in horizon_rows)
    fd_ok=all(r['smooth_relative_error_mean'] is not None and r['smooth_relative_error_mean']<=.05 for r in horizon_rows)
    return dict(passed=comparison['passed'] and finite_ok and fd_ok,physical_direction=comparison,
        multi_horizon=horizon_rows,details=detail,checks=dict(finite_nonzero_bounded=finite_ok,finite_difference=fd_ok),
        context_switch_scope='separately recorded, not required to have a unique contact derivative')


def evaluated_summary(arrays,report,case):
    # Shared0..0.5/0.5..2/2..30 metrics plus all-window diagnostics for selection.
    origin=case.get('warmup',0);mask=(arrays['time']>origin+2)&(arrays['time']<=origin+30+1e-8)
    x=arrays['state'][mask];cmd=np.array(case['command']);axis=2 if cmd[1] else 0
    hold=(arrays['time']>origin+10)&(arrays['time']<=origin+18+1e-8)
    h=arrays['state'][hold]
    expected=arrays['effective_command'][hold,1 if cmd[1] else 0]
    def longest(lost):
        peak=run=0
        for value in lost:
            run=run+1 if value else 0;peak=max(peak,run)
        return peak*.01
    summary=dict(physical_seconds=min(30.,max(0.,report['result']['physical_seconds']-origin)),
        tracking_MAE=float(np.abs(h[:,axis]-expected).mean()) if len(h) else 1e9,
        vx_abs_mean=float(np.abs(x[:,0]).mean()) if len(x) else 1e9,
        contact_fraction=float((arrays['contact'][mask]>2).all(-1).mean()) if len(x) else 0.,
        roll_peak=float(np.abs(arrays['state'][:,5]).max()),pitch_peak=float(np.abs(arrays['state'][:,4]).max()),
        height_MAE=float(np.abs(x[:,3]-.34).mean()) if len(x) else 1e9,
        leg_difference_MAE=float(np.abs(arrays['leg_difference'][mask]).mean()) if len(x) else 1e9,
        leg_dq_p95=float(np.quantile(np.abs(arrays['dq'][mask][:,[0,1,3,4]]),.95)) if len(x) else 1e9,
        torque_limit_fraction=float((np.abs(arrays['driver'][mask])>=arrays['driver_cap'][mask]-1e-4).mean()) if len(x) else 1.,
        closure_peak=float(arrays['state'][:,6].max()),full_window=bool(len(x)==2800),
        swing_peak=float(np.abs(arrays['leg_swing_difference_rad']).max()),
        slip_speed_mean=float(np.linalg.norm(arrays['support_point_velocity_xy_mps'][mask],axis=-1).mean()) if len(x) else 1e9)
    extra={}
    for lo,hi in ((0.,.5),(.5,2.),(2.,30.)):
        valid=(arrays['time']>origin+lo+1e-8)&(arrays['time']<=origin+hi+1e-8)
        if not valid.any():continue
        state=arrays['state'][valid];q=arrays['q'][valid];command=arrays['effective_command'][valid]
        lost=arrays['contact'][valid]<=2
        extra[f'{lo:g}_{hi:g}']=dict(vx_tracking_MAE=float(np.abs(state[:,0]-command[:,0]).mean()),
            yaw_tracking_MAE=float(np.abs(state[:,2]-command[:,1]).mean()),
            height_MAE=float(np.abs(state[:,3]-.34).mean()),root_xy_displacement_m=(arrays['root_xy'][valid][-1]-arrays['root_xy'][valid][0]).tolist(),
            joint_mirror_MAE_rad=np.abs(np.stack((q[:,0]+q[:,3],q[:,1]-q[:,4]),-1)).mean(0).tolist(),
            leg_difference_MAE_m=float(np.abs(arrays['leg_difference'][valid]).mean()),
            swing_peak_rad=float(np.abs(arrays['leg_swing_difference_rad'][valid]).max()),
            geometric_support_point_speed_mean_mps=np.linalg.norm(arrays['support_point_velocity_xy_mps'][valid],axis=-1).mean(0).tolist(),
            single_or_both_lost_fraction=lost.mean(0).tolist(),longest_lost_s=[longest(lost[:,i]) for i in range(2)],
            torque_abs_peak_Nm={k:np.abs(arrays[k][valid]).max(0).tolist() for k in ('pd_request','driver','motor','spring','generalized_actual')},
            adopted_step_peak=np.abs(np.diff(arrays['adopted_target'][valid],axis=0)).max(0).tolist() if valid.sum()>1 else None,
            motor_step_peak_Nm=np.abs(np.diff(arrays['motor'][valid],axis=0)).max(0).tolist() if valid.sum()>1 else None)
    identity={k:v for k,v in case.items() if k!='controller'}
    identity=hashlib_json(identity)
    report.update(condition_identity=identity,command=case['command'],summary=summary,physical_diagnostics=extra,
        absolute_stationary=bool(case['command']==[0.,0.] and summary['physical_seconds']>=30 and summary['vx_abs_mean']<=.05
            and summary['roll_peak']<=.05 and summary['pitch_peak']<=.1 and summary['height_MAE']<=.01
            and summary['leg_difference_MAE']<=.01 and summary['swing_peak']<.05 and summary['contact_fraction']>=.99),
        static_threshold_scope='new explicit quality diagnostic; historical acceptance unchanged',
        command_clock='existing run_case absolute start including teacher prefix; tracking uses actual logged effective command',
        contact_scope='geometric support velocity is reported separately from contact; not certified solver contact-patch slip; solver measurement precedes integrated qpos by1ms')
    return report


def hashlib_json(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def world_fit_arguments(spec):
    """Single read-back point from the frozen declaration into fit_world.

    Kept as a named function so the declaration->learner handoff can be tested
    without launching anything; no dimension is ever inferred from tensors.
    """
    declared = spec['world_model']
    arguments = {k: declared[k] for k in ('members', 'width', 'blocks', 'batch_size', 'seed',
                  'fit_window_substeps', 'activation_checkpoint', 'training_revision', 'sampling',
                  'supervision_geometry') if k in declared}
    from .models import contact_force_config
    arguments['contact_force'] = contact_force_config(declared.get('contact_force_supervision'))
    return arguments


def dispatch(request,budget):
    spec=budget.spec;operation=request['operation'];out=Path(request['output'])
    out.mkdir(parents=True,exist_ok=False)
    device=resources(spec,operation);torch.set_num_threads(spec['resources']['max_torch_threads'])
    # A World-Model-only block declares zero physics and never reads the plant, so it must
    # not fail on historical plant files an external prune removed. Every physical or Actor
    # operation still loads and re-verifies the plant before doing anything.
    wm_only=operation in ('train-world','predict')
    if wm_only:
        model=refs=cfg=None
        readback=dict(not_loaded=True,reason='operation declares zero physics steps and reads no plant',
                      declared_plant_sha256=hashlib_json(spec['plant_readback']))
    else:
        model,refs,cfg,readback=load_plant()
        if readback!=spec['plant_readback'] or description(model,refs)!=spec['geometry']:
            raise ValueError('effective shared public plant/solver readback differs')
    geometry=Kinematics(spec['geometry']);actor_map,source=actors()
    parent=json.loads((ROOT/'configs/experiments/2026_10_25-23_52__motion1500_posture_retention312_10000/spec.json').read_text())
    reward=MotionReward(parent,source)
    write_json(out/'input_identity.json',dict(inputs=identity_report(request),spec_sha256=sha(budget.path),
        code_inputs=spec['code_inputs'],plant=readback,device=device))
    provenance=dict(spec_sha256=sha(budget.path),inputs=identity_report(request),state_version=spec['state_version'],
        geometry_sha256=hashlib_json(spec['geometry']),
        plant_sha256=hashlib_json(spec['plant_readback']) if wm_only else hashlib_json(readback),
        plant_loaded=(None if wm_only else readback),
        plant_verification=('declaration only: this block declares zero physics steps' if wm_only
                            else 're-verified against the shared public plant at dispatch'))
    if request.get('actor'):actor_map['candidate']=load_actor(request['actor'],actor_map['teacher4500'],budget.path)
    if operation=='engineering':
        from .engineering import run
        return run(model,refs,cfg,actor_map,geometry,reward,budget,out)
    if operation=='reuse-data':
        declaration=spec['prediction_continuation']
        evidence=json.loads(Path(declaration['engineering']).read_text())
        if evidence['passed'] is not True or evidence['spec_sha256']!=declaration.get('engineering_spec_sha256',declaration['parent_spec_sha256']):
            raise ValueError('parent engineering evidence identity differs')
        # verify() bound every complete episode manifest and trace SHA. This stage
        # neither copies the source/data tree nor replays physics or a source Actor.
        return dict(passed=True,reused=True,physics_steps=0,old_physics_steps=evidence['physics_steps'],
                    dataset=declaration['dataset'],world=declaration['world'],declaration=declaration)
    if operation=='collect':
        from .real import collect_episode
        if request.get('collection','initial')=='initial':chosen=spec['collection']
        else:
            selection=json.loads(Path(request['selection']).read_text()) if request.get('selection') else {}
            if selection.get('selected') is not True or selection.get('candidate_sha256')!=sha(request['actor']):
                raise ValueError('new current-Actor distribution requires real complete-matrix improvement evidence')
            if 'candidate' not in actor_map:raise ValueError('refresh must use physically assessed current Actor')
            old=[c for c in spec['collection'] if c['split']=='train' and c['controller']=='teacher4500']
            old=old+[dict(old[0],predecessor='teacher4500',handover_ms=1000),
                     dict(old[3],predecessor='teacher4500',handover_ms=1000)]
            chosen=[]
            for i,row in enumerate(old):
                chosen.append(dict(row,id=f'{out.parent.name}_refresh_{i}',controller='candidate'))
        results=[]
        for case in chosen:
            known={'teacher4500':TEACHER_SHA,'small4500':SMALL_SHA,
                   'old500':'5b67781bdaa86b666683e315f685a0f19c45fee918da70439ea3bb43d0fbb03a'}
            digest=sha(request['actor']) if case['controller']=='candidate' else known[case['controller']]
            case=dict(case,controller_sha256=digest,spec_sha256=sha(budget.path),plant_identity=hashlib_json(readback))
            budget.check();results.append(collect_episode(model,refs,cfg,actor_map,case,geometry,reward,budget,output=out/case['id']))
        return dict(episodes=results,physics_steps=budget.local['physics_steps'],source='actual MuJoCo1ms')
    if operation=='build-data':
        paths=[p.parent for folder in request['episodes'] for p in Path(folder).glob('*/episode.json')]
        return build_manifest(paths,out/'dataset.json',spec['source_inputs'])
    replay=Replay(request['dataset'],request.get('split','train'),permit_test=request.get('final_test',False)) if request.get('dataset') else None
    # The dataset must still be the one collected under the DECLARED plant; a WM-only block
    # skips loading the plant but does not relax the dataset-to-declaration binding.
    plant_digest=hashlib_json(spec['plant_readback'] if wm_only else readback)
    if replay is not None and (replay.manifest['identities']!=spec['source_inputs'] or
        replay.manifest.get('plant_identities')!=[plant_digest]):
        raise ValueError('dataset plant/asset/execution lineage differs')
    if operation=='train-world':
        from .training import fit_world
        val=Replay(request['dataset'],'validation')
        if request.get('world'):
            previous=torch.load(request['world'],map_location='cpu',weights_only=False)
            continuation=spec.get('prediction_continuation',{})
            if request['world']==continuation.get('world'):
                expected=continuation.get('world_provenance_spec_sha256',continuation['parent_spec_sha256'])
                if sha(request['world'])!=continuation['world_sha256'] or previous.get('provenance',{}).get('spec_sha256')!=expected:
                    raise ValueError('declared prior World Model lineage differs')
            else:check_provenance(previous,budget.path)
        return fit_world(replay,val,geometry,budget,out/'fit',steps=request['steps'],
            previous=request.get('world'),device=device,provenance=provenance,
            **world_fit_arguments(spec))
    if operation=='train-critic':
        from .training import fit_critic
        if request.get('critic'):check_provenance(torch.load(request['critic'],map_location='cpu',weights_only=False),budget.path)
        return fit_critic(replay,budget,out/'fit',steps=request['steps'],device=device,previous=request.get('critic'),provenance=provenance,
            controller_sha=sha(request['actor']) if request.get('actor') else TEACHER_SHA)
    if operation in ('predict','gradient','train-actor'):
        wm=load_world(request['world'],device,budget.path);geometry=geometry.to(device)
        if operation=='predict':
            from .validation import predict_report
            return predict_report(replay,wm,geometry,reward,budget)
        critic=load_critic(request['critic'],device,budget.path)
        if operation=='gradient':
            result=gradients(request,replay,wm,critic,actor_map.get('candidate',actor_map['teacher4500']),geometry,reward,budget,model,refs,cfg)
            result['actor_sha256']=sha(request['actor']) if request.get('actor') else TEACHER_SHA
            return result
        for key in ('prediction','gradient'):
            report=json.loads(Path(request[key]).read_text())
            if report.get('passed') is not True or report.get('inputs',{}).get('world')!=sha(request['world']):
                raise ValueError('exact-World-Model held-out prediction and physical gradient evidence required')
            if report.get('spec_sha256')!=sha(budget.path) or report.get('split','validation')!='validation':
                raise ValueError('validation gate cannot consume foreign or final-test results')
            if report.get('inputs',{}).get('dataset')!=sha(request['dataset']):raise ValueError('gradient/prediction dataset identity differs')
            if key=='gradient' and (report.get('inputs',{}).get('critic')!=sha(request['critic']) or
                report.get('actor_sha256')!=(sha(request['actor']) if request.get('actor') else TEACHER_SHA)):
                raise ValueError('physical gradient Actor/Critic identity differs')
        from .training import fit_actor
        actor=actor_map.get('candidate',copy.deepcopy(actor_map['teacher4500']))
        critic_metadata=torch.load(request['critic'],map_location='cpu',weights_only=False)
        expected_actor=sha(request['actor']) if request.get('actor') else TEACHER_SHA
        if critic_metadata.get('controller_sha256')!=[expected_actor]:raise ValueError('Critic must use actual current-policy real transitions')
        retain=retention_inputs()
        return fit_actor(replay,actor,actor_map['teacher4500'],wm,critic,geometry,reward,budget,out/'fit',
            steps=request['steps'],device=device,horizon=spec['bptt']['horizon'],retention_observations=retain,
            provenance=provenance,previous=request.get('actor'))
    if operation=='evaluate':
        from .real import evaluate_case
        if request.get('final_test'):
            selected=json.loads(Path(request['selection']).read_text())
            if selected.get('selected') is not True or selected.get('candidate_sha256')!=sha(request['actor']):
                raise ValueError('final test candidate differs from frozen physical selection')
        chosen=spec['final_test_conditions'] if request.get('final_test') else spec['validation_conditions']
        names=request.get('controllers',['teacher4500','small4500','candidate'])
        results={}
        for name in names:
            results[name]={}
            for case in chosen:
                budget.check();actual=dict(case,controller=name)
                arrays,report=evaluate_case(model,refs,cfg,actor_map,actual,budget)
                budget.storage_guard(sum(x.nbytes for x in arrays.values())+1024**2)
                np.savez_compressed(out/f'{name}__{case["id"]}.npz',**arrays)
                results[name][case['id']]=evaluated_summary(arrays,report,actual)
        return dict(models=results,split='test' if request.get('final_test') else 'validation',
            selected=False if request.get('final_test') else None,
            source='actual MuJoCo physical closed loop, existing public run_case/metrics',Native='NOT_RUN',hardware='NOT_RUN')
    if operation=='export':
        return export_actor(request,actor_map['teacher4500'],out)
    raise ValueError('unknown independent algorithm operation')


def retention_inputs():
    """Read-only original nominal Teacher trajectories, all22 old commands."""
    root=ROOT/'artifacts/2026_10_26-20_29__motion4500_precise_fidelity_dagger/teacher'
    from native_stand.students.motion_posture_design import commands
    from native_stand.students.motion_execution_uncertainty import nominal
    values=[];seen=set()
    for p in sorted(root.rglob('*.npz')):
        meta=json.loads(p.with_suffix('.json').read_text())
        if meta.get('variant')!=0 or meta.get('initial_yaw_rad')!=0.:continue
        if meta.get('uncertainty')!=nominal():raise ValueError('nominal retention execution metadata differs')
        if meta.get('solver_profile')!='closure_contact_tight_v1' or meta.get('plant_profile')!='motion_mcu_currentplant_v1':
            raise ValueError('retention solver/plant identity differs')
        if sha(p)!=meta['trace_sha256']:raise ValueError('retention trace SHA differs')
        target=meta['target'][:2]
        if target not in commands():raise ValueError('retention command outside certified source set')
        seen.add(tuple(target))
        with np.load(p,allow_pickle=False) as z:
            x=z['actor_input']
            if x.ndim!=2 or x.shape[1]!=31 or not np.isfinite(x).all():raise ValueError('retention31D field differs')
            values.append(x[::max(1,len(x)//64)])
    if seen!=set(map(tuple,commands())):raise ValueError('all22 nominal source commands required for retention')
    return torch.from_numpy(np.concatenate(values).astype(np.float32))


def export_actor(request,teacher,out):
    # No small-model distillation and no fictional PPO checkpoint metadata.
    actor=load_actor(request['actor'],teacher).eval()
    selected=json.loads(Path(request['selection']).read_text())
    if selected.get('selected') is not True or selected.get('candidate_sha256')!=sha(request['actor']):
        raise ValueError('only the physically selected candidate can be exported')
    from native_stand.students.export import DeterministicActor
    wrapper=DeterministicActor(actor)
    torch.onnx.export(wrapper,torch.zeros(1,31),out/'actor.onnx',input_names=['observation'],
        output_names=['mean'],opset_version=17,dynamo=False)
    import onnx
    from onnx.reference import ReferenceEvaluator
    session=ReferenceEvaluator(onnx.load(str(out/'actor.onnx')))
    x=retention_inputs()[:256]
    expected=actor.mean(x).detach().numpy();actual=session.run(None,{'observation':x.numpy()})[0]
    maximum=float(np.max(np.abs(expected-actual)))
    if maximum>1e-5:raise ValueError('BPTT ONNX golden error exceeds tolerance')
    return dict(onnx_sha256=sha(out/'actor.onnx'),golden_max_error=maximum,input_version='multiskill_mcu_v2',
                width=[31,512,256,128,6],hardware_validation='NOT_RUN')


def run_owned(request_path):
    request=json.loads(Path(request_path).read_text())
    if os.environ.get('WHEEL_WORLD_SUPERVISOR_REQUEST')!=sha(request_path):
        raise PermissionError('World Model runs must be owned by public command_v6.supervisor')
    spec,book=verify(request['spec'],authorization=True)
    validate_request(request,spec,book)
    if request['operation']=='cycle':
        from .cycle import run
        return run(request)
    budget=Budget(request['spec'],request['operation'],request['output'])
    def stop(sig,frame):raise InterruptedError('owned World Model experiment received signal '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop)
    if math.isfinite(budget.deadline):signal.alarm(max(1,int(budget.deadline-time.monotonic())))
    status='FAILED';error=None
    try:
        fd=os.environ.get('WHEEL_WORLD_HANDSHAKE_FD')
        if fd:
            os.write(int(fd),(json.dumps(dict(attempt_id=budget.id,status='RESERVED',pid=os.getpid()))+'\n').encode());os.close(int(fd))
        result=dispatch(request,budget)
        result.update(inputs=identity_report(request),spec_sha256=sha(request['spec']),actual_counts=budget.local)
        write_json(Path(request['output'])/'result.json',result)
        status='COMPLETED';return 0
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        out=Path(request['output']);out.mkdir(parents=True,exist_ok=True)
        write_json(out/'failure.json',dict(error=error,actual_counts=budget.local,attempt_id=budget.id))
        raise
    finally:
        signal.alarm(0);budget.finalize(status,error)


def supervisor_entry():
    parser=argparse.ArgumentParser(description='Public supervisor: independent finite World Model operation')
    parser.add_argument('--world-model-request',type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();request=json.loads(args.world_model_request.read_text())
    spec,book=verify(request['spec'],authorization=True)
    validate_request(request,spec,book)
    if request['operation']=='cycle':
        from .cycle import claim
        claim(request,args.world_model_request)
    out=Path(request['output']);out.parent.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'WHEEL_WORLD_SUPERVISOR_REQUEST':sha(args.world_model_request),
        'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT/'scripts')}
    r,w=os.pipe();env['WHEEL_WORLD_HANDSHAKE_FD']=str(w)
    log=out.with_suffix('.owner.log')
    with log.open('x') as stream:
        child=subprocess.Popen([sys.executable,'-m','native_stand.world_model.owner','--owned-request',
            str(args.world_model_request.resolve())],env=env,stdout=stream,stderr=subprocess.STDOUT,
            pass_fds=(w,),start_new_session=True)
    os.close(w)
    import select
    ready,_,_=select.select([r],[],[],45. if request['operation']=='cycle' else 10.)  # one bounded handshake
    receipt=os.read(r,4096).decode() if ready else None;os.close(r)
    report=dict(pid=child.pid,handshake=json.loads(receipt) if receipt else None,first_optimizer_update_verified=False,log=str(log))
    write_json(out.with_suffix('.handshake.json'),report)
    print(json.dumps(report))
    return child.wait() if args.wait else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--owned-request',type=Path,required=True)
    raise SystemExit(run_owned(parser.parse_args().owned_request))
