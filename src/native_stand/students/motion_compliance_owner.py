"""Explicit CPU MuJoCo PPO owner, submitted only through the public supervisor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import select
import signal
import subprocess
import sys
import time
import traceback
import uuid
import numpy as np
import torch
from .motion_compliance_requests import ROOT,write,sha,verify
from .motion_compliance_training import VERSION,CRITIC,assignments

STOP=False


def _stop(*args):
    global STOP
    STOP=True


def assemble(spec,obs):
    from rsl_rl.storage import RolloutStorage
    from native_stand.recovery.controlled_ppo import ControlledPPO
    from .model import Student,load
    actor,_=load(spec['source_checkpoint'],'MotionZeroStudent');actor.log_std.requires_grad_(False)
    critic=Student(False,version='multiskill_mcu_v2',critic_privileged_version=CRITIC)
    storage=RolloutStorage('rl',len(obs),128,obs,[6],device='cpu')
    ppo=ControlledPPO(actor,critic,storage,learning_rate=3e-5,schedule='fixed',
        num_learning_epochs=4,num_mini_batches=4,clip_param=.2,gamma=.99,lam=.95,
        value_loss_coef=.5,entropy_coef=.001,max_grad_norm=1.,use_clipped_value_loss=False,device='cpu')
    ppo.critic_control_only=True;ppo.record_gradient_norms=True
    return ppo


def save(run,spec,ppo,update,course,spec_sha):
    path=run/'formal'/f'model_update_{update}.pt';path.parent.mkdir(exist_ok=True)
    data=dict(interface_version='multiskill_mcu_v2',role='MotionZeroStudent',recipe_version=VERSION,
        training_recipe=dict(version=VERSION,source_actor_sha256=spec['source_sha256']),
        source_sha256=spec['source_sha256'],spec_sha256=spec_sha,
        source_code_sha256=hashlib.sha256(json.dumps(spec['inputs'],sort_keys=True).encode()).hexdigest(),
        critic_privileged_version=CRITIC,actor=ppo.actor.state_dict(),critic=ppo.critic.state_dict(),
        optimizer=ppo.optimizer.state_dict(),fixed_std=True,BC_coefficient=0.,update=update,
        motion_course_state=course,training_rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state()))
    temp=path.with_suffix('.tmp');torch.save(data,temp);temp.replace(path)
    from .model import load
    check,_=load(path,'MotionZeroStudent')
    if any(not torch.equal(v,check.state_dict()[k]) for k,v in ppo.actor.state_dict().items()):raise RuntimeError('compliance checkpoint roundtrip differs')
    return dict(checkpoint=str(path),sha256=sha(path),update=update)


def scalar(value):
    if isinstance(value,dict):return {k:scalar(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [scalar(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def export_actor(run,spec,actor,checkpoint):
    """Actor only; reference vectors and actual new execution parameters."""
    from onnx.reference import ReferenceEvaluator
    class Mean(torch.nn.Module):
        def __init__(self,model):super().__init__();self.model=model
        def forward(self,observation):return self.model(observation)
    out=run/'export';out.mkdir(exist_ok=False)
    path=out/'actor.onnx'
    torch.onnx.export(Mean(actor.eval()),torch.zeros(1,31),str(path),input_names=['observation'],
        output_names=['mean_action'],opset_version=17,dynamo=False)
    with np.load(Path(spec['source_ONNX']).with_name('golden_vectors.npz')) as source:inputs=source['observation']
    with torch.inference_mode():wanted=actor(torch.from_numpy(inputs)).numpy()
    evaluator=ReferenceEvaluator(str(path))
    actual=np.concatenate([evaluator.run(None,{'observation':r[None]})[0] for r in inputs])
    error=float(abs(actual-wanted).max())
    if error>1e-5:raise RuntimeError('new ONNX reference parity failed')
    np.savez_compressed(out/'golden_vectors.npz',observation=inputs,pytorch_mean_action=wanted,onnx_mean_action=actual)
    write(out/'manifest.json',dict(version='motion_compliance_small31_onnx_v1',checkpoint=checkpoint,
        source_sha256=spec['source_sha256'],input=[1,31],output=[1,6],dtype='float32',
        interface_version='multiskill_mcu_v2',deterministic_Tanh_included=True,critic_exported=False,
        PD=spec['PD'],clocks_ms=[1,5,10],leg_scale_rad=1.5,wheel_scale_rad_s=100.,
        motor_limits_Nm=[40,40,2.44694786377709,40,40,2.44694786377709],
        simulation_backend='MuJoCo_CPU',force_route='separated',hardware_verified=False,
        ONNX_sha256=sha(path),golden_sha256=sha(out/'golden_vectors.npz'),golden_count=len(inputs),maximum_normalized_error=error))
    (out/'README.md').write_text('''# Motion31柔顺站立／低速候选\n\n本包参数与身份以manifest.json为准，独立于旧4500包。输入float32 [1,31]，输出float32 [1,6]，确定性mean已含Tanh；部署不再Tanh、不采样、不乘探索std。\n\n策略10ms、PD5ms。腿Kp20/Kd2、轮Kv0.15；腿位置目标=1.5×action rad，轮速度目标=100×action rad/s。电机腿限40Nm、轮限2.44694786Nm；弹簧独立施力，67或200Nm机械数值包络不是电机能力。\n\n通道仍为model_left hip/active/wheel、model_right hip/active/wheel；model_left对应几何右，model_right对应几何左，不能按几何左右重排。几何左右从车尾同向向车头看。Actor gyro/gravity同USD body轴，速度与关节方向沿用原已对齐电控标定。\n\nobs0:3=gyro×.25；3:6=gravity；6:10=[q0,q1,q3,q4]/1.5；10:16=dq六通道×.05；16:20=上次实际采用腿目标[0,1,3,4]/1.5；20:22=实际采用两轮目标[2,5]/100；22:25=[vx/4,yaw/4,height/.5]；25:31=[1,0,0,1,0,0]。固定height=.34，目标高度是命令，不是实测高度。\n\nprevious只来自PD实际采用的物理目标：缓存存rad/rad/s，构造输入时才归一化；不回写shadow预测，不把测量轮速替代previous。纯力矩LQR不能凭空反推该缓存，交接须单独定义并记录；不能把实机姿态清零或假造gravity/速度。编码器零位、减速比、hip wrap与active闭链分支、IMU安装变换、电流换算仍需沿用已核验硬件协议。\n\n逐帧用golden_vectors.npz验证板端31→6推理，归一化最大误差≤1e−5；再验证真实输入、执行目标、PD/补偿与最终命令。学习型模型只在所声明柔顺假设和课程内评估；本包不授实机稳定认证，已学能力见本run评估JSON与completion_report。禁止将Critic64特权字段放进31D Actor输入。\n''')
    return dict(directory=str(out),onnx_sha256=sha(path),maximum_error=error)


def run_owned(path):
    global STOP
    path=Path(path).resolve();spec=verify(path);run=Path(spec['run']);book_path=path.with_name('budget.json')
    if os.environ.get('MOTION_COMPLIANCE_SUBMISSION_SHA')!=sha(path):raise ValueError('public supervisor ownership required')
    from native_stand.command_v6.ledger import Ledger
    from .motion_compliance_env import VectorEnv
    from .motion_compliance_eval import evaluate,cases
    from .model import load
    from .motion_compliance_reward import reward_snapshot
    book=Ledger(book_path);env=eval_env=ppo=writer=None;failure=None;attempt=None;phase=None
    started=time.monotonic();phase_started=started;candidates=[];source_results=[];old_pd_results=[];latest=None;completed=0
    course=dict(stage=0,consecutive_passes=0,last_full_update=0)
    physical_reserved=0;finished_physical=0;actual_samples=0;actor_samples=0
    run.mkdir(parents=True,exist_ok=False);(run/'source_snapshot').mkdir()
    write(run/'source_snapshot/spec.json',spec)
    manifest=dict(spec_path=str(path),spec_sha256=sha(path),pid=os.getpid(),hapi_session=spec['authorization']['hapi_session'],
        backend=spec['backend'],source_sha256=spec['source_sha256'],input_sha256=spec['inputs'])
    write(run/'manifest.json',manifest)
    source=json.loads(Path(spec['parent_spec']).read_text());snapshot=reward_snapshot(source)
    snapshot.update(formula_sha256={k:v for k,v in spec['inputs'].items() if any(n in k for n in ('reward','antioscillation','posture_design','motion_quality'))})
    write(run/'reward_snapshot.json',snapshot)
    def guard(steps):
        nonlocal physical_reserved
        if STOP:raise InterruptedError('user requested compliance stop')
        physical_reserved+=steps
        if physical_reserved>spec['budget']['physical_steps_1ms']:raise TimeoutError('physical step ceiling reached')
    def disk_guard():
        import shutil
        size=sum(f.stat().st_size for f in run.rglob('*') if f.is_file() and not f.is_symlink())
        if size>spec['artifacts']['storage_max_bytes']-spec['artifacts']['terminal_reserve_bytes']:
            raise TimeoutError('compliance storage ceiling reached')
        if shutil.disk_usage(run).free<spec['artifacts']['terminal_reserve_bytes']:raise TimeoutError('compliance disk reserve reached')
    def reserve(newphase):
        nonlocal attempt,phase,phase_started
        phase=newphase;attempt=uuid.uuid4().hex;folder=run/phase;folder.mkdir(exist_ok=True)
        book.reserve(attempt,phase,folder)
        phase_started=time.monotonic()
    def settle(status):
        nonlocal attempt
        if attempt:
            book.finalize(attempt,run/phase,status,exit_code=0 if status in ('COMPLETED','STOPPED_BOUNDED') else 1,wall_s=time.monotonic()-phase_started)
            attempt=None
    def assessment(actor,rows,label):
        nonlocal eval_env
        if eval_env is None:eval_env=VectorEnv(spec,8)
        result=evaluate(eval_env,actor,rows,guard)
        state_hash=hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in actor.state_dict().items())).hexdigest()
        write(run/'evaluations'/f'{label}.json',dict(label=label,records=scalar(result),PPO_updates=0,backend='MuJoCo_CPU',
            actor_state_sha256=state_hash,source_sha256=spec['source_sha256'],PD=spec['PD']))
        return result
    def bootstrap(transitions):
        r=torch.tensor([tr['reward'] for tr in transitions]);done=torch.tensor([tr['terminated'] or tr['timeout'] for tr in transitions])
        for i,tr in enumerate(transitions):
            if tr['timeout']:
                with torch.inference_mode():value=ppo.critic({'student_critic':torch.tensor(tr['terminal_critic'])[None]})[0,0]
                r[i]+=ppo.gamma*value
        return r,done
    try:
        signal.signal(signal.SIGTERM,_stop);signal.signal(signal.SIGINT,_stop)
        torch.set_num_threads(4);torch.manual_seed(spec['seed']);np.random.seed(spec['seed']);random.seed(spec['seed'])
        reserve('source_eval')
        fd=os.environ.get('MOTION_COMPLIANCE_HANDSHAKE_FD')
        if fd:
            os.write(int(fd),(json.dumps(dict(phase='SOURCE_EVAL',pid=os.getpid(),run=str(run),online_updates=0))+'\n').encode());os.close(int(fd))
        actor,_=load(spec['source_checkpoint'],'MotionZeroStudent')
        source_results=assessment(actor,cases(),'old_actor_new_PD')
        # Actual same-plant original-gain comparator, separate isolated worlds.
        oldspec={**spec,'PD':dict(leg_Kp=40.,leg_Kd=2.,wheel_Kv=.2)}
        original=VectorEnv(oldspec,8)
        try:old_pd_results=evaluate(original,actor,cases(0),guard)
        finally:finished_physical+=original.physical_steps;original.close()
        write(run/'evaluations/old_actor_original_PD.json',dict(records=scalar(old_pd_results),scope='12standing conditions, original gains, otherwise same separated mechanics'))
        # Explicit timeout/reset path covers what12x128 steps cannot guarantee.
        eng=VectorEnv(spec,8)
        try:
            assigned=assignments(0,0)
            for r in assigned:r.update(episode_s=.04,training=False)
            obs=eng.reset(assigned);seen=set();timeout_seen=False
            for _ in range(105):
                guard(80)
                with torch.inference_mode():a=actor(obs['student_actor']).numpy()
                obs,tr=eng.step(a)
                for i,t in enumerate(tr):
                    if t['timeout']:seen.add(i);timeout_seen=True
            if not seen or not timeout_seen:raise RuntimeError('explicit short engineering timeout/reset not exercised')
            write(run/'engineering_readback.json',dict(status='PASSED',worker_pids=[p.pid for p in eng.processes],
                independent_processes=len({p.pid for p in eng.processes})==8,actor_shape=list(obs['student_actor'].shape),
                critic_shape=list(obs['student_critic'].shape),explicit_short_timeout_envs=sorted(seen),
                natural30s_timeout_claimed=False,source_ability_prerequisite=False,
                scope='short engineering horizon only; not survival certificate'))
        finally:finished_physical+=eng.physical_steps;eng.close()
        settle('COMPLETED');reserve('smoke')
        env=VectorEnv(spec,8);obs=env.reset([{**r,'training':True,'stage':0,'cycle':0} for r in assignments(0,0)])
        ppo=assemble(spec,obs);initial_actor={k:v.clone() for k,v in ppo.actor.state_dict().items()}
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(run/'formal/tensorboard'),flush_secs=10)
        # Preserve existing6006 log root; only a small symlink, no duplicated events.
        tb=ROOT/'artifacts/standing_survival_v14_20261014/tensorboard'/spec['experiment']
        if tb.parent.is_dir() and not tb.exists():tb.symlink_to(run/'formal/tensorboard',target_is_directory=True)
        metrics_path=run/'metrics.jsonl'
        for update in range(1,3001):
            disk_guard()
            if update==13:
                settle('COMPLETED');reserve('formal')
            tic=time.monotonic();rows=[];reward_sums={};raw_sums={};count=0
            ppo.actor_update_enabled=update>5
            for step in range(128):
                guard(80)
                pre_valid=int(obs['recovery_actor_valid'].sum())
                with torch.inference_mode():actions=ppo.act(obs)
                obs,tr=env.step(actions.numpy());rewards,dones=bootstrap(tr)
                ppo.process_env_step(obs,rewards,dones,{})
                actual_samples+=8;actor_samples+=pre_valid if update>5 else 0
                for t in tr:
                    m=t.get('metrics')
                    if m and m['controlled']:
                        rows.append(m);count+=1
                        for name,den,raw in zip(m['fields'],m['density'],m['raw']):
                            reward_sums[name]=reward_sums.get(name,0.)+float(den);raw_sums[name]=raw_sums.get(name,0.)+float(raw)
            with torch.inference_mode():ppo.compute_returns(obs)
            valid=ppo.storage.observations['recovery_actor_valid'];returns=ppo.storage.returns[valid];values=ppo.storage.values[valid]
            var=float(returns.var(unbiased=False)) if returns.numel() else 0.
            ev=1-float((returns-values).var(unbiased=False))/var if var>1e-12 else None
            cohort=ppo.storage.observations['compliance_cohort'];group_EV=[]
            for key in torch.unique(cohort.reshape(-1,3),dim=0):
                m=(cohort==key).all(-1)&valid.squeeze(-1)
                r=ppo.storage.returns.squeeze(-1)[m];v=ppo.storage.values.squeeze(-1)[m]
                rv=float(r.var(unbiased=False)) if r.numel()>1 else 0.
                group_EV.append(dict(cohort=key.tolist(),valid_samples=int(m.sum()),
                    EV=1-float((r-v).var(unbiased=False))/rv if rv>1e-12 else None))
            values_loss=ppo.update();book.charge(attempt,run/phase);completed=update
            changed=any(not torch.equal(v,ppo.actor.state_dict()[k]) for k,v in initial_actor.items() if k!='log_std')
            metric=dict(update=update,phase=phase,samples=actual_samples,effective_actor_samples=actor_samples,
                update_wall_s=time.monotonic()-tic,stage=course['stage'],loss=values_loss,critic_EV=ev,cohort_EV=group_EV,
                PPO=ppo.mask_summary,actor_changed=changed,
                tracking=dict(vx_abs_mean=float(np.mean([abs(m['velocity'][0]) for m in rows])) if rows else None,
                    yaw_abs_mean=float(np.mean([abs(m['yaw']) for m in rows])) if rows else None,
                    height_MAE=float(np.mean([abs(m['height']-.34) for m in rows])) if rows else None),
                physical_steps_observed=finished_physical+env.physical_steps+(eval_env.physical_steps if eval_env else 0),
                physical_steps_reserved_upper=physical_reserved)
            with metrics_path.open('a') as f:f.write(json.dumps(scalar(metric),allow_nan=False)+'\n')
            for name in ('value','surrogate','entropy'):writer.add_scalar('Loss/'+name,values_loss[name],update)
            for group in group_EV:
                if group['EV'] is not None:writer.add_scalar('Critic/cohort_'+str(group['cohort'])+'/EV',group['EV'],update)
            if ev is not None:writer.add_scalar('Critic/EV',ev,update)
            for name in ('kl_valid_mean','clip_fraction_valid_mean','actor_gradient_norm_mean','critic_gradient_norm_mean'):
                if ppo.mask_summary[name] is not None:writer.add_scalar('PPO/'+name,ppo.mask_summary[name],update)
            for name in reward_sums:
                writer.add_scalar('Reward/raw/'+name,raw_sums[name]/max(count,1),update)
                writer.add_scalar('Reward/density/'+name,reward_sums[name]/max(count,1),update)
                writer.add_scalar('Reward/dt/'+name,reward_sums[name]/max(count,1)*.01,update)
            if update==12:
                if not changed:raise RuntimeError('smoke did not update Actor')
                latest=save(run,spec,ppo,update,course,sha(path))
                write(run/'smoke/wiring_certificate.json',dict(status='PASSED',updates=12,critic_only_first=5,
                    actual_actor_update=changed,actual_critic_update=bool(ppo.optimizer.state),BC=0,
                    checkpoint=latest,engineering_readback='engineering_readback.json',
                    episode_reset_scope='explicit short engineering timeout, not natural30s guarantee'))
            if update%50==0 or update==3000:
                entry={'update':update,'evaluated':False}
                if update%100==0 or update==3000:
                    full=assessment(ppo.actor,cases(course['stage']),f'update_{update}_full')
                    all_pass=all(r['accepted'] for r in full)
                    entry.update(evaluated=True,complete_current_matrix=True,stage=course['stage'],passed=sum(r['accepted'] for r in full),total=len(full),accepted=all_pass)
                    course['last_full_update']=update;course['consecutive_passes']=course['consecutive_passes']+1 if all_pass else 0
                    if course['consecutive_passes']>=2 and course['stage']<3:
                        course['stage']+=1;course['consecutive_passes']=0
                        obs=env.reset([{**r,'training':True,'stage':course['stage'],'cycle':0} for r in assignments(course['stage'],0)])
                else:
                    quick=[dict(profile=p,command=[0.,0.,.34],yaw=0.,prefix_s=0.,training=False) for p in ('R','F20','F40','F120')]
                    magnitude=.25 if course['stage']<3 else .5
                    quick += [dict(profile='R',command=[s*magnitude,0.,.34],yaw=0.,prefix_s=0.,training=False) for s in (1,-1)]
                    q=assessment(ppo.actor,quick,f'update_{update}_quick')
                    entry.update(quick_passed=sum(r['accepted'] for r in q),quick_total=len(q))
                latest=save(run,spec,ppo,update,course,sha(path));entry.update(latest)
                candidates.append(entry);write(run/'candidate_registry.json',candidates)
                write(run/'course_progress.json',course)
            write(run/'progress.json',dict(status='RUNNING',phase=phase,actual_updates=update,
                smoke_updates=min(update,12),formal_updates=max(0,update-12),latest_checkpoint=latest,
                stage=course['stage'],latest_metric=scalar(metric)))
        best=max([c for c in candidates if c.get('evaluated')],key=lambda c:(c.get('accepted',False),c.get('stage',0),c.get('passed',0)/max(c.get('total',1),1)))
        final_actor,_=load(best['checkpoint'],'MotionZeroStudent')
        final_matrix=assessment(final_actor,cases(),'selected_full60')
        handover=assessment(final_actor,cases(0,prefix=1.),'selected_handover12')
        write(run/'candidate_selection.json',dict(selected=best,source_new_PD_passed=sum(r['accepted'] for r in source_results),
            selected_full60_passed=sum(r['accepted'] for r in final_matrix),selected_handover12_passed=sum(r['accepted'] for r in handover)))
        export_actor(run,spec,final_actor,best)
        settle('COMPLETED');status='COMPLETED'
    except BaseException as exc:
        failure=dict(exception=repr(exc),traceback=traceback.format_exc())
        status='STOPPED_BOUNDED' if isinstance(exc,(TimeoutError,InterruptedError)) else 'FAILED'
        settle(status)
    finally:
        if ppo is not None and completed and (latest is None or latest['update']!=completed):
            try:latest=save(run,spec,ppo,completed,course,sha(path))
            except BaseException as exc:failure={**(failure or {}),'checkpoint_error':repr(exc)}
        if writer is not None:writer.close()
        actual_physics=finished_physical
        for world in (env,eval_env):
            if world is not None:actual_physics+=world.physical_steps;world.close()
        report=dict(status=locals().get('status','FAILED'),failure=failure,actual_updates=completed,
            smoke_updates=min(completed,12),formal_updates=max(0,completed-12),samples=actual_samples,
            effective_actor_samples=actor_samples,physical_steps_observed=actual_physics,
            physical_steps_reserved_upper=physical_reserved,latest_checkpoint=latest,course=course,
            wall_s=time.monotonic()-started,hardware_verified=False)
        write(run/'chain_terminal_report.json',report)
        write(run/'completion_report.md','# 柔顺站立／低速训练终态\n\n'+json.dumps(scalar(report),indent=2,ensure_ascii=False))
    return 0 if status in ('COMPLETED','STOPPED_BOUNDED') else 1


def supervisor_entry():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-compliance-spec',type=Path,required=True)
    parser.add_argument('--launch',action='store_true');parser.add_argument('--check-submission',action='store_true')
    args=parser.parse_args();s=verify(args.motion_compliance_spec);run=Path(s['run'])
    from native_stand.command_v6.tools.storage_budget import check
    capacity=check(s,run,'formal')
    plan=dict(status='READY' if capacity['status']=='PASS' and not run.exists() else 'NOT_READY',
        spec=str(args.motion_compliance_spec.resolve()),run=str(run),backend='MuJoCo_CPU',storage=capacity,
        budget=s['budget'],simulation=0,PPO_updates=0)
    if not args.launch:print(json.dumps(plan,indent=2));return 0 if plan['status']=='READY' else 2
    if plan['status']!='READY':raise ValueError(plan)
    claim=args.motion_compliance_spec.with_name('chain_submission.json')
    with claim.open('x') as f:json.dump(dict(spec_sha256=sha(args.motion_compliance_spec),run=str(run),owner=s['authorization']['hapi_session']),f)
    environment={**os.environ,'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT/'scripts'),
        'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'MOTION_COMPLIANCE_SUBMISSION_SHA':sha(args.motion_compliance_spec)}
    r,w=os.pipe();environment['MOTION_COMPLIANCE_HANDSHAKE_FD']=str(w)
    log=run.with_suffix('.owner.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as stream:
        child=subprocess.Popen([sys.executable,'-m','native_stand.students.motion_compliance_owner',
            '--owned-spec',str(args.motion_compliance_spec.resolve())],env=environment,stdout=stream,stderr=subprocess.STDOUT,
            pass_fds=(w,),start_new_session=True)
    os.close(w);ready,_,_=select.select([r],[],[],45.)
    payload=os.read(r,8192).decode() if ready else None;os.close(r)
    receipt=dict(owner_pid=child.pid,handshake=json.loads(payload) if payload else None,log=str(log),first_optimizer_update_verified=False)
    write(args.motion_compliance_spec.with_name('submission_receipt.json'),receipt)
    print(json.dumps(receipt,indent=2));return 0 if receipt['handshake'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--owned-spec',type=Path,required=True)
    raise SystemExit(run_owned(parser.parse_args().owned_spec))
