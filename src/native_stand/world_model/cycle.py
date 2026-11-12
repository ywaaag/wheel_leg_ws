"""Serial learning owner, not a monitor: every child is a directly owned stage.

The parent blocks on its child's exit, then consumes the completed report. No
PID/GPU/log polling, watcher, speculative retry or reused historical allowance.
"""
from pathlib import Path
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from .dataset import write_json
from .experiment import verify,validate_request
from .real import ROOT,sha


def claim(request,request_path):
    spec,book=verify(request['spec'],authorization=True)
    if not spec['workflow'].get('automatic_cycle'):raise ValueError('automatic cycle not declared')
    p=Path(request['spec']).with_name('chain_submission.json')
    with p.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if p.exists() or book['attempts']:raise ValueError('fresh independent cycle already claimed or used')
        continuation=spec.get('prediction_continuation')
        if continuation:
            claims=ROOT/'configs/experiments/.world_model_continuation_claims';claims.mkdir(exist_ok=True)
            with (claims/(continuation['parent_spec_sha256']+'.json')).open('x') as stream:
                json.dump(dict(parent_spec=continuation['parent_spec'],successor_spec=request['spec'],
                    parent_event=continuation['event_id'],time=time.time()),stream,indent=2)
        root=Path(spec['run']);root.mkdir(parents=True,exist_ok=True)
        session=spec['authorization']['owner_session'];attempt=uuid.uuid4().hex
        write_json(p,dict(chain=str(root),hapi_session=session,attempt_id=attempt,
            request_sha256=sha(request_path),submitted_unix_s=time.time()))
        write_json(root/'manifest.json',dict(kind='world_model_bptt_cycle',spec_path=request['spec'],
            spec_sha256=sha(request['spec']),hapi_session=session,attempt_id=attempt,
            source_code_copied_bytes=0,notification_route=dict(owner_session=session,target_session=session)))
        snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=False)
        write_json(snapshot/'spec.json',spec)  # one immutable declaration, never a source-tree copy


def stage(request):
    """No polling: wait for one owned operation and read its terminal file once."""
    spec,book=verify(request['spec'],authorization=True)
    validate_request(request,spec,book)
    out=Path(request['output']);out.parent.mkdir(parents=True,exist_ok=True)
    path=out.with_suffix('.request.json');write_json(path,request)
    env={**os.environ,'WHEEL_WORLD_SUPERVISOR_REQUEST':sha(path)}
    env.pop('WHEEL_WORLD_HANDSHAKE_FD',None)
    with out.with_suffix('.owner.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-m','native_stand.world_model.owner','--owned-request',str(path)],
            env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait()
        except BaseException:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # No force kill/refund: unsettled ledger prevents a replacement.
                pass
            raise
    if code:
        failure=out/'failure.json'
        raise RuntimeError(f"{request['operation']} failed ({code}): "+
            (failure.read_text() if failure.is_file() else str(out.with_suffix('.owner.log'))))
    return json.loads((out/'result.json').read_text())


def execute(request,*,execute_stage=stage,engineering_ready=None):
    """Algorithm state machine. Selection/gates use only validation, never test."""
    spec,_=verify(request['spec'],authorization=True)
    root=Path(spec['run']);progress=Path(request['output']);progress.mkdir(parents=True,exist_ok=False)
    records=[];collections=[];world=critic=actor=selection=None;baseline=None;accepted=None
    def call(operation,folder,**kwargs):
        row=dict(operation=operation,spec=request['spec'],output=str(folder),**kwargs)
        result=execute_stage(row)
        records.append(dict(operation=operation,output=str(folder),result_sha256=sha(folder/'result.json')))
        write_json(progress/f'stage_{len(records):03}.json',records[-1])
        return result
    continuation=spec.get('prediction_continuation')
    engineering=call('reuse-data' if continuation else 'engineering',root/'evidence_reuse' if continuation else root/'engineering')
    if engineering.get('passed') is not True:raise ValueError('physical engineering parity failed')
    if engineering_ready is not None:engineering_ready(engineering)
    if continuation:
        world=continuation['world'];collections=[continuation['collection_root']]
    count=0;reason=None
    while spec['workflow']['rounds_max'] is None or count<spec['workflow']['rounds_max']:
        folder=root/f'round_{count:03}';phase='initial' if count==0 else 'refresh';limits=spec['workflow'][phase]
        args={} if count==0 else dict(actor=actor,selection=selection)
        if count==0 and continuation:dataset=continuation['dataset']
        else:
            call('collect',folder/'collect',collection=phase,**args)
            collections.append(str(folder/'collect'))
            call('build-data',folder/'data',episodes=collections)
            dataset=str(folder/'data/dataset.json')
        fit=call('train-world',folder/'world',dataset=dataset,steps=limits['wm_optimizer_steps'],**({'world':world} if world else {}))
        world=fit['checkpoint']
        pred=call('predict',folder/'prediction',dataset=dataset,world=world,split='validation')
        if pred.get('passed') is not True:
            reason='PREDICTION_GATE';break
        if spec['workflow'].get('prediction_only'):
            # This declaration cannot proceed to policy training simply because
            # a prediction threshold passes. Independent value/gradient gates remain.
            break
        fit=call('train-critic',folder/'critic',dataset=dataset,steps=limits['critic_optimizer_steps'],
                 **({'critic':critic} if critic else {}),**({'actor':actor} if actor else {}))
        critic=fit['checkpoint']
        grad=call('gradient',folder/'gradient',dataset=dataset,world=world,critic=critic,split='validation',
                  **({'actor':actor} if actor else {}))
        if grad.get('passed') is not True:
            reason='PHYSICAL_GRADIENT_GATE';break
        fit=call('train-actor',folder/'actor',dataset=dataset,world=world,critic=critic,
            prediction=str(folder/'prediction/result.json'),gradient=str(folder/'gradient/result.json'),
            steps=limits['actor_optimizer_steps'],**({'actor':actor} if actor else {}))
        candidate=fit['checkpoint']
        evaluation=call('evaluate',folder/'evaluation',actor=candidate,
                        controllers=['teacher4500','small4500','candidate'] if baseline is None else ['candidate'])
        if baseline is None:baseline=evaluation
        from .validation import choose_physical
        chosen=choose_physical(baseline['models']['teacher4500'],evaluation['models']['candidate'])
        if accepted is not None:
            current=choose_physical(accepted['models']['candidate'],evaluation['models']['candidate'])
            chosen.update(previous_accepted=current,selected=chosen['selected'] and current['selected'])
        chosen.update(candidate_sha256=sha(candidate),evaluation_sha256=sha(folder/'evaluation/result.json'))
        candidate_selection=folder/'selection.json';write_json(candidate_selection,chosen)
        if not chosen['selected']:
            reason='NO_PHYSICAL_IMPROVEMENT';break
        actor=candidate;selection=str(candidate_selection);accepted=evaluation
        count+=1
    # Freeze the selected candidate before one independent final test. Test results
    # cannot trigger another training iteration or change the selected checkpoint.
    final_test=None;export=None
    if actor is not None:
        final_test=call('evaluate',root/'final_test',actor=actor,selection=selection,final_test=True,
                        controllers=['teacher4500','small4500','candidate'])
        export=call('export',root/'export',actor=actor,selection=selection)
    return dict(status='STOPPED_GATE' if reason else 'COMPLETED_BLOCKS',reason=reason,accepted_rounds=count,
        world=world,critic=critic,actor=actor,selection=selection,records=records,final_test=final_test is not None,
        export=export,prediction_only=spec['workflow'].get('prediction_only',False),
        Native='NOT_RUN',hardware='NOT_RUN',Recovery='NOT_COVERED',Jump='NOT_COVERED')


def terminal(request,result,error,start):
    from .experiment import COUNTERS
    from native_stand.command_v6.completion import notify
    root=Path(request['output']).parent
    # Output is constrained to run/automatic_chain by the caller contract.
    spec=json.loads(Path(request['spec']).read_text());root=Path(spec['run'])
    book=json.loads(Path(request['spec']).with_name('budget.json').read_text())
    actual={key:book['used'][key] for key in COUNTERS}
    unfinished={key:sum(a.get('actual',{}).get(key,0) for a in book['attempts'].values() if a['status']=='RUNNING') for key in COUNTERS}
    result=dict(result or {},status='FAILED' if error else result['status'],error=error,settled_actual=actual,
                unsettled_actual=unfinished,chain_wall_s=time.monotonic()-start)
    result['historical_actual']=book.get('historical_used',{key:0 for key in COUNTERS})
    result['lineage_actual']={key:actual[key]+result['historical_actual'][key] for key in COUNTERS}
    paths={}
    for name in ('world','critic','actor'):
        p=result.get(name)
        if p and Path(p).is_file():paths[name]=dict(checkpoint=p,sha256=sha(p))
    result['checkpoints']=paths
    write_json(root/'analysis.json',result)
    selected=paths.get('actor')
    write_json(root/'candidate_selection.json',dict(best_observed_candidate=selected,
        accepted_candidate=selected,scope='MuJoCo validation only; final test separately reported; hardware not tested'))
    lines=['# World Model / BPTT 实验结算',f"终态：{result['status']}；原因：{result.get('reason') or error}",
        '当前会话累计采样、optimizer与墙钟无限授权；本声明的有限执行块见spec/ledger。整条路线3GiB产物与1.5GiB可用空间保护保持。',
        '来源 Teacher4500；Actor31D、1/5/10ms MCU执行链；原PPO链未接管。',
        '实际消耗：\n```json\n'+json.dumps(actual,indent=2)+'\n```',
        '本路线历史＋本声明累计：\n```json\n'+json.dumps(result['lineage_actual'],indent=2)+'\n```',
        '未结算消耗（不退款、不自动重启）：\n```json\n'+json.dumps(unfinished,indent=2)+'\n```',
        'Checkpoint：\n```json\n'+json.dumps(paths,indent=2)+'\n```',
        '完整结果：'+str(root/'analysis.json'),
        '分阶段 result.json / failure.json、实际训练曲线、逐项预测误差和物理闭环轨迹保存在本run。',
        '预测通过不等于梯度可用；MuJoCo通过不等于Native/硬件通过；Recovery、Jump尚未覆盖。']
    if spec.get('workflow',{}).get('prediction_only'):
        lines.insert(2,'本块仅World Model训练与预测验收；即使预测门通过也不启动Actor/Critic。动作响应幅值与独立Critic验收待补齐。')
    report=root/'completion_report.md';report.write_text('\n\n'.join(lines)+'\n')
    write_json(root/'report_ready.json',dict(event_id='world_model_'+uuid.uuid4().hex,status='report_ready',
                                          report_sha256=sha(report),time=time.time()))
    if spec['notifications']['enabled']:notify(root)
    return result


def run(request):
    start=time.monotonic();result=None;error=None
    spec,_=verify(request['spec'],authorization=True)
    receipt=json.loads(Path(request['spec']).with_name('chain_submission.json').read_text())
    os.environ['WHEEL_WORLD_CYCLE_ID']=receipt['attempt_id']
    def stop(sig,frame):raise InterruptedError(f'World Model cycle signal {sig}')
    signal.signal(signal.SIGTERM,stop)
    fd=os.environ.pop('WHEEL_WORLD_HANDSHAKE_FD',None)
    def handshake(status,**detail):
        nonlocal fd
        if not fd:return
        try:os.write(int(fd),(json.dumps(dict(status=status,pid=os.getpid(),attempt_id=receipt['attempt_id'],
            hapi_session=spec['authorization']['owner_session'],**detail))+'\n').encode())
        except BrokenPipeError:pass  # bounded caller already yielded; owner must continue
        finally:os.close(int(fd));fd=None
    try:result=execute(request,engineering_ready=lambda r:handshake('EVIDENCE_REUSE_VERIFIED' if r.get('reused') else 'ENGINEERING_PASSED',
        stage='train-world' if r.get('reused') else 'collect',engineering_physics_steps=r['physics_steps'],first_optimizer_update_verified=False))
    except BaseException as exc:
        import traceback
        traceback.print_exc();error=type(exc).__name__+': '+str(exc)
    finally:
        handshake('FAILED' if error else 'COMPLETED',error=error)
        terminal(request,result,error,start)
    return 1 if error else 0
