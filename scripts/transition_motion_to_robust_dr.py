"""One explicitly authorized kernel-event promotion -> robust Motion successor."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('standing_transition_helpers', ROOT/'scripts/transition_motion_after_standing.py')
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
read, write, sha, Events = _helpers.read, _helpers.write, _helpers.sha, _helpers.Events


def promotion(plan):
    """Require parent v1's actual completed promotion; never rewrite its gate."""
    from native_stand.students.motion_reacquisition import commands, refinement_records
    run, p = Path(plan['parent_run']), Path(plan['parent_spec'])
    if sha(p) != plan['parent_spec_sha256']:
        raise ValueError('parent declaration changed')
    formal = run/'formal'
    progress = read(formal/'course_progress.json')
    stage = plan['watch_stage']
    certificate = progress['state']['accepted_by_stage'].get(str(stage))
    if not certificate or certificate['update'] <= plan['after_evaluation_update']:
        return None
    update = certificate['update']
    marker = formal/f'reward_refinement_after_{update}.json'
    if not marker.exists():
        return None
    registry = read(formal/'candidate_registry.json')
    pair = sorted((c for c in registry if c.get('stage') == stage and c.get('evaluations')
                   and c.get('update') in (update-250, update)), key=lambda c: c['update'])
    if len(pair) != 2 or pair[0]['sha256'] == pair[1]['sha256']:
        raise ValueError('parent two-checkpoint stage confirmation missing')
    inputs = {}
    for c in pair:
        records = c['evaluations']
        expected = refinement_records(records, stage)
        if (not c.get('complete_matrix') or len(records) != len(commands(stage))
                or any(len(r['per_env']) != 8 or not all(e.get('accepted') is True for e in r['per_env']) for r in expected)):
            raise ValueError('parent promoted stage lacks complete passing matrix')
        cp = Path(c['checkpoint'])
        if cp.parent != formal or sha(cp) != c['sha256']:
            raise ValueError('parent confirmation checkpoint SHA/path differs')
        inputs[str(cp)] = c['sha256']
        for r in records:
            raw_path = formal/r['path']
            if raw_path.resolve().parent != formal or sha(raw_path) != r['sha256']:
                raise ValueError('parent evaluation SHA/path differs')
            raw = read(raw_path)
            if (raw['spec_sha256'] != plan['parent_spec_sha256']
                    or raw['checkpoint_sha256'] != c['sha256'] or raw['case'] != r['case']
                    or raw['per_env'] != r['per_env'] or raw.get('model_unchanged') is not True):
                raise ValueError('parent raw evaluation identity differs')
            trace = formal/raw['trace']['path']
            if trace.resolve().parent != formal or sha(trace) != raw['trace']['sha256']:
                raise ValueError('parent raw trace SHA/path differs')
            inputs[str(raw_path)] = r['sha256']; inputs[str(trace)] = raw['trace']['sha256']
        m = read(formal/f"reward_refinement_after_{c['update']}.json")
        if m['evaluation_inputs'] != [{'path':r['path'],'sha256':r['sha256']} for r in records]:
            raise ValueError('post-evaluation marker does not bind matrix')
    if certificate['sha256'] != pair[-1]['sha256']:
        raise ValueError('accepted-stage receipt differs')
    inputs[str(marker)] = sha(marker)
    return dict(status='CONFIRMED', certified_skill_stage=stage,
                checkpoints=[{k:c[k] for k in ('checkpoint','sha256','update','stage')} for c in pair],
                parent_spec_sha256=plan['parent_spec_sha256'], inputs=inputs)


def latest_stopped(plan):
    import torch
    formal = Path(plan['parent_run'])/'formal'
    term = read(formal/'terminal_report.json')
    latest = formal/f"model_update_{term['actual_updates']}.pt"
    cp = torch.load(latest, map_location='cpu', weights_only=False)
    if (cp['update'] != term['actual_updates'] or cp['spec_sha256'] != plan['parent_spec_sha256']
            or cp['critic_privileged_version'] != 'motion_antiosc_privileged140_v1'
            or not all(k in cp for k in ('actor','critic','optimizer','training_rng','reward_refinement_state'))):
        raise ValueError('latest stopped complete checkpoint required')
    return {'checkpoint':str(latest),'sha256':sha(latest),'update':cp['update'],
            'evaluated':any(c.get('sha256') == sha(latest) and c.get('evaluated',True)
                            and c.get('evaluations') for c in read(formal/'candidate_registry.json'))}


def prepare(plan, proof, latest, out):
    from native_stand.students.motion_robustness import build, verify, reward_snapshot
    from native_stand.command_v6.domain.experiment_naming import new_experiment_name
    from native_stand.command_v6.tools.experiment_preparation import publish
    from native_stand.command_v6.tools.register_authorization import register
    name = new_experiment_name('motion_alternating_standing_dr_10000')
    folder = ROOT/'configs/experiments'/name
    s = build(plan['parent_spec'], latest['checkpoint'], experiment=name,
              spec_path=folder/'spec.json', confirmation=proof, owner_session=plan['session'],
              storage_max_bytes=plan['authorization']['storage_max_bytes'])
    verify(s)
    settlement = _helpers.verify_settlement(plan)
    phases = ('source_eval','smoke','formal','evaluate','play')
    ledger = dict(authorization='UNAUTHORIZED',experiment=name,hapi_session=plan['session'],
                  limits=dict(zip(phases,(1,5,9995,0,0))),used=dict.fromkeys(phases,0),attempts={},
                  source_eval_wall_used_s=0.,total_online_PPO_limit=10000,
                  previous_experiment_consumption=settlement)
    publish(folder,{'spec.json':s,'budget.json':ledger,'reward_snapshot.json':reward_snapshot(s),
                    'preparation.json':{'PPO':0,'simulation':0,'parent_confirmation':proof,'latest_source':latest}},
            s['student_dependencies'],completion_marker=True)
    registration = register(dict(approval_id='user_motion_robust_dr_20261023_'+name,
        user_scope=s['authorization']['user_scope'],session=plan['session'],physical_gpu=0,
        gpu_uuid=plan['gpu_uuid'],max_online_updates=10000,max_phase_wall_s=None,
        arms=[dict(spec=str(folder/'spec.json'),spec_sha256=sha(folder/'spec.json'),budget_sha256=sha(folder/'budget.json'))]),
        current_session=plan['session'],write=True)
    write(out/'successor_preparation.json',dict(spec=str(folder/'spec.json'),run=str(ROOT/'runs'/name),registration=registration))
    return folder/'spec.json',ROOT/'runs'/name


def worker(path, fd):
    plan = read(path); out = path.parent; run = Path(plan['parent_run']); bundle = Path(plan['bundle_dir'])
    if sha(__file__) != plan['controller_sha256'] or os.environ.get('HAPI_SESSION_ID') != plan['session']:
        raise ValueError('controller/session identity differs')
    with (out/'execution_claim.json').open('x') as f:
        json.dump({'pid':os.getpid(),'plan_sha256':sha(path)},f)
    events = Events([run/'formal',run,bundle])
    result = dict(status='FAILED',signal_sent=False,new_training_submitted=False)
    try:
        write(out/'task_state.json',dict(status='ARMED',pid=os.getpid(),mechanism='kernel inotify; no timer polling',
                                        watched_stage=plan['watch_stage'],automatic_successors=1))
        os.write(fd,b'{"status":"ARMED"}\n');os.close(fd)
        proof = promotion(plan)
        while proof is None:
            if (run/'chain_terminal_report.json').exists():
                raise ValueError('parent ended before required promotion; review instead of blind successor')
            events.wait(lambda p:p.name=='chain_terminal_report.json' or p.name.startswith('reward_refinement_after_'))
            proof = promotion(plan)
        write(out/'parent_confirmation.json',proof)
        if not (run/'chain_terminal_report.json').exists():
            _helpers.stop_parent(plan,out);result['signal_sent']=True
            write(out/'task_state.json',dict(**result,phase='WAITING_NORMAL_SETTLEMENT'))
            while not (run/'chain_terminal_report.json').exists():
                events.wait(lambda p:p==run/'chain_terminal_report.json')
        result['parent_settlement']=_helpers.verify_settlement(plan)
        latest=latest_stopped(plan);write(out/'latest_source.json',latest)
        write(out/'task_state.json',dict(**result,phase='PARENT_SETTLED_WAITING_CPU_BUNDLE',latest_source=latest))
        while not (bundle/'bundle_ready.json').exists():
            events.wait(lambda p:p==bundle/'bundle_ready.json')
        result['installed_bundle']=_helpers.install_bundle(bundle)
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src'),'HAPI_SESSION_ID':plan['session']}
        with (out/'installed_cpu_tests.log').open('x') as log:
            subprocess.run([sys.executable,'scripts/test_portable_cpu.py'],cwd=ROOT,env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        spec,destination=prepare(plan,proof,latest,out)
        command=[sys.executable,'-m','native_stand.command_v6.supervisor','--spec',str(spec),
                 '--output-dir',str(destination),'--physical-gpu','0','--hapi-session',plan['session']]
        with (out/'submission_check.json').open('x') as log:
            subprocess.run(command+['--check-submission'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        with (out/'successor_receipt.json').open('x') as log:
            subprocess.run(command+['--launch','--handshake-wait-s','30'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        receipt=read(out/'successor_receipt.json')
        if not receipt.get('submitted') or not receipt.get('submission_verified'):
            raise ValueError('no verified successor submission receipt')
        result.update(status='COMPLETED',new_training_submitted=True,successor_spec=str(spec),successor_run=str(destination))
        if receipt.get('run_state')=='FAILED' or receipt.get('status')=='FAILED':
            result['status']='FAILED'
            raise RuntimeError('successor submitted but failed during startup; inspect its terminal before recovery')
    except BaseException as error:
        result['error']=repr(error)
        raise
    finally:
        events.close();write(out/'task_terminal.json',result);write(out/'task_state.json',result)
        if result['status']=='FAILED' and not (out/'failure_notification.json').exists():
            write(out/'failure_notification.json',dict(status='delivery_started',session=plan['session']))
            try:
                from native_stand.command_v6.completion import transport
                delivery=transport(plan['session'],f"Motion DR交接异常：{out}\n原因：{result.get('error')}\n用户已授权排查接线错误、保持累计10000额度后恢复；请读取task_terminal.json，不重复arm。")
                write(out/'failure_notification.json',delivery)
            except BaseException as error:
                write(out/'failure_notification.json',dict(status='delivery_unknown',error=repr(error)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--arm',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--handshake-fd',type=int,help=argparse.SUPPRESS);a=p.parse_args();path=a.plan.resolve()
    if a.worker:return worker(path,a.handshake_fd)
    plan=read(path)
    if not a.arm:
        print(json.dumps(dict(plan=plan,files_written=0,simulation=0,PPO=0),indent=2));return
    if sha(__file__)!=plan['controller_sha256'] or sha(plan['parent_spec'])!=plan['parent_spec_sha256']:
        raise ValueError('reviewed controller/parent SHA changed')
    with (path.parent/'arm_claim.json').open('x') as f:json.dump({'plan_sha256':sha(path)},f)
    reader,writer=os.pipe()
    with (path.parent/'task.log').open('x') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--plan',str(path),
            '--worker','--handshake-fd',str(writer)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=(writer,))
    os.close(writer)
    try:
        if not select.select([reader],[],[],10)[0]:raise TimeoutError('unknown event-worker handshake; do not duplicate')
        if json.loads(os.read(reader,4096)).get('status')!='ARMED':raise RuntimeError('event-worker not armed')
    finally:os.close(reader)
    receipt=dict(status='ARMED',pid=process.pid,plan=str(path),automatic_successors=1,online_limit=10000,
                 watch_stage=plan['watch_stage'],notification_session=plan['session'],new_training_started=False)
    write(path.parent/'launch_receipt.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
