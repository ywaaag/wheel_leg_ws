"""One authorized, event-driven standing completion -> Motion successor.

No training-log polling. Runtime changes are installed only after the exact
parent trainer has normally exited and its ledger has settled.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


class Events:
    """Block on kernel events; no repeated timer/file/process inventories."""
    def __init__(self, directories):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.fd = self.libc.inotify_init1(os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), 'inotify_init1')
        self.paths = {}
        for directory in directories:
            wd = self.libc.inotify_add_watch(self.fd, os.fsencode(directory), 0x8 | 0x80)
            if wd < 0:
                self.close()
                raise OSError(ctypes.get_errno(), 'inotify_add_watch')
            self.paths[wd] = Path(directory)

    def wait(self, relevant):
        while True:
            select.select([self.fd], [], [])
            data = os.read(self.fd, 65536)
            offset = 0
            changed = []
            while offset < len(data):
                wd, mask, _, size = struct.unpack_from('iIII', data, offset)
                offset += 16
                name = os.fsdecode(data[offset:offset + size].split(b'\0', 1)[0])
                offset += size
                if mask & 0x4000:
                    raise RuntimeError('inotify overflow; uncertain evidence, no blind signal/launch')
                path = self.paths.get(wd, Path('/')) / name
                if relevant(path):
                    changed.append(path)
            if changed:
                return changed

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def completed_pair(plan):
    """Verify two distinct consecutive final-stage matrices and their raw SHA."""
    from native_stand.students.motion_antioscillation import refinement_records
    from native_stand.students.motion_parallel_assessment import safety_protocol
    spec_path = Path(plan['parent_spec'])
    if sha(spec_path) != plan['parent_spec_sha256']:
        raise ValueError('parent spec changed')
    run = Path(plan['parent_run'])
    formal = run / 'formal'
    progress_path = formal / 'course_progress.json'
    if not progress_path.exists():
        return None
    progress = read(progress_path)
    state, decision = progress['state'], progress['decision']
    if not (state['stage'] == 4 and state.get('consecutive_passes', 0) >= 2
            and decision.get('all_stages_passed') is True):
        return None
    update = state['last_evaluation_update']
    marker_path = formal / f'reward_refinement_after_{update}.json'
    # This marker is written AFTER the complete evaluation and CPU bookkeeping.
    if not marker_path.exists():
        if (formal / 'terminal_report.json').exists():
            raise ValueError('final confirmation lacks completed post-evaluation marker')
        return None
    registry = read(formal / 'candidate_registry.json')
    candidates = sorted((c for c in registry if c.get('stage') == 4
                         and c.get('evaluated', True) and c.get('evaluations')
                         and c.get('update', 0) <= update), key=lambda c: c['update'])
    if len(candidates) < 2 or candidates[-1]['update'] != update:
        raise ValueError('two final-stage evaluated checkpoints missing')
    pair = candidates[-2:]
    if pair[1]['update'] - pair[0]['update'] != 250 or pair[0]['sha256'] == pair[1]['sha256']:
        raise ValueError('need two consecutive real 250-update checkpoint intervals')
    spec = read(spec_path)
    proof = {}
    for candidate in pair:
        cp = Path(candidate['checkpoint'])
        if cp != formal / f"model_update_{candidate['update']}.pt" or sha(cp) != candidate['sha256']:
            raise ValueError('final-stage checkpoint path/SHA differs')
        records = candidate.get('evaluations', [])
        merged = refinement_records(records, 4)[0]['per_env']
        if (candidate.get('complete_matrix') is not True or len(merged) != 8
                or not all(e.get('accepted') is True for e in merged)):
            raise ValueError('final-stage complete passing matrix required')
        marker = formal / f"reward_refinement_after_{candidate['update']}.json"
        m = read(marker)
        if m['update'] != candidate['update'] or m['evaluation_inputs'] != [
                {'path': r['path'], 'sha256': r['sha256']} for r in records]:
            raise ValueError('post-evaluation marker does not bind raw matrix')
        proof[str(marker)] = sha(marker)
        for record in records:
            path = (formal / record['path']).resolve()
            if path.parent != formal or sha(path) != record['sha256']:
                raise ValueError('raw evaluation path/SHA differs')
            raw = read(path)
            level = record['case']['_timing_level']
            timing = raw['motion_timing']
            levels = timing['actuator_readback']['levels']
            if (raw['checkpoint_sha256'] != candidate['sha256']
                    or raw['spec_sha256'] != plan['parent_spec_sha256']
                    or raw['case'] != record['case'] or raw.get('model_unchanged') is not True
                    or raw['protocol'] != safety_protocol(spec)
                    or raw.get('scored_envs') != 8 or raw['per_env'] != record['per_env']
                    or timing['override'] != level or levels[level] != sum(levels)
                    or sum(levels) != 64):
                raise ValueError('final evaluation runtime/candidate identity differs')
            trace = path.parent / raw['trace']['path']
            if sha(trace) != raw['trace']['sha256']:
                raise ValueError('final evaluation trace SHA differs')
            proof[str(path)] = sha(path)
            proof[str(trace)] = sha(trace)
        proof[str(cp)] = candidate['sha256']
    if progress['checkpoint_sha256'] != pair[-1]['sha256']:
        raise ValueError('course receipt checkpoint differs')
    return {'status': 'CONFIRMED', 'checkpoints': [
        {k: c[k] for k in ('checkpoint', 'sha256', 'update', 'stage')} for c in pair],
        'inputs': proof, 'parent_spec_sha256': plan['parent_spec_sha256']}


def stop_parent(plan, out):
    from native_stand.command_v6.tools.health_check import process_identity
    args = dict(output_dir=Path(plan['parent_run']) / 'formal',
                attempt_id=plan['attempt_id'], started_unix_s=plan['process_started_unix_s'],
                spec_path=Path(plan['parent_spec']), command_token='train.py',
                physical_gpu=plan['physical_gpu'])
    identity = process_identity(plan['trainer_pid'], **args)
    if identity['status'] != 'OK' or identity.get('environment', {}).get('HAPI_SESSION_ID') != plan['session']:
        raise ValueError(identity)
    libc = ctypes.CDLL(None, use_errno=True)
    pidfd = libc.syscall(434, plan['trainer_pid'], 0)
    if pidfd < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_open')
    try:
        if process_identity(plan['trainer_pid'], **args)['status'] != 'OK':
            raise ValueError('trainer identity changed while acquiring pidfd')
        with (out / 'stop_claim.json').open('x') as f:
            json.dump({'pid': plan['trainer_pid'], 'attempt_id': plan['attempt_id'],
                       'signal': 'SIGTERM', 'repeat_signal': False}, f)
        if libc.syscall(424, pidfd, signal.SIGTERM, 0, 0) < 0:
            raise OSError(ctypes.get_errno(), 'pidfd_send_signal')
    finally:
        os.close(pidfd)


def verify_settlement(plan):
    run = Path(plan['parent_run'])
    chain = read(run / 'chain_terminal_report.json')
    terminal = read(run / 'formal/terminal_report.json')
    book = read(Path(plan['parent_spec']).with_name('budget.json'))
    if (chain['status'] not in ('STOPPED_BOUNDED', 'COMPLETED') or chain.get('failure')
            or terminal.get('failure') or terminal['status'] not in ('STOPPED_BOUNDED', 'COMPLETED')
            or terminal['attempt_id'] != plan['attempt_id']
            or book['used']['formal'] != terminal['actual_updates']
            or any(a['status'] == 'RUNNING' for a in book['attempts'].values())):
        raise ValueError('normal parent settlement required; no automatic infrastructure retry')
    return {'formal_updates': book['used']['formal'], 'smoke_updates': book['used']['smoke'],
            'chain_status': chain['status'], 'stop_kind': terminal['stop_kind'],
            'historical_consumption_retained': True}


def install_bundle(bundle, root=ROOT):
    """Compare all bases first; apply only the reviewed explicit Python files."""
    manifest = read(bundle / 'manifest.json')
    ready = read(bundle / 'bundle_ready.json')
    if ready['manifest_sha256'] != sha(bundle / 'manifest.json') or ready['status'] != 'CPU_PASSED':
        raise ValueError('validated bundle identity differs')
    files = manifest['files']
    for row in files:
        rel = Path(row['path'])
        if (rel.is_absolute() or '..' in rel.parts or rel.suffix != '.py'
                or rel.parts[0] not in ('src', 'tests')):
            raise ValueError('unsupported bundle destination')
        target, payload = root / rel, bundle / 'files' / rel
        actual = sha(target) if target.exists() else None
        if actual != row['before_sha256'] or sha(payload) != row['after_sha256']:
            raise ValueError('bundle source changed; refusing overwrite: ' + str(rel))
    previous, staged, installed = {}, [], []
    try:
        for row in files:
            target = root / row['path']
            previous[target] = target.read_bytes() if target.exists() else None
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + '.transition.tmp')
            with temp.open('xb') as stream:
                stream.write((bundle / 'files' / row['path']).read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((target, temp))
        for target, temp in staged:
            temp.replace(target)
            installed.append(target)
    except BaseException:
        for target in reversed(installed):
            if previous[target] is None:
                target.unlink()
            else:
                target.write_bytes(previous[target])
        raise
    finally:
        for _, temp in staged:
            if temp.exists():
                temp.unlink()
    return manifest


def prepare_successor(plan, proof, out):
    from native_stand.command_v6.domain.experiment_naming import new_experiment_name
    from native_stand.command_v6.tools.experiment_preparation import publish
    from native_stand.command_v6.tools.register_authorization import register
    from native_stand.students.motion_reacquisition import build, verify, reward_snapshot
    cp = proof['checkpoints'][-1]['checkpoint']
    name = new_experiment_name('motion_posture_reacquisition140_10000')
    folder = ROOT / 'configs/experiments' / name
    spec = build(plan['parent_spec'], cp, experiment=name, spec_path=folder / 'spec.json',
                 confirmation=proof, owner_session=plan['session'])
    verify(spec)
    if spec['artifacts']['storage_max_bytes'] != plan['authorization']['storage_max_bytes']:
        raise ValueError('successor storage differs from the explicit transition plan')
    phases = ('source_eval', 'smoke', 'formal', 'evaluate', 'play')
    ledger = {'authorization': 'UNAUTHORIZED', 'experiment': name, 'hapi_session': plan['session'],
              'limits': dict(zip(phases, (1, 5, 9995, 0, 0))),
              'used': dict.fromkeys(phases, 0), 'attempts': {}, 'source_eval_wall_used_s': 0.,
              'total_online_PPO_limit': 10000, 'previous_experiment_consumption': verify_settlement(plan)}
    publish(folder, {'spec.json': spec, 'budget.json': ledger,
                    'reward_snapshot.json': reward_snapshot(spec),
                    'preparation.json': {'simulation': 0, 'PPO': 0, 'parent_confirmation': proof}},
            spec['student_dependencies'], completion_marker=True)
    request = {'approval_id': 'user_motion_reacquisition10000_20261022_' + name,
               'user_scope': spec['authorization']['user_scope'], 'session': plan['session'],
               'physical_gpu': plan['physical_gpu'], 'gpu_uuid': plan['gpu_uuid'],
               'max_online_updates': 10000, 'max_phase_wall_s': None,
               'arms': [{'spec': str(folder / 'spec.json'), 'spec_sha256': sha(folder / 'spec.json'),
                         'budget_sha256': sha(folder / 'budget.json')}]}
    registration = register(request, current_session=plan['session'], write=True)
    write(out / 'successor_preparation.json', {'spec': str(folder / 'spec.json'),
          'run': str(ROOT / 'runs' / name), 'registration': registration})
    return folder / 'spec.json', ROOT / 'runs' / name


def worker(plan_path, handshake_fd=None):
    plan = read(plan_path)
    if (plan.get('controller_sha256') and sha(__file__) != plan['controller_sha256']) or os.environ.get('HAPI_SESSION_ID') != plan['session']:
        raise ValueError('controller code/session identity differs')
    out, run = plan_path.parent, Path(plan['parent_run'])
    bundle = Path(plan['bundle_dir'])
    result = {'status': 'FAILED', 'signal_sent': False, 'new_training_submitted': False}
    with (out / 'execution_claim.json').open('x') as f:
        json.dump({'pid': os.getpid(), 'plan_sha256': sha(plan_path)}, f)
    events = Events([run / 'formal', run, bundle])
    try:
        write(out / 'task_state.json', {'status': 'ARMED', 'pid': os.getpid(),
              'mechanism': 'kernel inotify; no log/progress polling', 'plan_sha256': sha(plan_path)})
        if handshake_fd is not None:
            os.write(handshake_fd, b'{"status":"ARMED"}\n')
            os.close(handshake_fd)
        proof = completed_pair(plan)
        while proof is None:
            if (run / 'chain_terminal_report.json').exists():
                raise ValueError('parent terminated without completed final-stage confirmation')
            changed = events.wait(lambda p: p.name == 'chain_terminal_report.json'
                                 or (p.name.startswith('reward_refinement_after_') and p.suffix == '.json'))
            if any(p.name == 'chain_terminal_report.json' for p in changed):
                proof = completed_pair(plan)
                if proof is None:
                    raise ValueError('parent ended before final confirmation')
            else:
                proof = completed_pair(plan)
        write(out / 'parent_confirmation.json', proof)
        if not (run / 'chain_terminal_report.json').exists():
            stop_parent(plan, out)
            result['signal_sent'] = True
            write(out / 'task_state.json', {**result, 'status': 'WAITING_NORMAL_SETTLEMENT'})
            while not (run / 'chain_terminal_report.json').exists():
                events.wait(lambda p: p == run / 'chain_terminal_report.json')
        result['parent_settlement'] = verify_settlement(plan)
        write(out / 'task_state.json', {**result, 'status': 'PARENT_SETTLED_WAITING_CPU_BUNDLE'})
        while not (bundle / 'bundle_ready.json').exists():
            events.wait(lambda p: p == bundle / 'bundle_ready.json')
        result['installed_bundle'] = install_bundle(bundle)
        write(out / 'task_state.json', {**result, 'status': 'VALIDATING_INSTALLED_CODE'})
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(ROOT / 'src'),
               'HAPI_SESSION_ID': plan['session']}
        with (out / 'installed_cpu_tests.log').open('x') as log:
            subprocess.run([sys.executable, 'scripts/test_portable_cpu.py'], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        spec, destination = prepare_successor(plan, proof, out)
        command = [sys.executable, '-m', 'native_stand.command_v6.supervisor', '--spec', str(spec),
                   '--output-dir', str(destination), '--physical-gpu', str(plan['physical_gpu']),
                   '--hapi-session', plan['session']]
        with (out / 'submission_check.json').open('x') as log:
            subprocess.run(command + ['--check-submission'], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        write(out / 'task_state.json', {**result, 'status': 'SUBMITTING_SUCCESSOR', 'spec': str(spec)})
        with (out / 'successor_receipt.json').open('x') as log:
            subprocess.run(command + ['--launch', '--handshake-wait-s', '30'], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        receipt = read(out / 'successor_receipt.json')
        if receipt.get('submitted') is not True or receipt.get('submission_verified') is not True:
            raise ValueError('successor has no verified submission receipt')
        result.update(status='COMPLETED', new_training_submitted=True, successor_spec=str(spec),
                      successor_run=str(destination), receipt=str(out / 'successor_receipt.json'),
                      formal_first_update_verified=receipt.get('formal_first_update_verified', False))
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        events.close()
        write(out / 'task_terminal.json', result)
        write(out / 'task_state.json', result)
        if result['status'] == 'FAILED':
            # Current owner has explicitly authorized notifications. At most one
            # attempt, no resend when delivery is uncertain, no new watcher.
            notification = out / 'failure_notification.json'
            if not notification.exists():
                write(notification, {'status': 'delivery_started', 'session': plan['session']})
                try:
                    from native_stand.command_v6.completion import transport
                    delivery = transport(plan['session'],
                        f"Motion站立转运动事件任务失败：{out}\n原因：{result.get('error')}\n"
                        "请读取task_terminal.json与task.log；该事件任务不会自行重复提交。")
                    write(notification, delivery)
                except BaseException as error:
                    write(notification, {'status': 'delivery_unknown', 'error': repr(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--arm', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--handshake-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.plan.resolve(), args.handshake_fd)
    plan = read(args.plan)
    if sha(plan['parent_spec']) != plan['parent_spec_sha256']:
        raise ValueError('parent spec SHA differs')
    if not args.arm:
        print(json.dumps({'plan': plan, 'files_written': 0, 'simulation': 0, 'PPO': 0}, indent=2))
        return
    out = args.plan.resolve().parent
    with (out / 'arm_claim.json').open('x') as f:
        json.dump({'plan_sha256': sha(args.plan), 'session': plan['session']}, f)
    reader, writer = os.pipe()
    with (out / 'task.log').open('x') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            '--plan', str(args.plan.resolve()), '--worker', '--handshake-fd', str(writer)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, pass_fds=(writer,), start_new_session=True)
    os.close(writer)
    try:
        if not select.select([reader], [], [], 10)[0]:
            raise TimeoutError('event task handshake unknown; do not submit a duplicate')
        handshake = json.loads(os.read(reader, 4096))
        if handshake.get('status') != 'ARMED':
            raise ValueError(handshake)
    finally:
        os.close(reader)
    receipt = {'status': 'ARMED', 'pid': process.pid, 'plan': str(args.plan.resolve()),
               'automatic_successors': 1, 'new_online_limit': 10000,
               'not_started_yet': 'waiting for final .012 confirmation and normal settlement'}
    write(out / 'launch_receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
