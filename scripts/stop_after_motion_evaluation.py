"""Explicit one-shot inotify stop/CPU MuJoCo task; no progress/log polling."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import time

from native_stand.command_v6.tools.health_check import process_identity, digest


def read(path):
    return json.loads(Path(path).read_text())


def verify_evaluation(run, update, spec):
    formal = run / 'formal'
    marker = read(formal / f'reward_refinement_after_{update}.json')
    if marker['update'] != update:
        raise ValueError('evaluation marker update differs')
    candidates = read(formal / 'candidate_registry.json')
    matches = [c for c in candidates if c.get('update') == update
               and Path(c['checkpoint']).parent == formal]
    if len(matches) != 1 or not matches[0].get('complete_matrix'):
        raise ValueError('complete evaluated candidate missing')
    candidate = matches[0]
    if digest(candidate['checkpoint']) != candidate['sha256']:
        raise ValueError('evaluated checkpoint SHA differs')
    records = candidate['evaluations']
    if len(records) != len(spec['student_evaluation']['cases']):
        raise ValueError('evaluation matrix incomplete')
    expected = {(r['path'], r['sha256']) for r in records}
    if expected != {(r['path'], r['sha256']) for r in marker['evaluation_inputs']}:
        raise ValueError('final evaluation marker evidence differs')
    for record in records:
        if digest(formal / record['path']) != record['sha256']:
            raise ValueError('raw evaluation SHA differs')
        rows = record['per_env']
        if len(rows) != 8 or sorted(r['env'] for r in rows) != list(range(8)):
            raise ValueError('expected complete eight-environment evidence')
    return candidate


class Events:
    def __init__(self, directories, deadline):
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
            self.paths[wd] = directory
        self.deadline = deadline

    def wait_for(self, path, abort=None):
        # Register before checking existence, so an atomic rename cannot be missed.
        while not path.exists():
            if abort is not None and abort.exists():
                raise RuntimeError('trainer terminated before requested evaluation')
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.fd], [], [], remaining)[0]:
                raise TimeoutError('one-shot event deadline; no signal sent by timeout')
            block = os.read(self.fd, 65536)
            offset = 0
            relevant = False
            while offset < len(block):
                wd, mask, cookie, size = struct.unpack_from('iIII', block, offset)
                offset += 16
                name = os.fsdecode(block[offset:offset+size].split(b'\0', 1)[0])
                offset += size
                if mask & 0x4000:
                    raise RuntimeError('inotify event overflow; refusing uncertain stop')
                event = self.paths.get(wd, Path('/')) / name
                relevant |= event == path or (abort is not None and event == abort)
            if not relevant:
                # Wait again without opening log/progress/PID files.
                continue

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    plan = read(a.plan)
    run, spec_path = Path(plan['run']), Path(plan['spec'])
    if digest(spec_path) != plan['spec_sha256']:
        raise ValueError('frozen spec SHA differs')
    identity = process_identity(plan['trainer_pid'], output_dir=run/'formal',
        attempt_id=plan['attempt_id'], started_unix_s=plan['process_started_unix_s'],
        spec_path=spec_path, command_token='train.py', physical_gpu=0)
    if identity['status'] != 'OK':
        raise ValueError(identity)
    print(json.dumps({'execute': a.execute, 'identity': identity, 'PPO': 0}), flush=True)
    if not a.execute:
        return
    out = a.plan.parent
    with (out/'execution_claim.json').open('x') as f:
        json.dump({'pid': os.getpid(), 'authorization': 'user explicitly permits this one-shot event task'}, f)
    def write(name, data):
        tmp = out / (name+'.tmp')
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
        tmp.replace(out/name)
    events = Events([run/'formal', run], time.monotonic()+10800)
    result = {'status': 'FAILED', 'signal_sent': False, 'PPO_new': 0}
    try:
        write('task_state.json', {'status': 'ARMED', 'pid': os.getpid(), 'target_update': 500,
              'mechanism': 'kernel inotify, no log/progress polling', 'run': str(run)})
        events.wait_for(run/'formal/reward_refinement_after_500.json', run/'formal/terminal_report.json')
        candidate = verify_evaluation(run, 500, read(spec_path))
        identity = process_identity(plan['trainer_pid'], output_dir=run/'formal',
            attempt_id=plan['attempt_id'], started_unix_s=plan['process_started_unix_s'],
            spec_path=spec_path, command_token='train.py', physical_gpu=0)
        if identity['status'] != 'OK' or digest(spec_path) != plan['spec_sha256']:
            raise ValueError('trainer/spec identity changed before signal')
        libc = events.libc
        pidfd = libc.syscall(434, plan['trainer_pid'], 0)
        if pidfd < 0:
            raise OSError(ctypes.get_errno(), 'pidfd_open')
        try:
            # Recheck while holding the pidfd; never signal a reused numeric PID.
            current = process_identity(plan['trainer_pid'], output_dir=run/'formal',
                attempt_id=plan['attempt_id'], started_unix_s=plan['process_started_unix_s'], spec_path=spec_path)
            if current['status'] != 'OK':
                raise ValueError('trainer changed while acquiring pidfd')
            if libc.syscall(424, pidfd, signal.SIGTERM, 0, 0) < 0:
                raise OSError(ctypes.get_errno(), 'pidfd_send_signal')
        finally:
            os.close(pidfd)
        result.update(signal_sent=True, evaluated_checkpoint=candidate['checkpoint'],
                      evaluated_checkpoint_sha256=candidate['sha256'])
        write('task_state.json', {'status': 'STOP_SIGNAL_SENT_WAITING_SETTLEMENT', **result})
        events.wait_for(run/'chain_terminal_report.json')
        terminal = read(run/'formal/terminal_report.json')
        budget = read(spec_path.with_name('budget.json'))
        if (terminal['status'] != 'STOPPED_BOUNDED' or terminal['stop_kind'] != 'user_signal'
            or terminal['attempt_id'] != plan['attempt_id']
            or budget['attempts'][plan['attempt_id']]['status'] == 'RUNNING'):
            raise ValueError('normal user-stop settlement not verified')
        update = terminal['actual_updates']
        cp = run/'formal'/f'model_update_{update}.pt'
        import torch
        saved = torch.load(cp, map_location='cpu', weights_only=False)
        if not all(k in saved for k in ('actor','critic','optimizer','training_rng','training_recipe')):
            raise ValueError('final checkpoint incomplete')
        if saved['spec_sha256'] != plan['spec_sha256'] or saved['update'] != update:
            raise ValueError('final checkpoint identity/update differs')
        from native_stand.command_v6.domain.experiment_naming import new_experiment_name
        screen = Path.cwd()/'sim2sim'/new_experiment_name('motion_final_low_armature')
        command = [sys.executable, 'scripts/compare_motion_timing_profiles.py',
            '--candidate-spec', str(spec_path), '--candidate-checkpoint', str(cp),
            '--output-dir', str(screen), '--run']
        write('task_state.json', {'status': 'MUJOCO_SCREEN', 'final_update': update,
              'checkpoint': str(cp), 'checkpoint_sha256': digest(cp), 'screen': str(screen)})
        with (out/'screen.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=330)
        summary = read(screen/'summary.json')
        if summary['status'] != 'COMPLETED' or len(summary['results']) != 10:
            raise ValueError('bounded screen incomplete')
        result.update(status='COMPLETED', final_update=update, final_checkpoint=str(cp),
                      final_checkpoint_sha256=digest(cp), screen=str(screen),
                      screen_results=summary['results'])
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        events.close()
        write('task_terminal.json', result)
        write('task_state.json', result)


if __name__ == '__main__':
    main()
