"""Read a recorded distillation workflow, optionally checking its existing files.

Standard-library only: never imports an Actor, trainer or simulator; never writes.
This is an optional handover/reuse check, not another training admission gate.
"""
import argparse
import hashlib
import json
from pathlib import Path

VERSION = 'motion31_distillation_workflow_reference_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inspect(reference_path, root, verify=False):
    root = Path(root).resolve()
    reference = read(reference_path)
    if reference.get('version') != VERSION:
        raise ValueError('unsupported workflow reference version')
    if reference.get('kind') != 'completed_reference_not_training_request':
        raise ValueError('a completed reference is required, not an execution request')
    task = (root / reference['task_directory']).resolve()
    bindings = reference['bindings']
    required = {'terminal', 'source_identity', 'selected_student', 'candidate_frozen', 'export', 'package_checksums'}
    if not required <= bindings.keys():
        raise ValueError('reference is missing terminal/source/selection/freeze bindings')
    failures, hashes = [], {}

    def check(path, expected):
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('invalid SHA256 for ' + str(path))
        if str(path) not in hashes:
            if not path.is_file():
                failures.append({'path': str(path), 'reason': 'MISSING'})
                hashes[str(path)] = None
                return
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            hashes[str(path)] = digest.hexdigest()
        actual = hashes[str(path)]
        if actual is not None and actual != expected:
            failures.append({'path': str(path), 'reason': 'SHA_MISMATCH', 'expected': expected, 'actual': actual})

    if verify:
        for binding in bindings.values():
            check(task / binding['file'], binding['sha256'])
        # Never follow a changed identity document to report a misleading success.
        if failures:
            return {'status': 'REFERENCE_FILES_DIFFER', 'files_written': 0, 'failures': failures}

    terminal = read(task / bindings['terminal']['file'])
    selected = read(task / bindings['selected_student']['file'])
    frozen = read(task / bindings['candidate_frozen']['file'])
    student = terminal['student']
    if any(selected[k] != student[k] for k in ('checkpoint', 'sha256', 'round')):
        raise ValueError('terminal and selected Student differ')
    if frozen['selected'] != selected or frozen['teacher_sha256'] != terminal['teacher']['sha256']:
        raise ValueError('frozen candidate differs; no implicit reselection allowed')
    if verify:
        identity = read(task / bindings['source_identity']['file'])
        for group in ('task_local_sources', 'protected_sources'):
            for path, digest in identity[group].items():
                check(path, digest)
        teacher = terminal['teacher']
        for key, digest_key in [('checkpoint', 'sha256'), ('parent_spec', 'parent_spec_sha256'), ('contract', 'contract_sha256')]:
            check(teacher[key], teacher[digest_key])
        check(student['checkpoint'], student['sha256'])
        check(terminal['onnx']['path'], terminal['onnx']['sha256'])
        package = terminal['package']
        check(package['archive'], package['archive_sha256'])
        for settlement in terminal.get('block_settlements', []):
            check(settlement['file'], settlement['sha256'])
        # The finalized export manifest is already bound by export.json; reuse it.
        exported = read(task / bindings['export']['file'])
        check(exported['manifest'], exported['manifest_sha256'])
        checksum_path = (task / bindings['package_checksums']['file']).resolve()
        for line in checksum_path.read_text(encoding='utf-8').splitlines():
            digest, name = line.split(maxsplit=1)
            member = (checksum_path.parent / name.removeprefix('*')).resolve()
            if not member.is_relative_to(checksum_path.parent):
                raise ValueError('package checksum member escapes its directory')
            check(member, digest)
    return {
        'status': ('REFERENCE_FILES_DIFFER' if failures else 'REFERENCE_FILES_MATCH') if verify else 'READ_ONLY_REVIEW',
        'files_written': 0,
        'physics_steps_executed': 0,
        'optimizer_updates_executed': 0,
        'task_status': terminal['status'],
        'pipeline_completed': terminal['pipeline_completed'],
        'full_teacher_fidelity_passed': terminal['full_teacher_fidelity_passed'],
        'hardware_verified': terminal['hardware_verified'],
        'teacher': terminal['teacher'],
        'student': student,
        'onnx': terminal['onnx'],
        'recorded_matrix_summary': terminal['matrix_summary'],
        'recorded_actual_totals': terminal['actual_totals'],
        'NOT_RUN': terminal['NOT_RUN'],
        'tools': reference['tools'],
        'manual_stages': reference['manual_stages'],
        'files_hashed': len(hashes),
        'failures': failures,
        'scope': 'Read existing evidence only. SHA matches do not repeat numerical/physics tests or certify fidelity/hardware. No training or simulation dispatch.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--verify-files', action='store_true', help='Optionally hash referenced sources/models/results; never executes them')
    args = parser.parse_args()
    try:
        result = inspect(args.reference, args.root, args.verify_files)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {'status': 'INVALID_REFERENCE', 'files_written': 0, 'error': str(error)}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if result['status'] in ('REFERENCE_FILES_DIFFER', 'INVALID_REFERENCE') else 0


if __name__ == '__main__':
    raise SystemExit(main())
