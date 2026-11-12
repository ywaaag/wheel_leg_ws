"""Read-only handover boundary tests; no model inference, optimizer or physics."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

FROZEN_TEST_SCOPE = 'PORTABLE'
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('distillation_reference_review', ROOT / 'scripts/review_distillation_workflow.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def reference(tmp_path):
    task = tmp_path / 'task'
    task.mkdir()
    model = tmp_path / 'actor'
    model.write_bytes(b'fixture, never executed')
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    selected = {'checkpoint': str(model), 'sha256': digest, 'round': 2}
    teacher = {'checkpoint': str(model), 'sha256': digest, 'parent_spec': str(model),
               'parent_spec_sha256': digest, 'contract': str(model), 'contract_sha256': digest}
    terminal = {'student': selected, 'teacher': teacher, 'onnx': {'path': str(model), 'sha256': digest},
                'package': {'archive': str(model), 'archive_sha256': digest},
                'status': 'COMPLETED_WITH_FIDELITY_GAPS', 'pipeline_completed': True,
                'full_teacher_fidelity_passed': False, 'hardware_verified': False,
                'matrix_summary': {'validation': {'passed': 2, 'count': 3}},
                'actual_totals': {'offline_optimizer_updates': 7}, 'NOT_RUN': ['hardware']}
    data = {
        'terminal': terminal, 'selected_student': selected,
        'candidate_frozen': {'selected': selected, 'teacher_sha256': digest},
        'source_identity': {'task_local_sources': {str(model): digest}, 'protected_sources': {}},
        'export': {'manifest': str(model), 'manifest_sha256': digest},
    }
    bindings = {key: {'file': key + '.json', 'sha256': save(task / (key + '.json'), value)} for key, value in data.items()}
    checksum = task / 'SHA256SUMS'
    member = task / 'weights.h'
    member.write_bytes(b'fixture C weights')
    checksum.write_text(hashlib.sha256(member.read_bytes()).hexdigest() + '  weights.h\n')
    bindings['package_checksums'] = {'file': checksum.name, 'sha256': hashlib.sha256(checksum.read_bytes()).hexdigest()}
    value = {'version': review.VERSION, 'kind': 'completed_reference_not_training_request',
             'task_directory': 'task', 'bindings': bindings, 'tools': [], 'manual_stages': []}
    path = tmp_path / 'reference.json'
    save(path, value)
    return path, tmp_path, value


def test_valid_reference_preserves_fidelity_gaps_without_executing_or_writing(reference):
    path, root, _ = reference
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = review.inspect(path, root, verify=True)
    after = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert before == after
    assert result['status'] == 'REFERENCE_FILES_MATCH'
    assert result['full_teacher_fidelity_passed'] is False
    assert result['pipeline_completed'] is True
    assert result['files_written'] == result['physics_steps_executed'] == result['optimizer_updates_executed'] == 0
    assert result['recorded_actual_totals']['offline_optimizer_updates'] == 7


def test_changed_or_missing_frozen_dependency_is_reported(reference):
    path, root, _ = reference
    model = root / 'actor'
    model.write_bytes(b'changed')
    result = review.inspect(path, root, verify=True)
    assert result['status'] == 'REFERENCE_FILES_DIFFER'
    assert any(f['reason'] == 'SHA_MISMATCH' and f['path'] == str(model) for f in result['failures'])
    model.unlink()
    result = review.inspect(path, root, verify=True)
    assert any(f['reason'] == 'MISSING' and f['path'] == str(model) for f in result['failures'])


def test_changed_manifest_is_not_followed(reference):
    path, root, _ = reference
    (root / 'task/source_identity.json').write_text('{"bad": true}')
    result = review.inspect(path, root, verify=True)
    assert result['status'] == 'REFERENCE_FILES_DIFFER'
    assert result['failures'][0]['reason'] == 'SHA_MISMATCH'


def test_selection_cannot_change_after_freeze(reference):
    path, root, value = reference
    selected = json.loads((root / 'task/selected_student.json').read_text())
    selected['round'] = 3
    value['bindings']['selected_student']['sha256'] = save(root / 'task/selected_student.json', selected)
    save(path, value)
    with pytest.raises(ValueError, match='selected Student differ'):
        review.inspect(path, root)


def test_package_checksum_checks_current_directory_not_only_archive(reference):
    path, root, _ = reference
    (root / 'task/weights.h').write_bytes(b'changed after archiving')
    result = review.inspect(path, root, verify=True)
    assert result['status'] == 'REFERENCE_FILES_DIFFER'
    assert any(f['path'].endswith('weights.h') for f in result['failures'])


def test_package_checksum_cannot_escape_package(reference):
    path, root, value = reference
    sums = root / 'task/SHA256SUMS'
    sums.write_text('a' * 64 + '  ../actor\n')
    value['bindings']['package_checksums']['sha256'] = hashlib.sha256(sums.read_bytes()).hexdigest()
    save(path, value)
    with pytest.raises(ValueError, match='escapes'):
        review.inspect(path, root, verify=True)


def test_execution_request_is_never_dispatched(reference):
    path, root, value = reference
    value = copy.deepcopy(value)
    value['kind'] = 'training_request'
    save(path, value)
    with pytest.raises(ValueError, match='not an execution request'):
        review.inspect(path, root)
