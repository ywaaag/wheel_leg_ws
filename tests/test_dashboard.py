"""Portable read-only dashboard consumer contracts."""

import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from native_stand.artifacts.dashboard import Catalog, compare
from native_stand.integrations.dashboard import server

FROZEN_TEST_SCOPE = "PORTABLE"


def fixture(tmp_path):
    run = tmp_path / "run"
    (run / "diagnostics").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": "one", "task": "motion"}))
    (run / "state.json").write_text('{"status":"RUNNING","updates_used":7}')
    (run / "budget.json").write_text('{"used":{"formal_updates":9}}')
    (run / "metrics.jsonl").write_text('{"update":1,"reward_mean":2}\n{"update":2')
    (run / "diagnostics/update_00100.json").write_text(json.dumps({"onpolicy": {
        "through_update": 100, "version": "real-schema", "rows": [
            {"key": "forward", "ended_episodes": 3, "completed_episodes": 1,
             "pass_fraction": 1/3, "quality_pass_fraction": None}]}}))
    return run


def test_snapshot_and_half_line(tmp_path):
    run = fixture(tmp_path)
    from native_stand.tasks.motion.dashboard import training_windows

    catalog = Catalog([tmp_path], adapters={"motion": training_windows})
    row = catalog.snapshot()["runs"][0]
    assert row["updates_used"] == 7 and row["used"]["formal_updates"] == 9
    assert "未核验" in row["process"]
    assert len(row["metrics"]) == 1
    assert row["windows"][0]["quality_pass_fraction"] is None
    assert row["evaluations"] == [] and row["hardware"] == []
    with (run / "metrics.jsonl").open("a") as stream:
        stream.write(',"reward_mean":3}\n')
    assert len(catalog.snapshot()["runs"][0]["metrics"]) == 2
    replacement = run / "replacement"
    replacement.write_text('{"update":1,"reward_mean":8}\n')
    replacement.replace(run / "metrics.jsonl")
    assert catalog.snapshot()["runs"][0]["metrics"][0]["reward_mean"] == 8


def test_comparison():
    a = dict(layer="training", task="motion", command=[2,.65], height=.34,
             terrain="flat", DR="nominal", execution_contract="C0", protocol="v1",
             ended_episodes=20, pass_fraction=.5)
    assert compare(a, {**a, "pass_fraction": .6}) == "改善"
    assert compare(a, {**a, "command": [2,.8]}) == "不可比较"
    assert compare(a, {**a, "terrain": None}) == "证据不足"
    assert compare(a, {**a, "layer": "fixed"}) == "不可比较"


def test_http_allowlist(tmp_path):
    fixture(tmp_path)
    with server([tmp_path], port=0) as http:
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{http.server_port}"
        try:
            assert "训练与能力看板" in urlopen(base).read().decode()
            data = json.load(urlopen(base + "/api/snapshot"))
            token = data["runs"][0]["windows"][0]["evidence_id"]
            assert json.load(urlopen(base + "/api/evidence/" + token))["onpolicy"]["through_update"] == 100
            for route in ("/../etc/passwd", "/api/evidence/../../etc/passwd", "/api/stop"):
                with pytest.raises(HTTPError):
                    urlopen(base + route)
        finally:
            http.shutdown()
            thread.join()
    with pytest.raises(ValueError):
        Catalog([tmp_path]).read('/etc/passwd')
    with pytest.raises(ValueError):
        server([tmp_path], host="0.0.0.0")


def test_cli_dry_run_no_engine(tmp_path):
    import os
    import subprocess
    import sys

    command = '''import sys
from native_stand.cli import main
assert main(['dashboard','--root','/does/not/exist','--dry-run']) == 0
assert not any(x in sys.modules for x in ('torch','mujoco','isaaclab'))
from native_stand.integrations.dashboard import server
with server(['/tmp'],port=0):
    assert not any(x in sys.modules for x in ('torch','mujoco','isaaclab'))
'''
    result = subprocess.run([sys.executable, '-c', command], env=os.environ.copy(), capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_real_writers_and_evaluation_identity(tmp_path):
    from native_stand.artifacts.evaluation import build_result, register, write_result
    from native_stand.artifacts.metrics import MetricsWriter
    from native_stand.artifacts.report import build_report

    run = fixture(tmp_path)
    (run / 'metrics.jsonl').unlink()
    writer = MetricsWriter(run / 'metrics.jsonl', tensorboard=False, every_updates=1)
    writer.write({'update': 1, 'reward_mean': 4})
    result = build_result(model={'sha256': 'a' * 64}, protocol={'version': 'one'},
                          conditions=[{'condition': 'flat', 'pass_fraction': .8}])
    target = run / 'evaluations/result.json'
    write_result(target, result)
    register(target.parent, kind='final', path=target, result=result)
    row = Catalog([tmp_path]).snapshot()['runs'][0]
    assert row['metrics'][0]['reward_mean'] == 4
    assert row['evaluations'][0]['missing'] == ['backend', 'seeds']
    assert row['fixed_rows'][0]['layer'] == 'fixed'
    assert 'final' in build_report(run)
    target.write_text('{}')
    assert Catalog([tmp_path]).snapshot()['runs'][0]['evaluations'] == []


def test_symlink_and_credentials(tmp_path):
    from native_stand.artifacts.dashboard import public

    run = fixture(tmp_path)
    (run / 'outside.json').symlink_to('/etc/passwd')
    with pytest.raises(ValueError):
        Catalog([tmp_path]).read(run / 'outside.json')
    assert public({'password': 'secret', 'environment': {'KEY': 'value'}, 'value': None}) == {'value': None}


def test_attempts_runs_and_corruption(tmp_path):
    run = fixture(tmp_path)
    rows = [{'attempt': 'a', 'stage': 'formal', 'update': 1, 'reward_mean': 2},
            {'attempt': 'a', 'stage': 'formal', 'update': 1, 'reward_mean': 3},
            {'attempt': 'b', 'stage': 'formal', 'update': 1, 'reward_mean': 4}]
    (run / 'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    catalog = Catalog([tmp_path])
    assert [x['reward_mean'] for x in catalog.snapshot()['runs'][0]['metrics']] == [3, 4]
    second = tmp_path / 'other'
    second.mkdir()
    (second / 'run.json').write_text('{"task":"jump"}')
    assert len(catalog.snapshot()['runs']) == 2
    (run / 'metrics.jsonl').write_text('{"update":1,"loss":NaN}\n')
    snapshot = catalog.snapshot()
    row = next(x for x in snapshot['runs'] if x['task'] == 'motion')
    assert row['status'] == 'RUNNING' and row['metrics'] == []
    assert any(x['error'] == 'ValueError' for x in row['errors'])


def test_evidence_replacement(tmp_path):
    run = fixture(tmp_path)
    catalog = Catalog([tmp_path])
    token = catalog.register(run / 'run.json')
    (run / 'run.json').write_text('{"task":"recovery"}')
    with pytest.raises(ValueError):
        catalog.read_evidence(token)


def test_domain_target_and_legacy_counts():
    from native_stand.tasks.motion.dashboard import target_description, training_windows

    rows = training_windows({'onpolicy': {'rows': [{'actual_student_policy_samples': 10000}]}}, {}, 'evidence')
    assert rows[0]['sample_count_status'] == 'UNKNOWN'
    assert rows[0]['actual_student_policy_samples'] is None
    assert target_description('| M01 | standing | keep | P0 |')['rows'][0]['capability_id'] == 'M01'


def test_comparison_http_consumer(tmp_path):
    fixture(tmp_path)
    with server([tmp_path], port=0) as http:
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{http.server_port}'
        try:
            snapshot = json.load(urlopen(base+'/api/snapshot'))
            run_id = snapshot['runs'][0]['id']
            response = json.load(urlopen(base+f'/api/compare?run={run_id}&layer=windows&left=0&right=0'))
            assert response['verdict'] == '证据不足'
            assert len(snapshot['targets']['motion']['rows']) == 8
        finally:
            http.shutdown()
            thread.join()
    with pytest.raises(ValueError):
        server([tmp_path], tensorboard_url='http://password:secret@example.org')


def test_explicit_proxy_host(tmp_path):
    from urllib.request import Request

    fixture(tmp_path)
    with server([tmp_path], port=0, public_host='100.127.219.54') as http:
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{http.server_port}'
        try:
            assert urlopen(Request(base, headers={'Host': '100.127.219.54'})).status == 200
            with pytest.raises(HTTPError) as denied:
                urlopen(Request(base, headers={'Host': 'untrusted.example'}))
            assert denied.value.code == 403
        finally:
            http.shutdown()
            thread.join()
