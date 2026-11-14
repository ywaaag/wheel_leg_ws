"""Read-only remote collector contracts; no network, model or engine."""
import base64
import hashlib
import json

import pytest

from native_stand.integrations.remote_dashboard import RemoteError, RemoteSource, classify, process_identity

FROZEN_TEST_SCOPE = 'PORTABLE'


def source_factory(**kwargs):
    return RemoteSource(host='example.test', port=2222, username='reader', root='/allowed/run', fingerprint='SHA256:'+'A'*43, **kwargs)


def payload():
    files = {'run.json': {'run_id': 'remote', 'task': 'motion'},
             'state.json': {'status': 'RUNNING', 'owner': {'pid': 12, 'start_ticks': 42}},
             'budget.json': {'used': {'formal': 9}, 'limits': {'formal': 10}}}
    encoded = {}
    for name, value in files.items():
        raw = json.dumps(value).encode()
        encoded[name] = {'base64': base64.b64encode(raw).decode(), 'sha256': hashlib.sha256(raw).hexdigest(), 'size': len(raw)}
    raw = b'{"update":9,"reward_mean":2}\n{"update":10'
    encoded['metrics.jsonl'] = {'base64': base64.b64encode(raw).decode(), 'sha256': hashlib.sha256(raw).hexdigest(), 'size': len(raw)}
    return {'files': encoded, 'processes': [{'pid': 12, 'start_ticks': 42, 'state': 'S', 'cwd': '/source', 'argv': 'worker'}], 'resources': {'gpu_pid_mapping': 'UNKNOWN'}}


def test_snapshot_and_failed_cache():
    source = source_factory(runner=payload)
    first = source.refresh()
    assert first['status'] == '成功'
    run = first['snapshot']['runs'][0]
    assert run['process'] == '进程已核验'
    assert len(run['metrics']) == 1
    assert run['remaining']['formal'] == 1
    def fail():
        raise RemoteError('连接拒绝')
    source.runner = fail
    failed = source.refresh()
    assert failed['stale'] and failed['snapshot'] == first['snapshot']
    assert failed['age_seconds'] >= 0


@pytest.mark.parametrize(('text','expected'), [('Permission denied','认证失败'),('Connection refused','连接拒绝'),('timed out','超时'),('REMOTE HOST IDENTIFICATION HAS CHANGED','主机密钥不符')])
def test_failures(text, expected):
    assert classify(text) == expected


def test_pid_reuse_zombie_and_unknown():
    state = {'owner': {'pid': 12, 'start_ticks': 42}}
    assert '复用' in process_identity(state,[{'pid':12,'start_ticks':41,'state':'S'}])['status']
    assert '僵尸' in process_identity(state,[{'pid':12,'start_ticks':42,'state':'Z'}])['status']
    assert 'UNKNOWN' in process_identity({'owner':{'pid':12}},payload()['processes'])['status']
    assert '未在' in process_identity(state,[])['status']


def test_path_and_integrity_keep_last():
    source = source_factory(runner=payload)
    source.refresh()
    good = source.last
    wrong = payload()
    wrong['files']['../secret'] = wrong['files']['run.json']
    source.runner = lambda: wrong
    assert source.refresh()['stale']
    assert source.last == good
    wrong = payload()
    wrong['files']['run.json']['sha256'] = '0'*64
    source.runner = lambda: wrong
    assert '身份' in source.refresh()['status']
    assert source.last == good


def test_credentials_never_serialized():
    source = source_factory(password='synthetic-private-input',runner=payload)
    source.refresh()
    assert 'synthetic-private-input' not in json.dumps(source.view())
    source.lock.acquire()
    try:
        assert '连接中' in source.refresh()['status']
    finally:
        source.lock.release()


def test_connection_not_hardcoded():
    source = source_factory(runner=payload)
    source.configure(host='new.example.test', port=23000, username='another', fingerprint='SHA256:'+'B'*43)
    view = source.view()
    assert view['host'] == 'new.example.test' and view['port'] == 23000
    with pytest.raises(ValueError):
        source.configure(host='x;touch', port=22, username='reader', fingerprint='SHA256:'+'B'*43)
    assert source.view()['host'] == 'new.example.test'


def test_declared_state_conflict_and_handshake():
    data = payload()
    data['files']['state.json']['base64'] = base64.b64encode(b'{"status":"STOPPED_BOUNDED","owner":{"pid":12}}').decode()
    raw = base64.b64decode(data['files']['state.json']['base64'])
    data['files']['state.json'].update(sha256=hashlib.sha256(raw).hexdigest(),size=len(raw))
    raw = b'{"worker_pid":12,"worker_start_ticks":42}'
    data['files']['startup_handshake.json'] = {'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)}
    source = source_factory(runner=lambda:data)
    assert '不一致' in source.refresh()['snapshot']['runs'][0]['process']


def test_http_refresh_metadata_and_origin(tmp_path):
    import threading
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    from native_stand.integrations.dashboard import server

    source=source_factory(runner=payload)
    with server([tmp_path],port=0,remote=source) as http:
        thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{http.server_port}'
        try:
            assert json.load(urlopen(base+'/api/snapshot'))['remote']['last_success_unix_s'] is None
            result=json.load(urlopen(Request(base+'/api/remote/refresh',data=b'{}',headers={'Content-Type':'application/json'})))
            assert result['status']=='成功'
            rows=json.load(urlopen(base+'/api/snapshot'))['runs']
            assert rows[0]['remote'] and rows[0]['metrics_file_updated_at'] is None
            with pytest.raises(HTTPError):
                urlopen(Request(base+'/api/remote/refresh',data=b'{}',headers={'Origin':'http://untrusted.example'}))
            with pytest.raises(HTTPError):
                urlopen(Request(base+'/api/remote/refresh',data=b'{"command":"unsafe"}'))
            assert source.last is not None
        finally:
            http.shutdown();thread.join()


def test_wrong_known_host_pin_never_connects(monkeypatch):
    from types import SimpleNamespace
    import native_stand.integrations.remote_dashboard as module

    fake_key=base64.b64encode(b'not-the-pinned-key').decode()
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='example.test ssh-ed25519 '+fake_key+'\n'))
    source=source_factory()
    with pytest.raises(RemoteError,match='主机密钥不符'):
        source._ssh()


def test_expired_cached_snapshot(monkeypatch):
    import native_stand.integrations.remote_dashboard as module

    source=source_factory(runner=payload);source.refresh()
    captured=source.last_success
    monkeypatch.setattr(module.time,'time',lambda:captured+301)
    assert source.view()['stale'] and source.view()['age_seconds']==301


def test_authorized_runs_discovery_and_stable_identity():
    data={'discovered_runs':[{'root':'/allowed/run/first','payload':payload()},
                             {'root':'/allowed/run/second','payload':payload()}]}
    source=source_factory(runner=lambda:data)
    first=source.refresh()
    assert first['status']=='成功'
    assert len(first['snapshot']['runs'])==2
    identities=[row['id'] for row in first['snapshot']['runs']]
    assert len(set(identities))==2
    assert [row['id'] for row in source.refresh()['snapshot']['runs']]==identities
    assert len(source.catalog.roots)==2
    bad={'discovered_runs':[{'root':'/outside/run','payload':payload()}]}
    source.runner=lambda:bad
    assert source.refresh()['stale']
    assert [row['id'] for row in source.last['runs']]==identities


def test_probe_discovery_direct_children_only(tmp_path, monkeypatch):
    import native_stand.integrations.remote_probe as probe
    for name in ('one','two'):
        path=tmp_path/name;path.mkdir();(path/'run.json').write_text('{}')
    nested=tmp_path/'one'/'nested';nested.mkdir();(nested/'run.json').write_text('{}')
    calls=[]
    def collect(root, *, resources=True):
        calls.append((root,resources))
        return {'files':{}}
    monkeypatch.setattr(probe,'collect',collect)
    result=probe.discover(str(tmp_path))
    assert len(result['discovered_runs'])==2
    assert [value for _,value in calls]==[True,False]
