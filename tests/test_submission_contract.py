import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from native_stand.command_v6.domain.submission_contract import authorization, metadata
from native_stand.command_v6 import supervisor


@pytest.mark.parametrize('value', ['AUTHORIZED', 'UNAUTHORIZED', {'status':'AUTHORIZED'},
                                  {'hapi_session':'current'}, None])
def test_legacy_authorization_is_normalized_without_rewriting(value):
    spec={'authorization':value}
    original=deepcopy(spec)
    effective=authorization(spec)
    assert isinstance(effective,dict)
    assert effective['status'] in ('AUTHORIZED','UNAUTHORIZED','UNKNOWN')
    assert spec==original


@pytest.mark.parametrize('value', [True, [], 1, 'YES', {'status':False},
                                  {'hapi_session':''}, {'hapi_session':7}])
def test_invalid_authorization_rejected(value):
    with pytest.raises(ValueError):authorization({'authorization':value})


def args_for(tmp_path):
    return SimpleNamespace(spec=tmp_path/'spec.json',output_dir=tmp_path/'new',
        formal_only=False,evaluation_only=False,hapi_session='current',physical_gpu=1,
        resource_policy={'mode':'exclusive'})


def health():
    return {'findings':[],'gpu':{'status':'OK','devices':[{'index':1,'uuid':'gpu1'}],
                               'compute_processes':[]}}


def test_readiness_needs_resource_evidence_and_preflight(tmp_path,monkeypatch):
    monkeypatch.setenv('HAPI_SESSION_ID','current')
    args=args_for(tmp_path)
    spec={'authorization':'AUTHORIZED','device':{'physical_index':1,'uuid':'gpu1'}}
    result=supervisor.submission_readiness(args,spec,{'status':'PASS'})
    assert result['status']=='READY_METADATA' and not result['submit_ready']
    assert supervisor.submission_readiness(args,spec,{'status':'PASS'},health())['submit_ready']
    rejected=supervisor.submission_readiness(args,spec,{'status':'READY_NEEDS_AUTH',
        'reason_code':'BUDGET_EXHAUSTED'},health())
    assert not rejected['submit_ready'] and rejected['reason_code']=='BUDGET_EXHAUSTED'
    spec['device']['uuid']='wrong'
    assert not supervisor.submission_readiness(args,spec,{'status':'PASS'},health())['submit_ready']
    assert not args.output_dir.exists()


@pytest.mark.parametrize('changes',[dict(current_session=None),dict(target_session=None),
    dict(target_session='historical'),dict(physical_gpu=None),dict(physical_gpu=0),
    dict(output_available=False),dict(duplicate_submission=True)])
def test_submission_metadata_blocks_incomplete_command(changes):
    kwargs=dict(target_session='current',current_session='current',physical_gpu=1)
    kwargs.update(changes)
    result=metadata({'authorization':{'status':'AUTHORIZED','hapi_session':'current'},
                     'device':{'physical_index':1}},**kwargs)
    assert result['status']=='NOT_READY'


def test_unauthorized_declaration_cannot_be_promoted_by_session():
    assert metadata({'authorization':'UNAUTHORIZED'},target_session='current',
        current_session='current',physical_gpu=1)['status']=='NOT_READY'

# 旧 supervisor 多轮CLI已退役（2026-11-05）；等价的无写计划契约见 tests/test_new_cli.py。
