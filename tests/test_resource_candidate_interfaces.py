import json
import pytest

from native_stand.command_v6.domain.resource_policy import resolve_resources
from native_stand.command_v6.tools.health_check import submission_health
from native_stand.command_v6.completion import choose
from test_completion import candidate,EXPECTED


@pytest.mark.parametrize('name',['/usr/NX/bin/nxnode.bin','python','unclassified_owner'])
def test_exclusive_resource_mode_never_uses_process_name_whitelist(name):
    report={'findings':[],'gpu':{'status':'OK','devices':[{'index':1,'uuid':'GPU-1'}],
        'compute_processes':[{'pid':7,'gpu_uuid':'GPU-1','process_name':name}]}}
    with pytest.raises(ValueError,match='RESOURCE_OCCUPIED'):
        submission_health(report,current_session='now',target_session='now',physical_gpu=1)
    policy=resolve_resources({'resources':{'mode':'shared','sharing_authorized':True,
        'authorization_note':'explicit user permission for concurrent use'}})
    result=submission_health(report,current_session='now',target_session='now',physical_gpu=1,resource_policy=policy)
    assert result['resource_policy']==policy and result['gpu_snapshot']==report['gpu']
    assert not result['preemption']


@pytest.fixture
def declaration():
    """声明了shared模式但缺少非空授权说明。"""
    return {'mode':'shared','sharing_authorized':True}


def test_shared_resources_require_valid_explicit_consent(declaration):
    with pytest.raises(ValueError):resolve_resources({'resources':declaration})
    assert resolve_resources({'resources':{**declaration,'authorization_note':'explicit user permission'}})['mode']=='shared'


def test_resource_policy_does_not_bypass_authorization_identity_or_unknown_gpu():
    policy=resolve_resources({},mode='shared',sharing_authorization='user permits sharing')
    for report in ({'findings':[],'gpu':{'status':'UNKNOWN'}},
        {'findings':[{'reason_code':'IDENTITY_ERROR'}],'gpu':{'status':'OK','devices':[{'index':1,'uuid':'GPU-1'}]}}):
        with pytest.raises(ValueError):submission_health(report,current_session='now',target_session='now',physical_gpu=1,resource_policy=policy)
    # HAPI会话不一致只记录告警，不再阻断（可选通知通道）。
    warned=submission_health({'findings':[]},current_session='now',target_session='old',resource_policy=policy)
    assert warned['warnings']


def test_source_best_overall_is_not_best_new_and_latest_can_be_unreviewed():
    source=candidate('source',drift=.01,role='source')
    new=candidate('new',drift=.15)
    latest={'path':'not_evaluated','sha256':'not_evaluated'}
    result=choose([source,new],latest,EXPECTED)
    assert result['best_overall_candidate']['checkpoint']=='source'
    assert result['best_new_evaluated_candidate']['checkpoint']=='new'
    assert result['best_observed_candidate']['checkpoint']=='source'
    assert result['accepted_candidate']['checkpoint']=='source'
    assert result['model_roles']['latest']==latest


def test_source_copied_to_candidate_path_not_new_result():
    source=candidate('source',drift=.01,role='source')
    copied=candidate('copy',drift=.01)
    copied['checkpoint_sha256']=EXPECTED['source_sha256']
    result=choose([copied],None,EXPECTED)
    assert result['best_new_evaluated_candidate'] is None
    assert result['best_overall_candidate']['checkpoint']=='copy'

# 旧 supervisor 多轮CLI已退役（2026-11-05）；等价的无写计划契约见 tests/test_new_cli.py。
