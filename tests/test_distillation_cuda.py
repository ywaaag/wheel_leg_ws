"""Explicit authorized CUDA consumer check; excluded from portable CPU runs."""

import dataclasses

import pytest
import torch
from test_public_migration import supervision_job

from native_stand.experiments.supervision import PhaseConfig
from native_stand.experiments.supervision_flow import execute

FROZEN_TEST_SCOPE = 'CUDA'


def test_cuda_supervision_round_boundary_resume_preserves_adam_rng_and_weights(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('Explicit CUDA validation needs the authorized physical GPU')
    base=supervision_job(tmp_path)
    job=dataclasses.replace(base,runtime=dataclasses.replace(base.runtime,device='cuda:0'),
                            supervision=dataclasses.replace(base.supervision,phases=(PhaseConfig('label'),PhaseConfig('fit',rounds=2))))
    def run(folder,stop):
        state,done=execute(job,folder,stop_requested=stop,on_round_complete=lambda x:None,
                           on_round_saved=lambda x:None)
        return state,done
    continuous,done=run(tmp_path/'continuous',lambda:False)
    assert done
    count=[]
    state,done=execute(job,tmp_path/'segmented',stop_requested=lambda:len(count)>=1,
                       on_round_complete=count.append,on_round_saved=lambda x:None)
    assert not done
    first=torch.load(state['latest']['path'],map_location='cpu',weights_only=False)
    assert first['round']==1 and first['optimization_device']=='cuda:0'
    assert 'cuda' in first['rng'] and first['rng']['cuda_device']==0
    state,done=run(tmp_path/'segmented',lambda:False)
    assert done
    a=torch.load(continuous['latest']['path'],map_location='cpu',weights_only=False)
    b=torch.load(state['latest']['path'],map_location='cpu',weights_only=False)
    assert a['round']==b['round']==2
    for key in a['student']:torch.testing.assert_close(a['student'][key],b['student'][key],rtol=0,atol=0)
    for key in a['optimizer']['state']:
        for name,value in a['optimizer']['state'][key].items():
            torch.testing.assert_close(value,b['optimizer']['state'][key][name],rtol=0,atol=0)
    assert torch.equal(a['rng']['cuda'],b['rng']['cuda'])
