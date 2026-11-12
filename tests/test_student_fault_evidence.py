"""Fault rows outside the trace preview and legacy incomplete evidence."""
FROZEN_TEST_SCOPE='PORTABLE'
import torch
from native_stand.command_v6.domain.trace_schema import STATE
from native_stand.students.fault_evidence import capture,describe

def test_fault_outside_first_eight_and_missing_evidence():
 state=torch.zeros(64,39);reward=torch.zeros(64)
 state[51,STATE['runtime_fault']]=1
 data=capture(state,reward,{})
 assert data['env_ids'].tolist()==[51]
 report=describe(data)
 assert not report['diagnostics_complete']
 assert 'component_joint_outside' in report['missing_evidence']
 assert report['trigger_components']=={}
 # Nonfinite reward is a separate trigger, including a different environment.
 reward[63]=float('nan')
 data=capture(state,reward,{})
 assert data['env_ids'].tolist()==[51,63]
 assert data['nonfinite_reward'].tolist()==[False,True]
