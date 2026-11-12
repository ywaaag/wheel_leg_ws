FROZEN_TEST_SCOPE='PORTABLE'
from native_stand.command_v6.stand_chain import source_engineering_error


def test_source_timeout_retains_actual_cause():
 row={'status':'STOPPED_BOUNDED','terminal':{'budget_interruption':{'exception':"TimeoutError('Student evaluation wall budget exhausted')"}},'progress':{}}
 reason=source_engineering_error(row)
 assert reason.startswith('SOURCE_EVALUATION_BUDGET_EXHAUSTED:')
 assert 'wall budget exhausted' in reason and 'not a capability failure' in reason


def test_behavior_failure_does_not_block_valid_source_wiring():
 assert source_engineering_error({'status':'COMPLETED','progress':{'engineering_valid':True,'accepted':False}}) is None


def test_missing_and_runtime_failures_distinct():
 assert source_engineering_error({}).startswith('SOURCE_EVALUATION_MISSING_EVIDENCE:')
 assert source_engineering_error({'terminal':{'failure':'nonfinite'}}).startswith('SOURCE_EVALUATION_RUNTIME_FAILURE:')
