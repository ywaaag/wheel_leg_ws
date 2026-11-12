FROZEN_TEST_SCOPE='PORTABLE'
import math
import pytest
from native_stand.command_v6.domain.execution_limits import ExecutionLimits,validate_budget
from native_stand.students.motion_quick_evaluation import parent_deadline


def spec():
 return {'ppo':{'envs':4096,'steps_per_env':128},'budget':{'formal_updates':2000,'smoke_updates':0,'formal_wall_s':None,'chain_wall_s':None,'sample_limit':2000*4096*128},'motion_large_continuation':{'execution_policy':'motion31_unlimited_wall_authorized_retry_v1'}}


def test_null_wall_has_real_update_bound_and_no_monitor_timeout():
 s=spec();assert validate_budget(s)['updates']==2000
 limits=ExecutionLimits.resolve(s,'formal');assert limits.wall_s is None
 assert math.isinf(limits.deadline(100)) and limits.monitor_bound() is None
 assert limits.samples==1048576000


def test_generic_update_only_contract_remains_compatible():
 s=spec();s.pop('motion_large_continuation')
 assert validate_budget(s)['updates']==2000
 s['budget']['chain_wall_s']=-1
 with pytest.raises(ValueError):validate_budget(s)


def test_quick_keeps_explicit_global_cap_and_finite_legacy_wall():
 s=spec();m={'submitted_unix_s':100}
 assert math.isinf(parent_deadline(s,m,float('inf')))
 m['global_deadline_unix_s']=500;assert parent_deadline(s,m,float('inf'))==500
 s['budget']['chain_wall_s']=200;assert parent_deadline(s,m,999)==300
