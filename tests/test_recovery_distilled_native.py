"""Public diagnostic dispatch and fixed first-window phase accounting."""
FROZEN_TEST_SCOPE='PORTABLE'
import numpy as np
import pytest
from native_stand.students import motion_single_screen as public
from native_stand.students import recovery_distilled_native as recovery


@pytest.mark.parametrize('version',[recovery.VERSION,recovery.STUDENT_ONLY])
def test_public_dispatch_does_not_send_small_recovery_to_motion(monkeypatch,version):
 request={'version':version}
 seen=[]
 monkeypatch.setattr(recovery,'validate',lambda r:seen.append(r) or {'explicit':'Recovery'})
 assert public.validate(request)=={'explicit':'Recovery'}
 assert seen==[request]
 monkeypatch.setattr(recovery,'worker',lambda r:17)
 assert public.worker(request)==17


def test_unknown_or_incomplete_request_fails_before_runtime_import():
 with pytest.raises(ValueError,match='schema'):
  recovery.validate({'version':recovery.VERSION,'unexpected':1})


def test_hold_statistics_use_first_fixed_window_and_exclude_reset_episode():
 n=1100
 a={'first_episode_valid':np.ones((n,1),bool),'actor_valid':np.ones((n,1,1),bool),
    'state':np.zeros((n,1,39)),'posture':np.zeros((n,1,2)),'control':np.zeros((n,1,54))}
 a['actor_valid'][:40]=False
 a['first_episode_valid'][1060:]=False
 a['state'][:,:,6]=.34;a['state'][:,:,10:12]=3
 a['state'][1060:,:,7]=100
 rows=recovery.phase_metrics(a,{'entry_index':84})
 assert rows['pre_entry']['samples']==45
 assert rows['first_fixed_hold']['samples']==975
 assert rows['first_fixed_hold']['mean_abs_vx_vy_yaw']==[0,0,0]
 assert rows['first_fixed_hold']['seconds']==9.75
