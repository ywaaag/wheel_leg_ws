import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1]/'scripts/review_recovery_entry_windows.py'
s=importlib.util.spec_from_file_location('entry_windows',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_force_saturation_is_separate_from_action_and_mechanical_clipping():
 a={'state':np.zeros((3,1,39)),'posture':np.zeros((3,1,10)), 'control':np.zeros((3,1,54)),
 'joint_q_pre':np.zeros((3,1,6)),'raw_action':np.ones((3,1,6))*.2,'mean':np.ones((3,1,6))*.2}
 a['state'][:,:,17]=1;a['control'][:,:,43]=5.5
 r=m.measure(a,0,np.arange(3),list(range(6)))
 assert r['saturation_any']==1 and r['channel_torque_at_limit'][-1]==1
 assert r['raw_action_outside_envelope']==[0]*6
 assert r['physical_mechanical_clip_rad_abs_mean']==[0]*4
