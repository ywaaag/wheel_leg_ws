"""Named telemetry indices must override physical CSV column order."""
FROZEN_TEST_SCOPE='PORTABLE'
import csv,importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
 p=Path(__file__).resolve().parents[1]/'scripts/replay_motion31_shadow_csv.py'
 spec=importlib.util.spec_from_file_location('shadow_replay',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def make(path):
 fields=['Index','Time']+[f'(g_rl_shadow_debug).observation[{i}]' for i in list(range(22))+[23,22,24]]+[f'(g_rl_shadow_debug).mean_action[{i}]' for i in range(6)]
 r=dict.fromkeys(fields,0);r['(g_rl_shadow_debug).observation[22]']=.125;r['(g_rl_shadow_debug).observation[23]']=-.25;r['(g_rl_shadow_debug).observation[24]']=.68
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerow(r)
 return fields,r


def test_real_csv_reader_uses_named_indices_and_declares_missing_skill_flags(tmp_path):
 path=tmp_path/'log.csv';make(path)
 rows,x,action,t,columns=module().read_csv(path)
 assert x[0,22]==.125 and x[0,23]==-.25
 np.testing.assert_array_equal(x[0,25:31],[1,0,0,1,0,0])
 assert set(columns['observation'])==set(range(25)) and action.shape==(1,6)


def test_nonfinite_and_duplicate_time_not_silently_repaired(tmp_path):
 path=tmp_path/'log.csv';fields,row=make(path)
 with path.open('a',newline='') as f:csv.DictWriter(f,fields).writerow(row)
 with pytest.raises(ValueError,match='timestamps'):module().read_csv(path)
 fields,row=make(path);row['(g_rl_shadow_debug).observation[6]']='NaN'
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerow(row)
 with pytest.raises(ValueError,match='nonfinite'):module().read_csv(path)
