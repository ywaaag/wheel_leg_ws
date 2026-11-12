"""Behavior checks for the actual capture ABI and two independent time axes."""
FROZEN_TEST_SCOPE='PORTABLE'
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import decode_motion_capture as decoder
import review_motion_capture_sync as tool


def fixture(path):
    a=np.zeros(2,dtype=decoder.DTYPE)
    for j in range(2):
        a['sequence'][j]=j+1;a['frame_begin'][j]=a['frame_end'][j]=2*(j+1)
        a['tick_ms'][j]=105+5*j;a['feedback_receive_tick_ms'][j]=103+5*j
        a['feedback_valid_mask'][j]=63;a['owner'][j]=1
    a['valid_mask'][0]=7;a['policy_sequence'][0]=1
    a['policy_snapshot_tick_ms'][0]=100;a['policy_output_tick_ms'][0]=102
    a['observation'][0,10:16]=np.arange(6)*.05
    a['pd_q'][0]=[.1,.2,-.1,.2];a['pd_dq'][0]=np.arange(6)*2
    a['adopted_target'][0]=[.2,.3,4.,-.2,.3,-5.]
    P=np.zeros(6);D=np.zeros(6);legs=[0,1,3,4]
    P[legs]=40*(a['adopted_target'][0,legs]-a['pd_q'][0]);D[legs]=-2*a['pd_dq'][0,legs]
    P[[2,5]]=.2*a['adopted_target'][0,[2,5]];D[[2,5]]=-.2*a['pd_dq'][0,[2,5]]
    a['pd_request_nm'][0]=P+D
    a['valid_mask'][1]=4 # No policy/PD: stale arrays must not become valid values.
    a['observation'][1,10:16]=99
    head=dict(schema_version=1,record_bytes=decoder.DTYPE.itemsize,capacity=2,count=2,write_index=0,
        armed=1,triggered=1,frozen=1,remaining=0,post_records=1,records_written=2,owner_previous=1)
    path.write_bytes(np.array([head[k] for k in decoder.HEADER],dtype='<u4').tobytes()+a.tobytes())


def test_PD_and_Actor_velocities_are_not_mixed(tmp_path):
    p=tmp_path/'capture.bin';fixture(p);result,rows=tool.review(p)
    assert result['CPU_consistent'] and result['RL_PD_rows']==1
    assert rows[0]['Actor_dq_ch1_rad_s']==pytest.approx(1)
    assert rows[0]['PD_dq_ch1_rad_s']==2
    assert rows[0]['policy_snapshot_tick_ms']==100 and rows[0]['pd_tick_ms']==105
    assert rows[0]['PD_feedback_receive_age_ch1_ms']==2
    assert 'Actor_dq_ch1_rad_s' not in rows[1] and 'P_ch0_Nm' not in rows[1]


def test_clock_wrap_is_supported_but_future_is_rejected():
    assert tool.age32(2,0xfffffffe)==4
    with pytest.raises(ValueError):tool.age32(5,6)


def test_torn_actual_capture_still_refused(tmp_path):
    p=tmp_path/'capture.bin';fixture(p)
    a=bytearray(p.read_bytes());start=4*len(decoder.HEADER)
    a[start:start+4]=(3).to_bytes(4,'little');p.write_bytes(a)
    with pytest.raises(ValueError,match='torn'):tool.review(p)


def test_nonfinite_Actor_data_cannot_be_exported_as_valid(tmp_path):
    p=tmp_path/'capture.bin';fixture(p)
    raw=bytearray(p.read_bytes());offset=4*len(decoder.HEADER)
    a=np.ndarray((2,),dtype=decoder.DTYPE,buffer=raw,offset=offset)
    a['observation'][0,11]=np.nan;p.write_bytes(raw)
    with pytest.raises(ValueError,match='nonfinite valid'):tool.review(p)


def test_current_raw_and_requested_targets_keep_separate_meaning(tmp_path):
    p=tmp_path/'capture.bin';fixture(p)
    raw=bytearray(p.read_bytes());a=np.ndarray((2,),dtype=decoder.DTYPE,buffer=raw,offset=4*len(decoder.HEADER))
    a['wheel_command_raw'][0]=[100,200];a['driver_reported_wheel_current_raw'][0]=[90,180]
    a['requested_target'][0,0]=.31;p.write_bytes(raw)
    result,rows=tool.review(p)
    assert result['all_recorded_policy_commands_zero']
    assert rows[0]['wheel_command_raw_0']==100 and rows[0]['driver_reported_wheel_current_raw_0']==90
    assert rows[0]['requested_ch0']==pytest.approx(.31) and rows[0]['adopted_ch0']==pytest.approx(.2)
