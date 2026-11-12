"""Decode the firmware's frozen RAM capture; no debugger/device access.

Use --synthetic only for a host-generated fixture. Missing policy/PD quantities
are exported empty according to valid_mask, never zero-filled as measurements.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HEADER=('schema_version','record_bytes','capacity','count','write_index','armed','triggered','frozen',
        'remaining','post_records','records_written','owner_previous')
SCALARS=('frame_begin','sequence','tick_ms','owner','fault','control_epoch','policy_sequence',
         'policy_snapshot_tick_ms','policy_output_tick_ms','adopted_sequence','adopted_tick_ms',
         'previous_source','previous_sequence','valid_mask','feedback_valid_mask')
ARRAYS=(('feedback_receive_tick_ms',6,'<u4'),('imu_update_tick_ms',1,'<u4'),
        ('observation',31,'<f4'),('mean_action',6,'<f4'),('requested_target',6,'<f4'),('adopted_target',6,'<f4'),
        ('pd_q',4,'<f4'),('pd_dq',6,'<f4'),('pd_request_nm',6,'<f4'),('pd_limited_nm',6,'<f4'),
        ('spring_motor_nm',4,'<f4'),('final_motor_command_nm',4,'<f4'),('wheel_command_raw',2,'<f4'),
        ('driver_reported_leg_torque_motor_nm',4,'<f4'),('driver_reported_wheel_current_raw',2,'<f4'))
DTYPE=np.dtype([(k,'<u4') for k in SCALARS]+[(k,dtype,(n,)) if n>1 else (k,dtype) for k,n,dtype in ARRAYS]+[('frame_end','<u4')])

def decode(path):
    raw=Path(path).read_bytes()
    if len(raw)<4*len(HEADER):raise ValueError('capture header missing')
    header=dict(zip(HEADER,map(int,np.frombuffer(raw,dtype='<u4',count=len(HEADER)))))
    if header['schema_version']!=1 or header['record_bytes']!=DTYPE.itemsize:raise ValueError('unsupported schema/record size')
    cap,count,index=header['capacity'],header['count'],header['write_index']
    if cap<1 or count>cap or index>=cap or len(raw)!=4*len(HEADER)+cap*DTYPE.itemsize:raise ValueError('invalid capture extent')
    if not header['frozen']:raise ValueError('capture is not frozen; refuse a live/torn RAM image')
    records=np.frombuffer(raw,dtype=DTYPE,offset=4*len(HEADER))
    records=records[(np.arange(count)+index-count)%cap].copy()
    if np.any(records['frame_begin']!=records['frame_end']) or np.any(records['frame_begin']&1):raise ValueError('in-progress/torn capture frame')
    seq=records['sequence'].astype(np.int64)
    if len(seq)>1 and np.any(np.diff(seq)%(2**32)!=1):raise ValueError('capture sequence gap/reordering')
    t=records['tick_ms'].astype(np.int64)
    dt=np.diff(t)%(2**32)
    if len(t)>1 and np.any((dt==0)|(dt>1000)):raise ValueError('nonmonotonic or >1s discontinuous capture time')
    return header,records

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--synthetic',action='store_true');a=p.parse_args()
    h,records=decode(a.input);a.output_dir.mkdir(parents=True,exist_ok=False)
    columns=list(SCALARS[1:])+[k if n==1 else f'{k}[{i}]' for k,n,_ in ARRAYS for i in range(n)]
    policy={'observation','mean_action','requested_target'}
    pd={'pd_request_nm','pd_limited_nm','spring_motor_nm','adopted_target'}
    with (a.output_dir/'capture.csv').open('w') as f:
        w=csv.writer(f);w.writerow(columns)
        for r in records:
            row=[int(r[k]) for k in SCALARS[1:]]
            for k,n,dtype in ARRAYS:
                valid=not((k in policy and not int(r['valid_mask'])&1) or (k in pd and not int(r['valid_mask'])&2))
                value=np.atleast_1d(r[k]);row.extend(value.tolist() if valid else ['']*n)
            w.writerow(row)
    manifest=dict(version='motion_capture_decode_v1',input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
                  input=str(a.input.resolve()),header=h,rows=len(records),synthetic=a.synthetic,
                  firmware_build='UNKNOWN unless supplied separately by electrical owner',
                  independently_calibrated_torque=False,current_in_amperes=False,
                  semantics='native model arrays; motor arrays lf/lb/rf/rb; wheel driver raw is physical L/R; timestamps are receipt/update times, not sensor acquisition')
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(rows=len(records),synthetic=a.synthetic,output=str(a.output_dir))))

if __name__=='__main__':main()
