"""Review frozen existing capture-v1 using its real policy/PD time axes.

No device access, physics or network replacement. Derive P/D from declared
40/2/.2; do not invent missing raw motor velocity or sensor acquisition times.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from decode_motion_capture import decode

LEGS=[0,1,3,4]
WHEELS=[2,5]


def age32(now,then):
    age=(int(now)-int(then)) & 0xffffffff
    if age>=0x80000000:raise ValueError('future timestamp or ambiguous clock span')
    return age


def review(path):
    header,records=decode(path)
    rows=[];policy_seen=set();failures=[];raw_errors=[]
    for index,r in enumerate(records):
        valid=int(r['valid_mask']);tick=int(r['tick_ms']);row=dict(
            row=index,frame_sequence=int(r['sequence']),pd_tick_ms=tick,
            owner=int(r['owner']),fault=int(r['fault']),control_epoch=int(r['control_epoch']),
            adopted_sequence=int(r['adopted_sequence']),adopted_tick_ms=int(r['adopted_tick_ms']),
            policy_valid=bool(valid&1),PD_valid=bool(valid&2),final_command_valid=bool(valid&4))
        for bit,fields in ((1,('observation','mean_action','requested_target')),
                (2,('pd_q','pd_dq','adopted_target','pd_request_nm','pd_limited_nm','spring_motor_nm')),
                (4,('final_motor_command_nm','wheel_command_raw','driver_reported_leg_torque_motor_nm','driver_reported_wheel_current_raw'))):
            if valid&bit and any(not np.isfinite(r[field]).all() for field in fields):
                raise ValueError(f'nonfinite valid capture quantities at row{index}, mask group{bit}')
        for k in ('policy_snapshot_tick_ms','policy_output_tick_ms','policy_sequence','previous_sequence'):
            row[k]=int(r[k]) if valid&1 else None
        if valid&1:
            key=(int(r['control_epoch']),int(r['policy_sequence']))
            row['first_record_for_policy']=key not in policy_seen;policy_seen.add(key)
            try:
                row['snapshot_to_output_ms']=age32(r['policy_output_tick_ms'],r['policy_snapshot_tick_ms'])
                row['output_to_this_PD_record_ms']=age32(tick,r['policy_output_tick_ms'])
            except ValueError as e:failures.append(dict(row=index,error=str(e)))
            for ch in range(6):row[f'Actor_dq_ch{ch}_rad_s']=float(r['observation'][10+ch])*20
            for ch in range(31):row[f'observation{ch}']=float(r['observation'][ch])
            for ch in range(6):
                row[f'mean{ch}']=float(r['mean_action'][ch])
                row[f'requested_ch{ch}']=float(r['requested_target'][ch])
            row['previous_source']=int(r['previous_source'])
            row['vx_command_m_s']=float(r['observation'][22])*4
            row['yaw_command_rad_s']=float(r['observation'][23])*4
            row['height_command_m']=float(r['observation'][24])*.5
        if valid&2:
            try:row['adopted_age_ms']=age32(tick,r['adopted_tick_ms'])
            except ValueError as e:failures.append(dict(row=index,error=str(e)))
            q=np.asarray(r['pd_q'],float);dq=np.asarray(r['pd_dq'],float);target=np.asarray(r['adopted_target'],float)
            P=np.zeros(6);D=np.zeros(6)
            P[LEGS]=40*(target[LEGS]-q);D[LEGS]=-2*dq[LEGS]
            P[WHEELS]=.2*target[WHEELS];D[WHEELS]=-.2*dq[WHEELS]
            error=float(abs(P+D-r['pd_request_nm']).max())
            if not np.isfinite(error):raise ValueError('nonfinite PD quantities in valid row')
            raw_errors.append(error)
            if error>1e-4:failures.append(dict(row=index,error='40/2/.2 PD reconstruction differs',max_Nm=error))
            for ch in range(6):
                row[f'PD_dq_ch{ch}_rad_s']=float(dq[ch]);row[f'P_ch{ch}_Nm']=float(P[ch]);row[f'D_ch{ch}_Nm']=float(D[ch])
                row[f'PD_requested_ch{ch}_Nm']=float(r['pd_request_nm'][ch])
                row[f'PD_limited_ch{ch}_Nm']=float(r['pd_limited_nm'][ch])
                row[f'adopted_ch{ch}']=float(target[ch])
        for ch in range(6):
            row[f'PD_feedback_receive_age_ch{ch}_ms']=None
            if int(r['feedback_valid_mask']) & (1<<ch):
                try:row[f'PD_feedback_receive_age_ch{ch}_ms']=age32(tick,r['feedback_receive_tick_ms'][ch])
                except ValueError as e:failures.append(dict(row=index,error=str(e),channel=ch))
        if valid&4:
            for ch in range(4):
                row[f'final_motor_{ch}_Nm']=float(r['final_motor_command_nm'][ch])
                row[f'driver_reported_motor_{ch}_Nm']=float(r['driver_reported_leg_torque_motor_nm'][ch])
            for ch in range(2):
                row[f'wheel_command_raw_{ch}']=float(r['wheel_command_raw'][ch])
                row[f'driver_reported_wheel_current_raw_{ch}']=float(r['driver_reported_wheel_current_raw'][ch])
        if valid&2:
            for ch in range(4):row[f'spring_motor_{ch}_Nm']=float(r['spring_motor_nm'][ch])
        rows.append(row)
    result=dict(version='motion_capture_sync_review_v1',input=str(Path(path).resolve()),
        input_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),header=header,rows=len(rows),
        policy_count=len(policy_seen),RL_PD_rows=len(raw_errors),
        max_declared_PD_reconstruction_error_Nm=max(raw_errors) if raw_errors else None,
        all_recorded_policy_commands_zero=all(abs(row['vx_command_m_s'])<=1e-6 and
            abs(row['yaw_command_rad_s'])<=1e-6 for row in rows if row['policy_valid']) if policy_seen else None,
        validation_errors=failures,CPU_consistent=not failures,
        missing=['actual flashed build and model identity','raw motor dq before mapping',
            'Actor-specific feedback receipt/sample times','actual sensor acquisition times',
            'actual remote target acceptance and independently calibrated motor torque'],
        semantics='Actor values use policy_snapshot/output time; PD values use record PD time; receipt age is NOT sensor sample age',
        physics_steps=0,optimizer_steps=0,hardware_certified=False)
    return result,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--synthetic',action='store_true')
    args=parser.parse_args();result,rows=review(args.input)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    result['synthetic']=args.synthetic
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (args.output_dir/'sync.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    (args.output_dir/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('rows','policy_count','RL_PD_rows','CPU_consistent','synthetic')},ensure_ascii=False))
    if result['validation_errors']:raise SystemExit(2)


if __name__=='__main__':main()
