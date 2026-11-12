"""CPU-build the actual electrical drivers and core with causal packet adapters.

No board build, flashing, simulation, training or writes to the electrical repo.
Default is a zero-write plan; --build needs a new independent output directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_body(source,signature):
    start=source.index(signature);opening=source.index('{',start);end=opening+1;depth=1
    while depth:
        if source[end]=='{':depth+=1
        elif source[end]=='}':depth-=1
        end+=1
    return source[start:end]


def build(repository,output):
    from validate_motion_firmware import build as build_core
    repo=Path(repository).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();commands=[];result=dict(status='INCOMPLETE',simulation=0,optimizer_steps=0,
        version='motion31_firmware_packet_sampling_v1',repository=str(repo),commands=commands,
        hardware_flash_ready=False,scope='real DM/DJI decoding, packet filters, MotorControlTask body and control core; HAL/RTOS mocked')
    files=[ROOT/'src/native_stand/students'/n for n in ('firmware_sampling_bridge.c','firmware_bridge.c')]
    relative=['application/chassis/chassis.c','application/chassis/rl_motion31.c','application/chassis/rl_motion31.h',
        'application/chassis/rl_control.c','application/chassis/rl_control.h','application/chassis/rl_shadow.h',
        'application/chassis/rl_capture.h','application/robot_task.h','Core/Inc/FreeRTOSConfig.h',
        'application/balance/legalgorithm.c','application/balance/leg_const.hpp',
        'modules/motor/DMmotor/dmmotor.c','modules/motor/DMmotor/dmmotor.h','modules/motor/DMmotor/dm_position_pd.h',
        'modules/motor/DJImotor/dji_motor.c','modules/motor/DJImotor/dji_motor.h','modules/motor/motor_task.c',
        'modules/motor/motor_def.h','modules/general_def.h']
    files += [repo/p for p in relative]+list((repo/'tests/host_driver').glob('*'))+[Path(__file__).resolve()]
    identities={str(p):sha(p) for p in files if p.is_file()};result['source_sha256']=identities
    def run(args):
        commands.append([str(x) for x in args]);r=subprocess.run(commands[-1],text=True,capture_output=True,timeout=90)
        if r.returncode:raise RuntimeError('compile failed\n'+r.stdout+r.stderr)
        return r
    try:
        baseline=build_core(repo,out/'core',extended=True)
        result['core_validation']=str(out/'core/validation.json');result['base_commit']=baseline['base_commit']
        chassis=(repo/'application/chassis/chassis.c').read_text()
        names=('LF_ENCODE_OFFSET','LB_ENCODE_OFFSET','RF_ENCODE_OFFSET','RB_ENCODE_OFFSET','REDUCE_RATIO')
        constants=[]
        for name in names:
            found=re.findall(r'^#define\s+'+name+r'\s+(.+)$',chassis,re.M)
            if len(found)!=1:raise ValueError('exactly one source constant required: '+name)
            constants.append('#define '+name+' ('+found[0].split('//')[0].strip()+')')
        (out/'sampling_source_constants.inc').write_text('\n'.join(constants)+'\n')
        task=function_body((repo/'modules/motor/motor_task.c').read_text(),'void MotorControlTask()')
        (out/'sampling_motor_task.inc').write_text(task+'\n')
        library=out/'motion_sampling.so'
        include=[out,repo/'tests/host_driver',repo/'application/chassis',repo/'application/balance',
                 repo/'modules/motor/DMmotor',repo/'modules/motor/DJImotor',repo/'modules/motor',
                 repo/'modules/daemon',repo/'modules']
        sources=[ROOT/'src/native_stand/students/firmware_sampling_bridge.c']+[repo/p for p in (
            'application/chassis/rl_motion31.c','application/chassis/rl_control.c',
            'application/balance/legalgorithm.c','modules/motor/DMmotor/dmmotor.c','modules/motor/DJImotor/dji_motor.c')]
        run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC','-DRL_KINEMATICS_HOST',
             *['-I'+str(p) for p in include],*sources,'-lm','-o',library])
        result.update(status='CPU_PASS',library=str(library),library_sha256=sha(library),
            extracted_function_sha256=hashlib.sha256(task.encode()).hexdigest(),
            generated_sha256={str(out/n):sha(out/n) for n in ('sampling_source_constants.inc','sampling_motor_task.inc')},
            source_facts=dict(policy_period_ms=10,pd_period_ms=5,motor_task_delay_ticks=1,
                leg_send_every_motor_task_calls=2,wheel_send_every_motor_task_calls=1,
                leg_q_bits=16,leg_q_max_rad=6.283,leg_dq_bits=12,leg_dq_max_rad_s=45,
                wheel_speed_alpha_per_received_packet=.85,driver_torque_bits=12,leg_torque_cap_nm=40,
                feedback_guard_ms=24,explicit_fixed_feedback_delay_ms=None),
            unknown=['actual flashed build','motor internal velocity estimator','packet production cadence',
                     'physical sample time before receipt','CAN arbitration','scheduler jitter','actual torque establishment'])
    except BaseException as exc:
        result.update(status='FAILED',failure=type(exc).__name__+': '+str(exc));raise
    finally:
        result['source_unchanged']=all(sha(p)==v for p,v in identities.items())
        result['wall_s']=time.monotonic()-started
        (out/'validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,default=Path('/home/kellen/combat-dm-balance-main'))
    p.add_argument('--output-dir',type=Path);p.add_argument('--build',action='store_true');a=p.parse_args()
    if not a.build:print(json.dumps(dict(version='motion31_firmware_packet_sampling_v1',files_written=0,physics_steps=0)));return
    if a.output_dir is None:p.error('--build requires new --output-dir')
    r=build(a.repository,a.output_dir);print(json.dumps({k:r[k] for k in ('status','library','wall_s','source_unchanged')},indent=2))


if __name__=='__main__':main()
