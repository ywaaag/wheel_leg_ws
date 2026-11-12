"""Build/test the electrical control core on CPU. Default is a zero-write plan.

Does not build/flash board firmware, run a simulator, access a motor or optimize.
The resulting .so is used by the existing MuJoCo diagnostic adapter.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build(repository,output,*,extended=False):
    repo=repository.resolve();out=output.resolve()
    core=[repo/'application/chassis'/n for n in ('rl_motion31.c','rl_control.c')]
    headers=[repo/'application/chassis'/n for n in ('rl_motion31.h','rl_control.h','rl_shadow.h','rl_capture.h')]
    bridge=ROOT/'src/native_stand/students/firmware_bridge.c'
    names=('test_rl_motion31','test_rl_spring','test_rl_execution','test_rl_capture','test_rl_shadow_boundary')
    if extended:names=names+('test_rl_timing',)
    geometry=repo/'application/balance/legalgorithm.c'
    sources=core+headers+[bridge,geometry,repo/'application/balance/leg_const.hpp',repo/'application/chassis/rl_shadow.c',repo/'application/chassis/rl_capture.c']
    sources+=[repo/'tests'/f'{n}.c' for n in names]+list((repo/'tests/host').glob('*.h'))+list((repo/'tests/host').glob('*.hpp'))
    if extended:
        sources+=[repo/p for p in ('application/chassis/chassis.c',
            'application/chassis_can_link.h','application/robot_def.h',
            'modules/motor/DMmotor/dmmotor.c','modules/motor/DMmotor/dmmotor.h',
            'modules/motor/DMmotor/dm_position_pd.h','modules/motor/DJImotor/dji_motor.c',
            'modules/motor/DJImotor/dji_motor.h','modules/motor/motor_def.h',
            'tests/test_motor_driver_boundary.c','tests/run_motor_driver_host_tests.sh',
            'tests/run_chassis_snapshot_host_test.py')]
        sources+=list((repo/'tests/host_driver').glob('*'))
    identities={str(p):sha(p) for p in sources}
    git=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    out.mkdir(parents=True,exist_ok=False)
    result=dict(version='motion_firmware_cpu_validation_v1',repository=str(repo),base_commit=git,
                source_sha256=identities,status='INCOMPLETE',tests=[],simulation=0,PPO_updates=0,optimizer_steps=0,
                target_firmware_build='NOT_RUN: arm-none-eabi-gcc not available; no board flash',
                authorization='user authorized S1 electrical repair and CPU validation',
                commands=[],extended_driver_and_snapshot_checks=extended)
    started=time.monotonic()
    def run(args):
        result['commands'].append([str(x) for x in args])
        r=subprocess.run([str(x) for x in args],text=True,capture_output=True,timeout=60)
        if r.returncode:raise RuntimeError('command failed: '+' '.join(str(x) for x in args)+'\n'+r.stdout+r.stderr)
        return r
    try:
        flags=['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-I'+str(repo/'application/chassis')]
        for name in names:
            files=[repo/'tests'/f'{name}.c']+core
            extra=[]
            if name=='test_rl_capture':files.append(repo/'application/chassis/rl_capture.c')
            if name=='test_rl_shadow_boundary':
                files.append(repo/'application/chassis/rl_shadow.c');extra=['-I'+str(repo/'tests/host')]
            binary=out/name
            run(flags+extra+files+['-lm','-o',binary])
            command=[binary]+([out/'capture_synthetic.bin'] if name=='test_rl_capture' else [])
            execution=run(command)
            result['tests'].append(dict(name=name,status='PASS',stdout=execution.stdout))
            if extended and name=='test_rl_shadow_boundary':
                result['shadow_modes']=[]
                for mode in ('shadow','inactive','stale','invalid'):
                    r=run([binary,mode]);result['shadow_modes'].append(dict(mode=mode,status='PASS',stdout=r.stdout))
        if extended:
            execution=run(['sh',repo/'tests/run_motor_driver_host_tests.sh'])
            result['driver_boundary']=dict(status='PASS',stdout=execution.stdout,
                                          scope='real drivers, mocked HAL/RTOS/PID, no physical CAN')
            execution=run([sys.executable,repo/'tests/run_chassis_snapshot_host_test.py'])
            result['chassis_snapshot_boundary']=dict(status='PASS',result=json.loads(execution.stdout),
                scope='current production snapshot body; geometry and RTOS mocked; not whole-board build')
            result['chassis_output_boundary']=[]
            for enabled in ('0','1'):
                execution=run([sys.executable,repo/'tests/run_chassis_snapshot_host_test.py',
                               '--mode','output','--output-enabled',enabled])
                result['chassis_output_boundary'].append(json.loads(execution.stdout))
            execution=run([sys.executable,repo/'tests/run_chassis_snapshot_host_test.py','--mode','commands'])
            result['chassis_command_boundary']=json.loads(execution.stdout)
        library=out/'motion_firmware.so'
        run(flags+['-shared','-fPIC','-DRL_KINEMATICS_HOST','-I'+str(repo/'application/balance')]+core+[geometry,bridge,'-lm','-o',library])
        lib=ctypes.CDLL(str(library))
        if lib.motion_fw_abi_version()!=1:raise ValueError('host ABI differs')
        result.update(status='CPU_PASS',library=str(library),library_sha256=sha(library),
                      capture_record_bytes=lib.motion_fw_capture_record_bytes(),
                      capture_buffer_bytes=lib.motion_fw_capture_buffer_bytes(),
                      hardware_flash_ready=False)
    except BaseException as e:
        result['status']='FAILED';result['error']=type(e).__name__+': '+str(e)
        raise
    finally:
        result['source_unchanged_during_build']=all(sha(p)==v for p,v in identities.items())
        if not result['source_unchanged_during_build']:result['status']='FAILED_SOURCE_CHANGED'
        result['wall_s']=time.monotonic()-started
        result['artifact_bytes_before_manifest']=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        target=out/'validation.json';temp=target.with_suffix('.tmp')
        temp.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(target)
    if result['status']!='CPU_PASS':raise RuntimeError(result['status'])
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,default=Path('/home/kellen/combat-dm-balance-main'))
    p.add_argument('--output-dir',type=Path);p.add_argument('--build',action='store_true')
    p.add_argument('--extended',action='store_true',help='Also run real motor drivers, chassis snapshot and timing regressions')
    a=p.parse_args()
    if not a.build:
        print(json.dumps(dict(repository=str(a.repository),tests=6 if a.extended else 5,
                              extended=a.extended,simulation=0,PPO=0,files_written=0)));return
    if a.output_dir is None:p.error('--build needs independent --output-dir')
    result=build(a.repository,a.output_dir,extended=a.extended)
    print(json.dumps({k:result[k] for k in ('status','wall_s','library','capture_record_bytes','capture_buffer_bytes','hardware_flash_ready')},indent=2))

if __name__=='__main__':main()
