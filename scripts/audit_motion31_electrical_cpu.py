"""Motion500 CPU-only interface/weight audit; no simulator or optimizer.

Default is a zero-write plan. --run records small independent evidence. The
STM32-only AI runtime and the actually flashed build remain unverified.
"""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import resource
import struct
import subprocess
import tempfile
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODEL_SHA='5b67781bdaa86b666683e315f685a0f19c45fee918da70439ea3bb43d0fbb03a'
F=lambda n:C.c_float*n
FP=C.POINTER(C.c_float)


class Snapshot(C.Structure):
    _fields_=[('gyro',F(3)),('gravity',F(3)),('q',F(4)),('dq',F(6)),
              ('previous',F(6)),('previous_valid',C.c_uint8),('lqr_leg',F(4)),
              ('lqr_wheel',F(2)),('lqr_valid',C.c_uint8),
              ('vx',C.c_float),('yaw',C.c_float),('height',C.c_float)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args):
    start=time.monotonic()
    r=subprocess.run(list(map(str,args)),text=True,capture_output=True,timeout=60)
    return dict(command=list(map(str,args)),returncode=r.returncode,
                stdout=r.stdout,stderr=r.stderr,wall_s=time.monotonic()-start)


def checked(args):
    r=command(args)
    if r['returncode']:raise RuntimeError(json.dumps(r,ensure_ascii=False))
    return r


def math_library(source,repo,temp,name):
    output=temp/(name+'.so')
    checked(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
             '-I'+str(repo/'application/chassis'),source,'-lm','-o',output])
    lib=C.CDLL(str(output))
    lib.RLMotion31BuildObservation.argtypes=[C.POINTER(Snapshot),FP]
    lib.RLMotion31MapLqrAttitude.argtypes=[C.POINTER(Snapshot)]+[C.c_float]*5
    lib.RLMotion31DecodeOutput.argtypes=[FP,FP]
    lib.RLMotion31MapPolicyTargets.argtypes=[FP,FP,FP]
    lib.RLMotion31MapPolicyLegTorqueToMotorOrder.argtypes=[FP,FP]
    lib.RLMotion31MapPolicyWheelTorqueToForwardOrder.argtypes=[FP,FP]
    lib.RLMotion31ComputePdTorque.argtypes=[C.POINTER(Snapshot),FP,FP,FP]
    return lib


def quaternion_gravity(lib):
    """QEKF uses sensor X Euler as Pitch, sensor Y Euler as Roll."""
    worst=0.;count=0
    for x in np.deg2rad(np.linspace(-45,45,7)):
        for y in np.deg2rad(np.linspace(-45,45,7)):
            for z in (0.,.8,-1.7):
                cx,sx=np.cos(x),np.sin(x);cy,sy=np.cos(y),np.sin(y);cz,sz=np.cos(z),np.sin(z)
                rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
                ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
                rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
                down=(rz@ry@rx).T@np.array([0.,0.,-1.])
                expected=down[[0,2,1]]*np.array([-1.,1.,1.])
                sample=Snapshot(height=.34)
                lib.RLMotion31MapLqrAttitude(C.byref(sample),-y,x,.2,.7,.4)
                worst=max(worst,float(abs(np.array(sample.gravity)-expected).max()));count+=1
    return dict(samples=count,max_abs=worst,
                scope='sensor orientation mathematics; physical IMU mounting remains uncalibrated')


def interface_math(lib):
    import torch
    from native_stand.students.mcu_interface import encode
    gen=np.random.default_rng(20261026);n=256
    gyro=gen.uniform(-1,1,(n,3)).astype('float32')
    gravity=gen.normal(size=(n,3));gravity=(gravity/np.linalg.norm(gravity,axis=1)[:,None]).astype('float32')
    q=gen.uniform(-.9,.9,(n,6)).astype('float32');dq=gen.uniform(-5,5,(n,6)).astype('float32')
    previous=gen.uniform(-.8,.8,(n,6)).astype('float32');previous[:,[2,5]]*=20
    commands=gen.uniform(-2,2,(n,3)).astype('float32');commands[:,2]=.34
    obs=encode(*map(torch.from_numpy,(gyro,gravity,q,dq,previous[:,[0,1,3,4]],previous[:,[2,5]])),
               torch.zeros(n,6),'Motion',torch.from_numpy(commands)).numpy()
    maximum=dict(observation=0.,decode=0.,pd_request=0.,angle_roundtrip=0.,torque_power_mapping=0.)
    for i in range(n):
        sample=Snapshot(gyro=F(3)(*gyro[i]),gravity=F(3)(*gravity[i]),q=F(4)(*q[i,[0,1,3,4]]),
                        dq=F(6)(*dq[i]),previous=F(6)(*previous[i]),previous_valid=1,
                        vx=float(commands[i,0]),yaw=float(commands[i,1]),height=.34)
        out=F(31)();assert lib.RLMotion31BuildObservation(C.byref(sample),out)==0
        maximum['observation']=max(maximum['observation'],float(abs(np.array(out)-obs[i]).max()))
        raw=gen.uniform(-1.2,1.2,6).astype('float32');target=F(6)()
        assert lib.RLMotion31DecodeOutput(F(6)(*raw),target)==0
        expected=np.clip(raw,-1,1)*np.array([1.5,1.5,100,1.5,1.5,100],np.float32)
        maximum['decode']=max(maximum['decode'],float(abs(np.array(target)-expected).max()))
        req=F(6)();lim=F(6)();assert lib.RLMotion31ComputePdTorque(C.byref(sample),target,req,lim)==0
        request=(expected-q[i])*40-2*dq[i]
        request[[2,5]]=.2*(expected[[2,5]]-dq[i,[2,5]])
        maximum['pd_request']=max(maximum['pd_request'],float(abs(np.array(req)-request).max()))
        motors=F(4)();wheels=F(2)()
        assert lib.RLMotion31MapPolicyLegTorqueToMotorOrder(lim,motors)==0
        assert lib.RLMotion31MapPolicyWheelTorqueToForwardOrder(lim,wheels)==0
        # Physical motor rates [lf,lb,rf,rb,Lwheel,Rwheel] from the declared q map.
        motor_dq=np.r_[-dq[i,4],-dq[i,3],dq[i,1],-dq[i,0],dq[i,5],-dq[i,2]]
        mapped_torque=np.r_[np.array(motors),-wheels[0],wheels[1]]
        power=abs(float(mapped_torque@motor_dq)-float(np.array(lim)@dq[i]))
        maximum['torque_power_mapping']=max(maximum['torque_power_mapping'],power)
        h=F(4)();w=F(2)();assert lib.RLMotion31MapPolicyTargets(target,h,w)==0
        inverse=Snapshot()
        # Hardware order is lf/lb/rf/rb; MapJointAngles expects rb/rf/lb/lf.
        lib.RLMotion31MapJointAngles.argtypes=[C.POINTER(Snapshot)]+[C.c_float]*8
        lib.RLMotion31MapJointAngles(C.byref(inverse),h[3],h[2],h[1],h[0],0,0,0,0)
        maximum['angle_roundtrip']=max(maximum['angle_roundtrip'],
            float(abs(np.array(inverse.q)-np.array(target)[[0,1,3,4]]).max()))
    sample=Snapshot(height=.34);sample.gravity[1]=-1;sample.q[0]=1e30
    out=F(31)();invalid_return=lib.RLMotion31BuildObservation(C.byref(sample),out)
    assert invalid_return!=0
    assert maximum['observation']<1e-6 and maximum['decode']<1e-6
    assert maximum['pd_request']<2e-5 and maximum['angle_roundtrip']<1e-6
    assert maximum['torque_power_mapping']<1e-4
    return dict(samples=n,max_abs=maximum,huge_finite_q_rejected=True,
                coordinates='model ch0..5; geometric R/R/R/L/L/L',
                scope='software mappings and units; encoder zeros/physical polarities remain unverified')


def weights_and_inference(repo,package,temp):
    import torch
    import onnx
    from onnx.reference import ReferenceEvaluator
    from native_stand.students.model import Student
    graph=onnx.load(str(package/'actor.onnx'));arrays={t.name:onnx.numpy_helper.to_array(t) for t in graph.graph.initializer}
    text=(repo/'X-CUBE-AI/App/tjb_network_data_params.c').read_text()
    section=re.search(r's_tjb_network_weights_array_u64\[(\d+)\]\s*=\s*\{(.*?)\};',text,re.S)
    if section is None:raise ValueError('unknown ST weight container; do not guess layout')
    words=re.findall(r'0x([0-9a-fA-F]+)U',section[2]);assert len(words)==int(section[1])
    blob=b''.join(struct.pack('<Q',int(v,16)) for v in words)
    network=(repo/'X-CUBE-AI/App/tjb_network.c').read_text();comparisons=[];weights={}
    for name,array in arrays.items():
        index=name.split('.')[1];kind='weights' if name.endswith('weight') else 'bias'
        symbol=f'_mlp_mlp_{index}_Gemm_output_0_{kind}_array'
        match=re.search(re.escape(symbol)+r'\.data_start\s*=\s*AI_PTR\(g_tjb_network_weights_map\[0\]\s*\+\s*(\d+)\)',network)
        if match is None:raise ValueError('missing generated C tensor offset: '+name)
        offset=int(match[1]);loaded=np.frombuffer(blob,dtype='<f4',count=array.size,offset=offset).reshape(array.shape).copy()
        same=array.tobytes()==loaded.tobytes()
        if not same:raise ValueError('generated firmware weight differs from ONNX: '+name)
        weights[name]=torch.from_numpy(loaded)
        comparisons.append(dict(name=name,count=array.size,offset_bytes=offset,bitwise_equal=same))
    with torch.random.fork_rng(devices=[]):actor=Student(True,version='multiskill_mcu_v2')
    loaded=actor.load_state_dict(weights,strict=False)  # log_std is irrelevant to deterministic mean.
    assert loaded.missing_keys==['log_std'] and not loaded.unexpected_keys
    actor.eval();ref=ReferenceEvaluator(graph)
    with np.load(package/'golden_vectors.npz',allow_pickle=False) as a:
        x=a['observation'].astype('float32');gold=a['pytorch_mean_action'].astype('float32')
        saved_onnx=a['onnx_mean_action'].astype('float32')
    assert x.shape==(810,31) and gold.shape==(810,6)
    # Keep the deployment's real [1,31] inference boundary in the export backend.
    predicted=np.concatenate([ref.run(None,{'observation':row[None]})[0] for row in x])
    with torch.inference_mode():student=actor(torch.from_numpy(x)).numpy()
    error=float(abs(predicted-student).max());gold_error=float(abs(predicted-gold).max())
    assert error<=1e-5 and gold_error<=1e-5
    fixed=np.zeros((1,31),np.float32);fixed[0,[4,24,25,28]]=[-1,.68,1,1]
    repeated=np.concatenate([ref.run(None,{'observation':fixed})[0] for _ in range(100)])
    assert np.array_equal(repeated,np.broadcast_to(repeated[0],repeated.shape))
    reset=fixed.copy();reset[0,6:10]=np.array([.21648856,.21663799,-.21648856,.21663799])/1.5
    reset[0,16:20]=reset[0,6:10]
    reset_action=ref.run(None,{'observation':reset})[0]
    np.savez(temp/'inference_probe.npz',fixed_input=fixed,fixed_mean=repeated[:1],
             reset_input=reset,reset_mean=reset_action)
    return dict(parameters=sum(a.size for a in arrays.values()),weight_bytes=len(blob),
        weight_blob_sha256=hashlib.sha256(blob).hexdigest(),tensors=comparisons,
        golden_samples=len(x),golden_source_sha256=sha(package/'golden_vectors.npz'),
        reference_backend='onnx.reference.ReferenceEvaluator CPU; fixed batch1',
        max_abs_onnx_vs_torch_factory=error,max_abs_onnx_vs_saved_pytorch=gold_error,
        max_abs_onnx_vs_saved_onnx=float(abs(predicted-saved_onnx).max()),
        per_channel_max_abs=(abs(predicted-student).max(0)).tolist(),
        fixed_input_100_repeats_max_abs=float(abs(repeated-repeated[0]).max()),
        fixed_zero_pose_mean=repeated[0].tolist(),
        reset_declared_pose_mean=reset_action[0].tolist(),
        stm32_runtime='NOT_VERIFIED: ARM-only runtime, no MCU execution or flashed-build record'),ref


def log_replay(ref,source,lib):
    spec=importlib.util.spec_from_file_location('motion31_csv',ROOT/'scripts/replay_motion31_shadow_csv.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows,x,logged,times,columns=module.read_csv(source)
    active_fields=[name for name in rows[0] if name.endswith('.rl_active')]
    if len(active_fields)>1:raise ValueError('ambiguous rl_active fields')
    if active_fields:
        active=np.array([int(float(row[active_fields[0]]))!=0 for row in rows])
        selection='explicit rl_active inference-permission flag; not controller ownership'
    else:
        # Some old exports have no inference flag. Compare only rows whose
        # stored output changed; do not call those rows active RL control.
        active=np.r_[np.any(logged[0]!=0),np.any(np.diff(logged,axis=0)!=0,axis=1)]
        selection='changed stored mean only; inference permission and owner UNKNOWN; unchanged outputs not checked'
    valid=np.zeros(len(x),bool)
    for i in np.flatnonzero(active):
        sample=Snapshot(gyro=F(3)(*(x[i,:3]*4)),gravity=F(3)(*x[i,3:6]),
            q=F(4)(*(x[i,6:10]*1.5)),dq=F(6)(*(x[i,10:16]*20)),
            previous=F(6)(*(np.r_[x[i,16:18]*1.5,x[i,20]*100,x[i,18:20]*1.5,x[i,21]*100])),
            previous_valid=1,vx=float(x[i,22]*4),yaw=float(x[i,23]*4),height=float(x[i,24]*.5))
        out=F(31)();valid[i]=lib.RLMotion31BuildObservation(C.byref(sample),out)==0 and abs(logged[i]).max()<=1.00001
    index=np.flatnonzero(valid)
    if not len(index):raise ValueError('no numerically valid active inference frame in '+str(source))
    prediction=np.concatenate([ref.run(None,{'observation':x[i:i+1]})[0] for i in index])
    delta=abs(prediction-logged[index]);assert np.isfinite(delta).all()
    mismatched=index[delta.max(1)>1e-5]
    return dict(source=str(source),source_sha256=sha(source),rows=len(x),
        inference_flag_present=bool(active_fields),selection_rule=selection,selected_rows=int(active.sum()),
        compared_rows=len(index),selected_rows_rejected_as_invalid=int((active&~valid).sum()),
        max_abs=float(delta.max()),per_channel_max_abs=delta.max(0).tolist(),
        rows_with_error_over_1e_5=len(mismatched),
        mismatched_rows=[dict(index=int(i),time_s=float(times[i]),
            logged_mean=logged[i].tolist(),previous_logged_mean=logged[i-1].tolist() if i else None,
            input_used=x[i].tolist()) for i in mismatched[:8]],
        interpretation='numerically matching selected frames' if not len(mismatched) else
            'unmatched rows retained; no inference/owner flags to identify reset/frozen-output boundary',
        task_flags='25..30 absent; Motion/NONE explicitly assumed from deployment contract',
        scope='numerical consistency only; rl_active does not prove physical RL control')


def standing_pd_probe(library,ref):
    from native_stand.students.firmware_control import FirmwareControl
    publication=json.loads(library.with_name('validation.json').read_text())
    if publication.get('status')!='CPU_PASS' or publication.get('library_sha256')!=sha(library):
        raise ValueError('public firmware library validation identity differs')
    if not all(sha(path)==value for path,value in publication['source_sha256'].items()):
        raise ValueError('firmware library sources have changed; rebuild through the public validator')
    native=np.zeros(16,np.float32);native[4]=-1
    native[6:10]=[.21648856,.21663799,-.21648856,.21663799]
    with FirmwareControl(library) as core:
        x=core.observe(0,0,native,[0,0,.34]);mean=ref.run(None,{'observation':x[None]})[0][0]
        core.publish(0,mean);length,spread=core.geometry(native[6:10])
        values=core.pd(0,0,native,[0,0,.34],length,spread)
    def motor_order(t):return [-float(t[4]),-float(t[3]),float(t[1]),-float(t[0])]
    return dict(scope='one synthetic declared standing input and real C math; no physical equilibrium or LQR simulation',
        library_sha256=sha(library),input31=x.tolist(),mean=mean.tolist(),
        adopted_targets_model_order=values['adopted_target'].tolist(),
        leg_length_m=length.tolist(),leg_spread_rad=spread.tolist(),
        pd_request_model_order_nm=values['pd_request_nm'].tolist(),
        spring_compensation_motor_order_nm=motor_order(values['spring_compensation_model_nm']),
        motor_command_lf_lb_rf_rb_nm=motor_order(values['motor_model_nm']),
        motor_order='lf=geometric left active,lb=left hip,rf=right active,rb=right hip',
        actual_flashed_build='UNKNOWN',measured_motor_torque='NOT_MEASURED')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,default=Path('/home/kellen/combat-dm-balance-main'))
    parser.add_argument('--package',type=Path,default=ROOT/'deployment_protocol/2026_10_23-19_53__motion500_currentplant_small31_onnx')
    parser.add_argument('--before-dir',type=Path)
    parser.add_argument('--firmware-library',type=Path,help='Optional output of public validate_motion_firmware --extended')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:
        print(json.dumps(dict(task='CPU weights/interface/driver audit',files_written=0,simulation=0,optimizer_steps=0)));return
    if args.output_dir is None:parser.error('--run requires a new independent --output-dir')
    import torch
    torch.set_num_threads(4);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    repo=args.repository.resolve();package=args.package.resolve();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    if sha(package/'actor.onnx')!=MODEL_SHA:raise ValueError('Motion500 deployment ONNX SHA differs')
    tracked=['application/chassis/'+n for n in ('rl_motion31.c','rl_motion31.h','rl_control.c','rl_control.h','rl_shadow.c','rl_shadow.h','chassis.c')]
    tracked+=['modules/motor/DMmotor/dmmotor.c','modules/motor/DMmotor/dmmotor.h',
              'application/chassis_can_link.h','application/robot_def.h',
              'modules/motor/DJImotor/dji_motor.c','X-CUBE-AI/App/tjb_network.c',
              'X-CUBE-AI/App/tjb_network_data_params.c','Makefile']
    identities={name:sha(repo/name) for name in tracked}
    result=dict(version='motion31_electrical_cpu_audit_v1',status='INCOMPLETE',
                repository=str(repo),firmware_head=checked(['git','-C',repo,'rev-parse','HEAD'])['stdout'].strip(),
                source_sha256=identities,actor_onnx_sha256=MODEL_SHA,
                simulation_steps=0,optimizer_steps=0,hardware_access=False)
    try:
        with tempfile.TemporaryDirectory(prefix='motion31-cpu-audit-') as folder:
            temp=Path(folder);lib=math_library(repo/'application/chassis/rl_motion31.c',repo,temp,'current')
            result['interface_math']=interface_math(lib)
            result['gravity_current']=quaternion_gravity(lib)
            assert result['gravity_current']['max_abs']<1e-6
            original=temp/'head_motion.c'
            original.write_text(checked(['git','-C',repo,'show','HEAD:application/chassis/rl_motion31.c'])['stdout'])
            result['gravity_committed_head']=quaternion_gravity(math_library(original,repo,temp,'head'))
            original_driver=temp/'head_dmmotor.c'
            original_driver.write_text(checked(['git','-C',repo,'show','HEAD:modules/motor/DMmotor/dmmotor.c'])['stdout'])
            driver_flags=['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-Wno-unused-parameter',
                '-DHOST_LEGACY_PACKING_PROBE',*['-I'+str(repo/p) for p in ('tests/host_driver',
                'modules/motor/DMmotor','modules/motor/DJImotor','modules/motor','modules/daemon','modules')]]
            result['driver_packing_observation']={}
            for name,source in (('committed_head',original_driver),('current',repo/'modules/motor/DMmotor/dmmotor.c')):
                binary=temp/('driver_'+name)
                checked(driver_flags+[repo/'tests/test_motor_driver_boundary.c',source,
                    repo/'modules/motor/DJImotor/dji_motor.c','-lm','-o',binary])
                result['driver_packing_observation'][name]=checked([binary,'packing_observe'])
            result['weights_and_inference'],ref=weights_and_inference(repo,package,out)
            if args.firmware_library:result['standing_pd_probe']=standing_pd_probe(args.firmware_library,ref)
            result['existing_log_replay']=[]
            for name in ('Ozone_DataSampling_261023.csv','Ozone_DataSampling_261024_17_35.csv'):
                path=ROOT/'logs'/name
                if path.exists():result['existing_log_replay'].append(log_replay(ref,path,lib))
            result['before_after_regressions']=[]
            if args.before_dir is not None:
                before=args.before_dir.resolve()
                binary=temp/'before_timing'
                checked(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-I'+str(repo/'application/chassis'),
                    repo/'tests/test_rl_timing.c',before/'application/chassis/rl_control.c',
                    repo/'application/chassis/rl_motion31.c','-lm','-o',binary])
                for probe in ('sequence','wrap','schedules'):
                    r=command([binary,probe]);r['probe']='before_'+probe
                    assert r['returncode']==(0 if probe=='schedules' else 1)
                    result['before_after_regressions'].append(r)
                r=command([__import__('sys').executable,repo/'tests/run_chassis_snapshot_host_test.py',
                           '--source',before/'application/chassis/chassis.c'])
                assert r['returncode']==1;result['before_after_regressions'].append(r)
                r=command([__import__('sys').executable,repo/'tests/run_chassis_snapshot_host_test.py',
                    '--mode','commands','--link-header',before/'application/chassis_can_link.h'])
                assert r['returncode']==1;result['before_after_regressions'].append(r)
        result['source_unchanged_during_audit']=all(sha(repo/k)==v for k,v in identities.items())
        assert result['source_unchanged_during_audit']
        result['status']='CPU_PASS_WITH_LOG_BOUNDARIES_AND_HARDWARE_UNVERIFIED' if any(
            x['rows_with_error_over_1e_5'] for x in result['existing_log_replay']) else 'CPU_PASS_WITH_HARDWARE_UNVERIFIED'
    except BaseException as error:
        result['status']='FAILED'
        result['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        result['wall_s']=time.monotonic()-started
        (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','interface_math','gravity_current','gravity_committed_head','existing_log_replay','wall_s')},indent=2))


if __name__=='__main__':main()
