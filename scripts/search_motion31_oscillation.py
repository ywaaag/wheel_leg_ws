"""Bounded CPU MuJoCo search for logged closed-loop oscillation, not identification.

Default prints a zero-write plan. --run is a new physical diagnostic attempt.
Uses current-Plant/tight solver, the exported ONNX, and the actual host C owner,
adoption, PD, compensation and output mappings. Does not simulate full FreeRTOS,
CAN or LQR, nor silently replace missing hardware measurements.
"""
import argparse
import copy
import ctypes as C
import json
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import numpy as np

from replay_motion31_shadow_csv import read_csv, sha
from native_stand.students.oscillation_search import (
    VERSION, CExecution, LEGS, WHEELS, VelocityFeedback, dynamic_cases, dynamic_nominal,
    validate_dynamics, features, nominal, singles, combinations, similarity)
from native_stand.students.real2sim_delay import ObservationHistory
from native_stand.students.real2sim_handover import initialize_from_log
from native_stand.students.motion_mcu_execution import WHEEL_CAP


def c_input_valid(ex, native, command):
    """Ask the same C validator; do not reimplement or relax its thresholds."""
    from audit_motion31_electrical_cpu import Snapshot,F
    q=np.asarray(native,np.float32)
    sample=Snapshot(gyro=F(3)(*q[:3]),gravity=F(3)(*q[3:6]),q=F(4)(*q[6:10]),
                    dq=F(6)(*q[10:16]),previous=F(6)(*ex.adopted),previous_valid=1,
                    vx=float(command[0]),yaw=float(command[1]),height=float(command[2]))
    ex.lib.RLMotion31ValidateSnapshot.argtypes=[C.POINTER(Snapshot)]
    return ex.lib.RLMotion31ValidateSnapshot(C.byref(sample))==0


def write(path, value):
    p = Path(path); temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(p)


def inputs(args):
    spec = json.loads(args.spec.read_text())
    manifest = json.loads((args.onnx.parent/'manifest.json').read_text())
    if sha(args.onnx) != manifest['actor_onnx_sha256'] or sha(args.spec) != manifest['spec_sha256']:
        raise ValueError('model/parent spec identity differs from deployment manifest')
    validation = json.loads(args.firmware_validation.read_text())
    if validation['status'] != 'CPU_PASS' or sha(validation['library']) != validation['library_sha256']:
        raise ValueError('validated host C library required')
    for path, expected in validation['source_sha256'].items():
        if sha(path) != expected: raise ValueError('host C source changed: '+path)
    rows, x, y, t, columns = read_csv(args.csv)
    names = [n for n in rows[0] if n.endswith('.rl_active')]
    if len(names) != 1: raise ValueError('one inference flag required')
    enabled = np.array([float(r[names[0]]) != 0 for r in rows])
    rising = np.flatnonzero(enabled & ~np.r_[False, enabled[:-1]])
    if len(rising) != 1: raise ValueError('select one explicit continuous inference interval')
    ix = int(rising[0]); active_ids = np.flatnonzero(enabled)
    if not np.all(np.diff(active_ids) == 1) or ix < 1:
        raise ValueError('continuous interval with preceding LQR row required')
    if t[active_ids[-1]]-t[ix] < args.seconds-.01:
        raise ValueError('comparison window exceeds continuously enabled log')
    def col(name):
        found = [n for n in rows[0] if n.endswith(name)]
        if len(found) != 1: raise ValueError('missing/ambiguous '+name)
        return np.array([float(r[found[0]]) for r in rows])
    leg = np.stack([col(f'.rl_leg_torque_nm[{j}]') for j in range(4)], -1)
    wh = np.stack([col(f'.rl_wheel_torque_nm[{j}]') for j in range(2)], -1)
    tau = np.zeros((len(t), 6)); tau[:, [4,3,1,0]] = leg*[-1,-1,1,-1]; tau[:, [5,2]] = -wh
    initial_motor = np.zeros(6,np.float32)
    initial_motor[[4,3,1,0]] = [col(f'.lqr_leg_torque_nm[{j}]')[ix-1]*s for j,s in enumerate([-1,-1,1,-1])]
    initial_motor[[5,2]] = [-col(f'.lqr_wheel_torque_nm[{j}]')[ix-1] for j in range(2)]
    return spec, validation, dict(x=x, y=y, t=t, ix=ix, tau=tau, initial_motor=initial_motor,
                                  missing_task_indices=[i for i in range(25,31) if i not in columns['observation']])


def plan(args, data):
    if args.matrix=='separated_forces_v1':
        from native_stand.students.firmware_mechanics import mechanical_cases
        cases=mechanical_cases()
    elif args.matrix=='source_packets_v1':
        from native_stand.students.firmware_sampling import cases as source_cases
        cases=source_cases()
    else:cases=dynamic_cases() if args.matrix=='dynamics_v2' else singles()
    count = len(cases)
    ceiling=160 if args.matrix=='separated_forces_v1' else 128
    if not 6 <= args.seconds <= 6.2 or not count <= args.max_cases <= ceiling:
        raise ValueError(f'seconds6..6.2, cases{count}..{ceiling} required')
    if not 60 <= args.wall_seconds <= 1800 or not 32 <= args.storage_mib <= 256:
        raise ValueError('bounded wall60..1800s/storage32..256MiB required')
    request=dict(version=VERSION if args.matrix=='single_v1' else 'motion31_dynamic_oscillation_search_v2',
                matrix=args.matrix,model=str(args.onnx.resolve()), model_sha256=sha(args.onnx),
                score_version='raw_moments_fft_grid_v2',
                source_log=str(args.csv.resolve()), source_log_sha256=sha(args.csv),
                first_log_time_s=float(data['t'][data['ix']]), duration_s=args.seconds,
                fit_window_s=[.5,2.], holdout_window_s=[2.,args.seconds],
                single_cases=count, maximum_cases=args.max_cases,
                maximum_physics_steps=args.max_cases*round(args.seconds*1000),
                wall_limit_s=args.wall_seconds, storage_limit_bytes=args.storage_mib*2**20,
                resources=dict(engine='MuJoCo CPU', cpu_threads=1, GPU=False, viewer=False),
                optimizer_steps=0, online_PPO=0, physical_attempts=1,
                source_contract='multiskill_mcu_v2; same deterministic ONNX throughout',
                nominal=nominal(),dynamic_nominal=dynamic_nominal(),single_matrix=cases,
                command_source='logged command trajectory only; Actor measurements from simulated feedback',
                initial_previous='logged first previous, once; subsequent previous from C adoption',
                initial_motor='preceding LQR debug command proxy; NOT actual motor torque or measured queue',
                unknown=['flashed build', 'true sensor/driver timing', 'real torque', 'root vx',
                         'actual prehistory/queue', 'encoder calibration', 'spring curve'],
                certificate_eligible=False, hardware_parameters_identified=False)
    if args.matrix in ('source_packets_v1','separated_forces_v1'):
        from native_stand.students.firmware_sampling import VERSION as sampling_version,profile,source_parameters
        request.update(version=sampling_version,nominal=source_parameters(),sampling_nominal=profile(),
            adaptive_combinations=False,source_software=dict(leg_feedback='real DM 16/12bit decoder, packet hold',
                wheel_feedback='real DJI per-packet .85 IIR and integer RPM',
                task_body='real MotorControlTask including DM every-other-call counter',
                control='real C observation/adoption/PD/compensation; explicit Robot->Motor->Actor event order',
                actual_RTOS_and_CAN_emulated=False),
            hypothesis_boundary=dict(packet_cadence='2ms leg,1ms wheel are hypotheses; also test1/5/10ms',
                packet_rounding='nearest-bin generating hypothetical received words; actual C decodes',
                prehistory='32ms constant initial state to initialize packet filters; no physics',
                startup_control='preceding LQR debug torque proxy until first target adoption; no full LQR',
                estimator='unknown internal DM estimator, additional ages/tau explicitly isolated',
                compliance='optional upstream-encoder two-inertia; rotor reallocated from .012 armature',
                fixed_old_10ms_feedback_age='removed in this version, not assumed to be hardware truth'))
    if args.matrix=='separated_forces_v1':
        from native_stand.students.firmware_mechanics import VERSION as mechanics_version
        request.update(version=mechanics_version,
            force_contract=dict(leg_motor_cap_nm=40,wheel_motor_cap_nm=float(WHEEL_CAP),
                legacy='motor transmission + spring combined in the old actuator with40Nm secondary cap',
                separated='rigid motor uses unchanged actuator; spring is independent qfrc_applied; flex load receives passive transmission through qfrc_applied',
                flexible_motor='40Nm upstream motor acts on virtual rotor; spring/transmission may store/release energy',
                force_readback='after mj_step before any forward; actuator/external/passive/constraint/bias/qacc separately',
                total_armature=.012,mechanical_hypothesis_envelope_nm=200),
            followup_rule='up to4 fully covered .5-2s fit-window leaders, each lowgain/preload/motor_tau2; at most12; no holdout selection',
            reproduction_rule=dict(full_holdout_coverage=.98,all_four_frequency_error_hz_max=4.,
                all_four_dq_std_ratio=[.5,2.],mean_band20_27_fraction_min=.25,tilt_mean_error_rad_max=.1,
                hardware_identified=False),
            hypothesis_boundary={**request['hypothesis_boundary'],
                'initial_preload':'0 or preceding LQR debug motor torque/K; proxy only; encoder stays at logged angle'},
            adaptive_combinations='predeclared fit-only followups, at most12')
    return request


def setup(spec, args):
    import mujoco
    from validate_native_mujoco import binding
    from motion_mcu_playback import declared, configuration, apply_current_plant
    from native_stand.students.mujoco_solver import apply as solver
    model = mujoco.MjModel.from_xml_path(str(args.xml)); refs = binding(model)
    if model.neq != 4 or model.nu != 6 or abs(model.opt.timestep-.001) > 1e-10:
        raise ValueError('native closed-chain/timing topology differs')
    cfg = configuration(SimpleNamespace(leg_armature_m=None,zero_latency=False),declared(spec),spec)
    plant = apply_current_plant(model,refs,spec,cfg)
    solved = solver(model,refs,'closure_contact_tight_v1')
    keys = ('body_mass','body_inertia','dof_armature','dof_damping','dof_frictionloss','geom_friction')
    base = {k:getattr(model,k).copy() for k in keys}
    return model, refs, base, dict(plant=plant,solver=solved)


def run_case(model, refs, base, actor, data, library, case, seconds, deadline):
    if 'sampling' in case:
        return run_sampled_case(model,refs,base,actor,data,library,case,seconds,deadline)
    import mujoco
    from diagnose_motion_real2sim import sensors
    from validate_native_mujoco import state
    p = case['parameters']
    dynamic=validate_dynamics(case.get('dynamics'))
    for k,v in base.items(): getattr(model,k)[:] = v
    dofs = np.array(refs['dofs'])[LEGS]
    model.dof_armature[dofs] = p['armature']
    model.dof_damping[dofs] += p['viscous']; model.dof_frictionloss[dofs] += p['frictionloss']
    model.body_mass[refs['root']] *= p['root_mass_scale']
    model.body_inertia[refs['root']] *= p['root_inertia_scale']
    model.geom_friction[:,0] = p['friction']
    mujoco.mj_setConst(model,mujoco.MjData(model))
    ix = data['ix']; x,t = data['x'],data['t']
    # Reported q = true q + offset; initial true q must change as well if the log
    # itself is interpreted as a miscalibrated measurement of the SAME robot.
    initial_obs = x[ix].copy(); initial_obs[6:10] -= np.asarray(p['zero_bias'])/1.5
    d, fit = initialize_from_log(model,refs,initial_obs,p['root_vx'])
    bias = np.zeros(6); bias[LEGS] = p['zero_bias']
    def reported(s):
        result = s.copy(); result[6:12] += bias
        return result
    initial = sensors(model,d,refs)
    history = ObservationHistory(reported(initial),initial_tick=-32)
    velocity=VelocityFeedback(dynamic,reported(initial))
    ex = CExecution(library,p,data['initial_motor'],dynamics=dynamic)
    rows, fine = [], []; termination = None; steps=0;guard=None
    try:
        for tick in range(round(seconds*1000)):
            if tick%100 == 0 and time.monotonic() >= deadline:
                termination='WALL_BUDGET'; break
            # Refresh current kinematics before reading mixed q/dq/body fields.
            # mj_forward performs no integration; actual physics remains1ms.
            mujoco.mj_forward(model,d)
            current = sensors(model,d,refs)
            if not np.isfinite(current).all(): raise RuntimeError('NONFINITE_SENSORS')
            history.append(tick,velocity.step(tick,reported(current)))
            if tick%10 == 0:
                query = t[ix]+tick*.001
                j = min(len(t)-1,max(ix,int(np.searchsorted(t,query,side='right')-1)))
                command = (x[j,22:25]*[4,4,.5]).astype(np.float32)
                s = history.at(tick-p['actor_age_ms'])
                native = np.r_[s[:6],s[6:12][LEGS],s[12:18]].astype(np.float32)
                try:
                    obs = ex.core.observe(tick,tick-p['actor_age_ms'],native,command)
                except ValueError:
                    if c_input_valid(ex,native,command): raise
                    termination='FIRMWARE_INPUT_GUARD'
                    guard=dict(tick_ms=tick,phase='Actor_input',native16=native.tolist(),validator='production C rejected; no clipping or bypass')
                    break
                if tick == 0: obs[16:22] = x[ix,16:22]  # explicit unknown startup cache hypothesis
                mean = actor(obs[None])[0]
                if not np.isfinite(mean).all(): raise RuntimeError('NONFINITE_ACTOR')
                ex.core.publish(tick,mean)
                pre = dict(time=tick*.001,actor_input=obs.copy(),mean=mean.copy(),
                           q=current[6:12].copy(),dq=current[12:18].copy(),gravity=current[3:6].copy())
            feedback = history.at(tick-p['pd_age_ms'])
            native = np.r_[feedback[:6],feedback[6:12][LEGS],feedback[12:18]].astype(np.float32)
            try:
                torque = ex.step(tick,native,current[6:12][LEGS],command)
            except RuntimeError:
                if c_input_valid(ex,native,command): raise
                termination='FIRMWARE_INPUT_GUARD'
                guard=dict(tick_ms=tick,phase='PD_input',native16=native.tolist(),validator='production C rejected; no clipping or bypass')
                break
            # Preserve solver capacity and detect clipping instead of silently
            # changing mechanical/motor limits while searching.
            ctrl = torque/refs['gears']
            bounds = model.actuator_ctrlrange[refs['actuators']]
            if np.any(ctrl < bounds[:,0]-1e-5) or np.any(ctrl > bounds[:,1]+1e-5):
                raise RuntimeError('NET_TORQUE_EXCEEDS_DECLARED_SOLVER_CAPACITY')
            d.ctrl[refs['actuators']] = ctrl
            fine.append(np.r_[tick*.001,current[12:18],ex.motor])
            if tick%10 == 0:
                pre.update(command=ex.driver.copy(),motor=ex.motor.copy(),pd_request=ex.raw.copy(),
                           adopted=ex.adopted.copy(),pd_feedback=native.copy())
            mujoco.mj_step(model,d); steps+=1
            if tick%10 == 9:
                mujoco.mj_forward(model,d)
                _,_,st = state(model,d,refs)
                pre.update(post_state=np.array([st[k] for k in ('vx','yaw_rate','height','pitch','roll','closure_m')]),
                           contact=np.array(st['contact_force_n']),qpos=d.qpos.copy(),qvel=d.qvel.copy())
                rows.append(pre)
                if st['closure_m']>.005: raise RuntimeError('CLOSURE_EXCEEDED_5MM')
                if st['up']<.5: termination='UPRIGHT'; break
        arrays = {k:np.stack([row[k] for row in rows]) for k in rows[0]} if rows else {}
        if arrays:
            arrays['fine_1ms'] = np.stack(fine)
        return dict(**case,physics_steps=steps,termination=termination,initial_fit=fit,
                    input_guard=guard,
                    nominal_C_max_error_nm=ex.nominal_error,
                    first_actor_vs_log_max_abs=(float(abs(arrays['actor_input'][0]-x[ix]).max()) if arrays else None),
                    effective_plant=dict(leg_armature=model.dof_armature[dofs].tolist(),
                        leg_damping=model.dof_damping[dofs].tolist(),leg_frictionloss=model.dof_frictionloss[dofs].tolist(),
                        root_mass=float(model.body_mass[refs['root']]),root_inertia=model.body_inertia[refs['root']].tolist(),
                        friction=float(model.geom_friction[refs['ground'],0]))), arrays
    except Exception as exc:
        arrays={k:np.stack([row[k] for row in rows]) for k in rows[0]} if rows else {}
        if fine: arrays['fine_1ms']=np.stack(fine)
        return dict(**case,physics_steps=steps,termination='RUNTIME_FAULT',initial_fit=fit,
                    runtime_fault=type(exc).__name__+': '+str(exc)),arrays
    finally:
        ex.close()


def run_sampled_case(model,refs,base,actor,data,library,case,seconds,deadline):
    """Source-exact packet/packing boundaries; explicitly phased task schedule."""
    import mujoco
    from diagnose_motion_real2sim import sensors
    from validate_native_mujoco import state
    from native_stand.students.firmware_sampling import SourceExecution,PacketEnvelopeError
    mechanics='force_route' in case
    if mechanics:
        from native_stand.students.firmware_mechanics import MechanicalExecution,ForceRouter,MechanicalEnvelopeError
    p,s=case['parameters'],case['sampling']
    for k,v in base.items():getattr(model,k)[:]=v
    dofs=np.asarray(refs['dofs'])[LEGS]
    model.dof_armature[dofs]=p['armature']-(s['rotor_inertia'] if s['coupling_stiffness_nm_rad'] else 0)
    mujoco.mj_setConst(model,mujoco.MjData(model))
    x,t,ix=data['x'],data['t'],data['ix']
    initial_obs=x[ix].copy();twist=np.zeros(4)
    if mechanics and s['coupling_stiffness_nm_rad']:
        twist=case['preload_fraction']*data['initial_motor'][LEGS]/s['coupling_stiffness_nm_rad']
        if abs(twist).max()>.15:raise ValueError('initial preload outside declared .15rad hypothesis')
        initial_obs[6:10]-=twist/1.5
    d,fit=initialize_from_log(model,refs,initial_obs,0.)
    initial=sensors(model,d,refs)
    if mechanics:
        upstream=initial.copy();upstream[6:12][LEGS]=x[ix,6:10]*1.5
        ex=MechanicalExecution(library,p,data['initial_motor'],s,upstream,x[ix,16:22])
        router=ForceRouter(model,refs,case['force_route'],bool(ex.flex.k),envelope_nm=case['mechanical_envelope_nm'])
    else:ex=SourceExecution(library,p,data['initial_motor'],s,initial,x[ix,16:22])
    rows=[];policies=[];pd_rows=[];fine=[];steps=0;termination=None;guard=None;fault=None
    force_rows=[]
    counts=(0,0);receipt_ages=[];sample_ages=[]
    try:
        for tick in range(round(seconds*1000)):
            if tick%100==0 and time.monotonic()>=deadline:termination='WALL_BUDGET';break
            mujoco.mj_forward(model,d);current=sensors(model,d,refs)
            if not np.isfinite(current).all():raise RuntimeError('NONFINITE_SENSORS')
            query=t[ix]+tick*.001
            j=min(len(t)-1,max(ix,int(np.searchsorted(t,query,side='right')-1)))
            command=(x[j,22:25]*[4,4,.5]).astype(np.float32)
            try:torque=ex.step_sampled(tick,current,command,actor)
            except PacketEnvelopeError as exc:
                termination='HYPOTHESIS_SENSOR_ENVELOPE'
                guard=dict(tick_ms=tick,reason=str(exc),scope='hypothetical internal shaft speed/angle exceeds declared packet range; not clipped');break
            except (RuntimeError,ValueError):
                native=ex.io.current if ex.io.current is not None else np.r_[current[:6],current[6:12][LEGS],current[12:18]]
                if c_input_valid(ex,native,command):raise
                termination='FIRMWARE_INPUT_GUARD'
                guard=dict(tick_ms=tick,native16=native.tolist(),validator='production C, no threshold change');break
            if ex.policy_count!=counts[0]:
                po=ex.policy
                policies.append(dict(tick_ms=po['tick_ms'],snapshot_tick_ms=po['snapshot_tick_ms'],
                    scheduled_publish_tick_ms=po['publish_tick_ms'],receipt_tick_ms=po['receipt_tick_ms'].copy(),
                    sample_tick_ms=po['sample_tick_ms'].copy(),actor_input=po['actor_input'].copy(),mean=po['mean'].copy()))
                if (po['receipt_tick_ms']>po['snapshot_tick_ms']).any():raise RuntimeError('FUTURE_PACKET_IN_ACTOR_SNAPSHOT')
            if ex.pd_count!=counts[1]:
                pr=ex.pd;receipt_ages.append(tick-pr['receipt_tick_ms'][:6]);sample_ages.append(tick-pr['sample_tick_ms'])
                pd_rows.append(dict(tick_ms=tick,native16=pr['native16'].copy(),receipt_tick_ms=pr['receipt_tick_ms'].copy(),
                    sample_tick_ms=pr['sample_tick_ms'].copy(),metadata=pr['metadata'].copy(),
                    requested=ex.raw.copy(),adopted=ex.adopted.copy(),motor_command=ex.driver.copy()))
            counts=(ex.policy_count,ex.pd_count)
            if mechanics:
                try:router.apply(d,ex.motor,ex.transmitted,ex.spring)
                except MechanicalEnvelopeError as exc:
                    termination='HYPOTHESIS_FORCE_ENVELOPE';guard=dict(tick_ms=tick,reason=str(exc));break
            else:
                ctrl=torque/refs['gears'];bounds=model.actuator_ctrlrange[refs['actuators']]
                if np.any(ctrl<bounds[:,0]-1e-5) or np.any(ctrl>bounds[:,1]+1e-5):
                    if s['coupling_stiffness_nm_rad']:
                        termination='HYPOTHESIS_FORCE_ENVELOPE';break
                    raise RuntimeError('NET_TORQUE_EXCEEDS_DECLARED_SOLVER_CAPACITY')
                d.ctrl[refs['actuators']]=ctrl
            fine.append(np.r_[tick*.001,current[12:18],ex.motor,ex.io.current[10:16],ex.io.wire,torque])
            measured_q=current[6:12].copy();measured_q[LEGS]=ex.io.current[6:10]
            pre=dict(time=tick*.001,q=measured_q,dq=ex.io.current[10:16].copy(),q_true=current[6:12].copy(),
                dq_true=current[12:18].copy(),gravity=current[3:6].copy(),
                mean=ex.policy['mean'].copy() if ex.policy else np.zeros(6),
                command=ex.driver.copy(),motor=ex.motor.copy(),wire=ex.io.wire.copy(),
                wire_tick_ms=ex.io.wire_tick.copy(),wire_sequence=ex.io.wire_sequence.copy(),
                adopted=ex.adopted.copy(),pd_request=ex.raw.copy(),pd_valid=ex.pd is not None)
            mujoco.mj_step(model,d);steps+=1
            if mechanics:force_rows.append(router.readback(d))
            if tick%10==9:
                mujoco.mj_forward(model,d);_,_,st=state(model,d,refs)
                pre.update(post_state=np.array([st[k] for k in ('vx','yaw_rate','height','pitch','roll','closure_m')]),
                    contact=np.array(st['contact_force_n']),qpos=d.qpos.copy(),qvel=d.qvel.copy())
                rows.append(pre)
                if st['closure_m']>.005:raise RuntimeError('CLOSURE_EXCEEDED_5MM')
                if st['up']<.5:termination='UPRIGHT';break
    except Exception as exc:
        fault=type(exc).__name__+': '+str(exc);termination='RUNTIME_FAULT'
    finally:
        ex.close()
    arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]} if rows else {}
    for prefix,items in (('policy',policies),('pd',pd_rows)):
        if items:
            for k in items[0]:arrays[prefix+'_'+k]=np.stack([r[k] for r in items])
    if fine:arrays['fine_1ms']=np.stack(fine)
    force_stats={}
    if force_rows:
        for key in force_rows[0]:
            value=np.stack([row[key] for row in force_rows]);arrays['force_'+key+'_1ms']=value
            force_stats[key]=dict(abs_max=np.max(abs(value),axis=0).tolist(),rms=np.sqrt(np.mean(value**2,axis=0)).tolist())
        applied=arrays['force_total_1ms'];requested=arrays['force_transmission_1ms']+arrays['force_spring_requested_1ms']
        force_stats['total_vs_requested_max_abs_nm']=float(abs(applied-requested).max())
    arrays['publication_events_ms']=np.array([e[0] for e in ex.events],dtype=int)
    arrays['source_timestamp_semantics']=np.array('sample: assumed motor measurement; receipt: actual simulated C callback; snapshot: Robot copy; publication: Actor completes; adoption: real C owner; wire: packed CAN enqueue, not actual torque')
    timing={}
    if receipt_ages:
        for key,values in (('receipt_age_ms',receipt_ages),('sample_age_ms',sample_ages)):
            a=np.stack(values);timing[key]=dict(mean=a.mean(0).tolist(),maximum=a.max(0).tolist())
    row=dict(**case,physics_steps=steps,termination=termination,initial_fit=fit,input_guard=guard,
        nominal_C_max_error_nm=ex.nominal_error,first_actor_vs_log_max_abs=(float(abs(policies[0]['actor_input']-x[ix]).max()) if policies else None),
        effective_plant=dict(leg_armature=model.dof_armature[dofs].tolist(),split_rotor_inertia=s['rotor_inertia'] if ex.flex.k else 0,
            total_leg_armature=p['armature'],spring='same unmeasured ideal counter-compensation hypothesis'),
        execution=dict(policy_calls=ex.policy_count,pd_calls=ex.pd_count,first_adoption_tick_ms=ex.first_adoption_tick,
            driver_packet_counts=ex.io.counts,prehistory_packet_counts=ex.io.bootstrap_counts,
            timeline=timing,physics_torque='simulated, not measured hardware',RTOS_jitter_emulated=False))
    if mechanics:
        row.update(force_readback=force_stats,initial_twist_rad=twist.tolist(),
            effective_force_limits=dict(ctrlrange=model.actuator_ctrlrange[refs['actuators']].tolist(),
                forcerange=model.actuator_forcerange[refs['actuators']].tolist(),
                forcelimited=model.actuator_forcelimited[refs['actuators']].tolist(),
                note='unchanged actuator motor limits; separated passive forces are independent generalized loads'))
    if fault:row['runtime_fault']=fault
    return row,arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('csv','onnx','spec','firmware-validation'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--xml',type=Path,default=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml'))
    parser.add_argument('--seconds',type=float,default=6.)
    parser.add_argument('--max-cases',type=int,default=64)
    parser.add_argument('--matrix',choices=['single_v1','dynamics_v2','source_packets_v1','separated_forces_v1'],default='single_v1')
    parser.add_argument('--wall-seconds',type=float,default=1200.)
    parser.add_argument('--storage-mib',type=int,default=128)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--continue-from',type=Path,help='Continue only unexecuted cases after the specifically CPU-confirmed input-guard classification; inherit all costs')
    parser.add_argument('--complete-source-packets-from',type=Path,
        help='Finish only the original missing packet cases after the CPU-proven finite49.78rad/s envelope exit; original FAILED/raw retained')
    parser.add_argument('--envelope-proof',type=Path,help='CPU-only replay proof required by the exact packet-envelope completion')
    parser.add_argument('--check',action='store_true',help='CPU preflight/materialization only, zero integration and zero writes')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    spec, validation, data = inputs(args)
    if args.matrix in ('source_packets_v1','separated_forces_v1') and validation.get('version')!='motion31_firmware_packet_sampling_v1':
        raise ValueError('source packet matrix requires actual DM/DJI sampling library validation')
    request = plan(args,data)
    cases=request['single_matrix']
    inherited=[];inherited_steps=0;inherited_wall=0.;parent=None
    if args.complete_source_packets_from:
        if args.continue_from or args.matrix!='source_packets_v1':raise ValueError('source-only completion cannot mix recovery recipes')
        parent=args.complete_source_packets_from.resolve()
        old=json.loads((parent/'spec.json').read_text());ledger=json.loads((parent/'ledger.json').read_text())
        summary=json.loads((parent/'summary.json').read_text());identity=json.loads((parent/'identity.json').read_text())
        for key in ('version','model_sha256','source_log_sha256','maximum_cases','maximum_physics_steps','duration_s','wall_limit_s','storage_limit_bytes','single_matrix'):
            if old[key]!=request[key]:raise ValueError('completion changes frozen packet matrix: '+key)
        if ledger['status']!='FAILED' or ledger['failure']!='RuntimeError: ValueError: hypothetical encoder outside declared CAN range':
            raise ValueError('completion requires exact settled finite packet-envelope failure')
        allowed={Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'src/native_stand/students/firmware_sampling.py'}
        for path,expected in identity['input_sha256'].items():
            if Path(path).resolve() not in allowed and sha(path)!=expected:raise ValueError('unchanged parent input required: '+path)
        inherited=copy.deepcopy(summary['results'])
        if len(inherited)!=36 or [r['name'] for r in inherited]!=[r['name'] for r in cases[:36]]:
            raise ValueError('completion must preserve exactly the36 already executed packet cases')
        last=inherited[-1]
        if last['name']!='flexK300_B0.2' or last['physics_steps']!=60:raise ValueError('wrong interrupted physical boundary')
        if args.envelope_proof is None:raise ValueError('explicit CPU envelope proof required')
        proof_path=args.envelope_proof
        proof=json.loads(proof_path.read_text())
        if proof['parent']!=str(parent.relative_to(Path.cwd())) or proof['case']!=last['name'] or not proof['finite'] or not proof['exceeds'] or proof['physics_steps_added']!=0 or proof['position_reconstruction_max_error_rad']>1e-7:
            raise ValueError('CPU-only envelope proof required, not a guessed recovery')
        from native_stand.students.firmware_sampling import DriverPackets,PacketEnvelopeError
        try:DriverPackets.word(max(proof['next_rotor_velocity_rad_s'],key=abs),45.,12)
        except PacketEnvelopeError:pass
        else:raise ValueError('new packet envelope guard does not reproduce the saved boundary')
        last.update(original_runtime_fault=last.pop('runtime_fault'),original_termination=last['termination'],
            termination='HYPOTHESIS_SENSOR_ENVELOPE',child_interpretation_only=True)
        inherited_steps=sum(r['physics_steps'] for r in inherited);inherited_wall=ledger['wall_used_s']
        if inherited_steps!=ledger['used_physics_steps']:raise ValueError('parent physical consumption differs')
        request['completion']=dict(parent=str(parent),parent_attempt=ledger['attempt_id'],
            parent_ledger_sha256=sha(parent/'ledger.json'),parent_summary_sha256=sha(parent/'summary.json'),
            proof=str(proof_path.resolve()),proof_sha256=sha(proof_path),inherited_physics_steps=inherited_steps,
            inherited_wall_s=inherited_wall,repeated_cases=0,remaining_original_cases=11,
            inherited_artifact_bytes=sum(p.stat().st_size for p in parent.rglob('*') if p.is_file()),
            interpretation='finite rotor speed49.778rad/s exceeds45rad/s packet hypothesis; original FAILED/raw unmodified')
        request['physical_attempts']=2
    if args.continue_from:
        parent=args.continue_from.resolve()
        old=json.loads((parent/'spec.json').read_text());ledger=json.loads((parent/'ledger.json').read_text())
        summary=json.loads((parent/'summary.json').read_text());identity=json.loads((parent/'identity.json').read_text())
        for key in ('version','model_sha256','source_log_sha256','maximum_cases','maximum_physics_steps','duration_s','wall_limit_s','storage_limit_bytes'):
            if old[key]!=request[key]:raise ValueError('continuation changes bounded search: '+key)
        if old.get('score_version','grid_moments_v1')!=request['score_version']:
            raise ValueError('old scoring search is finished; use an offline review, do not mix scoring versions in physical continuation')
        if ledger['status']!='FAILED' or not ledger['failure'].endswith('firmware observation rejected: -1'):
            raise ValueError('only the known C input-guard classification is supported')
        for path,value in identity['input_sha256'].items():
            if Path(path).resolve()!=Path(__file__).resolve() and sha(path)!=value:
                raise ValueError('parent input changed: '+path)
        inherited=copy.deepcopy(summary['results']);last=inherited[-1]
        if last['name']!='pd_age20' or last['termination']!='RUNTIME_FAULT':
            raise ValueError('unexpected parent failure; do not auto recover')
        # The final pre-action q is exactly the next aged Actor measurement at
        # the rejection boundary. Confirm invalidity with the actual C validator.
        with np.load(parent/'pd_age20.npz') as trace:
            native=np.r_[trace['actor_input'][-1,:3]*4,trace['gravity'][-1],trace['q'][-1,LEGS],trace['dq'][-1]].astype(np.float32)
            probe=CExecution(validation['library'],nominal(),data['initial_motor'])
            try:
                if c_input_valid(probe,native,data['x'][data['ix'],22:25]*[4,4,.5]):
                    raise ValueError('parent input-guard explanation not reproduced')
            finally:probe.close()
        inherited_steps=sum(r['physics_steps'] for r in inherited)
        if inherited_steps!=ledger['used_physics_steps']:raise ValueError('parent physical use mismatch')
        inherited_wall=ledger['wall_used_s']
        request['continuation']=dict(parent=str(parent),parent_attempt=ledger['attempt_id'],
            parent_ledger_sha256=sha(parent/'ledger.json'),parent_summary_sha256=sha(parent/'summary.json'),
            inherited_physics_steps=inherited_steps,inherited_wall_s=inherited_wall,
            reinterpretation='only child interpretation: pd_age20 ended at C input guard; original FAILED/raw unchanged',
            repeated_physics_steps=0)
    if args.check:
        if args.run: raise ValueError('--check and --run are mutually exclusive')
        model,refs,base,readback=setup(spec,args)
        d,fit=initialize_from_log(model,refs,data['x'][data['ix']],0.)
        print(json.dumps(dict(status='CPU_PREFLIGHT_PASS',physics_steps=0,files_written=0,
                              initial_fit=fit,plant_version=readback['plant']['version'],
                              seconds=args.seconds,single_cases=len(cases)),indent=2));return
    if not args.run:
        print(json.dumps(request,indent=2,ensure_ascii=False)); return
    from native_stand.command_v6.domain.experiment_naming import new_experiment_name,validate_new_experiment_name
    purpose='motion31_force_compliance_search' if args.matrix=='separated_forces_v1' else ('motion31_source_packet_search' if args.matrix=='source_packets_v1' else 'motion31_oscillation_search')
    out=args.output_dir or Path('sim2sim')/new_experiment_name(purpose)
    validate_new_experiment_name(out.name); out.mkdir(parents=True,exist_ok=False)
    if parent:
        claimdir=parent.parent/'.motion31_oscillation_claims';claimdir.mkdir(exist_ok=True)
        record=request.get('completion',request.get('continuation'))
        with (claimdir/(record['parent_attempt']+'.json')).open('x') as claim:
            json.dump(dict(child=str(out.resolve()),parent=str(parent)),claim)
    import shutil
    if shutil.disk_usage(out).free < (args.storage_mib+512)*2**20:
        raise RuntimeError('storage reserve insufficient')
    import torch,onnx
    from onnx.reference import ReferenceEvaluator
    torch.set_num_threads(1)
    runtime=ReferenceEvaluator(onnx.load(args.onnx))
    actor=lambda z: runtime.run(None,{'observation':z.astype(np.float32)})[0]
    active = np.flatnonzero((data['t']>=data['t'][data['ix']])&(data['t']<data['t'][data['ix']]+args.seconds))
    prediction=np.concatenate([actor(data['x'][i:i+1]) for i in active])
    if abs(prediction-data['y'][active]).max()>1e-5:
        raise ValueError('selected ONNX does not reproduce this active log')
    root=Path(__file__).resolve().parents[1]
    paths=[args.csv,args.onnx,args.spec,args.xml,args.firmware_validation,Path(__file__),
           Path(spec['asset_path']),Path(spec['colleague_config_path'])]
    paths += [root/p for p in ('src/native_stand/students/oscillation_search.py',
               'src/native_stand/students/firmware_control.py','src/native_stand/students/real2sim_delay.py',
               'src/native_stand/students/real2sim_handover.py','src/native_stand/students/mujoco_solver.py',
               'src/native_stand/students/motion_mcu_execution.py',
               'scripts/motion_mcu_playback.py','scripts/compare_motion_timing_profiles.py',
               'scripts/validate_native_mujoco.py','scripts/diagnose_motion_real2sim.py')]
    if args.matrix in ('source_packets_v1','separated_forces_v1'):
        paths+=[root/p for p in ('src/native_stand/students/firmware_sampling.py',
            'src/native_stand/students/firmware_sampling_bridge.c','scripts/validate_motion_sampling.py')]
    if args.matrix=='separated_forces_v1':paths.append(root/'src/native_stand/students/firmware_mechanics.py')
    source_hashes={str(p.resolve()):sha(p) for p in paths}
    started=time.monotonic();deadline=started+max(0.,args.wall_seconds-inherited_wall)
    budget=dict(attempt_id=uuid.uuid4().hex,status='RUNNING',used_physics_steps=inherited_steps,completed_cases=len(inherited),
                maximum_physics_steps=request['maximum_physics_steps'],maximum_cases=args.max_cases,
                wall_limit_s=args.wall_seconds,storage_limit_bytes=request['storage_limit_bytes'],
                online_PPO=0,optimizer_steps=0,attempts=2 if parent else 1,
                inherited_physics_steps=inherited_steps,inherited_wall_s=inherited_wall)
    write(out/'spec.json',request); write(out/'ledger.json',budget)
    write(out/'identity.json',dict(input_sha256=source_hashes,firmware=validation,
                                 log_network_max_error=float(abs(prediction-data['y'][active]).max())))
    try:
        model,refs,base,readback=setup(spec,args);write(out/'plant_readback.json',readback)
        real_args=(data['t']-data['t'][data['ix']],data['x'][:,6:10]*1.5,data['x'][:,10:16]*20,
                   data['y'],data['tau'],data['x'][:,3:6])
        real_fit=features(*real_args,window=(.5,2.));real_hold=features(*real_args,window=(2.,args.seconds))
        write(out/'real_features.json',dict(fit=real_fit,holdout=real_hold))
    except BaseException as exc:
        budget.update(status='FAILED',failure=type(exc).__name__+': '+str(exc),wall_used_s=time.monotonic()-started)
        write(out/'ledger.json',budget)
        write(out/'summary.json',dict(status='FAILED',failure=budget['failure'],results=[],physical_steps=0))
        raise
    results=inherited; traces={}; terminal='FAILED';failure=None
    def evaluate(case):
        if len(results)>=args.max_cases or time.monotonic()>=deadline: return False
        row,arrays=run_case(model,refs,base,actor,data,validation['library'],case,args.seconds,deadline)
        budget['used_physics_steps']+=row['physics_steps']
        if row.get('runtime_fault'):
            row.update(fit=dict(score=100.,coverage=0.,oscillation_candidate=False),
                       holdout=dict(score=100.,coverage=0.,oscillation_candidate=False))
            results.append(row);budget['completed_cases']=len(results)
            traces[case['name']]=arrays
            write(out/(case['name']+'.json'),row);write(out/'ledger.json',budget)
            raise RuntimeError(row['runtime_fault'])
        if arrays:
            a=arrays
            common=(a['time'],a['q'][:,LEGS],a['dq'],a['mean'],a['command'],a['gravity'])
            row['features_fit']=features(*common,window=(.5,2.))
            row['features_holdout']=features(*common,window=(2.,args.seconds))
        else: row.update(features_fit=dict(valid=False),features_holdout=dict(valid=False))
        row['fit']=similarity(real_fit,row['features_fit'])
        row['holdout']=similarity(real_hold,row['features_holdout'])
        if args.matrix=='separated_forces_v1':
            from native_stand.students.firmware_mechanics import reproduction
            row['reproduction_fit']=reproduction(real_fit,row['features_fit'])
            row['reproduction_holdout']=reproduction(real_hold,row['features_holdout'])
        results.append(row);budget['completed_cases']=len(results)
        write(out/(case['name']+'.json'),row);write(out/'ledger.json',budget)
        traces[case['name']]=arrays
        ranked=sorted(results,key=lambda r:r['fit']['score'])[:6]
        keep={r['name'] for r in ranked}|{'nominal','legP20','wheelKv015','legP20_wheelKv015',
            'dyn_nominal','dyn_legP20','dyn_wheelKv015','dyn_legP20_wheelKv015',
            'source_nominal','source_P20','source_Kv015','source_P20_Kv015'}
        if args.matrix=='separated_forces_v1':
            keep|={r['name'] for r in results if r['family']=='force_route_pair'}
            keep|=set([r['name'] for r in results if r.get('holdout',{}).get('oscillation_candidate')][:4])
        # Retain a physically meaningful partial failure boundary for inspection.
        if row['termination'] is not None:keep.add(case['name'])
        for name in list(traces):
            if name not in keep: del traces[name]
        parent_bytes=request.get('completion',{}).get('inherited_artifact_bytes',0)
        if parent_bytes+sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>request['storage_limit_bytes']-16*2**20:
            raise RuntimeError('storage reserve reached')
        print(f"{len(results):02d} {case['name']} fit={row['fit']['score']:.3f} hold={row['holdout']['score']:.3f} {row['termination']}",flush=True)
        return row['termination']!='WALL_BUDGET'
    try:
        for c in cases:
            if c['name'] in {r['name'] for r in results}:continue
            if not evaluate(c): break
        if args.matrix=='separated_forces_v1' and len(results)==len(cases) and time.monotonic()<deadline:
            from native_stand.students.firmware_mechanics import followup_cases
            for case in followup_cases(results,limit=12,initial_motor=data['initial_motor']):
                if not evaluate(case):break
        if args.matrix not in ('source_packets_v1','separated_forces_v1') and len(results)==len(cases) and time.monotonic()<deadline:
            leaders=sorted([r for r in results if r['family'] not in ('gain','baseline')],key=lambda r:r['fit']['score'])[:6]
            for c in combinations(leaders,limit=8):
                if not evaluate(c): break
            leaders=sorted([r for r in results if r['family']!='gain'],key=lambda r:r['fit']['score'])[:4]
            # Gain response is a qualitative user observation, not fit evidence.
            for leader in leaders:
                p=copy.deepcopy(leader['parameters']);p.update(kp=20.,kv=.15)
                c=dict(name='lowgain_'+leader['name'],family='gain_confirmation',parameters=p,parent=leader['name'])
                if 'dynamics' in leader:c['dynamics']=copy.deepcopy(leader['dynamics'])
                if not evaluate(c): break
        terminal='STOPPED_BOUNDED' if len(results)>=args.max_cases or time.monotonic()>=deadline else 'COMPLETED'
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        raise
    finally:
        for name,arrays in traces.items():
            if arrays: np.savez_compressed(out/(name+'.npz'),**arrays)
        unchanged=all(sha(p)==v for p,v in source_hashes.items())
        if not unchanged: terminal='FAILED';failure='source identity changed during execution'
        budget.update(status=terminal,failure=failure,wall_used_s=time.monotonic()-started+inherited_wall,
                      new_physics_steps=budget['used_physics_steps']-inherited_steps,
                      retained_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                      inherited_artifact_bytes=request.get('completion',{}).get('inherited_artifact_bytes',0))
        write(out/'ledger.json',budget)
        write(out/'summary.json',dict(status=terminal,failure=failure,results=results,
            ranked_by_fit=[r['name'] for r in sorted(results,key=lambda r:r['fit']['score'])],
            source_unchanged=unchanged,hardware_identified=False,policy_accepted=False,
            interpretation='Only candidate mechanisms; no actual motor/feedback timing identification'))
    print('OUTPUT='+str(out.resolve()),flush=True)


if __name__=='__main__': main()
