"""Adapters to the public MuJoCo plant, MCU chain and closed-loop evaluator.

Loading/forward kinematics is CPU inspection. Only collect/pair/evaluate call
mj_step, and these entrypoints require a ledger-owned experiment attempt.
"""
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import torch
from .state import initial


ROOT = Path(__file__).resolve().parents[3]
TEACHER = 'runs/2026_10_25-23_52__motion1500_posture_retention312_10000/formal/model_update_4500.pt'
TEACHER_SHA = '0be977b308cfe493fdddee54823e21c713508fa582ce1618ab34e57b13fbc619'
PARENT = 'configs/experiments/2026_10_25-23_52__motion1500_posture_retention312_10000/spec.json'
SMALL = 'artifacts/2026_10_26-20_29__motion4500_precise_fidelity_dagger/round_00/fit/best_validation.pt'
SMALL_SHA = 'efad012efcced3a9ac58dc51839dea98b81598de289dea8ef0986a0813d139a1'
XML = '/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sources():
    result = {}
    for p, expected in ((ROOT/TEACHER, TEACHER_SHA), (ROOT/SMALL, SMALL_SHA)):
        found = sha(p)
        if found != expected:
            raise ValueError('frozen source checkpoint identity differs: '+str(p))
        result[str(p)] = found
    parent = json.loads((ROOT/PARENT).read_text())
    for p, expected in ((Path(parent['asset_path']), parent['asset_sha256']),
                        (Path(parent['colleague_config_path']), parent['colleague_config_sha256'])):
        if sha(p) != expected:
            raise ValueError('source plant identity differs: '+str(p))
        result[str(p)] = expected
    result[str(ROOT/PARENT)] = sha(ROOT/PARENT)
    result[XML] = sha(XML)
    extra={
        ROOT/'deployment_protocol/2026_10_26-20_29__motion4500_precise_small31_onnx/actor.onnx':
            '776201d861dbf1f00b6824f2c2615068a3206eb45b38529e65e7591bc150e617',
        ROOT/'deployment_protocol/2026_10_23-19_53__motion500_currentplant_small31_onnx/actor.onnx':
            '5b67781bdaa86b666683e315f685a0f19c45fee918da70439ea3bb43d0fbb03a'}
    for p,h in extra.items():
        if sha(p)!=h: raise ValueError('deployment comparison identity differs')
        result[str(p)]=h
    # Bind all mesh bytes too; XML alone does not identify the effective mechanics.
    import xml.etree.ElementTree as ET
    xml = ET.parse(XML).getroot()
    meshdir = xml.find('compiler').get('meshdir', '')
    for element in xml.findall('./asset/mesh'):
        p = Path(XML).parent/meshdir/element.get('file')
        result[str(p.resolve())] = sha(p)
    retention=ROOT/'artifacts/2026_10_26-20_29__motion4500_precise_fidelity_dagger/teacher'
    for p in sorted(retention.glob('split0_case*.json')):
        r=json.loads(p.read_text())
        if r.get('variant')!=0 or r.get('initial_yaw_rad')!=0.:continue
        trace=p.parent/r['trace']
        if sha(trace)!=r['trace_sha256']:raise ValueError('frozen retention episode SHA differs')
        result[str(p)]=sha(p);result[str(trace)]=r['trace_sha256']
    return parent, result


def actors():
    from native_stand.students.model import Student
    from native_stand.students.motion_distillation_pipeline import validate_teacher
    teacher, metadata, _ = validate_teacher(ROOT/TEACHER, ROOT/PARENT)
    student_data = torch.load(ROOT/SMALL, map_location='cpu', weights_only=False)
    # BC has explicit distillation identity and intentionally no recoverable PPO Critic.
    student = Student(True, version='multiskill_mcu_v2')
    weights = student_data.get('actor', student_data.get('model_state_dict'))
    if weights is None:
        raise ValueError('explicit distillation Actor weights absent')
    student.load_state_dict(weights)
    for model in (teacher, student):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    from diagnose_motion_closed_loop_robustness import OnnxMean
    old_path=ROOT/'deployment_protocol/2026_10_23-19_53__motion500_currentplant_small31_onnx/actor.onnx'
    if sha(old_path)!='5b67781bdaa86b666683e315f685a0f19c45fee918da70439ea3bb43d0fbb03a':
        raise ValueError('historical failure-control identity differs')
    return {'teacher4500': teacher, 'small4500': student,'old500':OnnxMean(old_path)}, metadata


def load_plant():
    import mujoco
    from motion_mcu_playback import configuration, declared, apply_current_plant
    from validate_native_mujoco import binding
    from native_stand.students.mujoco_solver import apply
    parent, identities = sources()
    cfg = configuration(SimpleNamespace(leg_armature_m=.012, zero_latency=False), declared(parent), parent)
    model = mujoco.MjModel.from_xml_path(XML)
    refs = binding(model)
    plant = apply_current_plant(model, refs, parent, cfg)
    solver = apply(model, refs, 'closure_contact_tight_v1')
    # Support the audited one-free-joint/hinge topology; fail closed on unknown backends.
    if (model.na or model.nmocap or model.nplugin or model.nflex or
        np.any(model.tendon_stiffness) or np.any(model.tendon_damping) or
        np.any(model.tendon_frictionloss) or np.any(model.tendon_limited)):
        raise ValueError('unsupported hidden mechanical/activation/plugin state')
    if int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST):
        raise ValueError('effective integrator identity drift')
    return model, refs, cfg, dict(input_sha256=identities, plant=plant, solver=solver,
        actuator_force_range=model.actuator_forcerange.tolist(),actuator_force_limited=model.actuator_forcelimited.tolist(),
        integrator='implicitfast', engine_version=mujoco.__version__)


def description(model, refs):
    """Small numeric kinematic description, never a copied source/asset tree."""
    import mujoco
    body_jq, body_jv = [-1]*model.nbody, [-1]*model.nbody
    axes, anchors = np.zeros((model.nbody, 3)), np.zeros((model.nbody, 3))
    for body in range(1, model.nbody):
        if model.body_jntnum[body] > 1:
            raise ValueError('multiple joints per body need a new kinematic schema')
        if model.body_jntnum[body]:
            j = int(model.body_jntadr[body])
            if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError('unsupported joint type')
            body_jq[body] = int(model.jnt_qposadr[j]); body_jv[body] = int(model.jnt_dofadr[j])
            axes[body], anchors[body] = model.jnt_axis[j], model.jnt_pos[j]
    limits = model.jnt_range[refs['joints']].copy()
    limits[~model.jnt_limited[refs['joints']].astype(bool)] = [-float('inf'), float('inf')]
    # JSON finite infinities represented by deployment envelope for unbounded wheels;
    # only LEGS are clipped, so wheel entries are immaterial and explicitly finite.
    limits[[2, 5]] = [-1e9, 1e9]
    contact_geoms = [i for i in range(model.ngeom)
                     if model.geom_contype[i] == 1 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER]
    all_limits=np.zeros((model.nq-7,2));all_limited=np.zeros(model.nq-7,dtype=bool)
    for j in range(model.njnt):
        address=int(model.jnt_qposadr[j])
        if address>=7:all_limits[address-7]=model.jnt_range[j];all_limited[address-7]=bool(model.jnt_limited[j])
    result={k: np.asarray(v).tolist() for k, v in dict(
        parent=model.body_parentid, pos=model.body_pos, quat=model.body_quat, axis=axes, anchor=anchors,
        ipos=model.body_ipos, iquat=model.body_iquat, mass=model.body_mass, inertia=model.body_inertia,
        q0=model.qpos0, joint_q=body_jq, joint_v=body_jv, site_body=model.site_bodyid,
        site_pos=model.site_pos, geom_pos=model.geom_pos, geom_quat=model.geom_quat, geom_size=model.geom_size,
        geom_body=model.geom_bodyid, armature=model.dof_armature, active_q=refs['qpos'], active_v=refs['dofs'],
        root=refs['root'], wheels=refs['wheels'], hips=model.jnt_bodyid[np.asarray(refs['joints'])[[0, 3]]],
        loops=refs['loops'], contact_geoms=contact_geoms, limits=limits,
        all_joint_limits=all_limits,all_joint_limited=all_limited,
        solver_actuator_limits=np.minimum(np.abs(model.actuator_ctrlrange[refs['actuators']]).min(-1),
            np.abs(model.actuator_forcerange[refs['actuators']]).min(-1))*np.abs(refs['gears'])).items()}
    result['joint_names']=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt)]
    result['body_names']=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(model.nbody)]
    result['qpos_address']=model.jnt_qposadr.tolist();result['qvel_address']=model.jnt_dofadr.tolist()
    return result


CONTEXT_ARRAYS = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_pos', 'body_quat',
    'dof_armature', 'dof_damping', 'dof_frictionloss', 'jnt_range', 'jnt_limited', 'jnt_stiffness',
    'qpos_spring', 'geom_size', 'geom_pos', 'geom_quat', 'geom_friction', 'geom_margin', 'geom_gap',
    'geom_solref', 'geom_solimp', 'geom_condim', 'geom_contype', 'geom_conaffinity',
    'eq_solref', 'eq_solimp', 'eq_data', 'actuator_gear', 'actuator_ctrlrange', 'actuator_forcerange',
    'actuator_ctrllimited','actuator_forcelimited','actuator_gainprm','actuator_biasprm',
    'tendon_stiffness','tendon_damping','tendon_frictionloss','tendon_limited')


def context(model):
    arrays = {k: np.asarray(getattr(model, k), dtype=np.float32).reshape(-1) for k in CONTEXT_ARRAYS}
    arrays['options'] = np.asarray([model.opt.timestep, model.opt.integrator, model.opt.solver,
        model.opt.iterations, model.opt.tolerance, model.opt.ls_iterations, model.opt.ls_tolerance,
        model.opt.noslip_iterations, model.opt.disableflags, model.opt.enableflags, *model.opt.gravity], np.float32)
    flat = np.concatenate(list(arrays.values()))
    return torch.from_numpy(flat)[None], {k: len(v) for k, v in arrays.items()}


def integration_state(model, data):
    import mujoco
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, kind))
    mujoco.mj_getState(model, data, state, kind)
    return state


def root_contact_force(model,data,refs):
    import mujoco
    total=np.zeros(3)
    for index,c in enumerate(data.contact):
        if refs['ground'] not in (c.geom1,c.geom2):continue
        other=c.geom2 if c.geom1==refs['ground'] else c.geom1
        if int(model.geom_bodyid[other])!=refs['root']:continue
        force=np.zeros(6);mujoco.mj_contactForce(model,data,index,force)
        total+=(np.asarray(c.frame).reshape(3,3).T@force[:3])*(1 if c.geom1==refs['ground'] else -1)
    return np.asarray([np.linalg.norm(total)])


def export_state(model, data, refs, engine, history, template, next_tick):
    """No reset or queue reconstruction by guessing: read every causal bank."""
    from validate_native_mujoco import state as measured_state
    def tensor(x):
        return torch.as_tensor(np.array(x, copy=True), dtype=torch.float32)[None]
    age = np.arange(1, 65)
    sensors = np.stack([history.at(next_tick-int(a)) for a in age])
    queue = engine.ring.history[(next_tick-torch.arange(1, 65)) % 64, 0].clone()[None]
    _, _, measured = measured_state(model, data, refs)
    values = dict(q=tensor(data.qpos), v=tensor(data.qvel), warm=tensor(data.qacc_warmstart),
        contact=tensor(measured['contact_force_n']),body_contact=tensor(root_contact_force(model,data,refs)), sensors=tensor(sensors), queue=queue,
        tick=torch.tensor([next_tick], dtype=torch.long), driver_cap=engine.driver_cap[None].clone(),
        delay=engine.ring.delay.clone(), tau=engine.ring.tau.clone(),
        generalized=tensor(data.actuator_force[refs['actuators']]*refs['gears']))
    for src, dst in (('pending','pending'), ('adopted','adopted'), ('driver','driver'),
                     ('motor_actual','motor'), ('pre','pd_request'), ('pd_limited','pd_limited'),
                     ('compensation','compensation'), ('spring_actual','spring'), ('actual','applied')):
        values[dst] = getattr(engine, src)[None].clone()
    return replace(template, **values)


def restore(model, refs, cfg, s, raw_integration):
    """Exact simulator integration blob + explicit MCU/feedback state, no settling."""
    import mujoco
    from diagnose_motion_closed_loop_robustness import UncertainExecution
    from native_stand.students.motion_execution_uncertainty import nominal
    from native_stand.students.real2sim_delay import ObservationHistory
    data = mujoco.MjData(model)
    blobs=np.asarray(raw_integration,dtype=np.float64)
    if blobs.ndim!=2 or blobs.shape[0]!=2:raise ValueError('both current and last-solver integration blobs required')
    mujoco.mj_setState(model, data, blobs[1], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    # Restore post-integration coordinates while preserving original cached
    # last-solver body/contact readback. No extra forward in the control path.
    mujoco.mj_setState(model, data, blobs[0], mujoco.mjtState.mjSTATE_INTEGRATION)
    u = nominal(); u.update(PD_age_ms=int(s.pd_age[0]), actor_age_ms=int(s.actor_age[0]),
        motor_delay_ms=s.delay[0].tolist(), motor_tau_ms=s.tau[0].tolist(), spring_scale=float(s.spring_scale[0, 0]))
    calibration = dict(version='motion31_split_encoder_offsets_v1',
        actor_leg_zero_bias_rad=s.actor_bias[0, [0,1,3,4]].tolist(),
        pd_leg_zero_bias_rad=s.pd_bias[0, [0,1,3,4]].tolist())
    engine = UncertainExecution(cfg['mcu_execution'], data.qpos[refs['qpos']], s.adopted[0].numpy(), u,
                                calibration=calibration)
    for src, dst in (('pending','pending'), ('adopted','adopted'), ('adopted','previous'),
                     ('driver','driver'), ('motor','motor_actual'), ('pd_request','pre'),
                     ('pd_limited','pd_limited'), ('compensation','compensation'), ('spring','spring_actual'), ('applied','actual')):
        setattr(engine, dst, getattr(s, src)[0].clone())
    tick = int(s.tick[0]); engine.ring.last_tick = tick-1 if tick else None
    for age in range(1, 65):
        engine.ring.history[(tick-age) % 64, 0] = s.queue[0, age-1]
    engine.ring.actual = s.motor.clone(); engine.driver_cap = s.driver_cap[0].clone()
    history = ObservationHistory(s.sensors[0, -1].numpy(), initial_tick=tick-64)
    for age in range(63, 0, -1):
        history.append(tick-age, s.sensors[0, age-1].numpy())
    engine.history = ObservationHistory(np.r_[s.sensors[0, -1, 6:12].numpy()+s.pd_bias[0].numpy(),
                                             s.sensors[0, -1, 12:18].numpy()], initial_tick=tick-64)
    for age in range(63, 0, -1):
        engine.history.append(tick-age, np.r_[s.sensors[0, age-1, 6:12].numpy()+s.pd_bias[0].numpy(),
                                             s.sensors[0, age-1, 12:18].numpy()])
    return data, engine, history


def collect_episode(model, refs, cfg, actor_map, case, geometry, reward, budget, *, output):
    """Persist last complete state on engineering faults, without labeling a reset."""
    capture={}
    try:
        return _collect_episode(model,refs,cfg,actor_map,case,geometry,reward,budget,output=output,capture=capture)
    except BaseException as error:
        if capture:
            try:
                p=Path(output);p.mkdir(parents=True,exist_ok=True)
                torch.save(dict(version='FAILED_PARTIAL_NOT_TRAINING_EPISODE',error=repr(error),
                    case=case,**capture),p/'fault_last_state.pt')
            except Exception as write_error:
                import sys
                print('fault_state_save_failure: '+repr(write_error),file=sys.stderr,flush=True)
        raise


def _collect_episode(model, refs, cfg, actor_map, case, geometry, reward, budget, *, output,capture):
    """One finite real episode, dense1ms labels + complete100Hz imagination starts."""
    import mujoco
    from diagnose_motion_closed_loop_robustness import UncertainExecution
    from motion_mcu_playback import apply_initial_pose
    from diagnose_motion_real2sim import sensors
    from validate_native_mujoco import state as measured_state
    from native_stand.students.real2sim_delay import ObservationHistory
    from .execution import observation, decode,measurements
    from .dataset import write_episode
    budget.ensure('episodes',1)
    data = mujoco.MjData(model)
    q0 = apply_initial_pose(model, data, refs, cfg['mcu_execution'])
    if case['yaw']:
        from scipy.spatial.transform import Rotation
        r = Rotation.from_rotvec([0., 0., case['yaw']])*Rotation.from_quat(data.qpos[3:7][[1,2,3,0]])
        data.qpos[3:7] = r.as_quat()[[3,0,1,2]]
    mujoco.mj_forward(model, data)
    u = case['uncertainty']
    root = refs['root']; free = int(model.body_jntadr[root]); dof = int(model.jnt_dofadr[free])
    data.qvel[dof:dof+3] = data.xmat[root].reshape(3,3) @ np.array([0.,0.,-u['initial_vx_mps']])
    data.qvel[dof+3:dof+6] = [u['initial_pitch_rate_rad_s'],0.,0.]
    mujoco.mj_forward(model, data)
    target = q0.astype(np.float32); target[[2,5]] = 0
    engine = UncertainExecution(cfg['mcu_execution'], q0, target, u)
    history = ObservationHistory(sensors(model, data, refs), initial_tick=0)
    ctx, _ = context(model)
    template = initial(torch.tensor(data.qpos, dtype=torch.float32)[None],
        torch.tensor(data.qvel, dtype=torch.float32)[None], torch.tensor(sensors(model,data,refs), dtype=torch.float32)[None],
        ctx, torch.tensor([[0.,0.,.34,0.]]), compensation=engine.compensation[None],
        cap=engine.driver_cap[None], delay=engine.ring.delay, tau=engine.ring.tau,
        actor_age=u['actor_age_ms'], pd_age=u['PD_age_ms'])
    template = replace(template, target=torch.tensor([case['command']], dtype=torch.float32),
                       spring_scale=torch.tensor([[u['spring_scale']]], dtype=torch.float32))
    rows, micro, blobs = [], [], []
    solver_blob=integration_state(model,data)
    rng = np.random.default_rng(case['seed'])
    duration = int(round(case['seconds']*1000))
    for tick in range(0, duration, 10):
        budget.check()
        history.append(tick, sensors(model, data, refs))
        before = export_state(model,data,refs,engine,history,template,tick)
        predecessor = case.get('predecessor') if tick < case.get('handover_ms',0) else None
        name = predecessor or case['controller']
        before = replace(before, owner=torch.tensor([0 if predecessor else 1]))
        m = measurements(before,geometry); obs = observation(before,m)
        with torch.no_grad():
            action = actor_map[name].mean(obs)
        if case.get('action_noise',0):
            noise = np.clip(rng.normal(size=(1,6)), -2,2)*case['action_noise']
            action = (action+torch.tensor(noise,dtype=action.dtype)).clamp(-1,1)
        target = decode(action, geometry, before)[0].detach().numpy()
        raw = integration_state(model,data)
        blobs.append(np.stack((raw,solver_blob)))
        capture.update(before=before.as_dict(),integration_state=blobs[-1],tick=tick)
        for sub in range(10):
            t = tick+sub
            current = sensors(model,data,refs); history.append(t,current)
            pre = (data.qpos.copy(),data.qvel.copy(),data.qacc_warmstart.copy())
            engine.step(t,target,data.qpos[refs['qpos']].copy(),data.qvel[refs['dofs']].copy())
            if not engine.geometry_valid:
                raise RuntimeError('physical execution spring geometry invalid')
            data.ctrl[refs['actuators']] = engine.actual.numpy()/refs['gears']
            solver_blob=integration_state(model,data)
            budget.charge('physics_steps',1)  # reserve/charge BEFORE dispatch, even if engine raises
            mujoco.mj_step(model,data)
            numerical = integration_state(model,data)
            _,_,measured = measured_state(model,data,refs)
            if not np.isfinite(numerical).all() or measured['closure_m'] > .005:
                raise RuntimeError('nonfinite/closure physical fault; stop experiment')
            micro.append(dict(q=pre[0],v=pre[1],warm=pre[2],
                applied=(data.actuator_force[refs['actuators']]*refs['gears']).copy(),
                next_q=data.qpos.copy(),next_v=data.qvel.copy(),next_warm=data.qacc_warmstart.copy(),
                contact=np.asarray(measured['contact_force_n']),body_contact=root_contact_force(model,data,refs)))
        latest=replace(before,solver_q=torch.tensor(pre[0],dtype=torch.float32)[None],
                       solver_v=torch.tensor(pre[1],dtype=torch.float32)[None])
        after = export_state(model,data,refs,engine,history,latest,tick+10)
        measured = measurements(after,geometry)
        after, scores, terminal = reward(before,after,measured)
        truncated = tick+10 >= duration and not bool(terminal[0])
        rows.append(dict(before=before,after=after,action=action.detach(),observation=obs.detach(),
                         reward=scores['total'].detach(),terminated=terminal.detach(),
                         truncated=torch.tensor([truncated]),reward_raw=scores['raw'].detach(),
                         reward_density=scores['density'].detach(),reward_dt=scores['dt'].detach(),
                         reward_fields=scores['fields']))
        template = after
        if bool(terminal[0]):
            break                         # terminal stays paired with its PRE-RESET next state
    budget.charge('episodes',1)
    return write_episode(output,case,rows,micro,blobs,geometry.description,budget)


def evaluate_case(model, refs, cfg, actors_, case, budget):
    from diagnose_motion_closed_loop_robustness import run_case, metrics
    from native_stand.students.mujoco_solver import ConstraintRecorder
    recorder=ConstraintRecorder()
    def observe(m,d,r):
        import mujoco
        from native_stand.students.motion_yaw_quality import measurement
        budget.check()
        extra=recorder(m,d,r)
        rotation=d.xmat[r['root']].reshape(3,3)
        hips=d.xanchor[np.asarray(r['joints'])[[0,3]]]
        legs=(d.xpos[r['wheels']]-hips)@rotation
        swing=np.arctan2(-legs[:,2],-legs[:,1])
        velocities=[]
        for body in r['wheels']:
            value=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body,value,0);velocities.append(value)
        velocities=np.asarray(velocities)
        tensor=lambda a:torch.tensor(np.asarray(a),dtype=torch.float32)[None]
        spin=measurement(tensor(d.xpos[r['wheels']]),tensor(d.xipos[r['wheels']]),
            tensor(velocities[:,3:]),tensor(velocities[:,:3]),tensor(d.xmat[r['wheels']].reshape(-1,3,3)[:,:,0]),
            tensor(-rotation[:,2]))
        extra.update(qpos_full=d.qpos.copy(),qvel_full=d.qvel.copy(),
            generalized_actual=(d.actuator_force[r['actuators']]*r['gears']).copy(),
            leg_swing_difference_rad=np.arctan2(np.sin(swing[0]-swing[1]),np.cos(swing[0]-swing[1])),
            support_point_velocity_xy_mps=spin['tangent_velocity_w'][0].numpy(),
            rolling_speed_mps=spin['rolling_speed_forward_mps'][0].numpy())
        return extra
    # Existing authoritative loop and window scorer; no imagined state is fed here.
    # Reserve upper bound BEFORE the existing evaluator, settle its actual steps below.
    max_steps = round((case['seconds']+case.get('warmup',0))*1000)
    budget.ensure('physics_steps', max_steps)
    budget.ensure('episodes',1)
    with torch.inference_mode():
        result, arrays = run_case(model, refs, actors_, cfg, case['uncertainty'], case['controller'],
            case['seconds'], budget.deadline, predecessor=case.get('predecessor'),
            warmup=case.get('warmup',0), target=case['command']+[.34], initial_yaw=case['yaw'],constraint_observer=observe)
    # run_case records100Hz, including early behavior terminations. On failure charge reservation.
    actual = int(round(float(arrays['time'][-1])*1000)) if len(arrays.get('time',[])) else max_steps
    budget.charge('physics_steps',actual)
    budget.charge('episodes',1)
    origin = case.get('warmup',0)
    report = dict(result=result,windows={f'{a:g}_{b:g}':metrics(arrays,a,b,origin)
        for a,b in ((0.,.5),(.5,2.),(2.,30.))}, case=case,
        imagined=False,scope='same-MuJoCo-condition comparison; Native/hardware NOT_RUN')
    return arrays, report


def paired_return(model,refs,cfg,start,integration,first_action,actor,critic,geometry,reward,budget,horizon=8):
    """Restore the SAME full physical state; no reset or controller prefix."""
    import mujoco
    from diagnose_motion_real2sim import sensors
    from .execution import observation,decode,measurements
    budget.ensure('paired_physics_rollouts',1)
    data,engine,history=restore(model,refs,cfg,start,integration)
    s=start.clone();total=0.;alive=True;events=[]
    with torch.no_grad():
        for k in range(horizon):
            budget.check()
            tick=int(s.tick[0]);history.append(tick,sensors(model,data,refs))
            b=export_state(model,data,refs,engine,history,s,tick)
            action=first_action if k==0 else actor.mean(observation(b,measurements(b,geometry)))
            requested=decode(action,geometry,b)[0].numpy()
            for sub in range(10):
                t=tick+sub;history.append(t,sensors(model,data,refs))
                engine.step(t,requested,data.qpos[refs['qpos']],data.qvel[refs['dofs']])
                data.ctrl[refs['actuators']]=engine.actual.numpy()/refs['gears']
                pq,pv=data.qpos.copy(),data.qvel.copy()
                budget.charge('physics_steps',1);mujoco.mj_step(model,data)
                numerical=integration_state(model,data)
                if not np.isfinite(numerical).all() or not engine.geometry_valid:
                    raise RuntimeError('physical paired-gradient numerical fault')
            latest=replace(b,solver_q=torch.tensor(pq,dtype=torch.float32)[None],solver_v=torch.tensor(pv,dtype=torch.float32)[None])
            a=export_state(model,data,refs,engine,history,latest,tick+10)
            m=measurements(a,geometry)
            if float(m['closure'].norm(dim=-1).max())>.005:raise RuntimeError('physical paired-gradient closure fault')
            s,terms,done=reward(b,a,m)
            total+=(.99**k)*float(terms['total'][0])*alive
            events.append((s.contact>2).numpy())
            alive=alive and not bool(done[0])
            if not alive:break
        value=float(critic(s)[0]) if alive else 0.
    budget.charge('paired_physics_rollouts',1)
    return dict(return_with_value=total+(.99**horizon)*value,reward_only=total,
                contact_events=np.stack(events),terminated=not alive)
