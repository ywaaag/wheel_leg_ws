"""Process-isolated MuJoCo sampling through the existing real C execution path.

Workers pause inside the10ms Actor callback. The learner sees the exact C input
before supplying a raw Gaussian action; only the environment clips publication.
"""
import multiprocessing as mp
import os
import traceback
from collections import deque
from pathlib import Path
import numpy as np
import torch
from .motion_compliance_training import CRITIC,PROFILES,assignments,critic_input

LEGS=[0,1,3,4];WHEELS=[2,5]


class _Reset(Exception):
    def __init__(self,row):self.row=row


def _worker(connection,spec,index):
    # spawn gives this worker its own DSO globals and actual driver filters.
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ex=None
    try:
        import json,sys,mujoco
        from types import SimpleNamespace as S
        root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'scripts'))
        from validate_native_mujoco import binding,state
        from motion_mcu_playback import apply_current_plant,configuration,declared,apply_initial_pose
        from native_stand.students.mujoco_solver import apply as solver
        from .firmware_sampling import profile,source_parameters,PacketEnvelopeError
        from .firmware_mechanics import MechanicalExecution,ForceRouter,MechanicalEnvelopeError
        from .motion_quality import QualityTermination,CAUSES
        from .motion_yaw_quality import measurement
        from .model import load
        from .motion_compliance_reward import compute
        from search_motion31_oscillation import c_input_valid
        from .motion_mcu_execution import WHEEL_CAP
        from .motion_response_probe import declaration,bootstrap_case,episode_parameters,prepare_colleague,array_summary
        probe=declaration(spec)
        source=json.loads(Path(spec['parent_spec']).read_text())
        frozen,_=load(spec['source_checkpoint'],'MotionZeroStudent');frozen.eval()
        model=mujoco.MjModel.from_xml_path(spec['xml']);refs=binding(model)
        cfg=configuration(S(leg_armature_m=None,zero_latency=False),declared(source),source)
        if probe and probe['asset']=='colleague':
            plant=prepare_colleague(model,refs)
        else:
            plant=apply_current_plant(model,refs,source,cfg);solver(model,refs,'closure_contact_tight_v1')
        phys=source_parameters();phys.update(kp=spec['PD']['leg_Kp'],kd=spec['PD']['leg_Kd'],kv=spec['PD']['wheel_Kv'])
        pindex=index//2;row=assignments(0,0)[index];pending=None;total_physics=0;episode_counter=0
        row=bootstrap_case(spec,row)
        def sensors(d):
            rotation=d.xmat[refs['root']].reshape(3,3);velocity=np.zeros(6)
            mujoco.mj_objectVelocity(model,d,mujoco.mjtObj.mjOBJ_BODY,refs['root'],velocity,0)
            return np.r_[rotation.T@velocity[:3],rotation.T@[0.,0.,-1.],d.qpos[refs['qpos']],d.qvel[refs['dofs']]]
        def t(value):return torch.as_tensor(np.asarray(value),dtype=torch.float32)[None]
        def measured(d):
            angular,gravity,m=state(model,d,refs);rot=d.xmat[refs['root']].reshape(3,3)
            # Root COM native velocity matches the existing Actor/critic frame.
            body=np.asarray(m['COM_linear_velocity_w']);native=rot.T@body
            velocity=np.array([-native[2],-native[0],native[1]])
            centers=d.xpos[refs['wheels']].copy();com=d.xipos[refs['wheels']].copy()
            motions=[]
            for b in refs['wheels']:
                v=np.zeros(6);mujoco.mj_objectVelocity(model,d,mujoco.mjtObj.mjOBJ_BODY,b,v,0);motions.append(v)
            motions=np.asarray(motions);axes=d.xmat[refs['wheels']].reshape(2,3,3)[:,:,0]
            spin=measurement(t(centers),t(com),t(motions[:,3:]),t(motions[:,:3]),t(axes),t(-rot[:,2]))
            hips=d.xanchor[np.asarray(refs['joints'])[[0,3]]];local=(centers-hips)@rot
            lengths=np.linalg.norm(local,axis=-1);swing=np.arctan2(-local[:,2],-local[:,1])
            diff=float(np.arctan2(np.sin(swing[0]-swing[1]),np.cos(swing[0]-swing[1])))
            angles=np.array([np.arctan2(gravity[2],-gravity[1]),np.arctan2(gravity[0],-gravity[1])])
            return m,velocity,angular,angles,lengths,diff,spin
        while True:
            if ex is not None:ex.close()
            mechanical,episode_phys=episode_parameters(spec,row,next(p for p in PROFILES if p['name']==row['profile']),phys)
            settings=profile();settings.update(coupling_stiffness_nm_rad=mechanical['K'],
                coupling_damping_nm_s_rad=mechanical['B'],rotor_inertia=mechanical['rotor_J'] or .006)
            model.dof_armature[np.asarray(refs['dofs'])[LEGS]]=mechanical['load_armature']
            mujoco.mj_setConst(model,mujoco.MjData(model));d=mujoco.MjData(model)
            apply_initial_pose(model,d,refs,source['mcu_execution'])
            # Small initial world yaw, not a change to body-axis conventions.
            yaw=float(row.get('yaw',0.));w,x,y,z=d.qpos[3:7].copy();c=np.cos(yaw/2);s=np.sin(yaw/2)
            d.qpos[3:7]=[c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w]
            mujoco.mj_forward(model,d);current=sensors(d)
            previous=np.r_[current[6:12][LEGS]/1.5,[0.,0.]].astype(np.float32)
            # Same declared hypothesis as diagnostics: initial motor balances
            # independent spring. Not a reconstruction of a real LQR sequence.
            ex=MechanicalExecution(spec['firmware_library'],episode_phys,np.zeros(6),settings,current,previous)
            initial_motor=ex.compensation(current[6:12][LEGS])
            ex.driver=initial_motor.copy();ex.motor=initial_motor.copy()
            from .real2sim_delay import TargetTransport
            ex.fifo=TargetTransport(episode_phys['motor_delay_ms'],initial_motor)
            ex.adopted=current[6:12].copy();ex.adopted[WHEELS]=0.
            # Normalize only the host task's reset phase, with explicit
            # nonphysical initialization sends. No extra simulated force.
            for stamp in (-3,-2,-1):
                before=ex.io.wire_sequence[LEGS].copy();ex.io.send(stamp,initial_motor)
                if stamp>-3 and np.array_equal(before,ex.io.wire_sequence[LEGS]):break
            router=ForceRouter(model,refs,'separated',bool(mechanical['K']))
            monitor=QualityTermination(1,'cpu',source['reward']['quality_termination'])
            step=0;tick=0;prefix=round(float(row['prefix_s'])*100);target=np.asarray(row['command'][:2])
            fine=deque(maxlen=1000)
            prev_target=ex.adopted.copy();older_target=prev_target.copy();prev_motor=initial_motor.copy()
            history=False;last_action=None;policy_observation=None;current_command=np.array([0.,0.,.34]);last_measure=None
            episode_counter+=1
            def callback(x):
                nonlocal pending,policy_observation,last_action,last_measure
                policy_observation=x[0].copy();last_measure=measured(d)
                m,velocity,angular,angles,lengths,diff,spin=last_measure
                age=max(0,step-prefix)*.01;valid=step>=prefix
                actor=t(policy_observation)
                k=torch.full((1,4),mechanical['K']);b=torch.full((1,4),mechanical['B'])
                critic=critic_input(actor,height=t([m['height']]),velocity_native=t(velocity),
                    contacts=t(m['contact_force_n']),controlled_s=t([age]),
                    load_q=t(d.qpos[refs['qpos']][LEGS]),load_dq=t(d.qvel[refs['dofs']][LEGS]),
                    encoder_q=t(ex.flex.theta),encoder_dq=t(ex.flex.omega),K=k,B=b,
                    rotor_J=t([mechanical['rotor_J']]),load_armature=t([mechanical['load_armature']]),version=CRITIC)
                connection.send(dict(kind='observation',actor=policy_observation,critic=critic[0].numpy(),valid=valid,
                    transition=pending,pid=os.getpid(),episode=episode_counter,total_physics=total_physics,
                    cohort=np.r_[next(i for i,p in enumerate(PROFILES) if p['name']==row['profile']),target],
                    **(dict(probe_initial=dict(plant=plant,model=array_summary(model,refs),
                        qpos=d.qpos.copy(),qvel=d.qvel.copy(),initial_previous=previous.copy(),
                        physical=episode_phys,mechanical=mechanical)) if probe and tick==0 else {})))
                pending=None;message=connection.recv()
                if message['kind']=='close':raise EOFError()
                if message['kind']=='reset':raise _Reset(message['row'])
                raw=np.asarray(message['action'],np.float32)
                if raw.shape!=(6,) or not np.isfinite(raw).all():raise RuntimeError('invalid raw Gaussian action')
                last_action=raw.copy()
                if not valid:
                    with torch.inference_mode():return frozen(torch.from_numpy(x.astype(np.float32))).numpy()
                return np.clip(raw,-1.,1.)[None]
            try:
                while True:
                    age=max(0,step-prefix)*.01
                    u=np.clip((age-.5)/1.,0,1)*np.clip((28.-age)/1.,0,1)
                    current_command=np.r_[target*u,.34].astype(np.float32)
                    current=sensors(d)
                    try:
                        ex.step_sampled(tick,current,current_command,callback)
                        router.apply(d,ex.motor,ex.transmitted,ex.spring)
                    except (PacketEnvelopeError,MechanicalEnvelopeError) as exc:
                        # Known finite hypothesis/task envelope failure; sample
                        # remains in storage, no fabricated valid observation.
                        pending=dict(reward=-1. if step>=prefix else 0.,terminated=True,timeout=False,
                            reason=str(exc),metrics=None,raw_action=last_action,physical_ticks=tick%10)
                        row=dict(row);row['cycle']=row.get('cycle',0)+1;break
                    except (ValueError,RuntimeError) as exc:
                        if ex.io.current is None or c_input_valid(ex,ex.io.current,current_command):raise
                        pending=dict(reward=-1. if step>=prefix else 0.,terminated=True,timeout=False,
                            reason='finite firmware input envelope: '+str(exc),metrics=None,
                            raw_action=last_action,physical_ticks=tick%10)
                        row=dict(row);row['cycle']=row.get('cycle',0)+1;break
                    mujoco.mj_step(model,d);total_physics+=1;tick+=1
                    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise RuntimeError('nonfinite MuJoCo state')
                    forces=router.readback(d) # force readback before forward refresh
                    mujoco.mj_forward(model,d)
                    sample=np.r_[(tick-prefix*10)*.001,d.qvel[refs['dofs']],ex.io.current[10:16],ex.motor,ex.adopted]
                    if probe:sample=np.r_[sample,ex.raw,ex.driver,ex.io.wire,ex.spring,ex.transmitted,
                        d.qpos[refs['qpos']],ex.flex.theta,ex.flex.omega]
                    fine.append(sample)
                    if tick%10:continue
                    m,velocity,angular,angles,lengths,diff,spin=measured(d)
                    if m['closure_m']>.005:raise RuntimeError('closure numerical fault above5mm')
                    controlled=step>=prefix;q=d.qpos[refs['qpos']].copy();dq=d.qvel[refs['dofs']].copy()
                    posture=np.r_[angles,lengths,q[0]+q[3],q[1]-q[4],np.zeros(4)]
                    decision=monitor.observe(posture=t(posture),height=torch.tensor([m['height']],dtype=torch.float32),
                        wheel_offset=torch.tensor([m['wheel_offset_m']],dtype=torch.float32),contacts=t(m['contact_force_n']),
                        target=t(target),controlled=torch.tensor([controlled]),valid=torch.tensor([True]))
                    failed=bool(decision['terminated'][0]) or m['up']<.5 or m['height']<.20
                    origin_offset=np.asarray(m['root_position_w'])-np.asarray(m['COM_position_w'])
                    origin_velocity=np.asarray(m['COM_linear_velocity_w'])+np.cross(np.asarray(m['angular_velocity_w']),origin_offset)
                    rr=compute(source,command=t(current_command),target=t(target),velocity=t(velocity),angular=t(angular),
                        height=torch.tensor([m['height']],dtype=torch.float32),angles=t(angles),contacts=t(m['contact_force_n']),
                        q=t(q),dq=t(dq),motor=t(ex.motor),request=t(ex.raw),previous_motor=t(prev_motor),
                        adopted=t(ex.adopted),previous_target=t(prev_target),older_target=t(older_target),
                        lengths=t(lengths),wheel_offset=torch.tensor([m['wheel_offset_m']],dtype=torch.float32),
                        swing_difference=torch.tensor([diff],dtype=torch.float32),spin=spin,controlled=controlled,
                        failed=failed,history_valid=torch.tensor([history]),
                        vertical_velocity=torch.tensor([origin_velocity[2]],dtype=torch.float32))
                    if not bool(torch.isfinite(rr['total']).all()):raise RuntimeError('nonfinite reward')
                    step+=1;elapsed=max(0,step-prefix)*.01
                    timeout=not failed and step>=prefix+round(row.get('episode_s',30.)*100)
                    cap=np.array([40,40,WHEEL_CAP,40,40,WHEEL_CAP])
                    metric=dict(t=elapsed,controlled=controlled,velocity=velocity,yaw=float(angular[1]),command=current_command,
                        profile=row['profile'],episode_target=target.copy(),
                        height=m['height'],angles=angles,lengths=lengths,swing=diff,wheel_offset=m['wheel_offset_m'],
                        contact=np.asarray(m['contact_force_n']),encoder_dq=ex.io.current[10:16].copy(),load_dq=dq,
                        motor=ex.motor.copy(),transmission=ex.transmitted.copy(),spring=ex.spring.copy(),
                        adopted=ex.adopted.copy(),saturated=(np.abs(ex.motor)>=cap-1e-4),
                        slip=torch.linalg.vector_norm(spin['tangent_velocity_w'][0],dim=-1).numpy(),
                        raw=rr['raw'][0].numpy(),density=rr['density'][0].numpy(),fields=rr['fields'])
                    if probe:
                        metric.update(actor_input=policy_observation.copy(),mean=ex.policy['mean'].copy(),
                            requested_target=ex.policy['mean']*np.array([1.5,1.5,100.,1.5,1.5,100.]),
                            load_q=q.copy(),encoder_q=ex.io.current[6:10].copy(),
                            PD_input=ex.pd['native16'].copy() if ex.pd else np.zeros(16),
                            PD_valid=ex.pd is not None,PD_request=ex.raw.copy(),
                            driver=ex.driver.copy(),wire=ex.io.wire.copy(),force=forces)
                    history=True;older_target=prev_target.copy();prev_target=ex.adopted.copy();prev_motor=ex.motor.copy()
                    pending=dict(reward=float(rr['total'][0]),terminated=bool(failed),timeout=bool(timeout),
                        reason=[CAUSES[i] for i,v in enumerate(decision['causes'][0]) if v],metrics=metric,
                        raw_action=last_action,physical_ticks=10)
                    if failed or timeout:
                        # True terminal Critic, before any reset; timeout only
                        # bootstraps this observation, never the reset state.
                        pending['terminal_actor']=policy_observation.copy()
                        pending['terminal_height']=m['height'];pending['terminal_velocity']=velocity.copy()
                        pending['terminal_critic']=critic_input(t(policy_observation),height=t([m['height']]),
                            velocity_native=t(velocity),contacts=t(m['contact_force_n']),controlled_s=t([elapsed]),
                            load_q=t(q[LEGS]),load_dq=t(dq[LEGS]),encoder_q=t(ex.flex.theta),encoder_dq=t(ex.flex.omega),
                            K=torch.full((1,4),mechanical['K']),B=torch.full((1,4),mechanical['B']),
                            rotor_J=t([mechanical['rotor_J']]),load_armature=t([mechanical['load_armature']]),version=CRITIC)[0].numpy()
                        pending['fine_tail']=np.asarray(fine)
                        row=dict(row);row['cycle']=row.get('cycle',0)+1
                        if row.get('training',False):
                            new=assignments(row.get('stage',0),row['cycle'])[index]
                            new.update(stage=row.get('stage',0),cycle=row['cycle'],training=True)
                            row=new
                        break
            except _Reset as reset:
                row=reset.row;pending=None
    except EOFError:pass
    except BaseException as exc:
        try:connection.send(dict(kind='error',error=repr(exc),traceback=traceback.format_exc(),physical_steps=locals().get('total_physics')))
        except (BrokenPipeError,EOFError):pass
    finally:
        if ex is not None:ex.close()
        connection.close()


class VectorEnv:
    """One sampler process per world. Blocking RPC, no polling/watchers."""
    def __init__(self,spec,n=8):
        self.spec=spec;self.n=n;self.processes=[];self.connections=[];self.current=[]
        context=mp.get_context('spawn')
        try:
            for i in range(n):
                parent,child=context.Pipe();process=context.Process(target=_worker,args=(child,spec,i),daemon=True)
                process.start();child.close();self.connections.append(parent);self.processes.append(process)
            self.current=self._receive()
        except BaseException:self.close();raise
    def _receive(self):
        rows=[]
        for connection in self.connections:
            if not connection.poll(180):raise TimeoutError('MuJoCo worker response timeout')
            row=connection.recv()
            if row['kind']=='error':raise RuntimeError(row['error']+'\n'+row['traceback'])
            rows.append(row)
        return rows
    def observations(self):
        from tensordict import TensorDict
        from native_stand.recovery.controlled_ppo import KEY
        return TensorDict({'student_actor':torch.tensor(np.stack([r['actor'] for r in self.current])),
            'student_critic':torch.tensor(np.stack([r['critic'] for r in self.current])),
            'compliance_cohort':torch.tensor(np.stack([r['cohort'] for r in self.current]),dtype=torch.float32),
            KEY:torch.tensor([[r['valid']] for r in self.current],dtype=torch.bool)},batch_size=[self.n])
    def reset(self,rows):
        if len(rows)!=self.n:raise ValueError('explicit assignment per worker required')
        for conn,row in zip(self.connections,rows):conn.send(dict(kind='reset',row=row))
        self.current=self._receive();return self.observations()
    def step(self,actions):
        values=np.asarray(actions)
        for conn,action in zip(self.connections,values):conn.send(dict(kind='action',action=action))
        self.current=self._receive();trans=[r['transition'] for r in self.current]
        if any(t is None for t in trans):raise RuntimeError('missing completed policy transition')
        return self.observations(),trans
    @property
    def physical_steps(self):return sum(r['total_physics'] for r in self.current)
    def close(self):
        for conn in self.connections:
            try:conn.send(dict(kind='close'))
            except (BrokenPipeError,EOFError,OSError):pass
        for process in self.processes:
            process.join(10)
            if process.is_alive():process.terminate();process.join(5)
        for conn in self.connections:conn.close()
