"""Explicit Recovery31 Swing migration diagnostic behind the shared MuJoCo CLI."""
import copy,json,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import mujoco
import threading
from native_stand.students.data import sha


class Playback:
    """Viewer thread only changes intent; main thread owns all physics steps."""
    def __init__(self,paused=False):
        self.lock=threading.Lock();self.paused=paused;self.remaining=0
    def key(self,key):
        with self.lock:
            if key in (32,ord('P'),ord('p')):self.paused=not self.paused;self.remaining=0
            elif key in (ord('N'),ord('n')):self.paused=True;self.remaining=10
    def tick(self):
        with self.lock:
            if not self.paused:return True
            if self.remaining:self.remaining-=1;return True
            return False

def wheel_ground_friction(xml,value):
    """Only explicit ground/main-wheel sliding coefficients; all other attrs preserved."""
    if not np.isfinite(value) or not 0<value<=2:raise ValueError('sliding friction must be finite in (0,2]')
    tree=ET.fromstring(xml);names={'ground','left_wheel_contact','right_wheel_contact'};changes=[]
    for geom in tree.findall('.//geom'):
        if geom.get('name') not in names:continue
        friction=geom.get('friction');parts=friction.split() if friction else []
        if len(parts)!=3:raise ValueError('explicit three-slot friction required')
        changes.append({'geom':geom.get('name'),'before':list(map(float,parts)),'after':[value,*map(float,parts[1:])]})
        geom.set('friction',' '.join([format(value,'.16g'),*parts[1:]]))
    if {c['geom'] for c in changes}!=names:raise ValueError('unique wheel/ground colliders required')
    return ET.tostring(tree,encoding='unicode'),changes


def xml_with_native_limits(path,usd_path):
    """Reference geometry/PD unchanged; author the current USD mechanical ranges."""
    from pxr import Usd,UsdPhysics
    stage=Usd.Stage.Open(str(usd_path));limits={}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint=UsdPhysics.RevoluteJoint(prim)
            lo,hi=joint.GetLowerLimitAttr().Get(),joint.GetUpperLimitAttr().Get()
            if np.isfinite([lo,hi]).all():limits[prim.GetName()]=np.deg2rad([lo,hi])
    tree=ET.parse(path);root=tree.getroot();compiler=root.find('compiler')
    if compiler.get('angle','degree')!='radian':raise ValueError('MJCF joint angle units must be radians')
    for key in ['meshdir','texturedir']:
        if compiler.get(key):compiler.set(key,str((path.parent/compiler.get(key)).resolve()))
    changes=[]
    for joint in root.findall('.//joint'):
        name=joint.get('name')
        if name in limits:
            old=joint.get('range');new=limits[name]
            if old is None or not np.allclose(np.fromstring(old,sep=' '),new,atol=1e-7):
                changes.append({'joint':name,'reference_range_rad':old,'native_range_rad':new.tolist()})
            joint.set('limited','true');joint.set('range',' '.join(format(v,'.16g') for v in new))
    return ET.tostring(root,encoding='unicode'),changes


def initialize(model,refs,pose):
    data=mujoco.MjData(model);missing=[]
    for name,value in pose['joint_pos'].items():
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if j<0:missing.append(name);continue
        if model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE:raise ValueError('pose joint type differs')
        data.qpos[model.jnt_qposadr[j]]=value
    if missing:raise ValueError('Missing complete Swing pose joints: '+str(missing))
    q=pose['root_quaternion_xyzw'];data.qpos[:3]=[0,0,pose['root_height_m']]
    data.qpos[3:7]=[q[3],*q[:3]];data.qvel[:]=0
    mujoco.mj_forward(model,data);return data


def clearance(model,data,refs):
    values=[]
    for i in range(model.ngeom):
        if model.geom_bodyid[i]!=refs['root'] or not (model.geom_contype[i] or model.geom_conaffinity[i]):continue
        axis=data.geom_xmat[i].reshape(3,3)[2];size=model.geom_size[i];kind=model.geom_type[i]
        if kind==mujoco.mjtGeom.mjGEOM_BOX:extent=np.abs(axis)@size
        elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:extent=size[0]
        elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:extent=size[0]+abs(axis[2])*size[1]
        else:raise ValueError('Unsupported chassis collider clearance shape')
        values.append(data.geom_xpos[i,2]-extent)
    if not values:raise ValueError('No chassis collision geometry')
    return float(min(values))


def observation(model,data,refs,legs,wheels):
    from validate_native_mujoco import state
    from native_stand.students.mcu_interface import encode
    # mj_step integrates qpos/qvel after computing derived kinematics/cvel.
    # Refresh at the policy boundary so IMU and encoders refer to the same instant.
    mujoco.mj_forward(model,data)
    angular,gravity,s=state(model,data,refs)
    q=torch.tensor(data.qpos[refs['qpos']],dtype=torch.float32)[None]
    dq=torch.tensor(data.qvel[refs['dofs']],dtype=torch.float32)[None]
    x=encode(torch.tensor(angular,dtype=torch.float32)[None],torch.tensor(gravity,dtype=torch.float32)[None],
             q,dq,legs,wheels,torch.zeros(1,6),'Recovery',torch.tensor([[0.,0.,.34]]))
    return x,s


def plan(args):
    from native_stand.students.model import load
    from native_stand.students.requests import validate
    from validate_native_mujoco import binding,state,JOINTS
    spec=json.loads(args.spec.read_text());errors=validate(spec)
    if errors:raise ValueError(errors)
    actor,saved=load(args.checkpoint,'RecoveryStudent');actor.eval();actor.requires_grad_(False)
    small_dataset=getattr(args,'student_dataset',None)
    if small_dataset:
        from native_stand.students.recovery_distillation import validate_small_lineage
        actor,saved,_=validate_small_lineage(args.checkpoint,small_dataset,args.spec)
        actor.eval().requires_grad_(False)
    elif saved['spec_sha256']!=sha(args.spec) or saved['asset_sha256']!=sha(spec['asset_path']):raise ValueError('checkpoint/spec/asset identity differs')
    expected_critic={'student_recovery_broad_narrow_ppo_v1':'recovery_privileged64_v1',
                     'student_recovery_contact_quality_ppo_v1':'recovery_contact64_v2'}.get(spec.get('student_training',{}).get('version'))
    if expected_critic is None or (not small_dataset and saved['critic_privileged_version']!=expected_critic):
        raise ValueError('Explicit Recovery recipe/Critic contract differs')
    if not 0<args.wall_seconds<=1800:raise ValueError('Wall bound0..1800required')
    if not np.isfinite(getattr(args,'playback_speed',1.)) or not .05<=getattr(args,'playback_speed',1.)<=4:
        raise ValueError('playback speed must be .05..4')
    if not getattr(args,'viewer',False) and (getattr(args,'start_paused',False) or getattr(args,'hold_viewer_on_end',False) or getattr(args,'playback_speed',1.)!=1.):raise ValueError('Playback controls require --viewer')
    if args.commands not in (None,[0.]) or args.yaw_commands:raise ValueError('Recovery only supports fixed zero command')
    poses=Path(spec['recovery']['pose_library'])
    if sha(poses)!=spec['recovery']['pose_library_sha256']:raise ValueError('pose library changed')
    choices=[p for p in json.loads(poses.read_text())['poses'] if p['family']=='swing_candidate_from_historical_r1' and p['split']=='holdout']
    if len(choices)!=1:raise ValueError('Explicit unique native Swing required')
    pose=copy.deepcopy(choices[0])
    yaw=getattr(args,'initial_yaw_rad',0.)
    if not np.isfinite(yaw) or abs(yaw)>.05:raise ValueError('Recovery initialization yaw must be within .05rad')
    q=np.asarray(pose['root_quaternion_xyzw']);v=np.array([0.,0.,np.sin(yaw/2)]);w=np.cos(yaw/2)
    pose['root_quaternion_xyzw']=[*(w*q[:3]+q[3]*v+np.cross(v,q[:3])),w*q[3]-v@q[:3]]
    xml,changes=xml_with_native_limits(args.xml,Path(spec['asset_path']))
    damping_changes=[]
    if getattr(args,'damping_readback',None):
        from audit_recovery_mujoco_parity import align_joint_damping
        xml,damping_changes=align_joint_damping(xml,json.loads(args.damping_readback.read_text()))
    friction_changes=[]
    if getattr(args,'wheel_ground_sliding_friction',None) is not None:
        xml,friction_changes=wheel_ground_friction(xml,args.wheel_ground_sliding_friction)
    model=mujoco.MjModel.from_xml_string(xml);refs=binding(model)
    if model.nu!=6 or model.neq!=4 or abs(model.opt.timestep-.001)>1e-12:raise ValueError('topology/timestep differs')
    if [spec['control'][k] for k in ('physics_dt_s','pd_dt_s','policy_dt_s','leg_kp','leg_kd','wheel_kv')]!=[.001,.005,.01,60.,2.,1.] or spec['control']['effort_limits_nm']!=[40,40,5.5,40,40,5.5]:raise ValueError('PD/physical contract differs')
    ctrl=np.array([40,40,5.5,40,40,5.5])/np.abs(refs['gears'])
    if np.any(ctrl>np.min(np.abs(model.actuator_ctrlrange[refs['actuators']]),axis=1)):raise ValueError('MJCF actuator range cannot execute declared torque')
    if spec['action']['order']!=list(JOINTS) or spec['action']['leg_scale_rad']!=1.5 or spec['action']['wheel_scale_rad_s']!=100.:raise ValueError('action contract differs')
    data=initialize(model,refs,pose);_,_,s=state(model,data,refs)
    if s['closure_m']>spec['recovery']['initialization']['offline_closure_max_m']:raise ValueError('Swing offline closure invalid')
    legs=torch.tensor(data.qpos[refs['qpos']][[0,1,3,4]],dtype=torch.float32)[None];wheels=torch.zeros(1,2)
    obs,_=observation(model,data,refs,legs,wheels)
    files=[args.spec,args.checkpoint,args.xml,poses,Path(spec['asset_path']),Path(__file__),Path(__file__).with_name('validate_native_mujoco.py')]
    if small_dataset:files.extend([small_dataset,small_dataset.with_name('manifest.json')])
    if getattr(args,'damping_readback',None):files.extend([args.damping_readback,Path(__file__).with_name('audit_recovery_mujoco_parity.py')])
    tree=ET.fromstring(xml);meshdir=Path(tree.find('compiler').get('meshdir',''))
    files += [meshdir/m.get('file') for m in tree.findall('./asset/mesh')]
    manifest={'version':'recovery31_mujoco_swing_v1','checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':sha(args.checkpoint),
       'inputs_sha256':{str(f.resolve()):sha(f) for f in files},'pose_id':pose['pose_id'],'initial_state':s,
       'initial_clearance_m':clearance(model,data,refs),'initial_input':obs[0].tolist(),
       'initial_targets_fourlegs_then_wheels':[ *legs[0].tolist(),0.,0.],
       'joint_names':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(model.njnt)],
       'limit_changes_to_match_USD':changes,'model_xml_sha256':__import__('hashlib').sha256(xml.encode()).hexdigest(),
       'protocol':{'entry_deadline_s':2.5,'entry_dwell_s':.2,'first_hold_s':10.,'policy_dt_s':.01,'PD_dt_s':.005,'physics_dt_s':.001,'control':['PD60/2/1','40/5.5Nm'],'height_command_m':.34},
       'differences':['Reference MJCF colliders/friction/damping/armature/soft equality remain unchanged.',
          'Only local diagnostic mechanical ranges match current USD; external source remains read-only.',
          'Wheel contacts are MuJoCo ground normal forces, unlike native resultant sensors; no direct force-certificate equivalence.',
          'Native CAD Actor axes; root linear velocity measured at body COM; no compensator or teacher.',
          'This is a single deterministic migration diagnostic, not native certification.'],
       'initial_yaw_rad':yaw,'wall_limit_s':args.wall_seconds,'simulation':0,'PPO':0,'optimizer':0}
    if damping_changes:
        manifest.update(damping_alignment={'version':'recovery_mj_damping_group_alignment_v1','readback_sha256':sha(args.damping_readback),'changes':damping_changes,'scope':'nominal joint damping only; explicit native passive PD versus MJ mechanical damping still use different solvers'},cases=['normal_standing','swing'],max_episodes=2)
        manifest['differences'][0]='Reference colliders/friction/armature/soft equality unchanged; only named joint damping aligned to saved runtime coefficients.'
    if friction_changes:
        manifest.update(friction_diagnostic={'version':'recovery_wheel_ground_sliding_friction_v1','changes':friction_changes,'scope':'only three sliding coefficient attributes; does not certify PhysX/MuJoCo contact equivalence'},cases=['normal_standing','swing'],max_episodes=2)
        manifest['differences'][0]='Reference geometry/damping/armature/solver unchanged; only main wheels and ground sliding friction changed.'
    return spec,model,refs,actor,pose,xml,manifest


def run_case(spec,model,refs,actor,pose,deadline,viewer=False,*,start_paused=False,playback_speed=1.,hold_viewer_on_end=False,normal_start=False,decode_targets=None):
    from validate_native_mujoco import state
    from native_stand.students.interface import targets
    from native_stand.command_v6.domain.contract import effort
    from native_stand.students.recovery_task import Certificate
    from native_stand.students.protocol import score
    data=mujoco.MjData(model) if normal_start else initialize(model,refs,pose)
    mujoco.mj_forward(model,data)
    initial=spec['recovery']['initialization'];clock=Certificate(1,'cpu')
    controlled=normal_start;control_start=0. if normal_start else None;zeroed=False;dwell=0.;reason=None;rows={};display=None;started=time.monotonic()
    legs=torch.tensor(data.qpos[refs['qpos']][[0,1,3,4]],dtype=torch.float32)[None];wheels=torch.zeros(1,2)
    requested=applied=torch.zeros(1,6);raw=torch.zeros(1,6);obs=None
    settle_dofs=[model.jnt_dofadr[j] for j in range(model.njnt) if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE and not mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j).startswith('guide_wheel_')]
    ranges=torch.tensor(model.jnt_range[refs['joints']],dtype=torch.float32)[None]
    for c,j in enumerate(refs['joints']):
        if not model.jnt_limited[j]:ranges[0,c]=torch.tensor([-float('inf'),float('inf')])
    if viewer:
        from mujoco import viewer as v
        playback=Playback(start_paused)
        display=v.launch_passive(model,data,key_callback=playback.key);display.cam.trackbodyid=refs['root'];display.cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING;display.cam.distance=2.
        print('MuJoCo Recovery: Space/P pause-resume; N advance10ms; close window to exit. Playback speed='+str(playback_speed),flush=True)
        pace_wall=time.monotonic();pace_sim=data.time
    try:
        for step in range(14000):
            if time.monotonic()>=deadline:reason='WALL_BUDGET';break
            if display and not display.is_running():reason='VIEWER_CLOSED';break
            while display and not playback.tick():
                display.sync();time.sleep(.01);pace_wall=time.monotonic();pace_sim=data.time
                if not display.is_running() or time.monotonic()>=deadline:break
            if display and (not display.is_running() or time.monotonic()>=deadline):
                reason='VIEWER_CLOSED' if not display.is_running() else 'WALL_BUDGET';break
            was_controlled=controlled
            if step%10==0:
                obs,pre=observation(model,data,refs,legs,wheels)
                previous=torch.cat((legs,wheels),-1).numpy()[0].copy();root_pre=np.r_[data.xpos[refs['root']],data.xquat[refs['root']]].copy()
                qpre=data.qpos.copy();dqpre=data.qvel.copy()
                raw=actor.mean(obs) if controlled else torch.zeros(1,6)
                if not torch.isfinite(raw).all():reason='NONFINITE_ACTION';break
                if controlled:
                    output=(decode_targets or targets)(raw,torch.zeros(1,6),ranges);legs=output['leg_executed'];wheels=output['wheel_executed']
            if step%5==0:
                if controlled:
                    requested,applied=effort(torch.tensor(data.qpos[refs['qpos']],dtype=torch.float32)[None],torch.tensor(data.qvel[refs['dofs']],dtype=torch.float32)[None],legs,wheels,leg_kp=60.,leg_kd=2.,wheel_kv=1.)
                else:requested=torch.zeros(1,6);applied=requested
                data.ctrl[refs['actuators']]=applied[0].numpy()/refs['gears']
            mujoco.mj_step(model,data)
            if step%10!=9:continue
            mujoco.mj_forward(model,data)
            angular,gravity,s=state(model,data,refs);gap=clearance(model,data,refs)
            angles=np.array([np.arctan2(gravity[2],-gravity[1]),np.arctan2(gravity[0],-gravity[1])])
            finite=np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            penetration=min([c.dist for c in data.contact]+[0.])
            fault=not finite or s['closure_m']>.005 or penetration<-.005
            safety=s['up']<.5
            if not was_controlled:
                if fault or safety:reason='INITIALIZATION_PHYSICAL_FAULT';break
                if not zeroed and data.time>=initial['settle_min_s']-1e-9:
                    data.qvel[:]=0;mujoco.mj_forward(model,data);zeroed=True;dwell=0.
                elif zeroed:
                    good=(np.linalg.norm(s['COM_linear_velocity_w'])<=initial['settled_linear_speed_max_mps'] and np.linalg.norm(s['angular_velocity_w'])<=initial['settled_angular_speed_max_rad_s'] and np.abs(data.qvel[settle_dofs]).max()<=initial['settled_joint_speed_max_rad_s'] and s['tilt']<=initial['tilt_max_rad'] and pose['initial_root_height_m'][0]<=s['height']<=pose['initial_root_height_m'][1] and pose['initial_chassis_clearance_m'][0]<=gap<=pose['initial_chassis_clearance_m'][1])
                    dwell=dwell+.01 if good else 0.
                    if dwell>=initial['settle_dwell_s']-1e-9:controlled=True;control_start=data.time
                if not controlled and data.time>=initial['settle_max_s']-1e-9:reason='INITIALIZATION_NOT_SETTLED';break
            else:
                sat=bool((applied.abs()>=applied.new_tensor([40,40,5.5,40,40,5.5])-1e-4).any())
                clock.set_velocity(torch.tensor([[s['vx'],s['vy'],s['yaw_rate']]],dtype=torch.float32))
                success,failure,end,_=clock.observe(torch.tensor([s['height']]),torch.tensor(angles,dtype=torch.float32)[None],torch.tensor([s['tilt']]),torch.tensor([s['contact_force_n']]),torch.tensor([gap]),torch.tensor([s['closure_m']]),torch.tensor([float(sat)]),active=torch.tensor([True]),valid=torch.tensor([True]),safety_failed=torch.tensor([safety]),fault=torch.tensor([fault]))
                sr=np.zeros(39,dtype=np.float32);sr[[0,3,4,6,7,8,9,10,11,12,13,15,17,29,31,32]]=[data.time-control_start,bool(failure),fault,s['height'],s['vx'],s['vy'],s['yaw_rate'],*s['contact_force_n'],s['closure_m'],gap,s['tilt'],sat,1,1,2]
                posture=np.zeros(10,dtype=np.float32);posture[:2]=angles
                record=dict(policy_time_s=np.array(data.time-control_start),state=sr,posture=posture,actor_input=obs[0].numpy(),mean=raw[0].numpy(),previous_targets=previous,
                    requested_leg_target=output['leg_requested'][0].numpy() if 'leg_requested' in output else (raw[0,[0,1,3,4]]*1.5).numpy(),
                    executed_leg_target=legs[0].numpy(),executed_wheel_target=wheels[0].numpy(),requested_torque=requested[0].numpy(),clipped_torque=applied[0].numpy(),actuator_force=data.qfrc_actuator[refs['dofs']].copy(),root_pre_wxyz=root_pre,root_post_wxyz=np.r_[data.xpos[refs['root']],data.xquat[refs['root']]],joint_q_pre=qpre,joint_dq_pre=dqpre,joint_q_post=data.qpos.copy(),joint_dq_post=data.qvel.copy())
                for k,v in record.items():rows.setdefault(k,[]).append(v)
                if bool(end):reason='RUNTIME_FAULT' if fault else 'TILT' if safety else 'SUCCESS' if bool(success) else 'TASK_FAILED';break
            if display:
                display.sync();delay=(data.time-pace_sim)/playback_speed-(time.monotonic()-pace_wall)
                if delay>0:time.sleep(delay)
                pace_wall=time.monotonic();pace_sim=data.time
        if display and hold_viewer_on_end and reason not in ('VIEWER_CLOSED','WALL_BUDGET'):
            print('Episode ended: '+str(reason)+'; final state frozen, close window to exit.',flush=True)
            while display.is_running() and time.monotonic()<deadline:
                display.sync();time.sleep(.02)
    finally:
        if display:display.close()
    arrays={k:np.stack(v) for k,v in rows.items()}
    result={'initialization_valid':control_start is not None,'initialization_s':control_start,'termination':reason,'controlled_s':len(rows.get('state',[]))*.01,'accepted':False}
    if arrays:
        scored=score(torch.tensor(arrays['state']),torch.tensor(arrays['posture']),torch.ones(len(arrays['state']),dtype=torch.bool),policy_time_s=torch.tensor(arrays['policy_time_s'],dtype=torch.float32))
        result['score']=scored;result['accepted']=scored['accepted']
        result['height_peak_m']=float(arrays['state'][:,6].max());result['pitch_roll_peak_rad']=np.abs(arrays['posture'][:,:2]).max(0).tolist()
        result['closure_peak_m']=float(arrays['state'][:,12].max());result['saturation_fraction']=float(arrays['state'][:,17].mean())
        if bool(clock.certificate_passed[0])!=scored['accepted']:raise RuntimeError('MuJoCo online/offline certificate mismatch')
    result['last_state']=s;result['last_clearance_m']=gap
    return result,arrays


def execute(args):
    from native_stand.command_v6.tools.artifact_io import npz
    from native_stand.command_v6.completion import write
    from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
    started=time.monotonic();torch.set_num_threads(1)
    spec,model,refs,actor,pose,xml,manifest=plan(args)
    if not args.run:print(json.dumps(manifest,indent=2));return
    validate_new_experiment_name(args.output_dir.name);args.output_dir.mkdir(parents=True,exist_ok=False)
    write(args.output_dir/'manifest.json',manifest);(args.output_dir/'diagnostic_model.xml').write_text(xml)
    cases=manifest.get('cases',['swing']);status='FAILED';result=None;error=None;before={k:v.clone() for k,v in actor.state_dict().items()};results={};attempted=0
    write(args.output_dir/'budget.json',{'status':'RUNNING','wall_limit_s':args.wall_seconds,'max_episodes':len(cases),'PPO':0,'optimizer':0})
    try:
        for case in cases:
            attempted+=1
            with torch.inference_mode():result,arrays=run_case(spec,model,refs,actor,pose,started+args.wall_seconds-5,args.viewer,
                start_paused=getattr(args,'start_paused',False),playback_speed=getattr(args,'playback_speed',1.),hold_viewer_on_end=getattr(args,'hold_viewer_on_end',False),normal_start=case=='normal_standing')
            if arrays:npz(args.output_dir/(case+'_trajectory.npz' if len(cases)>1 else 'trajectory.npz'),arrays)
            results[case]=result;write(args.output_dir/(case+'_result.json'),result)
            status='COMPLETED' if result['termination'] not in ('RUNTIME_FAULT','INITIALIZATION_PHYSICAL_FAULT','INITIALIZATION_NOT_SETTLED','NONFINITE_ACTION','WALL_BUDGET') else 'FAILED'
            if status=='FAILED':
                for missing in cases[attempted:]:results[missing]={'status':'NOT_RUN','reason':'preceding engineering fault or budget'}
                break
    except BaseException as exc:error=repr(exc);raise
    finally:
        unchanged=all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
        for file,h in manifest['inputs_sha256'].items():
            if sha(file)!=h:error='Read-only input drift: '+file;status='FAILED'
        payload={'status':status,'wall_s':time.monotonic()-started,'PPO':0,'optimizer':0,'model_unchanged':unchanged,'error':error,'result':result,'results':results,'native_certification':False}
        write(args.output_dir/'terminal_report.json',payload);write(args.output_dir/'budget.json',{'status':status,'wall_limit_s':args.wall_seconds,'wall_used_s':payload['wall_s'],'max_episodes':len(cases),'episodes_attempted':attempted,'PPO':0,'optimizer':0})
        (args.output_dir/'report.md').write_text('# Recovery31 → MuJoCo\n\n单环境原生Swing迁移诊断，非native认证；接触为MuJoCo法向力，非native传感器合力。\n\n```json\n'+json.dumps(payload,ensure_ascii=False,indent=2)+'\n```\n')
        print(json.dumps(payload,ensure_ascii=False,indent=2))
