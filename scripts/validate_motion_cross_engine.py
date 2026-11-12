"""Finite, request-bound Motion31 current-plant comparison; plan unless --run.

Models retain their own parent spec identities; a separate common test protocol
selects height/initial state/PD age. No PPO, GUI, teacher takeover beyond the
declared one-second prefix, or firmware writes are performed.
"""
import argparse
import copy
import io
import json
import os
import shutil
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from native_stand.command_v6.completion import write
from native_stand.students.data import sha
from native_stand.students.motion_cross_engine import VERSION, cases, scenario, score, ranking


def read_request(path):
    request=json.loads(Path(path).read_text())
    if request['version'] != VERSION or not request.get('authorization'):
        raise ValueError('explicit comparison request/authorization required')
    limits=request['limits']
    if not (0 < limits['wall_seconds'] <= 5400 and limits['physics_seconds'] <= 6000
            and limits['storage_bytes'] <= 512*2**20 and limits['free_reserve_bytes'] >= 1536*2**20):
        raise ValueError('finite approved comparison limits required')
    if limits['seconds_per_case'] != 30 or len(request['models']) != 3:
        raise ValueError('this comparison freezes three models and30s per condition')
    names=[r['name'] for r in request['models']]
    if len(set(names)) != 3:raise ValueError('three unique model names required')
    if 'source4500' not in names:raise ValueError('explicit frozen source4500 prefix controller required')
    paths={request['reference_spec']:request['reference_spec_sha256']}
    for model in request['models']:
        paths.update({model['checkpoint']:model['sha256'], model['spec']:model['spec_sha256']})
    for path,digest in paths.items():
        if sha(path) != digest:raise ValueError('source identity differs: '+path)
    spec=json.loads(Path(request['reference_spec']).read_text())
    rows=cases(spec)
    maximum=sum(30+scenario(r)[2] for r in rows)*len(names)
    if maximum > limits['physics_seconds']:raise ValueError('matrix exceeds declared physics limit')
    return request,spec,rows,paths


def load_actors(request):
    from native_stand.students.model import load
    actors={}
    for row in request['models']:
        parent=json.loads(Path(row['spec']).read_text())
        actor,saved=load(row['checkpoint'],'MotionZeroStudent')
        if (saved['spec_sha256'] != sha(row['spec']) or saved['asset_sha256'] != parent['asset_sha256']
                or saved['critic_privileged_version'] != 'motion_mcu_motorqueue_privileged312_v1'):
            raise ValueError('exact model/own-parent-spec/asset required: '+row['name'])
        actors[row['name']]=actor.eval()
    return actors


def plant(request,spec):
    import mujoco
    from validate_native_mujoco import binding
    from motion_mcu_playback import apply_current_plant,configuration,declared
    from native_stand.students.mujoco_solver import apply
    if (request['plant_profile'] != 'motion_mcu_currentplant_v1' or
            request['solver_profile'] != 'closure_contact_tight_v1'):
        raise ValueError('explicit current MCU plant/tight solver required')
    cfg=configuration(SimpleNamespace(leg_armature_m=None,zero_latency=False),declared(spec),spec)
    model=mujoco.MjModel.from_xml_path(request['xml']);refs=binding(model)
    readback=apply_current_plant(model,refs,spec,cfg)
    solver=apply(model,refs,request['solver_profile'])
    return model,refs,cfg,dict(plant=readback,solver=solver,execution=cfg)


def case_configuration(spec,cfg,row):
    poses=spec['motion_height_dr']['height_poses']
    target=row['command'][2]
    pose=next((p for p in poses if p['height_m']==target),None)
    if pose is None or pose.get('offline_valid') is not True:
        raise ValueError('verified closed-chain pose required for height '+str(target))
    current=copy.deepcopy(cfg);current['mcu_execution']['pose']=copy.deepcopy(pose)
    return current


def initial_readback(model,refs,cfg,row):
    import mujoco
    from motion_mcu_playback import apply_initial_pose
    from native_stand.students.mcu_interface import encode
    from validate_native_mujoco import state
    from native_stand.students.motion_cross_engine import free_velocity
    data=mujoco.MjData(model);q=apply_initial_pose(model,data,refs,cfg['mcu_execution'])
    mujoco.mj_forward(model,data);ang,gravity,st=state(model,data,refs)
    h=row['command'][2]
    obs=encode(torch.tensor(ang)[None].float(),torch.tensor(gravity)[None].float(),
        torch.tensor(q)[None].float(),torch.zeros(1,6),torch.tensor(q[[0,1,3,4]])[None].float(),
        torch.zeros(1,2),torch.zeros(1,6),'Motion',torch.tensor([[0.,0.,h]]))
    if st['closure_m']>1e-6 or abs(st['height']-h)>1e-5 or abs(float(obs[0,24])-h/.5)>1e-6:
        raise ValueError('height/closure/Actor command initial readback differs')
    _,velocity,_=scenario(row)
    free=next(j for j in range(model.njnt) if model.jnt_bodyid[j]==refs['root'] and model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE)
    dof=int(model.jnt_dofadr[free]);rotation=data.xmat[refs['root']].reshape(3,3)
    data.qvel[dof:dof+6]=free_velocity(velocity,rotation);mujoco.mj_forward(model,data)
    actual=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,refs['root'],actual,0)
    if not np.allclose(actual[:3],velocity[3:],atol=1e-9,rtol=0):
        raise ValueError('initial WORLD angular velocity readback differs')
    return dict(case_id=row['case_id'],height_m=st['height'],closure_m=st['closure_m'],
                q6=q.tolist(),actor_height=float(obs[0,24]),previous_leg=obs[0,16:20].tolist(),
                requested_initial_velocity_w=velocity,actual_angular_velocity_w=actual[:3].tolist())


def reusable_records(request,matrix,actors):
    """Limited engineering recovery: zero-velocity episodes are unaffected.

    Parent raw files stay read-only. The corrected world→body angular mapping
    cannot be used to reuse any nonzero-velocity episode from the old process.
    """
    recovery=request.get('recovery')
    if not recovery:return [],{},0.,0.,0.
    parent=Path(recovery['parent_output']);terminal=parent/'terminal_report.json'
    if recovery['version']!='motion31_mujoco_zero_velocity_resume_v1' or sha(terminal)!=recovery['parent_terminal_sha256']:
        raise ValueError('explicit settled engineering recovery required')
    old=json.loads(terminal.read_text());budget=json.loads((parent/'budget.json').read_text())
    manifest=json.loads((parent/'manifest.json').read_text())
    if old['status']!='STOPPED_BOUNDED' or old['models_unchanged'] is not True or manifest['models']!=request['models']:
        raise ValueError('recovery model/settlement differs')
    by_id={r['case_id']:r for r in matrix};records=[];inputs={str(terminal.resolve()):sha(terminal)}
    def comparison_key(row):
        return {k:v for k,v in row.items() if k not in ('checkpoint','sha256','_template_source')}
    for row in old['results']:
        new=by_id[row['case']['case_id']]
        if comparison_key(row['case'])!=comparison_key(new) or scenario(new)[1]!=[0.]*6:
            raise ValueError('only identical zero-world-velocity conditions can be reused')
        path=parent/row['trace']['path']
        if sha(path)!=row['trace']['sha256']:raise ValueError('parent trace identity differs')
        with np.load(path) as saved:arrays={k:saved[k] for k in saved.files}
        prefix=scenario(new)[2]
        audit=inspect_trace(arrays,actors[row['model']],prefix,teacher=actors['source4500'])
        derived=copy.deepcopy(row);derived.update(case=new,readback=audit,
            tested_model=next(m for m in request['models'] if m['name']==row['model']),
            trace=dict(path=str(path.resolve()),sha256=sha(path)),reused_without_simulation=True,
            reuse_scope='zero initial angular velocity, identical commands/Plant/decoder/previous; corrected transform maps zero to zero')
        derived['assessment']=score(arrays,new,prefix=prefix,termination=row['termination'])
        derived['assessment']['metrics']['spectra']=row['assessment']['metrics'].get('spectra',{})
        records.append(derived);inputs[str(path.resolve())]=sha(path)
    if any(a.get('settled') is not True for a in budget['attempts'].values()):
        raise ValueError('parent attempt not settled')
    # A signal can interrupt between100Hz telemetry samples. Conservatively
    # charge the unobserved final <=10ms instead of refunding or claiming exactness.
    uncertain=sum(a['status']=='FAILED' for a in budget['attempts'].values())*.01
    return records,inputs,float(old['physical_seconds']),float(old['physical_seconds'])+uncertain,float(old['wall_seconds'])


class Telemetry:
    def __init__(self,spec,row,prefix):
        self.anchors=np.asarray(spec['recovery']['standing_task']['hip_anchors_root_m'])
        self.row=row;self.prefix=prefix;self.last_physics_s=0.;self.tail=[]

    def __call__(self,model,data,refs):
        import mujoco
        from native_stand.students.motion_yaw_quality import measurement
        self.last_physics_s=float(data.time)
        rotation=data.xmat[refs['root']].reshape(3,3)
        gravity=rotation.T@np.array([0.,0.,-1.]);forward=-rotation[:,2]
        heading=forward.copy();heading[2]=0.;heading/=np.linalg.norm(heading)
        centers=data.xpos[refs['wheels']].copy();com=data.xipos[refs['wheels']].copy()
        local=(centers-data.xpos[refs['root']])@rotation
        vectors=local-self.anchors;swing=np.arctan2(-vectors[:,2],-vectors[:,1])
        velocity=np.zeros((3,6))
        for i,body in enumerate([*refs['wheels'],refs['root']]):
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body,velocity[i],0)
        axis=data.xaxis[np.asarray(refs['joints'])[[2,5]]].copy()
        m=measurement(*(torch.tensor(x)[None] for x in (centers,com,velocity[:2,3:],velocity[:2,:3],axis)),torch.tensor(forward)[None])
        force=np.zeros((2,3));undesired=False
        for i in range(data.ncon):
            c=data.contact[i]
            if refs['ground'] not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==refs['ground'] else c.geom1;body=int(model.geom_bodyid[other])
            if body not in refs['wheels']:
                undesired=True;continue
            value=np.zeros(6);mujoco.mj_contactForce(model,data,i,value)
            force[refs['wheels'].index(body)]+=c.frame.reshape(3,3).T@value[:3]*(1 if c.geom1==refs['ground'] else -1)
        # Use the pre-action command held over the just-completed policy interval.
        age=max(0.,round(data.time*1000-10)*.001-self.prefix)
        gain=min(np.clip((age-2)/8,0,1),np.clip((26-age)/8,0,1))
        target=.5*self.row['command'][1]*gain*np.dot(np.cross([0.,0.,1.],centers[0]-centers[1]),heading)
        result=dict(native_posture=np.array([np.arctan2(gravity[2],-gravity[1]),np.arctan2(gravity[0],-gravity[1])]),
            root_velocity_w=velocity[2,3:].copy(),lengths=np.linalg.norm(vectors,axis=-1),
            heading_offset=np.dot(centers[0]-centers[1],heading),swing_rad=swing,
            swing_difference=abs(np.arctan2(np.sin(swing[0]-swing[1]),np.cos(swing[0]-swing[1]))),
            rolling=m['rolling_speed_forward_mps'][0].numpy(),
            rolling_geometry_valid=m['geometry_valid'][0].numpy(),
            support_velocity_xy=m['tangent_velocity_w'][0].numpy(),differential_target=np.array(target),
            contact_resultant=np.linalg.norm(force,axis=-1),undesired_ground_contact=np.array(undesired))
        self.tail.append(dict(time=self.last_physics_s,q=data.qpos[refs['qpos']].tolist(),dq=data.qvel[refs['dofs']].tolist()))
        self.tail=self.tail[-10:]
        return result


def inspect_trace(arrays,actor,prefix,teacher=None):
    obs=arrays['actor_input']
    with torch.inference_mode():mean=actor.mean(torch.tensor(obs,dtype=torch.float32)).numpy()
    prefix_rows=arrays['owner']==0
    if prefix_rows.any():
        if teacher is None:raise ValueError('prefix Actor required for trace readback')
        with torch.inference_mode():
            mean[prefix_rows]=teacher.mean(torch.tensor(obs[prefix_rows],dtype=torch.float32)).numpy()
    infer_error=float(np.max(np.abs(mean-arrays['mean'])))
    previous=arrays['adopted_target'][:-1][:,[0,1,3,4,2,5]]/np.array([1.5]*4+[100.]*2)
    cache_error=float(np.max(np.abs(obs[1:,16:22]-previous))) if len(previous) else 0.
    if infer_error>2e-5 or cache_error>2e-6:raise ValueError('Actor/actual-adopted previous trace readback differs')
    return dict(actor_error=infer_error,previous_error=cache_error,source='PD adopted physical target',
                handover_resets=False,controlled_clock_origin_s=prefix)


def prepare_wide(request,results,output,actors):
    from native_stand.students.actor_widening import expand_mcu_checkpoint,wide_network
    from native_stand.students.model import load
    ranks=ranking(results,list(actors))
    eligible=[r for r in ranks if r['matrix_complete']]
    if not eligible:return dict(status='NOT_PREPARED',reason='no complete model matrix',ranking=ranks)
    best=eligible[0]['model'];source=next(r for r in request['models'] if r['name']==best)
    saved=torch.load(source['checkpoint'],map_location='cpu',weights_only=False)
    wide=expand_mcu_checkpoint(saved,source_checkpoint_sha256=source['sha256'])
    path=output/'actor1024_initial.pt';torch.save(wide,path)
    actor,_=load(path,'MotionZeroStudent');actor.eval()
    inputs=[]
    for result in results:
        if result['model']==best:
            with np.load(output/result['trace']['path']) as data:inputs.append(data['actor_input'][::100])
    x=torch.tensor(np.concatenate(inputs),dtype=torch.float32)
    with torch.inference_mode():error=float((actors[best].mean(x)-actor.mean(x)).abs().max())
    if error>2e-5:raise ValueError('wide saved/read-back function differs')
    # Six output derivatives, on actual visited states; do not mutate either controller.
    old=copy.deepcopy(actors[best]).double();new=copy.deepcopy(actor).double()
    probe=x[::max(1,len(x)//8)][:8].double().requires_grad_(True)
    y0,y1=old.mean(probe),new.mean(probe);jacobian_error=0.
    for channel in range(6):
        j0=torch.autograd.grad(y0[:,channel].sum(),probe,retain_graph=True)[0]
        j1=torch.autograd.grad(y1[:,channel].sum(),probe,retain_graph=True)[0]
        jacobian_error=max(jacobian_error,float((j0-j1).abs().max()))
    if jacobian_error>1e-4:raise ValueError('wide saved/read-back local Jacobian differs')
    report=dict(status='PREPARED_CPU_ONLY',selected_model=best,ranking=ranks,
        checkpoint=str(path.resolve()),sha256=sha(path),network=wide_network(),
        source=source,states_audited=len(x),maximum_action_error=error,
        maximum_local_Jacobian_error=jacobian_error,
        online_updates_after_expansion=0,runtime_recipe_ready=False,
        source_critic_Adam_std_RNG='retained; Actor moments explicitly mapped; not equal optimization trajectory',
        paired_training=dict(status='DESIGN_ONLY',authorization_online_updates=0,
            source=source,arms=['512_256_128','1024_512_256'],
            same=['31D interface','Critic312','reward','sampling','LR','std','physics','sample budget','seed'],
            comparison=['full same-case capability','per-cohort EV','KL','clip','retention KL','samples and wall time']))
    write(output/'wide_candidate.json',report)
    return report


def report(output,terminal):
    lines=['# Motion MuJoCo跨引擎比较','',f"状态：{terminal['status']}；PPO0；预算物理秒：{terminal['physical_seconds']:.2f}（中断尾部按上界收费）。",
        '', '三个模型使用各自父spec身份、同一current Plant/solver与60个条件。每条件单环境，完整30秒受控窗口；不是Native八环境认证。',
        '', '|模型|完整矩阵|跌倒|旧22命令通过|全部通过|','|---|---:|---:|---:|---:|']
    for row in terminal['ranking']:
        lines.append(f"|{row['model']}|{row['matrix_complete']}|{row['falls']}|{row['nominal_old_passed']}/22|{row['passed']}/{row['cases']}|")
    lines += ['', '排序：完整证据优先，其次跌倒更少、旧运动通过更多、总通过更多；完全同分保留请求中的顺序。各任务分项及全部失败原因见results.json，轨迹只存NPZ。',
        '', '## 分条件结果','', '|模型|条件|命令|受控秒|通过|失败分项|','|---|---|---|---:|---|---|']
    for r in terminal['results']:
        a=r['assessment'];lines.append(f"|{r['model']}|{r['case']['case_id']}|{r['case']['command']}|{a['metrics']['controlled_seconds']:.2f}|{a['accepted']}|{', '.join(a['failed_checks'])}|")
    lines += ['', '## 扩容与能力边界','',
        'Actor扩宽仅验证初始化动作与局部反馈的一致性，不是新技能或收敛保证。原小→大实验同时修改过Critic、奖励与环境数，未单独证明容量因果。',
        '固定高度运动、急停急启、压弯、波浪地形需要对应数据与任务参考；压弯不应全局罚零roll，地形不能无条件罚等腿长。Recovery/Jump继续独立FSM策略。',
        '单帧31D扩宽不会增加隐藏执行状态的可观测性；如同配方扩宽后仍明确停滞，再研究短历史。',
        '', '扩宽候选：'+json.dumps(terminal.get('wide_candidate',{}),ensure_ascii=False)]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def execute(request_path):
    request,spec,matrix,paths=read_request(request_path)
    torch.set_num_threads(1);actors=load_actors(request)
    model,refs,cfg,readback=plant(request,spec)
    readbacks=[initial_readback(model,refs,case_configuration(spec,cfg,row),row) for row in matrix]
    paths.update(readback['plant'].get('input_sha256',{}))
    paths[str(Path(request['xml']).resolve())]=sha(request['xml'])
    for file in [Path(__file__),Path(__file__).with_name('diagnose_motion_closed_loop_robustness.py'),
                 Path(__file__).with_name('motion_mcu_playback.py')]:paths[str(file.resolve())]=sha(file)
    from native_stand.students import motion_cross_engine,actor_widening,mcu_interface,model as model_module
    from native_stand.students import motion_mcu_execution,motion_yaw_quality,mujoco_solver
    for module in (motion_cross_engine,actor_widening,mcu_interface,model_module,motion_mcu_execution,motion_yaw_quality,mujoco_solver):
        paths[str(Path(module.__file__).resolve())]=sha(module.__file__)
    return request,spec,matrix,paths,actors,model,refs,cfg,readback,readbacks


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--run',action='store_true');args=parser.parse_args()
    values=execute(args.request)
    request,spec,matrix,paths,actors,model,refs,cfg,readback,initial=values
    if not args.run:
        print(json.dumps(dict(status='PLAN_ONLY',models=request['models'],cases_per_model=len(matrix),
            total_cases=len(matrix)*len(actors),limits=request['limits'],initial_readback=initial,
            execution=readback['execution'],simulation_steps=0,online_PPO=0),ensure_ascii=False));return
    from diagnose_motion_closed_loop_robustness import run_case
    from diagnose_motion_mcu_gap import spectrum
    output=Path(request['output_dir']);limits=request['limits']
    if output.exists():raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free<limits['free_reserve_bytes']+limits['storage_bytes']:
        raise RuntimeError('comparison storage plus free reserve unavailable')
    output.mkdir();write(output/'manifest.json',dict(version=VERSION,request=str(args.request.resolve()),
        request_sha256=sha(args.request),models=request['models'],cases=matrix,inputs_sha256=paths,
        protocol=dict(controlled_seconds=30,hold_window_s=[10,18],standing_window_s=[2,30],
                      posture='native gravity angles',contact='actual wheel/ground resultant',envs=1),
        PPO=0,hardware_identified=False))
    write(output/'plant_readback.json',readback);write(output/'initial_readback.json',initial)
    results,reused_inputs,history_lower,history_upper,history_wall=reusable_records(request,matrix,actors)
    paths.update(reused_inputs)
    write(output/'evidence_reuse.json',dict(records=len(results),inputs_sha256=reused_inputs,
        historical_physical_interval_s=[history_lower,history_upper],historical_wall_seconds=history_wall))
    start=time.monotonic();deadline=start+limits['wall_seconds']-15
    attempts={};physical=history_upper;status='RUNNING';failure=None;wide=None
    completed_keys={(r['model'],r['case']['case_id']) for r in results}
    snapshots={n:{k:v.clone() for k,v in a.state_dict().items()} for n,a in actors.items()}
    def settle():
        write(output/'budget.json',dict(status=status,wall_used_s=time.monotonic()-start,
            historical_wall_seconds=history_wall,cumulative_wall_used_s=history_wall+time.monotonic()-start,
            historical_physical_interval_s=[history_lower,history_upper],reused_cases=len(completed_keys),
            physical_seconds=physical,limits=limits,attempts=attempts,online_PPO=0,optimizer_steps=0))
    def check_budget():
        used=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if request.get('recovery'):
            used+=sum(p.stat().st_size for p in Path(request['recovery']['parent_output']).rglob('*') if p.is_file())
        if used>limits['storage_bytes']-32*2**20 or shutil.disk_usage(output).free<limits['free_reserve_bytes']:
            raise RuntimeError('comparison runtime storage reserve')
        if time.monotonic()>deadline:raise TimeoutError('comparison wall limit')
    def interrupted(signum,frame):raise InterruptedError('comparison signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    settle()
    try:
        for row in matrix:
            for name,actor in actors.items():
                if (name,row['case_id']) in completed_keys:continue
                check_budget();uncertainty,velocity,prefix=scenario(row)
                if physical+30+prefix>limits['physics_seconds']:raise TimeoutError('comparison physical limit')
                key=f'{len(results):03}_{name}_{row["case_id"]}'
                attempts[key]=dict(status='RUNNING',case_id=row['case_id'],model=name,settled=False)
                settle();telemetry=Telemetry(spec,row,prefix)
                current=case_configuration(spec,cfg,row)
                try:
                    with torch.inference_mode():
                        summary,arrays=run_case(model,refs,actors,current,uncertainty,name,30.,deadline,
                            predecessor='source4500' if prefix else None,warmup=prefix,target=row['command'],
                            constraint_observer=telemetry,command_age_from_handover=True,initial_velocity_w=velocity)
                except BaseException as exc:
                    observed=telemetry.last_physics_s
                    physical+=float(getattr(exc,'physical_seconds',observed))
                    attempts[key].update(status='FAILED',settled=True,error=repr(exc),
                        observed_physical_seconds=observed,partial_physics_upper_bound_s=observed+.01)
                    write(output/(key+'_fault_tail.json'),dict(error=repr(exc),tail=telemetry.tail))
                    raise
                physical+=summary['physical_seconds'];audit=inspect_trace(arrays,actor,prefix,teacher=actors['source4500'])
                arrays_path=output/(key+'.npz');np.savez_compressed(arrays_path,**arrays)
                assessment=score(arrays,row,prefix=prefix,termination=summary['termination'])
                mask=(arrays['time']>prefix+2+1e-8)&(arrays['time']<=prefix+30+1e-8)
                assessment['metrics']['spectra']={k:spectrum(arrays['native_posture'][mask,i],.01)
                    for i,k in enumerate(('pitch','roll'))} if int(mask.sum())>=16 else {}
                record=dict(model=name,tested_model=next(m for m in request['models'] if m['name']==name),
                    case=row,status='COMPLETED',termination=summary['termination'],
                    physical_seconds=summary['physical_seconds'],assessment=assessment,readback=audit,
                    handover=summary['handover'],effective_uncertainty=uncertainty,initial_velocity_w=velocity,
                    trace=dict(path=arrays_path.name,sha256=sha(arrays_path)))
                write(output/(key+'.json'),record);results.append(record)
                attempts[key].update(status='COMPLETED',settled=True,physical_seconds=summary['physical_seconds'])
                settle();print(json.dumps(dict(completed=len(results),total=180,model=name,
                    case=row['case_id'],accepted=assessment['accepted'],failed=assessment['failed_checks'])),flush=True)
        status='COMPLETED'
        if request.get('prepare_wide'):
            check_budget();wide=prepare_wide(request,results,output,actors)
    except (TimeoutError,InterruptedError) as exc:
        status='STOPPED_BOUNDED';failure=repr(exc)
    except BaseException as exc:
        status='FAILED';failure=repr(exc)
    finally:
        unchanged=all(torch.equal(v,actors[n].state_dict()[k]) for n,a in snapshots.items() for k,v in a.items())
        if not unchanged:status='FAILED';failure='source Actor mutated'
        changed=[p for p,d in paths.items() if sha(p)!=d]
        if changed:status='FAILED';failure='read-only source changed: '+str(changed)
        terminal=dict(status=status,failure=failure,wall_seconds=time.monotonic()-start,
            historical_wall_seconds=history_wall,cumulative_wall_seconds=history_wall+time.monotonic()-start,
            actual_physical_seconds_interval=[physical-history_upper+history_lower,physical],
            physical_seconds=physical,completed_cases=len(results),expected_cases=180,results=results,
            ranking=ranking(results,list(actors)),wide_candidate=wide,models_unchanged=unchanged,
            online_PPO=0,optimizer_steps=0,capability_certified=False)
        write(output/'results.json',dict(results=results));write(output/'terminal_report.json',terminal)
        settle();report(output,terminal)
    if status=='FAILED':raise RuntimeError(failure)


if __name__=='__main__':main()
