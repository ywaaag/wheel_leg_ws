"""Prepare once, then replay handover bug candidate and ideal control in MuJoCo GUI.

--prepare performs two bounded CPU rollouts under explicit user authorization.
--viewer only replays full stored qpos; no policy inference or physics stepping.
Default prints a zero-write plan. Existing search evidence stays read-only.
"""
import argparse
import copy
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from diagnose_motion_handover import ROOT, decode_log, initialize_from_log, model_plan, read_csv, run_case, sha
from native_stand.command_v6.domain.experiment_naming import new_experiment_name


def geometric_wheel_mapping(configuration):
    """Electrical Left=geometric left/ch5(11ms), Right=ch2(7ms)."""
    cfg=copy.deepcopy(configuration)
    transport=cfg['target_delay_ms'];rise=cfg['torque_tau_ms']
    if any(transport[j] for j in (2,5)) and any(rise[j] for j in (2,5)):
        raise ValueError('cannot reinterpret combined wheel transport and response')
    if any(transport[j] for j in (2,5)):
        if sorted([transport[2],transport[5]])!=[7,11]:raise ValueError('unknown wheel delay values')
        transport[2],transport[5]=7,11
        wheel='transport7_11'
    elif any(rise[j] for j in (2,5)):
        if sorted([rise[2],rise[5]])!=[7,11]:raise ValueError('unknown wheel response values')
        rise[2],rise[5]=7,11
        wheel='rise7_11'
    else:wheel='ideal'
    cfg['name']=cfg['name']+'_geometric_wheels'
    if 'search_values' in cfg:cfg['search_values']['wheel']=wheel
    return cfg


def prepare(search_dir, case_name, out, *, wheel_mapping='geometric'):
    manifest=json.loads((search_dir/'manifest.json').read_text())
    case=json.loads((search_dir/(case_name+'.json')).read_text())
    configuration=(geometric_wheel_mapping(case['configuration']) if wheel_mapping=='geometric'
                   else copy.deepcopy(case['configuration']))
    changed=any(configuration[k]!=case['configuration'][k] for k in ('target_delay_ms','torque_tau_ms'))
    if manifest['version']!='motion31_handover_coarse_local_search_v1':raise ValueError('explicit search run required')
    inputs=manifest['input_sha256']
    def find(suffix):
        paths=[Path(p) for p in inputs if p.endswith(suffix)]
        if len(paths)!=1:raise ValueError('unambiguous input required '+suffix)
        path=paths[0]
        if sha(path)!=inputs[str(path)]:raise ValueError('original input SHA changed '+str(path))
        return path
    csv=find('.csv')
    checkpoints=[Path(p) for p in inputs if p.endswith('.pt')]
    if len(checkpoints)!=1:raise ValueError('exact one checkpoint required')
    checkpoint=checkpoints[0]
    if sha(checkpoint)!=manifest['checkpoint_sha256']:raise ValueError('checkpoint identity differs')
    spec_path=find('/spec.json');xml=find('.xml');spec=json.loads(spec_path.read_text())
    _,obs,_,t,_=read_csv(csv)
    torch.set_num_threads(1)
    model,refs,actor,identity=model_plan(SimpleNamespace(spec=spec_path,checkpoint=checkpoint,xml=xml,
        wall_seconds=120,commands=[0.],yaw_commands=[],viewer=False),spec)
    at=case['event_sample_time_s'];ix=int(np.argmin(abs(t-at)))
    if abs(t[ix]-at)>1e-9:raise ValueError('original event sample missing')
    initial,fit=initialize_from_log(model,refs,obs[ix],case['assumed_root_vx_mps'])
    history=[]
    for j in range(ix-15,ix):
        s=decode_log(obs[j]);history.append((int(np.floor((t[j]-at)*1000)),np.r_[s['gyro'],s['gravity'],s['q'],s['dq']]))
    ideal=dict(name='ideal_same_initial_state',target_delay_ms=[0]*6,pd_feedback_delay_ms=0,
               torque_tau_ms=[0]*6,dq_delay_ms=0,imu_delay_ms=0)
    out.mkdir(parents=True,exist_ok=False)
    def write(name,data):
        with (out/name).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
    saved=dict(version='motion31_handover_visual_replay_v1',xml=str(xml),xml_sha256=sha(xml),
        parent_search=str(search_dir.resolve()),parent_manifest_sha256=sha(search_dir/'manifest.json'),
        source_case=case_name,source_case_sha256=sha(search_dir/(case_name+'.json')),
        wheel_mapping=wheel_mapping,source_configuration=case['configuration'],effective_configuration=configuration,
        electrical_sides={'geometric_left':'model_right/ch5/11ms','geometric_right':'model_left/ch2/7ms'},
        mapping_corrected_from_source=changed,
        checkpoint_sha256=sha(checkpoint),csv_sha256=sha(csv),input_sha256=identity['input_sha256'],
        source_sha256={str(Path(__file__).resolve()):sha(__file__),
            str(ROOT/'scripts/diagnose_motion_handover.py'):sha(ROOT/'scripts/diagnose_motion_handover.py')},
        initial_fit=fit,physics_dt_s=.001,record_dt_s=.01,
        replay_semantics='qpos/qvel post-step100Hz, mj_forward only; no physical resimulation in viewer',
        authorization='user requested GUI comparison of authorized handover search',
        online_PPO=0,wall_limit_s=120,storage_max_bytes=16*2**20)
    write('manifest.json',saved);started=time.monotonic();results=[]
    before={k:v.clone() for k,v in actor.state_dict().items()}
    status='FAILED'
    try:
        with torch.inference_mode():
            for name,cfg in [('candidate',configuration),('ideal',ideal)]:
                summary,arrays=run_case(model,refs,actor,initial,history,obs[ix],cfg=cfg,cache_mode=case['cache_hypothesis'],
                    trace_time=t,observations=obs,sample_time=at,duration=2.5,compare_seconds=.8,
                    deadline=started+120,record_full_state=True)
                if summary['termination']=='WALL_BUDGET':raise TimeoutError('visual preparation budget')
                # At reset both clips start with exactly the same physical state.
                arrays['replay_time']=np.r_[0.,arrays['state_time']]
                arrays['replay_qpos']=np.vstack([initial.qpos,arrays['full_qpos_post']])
                arrays['replay_qvel']=np.vstack([initial.qvel,arrays['full_qvel_post']])
                if name=='candidate' and not changed:
                    with np.load(search_dir/(case_name+'.npz')) as old:
                        if old['target'].shape!=arrays['target'].shape:raise ValueError('candidate replay duration differs')
                        error=float(abs(old['target']-arrays['target']).max())
                    summary['original_search_target_max_error']=error
                    if error>1e-6:raise ValueError('visual reproduction differs from original search')
                elif name=='candidate':
                    summary['original_search_target_max_error']=None
                    summary['difference_reason']='new geometric wheel mapping ch2=7/ch5=11ms; expected new trajectory, old source read-only'
                np.savez_compressed(out/(name+'.npz'),**arrays)
                write(name+'.json',summary);results.append(dict(name=name,**summary))
        status='COMPLETED'
    finally:
        unchanged=all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
        write('budget.json',dict(version='mujoco_visual_preparation_budget_v1',attempt_id=uuid.uuid4().hex,
            status=status,wall_used_s=time.monotonic()-started,wall_limit_s=120,maximum_cases=2,
            physics_steps_used=sum(r['physics_steps'] for r in results),maximum_physics_steps=5000,
            online_PPO=0,storage_max_bytes=16*2**20))
        write('terminal_report.json',dict(status=status,model_unchanged=unchanged,results=results,online_PPO=0))
        if not unchanged:raise RuntimeError('Actor changed')
    print('REPLAY_DIR='+str(out.resolve()),flush=True)


def load_replay(replay_dir,initial_clip=None):
    """Validate stored physical trajectories before opening a window."""
    meta=json.loads((replay_dir/'manifest.json').read_text())
    deployment=meta['version']=='motion31_deployment_log_visual_replay_v1'
    terminal=json.loads((replay_dir/('terminal.json' if deployment else 'terminal_report.json')).read_text())
    if meta['version'] not in ('motion31_handover_visual_replay_v1','motion31_deployment_log_visual_replay_v1') or terminal['status']!='COMPLETED':
        raise ValueError('completed explicit replay required')
    xml=Path(meta['xml'])
    if sha(xml)!=meta['xml_sha256']:raise ValueError('MJCF changed')
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    clips=[]
    names=meta['clips'] if deployment else ('candidate','ideal')
    if initial_clip is not None and initial_clip not in names:raise ValueError('unknown clip '+initial_clip)
    for name in names:
        if deployment and sha(replay_dir/(name+'.npz'))!=meta['clip_sha256'][name]:raise ValueError('replay clip SHA differs')
        with np.load(replay_dir/(name+'.npz')) as z:
            clips.append(dict(name=name,t=z['replay_time'].copy(),q=z['replay_qpos'].copy(),dq=z['replay_qvel'].copy(),
                              telemetry={k:z[k].copy() for k in ('actor_input','mean','requested','adopted','q','dq','torque','preclip','state','contact') if k in z} if deployment else {},
                              physics=[(k,str(v)) for k,v in meta.get('configuration',{}).items()]))
        c=clips[-1]
        expected={'actor_input':31,'mean':6,'requested':6,'adopted':6,'q':6,'dq':6,'torque':6,'preclip':6,'state':6,'contact':2}
        for k,v in c['telemetry'].items():
            if v.shape!=(len(c['t'])-1,expected[k]) or not np.isfinite(v).all():raise ValueError('invalid replay telemetry '+k)
        if (c['t'].ndim!=1 or len(c['t'])<2 or c['t'][0]!=0 or np.any(np.diff(c['t'])<=0)
            or c['q'].shape!=(len(c['t']),model.nq) or c['dq'].shape!=(len(c['t']),model.nv)
            or not all(np.isfinite(c[k]).all() for k in ('t','q','dq'))):
            raise ValueError('invalid full-state replay '+name)
    return model,data,list(names),clips


def viewer(replay_dir,speed,wall,*,initial_clip=None,start_paused=False,panel=False,panel_host='127.0.0.1',panel_port=8766):
    from native_stand.students.replay_panel import ReplayPlan
    from native_stand.students.motion_panel import start, stop
    loaded=load_replay(replay_dir,initial_clip)
    plan=ReplayPlan(loaded[2],initial_clip or loaded[2][0],speed,start_paused)
    server=None
    try:
        if panel:
            server,_=start(plan,panel_port,panel_host)
            print(f'REPLAY PANEL http://{panel_host}:{panel_port}/',flush=True)
        return _viewer(loaded,plan,wall)
    finally:
        stop(server)


def _viewer(loaded,plan,wall):
    """Pure kinematic playback; Space pause, N/1/2 clip selection, R restart."""
    import mujoco.viewer
    import threading
    from native_stand.students.replay_panel import frame_telemetry, kinematic_telemetry, ReplayTiming
    from validate_native_mujoco import binding
    model,data,names,clips=loaded
    refs=binding(model)
    root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot')
    def key(code):
        current=plan.snapshot();index=names.index(current['model'])
        if code in (32,80,112):plan.apply({'paused':not current['paused']})
        elif code in (78,110):plan.apply({'model':names[(index+1)%len(names)]})
        elif 49<=code<49+len(names):plan.apply({'model':names[code-49]})
        elif code in (82,114):plan.apply({'reset':True})
    start=time.monotonic();last=start;phase=0.;last_index=-1
    timing=ReplayTiming(start)
    existing_threads=set(threading.enumerate())
    with mujoco.viewer.launch_passive(model,data,key_callback=key) as display:
        viewer_threads=[t for t in threading.enumerate() if t not in existing_threads]
        display.cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING;display.cam.trackbodyid=root
        display.cam.distance=1.7;display.cam.azimuth=135;display.cam.elevation=-15
        print('GUI: '+', '.join(f'{i+1}={n}' for i,n in enumerate(names))+'; N=next, Space/P=pause, R=restart; stored physical trajectories, viewer does not step physics',flush=True)
        while display.is_running() and time.monotonic()-start<wall:
            now=time.monotonic()
            controls=plan.take()
            if controls['restart']:phase=0.
            if not controls['paused']:phase+=(now-last)*controls['playback_speed']
            index=names.index(controls['model'])
            last=now;clip=clips[index]
            # One-second final-frame dwell before repeating.
            elapsed=phase%(float(clip['t'][-1])+1.)
            frame=min(np.searchsorted(clip['t'],elapsed,side='right')-1,len(clip['t'])-1)
            frame=max(0,frame)
            with display.lock():
                data.qpos[:]=clip['q'][frame];data.qvel[:]=clip['dq'][frame];data.time=float(clip['t'][frame])
                mujoco.mj_forward(model,data)
            if index!=last_index:
                print('PLAYING '+clip['name'],flush=True);last_index=index
            telemetry=frame_telemetry(clip,frame)
            telemetry.update(kinematic_telemetry(model,data,refs))
            telemetry.update(clip_time_s=float(clip['t'][frame]),clip_duration_s=float(clip['t'][-1]),
                             simulation_time_s=float(clip['t'][frame]),wall_time_s=now-start,
                             playback_speed=controls['playback_speed'],
                             physics=clip['physics'])
            telemetry.update(timing.sample(clip['name'],float(clip['t'][frame]),now,controls['restart']))
            plan.publish(telemetry)
            display.sync();time.sleep(.005)
    # Handle.close only requests exit; join GUI cleanup before Python unloads GL.
    for thread in viewer_threads:
        thread.join(timeout=3.)
        if thread.is_alive():raise RuntimeError('MuJoCo viewer did not complete shutdown')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search-dir',type=Path);p.add_argument('--case',default='case_039')
    p.add_argument('--output-dir',type=Path);p.add_argument('--replay-dir',type=Path)
    p.add_argument('--wheel-mapping',choices=['geometric','source'],default='geometric',
                   help='new preparations default confirmed geometric sides; source only replays historical assumption')
    p.add_argument('--prepare',action='store_true');p.add_argument('--viewer',action='store_true')
    p.add_argument('--speed','--playback-speed',dest='speed',type=float,default=.5);p.add_argument('--wall-seconds',type=float,default=600)
    p.add_argument('--clip');p.add_argument('--start-paused',action='store_true')
    p.add_argument('--panel',action='store_true');p.add_argument('--no-panel',action='store_true')
    p.add_argument('--panel-host',default='127.0.0.1');p.add_argument('--panel-port',type=int,default=8766)
    a=p.parse_args()
    if not .1<=a.speed<=2 or not 1<=a.wall_seconds<=1800:raise ValueError('speed.1..2 and wall1..1800 required')
    replay=a.replay_dir
    if a.prepare:
        if not a.search_dir:raise ValueError('--prepare needs --search-dir')
        replay=a.output_dir or ROOT/'sim2sim'/new_experiment_name('motion_handover_visual_'+uuid.uuid4().hex[:6])
        prepare(a.search_dir,a.case,replay,wheel_mapping=a.wheel_mapping)
    if a.viewer:
        if replay is None:raise ValueError('--viewer needs --replay-dir or --prepare')
        viewer(replay,a.speed,a.wall_seconds,initial_clip=a.clip,start_paused=a.start_paused,panel=a.panel and not a.no_panel,panel_host=a.panel_host,panel_port=a.panel_port)
    if not a.prepare and not a.viewer:
        print(json.dumps(dict(prepare_simulation_cases=2,prepare_max_physics_steps=5000,
            viewer='stored full qpos only; repeat without output writes',simulation=0,online_PPO=0,files_written=0)))


if __name__=='__main__':main()
