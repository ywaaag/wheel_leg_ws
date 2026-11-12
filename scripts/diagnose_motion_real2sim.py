"""CPU MuJoCo timing/previous sensitivity sweep. Plan is zero-write; --run samples physics.

No Isaac, PPO, hardware controller or spring compensation is started. This uses
one stable simulated starting state across cases, NOT a fabricated LQR controller.
"""
import argparse,json,time,uuid
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from native_stand.students.data import sha
from native_stand.students.real2sim_delay import TargetTransport,ObservationHistory,torque_response,scenarios
from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from validate_native_mujoco import state
from validate_motion31_mujoco import plan as model_plan

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SPEC=ROOT/'configs/experiments/2026_10_18-23_10__motion1000_differential85_5000_retry/spec.json'
DEFAULT_CP=ROOT/'isolated/2026_10_19-19_55__motion3250_small31_dagger_round2/round_01/fit/best_validation.pt'


def sensors(model,data,refs):
    rotation=data.xmat[refs['root']].reshape(3,3)
    velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,refs['root'],velocity,0)
    return np.r_[rotation.T@velocity[:3],rotation.T@np.array([0.,0.,-1.]),data.qpos[refs['qpos']],data.qvel[refs['dofs']]]


def run_case(model,refs,actor,cfg,*,warmup_s,duration_s,deadline,shadow_seed):
    from native_stand.students.mcu_interface import encode
    from native_stand.students.interface import targets
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    limit=np.array([40,40,5.5,40,40,5.5]);legs=[0,1,3,4];wheels=[2,5]
    bounds=torch.tensor(model.jnt_range[refs['joints']],dtype=torch.float32)[None]
    for i,j in enumerate(refs['joints']):
        if not model.jnt_limited[j]:bounds[0,i]=torch.tensor([-float('inf'),float('inf')])
    adopted=np.zeros(6);requested=adopted.copy();demand=adopted.copy();actual=demand.copy();previous=adopted.copy()
    transport=TargetTransport([0]*6,adopted);history=ObservationHistory(sensors(model,data,refs))
    rows=[];failure=None;event_tick=round(warmup_s*1000);total=round((warmup_s+duration_s)*1000);started=time.monotonic()
    physics_steps=0;actor_calls=0
    for tick in range(total):
        if tick%100==0 and time.monotonic()>deadline:failure='WALL_BUDGET';break
        pre=sensors(model,data,refs);history.append(tick,pre)
        event=tick>=event_tick
        if tick==event_tick:
            transport=TargetTransport(cfg['target_delay_ms'],adopted)
            if cfg['cache']=='shadow_seed_once':previous=shadow_seed.copy()
        if tick%10==0:
            policy_pre=pre.copy()
            sensed=pre.copy()
            if event:
                sensed[12:18]=history.at(tick-cfg['dq_delay_ms'])[12:18]
                sensed[:6]=history.at(tick-cfg['imu_delay_ms'])[:6]
            # previous is actual PD-adopted target unless this case explicitly changes it.
            cache=previous.copy()
            if event and cfg['cache']=='measured_wheels':cache[wheels]=pre[12:18][wheels]
            obs=encode(torch.tensor(sensed[:3],dtype=torch.float32)[None],torch.tensor(sensed[3:6],dtype=torch.float32)[None],
                torch.tensor(sensed[6:12],dtype=torch.float32)[None],torch.tensor(sensed[12:18],dtype=torch.float32)[None],
                torch.tensor(cache[legs],dtype=torch.float32)[None],torch.tensor(cache[wheels],dtype=torch.float32)[None],
                torch.zeros(1,6),'Motion',torch.tensor([[cfg['command_vx'] if event else 0.,0.,.34]]))
            raw=actor.mean(obs)
            actor_calls+=1
            if not torch.isfinite(raw).all():failure='NONFINITE_ACTION';break
            decoded=targets(raw,torch.zeros(1,6),bounds);requested=np.zeros(6)
            requested[legs]=decoded['leg_executed'][0].numpy();requested[wheels]=decoded['wheel_executed'][0].numpy()
            transport.submit(tick,requested)
        delivered=transport.delivered(tick)
        if tick%5==0:
            adopted=delivered.copy()
            feedback=history.at(tick-cfg['pd_feedback_delay_ms']) if event else pre
            q=feedback[6:12];dq=feedback[12:18]
            demand=np.zeros(6);demand[legs]=60*(adopted[legs]-q[legs])-2*dq[legs];demand[wheels]=adopted[wheels]-dq[wheels]
            demand=np.clip(demand,-limit,limit)
        actual=torque_response(actual,demand,cfg['torque_tau_ms'] if event else [0]*6)
        data.ctrl[refs['actuators']]=actual/refs['gears']
        mujoco.mj_step(model,data)
        physics_steps+=1
        previous=adopted.copy()
        if tick%10==9:
            angular,gravity,s=state(model,data,refs)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():failure='NONFINITE_STATE'
            elif s['closure_m']>.005:failure='CLOSURE'
            elif s['up']<.5:failure='UPRIGHT'
            if event:
                rows.append({'time':(tick-event_tick+1)/1000.,'actor_input':obs[0].numpy().copy(),
                    'requested_target':requested.copy(),'delivered_target':delivered.copy(),'PD_adopted_target':adopted.copy(),
                    'PD_limited_demand':demand.copy(),'actual_applied_torque':actual.copy(),
                    'PD_feedback_q':feedback[6:12].copy(),'PD_feedback_dq':feedback[12:18].copy(),
                    'joint_q_pre':policy_pre[6:12].copy(),'joint_dq_pre':policy_pre[12:18].copy(),
                    'state':np.array([s['vx'],s['vy'],s['yaw_rate'],s['height'],np.arctan2(gravity[2],-gravity[1]),np.arctan2(gravity[0],-gravity[1]),s['closure_m']]),
                    'contact':np.array(s['contact_force_n'])})
            if failure:break
    arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]} if rows else {}
    summary={'name':cfg['name'],'configuration':cfg,'termination':failure,'samples':len(rows),'duration_s':len(rows)*.01,
        'physics_steps':physics_steps,'policy_actor_calls':actor_calls,'simulation_envs':1,'PPO_updates':0,
        'wall_s':time.monotonic()-started,'warmup_s':warmup_s,'initialization':'native normal zero geometry; common policy-driven warmup, no LQR or log-pose teleport','capability_certified':False}
    if rows:
        s=arrays['state'];qt=arrays['requested_target'][:,legs];delta=np.diff(qt,axis=0)
        summary.update(pitch_abs_peak_rad=float(abs(s[:,4]).max()),roll_abs_peak_rad=float(abs(s[:,5]).max()),vx_mean_mps=float(s[:,0].mean()),
            both_contact_fraction=float((arrays['contact']>2).all(1).mean()),leg_target_step_p95_rad=np.percentile(abs(delta),95,axis=0).tolist() if len(delta) else None,
            max_leg_target_step_rad=float(abs(delta).max()) if len(delta) else None,
            torque_saturation_fraction=(abs(arrays['PD_limited_demand'])>=limit-1e-6).mean(0).tolist(),
            warmup_valid=bool(tick>=event_tick),first_event_actor_input=arrays['actor_input'][0].tolist(),
            first_requested_target=arrays['requested_target'][0].tolist())
    else:summary['warmup_valid']=False
    return summary,arrays


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,default=DEFAULT_SPEC);p.add_argument('--checkpoint',type=Path,default=DEFAULT_CP)
    p.add_argument('--xml',type=Path,default=Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml'))
    p.add_argument('--output-dir',type=Path);p.add_argument('--run',action='store_true');p.add_argument('--wall-seconds',type=float,default=600)
    p.add_argument('--warmup-seconds',type=float,default=8);p.add_argument('--duration-seconds',type=float,default=6)
    p.add_argument('--cases',nargs='+',choices=[r['name'] for r in scenarios()]);a=p.parse_args()
    if not 1<=a.duration_seconds<=30 or not 1<=a.warmup_seconds<=20 or not 10<=a.wall_seconds<=1800:raise ValueError('bounded duration/warmup/wall required')
    torch.set_num_threads(1);spec=json.loads(a.spec.read_text())
    model,refs,actor,manifest=model_plan(SimpleNamespace(spec=a.spec,checkpoint=a.checkpoint,xml=a.xml,wall_seconds=a.wall_seconds,commands=[0.],yaw_commands=[],viewer=False),spec)
    configs=[c for c in scenarios() if not a.cases or c['name'] in a.cases]
    # Last pre-oscillation logged shadow targets; exact native channel units. This
    # is a one-time inconsistent-cache hypothesis, not a reconstructed LQR state.
    shadow_seed=np.array([.05166947,.02855312,-3.5272114,-.00518679,.01055550,-5.9060574])
    manifest.update(version='motion31_real2sim_delay_diagnostic_v1',cases=configs,warmup_s=a.warmup_seconds,duration_s=a.duration_seconds,
        trace_schema={'state':['vx_mps','vy_mps','yaw_rad_s','root_height_m','native_pitch_rad','native_roll_rad','closure_m'],
            'input':'policy pre-physics; sensor history sampled at1ms','state_contact':'post-physics at100Hz',
            'target_channels':['model_left_hip','model_left_active','model_left_wheel','model_right_hip','model_right_active','model_right_wheel']},
        shadow_seed=shadow_seed.tolist(),cache='last PD-adopted target; transport at1ms, PD uses delivery at5ms boundaries',
        limits='no identified hardware dynamics; no J/B/Tc or spring changes; no LQR emulation; sensitivity NOT causal identification',
        online_PPO=0,output=str(a.output_dir) if a.output_dir else 'new date-named sim2sim directory on --run',
        extra_source_sha256={str(Path(__file__).resolve()):sha(__file__),str(ROOT/'src/native_stand/students/real2sim_delay.py'):sha(ROOT/'src/native_stand/students/real2sim_delay.py')})
    if not a.run:print(json.dumps(manifest,indent=2));return
    out=a.output_dir or ROOT/'sim2sim'/new_experiment_name('motion3250_real2sim_delay_sweep')
    from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
    validate_new_experiment_name(out.name);out.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    write('manifest.json',manifest);started=time.monotonic();results=[];status='FAILED';weights={k:v.clone() for k,v in actor.state_dict().items()}
    from native_stand.command_v6.completion import write as atomic_json
    book={'version':'mujoco_real2sim_diagnostic_budget_v1','attempt_id':uuid.uuid4().hex,
        'authorization':'user approved bounded real2sim diagnosis; no PPO','PPO_updates':0,
        'wall_limit_s':a.wall_seconds,'maximum_cases':len(configs),'maximum_physics_steps':len(configs)*round((a.warmup_seconds+a.duration_seconds)*1000),
        'storage_max_bytes':128*2**20,'attempts':{},'status':'RUNNING'}
    atomic_json(out/'budget.json',book)
    try:
        with torch.inference_mode():
            for c in configs:
                book['attempts'][c['name']]={'status':'RUNNING'};atomic_json(out/'budget.json',book)
                summary,arrays=run_case(model,refs,actor,c,warmup_s=a.warmup_seconds,duration_s=a.duration_seconds,deadline=started+a.wall_seconds,shadow_seed=shadow_seed)
                if arrays:np.savez_compressed(out/(c['name']+'.npz'),**arrays)
                write(c['name']+'.json',summary);results.append(summary)
                book['attempts'][c['name']]={'status':'COMPLETED_DIAGNOSTIC','physics_steps':summary['physics_steps'],'policy_actor_calls':summary['policy_actor_calls'],'physical_termination':summary['termination']}
                atomic_json(out/'budget.json',book)
                print(json.dumps(summary),flush=True)
                if summary['termination']=='WALL_BUDGET':break
                if sum(f.stat().st_size for f in out.iterdir() if f.is_file())>book['storage_max_bytes']:raise RuntimeError('diagnostic storage budget exhausted')
        status='COMPLETED' if len(results)==len(configs) and not any(r['termination']=='WALL_BUDGET' for r in results) else 'STOPPED_BOUNDED'
    finally:
        unchanged=all(torch.equal(v,actor.state_dict()[k]) for k,v in weights.items())
        for entry in book['attempts'].values():
            if entry['status']=='RUNNING':entry['status']='FAILED'
        book.update(status=status,wall_used_s=time.monotonic()-started,physics_steps_used=sum(r['physics_steps'] for r in results))
        atomic_json(out/'budget.json',book)
        write('terminal_report.json',{'status':status,'model_unchanged':unchanged,'online_PPO':0,'wall_s':time.monotonic()-started,'results':results})
        if not unchanged:raise RuntimeError('Actor changed')
        for path,digest in {**manifest['input_sha256'],**manifest['extra_source_sha256']}.items():
            if sha(path)!=digest:raise RuntimeError('read-only input changed: '+path)
    print('OUTPUT='+str(out.resolve()),flush=True)
if __name__=='__main__':main()
