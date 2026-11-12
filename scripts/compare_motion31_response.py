"""Bounded CPU tau0/2 pairing and a read-only colleague-MJCF bridge.

--check prints a zero-integration plan. --run consumes one independent bounded
diagnostic attempt; no PPO, no retry, no changes to either source MJCF.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch

from native_stand.students.motion_response_probe import VERSION,prepare_colleague

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'configs/experiments/2026_10_29-11_55__motion4500_small_compliance3000/spec.json'
COLLEAGUE=Path('/home/kellen/桌面/11 (1)/11/wheeled-legged_RL/sim2real_RL/sim2sim/rmuc_peak489_v9.xml')
ONNX=ROOT/'deployment_protocol/2026_10_26-20_29__motion4500_precise_small31_onnx/actor.onnx'
LOG=ROOT/'logs/Ozone_DataSampling_261028.csv'
VALIDATION=ROOT/'artifacts/2026_10_29__motion4500_compliance3000_engineering/firmware_cpu/validation.json'
PT_SHA='efad012efcced3a9ac58dc51839dea98b81598de289dea8ef0986a0813d139a1'
ONNX_SHA='776201d861dbf1f00b6824f2c2615068a3206eb45b38529e65e7591bc150e617'
LEGS=[0,1,3,4]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def write(path,value):
    tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=plain)+'\n')
    tmp.replace(path)


def matrix():
    rows=[]
    for asset,profiles in [('current',('R','F40')),('colleague',('R',))]:
        for profile in profiles:
            for yaw,prefix in [(0.,0.),(.015,0.),(-.015,0.),(0.,1.)]:
                pair=f'{asset}_{profile}_yaw{yaw:+g}_prefix{prefix:g}'
                for tau in (0.,2.):
                    rows.append(dict(id=pair+f'_tau{tau:g}',pair=pair,asset=asset,
                        profile=profile,command=[0.,0.,.34],yaw=yaw,prefix_s=prefix,
                        episode_s=30.,training=False,motor_tau_ms=tau))
    return rows


def xml_dependencies(path):
    path=Path(path).resolve();root=ET.parse(path).getroot();out={path}
    compiler=root.find('compiler');meshdir=compiler.get('meshdir','') if compiler is not None else ''
    for node in root.findall('.//mesh'):
        if node.get('file'):out.add((path.parent/meshdir/node.get('file')).resolve())
    for node in root.findall('.//include'):
        out.update(xml_dependencies(path.parent/node.get('file')))
    return out


def prepare():
    from types import SimpleNamespace as S
    from search_motion31_oscillation import inputs,setup
    from native_stand.students.model import load
    from validate_motion31_mujoco import OnnxActor
    import mujoco
    from validate_native_mujoco import binding
    spec=json.loads(BASE.read_text());parent=Path(spec['parent_spec'])
    source,validation,data=inputs(S(spec=parent,onnx=ONNX,firmware_validation=VALIDATION,csv=LOG,seconds=6.))
    pt=Path(spec['source_checkpoint'])
    if sha(pt)!=PT_SHA or sha(ONNX)!=ONNX_SHA:raise ValueError('fixed original4500 identity differs')
    actor,_=load(pt,'MotionZeroStudent');actor.eval()
    exported=OnnxActor(ONNX,ONNX.parent/'manifest.json',parent)
    with torch.inference_mode():
        check=torch.from_numpy(data['x'][[data['ix'],data['ix']+20,data['ix']+200]].astype(np.float32))
        error=float((actor(check)-torch.cat([exported.mean(v[None]) for v in check])).abs().max())
    if error>1e-5:raise ValueError('PT and exported mean differ')
    # Load/forward only. No integration during preflight.
    model,refs,base,own=setup(source,S(xml=Path(spec['xml'])))
    cm=mujoco.MjModel.from_xml_path(str(COLLEAGUE));cr=binding(cm);bridge=prepare_colleague(cm,cr)
    from motion_mcu_playback import apply_initial_pose
    cd=mujoco.MjData(cm);apply_initial_pose(cm,cd,cr,source['mcu_execution'])
    paths={BASE,parent,pt,ONNX,ONNX.parent/'manifest.json',LOG,VALIDATION,Path(validation['library']),Path(__file__).resolve()}
    paths.update(Path(p) for p in validation['source_sha256'])
    paths.update(Path(p) for p in validation.get('generated_sha256',{}))
    paths.update(xml_dependencies(spec['xml']));paths.update(xml_dependencies(COLLEAGUE))
    for name in ('motion_response_probe','motion_compliance_env','motion_compliance_eval','motion_compliance_training',
                 'motion_compliance_reward','firmware_sampling','firmware_mechanics','firmware_control','oscillation_search',
                 'real2sim_delay','real2sim_handover','model','mcu_interface','interface','motion_quality','motion_yaw_quality','mujoco_solver'):
        paths.add(ROOT/f'src/native_stand/students/{name}.py')
    for name in ('search_motion31_oscillation','validate_native_mujoco','validate_motion31_mujoco','motion_mcu_playback','diagnose_motion_real2sim','replay_motion31_shadow_csv'):
        paths.add(ROOT/f'scripts/{name}.py')
    identity={str(p):sha(p) for p in sorted(paths)}
    return spec,source,validation,data,actor,dict(input_sha256=identity,PT_ONNX_max_abs=error,
        current_plant=own,colleague_bridge=bridge,resource=dict(cpu_affinity=len(os.sched_getaffinity(0)),
        free_disk_bytes=shutil.disk_usage(ROOT).free,workers=4,worker_threads=1,parent_threads=1,GPU=False)),(model,refs,base)


def request(spec,identity):
    return dict(version=VERSION,model=str(ONNX),model_sha256=ONNX_SHA,interface='multiskill_mcu_v2',
        PD=dict(leg_Kp=20.,leg_Kd=2.,wheel_Kv=.15),matrix=matrix(),
        log_matrix=[dict(asset=a,profile=p,motor_tau_ms=tau) for a,p in [('current','R'),('current','F40'),('colleague','R')] for tau in (0.,2.)],
        budget=dict(attempts=1,scored_episodes=30,physics_steps=900000,wall_s=1200.,storage_bytes=256*2**20,
                    online_updates=0,optimizer_steps=0),
        scoring='existing motion_compliance_eval, fixed full30s; early exits remain failures',
        log_window=dict(fit=[.5,2.],holdout=[2.,6.],root_vx_assumed=0.,
            commands='recorded commands, simulated measurements, closed-loop Actor',
            initial_motor='preceding LQR debug proxy; not measured motor state/queue'),
        prefix='same frozen4500 for1s, no LQR; retain physical/execution/cache state at student boundary',
        colleague_scope=identity['colleague_bridge'],certificate_eligible=False,hardware_identified=False)


def stack_rows(rows):
    rows=[r for r in rows if r is not None]
    if not rows:return {}
    out={}
    for key in rows[0]:
        if key=='force':
            out.update({'force_'+k:np.stack([r[key][k] for r in rows]) for k in rows[0][key]})
        elif key not in ('fields','profile'):
            out[key]=np.asarray([r[key] for r in rows])
    return out


def pair_summary(results):
    pairs=[]
    for key in dict.fromkeys(r['case']['pair'] for r in results):
        a,b=sorted([r for r in results if r['case']['pair']==key],key=lambda r:r['case']['motor_tau_ms'])
        x,y=a['first_actor'],b['first_actor']
        pairs.append(dict(pair=key,initial_actor_max_abs=float(np.max(abs(np.asarray(x)-y))),
            initial_physical_identical=bool(np.array_equal(a['initial']['qpos'],b['initial']['qpos']) and
                np.array_equal(a['initial']['qvel'],b['initial']['qvel'])),
            tau0=dict(accepted=a['accepted'],survival30=a['survival30'],failed_checks=a['failed_checks'],metrics=a.get('metrics')),
            tau2=dict(accepted=b['accepted'],survival30=b['survival30'],failed_checks=b['failed_checks'],metrics=b.get('metrics'))))
    return pairs


def execute(output,spec,source,validation,data,actor,identity,setup_values,plan,inherited=None):
    from native_stand.students.motion_compliance_env import VectorEnv
    from native_stand.students.motion_compliance_eval import summarize
    from native_stand.students.firmware_sampling import profile,source_parameters
    from native_stand.students.motion_compliance_training import PROFILES
    from native_stand.students.firmware_mechanics import reproduction
    from native_stand.students.oscillation_search import features
    from search_motion31_oscillation import run_sampled_case
    from validate_native_mujoco import binding
    import mujoco
    started=time.monotonic();deadline=started+plan['budget']['wall_s'];used=0;results=[];logs=[]
    ledger=dict(status='RUNNING',attempts=1,physics_steps=0,scored_episodes=0,optimizer_steps=0,online_updates=0,
        wall_s=0.,failure=None,output_bytes=0)
    write(output/'spec.json',plan);write(output/'identity.json',identity);write(output/'ledger.json',ledger)
    def budget(reserve=0):
        if used+reserve>plan['budget']['physics_steps']:raise RuntimeError('physical budget reached')
        if time.monotonic()>=deadline:raise RuntimeError('wall budget reached')
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if size+8*2**20>plan['budget']['storage_bytes']:raise RuntimeError('output reserve reached')
        if shutil.disk_usage(output).free<256*2**20:raise RuntimeError('free disk reserve reached')
        ledger.update(physics_steps=used,scored_episodes=len(results)+len(logs),wall_s=time.monotonic()-started,output_bytes=size)
        write(output/'ledger.json',ledger)
    try:
        for asset in ('current','colleague'):
            config={k:spec[k] for k in ('parent_spec','source_checkpoint','firmware_library','xml')}
            config.update(PD=plan['PD'],response_probe=dict(version=VERSION,asset=asset))
            if asset=='colleague':config['xml']=str(COLLEAGUE)
            rows=[r for r in plan['matrix'] if r['asset']==asset]
            if not rows:continue
            env=None;old_steps=0
            try:
                env=VectorEnv(config,n=4)
                for offset in range(0,len(rows),4):
                    group=rows[offset:offset+4];obs=env.reset(group)
                    initials=[copy.deepcopy(r['probe_initial']) for r in env.current]
                    first=obs['student_actor'].numpy().copy();hist=[[] for _ in group];done=[False]*4
                    for step in range(3102):
                        if step%100==0:budget(4000)
                        with torch.inference_mode():actions=actor(obs['student_actor']).numpy()
                        obs,trans=env.step(actions)
                        new_steps=env.physical_steps;used+=new_steps-old_steps;old_steps=new_steps
                        for i,tr in enumerate(trans):
                            if done[i]:continue
                            hist[i].append(tr.get('metrics'))
                            if not(tr['terminated'] or tr['timeout']):continue
                            row=summarize(hist[i],tr,group[i]);row.update(initial=initials[i],first_actor=first[i].tolist(),
                                complete_controlled_s=sum(m is not None and m['controlled'] for m in hist[i])*.01)
                            arrays=stack_rows(hist[i]);arrays['fine_tail']=np.asarray(tr.get('fine_tail',[]))
                            path=output/(group[i]['id']+'.npz');np.savez_compressed(path,**arrays)
                            row['trace']=dict(path=path.name,sha256=sha(path));results.append(row);done[i]=True
                            write(output/'standing_results.json',results);budget()
                        if all(done):break
                    if not all(done):raise RuntimeError('missing terminal by fixed30s horizon')
                    print(json.dumps(dict(completed=len(results),asset=asset,physical_steps=used),ensure_ascii=False),flush=True)
            finally:
                if env is not None:env.close()
        real={k:features(data['t']-data['t'][data['ix']],data['x'][:,6:10]*1.5,data['x'][:,10:16]*20,
            data['y'],data['tau'],data['x'][:,3:6],window=window) for k,window in [('fit',(.5,2.)),('holdout',(2.,6.))]}
        def mean(x):
            with torch.inference_mode():return actor(torch.from_numpy(x.astype(np.float32))).numpy()
        for case in plan['log_matrix']:
            budget(6000);asset,pname,tau=case['asset'],case['profile'],case['motor_tau_ms']
            if asset=='current':model,refs,base=setup_values
            else:
                model=mujoco.MjModel.from_xml_path(str(COLLEAGUE));refs=binding(model);prepare_colleague(model,refs)
                base={k:getattr(model,k).copy() for k in ('body_mass','body_inertia','dof_armature','dof_damping','dof_frictionloss','geom_friction')}
            mechanical=next(p for p in PROFILES if p['name']==pname)
            sampling=profile();sampling.update(coupling_stiffness_nm_rad=mechanical['K'],
                coupling_damping_nm_s_rad=mechanical['B'],rotor_inertia=mechanical['rotor_J'] or .006)
            physical=source_parameters();physical.update(kp=20.,kd=2.,kv=.15,
                armature=.01 if asset=='colleague' else .012,motor_tau_ms=[tau,tau,0.,tau,tau,0.])
            entry=dict(name=f'log_{asset}_{pname}_tau{tau:g}',parameters=physical,sampling=sampling,
                force_route='separated',preload_fraction=0.,mechanical_envelope_nm=200.)
            row,arrays=run_sampled_case(model,refs,base,mean,data,validation['library'],entry,6.,deadline)
            used+=row['physics_steps']
            if row['termination']=='RUNTIME_FAULT':raise RuntimeError(row.get('runtime_fault'))
            if row['termination']=='WALL_BUDGET':raise RuntimeError('wall budget reached')
            for label,window in [('fit',(.5,2.)),('holdout',(2.,6.))]:
                f=features(arrays['time'],arrays['q'][:,LEGS],arrays['dq'],arrays['mean'],arrays['command'],arrays['gravity'],window=window) if arrays else dict(valid=False)
                row['features_'+label]=f;row['match_'+label]=reproduction(real[label],f)
            path=output/(entry['name']+'.npz');np.savez_compressed(path,**arrays)
            row.update(asset=asset,profile=pname,motor_tau_ms=tau,trace=dict(path=path.name,sha256=sha(path)))
            logs.append(row);write(output/'log_results.json',dict(real=real,results=logs));budget()
            print(json.dumps(dict(completed=len(results)+len(logs),log=entry['name'],termination=row['termination'],physics_steps=used)),flush=True)
        changed=[p for p,h in identity['input_sha256'].items() if sha(p)!=h]
        if changed:raise RuntimeError('input changed during run: '+repr(changed))
        combined=(inherited or [])+results
        write(output/'combined_standing_results.json',combined)
        write(output/'paired_results.json',pair_summary(combined));ledger['status']='COMPLETED';budget()
    except BaseException as exc:
        ledger.update(status='FAILED',failure=type(exc).__name__+': '+str(exc));raise
    finally:
        ledger.update(physics_steps=used,scored_episodes=len(results)+len(logs),wall_s=time.monotonic()-started)
        write(output/'ledger.json',ledger)
        write(output/'terminal.json',dict(**ledger,certificate_eligible=False,hardware_identified=False))
    return ledger


def completion(parent,plan,identity):
    """Continue ONLY unexecuted branches; retain failure and every prior cost."""
    old=json.loads((parent/'ledger.json').read_text());oldplan=json.loads((parent/'spec.json').read_text())
    oldid=json.loads((parent/'identity.json').read_text());results=json.loads((parent/'standing_results.json').read_text())
    if old['status']!='FAILED' or 'colleague bridge only tests its original rigid mechanism' not in str(old['failure']):
        raise ValueError('completion requires the exact pre-integration colleague bootstrap error')
    expected=[r for r in plan['matrix'] if r['asset']=='current']
    if old['scored_episodes']!=16 or [r['case'] for r in results]!=expected:
        # Within a parallel batch completion order may differ.
        if old['scored_episodes']!=16 or sorted(r['case']['id'] for r in results)!=sorted(r['id'] for r in expected):
            raise ValueError('exact original16 current-plant cases required')
    if oldplan['matrix']!=plan['matrix'] or oldplan['log_matrix']!=plan['log_matrix'] or oldplan['PD']!=plan['PD']:
        raise ValueError('completion cannot change diagnostic conditions')
    changed={p for p,h in oldid['input_sha256'].items() if sha(p)!=h}
    allowed={str(ROOT/p) for p in ('scripts/compare_motion31_response.py','src/native_stand/students/motion_compliance_env.py','src/native_stand/students/motion_response_probe.py')}
    if changed!=allowed:raise ValueError('unexpected changed inputs since failure: '+repr(changed))
    for p in changed:
        snapshot=ROOT/'artifacts/2026_10_29__motion4500_tau_pair/failed_execution_sources'/Path(p).name
        if sha(snapshot)!=oldid['input_sha256'][p]:raise ValueError('failed source snapshot differs')
    for r in results:
        path=parent/r['trace']['path']
        if sha(path)!=r['trace']['sha256']:raise ValueError('completed parent trace differs')
        r['trace']['path']=str(Path('..')/parent.name/r['trace']['path'])
    used_bytes=sum(p.stat().st_size for p in parent.rglob('*') if p.is_file())
    revised=copy.deepcopy(plan);revised['matrix']=[r for r in plan['matrix'] if r['asset']=='colleague']
    revised['budget'].update(scored_episodes=14,physics_steps=plan['budget']['physics_steps']-old['physics_steps'],
        wall_s=plan['budget']['wall_s']-old['wall_s'],storage_bytes=plan['budget']['storage_bytes']-used_bytes)
    revised['continuation']=dict(parent=str(parent),parent_ledger_sha256=sha(parent/'ledger.json'),
        parent_spec_sha256=sha(parent/'spec.json'),used_physics_steps=old['physics_steps'],used_wall_s=old['wall_s'],
        used_output_bytes=used_bytes,used_scored_episodes=16,prior_attempts=1,
        reason='zero-integration constructor used F20 training slot; only missing14 cases continue, no repeated physics',
        source_changes={p:dict(before=oldid['input_sha256'][p],after=identity['input_sha256'][p]) for p in changed})
    return revised,results


def check_workers(spec,plan):
    from native_stand.students.motion_compliance_env import VectorEnv
    records=[]
    for asset in ('current','colleague'):
        config={k:spec[k] for k in ('parent_spec','source_checkpoint','firmware_library','xml')}
        config.update(PD=plan['PD'],response_probe=dict(version=VERSION,asset=asset))
        if asset=='colleague':config['xml']=str(COLLEAGUE)
        env=None
        try:
            env=VectorEnv(config,n=4)
            group=[r for r in plan['matrix'] if r['asset']==asset][:4]
            obs=env.reset(group)
            if env.physical_steps!=0 or obs['student_actor'].shape!=(4,31):raise ValueError('zero-step worker contract failed')
            records.append(dict(asset=asset,workers=4,physics_steps=env.physical_steps,actor_shape=list(obs['student_actor'].shape),
                profiles=[r['probe_initial']['mechanical']['name'] for r in env.current]))
        finally:
            if env is not None:env.close()
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--check-workers',action='store_true',help='Construct/reset real C workers, zero integration')
    parser.add_argument('--complete-from',type=Path,help='Only unexecuted branches after the exact zero-step bootstrap failure')
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    spec,source,validation,data,actor,identity,models=prepare();plan=request(spec,identity)
    if args.check_workers:
        print(json.dumps(dict(worker_check=check_workers(spec,plan),simulation_steps=0),indent=2));return
    if not args.run:
        print(json.dumps(dict(plan=plan,CPU_preflight='PASS',PT_ONNX_max_abs=identity['PT_ONNX_max_abs'],simulation=0),ensure_ascii=False,indent=2));return
    if identity['resource']['free_disk_bytes']<512*2**20:raise ValueError('insufficient free disk')
    inherited=None
    if args.complete_from:plan,inherited=completion(args.complete_from.resolve(),plan,identity)
    from native_stand.command_v6.domain.experiment_naming import new_experiment_name,validate_new_experiment_name
    output=(args.output_dir or ROOT/'sim2sim'/new_experiment_name('motion4500_tau_pair')).resolve()
    if output.parent!=ROOT/'sim2sim':raise ValueError('new run must be directly under project sim2sim')
    validate_new_experiment_name(output.name);output.mkdir(exist_ok=False)
    if args.complete_from:
        claim=ROOT/'artifacts/2026_10_29__motion4500_tau_pair/completion_claim.json'
        with claim.open('x') as stream:
            json.dump(dict(parent=str(args.complete_from.resolve()),child=str(output),
                parent_ledger_sha256=plan['continuation']['parent_ledger_sha256']),stream,indent=2)
    ledger=execute(output,spec,source,validation,data,actor,identity,models,plan,inherited)
    print(json.dumps(dict(output=str(output),**ledger),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
