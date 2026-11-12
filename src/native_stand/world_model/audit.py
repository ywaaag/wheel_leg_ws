"""CPU identity/kinematics/state/gradient inspection, zero physics integration."""
from dataclasses import replace
from pathlib import Path
import copy
import json
import numpy as np
import torch
from .dataset import write_json


def run(output):
    import mujoco
    from .real import load_plant,description,context,actors,export_state,integration_state,restore,sha,ROOT
    from .geometry import Kinematics
    from .state import initial
    from .models import Ensemble,Critic,parameter_count,value_features
    from .reward import MotionReward
    from .owner import retention_inputs
    from native_stand.students.motion_execution_uncertainty import nominal
    from native_stand.students.real2sim_delay import ObservationHistory
    from native_stand.students.motion_tracking_reward import effective_reward_fields
    from motion_mcu_playback import apply_initial_pose
    from diagnose_motion_closed_loop_robustness import UncertainExecution
    from diagnose_motion_real2sim import sensors
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    model,refs,cfg,rb=load_plant();data=mujoco.MjData(model)
    q0=apply_initial_pose(model,data,refs,cfg['mcu_execution']);mujoco.mj_forward(model,data)
    geometry=Kinematics(description(model,refs));ctx,layout=context(model)
    q=torch.tensor(data.qpos,dtype=torch.float32)[None];v=torch.tensor(data.qvel,dtype=torch.float32)[None]
    measured=geometry(q,v)
    kin=dict(body_position_max_m=float(np.max(np.abs(measured['p'][0].numpy()-data.xpos))),
        body_rotation_max=float(np.max(np.abs(measured['r'][0].numpy()-data.xmat.reshape(-1,3,3)))),
        actor_sensor_max=float(np.max(np.abs(measured['sensors'][0].numpy()-sensors(model,data,refs)))),
        closure_peak_m=float(measured['closure'].norm(dim=-1).max()))
    # Nonzero-velocity COM/axis semantics checked at fixed q, no integrated time.
    data.qvel[:]=np.sin(np.arange(model.nv))*.03;mujoco.mj_forward(model,data)
    motion=geometry(q,torch.tensor(data.qvel,dtype=torch.float32)[None])
    velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,refs['root'],velocity,0)
    kin['root_com_velocity_max_mps']=float(np.max(np.abs(motion['com_velocity'][0,refs['root']].numpy()-velocity[3:])))
    kin['nonzero_actor_sensor_max']=float(np.max(np.abs(motion['sensors'][0].numpy()-sensors(model,data,refs))))
    data.qvel[:]=0;mujoco.mj_forward(model,data)
    target=q0.astype(np.float32);target[[2,5]]=0
    engine=UncertainExecution(cfg['mcu_execution'],q0,target,nominal())
    history=ObservationHistory(sensors(model,data,refs))
    state=initial(q,v,measured['sensors'],ctx,torch.tensor([[0.,0.,.34,0.]]),
        compensation=engine.compensation[None],cap=engine.driver_cap[None],delay=engine.ring.delay,tau=engine.ring.tau)
    state=export_state(model,data,refs,engine,history,state,0)
    raw=integration_state(model,data)
    rd,re,rh=restore(model,refs,cfg,state,np.stack((raw,raw)))
    restored=export_state(model,rd,refs,re,rh,state,0)
    differences={k:float((getattr(state,k).to(torch.float64)-getattr(restored,k).to(torch.float64)).abs().max()) for k in state.as_dict()}
    blob_equal=bool(np.array_equal(raw,integration_state(model,rd)))
    actor_map,source=actors();teacher=actor_map['teacher4500'];retained=retention_inputs()
    clone=copy.deepcopy(teacher)
    with torch.no_grad():source_error=float((clone.mean(retained)-teacher.mean(retained)).abs().max())
    parent=json.loads((ROOT/'configs/experiments/2026_10_25-23_52__motion1500_posture_retention312_10000/spec.json').read_text())
    reward=MotionReward(parent,source)
    q_grad=q.clone().requires_grad_(True)
    pre=replace(state,q=q_grad)
    post=replace(pre,v=pre.v+torch.linspace(-.01,.01,model.nv)[None])
    m=geometry(post.q,post.v);m['root_com_velocity']=m['com_velocity'][:,geometry.root]
    _,scores,terminal=reward(pre,post,m)
    reward_grad=torch.autograd.grad(scores['total'].sum(),q_grad)[0]
    if tuple(scores['fields'])!=tuple(effective_reward_fields(parent)):raise ValueError('source reward field/schema differs')
    torch.testing.assert_close(scores['dt'],scores['density']*.01)
    wm=Ensemble(model.nq,model.nv,ctx.shape[-1])
    critic=Critic(value_features(state).shape[-1])
    report=dict(status='CPU_VERIFIED_NOT_PHYSICALLY_TRAINED',physics_steps=0,
        real_world_optimizer_steps=0,real_actor_optimizer_steps=0,real_critic_optimizer_steps=0,ppo_updates=0,
        effective_plant=rb,kinematics=kin,full_markov_restore=dict(integration_blob_exact=blob_equal,max_field_errors=differences),
        source_actor_function_preserved_max=source_error,retention_samples=len(retained),
        actor_parameters=parameter_count(teacher),world_parameters=parameter_count(wm),
        world_member_parameters=parameter_count(wm.members[0]),critic_parameters=parameter_count(critic),
        critic_input_dim=value_features(state).shape[-1],context_dim=ctx.shape[-1],context_layout=layout,
        reward=dict(version='motion4500_model_gradient_reward_v1',fields=list(scores['fields']),
            reward=float(scores['total'][0].detach()),dt_once=True,finite_q_gradient=bool(torch.isfinite(reward_grad).all()),
            q_gradient_norm=float(reward_grad.norm()),hard_terminal_derivative='zero/undefined; no smooth success surrogate',
            source_source_implementation='shared density functions; pure symmetry adapter checked in portable tests'),
        scope='CPU kinematics/causal execution and fixture BPTT; learned prediction/physical gradient/closed-loop benefits NOT_RUN')
    registry=ROOT/'runs/2026_10_25-23_52__motion1500_posture_retention312_10000/formal/candidate_registry.json'
    candidate=next(r for r in json.loads(registry.read_text()) if r['update']==4500)
    if (candidate['sha256']!='0be977b308cfe493fdddee54823e21c713508fa582ce1618ab34e57b13fbc619' or
        candidate.get('complete_matrix') is not True or len(candidate['per_env'])!=192 or
        not all(r['accepted'] is True for r in candidate['per_env'])):
        raise ValueError('Teacher4500 complete native evaluation identity differs')
    raw_records=[]
    for item in candidate['evaluations']:
        p=registry.parent/item['path']
        if sha(p)!=item['sha256']:raise ValueError('native source evaluation JSON changed')
        raw=json.loads(p.read_text());trace=p.parent/raw['trace']['path']
        if sha(trace)!=raw['trace']['sha256']:raise ValueError('native source evaluation trace changed')
        raw_records.append(dict(json=str(p),json_sha256=sha(p),trace=str(trace),trace_sha256=sha(trace)))
    report['Teacher4500_native_evidence']=dict(conditions=len(raw_records),scored_envs=192,accepted=192,
        native_protocol_only=True,records=raw_records)
    manifest_path=ROOT/'isolated/recovery31_contact_quality_u1500_frozen_20261018/manifest.json'
    recovery=json.loads(manifest_path.read_text());rpath=ROOT/recovery['source_spec'];rspec=json.loads(rpath.read_text())
    rcp=ROOT/recovery['source_checkpoint']
    if sha(rcp)!=recovery['checkpoint_sha256'] or sha(rpath)!=recovery['source_spec_sha256']:
        raise ValueError('Recovery frozen checkpoint/spec identity differs')
    if sha(rspec['asset_path'])!=rspec['asset_sha256']:raise ValueError('Recovery frozen asset identity differs')
    report['Recovery_identity']=dict(checkpoint=str(rcp),checkpoint_sha256=sha(rcp),spec=str(rpath),spec_sha256=sha(rpath),
        asset=rspec['asset_path'],asset_sha256=sha(rspec['asset_path']),control=rspec['control'],
        status='VERIFIED_IDENTITY_NOT_IMPORTED_INTO_WORLD_MODEL',
        difference='Recovery60/2/1 and wheel5.5Nm versus MCU40/2/.2 with final2.4469479Nm, delays and independent spring')
    report['state_fields']={k:dict(shape=list(v.shape[1:]),dtype=str(v.dtype)) for k,v in state.as_dict().items()}
    if not blob_equal or max(differences.values())>1e-6 or kin['actor_sensor_max']>1e-5:
        raise ValueError('CPU Markov/kinematic readback differs')
    write_json(out/'cpu_audit.json',report)
    return report
