"""Owned, charged physical parity before learning; never a capability certificate."""
from dataclasses import fields,replace
import numpy as np
import torch
from .state import State
from .execution import step,decode,measurements,observation
from .real import collect_episode,restore,sha,integration_state,TEACHER_SHA,SMALL_SHA
from .dataset import write_json


def run(model,refs,cfg,actor_map,geometry,reward,budget,out):
    import mujoco
    from diagnose_motion_real2sim import sensors
    from .owner import hashlib_json
    from .experiment import cases
    from native_stand.students.mcu_interface import encode
    chosen=[cases()[0],cases()[5],cases()[10]]
    maxima={k:0. for k in ('observation','action','pending','adopted','driver','motor','pd_request',
                          'pd_limited','compensation','spring','applied','sensors','queue','restore_q','restore_v')}
    records=[]
    for index,row in enumerate(chosen):
        name=row['controller']
        case=dict(row,id=f'engineering_{index}',seconds=.4,handover_ms=200 if index==1 else 0,
            controller_sha256=TEACHER_SHA if name=='teacher4500' else SMALL_SHA,
            spec_sha256=sha(budget.path),plant_identity=hashlib_json(budget.spec['plant_readback']))
        folder=out/case['id']
        records.append(collect_episode(model,refs,cfg,actor_map,case,geometry,reward,budget,output=folder))
        with np.load(folder/'transitions.npz',allow_pickle=False) as z:
            def state(i,side):
                return State(**{f.name:torch.from_numpy(z[f'{side}_{f.name}'][i:i+1]) for f in fields(State)})
            def maximum(key,a,b):
                maxima[key]=max(maxima[key],float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
            with torch.no_grad():
                for i in range(len(z['action'])):
                    b=state(i,'before');a=state(i,'after');s=b.clone()
                    sensed=b.sensors[:,int(b.actor_age[0])-1]
                    public=encode(sensed[:,:3],sensed[:,3:6],sensed[:,6:12]+b.actor_bias,sensed[:,12:18],
                        b.adopted[:,[0,1,3,4]],b.adopted[:,[2,5]],torch.zeros(1,6),'Motion',b.task[:,:3])
                    actual=observation(b,measurements(b,geometry))
                    maximum('observation',actual,public)
                    owner=case.get('predecessor') if int(b.tick[0])<case.get('handover_ms',0) else name
                    maximum('action',actor_map[owner].mean(actual),actor_map[owner].mean(public))
                    target=decode(torch.from_numpy(z['action'][i:i+1]),geometry,b)
                    for sub in range(10):
                        j=i*10+sub
                        sensor=measurements(s,geometry)['sensors']
                        s,valid=step(s,target,sensor)
                        if not bool(valid.all()):raise ValueError('same-state differentiated execution invalid')
                        tensor=lambda k:torch.tensor(z[k][j:j+1],dtype=torch.float32)
                        s=replace(s,solver_q=tensor('micro_q'),solver_v=tensor('micro_v'),
                            q=tensor('micro_next_q'),v=tensor('micro_next_v'),warm=tensor('micro_next_warm'))
                    for key in ('pending','adopted','driver','motor','pd_request','pd_limited','compensation',
                                'spring','applied','sensors','queue'):
                        maximum(key,getattr(s,key),getattr(a,key))
                    if i%4==0:
                        data,engine,history=restore(model,refs,cfg,b,z['integration_state'][i])
                        for sub in range(10):
                            t=int(b.tick[0])+sub;history.append(t,sensors(model,data,refs))
                            engine.step(t,target[0].numpy(),data.qpos[refs['qpos']],data.qvel[refs['dofs']])
                            data.ctrl[refs['actuators']]=engine.actual.numpy()/refs['gears']
                            budget.charge('physics_steps',1);mujoco.mj_step(model,data)
                            if not np.isfinite(integration_state(model,data)).all():raise FloatingPointError('restored physical state nonfinite')
                        j=i*10+9
                        maximum('restore_q',data.qpos,z['micro_next_q'][j])
                        maximum('restore_v',data.qvel,z['micro_next_v'][j])
    tolerances={k:4e-4 for k in maxima}
    tolerances.update(observation=2e-6,action=2e-6,restore_q=1e-6,restore_v=1e-3)
    checks={key:maxima[key]<=tolerances[key] for key in maxima}
    result=dict(passed=all(checks.values()),max_abs_error=maxima,tolerances=tolerances,checks=checks,
        episodes=records,physics_steps=budget.local['physics_steps'],controller_handover_ms=200,
        status='ENGINEERING_ONLY_NOT_30S_BEHAVIOR_EVIDENCE',reset_labels=False,source='actual MuJoCo integration')
    write_json(out/'parity.json',result)
    if not result['passed']:raise ValueError('physical same-state parity failed: '+str({k:v for k,v in checks.items() if not v}))
    return result
