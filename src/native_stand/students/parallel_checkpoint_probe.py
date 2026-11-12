"""One real environment per frozen Actor, one native Swing episode each."""
import torch
from .model import load
from .grouped_evaluation import split_arrays


class Actors(torch.nn.Module):
    def __init__(self,actors):
        super().__init__();self.actors=torch.nn.ModuleList(actors)
    def act_for_envs(self,obs,ids):
        out=obs['student_actor'].new_zeros((len(ids),6))
        for i,actor in enumerate(self.actors):
            selected=ids==i
            if selected.any():out[selected]=actor({k:v[selected] for k,v in obs.items()},stochastic_output=False)
        return out


class ParallelCheckpoints:
    def __init__(self,cases):self.cases=cases;self.pending={}
    def collecting_cases(self,row):return [] if row['case_id'] in self.pending else self.cases
    def acquire(self,env,actor,row,*,stop=None):
        if row['case_id'] not in self.pending:
            from .evaluator import rollout,score_arrays
            actors=[]
            for case in self.cases:
                model,_=load(case['checkpoint'],case['role'])
                model=model.to(env.device).eval();model.requires_grad_(False);actors.append(model)
            dispatch=Actors(actors)
            before={k:v.clone() for k,v in dispatch.state_dict().items()}
            arrays=rollout(env,dispatch,self.cases[0],stop=stop,scored_envs=len(self.cases),collect_only=True)
            if not all(torch.equal(v,dispatch.state_dict()[k]) for k,v in before.items()):
                raise RuntimeError('parallel evaluation changed a frozen Actor')
            for i,case in enumerate(self.cases):
                part=split_arrays(arrays,i,1)
                payload,_=score_arrays(env,case,part,scored_envs=1)
                payload['execution_layout']={'version':'recovery_parallel_models_v1','global_env_id':i,
                    'model_count':len(self.cases),'scored_envs_per_model':1,'physical_envs':env.num_envs,
                    'all_simulated_env_steps_attributed':len(arrays['first_episode_valid'])*env.num_envs if i==0 else 0,
                    'scope':'single native Swing screening, not multiseed certification',
                    'physical_serial_equivalence':'NOT_VERIFIED','frozen_all_models_verified':True}
                self.pending[case['case_id']]=(payload,part)
        return self.pending.pop(row['case_id'])


def finalize(output,spec,status):
    """Called once by the existing monitor after exact-attempt settlement."""
    import json
    from pathlib import Path
    import numpy as np
    from .data import sha
    from native_stand.command_v6.completion import write,identity
    output=Path(output);terminal=json.loads((output/'terminal_report.json').read_text())
    rows=[]
    for c in terminal['conditions']:
        if c['status']!='COMPLETED':continue
        path=output/c['result'];d=json.loads(path.read_text());trace=output/d['trace']['path']
        if sha(trace)!=d['trace']['sha256']:raise ValueError('terminal trace SHA differs')
        r=d['per_env'][0]
        with np.load(trace) as a:
            ids=np.flatnonzero(a['first_episode_valid'][:,0]&(a['controller'][:,0,0]==3))
            entry=r.get('entry_index');hold=np.arange(entry+1,min(entry+1001,len(a['state']))) if entry is not None else np.array([],dtype=int)
            valid_hold=hold[a['first_episode_valid'][hold,0]]
            q=valid_hold if len(valid_hold) else ids
            s=a['state'][q,0];p=a['posture'][q,0,:2]
            metric={'observed_s':len(q)*.01,'velocity_abs_mean':np.abs(s[:,7:10]).mean(0).tolist(),
                    'pitch_roll_abs_mean':np.abs(p).mean(0).tolist(),'saturation':float(s[:,17].mean())}
        rows.append({'checkpoint':d['case']['checkpoint'],'sha256':d['checkpoint_sha256'],
                     'update':int(Path(d['case']['checkpoint']).stem.split('_')[-1]),'result':r,'metrics':metric})
    event=identity({'attempt':terminal['attempt_id'],'session':spec['authorization']['hapi_session']})[:24]
    lines=['# Recovery单环境并行快速筛选','',f'终态：{status}；完成{len(rows)}/{len(terminal["conditions"])}模型；PPO/optimizer0。',
           '每模型一个原生Swing episode；不作为多条件能力认证。未完整窗口的均值只覆盖已观察部分。','',
           '| update | 入口 | 满10秒窗口 | accepted | 观察秒数 | abs(vx) | abs(yaw) | 力矩触限 | 失败项 |',
           '|---|---|---|---|---|---|---|---|---|']
    for row in rows:
        r=row['result'];m=row['metrics'];v=m['velocity_abs_mean']
        lines.append(f"| {row['update']} | {r.get('entry_index') is not None} | {r.get('hold_complete',False)} | {r['accepted']} | {m['observed_s']:.2f} | {v[0]:.4f} | {v[2]:.4f} | {m['saturation']:.2%} | {r.get('failed_checks')} |")
    write(output/'screening_summary.json',{'status':status,'rows':rows,'scoring_episode_limit':len(terminal['conditions']),'PPO':0})
    (output/'completion_report.md').write_text('\n'.join(lines)+'\n')
    write(output/'analysis.json',{'event_id':event,'status':status,'actual_updates':0,'samples':0,'accepted':all(x['result']['accepted'] for x in rows) and len(rows)==len(terminal['conditions'])})
    # Do not rank models using incomplete windows or reward proxies.
    write(output/'candidate_selection.json',{'best_observed_candidate':None,'accepted_candidate':None,'scope':'single-native screening; inspect per-model metrics'})
    write(output/'report_ready.json',{'event_id':event,'status':'report_ready','report_sha256':sha(output/'completion_report.md')})


def finalize_full(output,spec,status):
    """Full matrix reporting only; scores/gates remain those in original results."""
    import json
    from pathlib import Path
    import numpy as np
    from .data import sha
    from native_stand.command_v6.completion import write,identity
    from native_stand.command_v6.domain.trace_schema import CONTROL
    output=Path(output);terminal=json.loads((output/'terminal_report.json').read_text())
    rows=[];sources={}
    for c in terminal['conditions']:
        if c['status']!='COMPLETED':continue
        file=output/c['result'];d=json.loads(file.read_text());trace=file.parent/d['trace']['path']
        if sha(trace)!=d['trace']['sha256']:raise ValueError('terminal trace SHA differs')
        sources[str(file)]=sha(file);sources[str(trace)]=sha(trace)
        with np.load(trace) as a:
            for e,r in enumerate(d['per_env']):
                windows={};entry=r.get('entry_index')
                for name,lo,hi in [('controlled',None,None),('entry0_2',0,200),('entry2_10',200,1000),('entry0_10',0,1000)]:
                    if lo is None:ids=np.flatnonzero(a['first_episode_valid'][:,e]&(a['controller'][:,e,0]==3))
                    else:
                        if entry is None:windows[name]=None;continue
                        ids=np.arange(entry+1+lo,min(entry+1+hi,len(a['state'])))
                        if len(ids)!=hi-lo or not a['first_episode_valid'][ids,e].all():windows[name]=None;continue
                    if not len(ids):windows[name]=None;continue
                    s=a['state'][ids,e];p=a['posture'][ids,e,:2];control=a['control'][ids,e]
                    q=a['joint_q_pre'][ids,e][:,d['field_schema']['active_indices']]
                    mirror=np.stack((q[:,0]+q[:,3],q[:,1]-q[:,4]),-1)
                    wheel=control[:,CONTROL['clipped_action']][:,[2,5]]*100
                    torque=control[:,CONTROL['physical_clipped_torque']]
                    windows[name]={'observed_s':len(ids)*.01,'height_mean':float(s[:,6].mean()),
                        'velocity_abs_mean':np.abs(s[:,7:10]).mean(0).tolist(),'pitch_roll_abs_mean':np.abs(p).mean(0).tolist(),
                        'pitch_roll_peak':np.abs(p).max(0).tolist(),'mirror_mae_rad':np.abs(mirror).mean(0).tolist(),
                        'mirror_p95_rad':np.quantile(np.abs(mirror),.95,axis=0).tolist(),'saturation':float(s[:,17].mean()),
                        'channel_torque_limit_fraction':(np.abs(torque)>=np.array([40,40,5.5,40,40,5.5])-1e-4).mean(0).tolist(),
                        'torque_rms_nm':np.sqrt((torque**2).mean(0)).tolist(),
                        'wheel_target_delta_rms':np.sqrt(np.mean(np.diff(wheel,axis=0)**2,axis=0)).tolist() if len(ids)>1 else None,
                        'dual_contact_fraction':float((s[:,10:12]>2).all(-1).mean()),'closure_peak_m':float(s[:,12].max()),
                        'root_displacement_xyz_m':(a['root_post'][ids[-1],e,:3]-a['root_pre'][ids[0],e,:3]).tolist()}
                rows.append({'case_id':d['case']['case_id'],'checkpoint':d['case']['checkpoint'],
                             'sha256':d['checkpoint_sha256'],'env':e,'result':r,'windows':windows})
    expected=len(terminal['conditions'])*spec['evaluation']['scored_envs']
    passed=sum(r['result']['accepted'] for r in rows);all_passed=len(rows)==expected and passed==expected
    event=identity({'attempt':terminal['attempt_id'],'session':spec['authorization']['hapi_session']})[:24]
    write(output/'full_assessment_summary.json',{'status':status,'scored':len(rows),'expected':expected,
        'passed':passed,'accepted':all_passed,'rows':rows,'sources_sha256':sources,'PPO':0,'optimizer':0})
    lines=['# Recovery四条件完整评估','',f'终态：{status}；完整通过{passed}/{expected}；PPO/optimizer0。',
        '范围：原生Swing与既有小yaw条件；不是地形或大扰动认证。入口2～10秒仅8秒诊断，不改变完整10秒验收。','',
        '| case | env | 入口/s | 完整通过 | abs vx | abs yaw | 力矩触限 | 失败项 |',
        '|---|---|---|---|---|---|---|---|']
    for row in rows:
        r=row['result'];m=row['windows']['entry0_10'];v=m['velocity_abs_mean'] if m else [None]*3
        lines.append(f"| {row['case_id']} | {row['env']} | {r.get('settle_s')} | {r['accepted']} | {v[0]} | {v[2]} | {m['saturation'] if m else None} | {r.get('failed_checks')} |")
    (output/'completion_report.md').write_text('\n'.join(lines)+'\n')
    candidate={'checkpoint':rows[0]['checkpoint'],'sha256':rows[0]['sha256']} if rows and len({r['checkpoint'] for r in rows})==1 else None
    write(output/'candidate_selection.json',{'best_observed_candidate':candidate,'accepted_candidate':candidate if all_passed else None,'scope':'same declared four-case protocol'})
    write(output/'analysis.json',{'event_id':event,'status':status,'actual_updates':0,'samples':0,'accepted':all_passed})
    write(output/'report_ready.json',{'event_id':event,'status':'report_ready','report_sha256':sha(output/'completion_report.md')})
