"""Replay fixed spring anchors through saved USD joint coordinates; no integration."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd
from audit_recovery_same_state import joint_tree

def batch_fk(tree,angles,n):
    poses={'robot':np.broadcast_to(np.eye(4),(n,4,4)).copy()};pending=list(tree)
    while pending:
        remaining=[]
        for name,parent,child,l0,l1,axis in pending:
            if parent not in poses:remaining.append((name,parent,child,l0,l1,axis));continue
            vec=np.zeros((n,3));vec[:,axis]=angles[name]
            turn=np.broadcast_to(np.eye(4),(n,4,4)).copy();turn[:,:3,:3]=Rotation.from_rotvec(vec).as_matrix()
            poses[child]=poses[parent]@l0@turn@np.linalg.inv(l1)
        if len(remaining)==len(pending):raise ValueError('disconnected joint tree')
        pending=remaining
    return poses

def stats(x):return dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()),p95=float(np.percentile(x,95)))
def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    g=json.loads(a.geometry.read_text());left=json.loads(Path(g['left_source']).read_text());usd=Path(left['source_usd'])
    if hashlib.sha256(usd.read_bytes()).hexdigest()!=left['source_usd_sha256']:raise ValueError('USD identity drift')
    tree=joint_tree(Usd.Stage.Open(str(usd)));rows=[];arrays={};sources={str(a.geometry):hashlib.sha256(a.geometry.read_bytes()).hexdigest(),str(usd):left['source_usd_sha256']}
    for ep in sorted((a.run/'formal').glob('evaluation_*.json')):
        d=json.loads(ep.read_text())
        if not isinstance(d,dict) or Path(d.get('case',{}).get('checkpoint','')).name!='model_update_1500.pt' or not d.get('per_env'):continue
        tp=ep.parent/d['trace']['path'];h=hashlib.sha256(tp.read_bytes()).hexdigest()
        if h!=d['trace']['sha256']:raise ValueError('trace identity drift')
        sources[str(ep)]=hashlib.sha256(ep.read_bytes()).hexdigest();sources[str(tp)]=h
        with np.load(tp) as z:
            names=d.get('joint_names') or json.loads((a.run/'source_eval/evaluation_readback_00.json').read_text())['joint_names']
            for env in d['per_env']:
                eid=env['env'];valid=z['first_episode_valid'][:,eid].astype(bool)&z['actor_valid'][:,eid,0].astype(bool);idx=np.flatnonzero(valid)
                if not len(idx) or np.any(np.diff(idx)!=1):raise ValueError('noncontinuous student episode')
                q=z['joint_q_pre'][idx,eid];poses=batch_fk(tree,{name:q[:,i] for i,name in enumerate(names)},len(idx));holdlo,holdhi=env['hold_index_half_open'];hold=(idx>=holdlo)&(idx<holdhi)
                for spring in g['springs']:
                    side=spring['side'];ends=[]
                    for anchor in spring['anchors']:
                        body=anchor['body'].rsplit('/',1)[-1];t=poses[body];ends.append(np.einsum('nij,j->ni',t[:,:3,:3],anchor['local_m'])+t[:,:3,3])
                    delta=ends[1]-ends[0];length=np.linalg.norm(delta,axis=1);axis=delta/length[:,None];rate=np.diff(length)/.01
                    knee=next(x for x in tree if x[0]==side+'_knee_pitch');_,parent,child,l0,l1,ax=knee;frame=poses[parent]@l0
                    lever=np.cross(ends[0]-frame[:,:3,3],-axis);moment=np.sum(lever*frame[:,:3,ax],axis=1)
                    # Unit outward force lower attachment moment about knee, not closed-chain motor torque.
                    row=dict(case=d['case']['case_id'],env=eid,side=side,samples=len(idx),length_m=stats(length),hold_length_m=stats(length[hold]),length_rate_mps=stats(rate),knee_unit_push_moment_arm_signed_m=stats(moment),fraction_below_160mm=float(np.mean(length<.16)),fraction_above_200mm=float(np.mean(length>.20)),fraction_above_210mm=float(np.mean(length>.21)))
                    rows.append(row);key=f'{ep.stem}_env{eid}_{side}';arrays[key+'_length_m']=length;arrays[key+'_index']=idx;arrays[key+'_knee_signed_arm_m']=moment
    if len(rows)!=128:raise ValueError('expected 64 episodes times two springs')
    all_lengths=np.concatenate([v for k,v in arrays.items() if k.endswith('_length_m')]);result=dict(version='gas_spring_geometry_replay_v1',sources_sha256=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),episodes=64,spring_episode_count=128,simulation_steps=0,optimizer_steps=0,coordinate='root-relative FK; global rotation/translation omitted as length/axis moment invariant',global_length_m=stats(all_lengths),rows=rows,hardware_limits='fixed physical endpoints; compressed approx .16m, extended exact endpoint unknown within user .20-.21m measurement interval',force_model='NOT_APPLIED',limitations='pre-action100Hz sampled; finite-difference length rate not continuous peak; unit knee moment is not active motor generalized torque through closed chains; previous no-spring trajectory may be unreachable with fixed-stroke springs')
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'geometry_samples.npz',**arrays);(a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(length=result['global_length_m'],max_below160_fraction=max(r['fraction_below_160mm'] for r in rows),max_above210_fraction=max(r['fraction_above_210mm'] for r in rows),hold_global_min=min(r['hold_length_m']['min'] for r in rows),hold_global_max=max(r['hold_length_m']['max'] for r in rows),knee_arm_min=min(r['knee_unit_push_moment_arm_signed_m']['min'] for r in rows),knee_arm_max=max(r['knee_unit_push_moment_arm_signed_m']['max'] for r in rows)),indent=2))

if __name__=='__main__':main()
