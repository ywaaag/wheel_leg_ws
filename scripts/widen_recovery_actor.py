"""CPU-only Recovery4100 widening. Default is zero-write; no simulation/PPO."""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch

from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.students.data import sha
from native_stand.students.model import Student,load
from native_stand.students.recovery_large import VERSION,SOURCE_SHA,network,expand,validate_candidate

ROOT=Path(__file__).resolve().parents[1]


def migrate(checkpoint,parent_spec,evaluations,write=False):
    cp,parent=Path(checkpoint).resolve(),Path(parent_spec).resolve()
    evaluations=[Path(p).resolve() for p in evaluations]
    if not evaluations or len(set(evaluations))!=len(evaluations):
        raise ValueError('unique reference evaluations required')
    if any(not p.is_relative_to(ROOT) for p in [cp,parent,*evaluations]):
        raise ValueError('inputs must be inside this project')
    if sha(cp)!=SOURCE_SHA:raise ValueError('exact Recovery4100 SHA required')
    actor,data=load(cp,'RecoveryStudent');spec=json.loads(parent.read_text())
    if (data.get('critic_privileged_version')!='recovery_privileged64_v1' or
        data.get('network_architecture') is not None or data['spec_sha256']!=sha(parent) or
        data['asset_sha256']!=spec['asset_sha256'] or sha(spec['asset_path'])!=spec['asset_sha256']):
        raise ValueError('Recovery source spec/asset/interface binding differs')
    if spec['action']['leg_scale_rad']!=1.5 or spec['action']['wheel_scale_rad_s']!=100.:
        raise ValueError('Recovery source physical action scales differ')
    inputs={str(p):sha(p) for p in [cp,parent,Path(spec['asset_path'])]}
    for path in [Path(__file__).resolve(),ROOT/'src/native_stand/students/recovery_large.py',
                 ROOT/'src/native_stand/students/actor_widening.py',ROOT/'src/native_stand/students/model.py']:
        inputs[str(path)]=sha(path)
    with torch.random.fork_rng(devices=[]):
        critic=Student(False,version='multiskill_mcu_v2',critic_privileged_version='recovery_privileged64_v1')
        large=Student(True,version='multiskill_mcu_v2',network_architecture=network())
        large_critic=Student(False,version='multiskill_mcu_v2',critic_privileged_version='recovery_privileged64_v1',network_architecture=network())
    critic.load_state_dict(data['critic'],strict=True)
    a,c=expand(data['actor'],data['critic']);large.load_state_dict(a);large_critic.load_state_dict(c)
    rng=torch.get_rng_state().clone();rows=[];maximum=torch.zeros(6);value_max=0.
    for path in evaluations:
        raw=json.loads(path.read_text());trace=(path.parent/raw['trace']['path']).resolve()
        if not trace.is_relative_to(path.parent):raise ValueError('trace outside evaluation directory')
        if (raw['checkpoint_sha256']!=SOURCE_SHA or raw['asset_sha256']!=data['asset_sha256'] or
            raw['case']['role']!='RecoveryStudent' or not all(e['accepted'] for e in raw['per_env']) or
            raw['field_schema']['state']!='recovery_policy_trace_v1' or sha(trace)!=raw['trace']['sha256']):
            raise ValueError('reference evaluation/trace identity or certificate differs')
        inputs.update({str(path):sha(path),str(trace):sha(trace)})
        with np.load(trace,allow_pickle=False) as z:
            if raw['trace']['axes']!='policy_sample,scored_env,feature':raise ValueError('trace axes differ')
            mask=z['first_episode_valid'].astype(bool)
            controlled=mask & z['actor_valid'][...,0].astype(bool)
            if not np.all(z['controller'][...,0][controlled]==raw['field_schema']['controller']['student']):
                raise ValueError('reference contains teacher control')
            times=z['policy_time_s']
            if not np.isfinite(times).all() or not np.allclose(np.diff(times),.01,atol=1e-5,rtol=0):
                raise ValueError('reference time axis differs')
            x=torch.from_numpy(z['actor_input'][mask].copy()).float()
            v=torch.from_numpy(z['critic_input'][mask].copy()).float()
            actual_mean=torch.from_numpy(z['mean'][controlled].copy()).float()
            controlled_x=torch.from_numpy(z['actor_input'][controlled].copy()).float()
        if (x.ndim!=2 or x.shape[1]!=31 or v.shape!=(len(x),64) or not len(x) or
            not torch.isfinite(x).all() or not torch.isfinite(v).all()):raise ValueError('named finite31/64D evidence required')
        if not torch.all(x[:,25:28]==x.new_tensor([0.,1.,0.])):raise ValueError('Recovery skill identity differs')
        error=torch.zeros(6);v_error=0.;trace_error=0.
        with torch.inference_mode():
            for start in range(0,len(x),512):
                end=start+512
                error=torch.maximum(error,(actor(x[start:end])-large(x[start:end])).abs().amax(0))
                v_error=max(v_error,float((critic(v[start:end])-large_critic(v[start:end])).abs().max()))
            for start in range(0,len(controlled_x),512):
                trace_error=max(trace_error,float((actor(controlled_x[start:start+512])-actual_mean[start:start+512]).abs().max()))
        if error.max()>1e-5 or v_error>1e-4 or trace_error>1e-5:raise ValueError('CPU equivalence or source trace mean tolerance exceeded')
        maximum=torch.maximum(maximum,error);value_max=max(value_max,v_error)
        rows.append({'evaluation':str(path),'case_id':raw['case']['case_id'],'samples':len(x),
                     'controlled_samples':int(controlled.sum()),'episodes':len(raw['per_env']),
                     'actor_max_error':float(error.max()),'critic_max_error':v_error,'source_vs_trace_mean_max_error':trace_error})
    if not torch.equal(rng,torch.get_rng_state()):raise ValueError('widening validation changed CPU RNG')
    if not torch.equal(actor.log_std,large.log_std):raise ValueError('std changed')
    candidate={'initialization_kind':VERSION,'interface_version':'multiskill_mcu_v2','role':'RecoveryStudent',
        'network_architecture':network(),'critic_privileged_version':'recovery_privileged64_v1',
        'actor':a,'critic':c,'source_sha256':SOURCE_SHA,'source_checkpoint':str(cp),
        'source_spec_sha256':sha(parent),'source_spec':str(parent),'source_update':data['update'],
        'source_training_recipe':data['training_recipe'],
        **{k:data[k] for k in ('asset_sha256','dataset_sha256','manifest_sha256')},
        'accepted':False,'online_updates_this_stage':0,'offline_optimizer_steps':0,
        'optimizer_transferred':False,'RNG_restored':False,
        'critic_scope':'expanded source-reward value function; fresh Critic required for a new reward recipe'}
    from native_stand.students.recovery_privileged import contract
    candidate['actor_interface']={**contract(),'actor_hidden_dims':[512,256,128],'critic_hidden_dims':[512,512,256]}
    candidate['physical_contract']={'action':spec['action'],'control':spec['control'],
                                    'ground':spec['ground'],'asset_path':spec['asset_path']}
    validate_candidate(candidate)
    name=new_experiment_name('recovery31_large_from4100');output=ROOT/'isolated'/name
    stream=io.BytesIO();torch.save(candidate,stream);content=stream.getvalue()
    import hashlib
    proof={'actor_normalized_max_error':float(maximum.max()),'actor_per_channel_max_error':maximum.tolist(),
           'physical_per_channel_max_error':(maximum*maximum.new_tensor([1.5,1.5,100.,1.5,1.5,100.])).tolist(),
           'critic_max_error':value_max,'std_exact':True,'CPU_RNG_unchanged':True,'evaluations':rows,
           'scope':'saved-input numerical equivalence only; closed-loop capability NOT_RUN'}
    report={'version':VERSION,'checkpoint':str(output/'initialization_candidate.pt'),
            'checkpoint_sha256':hashlib.sha256(content).hexdigest(),'network':network(),
            'actor_parameters_excluding_std':sum(p.numel() for n,p in large.named_parameters() if n!='log_std'),
            'critic_parameters':sum(p.numel() for p in large_critic.parameters()),
            'inputs':inputs,'proof':proof,'optimizer_steps':0,'online_PPO':0,'simulation':0,
            'training_ready':False,'remaining':'new explicit Recovery large recipe/spec and independent physical validation'}
    if write:
        output.mkdir(parents=True,exist_ok=False)
        (output/'initialization_candidate.pt').write_bytes(content)
        reloaded,_=load(output/'initialization_candidate.pt','RecoveryStudent')
        if any(not torch.equal(v,reloaded.state_dict()[k]) for k,v in a.items()):raise ValueError('public loader roundtrip differs')
        if any(sha(p)!=h for p,h in inputs.items()):raise ValueError('source changed during migration')
        (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        (output/'publication_complete.json').write_text(json.dumps({'status':'COMPLETE','files':{
            'initialization_candidate.pt':report['checkpoint_sha256'],'manifest.json':sha(output/'manifest.json')}},indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--parent-spec',required=True)
    p.add_argument('--reference-evaluation',action='append',required=True)
    p.add_argument('--write',action='store_true');args=p.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(migrate(args.checkpoint,args.parent_spec,args.reference_evaluation,args.write),indent=2))
