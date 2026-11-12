"""CPU-only Motion31 -> Recovery Actor initialization; never starts PPO."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.students.data import sha
from native_stand.students.mcu_interface import contract
from native_stand.students.model import load
from native_stand.students.motion_transfer import transfer_actor

ROOT=Path(__file__).resolve().parents[1]


def migrate(checkpoint,parent_spec,expected_sha,evaluation,write=False):
    cp,parent,evaluation=(Path(p).resolve() for p in (checkpoint,parent_spec,evaluation))
    raw=json.loads(evaluation.read_text());trace=(evaluation.parent/raw['trace']['path']).resolve()
    if not trace.is_relative_to(evaluation.parent):raise ValueError('trace outside evaluation directory')
    for path in (cp,parent,evaluation,trace):
        if not path.is_relative_to(ROOT):raise ValueError('inputs must be in this project')
    if sha(cp)!=expected_sha or raw['checkpoint_sha256']!=expected_sha or sha(trace)!=raw['trace']['sha256']:
        raise ValueError('checkpoint/evaluation/trace SHA differs')
    source,data=load(cp,'MotionZeroStudent');spec=json.loads(parent.read_text())
    if data['interface_version']!='multiskill_mcu_v2' or data['spec_sha256']!=sha(parent) or data['asset_sha256']!=spec['asset_sha256']:
        raise ValueError('source interface/spec/asset binding differs')
    if sha(spec['asset_path'])!=spec['asset_sha256'] or spec['action']['leg_scale_rad']!=1.5 or spec['action']['wheel_scale_rad_s']!=100.:
        raise ValueError('physical asset/action contract differs')
    if raw['field_schema']['controller']['student']!=3:raise ValueError('unknown controller schema')
    with np.load(trace,allow_pickle=False) as a:
        valid=a['first_episode_valid'].astype(bool)&a['actor_valid'][...,0].astype(bool)
        if not np.all(a['controller'][...,0][valid]==3):raise ValueError('non-student executed samples')
        x=torch.from_numpy(a['actor_input'][valid].copy()).float()
    if x.ndim!=2 or x.shape[1]!=31 or not len(x) or not torch.isfinite(x).all():raise ValueError('finite actual31D inputs required')
    if not torch.all(x[:,25:28]==x.new_tensor([1,0,0])):raise ValueError('actual inputs not Motion')
    student=transfer_actor(source,'Motion','Recovery');recovery=x.clone();recovery[:,25]=0;recovery[:,26]=1
    with torch.inference_mode():
        original=source(x);mapped=student(recovery);naive=source(recovery)
    difference=(original-mapped).abs();scale=difference.new_tensor([1.5,1.5,100.,1.5,1.5,100.])
    proof={'controlled_trace_samples':len(x),'action_max_abs_error':float(difference.max()),
           'per_channel_physical_max_error':(difference*scale).amax(0).tolist(),
           'naive_role_switch_action_max_error':float((original-naive).abs().max()),
           'std_unchanged':torch.equal(source.log_std,student.log_std),
           'scope':'CPU function equivalence on saved inputs; no new closed-loop capability'}
    if proof['action_max_abs_error']>1e-6 or not proof['std_unchanged']:raise ValueError('function preservation failed')
    for key,value in source.state_dict().items():
        if key!='mlp.0.weight' and not torch.equal(value,student.state_dict()[key]):raise ValueError('unexpected parameter change')
    for col in range(31):
        if col!=26 and not torch.equal(source.mlp[0].weight[:,col],student.mlp[0].weight[:,col]):raise ValueError('non-Recovery column changed')
    inputs={str(p):sha(p) for p in (cp,parent,evaluation,trace,Path(spec['asset_path']),Path(__file__),ROOT/'src/native_stand/students/motion_transfer.py')}
    name=new_experiment_name('recovery31_from_motion_u1600');output=ROOT/'isolated'/name
    migration={'version':'motion_to_recovery_actor_column_transfer_v1','initialization_only':True,
        'source_checkpoint':str(cp),'source_sha256':expected_sha,'source_spec':str(parent),'source_spec_sha256':sha(parent),
        'source_skill':'Motion','destination_skill':'Recovery','mapping':{'source_first_layer_column':25,'destination_first_layer_column':26},
        'target_height_m':.34,'critic_transferred':False,'optimizer_transferred':False,'RNG_restored':False,
        'source_capability':'Motion teacher; native Swing independent recovery failed0/8; not accepted Recovery'}
    saved={'interface_version':data['interface_version'],'role':'RecoveryStudent','actor':student.state_dict(),
        'asset_sha256':data['asset_sha256'],'source_sha256':expected_sha,'source_spec_sha256':sha(parent),
        'actor_interface':contract(),'migration':migration,'dataset_sha256':data['dataset_sha256'],
        'manifest_sha256':data['manifest_sha256'],'offline_bc_steps':data.get('offline_bc_steps',0),
        'accepted':False,'online_updates':0,'distillation_steps':0}
    buffer=io.BytesIO();torch.save(saved,buffer);content=buffer.getvalue();digest=hashlib.sha256(content).hexdigest()
    result={'experiment':name,'checkpoint':str(output/'recovery_actor_initial.pt'),'checkpoint_sha256':digest,
        'inputs':inputs,'proof':proof,'migration':migration,'actor_layers':[31,128,64,32,6],
        'changed_parameters':['mlp.0.weight[:,26] <- source.mlp.0.weight[:,25]'],
        'physical_action':{'leg_scale_rad':1.5,'wheel_scale_rad_s':100.,'target_height_m':.34},
        'training_ready':False,'remaining':'New Recovery reward/recipe/spec/ledger; fresh Critic/PPO Adam; public preflight and source evaluation',
        'simulation':0,'online_PPO':0,'offline_optimizer_updates':0,'files_written':0}
    if write:
        output.mkdir(parents=True,exist_ok=False)
        (output/'recovery_actor_initial.pt').write_bytes(content)
        reloaded,_=load(output/'recovery_actor_initial.pt','RecoveryStudent')
        if any(not torch.equal(v,reloaded.state_dict()[k]) for k,v in student.state_dict().items()):raise ValueError('roundtrip differs')
        if any(sha(p)!=h for p,h in inputs.items()):raise ValueError('inputs changed')
        result['files_written']=3
        (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        (output/'publication_complete.json').write_text(json.dumps({'status':'COMPLETE','files':{
            'recovery_actor_initial.pt':digest,'manifest.json':sha(output/'manifest.json')}},indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--parent-spec',required=True)
    p.add_argument('--expected-sha256',required=True);p.add_argument('--reference-evaluation',required=True)
    p.add_argument('--write',action='store_true');a=p.parse_args()
    print(json.dumps(migrate(a.checkpoint,a.parent_spec,a.expected_sha256,a.reference_evaluation,a.write),indent=2))
