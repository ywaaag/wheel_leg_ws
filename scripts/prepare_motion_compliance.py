"""Verify the deployed small source and print a zero-write training proposal.

This is CPU preparation only. There is intentionally no run/authorize switch
until the MuJoCo PPO adapter and public submission route are implemented.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from onnx.reference import ReferenceEvaluator
from native_stand.students.model import load
from native_stand.students.motion_compliance_training import declaration

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'deployment_protocol/2026_10_26-20_29__motion4500_precise_small31_onnx'
SOURCE_SHA='efad012efcced3a9ac58dc51839dea98b81598de289dea8ef0986a0813d139a1'
ONNX_SHA='776201d861dbf1f00b6824f2c2615068a3206eb45b38529e65e7591bc150e617'


def plan(package=PACKAGE):
    package=Path(package);manifest=package/'manifest.json'
    m=json.loads(manifest.read_text());cp=Path(m['source_checkpoint']);onnx=package/'actor.onnx'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(cp)!=SOURCE_SHA or sha(onnx)!=ONNX_SHA or m['source_sha256']!=SOURCE_SHA:
        raise ValueError('exact deployed4500 source identity differs')
    golden=package/'golden_vectors.npz'
    if sha(golden)!=m['files_sha256']['golden_vectors.npz']:raise ValueError('source golden identity differs')
    parent=Path(m['parent_spec'])
    if sha(parent)!=m['parent_spec_sha256']:raise ValueError('source reward/plant parent differs')
    actor,data=load(cp,'MotionZeroStudent')
    with np.load(golden) as g:
        x=g['input'] if 'input' in g else g['observation'] if 'observation' in g else g['observations']
    x=np.asarray(x,np.float32).reshape(-1,31)
    evaluator=ReferenceEvaluator(str(onnx));name=evaluator.input_names[0]
    with torch.inference_mode():want=actor(torch.from_numpy(x)).numpy()
    got=np.concatenate([evaluator.run(None,{name:r[None]})[0] for r in x])
    error=float(np.abs(got-want).max())
    if error>1e-5:raise ValueError('source ONNX/PyTorch output differs')
    s=declaration();s.update(authorization='PENDING_NEW_BUDGET',files_written=0,
        source_checkpoint=str(cp),source_sha256=SOURCE_SHA,source_ONNX=str(onnx),source_ONNX_sha256=ONNX_SHA,
        parent_spec=str(parent),parent_spec_sha256=sha(parent),
        source_validation={'golden_count':len(x),'maximum_normalized_action_error':error,
            'maximum_leg_target_error_rad':float(np.abs(got[:,[0,1,3,4]]-want[:,[0,1,3,4]]).max()*1.5),
            'maximum_wheel_target_error_rad_s':float(np.abs(got[:,[2,5]]-want[:,[2,5]]).max()*100)},
        source_reward=json.loads(parent.read_text())['reward'],
        online_PPO=0,physics_steps=0,optimizer_steps=0)
    return s


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,default=PACKAGE)
    p.add_argument('--write-plan',type=Path,help='Explicitly write this CPU proposal to a new file; no spec/ledger')
    p.add_argument('--write',action='store_true',help='Publish the explicit3000-update recipe after CPU checks')
    p.add_argument('--authorize',action='store_true')
    p.add_argument('--firmware-library',type=Path)
    p.add_argument('--tag',default='motion4500_small_compliance3000')
    args=p.parse_args();result=plan(args.package)
    if args.write:
        import datetime,os
        from native_stand.students.motion_compliance_requests import build,write
        if not args.authorize or args.firmware_library is None:raise ValueError('--write requires new authorization and compiled firmware library')
        experiment=datetime.datetime.now().strftime('%Y_%m_%d-%H_%M__')+args.tag
        destination=ROOT/'configs/experiments'/experiment
        destination.mkdir(exist_ok=False)
        spec=build(result,experiment=experiment,library=args.firmware_library,session=os.environ.get('HAPI_SESSION_ID'))
        spec['run']=str(ROOT/'runs'/experiment);spec['authorization']['status']='AUTHORIZED'
        write(destination/'spec.json',spec)
        write(destination/'budget.json',dict(authorization='AUTHORIZED',spec_sha256=hashlib.sha256((destination/'spec.json').read_bytes()).hexdigest(),
            limits=dict(source_eval=1,smoke=12,formal=2988),used=dict(source_eval=0,smoke=0,formal=0),attempts={},
            historical_online_updates_separate=3901,unused_old6099_not_transferred=True))
        print(json.dumps(dict(status='PREPARED',spec=str(destination/'spec.json'),run=spec['run'],budget=spec['budget']),indent=2));return
    if args.authorize:raise ValueError('authorization requires --write')
    if args.write_plan:
        result['files_written']=1
        args.write_plan.parent.mkdir(parents=True,exist_ok=True)
        with args.write_plan.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
