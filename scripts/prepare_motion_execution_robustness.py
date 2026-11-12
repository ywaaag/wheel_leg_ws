"""Prepare the separate4500 handover/PD-age experiment, public supervisor only."""
import argparse
import json
import os
from pathlib import Path

from native_stand.command_v6.domain.experiment_naming import new_experiment_name
from native_stand.command_v6.tools.experiment_preparation import publish
from native_stand.students.motion_handover_robustness import build, verify, ONLINE, FORMAL
from native_stand.students.data import sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-spec',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--diagnostic-result',type=Path,required=True)
    p.add_argument('--write',action='store_true')
    p.add_argument('--authorize',action='store_true')
    a=p.parse_args()
    if a.authorize and not a.write:p.error('--authorize requires --write')
    session=os.environ['HAPI_SESSION_ID']
    name=new_experiment_name('motion4500_handover_PDage1000')
    folder=ROOT/'configs/experiments'/name
    spec=build(a.parent_spec,a.checkpoint,a.diagnostic_result,experiment=name,spec_path=folder/'spec.json',session=session)
    verify(spec)
    from native_stand.command_v6.tools.storage_budget import check
    capacity=check(spec,ROOT/'runs'/name/'formal','formal')
    result=dict(spec=str(folder/'spec.json'),run=str(ROOT/'runs'/name),source_sha256=spec['source_sha256'],
        smoke_updates=5,formal_updates=FORMAL,inclusive_updates=ONLINE,max_samples=ONLINE*4096*64,
        envs=4096,steps=64,wall_seconds=spec['budget']['chain_wall_s'],storage_bytes=spec['artifacts']['storage_max_bytes'],
        conditions=len(spec['student_evaluation']['cases']),scored_envs_per_condition=8,
        storage_preflight=capacity,simulation=0,ppo=0,files_written=0)
    if a.write:
        if capacity['status']!='PASS':raise ValueError(capacity)
        claim=ROOT/'configs/experiments/.motion_handover_robustness_claims'/spec['source_sha256']
        if claim.exists():raise ValueError('4500 robustness authorization already prepared; no duplicate')
        phases=['source_eval','smoke','formal','evaluate','play']
        book=dict(authorization='UNAUTHORIZED',experiment=name,hapi_session=session,
            limits=dict(zip(phases,[1,5,FORMAL,0,0])),used={k:0 for k in phases},attempts={},
            source_eval_wall_used_s=0.,total_online_PPO_limit=ONLINE,
            prior_training_consumption='independent active run retains its own budget; no transfer')
        from native_stand.students.motion_reward_audit import audit
        from native_stand.students.motion_antioscillation import reward_snapshot
        from native_stand.students.motion_posture_preparation import runtime_reward_snapshot
        snapshot=audit(spec['reward']);snapshot['runtime_reward_extension']=reward_snapshot(spec['motion_timing']['configuration'])
        snapshot['posture_reward']=runtime_reward_snapshot()
        publish(folder,{'spec.json':spec,'budget.json':book,'preparation.json':result,'reward_snapshot.json':snapshot},
                spec['student_dependencies'],completion_marker=True)
        claim.parent.mkdir(parents=True,exist_ok=True)
        with claim.open('x') as f:json.dump({'spec':str(folder/'spec.json'),'experiment':name,'online_limit':ONLINE},f)
        if a.authorize:
            from native_stand.command_v6.tools.register_authorization import register
            result['registration']=register(dict(approval_id='user_motion4500_handover_20261026_'+name,
                user_scope=spec['authorization']['user_scope'],session=session,physical_gpu=0,gpu_uuid=spec['device']['uuid'],
                max_online_updates=ONLINE,max_phase_wall_s=18000,
                arms=[dict(spec=str(folder/'spec.json'),spec_sha256=sha(folder/'spec.json'),budget_sha256=sha(folder/'budget.json'))]),
                current_session=session,write=True)
        result['files_written']=5
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
