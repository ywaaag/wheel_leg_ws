"""Stepwise independent World Model/BPTT CLI. Runtime defaults to ZERO-WRITE plan."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import os

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='operation',required=True)
    audit=subs.add_parser('audit');audit.add_argument('--output',type=Path,required=True)
    prepare=subs.add_parser('prepare');prepare.add_argument('--folder',type=Path)
    prepare.add_argument('--contact-force',choices=('none','contact_force_v1'),default='none',
        help='explicit loss-space dispatch recorded in the spec; none keeps the historical supervision')
    continuation=subs.add_parser('prepare-window80');continuation.add_argument('--parent-spec',type=Path,required=True)
    continuation.add_argument('--review',type=Path,required=True)
    more=subs.add_parser('prepare-window80-continue');more.add_argument('--parent-spec',type=Path,required=True)
    more.add_argument('--review',type=Path,required=True)
    contact=subs.add_parser('prepare-contact25');contact.add_argument('--parent-spec',type=Path,required=True)
    contact.add_argument('--review',type=Path,required=True)
    force=subs.add_parser('prepare-contact-force');force.add_argument('--parent-spec',type=Path,required=True)
    force.add_argument('--review',type=Path,required=True)
    force.add_argument('--arm',choices=('control','force'),required=True,
        help='A(control) or B(force); run once per arm, sequential, never in parallel')
    force.add_argument('--pilot-artifacts-bytes',type=int,
        help='explicit reviewer-granted per-arm artifact allowance; absent means the lineage remainder')
    force.add_argument('--initialization-world',type=Path,
        help='explicit initialization checkpoint for BOTH arms (settlement parent stays recorded separately)')
    force.add_argument('--owner-handoff',type=Path,
        help='frozen owner handoff record; the only alternative to a matching HAPI owner session')
    force.add_argument('--reviewed-code-changes',type=Path,
        help='JSON map {path: {old_sha256,new_sha256,review}} for shared code changed by other workers')
    force.add_argument('--reviewed-source-deletions',type=Path,
        help='JSON map {path: {sha256_recorded,reason,evidence}} for frozen inputs removed by an external prune')
    for command in (prepare,continuation,more,contact,force):
        command.add_argument('--supervision-performance',choices=('legacy_full_fk_v1','closure_no_checkpoint_v1'),default='closure_no_checkpoint_v1',
            help='explicit equivalent supervised FK/checkpoint implementation; does not alter physics/evaluation')
    auth=subs.add_parser('authorize');auth.add_argument('--spec',type=Path,required=True);auth.add_argument('--scope',required=True)
    auth.add_argument('--unlimited',action='store_true',help='explicit human authorization: unlimited total sampling/updates/wall; storage guards retained')
    auth.add_argument('--notify-owner',action='store_true',help='explicit human authorization: terminal event to current HAPI owner')
    auth.add_argument('--owner-handoff',type=Path,help='frozen owner handoff record when the executor is not the parent HAPI owner')
    settle=subs.add_parser('settle-attempt')
    settle.add_argument('--spec',type=Path,required=True)
    settle.add_argument('--attempt-id',required=True)
    settle.add_argument('--status',choices=('FAILED','UNKNOWN'),default='FAILED')
    settle.add_argument('--error',required=True)
    handoff=subs.add_parser('record-handoff')
    handoff.add_argument('--parent-spec',type=Path,required=True)
    handoff.add_argument('--delegation',required=True,help='verbatim user delegation text')
    handoff.add_argument('--output',type=Path,required=True)
    handoff.add_argument('--analysis',type=Path,help='parent terminal analysis.json')
    handoff.add_argument('--arms',default='control,force')
    handoff.add_argument('--member-dispatches-per-arm',type=int,default=300)
    handoff.add_argument('--artifacts-bytes-total',type=int,required=True)
    selection=subs.add_parser('select');selection.add_argument('--evaluation',type=Path,required=True)
    selection.add_argument('--baseline-evaluation',type=Path,help='original same-spec baseline for candidate-only refresh evaluation')
    selection.add_argument('--actor',type=Path,required=True);selection.add_argument('--output',type=Path,required=True)
    for name in ('cycle','engineering','reuse-data','collect','build-data','train-world','train-critic','predict','gradient','train-actor','evaluate','export'):
        p=subs.add_parser(name);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        p.add_argument('--run',action='store_true',help='submit this one owned bounded operation via public supervisor')
        p.add_argument('--detach',action='store_true',help='one bounded startup handshake, then owner continues')
        p.add_argument('--emit-request',type=Path,help='CPU-only explicit request file, no reservation/launch')
        if name in ('train-world','train-critic','predict','gradient','train-actor'):p.add_argument('--dataset',type=Path,required=True)
        if name in ('predict','gradient','train-actor'):p.add_argument('--world',type=Path,required=True)
        elif name=='train-world':p.add_argument('--world',type=Path,help='same-algorithm model for supervised refresh')
        if name in ('gradient','train-actor'):p.add_argument('--critic',type=Path,required=True)
        elif name=='train-critic':p.add_argument('--critic',type=Path)
        if name in ('train-world','train-critic','train-actor'):p.add_argument('--steps',type=int,default={'train-world':3000,'train-critic':1000,'train-actor':20}[name])
        if name in ('collect','train-critic','gradient','train-actor','evaluate','export'):p.add_argument('--actor',type=Path)
        if name=='collect':p.add_argument('--collection',choices=('initial','refresh'),default='initial');p.add_argument('--selection',type=Path)
        if name=='build-data':p.add_argument('--episodes',type=Path,nargs='+',required=True)
        if name in ('predict','gradient'):p.add_argument('--split',choices=('validation','test'),default='validation');p.add_argument('--final-test',action='store_true')
        if name=='train-actor':
            p.add_argument('--prediction',type=Path,required=True);p.add_argument('--gradient',type=Path,required=True)
        if name=='evaluate':
            p.add_argument('--final-test',action='store_true');p.add_argument('--controllers',nargs='+',default=['teacher4500','small4500','candidate'])
            p.add_argument('--selection',type=Path)
        if name=='export':p.add_argument('--selection',type=Path,required=True)
    args=parser.parse_args()
    if args.operation=='audit':
        from native_stand.world_model.audit import run
        r=run(args.output);print(json.dumps({k:r[k] for k in ('status','physics_steps','world_parameters','critic_parameters','critic_input_dim','kinematics')},indent=2));return 0
    if args.operation=='settle-attempt':
        from native_stand.world_model.experiment import settle_interrupted_attempt
        print(json.dumps(settle_interrupted_attempt(args.spec,args.attempt_id,error=args.error,
            status=args.status),indent=2));return 0
    if args.operation=='record-handoff':
        from native_stand.world_model.experiment import record_owner_handoff
        arms=[a.strip() for a in args.arms.split(',') if a.strip()]
        record=record_owner_handoff(args.parent_spec,args.delegation,output=args.output,analysis=args.analysis,
            scope=dict(route='shared_world_model_bptt',arms=arms,
                actions=['prepare','authorize','execute','predict'],
                member_dispatches_per_arm=args.member_dispatches_per_arm,
                member_dispatches_total=args.member_dispatches_per_arm*len(arms),
                artifacts_bytes_total=args.artifacts_bytes_total,
                physics_steps=0,new_sampling_episodes=0,actor_optimizer_steps=0,critic_optimizer_steps=0,
                ppo_updates=0,test_split=False,horizons=[8,16,32],prediction_starts=51,
                stop='gate failure, non-finite loss/gradient, identity or storage fault; no blind retry'))
        print(json.dumps(record,indent=2,ensure_ascii=False));return 0
    if args.operation=='prepare':
        from native_stand.world_model.experiment import prepare
        print(prepare(args.folder,supervision_performance_version=args.supervision_performance,
            contact_force=None if args.contact_force=='none' else dict(version=args.contact_force)));return 0
    if args.operation in ('prepare-window80','prepare-window80-continue','prepare-contact25','prepare-contact-force'):
        from native_stand.world_model.experiment import prepare_prediction_continuation
        print(prepare_prediction_continuation(args.parent_spec,args.review,
            continue_window80=args.operation=='prepare-window80-continue',contact_focus=args.operation=='prepare-contact25',
            contact_force_arm=getattr(args,'arm',None),
            pilot_artifacts_bytes=getattr(args,'pilot_artifacts_bytes',None),
            initialization_world=getattr(args,'initialization_world',None),
            owner_handoff=getattr(args,'owner_handoff',None),
            reviewed_code_changes=(json.loads(args.reviewed_code_changes.read_text())
                                   if getattr(args,'reviewed_code_changes',None) else None),
            reviewed_source_deletions=(json.loads(args.reviewed_source_deletions.read_text())
                                   if getattr(args,'reviewed_source_deletions',None) else None),
            supervision_performance_version=args.supervision_performance));return 0
    if args.operation=='authorize':
        from native_stand.world_model.experiment import authorize_prepared
        authorize_prepared(args.spec,args.scope,unlimited=args.unlimited,notify_owner=args.notify_owner,
            owner_handoff=getattr(args,'owner_handoff',None));return 0
    if args.operation=='select':
        from native_stand.world_model.validation import choose_physical
        from native_stand.world_model.real import sha
        from native_stand.world_model.dataset import write_json
        d=json.loads(args.evaluation.read_text())
        if d['split']!='validation':raise ValueError('final test cannot select/tune a model')
        if d.get('inputs',{}).get('actor')!=sha(args.actor):raise ValueError('physical evaluation candidate SHA differs')
        base=json.loads(args.baseline_evaluation.read_text()) if args.baseline_evaluation else d
        if base['split']!='validation' or base['spec_sha256']!=d['spec_sha256']:raise ValueError('baseline validation spec differs')
        r=choose_physical(base['models']['teacher4500'],d['models']['candidate'])
        r.update(candidate_sha256=sha(args.actor),evaluation_sha256=sha(args.evaluation))
        write_json(args.output,r);print(json.dumps(r,indent=2));return 0
    from native_stand.world_model.experiment import verify,validate_request
    from native_stand.world_model.dataset import write_json
    spec,book=verify(args.spec,authorization=args.run)
    request={k:(str(v.resolve()) if isinstance(v,Path) else [str(x.resolve()) if isinstance(x,Path) else x for x in v] if isinstance(v,list) else v)
             for k,v in vars(args).items() if k not in ('run','detach','emit_request') and v is not None}
    if 'steps' in request and request['steps']<=0:raise ValueError('positive finite optimizer step count required')
    if args.run:validate_request(request,spec,book)
    plan=dict(request=request,authorization=spec['authorization']['status'],new_algorithm=True,
        prior_ppo_quota_reused=False,files_written=0,simulation_steps=0,optimizer_steps=0,
        public_entry='native_stand.command_v6.supervisor --world-model-request',limits=spec['limits'])
    if not args.run:
        if args.emit_request:write_json(args.emit_request,request);plan['files_written']=1
        print(json.dumps(plan,indent=2));return 0
    args.output.parent.mkdir(parents=True,exist_ok=True)
    request_file=args.output.with_suffix('.request.json')
    write_json(request_file,request)
    env={**os.environ,'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT/'scripts')}
    command=[sys.executable,'-m','native_stand.command_v6.supervisor','--world-model-request',str(request_file)]
    if not args.detach:command.append('--wait')
    return subprocess.run(command,env=env,cwd=ROOT).returncode


if __name__=='__main__':raise SystemExit(main())
