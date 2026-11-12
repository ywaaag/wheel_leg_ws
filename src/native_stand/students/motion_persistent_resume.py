"""Identity-bound normal course transition, preserving the current10000 budget."""
import copy
import json
from pathlib import Path
import torch
from .data import sha

VERSION='motion_fixed_height_persistent_resume_v1'


def evidence(parent,checkpoint):
    from . import motion_fixed_height as fixed
    parent=Path(parent).resolve();checkpoint=Path(checkpoint).resolve()
    old=json.loads(parent.read_text());fixed.verify(old,require_current_claim=False)
    run=parent.parents[3]/'runs'/old['experiment'];formal=run/'formal'
    book=json.loads(parent.with_name('budget.json').read_text())
    terminal=json.loads((run/'chain_terminal_report.json').read_text())
    end=json.loads((formal/'terminal_report.json').read_text())
    source=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (terminal['status']!='STOPPED_BOUNDED' or terminal.get('failure')
        or end.get('stop_kind')!='user_signal' or any(a['status']=='RUNNING' for a in book['attempts'].values())
        or source['update']!=end['actual_updates'] or book['used']['formal']!=source['update']
        or checkpoint!=formal/f"model_update_{source['update']}.pt" or source['spec_sha256']!=sha(parent)
        or source['network_architecture']!=fixed.network() or source['critic_privileged_version']!=fixed.CRITIC
        or any(k not in source for k in ('actor','critic','optimizer','training_rng','reward_refinement_state','symmetry_mix_state'))):
        raise ValueError('persistent transition needs settled exact latest full184 parent')
    owner=json.loads((formal/'course_progress.json').read_text());st=source['motion_course_state']
    for k in ('stage','levels','last_evaluation_update','assessment_keys'):
        if st[k]!=owner['state'][k]:raise ValueError('checkpoint and completed owner course differ: '+k)
    if owner['decision'].get('evidence_valid') is not True:raise ValueError('latest completed boundary has invalid evidence')
    registry=json.loads((formal/'candidate_registry.json').read_text())
    matrix=next(c for c in registry if c['update']==st['last_evaluation_update'] and c.get('complete_matrix') is True)
    if matrix['update']<1500 or sha(matrix['checkpoint'])!=matrix['sha256']:raise ValueError('1500 full matrix not settled')
    paths=[parent,parent.with_name('budget.json'),run/'chain_terminal_report.json',formal/'terminal_report.json',
           formal/'manifest.json',formal/'course_progress.json',formal/'candidate_registry.json',checkpoint,Path(matrix['checkpoint'])]
    for rec in matrix['evaluations']:
        p=(formal/rec['path']).resolve();raw=json.loads(p.read_text());trace=(formal/raw['trace']['path']).resolve()
        if (not p.is_relative_to(formal) or not trace.is_relative_to(formal) or sha(p)!=rec['sha256']
            or sha(trace)!=raw['trace']['sha256'] or raw['checkpoint_sha256']!=matrix['sha256']
            or raw['spec_sha256']!=sha(parent) or raw['asset_sha256']!=old['asset_sha256']
            or raw['protocol']!=old['motion_assessment'] or raw['case']!=rec['case']
            or raw.get('model_unchanged') is not True or len(raw['per_env'])!=8
            or {e['env'] for e in raw['per_env']}!=set(range(8))
            or any(type(e.get('accepted')) is not bool or e.get('controlled_seconds')!=30. for e in raw['per_env'])):
            raise ValueError('persistent transition completed matrix identity differs')
        paths += [p,trace]
    fallback=next(c for c in registry if c['update']==1250 and c.get('complete_matrix'))
    paths.append(Path(fallback['checkpoint']))
    if sha(fallback['checkpoint'])!=fallback['sha256']:raise ValueError('1250 fallback changed')
    used=book['used']['smoke']+book['used']['formal'];limit=book['total_online_PPO_limit']
    remaining=limit-used
    if limit!=10000 or remaining<=0:raise ValueError('no authorized updates remain')
    prior=old['motion_decoupled']['previous_consumption']
    size=sum(p.stat().st_size for p in run.rglob('*') if p.is_file() and not p.is_symlink())
    total_storage=old['artifacts']['storage_max_bytes']
    return old,source,dict(version=VERSION,parent_spec=str(parent),parent_run=str(run),
        inputs={str(p):sha(p) for p in paths},checkpoint_update=source['update'],matrix_update=matrix['update'],
        formal_update_offset=old['student_training'].get('formal_resume_offset',0)+source['update'],
        original_online_limit=limit,current_block_used_updates=used,remaining_formal_updates=remaining,
        ancestral_online_updates=prior['historical_online_updates'],
        historical_online_updates=prior['historical_online_updates']+used,
        historical_formal_updates=prior['historical_formal_updates']+book['used']['formal'],
        historical_smoke_updates=prior['historical_smoke_updates']+book['used']['smoke'],
        historical_wall_s=prior['historical_wall_s']+terminal['chain_wall_s'],
        historical_artifact_bytes=prior['historical_artifact_bytes']+size,block_artifact_bytes=size,
        storage_max_bytes=total_storage-size,parent_wiring_proof=old['motion_training']['wiring_reuse'],
        source_evidence_run=old['motion_training']['wiring_reuse']['source_evidence_run'],
        fallback_checkpoint=fallback['checkpoint'],fallback_sha256=fallback['sha256'],
        capability_accepted=False,physical_state_restored=False,warmup_repeated=False,
        scope='new scheduler; current10000 remaining only; old matrix certifies old tasks, never new frontiers')


def build(parent,checkpoint,*,experiment,spec_path,owner_session,**unused):
    from . import motion_persistent as course
    old,source,proof=evidence(parent,checkpoint);s=copy.deepcopy(old)
    state=course.initial_state(source['motion_course_state']);remaining=proof['remaining_formal_updates']
    s.update(experiment=experiment,source_checkpoint=str(Path(checkpoint).resolve()),source_sha256=sha(checkpoint))
    cfg=s['motion_decoupled'];cfg.pop('assessment_keys_resume',None)
    cfg.update(version=course.VERSION,parent_spec=str(Path(parent).resolve()),parent_spec_sha256=sha(parent),
        state=state,previous_consumption=proof,engineering_recovery=proof,persistent_resume=proof,
        formal_engineering_probe=True,configuration=course.configuration())
    cfg['configuration'].update(formal_updates_max=remaining,smoke_updates_max=0,storage_max_bytes=proof['storage_max_bytes'])
    s['motion_course'].update(version=course.VERSION,source_recipe_updates=source['recipe_updates'],configuration=copy.deepcopy(cfg['configuration']))
    s['motion_training'].update(spec_path=str(Path(spec_path).resolve()),stage=state['stage'],resume=None,
                              wiring_reuse=proof,prior_consumption=proof)
    milestones=list(range(250,remaining+1,250))
    if not milestones or milestones[-1]!=remaining:milestones.append(remaining)
    s['student_training'].update(course_version=course.VERSION,source_actor_sha256=sha(checkpoint),
        formal_resume_offset=proof['formal_update_offset'],milestones=milestones,
        height_frontier_reward=course.configuration()['height_frontier_reward'],
        critic='source_same_semantic184_full_state',optimizer='source_full_Adam',storage_max_bytes=proof['storage_max_bytes'])
    s['student_evaluation']['cases'][0].update(checkpoint=s['source_checkpoint'],sha256=s['source_sha256'])
    s['motion_timing'].update(parent_spec=str(Path(parent).resolve()),parent_spec_sha256=sha(parent),source_checkpoint_sha256=sha(checkpoint))
    samples=remaining*s['ppo']['envs']*s['ppo']['steps_per_env']
    s['budget'].update(source_eval_updates=0,smoke_updates=0,formal_updates=remaining,formal_samples=samples,
                      sample_limit=samples,formal_wall_s=None,chain_wall_s=None)
    s['artifacts'].update(run_name=experiment,storage_max_bytes=proof['storage_max_bytes'])
    s['authorization']=dict(status='UNAUTHORIZED',hapi_session=owner_session,
        user_scope='User2026-10-24 approves normal stop after1500 matrix and persistent multi-axis frontier training; inherit current10000 remaining only, no budget refresh; same31/184 reward PPO physicalGPU0 shared, existing owner notification and TensorBoard')
    s['student_dependencies'].update(proof['inputs'])
    return course.effective_spec(s,state['stage'],state)


def verify(spec,*,require_current_claim=True):
    proof=spec['motion_decoupled']['persistent_resume']
    expected=build(proof['parent_spec'],spec['source_checkpoint'],experiment=spec['experiment'],
        spec_path=spec['motion_training']['spec_path'],owner_session=spec['authorization']['hapi_session'])
    for k in ('authorization','resources','notifications'):expected[k]=spec[k]
    if spec!=expected:raise ValueError('persistent frontier declaration drift')
    if require_current_claim:
        claim=Path(spec['motion_training']['spec_path']).parents[1]/'.motion_persistent_claims'/(sha(proof['parent_spec'])+'.json')
        if claim.exists() and json.loads(claim.read_text())['destination']!=spec['experiment']:
            raise ValueError('persistent parent already consumed')
    return True


def verify_wiring(spec):
    verify(spec)
    return dict(status='VERIFIED',scope='same31/184 full-state plant wiring; new curriculum requires real readback',
                capability_accepted=False,remaining_formal_updates=spec['budget']['formal_updates'])
