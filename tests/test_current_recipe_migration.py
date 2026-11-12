"""Current-recipe CPU contracts; no physics integration or extra behavior rollout."""
from pathlib import Path
from types import SimpleNamespace as NS
import copy

import numpy as np
import pytest
import torch

from native_stand.artifacts.checkpoint import CheckpointError, load_source, save_full
from native_stand.artifacts.handoff import VERSION as HANDOFF, import_checkpoint, validate_resume_identity
from native_stand.artifacts.io import file_sha256, write_json
from native_stand.experiments.config import load_job_config
from native_stand.learning.ppo.command_bank import CommandBankAnchor
from native_stand.learning.ppo.networks import StudentNetwork
from native_stand.tasks.motion import maneuver_course, maneuver_course_v2
from native_stand.tasks.motion.course_strategy import resolve

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('name,module',[('motion48_turn_response_DR_full_resume','native_stand.simulation.isaac.motion_backend'),('recovery_preentry48_remaining','native_stand.simulation.recovery_isaac.backend')])
def test_current_config_real_dispatch(name,module):
    from native_stand.simulation.dispatch import execution_service
    job=load_job_config(ROOT/'configs/jobs'/f'{name}.json')
    service=execution_service(job)
    assert service['module']==module
    __import__(module,fromlist=['validate_job']).validate_job(job)
    assert job.control.effort_limits_nm==(40.,40.,4.8,40.,40.,4.8)


def test_course_migrates_keys_and_mixes_not_positions():
    old=maneuver_course.initial_state();old['stage']=19
    layout=maneuver_course.stage_config(19,2048,state=old)
    old['assessment_keys']=layout['cohort_keys']
    mixes=maneuver_course.mixtures(layout)
    source=dict(motion_course_state=old,online_updates_including_smoke=2097,
                reward_refinement_state=mixes,symmetry_mix_state=maneuver_course.mixtures(layout,symmetry=True))
    new=resolve(maneuver_course_v2.VERSION).prepare(source,2048)
    assert new['stage']==20
    assert len(source['reward_refinement_state'])==len(new['assessment_keys'])
    before=copy.deepcopy(source)
    resolve(maneuver_course_v2.VERSION).prepare(source,2048)
    assert source==before
    with pytest.raises(ValueError,match='未知Motion'):resolve('unregistered')


def test_command_bank_gradient_rotation_and_no_rng_cache_pollution():
    job=load_job_config(ROOT/'configs/jobs/motion48_turn_response_DR_full_resume.json')
    config=job.assets['source_policy_retention']
    actor=StudentNetwork(actor=True,input_dim=31,hidden_dims=(8,),output_dim=6,init_std=torch.ones(6)*.1)
    teacher=copy.deepcopy(actor)
    x=torch.zeros(384,31)
    x[:,22:25]=torch.tensor([[a/4,b/4,.68] for a,b in config['commands']]).repeat_interleave(128,0)
    data=dict(actor_obs=x.numpy(),source_mean=teacher.mean(x).detach().numpy(),cohort=np.arange(3).repeat(128))
    anchor=CommandBankAnchor(teacher,config,data)
    with torch.no_grad():actor.mlp[-1].bias.add_(.02)
    from native_stand.learning.ppo.source_anchor import KEY
    obs={'student_actor':torch.zeros(4,31),KEY:torch.full((4,1),-1,dtype=torch.long)}
    rng=torch.get_rng_state().clone()
    loss=anchor.loss(actor,obs,torch.ones(4,dtype=torch.bool));loss.backward()
    assert loss>0 and actor.mlp[-1].bias.grad.abs().sum()>0
    assert anchor.bank_uses==[64]*3
    assert torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None for p in anchor.teacher.parameters())
    restored=CommandBankAnchor(teacher,config,data);restored.load_state_dict(anchor.state_dict())
    a=anchor.loss(actor,obs,torch.ones(4,dtype=torch.bool))
    b=restored.loss(actor,obs,torch.ones(4,dtype=torch.bool))
    assert torch.equal(a,b) and anchor.state_dict()==restored.state_dict()


def test_motor_cap_actual_buffer_and_partial_rows():
    from native_stand.simulation.isaac.motion48_execution import apply_limits,readback
    from native_stand.tasks.motion.execution_limits import MotorLimits
    timing=NS(base=NS(limits=torch.full((3,6),5.5)),dr=NS(caps=torch.full((3,6),5.5)))
    config=MotorLimits().as_dict()
    apply_limits(timing,config,torch.tensor([1]))
    assert timing.dr.caps[1,2]==pytest.approx(4.8)
    assert timing.dr.caps[0,2]==5.5
    assert timing.base.limits[:,2].tolist()==pytest.approx([4.8]*3)
    apply_limits(timing,config)
    assert readback(timing,config)['actual_PD_and_driver_caps_verified']


def test_full_pool_anchor_all_rows_have_sampling_probability():
    from native_stand.learning.ppo.physical_anchor import PhysicalAnchor
    x=torch.zeros(1000,31);x[:,24]=.68;x[500:,10]=.2
    anchor=PhysicalAnchor(x,torch.zeros(1000,6),.1,512)
    assert len(anchor.weights)==1000 and (anchor.weights>0).all()
    rng=torch.get_rng_state();a=anchor()[0];ids=anchor.last_indices.clone()
    torch.set_rng_state(rng);anchor()
    assert torch.equal(ids,anchor.last_indices) and a.shape==(512,31)


def test_recovery_swing_first_and_preentry_density():
    from native_stand.tasks.recovery.swing_priority_course import RecoveryCourse
    from native_stand.simulation.recovery_isaac.preentry_lifecycle import runaway_density
    course=RecoveryCourse(1)
    assert course.choose()['group']=='full_Swing'
    vx=torch.tensor([2.,2.,2.]);cost=runaway_density(vx,torch.tensor([False,True,False]),torch.tensor([True,True,False]))
    assert cost.tolist()==pytest.approx([-.05,0,0])


def fixture_handoff(tmp_path):
    old=tmp_path/'old';cp=old/'checkpoints/x.pt'
    save_full(cp,dict(actor={},critic={},optimizer={},std=torch.ones(6),rng=torch.get_rng_state(),
        counters={'updates':7},run_identity=str(old),config_contract_sha256='c'*64,
        warmup_updates_done=50,pending_training_candidate={'window':7}))
    dependency=tmp_path/'bank.npz';dependency.write_bytes(b'fixture')
    manifest=dict(version=HANDOFF,run_id='logical',owner_terminated=True,
        checkpoint={'path':str(cp),'sha256':file_sha256(cp)},original_run_identity=str(old),
        config_contract_sha256='c'*64,budget={'updates_used':7,'remaining':93,'limits':{'formal':100}},
        dependencies=[{'path':str(dependency),'sha256':file_sha256(dependency)}],parent_settlement={'updates':2097})
    file=tmp_path/'handoff.json';write_json(file,manifest)
    return file,manifest,{str(cp):str(cp),str(dependency):str(dependency)}


def test_cross_directory_full_import_preserves_budget_window_warmup(tmp_path):
    file,manifest,mapping=fixture_handoff(tmp_path)
    target=tmp_path/'new/checkpoints/import.pt'
    imported=import_checkpoint(file,destination=target,mapping=mapping,expected_run_id='logical')
    write_json(target.parent.parent/'run.json',{'logical_run_id':'logical'})
    validate_resume_identity(imported,target.parent.parent)
    assert imported.payload['warmup_updates_done']==50
    assert imported.payload['counters']['updates']==7
    assert imported.payload['pending_training_candidate']=={'window':7}
    assert imported.payload['handoff']['budget']['remaining']==93
    assert file_sha256(manifest['checkpoint']['path'])==manifest['checkpoint']['sha256']
    assert load_source(target).sha256==imported.sha256

@pytest.mark.parametrize('change',[{'owner_terminated':False},{'run_id':'other'},{'budget':{'updates_used':8}},{'config_contract_sha256':'b'*64}])
def test_handoff_refuses_unsafe_or_conflicting_import(tmp_path,change):
    file,manifest,mapping=fixture_handoff(tmp_path);manifest.update(change);write_json(file,manifest)
    with pytest.raises(CheckpointError):import_checkpoint(file,destination=tmp_path/'new/checkpoints/import.pt',mapping=mapping,expected_run_id='logical')


def test_current_motion_real_source_adapter_save_publish():
    import subprocess
    import sys
    import json
    result=subprocess.run([sys.executable,str(ROOT/'tests/fixtures/motion48_training_probe.py')],capture_output=True,text=True,cwd=ROOT)
    assert result.returncode==0,result.stderr
    receipt=json.loads(result.stdout)
    assert receipt['first_publication'] and receipt['KL']=='CommandBankAnchor'
    assert receipt['cohorts']==75
    assert receipt['PPO_updates']==receipt['physics_steps']==0


def test_current_recovery_actual_actor_factory_auxiliary_losses():
    from native_stand.artifacts.checkpoint import load_actor_source
    from native_stand.learning.ppo.recovery_cubic import build_policy
    from tensordict import TensorDict
    job=load_job_config(ROOT/'configs/jobs/recovery_preentry48_remaining.json')
    loaded=load_actor_source(job.initialization.checkpoint)
    obs=TensorDict({'student_actor':torch.zeros(2,31),'student_critic':torch.zeros(2,909),'recovery_actor_valid':torch.ones(2,1,dtype=torch.bool)},[2])
    handle=build_policy(job,obs,source=loaded)
    ppo=handle['ppo']
    assert ppo.actor_update_enabled
    assert len(ppo.optimizer.actor.state)==len(ppo.optimizer.critic.state)==0
    before=ppo.actor.physical_mean(obs['student_actor']).detach()
    assert torch.isfinite(before).all()
    batch=NS(observations=obs)
    loss=ppo.actor_auxiliary_loss(batch,torch.ones(2,dtype=torch.bool))
    assert torch.isfinite(loss) and loss.requires_grad
    assert ppo.action_prior_readback['rows']==2
    assert 'UNKNOWN' in ppo.action_prior_readback['source']


def test_recovery_training_window_cursor_and_separate_candidates(tmp_path):
    from native_stand.simulation.recovery_adapter import RecoveryAdapter
    from native_stand.artifacts.model_refs import ModelIndex
    from native_stand.artifacts.io import json_digest
    base=dict(group='full_Swing',nominal_anchor=True,command_height_m=.34,height_dynamic=False,
        physical_quality=True,success=False,student_seconds=1.,teacher_seconds=0.,DR_strength=0.,steps=100)
    rows=[dict(base) for _ in range(16)]
    rows += [dict(base,group='standing_review',success=True,student_seconds=23.) for _ in range(16)]
    rows += [dict(base,success=True,student_seconds=0.,teacher_seconds=23.)]
    env=NS(completed=rows,completed_count=len(rows))
    context=NS(updates_used=100,run_dir=tmp_path/'run',refs=ModelIndex(tmp_path/'models.json'),store=NS(update=lambda **kw:None))
    context.run_dir.mkdir()
    a=RecoveryAdapter(None);a.env=NS(env=env);a.cursor=0;a.window={};a.pending=None
    a.selection=dict(every_updates=100,minimum_ended=16);a.domain='c'*64
    a.candidates=dict(version='recovery_own_hold_training_candidates_v1',comparison_sha256=a.domain,
        candidates_by_objective={'Swing':{},'standing':{}},best_candidate=None,best_standing_candidate=None)
    a.after_update(context,100,{})
    assert a.pending['rows'][json_digest(['Swing','full_Swing',True,.34,False,0.])]['ended']==17
    cp=context.run_dir/'checkpoints/model_update_100.pt'
    save_full(cp,dict(actor={},critic={},optimizer={},std=torch.ones(6),rng=torch.get_rng_state(),counters={'updates':100}))
    ref=dict(checkpoint=str(cp),sha256=file_sha256(cp),update=100)
    a.after_save(context,ref)
    assert a.candidates['best_candidate'] is None
    assert a.candidates['best_standing_candidate']['checkpoint']==str(cp)
    generation=context.refs.candidate_index('run')
    a.after_save(context,ref)
    assert context.refs.candidate_index('run')==generation
    context.updates_used=101;a.after_update(context,101,{})
    assert a.window=={}


def test_handoff_bad_checkpoint_sha(tmp_path):
    file,manifest,mapping=fixture_handoff(tmp_path)
    manifest['checkpoint']['sha256']='a'*64;write_json(file,manifest)
    with pytest.raises(CheckpointError,match='SHA'):
        import_checkpoint(file,destination=tmp_path/'new/checkpoints/import.pt',mapping=mapping,expected_run_id='logical')


def test_actual_877_critic_and_terminal_encoder_host():
    import subprocess
    import sys
    import json
    result=subprocess.run([sys.executable,str(ROOT/'tests/fixtures/motion48_critic_probe.py')],capture_output=True,text=True,cwd=ROOT)
    assert result.returncode==0,result.stderr
    receipt=json.loads(result.stdout)
    assert receipt['Critic_width']==receipt['terminal_width']==877
    assert receipt['wheel_caps'][0]==pytest.approx([4.8,4.8])
    assert receipt['physics_steps']==0


def test_public_cli_import_checks_target_budget_and_registers_resume(tmp_path,capsys):
    from native_stand.cli import main
    from native_stand.artifacts.model_refs import ModelIndex
    file,manifest,mapping=fixture_handoff(tmp_path)
    run=tmp_path/'new';write_json(run/'run.json',dict(logical_run_id='logical',model_index=str(tmp_path/'models.json')))
    write_json(run/'budget.json',dict(used={'formal':7},limits={'formal':100}))
    mapfile=tmp_path/'mapping.json';write_json(mapfile,mapping)
    target=run/'checkpoints/import.pt'
    assert main(['import-checkpoint','--manifest',str(file),'--mapping',str(mapfile),'--destination',str(target),'--run-id','logical','--json'])==0
    assert str(target) in ModelIndex(tmp_path/'models.json').consumer_paths()
    validate_resume_identity(load_source(target),run)


def test_recovery_actual48_execution_arrivals_partial_reset_and_restore():
    from native_stand.simulation.recovery_isaac.execution import Execution
    from native_stand.simulation.v4_response import Response
    names=[f'p{i}' for i in range(13)]
    row=dict(parameters=dict(zip(names,[1.,.1,.2,4.,6.,1.,-1.,2.,3.,.4,0.,0.,0.])),
        effective_observer=dict(transport_ms=8,q_age_ms=2,v_age_ms=4,q_reference=[0.,0.],velocity_tau_ms=3.))
    contract=dict(parameter_names=names,sides={'L':row,'R':row})
    caps=[40.,40.,4.8,40.,40.,4.8]
    ex=Execution(contract,Response,2,effort_caps=caps)
    q=torch.zeros(2,6);ex.reset(torch.arange(2),q,q)
    target=q.clone();target[:,[2,5]]=100;ex.submit(target)
    motors=[]
    for _ in range(20):
        ex.step(q,q);motors.append(ex.motor.clone())
    assert motors[11][0,2]==0 and motors[12][0,2]==pytest.approx(4.8)
    assert motors[15][0,5]==0 and motors[16][0,5]==pytest.approx(4.8)
    unaffected=ex.wheel_queue[:,1].clone();ex.reset(torch.tensor([0]),q[:1],q[:1])
    assert torch.equal(unaffected,ex.wheel_queue[:,1])
    assert ex.caps[:,2].tolist()==pytest.approx([4.8,4.8])
    state=ex.state_dict();other=Execution(contract,Response,2,effort_caps=caps);other.load_state_dict(state)
    assert other.tick==ex.tick and torch.equal(other.adopted,ex.adopted)
    historical=Execution(contract,Response,2)
    with pytest.raises(ValueError,match='限矩'):historical.load_state_dict(state)


def test_unknown_current_motion_course_rejected_by_real_cli_before_engine(tmp_path):
    import json
    from native_stand.cli import main
    data=json.loads((ROOT/'configs/jobs/motion48_turn_response_DR_full_resume.json').read_text())
    data['curriculum']['version']='unknown_version'
    config=tmp_path/'bad-course.json';write_json(config,data)
    assert main(['train','--config',str(config),'--dry-run'])==2


def test_jump_shared_owner_has_cpu_preflight_only():
    from native_stand.simulation.jump_backend import preflight
    job=load_job_config(ROOT/'configs/jobs/jump_unified_raw37_wheel48.json')
    result=preflight(job)
    assert result['Isaac_GPU'] is False and result['physical_steps']==result['PPO_updates']==0


def test_current_recipe_cpu_preflights_before_any_engine(monkeypatch):
    import dataclasses
    from native_stand.simulation.isaac import motion_backend
    from native_stand.simulation.recovery_isaac import backend
    motion=load_job_config(ROOT/'configs/jobs/motion48_turn_response_DR_full_resume.json')
    motion=dataclasses.replace(motion,name='cpu_only_preflight_fixture')
    monkeypatch.setattr(motion_backend.subprocess,'check_output',lambda *a,**k:f"0, {motion.assets['physical_gpu_uuid']}, 20000\n")
    from types import SimpleNamespace
    monkeypatch.setattr(motion_backend.shutil,'disk_usage',lambda *a:SimpleNamespace(free=20*1024**3))
    assert motion_backend.preflight(motion)['status']=='PASSED'
    # Keep the actual low-disk rejection covered alongside the positive I/O fixture.
    monkeypatch.setattr(motion_backend.shutil,'disk_usage',lambda *a:SimpleNamespace(free=0))
    with pytest.raises(RuntimeError,match='磁盘不足2GiB'):
        motion_backend.preflight(motion)
    recovery=load_job_config(ROOT/'configs/jobs/recovery_preentry48_remaining.json')
    assert backend.preflight(recovery)['status']=='CPU_PREFLIGHT_PASS'


def test_recovery_construction_failure_closes_owned_environment(monkeypatch,tmp_path):
    from native_stand.simulation.recovery_isaac.backend import IsaacRecovery
    job=load_job_config(ROOT/'configs/jobs/recovery_preentry48_remaining.json')
    closed=[]
    def initialize(self,job,output):
        self.env=NS(close=lambda:closed.append('env'))
        raise RuntimeError('script engine I/O construction failure')
    monkeypatch.setattr(IsaacRecovery,'_initialize',initialize)
    external=NS(close=lambda:closed.append('external_app'))
    with pytest.raises(RuntimeError,match='construction failure'):
        IsaacRecovery(job,output=tmp_path,application=external)
    assert closed==['env']


def test_recovery_real_full_state_factory_capture_directory_resume(tmp_path):
    import dataclasses
    from native_stand.artifacts.io import read_json,json_digest
    from native_stand.experiments.config import with_initialization
    from native_stand.learning.ppo.recovery_cubic import build_policy
    from tensordict import TensorDict
    run=Path('/home/kellen/wheel_leg_native_train/runs/2026_11_11__recovery_preentry48_remaining')
    latest=read_json(run/'state.json')['latest_model']
    source=load_source(latest['checkpoint']);assert source.sha256==latest['sha256']
    job=with_initialization(load_job_config(ROOT/'configs/jobs/recovery_preentry48_remaining.json'),source.path,source.sha256)
    job=dataclasses.replace(job,runtime=dataclasses.replace(job.runtime,envs=2,steps_per_env=2,device='cpu'))
    obs=TensorDict({'student_actor':torch.zeros(2,31),'student_critic':torch.zeros(2,909),'recovery_actor_valid':torch.ones(2,1,dtype=torch.bool)},[2])
    handle=build_policy(job,obs,source=source);handle['restore'](source.payload)
    state=handle['capture'](source.payload['counters'],curriculum=source.payload['curriculum'],execution_state=source.payload['execution_state'])
    state['runtime_state']=source.payload['runtime_state']
    state['run_identity']=str(tmp_path/'old')
    recipe=job.as_run_record();state['config_contract_sha256']=json_digest({k:recipe[k] for k in ('model','observation','action','control','reward','curriculum','ppo')})
    for name in ('actor','critic'):
        assert all(torch.equal(v,source.payload[name][k]) for k,v in state[name].items())
    assert state['warmup_updates_done']==5
    assert torch.equal(state['CUDA_rng'][0],source.payload['CUDA_rng'][0])
    checkpoint=tmp_path/'old/checkpoints/full.pt';save_full(checkpoint,state)
    used=state['counters']['updates']
    manifest=dict(version=HANDOFF,run_id='recovery-fixture',owner_terminated=True,
        checkpoint=dict(path=str(checkpoint),sha256=file_sha256(checkpoint)),dependencies=[],
        original_run_identity=str(tmp_path/'old'),config_contract_sha256=state['config_contract_sha256'],
        budget=dict(updates_used=used,remaining=8334-used,limits={'formal':8334}),parent_settlement={'updates':1666})
    manifest_path=tmp_path/'import.json';write_json(manifest_path,manifest)
    imported=import_checkpoint(manifest_path,destination=tmp_path/'new/checkpoints/full.pt',mapping={str(checkpoint):str(checkpoint)},expected_run_id='recovery-fixture')
    second=build_policy(job,obs,source=imported);second['restore'](imported.payload)
    assert second['ppo'].actor_update_enabled
    for owner in ('actor','critic'):
        first_adam=getattr(handle['ppo'].optimizer,owner).state_dict()
        second_adam=getattr(second['ppo'].optimizer,owner).state_dict()
        assert first_adam['param_groups']==second_adam['param_groups']
        for key,value in first_adam['state'].items():
            for field,tensor in value.items():
                assert torch.equal(tensor,second_adam['state'][key][field])


def test_explicit_runtime_path_view_preserves_frozen_json_and_sha(tmp_path):
    from native_stand.artifacts.runtime_paths import RuntimePaths
    old='/unavailable/old-root'
    target=tmp_path/'persistent';target.mkdir()
    file=target/'runtime.json';write_json(file,dict(asset_path=old+'/asset.usd',ground={'material_path':'/World/material'}))
    mapping=tmp_path/'map.json';write_json(mapping,dict(version='runtime_path_mapping_v1',prefixes={old:str(target)}))
    paths=RuntimePaths.from_assets({'path_mapping':dict(path=str(mapping),sha256=file_sha256(mapping))})
    before=file_sha256(file)
    view=paths.read(old+'/runtime.json')
    assert view['asset_path']==str(target/'asset.usd')
    assert view['ground']['material_path']=='/World/material'
    assert file_sha256(file)==before and old in file.read_text()
    with pytest.raises(ValueError,match='SHA'):
        RuntimePaths.from_assets({'path_mapping':dict(path=str(mapping),sha256='f'*64)})


@pytest.mark.parametrize('name',['motion48_turn_response_DR_full_resume','recovery_preentry48_remaining','jump_unified_raw37_wheel48'])
def test_positive_cli_owner_preflight_to_spawn_boundary(tmp_path,monkeypatch,name):
    from native_stand.cli import main
    from native_stand.experiments import owner
    from native_stand.artifacts.io import read_json
    job=load_job_config(ROOT/'configs/jobs'/f'{name}.json')
    monkeypatch.setattr(owner.subprocess,'check_output',lambda *a,**k:f"0, {job.assets.get('physical_gpu_uuid','fixture')}, 20000\n")
    # Resource I/O is a fixture boundary; the test must not depend on free host disk.
    from types import SimpleNamespace
    import shutil
    monkeypatch.setattr(shutil,'disk_usage',lambda *a:SimpleNamespace(free=20*1024**3))
    seen=[]
    def spawn(*args,**kwargs):
        seen.append(args[0]);raise OSError('authorized CPU test stops before spawning engine')
    monkeypatch.setattr(owner.subprocess,'Popen',spawn)
    import json
    raw=json.loads((ROOT/'configs/jobs'/f'{name}.json').read_text())
    raw['name']='cpu-preflight-independent-fixture'
    if name=='jump_unified_raw37_wheel48':
        raw['limits'].update(smoke_updates=1,formal_updates=0,sample_limit=None)
    config=tmp_path/'positive.json';write_json(config,raw)
    assert main(['train','--config',str(config),'--runs-root',str(tmp_path/'runs'),'--run-id','cpu-preflight','--wait-seconds','0'])==1
    assert len(seen)==1
    record=read_json(tmp_path/'runs/cpu-preflight/run.json')
    assert record['logical_run_id']
    assert record['model_index']==str(tmp_path/'runs/models.json')
    assert read_json(tmp_path/'runs/cpu-preflight/state.json')['status']=='FAILED'


def test_maneuver_unknown_interrupts_confirmation_without_losing_levels():
    from native_stand.tasks.motion.maneuver_statistics import RESPONSE_VERSION
    state=maneuver_course_v2.initial_state();state['height_level']=1
    state['streaks'].update(height=1,turn=1,positive=1,negative=1,DR=1)
    state['regressions'].update(height=2,turn=2,positive=2,negative=2)
    stage=maneuver_course_v2.stage_config(state['stage'],2048,state=state)
    rows=[dict(key=key,axis=axis,slot_count=slots,ended_episodes=0,completed_episodes=0,pass_fraction=None)
        for key,axis,slots in zip(stage['cohort_keys'],stage['maneuver_tasks'],stage['slot_counts'])]
    report=dict(version=RESPONSE_VERSION,stage_revision=state['stage'],rows=rows)
    next_state,decision=resolve(maneuver_course_v2.VERSION).advance(state,report,100,envs=2048)
    assert next_state['height_level']==1
    assert set(next_state['streaks'].values())=={0} and set(next_state['regressions'].values())=={0}
    broken=copy.deepcopy(report);broken['rows'][0]['ended_episodes']=16;broken['rows'][0]['pass_fraction']=float('nan')
    with pytest.raises(ValueError,match='malformed'):
        resolve(maneuver_course_v2.VERSION).advance(state,broken,100,envs=2048)


def test_worker_bounded_resume_stop_uses_completed_updates_only():
    from native_stand.experiments.worker import RunContext
    context=NS(updates_used=10,stop_at_update=12,run_dir=Path('/tmp/nonexistent-cpu-bound-fixture'))
    assert not RunContext.stop_requested(context)
    context.updates_used=11;assert not RunContext.stop_requested(context)
    context.updates_used=12;assert RunContext.stop_requested(context)


def test_v8_lifecycle_real_finite_timeout_and_student_ten_second_hold():
    from native_stand.simulation.recovery_isaac.preentry_lifecycle import Lifecycle
    life=Lifecycle(2);life.height_conditioned=True;life.command_height=torch.tensor([.34,.34])
    life.effort_caps=torch.tensor([[40.,40.,4.8,40.,40.,4.8]]).expand(2,-1)
    life.reset(torch.arange(2),torch.tensor([.18,.18]),torch.zeros(2))
    state=torch.zeros(2,13);state[:,3]=.34;state[:,9:11]=3.;state[:,12]=1.
    q=torch.zeros(2,6);q[:,[0,3]]=0.1
    entered=0;success=0
    for step in range(1,1050):
        result=life.observe(time_s=torch.full((2,),step*.01),state=state,target=q,
            motor=torch.zeros(2,6),student=torch.tensor([True,False]),terminal=torch.zeros(2,dtype=torch.bool),
            truncated=torch.zeros(2,dtype=torch.bool),hp10=torch.zeros(2,4),angular_velocity=torch.zeros(2,3),
            physical_q=q,clearance=torch.ones(2)*.15)
        entered+=int(result['entered'][0]);success+=int(result['success'][0])
        assert result['reward'].shape==(2,) and torch.isfinite(result['reward']).all()
    assert entered==success==1
    assert life.success.tolist()==[True,False]
    assert life.hold[1]==0
    timeout=Lifecycle(1);timeout.effort_caps=life.effort_caps[:1]
    timeout.reset(torch.tensor([0]),torch.tensor([.18]),torch.zeros(1))
    low=state[:1].clone();low[:,3]=.20
    result=timeout.observe(time_s=torch.tensor([23.]),state=low,target=q[:1],motor=torch.zeros(1,6),
        student=torch.ones(1,dtype=torch.bool),terminal=torch.zeros(1,dtype=torch.bool),truncated=torch.ones(1,dtype=torch.bool),
        hp10=torch.zeros(1,4),angular_velocity=torch.zeros(1,3),physical_q=q[:1],clearance=torch.ones(1)*.05)
    assert result['truncated'].item() and not result['success'].item()


def test_stop_request_only_never_sends_signal(tmp_path,monkeypatch):
    from native_stand.experiments import owner
    from native_stand.experiments.state import RunStateStore
    from native_stand.cli import main
    store=RunStateStore(tmp_path);store.create(run_id='fixture',owner={'pid':123,'role':'fixture'},stage='formal')
    monkeypatch.setattr(owner,'_candidate_pids',lambda *a:[123])
    monkeypatch.setattr(owner,'process_identity',lambda *a:{'status':'OK','pid':123})
    def forbidden(*a):raise AssertionError('must not signal')
    monkeypatch.setattr(owner.os,'kill',forbidden)
    assert main(['stop','--run',str(tmp_path),'--request-only','--json'])==0
    assert (tmp_path/'stop_request.json').is_file()
    assert store.read()['status']=='SUBMITTED'


def test_selected_branch_fixture_distinguishes_model_progress_and_spent_budget(tmp_path):
    from test_motion_spin_branches import fixture, equal_tree
    from native_stand.simulation.isaac.motion_backend import _resolve_source, _check_source_progress
    job, original, _ = fixture(tmp_path)
    loaded = _resolve_source(job, None)
    assert _check_source_progress(job, loaded.payload, None) == 0
    assert loaded.payload['model_online_updates'] == 5097
    assert job.assets['historical_online_updates'] == 5111
    assert loaded.payload['motion_course_state']['stage'] == 0
    assert loaded.payload['warmup_updates_done'] == 50
    equal_tree(original['actor'], loaded.payload['actor'])
    equal_tree(original['critic'], loaded.payload['critic'])
    equal_tree(original['optimizer'], loaded.payload['optimizer'])
    # The separate HOST suite verifies the pinned u3000 archive through save/resume.


def test_fixed_pin_path_mapping_keeps_actual_file_consumer(tmp_path):
    from native_stand.artifacts.model_refs import ModelIndex,initialize_fixed
    from native_stand.artifacts.runtime_paths import RuntimePaths
    file=tmp_path/'pin.pt';file.write_bytes(b'fixed fixture')
    manifest=tmp_path/'fixed.json';write_json(manifest,dict(fixed_models=[dict(path='/old/pin.pt',sha256=file_sha256(file),consumer='user-pin')]))
    index=ModelIndex(tmp_path/'models.json')
    initialize_fixed(index,manifest,paths=RuntimePaths(files=(('/old/pin.pt',str(file)),)))
    assert str(file) in index.consumer_paths() and '/old/pin.pt' not in index.consumer_paths()
