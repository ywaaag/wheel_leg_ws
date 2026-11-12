"""Portable worker→production adapter→real course/decoder/PPO/state integration.

Only engine I/O and restoration report I/O are replaced; no physics integration.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from tensordict import TensorDict

from native_stand.artifacts.checkpoint import load_complete_state, save_full
from native_stand.artifacts.io import file_sha256
from native_stand.experiments.config import load_job_config
from native_stand.experiments.worker import _execute_stage
from native_stand.learning.ppo.assembly import build_actor, build_critic
from native_stand.learning.ppo.dual_adam import DualAdam
from native_stand.learning.ppo.source_anchor import CAUSAL_VERSION, KEY, SourceAnchor, qualified_valid, metadata
from native_stand.learning.ppo.supported import assemble_from_full, auxiliary_loss
from native_stand.simulation.isaac.motion_backend import MotionProductionAdapter, _check_source_progress
from native_stand.simulation.isaac.environment import control_router
from native_stand.students.resume import rng_state,restore_rng
from native_stand.tasks.motion import c0_boundary_course
from native_stand.tasks.motion.course_strategy import resolve
from native_stand.tasks.motion.scale_corrections import configuration

FROZEN_TEST_SCOPE='PORTABLE'
FIXTURES=Path(__file__).parent/'fixtures'


def make_job(tmp_path, *, warmup=5, retention=True):
    raw=json.loads((FIXTURES/'cpu_motion_job.json').read_text())
    contract=json.loads((FIXTURES/'motion_C0_control_contract.json').read_text())
    contract.update(actor_dimensions=[31,8,6],hidden_activation='ELU(alpha=1)',mean_activation='Tanh')
    cp=tmp_path/'contract.json';cp.write_text(json.dumps(contract))
    template={'control':raw['control'],'motion_height_dr':{},'motion_training':{},
        'student_training':{},'motion_identified_response':{},'curriculum':{}}
    tp=tmp_path/'template.json';tp.write_text(json.dumps(template))
    center={k:v for k,v in contract['center_formula'].items() if k not in ('version','clearance_m')}
    center.update(height_range_m=[.3,.4],support_offset_table=contract['support_offset_table'])
    raw.update(backend='isaac',initialization={'mode':'full','checkpoint':str(tmp_path/'source.pt'),'sha256':'a'*64})
    raw['model'].update(actor_hidden_dims=[8],critic_hidden_dims=[8],action_version=contract['action_version'])
    raw['action'].update(version=contract['action_version'],wheel_scale_rad_s=80.)
    raw['runtime'].update(envs=4,steps_per_env=4)
    raw['ppo'].update(epochs=1,minibatches=1,critic_warmup_updates=warmup,dq_gain_coefficient=0.,boundary_coefficient=0.,exploration_leg_rad=.02,exploration_wheel_rad_s=2.)
    raw['curriculum'].update(version=c0_boundary_course.VERSION,stages=[c0_boundary_course.configuration()])
    scale=configuration();scale['output_scales']=[.15,80.]
    raw['reward']['fields']={'scale_corrections':scale}
    raw['assets'].update(center=center,runtime_template=str(tp),runtime_template_sha256=file_sha256(tp),
        frozen_source_root=str(tmp_path),frozen_bundle=str(tmp_path),prefix_onnx=str(tmp_path/'unused.onnx'),
        prefix_sha256='b'*64,physical_gpu_uuid='not-used-CPU',historical_online_updates=0,
        original_online_limit=10000,prefix_output_contract=str(cp),prefix_output_contract_sha256=file_sha256(cp),
        output_contract=str(cp),output_contract_sha256=file_sha256(cp),center_definition_canonical_json_sha256='c'*64,
        source_boundary_frontiers=dict.fromkeys(('forward','backward','yaw_positive','yaw_negative'),0))
    if retention:
        raw['assets']['source_policy_retention']=dict(version=CAUSAL_VERSION,coefficient=.01,
            source_sha256='a'*64,commands=[[0.,0.]],maximum_states_per_cohort=4,record_first_gradient=True)
    path=tmp_path/'job.json';path.write_text(json.dumps(raw))
    return load_job_config(path),contract


def observations(job):
    return TensorDict({'student_actor':torch.randn(4,31),'student_critic':torch.randn(4,64),
        'recovery_actor_valid':torch.ones(4,1,dtype=torch.bool),'motion_diagnostic_cohort':torch.zeros(4,1),KEY:torch.zeros(4,1,dtype=torch.long)},batch_size=[4])


def full_source(job,contract,obs):
    actor,critic=build_actor(job),build_critic(job)
    opt=DualAdam(actor.parameters(),critic.parameters(),actor_lr=job.ppo.learning_rate,critic_lr=job.ppo.critic_learning_rate)
    state=c0_boundary_course.initial_state(dict.fromkeys(('forward','backward','yaw_positive','yaw_negative'),0))
    source=dict(actor=actor.state_dict(),critic=critic.state_dict(),optimizer=opt.state_dict(),
        training_rng=rng_state(),rng=torch.get_rng_state(),std=actor.log_std.exp().detach(),counters={},
        network_architecture={'actor_hidden_dims':[8],'critic_hidden_dims':[8]},
        interface_version=job.model.interface,critic_privileged_version=job.model.critic_privileged_version,
        output_contract=contract,action_version=job.action.version,motion_course_state=state,
        reward_refinement_state=[],symmetry_mix_state=[],warmup_updates_done=5,
        update=0,online_updates_including_smoke=0,recipe_updates=0)
    resolve(job.curriculum.version).prepare(source,job.runtime.envs)
    return source


def rollout(algorithm,obs,steps):
    with torch.inference_mode():
        for _ in range(steps):
            action=algorithm.act(obs)
            algorithm.process_env_step(obs,1.-action.square().mean(-1),torch.zeros(4,dtype=torch.bool),{})
        algorithm.compute_returns(obs)
    return algorithm.update()


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b,strict=True):equal_tree(x,y)
    else:assert a==b


def test_dual_adam_continuous_matches_save_restore(tmp_path):
    job,contract=make_job(tmp_path)
    torch.manual_seed(41);obs=observations(job);source=full_source(job,contract,obs)
    first=assemble_from_full(source,obs,job);restore_rng(source['training_rng'])
    rollout(first,obs,4)
    state=copy.deepcopy(source)
    state.update(actor=copy.deepcopy(first.actor.state_dict()),critic=copy.deepcopy(first.critic.state_dict()),
        optimizer=copy.deepcopy(first.optimizer.state_dict()),training_rng=rng_state(),
        source_anchor_actor=copy.deepcopy(first.source_anchor.teacher.state_dict()),source_policy_retention=dict(job.assets['source_policy_retention']))
    save_full(tmp_path/'split.pt',state)
    rollout(first,obs,4)
    expected_rng=torch.get_rng_state()
    loaded=load_complete_state(tmp_path/'split.pt')
    second=assemble_from_full(loaded.payload,obs,job)
    restore_rng(loaded.payload['training_rng'])
    rollout(second,obs,4)
    equal_tree(first.actor.state_dict(),second.actor.state_dict())
    equal_tree(first.critic.state_dict(),second.critic.state_dict())
    equal_tree(first.optimizer.state_dict(),second.optimizer.state_dict())
    equal_tree(first.source_anchor.teacher.state_dict(),second.source_anchor.teacher.state_dict())
    assert torch.equal(first.actor.log_std,second.actor.log_std)
    assert torch.equal(expected_rng,torch.get_rng_state())


def test_actor_only_real_optimizer_first5_critic_then6_actor(tmp_path):
    job,contract=make_job(tmp_path,retention=False)
    obs=observations(job);source=full_source(job,contract,obs)
    source['warmup_updates_done']=0
    algorithm=assemble_from_full(source,obs,job,fresh=True)
    actor=copy.deepcopy(algorithm.actor.state_dict());critic=copy.deepcopy(algorithm.critic.state_dict())
    for index in range(6):
        algorithm.actor_update_enabled=index>=5
        rollout(algorithm,obs,4)
        if index<5:equal_tree(actor,algorithm.actor.state_dict())
    assert any(not torch.equal(v,algorithm.actor.state_dict()[k]) for k,v in actor.items())
    assert any(not torch.equal(v,algorithm.critic.state_dict()[k]) for k,v in critic.items())
    assert len(algorithm.optimizer.actor.state)>0 and len(algorithm.optimizer.critic.state)>0


def test_source_KL_real_actor_loss_nonzero_and_diagnostics_do_not_mutate(tmp_path):
    job,contract=make_job(tmp_path)
    obs=observations(job);source=full_source(job,contract,obs)
    algorithm=assemble_from_full(source,obs,job)
    frozen=copy.deepcopy(algorithm.source_anchor.teacher.state_dict())
    with torch.no_grad():algorithm.actor.mlp[-1].bias.add_(.1)
    rollout(algorithm,obs,4)
    summary=algorithm.mask_summary['source_policy_retention']
    assert summary['weighted_loss_mean']>0
    assert summary['gradient_comparison']['weighted_source_KL_gradient_norm']>0
    equal_tree(frozen,algorithm.source_anchor.teacher.state_dict())


@pytest.mark.parametrize('blocked',['DR','event','teacher','empty'])
def test_source_constraint_causal_mask(blocked,tmp_path):
    job,_=make_job(tmp_path);actor=build_actor(job);anchor=SourceAnchor(actor,job.assets['source_policy_retention'])
    with torch.no_grad():actor.mlp[-1].bias.add_(.1)
    obs=observations(job);valid=torch.ones(4,dtype=torch.bool)
    profile=torch.zeros(4,dtype=torch.long) if blocked=='DR' else torch.full((4,),-1)
    if blocked in ('teacher','empty'):valid.zero_()
    valid=qualified_valid(valid,event_enabled=torch.full((4,),blocked=='event'),age_s=torch.full((4,),3.),
        pitch=torch.zeros(4),roll=torch.zeros(4),wheel_support=torch.full((4,2),10.))
    metadata(obs,torch.zeros(4,2),profile,valid,[[0.,0.]])
    before=torch.get_rng_state();cache=actor._current_distribution
    loss=anchor.loss(actor,obs,valid)
    assert loss==0 and not loss.requires_grad
    assert torch.equal(before,torch.get_rng_state()) and actor._current_distribution is cache


def test_disabled_gain_skips_FD_and_preserves_RNG_cache(tmp_path,monkeypatch):
    job,_=make_job(tmp_path);actor=build_actor(job)
    monkeypatch.setattr('native_stand.learning.ppo.supported.physical_gain',lambda *a,**k:pytest.fail('disabled FD executed'))
    obs=torch.randn(4,31);before=torch.get_rng_state();cache=actor._current_distribution
    loss,detail=auxiliary_loss(actor,obs,torch.ones(4,dtype=torch.bool),job.ppo)
    assert loss==0 and detail['gain_skipped']
    assert torch.equal(before,torch.get_rng_state()) and actor._current_distribution is cache


def test_rebuild_closes_before_creation_and_failure_has_no_zombie_env():
    adapter=MotionProductionAdapter();old=NS(closed=False)
    old.close=lambda:setattr(old,'closed',True)
    adapter.env=old
    def fail():
        assert old.closed and adapter.env is None and adapter.obs is None
        raise RuntimeError('construction failed')
    adapter._make_env=fail
    with pytest.raises(RuntimeError,match='construction failed'):adapter._rebuild_environment()
    assert adapter.env is None


def test_independent_and_resume_budget_semantics(tmp_path):
    job,_=make_job(tmp_path)
    from dataclasses import replace
    assets=dict(job.assets,new_independent_authorization=True,source_online_updates=3302)
    job=replace(job,assets=assets)
    assert _check_source_progress(job,{'online_updates_including_smoke':3302},None)==0
    source={'online_updates_including_smoke':12,'counters':{'new_run_updates':12}}
    assert _check_source_progress(job,source,NS(payload=source))==12


def test_public_worker_uses_production_course_decoder_optimizer_and_capture(tmp_path,monkeypatch):
    from dataclasses import replace
    from test_phase3_fixes import _production_context
    job,contract=make_job(tmp_path)
    obs=observations(job);source=full_source(job,contract,obs)
    source.update(source_anchor_actor=copy.deepcopy(source['actor']),source_policy_retention=dict(job.assets['source_policy_retention']))
    teacher_path=tmp_path/'teacher.pt';torch.save({'actor':source['actor'],'actor_only':True},teacher_path)
    identity=dict(path=str(teacher_path),sha256=file_sha256(teacher_path))
    retention=dict(job.assets['source_policy_retention'],source_sha256=identity['sha256'])
    job=replace(job,assets=dict(job.assets,source_anchor_identity=identity,source_policy_retention=retention))
    source['source_policy_retention']=retention
    save_full(job.initialization.checkpoint,source)
    job=replace(job,initialization=replace(job.initialization,sha256=file_sha256(job.initialization.checkpoint)))
    context=_production_context(tmp_path,job,smoke=1)
    seen={}
    class Teacher(torch.nn.Module):
        def mean(self,x):return x.new_tensor([.2,.2,.5,.2,.2,-.5]).expand(len(x),-1)
    def engine_io(adapter):
        # Replace engine allocation only. All prepared semantic state is real.
        stage=adapter.spec['curriculum']['stages'][0]
        assert len(stage['commands'])==len(adapter.source['symmetry_mix_state'])
        env=NS(num_envs=4,device='cpu',fsm=NS(mode=torch.full((4,),2,dtype=torch.long)),
            initialization_checked=torch.ones(4,dtype=torch.bool),height_adapter=NS(target=torch.full((4,),.34)),
            student_obs=obs,identified_dr=NS(episodes=torch.zeros(4,dtype=torch.long)))
        router=control_router(adapter.job,adapter.source,Teacher())
        router.initialize(env,{})
        requested,student=router.actions(env,torch.tensor([[0.,0.,1.,0.,0.,-1.]]).expand(4,-1))
        seen['wheels']=(requested[:,[2,5]]*100).tolist()
        seen['student']=student.tolist()
        env.close=lambda:seen.update(closed=True)
        adapter.env=env;adapter.obs=obs
        adapter.modules={'rng_state':rng_state}
        seen['adapter']=adapter
    monkeypatch.setattr(MotionProductionAdapter,'_open_environment',engine_io)
    monkeypatch.setattr(MotionProductionAdapter,'_write_restoration',lambda self:None)
    # Stop at the first boundary: exercise actual capture without physics or PPO.
    monkeypatch.setattr(context,'stop_requested',lambda:True)
    try:
        result=_execute_stage(context,'smoke',None)
        assert result.updates==0 and result.last_checkpoint is not None
        assert seen['wheels']==[[80.,-80.]]*4 and all(seen['student'])
        assert seen['closed']
        saved=load_complete_state(result.last_checkpoint['checkpoint']).payload
        assert saved['optimizer']['version']==DualAdam.VERSION
        assert saved['warmup_updates_done']==5
        assert saved['source_policy_retention']==dict(job.assets['source_policy_retention'])
        assert saved['motion_course_state']['version']==c0_boundary_course.VERSION
        assert len(saved['motion_course_state']['assessment_keys'])==len(saved['reward_refinement_state'])
    finally:
        context.metrics.close()


def test_save_publish_reference_prune_order_and_retained_set(tmp_path):
    from test_phase3_fixes import _production_context
    from native_stand.simulation.driver import run_stage
    from native_stand.tasks.motion.motion_candidates import FAST_YAW_VERSION
    job,_=make_job(tmp_path,retention=False)
    context=_production_context(tmp_path,job,smoke=5)
    context.policy=type(context.policy)(every_updates=1,keep_recent=2,milestones=())
    sequence=[]
    class Adapter(MotionProductionAdapter):
        def build(self,context,stage,resume_from):
            self.source={};self.source_path=tmp_path/'ancestor.pt';self.source_sha='f'*64
        def update_once(self,context,index):
            sequence.append(('update',index))
            return dict(actor_steps=1,reward_mean=1.,seconds_per_update=.123)
        def after_update(self,context,index,outcome):
            sequence.append(('after',index))
            # Movement winner at2, standing winner at3;4/5 should not replace them.
            self._candidate_pending=dict(version=FAST_YAW_VERSION,comparison_sha256='e'*64,
                eligible=True,DR_level=0,through_update=index,reason='CPU fixture',
                motion_rank=[10 if index==2 else 0],standing_rank=[10 if index==3 else 0])
        def capture(self,context,counters):
            sequence.append(('capture',context.updates_used))
            return dict(actor={},critic={},optimizer={},std=torch.ones(6),rng=torch.get_rng_state(),counters=counters)
        def after_save(self,context,reference):
            assert Path(reference['checkpoint']).is_file()
            assert file_sha256(reference['checkpoint'])==reference['sha256']
            sequence.append(('published',reference['update']))
            super().after_save(context,reference)
        def close(self):pass
    original=context.prune_checkpoints
    def prune():
        assert sequence[-1][0]=='published'
        sequence.append(('prune',context.updates_used))
        return original()
    context.prune_checkpoints=prune
    try:
        run_stage(context,Adapter(),stage='smoke')
        assert {p.name for p in (context.run_dir/'checkpoints').glob('*.pt')}=={
            'model_update_2.pt','model_update_3.pt','model_update_4.pt','model_update_5.pt'}
        assert sequence==[(name,i) for i in range(1,6) for name in ('update','after','capture','published','prune')]
        assert context.store.read()['best_model'] is None
        assert not context.store.read()['capability_accepted']
        rows=context.refs.rows()
        assert {Path(r['path']).name for r in rows if 'training_candidate' in r['roles']}=={'model_update_2.pt','model_update_3.pt'}
        metrics=[json.loads(line) for line in (context.run_dir/'metrics.jsonl').read_text().splitlines()]
        assert all(row['seconds_per_update']==.123 for row in metrics)
    finally:context.metrics.close()


def test_production_update_crosses_30s_terminal_and_bootstraps_once(tmp_path):
    from dataclasses import replace
    from native_stand.students.motion_hotpath import runner_flags
    from native_stand.students.motion_statistics import MotionStatistics
    from native_stand.students.standing_survival_task import bootstrap_reward
    from native_stand.students.step_checks import StepChecks
    from native_stand.tasks.motion.value_diagnostics import from_storage
    job,contract=make_job(tmp_path,retention=False)
    job=replace(job,runtime=replace(job.runtime,steps_per_env=3001))
    obs=observations(job);source=full_source(job,contract,obs)
    algorithm=assemble_from_full(source,obs,job)
    called=[]
    class EngineIO:
        device='cpu'
        num_envs=4
        stage_spec={'commands':[[0.,0.]]}
        command_index=torch.zeros(4,dtype=torch.long)
        rise_hold=NS(episode=NS(steps=torch.zeros(4,dtype=torch.long)))
        student_actor_valid=torch.ones(4,dtype=torch.bool)
        recovery_snapshot=torch.zeros(4,31)
        standing_snapshot=torch.zeros(4,6)
        wheel_geometry_snapshot=torch.zeros(4,9)
        motion_action_command=torch.tensor([[0.,0.,.34]]).expand(4,-1)
        ticks=0
        def step(self,raw):
            self.ticks+=1;self.rise_hold.episode.steps.add_(1)
            terminated=torch.zeros(4,dtype=torch.bool);truncated=terminated.clone()
            if self.ticks==3000:
                terminated[1]=True;truncated[0]=True
                self.student_terminal_critic=torch.full((4,64),99.)
                self.rise_hold.episode.steps[:2]=0
            next_obs=obs.clone()
            next_obs['student_critic'].fill_(-99.)
            return next_obs,torch.ones(4),terminated,truncated,{'time_outs':truncated}
    env=EngineIO();adapter=MotionProductionAdapter()
    adapter.job=job;adapter.env=env;adapter.algorithm=algorithm;adapter.obs=obs
    adapter.source=source;adapter.restore_point_updates=0
    adapter.fixed_std=algorithm.actor.log_std.detach().clone()
    adapter.spec={'student_training':{'train_schedule_s':[2.,8.,8.,8.],'parking_s':4.}}
    def bootstrap(reward,terminated,truncated,value,*,gamma):
        called.append((value.clone(),terminated.clone(),truncated.clone()))
        expected=algorithm.critic({'student_critic':torch.full((4,64),99.)}).squeeze(-1)
        assert torch.equal(value,expected)
        return bootstrap_reward(reward,terminated,truncated,value,gamma=gamma)
    adapter.modules=dict(KEY='recovery_actor_valid',MotionStatistics=MotionStatistics,
        StepChecks=StepChecks,runner_flags=runner_flags,bootstrap_reward=bootstrap,
        observations=lambda env,value:value,value_diagnostics=from_storage)
    outcome=adapter.update_once(NS(stop_requested=lambda:False,updates_used=0),1)
    assert env.ticks==3001 and outcome['episode_completed']==2
    assert len(called)==1
    assert called[0][1].tolist()==[False,True,False,False]
    assert called[0][2].tolist()==[True,False,False,False]
    assert outcome['actor_steps']>0


def test_execution_buffers_construct_only_consumed_state_and_partial_reset():
    from native_stand.simulation.isaac.execution_buffers import ExecutionBuffers
    from native_stand.students.motion_wheel_matching import profile
    armature=torch.zeros(2,6)
    def write(armature:torch.Tensor,joint_ids):
        values=armature
        # Explicit engine-I/O boundary; state readback is exercised by the real factory.
        stored[:,joint_ids.long()]=values
    stored=armature
    robot=NS(data=NS(joint_pos=NS(torch=torch.arange(12,dtype=torch.float32).reshape(2,6)*.01),
        joint_vel=NS(torch=torch.zeros(2,6))),write_joint_armature_to_sim_index=write,
        root_view=NS(get_dof_armatures=lambda:stored,get_dof_dampings=lambda:torch.zeros_like(stored)))
    env=NS(num_envs=2,device='cpu',ids=list(range(6)),robot=robot,cfg=NS(wheel_kv=.2))
    buffers=ExecutionBuffers(env,{'wheel_execution':profile()},armature_level=4)
    buffers.reset(torch.arange(2))
    assert buffers.episodes.tolist()==[1,1]
    assert torch.all(buffers.actual_armature==.012)
    buffers.tick=129
    untouched=buffers.targets[:,1].clone()
    buffers.reset(torch.tensor([0]))
    assert buffers.tick==129 and buffers.episodes.tolist()==[2,1]
    assert torch.equal(buffers.targets[:,1],untouched)
    assert buffers.targets.shape==(32,2,6)
    assert not buffers.compensation.any()
