"""Synthetic engineering tests only: zero physical engine integration steps."""
FROZEN_TEST_SCOPE = 'PORTABLE'

from dataclasses import fields, replace
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from native_stand.world_model.state import State, initial
from native_stand.world_model.execution import step, observation
from native_stand.world_model.geometry import quat_exp, quat_matrix, integrate, rotation_error
from native_stand.world_model.models import Ensemble, Critic, value_features
from native_stand.world_model.learner import actor_step, critic_step, bootstrap_target


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(n=2):
    from native_stand.students.motion_mcu_execution import profile,spring_compensation
    p=profile();q=torch.zeros(n,13);q[:,2]=.34;q[:,3:7]=q.new_tensor([2**-.5,2**-.5,0.,0.])
    sensors=torch.zeros(n,18);sensors[:,4]=-1
    comp,_,_,_=spring_compensation(torch.zeros(n,4))
    return initial(q,torch.zeros(n,12),sensors,torch.zeros(n,3),torch.tensor([[0.,0.,.34,0.]]).expand(n,-1),
        compensation=comp,cap=torch.tensor(p['driver_limits_nm']).expand(n,-1),
        delay=torch.tensor(p['motor_delay_ms']).expand(n,-1),tau=torch.tensor(p['motor_tau_ms'],dtype=torch.float32).expand(n,-1))


class TinyGeometry(nn.Module):
    """Declared artificial differentiable fixture, no robot capability evidence."""
    active_q=list(range(7,13));active_v=list(range(6,12));root=1
    def __init__(self):
        super().__init__();self.register_buffer('limits',torch.tensor([[-2.,2.]]*6))
    def forward(self,q,v):
        gravity=torch.zeros_like(q[:,:3]);gravity[:,1]=-1
        sensor=torch.cat((v[:,3:6],gravity,q[:,7:],v[:,6:]),-1)
        return dict(sensors=sensor,com_velocity=v[:,:3,None].transpose(-1,-2).expand(-1,2,-1),
            closure=q.new_zeros(len(q),1,3),energy=v.square().sum(-1),velocity=v[:,:3],
            height=q[:,2],penetration=q.new_zeros(len(q),1))
    def valid(self,q,v,m):return torch.isfinite(q).all(-1)&torch.isfinite(v).all(-1)
    def effective_torque(self,requested):return requested
    def contact_valid(self,state):return torch.ones(len(state.q),dtype=torch.bool,device=state.q.device)


class TinyMechanics(nn.Module):
    def __init__(self):
        super().__init__();self.gain=nn.Parameter(torch.tensor(.05));self.members=[None]
    def predict(self,s,applied,member=0):
        # A known smooth artificial plant where action gradients can be checked.
        dv=torch.cat((applied[:,:3],applied[:,3:6],applied),-1)*self.gain*.001
        v=s.v+dv
        return replace(s,q=integrate(s.q,v),v=v,solver_q=s.q,solver_v=s.v)
    def disagreement(self,s,applied):return (applied*0).sum(-1)


class TinyReward:
    def __call__(self,b,a,m):
        r=-(a.v[:,0]-.05).square()-.01*a.v[:,6:].square().mean(-1)
        return a,{'total':r},torch.zeros(len(a.q),dtype=torch.bool)


class Actor(nn.Module):
    def __init__(self):
        super().__init__();self.mlp=nn.Linear(31,6);self.log_std=nn.Parameter(torch.zeros(6),requires_grad=False)
        nn.init.zeros_(self.mlp.weight);nn.init.constant_(self.mlp.bias,.02)
    def mean(self,x):return self.mlp(x).tanh()


def test_actor31_fields_left_first_and_source_contract():
    s=fixture(1);m=TinyGeometry()(s.q,s.v)
    s=replace(s,adopted=torch.tensor([[.15,.3,20.,-.45,.6,-30.]]),task=torch.tensor([[.5,-.5,.34,0.]]))
    obs=observation(s,m)
    assert obs.shape==(1,31)
    torch.testing.assert_close(obs[0,16:20],torch.tensor([.1,.2,-.3,.4]))
    torch.testing.assert_close(obs[0,20:22],torch.tensor([.2,-.3]))
    torch.testing.assert_close(obs[0,22:25],torch.tensor([.125,-.125,.68]))
    assert obs[0,25:].tolist()==[1,0,0,1,0,0]
    assert obs[0,4]==-1
    from native_stand.students.interface import ORDER
    assert ORDER[0]=='left_hip' and ORDER[3]=='right_hip'


def test_differentiable_execution_matches_authoritative_chain_without_physics():
    # Test MCU math/time using scripted feedback only; no mj_step or physics sampler.
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from motion_mcu_playback import MCUPlayback
    from native_stand.students.motion_mcu_execution import profile
    oracle=MCUPlayback({'profile':profile()},np.zeros(6),np.zeros(6))
    s=fixture(1)
    for tick in range(137):
        q=np.sin(tick*.01+np.arange(6))*.01;dq=np.cos(tick*.01+np.arange(6))*.1
        target=np.array([.04,.02,4.,-.03,.01,-3.],np.float32)*(1 if tick<70 else -.5)
        sensor=torch.zeros(1,18);sensor[:,4]=-1;sensor[:,6:12]=torch.tensor(q,dtype=torch.float32)
        sensor[:,12:]=torch.tensor(dq,dtype=torch.float32)
        expected=oracle.step(tick,target,q,dq)
        s,valid=step(s,torch.from_numpy(target)[None],sensor)
        assert valid.all()
        for value,want in ((s.applied,expected[None]),(s.adopted,oracle.previous[None]),
                            (s.driver,oracle.driver[None]),(s.motor,oracle.motor_actual[None]),
                            (s.compensation,oracle.compensation[None]),(s.pd_request,oracle.pre[None])):
            torch.testing.assert_close(value,want,atol=3e-6,rtol=2e-6)


def test_delays_previous_and_partial_reset_are_causal():
    s=fixture();current=s.sensors[:,0];target=s.pending.clone();target[:,2]=10
    # Tick1 isn't adoption. Pending/shadow requested target must not become previous.
    one=replace(s,tick=s.tick+1)
    after,_=step(one,target,current)
    torch.testing.assert_close(after.adopted,s.adopted)
    after,_=step(s,target,current)
    assert after.adopted[:,2].eq(10).all()
    # Motor torque is still the prehistory until ch2's7ms transport has elapsed.
    assert after.motor[:,2].eq(0).all()
    untouched=after.rows([1]).clone()
    reset=after.reset_rows([0],fixture(1))
    torch.testing.assert_close(reset.tick,after.tick) # global1ms/PD phase survives partial reset
    for f in fields(State):torch.testing.assert_close(getattr(reset,f.name)[1:2],getattr(untouched,f.name))
    # Controller handover changes only owner; queues, physics and previous survive.
    changed=replace(after,owner=after.owner+1)
    for k in ('q','v','sensors','queue','adopted','motor'):torch.testing.assert_close(getattr(changed,k),getattr(after,k))


def test_state_roundtrip_has_full_execution_memory(tmp_path):
    s=fixture();p=tmp_path/'state.pt';torch.save(s.as_dict(),p)
    restored=State(**torch.load(p,weights_only=True))
    for f in fields(State):torch.testing.assert_close(getattr(s,f.name),getattr(restored,f.name))
    assert restored.sensors.shape==(2,64,18) and restored.queue.shape==(2,64,6)
    assert restored.quality_ticks.shape==(2,7)


def test_quaternion_double_cover_and_integrator_autograd():
    vector=torch.tensor([[.01,.02,-.03]],dtype=torch.float64,requires_grad=True)
    q=quat_exp(vector)
    torch.testing.assert_close(quat_matrix(q),quat_matrix(-q))
    assert rotation_error(q,-q).item()==0
    assert torch.autograd.gradcheck(lambda x:quat_matrix(quat_exp(x)),(vector,),atol=1e-6)
    s=fixture(1);out=integrate(s.q,torch.ones_like(s.v),.001)
    torch.testing.assert_close(out[:,3:7].norm(dim=-1),torch.ones(1))


def test_model_shared_across_commands_but_critic_sees_task_and_banks():
    s=fixture();wm=Ensemble(13,12,3,members=3,width=16,blocks=2)
    other=replace(s,task=torch.tensor([[.5,0.,.34,0.],[0.,-.5,.34,0.]]))
    torch.testing.assert_close(wm.predict(s,s.applied).v,wm.predict(other,other.applied).v)
    assert not torch.equal(value_features(s),value_features(other))
    queued=replace(s,queue=s.queue+1)
    assert not torch.equal(value_features(s),value_features(queued))
    assert len(wm.members)==3


def test_world_supervision_really_updates_mechanics():
    s=fixture();wm=Ensemble(13,12,3,members=1,width=16,blocks=2)
    opt=torch.optim.Adam(wm.parameters(),lr=.001)
    before=copy.deepcopy(wm.state_dict());loss=wm.predict(s,s.applied).v.square().mean()
    opt.zero_grad();loss.backward();opt.step()
    assert any(not torch.equal(before[k],v) for k,v in wm.state_dict().items() if v.dtype==torch.float32)


def test_contact_head_has_no_artificial_force_floor():
    from native_stand.world_model.models import mechanics_targets
    s=fixture(2);wm=Ensemble(13,12,3,members=1,width=16,blocks=1)
    target=mechanics_targets(s.v,s.v,s.warm,s.contact)
    wm.target.fit(target)
    for p in wm.members[0].parameters():nn.init.zeros_(p)
    predicted=wm.predict(s,s.generalized)
    assert predicted.contact.abs().max()<2e-6


def test_bptt_updates_actor_while_world_and_critic_remain_frozen():
    s=fixture(2);actor=Actor();teacher=copy.deepcopy(actor);wm=TinyMechanics();geometry=TinyGeometry()
    critic=Critic(value_features(s).shape[-1],width=16)
    optimizer=torch.optim.Adam(actor.mlp.parameters(),lr=1e-5)
    w=copy.deepcopy(wm.state_dict());c=copy.deepcopy(critic.state_dict());a=copy.deepcopy(actor.state_dict())
    result=actor_step(actor,teacher,wm,critic,geometry,TinyReward(),s,optimizer,horizon=8,
                      retention=0.,max_normalized_action_change=1.)
    assert result['updated'] and result['grad_norm']>0 and result['actor_optimizer_steps']==1
    assert any(not torch.equal(a[k],v) for k,v in actor.state_dict().items())
    assert all(torch.equal(w[k],v) for k,v in wm.state_dict().items())
    assert all(torch.equal(c[k],v) for k,v in critic.state_dict().items())
    assert wm.gain.grad is None and all(p.grad is None for p in critic.parameters())


def test_multistep_autograd_matches_finite_difference():
    from native_stand.world_model.validation import imagined_objective
    from native_stand.world_model.learner import frozen
    s=fixture(1).map(lambda x:x.double() if x.is_floating_point() else x)
    actor=Actor().double();wm=TinyMechanics().double();kin=TinyGeometry().double()
    critic=Critic(value_features(s).shape[-1],width=16).double()
    action=torch.full((1,6),.02,dtype=torch.float64,requires_grad=True)
    with frozen(actor,wm,critic):
        f=lambda a:imagined_objective(s,a,actor,wm,critic,kin,TinyReward(),horizon=8)[0]
        value=f(action);gradient=torch.autograd.grad(value,action)[0]
        delta=torch.zeros_like(action);delta[:,2]=1e-5
        finite=(f(action+delta)-f(action-delta))/(2e-5)
    torch.testing.assert_close(gradient[:,2],finite,atol=2e-6,rtol=2e-3)
    assert torch.isfinite(gradient).all() and gradient.norm()>0


def test_gradient_report_keeps_batch_and_contact_axes_separate():
    from native_stand.world_model.validation import action_gradient
    s=fixture(2).map(lambda x:x.double() if x.is_floating_point() else x)
    actor=Actor().double();wm=TinyMechanics().double();kin=TinyGeometry().double()
    critic=Critic(value_features(s).shape[-1],width=16).double()
    report,gradient=action_gradient(s,actor,wm,critic,kin,TinyReward(),horizon=8)
    assert np.asarray(report['smooth_mask']).shape==(2,6) and gradient.shape==(2,6)
    assert report['finite'] and report['smooth_relative_error_mean']<.01


def test_one_and_eight_step_mechanics_loss_backpropagates():
    from native_stand.world_model.training import supervised_loss
    s=fixture(2);wm=Ensemble(13,12,3,members=1,width=16,blocks=2);kin=TinyGeometry()
    batch={k:v for k,v in dict(q=s.q,v=s.v,warm=s.warm,applied=s.generalized,next_q=s.q,
        next_v=s.v,next_warm=s.warm,contact=s.contact,context=s.context).items()}
    one,_=supervised_loss(wm,0,batch,kin)
    window={k:(v if k=='context' else v[:,None].expand(-1,8,-1)) for k,v in batch.items()}
    many,_=supervised_loss(wm,0,window,kin)
    (one+many).backward()
    assert torch.isfinite(one) and torch.isfinite(many)
    assert sum(float(p.grad.abs().sum()) for p in wm.members[0].parameters() if p.grad is not None)>0


def test_invalid_imagination_cannot_dispatch_actor_optimizer():
    s=fixture(1);actor=Actor();teacher=copy.deepcopy(actor);wm=TinyMechanics();kin=TinyGeometry()
    kin.valid=lambda q,v,m:torch.zeros(len(q),dtype=torch.bool)
    critic=Critic(value_features(s).shape[-1],width=16);opt=torch.optim.Adam(actor.mlp.parameters(),lr=.001)
    before=copy.deepcopy(actor.state_dict());calls=[]
    result=actor_step(actor,teacher,wm,critic,kin,TinyReward(),s,opt,before_optimizer=lambda:calls.append(1))
    assert not result['updated'] and result['actor_optimizer_steps']==0 and calls==[]
    assert all(torch.equal(before[k],v) for k,v in actor.state_dict().items())


def test_critic_terminal_and_timeout_bootstrap_and_update():
    y=bootstrap_target(torch.tensor([1.,1.,1.]),torch.tensor([True,False,True]),
        torch.tensor([False,True,True]),torch.tensor([10.,10.,10.]),.9)
    torch.testing.assert_close(y,torch.tensor([1.,10.,1.]))
    s=fixture();critic=Critic(value_features(s).shape[-1],width=16);target=copy.deepcopy(critic)
    before=copy.deepcopy(critic.state_dict());opt=torch.optim.Adam(critic.parameters(),lr=.001)
    result=critic_step(critic,target,opt,s,s,torch.ones(2),torch.tensor([True,False]),torch.tensor([False,True]))
    assert result['grad_norm']>0 and any(not torch.equal(before[k],v) for k,v in critic.state_dict().items())


def test_real_episode_split_rejects_tuple_leak_and_reset_splice(tmp_path):
    from native_stand.world_model.dataset import write_episode,build_manifest
    s=fixture(1)
    def ep(name,split):
        row=dict(before=s,after=replace(s,tick=s.tick+10),action=torch.zeros(1,6),observation=torch.zeros(1,31),reward=torch.ones(1),
            terminated=torch.tensor([False]),truncated=torch.tensor([True]),reward_raw=torch.zeros(1,1),
            reward_density=torch.zeros(1,1),reward_dt=torch.zeros(1,1),reward_fields=['fixture'])
        micro=[dict(q=s.q[0].numpy(),v=s.v[0].numpy(),warm=s.warm[0].numpy(),applied=s.applied[0].numpy(),
            next_q=s.q[0].numpy(),next_v=s.v[0].numpy(),next_warm=s.warm[0].numpy(),contact=s.contact[0].numpy(),
            body_contact=s.body_contact[0].numpy())]*10
        case=dict(id=name,split=split,tuple_id='same-tuple')
        p=tmp_path/name;write_episode(p,case,[row],micro,[np.zeros(2)],{'fixture':True});return p
    a,b=ep('a','train'),ep('b','validation')
    with pytest.raises(ValueError,match='tuple leaks'):build_manifest([a,b],tmp_path/'manifest.json',{})
    result=build_manifest([a],tmp_path/'safe.json',{})
    assert result['tasks']==['Motion'] and result['Jump']=='NOT_COVERED_NO_FLIGHT_OR_LANDING_DATA'


def test_legacy_trace_cannot_be_called_complete_markov(tmp_path):
    from native_stand.world_model.dataset import build_manifest
    p=tmp_path/'legacy';p.mkdir();(p/'episode.json').write_text(json.dumps({'version':'old31D'}))
    with pytest.raises(ValueError,match='Markov episode schema'):build_manifest([p],tmp_path/'m.json',{})


def test_selection_uses_real_same_conditions_and_old_ability():
    from native_stand.world_model.validation import choose_physical
    def result(v):return {'a':dict(imagined=False,condition_identity='fixed',command=[0.,0.],summary=dict(
        physical_seconds=30.,tracking_MAE=v,contact_fraction=1.,roll_peak=.01,pitch_peak=.01,vx_abs_mean=v))}
    assert choose_physical(result(.14),result(.10))['selected']
    worse=result(.1);worse['a']['summary']['contact_fraction']=.5
    assert not choose_physical(result(.14),worse)['selected']
    imagined=result(.1);imagined['a']['imagined']=True
    with pytest.raises(ValueError,match='imagined reward'):choose_physical(result(.14),imagined)


def test_old_interfaces_remain_explicit_and_unchanged():
    from native_stand.students.model import Student
    from native_stand.students.interface import ACTOR_DIM
    from native_stand.students.mcu_interface import ACTOR_DIM as MCU_DIM
    assert ACTOR_DIM==1182 and MCU_DIM==31
    assert Student(version='multiskill_mcu_v2').mean(torch.zeros(2,31)).shape==(2,6)
    assert Student(version='multiskill_student_v1').mean(torch.zeros(2,1182)).shape==(2,6)
    from native_stand.command_v6.domain.contract import frame,history
    f=frame(torch.zeros(2,3),torch.tensor([[0.,-1.,0.]]).expand(2,-1),torch.zeros(2,3),
            torch.zeros(2,6),torch.zeros(2,6),torch.zeros(2,6),torch.zeros(2,6))
    assert f.shape==(2,25) and history(torch.zeros(2,125),f,torch.zeros(2,dtype=torch.bool)).shape==(2,125)


def test_independent_authorization_and_exact_counter_limits(tmp_path,monkeypatch):
    from native_stand.world_model import experiment
    from native_stand.world_model import VERSION,STATE_VERSION
    from native_stand.world_model.real import sha
    monkeypatch.setattr(experiment,'ROOT',tmp_path);monkeypatch.setenv('HAPI_SESSION_ID','fixture-owner')
    folder=tmp_path/'configs'/'fixture';folder.mkdir(parents=True)
    spec=dict(version=VERSION,state_version=STATE_VERSION,experiment='fixture',run=str(tmp_path/'runs'/'fixture'),
        source_inputs={},code_inputs={},limits={**{k:2 for k in experiment.COUNTERS},'ppo_updates':0,
            'wall_s':60,'artifacts_bytes':1000000,'min_free_bytes':0},
        authorization=dict(status='UNAUTHORIZED',owner_session='fixture-owner',scope=None))
    path=folder/'spec.json';path.write_text(json.dumps(spec))
    book=dict(spec_sha256=sha(path),authorization='UNAUTHORIZED',limits=spec['limits'],
        used={k:0 for k in experiment.COUNTERS},wall_used_s=0.,attempts={})
    (folder/'budget.json').write_text(json.dumps(book));(folder/'manifest.json').write_text('{}')
    with pytest.raises(PermissionError,match='authorization missing'):
        experiment.Budget(path,'fixture',tmp_path/'runs'/'fixture'/'a')
    experiment.authorize_prepared(path,'CPU fixture only, no robot data or physics')
    b=experiment.Budget(path,'fixture',tmp_path/'runs'/'fixture'/'a')
    b.charge('wm_optimizer_steps',1);b.charge('actor_optimizer_steps',1)
    with pytest.raises(RuntimeError,match='ppo_updates'):b.charge('ppo_updates',1)
    b.finalize('COMPLETED')
    b2=experiment.Budget(path,'fixture2',tmp_path/'runs'/'fixture'/'b')
    with pytest.raises(RuntimeError,match='wm_optimizer_steps'):b2.ensure('wm_optimizer_steps',2)
    b2.finalize('COMPLETED')
    data=json.loads((folder/'budget.json').read_text())
    assert data['used']['wm_optimizer_steps']==1 and data['used']['actor_optimizer_steps']==1
    assert data['used']['critic_optimizer_steps']==0 and data['used']['ppo_updates']==0


def test_world_fit_critic_and_actor_cli_functions_on_named_fixture_replay(tmp_path):
    """Executes the actual fit functions on artificial tensors, never robot data."""
    from native_stand.world_model.dataset import Replay,write_episode,build_manifest
    from native_stand.world_model.training import fit_world,fit_critic,fit_actor
    from native_stand.world_model.owner import load_world,load_critic
    class FixtureBudget:
        def __init__(self):self.local={}
        def check(self):pass
        def storage_guard(self,n):assert n>0
        def charge(self,k,n):self.local[k]=self.local.get(k,0)+n
    episode_paths=[]
    s=fixture(1)
    for split in ('train','validation'):
        rows=[];micro=[];blobs=[]
        for i in range(40):
            b=replace(s,tick=torch.tensor([i*10]),episode_step=torch.tensor([i]),owner=torch.tensor([1]))
            a=replace(b,tick=b.tick+10,episode_step=b.episode_step+1)
            rows.append(dict(before=b,after=a,action=torch.zeros(1,6),observation=torch.zeros(1,31),
                reward=torch.ones(1)*.01,terminated=torch.tensor([False]),truncated=torch.tensor([i==39]),
                reward_raw=torch.ones(1,1),reward_density=torch.ones(1,1),reward_dt=torch.ones(1,1)*.01,
                reward_fields=['synthetic']))
            micro.extend([dict(q=s.q[0].numpy(),v=s.v[0].numpy(),warm=s.warm[0].numpy(),applied=s.generalized[0].numpy(),
                next_q=s.q[0].numpy(),next_v=s.v[0].numpy(),next_warm=s.warm[0].numpy(),contact=s.contact[0].numpy(),
                body_contact=s.body_contact[0].numpy())]*10)
            blobs.append(np.zeros(2))
        path=tmp_path/split
        write_episode(path,dict(id=split,split=split,tuple_id=split,controller='teacher4500'),rows,micro,blobs,{'synthetic':True})
        episode_paths.append(path)
    manifest=build_manifest(episode_paths,tmp_path/'dataset.json',{'fixture':'NO_ROBOT_DATA'})
    train=Replay(manifest,'train');val=Replay(manifest,'validation')
    budget=FixtureBudget()
    world=fit_world(train,val,TinyGeometry(),budget,tmp_path/'wm',steps=3,batch_size=2,width=16,members=3)
    assert budget.local['wm_optimizer_steps']==3
    wm=load_world(world['checkpoint'],'cpu')
    from native_stand.world_model.validation import predict_report
    prediction=predict_report(val,wm,TinyGeometry(),TinyReward(),budget,samples=1)
    assert prediction['single_step_1ms']['count']==1
    assert set(prediction['free_rollout'])=={'8','16','32'}
    assert all(row['count']==3 for row in prediction['free_rollout'].values())
    critic=fit_critic(train,budget,tmp_path/'critic',steps=2,batch_size=2,width=16)
    assert budget.local['critic_optimizer_steps']==2
    value=load_critic(critic['checkpoint'],'cpu')
    actor=Actor();actor.network_architecture={'version':'SYNTHETIC_FIXTURE_ONLY'}
    teacher=copy.deepcopy(actor)
    result=fit_actor(train,actor,teacher,TinyMechanics(),value,TinyGeometry(),TinyReward(),budget,
                     tmp_path/'actor',steps=1,batch_size=1)
    assert budget.local['actor_optimizer_steps']==1
    assert Path(result['checkpoint']).is_file() and wm.config['members']==3
    longer=fit_world(train,val,TinyGeometry(),budget,tmp_path/'wm80',steps=1,batch_size=2,width=16,members=3,
        previous=world['checkpoint'],fit_window_substeps=80,activation_checkpoint=True,training_revision='supervised_window80_v2')
    assert budget.local['wm_optimizer_steps']==4
    saved=torch.load(longer['checkpoint'],weights_only=False)
    assert saved['training_recipe']['fit_window_substeps']==80 and longer['history'][0]['step']==0
    val.cases[0]['handover_ms']=200
    targeted=predict_report(val,wm,TinyGeometry(),TinyReward(),budget,samples=0,horizons=(8,))
    assert targeted['by_regime']['handover']['8']['count']>0
    assert targeted['sampling']['extra_handover_starts']==2 and targeted['sampling']['random_starts']==0
    focused=fit_world(train,val,TinyGeometry(),budget,tmp_path/'wm_contact',steps=1,batch_size=4,width=16,members=3,
        previous=longer['checkpoint'],fit_window_substeps=80,activation_checkpoint=True,
        training_revision='supervised_window80_contact25_v4',sampling='contact_event25_v1')
    assert budget.local['wm_optimizer_steps']==5
    assert focused['training_recipe']['sampling']=='contact_event25_v1'
    assert focused['training_recipe']['validation_sampling']=='uniform_episode_v1'
    assert all(row['extra_event_samples_per_batch']==1 for row in focused['training_recipe']['event_coverage'])
    saved=torch.load(focused['checkpoint'],weights_only=False)
    assert saved['best_local_optimizer_step'] in (0,1) and saved['steps']==1


def test_unlimited_budget_still_accounts_and_refuses_ppo(tmp_path,monkeypatch):
    import native_stand.world_model.experiment as e
    spec=dict(run=str(tmp_path/'run'),limits={k:None for k in e.COUNTERS})
    spec['limits'].update(ppo_updates=0,wall_s=None,artifacts_bytes=10**6,min_free_bytes=0)
    book=dict(authorization='AUTHORIZED',used={k:0 for k in e.COUNTERS},attempts={},wall_used_s=0.)
    path=tmp_path/'spec.json';path.write_text('{}');path.with_name('budget.json').write_text(json.dumps(book))
    monkeypatch.setattr(e,'verify',lambda *a,**k:(spec,book))
    budget=e.Budget(path,'engineering',tmp_path/'run'/'engineering')
    assert budget.deadline==float('inf')
    budget.charge('physics_steps',1234);budget.charge('wm_optimizer_steps',3)
    with pytest.raises(RuntimeError,match='ppo_updates'):budget.charge('ppo_updates',1)
    budget.finalize('COMPLETED')
    data=json.loads(path.with_name('budget.json').read_text())
    assert data['used']['physics_steps']==1234 and data['used']['wm_optimizer_steps']==3
    assert data['used']['ppo_updates']==0 and data['wall_used_s']>0


def test_cycle_prediction_failure_stops_before_critic_actor_and_physics_eval(tmp_path,monkeypatch):
    import native_stand.world_model.cycle as cycle
    spec=dict(run=str(tmp_path),workflow=dict(rounds_max=None,initial=dict(wm_optimizer_steps=3000)))
    monkeypatch.setattr(cycle,'verify',lambda *a,**k:(spec,{}))
    seen=[]
    def fake(request):
        seen.append(request['operation']);out=Path(request['output']);out.mkdir(parents=True)
        result=dict(passed=request['operation']!='predict',checkpoint=str(out/'fixture.pt'))
        (out/'result.json').write_text(json.dumps(result));return result
    result=cycle.execute(dict(spec='fixture',output=str(tmp_path/'automatic_chain')),execute_stage=fake)
    assert seen==['engineering','collect','build-data','train-world','predict']
    assert result['reason']=='PREDICTION_GATE' and result['actor'] is None


def test_real_critic_filters_exact_current_actor_checkpoint():
    from native_stand.world_model.dataset import Replay
    replay=object.__new__(Replay);replay.split='train'
    replay.cases=[dict(controller='candidate',controller_sha256=x) for x in ('old','current')]
    replay.rows=[0,1];replay.data=['old data','current data']
    selected=replay.for_controllers(['candidate'],'current')
    assert selected.data==['current data'] and replay.data==['old data','current data']


def test_cycle_terminal_reuses_at_most_once_owner_notification(tmp_path):
    import time
    from native_stand.world_model.cycle import terminal
    from native_stand.world_model.experiment import COUNTERS
    from native_stand.command_v6.completion import notify
    path=tmp_path/'spec.json';session='explicit-fixture-session'
    path.write_text(json.dumps(dict(run=str(tmp_path),notifications=dict(enabled=False))))
    path.with_name('budget.json').write_text(json.dumps(dict(used={k:0 for k in COUNTERS},attempts={})))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(hapi_session=session,
        notification_route=dict(owner_session=session,target_session=session))))
    terminal(dict(spec=str(path),output=str(tmp_path/'automatic_chain')),None,'fixture failure',time.monotonic())
    sent=[]
    def send(owner,message):sent.append((owner,message));return dict(status='sent')
    first=notify(tmp_path,send=send);second=notify(tmp_path,send=send)
    assert first==second and len(sent)==1 and sent[0][0]==session
    assert 'FAILED' in sent[0][1] and 'completion_report.md' in sent[0][1]


def test_checkpointed_long_supervision_keeps_loss_and_all_parameter_gradients():
    from native_stand.world_model.training import supervised_loss
    torch.manual_seed(821)
    s=fixture(2);wm=Ensemble(13,12,3,members=1,width=16,blocks=1);geometry=TinyGeometry()
    row=dict(q=s.q,v=s.v,warm=s.warm,applied=s.generalized,next_q=s.q,next_v=s.v,
             next_warm=s.warm,contact=s.contact,body_contact=s.body_contact,context=s.context)
    batch={k:v if k=='context' else v[:,None].expand(-1,80,-1) for k,v in row.items()}
    reference,detail=supervised_loss(wm,0,batch,geometry);reference.backward()
    expected=[p.grad.clone() for p in wm.members[0].parameters()]
    wm.zero_grad()
    actual,measured=supervised_loss(wm,0,batch,geometry,activation_checkpoint=True);actual.backward()
    torch.testing.assert_close(actual,reference)
    assert measured==pytest.approx(detail)
    for p,want in zip(wm.members[0].parameters(),expected):
        torch.testing.assert_close(p.grad,want)
        assert torch.isfinite(p.grad).all()


def test_continuation_requires_declared_world_and_no_duplicate_collection(tmp_path):
    from native_stand.world_model.experiment import validate_request
    spec=dict(run=str(tmp_path),workflow=dict(rounds_max=None,initial=dict(wm_optimizer_steps=1000)),
              prediction_continuation=dict(world='/verified/old.pt'))
    book=dict(attempts={})
    request=dict(operation='train-world',world='/verified/old.pt',steps=1000,output=str(tmp_path/'wm'))
    assert validate_request(request,spec,book,files=False)
    with pytest.raises(ValueError,match='declared exact World Model'):
        validate_request(dict(request,world='/unrelated.pt'),spec,book,files=False)
    with pytest.raises(ValueError,match='initial collection'):
        validate_request(dict(operation='collect',collection='initial',output=str(tmp_path/'data')),spec,book,files=False)


def test_continuation_cycle_reuses_evidence_and_does_not_resample(tmp_path,monkeypatch):
    import native_stand.world_model.cycle as cycle
    spec=dict(run=str(tmp_path),workflow=dict(rounds_max=None,initial=dict(wm_optimizer_steps=1000)),
              prediction_continuation=dict(world='old.pt',dataset='old_data.json',collection_root='old_data'))
    monkeypatch.setattr(cycle,'verify',lambda *a,**k:(spec,{}))
    seen=[]
    def fake(request):
        seen.append(request);out=Path(request['output']);out.mkdir(parents=True)
        result=dict(passed=request['operation']!='predict',checkpoint=str(out/'fixture.pt'))
        (out/'result.json').write_text(json.dumps(result));return result
    result=cycle.execute(dict(spec='fixture',output=str(tmp_path/'automatic_chain')),execute_stage=fake)
    assert [r['operation'] for r in seen]==['reuse-data','train-world','predict']
    assert seen[1]['world']=='old.pt' and seen[1]['dataset']=='old_data.json'
    assert result['reason']=='PREDICTION_GATE' and result['actor'] is None


@pytest.mark.parametrize('operation',('collect','engineering','train-actor','train-critic','gradient','evaluate','export'))
def test_prediction_only_declaration_cannot_dispatch_policy_or_physics(tmp_path,operation):
    from native_stand.world_model.experiment import validate_request
    spec=dict(run=str(tmp_path),workflow=dict(prediction_only=True))
    with pytest.raises(ValueError,match='prediction-only block'):
        validate_request(dict(operation=operation,output=str(tmp_path/'stage')),spec,dict(attempts={}),files=False)


@pytest.mark.parametrize('contact_focus',(False,True))
def test_second_generation_continuation_keeps_ancestor_identity_counts_and_stops_after_prediction(tmp_path,monkeypatch,contact_focus):
    import native_stand.world_model.experiment as e
    import native_stand.world_model.cycle as cycle
    import native_stand.command_v6.domain.experiment_naming as naming
    from native_stand.world_model import VERSION,STATE_VERSION
    from native_stand.world_model.real import sha
    monkeypatch.setattr(e,'ROOT',tmp_path);monkeypatch.setattr(e,'code_identities',lambda:{})
    monkeypatch.setenv('HAPI_SESSION_ID','fixture-owner')
    monkeypatch.setattr(naming,'new_experiment_name',lambda suffix:'2026_10_28-00_00__'+suffix)
    folder=tmp_path/'configs/experiments/parent';folder.mkdir(parents=True)
    run=tmp_path/'runs/parent';(run/'round_000/world').mkdir(parents=True)
    (run/'round_000/prediction').mkdir()
    source=tmp_path/'frozen_source';source.write_text('SYNTHETIC_NO_PHYSICS')
    ancestor=tmp_path/'settled_ancestor';ancestor.write_text('ANCESTOR_READ_ONLY')
    world=run/'best_world.pt';world.write_text('SYNTHETIC_CHECKPOINT_IDENTITY_ONLY')
    if contact_focus:
        tiny=Ensemble(13,12,3,members=3,width=16,blocks=1)
        torch.save(dict(config=tiny.config,model=tiny.state_dict()),world)
    dataset=tmp_path/'dataset.json';dataset.write_text(json.dumps(dict(episodes=[])))
    engineering=tmp_path/'engineering.json'
    engineering.write_text(json.dumps(dict(passed=True,spec_sha256=sha(ancestor),physics_steps=1500)))
    used={k:0 for k in e.COUNTERS};used['wm_optimizer_steps']=1000
    historical=dict(used,wm_optimizer_steps=4000 if contact_focus else 3000,physics_steps=497020,episodes=49)
    limits={k:None for k in e.COUNTERS};limits.update(ppo_updates=0,artifacts_bytes=20*1024**2 if contact_focus else 2*1024**3,wall_s=None)
    spec=dict(version=VERSION,state_version=STATE_VERSION,experiment='parent',run=str(run),
        source_inputs={str(source):sha(source)},code_inputs={},resume_inputs={str(ancestor):sha(ancestor)},limits=limits,
        authorization=dict(status='AUTHORIZED',unlimited_updates_and_wall=True,owner_session='fixture-owner',scope='fixture'),
        optimization_version='wheel_leg_shared_dynamics_bptt_window80_continue_v3' if contact_focus else 'wheel_leg_shared_dynamics_bptt_window80_v2',
        world_model=dict(fit_window_substeps=80),
        prediction_continuation=dict(dataset=str(dataset),collection_root=str(tmp_path/'source_collect'),engineering=str(engineering)),
        workflow=dict(rounds_max=None,automatic_cycle=True,initial=dict(wm_optimizer_steps=1000),physical_budget_breakdown={}))
    path=folder/'spec.json';path.write_text(json.dumps(spec))
    ledger=folder/'budget.json'
    ledger.write_text(json.dumps(dict(spec_sha256=sha(path),limits=limits,used=used,historical_used=historical,
        wall_used_s=123.,historical_wall_s=456.,historical_artifacts_bytes=789,
        attempts={'settled':dict(status='COMPLETED')})))
    (run/'completion_report.md').write_text('SYNTHETIC_STOPPED_GATE')
    (run/'report_ready.json').write_text(json.dumps(dict(event_id='fixture-event',report_sha256=sha(run/'completion_report.md'))))
    (run/'analysis.json').write_text(json.dumps(dict(status='STOPPED_GATE',reason='PREDICTION_GATE',error=None,
        checkpoints=dict(world=dict(checkpoint=str(world),sha256=sha(world))))))
    (run/'round_000/world/result.json').write_text(json.dumps(dict(history=[dict(step=0,validation_loss=.5),dict(step=1000,validation_loss=.3)])))
    (run/'round_000/prediction/result.json').write_text(json.dumps(dict(passed=False)))
    review=tmp_path/'diagnosis.json'
    review.write_text(json.dumps(dict(physics_steps=0,optimizer_steps=0,models=dict(window80=dict(sha256=sha(world))),
        reviewed_parent_spec_sha256=sha(path),dataset_sha256=sha(dataset),actual_initial_invalid=dict(closure=0))))
    coverage=tmp_path/'contact_coverage.json'
    coverage.write_text(json.dumps(dict(physics_steps=0,optimizer_steps=0,dataset_sha256=sha(dataset),
        actual_selected_substeps=dict(invalid_contact_substeps=0))))
    if contact_focus:
        review.write_text(json.dumps(dict(physics_steps=0,optimizer_steps=0,world_sha256=sha(world),dataset_sha256=sha(dataset),faults=[dict(reasons=['contact_geometry'])])))
        audit=tmp_path/'sampling_audit.json'
        audit.write_text(json.dumps(dict(physics_steps=0,optimizer_steps=0,world_sha256=sha(world),
            dataset_sha256=sha(dataset),parent_spec_sha256=sha(path),gradient_finite=True,gradient_norm=1.,
            all_members_have_events=True,world_config=tiny.config)))
    originals={str(p):sha(p) for p in (path,ledger,world,dataset,engineering,ancestor,source)}
    parent_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    child=e.prepare_prediction_continuation(path,review,continue_window80=not contact_focus,contact_focus=contact_focus)
    s,b=e.verify(child,authorization=True)
    assert b['historical_used']['wm_optimizer_steps']==(5000 if contact_focus else 4000)
    assert b['historical_used']['physics_steps']==497020 and b['historical_used']['episodes']==49
    assert b['historical_wall_s']==579. and b['historical_artifacts_bytes']==789+parent_bytes
    assert s['limits']['artifacts_bytes']+parent_bytes==limits['artifacts_bytes']
    assert s['limits']['physics_steps']==s['limits']['actor_optimizer_steps']==s['limits']['critic_optimizer_steps']==0
    assert s['limits']['wm_optimizer_steps']==1000 and s['workflow']['prediction_only']
    assert s['prediction_continuation']['engineering_spec_sha256']==sha(ancestor)!=sha(path)
    assert s['resume_inputs'][str(ancestor)]==sha(ancestor) and s['world_model']['seed']==(2017 if contact_focus else 1017)
    if contact_focus:
        assert s['world_model']['sampling']=='contact_event25_v1'
        assert s['prediction_continuation']['measured_save_guard_bytes']<s['limits']['artifacts_bytes']<1024**3
    assert all(sha(p)==h for p,h in originals.items())
    seen=[]
    def fake(request):
        seen.append(request);out=Path(request['output']);out.mkdir(parents=True)
        result=dict(passed=True,checkpoint=str(out/'fixture.pt'))
        (out/'result.json').write_text(json.dumps(result));return result
    result=cycle.execute(dict(spec=str(child),output=str(Path(s['run'])/'automatic_chain')),execute_stage=fake)
    assert [r['operation'] for r in seen]==['reuse-data','train-world','predict']
    assert result['status']=='COMPLETED_BLOCKS' and result['prediction_only'] and result['actor'] is None
    if contact_focus:
        bad=json.loads(audit.read_text());bad['gradient_finite']=False;audit.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='finite offline gradient'):
            e.prepare_prediction_continuation(path,review,contact_focus=True)
    else:
        bad=json.loads(coverage.read_text());bad['actual_selected_substeps']['invalid_contact_substeps']=1
        coverage.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='real contact labels'):
            e.prepare_prediction_continuation(path,review,continue_window80=True)
    assert all(sha(p)==h for p,h in originals.items())


def test_contact_event_labels_exclude_initialization_and_require_complete_future():
    from native_stand.world_model.dataset import contact_event_starts
    contact=np.full((400,2),20.)
    contact[:40]=0  # initial settling remains in the uniform quota
    contact[130:134,0]=0  #four1ms samples do not define sustained loss
    contact[200:206,1]=0
    contact[360:]=0  # insufficient80ms future cannot supply a supervised start
    starts=contact_event_starts(contact,80)
    np.testing.assert_array_equal(starts,np.arange(180,226))
    assert starts.min()>=100 and starts.max()+80<=len(contact)
    contact[20,0]=np.nan
    with pytest.raises(ValueError,match='finite nonnegative'):contact_event_starts(contact,80)


def test_contact_sampler_keeps_exact_uniform_default_and_member_bootstrap():
    from native_stand.world_model.dataset import mechanics_indices,contact_event_starts
    data=[]
    for i in range(3):
        force=np.full((400,2),20.)
        if i:force[200:240]=0
        data.append(dict(micro_q=np.zeros((400,13)),micro_contact=force))
    pool=[0,0,1];rng=np.random.default_rng(44)
    ep=[int(rng.choice(pool)) for _ in range(256)]
    ids=[int(rng.integers(321)) for _ in ep];next_value=int(rng.integers(10000))
    rng=np.random.default_rng(44)
    actual_e,actual_i,count=mechanics_indices(data,256,rng,episode_pool=pool,window=80)
    assert actual_e==ep and actual_i==ids and count==0 and int(rng.integers(10000))==next_value
    e,i,count=mechanics_indices(data,256,np.random.default_rng(44),episode_pool=pool,window=80,sampling='contact_event25_v1')
    assert count==64 and len(e)==256 and len(i)==256
    assert all(x==1 for x in e[-64:]) and 2 not in e  #excluded bootstrap episode stays excluded
    assert all(x in contact_event_starts(data[1]['micro_contact'],80) for x in i[-64:])
    assert set(e[:192])=={0,1}
    with pytest.raises(ValueError,match='no train contact events'):
        mechanics_indices(data,256,np.random.default_rng(44),episode_pool=[0],window=80,sampling='contact_event25_v1')


@pytest.mark.parametrize('split',('validation','test'))
def test_contact_sampler_never_changes_held_out_distribution(split):
    from native_stand.world_model.dataset import Replay
    r=Replay.__new__(Replay);r.split=split
    with pytest.raises(ValueError,match='train-only'):
        r.mechanics(256,np.random.default_rng(1),window=80,sampling='contact_event25_v1')


def closure_test_geometry():
    """Two synthetic arms with loop sites, plus unneeded rotating wheel leaves."""
    from native_stand.world_model.geometry import Kinematics
    n=8;unit=[1.,0.,0.,0.]
    return Kinematics(dict(parent=[0,0,1,2,3,1,5,6],root=1,joint_q=[-1,-1,7,8,9,10,11,12],
        joint_v=[-1,-1,6,7,8,9,10,11],pos=[[0,0,0],[0,0,0],[0,.15,0],[0,0,-.1],[0,0,-.1],
            [0,-.15,0],[0,0,-.1],[0,0,-.1]],quat=[unit]*n,axis=[[1.,0,0]]*n,
        anchor=[[0,.01,.02]]*n,ipos=[[0,.002,.01]]*n,iquat=[unit]*n,mass=[0]+[1.]*(n-1),
        inertia=[[.01,.02,.03]]*n,q0=[0,0,.34,1,0,0,0]+[0]*6,
        site_pos=[[0,0,-.03],[0,0,.02],[.01,0,-.04],[0,.02,.01]],site_body=[2,3,5,6],
        geom_pos=[[0,0,0]]*2,geom_quat=[unit]*2,geom_size=[[.05,.01,.0]]*2,
        geom_body=[4,7],contact_geoms=[0,1],loops=[[0,1],[2,3]],
        wheels=[4,7],hips=[2,5],armature=[.01]*12,limits=[[-2,2]]*6,
        all_joint_limits=[[-2,2]]*6,all_joint_limited=[True]*6,solver_actuator_limits=[40]*6,
        active_q=list(range(7,13)),active_v=list(range(6,12))))


def test_closure_only_fk_matches_full_geometry_and_coordinate_gradients():
    from native_stand.world_model.geometry import ClosureOnlyKinematics
    torch.manual_seed(792);g=closure_test_geometry().double();fast=ClosureOnlyKinematics(g)
    s=fixture(7);q=s.q.double().clone();q[:,7:]=torch.randn(7,6,dtype=torch.double)*.2
    q[:,3:7]=quat_exp(torch.randn(7,3,dtype=torch.double)*.2);q.requires_grad_(True)
    v=torch.randn(7,12,dtype=torch.double,requires_grad=True)
    reference=g(q,v)['closure'];actual=fast(q,v)['closure']
    torch.testing.assert_close(reference,actual,atol=0,rtol=0)
    weights=torch.randn_like(actual)
    first=torch.autograd.grad((reference*weights).sum(),q,retain_graph=True)[0]
    second=torch.autograd.grad((actual*weights).sum(),q)[0]
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    assert len(fast.bodies)==5 and 4 not in fast.bodies and 7 not in fast.bodies
    assert torch.count_nonzero(second[:,[9,12]])==0
    assert torch.autograd.gradcheck(lambda x:fast(x,v)['closure'],(q,),atol=1e-6)
    # Full acceptance still exposes energy/penetration/contact, no fast FK shortcut.
    assert {'energy','penetration','wheel_ground_gap'}<=set(g(q,v))


def test_fast_supervision_preserves_multistep_loss_parameter_gradient_and_adam_step():
    from native_stand.world_model.geometry import ClosureOnlyKinematics
    from native_stand.world_model.training import supervised_loss
    torch.manual_seed(790);s=fixture(4);g=closure_test_geometry()
    source=Ensemble(13,12,3,members=1,width=16,blocks=1)
    a=copy.deepcopy(source);b=copy.deepcopy(source)
    row=dict(q=s.q,v=s.v,warm=s.warm,applied=s.generalized,next_q=s.q,next_v=s.v,
        next_warm=s.warm,contact=s.contact,body_contact=s.body_contact,context=s.context)
    batch={k:v if k=='context' else v[:,None].expand(-1,80,-1) for k,v in row.items()}
    oa=torch.optim.Adam(a.parameters(),lr=1e-4);ob=torch.optim.Adam(b.parameters(),lr=1e-4)
    loss_a,detail_a=supervised_loss(a,0,batch,g,activation_checkpoint=True);loss_a.backward()
    loss_b,detail_b=supervised_loss(b,0,batch,ClosureOnlyKinematics(g),activation_checkpoint=False);loss_b.backward()
    torch.testing.assert_close(loss_a,loss_b,atol=0,rtol=0)
    assert detail_a==detail_b
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
    oa.step();ob.step()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)


def test_supervision_performance_is_explicit_and_preserves_algorithm_configuration():
    from native_stand.world_model.experiment import supervision_performance
    original=dict(fit_window_substeps=80,width=1024,members=3,context_dim=2888,sampling='contact_event25_v1',
        lr=1e-4,batch_size=256,activation_checkpoint=True)
    assert supervision_performance(original)==original
    fast=supervision_performance(original,'closure_no_checkpoint_v1')
    assert fast['activation_checkpoint'] is False and fast['supervision_geometry']=='closure_fk_v1'
    assert fast['performance_version']=='closure_no_checkpoint_v1' and original['activation_checkpoint'] is True
    assert {k:fast[k] for k in original if k!='activation_checkpoint'}=={k:v for k,v in original.items() if k!='activation_checkpoint'}
    with pytest.raises(ValueError,match='unknown supervision'):supervision_performance(original,'smaller_network')
