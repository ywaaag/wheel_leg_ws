"""Bounded train functions. Runtime budget callbacks count distinct optimizers."""
from dataclasses import replace
from pathlib import Path
import copy
import hashlib
import sys
import numpy as np
import torch
from .models import Ensemble, Critic, mechanical_features, value_features, mechanics_targets, contact_force_config, force_supervision
from .geometry import integrate, rotation_error,ClosureOnlyKinematics
from .learner import critic_step, actor_step


def save_checkpoint(path, payload, budget=None):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    if path.exists(): raise FileExistsError(path)
    if budget is not None:
        def size(value):
            if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
            if isinstance(value,dict):return sum(size(v) for v in value.values())
            if isinstance(value,(tuple,list)):return sum(size(v) for v in value)
            return 0
        budget.storage_guard(int(size(payload)*1.1)+1024**2)
    torch.save(payload,temporary)
    temporary.replace(path)


def resume_state(optimizer_steps,pools,rng):
    """Recorded continuation bookkeeping. `fit_world` does NOT consume it yet.

    Saving it stops future documentation from claiming an exact replay-RNG restore
    that the code cannot perform: only models/Adam/episode bootstrap were stored before.
    """
    return dict(optimizer_steps=int(optimizer_steps),next_member=int(optimizer_steps)%len(pools),
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state().clone(),
                episode_pools=[list(pool) for pool in pools],
                consumed_by='NOT_CONSUMED: fit_world still expects an explicit new seed per block',
                restored_scope='models,Adam,normalizer,episode bootstrap; sampler stream not reconstructed')


def save_last(path,payload,budget):
    """Preserve the original training failure if emergency persistence also fails."""
    original=sys.exc_info()[1]
    try:save_checkpoint(path,payload,budget)
    except Exception as error:
        if original is None:raise
        print('checkpoint_save_failure: '+repr(error),file=sys.stderr,flush=True)


def supervised_loss(wm, member, batch, geometry, *, activation_checkpoint=False, contact_force=None):
    if batch['q'].ndim==3:
        # Autoregressive mechanics windows of declared consecutive1ms transitions. Applied
        # forces stay recorded labels; predicted q/v/warm are fed back without detach.
        losses=[];details=[];predicted=None
        for k in range(batch['q'].shape[1]):
            row={name:(value if name=='context' else value[:,k]) for name,value in batch.items()}
            if predicted is not None:
                row.update(q=predicted[0],v=predicted[1],warm=predicted[2])
            if activation_checkpoint and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                loss,detail,predicted=checkpoint(lambda values:_single_loss(wm,member,values,geometry,contact_force),
                    row,use_reentrant=False,preserve_rng_state=False)
            else:loss,detail,predicted=_single_loss(wm,member,row,geometry,contact_force)
            losses.append(loss);details.append(detail)
        return torch.stack(losses).mean(),{key:float(torch.stack([r[key] for r in details]).mean()) for key in details[0]}
    loss,detail,_=_single_loss(wm,member,batch,geometry,contact_force)
    return loss,{k:float(v) for k,v in detail.items()}


def _single_loss(wm,member,batch,geometry,contact_force=None):
    features=mechanical_features(batch['q'],batch['v'],batch['warm'],batch['applied'],batch['context'])
    predicted=wm.raw(features,member)
    target=mechanics_targets(batch['next_v'],batch['v'],batch['next_warm'],batch['contact'],batch.get('body_contact'))
    # Every output keeps a named physical target; normalization only sets loss scales.
    normalized=wm.target(target)
    error=predicted-normalized
    value=predicted*wm.target.scale+wm.target.mean
    v=batch['v']+value[:,:wm.nv]
    q=integrate(batch['q'],v)
    measured=geometry(q,v)
    closure=(measured['closure']/.001).square().mean()
    # Rotation integrated on SO(3), passive coordinates explicitly supervised too.
    qloss=(q[:,7:]-batch['next_q'][:,7:]).square().mean()
    rloss=rotation_error(q[:,3:7],batch['next_q'][:,3:7]).mean()
    base=torch.nn.functional.smooth_l1_loss(error,torch.zeros_like(error))+closure*.01+qloss+rloss
    detail=dict(normalized_error=error.detach().square().mean(),
                closure_rms_m=measured['closure'].detach().square().mean().sqrt())
    loss=base
    if contact_force is not None:
        extra,diagnostics=force_supervision(value,batch['contact'],wm.nv,contact_force)
        loss=base+contact_force['lambda_force']*extra
        detail.update(diagnostics)
        detail['contact_force_base_loss']=base.detach()
        detail['contact_force_weighted']=contact_force['lambda_force']*extra.detach()
    return loss,detail,(q,v,value[:,wm.nv:2*wm.nv])


def fit_world(train, validation, geometry, budget, output, *, steps=3000, seed=17,
              width=1024, blocks=2, members=3, batch_size=256, previous=None, device='cpu',provenance=None,
              fit_window_substeps=8,activation_checkpoint=False,training_revision='supervised_window8_v1',
              sampling='uniform_episode_v1',supervision_geometry='full_fk_v1',contact_force=None):
    if train.split!='train' or validation.split!='validation':
        raise ValueError('train/validation episodes must remain separate')
    if fit_window_substeps not in (8,80):raise ValueError('explicit supervised substep window must be8 or80')
    if supervision_geometry not in ('full_fk_v1','closure_fk_v1'):raise ValueError('unknown supervised FK version')
    contact_force=contact_force_config(contact_force)  # explicit; never inferred from shapes
    from .dataset import UNIFORM_MECHANICS,CONTACT_MECHANICS,contact_event_starts
    if sampling not in (UNIFORM_MECHANICS,CONTACT_MECHANICS):raise ValueError('unknown mechanics sampling recipe')
    if sampling==CONTACT_MECHANICS and (fit_window_substeps!=80 or batch_size<4 or previous is None):
        raise ValueError('contact25 continuation requires prior WM,80ms and batch>=4')
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    geometry=geometry.to(device)
    loss_geometry=ClosureOnlyKinematics(geometry) if supervision_geometry=='closure_fk_v1' else geometry
    sample=train.mechanics(min(8192,batch_size*16),rng,device=device)
    if previous is None:
        wm=Ensemble(sample['q'].shape[-1],sample['v'].shape[-1],sample['context'].shape[-1],
                    width=width,blocks=blocks,members=members).to(device)
        wm.normalizer.fit(mechanical_features(sample['q'],sample['v'],sample['warm'],sample['applied'],sample['context']))
        wm.target.fit(mechanics_targets(sample['next_v'],sample['v'],sample['next_warm'],sample['contact'],sample['body_contact']))
    else:
        saved=torch.load(previous,map_location=device,weights_only=False)
        wm=Ensemble(**saved['config']).to(device);wm.load_state_dict(saved['model'])
    optimizers=[torch.optim.Adam(m.parameters(),lr=1e-4) for m in wm.members]
    # Bootstrap WHOLE EPISODES per member; adjacent frames are not held out.
    pools=[rng.integers(len(train.data),size=len(train.data)).tolist() for _ in wm.members]
    if previous is not None and 'optimizers' in saved:
        for optimizer,state in zip(optimizers,saved['optimizers']): optimizer.load_state_dict(state)
    if previous is not None and 'ensemble_episode_bootstrap' in saved:
        previous_pools=saved['ensemble_episode_bootstrap']
        if len(previous_pools)==members and all(len(pool)==len(train.data) for pool in previous_pools):pools=previous_pools
    coverage=[]
    if sampling==CONTACT_MECHANICS:
        counts=[len(contact_event_starts(z['micro_contact'],fit_window_substeps)) for z in train.data]
        for member,pool in enumerate(pools):
            event_pool=[e for e in pool if counts[e]]
            if not event_pool:raise ValueError('bootstrap member has no train contact-event coverage')
            coverage.append(dict(member=member,extra_event_samples_per_batch=batch_size//4,
                eligible_episodes={train.rows[e]['episode_id']:dict(starts=counts[e],multiplicity=pool.count(e)) for e in set(event_pool)}))
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    budget.storage_guard(sum(p.numel()*p.element_size() for p in wm.parameters())*7+2*1024**2)
    best=float('inf'); history=[]; checkpoint=None;best_optimizers=None;best_step=0
    recipe=dict(fit_window_substeps=fit_window_substeps,activation_checkpoint=activation_checkpoint,
                training_revision=training_revision,effective_batch_size=batch_size,seed=seed,
                replay_rng='new explicit seed per block; historical checkpoints contain no replay RNG state',
                sampling=sampling,event_coverage=coverage,validation_sampling=UNIFORM_MECHANICS,
                supervision_geometry=supervision_geometry,physics_and_prediction_geometry='full_fk_unchanged',
                contact_force_supervision=contact_force,
                loss_definition=('smooth_l1(normalized dv/warm/contact)+0.01 closure+active q SO(3) rotation'
                                 +(' + lambda_force*huber((softplus(wheel head)-F_real)/F_scale)' if contact_force else '')),
                validation_loss_scope='held-out episode loss including the declared contact force term when enabled')
    def validation_score():
        with torch.no_grad():
            val=validation.mechanics(min(512,batch_size*2),np.random.default_rng(991),device=device,window=fit_window_substeps)
            return float(np.mean([float(supervised_loss(wm,j,val,loss_geometry,contact_force=contact_force)[0])
                                  for j in range(len(wm.members))]))
    try:
        if previous is not None:
            best=validation_score();checkpoint=copy.deepcopy(wm.state_dict())
            best_optimizers=copy.deepcopy([o.state_dict() for o in optimizers])
            history.append(dict(step=0,validation_loss=best,origin='preserved prior WM; re-scored at new window'))
        for index in range(steps):
            budget.check()
            k=index%len(wm.members)
            batch=train.mechanics(batch_size,rng,device=device,episode_pool=pools[k],window=fit_window_substeps,sampling=sampling)
            loss,detail=supervised_loss(wm,k,batch,loss_geometry,activation_checkpoint=activation_checkpoint,
                                        contact_force=contact_force)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite learned mechanics loss')
            optimizers[k].zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.members[k].parameters(),10.,error_if_nonfinite=True)
            budget.charge('wm_optimizer_steps',1)
            optimizers[k].step()
            if (index+1)%100==0 or index+1==steps:
                score=validation_score()
                history.append(dict(step=index+1,train_loss=float(loss.detach()),validation_loss=float(score),
                    member=k,sampler_state_sha=hashlib.sha256(repr(rng.bit_generator.state).encode()).hexdigest(),
                    **detail))
                if score<best:
                    best=score;checkpoint=copy.deepcopy(wm.state_dict())
                    best_step=index+1
                    best_optimizers=copy.deepcopy([o.state_dict() for o in optimizers])
        save_checkpoint(folder/'best_world.pt',dict(version='wheel_leg_shared_dynamics_bptt_v1',config=wm.config,
            model=checkpoint,optimizers=best_optimizers,selection='held-out-episode prediction loss; not policy selection',
            steps=steps,best_local_optimizer_step=best_step,validation_loss=best,ensemble_episode_bootstrap=pools,
            episode_ids=[row['episode_id'] for row in train.rows],
            resume_state=resume_state(best_step,pools,rng),provenance=provenance,training_recipe=recipe),budget)
    finally:
        save_last(folder/'last_world.pt',dict(version='wheel_leg_shared_dynamics_bptt_v1',config=wm.config,
            model=wm.state_dict(),optimizers=[o.state_dict() for o in optimizers],
            actual_optimizer_steps=budget.local.get('wm_optimizer_steps',0),history=history,
            episode_ids=[row['episode_id'] for row in train.rows],
            resume_state=resume_state(steps,pools,rng),provenance=provenance,training_recipe=recipe),budget)
    return dict(checkpoint=str(folder/'best_world.pt'),history=history,optimizer_steps=steps,best_local_optimizer_step=best_step,training_recipe=recipe)


def fit_critic(train, budget, output, *, steps=1000, seed=18, width=512, batch_size=128,
               device='cpu', previous=None,provenance=None,controller_sha=None):
    if train.split!='train': raise ValueError('Critic can only fit train episodes')
    torch.manual_seed(seed)
    roles=['candidate'] if any(c.get('controller')=='candidate' for c in train.cases) else ['teacher4500']
    train=train.for_controllers(roles,controller_sha)
    rng=np.random.default_rng(seed)
    before,_,_,_=train.batch(min(2048,batch_size*8),rng,device=device,actor_only=True)
    critic=Critic(value_features(before).shape[-1],width).to(device)
    critic.normalizer.fit(value_features(before))
    optimizer=torch.optim.Adam(critic.parameters(),lr=1e-4)
    target=copy.deepcopy(critic)
    if previous:
        saved=torch.load(previous,map_location=device,weights_only=False)
        if saved['config']!=critic.config: raise ValueError('Critic semantic/network identity differs')
        critic.load_state_dict(saved['model']);target.load_state_dict(saved['target'])
        optimizer.load_state_dict(saved['optimizer'])
    history=[];folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    try:
        for i in range(steps):
            budget.check()
            b,a,t,_=train.batch(batch_size,rng,device=device,actor_only=True)
            detail=critic_step(critic,target,optimizer,b,a,t['reward'],t['terminated'],t['truncated'],
                              before_optimizer=lambda:budget.charge('critic_optimizer_steps',1))
            if (i+1)%100==0 or i+1==steps: history.append(dict(step=i+1,**detail))
    finally:
        save_last(folder/'critic.pt',dict(version='full_execution_real_TD_critic_v1',config=critic.config,
            model=critic.state_dict(),target=target.state_dict(),optimizer=optimizer.state_dict(),history=history,
            gamma=.99,source='real post-physics/pre-reset current-policy train replay only',
            controllers=roles,controller_sha256=sorted({c.get('controller_sha256','FIXTURE_ONLY') for c in train.cases}),
            terminal_bootstrap=False,timeout_bootstrap=True,provenance=provenance),budget)
    return dict(checkpoint=str(folder/'critic.pt'),history=history)


def fit_actor(train, actor, teacher, wm, critic, geometry, reward, budget, output, *,
              steps=20, batch_size=16, horizon=8, device='cpu', seed=19, retention_observations=None,provenance=None,previous=None):
    if train.split!='train': raise ValueError('Actor starts must be real train-episode states')
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    actor=actor.to(device);teacher=teacher.to(device);wm=wm.to(device);critic=critic.to(device);geometry=geometry.to(device)
    for p in actor.mlp.parameters(): p.requires_grad_(True)
    actor.log_std.requires_grad_(False)  # source std retained, not a new exploration change
    optimizer=torch.optim.Adam(actor.mlp.parameters(),lr=3e-5)
    if previous:
        saved=torch.load(previous,map_location=device,weights_only=False)
        if saved.get('version')!='motion31_bptt_actor_v1':raise ValueError('cannot import PPO Adam into BPTT')
        optimizer.load_state_dict(saved['optimizer'])
    history=[];folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    try:
        for i in range(steps):
            budget.check()
            start,_,_,_=train.batch(batch_size,rng,device=device,horizon=horizon,actor_only=True)
            # The budget accounts attempted optimizer dispatch separately from accepted policy steps.
            def charge(): budget.charge('actor_optimizer_steps',1)
            result=actor_step(actor,teacher,wm,critic,geometry,reward,start,optimizer,
                horizon=horizon,member=i%len(wm.members),before_optimizer=charge,
                retention_observations=retention_observations)
            history.append(dict(iteration=i+1,**result))
        save_checkpoint(folder/'candidate_actor.pt',dict(version='motion31_bptt_actor_v1',
            actor=actor.state_dict(),optimizer=optimizer.state_dict(),interface_version='multiskill_mcu_v2',network_architecture=actor.network_architecture,
            source_checkpoint_sha256='0be977b308cfe493fdddee54823e21c713508fa582ce1618ab34e57b13fbc619',
            selection='UNSELECTED_REQUIRES_PHYSICAL_CLOSED_LOOP',history=history,provenance=provenance),budget)
    finally:
        save_last(folder/'last_actor.pt',dict(version='motion31_bptt_actor_v1',
            actor=actor.state_dict(),optimizer=optimizer.state_dict(),history=history,
            interface_version='multiskill_mcu_v2',network_architecture=actor.network_architecture,
            selection='UNSELECTED_REQUIRES_PHYSICAL_CLOSED_LOOP',provenance=provenance),budget)
    return dict(checkpoint=str(folder/'candidate_actor.pt'),history=history)
