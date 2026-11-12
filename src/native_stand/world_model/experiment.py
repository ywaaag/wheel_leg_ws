"""Independent version/spec/ledger and finite public-supervisor owner.

No historical authorization or PPO balance is consumed. Preparation is CPU only;
UNAUTHORIZED specs cannot reserve even a single physical or optimizer step.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import time
import uuid
from . import VERSION, STATE_VERSION, REWARD_VERSION
from .dataset import write_json
from .real import ROOT, sha

COUNTERS=('physics_steps','episodes','paired_physics_rollouts','wm_optimizer_steps',
          'actor_optimizer_steps','critic_optimizer_steps','ppo_updates')


def cases():
    from native_stand.students.motion_execution_uncertainty import nominal
    commands=[[0.,0.],[.5,0.],[-.5,0.],[0.,.5],[0.,-.5]]
    rows=[]
    # A tuple is execution parameters PLUS initialization, split as a whole.
    settings=[('train',0.,False),('train',0.,True),('validation',.015,False),('test',-.015,True)]
    for group,(split,yaw,stress) in enumerate(settings):
        u=nominal()
        if stress:
            u.update(PD_age_ms=12,motor_delay_ms=[12,12,9,12,12,13],spring_scale=1.1,
                     initial_vx_mps=.05,initial_pitch_rate_rad_s=.1)
        if split=='validation':
            u.update(PD_age_ms=11,motor_delay_ms=[11,11,8,11,11,12],spring_scale=1.05)
        for actor in ('teacher4500','small4500'):
            for i,cmd in enumerate(commands):
                rows.append(dict(id=f'initial_{group}_{actor}_{i}',split=split,tuple_id=f'execution_init_{group}',
                    controller=actor,command=cmd,yaw=yaw,uncertainty=u,seconds=12.,seed=104+group,
                    predecessor='teacher4500' if actor=='small4500' and i==0 else None,
                    handover_ms=1000 if actor=='small4500' and i==0 else 0,
                    action_noise=.002 if split=='train' and actor=='small4500' and i!=0 else 0.))
    for i in range(6):
        split=('train','validation','test')[i//2]
        yaw=(0.,.015,-.015)[i//2]
        u=nominal();u.update(PD_age_ms=12,motor_delay_ms=[12,12,9,12,12,13],spring_scale=1.1,
            initial_vx_mps=.05,initial_pitch_rate_rad_s=.1)
        # Controller/noise is not a physics parameter. Identical initial/plant
        # tuples share the same group across teachers and historical failures.
        group_id='execution_init_1' if split=='train' else 'failure_init_1' if split=='validation' else 'execution_init_3'
        rows.append(dict(id=f'failure_{i}',split=split,tuple_id=group_id,controller='old500',
            command=[0.,0.],yaw=yaw,uncertainty=u,seconds=6.,seed=201+i,action_noise=.002*(i%2)))
    return rows


def evaluation_cases(*, test=False):
    from native_stand.students.motion_posture_design import commands
    from native_stand.students.motion_execution_uncertainty import nominal
    rows=[]
    values=commands() # all22 nominal old command identities, not just the five pilot commands
    for i,cmd in enumerate(values):
        rows.append(dict(id=f'old_{i}',command=cmd,uncertainty=nominal(),yaw=0. if not test else -.015,
                         seconds=30.,warmup=0.))
    # Fixed low-speed uncertainty and continuous handover; no physical reset at switch.
    for i,cmd in enumerate(([0.,0.],[.5,0.],[-.5,0.],[0.,.5],[0.,-.5])):
        u=nominal();u.update(PD_age_ms=12,motor_delay_ms=[12,12,9,12,12,13],spring_scale=1.1,
            initial_vx_mps=.05,initial_pitch_rate_rad_s=.1)
        rows.append(dict(id=f'execution_{i}',command=cmd,uncertainty=u,yaw=0. if not test else -.015,
                         seconds=30.,warmup=0.))
    for i,cmd in enumerate(([0.,0.],[.5,0.],[0.,.5])):
        rows.append(dict(id=f'handover_{i}',command=cmd,uncertainty=nominal(),yaw=0. if not test else -.015,
            seconds=30.,warmup=1.,predecessor='teacher4500'))
    return rows  #30 conditions; test subset selected a priori below


def code_identities():
    # SHA references only, no copy of the existing source/test tree.
    paths=list((ROOT/'src/native_stand/world_model').glob('*.py'))
    paths += [ROOT/'scripts/world_model.py',ROOT/'src/native_stand/command_v6/supervisor.py',
              ROOT/'src/native_stand/command_v6/completion.py']
    paths += [ROOT/'scripts'/n for n in ('motion_mcu_playback.py','diagnose_motion_closed_loop_robustness.py',
        'diagnose_motion_real2sim.py','validate_native_mujoco.py','compare_motion_timing_profiles.py')]
    paths += [ROOT/'src/native_stand/students'/n for n in ('mcu_interface.py','interface.py','model.py',
        'motion_mcu_execution.py','real2sim_delay.py','mujoco_solver.py','motion_execution_uncertainty.py',
        'motion_tracking_reward.py','motion_quality.py','motion_parallel_course.py','motion_parallel_design.py',
        'motion_yaw_quality.py','motion_sensor_transfer.py','motion_antioscillation.py','motion_posture_design.py',
        'motion_differential.py','motion_distillation_pipeline.py','export.py')]
    paths += [ROOT/'src/native_stand/command_v6/domain'/n for n in ('command_schedule.py','posture_geometry.py')]
    return {str(p):sha(p) for p in paths}


def supervision_performance(config,version='legacy_full_fk_v1'):
    """Explicit compute-only profile; never shrinks the dynamics or time window."""
    result=dict(config)
    if version=='legacy_full_fk_v1':return result
    if version!='closure_no_checkpoint_v1':raise ValueError('unknown supervision performance profile')
    result.update(activation_checkpoint=False,supervision_geometry='closure_fk_v1',performance_version=version)
    return result


def contact_force_supervision(config,declaration=None):
    """Explicit loss-space dispatch recorded in the spec; never inferred from shapes.

    Always writes the block (including the explicit `none` control marker) so a later
    reader can tell an intentionally disabled arm from a missing field.
    """
    from .models import contact_force_config
    result=dict(config)
    frozen=contact_force_config(declaration)
    result['contact_force_supervision']=frozen if frozen is not None else dict(version='none')
    return result


def prepare(folder=None,*,supervision_performance_version='closure_no_checkpoint_v1',contact_force=None):
    from native_stand.command_v6.domain.experiment_naming import new_experiment_name
    from .real import sources, load_plant, description, context
    parent,inputs=sources(); model,refs,cfg,readback=load_plant();d=description(model,refs);ctx,layout=context(model)
    name=new_experiment_name('motion4500_world_model_bptt')
    folder=ROOT/'configs/experiments'/name if folder is None else Path(folder).resolve()
    from native_stand.command_v6.domain.experiment_naming import validate_new_experiment_name
    validate_new_experiment_name(folder.name)
    if folder.parent!=ROOT/'configs/experiments':raise ValueError('new experiment must use the project experiment directory')
    run=ROOT/'runs'/folder.name
    spec=dict(version=VERSION,experiment=folder.name,interface_version='multiskill_mcu_v2',state_version=STATE_VERSION,
        reward_version=REWARD_VERSION,source_inputs=inputs,code_inputs=code_identities(),
        source_actor='teacher4500',actor_architecture=[31,512,256,128,6],actor_activation='ELU/tanh',
        physical_backend='MuJoCo',Native='NOT_AUTHORIZED',hardware='NOT_AUTHORIZED',
        execution=cfg,plant_readback=readback,geometry=d,context_layout=layout,context_dim=ctx.shape[-1],
        source_reward=parent['reward'],mixtures='fixed source4500 checkpoint; no curriculum/mix progression',
        world_model=dict(members=3,width=1024,blocks=2,hidden_layers=5,substep_s=.001,policy_s=.01,fit_window_substeps=8,
            prediction='all generalized velocity increments, next solver warmstart, wheel/root contact forces',
            deterministic_execution=True,task_name_input=False,lr=1e-4,batch_size=256),
        critic=dict(width=512,source='real replay TD only',gamma=parent['ppo']['gamma'],lr=1e-4,batch_size=128,
            input='all named causal state, complete queues, reward/terminal history, task/schedule',target_EMA=.01),
        bptt=dict(horizon=8,horizon_s=.08,test_horizons=[8,16,32],lr=3e-5,batch_size=16,
            uncertainty_penalty=.01,source_retention_KL=.01,max_source_scaled_action_change=.1,
            source_std='preserved; fixed',world_parameters='frozen, input gradients enabled',
            critic_parameters='frozen, input gradients enabled',actor_optimizer='independent Adam, no PPO Adam/log probability'),
        collection=cases(),validation_conditions=evaluation_cases(),
        final_test_conditions=[r for r in evaluation_cases(test=True) if r['id'] in
            ('old_0','old_1','old_6','old_11','old_17','old_21','execution_0','execution_1',
             'execution_2','execution_3','execution_4','handover_0')],
        workflow=dict(rounds_max=2,initial=dict(wm_optimizer_steps=3000,critic_optimizer_steps=1000,actor_optimizer_steps=20),
            refresh=dict(episodes_max=12,seconds_per_episode=12,wm_optimizer_steps=1000,
                         critic_optimizer_steps=500,actor_optimizer_steps=20),
            refresh_gate='same current Actor passes complete physical validation and improves zero drift; no automatic second run',
            actor_gate='held-out prediction and paired physical action-gradient reports for exact WM SHA both pass',
            selection='complete same-condition physical validation; final test never selects',
            physical_budget_breakdown=dict(initial_collection_steps=516000,refresh_collection_steps=144000,
                first_validation_3actors_steps=2709000,second_validation_candidate_only_steps=903000,
                final_test_3actors_steps=1083000,paired_gradient_steps=15360),
            refresh_validation='reuse frozen source/small same-condition physical results; evaluate refreshed candidate only'),
        limits=dict(physics_steps=5600000,episodes=220,paired_physics_rollouts=192,wm_optimizer_steps=4000,
            actor_optimizer_steps=40,critic_optimizer_steps=1500,ppo_updates=0,wall_s=10800,
            artifacts_bytes=3*1024**3,min_free_bytes=1536*1024**2),
        resources=dict(physical_gpu=0,uuid='GPU-e704e7d4-2965-4712-9759-ba6b9910bd72',logical_cuda=0,
            mode='shared',no_preemption=True,collection_device='cpu',optimizer_device='cuda:0',max_torch_threads=4),
        run=str(run),authorization=dict(status='UNAUTHORIZED',owner_session=os.environ.get('HAPI_SESSION_ID'),
            scope=None,old_ppo_allowances_reused=False),notifications=dict(enabled=False),
        Recovery=dict(status='INTERFACE_ONLY_NOT_COVERED',needs='same-state execution/plant reconciliation and autonomous Swing data'),
        Jump=dict(status='INTERFACE_ONLY_NOT_COVERED',needs='flight and landing transitions'))
    spec['world_model']=supervision_performance(spec['world_model'],supervision_performance_version)
    spec['world_model']=contact_force_supervision(spec['world_model'],contact_force)
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'spec.json',spec)
    write_json(folder/'budget.json',dict(version=VERSION,spec_sha256=sha(folder/'spec.json'),
        authorization='UNAUTHORIZED',limits=spec['limits'],used={k:0 for k in COUNTERS},wall_used_s=0.,attempts={}))
    write_json(folder/'manifest.json',dict(version=VERSION,spec_sha256=sha(folder/'spec.json'),
        state='CPU_PREPARED_UNAUTHORIZED',runtime_attempts=0,source_inputs=inputs,code_inputs=spec['code_inputs']))
    return folder/'spec.json'


def verify(spec_path, *, authorization=False):
    spec_path=Path(spec_path).resolve();s=json.loads(spec_path.read_text())
    if s['version']!=VERSION or s['state_version']!=STATE_VERSION or s['limits']['ppo_updates']!=0:
        raise ValueError('explicit new algorithm/state/PPO0 identity required')
    if Path(s['run']).parent!=ROOT/'runs' or Path(s['run']).name!=s['experiment']:
        raise ValueError('independent matching run name required')
    book=json.loads(spec_path.with_name('budget.json').read_text())
    if book['spec_sha256']!=sha(spec_path) or book['limits']!=s['limits']:
        raise ValueError('immutable spec/independent ledger identity differs')
    deletions=dict(s.get('reviewed_source_deletions') or {})
    for p,h in {**s['source_inputs'],**s['code_inputs'],**s.get('resume_inputs',{})}.items():
        if p in s['source_inputs'] and not Path(p).exists():
            # Only a reviewed deletion record frozen into this hash-bound spec may excuse
            # a missing historical input; code and resume inputs stay strict.
            entry=deletions.get(str(p))
            if (not isinstance(entry,dict) or entry.get('sha256_recorded')!=h
                    or not entry.get('reason') or not entry.get('evidence')):
                raise ValueError('frozen source missing without a reviewed deletion record: '+p)
            continue
        if sha(p)!=h:raise ValueError('source/code identity differs: '+p)
    if authorization:
        if s['authorization']['status']!='AUTHORIZED' or book['authorization']!='AUTHORIZED':
            raise PermissionError('new World Model/BPTT/sampling authorization missing; old PPO quotas cannot be used')
        current=os.environ.get('HAPI_SESSION_ID')
        if s['authorization']['owner_session']!=current:
            # No implicit takeover: only a frozen, hash-bound owner handoff may authorize
            # another session, and it must bind this exact parent declaration.
            declared=s.get('owner_handoff')
            if not isinstance(declared,dict) or not declared.get('sha256'):
                raise ValueError('current owner session differs; no implicit takeover')
            verify_owner_handoff(declared,parent_spec_sha=declared.get('parent_spec_sha256'),
                                 session_id=s['authorization']['owner_session'],action='execute')
        if not s['authorization']['scope']:raise ValueError('explicit authorization scope absent')
    return s,book


def validate_request(request,spec,book,*,files=True):
    """Validate a reviewable operation before its exact-ID reservation."""
    operation=request.get('operation')
    allowed={'cycle','engineering','reuse-data','collect','build-data','train-world','train-critic','predict','gradient','train-actor','evaluate','export'}
    if operation not in allowed:raise ValueError('unknown World Model operation')
    if spec['workflow'].get('prediction_only') and operation not in ('cycle','reuse-data','train-world','predict'):
        raise ValueError('prediction-only block cannot dispatch physics, Critic or Actor stages')
    root=Path(spec['run']);out=Path(request['output']).resolve()
    if out==root or not out.is_relative_to(root):raise ValueError('operation output must be a unique child of the independent run')
    if request.get('split','train') not in ('train','validation','test'):raise ValueError('unknown replay split')
    if operation.startswith('train-') and request.get('split','train')!='train':raise ValueError('learners cannot consume held-out splits')
    if operation=='gradient' and request.get('split')!='validation':raise ValueError('physical gradient gate uses validation only')
    prior=[a for a in book['attempts'].values() if a['stage']==operation]
    if operation=='cycle' and (not spec['workflow'].get('automatic_cycle') or book['attempts']):
        raise ValueError('fresh explicitly declared automatic cycle required')
    if operation=='engineering' and prior:raise ValueError('engineering attempt already consumed')
    if operation=='reuse-data' and (prior or not spec.get('prediction_continuation')):
        raise ValueError('one explicitly bound prediction continuation evidence required')
    rounds=spec['workflow']['rounds_max']
    if operation in ('train-world','train-critic','train-actor','gradient') and rounds is not None and len(prior)>=rounds:
        raise ValueError('declared finite learning blocks exhausted')
    if operation.startswith('train-'):
        counter={'train-world':'wm_optimizer_steps','train-critic':'critic_optimizer_steps','train-actor':'actor_optimizer_steps'}[operation]
        phase='refresh' if prior else 'initial'
        steps=request.get('steps')
        if type(steps) is not int or not 0<steps<=spec['workflow'][phase][counter]:raise ValueError('operation optimizer steps exceed declared finite block')
        previous={'train-world':'world','train-critic':'critic','train-actor':'actor'}[operation]
        warmstart=spec.get('prediction_continuation',{}).get('world') if operation=='train-world' and not prior else None
        if warmstart:
            if request.get(previous)!=warmstart:raise ValueError('first continuation must use declared exact World Model checkpoint')
        elif bool(prior)!=bool(request.get(previous)):raise ValueError('refresh must resume explicit same-algorithm checkpoint; initial uses declared source')
    if operation=='collect':
        refresh=request.get('collection','initial')=='refresh'
        reused=spec.get('prediction_continuation')
        if (not refresh and (prior or reused)) or (refresh and not prior and not reused):raise ValueError('initial collection must precede distribution refresh')
        if refresh and (not request.get('actor') or not request.get('selection')):raise ValueError('refresh needs current Actor and physical selection')
    if operation=='export' and (not request.get('actor') or not request.get('selection')):raise ValueError('selected Actor required for export')
    if operation=='evaluate':
        names=request.get('controllers',[])
        if not names or len(names)!=len(set(names)) or any(x not in ('teacher4500','small4500','candidate') for x in names):
            raise ValueError('explicit unique evaluation controllers required')
        if 'candidate' in names and not request.get('actor'):raise ValueError('candidate checkpoint absent')
        if request.get('final_test') and not request.get('selection'):raise ValueError('final test follows frozen physical validation selection')
    if files:
        for key in ('dataset','world','critic','actor','prediction','gradient','selection'):
            if request.get(key) and not Path(request[key]).is_file():raise ValueError('required input missing: '+key)
    return True


class Budget:
    """One exact-ID attempt; physical dispatch reservations survive abrupt death.

    Physics progress batches at most1000 ticks per journal write. An unclean death
    leaves its reservation UNKNOWN/held and blocks reuse, never silently refunded.
    Optimizer dispatch counts are persisted before every optimizer.step.
    """
    def __init__(self,spec_path,stage,output):
        self.path=Path(spec_path).resolve();self.spec,self.book=verify(self.path,authorization=True)
        self.stage=stage;self.output=Path(output).resolve();self.id=uuid.uuid4().hex
        if not self.output.is_relative_to(Path(self.spec['run'])):
            raise ValueError('attempt output must stay inside the new independent run')
        self.local={k:0 for k in COUNTERS};self.reserved={k:0 for k in COUNTERS}
        self.start=time.monotonic();self.start_unix=time.time()
        self.last_disk_check=0.
        allowance=self.spec['limits']['wall_s']
        self.deadline=float('inf') if allowance is None else self.start+max(0,allowance-self.book['wall_used_s'])
        self._change(self._reserve)

    def _change(self,fn):
        with self.path.with_name('budget.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            book=json.loads(self.path.with_name('budget.json').read_text());result=fn(book)
            tmp=self.path.with_name('budget.json.tmp')
            tmp.write_text(json.dumps(book,indent=2,allow_nan=False)+'\n')
            os.replace(tmp,self.path.with_name('budget.json'))
            self.book=book
            return result

    def _reserve(self,book):
        receipt=self.path.with_name('chain_submission.json')
        if receipt.is_file() and json.loads(receipt.read_text())['attempt_id']!=os.environ.get('WHEEL_WORLD_CYCLE_ID'):
            raise ValueError('this declaration belongs to its serial cycle owner')
        if book['authorization']!='AUTHORIZED' or any(a['status'] in ('RUNNING','FAILED','UNKNOWN') for a in book['attempts'].values()):
            raise ValueError('authorization absent or unsettled/failed attempt; no automatic retry')
        if any(a['output']==str(self.output) for a in book['attempts'].values()) or self.output.exists():
            raise ValueError('duplicate stage output/claim')
        root=Path(self.spec['run'])
        present=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
        required=max(0,self.spec['limits']['artifacts_bytes']-present)+self.spec['limits']['min_free_bytes']
        if shutil.disk_usage(ROOT).free<required:
            raise OSError('insufficient disk for remaining independent artifact allowance plus1.5GiB reserve; no deletion')
        book['attempts'][self.id]=dict(stage=self.stage,output=str(self.output),status='RUNNING',
            started_unix_s=self.start_unix,actual=self.local.copy(),reserved_upper=self.reserved.copy(),
            owner_session=os.environ.get('HAPI_SESSION_ID'),pid=os.getpid())

    def check(self):
        now=time.monotonic()
        if now>=self.deadline:raise TimeoutError('independent experiment wall allowance exhausted')
        if now-self.last_disk_check<1.:return
        self.last_disk_check=now
        free=shutil.disk_usage(ROOT).free
        if free<self.spec['limits']['min_free_bytes']:raise OSError('runtime free-space reserve exhausted')
        root=Path(self.spec['run'])
        total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
        if total>=self.spec['limits']['artifacts_bytes']-1024**2:raise OSError('new experiment artifact cap exhausted')

    def storage_guard(self,new_bytes):
        self.check()
        root=Path(self.spec['run'])
        total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
        if total+new_bytes>self.spec['limits']['artifacts_bytes']-1024**2:
            raise OSError('checkpoint atomic-write peak exceeds experiment artifact cap')
        if shutil.disk_usage(ROOT).free<self.spec['limits']['min_free_bytes']+new_bytes:
            raise OSError('checkpoint atomic-write peak exhausts runtime free reserve')

    def ensure(self,key,count):
        if key not in COUNTERS or type(count) is not int or count<0:raise ValueError('explicit nonnegative counter required')
        limit=self.spec['limits'][key]
        if limit is not None and self.book['used'][key]+self.local[key]+count>limit:
            raise RuntimeError('independent '+key+' budget exhausted')
        upper=self.local[key]+count
        if upper>self.reserved[key]:
            self.reserved[key]=upper
            self._change(lambda b:b['attempts'][self.id].update(actual=self.local.copy(),reserved_upper=self.reserved.copy()))

    def charge(self,key,count):
        if key=='physics_steps':
            if self.local[key]+count>self.reserved[key]:
                limit=self.spec['limits'][key]
                chunk=1000 if limit is None else min(1000,limit-self.book['used'][key]-self.local[key])
                self.ensure(key,max(count,chunk))
        else:self.ensure(key,count)
        self.local[key]+=count
        if key!='physics_steps' or self.local[key]%1000==0:
            self._change(lambda b:b['attempts'][self.id].update(actual=self.local.copy(),reserved_upper=self.reserved.copy()))

    def finalize(self,status,error=None):
        def settle(b):
            row=b['attempts'][self.id]
            if row['status']!='RUNNING':raise ValueError('attempt already settled')
            row.update(status=status,error=error,actual=self.local.copy(),reserved_upper=self.reserved.copy(),
                       wall_s=time.monotonic()-self.start,finished_unix_s=time.time())
            for k in COUNTERS:b['used'][k]+=self.local[k]
            b['wall_used_s']+=row['wall_s']
            row['unused_reservation_released']=status=='COMPLETED'
            row['uncertain_reserved']={k:max(0,self.reserved[k]-self.local[k]) for k in COUNTERS} if status!='COMPLETED' else {}
        self._change(settle)


def settle_interrupted_attempt(spec_path,attempt_id,*,error,status='FAILED'):
    """Reconcile one attempt whose worker died before it could settle itself.

    Bookkeeping only: counters are never touched, no output is created or removed, and
    the declaration/spec SHA chain is re-checked. A still-running pid is refused so this
    can never be used to steal a live attempt.
    """
    p=Path(spec_path).resolve();s=json.loads(p.read_text())
    book_path=p.with_name('budget.json');book=json.loads(book_path.read_text())
    if book['spec_sha256']!=sha(p):raise ValueError('spec/ledger identity differs')
    row=book['attempts'].get(attempt_id)
    if row is None:raise ValueError('unknown attempt id')
    if row['status']!='RUNNING':raise ValueError('attempt already settled')
    pid=row.get('pid');alive=False
    if isinstance(pid,int) and pid>0:
        try:os.kill(pid,0);alive=True
        except OSError:alive=False
    if alive:raise ValueError('attempt pid %d still alive; refusing to settle'%pid)
    if status not in ('FAILED','UNKNOWN'):raise ValueError('interrupted attempt settles as FAILED or UNKNOWN')
    if not isinstance(error,str) or len(error.strip())<10:raise ValueError('explicit interruption reason required')
    def settle(b):
        target=b['attempts'][attempt_id]
        if target['status']!='RUNNING':raise ValueError('attempt already settled')
        target.update(status=status,error=error.strip(),interrupted=True,
                      finished_unix_s=time.time(),settled_by='settle_interrupted_attempt')
    with book_path.with_name('budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        current=json.loads(book_path.read_text());settle(current)
        tmp=book_path.with_name('budget.json.tmp');tmp.write_text(json.dumps(current,indent=2,allow_nan=False)+'\n')
        os.replace(tmp,book_path)
    return dict(attempt_id=attempt_id,status=status,error=error.strip())


def executing_session():
    """Real executor identity; HAPI when the runtime provides it, else the IDE session.

    Never fabricates a HAPI id. The kind is recorded verbatim so a later reader can
    tell which surface authorized a block and whether hub notification exists.
    """
    hapi=os.environ.get('HAPI_SESSION_ID')
    if hapi:return dict(session_id=hapi,kind='hapi',hapi_session=hapi)
    for key in ('CODEBUDDY_SESSION_ID','CLAUDE_SESSION_ID'):
        value=os.environ.get(key)
        if value:return dict(session_id=value,kind='ide_session',hapi_session=None,source_env=key)
    raise ValueError('no explicit executor session identity available')


def record_owner_handoff(parent_spec,delegation,*,scope,output,analysis=None):
    """Strict, one-shot handoff of the World Model route to a new executing session.

    Binds the parent declaration + its terminal evidence + the user's delegation text
    to exactly one new session id. The parent spec, ledger and run stay read-only.
    """
    parent=Path(parent_spec).resolve();parent_spec_sha=sha(parent)
    spec=json.loads(parent.read_text())
    run=Path(spec['run'])
    analysis_path=Path(analysis) if analysis else run/'analysis.json'
    evidence=json.loads(analysis_path.read_text()) if analysis_path.is_file() else {}
    if not isinstance(delegation,str) or len(delegation.strip())<10:
        raise ValueError('handoff requires the verbatim user delegation text')
    if not isinstance(scope,dict) or not scope.get('arms'):
        raise ValueError('handoff requires an explicit bounded scope')
    current=executing_session()
    record=dict(version='wm_owner_handoff_v1',created_unix_s=time.time(),
        parent=dict(spec=str(parent),spec_sha256=parent_spec_sha,run=str(run),
                    event_id=evidence.get('event_id'),terminal_status=evidence.get('status'),
                    failure=evidence.get('failure'),analysis=str(analysis_path),
                    analysis_sha256=sha(analysis_path) if analysis_path.is_file() else None,
                    owner_session=spec['authorization'].get('owner_session')),
        new_owner=dict(**current,assumed_route='shared_world_model_bptt',
                       parent_owner_still_executing=False),
        delegation=dict(user_text=delegation.strip(),received_unix_s=time.time(),
                        channel='user_message_in_executing_session',
                        note='the original owner session terminated on a provider quota error; '
                             'the user delegated the route to this executor'),
        scope=scope,
        constraints=dict(parent_spec_read_only=True,parent_owner_session_unchanged=True,
                         no_hapi_session_fabrication=True,no_history_deletion=True,
                         notify_on_terminal='disabled: executor is not a HAPI session'),
        recorded_by=current)
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise ValueError('handoff record already exists; one-shot only')
    target.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    record['record_sha256']=sha(target)
    return record


def verify_owner_handoff(record,*,parent_spec_sha,session_id,action):
    """Re-check a frozen handoff on every use; any mismatch is fatal.

    The record carried in a declaration is only {path, sha256, owner_session,
    parent_spec_sha256}; the authoritative content is the hashed file on disk.
    """
    if not isinstance(record,dict) or not record.get('sha256'):
        raise ValueError('owner handoff record missing')
    path=Path(record.get('path',''))
    if not path.is_file() or sha(path)!=record.get('sha256'):
        raise ValueError('owner handoff record changed or missing')
    frozen=json.loads(path.read_text())
    if frozen.get('version')!='wm_owner_handoff_v1':
        raise ValueError('owner handoff record version differs')
    if frozen['parent']['spec_sha256']!=parent_spec_sha:
        raise ValueError('owner handoff binds a different parent declaration')
    if frozen['new_owner']['session_id']!=record.get('owner_session'):
        raise ValueError('owner handoff new-owner id differs from the declaration')
    if session_id and frozen['new_owner']['session_id']!=session_id:
        raise ValueError('owner handoff belongs to another executing session')
    if not frozen['delegation']['user_text']:
        raise ValueError('owner handoff delegation text missing')
    if action not in frozen['scope'].get('actions',[]):
        raise ValueError('owner handoff scope does not cover '+action)
    return frozen


def authorize_prepared(spec_path,scope,*,unlimited=False,notify_owner=False,owner_handoff=None):
    """Called only after the human grants the exact new experiment block."""
    p=Path(spec_path);s,b=verify(p)
    if b['attempts'] or s['authorization']['status']!='UNAUTHORIZED':
        raise ValueError('only a zero-attempt fresh declaration can be authorized')
    if not scope:raise ValueError('human authorization evidence required')
    current=executing_session()
    session=current['session_id']
    handoff=None
    if owner_handoff is not None:
        declared=s.get('owner_handoff') or {}
        if declared.get('path')!=str(Path(owner_handoff).resolve()) or declared.get('sha256')!=sha(owner_handoff):
            raise ValueError('authorize handoff differs from the one frozen into the declaration')
        handoff=verify_owner_handoff(declared,parent_spec_sha=s['prediction_continuation']['parent_spec_sha256'],
                                     session_id=session,action='authorize')
    elif os.environ.get('HAPI_SESSION_ID') is None:
        raise ValueError('explicit current HAPI owner required; a frozen owner handoff is the only alternative')
    notify_owner=notify_owner and current['hapi_session'] is not None
    if unlimited:
        for key in COUNTERS:
            if key!='ppo_updates':s['limits'][key]=None
        s['limits']['wall_s']=None
        s['workflow'].update(rounds_max=None,
            refresh_gate='complete current physical validation retains old skills and improves zero drift against both source and prior accepted Actor',
            automatic_cycle=True,engineering_physics_steps=1500,
            stop='prediction/gradient gate failure, no physical improvement, numerical/identity/storage fault; no blind retry')
    s['notifications']=dict(enabled=notify_owner,owner_session=session,target_session=session,
        events=['terminal','gate_stop','exception'],
        transport=('public completion.notify; at most once; no watcher' if current['hapi_session'] else
                   'disabled: executor is not a HAPI session; terminal files in the run directory are the delivery'))
    if handoff:
        s['owner_handoff'].update(consumed_at_unix_s=time.time(),consumed_by=session)
    s['authorization'].update(status='AUTHORIZED',scope=scope,owner_session=session,
        hapi_session=current['hapi_session'],session_kind=current['kind'],
        unlimited_updates_and_wall=unlimited,authorized_unix_s=time.time())
    p.write_text(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    b.update(authorization='AUTHORIZED',spec_sha256=sha(p),limits=s['limits'])
    p.with_name('budget.json').write_text(json.dumps(b,indent=2)+'\n')
    manifest=json.loads(p.with_name('manifest.json').read_text())
    manifest.update(state='AUTHORIZED_NOT_SUBMITTED',spec_sha256=sha(p),code_inputs=s['code_inputs'])
    p.with_name('manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def prepare_prediction_continuation(parent_path,review_path,*,continue_window80=False,contact_focus=False,
                                   contact_force_arm=None,pilot_artifacts_bytes=None,
                                   supervision_performance_version='closure_no_checkpoint_v1',
                                   plan_only=False,reviewed_code_changes=None,initialization_world=None,
                                   owner_handoff=None,reviewed_source_deletions=None):
    """New immutable declaration after an inspected prediction-only gate stop.

    This consumes the current user's unlimited World Model authorization, never
    a historical PPO balance or an implicit permission from a notification.
    """
    from native_stand.command_v6.domain.experiment_naming import new_experiment_name
    from .models import CONTACT_FORCE_FROZEN
    import copy
    if contact_force_arm not in (None,'control','force'):
        raise ValueError('contact force pilot arm must be explicit control or force')
    if continue_window80 and contact_focus:raise ValueError('choose one explicit continuation recipe')
    if contact_force_arm and (continue_window80 or contact_focus):
        raise ValueError('contact force pilot is one single-variable change; combine nothing else')
    if initialization_world is not None and not contact_force_arm:
        raise ValueError('explicit initialization checkpoint is declared only by the paired pilot recipe')
    p=Path(parent_path).resolve();parent=json.loads(p.read_text())
    book=json.loads(p.with_name('budget.json').read_text());run=Path(parent['run'])
    result=json.loads((run/'analysis.json').read_text());ready=json.loads((run/'report_ready.json').read_text())
    session=os.environ.get('HAPI_SESSION_ID');auth=parent['authorization']
    owner_matches=bool(session) and auth.get('owner_session')==session
    handoff=None
    if owner_handoff is not None:
        if continue_window80 or contact_focus or not contact_force_arm:
            raise ValueError('owner handoff is declared only for the paired contact force pilot')
        current=executing_session()
        handoff=verify_owner_handoff(dict(path=str(Path(owner_handoff).resolve()),
                                          sha256=sha(owner_handoff),owner_session=current['session_id'],
                                          parent_spec_sha256=sha(parent_path)),
                                     parent_spec_sha=sha(parent_path),session_id=current['session_id'],
                                     action='prepare')
        session=current['session_id'];owner_matches=handoff['new_owner']['session_id']==session
    if auth['status']!='AUTHORIZED' or not auth.get('unlimited_updates_and_wall') or (not plan_only and not owner_matches):
        raise PermissionError('current owner unlimited new-algorithm authorization required')
    # plan_only evaluates the real declaration path and returns the payload WITHOUT
    # writing. It never rewrites the parent owner and never fakes a session id. When the
    # executing session differs from the parent owner, the plan states the handoff need.
    if book['spec_sha256']!=sha(p) or book['limits']!=parent['limits'] or ready['report_sha256']!=sha(run/'completion_report.md'):
        raise ValueError('parent declaration/settlement identity differs')
    if (result['status']!='STOPPED_GATE' or result['reason']!='PREDICTION_GATE' or result['error'] is not None
            or any(a['status']!='COMPLETED' for a in book['attempts'].values())
            or any(book['used'][k] for k in ('actor_optimizer_steps','critic_optimizer_steps','ppo_updates'))):
        raise ValueError('only fully settled pre-policy prediction failure can use this continuation')
    deletions=dict(reviewed_source_deletions or {})
    for path,h in parent['source_inputs'].items():
        if not Path(path).exists():
            # An external prune is acceptable only with a per-file record binding the
            # recorded digest, the reason and the evidence receipt. Never skipped quietly.
            entry=deletions.get(str(path))
            if (not isinstance(entry,dict) or entry.get('sha256_recorded')!=h
                    or not entry.get('reason') or not entry.get('evidence')):
                raise ValueError('frozen source missing without a reviewed deletion record: '+path)
            continue
        if sha(path)!=h:raise ValueError('frozen source identity changed: '+path)
    for path,h in parent.get('resume_inputs',{}).items():
        if sha(path)!=h:raise ValueError('ancestor evidence changed: '+path)
    reviewed=dict(reviewed_code_changes or {})
    for path,h in parent['code_inputs'].items():
        if '/native_stand/world_model/' in path or path.endswith('/scripts/world_model.py'):
            continue
        if sha(path)==h:continue
        # Another worker's edit is acceptable only after a per-file review that names both
        # the recorded and the current digest; SHA sets are never ignored wholesale.
        entry=reviewed.get(str(path))
        if (not isinstance(entry,dict) or entry.get('old_sha256')!=h or entry.get('new_sha256')!=sha(path)
                or not entry.get('review')):
            raise ValueError('unreviewed shared code change: '+path)
    review_path=Path(review_path).resolve();review=json.loads(review_path.read_text())
    world=Path(result['checkpoints']['world']['checkpoint'])
    if sha(world)!=result['checkpoints']['world']['sha256']:
        raise ValueError('review/model identity mismatch')
    extra_inputs=[]
    if contact_force_arm:
        # One bounded paired pilot: same parent WM/Adam/normalizer/bootstrap, same
        # uniform80 sampling and the same fresh sampler seed in both arms.
        if parent['world_model']['fit_window_substeps']!=80:
            raise ValueError('contact force pilot requires the inspected80ms supervision parent')
        if parent.get('optimization_version') not in ('wheel_leg_shared_dynamics_bptt_window80_v2',
                'wheel_leg_shared_dynamics_bptt_window80_continue_v3',
                'wheel_leg_shared_dynamics_bptt_contact25_v4'):
            raise ValueError('contact force pilot takes the audited80ms World Model lineage only')
        inherited=parent['prediction_continuation']
        dataset=Path(inherited['dataset']);collection_root=Path(inherited['collection_root'])
        engineering=Path(inherited['engineering'])
        # Settlement identity (contact25) and initialization (uniform80 best) are separate.
        init_world=Path(initialization_world).resolve() if initialization_world else world
        if not init_world.is_file():raise ValueError('declared initialization World Model checkpoint absent')
        if (review['world_sha256']!=sha(init_world) or review.get('parent_world_sha256')!=sha(world)
                or review['dataset_sha256']!=sha(dataset)
                or review.get('physics_steps')!=0 or review.get('optimizer_steps')!=0):
            raise ValueError('contact pilot review must bind initialization WM, settled parent WM and this replay '
                             'with zero physics and optimizer work')
        if not review.get('contact_signal_audit'):
            raise ValueError('contact pilot requires the completed contact signal audit')
        extra_inputs=[run/'round_000/world/result.json',init_world]
    elif continue_window80 or contact_focus:
        expected='wheel_leg_shared_dynamics_bptt_window80_continue_v3' if contact_focus else 'wheel_leg_shared_dynamics_bptt_window80_v2'
        if parent['world_model']['fit_window_substeps']!=80 or parent.get('optimization_version')!=expected:
            raise ValueError('one further window80 block requires the inspected window80 experiment')
        inherited=parent['prediction_continuation']
        dataset=Path(inherited['dataset']);collection_root=Path(inherited['collection_root'])
        engineering=Path(inherited['engineering'])
        if contact_focus:
            audit_path=review_path.with_name('sampling_audit.json');audit=json.loads(audit_path.read_text())
            if (review['world_sha256']!=sha(world) or audit['world_sha256']!=sha(world)
                    or review['dataset_sha256']!=sha(dataset) or audit['dataset_sha256']!=sha(dataset)
                    or audit['parent_spec_sha256']!=sha(p)):
                raise ValueError('contact review/model/replay identity differs')
            if (any(x['physics_steps'] or x['optimizer_steps'] for x in (review,audit))
                    or not audit['gradient_finite'] or audit['gradient_norm']<=0
                    or not audit['all_members_have_events'] or len(review['faults'])==0):
                raise ValueError('contact continuation requires real train-event coverage and finite offline gradient evidence')
            extra_inputs=[audit_path,run/'round_000/world/result.json']
        else:
            if review['models']['window80']['sha256']!=sha(world) or any(review['actual_initial_invalid'].values()):
                raise ValueError('window80 review identity/initial validity differs')
            coverage_path=review_path.with_name('contact_coverage.json');coverage=json.loads(coverage_path.read_text())
            if coverage['actual_selected_substeps']['invalid_contact_substeps']!=0:
                raise ValueError('real contact labels disagree with the physical gate; investigate before more fitting')
            if any(x['physics_steps'] or x['optimizer_steps'] for x in (review,coverage)):
                raise ValueError('review must be offline inference only')
            fit=json.loads((run/'round_000/world/result.json').read_text());history=fit['history']
            if not (history[-1]['step']==1000 and history[-1]['validation_loss']==min(h['validation_loss'] for h in history)
                    and history[-1]['validation_loss']<history[0]['validation_loss']*.9):
                raise ValueError('one more unchanged fitting block requires continuing held-out improvement at block end')
            if (review['reviewed_parent_spec_sha256']!=sha(p) or review['dataset_sha256']!=sha(dataset)
                    or coverage['dataset_sha256']!=sha(dataset)):
                raise ValueError('offline review belongs to a different declaration or replay')
            extra_inputs=[coverage_path,run/'round_000/world/result.json']
    else:
        if parent['world_model']['fit_window_substeps']!=8:
            raise ValueError('window80 continuation may only replace the explicit window8 experiment')
        if review['original_world_sha256']!=sha(world):raise ValueError('review/model identity mismatch')
        if not review['window_80']['gradient_finite'] or review['physics_steps']!=0 or review['optimizer_steps']!=0:
            raise ValueError('review must include finite offline80ms gradient evidence')
        dataset=run/'round_000/data/dataset.json';collection_root=run/'round_000/collect';engineering=run/'engineering/result.json'
    data=json.loads(dataset.read_text());engineering_evidence=json.loads(engineering.read_text())
    if engineering_evidence['passed'] is not True:raise ValueError('physical engineering evidence did not pass')
    inputs={**parent.get('resume_inputs',{}),**{str(x):sha(x) for x in (p,p.with_name('budget.json'),
        run/'analysis.json',run/'report_ready.json',run/'completion_report.md',world,dataset,review_path,engineering,
        run/'round_000/prediction/result.json',*extra_inputs)}}
    for ep in data['episodes']:
        folder=Path(ep['path'])
        for file,key in (('episode.json','manifest_sha256'),('transitions.npz','trace_sha256')):
            if sha(folder/file)!=ep[key]:raise ValueError('completed replay evidence changed')
            inputs[str(folder/file)]=ep[key]
    suffix=('motion4500_wm_contactforce_' + ('a' if contact_force_arm=='control' else 'b')) if contact_force_arm else \
        'motion4500_wm_contact25' if contact_focus else 'motion4500_wm_window80_more1000' if continue_window80 else 'motion4500_wm_window80'
    s=copy.deepcopy(parent);name=new_experiment_name(suffix)
    target=ROOT/'configs/experiments'/name;newrun=ROOT/'runs'/name
    ancestor_bytes=sum(f.stat().st_size for f in run.rglob('*') if f.is_file())
    allowance=parent['limits']['artifacts_bytes']-ancestor_bytes
    required_bytes=1024**3
    if contact_focus or contact_force_arm:
        # Same22.2M architecture, no new collection: calculate the actual existing
        # train save guard (7xFP32 parameters), plus8MiB metadata/terminal reserve.
        # This replaces a coarse1GiB estimate, never the3GiB cumulative cap.
        import torch
        guard_world=init_world if contact_force_arm else world
        saved=torch.load(guard_world,map_location='cpu',weights_only=False)
        if contact_focus and saved['config']!=audit['world_config']:
            raise ValueError('audited World Model architecture differs')
        parameter_bytes=sum(v.numel()*v.element_size() for k,v in saved['model'].items() if k.startswith('members.'))
        required_bytes=7*parameter_bytes+8*1024**2
    granted=pilot_artifacts_bytes if contact_force_arm else None
    if granted is not None and (type(granted) is not int or granted<required_bytes):
        raise OSError('explicit pilot artifact allowance below the measured save guard')
    if (granted or allowance)<required_bytes:
        raise OSError('insufficient measured remaining lineage artifact capacity; no increase or deletion')
    s.update(experiment=name,run=str(newrun),code_inputs=code_identities(),resume_inputs=inputs,
        optimization_version='wheel_leg_shared_dynamics_bptt_window80_v2',
        prediction_continuation=dict(parent_spec=str(p),parent_spec_sha256=sha(p),parent_run=str(run),
            event_id=ready['event_id'],world=str(world),world_sha256=sha(world),dataset=str(dataset),
            collection_root=str(collection_root),engineering=str(engineering),
            engineering_spec_sha256=engineering_evidence['spec_sha256'],
            review=str(review_path),single_optimization_change='mechanical supervised rollout8ms ->80ms',
            preserved='network, normalization, Adam, episode bootstrap, batch256, LR, reward, Actor, physical plant and acceptance thresholds',
            parent_used=book['used'],parent_wall_s=book['wall_used_s'],parent_artifacts_bytes=ancestor_bytes,
            storage_scope='parent retained bytes plus new artifact allowance remains within original3GiB'))
    s['world_model'].update(fit_window_substeps=80,activation_checkpoint=True,training_revision='supervised_window80_v2')
    if reviewed:s['reviewed_code_changes']=reviewed
    if deletions:s['reviewed_source_deletions']=deletions
    if continue_window80 or contact_focus:
        s['optimization_version']='wheel_leg_shared_dynamics_bptt_contact25_v4' if contact_focus else 'wheel_leg_shared_dynamics_bptt_window80_continue_v3'
        # Prior checkpoints did not save the WM replay RNG. Use an explicit fresh
        # batch stream, never claim a complete RNG restore or independent training seed.
        s['world_model']['seed']=2017 if contact_focus else 1017
        s['workflow'].update(prediction_only=True,rounds_max=1,
            stop='after one1000-step WM block and fixed held-out prediction; no Actor/Critic/physics stages')
        s['workflow']['initial'].update(critic_optimizer_steps=0,actor_optimizer_steps=0)
        s['limits'].update({k:(1000 if k=='wm_optimizer_steps' else 0) for k in COUNTERS})
        s['prediction_continuation'].update(single_optimization_change='one further1000 optimizer dispatches at unchanged80ms supervision',
            preserved='network, weights, normalization, Adam, episode bootstrap, batch256, LR, loss, reward, Actor, physical plant and acceptance thresholds',
            sampler='fresh explicit seed1017; prior WM replay RNG absent; not an independent policy training seed',
            policy_gate_pending=['physical action-response amplitude/eligibility','independent Critic value validation'])
        if contact_focus:
            s['world_model'].update(sampling='contact_event25_v1',training_revision='supervised_window80_contact25_v4')
            s['prediction_continuation'].update(
                single_optimization_change='75% original uniform episode sampling plus25% sustained contact-loss/recontact train starts',
                sampler='fresh explicit seed2017; member bootstrap unchanged; validation sampling unchanged',
                preserved='network, checkpoint weights, Adam, normalization, episode bootstrap, batch256, LR,80ms loss, reward, physical plant and acceptance thresholds',
                contact_event_definition='either wheel<=2N for>=5ms; expand20ms each side; starts>=100ms; full80ms future; train split only',
                measured_save_guard_bytes=required_bytes,
                checkpoint_scope='parent best loss at local800; all1000 actual optimizer dispatches remain charged')
    if contact_force_arm:
        # Frozen pair declaration: the ONLY difference between the arms is the loss block.
        import torch
        if handoff:
            # Never notify the terminated owner session; the terminal files are the delivery.
            s['notifications']=dict(enabled=False,owner_session=session,target_session=session,
                events=['terminal','gate_stop','exception'],
                transport='disabled: executor is not a HAPI session; terminal files in the run directory are the delivery',
                note='historical owner session terminated on a provider quota error; it is never a notification target')
            s['owner_handoff']=dict(path=str(Path(owner_handoff).resolve()),sha256=sha(owner_handoff),
                owner_session=session,parent_spec_sha256=sha(parent_path),
                parent_owner_session=auth.get('owner_session'),
                user_delegation=handoff['delegation']['user_text'],
                assumed_unix_s=handoff['delegation']['received_unix_s'],
                note='frozen handoff consumed by this declaration; hashes re-verified at every run')
        initialization=torch.load(init_world,map_location='cpu',weights_only=False)
        s['optimization_version']='wheel_leg_shared_dynamics_bptt_contactforce_v5'
        s['prediction_continuation'].update(world=str(init_world),world_sha256=sha(init_world),
            world_provenance_spec_sha256=initialization.get('provenance',{}).get('spec_sha256'),
            initialization_checkpoint=str(init_world),initialization_sha256=sha(init_world),
            parent_settled_world=str(world),parent_settled_world_sha256=sha(world),
            initialization_scope='both arms resume models/Adam/normalizer/episode bootstrap from the declared '
                                 'initialization checkpoint; the settled parent supplies only lineage/accounting')
        s['world_model'].update(seed=3017,sampling='uniform_episode_v1',
            training_revision='supervised_window80_contactforce_' + ('control_v5' if contact_force_arm=='control' else 'force_v5'))
        s['workflow'].update(prediction_only=True,rounds_max=1,
            stop='after the declared300 member dispatches and fixed held-out prediction; no Actor/Critic/physics stages')
        s['workflow']['initial'].update(wm_optimizer_steps=300,critic_optimizer_steps=0,actor_optimizer_steps=0)
        s['limits'].update({k:(300 if k=='wm_optimizer_steps' else 0) for k in COUNTERS})
        s['prediction_continuation'].update(
            single_optimization_change=('none: control arm of the paired contact force pilot' if contact_force_arm=='control'
                else 'add lambda_force*huber((softplus(wheel rows)-F_real)/F_scale) on the two wheel contact outputs'),
            preserved=('network 1024x3 inputs3030 outputs91, normalization, Adam, episode bootstrap, batch256, LR, '
                       '80ms window, uniform_episode_v1 sampling, dv/warm/contact inverse-softplus supervision, closure/q/rotation loss, '
                       'reward, Actor, physical plant and prediction acceptance thresholds'),
            pair=dict(version='contact_force_pair_v1',
                pair_id=Path(p).parent.name,
                arms=dict(a='control (loss unchanged)',b='force (declared physical force term added)'),
                identity='the two arms differ only in world_model.contact_force_supervision',
                same=dict(initialization_world=str(init_world),initialization_world_sha256=sha(init_world),
                          dataset=str(dataset),dataset_sha256=sha(dataset),
                          seed=s['world_model']['seed'],sampler='uniform_episode_v1',
                          member_dispatches=300,batch_size=s['world_model']['batch_size'],
                          validation_sampling='uniform_episode_v1',validation_seed=991,
                          performance_profile='closure_no_checkpoint_v1'),
                frozen_force_loss=CONTACT_FORCE_FROZEN),
            scope='one300-dispatch arm; total paired pilot600 member dispatches; not extendable by the executor')
        s['prediction_continuation']['policy_gate_pending']=['physical action-response amplitude/eligibility',
            'independent Critic value validation']
    if not contact_force_arm:
        s['workflow']['initial']['wm_optimizer_steps']=1000
    s['workflow']['engineering_physics_steps']=0
    s['workflow']['physical_budget_breakdown'].update(initial_collection_steps=0)
    s['limits']['artifacts_bytes']=granted or allowance
    if contact_force_arm:
        s['storage_grant']=dict(version='explicit_pair_pilot_artifact_allowance_v1'
            if granted else 'lineage_remaining_allowance_no_increase',
            artifacts_bytes=s['limits']['artifacts_bytes'],lineage_remaining_bytes=allowance,
            measured_save_guard_bytes=required_bytes,parent_artifacts_bytes=ancestor_bytes,
            basis=('explicit reviewer/grant value' if granted else 'no increase over the parent lineage allowance'))
    s['world_model']=contact_force_supervision(s['world_model'],
        CONTACT_FORCE_FROZEN if contact_force_arm=='force' else (dict(version='none') if contact_force_arm else None))
    s['world_model']=supervision_performance(s['world_model'],supervision_performance_version)
    s['authorization'].update(authorized_unix_s=time.time(),continuation_basis='current-session unlimited new WM authorization; inspected prediction evidence',
        review_request='请读取报告复盘，并按当前有效用户授权处理后续工作。')
    if handoff:
        s['authorization'].update(owner_session=session,hapi_session=handoff['new_owner'].get('hapi_session'),
            session_kind=handoff['new_owner']['kind'],
            continuation_basis='owner handoff: '+handoff['delegation']['user_text'][:240],
            handoff_sha256=sha(owner_handoff))
    if plan_only:
        # Reviewable payload with the exact numbers, zero files written.
        return dict(spec=s,experiment=name,run=str(newrun),config_dir=str(target),written=0,
            artifacts_bytes=s['limits']['artifacts_bytes'],measured_save_guard_bytes=required_bytes,
            lineage_remaining_bytes=allowance,parent_artifacts_bytes=ancestor_bytes,
            free_disk_bytes=shutil.disk_usage(ROOT).free,
            min_free_bytes=s['limits']['min_free_bytes'],
            executing_session=session,parent_owner_session=auth.get('owner_session'),
            owner_handoff_required=not owner_matches,
            storage_ready=bool(shutil.disk_usage(ROOT).free>=s['limits']['min_free_bytes']+required_bytes))
    target.mkdir(parents=True,exist_ok=False)
    write_json(target/'spec.json',s)
    historical={k:book['used'][k]+book.get('historical_used',{}).get(k,0) for k in COUNTERS}
    write_json(target/'budget.json',dict(version=VERSION,spec_sha256=sha(target/'spec.json'),authorization='AUTHORIZED',
        limits=s['limits'],used={k:0 for k in COUNTERS},wall_used_s=0.,attempts={},historical_used=historical,
        historical_wall_s=book['wall_used_s']+book.get('historical_wall_s',0.),
        historical_artifacts_bytes=ancestor_bytes+book.get('historical_artifacts_bytes',0)))
    write_json(target/'manifest.json',dict(version=VERSION,state='AUTHORIZED_NOT_SUBMITTED',runtime_attempts=0,
        spec_sha256=sha(target/'spec.json'),source_inputs=s['source_inputs'],code_inputs=s['code_inputs'],resume_inputs=inputs))
    return target/'spec.json'
