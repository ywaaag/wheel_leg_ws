"""CPU preparation contract for a future explicitly authorized MuJoCo PPO task.

No simulator, filesystem, firmware globals or optimizer are used here. A valid
declaration is not evidence that the training adapter has been implemented.
"""
import torch

VERSION='motion31_compliance_low_speed_ppo_v1'
CRITIC='motion_compliance_privileged64_v1'
PROFILES=(
    {'name':'R','K':0.,'B':0.,'rotor_J':0.,'load_armature':.012},
    {'name':'F20','K':20.,'B':1.,'rotor_J':.009,'load_armature':.003},
    {'name':'F40','K':40.,'B':.4,'rotor_J':.009,'load_armature':.003},
    {'name':'F120','K':120.,'B':.4,'rotor_J':.009,'load_armature':.003},
)


def declaration(*,updates=300):
    if type(updates) is not int or updates<12:raise ValueError('inclusive online budget must include smoke12')
    result={
        'version':VERSION,'status':'CPU_PREPARATION','runtime_recipe_ready':False,
        'backend':'MuJoCo_CPU','interface':'multiskill_mcu_v2',
        'actor':[31,128,64,32,6],'critic':[64,256,128,64,1],'critic_version':CRITIC,
        'physics_PD_policy_ms':[1,5,10],'PD':{'leg_Kp':20.,'leg_Kd':2.,'wheel_Kv':.15},
        'height_target_m':.34,'force_route':'separated',
        'profiles':[dict(p) for p in PROFILES],
        'sampler':'8 spawn workers; one C library per process; no threads/shared C globals',
        'source_sampling':'source_packets_v1; leg2ms/wheel1ms; no added fixed feedback/transport age',
        'algorithm':{'implementation':'ControlledPPO','lr':3e-5,'epochs':4,'minibatches':4,
            'gamma':.99,'lambda':.95,'clip':.2,'value_coef':.5,'entropy':.001,
            'max_grad_norm':1.,'BC':0.,'Teacher_KL':0.,'critic_only_updates':5,
            'fixed_std_physical':[.05,.05,2.,.05,.05,2.]},
        'budget_proposal':{'online_updates':updates,'smoke':12,'formal':updates-12,'envs':8,
            'steps_per_env':128,'raw_rollout_samples':updates*8*128,'physical_steps_1ms':max(16000000,updates*53334),
            'wall_s':None if updates==3000 else 14400,'artifact_max_bytes':1073741824},
        'mask':'initialization/frozen-policy-prefix excluded from Actor and Critic; keep GAE timeline',
        'smoke_reset_gap':{'per_env_policy_seconds':15.36,'episode_seconds':30.,
            'timeout_coverage_guaranteed':False,
            'resolution':'explicit bounded reset/timeout engineering test before optimizer; charge physical steps; do not invent natural timeout coverage'},
        'entry_control':'half direct; half frozen4500 prefix1s; state/cache/queues continuous',
        'evaluation':{'checkpoint_every':50,'quick_every':50,'full_every':100,
            'profiles':4,'commands':5,'initial_yaws':[0.,.015,-.015],'controlled_seconds':30.,
            'comparators':['old_actor_original_PD','old_actor_new_PD','new_actor_new_PD']},
        'remaining_wiring':['MuJoCo training adapter and worker lifecycle',
            'raw Gaussian publication boundary and post-physics reward/pre-reset terminal',
            'explicit Critic64 producer/model/storage/checkpoint dispatch',
            'source reward measurement-layer audit and snapshot',
            'supervisor backend/reservation/settlement/notification dispatch',
            'bounded source/smoke/reset/isolation physical readbacks',
            'independent evaluation and PPO export identity'],
    }
    return result


def critic_input(actor,*,height,velocity_native,contacts,controlled_s,load_q,load_dq,
                 encoder_q,encoder_dq,K,B,rotor_J,load_armature,version):
    """64D, explicit new semantics; observed31 prefix is kept byte-for-byte."""
    if version!=CRITIC or actor.ndim!=2 or actor.shape[1]!=31:
        raise ValueError('explicit motion compliance64 contract required')
    n=len(actor)
    values={'height':(height,(n,1)),'velocity_native':(velocity_native,(n,3)),
        'contacts':(contacts,(n,2)),'controlled_s':(controlled_s,(n,1)),
        'load_q':(load_q,(n,4)),'load_dq':(load_dq,(n,4)),
        'encoder_q':(encoder_q,(n,4)),'encoder_dq':(encoder_dq,(n,4)),
        'K':(K,(n,4)),'B':(B,(n,4)),'rotor_J':(rotor_J,(n,1)),
        'load_armature':(load_armature,(n,1))}
    for key,(value,shape) in values.items():
        if value.shape!=shape or value.device!=actor.device or value.dtype!=actor.dtype:
            raise ValueError('compliance Critic field shape/device/dtype: '+key)
    if not all(bool(torch.isfinite(value).all()) for value,_ in values.values()) or not bool(torch.isfinite(actor).all()):
        raise ValueError('nonfinite compliance Critic input')
    rigid=(K==0).all(-1,keepdim=True)
    extra=torch.cat((height/.5,velocity_native/4.,contacts/100.,controlled_s/30.,
        load_q/1.5,load_dq*.05,
        torch.where(rigid,0.,encoder_q-load_q)/.1,
        torch.where(rigid,0.,encoder_dq-load_dq),K/120.,torch.where(rigid,0.,B),
        torch.where(rigid,0.,rotor_J)/.012,load_armature/.012),-1)
    return torch.cat((actor,extra),-1)


def command_set(stage):
    if stage==0:return [[0.,0.]]
    if stage==1:return [[0.,0.],[.25,0.],[-.25,0.]]
    if stage in (2,3):
        c=.25 if stage==2 else .5
        return [[0.,0.],[c,0.],[-c,0.],[0.,c],[0.,-c]]
    raise ValueError('unknown compliance stage')


def assignments(stage,cycle):
    """Eight equal profile slots, 50%zero and balanced nonzero over cycles.

    Cycle is an explicit sampler counter, not policy RNG. It changes at reset,
    never by switching mechanics inside an episode. Direct/prefix is crossed
    with profile and command instead of becoming another hidden task label.
    """
    if not isinstance(cycle,int) or cycle<0:raise ValueError('nonnegative cycle required')
    commands=command_set(stage);rows=[]
    for i in range(8):
        profile=i//2
        zero=(i+cycle)%2==0
        index=0 if zero or stage==0 else 1+(profile+cycle//2)%(len(commands)-1)
        rows.append({'env':i,'profile':PROFILES[profile]['name'],
            'command':commands[index]+[.34],'prefix_s':float((i+cycle//4)%2)})
    return rows


def validate_checkpoint(data):
    if (data.get('recipe_version')!=VERSION or data.get('critic_privileged_version')!=CRITIC
        or data.get('interface_version')!='multiskill_mcu_v2' or data.get('fixed_std') is not True
        or data.get('BC_coefficient')!=0. or data.get('training_recipe',{}).get('version')!=VERSION):
        raise ValueError('explicit compliance checkpoint semantics differ')
    for key in ('source_sha256','spec_sha256','source_code_sha256'):
        if not isinstance(data.get(key),str) or len(data[key])!=64:raise ValueError('missing compliance identity '+key)
    if data['actor']['mlp.0.weight'].shape!=(128,31) or data['critic']['mlp.0.weight'].shape!=(256,64):
        raise ValueError('compliance checkpoint network differs')
    return True
