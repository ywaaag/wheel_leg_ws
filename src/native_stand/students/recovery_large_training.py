"""Explicit large Recovery stop recipe; legacy reward and interfaces remain fixed."""
import copy
import torch
from .recovery_large import network
from .recovery_broad_narrow_reward import FIELDS as BASE_FIELDS,configuration as base_configuration,reward as base_reward,huber,STOP_VERSION
VERSION='student_recovery_large_stop_ppo_v1'
REWARD='student_recovery_large_stop_reward_v1'
SOURCE='0f4bebb2a763d8a44b863bf6017f3388a999e475af8c58f1dee07857d7aa27cd'
FIELDS=BASE_FIELDS+('post_entry_zero_wide','post_entry_zero_narrow')

def configuration():
    c=copy.deepcopy(base_configuration(STOP_VERSION));c['version']=REWARD
    c['weights'].update(post_entry_zero_wide=.5,post_entry_zero_narrow=1.5)
    c['stop_shaping']={'clock':'first_confirmed_entry_elapsed; never restart by leaving envelope',
        'ramp_s':1.,'ramp':'smoothstep(clamp(elapsed/1,0,1))',
        'vx_huber_density_weight':-.1,'vx_huber_scale_mps':.2,
        'wide_sigma_mps':.5,'narrow_sigma_mps':.15,
        'pre_entry':'unchanged from broad_narrow v1; no zero-speed reward',
        'post_entry_vy_yaw':'unchanged stop_v2, weight -.5'}
    return c

def recipe(source_sha256=None):
    from .recovery_privileged import recipe as base,STOP_RECIPE
    if source_sha256 is not None and (not isinstance(source_sha256,str) or len(source_sha256)!=64):raise ValueError('large Recovery source SHA required')
    r=base(version=STOP_RECIPE)
    r.update(version=VERSION,source_actor_sha256=source_sha256 or SOURCE,reward=configuration(),learning_rate=3e-5,
        network_architecture=network(),milestones=[500,1000,1500,2000,2488],
        storage_max_bytes=4*2**30,checkpoint_every_updates=100,
        checkpoint_quick_evaluation={'version':'recovery_single_native_screen_v1','every_updates':100,
            'case_id':'student_native','simulated_envs':1,'scored_envs':1,'selection_eligible':False},
        selection_scope='same four full conditions: capability first, then mean absolute vx, earlier checkpoint',
        selected_rollouts_smoke=[1,6],selected_rollouts_formal=[])
    return r

def reward(*,entry_elapsed,**kwargs):
    c=kwargs.pop('reward_configuration',None)
    if c!=configuration():raise ValueError('large stop reward configuration differs')
    result=base_reward(**kwargs,reward_configuration=base_configuration(STOP_VERSION))
    height=kwargs['height'];elapsed=entry_elapsed
    if elapsed.shape!=height.shape or elapsed.device!=height.device or elapsed.dtype!=height.dtype or not torch.isfinite(elapsed).all() or (elapsed<0).any():raise ValueError('invalid first-entry elapsed clock')
    t=(elapsed/c['stop_shaping']['ramp_s']).clamp(0,1);alpha=t.square()*(3-2*t)
    I=kwargs['entered'].to(height.dtype);v=kwargs['velocity_vx_vy_yaw']
    raw=result['raw'].clone()
    # Existing column keeps -.5 weight; .2 factor makes vx effective weight -.1.
    raw[:,7]=I*(.2*alpha*huber(v[:,0]/.2)+huber(v[:,1]/.2)+huber(v[:,2]/.3))*result['eligible']
    gate=I*alpha*result['eligible']
    raw=torch.cat((raw,(gate*torch.exp(-(v[:,0]/.5).square()))[:,None],
        (gate*torch.exp(-(v[:,0]/.15).square()))[:,None]),-1)
    density=raw*height.new_tensor([c['weights'][k] for k in FIELDS]);dt=density*c['dt']
    result.update(version=REWARD,fields=FIELDS,raw=raw,weighted_density=density,continuous_dt=dt,total=dt.sum(-1)+result['events'].sum(-1))
    result['diagnostics']['stop_alpha']=alpha
    if not torch.isfinite(result['total']).all():raise ValueError('nonfinite large Recovery reward')
    return result

def selection_rank(row):
    """Ability remains first. Only complete full matrices are candidates."""
    envs=row['per_env']
    if len(envs)!=32:raise ValueError('large stop selection requires full four-condition32 matrix')
    velocities=[float(e['velocity_means'][0]) for e in envs if e.get('velocity_means') is not None]
    quality=-sum(velocities)/len(velocities) if len(velocities)==32 else float('-inf')
    return (sum(e['accepted'] for e in envs),sum(e.get('entry_index') is not None for e in envs),
        sum(e.get('valid_hold_prefix_s',0.) for e in envs),quality,-row.get('update',0))

def quick_due(spec,phase,update,*,full_assessment):
    if spec.get('student_training',{}).get('version') not in (VERSION,'student_recovery_contact_quality_ppo_v1'):return False
    interval=spec['student_training']['checkpoint_quick_evaluation']['every_updates']
    return phase=='formal' and update>0 and update%interval==0 and not full_assessment
