"""Future PPO assembly only; no online rollout or optimizer step on import."""
import torch
from .model import load,Student

def assemble(checkpoint,observations,*,steps_per_env=128,recipe=None):
    from rsl_rl.storage import RolloutStorage
    from native_stand.recovery.controlled_ppo import ControlledPPO,KEY
    actor,data=load(checkpoint)
    if data.get('initialization_kind')=='recovery31_function_preserving_widening_v1' and (recipe or {}).get('version') not in ('student_recovery_large_stop_ppo_v1','student_recovery_contact_quality_ppo_v1'):
        raise ValueError('Recovery large initialization requires a new explicit training recipe; no inherited small/Motion recipe')
    reference_actor=actor
    if (recipe or {}).get('continuation_teacher'):
        teacher=recipe['continuation_teacher']
        from .data import sha
        if sha(teacher['checkpoint'])!=teacher['sha256']:raise ValueError('retention teacher SHA differs')
        reference_actor,_=load(teacher['checkpoint'])
    architecture=(recipe or {}).get('network_architecture')
    if architecture is not None and (recipe or {}).get('version') not in ('student_recovery_large_stop_ppo_v1','student_recovery_contact_quality_ppo_v1'):
        if architecture.get('version')=='motion31_actor2048_delaycritic411_v1':
            if (recipe or {}).get('course_version')!='motion31_standing_sensor20_target5_motor20_v1':
                raise ValueError('delay411 requires the explicit standing delay recipe')
            from .motion_standing_delay import expand_source as expanded_source
        elif architecture.get('version')=='motion31_mcu_actor2048_critic312_v1':
            if (recipe or {}).get('course_version')!='motion31_mcu_highspeed_standing2048_v1':
                raise ValueError('Actor2048 requires the explicit high-speed training recipe')
            from .motion_highspeed_standing import expanded_source
        elif architecture.get('version')=='motion31_mcu_motorqueue_critic312_v1':
            from .motion_mcu_privileged import expand_source as expanded_source
        elif architecture.get('version')=='motion31_fixed_height_critic184_v1':
            from .motion_fixed_height import expand_source as expanded_source
        elif architecture.get('version')=='motion31_height_critic144_v1':
            from .motion_height import expand_source as expanded_source
        elif architecture.get('version')=='motion31_antiosc_critic140_v1':
            from .motion_antiosc_privileged import expand_source as expanded_source
        elif architecture.get('version')=='motion31_timing_large_actor_critic_v1':
            from .motion_timing import expand_source as expanded_source
        elif architecture.get('version')=='motion31_symmetry_large_actor_critic_v3':
            from .motion_symmetry import expand_source as expanded_source
        elif architecture.get('version')=='motion31_spin_large_actor_critic_v2':
            from .motion_spin_privileged import expand_source as expanded_source
        else:
            from .motion_large_combo import expanded_source
        expanded=expanded_source(data)
        with torch.random.fork_rng(devices=[]):
            actor=Student(True,version=data['interface_version'],network_architecture=architecture)
        actor.load_state_dict(expanded['actor'])
    if (recipe or {}).get('version')=='student_recovery_contact_quality_ppo_v1':
        actor.network_architecture=architecture  # Same loaded31D Actor weights; explicit new Critic semantics.
    mcu=data['interface_version']=='multiskill_mcu_v2'
    if mcu:
        from .training_recipe import validate_instance
        if recipe is None:raise ValueError('MCU v2 reward undefined without explicit training recipe')
        validate_instance(recipe,(recipe or {}).get('source_actor_sha256'))
        actor.log_std.requires_grad_(False)
    critic=Student(False,version=data['interface_version'],critic_privileged_version=(recipe or {}).get('critic_privileged_version'),network_architecture=architecture)
    if (recipe or {}).get('critic')=='source_same_semantic_contact64_v2':
        from .recovery_dr import SOURCE as DR_SOURCE
        if recipe.get('source_actor_sha256')!=DR_SOURCE or data.get('critic_privileged_version')!='recovery_contact64_v2':raise ValueError('DR source Critic semantic binding differs')
        critic.load_state_dict(data['critic'],strict=True)
    critic_dim=411 if mcu and (recipe or {}).get('critic_privileged_version')=='motion_standing_delay_privileged411_v1' else 312 if mcu and (recipe or {}).get('critic_privileged_version')=='motion_mcu_motorqueue_privileged312_v1' else 184 if mcu and (recipe or {}).get('critic_privileged_version')=='motion_height_robust_privileged184_v1' else 144 if mcu and (recipe or {}).get('critic_privileged_version')=='motion_height_privileged144_v1' else 140 if mcu and (recipe or {}).get('critic_privileged_version')=='motion_antiosc_privileged140_v1' else 110 if mcu and (recipe or {}).get('critic_privileged_version')=='motion_timing_privileged110_v1' else 85 if mcu and (recipe or {}).get('critic_privileged_version')=='motion_symmetry_privileged85_v1' else 61 if mcu and (recipe or {}).get('critic_privileged_version')=='motion_spin_privileged61_v1' else 64 if mcu and (recipe or {}).get('critic_privileged_version') in ('recovery_privileged64_v1','recovery_contact64_v2') else 49 if mcu and (recipe or {}).get('critic_privileged_version') in ('standing_privileged49_v1','motion_privileged49_v1') else 40 if mcu else 1191
    if observations['student_actor'].shape[-1]!=(31 if mcu else 1182) or observations['student_critic'].shape[-1]!=critic_dim:raise ValueError('Student PPO observation dimensions differ')
    if KEY not in observations or observations[KEY].dtype!=torch.bool:raise ValueError('Student PPO requires pre-action control metadata')
    device=str(observations['student_actor'].device)
    storage=RolloutStorage('rl',observations.batch_size[0],steps_per_env,observations,[6],device=device)
    cls=ControlledPPO
    retention=(recipe or {}).get('source_policy_retention')
    if retention is not None:
        from .motion_source_retention import ppo_class,KEY as RETENTION_KEY
        if RETENTION_KEY not in observations:raise ValueError('Motion retention metadata required at storage assembly')
        cls=ppo_class()
    ppo=cls(actor,critic,storage,learning_rate=(recipe or {}).get('learning_rate',3e-5),schedule='fixed',num_learning_epochs=4,num_mini_batches=4,entropy_coef=.001,clip_param=.2,gamma=.99,lam=.95,value_loss_coef=.5,max_grad_norm=1.,use_clipped_value_loss=False,device=device)
    if retention is not None:
        from .motion_source_retention import SourcePolicyRetention
        reference_actor.log_std.requires_grad_(False)
        reference_actor=reference_actor.to(device)
        ppo.source_retention=SourcePolicyRetention(reference_actor,retention)
    if architecture is not None:
        ppo.record_gradient_norms=True
    if (recipe or {}).get('course_version') in ('motion31_mcu_handover_pd_age_v1','motion31_mcu_height_pd_parallel_v1','motion31_mcu_highspeed_standing2048_v1','motion31_standing_sensor20_target5_motor20_v1'):
        ppo.critic_control_only=True
    if ppo.optimizer.state:raise RuntimeError('New student PPO Adam must be empty')
    return ppo

def demonstration_loss(actor,independent_demo_batch):
    """Independent BC data only; never added to on-policy rollout storage."""
    from .distill import loss
    return loss(actor(independent_demo_batch['x']),independent_demo_batch['y'])
