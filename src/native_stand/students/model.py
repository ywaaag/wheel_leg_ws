"""Fresh student networks, bounded mean and unsquashed Gaussian sampling."""
import torch
from torch import nn
from .interface import ACTOR_DIM,CRITIC_DIM,VERSION
class Student(nn.Module):
    is_recurrent=False
    def __init__(self,actor=True,version=VERSION,critic_privileged_version=None,network_architecture=None):
        super().__init__();self.is_actor=actor;self.interface_version=version
        if version not in (VERSION,"multiskill_mcu_v2"):raise ValueError("Unknown student interface version")
        if critic_privileged_version not in (None,'standing_privileged49_v1','motion_privileged49_v1','motion_spin_privileged61_v1','motion_symmetry_privileged85_v1','motion_timing_privileged110_v1','motion_antiosc_privileged140_v1','motion_height_privileged144_v1','motion_height_robust_privileged184_v1','motion_mcu_motorqueue_privileged312_v1','motion_standing_delay_privileged411_v1','recovery_privileged64_v1','recovery_contact64_v2','motion_compliance_privileged64_v1') or (critic_privileged_version and version!='multiskill_mcu_v2'):raise ValueError('unknown Critic privileged version')
        if critic_privileged_version in ('motion_antiosc_privileged140_v1','motion_height_privileged144_v1','motion_height_robust_privileged184_v1','motion_mcu_motorqueue_privileged312_v1','motion_standing_delay_privileged411_v1') and network_architecture is None:raise ValueError('extended Critic requires explicit architecture')
        dims=([31,128,64,32,6] if actor else [61 if critic_privileged_version=='motion_spin_privileged61_v1' else 64 if critic_privileged_version in ('recovery_privileged64_v1','recovery_contact64_v2') else 49 if critic_privileged_version else 40,256,128,64,1]) if version=="multiskill_mcu_v2" else ([ACTOR_DIM,256,128,64,6] if actor else [CRITIC_DIM,256,128,64,1])
        if not actor and critic_privileged_version=='motion_compliance_privileged64_v1':dims=[64,256,128,64,1]
        self.network_architecture=network_architecture
        if network_architecture is not None:
            from .motion_large_combo import network
            from .recovery_large import network as recovery_network
            from .motion_spin_privileged import network as spin_network
            from .contact_quality import network as contact_network
            from .motion_symmetry import network as symmetry_network
            from .motion_timing import network as timing_network,CRITIC as TIMING_CRITIC
            from .motion_antiosc_privileged import network as anti_network,VERSION as ANTI_CRITIC
            from .motion_height import network as height_network,CRITIC as HEIGHT_CRITIC
            from .motion_fixed_height import network as fixed_height_network, CRITIC as FIXED_HEIGHT_CRITIC
            from .motion_mcu_privileged import network as mcu_network, VERSION as MCU_CRITIC
            from .actor_widening import wide_network
            from .motion_standing_delay import network as delay_network,CRITIC as DELAY_CRITIC
            if network_architecture==delay_network():
                if version!='multiskill_mcu_v2' or (not actor and critic_privileged_version!=DELAY_CRITIC):raise ValueError('explicit delay411 network required')
                dims=[31,2048,1024,512,6] if actor else [411,512,512,256,1]
            elif network_architecture==wide_network(2048):
                if version!='multiskill_mcu_v2' or (not actor and critic_privileged_version!=MCU_CRITIC):raise ValueError('explicit large MCU312 network required')
                dims=[31,2048,1024,512,6] if actor else [312,512,512,256,1]
            elif network_architecture==wide_network():
                if version!='multiskill_mcu_v2' or (not actor and critic_privileged_version!=MCU_CRITIC):raise ValueError('explicit wide MCU312 network required')
                dims=[31,1024,512,256,6] if actor else [312,512,512,256,1]
            elif network_architecture==mcu_network():
                if version!='multiskill_mcu_v2' or (not actor and critic_privileged_version!=MCU_CRITIC):raise ValueError('explicit MCU312 network required')
                dims=[31,512,256,128,6] if actor else [312,512,512,256,1]
            elif network_architecture==fixed_height_network():
                if version!='multiskill_mcu_v2' or (not actor and critic_privileged_version!=FIXED_HEIGHT_CRITIC):raise ValueError('explicit fixed-height184 required')
                dims=[31,512,256,128,6] if actor else [184,512,512,256,1]
            elif network_architecture==height_network():
                if version!='multiskill_mcu_v2' or (not actor and critic_privileged_version!=HEIGHT_CRITIC):raise ValueError('explicit Motion height144 required')
                dims=[31,512,256,128,6] if actor else [144,512,512,256,1]
            elif network_architecture==anti_network():
                if version!='multiskill_mcu_v2' or (not actor and critic_privileged_version!=ANTI_CRITIC):raise ValueError('explicit Motion anti-oscillation140 required')
                dims=[31,512,256,128,6] if actor else [140,512,512,256,1]
            elif network_architecture==timing_network():
                if version!='multiskill_mcu_v2' or (not actor and critic_privileged_version!=TIMING_CRITIC):raise ValueError('explicit Motion timing110 required')
                dims=[31,512,256,128,6] if actor else [110,512,512,256,1]
            elif network_architecture==symmetry_network():
                if version!='multiskill_mcu_v2' or (not actor and critic_privileged_version!='motion_symmetry_privileged85_v1'):raise ValueError('explicit symmetry85 network required')
                dims=[31,512,256,128,6] if actor else [85,512,512,256,1]
            elif network_architecture==spin_network():
                if version!='multiskill_mcu_v2' or (not actor and critic_privileged_version!='motion_spin_privileged61_v1'):raise ValueError('explicit Motion61 network declaration required')
                dims=[31,512,256,128,6] if actor else [61,512,512,256,1]
            elif network_architecture==contact_network():
                if version!='multiskill_mcu_v2' or (not actor and critic_privileged_version!='recovery_contact64_v2'):raise ValueError('explicit contact64 Critic required')
                dims=[31,512,256,128,6] if actor else [64,512,512,256,1]
            elif network_architecture==recovery_network():
                if version!='multiskill_mcu_v2' or (not actor and critic_privileged_version!='recovery_privileged64_v1'):
                    raise ValueError('explicit large Recovery64 network declaration required')
                dims=[31,512,256,128,6] if actor else [64,512,512,256,1]
            else:
                if network_architecture!=network() or version!='multiskill_mcu_v2' or (not actor and critic_privileged_version!='motion_privileged49_v1'):
                    raise ValueError('explicit large Motion network declaration required')
                dims=[31,512,256,128,6] if actor else [49,512,512,256,1]
        layers=[]
        for i,(a,b) in enumerate(zip(dims,dims[1:])):
            layers.append(nn.Linear(a,b))
            if i<len(dims)-2:layers.append(nn.ELU(alpha=1.))
        self.mlp=nn.Sequential(*layers)
        if actor:self.log_std=nn.Parameter(torch.tensor([.05/1.5,.05/1.5,2/100,.05/1.5,.05/1.5,2/100]).log())
    def mean(self,x):return self.mlp(x).tanh()
    def distribution(self,x):return torch.distributions.Normal(self.mean(x),self.log_std.exp())
    def forward(self,x,stochastic_output=False):
        if isinstance(x,dict) or hasattr(x,'keys'):x=x['student_actor' if self.is_actor else 'student_critic']
        if not self.is_actor:return self.mlp(x)
        if stochastic_output:
            self._current_distribution=self.distribution(x)
            return self._current_distribution.sample()
        return self.mean(x)
    def log_prob(self,x,raw_action):return self.distribution(x).log_prob(raw_action).sum(-1)

    def get_hidden_state(self):return None
    def reset(self,dones=None):pass
    def update_normalization(self,obs):pass
    def get_output_log_prob(self,raw):return self._current_distribution.log_prob(raw).sum(-1)
    @property
    def output_distribution_params(self):return (self._current_distribution.mean,self._current_distribution.stddev)
    @property
    def output_entropy(self):return self._current_distribution.entropy().sum(-1)
    @property
    def output_mean(self):return self._current_distribution.mean
    @property
    def output_std(self):return self._current_distribution.stddev
    @staticmethod
    def get_kl_divergence(old,new):
        return torch.distributions.kl_divergence(torch.distributions.Normal(*old),torch.distributions.Normal(*new)).sum(-1)

def load(path,role=None):
    data=torch.load(path,map_location='cpu',weights_only=False)
    if data.get('interface_version') not in (VERSION,'multiskill_mcu_v2') or (role and data['role']!=role):raise ValueError('student identity/role differs')
    if data.get('recipe_version')=='motion31_compliance_low_speed_ppo_v1':
        from .motion_compliance_training import validate_checkpoint
        validate_checkpoint(data)
    elif data.get('recipe_version'):
        from .training_recipe import validate_instance
        recipe=validate_instance(data.get('training_recipe'),data.get('source_sha256'))
        if recipe['version']!=data['recipe_version']:raise ValueError('student checkpoint recipe version differs')
        if data.get('temporal_regularization') is not None:
            from .temporal import declaration as temporal_declaration
            if data['temporal_regularization']!=temporal_declaration(data['temporal_regularization'].get('arm')):raise ValueError('temporal checkpoint declaration differs')
        for key in ('source_sha256','dataset_sha256','manifest_sha256','asset_sha256','spec_sha256','source_code_sha256'):
            if not isinstance(data.get(key),str) or len(data[key])!=64:raise ValueError('training checkpoint missing bound '+key)
        if data['source_sha256']!=recipe['source_actor_sha256'] or data.get('BC_coefficient')!=0. or data.get('fixed_std') is not True:raise ValueError('student trained checkpoint intervention differs')
        if recipe.get('source_policy_retention') is not None and data.get('source_policy_retention')!=recipe['source_policy_retention']:raise ValueError('Motion checkpoint source retention declaration missing/differs')
    with torch.random.fork_rng(devices=[]):
        if data.get('initialization_kind'):
            from .actor_widening import WIDE_INITIALIZATION, LARGER_INITIALIZATION, validate_wide_candidate
            if data['initialization_kind'] in (WIDE_INITIALIZATION,LARGER_INITIALIZATION):
                validate_wide_candidate(data)
            else:
                from .recovery_large import validate_candidate
                validate_candidate(data)
        elif data.get('network_architecture')!=data.get('training_recipe',{}).get('network_architecture'):
            raise ValueError('checkpoint/recipe network architecture differs')
        actor=Student(True,version=data["interface_version"],network_architecture=data.get('network_architecture'))
        from native_stand.command_v6.tools.model_adapter import ModelAdapter
        ModelAdapter(data).load(actor,'actor')
    return actor,data

def ppo_initialization(path):
    """Actor only; new critic and optimizer. BC Adam never imported."""
    actor,data=load(path)
    if data.get('initialization_kind')=='recovery31_function_preserving_widening_v1':
        raise ValueError('Recovery large initialization requires an explicit training recipe')
    critic=Student(False,version=data["interface_version"])
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=3e-5)
    return actor,critic,optimizer
