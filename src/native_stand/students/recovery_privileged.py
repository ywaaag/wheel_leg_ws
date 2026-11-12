"""Recovery-only64D pre-action Critic input; no IO or actor fields added."""
import torch

VERSION='recovery_privileged64_v1'
RECIPE='student_recovery_broad_narrow_ppo_v1'
SOURCE='47f0a56747e288c313d5925ca8860d7a40d761eefb8cc68bb5832b7fb73dbf8f'
STOP_RECIPE='student_recovery_stop_ppo_v2'
STOP_SOURCE='66243a253e7f1d595d2950f5686ec6f715fd6d79edd481c52bb34abcb36a4d3a'


def contract():
    from .mcu_interface import contract as base
    c=base();c.update(critic_dim=64,critic_privileged_version=VERSION,height_command_m=.34)
    c['critic_privileged']=['root_height/.5:1','native_root_velocity/[4,4,2]:3',
        'wheel_resultant_force/100:2','clearance/.1:1','wheel_root_CAD_position/.5:6',
        'hip_to_wheel_length/.5:2','backward_length_rate/2:2','controlled_h0/.5:1',
        'controlled_entry_hold_clocks/[2.5,.2,10]:3','hold_mean_abs_angles/.1:2',
        'hold_mean_abs_vx_vy_yaw/[.5,.02,.03]:3','hold_saturation_fraction/.05:1',
        'hold_height_min/.5:1','hold_tilt_max/.18:1','hold_contact_min/100:2',
        'hold_closure_max/.005:1','hold_hard_failed:1']
    return c


class HoldExtrema:
    def __init__(self,n,device):
        self.values=torch.zeros(n,5,device=device)
        self.seen=torch.zeros(n,dtype=torch.bool,device=device)
    def reset(self,ids):self.values[ids]=0;self.seen[ids]=False
    def observe(self,height,tilt,contacts,closure,holding):
        x=torch.cat((height[:,None],tilt[:,None],contacts,closure[:,None]),-1)
        first=holding&~self.seen
        self.values[first]=x[first]
        mask=holding&self.seen
        self.values[mask,0]=torch.minimum(self.values[mask,0],height[mask])
        self.values[mask,1]=torch.maximum(self.values[mask,1],tilt[mask])
        self.values[mask,2:4]=torch.minimum(self.values[mask,2:4],contacts[mask])
        self.values[mask,4]=torch.maximum(self.values[mask,4],closure[mask])
        self.seen|=holding


def critic_input(actor,height,velocity,contacts,clearance,wheels,lengths,rates,h0,clock,extrema):
    n=len(actor);count=clock.count.to(actor.dtype);den=count.clamp_min(1)
    sums=torch.cat((clock.angle_sum/den[:,None]/.1,
        clock.velocity_sum/den[:,None]/actor.new_tensor([.5,.02,.03]),
        (clock.saturation_sum/den/.05)[:,None]),-1)
    limits=extrema.values/actor.new_tensor([.5,.18,100.,100.,.005])
    extra=torch.cat((height[:,None]/.5,velocity/actor.new_tensor([4.,4.,2.]),contacts/100.,
        clearance[:,None]/.1,wheels.reshape(n,6)/.5,lengths/.5,rates/2.,h0[:,None]/.5,
        clock.age[:,None]/2.5,clock.dwell[:,None].to(actor.dtype)*.01/.2,count[:,None]*.01/10.,
        sums,limits,clock.hard_failed[:,None].to(actor.dtype)),-1)
    out=torch.cat((actor,extra),-1)
    if actor.shape!=(n,31) or out.shape!=(n,64) or not torch.isfinite(out).all():raise ValueError('Recovery64 privileged input differs')
    return out


def recipe(source_sha256=None,*,version=RECIPE):
    from .training_recipe import declaration,VERSION as BASE
    from .recovery_broad_narrow_reward import configuration
    if version not in (RECIPE,STOP_RECIPE):raise ValueError('unknown Recovery64 recipe')
    source=STOP_SOURCE if version==STOP_RECIPE else SOURCE
    if source_sha256 not in (None,source):raise ValueError('Recovery64 fixed source differs')
    r=declaration(BASE)
    r.update(version=version,source_actor_sha256=source,reward=configuration('student_recovery_stop_reward_v2' if version==STOP_RECIPE else 'student_recovery_broad_narrow_reward_v1'),
        critic='new_random_64D',critic_privileged_version=VERSION,optimizer='new_Adam',
        learning_rate=1e-4,checkpoint_every_updates=100,target_height_m=.34,
        source_teacher_baseline=False,selected_rollouts_formal=[],selected_rollouts_smoke=[1,6],
        milestones=list(range(500,10000,500))+[9988],storage_max_bytes=8*2**30)
    if version==STOP_RECIPE:r.update(milestones=list(range(100,1901,100))+[1988],storage_max_bytes=2*2**30)
    return r
