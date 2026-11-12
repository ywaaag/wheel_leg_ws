"""Pure explicit gas-spring geometry and balanced wrenches; no runtime hook."""
from dataclasses import dataclass
import math
import torch

@dataclass(frozen=True)
class SpringLaw:
    force_at_extended_n: float
    stiffness_n_m: float
    extended_m: float
    compressed_m: float
    damping_extension_ns_m: float = 0.
    damping_compression_ns_m: float = 0.

    def __post_init__(self):
        values=list(self.__dict__.values())
        if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite spring parameter')
        if not 0<self.compressed_m<self.extended_m:raise ValueError('invalid stroke')
        if min(self.force_at_extended_n,self.stiffness_n_m,self.damping_extension_ns_m,self.damping_compression_ns_m)<0:raise ValueError('negative spring parameter')

def force_pair(*,position_a,position_b,com_a,com_b,velocity_com_a,velocity_com_b,omega_a,omega_b,law):
    """All vectors world-frame metres/SI. Positive scalar pushes ends apart.

    Gas push cuts off above extension; this does not detach the two anchors.
    Mechanical stop reaction is supplied separately by solver limits/end stops.
    """
    vectors=(position_a,position_b,com_a,com_b,velocity_com_a,velocity_com_b,omega_a,omega_b)
    if any(v.shape!=position_a.shape or v.shape[-1]!=3 or not torch.isfinite(v).all() for v in vectors):raise ValueError('finite matching world3 vectors required')
    delta=position_b-position_a;length=delta.norm(dim=-1)
    if (length<=1e-8).any():raise ValueError('coincident spring anchors')
    axis=delta/length[...,None]
    va=velocity_com_a+torch.cross(omega_a,position_a-com_a,dim=-1)
    vb=velocity_com_b+torch.cross(omega_b,position_b-com_b,dim=-1)
    rate=((vb-va)*axis).sum(-1)
    elastic=law.force_at_extended_n+law.stiffness_n_m*(law.extended_m-length)
    damping=-law.damping_extension_ns_m*rate.clamp_min(0)-law.damping_compression_ns_m*rate.clamp_max(0)
    magnitude=torch.where(length<=law.extended_m,(elastic+damping).clamp_min(0),torch.zeros_like(length))
    fb=magnitude[...,None]*axis;fa=-fb
    ta=torch.cross(position_a-com_a,fa,dim=-1);tb=torch.cross(position_b-com_b,fb,dim=-1)
    return dict(length_m=length,length_rate_mps=rate,stroke_valid=(length>=law.compressed_m)&(length<=law.extended_m),
                elastic_force_n=elastic,damping_force_n=damping,force_n=magnitude,force_a_world=fa,force_b_world=fb,
                torque_a_about_com_world=ta,torque_b_about_com_world=tb,power_w=magnitude*rate,
                net_force_world=fa+fb,net_torque_about_origin_world=torch.cross(position_a,fa,dim=-1)+torch.cross(position_b,fb,dim=-1))

@dataclass(frozen=True)
class EndStops:
    stiffness_n_m: float
    damping_ns_m: float
    max_penetration_m: float
    max_reaction_n: float

    def __post_init__(self):
        if not all(math.isfinite(v) and v>0 for v in self.__dict__.values()):raise ValueError('explicit positive end-stop parameters required')

def with_end_stops(result,position_a,position_b,com_a,com_b,law,stops):
    """Finite-compliance unilateral end-stop approximation, NOT an exact hard constraint.

    Upper reaction can be tensile; this is the mechanical stop, not pulling gas.
    Safety violations abort before submitting an unbounded wrench.
    """
    length=result['length_m'];rate=result['length_rate_mps']
    lower=(law.compressed_m-length).clamp_min(0);upper=(length-law.extended_m).clamp_min(0)
    compression=stops.stiffness_n_m*lower+stops.damping_ns_m*(-rate).clamp_min(0)*(length<=law.compressed_m)
    extension=stops.stiffness_n_m*upper+stops.damping_ns_m*rate.clamp_min(0)*(length>=law.extended_m)
    reaction=compression-extension
    if (torch.maximum(lower,upper)>stops.max_penetration_m).any():raise ValueError('gas spring stroke penetration exceeds declared safety bound')
    if (reaction.abs()>stops.max_reaction_n).any():raise ValueError('gas spring end-stop reaction exceeds declared safety bound')
    axis=(position_b-position_a)/length[...,None];fb=(result['force_n']+reaction)[...,None]*axis;fa=-fb
    return {**result,'stop_compression_n':compression,'stop_extension_n':extension,'stop_reaction_n':reaction,
            'stop_power_w':reaction*rate,'force_a_world':fa,'force_b_world':fb,
            'torque_a_about_com_world':torch.cross(position_a-com_a,fa,dim=-1),
            'torque_b_about_com_world':torch.cross(position_b-com_b,fb,dim=-1),
            'power_w':(result['force_n']+reaction)*rate,'net_force_world':fa+fb,
            'net_torque_about_origin_world':torch.cross(position_a,fa,dim=-1)+torch.cross(position_b,fb,dim=-1)}
