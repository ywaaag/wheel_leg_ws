"""Explicit tensor-only MCU PD/compensation/transport model, CPU-oracle testable."""
import torch

VERSION = 'motion_mcu_post_pd_execution_v1'
LEGS = [0, 1, 3, 4]
WHEELS = [2, 5]
WHEEL_CURRENT_PER_NM = 19. / .00036 / (268. / 17.)
WHEEL_CAP = 8192. / WHEEL_CURRENT_PER_NM


def profile():
    return dict(version=VERSION, pd_period_ms=5, policy_period_ms=10, physics_period_ms=1,
                leg_kp=40., leg_kd=2., wheel_kv=.2, local_target_delay_ms=0,
                actor_age_ms=10, PD_age_ms=10,
                motor_delay_ms=[10,10,7,10,10,11], motor_tau_ms=[5,5,0,5,5,0],
                pd_limits_nm=[40,40,5.5,40,40,5.5],
                driver_limits_nm=[40,40,WHEEL_CAP,40,40,WHEEL_CAP],
                reference_limits_nm=[40,40,5.5,40,40,5.5], solver_limits_nm=[67,67,5.5,67,67,5.5],
                spring='negative_C_compensation_at_current_geometry; nominal_equilibrium_hypothesis',
                initial_motor='current_spring_compensation; zero_net_torque_hypothesis_not_measured_LQR',
                source='firmware mathematical contract; electrical timing/capability/spring parameters are hypotheses')


def spring_compensation(q_leg):
    """Torch port of actual getPhi + RLControlCompensateSpring, in model order.

    The returned value is an electrical feedforward command, NOT a measured
    gas-spring torque. The independent physical-spring hypothesis is its negative.
    """
    active = q_leg.new_tensor((90. + 47.1080837) * (torch.pi / 180.))
    hip = q_leg.new_tensor((90. - 47.1033888) * (torch.pi / 180.))
    phi1 = torch.stack((active + q_leg[:,3], active + q_leg[:,1]), -1)
    phi4 = torch.stack((hip + q_leg[:,2], hip - q_leg[:,0]), -1)
    xb, yb = .210 * phi1.cos(), .210 * phi1.sin()
    xd, yd = .210 * phi4.cos(), .210 * phi4.sin()
    a, b = .5 * (xd-xb), .5 * (yd-yb)
    c = (xd-xb).square() + (yd-yb).square()
    discriminant = a.square() + b.square() - c.square()
    # Tiny roundoff only; invalid geometry is separately flagged, never hidden.
    phi2 = 2 * torch.atan2(b + discriminant.clamp_min(0).sqrt(), a+c)
    xc, yc = xb + .250 * phi2.cos(), yb + .250 * phi2.sin()
    length = (xc.square() + yc.square()).sqrt()
    spread = phi1 - phi4
    s = ((-83.121 * length + 40.242) * .8).clamp(16.5,27.) * spread.sin()
    out = q_leg.new_zeros(len(q_leg),6)
    out[:,0] = s[:,1]; out[:,1] = s[:,1]
    out[:,3] = -s[:,0]; out[:,4] = s[:,0]
    valid = (discriminant >= -1e-7).all(-1) & torch.isfinite(out).all(-1)
    return out, length, spread, valid


def pd_and_motor(q, dq, adopted):
    request = torch.zeros_like(adopted)
    request[:,LEGS] = 40 * (adopted[:,LEGS]-q[:,LEGS]) - 2*dq[:,LEGS]
    request[:,WHEELS] = .2 * (adopted[:,WHEELS]-dq[:,WHEELS])
    pd_limits = request.new_tensor([40,40,5.5,40,40,5.5])
    limited = request.clamp(-pd_limits,pd_limits)
    compensation, length, spread, valid = spring_compensation(q[:,LEGS])
    cap = request.new_tensor([40,40,WHEEL_CAP,40,40,WHEEL_CAP])
    motor = (limited+compensation).clamp(-cap,cap)
    return request, limited, compensation, motor, valid


class MotorRing:
    """Causal1ms transport and first-order response; reset only selected rows."""
    def __init__(self,n,device):
        self.ids=torch.arange(n,device=device);self.channels=torch.arange(6,device=device)
        self.history=torch.zeros(64,n,6,device=device)
        self.actual=torch.zeros(n,6,device=device)
        self.delay=torch.tensor(profile()['motor_delay_ms'],device=device).expand(n,-1).clone()
        self.tau=torch.tensor(profile()['motor_tau_ms'],device=device).float().expand(n,-1).clone()
        self.last_tick=None

    def reset(self,ids,initial):
        self.history[:,ids]=initial[None];self.actual[ids]=initial

    def step(self,tick,command):
        if self.last_tick is not None and tick!=self.last_tick+1:
            raise ValueError('motor clock must advance exactly1ms')
        self.history[tick%64]=command
        delivered=self.history[(tick-self.delay)%64,self.ids[:,None],self.channels[None,:]]
        alpha=-torch.expm1(-1/self.tau.clamp_min(1e-9))
        self.actual.add_(alpha*(delivered-self.actual));self.last_tick=tick
        return self.actual
