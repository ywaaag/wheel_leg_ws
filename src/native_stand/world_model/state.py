"""Named, serializable causal state. Tick denotes the NEXT 1 ms physics step."""
from dataclasses import dataclass, fields, replace
import torch


@dataclass
class State:
    q: torch.Tensor                 # MuJoCo qpos, root quaternion wxyz
    v: torch.Tensor                 # MuJoCo qvel, free linear world / angular local
    solver_q: torch.Tensor          # last solver q, BEFORE its final integration
    solver_v: torch.Tensor          # last solver velocity used for body sensors
    warm: torch.Tensor              # solver qacc_warmstart (implicitfast is not explicit Euler)
    contact: torch.Tensor           # post-solver wheel ground normal force, N
    body_contact: torch.Tensor      # root-body ground resultant, N; source failure gate
    sensors: torch.Tensor           # [B,64,18], newest first, completed ticks t-1..t-64
    queue: torch.Tensor             # [B,64,6], published post-PD driver commands t-1..t-64
    pending: torch.Tensor
    adopted: torch.Tensor           # actual local PD adopted target; also Actor previous
    driver: torch.Tensor
    motor: torch.Tensor             # response internal state == actual motor torque
    pd_request: torch.Tensor
    pd_limited: torch.Tensor
    compensation: torch.Tensor
    spring: torch.Tensor
    applied: torch.Tensor           # requested combined motor + independent spring
    generalized: torch.Tensor       # actual solver actuator torque, after ctrl/force limits
    tick: torch.Tensor              # per-row integer, absolute continuous clock
    delay: torch.Tensor             # integer post-PD transportation, ms
    tau: torch.Tensor               # response time constants, ms
    pd_age: torch.Tensor
    actor_age: torch.Tensor
    spring_scale: torch.Tensor
    actor_bias: torch.Tensor        # reported four-encoder bias in model ch order (6 columns)
    pd_bias: torch.Tensor
    driver_cap: torch.Tensor
    last_target: torch.Tensor       # previous policy-boundary target, reward only
    older_target: torch.Tensor
    last_torque: torch.Tensor       # previous policy-boundary total generalized torque
    history_valid: torch.Tensor
    episode_step: torch.Tensor      # schedule/terminal history, never an Actor field
    quality_ticks: torch.Tensor     # seven persistent soft-violation counters
    quality_age: torch.Tensor
    quality_sent: torch.Tensor
    context: torch.Tensor           # full effective plant parameters, never task name
    task: torch.Tensor              # [vx,yaw,height,skill_id]; excluded from mechanics
    target: torch.Tensor            # episode target; ramps never relabel its cohort
    jump_kind: torch.Tensor         # reserved MCU NONE/SMALL/BIG=0/1/2; v1 runtime requires0
    owner: torch.Tensor             # actual controlling policy identity, excluded from mechanics

    def map(self, fn):
        return replace(self, **{f.name: fn(getattr(self, f.name)) for f in fields(self)})

    def detach(self):
        return self.map(lambda x: x.detach())

    def clone(self):
        return self.map(lambda x: x.clone())

    def to(self, device):
        return self.map(lambda x: x.to(device))

    def rows(self, ids):
        return self.map(lambda x: x[ids])

    def reset_rows(self, ids, initial):
        """Explicit row reset only; controller handover does NOT call this."""
        def merge(name):
            a = getattr(self, name).clone()
            if name!='tick':a[ids] = getattr(initial, name)
            return a
        return replace(self, **{f.name: merge(f.name) for f in fields(self)})

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def cat(cls, states):
        return cls(**{f.name: torch.cat([getattr(s, f.name) for s in states]) for f in fields(cls)})


def initial(q, v, sensors, context, task, *, compensation, cap, delay, tau,
            actor_age=10, pd_age=10):
    n = len(q)
    z = q.new_zeros(n, 6)
    target = z.clone()
    # Six physical targets, not normalized actions. Wheels start at zero.
    target[:, [0, 1, 3, 4]] = sensors[:, 6:12][:, [0, 1, 3, 4]]
    integer = torch.zeros(n, dtype=torch.long, device=q.device)
    return State(q, v, q.clone(), v.clone(), torch.zeros_like(v), q.new_zeros(n, 2), q.new_zeros(n,1),
        sensors[:, None].expand(-1, 64, -1).clone(),
        compensation[:, None].expand(-1, 64, -1).clone(), target, target.clone(),
        compensation, compensation.clone(), z.clone(), z.clone(), compensation.clone(),
        -compensation, z.clone(), z.clone(), integer.clone(), delay.long(), tau, integer + pd_age,
        integer + actor_age, q.new_ones(n, 1), z.clone(), z.clone(), cap,
        target.clone(), target.clone(), z.clone(), torch.zeros(n, dtype=torch.bool, device=q.device),
        integer.clone(), torch.zeros(n, 7, dtype=torch.long, device=q.device), integer.clone(),
        torch.zeros(n, dtype=torch.bool, device=q.device), context, task, task[:, :2].clone(), integer.clone(), integer.clone())
