"""Pure functional 1 ms MCU chain; published history remains in autograd."""
from dataclasses import replace
import torch
from native_stand.students.motion_mcu_execution import pd_and_motor, spring_compensation, LEGS, WHEELS
from native_stand.students.mcu_interface import encode
from native_stand.students.interface import targets


def gather_age(history, age):
    return history[torch.arange(len(history), device=history.device), age.long()]


def measurements(s,geometry):
    """Match existing public playback timestamps, including its solver readback lag.

    Public MuJoCo sensors() uses last-solver body kinematics but integrated active
    joint q/dq. Preserve that effective convention, never add mj_forward to the
    controller path to silently remove a millisecond of feedback age.
    """
    m=geometry(s.solver_q,s.solver_v)
    m['sensors']=torch.cat((m['sensors'][:,:6],s.q[:,geometry.active_q],s.v[:,geometry.active_v]),-1)
    m['root_com_velocity']=m['com_velocity'][:,geometry.root]
    return m


def observation(s, measured, reference=None):
    if reference is None:
        reference = torch.zeros_like(s.adopted)
    # View current sensor without changing the completed-tick bank twice.
    bank = torch.cat((measured['sensors'][:, None], s.sensors), 1)
    sensed = gather_age(bank, s.actor_age)
    q = sensed[:, 6:12]+s.actor_bias
    # Explicit skills, never inferred from tensor dimensions.
    if not bool(((s.task[:, 3] == 0)&(s.jump_kind==0)).all()):
        raise ValueError('v1 learner supports Motion only; Recovery/Jump need corresponding coverage')
    return encode(sensed[:, :3], sensed[:, 3:6], q, sensed[:, 12:18],
                  s.adopted[:, LEGS], s.adopted[:, WHEELS], reference, 'Motion', s.task[:, :3])


def decode(action, kinematics, s):
    d = targets(action, torch.zeros_like(action), kinematics.limits.expand(len(action), -1, -1)+s.pd_bias[:, :, None])
    # Match shared physical decoder, retaining native ch0..5 order.
    out = torch.zeros_like(action)
    out[:, LEGS] = d['leg_executed']; out[:, WHEELS] = d['wheel_executed']
    return out


def step(s, requested, true_sensors):
    """At t: sense -> local adoption/PD -> driver -> transport -> response -> spring."""
    if s.queue.shape[1:] != (64, 6) or s.sensors.shape[1:] != (64, 18):
        raise ValueError('complete 64 ms execution banks required')
    if bool(((s.delay < 0) | (s.delay >= 64)).any()) or bool(((s.pd_age < 0) | (s.pd_age >= 64)).any()):
        raise ValueError('delay/feedback age outside saved Markov bank')
    bank = torch.cat((true_sensors[:, None], s.sensors[:, :-1]), 1)
    pending = torch.where((s.tick % 10 == 0)[:, None], requested, s.pending)
    pd = (s.tick % 5 == 0)[:, None]
    adopted = torch.where(pd, pending, s.adopted)
    feedback = gather_age(bank, s.pd_age)
    request, limited, comp, motor, ok = pd_and_motor(feedback[:, 6:12]+s.pd_bias, feedback[:, 12:18], adopted)
    driver_new = (limited+comp).clamp(-s.driver_cap, s.driver_cap)
    driver = torch.where(pd, driver_new, s.driver)
    queue = torch.cat((driver[:, None], s.queue[:, :-1]), 1)
    rows = torch.arange(len(s.q), device=s.q.device)[:, None]
    channels = torch.arange(6, device=s.q.device)[None]
    delivered = queue[rows, s.delay.long(), channels]
    alpha = -torch.expm1(-1/s.tau.clamp_min(1e-9))
    actual = s.motor+alpha*(delivered-s.motor)
    spring, _, _, spring_ok = spring_compensation(true_sensors[:, 6:12][:, LEGS])
    spring = -spring*s.spring_scale
    out = replace(s, sensors=bank, queue=queue, pending=pending, adopted=adopted,
        driver=driver, motor=actual, pd_request=torch.where(pd, request, s.pd_request),
        pd_limited=torch.where(pd, limited, s.pd_limited),
        compensation=torch.where(pd, comp, s.compensation), spring=spring, applied=actual+spring,
        tick=s.tick+1)
    return out, torch.where(pd[:,0],ok,torch.ones_like(ok)) & spring_ok
