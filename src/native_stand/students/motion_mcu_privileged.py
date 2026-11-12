"""CPU candidate only: causal motor-state extension; not installed in training.

The Actor interface and the existing 184 Critic columns keep their meanings.
Only already-published commands enter this candidate. No simulator, file IO,
future policy actions or optimizer steps occur in feature construction.
"""
import copy
import torch

VERSION = 'motion_mcu_motorqueue_privileged312_v1'
NETWORK = 'motion31_mcu_motorqueue_critic312_v1'
BASE_DIM, EXTRA_DIM, DIM, HISTORY_MS = 184, 128, 312, 14


def network():
    from native_stand.students.motion_fixed_height import network as base
    declaration = copy.deepcopy(base())
    declaration.update(version=NETWORK, critic_input=DIM)
    return declaration


def fields():
    return [
        dict(name='published_motor_commands', dim=84, units='Nm/driver_limit',
             order='newest to oldest, ages1..14ms at next physics tick; each ch0..5'),
        dict(name='actual_motor_torque', dim=6, units='Nm/driver_limit'),
        dict(name='held_driver_command', dim=6, units='Nm/driver_limit'),
        dict(name='last_PD_spring_compensation', dim=6, units='Nm/driver_limit'),
        dict(name='current_independent_spring_torque', dim=6, units='Nm/driver_limit'),
        dict(name='motor_transport_delay', dim=6, units='ms/20'),
        dict(name='motor_response_tau', dim=6, units='ms/10'),
        dict(name='driver_output_capability', dim=6, units='Nm/[40,40,5.5,40,40,5.5]'),
        dict(name='PD_feedback_age', dim=1, units='ms/20'),
        dict(name='PD_clock_phase', dim=1, units='next_physics_tick_mod5/5'),
    ]


def contract():
    from native_stand.students.motion_fixed_height import contract as base
    declaration=copy.deepcopy(base())
    declaration.update(critic_dim=DIM,critic_privileged_version=VERSION)
    declaration['critic_privileged'] += [f"{r['name']}:{r['dim']} ({r['units']})" for r in fields()]
    declaration['privileged_scope']='causal published motor history and current simulator execution state; no future Actor predictions'
    return declaration


def past_motor_commands(history, next_tick):
    """Read the completed 14 ticks, excluding the not-yet-executed tick."""
    if history.ndim != 3 or history.shape[0] < HISTORY_MS or history.shape[-1] != 6:
        raise ValueError('motor history must be [ring>=14,env,6]')
    if type(next_tick) is not int or next_tick < 0:
        raise ValueError('next physics tick must be nonnegative int')
    ages = torch.arange(1, HISTORY_MS + 1, device=history.device)
    return history[(next_tick - ages) % history.shape[0]].permute(1, 0, 2)


def critic_input(base, *, published_commands, motor_actual, driver_command,
                 compensation, spring_torque, transport_ms, tau_ms,
                 driver_limits, PD_feedback_ms, PD_phase_ms):
    if base.ndim != 2 or base.shape[-1] != BASE_DIM:
        raise ValueError('explicit 184D base Critic input required')
    n = base.shape[0]
    inputs = [(published_commands, (n, HISTORY_MS, 6))]
    inputs += [(v, (n, 6)) for v in (motor_actual, driver_command, compensation,
                                   spring_torque, transport_ms, tau_ms, driver_limits)]
    inputs += [(v, (n, 1)) for v in (PD_feedback_ms, PD_phase_ms)]
    for value, shape in inputs:
        if value.shape != shape or value.device != base.device or value.dtype != base.dtype:
            raise ValueError('explicit motor-state shape/device/dtype differs')
    # Driver capabilities are strictly positive declared constants, not learned values.
    reference = base.new_tensor([40,40,5.5,40,40,5.5])
    extra = torch.cat(((published_commands / driver_limits[:, None]).flatten(1),
        motor_actual / driver_limits, driver_command / driver_limits,
        compensation / driver_limits, spring_torque / driver_limits,
        transport_ms / 20, tau_ms / 10, driver_limits / reference,
        PD_feedback_ms / 20, PD_phase_ms / 5), -1)
    if extra.shape[-1] != EXTRA_DIM:
        raise ValueError('motor-state declaration must total128')
    return torch.cat((base, extra), -1)


def expand_source(source):
    """Migration intermediate only; new recipe/runtime binding is still required."""
    from native_stand.students.motion_fixed_height import network as old_network, CRITIC
    if source.get('network_architecture') == network() and source.get('critic_privileged_version') == VERSION:
        return source
    if source.get('network_architecture') != old_network() or source.get('critic_privileged_version') != CRITIC:
        raise ValueError('explicit same-semantic 184D source required')
    result = copy.deepcopy(source)
    weight = result['critic']['mlp.0.weight']
    if weight.shape != (512, BASE_DIM):
        raise ValueError('source Critic first layer differs')
    def pad(value):
        if value.shape != weight.shape:
            raise ValueError('Adam first-layer tensor shape differs')
        return torch.cat((value, value.new_zeros(512, EXTRA_DIM)), -1)
    result['critic']['mlp.0.weight'] = pad(weight)
    ids = [pid for group in result['optimizer']['param_groups'] for pid in group['params']]
    if len(ids) != len(source['actor']) + len(source['critic']):
        raise ValueError('source optimizer parameter order differs')
    key = ids[len(source['actor']) + list(source['critic']).index('mlp.0.weight')]
    for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        state = result['optimizer']['state'].get(key, {})
        if name in state:
            state[name] = pad(state[name])
    result.update(network_architecture=network(), critic_privileged_version=VERSION)
    return result
