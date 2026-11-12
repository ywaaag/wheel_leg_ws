"""Explicit hypotheses for closed-loop MCU sensitivity, not hardware identification.

No files, simulator state or global random generators are accessed here.
The nominal point is the existing MCU recipe; one-factor screens change exactly
one family. Calibration offsets are reported-coordinate errors in BOTH Actor
and PD, not a change to mechanical zero or a runtime correction.
"""
import copy
import math

VERSION = 'motion_mcu_execution_uncertainty_v1'
LEGS = (0, 1, 3, 4)


def nominal():
    return dict(name='nominal', family='nominal', actor_age_ms=10, PD_age_ms=10,
                motor_delay_ms=[10, 10, 7, 10, 10, 11],
                motor_tau_ms=[5., 5., 0., 5., 5., 0.],
                leg_zero_bias_rad=[0.] * 4, spring_scale=1., wheel_cap_scale=1.,
                initial_vx_mps=0., initial_pitch_rate_rad_s=0., initial_roll_rad=0.)


def validate(row):
    if set(row) != set(nominal()):
        raise ValueError('uncertainty profile fields differ')
    for key in ('actor_age_ms', 'PD_age_ms'):
        if isinstance(row[key], bool) or int(row[key]) != row[key] or not 0 <= row[key] <= 30:
            raise ValueError('integer sensor age 0..30ms required')
    for key, length, lo, hi in [('motor_delay_ms', 6, 0, 30),
                                ('motor_tau_ms', 6, 0, 30),
                                ('leg_zero_bias_rad', 4, -.1, .1)]:
        if len(row[key]) != length or any(not math.isfinite(v) or not lo <= v <= hi for v in row[key]):
            raise ValueError('invalid ' + key)
    if any(int(v) != v for v in row['motor_delay_ms']):
        raise ValueError('motor transport requires integer milliseconds')
    for key, lo, hi in [('spring_scale', .8, 1.2), ('wheel_cap_scale', .5, 1.),
                        ('initial_vx_mps', -.1, .1), ('initial_pitch_rate_rad_s', -.2, .2),
                        ('initial_roll_rad', -.04, .04)]:
        if not math.isfinite(row[key]) or not lo <= row[key] <= hi:
            raise ValueError('invalid ' + key)
    if row['initial_roll_rad'] != 0.:
        raise ValueError('roll-pose reset not enabled in this screen; use declared velocity perturbations')
    return row


def screen():
    rows = [nominal()]

    def add(name, family, **change):
        row = dict(nominal(), name=name, family=family, **change)
        rows.append(validate(row))

    for value in (-.05, .05):
        add(f'initial_vx_{value:+g}', 'initial_state', initial_vx_mps=value)
    for value in (-.1, .1):
        add(f'initial_pitch_rate_{value:+g}', 'initial_state', initial_pitch_rate_rad_s=value)
    for age in (0, 20):
        add(f'actor_age_{age}', 'actor_age', actor_age_ms=age)
        add(f'PD_age_{age}', 'PD_age', PD_age_ms=age)
    for delta in (-5, 3):
        add(f'motor_transport_delta_{delta:+d}', 'motor_transport',
            motor_delay_ms=[v + delta for v in nominal()['motor_delay_ms']])
    for tau in (10., 20.):
        add(f'leg_tau_{tau:g}', 'motor_response', motor_tau_ms=[tau, tau, 0., tau, tau, 0.])
    for index, channel in enumerate(LEGS):
        for sign in (-1, 1):
            bias = [0.] * 4
            bias[index] = sign * .02
            add(f'zero_ch{channel}_{sign:+d}', 'calibration', leg_zero_bias_rad=bias)
    for scale in (.9, 1.1):
        add(f'spring_{scale:g}', 'spring_residual', spring_scale=scale)
    add('wheel_capacity_80pct', 'wheel_capacity', wheel_cap_scale=.8)
    return rows


def combine(a, b):
    """Combine different mechanisms only, never silently overwrite a perturbation."""
    validate(a); validate(b)
    if a['family'] == b['family'] or 'nominal' in (a['family'], b['family']):
        raise ValueError('two different non-nominal families required')
    result = nominal()
    for row in (a, b):
        for key, value in row.items():
            if key in ('name', 'family') or value == nominal()[key]:
                continue
            if result[key] != nominal()[key]:
                raise ValueError('overlapping uncertainty mechanisms')
            result[key] = copy.deepcopy(value)
    result.update(name=a['name'] + '__' + b['name'], family=a['family'] + '+' + b['family'])
    return validate(result)
