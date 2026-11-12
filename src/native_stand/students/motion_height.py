"""Explicit common-height task: reference, reward and value-state semantics.

No virtual-leg controller: the height reference is an Actor command only.
"""
import copy
import torch

VERSION = 'motion31_common_height_v1'
CRITIC = 'motion_height_privileged144_v1'
NETWORK = 'motion31_height_critic144_v1'
HEIGHTS = (.32, .33, .34, .35, .36)


def configuration():
    return dict(version=VERSION, heights_m=list(HEIGHTS), nominal_height_m=.34,
                transition_s=[2., 6.], moving_transition_s=[12., 16.],
                height_error_scale_m=.04, stable_error_max_m=.02,
                transition_error_p95_max_m=.04,
                levels=['H0_fixed034', 'H1_fixed_common_transition',
                        'H2_transition_then_motion_no_reset',
                        'H3_random_common_initial_target', 'H4_moving_transition'],
                random_initial='reset-only deterministic selection from five closure-solved common heights',
                reward='replace height reference and vertical-speed residual; other weights unchanged',
                previous='actual initialized joint target, zero wheel target; no shadow prediction')


def network():
    from .motion_antiosc_privileged import network as base
    result = base(); result.update(version=NETWORK, critic_input=144)
    return result


def contract():
    from .motion_antiosc_privileged import contract as base
    result = base(); result.update(critic_dim=144, critic_privileged_version=CRITIC)
    result['critic_privileged'][0] = 'root_height_minus_effective_height_command_m:1'
    result['critic_privileged'] += ['initial_common_height_m/.5:1', 'target_common_height_m/.5:1',
                                   'effective_common_height_m/.5:1', 'reference_vertical_speed_mps:1']
    result['height_measurement'] = 'world Z of USD root-link origin; reference rate uses the same point/axis'
    return result


def expand_source(source):
    from .motion_antiosc_privileged import network as old_network, VERSION as old_version
    if source.get('network_architecture') == network() and source.get('critic_privileged_version') == CRITIC:
        return source
    if source.get('network_architecture') != old_network() or source.get('critic_privileged_version') != old_version:
        raise ValueError('height144 requires explicitly declared Motion140 source')
    result = copy.deepcopy(source)
    weight = result['critic']['mlp.0.weight']
    if weight.shape != (512, 140): raise ValueError('height source140 shape differs')
    def pad(value): return torch.cat((value, value.new_zeros(512, 4)), -1)
    result['critic']['mlp.0.weight'] = pad(weight)
    ids = [pid for group in result['optimizer']['param_groups'] for pid in group['params']]
    if len(ids) != len(result['actor']) + len(result['critic']): raise ValueError('height Adam ordering differs')
    state = result['optimizer']['state'].get(ids[len(result['actor']) + list(result['critic']).index('mlp.0.weight')], {})
    for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        if name in state: state[name] = pad(state[name])
    result.update(network_architecture=network(), critic_privileged_version=CRITIC)
    return result


def reference(age, initial, target, moving):
    start = torch.where(moving, age.new_tensor(12.), age.new_tensor(2.))
    u = ((age - start) / 4.).clamp(0., 1.)
    height = initial + (target - initial) * u.square() * (3. - 2. * u)
    rate = (target - initial) * 6. * u * (1. - u) / 4.
    return height, rate


def command(target_xy, age, initial, target, moving, training):
    from .motion_training import effective
    result = effective(target_xy, age, training=training)
    # H2/H3 complete the common-height transition before beginning locomotion.
    shifted = (~moving) & ((initial - target).abs() > 1e-7)
    if training:
        gain = ((age - 6.) / 4.).clamp(0., 1.)
        gain = torch.minimum(gain, ((30. - age) / 1.).clamp(0., 1.))
    else:
        gain = torch.minimum(((age - 6.) / 4.).clamp(0., 1.), ((26. - age) / 8.).clamp(0., 1.))
    result[:, :2] = torch.where(shifted[:, None], target_xy * gain[:, None], result[:, :2])
    height, rate = reference(age, initial, target, moving)
    result[:, 2] = height
    return result, rate


def correct_reward(result, height, effective_height, vertical_velocity, reference_rate):
    """Replace the two existing terms, rather than silently adding more weight."""
    updated = dict(result)
    for key in ('raw', 'density', 'dt', 'total'): updated[key] = result[key].clone()
    new = {'height': 1. / (1. + ((height - effective_height) / .04).square()),
           'transfer_vertical_velocity': (vertical_velocity - reference_rate).square()}
    for name, raw in new.items():
        index = result['fields'].index(name)
        weight = 2. if name == 'height' else -.5
        raw = raw * result['eligible']
        delta = (raw * weight - result['density'][:, index]) * .01
        updated['raw'][:, index] = raw
        updated['density'][:, index] = raw * weight
        updated['dt'][:, index] = raw * weight * .01
        updated['total'] += delta
    return updated


def critic_input(base140, initial, target, effective_height, rate):
    if base140.shape[-1] != 140: raise ValueError('height critic needs explicit140 physical prefix')
    base = base140.clone()
    base[:, 31] += .34 - effective_height
    return torch.cat((base, initial[:, None] / .5, target[:, None] / .5,
                      effective_height[:, None] / .5, rate[:, None]), -1)


def reward_snapshot():
    from .data import sha
    curves = []
    for error in (-.08, -.04, -.02, -.01, 0., .01, .02, .04, .08):
        x = torch.tensor(error, dtype=torch.float64, requires_grad=True)
        value = 2. / (1. + (x / .04).square())
        curves.append(dict(error_m=error, weighted_density=value.item(),
                           derivative=torch.autograd.grad(value, x)[0].item(), dt_contribution=value.item() * .01))
    return dict(configuration=configuration(), formula_source_sha256=sha(__file__),
                height_curves=curves, vertical_density='-.5*(root_origin_world_vz - reference_dh_dt)**2',
                dt_s=.01, actor_interface_unchanged=True, convergence_guaranteed=False)


def prepare_poses(spec, heights=HEIGHTS):
    """Offline geometric reset candidates, never runtime validity claims."""
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from native_stand.recovery.asset_poses import AssetPoses
    asset = AssetPoses(spec['asset_path'])
    rotation = np.eye(4); rotation[:3, :3] = Rotation.from_quat([2**-.5, 0., 0., 2**-.5]).as_matrix()
    seed = {j['name']: 0. for j in asset.joints}
    result = []
    for height in heights:
        def solve(values):
            hip, active = values
            return asset.solve(dict(left_hip_pitch=hip, left_active_link_pitch=active,
                                    right_hip_pitch=-hip, right_active_link_pitch=active), seed)
        def residual(values):
            q, _ = solve(values); bodies = asset.forward(q)
            centers = np.array([bodies[side + '_wheel'][:3, 3] for side in ('left', 'right')])
            support = asset.support(q, rotation)
            wheel = [row['lowest_z_root'] for row in support if row['body'] in ('left_wheel', 'right_wheel')]
            return [height + np.mean(wheel) - .001, centers[:, 2].mean()]
        fit = least_squares(residual, [.1, .4], bounds=([-1.2, -1.5], [1.2, 1.5]),
                            xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=70)
        q, closure = solve(fit.x); support = asset.support(q, rotation)
        gaps = {row['body']: height + row['lowest_z_root'] for row in support}
        if max(abs(x) for x in residual(fit.x)) > 1e-4 or closure.max() > 1e-4:
            raise ValueError('height reset closure/height solve failed')
        if any(q[j['name']] < j['limits'][0] - 1e-6 or q[j['name']] > j['limits'][1] + 1e-6 for j in asset.joints):
            raise ValueError('height reset exceeds authored joint limit')
        if min(gaps.values()) < -.001: raise ValueError('height reset intersects ground')
        result.append(dict(height_m=height, joint_pos=q, root_quaternion_xyzw=[2**-.5, 0., 0., 2**-.5],
                           closure_max_m=float(closure.max()), minimum_gap_m=float(min(gaps.values())),
                           offline_valid=True, runtime_valid=None))
    return result
