"""4500 continuous reward adapter, exact shared costs and explicit event semantics.

All densities are measured post-physics and multiplied by .01 ONCE. The learned
contact force supplies contact-dependent continuous terms in imagination. Hard
termination/event predicates have zero/undefined derivative; no smooth success
surrogate is introduced. Real scoring always uses solver measurements.
"""
from dataclasses import replace
from types import SimpleNamespace
import torch
from native_stand.students import motion_tracking_reward as tracking
from native_stand.students.motion_tracking_reward import huber
from native_stand.students.motion_mcu_execution import LEGS, WHEELS, WHEEL_CAP
from native_stand.students import motion_sensor_transfer as sensor
from native_stand.students import motion_antioscillation as anti
from native_stand.students.motion_posture_design import apply_runtime_reward
from native_stand.students.motion_differential import target_from_geometry, terms
from native_stand.students.motion_parallel_course import apply_tracking
from native_stand.students.motion_quality import QualityTermination, configuration as quality_configuration


class MotionReward:
    def __init__(self, spec, source):
        if spec['reward']['version'] != tracking.PARALLEL_VERSION or not spec.get('mcu_posture'):
            raise ValueError('explicit 4500 dense-posture parallel reward required')
        self.spec = spec
        # Freeze the SOURCE mixtures, with nominal identity including duplicate zero.
        self.symmetry = source['symmetry_mix_state']
        self.linear = source.get('reward_refinement_state', [])
        self.config = spec['motion_timing']['configuration']

    def mixtures(self, state, key):
        rows = self.symmetry if key == 'symmetry' else self.linear
        out = state.q.new_zeros(len(state.q))
        for row in rows:
            target = row['target'][:2]
            selected = (state.target == state.q.new_tensor(target)).all(-1)
            # Same XY in stress/nominal must not import an unrelated cohort's mixture.
            if len(row['target']) >= 4:
                nominal = (state.pd_age == 10) & (state.spring_scale[:, 0] == 1)
                selected &= nominal if row['target'][3] == 0 else ~nominal
            out = torch.where(selected, out.new_tensor(row.get('mix', 0.)), out)
        return out

    def __call__(self, before, after, measured):
        n = len(after.q)
        yes = torch.ones(n, dtype=torch.bool, device=after.q.device)
        no = ~yes
        q6 = measured['sensors'][:, 6:12]
        angles, lengths = measured['angles'], measured['lengths']
        # Reuse source v6 -> spin -> exact symmetry/differential -> v10 order.
        result = tracking.reward(command=before.task[:, :3], target=before.target,
            velocity=measured['velocity'], yaw=measured['yaw'], height=measured['height'],
            angles=angles, contacts=after.contact, leg_q=q6[:, LEGS],
            wheel_offset=measured['wheel_offset'], leg_lengths=lengths,
            applied_torque=after.generalized, previous_targets=before.last_target,
            executed_targets=after.adopted, controlled=yes, valid=yes, physical_failure=no,
            fault=no, config=tracking.configuration(tracking.COMBO_VERSION),
            linear_narrow_mix=self.mixtures(before, 'linear'))
        from native_stand.students.motion_yaw_quality import apply as apply_spin, configuration as spin_config
        result = apply_spin(result, target=before.target, command=before.task[:, :3],
            velocity=measured['velocity'], contacts=after.contact, previous_targets=before.last_target,
            executed_targets=after.adopted, measurement=measured['spin'], config=spin_config())
        mix = self.mixtures(before, 'symmetry')
        # Source symmetry.apply has an Isaac-only measurement adapter. These three
        # tensor equations are its unchanged pure density, tested against fixtures.
        e = measured['swing_diff'].abs()
        swing = -.5*huber(torch.relu(e-.02)/.03)+.25*((1-mix)*torch.exp(-e.square()/.0064)
                                                            +mix*torch.exp(-e.square()/.0009)-1)
        spin = measured['spin']; gate = result['spin_diagnostics']['activation']
        common = spin['rolling_speed_forward_mps'].mean(-1)
        f = spin['forward_world'].clone(); f[:, 2] = 0
        f = f/f.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        left = torch.stack((-f[:, 1], f[:, 0], f[:, 2]*0), -1)
        cv = measured['root_com_velocity']
        horizontal = torch.stack((cv.mul(f).sum(-1), cv.mul(left).sum(-1)), -1)
        replacements = dict(spin_common_rolling=-.5*((1-mix)*huber(common/.15)+mix*huber(common/.08)),
            spin_translation=-.5*((1-mix)*huber(horizontal/.2).sum(-1)+mix*huber(horizontal/.1).sum(-1)))
        density, raw = result['density'].clone(), result['raw'].clone()
        for name, value in replacements.items():
            i = result['fields'].index(name)
            density[:, i] = value*gate; raw[:, i] = value*gate/-.5
        result = {**result, 'fields': (*result['fields'], 'leg_swing_symmetry'),
            'density': torch.cat((density, swing[:, None]), -1),
            'raw': torch.cat((raw, -swing[:, None]), -1)}
        result['dt'] = result['density']*.01; result['total'] = result['dt'].sum(-1)
        differential = terms(spin['rolling_speed_forward_mps'],
            target_from_geometry(spin['center_world'], spin['forward_world'], before.task[:, 1]), mix)
        result.update(fields=(*result['fields'], 'spin_differential_target'),
            raw=torch.cat((result['raw'], (differential['raw']*gate)[:, None]), -1),
            density=torch.cat((result['density'], (differential['density']*gate)[:, None]), -1))
        result['dt'] = result['density']*.01; result['total'] = result['dt'].sum(-1)
        env = SimpleNamespace(motion_action_command=before.task[:, :3], target_xy=before.target,
            vx=measured['velocity'][:, 0], vy=measured['velocity'][:, 1], yaw=measured['yaw'],
            motion_cohort_mix=self.mixtures(before, 'linear'), command_index=torch.arange(n, device=after.q.device))
        result = apply_tracking(result, env)
        result = sensor.apply_costs(result, targets=after.adopted, previous=before.last_target,
            dq=measured['sensors'][:, 12:18], vertical_velocity=measured['root_com_velocity'][:, 2],
            angular_body=measured['sensors'][:, :3], torque=after.generalized,
            config=self.config['reward_extension'])
        result = anti.apply_costs(result, after.pd_request, after.applied, before.last_torque,
            after.adopted, before.last_target, before.older_target, after.contact,
            before.history_valid, self.config)
        i = result['fields'].index('wheel_request_excess')
        excess = torch.relu(after.pd_request[:, WHEELS].abs()/WHEEL_CAP-.8)
        smooth_mix = self.config['smooth_mix']
        value = ((1-smooth_mix)*huber(excess/2)+smooth_mix*huber(excess)).mean(-1)
        result['raw'][:, i] = value; result['density'][:, i] = -.25*value
        result['dt'] = result['density']*.01; result['total'] = result['dt'].sum(-1)
        posture = torch.cat((angles, lengths, after.q.new_zeros(n, 6)), -1)
        result = apply_runtime_reward(result, posture)
        # Shared quality termination, with counters explicitly in Markov state.
        monitor = QualityTermination(n, after.q.device, quality_configuration())
        monitor.ticks = before.quality_ticks.clone(); monitor.age = before.quality_age.clone()
        monitor.sent = before.quality_sent.clone()
        posture[:, 4:6] = torch.stack((q6[:, 0]+q6[:, 3], q6[:, 1]-q6[:, 4]), -1)
        decision = monitor.observe(posture=posture, height=measured['height'],
            wheel_offset=measured['wheel_offset'], contacts=after.contact,
            target=before.target, controlled=yes, valid=yes)
        t=self.spec['termination']
        physical = ((measured['height']<t['fall_root_height_m']) |
            (-measured['sensors'][:,4]<t['minimum_up_cos']) |
            (q6[:,LEGS].abs().amax(-1)>t['active_leg_angle_max_rad']) |
            (after.body_contact[:,0]>t['body_contact_max_n']))
        terminated = decision['terminated'] | physical
        # Terminal transition is retained, positive densities suppressed, -1 event unscaled.
        result['total'] = torch.where(terminated, -torch.ones_like(result['total']), result['total'])
        after = replace(after, last_target=after.adopted, older_target=before.last_target,
            last_torque=after.applied, history_valid=yes, episode_step=before.episode_step+1,
            quality_ticks=monitor.ticks, quality_age=monitor.age, quality_sent=monitor.sent)
        from native_stand.command_v6.domain.command_schedule import effective
        command = effective(after.target, after.episode_step.to(after.q.dtype)*.01,
                            height=.34, entry=2., ramp=8., hold=8., exit=8.)
        after = replace(after, task=torch.cat((command, after.task[:, 3:]), -1))
        return after, result, terminated
