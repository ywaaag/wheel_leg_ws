"""构造时注入DR时序；在同一次reset完成后、历史队列填充前生效。"""

import torch

from native_stand.tasks.motion.delayed_execution import VERSION, apply_reset, validate_configuration


def dr_factory(base, noise, config):
    validate_configuration(config)

    class DelayedDR(base):
        def reset(self, ids):
            super().reset(ids)
            apply_reset(self, ids, noise, config=config)

        def summary(self):
            result = super().summary()
            wheels = self.t.base.motor_ring
            result["delayed_execution"] = {
                "version": config["version"],
                "actor_packet_nominal_ms": config["actor_packet_nominal_ms"],
                "actor_leg_packet_nominal_ms": config["actor_leg_packet_nominal_ms"],
                "target_nominal_ms": config["target_nominal_ms"],
                "wheel_transport_nominal_ms_ch2_ch5": config["wheel_transport_nominal_ms"],
                "wheel_delay_ms_min_ch2_ch5": wheels.delay[:, [2, 5]].amin(0).cpu().tolist(),
                "wheel_delay_ms_max_ch2_ch5": wheels.delay[:, [2, 5]].amax(0).cpu().tolist(),
                "wheel_tau_ms_max_ch2_ch5": wheels.tau[:, [2, 5]].amax(0).cpu().tolist(),
                "old_wheel20ms_tau2_added": False,
            }
            return result

    return DelayedDR


def readback(env, obs):
    """核对生效队列与Critic真实时序上下文，允许reset最初的因果历史填充。"""
    from critic_tensor import FIELDS
    from fixed_recovery import EXTRA_FIELDS

    timing = env.motion_timing
    dr = timing.dr
    wheels = timing.base.motor_ring
    profile = dr.profiles
    nominal = (profile != 0) & (profile != 3)
    expected = wheels.delay.new_tensor([7, 11]).expand_as(wheels.delay[:, [2, 5]])
    critic = obs["student_critic"]
    offset = sum(
        w
        for name, w in FIELDS[
            : next(i for i, x in enumerate(FIELDS) if x[0] == "effective_wheel_timing")
        ]
    )
    actual = torch.cat((wheels.delay[:, [2, 5]].float() / 30, wheels.tau[:, [2, 5]] / 10), -1)
    leg_age_offset = 562 + sum(
        w
        for name, w in EXTRA_FIELDS[
            : next(i for i, x in enumerate(EXTRA_FIELDS) if x[0] == "actor_leg_packet_age_ms")
        ]
    )
    checks = dict(
        wheel_geometric_right7_left11=bool(
            (wheels.delay[nominal][:, [2, 5]] == expected[nominal]).all()
        ),
        wheel_tau_zero_no_duplicate=bool((wheels.tau[:, [2, 5]] == 0).all()),
        wheel_DR_nonnegative_bounded=bool(
            ((wheels.delay[:, [2, 5]] >= 3) & (wheels.delay[:, [2, 5]] <= 15)).all()
        ),
        nominal_actor_all_physical_sensors20=bool(
            ((dr.packet_age[nominal] == 20) & (dr.leg_packet_age[nominal] == 20)).all()
        ),
        actual_Critic_wheel_timing=bool(
            torch.allclose(critic[:, offset : offset + 4], actual, atol=1e-6, rtol=0)
        ),
        actual_Critic_leg_packet_age=bool(
            torch.allclose(
                critic[:, leg_age_offset], env.identified_packets.leg_age / 20, atol=1e-6, rtol=0
            )
        ),
        separate_V4_motor_NET=bool((timing.base.compensation == 0).all()),
        DR_curriculum_enabled=True,
    )
    if not all(checks.values()):
        raise RuntimeError("延迟实际接线不一致: " + str(checks))
    return dict(
        version=VERSION,
        checks=checks,
        wheel_delay_ms_min_ch2_ch5=wheels.delay[:, [2, 5]].amin(0).cpu().tolist(),
        wheel_delay_ms_max_ch2_ch5=wheels.delay[:, [2, 5]].amax(0).cpu().tolist(),
        wheel_tau_ms_max_ch2_ch5=wheels.tau[:, [2, 5]].amax(0).cpu().tolist(),
        nominal_Actor_packet_ms=20,
        nominal_Actor_leg_packet_ms=20,
        configured_packet_age_histogram_ms={
            str(i): int((dr.packet_age == i).sum()) for i in (0, 10, 20)
        },
        configured_leg_packet_age_histogram_ms={
            str(i): int((dr.leg_packet_age == i).sum()) for i in (0, 10, 20)
        },
        effective_leg_packet_age_min_max_ms=[
            float(env.identified_packets.leg_age.min()),
            float(env.identified_packets.leg_age.max()),
        ],
        leg_motor_transport_ms_min=timing.leg.delay.reshape(env.num_envs, 2).amin(0).cpu().tolist(),
        leg_motor_transport_ms_max=timing.leg.delay.reshape(env.num_envs, 2).amax(0).cpu().tolist(),
        old_wheel20ms_tau2_added=False,
        hardware_timing_uniquely_identified=False,
    )
