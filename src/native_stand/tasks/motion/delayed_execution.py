"""完整V4腿响应与辨识轮时序的显式配方；不读取文件或修改物理状态。"""

from __future__ import annotations

import torch

VERSION = "motion_v4_wheel_reported_delay_actor20_dr_v1"


def configuration():
    return dict(
        version=VERSION,
        actor_packet_nominal_ms=20,
        actor_leg_packet_nominal_ms=20,
        actor_packet_DR_support_ms=[0, 10, 20],
        target_nominal_ms=5,
        wheel_channels=[2, 5],
        wheel_transport_nominal_ms=[7, 11],
        wheel_tau_ms=[0.0, 0.0],
        wheel_transport_DR_halfwidth_ms=4.0,
        wheel_delay_semantics="reported combined onset, represented once as post-PD transport; not an independently measured pure delay",
        leg_response="complete V4 tuple and existing DR; no additional20ms motor delay",
        DR="existing reset-only per-env curriculum, nominal anchors retained",
        previous="actual locally adopted physical target; not sensor-delayed",
        input_scales=[1.5, 100.0],
        motor_limits_unchanged=True,
    )


def validate_configuration(value):
    expected = configuration()
    # Motor caps have an independent, explicit execution contract.
    if value.get("motor_limits_unchanged") is False:
        expected["motor_limits_unchanged"] = False
        expected["version"] = "motion48_v4_wheel_reported_delay_actor20_dr_v2"
    if dict(value) != expected:
        raise ValueError("V4与辨识轮延迟配方声明不一致")
    return value


def reset_values(ids, episodes, profiles, amplitude, noise, *, config):
    """沿用原无状态DR随机流，不消耗策略RNG，不跨环境共享队列。"""
    validate_configuration(config)
    u = noise(episodes[ids], ids, 96, 509)
    # Profile0 is the combined group; profile3 is the single timing group.
    timing = amplitude[ids, 0] * ((profiles[ids] == 0) | (profiles[ids] == 3)).float()
    choose = ((u[:, 16] + 1) * 1.5).long().clamp_max(2) * 10
    packet = torch.where((u[:, 17] + 1) * 0.5 < timing, choose, 20)
    leg_packet = torch.where((u[:, 54] + 1) * 0.5 < timing, choose, 20)
    nominal = u.new_tensor(config["wheel_transport_nominal_ms"])
    wheel = (
        (nominal + u[:, 34:36] * config["wheel_transport_DR_halfwidth_ms"] * timing[:, None])
        .round()
        .long()
    )
    return dict(
        packet_age=packet,
        leg_packet_age=leg_packet,
        wheel_delay=wheel,
        wheel_tau=u.new_zeros(len(ids), 2),
    )


def apply_reset(dr, ids, noise, *, config):
    values = reset_values(ids, dr.episodes, dr.profiles, dr.amplitude, noise, config=config)
    dr.packet_age[ids] = values["packet_age"]
    dr.leg_packet_age[ids] = values["leg_packet_age"]
    channels = torch.tensor([2, 5], device=ids.device)
    dr.t.base.motor_ring.delay[ids[:, None], channels[None]] = values["wheel_delay"]
    dr.t.base.motor_ring.tau[ids[:, None], channels[None]] = values["wheel_tau"]
    return values
