"""40→80显式输出版本迁移。只变轮输出头与轮std，不修改源文件。"""

from __future__ import annotations

import copy
import math

import torch

OLD_ACTION = "motion31_kinematic_support_centered_015_40_v1"
NEW_ACTION = "motion31_kinematic_support_centered_015_80_v1"


def migrate_wheel80(source):
    if source["action_version"] != OLD_ACTION:
        raise ValueError("仅支持已声明的015/40来源，不能重复迁移")
    target = copy.deepcopy(source)
    # 保持同一MLP/Tanh拓扑；这是一阶保持，不宣称Tanh物理函数完全相同。
    for name in ("mlp.6.weight", "mlp.6.bias"):
        target["actor"][name][[2, 5]] *= 0.5
    target["actor"]["log_std"][[2, 5]] -= math.log(2.0)
    target["std"] = target["actor"]["log_std"].clone()
    actor_names = list(source["actor"])
    actor_ids = source["optimizer"]["param_groups"][0]["params"]
    if len(actor_names) != len(actor_ids):
        raise ValueError("Actor Adam参数身份与显式state_dict布局不匹配")
    for name, pid in zip(actor_names, actor_ids, strict=False):
        if name in ("mlp.6.weight", "mlp.6.bias"):
            for moment in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = target["optimizer"]["state"].get(pid, {}).get(moment)
                if value is not None:
                    value[[2, 5]] = 0
    target["action_version"] = NEW_ACTION
    target["output_contract"].update(
        action_version=NEW_ACTION,
        wheel_halfwidth_rad_s=80.0,
        kind="EXPLICIT_PPO_OUTPUT_CONTRACT",
        runtime_recipe_ready=True,
        deployment_ready=False,
    )
    target["output_migration"] = dict(
        version="wheel40_to80_head_first_order_v1",
        source_action=OLD_ACTION,
        action=NEW_ACTION,
        source_online_updates=source["online_updates_including_smoke"],
        source_recipe_updates=source["recipe_updates"],
        formula="wheel final Linear rows times0.5;80*tanh(z/2), versus old40*tanh(z)",
        exact_function_preservation=False,
        optimizer="all state preserved except wheel final-head moment rows zeroed; shared step counts preserved",
        exploration="physical leg0.02rad/wheel2rad_s unchanged; raw wheel std0.025",
        Critic="full causal863 preserved, no repeated warmup",
        input_scales=[1.5, 100.0],
        output_scales=[0.15, 80.0],
    )
    return target


def physical_migration_error(actor_before, actor_after, observations):
    with torch.no_grad():
        a = actor_before.mean(observations)
        b = actor_after.mean(observations)
        error = (b[:, [2, 5]] * 80 - a[:, [2, 5]] * 40).abs()
        return dict(
            states=len(observations),
            wheel_mean_abs_error_rad_s=float(error.mean()),
            wheel_max_abs_error_rad_s=float(error.max()),
            leg_raw_max_error=float((a[:, [0, 1, 3, 4]] - b[:, [0, 1, 3, 4]]).abs().max()),
        )
