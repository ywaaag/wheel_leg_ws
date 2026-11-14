"""Bounded supervised Jump collection; the public collector owns execution/data."""

import numpy as np
import torch


def collection_commands(job):
    config = job.assets["jump_balance_collection"]
    if (
        job.algorithm != "distillation"
        or job.supervision is None
        or job.backend != "isaac"
        or job.assets.get("jump_compute") != "cuda_firmware_warp_inverse_v1"
        or config.get("version") != "jump_balance_student_collection_v1"
    ):
        raise ValueError("Explicit supervised-only Jump collection required")
    commands = np.asarray(config.get("commands"), dtype=float)
    if commands.ndim != 2 or commands.shape[1] != 3 or not len(commands):
        raise ValueError("Explicit vx/yaw/height collection commands required")
    if not np.isfinite(commands).all() or (
        (abs(commands[:, 0]) > 0.5).any()
        or (abs(commands[:, 1]) > 0.5).any()
        or (abs(commands[:, 2] - 0.34) > 1e-8).any()
    ):
        raise ValueError("Bounded basic movement collection envelope")
    return commands[np.arange(job.runtime.envs) % len(commands)]


class BalanceCollection:
    """Student-only execution; Teacher inference never mutates physical caches."""

    def __init__(self, env):
        self.env, self.num_envs = env, env.num_envs

    def observations(self):
        return self.env.observations()

    def step(self, action):
        measured = self.env.scene.measured()
        # Teacher only labels supported, upright student states. Uncovered
        # states stay recorded/unlabelled, never converted to success labels.
        safe = (
            measured["contact"].all(-1)
            & ~measured["restricted"]
            & (measured["tilt"].abs().amax(-1) < 0.35)
            & (measured["root"][:, 2] > 0.25)
        ).clone()
        from native_stand.simulation.jump_device_c import abi_field
        from native_stand.simulation.jump_shared_c import Input

        inputs = self.env.execution.gpu.source_inputs
        length = abi_field(inputs, Input, "leg_length")
        spread = abi_field(inputs, Input, "leg_spread")
        spring = ((-83.121 * length + 40.242) * 0.8).clamp(16.5, 27.0) * spread.sin()
        # Exact policy-order map of the existing original C compensation.
        compensation = torch.stack(
            (spring[:, 1], spring[:, 1], -spring[:, 0], spring[:, 0]), -1
        ).clone()
        result = self.env.step(action)
        result["extras"]["collection_conditions"] = {"teacher_safe": safe}
        for i in range(4):
            result["extras"]["collection_conditions"]["jump_spring_model_nm_" + str(i)] = (
                compensation[:, i]
            )
        return result

    def close(self):
        self.env.close()
