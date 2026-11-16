"""Read-only Teacher routing and explicit conversion via physical targets."""

from dataclasses import dataclass

import torch

from native_stand.learning.inference import InferenceModel


@dataclass
class ActionLabels:
    """A declared invertible label transform, separate from execution protection."""

    kind: str
    gains: tuple = ()
    center: tuple = ()
    lower: tuple = ()
    upper: tuple = ()
    contract: dict | None = None
    support_shift: dict | None = None

    def __post_init__(self):
        if self.kind == "motion_support_teacher":
            from native_stand.tasks.motion.action_contract import ActionDecoder
            from native_stand.tasks.motion.support_teacher import validate

            validate(self.support_shift)
            ActionDecoder.from_contract(self.contract)
        elif self.support_shift is not None:
            raise ValueError("Support shift only belongs to explicit Motion Teacher labels")

    def physical(self, raw, observation=None):
        if self.kind in ("motion_contract", "motion_support_teacher"):
            from native_stand.tasks.motion.action_contract import ActionDecoder

            if observation is None:
                raise ValueError("Motion labels require actual height command")
            result = ActionDecoder.from_contract(self.contract).decode(
                raw, observation[:, 24] * 0.5
            )["requested_target"]
            if self.kind == "motion_support_teacher":
                from native_stand.tasks.motion.support_teacher import tensor_shift

                result[:, [0, 1, 3, 4]] += tensor_shift(
                    observation[:, 24] * 0.5, self.support_shift
                )
            return result
        if self.kind == "physical":
            return raw
        if self.kind == "affine":
            return raw * raw.new_tensor(self.gains) + raw.new_tensor(self.center)
        if self.kind == "recovery_cubic":
            out = raw * 100
            legs = [0, 1, 3, 4]
            x = raw[:, legs]
            c, lo, hi = (raw.new_tensor(v) for v in (self.center, self.lower, self.upper))
            reach = torch.where(x >= 0, hi - c, c - lo)
            out[:, legs] = c + 0.15 * x + (reach - 0.15) * x.pow(3)
            return out
        raise ValueError("Unsupported physical label transform")

    def labels(self, physical, observation=None):
        if self.kind == "motion_contract":
            from native_stand.tasks.motion.action_contract import ActionDecoder

            decoder = ActionDecoder.from_contract(self.contract)
            center = self.physical(torch.zeros_like(physical), observation)
            converted = (physical - center) / physical.new_tensor(
                [
                    decoder.leg_scale,
                    decoder.leg_scale,
                    decoder.wheel_scale,
                    decoder.leg_scale,
                    decoder.leg_scale,
                    decoder.wheel_scale,
                ]
            )
            if bool((converted.abs() > 1 + 1e-6).any()):
                raise ValueError("Teacher physical target outside Student Motion contract")
            return converted
        if self.kind == "physical":
            return physical
        if self.kind == "affine":
            gains = physical.new_tensor(self.gains)
            if bool((gains <= 0).any()):
                raise ValueError("Positive affine gains required")
            return (physical - physical.new_tensor(self.center)) / gains
        if self.kind == "recovery_cubic":
            legs = [0, 1, 3, 4]
            lo, hi = physical.new_tensor(self.lower), physical.new_tensor(self.upper)
            if bool(((physical[:, legs] < lo) | (physical[:, legs] > hi)).any()):
                raise ValueError("Teacher physical target outside invertible Student range")
            result = physical / 100
            a, b = torch.full_like(physical, -1), torch.ones_like(physical)
            for _ in range(40):
                mid = (a + b) / 2
                below = self.physical(mid) < physical
                a = torch.where(below, mid, a)
                b = torch.where(below, b, mid)
            result[:, legs] = ((a + b) / 2)[:, legs]
            return result
        raise ValueError("No explicit inverse label transform")


@dataclass
class Teacher:
    name: str
    model: InferenceModel
    codec: ActionLabels
    priority: int
    conditions: dict
    qualified: bool = True
    observation_adapter: str | None = None
    action_adapter: str | None = None

    def required_conditions(self):
        keys = set(self.conditions)
        if self.action_adapter is not None:
            from native_stand.learning.distillation.jump_labels import executor_condition_names

            keys.update(executor_condition_names(self.action_adapter))
        return keys

    def eligible(self, conditions, count, device):
        mask = torch.full((count,), self.qualified, dtype=torch.bool, device=device)
        for key, wanted in self.conditions.items():
            if key not in conditions:
                raise ValueError("Missing Teacher routing condition: " + key)
            value = conditions[key]
            if isinstance(wanted, str):
                import numpy as np

                condition = torch.as_tensor(np.asarray(value) == wanted, device=device)
            else:
                value = torch.as_tensor(value, device=device)
                if value.shape != (count,) or not bool(torch.isfinite(value).all()):
                    raise ValueError("Invalid routing condition rows")
                condition = value == wanted
            if condition.shape != (count,):
                raise ValueError("Invalid routing condition shape")
            mask &= condition
        return mask


def label(teachers, observation, conditions, student_codec, *, student_kind=None):
    """No Teacher writes owner/previous/FIFO; uncovered rows remain unlabelled."""
    if observation.ndim != 2 or not bool(torch.isfinite(observation).all()):
        raise ValueError("Finite Teacher observations required")
    labels = observation.new_full((len(observation), 6), float("nan"))
    owner = torch.full((len(observation),), -1, dtype=torch.long, device=observation.device)
    for index, teacher in sorted(enumerate(teachers), key=lambda row: -row[1].priority):
        mask = (owner < 0) & teacher.eligible(conditions, len(observation), observation.device)
        if teacher.observation_adapter is not None:
            from native_stand.learning.distillation.jump_labels import motion_teacher_eligible

            mask &= motion_teacher_eligible(observation, teacher.observation_adapter)
        if bool(mask.any()):
            teacher_observation = observation[mask]
            if teacher.observation_adapter is not None:
                from native_stand.learning.distillation.jump_labels import (
                    motion_teacher_observation,
                )

                teacher_observation = motion_teacher_observation(
                    teacher_observation, teacher.observation_adapter
                )
            physical = teacher.codec.physical(
                teacher.model.mean(teacher_observation), teacher_observation
            )
            if teacher.action_adapter is not None:
                from native_stand.learning.distillation.jump_labels import match_jump_executor

                physical = match_jump_executor(physical, conditions, mask, teacher.action_adapter)
            converted = student_codec.labels(physical, observation[mask])
            if student_kind == "tanh" and bool((converted.abs() > 1).any()):
                raise ValueError("Teacher target outside Student Tanh image; no clipping")
            if not bool(torch.isfinite(converted).all()):
                raise ValueError("Nonfinite Teacher label")
            labels[mask] = converted
            owner[mask] = index
    return labels, owner, owner >= 0
