"""Explicit per-command source retention; normalize distributions before tasks."""

from __future__ import annotations

import copy

import torch

from .nominal_bank import VERSION as LEGACY_VERSION
from .nominal_bank import NominalBankAnchor

VERSION = "motion48_per_command_nominal_source_KL_v4"
STATE_VERSION = "motion48_command_bank_rotation_v2"


class CommandBankAnchor(NominalBankAnchor):
    def __init__(self, teacher, config, data):
        if (
            config.get("version") != VERSION
            or config.get("bank_fraction_when_onpolicy") != 0.5
            or config.get("task_reduction") != "mean_over_declared_commands"
        ):
            raise ValueError("Explicit per-command source KL normalization required")
        super().__init__(teacher, {**config, "version": LEGACY_VERSION}, data)
        self.config = copy.deepcopy(config)

    def reset(self):
        super().reset()
        self.command_losses = None
        self.bank_fractions = None
        self.onpolicy_fractions = None

    def combine_losses(self, bank, onpolicy):
        fraction = self.config["bank_fraction_when_onpolicy"]
        bw = [fraction if x is not None else 1.0 for x in onpolicy]
        cw = [1.0 - fraction if x is not None else 0.0 for x in onpolicy]
        b = torch.stack([w * x for w, x in zip(bw, bank, strict=False)])
        c = torch.stack(
            [
                w * x if x is not None else bank[i].new_zeros(())
                for i, (w, x) in enumerate(zip(cw, onpolicy, strict=False))
            ]
        )
        terms = b + c
        detached = terms.detach()
        self.command_losses = (
            detached if self.command_losses is None else self.command_losses + detached
        )
        wb = detached.new_tensor(bw)
        wc = detached.new_tensor(cw)
        self.bank_fractions = wb if self.bank_fractions is None else self.bank_fractions + wb
        self.onpolicy_fractions = (
            wc if self.onpolicy_fractions is None else self.onpolicy_fractions + wc
        )
        return terms.mean(), b.mean(), c.mean()

    def state_dict(self):
        return dict(
            **super().state_dict(),
            loss_version=VERSION,
            command_state_version=STATE_VERSION,
            bank_fraction_when_onpolicy=self.config["bank_fraction_when_onpolicy"],
        )

    def load_state_dict(self, state):
        if (
            state.get("loss_version") != VERSION
            or state.get("command_state_version") != STATE_VERSION
            or state.get("bank_fraction_when_onpolicy")
            != self.config["bank_fraction_when_onpolicy"]
        ):
            raise ValueError("Per-command KL aggregation state/config differs")
        super().load_state_dict(state)

    def summary(self):
        value = super().summary()
        count = max(1, self.calls)
        value.update(
            version=VERSION,
            task_reduction=self.config["task_reduction"],
            bank_fraction_when_onpolicy=self.config["bank_fraction_when_onpolicy"],
            weighted_loss_by_command=None
            if self.command_losses is None
            else (self.command_losses * self.config["coefficient"] / count).cpu().tolist(),
            mean_bank_fraction_by_command=None
            if self.bank_fractions is None
            else (self.bank_fractions / count).cpu().tolist(),
            mean_onpolicy_fraction_by_command=None
            if self.onpolicy_fractions is None
            else (self.onpolicy_fractions / count).cpu().tolist(),
            missing_onpolicy="use the same command bank, without losing that task or raising another task weight",
            coefficient_calibration="0.02 times task mean matches old0.01 times two distribution means for equal errors when onpolicy exists",
        )
        return value
