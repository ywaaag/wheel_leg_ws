"""Balanced nominal source KL; replay never enters PPO storage or GAE."""

from __future__ import annotations

import copy

import torch

from .source_anchor import CAUSAL_VERSION, KEY, SourceAnchor

VERSION = "motion48_qualified_nominal_bank_source_KL_v3"
STATE_VERSION = "motion48_nominal_bank_rotation_v1"


class NominalBankAnchor(SourceAnchor):
    """Files and identities are checked by the entrypoint, not this loss."""

    def __init__(self, teacher, config, data):
        if config.get("version") != VERSION or config.get("bank_states_per_command") != 64:
            raise ValueError("Explicit balanced nominal-bank version/batch required")
        super().__init__(teacher, {**config, "version": CAUSAL_VERSION})
        self.config = copy.deepcopy(config)
        device = self.std.device
        x = torch.as_tensor(data["actor_obs"], device=device, dtype=torch.float32).detach().clone()
        y = (
            torch.as_tensor(data["source_mean"], device=device, dtype=torch.float32)
            .detach()
            .clone()
        )
        group = torch.as_tensor(data["cohort"], device=device, dtype=torch.long)
        if x.ndim != 2 or x.shape[1] != 31 or y.shape != (len(x), 6) or group.shape != (len(x),):
            raise ValueError("Nominal bank31D/6D/group shapes differ")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all() or (y.abs() > 1.00001).any():
            raise ValueError("Nonfinite or out-of-contract nominal bank")
        if (group < 0).any() or (group >= len(config["commands"])).any():
            raise ValueError("Nominal bank contains an undeclared command")
        self.bank = []
        with torch.no_grad():
            for i, command in enumerate(config["commands"]):
                ids = (group == i).nonzero(as_tuple=True)[0]
                if len(ids) < 128:
                    raise ValueError("Source protection command has insufficient qualified anchors")
                bx, by = x[ids], y[ids]
                expected = bx.new_tensor([command[0] / 4.0, command[1] / 4.0, 0.34 / 0.5])
                if not torch.allclose(bx[:, 22:25], expected.expand(len(bx), 3), atol=2e-6, rtol=0):
                    raise ValueError(
                        "Anchor inputs do not contain their declared settled nominal command"
                    )
                if not torch.allclose(self.teacher.mean(bx), by, atol=3e-6, rtol=1e-5):
                    raise ValueError("Anchor labels do not match the exact frozen source")
                self.bank.append((bx, by))
        self.bank_cursors = [0] * len(self.bank)
        self.bank_positions = torch.arange(config["bank_states_per_command"], device=device)
        self.reset()

    def reset(self):
        super().reset()
        self.bank_uses = [0] * len(self.config["commands"])
        self.onpolicy_total = 0.0
        self.bank_total = 0.0
        self.physical_error = None

    def loss(self, actor, observations, valid):
        x = observations["student_actor"]
        if not bool(valid.any()):
            return x.sum() * 0.0
        ids = observations[KEY].squeeze(-1)
        if ids.dtype != torch.long:
            raise ValueError("Retention cohort requires integer metadata")
        bank_losses = []
        onpolicy_losses = [None] * len(self.bank)
        errors = []
        for i, (bx, by) in enumerate(self.bank):
            positions = (self.bank_positions + self.bank_cursors[i]) % len(bx)
            target = by[positions]
            mean = actor.mean(bx[positions])
            self.bank_cursors[i] = (self.bank_cursors[i] + len(positions)) % len(bx)
            self.bank_uses[i] += len(positions)
            difference = mean - target
            bank_losses.append(0.5 * (difference / self.std).square().mean())
            physical = difference.detach().abs() * mean.new_tensor(
                self.config.get("physical_output_scales", [0.15, 0.15, 80.0, 0.15, 0.15, 80.0])
            )
            errors.append(
                torch.stack((physical[:, [0, 1, 3, 4]].mean(), physical[:, [2, 5]].mean()))
            )
            selected = (valid & (ids == i)).nonzero(as_tuple=True)[0]
            if not len(selected):
                continue
            n = min(self.config["maximum_states_per_cohort"], len(selected))
            offsets = torch.div(
                torch.arange(n, device=x.device) * len(selected), n, rounding_mode="floor"
            )
            sample = x[selected[offsets]].detach()
            with torch.no_grad():
                reference = self.teacher.mean(sample)
            onpolicy_losses[i] = 0.5 * ((actor.mean(sample) - reference) / self.std).square().mean()
            self.uses[i] += n
        combined, bank_loss, current_loss = self.combine_losses(bank_losses, onpolicy_losses)
        # Same KL definition, distinct input distributions. No second BC term
        # on the same rows; no labels are inserted into the on-policy rollout.
        result = self.config["coefficient"] * combined
        self.bank_total += float(bank_loss.detach())
        self.onpolicy_total += float(current_loss.detach())
        self.total += float(combined.detach())
        self.calls += 1
        error = torch.stack(errors)
        self.physical_error = error if self.physical_error is None else self.physical_error + error
        return result

    def combine_losses(self, bank, onpolicy):
        """Historical v3 reduction, kept exactly for old jobs and resumes."""
        b = torch.stack(bank).mean()
        present = [x for x in onpolicy if x is not None]
        c = torch.stack(present).mean() if present else b.new_zeros(())
        return b + c, b, c

    def state_dict(self):
        return dict(
            version=STATE_VERSION,
            bank_sha256=self.config["bank"]["sha256"],
            cursors=list(self.bank_cursors),
        )

    def load_state_dict(self, state):
        cursors = state.get("cursors", [])
        if (
            state.get("version") != STATE_VERSION
            or state.get("bank_sha256") != self.config["bank"]["sha256"]
            or len(cursors) != len(self.bank)
            or any(
                not isinstance(v, int) or not 0 <= v < len(self.bank[i][0])
                for i, v in enumerate(cursors)
            )
        ):
            raise ValueError("Nominal bank rotation/identity cannot be restored")
        self.bank_cursors = list(cursors)

    def summary(self):
        value = super().summary()
        value.update(
            version=VERSION,
            actual_bank_selected_visits_by_cohort=self.bank_uses,
            bank_rows_by_cohort=[len(x) for x, _ in self.bank],
            bank_sha256=self.config["bank"]["sha256"],
            bank_weighted_loss_mean=self.config["coefficient"]
            * self.bank_total
            / max(1, self.calls),
            onpolicy_weighted_loss_mean=self.config["coefficient"]
            * self.onpolicy_total
            / max(1, self.calls),
            bank_physical_target_MAE_by_command=None
            if self.physical_error is None
            else (self.physical_error / max(1, self.calls)).cpu().tolist(),
            physical_MAE_columns=["leg_rad", "wheel_rad_s"],
            bank_rotation=self.state_dict(),
            inputs="qualified training-only nominal anchors plus current causal qualified nominal states",
            replay_in_PPO_returns=False,
            extra_optimizer_steps=0,
            source_teacher_frozen=True,
        )
        return value
