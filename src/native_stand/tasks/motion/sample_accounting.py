"""Explicit window/lifetime sample counts; no inference from episode counts."""

from __future__ import annotations

import copy

import torch

VERSION = "motion_student_transition_accounting_v1"


def window_anchor(window):
    return {
        k: copy.deepcopy(window[k])
        for k in ("run", "comparison_sha256", "stage_revision", "start_update")
    }


class SampleAccounting:
    def __init__(self, cohorts, device):
        self.total = torch.zeros(2, dtype=torch.long, device=device)
        self.cohort = torch.zeros(2, cohorts, dtype=torch.long, device=device)
        self.window = torch.zeros_like(self.cohort)
        self.policy_steps = self.window_policy_steps = 0
        self.total_known = [True, True]
        self.policy_steps_known = True
        self.window_known = True
        self.anchor = None
        self.unknown_reason = None

    def bind(self, window):
        anchor = window_anchor(window)
        if self.anchor is not None and self.anchor != anchor:
            raise ValueError("Motion sample window origin differs")
        if self.anchor is None and self.window_policy_steps:
            self.window_known = False
            self.unknown_reason = "samples_precede_declared_window_origin"
        self.anchor = anchor

    def observe(self, cohort, student):
        students = student.long()
        all_rows = torch.ones_like(students)
        self.window[0].index_add_(0, cohort, students)
        self.window[1].index_add_(0, cohort, all_rows)
        self.cohort[0].index_add_(0, cohort, students)
        self.cohort[1].index_add_(0, cohort, all_rows)
        self.total[0].add_(students.sum())
        self.total[1].add_(student.numel())
        self.policy_steps += 1
        self.window_policy_steps += 1

    def summary(self):
        known = self.window_known and self.anchor is not None
        rows = self.window.cpu().tolist() if known else None
        totals = self.total.cpu().tolist()
        return dict(
            version=VERSION,
            window_origin=copy.deepcopy(self.anchor) if known else None,
            window_status="KNOWN" if known else "UNKNOWN",
            unknown_reason=None if known else self.unknown_reason or "window_origin_not_declared",
            student_policy_samples=sum(rows[0]) if known else None,
            all_transitions=sum(rows[1]) if known else None,
            student_samples_by_cohort=rows[0] if known else None,
            all_transitions_by_cohort=rows[1] if known else None,
            observed_policy_steps=self.window_policy_steps if known else None,
            cumulative_student_policy_samples=totals[0] if self.total_known[0] else None,
            cumulative_all_transitions=totals[1] if self.total_known[1] else None,
            cumulative_observed_policy_steps=self.policy_steps if self.policy_steps_known else None,
            cumulative_scope="declared counter lineage, including environment rebuilds",
            cohort_cumulative_scope="current environment/cohort layout only",
            cumulative_samples_by_cohort=[
                row if known else None
                for row, known in zip(self.cohort.cpu().tolist(), self.total_known, strict=True)
            ],
        )

    def advance(self, window):
        self.window.zero_()
        self.window_policy_steps = 0
        self.window_known = True
        self.unknown_reason = None
        self.anchor = {**window_anchor(window), "start_update": window["end_update"] + 1}

    def state_dict(self):
        return dict(
            version=VERSION,
            total=self.total.detach().cpu().clone(),
            cohort=self.cohort.detach().cpu().clone(),
            window=self.window.detach().cpu().clone(),
            policy_steps=self.policy_steps,
            window_policy_steps=self.window_policy_steps,
            total_known=list(self.total_known),
            policy_steps_known=self.policy_steps_known,
            window_known=self.window_known,
            anchor=copy.deepcopy(self.anchor),
            unknown_reason=self.unknown_reason,
        )

    def restore(self, record, *, totals_only=False):
        if record.get("version") != VERSION:
            raise ValueError("Unknown Motion sample accounting version")
        if (
            type(record.get("policy_steps")) is not int
            or record["policy_steps"] < 0
            or type(record.get("window_policy_steps")) is not int
            or not 0 <= record["window_policy_steps"] <= record["policy_steps"]
            or len(record.get("total_known", [])) != 2
            or any(type(x) is not bool for x in record["total_known"])
            or type(record.get("window_known")) is not bool
            or type(record.get("policy_steps_known")) is not bool
        ):
            raise ValueError("Damaged Motion sample accounting metadata")
        names = ("total",) if totals_only else ("total", "cohort", "window")
        for name in names:
            value, target = record[name], getattr(self, name)
            if value.shape != target.shape or value.dtype != target.dtype or (value < 0).any():
                raise ValueError("Damaged Motion sample counters: " + name)
        for name in names:
            getattr(self, name).copy_(record[name].to(self.total.device))
        self.policy_steps = record["policy_steps"]
        self.total_known = list(record["total_known"])
        self.policy_steps_known = record["policy_steps_known"]
        if not totals_only:
            self.window_policy_steps = record["window_policy_steps"]
            self.window_known = record["window_known"]
            self.anchor = copy.deepcopy(record["anchor"])
            self.unknown_reason = record["unknown_reason"]

    def restore_legacy(self, student_total=None, student_cohorts=None):
        # Legacy counters contain no all-transition count or window baseline.
        # Preserve their known Student total without fabricating the other data.
        self.total_known = [student_total is not None, False]
        self.policy_steps_known = False
        if student_total is not None:
            self.total[0].copy_(student_total.to(self.total))
            self.cohort[0].copy_(student_cohorts.to(self.cohort))
        self.window_known = False
        self.unknown_reason = "legacy_checkpoint_missing_window_origin"
