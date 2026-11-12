"""Motion consumers of the public Measurement; no private clock or sink."""

import copy
from contextlib import nullcontext

from native_stand.artifacts.io import json_digest


def rebuild_inputs(spec, job, *, stage=None):
    """Host-only factory inputs, not sampled physical readback or a rebuild decision."""
    stage = stage if stage is not None else spec["curriculum"]["stages"][0]
    slots = ("slot_counts", "sampling_slots", "sampling_mode", "weights")
    physical_stage = (
        "identified_profiles",
        "identified_amplitudes",
        "dq_amplitudes",
        "PD_feedback_max_ms",
        "mcu_variants",
    )
    excluded = {"id", "name", "cohort_keys", "terrain_families", *slots, *physical_stage}
    state = spec.get("student_training", {}).get("height_dr_state", {})
    return copy.deepcopy(
        dict(
            revision=stage.get("id"),
            keys=stage.get("cohort_keys"),
            slots={k: stage[k] for k in slots if k in stage},
            scene={
                **{
                    k: spec[k]
                    for k in (
                        "ground",
                        "asset_path",
                        "asset_sha256",
                        "asset_module_path",
                        "asset_module_sha256",
                        "colleague_config_path",
                        "colleague_config_sha256",
                    )
                    if k in spec
                },
                "terrain_scene": job.assets.get("terrain_scene"),
                "terrain_families": stage.get("terrain_families"),
            },
            physics={
                **{
                    k: spec[k]
                    for k in (
                        "control",
                        "motion_timing",
                        "mcu_execution",
                        "motion_identified_response",
                        "motion_comprehensive_dr",
                        "motion_dq_robustness",
                        "action_output",
                    )
                    if k in spec
                },
                **{
                    k: job.assets[k]
                    for k in (
                        "delayed_execution",
                        "motor_limits",
                        "wheel_contact_dr",
                        "motion_execution_optimization",
                    )
                    if k in job.assets
                },
                "stage": {k: stage[k] for k in physical_stage if k in stage},
                "DR_state": {
                    k: state[k]
                    for k in (
                        "DR_levels",
                        "dq_level",
                        "general_DR_frontier",
                        "standing_DR_frontier",
                    )
                    if k in state
                },
            },
            task={k: v for k, v in stage.items() if k not in excluded},
        )
    )


def rebuild_attribution(previous, current):
    """Multiple causes may coexist; no claim that a revision needed a rebuild."""
    if previous is None:
        return dict(reasons=["attribution_unavailable"], old=None, new=None)
    changed = {
        k: sorted(set(previous[k]) | set(current[k]))
        for k in ("slots", "scene", "physics", "task")
        if previous[k] != current[k]
    }
    changed = {
        k: [name for name in names if previous[k].get(name) != current[k].get(name)]
        for k, names in changed.items()
    }
    reasons = ["cohort_keys"] if previous["keys"] != current["keys"] else []
    reasons += [k for k in ("slots", "scene", "physics", "task") if k in changed]

    def summary(value):
        return dict(
            revision=value["revision"],
            cohort_keys=value["keys"],
            slot_counts=value["slots"].get("slot_counts"),
            input_sha256={k: json_digest(value[k]) for k in ("slots", "scene", "physics", "task")},
        )

    return dict(
        reasons=reasons or ["statistics_revision_only"],
        changed_fields=changed,
        old=summary(previous),
        new=summary(current),
    )


def section(measurement, name, *, parent=None, reason=None):
    if measurement is None or not measurement.enabled:
        return nullcontext()
    if reason is not None:
        measurement.count(name + ".calls", reason=reason)
    return measurement.section(name, parent=parent, synchronize=False)


def count(measurement, name, amount=1, *, reason=None):
    if measurement is not None and measurement.enabled:
        measurement.count(name, amount, reason=reason)


def cpu(value, measurement, *, reason):
    if value.device.type != "cpu":
        count(
            measurement,
            "motion.explicit_transfer_bytes",
            value.numel() * value.element_size(),
            reason=reason,
        )
    return value.cpu()
