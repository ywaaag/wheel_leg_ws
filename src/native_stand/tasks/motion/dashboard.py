"""Presentation only: expose actual maneuver statistics, never infer scores or conditions."""


def training_windows(data, config, evidence_id):
    review = data.get("onpolicy") or {}
    return [
        {
            **row,
            "actual_student_policy_samples": row.get("actual_student_policy_samples")
            if row.get("sample_count_status") == "KNOWN"
            else None,
            "actual_all_transitions": row.get("actual_all_transitions")
            if row.get("sample_count_status") == "KNOWN"
            else None,
            "sample_count_status": row.get("sample_count_status", "UNKNOWN"),
            "window_selection_evidence": data.get("training_candidate_evidence"),
            "layer": "training",
            "task": "motion",
            "window": review.get("window"),
            "update": review.get("through_update"),
            "protocol": review.get("version"),
            "DR": row.get("DR_level", review.get("DR_level")),
            "execution_contract": config.get("action_contract"),
            "evidence_id": evidence_id,
        }
        for row in review.get("rows", [])
    ]


def target_description(document):
    """Read the existing goal table, not a second capability or scoring registry."""
    rows = []
    for line in document.splitlines():
        if line.startswith("| M"):
            cells = [x.strip() for x in line.strip("|").split("|")]
            if len(cells) == 4:
                rows.append(
                    dict(
                        zip(
                            ("capability_id", "target", "relationship", "priority"),
                            cells,
                            strict=True,
                        )
                    )
                )
    return {
        "document": "docs/MOTION_CAPABILITY_MATRIX.md",
        "rows": rows,
        "scope": "目标规划，不代表本配方覆盖、课程解锁或模型通过",
    }
