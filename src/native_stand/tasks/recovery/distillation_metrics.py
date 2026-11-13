"""Closed-loop first-episode readback; incomplete episodes remain UNKNOWN."""

import numpy as np


def summarize(arrays):
    keys = arrays["physical_episode_keys"]
    rows = []
    for key in np.unique(keys):
        mask = keys == key
        # Subsequent resets do not replace a failed first attempt.
        if not np.all(arrays["episodes"][mask] == 0):
            continue
        ended = bool(np.any((arrays["terminated"] | arrays["truncated"])[mask]))
        entered = arrays["condition_entered"][mask].astype(bool)
        dual = arrays["condition_dual_contact"][mask].astype(bool)
        hold = arrays["condition_continuous_hold_s"][mask]
        rows.append(
            dict(
                episode=str(key),
                complete=ended,
                cohort=int(arrays["condition_validation_cohort"][mask][0]),
                environment_id=int(arrays["environment_id"][mask][0])
                if "environment_id" in arrays
                else None,
                Swing=bool(arrays["condition_initial_Swing"][mask][0]),
                rise=bool(entered.any()) if ended else None,
                lost_contact=bool((entered & ~dual).any()) if ended else None,
                hold10=bool(hold.max() >= 10 - 1e-5) if ended else None,
                HF_peak=float(arrays["condition_HF_peak"][mask].max()),
                raw_hold10=bool(arrays["condition_raw_longest_hold_s"][mask].max() >= 10 - 1e-5)
                if "condition_raw_longest_hold_s" in arrays and ended
                else None,
            )
        )
    complete = [r for r in rows if r["complete"]]
    swing = [row for row in complete if row["Swing"]]
    return dict(
        first_episodes=len(rows),
        completed=len(complete),
        UNKNOWN=len(rows) - len(complete),
        rise_success_rate=np.mean([r["rise"] for r in swing]).item() if swing else None,
        settling_contact_loss_rate=np.mean([r["lost_contact"] for r in complete]).item()
        if complete
        else None,
        continuous_10s_rate=np.mean([r["hold10"] for r in complete]).item() if complete else None,
        HF_peak=max((r["HF_peak"] for r in rows), default=None),
        raw_continuous_10s_rate=np.mean([r["raw_hold10"] for r in complete]).item()
        if complete and all(r["raw_hold10"] is not None for r in complete)
        else None,
        delay_robustness={
            str(c): dict(
                completed=sum(r["cohort"] == c for r in complete),
                continuous_10s_rate=np.mean(
                    [r["hold10"] for r in complete if r["cohort"] == c]
                ).item(),
            )
            for c in sorted({r["cohort"] for r in complete})
        },
        episodes=rows,
    )


def paired_retention(teacher, student):
    """Extra success on Teacher-failed cases never offsets lost Teacher capability."""

    def mapping(report):
        result = {}
        for row in report["episodes"]:
            key = (row.get("environment_id"), row["cohort"])
            if key in result:
                raise ValueError("Paired retention needs explicit unique environment/case identity")
            result[key] = row
        return result

    left, right = mapping(teacher), mapping(student)
    if left.keys() != right.keys():
        raise ValueError("Teacher/student condition identities differ")
    rows = []
    for key, t in left.items():
        s = right[key]
        if bool(t["Swing"]) != bool(s["Swing"]):
            raise ValueError("Teacher/student reset pose identity differs")
        if not t["complete"] or not s["complete"]:
            category = "UNKNOWN"
        else:
            expected = t["rise"] if t["Swing"] else t["hold10"]
            actual = s["rise"] if s["Swing"] else s["hold10"]
            category = (
                ("RETAINED" if actual else "STUDENT_ONLY_LOSS")
                if expected
                else ("EXTRA_NOT_REQUIRED" if actual else "TEACHER_BOUNDARY")
            )
        rows.append(
            dict(
                environment_id=key[0],
                cohort=key[1],
                Swing=t["Swing"],
                category=category,
                teacher_complete=t["complete"],
                student_complete=s["complete"],
                teacher_rise=t["rise"],
                student_rise=s["rise"],
                teacher_hold10=t["hold10"],
                student_hold10=s["hold10"],
            )
        )
    result = dict(version="teacher_success_condition_retention_v1", cases=rows)
    for metric, kind in [("teacher_rise_retention", "rise"), ("teacher_hold_retention", "hold10")]:
        eligible = [
            r
            for r in rows
            if r["teacher_complete"] and r["teacher_" + kind] and (kind != "rise" or r["Swing"])
        ]
        unknown = sum(not r["student_complete"] for r in eligible)
        kept = sum(bool(r["student_" + kind]) for r in eligible if r["student_complete"])
        result[metric] = kept / len(eligible) if eligible and not unknown else None
        result[metric + "_counts"] = dict(
            teacher_success=len(eligible), student_retained=kept, UNKNOWN=unknown
        )
    return result
