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
