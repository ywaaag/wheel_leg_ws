"""Pure fixed-Actor replication metrics; explicit windows, no physical execution."""

from __future__ import annotations

import numpy as np

LEGS = [0, 1, 3, 4]


def band_rms(value, rate=1000, low=10, high=40):
    value = np.asarray(value, dtype=float)
    if len(value) < rate // low:
        return None
    spectrum = np.fft.rfft(value - value.mean(0), axis=0)
    frequency = np.fft.rfftfreq(len(value), 1 / rate)
    spectrum[(frequency < low) | (frequency > high)] = 0
    filtered = np.fft.irfft(spectrum, n=len(value), axis=0)
    return np.sqrt(np.mean(filtered**2, axis=0)).tolist()


def longest(mask, dt=0.001):
    padded = np.r_[False, np.asarray(mask, bool), False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return float(max(changes[1::2] - changes[::2], default=0) * dt)


def summarize(trace, case, termination, gates):
    t = np.asarray(trace["time"])
    window = case["score_window_s"]
    mask = (t >= window[0]) & (t < window[1])
    complete = termination is None and len(t) == round(case["duration_s"] * 1000)
    if mask.sum() < 1000:
        return dict(
            status="FAIL" if termination == "attitude_or_height_boundary" else "UNKNOWN",
            completed=complete,
            termination=termination,
            capability_accepted=False,
            reason="insufficient scored physical samples",
            teacher_eligible=False,
        )

    def take(k):
        return np.asarray(trace[k])[mask]

    velocity, command = take("vx"), take("command")
    dq, q = take("dq")[:, LEGS], take("q")[:, LEGS]
    support = (take("normal_force_N") > gates["support_force_N"]).all(1)
    hf = band_rms(dq)
    result = dict(
        status="OBSERVED",
        completed=complete,
        termination=termination,
        score_window_s=window,
        samples_1ms=int(mask.sum()),
        sampling_Hz=1000,
        bandpass_method="mean removal + rectangular FFT 10-40Hz inclusive; scored window only; not HP10",
        vx_mean_mps=float(velocity.mean()),
        vy_mean_mps=float(take("vy").mean()),
        vx_MAE_mps=float(np.abs(velocity - command[:, 0]).mean()),
        yaw_mean_rad_s=float(take("yaw").mean()),
        yaw_MAE_rad_s=float(np.abs(take("yaw") - command[:, 1]).mean()),
        height_MAE_m=float(np.abs(take("height") - command[:, 2]).mean()),
        roll_abs_peak_rad=float(np.abs(take("roll")).max()),
        pitch_abs_peak_rad=float(np.abs(take("pitch")).max()),
        dual_support_fraction=float(support.mean()),
        longest_support_loss_s=longest(~support),
        q_band10_40_RMS_rad=band_rms(q),
        dq_band10_40_RMS_rad_s=hf,
        dq_full_RMS_rad_s=np.sqrt((dq**2).mean(0)).tolist(),
        velocity_std_mps=float(velocity.std()),
        yaw_std_rad_s=float(take("yaw").std()),
        drift_distance_m=float(np.linalg.norm(take("position_xy")[-1] - take("position_xy")[0])),
        motor_saturation_fraction=float(
            (np.abs(take("motor")) >= np.array([40, 40, 4.8, 40, 40, 4.8]) * 0.999).mean()
        ),
        target_saturation_fraction=float((np.abs(take("raw")) >= 0.999).mean()),
        target_adoption_error_mean=float(np.abs(take("requested") - take("adopted")).mean()),
        mirror_leg_error_mean_rad=float(np.abs(q[:, [0, 1]] - q[:, [2, 3]] * [-1, 1]).mean()),
        capability_accepted=False,
    )
    support_pass = (
        result["dual_support_fraction"] > gates["dual_support_fraction"]
        if gates.get("support_fraction_comparison", "ge") == "gt"
        else result["dual_support_fraction"] >= gates["dual_support_fraction"]
    )
    quality = (
        complete
        and result["height_MAE_m"] <= gates["height_MAE_m"]
        and support_pass
        and result["longest_support_loss_s"] <= gates["longest_support_loss_s"]
        and max(result["roll_abs_peak_rad"], result["pitch_abs_peak_rad"]) <= gates["attitude_rad"]
    )
    task = (
        result["vx_MAE_mps"]
        <= max(
            gates["vx_MAE_floor_mps"],
            gates["vx_MAE_fraction"] * float(np.abs(command[:, 0]).mean()),
        )
        and result["yaw_MAE_rad_s"] <= gates["yaw_MAE_rad_s"]
    )
    result["teacher_eligible"] = bool(quality and task)
    result["absolute_stationary_drift_pass"] = (
        bool(np.abs(velocity).mean() <= gates["absolute_stationary_speed_mps"])
        if np.all(command[:, :2] == 0)
        else None
    )
    # Per command phase: transient response and stopping must not disappear in a steady average.
    phases = []
    for start, end in zip(
        case["commands"], case["commands"][1:] + [[case["duration_s"], 0, 0, 0.34]], strict=True
    ):
        a, b = start[0], end[0]
        selected = (t >= a) & (t < b)
        if not selected.any():
            continue
        v = np.asarray(trace["vx"])[selected]
        target = start[1]
        error = np.abs(v - target)
        good = error <= gates["response_tolerance_mps"]
        stable = None
        for i in np.flatnonzero(good):
            if i + 1000 <= len(good) and good[i : i + 1000].all():
                stable = float(i * 0.001)
                break
        phases.append(
            dict(
                start_s=a,
                end_s=b,
                target_vx_mps=target,
                vx_MAE_mps=float(error.mean()),
                overshoot_mps=float(max(0, (np.sign(target) or 1) * v.max() - abs(target)))
                if target >= 0
                else float(max(0, -v.min() - abs(target))),
                settle_1s_s=stable,
                tail_velocity_mean_mps=float(v[-1000:].mean()),
            )
        )
    result["command_phases"] = phases
    events = []
    full_dq = np.asarray(trace["dq"])[:, LEGS]
    # Offline bandpass for event tails, with the same explicit FFT definition.
    spectrum = np.fft.rfft(full_dq - full_dq.mean(0), axis=0)
    hz = np.fft.rfftfreq(len(t), 0.001)
    spectrum[(hz < 10) | (hz > 40)] = 0
    filtered = np.fft.irfft(spectrum, n=len(t), axis=0)

    def smooth(x):
        return np.convolve(x, np.ones(100) / 100, mode="same")

    dq_energy = smooth((full_dq**2).mean(1))
    hf_energy = smooth((filtered**2).mean(1))
    quiet = (
        (dq_energy <= gates["quiet_dq_rad_s"] ** 2)
        & (hf_energy <= gates["quiet_hf_rad_s"] ** 2)
        & (np.asarray(trace["normal_force_N"]) > gates["support_force_N"]).all(1)
        & (np.hypot(trace["vx"], trace["vy"]) <= gates["quiet_speed_mps"])
        & (np.abs(trace["roll"]) <= gates["quiet_roll_rad"])
        & (np.abs(trace["pitch"]) <= gates["quiet_pitch_rad"])
    )
    for event in case.get("events", []):
        off = event["at_s"] + (event.get("duration_s", 0) if event["kind"] == "modal" else 0)
        after = np.flatnonzero(t >= off)
        confirmed = None
        for i in after:
            if i + 1000 <= len(t) - 50 and quiet[i : i + 1000].all():
                confirmed = float(t[i] - off)
                break
        events.append(
            dict(
                event=event,
                force_off_s=off,
                quiet_1s_recovery_s=confirmed,
                recovery_status="PASS"
                if confirmed is not None
                else "FAIL"
                if complete
                else "UNKNOWN",
                off_to_end_s=float(max(0, t[-1] - off)),
            )
        )
    result["events"] = events
    if events and any(e["recovery_status"] != "PASS" for e in events):
        result["teacher_eligible"] = False
    return result
