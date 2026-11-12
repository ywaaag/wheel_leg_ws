"""按 armature 扫描复刻同一初始状态,用于判断"重腿"是否是 sim 与实机分歧的来源。

用法(在项目根):
  PYTHONPATH=src:scripts /home/kellen/anaconda3/envs/issaclab/bin/python \
    scripts/sweep_motion31_armature.py \
    --csv logs/Ozone_DataSampling_261028.csv \
    --onnx deployment_protocol/2026_10_26-20_29__motion4500_precise_small31_onnx/actor.onnx \
    --armatures 0.35 0.11 0.04 0.02 0.016 0.012 \
    --seconds 6.25 --out-dir artifacts/NEW_armature_sweep

默认闭环(策略参与)。加 --replay-logged-actions 改为"同源命令开环回放"。
每个 armature 只改腿臂架惯量,其余(观测/动作/PD/限幅/时序/初始状态)逐项相同。
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--armatures", type=float, nargs="+", default=[0.35, 0.11, 0.04, 0.02, 0.016, 0.012])
    p.add_argument("--seconds", type=float, default=6.25)
    p.add_argument("--event-time", type=float, default=13.74)
    p.add_argument("--replay-logged-actions", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="src:scripts",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    rows = []
    for a in args.armatures:
        out = args.out_dir / f"arm_{a:g}.json"
        trace = args.out_dir / f"arm_{a:g}.trace.csv"
        cmd = [args.python, "scripts/play_motion31_handover_mujoco.py",
               "--csv", str(args.csv), "--onnx", str(args.onnx),
               "--event-time", str(args.event_time), "--seconds", str(args.seconds),
               "--leg-armature", str(a), "--previous-init", "zero",
               "--out", str(out), "--trace-csv", str(trace)]
        if args.replay_logged_actions:
            cmd.append("--replay-logged-actions")
        print("\n>>> armature = %g" % a, flush=True)
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
        d = json.loads(out.read_text())
        m = d["metrics"]
        rows.append(dict(
            armature=a,
            fell=d["fell"],
            height=d["final_height_m"],
            init_height=d["initial_fit"]["assumed_root_height_m"],
            pitch_std=m["pitch_deg_std"],
            roll_std=m["roll_deg_std"],
            dq_std=max(m["leg_dq_std"]),
            action_abs_max=m["action_abs_max"],
            action_std=max(m["action_std"]),
            flip=m["action_sign_flip_rate"],
            leg_tau_p90=m["leg_torque_abs_p90"],
            leg_sat=m["leg_torque_sat_frac"],
            wheel_sat=m["wheel_torque_sat_frac"],
            dom_hz=m["dominant_hz"]["dq0"],
            tail_pitch=m["tail_windows"].get("4s+", {}).get("pitch_deg_std"),
        ))

    print("\n" + "=" * 130)
    print("%-9s %-6s %-9s %-9s %-9s %-8s %-8s %-8s %-8s %-9s %-9s %-8s %-9s" % (
        "armature", "fell", "末态高度", "pitch_std", "roll_std", "dq_std",
        "|a|max", "a_std", "换号率", "腿tau_p90", "轮饱和", "主频Hz", "4s+pitch"))
    for r in rows:
        print("%-9g %-6s %-9.4f %-9.4f %-9.4f %-8.2f %-8.3f %-8.3f %-8.3f %-9.1f %-9.4f %-8.1f %s" % (
            r["armature"], r["fell"], r["height"], r["pitch_std"], r["roll_std"],
            r["dq_std"], r["action_abs_max"], r["action_std"], r["flip"],
            r["leg_tau_p90"], r["wheel_sat"], r["dom_hz"],
            "-" if r["tail_pitch"] is None else "%.4f" % r["tail_pitch"]))
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print("\n写入 %s" % (args.out_dir / "summary.json"))
    print("实机同帧对照: dq_std 4.6-5.8 rad/s, |a|max 0.75, a_std 0.25-0.35, 换号率 0.45-0.53, "
          "腿tau_p90 30-40, 轮饱和 0.589, 主频 23.4 Hz")


if __name__ == "__main__":
    main()
