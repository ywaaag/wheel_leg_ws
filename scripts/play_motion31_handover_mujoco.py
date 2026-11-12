"""把 Ozone 日志里 rl_active 上升沿那一帧当作 MuJoCo 初始状态,复刻实机闭环。

**气弹簧完全不建模,也不加补偿项**:实机上弹簧力矩与该前馈补偿互相抵消,
净弹簧作用为零,因此腿的净力矩就等于 PD 输出。仿真侧唯一自洽的写法是
两样都不加(直接把补偿项从力矩里去掉),而不是"加补偿但不建弹簧"。

目的:判断"该初始状态本身"会不会引发实机那样的 12 Hz 抖振。
  - 会  -> 问题在策略/状态,应换策略或改训练;
  - 不会-> 问题在执行/被控对象(限幅、时延、接触、dq 噪声)。

只复用仓库既有实现,不重新定义语义:
  初态      : native_stand.students.real2sim_handover.initialize_from_log
              (q/dq/gyro/gravity 严格拟合;被动闭链最小二乘;高度取较低圆柱轮刚好触地)
  观测 31D  : 与 rl_motion31.c:118-142 逐位一致
  动作      : 腿 target = 1.5*a,轮 target = 100*a (rl_motion31.c:175-187)
  力矩链路  : PD(Kp40/Kd2/Kv0.2) -> PD 限幅[40,40,5.5,40,40,5.5]
              -> 补偿项与弹簧净零(不加) -> 驱动限幅[40,40,2.4469,40,40,2.4469]
  运输+响应 : motion_mcu_execution.MotorRing (腿 10ms/tau5ms,轮 7,11ms/tau0)
  时钟      : 物理 1ms / PD 5ms / 策略 10ms

日志没有的量按显式假设给出并写进输出(不是实测):root 线速度(--root-vx,默认0)、yaw、真实接触、
LQR 力矩、执行目标。

用法(GUI 看效果):
  PYTHONPATH=src:scripts /home/kellen/anaconda3/envs/issaclab/bin/python \
    scripts/play_motion31_handover_mujoco.py \
    --csv logs/Ozone_DataSampling_261028.csv \
    --onnx deployment_protocol/2026_10_26-20_29__motion4500_precise_small31_onnx/actor.onnx \
    --event-time 13.74 --seconds 8 --viewer

无窗口取指标:
  ... --seconds 8 --out artifacts/NEW/handover.json --trace-csv artifacts/NEW/trace.csv
"""
import argparse
import csv
import os
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

LEG = (0, 1, 3, 4)
WHEEL = (2, 5)
DEFAULT_XML = Path("/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml")


def read_log_frame(path, event_time):
    """复用仓库解析器;返回该帧 31D 输入与其时间。"""
    from replay_motion31_shadow_csv import read_csv
    rows, x, y, t, columns = read_csv(path)
    index = int(np.searchsorted(t, event_time))
    if index >= len(t):
        raise ValueError(f"event_time={event_time} 超出日志 {t[-1]:.3f}s")
    return float(t[index]), x[index].astype(np.float64), index


def summarize(t, q, dq, action, tau_leg, tau_wheel, gravity):
    g = np.asarray(gravity)
    pitch = np.degrees(np.arcsin(np.clip(-g[:, 2], -1, 1)))
    roll = np.degrees(np.arcsin(np.clip(-g[:, 0], -1, 1)))
    out = dict(
        pitch_deg_max=float(np.abs(pitch).max()),
        roll_deg_max=float(np.abs(roll).max()),
        pitch_deg_std=float(pitch.std()),
        roll_deg_std=float(roll.std()),
        action_abs_max=float(np.abs(action).max()),
        action_std=action.std(0).tolist(),
        action_sign_flip_rate=float(np.mean(np.diff(np.sign(action), axis=0) != 0)),
        leg_dq_std=dq.std(0).tolist(),
        leg_torque_abs_p50=float(np.percentile(np.abs(tau_leg), 50)),
        leg_torque_abs_p90=float(np.percentile(np.abs(tau_leg), 90)),
        leg_torque_sat_frac=float(np.mean(np.abs(tau_leg) > 39.5)),
        wheel_torque_sat_frac=float(np.mean(np.abs(tau_wheel) > 2.4)),
    )
    spec = {}
    for name, sig in (("pitch_deg", pitch), ("roll_deg", roll), ("dq0", dq[:, 0]), ("action0", action[:, 0])):
        z = sig - sig.mean()
        freq = np.fft.rfftfreq(len(z), 0.01)
        power = np.abs(np.fft.rfft(z * np.hanning(len(z))))
        power[0] = 0.0
        spec[name] = float(freq[int(power.argmax())])
    out["dominant_hz"] = spec
    out["tail_windows"] = {}
    for lo in (0.0, 2.0, 4.0):
        m = t >= lo
        if m.sum() < 8:
            continue
        out["tail_windows"][f"{lo:.0f}s+"] = dict(
            pitch_deg_std=float(pitch[m].std()), roll_deg_std=float(roll[m].std()),
            max_abs_gx=float(np.abs(g[m, 0]).max()), max_abs_gz=float(np.abs(g[m, 2]).max()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True, help="部署包里的 actor.onnx")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--event-time", type=float, default=13.74, help="rl_active 上升沿时间(s)")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--root-vx", type=float, default=0.0, help="日志无 root 线速度,显式假设 m/s")
    p.add_argument("--leg-armature", type=float, default=None, help="显式覆盖腿 armature(仓库训练档为 .012)")
    p.add_argument("--solver-profile", default="closure_contact_tight_v1",
                   help="仓库 Motion31 默认闭链/接触紧求解器;legacy_xml_v1 复现原始 XML")
    p.add_argument("--zero-motor-timing", action="store_true",
                   help="诊断: 电机环运输延迟与一阶上升时间全部置 0,用于校核相位差")
    p.add_argument("--replay-logged-obs", action="store_true",
                   help="把日志记录的 31D 观测直接喂给策略(不再用仿真自身状态),plant 仍自行积分")
    p.add_argument("--replay-logged-actions", action="store_true",
                   help="开环回放: 用日志记录的 mean_action 代替策略输出,用于分离 plant 差异与策略差异")
    p.add_argument("--previous-init", choices=("zero", "proxy"), default="zero",
                   help="首帧 previous_target:zero=与日志一致(实测首帧为0);proxy=源码的 q 代理")
    p.add_argument("--viewer", action="store_true", help="MuJoCo 被动窗口,按实时节拍播放")
    p.add_argument("--viewer-hold", action="store_true", help="播完后保持窗口直到关闭")
    p.add_argument("--out", type=Path, help="结果 JSON(不写则不落盘)")
    p.add_argument("--trace-csv", type=Path, help="100Hz 原始轨迹")
    args = p.parse_args()

    import mujoco
    import torch
    from onnx.reference import ReferenceEvaluator
    import onnx as onnx_pb

    from native_stand.students.real2sim_handover import initialize_from_log
    from native_stand.students.motion_mcu_execution import WHEEL_CAP, MotorRing, pd_and_motor
    from validate_native_mujoco import binding, obj

    t_event, obs_log, index = read_log_frame(args.csv, args.event_time)

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    if model.nu != 6:
        raise ValueError(f"MJCF 需要 6 个执行器,实际 {model.nu}")
    refs = binding(model)
    from native_stand.students.mujoco_solver import apply as apply_solver
    apply_solver(model, refs, args.solver_profile)
    if args.leg_armature is not None:
        dofs = np.asarray(refs["dofs"])[list(LEG)]
        model.dof_armature[dofs] = float(args.leg_armature)

    data, fit = initialize_from_log(model, refs, obs_log, args.root_vx)
    mujoco.mj_forward(model, data)

    logged_actions = None
    logged_obs = None
    if args.replay_logged_obs or args.replay_logged_actions:
        with open(args.csv, newline="") as fh:
            rr = list(csv.reader(fh))
        hh = [h.replace("(g_rl_shadow_debug).", "") for h in rr[0]]
        ai = [hh.index(f"mean_action[{k}]") for k in range(6)]
        dd = np.array([[float(x) for x in r] for r in rr[1:] if len(r) == len(rr[0])])
        act_col = dd[:, hh.index("rl_active")]
        s0 = int(np.where(np.diff(act_col) > 0)[0][0]) + 1
        if args.replay_logged_actions:
            logged_actions = dd[s0:, ai]
        else:
            oi = [hh.index(f"observation[{k}]") for k in range(25)]
            logged_obs = np.zeros((len(dd) - s0, 31))
            logged_obs[:, :25] = dd[s0:, oi]
            logged_obs[:, 25] = 1.0
            logged_obs[:, 28] = 1.0
    graph = onnx_pb.load(str(args.onnx))
    runtime = ReferenceEvaluator(graph)
    onnx_in = graph.graph.input[0].name

    root = refs["root"]
    qpos_active = np.asarray(refs["qpos"])
    dofs = np.asarray(refs["dofs"])
    actuators = np.asarray(refs["actuators"])
    gears = np.asarray(refs["gears"])
    cap = torch.tensor([40., 40., WHEEL_CAP, 40., 40., WHEEL_CAP], dtype=torch.float32)

    def observe(previous):
        rotation = data.xmat[root].reshape(3, 3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root, velocity, 0)
        angular = rotation.T @ velocity[:3]
        gravity = rotation.T @ np.array([0., 0., -1.])
        q = data.qpos[qpos_active].copy()
        dq = data.qvel[dofs].copy()
        o = np.zeros(31)
        o[0:3] = angular * 0.25
        o[3:6] = gravity
        o[6:10] = q[list(LEG)] / 1.5
        o[10:16] = dq * 0.05
        o[16:20] = previous[list(LEG)] / 1.5
        o[20:22] = previous[list(WHEEL)] / 100.0
        o[22:25] = obs_log[22:25]
        o[25] = 1.0
        o[28] = 1.0
        return o, gravity, q, dq

    # 报告坐标:初始化已按日志拟合;previous 按仓库约定取本地已采用目标的初始化代理
    previous = torch.zeros(1, 6, dtype=torch.float32)
    _, _, q0, dq0 = observe(np.zeros(6))
    if args.previous_init == "proxy":
        previous[0, list(LEG)] = torch.tensor(q0[list(LEG)], dtype=torch.float32)
        previous[0, list(WHEEL)] = torch.tensor(dq0[list(WHEEL)], dtype=torch.float32)

    ring = MotorRing(1, "cpu")
    if args.zero_motor_timing:
        ring.delay.zero_(); ring.tau.fill_(1e-9)
    _, limited0, _, _, ok = pd_and_motor(
        torch.tensor(q0, dtype=torch.float32)[None], torch.tensor(dq0, dtype=torch.float32)[None], previous)
    if not bool(ok[0]):
        raise ValueError("日志初态不构成有效闭链")
    # 接管那一帧实机已经在施加 PD+补偿(日志 rl_leg 非零),净力矩=PD。
    # 电机环在 k=0 用该帧 PD 值预填,否则前 10ms 会白丢一段保持力矩。
    command = limited0.clamp(-cap, cap).clone()

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)

    steps = int(round(args.seconds / 0.001))
    trace = {k: [] for k in ("t", "obs", "q", "dq", "action", "tau_leg", "tau_wheel", "gravity")}
    wall_start = time.monotonic()
    for k in range(steps):
        log_this = False
        # 先出策略目标:100Hz 时 obs 用"上一帧目标",然后目标更新供本帧 PD 采用。
        if k % 10 == 0:
            o, gravity, q, dq = observe(previous[0].numpy())
            if logged_obs is not None:
                if k // 10 >= len(logged_obs):
                    raise ValueError("日志观测不足以覆盖回放时长")
                o = logged_obs[k // 10].copy()
            if logged_actions is not None:
                if k // 10 >= len(logged_actions):
                    raise ValueError("日志动作不足以覆盖回放时长")
                action = logged_actions[k // 10].astype(np.float64)
            else:
                action = runtime.run(None, {onnx_in: o[None].astype(np.float32)})[0][0].astype(np.float64)
            clipped = np.clip(action, -1.0, 1.0)
            target = torch.zeros(1, 6, dtype=torch.float32)
            target[0, list(LEG)] = torch.tensor(clipped[list(LEG)] * 1.5, dtype=torch.float32)
            target[0, list(WHEEL)] = torch.tensor(clipped[list(WHEEL)] * 100.0, dtype=torch.float32)
            previous = target.clone()
            log_this = True
        if k % 5 == 0:
            q_t = torch.tensor(data.qpos[qpos_active].copy(), dtype=torch.float32)[None]
            dq_t = torch.tensor(data.qvel[dofs].copy(), dtype=torch.float32)[None]
            _, limited, _, _, ok = pd_and_motor(q_t, dq_t, previous)
            if not bool(ok[0]):
                raise RuntimeError(f"t={k}ms 闭链无效")
            # 弹簧与其补偿在实机上互相抵消:仿真两样都不加,只保留 PD 限幅后的驱动限幅。
            command = limited.clamp(-cap, cap).clone()
            if k == 0:
                ring.reset(torch.tensor([0]), command.clone())
        torque = ring.step(torch.tensor([k]), command)[0].numpy()
        if log_this:
            trace["t"].append(k / 1000.0)
            trace["obs"].append(o.copy())
            trace["q"].append(q[list(LEG)].copy())
            trace["dq"].append(dq.copy())
            trace["action"].append(action.copy())
            trace["tau_leg"].append(torque[list(LEG)].copy())
            trace["tau_wheel"].append(torque[list(WHEEL)].copy())
            trace["gravity"].append(gravity.copy())
        data.ctrl[actuators] = torque / gears
        mujoco.mj_step(model, data)
        if viewer is not None and not viewer.is_running():
            break
        if viewer is not None:
            delay = wall_start + (k + 1) * 0.001 - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            viewer.sync()

    metrics = summarize(np.array(trace["t"]), np.array(trace["q"]), np.array(trace["dq"]),
                        np.array(trace["action"]), np.array(trace["tau_leg"]),
                        np.array(trace["tau_wheel"]), np.array(trace["gravity"]))
    result = dict(
        version="motion31_mujoco_handover_state_diagnostic_v1",
        not_a_capability_certificate=True,
        csv=str(args.csv), csv_row=index, event_time_s=t_event,
        previous_init=args.previous_init,
        onnx=str(args.onnx), xml=str(args.xml), spring_model='net zero: gas spring and its feedforward compensation both omitted',
        seconds=args.seconds,
        initial_fit=fit,
        solver_profile=args.solver_profile,
        declared_absent=['plant profile override (motion_mcu_currentplant_v1) not applied',
                         'gas spring residual (declared net zero; not identified)',
                         'root linear velocity/yaw measurement', 'LQR torque', 'contact state'],
        metrics=metrics,
        real_log_reference=dict(note="同一帧在实机上的表现,仅供对照,不是本仿真结果",
                                wheel_torque_sat_frac=0.589, leg_dq_std=0.229, action_std_leg=0.29,
                                pitch_std_deg=1.8, dominant_hz=12.0),
        final_height_m=float(data.xpos[root, 2]),
        fell=bool(data.xpos[root, 2] < 0.20),
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if args.trace_csv:
        args.trace_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.trace_csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t"] + [f"obs{k}" for k in range(31)] + [f"q{k}" for k in range(4)]
                       + [f"dq{k}" for k in range(6)] + [f"a{k}" for k in range(6)]
                       + [f"tau_leg{k}" for k in range(4)] + [f"tau_wheel{k}" for k in range(2)]
                       + ["gx", "gy", "gz"])
            for i in range(len(trace["t"])):
                w.writerow([trace["t"][i], *trace["obs"][i], *trace["q"][i], *trace["dq"][i],
                            *trace["action"][i], *trace["tau_leg"][i], *trace["tau_wheel"][i],
                            *trace["gravity"][i]])
    print(json.dumps({k: result[k] for k in ("spring_model", "fell", "final_height_m", "metrics")},
                     indent=2, ensure_ascii=False))
    if viewer is not None:
        while viewer.is_running() and args.viewer_hold:
            time.sleep(0.02)
        try:
            viewer.close()
        except Exception:
            pass
    # 该机器的 GLX 在解释器退出阶段会崩(GL teardown),输出已 flush,直接退出。
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
