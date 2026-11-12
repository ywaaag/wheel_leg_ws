"""把 Ozone 日志里 rl_active=1 的那一帧当作 Isaac 初始状态，闭环跑部署 ONNX。

用途:判断"该初始状态本身"是否会引发实机那样的 12 Hz 抖振。
  - 若是:问题在策略/状态;
  - 若否:问题在执行/被控对象(弹簧、限幅、时延、接触)。

严格按仓库既有契约执行,不重新定义任何语义:
  - 关节/轮顺序 = native_stand.contract.JOINT_NAMES(= rl_motion31 的模型序)
  - 观测 31D:见 rl_motion31.c:118-142,本脚本 obs31() 逐位对应
  - 动作:腿 target = 1.5*a,轮 target = 100*a(rl_motion31.c:175-187)
  - PD:腿 Kp40/Kd2、轮 Kv0.2,限幅 40/5.5,补偿项与总限幅走 motion_mcu_execution.pd_and_motor
  - 运输/一阶响应:motion_mcu_execution.MotorRing(腿10ms/tau5ms、轮7,11ms/tau0)
  - 时钟:物理 1 ms、PD 5 ms、策略 10 ms

日志里没有的量按显式假设给出并写进输出(不是实测):
  - root 线速度:默认 0(--root-vx 可改)
  - yaw:不可观测,取 0
  - root 高度:由闭链腿长 + 轮半径把轮子放到刚好触地
  - 初始电机力矩:取当前弹簧补偿(profile 的 initial_motor 约定)

用法:
  python scripts/diagnose_motion_isaac_handover_state.py --headless \
    --csv logs/Ozone_DataSampling_261028.csv \
    --onnx deployment_protocol/2026_10_26-20_29__motion4500_precise_small31_onnx/actor.onnx \
    --event-time 13.74 --seconds 8 \
    --out artifacts/2026_10_28__isaac_handover_state/handover.json

两个弹簧口径(见 firmware_control.py 的 spring_mode):
  --spring-mode motor_only        训练/评估现状:PD + 补偿,被控对象无弹簧(源码标注 known mismatch)
  --spring-mode ideal_cancellation 实机假设:补偿被真实弹簧抵消,净 = PD
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

R_W = 0.06                      # 轮半径 leg_const.hpp R_W
LEGS = (0, 1, 3, 4)
WHEELS = (2, 5)
HIP_ZERO = math.radians(90.0 - 47.1033888)
ACT_ZERO = math.radians(90.0 + 47.1080837)
L1, L2, L3, L4 = 0.210, 0.250, 0.250, 0.210


# ---------------------------------------------------------------- 日志解析
def read_frame(path, event_time, column="rl_active"):
    """返回 rl_active 0->1 之后第一帧的 31D 输入与时间。"""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    names = [h.replace("(g_rl_shadow_debug).", "") for h in head]
    act_i = names.index(column)
    obs_i = [names.index(f"observation[{k}]") for k in range(25)]
    if t_i := names.index("Time"):
        pass
    data = np.array([[float(v) for v in row] for row in body if len(row) == len(head)])
    t = data[:, t_i]
    active = data[:, act_i]
    rising = np.where(np.diff(active) > 0)[0]
    if len(rising) == 0:
        raise ValueError("日志里没有 rl_active 0->1")
    # 取 event_time 之后(含)第一个上升沿,便于显式指定
    candidate = [i + 1 for i in rising if t[i + 1] >= event_time - 1e-9]
    if not candidate:
        raise ValueError(f"event_time={event_time} 之后没有上升沿")
    i = candidate[0]
    obs25 = data[i, obs_i].copy()
    obs31 = np.zeros(31)
    obs31[:25] = obs25
    obs31[25] = 1.0
    obs31[28] = 1.0
    return float(t[i]), obs31, i


# ---------------------------------------------------------------- 运动学
def leg_length(q_rh, q_ra, q_lh, q_la):
    """固件 getPhi 的同一分支;返回左右腿 L0(m)。"""
    def fk(p1, p4):
        xb, yb = L1 * math.cos(p1), L1 * math.sin(p1)
        xd, yd = L4 * math.cos(p4), L4 * math.sin(p4)
        a0, b0 = 2 * L2 * (xd - xb), 2 * L2 * (yd - yb)
        lbd = math.hypot(xd - xb, yd - yb)
        c0 = L2 * L2 + lbd * lbd - L3 * L3
        disc = a0 * a0 + b0 * b0 - c0 * c0
        if disc < 0:
            return float("nan"), float("nan")
        p2 = 2 * math.atan2(b0 + math.sqrt(disc), a0 + c0)
        xc = L1 * math.cos(p1) + L2 * math.cos(p2)
        yc = L1 * math.sin(p1) + L2 * math.sin(p2)
        return math.hypot(xc, yc), math.atan2(yc, xc)
    left = fk(ACT_ZERO + q_la, HIP_ZERO + q_lh)
    right = fk(ACT_ZERO + q_ra, HIP_ZERO - q_rh)
    return left, right


def quat_down_to(q):
    """构造 q 使 quat_apply_inverse(q,(0,0,-1)) == 给定 body 重力;yaw 取最小旋转(未观测)。"""
    g = np.asarray(q, float)
    g = g / np.linalg.norm(g)
    down = np.array([0.0, 0.0, -1.0])
    dot = float(np.clip(down @ g, -1.0, 1.0))
    if dot > 1 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1 + 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(down, g)
    axis /= np.linalg.norm(axis)
    ang = math.acos(dot)
    return np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * axis])


# ---------------------------------------------------------------- 主流程
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--event-time", type=float, default=13.74, help="rl_active 上升沿时间(s)")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--spring-mode", choices=("motor_only", "ideal_cancellation"), default="motor_only")
    parser.add_argument("--root-vx", type=float, default=0.0, help="日志无 root 线速度,显式假设")
    parser.add_argument("--settle-ms", type=int, default=0, help="施加电机前先空跑物理(默认0,不改初态)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace-csv", type=Path)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import quat_apply_inverse
    from agent_world.assets.rmuc_wheel_leg import RMUC_WHEEL_LEG_CFG
    from native_stand.contract import JOINT_NAMES
    from native_stand.students.motion_mcu_execution import WHEEL_CAP, MotorRing, pd_and_motor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    t_event, obs, row = read_frame(args.csv, args.event_time)

    # ---- 日志帧 -> 初始关节/姿态
    q_leg = obs[6:10] * 1.5                      # 模型序 RH,RA,LH,LA
    dq6 = obs[10:16] / 0.05                      # 模型序 6 通道
    (L0l, thl), (L0r, thr) = leg_length(*q_leg)
    if not (np.isfinite(L0l) and np.isfinite(L0r)):
        raise ValueError("日志关节角无法闭链,拒绝伪造初态")
    quat_wxyz = quat_down_to(obs[3:6])
    ang_b = obs[0:3] / 0.25                      # body 角速度(模型系)

    # ---- 策略
    import onnx
    from onnx.reference import ReferenceEvaluator
    graph = onnx.load(str(args.onnx))
    runtime = ReferenceEvaluator(graph)
    onnx_in = graph.graph.input[0].name

    # ---- 场景
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.001, device=device))
    sim.set_camera_view([2.0, 2.0, 1.2], [0.0, 0.0, 0.3])
    import isaaclab.sim as _su
    from pxr import UsdGeom, UsdPhysics
    stage = sim.stage
    ground = UsdGeom.Plane.Define(stage, "/World/ground")
    ground.CreateAxisAttr("Z").CreateWidthAttr(128.0).CreateLengthAttr(128.0)
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    mat = _su.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5, restitution=0.0)
    mat.func("/World/ground_material", mat)
    _su.bind_physics_material("/World/ground", "/World/ground_material", stage=stage)
    robot = Articulation(RMUC_WHEEL_LEG_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    ids = [int(robot.find_joints(n)[0][0]) for n in JOINT_NAMES]
    if len(set(ids)) != 6:
        raise RuntimeError("JOINT_NAMES 绑定不是一一对应")
    wheel_body = [robot.body_names.index(f"{s}_wheel") for s in ("left", "right")]

    def write_state(root_z, q_leg_rad, dq6_val):
        pose = torch.tensor([[0.0, 0.0, root_z, *quat_wxyz]], device=device, dtype=torch.float32)
        vel = torch.tensor([[args.root_vx, 0.0, 0.0, *ang_b]], device=device, dtype=torch.float32)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(vel)
        pos = torch.zeros(1, len(ids), device=device)
        velj = torch.zeros(1, len(ids), device=device)
        for k, ch in enumerate(LEGS):
            pos[0, k] = float(q_leg_rad[k])
        for k, ch in enumerate(range(6)):
            velj[0, k] = float(dq6_val[ch])
        robot.write_joint_state_to_sim(pos, velj, joint_ids=ids)
        robot.update(0.0)

    # 轮子刚好触地:先放到估计高度,读实际轮心,再整体平移一次
    z0 = R_W + 0.5 * (L0l + L0r)
    write_state(z0, q_leg, dq6)
    low = float(robot.data.body_pos_w.torch[0, wheel_body, 2].min())
    z0 += R_W - low
    write_state(z0, q_leg, dq6)

    # ---- 时钟与执行器
    ring = MotorRing(1, device)
    _, _, comp0, motor0, ok = pd_and_motor(
        robot.data.joint_pos.torch[:, ids], robot.data.joint_vel.torch[:, ids], torch.zeros(1, 6, device=device))
    if not bool(ok[0]):
        raise RuntimeError("初始关节角不构成有效闭链")
    ring.reset(torch.tensor([0]), motor0.clone())

    prev_exec = torch.zeros(1, 6, device=device)
    prev_exec[0, list(LEGS)] = torch.tensor(q_leg, device=device, dtype=torch.float32)
    prev_exec[0, list(WHEELS)] = torch.tensor([dq6[2], dq6[5]], device=device, dtype=torch.float32)

    def obs31(q, dq, prev):
        o = np.zeros(31)
        o[0:3] = q.new_tensor(0.0).item() if False else 0.0
        g = quat_apply_inverse(robot.data.root_quat_w.torch, torch.tensor([[0.0, 0.0, -1.0]], device=device))
        w = quat_apply_inverse(robot.data.root_quat_w.torch, robot.data.root_ang_vel_w.torch)
        o[0:3] = (w[0].cpu().numpy() * 0.25)
        o[3:6] = g[0].cpu().numpy()
        o[6:10] = (q[0, list(LEGS)].cpu().numpy() / 1.5)
        o[10:16] = (dq[0].cpu().numpy() * 0.05)
        o[16:20] = (prev[0, list(LEGS)].cpu().numpy() / 1.5)
        o[20:22] = (prev[0, list(WHEELS)].cpu().numpy() / 100.0)
        o[22] = obs[22]
        o[23] = obs[23]
        o[24] = obs[24]
        o[25] = 1.0
        o[28] = 1.0
        return o

    # ---- 主循环
    steps = int(round(args.seconds / 0.001))
    trace = {"t": [], "q": [], "dq": [], "action": [], "torque": [], "motor": [], "gravity": []}
    action = np.zeros(6)
    sat_leg = sat_wheel = 0
    n_torque = 0
    for k in range(steps):
        if args.settle_ms and k * 1 < args.settle_ms:
            torque = ring.step(torch.tensor([k]), motor0).clone() * 0.0
            torque = motor0 if False else torque
        else:
            torque = ring.step(torch.tensor([k]), cmd)
        if k % 5 == 0 and (not args.settle_ms or k >= args.settle_ms):
            q = robot.data.joint_pos.torch
            dq = robot.data.joint_vel.torch
            _, limited, comp, motor, ok = pd_and_motor(q[:, ids], dq[:, ids], prev_exec)
            if not bool(ok[0]):
                raise RuntimeError(f"t={k}ms 闭链无效")
            if args.spring_mode == "ideal_cancellation":
                motor = (motor - comp).clamp(-torch.tensor([40., 40., WHEEL_CAP, 40., 40., WHEEL_CAP], device=device),
                                             torch.tensor([40., 40., WHEEL_CAP, 40., 40., WHEEL_CAP], device=device))
            cmd = motor.clone()
        if k % 10 == 0 and (not args.settle_ms or k >= args.settle_ms):
            q = robot.data.joint_pos.torch
            dq = robot.data.joint_vel.torch
            o = obs31(q, dq, prev_exec)[None].astype(np.float32)
            action = runtime.run(None, {onnx_in: o})[0][0].astype(np.float64)
            clipped = np.clip(action, -1.0, 1.0)
            tgt = torch.zeros(1, 6, device=device)
            tgt[0, list(LEGS)] = torch.tensor(clipped[list(LEGS)] * 1.5, device=device, dtype=torch.float32)
            tgt[0, list(WHEELS)] = torch.tensor(clipped[list(WHEELS)] * 100.0, device=device, dtype=torch.float32)
            prev_exec = tgt.clone()
            if k >= args.settle_ms:
                tq = torque[0].cpu().numpy()
                trace["t"].append(k / 1000.0)
                trace["q"].append(q[0, list(LEGS)].cpu().numpy().copy())
                trace["dq"].append(dq[0].cpu().numpy().copy())
                trace["action"].append(action.copy())
                trace["torque"].append(tq[list(LEGS)].copy())
                trace["motor"].append(tq[list(WHEELS)].copy())
                trace["gravity"].append(quat_apply_inverse(
                    robot.data.root_quat_w.torch, torch.tensor([[0.0, 0.0, -1.0]], device=device))[0].cpu().numpy().copy())
                n_torque += 1
                sat_leg += int(np.abs(tq[list(LEGS)]).max() >= 39.5)
                sat_wheel += int(np.abs(tq[list(WHEELS)]).max() >= 0.999 * WHEEL_CAP)
        robot.set_joint_effort_target(torque, joint_ids=ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(0.001)

    # ---- 汇总
    g = np.array(trace["gravity"])
    q = np.array(trace["q"])
    dq = np.array(trace["dq"])
    a = np.array(trace["action"])
    root_z = float(robot.data.root_pos_w.torch[0, 2])
    pitch = np.degrees(np.arcsin(np.clip(-g[:, 2], -1, 1)))
    roll = np.degrees(np.arcsin(np.clip(-g[:, 0], -1, 1)))
    spec_sum = {}
    if n_torque > 4:
        for name, sig in (("pitch_deg", pitch), ("roll_deg", roll), ("dq_leg", dq[:, 0])):
            x = sig - sig.mean()
            f = np.fft.rfftfreq(len(x), 0.01)
            p = np.abs(np.fft.rfft(x * np.hanning(len(x))))
            p[0] = 0
            spec_sum[name] = float(f[int(p.argmax())])
    result = dict(
        version="motion31_isaac_handover_state_diagnostic_v1",
        not_a_capability_certificate=True,
        csv=str(args.csv), onnx=str(args.onnx), event_time_s=t_event, csv_row=row,
        spring_mode=args.spring_mode, seconds=args.seconds,
        initial=dict(q_leg_rad=q_leg.tolist(), dq6_rad_s=dq6.tolist(),
                     gravity_model=obs[3:6].tolist(), ang_body_rad_s=ang_b.tolist(),
                     leg_length_m=[L0l, L0r], root_z_written=z0,
                     assumed=dict(root_linear_velocity_m_s=[args.root_vx, 0.0, 0.0], yaw_rad=0.0,
                                  source="日志缺线速度与 yaw;高度由闭链腿长+轮半径触地确定")),
        final=dict(root_z_m=root_z, fell=bool(root_z < 0.20)),
        metrics=dict(pitch_deg_max=float(np.abs(pitch).max()) if len(pitch) else None,
                     roll_deg_max=float(np.abs(roll).max()) if len(roll) else None,
                     pitch_deg_std=float(pitch.std()) if len(pitch) else None,
                     roll_deg_std=float(roll.std()) if len(roll) else None,
                     action_abs_max=float(np.abs(a).max()) if len(a) else None,
                     action_std=a.std(0).tolist() if len(a) else None,
                     action_sign_flip_rate=float(np.mean(np.diff(np.sign(a), axis=0) != 0)) if len(a) > 1 else None,
                     leg_dq_std=dq.std(0).tolist() if len(dq) else None,
                     leg_torque_sat_frac=sat_leg / max(n_torque, 1),
                     wheel_torque_sat_frac=sat_wheel / max(n_torque, 1),
                     dominant_hz=spec_sum),
        real_log_reference=dict(
            note="同一帧在实机上的表现,用于对照,不是本仿真结果",
            wheel_torque_sat_frac=0.589, leg_dq_std=0.229, action_std_leg=0.29,
            pitch_std_deg=1.8, dominant_hz=12.0),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if args.trace_csv:
        args.trace_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.trace_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t"] + [f"q{k}" for k in range(4)] + [f"dq{k}" for k in range(6)]
                       + [f"a{k}" for k in range(6)] + [f"tau_leg{k}" for k in range(4)]
                       + [f"tau_wheel{k}" for k in range(2)] + ["gx", "gy", "gz"])
            for i in range(len(trace["t"])):
                w.writerow([trace["t"][i], *trace["q"][i], *trace["dq"][i], *trace["action"][i],
                            *trace["torque"][i], *trace["motor"][i], *trace["gravity"][i]])
    print(json.dumps({k: result[k] for k in ("spring_mode", "final", "metrics")}, indent=2, ensure_ascii=False))
    sim.clear_instance()
    simulation_app.close()


if __name__ == "__main__":
    main()
