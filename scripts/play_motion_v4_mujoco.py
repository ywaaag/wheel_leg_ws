"""Read-only MuJoCo viewer for the full-v4 identified leg plant.

Reuses the frozen V4 comparison math (isolated/2026_10_31__fullchain_v4_mujoco_comparison):
body mass/inertia scale, PD40/2 -> per-side modal transport and first-order response,
separate NET load on qfrc_applied, motor-only actuator clamp and the 5 ms held observer.
No file is written, no training code or asset is modified; V4 is an equivalent
fixture-derived candidate, not measured whole-robot truth.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / 'isolated/2026_10_31__fullchain_v4_mujoco_comparison/compare.py'
CONTRACT = ROOT / 'artifacts/2026_10_30__bilateral_detailed_response_v4/candidate_contract.json'


def v4_module():
    spec = importlib.util.spec_from_file_location('v4_reference', V4)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='31D MotionActor checkpoint')
    parser.add_argument('--contract', type=Path, default=CONTRACT)
    parser.add_argument('--command', nargs=2, type=float, default=[0., 0.], metavar=('VX', 'YAW'))
    parser.add_argument('--speed', type=float, default=1., help='wall-clock multiplier')
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--no-viewer', action='store_true', help='run the same loop without a window')
    args = parser.parse_args()
    if not 0 < args.speed <= 8:
        parser.error('speed must be in (0, 8]')
    if not 0 < args.seconds <= 60:
        parser.error('seconds must be in (0, 60]')

    v4 = v4_module()
    contract = json.loads(args.contract.read_text(encoding='utf-8'))
    if v4.sha(args.contract) != contract.get('contract_sha256', v4.sha(args.contract)):
        raise ValueError('V4 contract SHA differs from its own record')
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    model, refs, readback = v4.build_model(spec, contract)
    actor, checkpoint, dims = v4.load_actor(args.checkpoint)
    target = np.asarray(args.command, dtype=float)

    data = mujoco.MjData(model)
    from motion_mcu_playback import apply_initial_pose
    from validate_native_mujoco import binding  # noqa: F401  (same refs object as build_model)
    apply_initial_pose(model, data, refs, spec['mcu_execution'])
    mujoco.mj_forward(model, data)

    frame, q, velocity, _ = v4.measured(model, data, refs)
    adopt = np.zeros(6); adopt[v4.LEGS] = q[v4.LEGS]
    request = adopt.copy()
    leg = v4.LegResponse(contract, q[v4.PAIRS] * v4.SIGN, velocity[v4.PAIRS] * v4.SIGN,
                         adopt[v4.PAIRS] * v4.SIGN)
    targets = np.tile(adopt, (32, 1))
    wheel_queue = np.zeros((128, 2)); wheel_state = np.zeros(2); wheel_raw = np.zeros(2)
    packets = np.tile(frame, (4, 1))

    def unpair(values):
        out = np.zeros(6); out[v4.PAIRS.flatten()] = (values * v4.SIGN).flatten(); return out

    viewer = None
    if not args.no_viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        with viewer.lock():
            viewer.cam.lookat[:] = [0., 0., .35]
            viewer.cam.distance = 2.2
            viewer.cam.elevation = -20.
    ticks = int(round(args.seconds * 1000))
    started = time.monotonic()
    failure = None
    step = 0
    try:
        for tick in range(ticks):
            if viewer is not None and not viewer.is_running():
                failure = 'viewer_closed'
                break
            if tick % 10 == 0:
                frame, _, _, _ = v4.measured(model, data, refs)
                packets[(tick // 10) % 4] = frame
                sensed = packets[((tick - min(20, tick)) // 10) % 4].copy()
                held_q = unpair(leg.heldq); held_v = unpair(leg.heldv)
                sensed[6:10] = held_q[v4.LEGS] / 1.5
                sensed[[10, 11, 13, 14]] = held_v[v4.LEGS] * .05
                command = v4.command_at(tick * .001, target)
                obs = np.r_[sensed, adopt[v4.LEGS] / 1.5, adopt[v4.WHEELS] / 100,
                            command / [4, 4, .5], [1, 0, 0, 1, 0, 0]].astype(np.float32)
                import torch
                with torch.inference_mode():
                    raw = actor(torch.from_numpy(obs)[None]).tanh()[0].numpy()
                if not np.isfinite(raw).all():
                    failure = 'nonfinite_action'; break
                request = np.clip(raw, -1, 1) * [1.5, 1.5, 100, 1.5, 1.5, 100]
                request[v4.LEGS] = np.clip(request[v4.LEGS],
                                           model.jnt_range[np.array(refs['joints'])[v4.LEGS], 0],
                                           model.jnt_range[np.array(refs['joints'])[v4.LEGS], 1])
            q = data.qpos[refs['qpos']].copy(); velocity = data.qvel[refs['dofs']].copy()
            targets[tick % 32] = request
            if tick % 5 == 0:
                adopt = targets[(tick - 5) % 32].copy()
                wheel_raw = np.clip(.2 * (adopt[v4.WHEELS] - velocity[v4.WHEELS]),
                                    -2.44694786377709, 2.44694786377709)
            motor_modal, net_modal = leg.step(q[v4.PAIRS] * v4.SIGN, velocity[v4.PAIRS] * v4.SIGN,
                                              adopt[v4.PAIRS] * v4.SIGN)
            wheel_queue[tick % 128] = wheel_raw
            wheel_state += (-np.expm1(-.5)) * (wheel_queue[(tick - 20) % 128] - wheel_state)
            motor = unpair(motor_modal); motor[v4.WHEELS] = wheel_state
            net = unpair(net_modal)
            data.ctrl[refs['actuators']] = motor / refs['gears']
            data.qfrc_applied[:] = 0.
            data.qfrc_applied[refs['dofs']] = net
            mujoco.mj_step(model, data)
            step = tick + 1
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                failure = 'nonfinite_state'; break
            if viewer is not None:
                if tick % 5 == 0:
                    viewer.sync()
                delay = started + .001 * step / args.speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            elif tick % 1000 == 999:
                print(json.dumps({'tick': step, 'orientation_cos': None}), flush=True)
    finally:
        if viewer is not None:
            viewer.close()
    print(json.dumps(dict(seconds=round(step / 1000., 3), termination=failure or 'WINDOW_COMPLETE',
                          command=target.tolist(), contract=str(args.contract), contract_sha256=v4.sha(args.contract),
                          checkpoint=str(args.checkpoint), actor_dims=dims,
                          body_scale=[contract['sides'][s]['parameters']['leg_mass_inertia_scale'] for s in ('L', 'R')],
                          transport_ms=[contract['sides'][s]['effective_observer']['transport_ms'] for s in ('L', 'R')],
                          motor_force_limit_Nm=readback['motor_force_limit_joint_Nm'],
                          separate_NET='qfrc_applied, motor only through the actuator',
                          read_only=True, files_written=0,
                          scope='full-v4 equivalent candidate, not measured hardware; single env, no capability certificate'),
                    ensure_ascii=False), flush=True)
    if viewer is not None:
        # The GL/EGL context teardown can abort after the window is closed; the run is over
        # and the summary is already flushed, so leave without running interpreter shutdown.
        sys.stdout.flush()
        import os
        os._exit(0)
    return 0


if __name__ == '__main__':
    sys.exit(main())
