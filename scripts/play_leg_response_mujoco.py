"""Read-only MuJoCo viewer for the delivered bilateral local leg-response model.

The model is the handoff in artifacts/2026_10_30__bilateral_leg_response_handoff:
two modal bodies per side (common/difference), PD40/2 recomputed every5ms, per-modal
post-PD transport, exact first-order motor response and a local constant load. It is a
LOCAL response target, not the robot plant: no gravity, wheel contact, closed chain or
IMU, so this viewer never drives the full robot and never writes files.

Default playback replays the delivered golden targets so the window reproduces the
frozen sim_q/sim_v; --source step|sine gives a quick look without logs.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / 'artifacts/2026_10_30__bilateral_leg_response_handoff'
CONTRACT = HANDOFF / 'contract.json'
GOLDEN = HANDOFF / 'golden_vectors.npz'


def side_block(contract_path, side):
    contract = json.loads(Path(contract_path).read_text(encoding='utf-8'))
    if side not in contract['sides']:
        raise ValueError(f'no delivered contract for side {side}')
    block = contract['sides'][side]
    params = np.array(block['parameters_J_B_K_tau_ms'], dtype=float)
    delays = np.array(block['modal_post_PD_transport_ms'], dtype=int)
    xml = (ROOT / block['run'] / 'fitted_local_model.xml').read_text(encoding='utf-8')
    return params, delays, xml, block


def golden_case(npz_path, case):
    with np.load(npz_path) as z:
        needed = [f'{case}__{name}' for name in
                  ('time', 'target_q', 'sim_q', 'sim_v', 'actual_q', 'actual_v', 'modal_state', 'loads_nm')]
        missing = [k for k in needed if k not in z.files]
        if missing:
            raise ValueError('golden vectors lack ' + ','.join(missing))
        return dict(target=z[f'{case}__target_q'].copy(), sim_q=z[f'{case}__sim_q'].copy(),
                    sim_v=z[f'{case}__sim_v'].copy(), actual_q=z[f'{case}__actual_q'].copy(),
                    actual_v=z[f'{case}__actual_v'].copy(), modal=z[f'{case}__modal_state'].copy(),
                    loads=z[f'{case}__loads_nm'].copy(), time=z[f'{case}__time'].copy())


def scripted(kind, samples, block):
    """Deterministic non-log target: a step around the delivered mid-range amplitude."""
    amplitude = float(np.max(np.abs(np.array(block['parameters_J_B_K_tau_ms'], dtype=float)[:, 2]))) / 10.
    gain = float(np.clip(amplitude, .05, .25))
    tick = np.arange(samples, dtype=float)
    if kind == 'step':
        return np.stack([gain * (tick >= 20), -gain * (tick >= 60)], axis=1)
    return np.stack([gain * np.sin(2 * np.pi * .5 * tick * .005),
                     -gain * np.sin(2 * np.pi * .3 * tick * .005)], axis=1)


def first_case(npz_path, side):
    """Delivered case ids are L_0/L_1/R_2/R_3, not a per-side counter."""
    with np.load(npz_path) as z:
        cases = sorted({k.split('__')[0] for k in z.files if k.endswith('__time') and k.startswith(f'{side}_')})
    if not cases:
        raise ValueError(f'golden vectors hold no case for side {side}')
    return cases[0]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def run_side(side, args):
    from native_stand.students.leg_response_rig import LocalLegResponseRig
    params, delays, xml, block = side_block(args.contract, side)
    rig = LocalLegResponseRig(params, delays, xml)
    if args.source == 'golden':
        if not Path(args.golden).exists():
            raise ValueError(f'golden vectors missing: {args.golden}')
        case = args.case or first_case(args.golden, side)
        data = golden_case(args.golden, case)
        sim_q, sim_v = data['sim_q'], data['sim_v']
        actual_q, actual_v = data['actual_q'], data['actual_v']
        # Golden targets are physical (hip,active); the rig's PD acts on the two modal
        # coordinates, so convert common=(hip+active)/2, differential=(active-hip)/2.
        physical = data['target']
        targets = np.stack([(physical[:, 0] + physical[:, 1]) * .5, (physical[:, 1] - physical[:, 0]) * .5], axis=1)
        initial = data['modal'][0]
        if initial.shape != (3, 2):
            raise ValueError('golden modal_state must be (3,2) of q,v,motor')
        rig.reset(initial[0], initial[1], initial[2], data['loads'])
        timeline = data['time']
    else:
        samples = 400
        targets = scripted(args.source, samples, block)
        sim_q = sim_v = actual_q = actual_v = None
        rig.reset(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
        timeline = np.arange(samples) * .005
    if args.max_seconds is not None:
        limit = int(np.searchsorted(timeline, timeline[0] + args.max_seconds, side='right'))
        targets = targets[:limit]
        if sim_q is not None:
            sim_q, sim_v = sim_q[:limit], sim_v[:limit]
            actual_q, actual_v = actual_q[:limit], actual_v[:limit]

    viewer = None
    if not args.headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(rig.model, rig.data)
        with viewer.lock():
            viewer.cam.lookat[:] = [0., 0., 0.]
            viewer.cam.distance = 1.2
            viewer.cam.elevation = -30.
    produced = []
    started = time.monotonic()
    try:
        for index, target in enumerate(targets):
            if viewer is not None and not viewer.is_running():
                break
            produced.append(rig.step(np.asarray(target, dtype=float)))
            if viewer is not None:
                viewer.sync()
                delay = started + 0.005 * (index + 1) / args.speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    finally:
        if viewer is not None:
            viewer.close()
    produced = np.asarray(produced)
    summary = dict(side=side, case=None if args.source != 'golden' else (args.case or f'{side}_0'),
                   stepped=len(produced), params_J_B_K_tau_ms=params.tolist(),
                   modal_post_PD_transport_ms=delays.tolist(),
                   max_motor_Nm=round(rig.max_motor_nm, 6), route_error_Nm=rig.max_route_error_nm,
                   physical_steps=rig.physical_steps,
                   scope='local two-modal response target; no gravity, wheel contact, closed chain or IMU')
    if sim_q is not None and len(produced) > 1:
        steps = len(produced) - 1
        # The rig integrates the two modal coordinates; the delivered golden vectors are in
        # physical order, hip=common-differential and active=common+differential.
        q_modal, v_modal = produced[:steps, 0], produced[:steps, 1]
        q_physical = np.stack([q_modal[:, 0] - q_modal[:, 1], q_modal[:, 0] + q_modal[:, 1]], axis=1)
        v_physical = np.stack([v_modal[:, 0] - v_modal[:, 1], v_modal[:, 0] + v_modal[:, 1]], axis=1)
        summary['identity_vs_golden_sim_max_abs_q'] = float(np.max(np.abs(q_physical - sim_q[1:steps + 1])))
        summary['identity_vs_golden_sim_max_abs_v'] = float(np.max(np.abs(v_physical - sim_v[1:steps + 1])))
        summary['reference_q_rmse_rad'] = [rmse(q_physical[:, j], actual_q[1:steps + 1][:, j]) for j in range(2)]
        summary['reference_v_rmse_rad_s'] = [rmse(v_physical[:, j], actual_v[1:steps + 1][:, j]) for j in range(2)]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, default=CONTRACT)
    parser.add_argument('--golden', type=Path, default=GOLDEN)
    parser.add_argument('--side', choices=('L', 'R', 'both'), default='both')
    parser.add_argument('--source', choices=('golden', 'step', 'sine'), default='golden')
    parser.add_argument('--case', default=None, help='golden case id, default <side>_0')
    parser.add_argument('--speed', type=float, default=1., help='wall-clock multiplier')
    parser.add_argument('--max-seconds', type=float, default=None)
    parser.add_argument('--headless', action='store_true', help='no window; still runs the delivered model')
    args = parser.parse_args()
    if not 0 < args.speed <= 8:
        parser.error('speed must be in (0, 8]')
    sides = ('L', 'R') if args.side == 'both' else (args.side,)
    for side in sides:
        print(json.dumps(run_side(side, args), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
