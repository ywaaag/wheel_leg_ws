"""Minimal MuJoCo viewer for a bare 31D->6D ONNX actor: no manifest, no spec, no SHA gate.

Only the ONNX graph is trusted for the policy. The plant is the reference MJCF with the
explicitly named playback assumptions below (armature, friction, PD40/2/.2, motor caps and
the recorded .34 m standing pose). Nothing is written and no training code is touched.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

from validate_native_mujoco import binding          # noqa: E402
from motion_mcu_playback import apply_initial_pose  # noqa: E402
from diagnose_motion_real2sim import sensors        # noqa: E402
from native_stand.students.mcu_interface import encode  # noqa: E402
from native_stand.students.interface import targets      # noqa: E402

XML = Path('/home/kellen/wheel_leg_sim2sim/assets/wheeled_infantry.xml')
V4_CONTRACT = ROOT / 'artifacts/2026_10_30__bilateral_detailed_response_v4/candidate_contract.json'
# Geometric left is the model's right_* pair; both sides use capture_to_model_signs [-1,+1].
V4_PAIRS = np.array([[3, 4], [0, 1]])
V4_SIGN = np.array([-1., 1.])
START_POSE = ROOT / 'configs/playback/motion_standing_pose_0p34.json'
LEGS = [0, 1, 3, 4]
WHEELS = [2, 5]
LEG_KP, LEG_KD, WHEEL_KV = 40., 2., .2
JOINT_CAP_NM = np.array([40., 40., 2.44694786377709, 40., 40., 2.44694786377709])


def load_v4_contract(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def apply_v4_plant(model, refs, contract):
    """Same physical parameters as the full-v4 plant: named-body scale and motor-only clamp."""
    touched = []
    for side in ('L', 'R'):
        row = contract['sides'][side]
        scale = float(row['parameters']['leg_mass_inertia_scale'])
        for body in row['body_changes']:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body['name'])
            if index < 0:
                raise ValueError(f"v4 body missing: {body['name']}")
            model.body_mass[index] *= scale
            model.body_inertia[index] *= scale
            touched.append([body['name'], scale])
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return touched


def v4_net_force(q6, dq6, contract):
    """Fitted NET load only (no motor transport or response); wheels get nothing."""
    names = contract['parameter_names']
    net = np.zeros(6)
    for index, side in enumerate(('L', 'R')):
        parameters = contract['sides'][side]['parameters']
        reference = np.asarray(contract['sides'][side]['effective_observer']['q_reference'], dtype=float)
        pair = V4_PAIRS[index]; sign = V4_SIGN
        q = sign * np.asarray(q6, dtype=float)[pair]
        dq = sign * np.asarray(dq6, dtype=float)[pair]
        z = np.array([q.mean(), .5 * (q[1] - q[0])]) - np.array([reference.mean(), .5 * (reference[1] - reference[0])])
        dz = np.array([dq.mean(), .5 * (dq[1] - dq[0])])
        eta = np.array([
            parameters['net_load_common_Nm'] - parameters['net_K_common'] * z[0]
            - parameters['net_K_cross'] * z[1] - parameters['extra_modal_B_common'] * dz[0],
            parameters['net_load_difference_Nm'] - parameters['net_K_difference'] * z[1]
            - parameters['net_K_cross'] * z[0] - parameters['extra_modal_B_difference'] * dz[1]])
        eta -= np.array([parameters['quadratic_net_load_common'], parameters['quadratic_net_load_difference']]) * z * z
        net[pair] = sign * np.array([eta[0] - eta[1], eta[0] + eta[1]])
    return net


def build_slope_grid(xml_path, grades, half=2., spacing=4.):
    """One tilted static plane per DR grade in a row; the middle grade stays flat and named
    'ground' so the usual bindings and contact metrics keep working. The original 128 m
    ground is parked far below instead of being deleted."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path); root = tree.getroot()
    compiler = root.find('compiler')
    if compiler is not None and compiler.get('meshdir'):
        compiler.set('meshdir', str((Path(xml_path).parent / compiler.get('meshdir')).resolve()))
    world = root.find('worldbody')
    ground = next(g for g in world.findall('geom') if g.get('name') == 'ground')
    ground.set('name', 'ground_orig')
    position = (ground.get('pos') or '0 0 0').split()
    ground.set('pos', f'{position[0]} {position[1]} -10')
    middle = len(grades) // 2
    for index, grade in enumerate(grades):
        geom = ET.SubElement(world, 'geom')
        # Thin boxes, not planes: MuJoCo planes collide as infinite surfaces, so nine tilted
        # planes would all cross the origin region and throw the robot. Boxes stay in their
        # own 4x4 m patch; the top face of the middle (grade 0) box sits at z=0.
        geom.set('name', 'ground' if index == middle else f'slope_{index}')
        geom.set('type', 'box'); geom.set('size', f'{half} {half} .05')
        geom.set('pos', f'{(index - middle) * spacing:.4g} 0 -0.05')
        quat = np.zeros(4); mujoco.mju_axisAngle2Quat(quat, np.array([0., 1., 0.]), float(grade))
        geom.set('quat', ' '.join(f'{v:.9g}' for v in quat))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def flat_two_link_centers(height_m, formula):
    """Commanded height -> geometric leg angles; no spec, no support offset unless declared."""
    length = height_m + float(formula['hip_above_root_m']) - float(formula['wheel_radius_m'])
    upper = float(formula['equivalent_upper_m']); lower = float(formula['equivalent_lower_m'])
    cosine = (length ** 2 + upper ** 2 - lower ** 2) / (2 * length * upper)
    beta = math.acos(max(-1., min(1., cosine)))
    hip = beta - math.radians(float(formula['hip_zero_deg']))
    active = beta - math.radians(float(formula['active_zero_deg']))
    return np.array([hip, active, -hip, active])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('onnx', type=Path, help='31D->6D ONNX graph or a checkpoint containing actor/mlp.*')
    parser.add_argument('--xml', type=Path, default=XML)
    parser.add_argument('--pose', type=Path, default=START_POSE)
    parser.add_argument('--leg-armature', type=float, default=.012)
    parser.add_argument('--friction', type=float, default=.5)
    parser.add_argument('--slope-pitch', type=float, default=0.,
                        help='ground plane pitch about Y in rad (training DR range is +/-0.1)')
    parser.add_argument('--slope-roll', type=float, default=0., help='ground plane roll about X in rad')
    parser.add_argument('--slope-grid', action='store_true',
                        help='show the DR grade set: nine tilted planes in a row across +/-pitch '
                             '(robot starts on the middle, flat grade)')
    parser.add_argument('--command', nargs=2, type=float, default=[0., 0.], metavar=('VX', 'YAW'))
    parser.add_argument('--seconds', type=float, default=None,
                        help='stop after this many seconds; default runs until the window is closed '
                             '(headless needs an explicit value)')
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--no-viewer', action='store_true')
    parser.add_argument('--display', default=None, help='X display used when DISPLAY is unset (default :2)')
    parser.add_argument('--v4-plant', dest='v4_plant', action='store_true', default=True,
                        help='apply the full-v4 identified plant parameters (default on)')
    parser.add_argument('--no-v4-plant', dest='v4_plant', action='store_false')
    parser.add_argument('--v4-contract', type=Path, default=V4_CONTRACT)
    parser.add_argument('--output-contract', type=Path, default=None,
                        help='explicit action contract; default: checkpoint output_contract, '
                             'or the actor output_contract.json beside an ONNX')
    parser.add_argument('--panel', dest='panel', action='store_true', default=True, help='browser command panel (default on)')
    parser.add_argument('--no-panel', dest='panel', action='store_false')
    parser.add_argument('--panel-host', default='192.168.250.1')
    parser.add_argument('--panel-port', type=int, default=8766)
    parser.add_argument('--max-speed', type=float, default=.5)
    parser.add_argument('--max-yaw', type=float, default=.5)
    args = parser.parse_args()
    if not args.onnx.is_file():
        parser.error(f'ONNX file not found: {args.onnx}')

    pitch, roll = float(args.slope_pitch), float(args.slope_roll)
    if args.slope_grid:
        if pitch == 0.:
            pitch = .1
        grades = np.linspace(-pitch, pitch, 9)
        model = build_slope_grid(args.xml, grades)
        print(f'play: slope grid {len(grades)} planes, grades '
              f'{[round(float(g), 4) for g in grades]} rad, spacing 4 m', file=sys.stderr, flush=True)
    else:
        model = mujoco.MjModel.from_xml_path(str(args.xml))
    refs = binding(model)
    dofs = np.asarray(refs['dofs'])
    model.dof_armature[dofs[LEGS]] = float(args.leg_armature)
    model.dof_damping[dofs] = 0.
    model.geom_friction[:, 0] = float(args.friction)
    if (pitch or roll) and not args.slope_grid:
        q_pitch = np.zeros(4); mujoco.mju_axisAngle2Quat(q_pitch, np.array([0., 1., 0.]), pitch)
        q_roll = np.zeros(4); mujoco.mju_axisAngle2Quat(q_roll, np.array([1., 0., 0.]), roll)
        quat = np.zeros(4); mujoco.mju_mulQuat(quat, q_roll, q_pitch)
        model.geom_quat[int(refs['ground'])] = quat
    actuators = np.asarray(refs['actuators'])
    gears = np.abs(np.asarray(refs['gears']))
    cap = JOINT_CAP_NM / gears
    model.actuator_ctrlrange[actuators] = np.stack((-cap, cap), -1)
    model.actuator_forcerange[actuators] = model.actuator_ctrlrange[actuators]
    model.actuator_forcelimited[actuators] = True
    v4_contract = None
    v4_bodies = None
    if args.v4_plant:
        v4_contract = load_v4_contract(args.v4_contract)
        v4_bodies = apply_v4_plant(model, refs, v4_contract)
    mujoco.mj_setConst(model, mujoco.MjData(model))

    data = mujoco.MjData(model)
    pose = json.loads(Path(args.pose).read_text(encoding='utf-8'))
    apply_initial_pose(model, data, refs, {'pose': pose})
    mujoco.mj_forward(model, data)

    if args.onnx.suffix.lower() in ('.pt', '.pth'):
        # Plain checkpoint support: build the Actor MLP from its own declared dims. No spec,
        # no Critic, no SHA gate -- exactly the bare-playback contract.
        from torch import nn
        checkpoint = torch.load(args.onnx, map_location='cpu', weights_only=False)
        state = checkpoint['actor']
        dims = list((checkpoint.get('network_architecture') or {}).get('actor_hidden_dims') or [2048, 1024, 512])
        layers = []
        width = 31
        for hidden in dims:
            layers += [nn.Linear(width, hidden), nn.ELU(1.0)]
            width = hidden
        layers.append(nn.Linear(width, 6))
        network = nn.Sequential(*layers).eval()
        network.load_state_dict({k.removeprefix('mlp.'): v for k, v in state.items() if k.startswith('mlp.')})

        def policy(obs):
            with torch.no_grad():
                out = network(torch.from_numpy(np.asarray(obs, dtype=np.float32))).tanh()
            return out.numpy().reshape(-1)
    else:
        import onnx
        graph = onnx.load(str(args.onnx))
        graph_input = graph.graph.input[0].name
        graph_output = graph.graph.output[0].name
        engine = ReferenceEvaluator(graph)

        def policy(obs):
            out = engine.run([graph_output], {graph_input: np.asarray(obs, dtype=np.float32)})[0]
            return np.asarray(out, dtype=np.float32).reshape(-1)

    plan = controls = panel_server = None
    if args.panel:
        from native_stand.students.motion_panel import Plan, port_number, start as start_panel
        from native_stand.students.mujoco_keyboard import MotionPanelControls
        port = port_number(args.panel_port)
        plan = Plan(args.max_speed, args.max_yaw, backend='MuJoCo', mode='Motion',
                    models=['onnx'], model='onnx')
        panel_server, _ = start_panel(plan, port, args.panel_host)
        controls = MotionPanelControls(plan, keyboard_enabled=True)
        print(f'play: web panel http://{args.panel_host}:{port}/', file=sys.stderr, flush=True)

    # The model may declare its own action contract (kinematic support-centred outputs):
    # explicit flag, else the checkpoint field, else the actor's sibling output_contract.json.
    if args.output_contract is not None:
        declared = json.loads(Path(args.output_contract).read_text(encoding='utf-8'))
    elif args.onnx.suffix.lower() in ('.pt', '.pth'):
        declared = checkpoint.get('output_contract') or {}
    else:
        sibling = Path(args.onnx).with_name('output_contract.json')
        declared = json.loads(sibling.read_text(encoding='utf-8')) if sibling.is_file() else {}
    contract = declared if any(tag in str(declared.get('action_version', ''))
                               for tag in ('support_centered', 'kinematic_centered')) else None
    reference = torch.zeros(1, 6)
    bounds = torch.tensor(model.jnt_range[refs['joints']], dtype=torch.float32)[None]
    for column, joint in enumerate(refs['joints']):
        if not model.jnt_limited[joint]:
            bounds[0, column] = torch.tensor([-float('inf'), float('inf')])
    target = np.asarray(args.command, dtype=float)
    legs = torch.zeros(1, 4); wheels = torch.zeros(1, 2)
    command = torch.zeros(1, 3); command[0, 0] = float(target[0]); command[0, 1] = float(target[1])
    command[0, 2] = .34

    viewer = None
    if not args.no_viewer:
        # The shell may not export a display (this desktop's X server is :2 by default);
        # MuJoCo/GLFW would otherwise abort before any window exists.
        if not os.environ.get('DISPLAY'):
            os.environ['DISPLAY'] = args.display or ':2'
            print(f'play: DISPLAY unset, using {os.environ["DISPLAY"]}', file=sys.stderr)
        if not os.environ.get('XAUTHORITY'):
            authority = Path.home() / '.Xauthority'
            if authority.is_file():
                os.environ['XAUTHORITY'] = str(authority)
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data,
                                          key_callback=controls.key if controls is not None else None)
        with viewer.lock():
            viewer.cam.trackbodyid = refs['root']
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.distance = 2.
    started = time.monotonic()
    ticks = int(round(args.seconds * 1000))
    applied = np.zeros(6)
    actions = 0
    try:
        for tick in range(ticks):
            if viewer is not None and not viewer.is_running():
                break
            if tick % 10 == 0:
                if controls is not None:
                    vx, yaw, reset = controls.read()
                    command[0, 0] = float(vx); command[0, 1] = float(yaw)
                    if reset:
                        mujoco.mj_resetData(model, data)
                        apply_initial_pose(model, data, refs, {'pose': pose})
                        mujoco.mj_forward(model, data)
                        legs.zero_(); wheels.zero_(); applied = np.zeros(6)
                q = torch.tensor(data.qpos[refs['qpos']], dtype=torch.float32)[None]
                dq = torch.tensor(data.qvel[refs['dofs']], dtype=torch.float32)[None]
                measured = sensors(model, data, refs)
                obs = encode(torch.tensor(measured[:3], dtype=torch.float32)[None],
                             torch.tensor(measured[3:6], dtype=torch.float32)[None],
                             q, dq, legs, wheels, reference, 'Motion', command)
                raw = torch.from_numpy(policy(obs.numpy()))[None]
                actions += 1
                if not torch.isfinite(raw).all():
                    break
                if contract is not None:
                    height_m = float(command[0, 2])
                    center = flat_two_link_centers(height_m, contract['center_formula'])
                    offset = contract.get('support_target_offset_rad')
                    if offset:
                        center = center + np.asarray(offset, dtype=float)
                    leg_target = center + raw[0].numpy()[LEGS] * float(contract.get('leg_halfwidth_rad', .15))
                    leg_target = np.clip(leg_target, model.jnt_range[np.array(refs['joints'])[LEGS], 0],
                                         model.jnt_range[np.array(refs['joints'])[LEGS], 1])
                    legs = torch.tensor(leg_target, dtype=torch.float32)[None]
                    wheels = torch.tensor(raw[0].numpy()[WHEELS] * float(contract.get('wheel_halfwidth_rad_s', 80.)),
                                          dtype=torch.float32)[None]
                else:
                    out = targets(raw, reference, bounds)
                    legs, wheels = out['leg_executed'], out['wheel_executed']
            if tick % 5 == 0:
                q6 = data.qpos[refs['qpos']]
                dq6 = data.qvel[refs['dofs']]
                effort = np.zeros(6)
                effort[LEGS] = LEG_KP * (legs[0].numpy() - q6[LEGS]) - LEG_KD * dq6[LEGS]
                effort[WHEELS] = WHEEL_KV * (wheels[0].numpy() - dq6[WHEELS])
                applied = np.clip(effort, -JOINT_CAP_NM, JOINT_CAP_NM)
            data.ctrl[refs['actuators']] = applied / refs['gears']
            data.qfrc_applied[:] = 0.
            if v4_contract is not None:
                data.qfrc_applied[refs['dofs']] = v4_net_force(data.qpos[refs['qpos']],
                                                               data.qvel[refs['dofs']], v4_contract)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                print(json.dumps({'termination': 'NONFINITE_STATE', 'tick': tick}), flush=True)
                break
            if plan is not None and tick % 100 == 0:
                plan.publish({'t': time.time(), 'steps': tick + 1, 'source': 'MuJoCo browser panel',
                              'height_m': float(data.xpos[refs['root'], 2]),
                              'vx_cmd': float(command[0, 0]), 'yaw_cmd': float(command[0, 1]),
                              'q': [float(v) for v in data.qpos[refs['qpos']]]})
            if viewer is not None:
                if tick % 5 == 0:
                    viewer.sync()
                delay = started + .001 * (tick + 1) / args.speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    finally:
        if viewer is not None:
            viewer.close()
        if panel_server is not None:
            from native_stand.students.motion_panel import stop as stop_panel
            stop_panel(panel_server)
    print(json.dumps(dict(onnx=str(args.onnx), steps=tick + 1, seconds=round((tick + 1) / 1000., 3),
                          actions=actions, command=target.tolist(),
                          height_m=round(float(data.xpos[refs['root'], 2]), 4),
                          read_only=True, files_written=0,
                          source_kind=args.onnx.suffix.lower(),
                          v4_plant=bool(args.v4_plant),
                          v4_bodies=(v4_bodies or [])[:4],
                          action_contract=(contract or {}).get('action_version'),
                          assumptions=dict(leg_armature_kg_m2=args.leg_armature, friction=args.friction,
                                           ground_slope_pitch_rad=pitch, ground_slope_roll_rad=roll,
                                           pd=[LEG_KP, LEG_KD, WHEEL_KV], joint_caps_Nm=JOINT_CAP_NM.tolist(),
                                           start_pose=str(args.pose), no_manifest=True, no_spec=True,
                                           v4_contract=str(args.v4_contract) if args.v4_plant else None,
                                           no_transport=True, no_sensor_age=True, no_mcu_timing=True,
                                           no_sha_gate=True, no_critic=True)),
                    ensure_ascii=False), flush=True)
    if viewer is not None:
        sys.stdout.flush()
        os._exit(0)
    return 0


if __name__ == '__main__':
    sys.exit(main())
