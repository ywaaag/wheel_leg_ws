"""Named, scaled full engineering state; Actor37 never receives these fields."""

import numpy as np
import torch

from native_stand.tasks.jump.controller_critic import (
    Field,
    NormalizedCritic,
    controller_fields,
    extend_fields,
)

VERSION = "jump_direct_full_causal_MuJoCo_critic_v1"
TARGET = np.array([1.5, 1.5, 100, 1.5, 1.5, 100])
CAP = np.array([40, 40, 2.44694786377709, 40, 40, 2.44694786377709])


class FullCritic(NormalizedCritic):
    def contract(self):
        return {
            **super().contract(),
            "version": VERSION,
            "scope": "complete declared MuJoCo/C/V4 engineering model; Isaac schema not certified",
        }


def pack(env, actor, *, wheel_cap=4.8, phase_count=10, schema_class=FullCritic, engine_fields=None):
    p = env.plant
    e = p.execution
    v = e.v4
    signature = (wheel_cap, phase_count, schema_class)
    if getattr(env, "_pack_signature", signature) != signature:
        raise ValueError("Critic packing contract changed")
    env._pack_signature = signature
    layout = getattr(env, "_controller_layout", None)
    values, fields = controller_fields(
        e.c.states[0], e.c.inputs[0], fault_count=14, phase_count=phase_count, layout=layout
    )
    if layout is None:
        env._controller_layout = fields
    values["C.raw_output"] = values.pop("C.mean")
    previous = getattr(env, "schema", None)
    if previous is None:
        fields = [f for f in fields if f.name != "C.mean"] + [
            Field("C.raw_output", (6,), np.array(1.0), "raw Normal action")
        ]
    extras = {}

    def add(name, value, scale, unit):
        if previous is None:
            extras[name] = (np.asarray(value), scale, unit)
        else:
            values[name] = np.asarray(value)

    add("Actor.delayed37", actor, 1.0, "deployed normalized fields")
    for name in (
        []
        if engine_fields is not None
        else [
            "qpos",
            "qvel",
            "qacc",
            "qacc_warmstart",
            "ctrl",
            "qfrc_applied",
            "xfrc_applied",
            "qfrc_actuator",
        ]
    ):
        a = np.asarray(getattr(p.data, name))
        scale = (
            10.0
            if name in ["qvel"]
            else 100.0
            if name in ["qacc", "qacc_warmstart"]
            else 40.0
            if name in ["ctrl", "qfrc_applied", "xfrc_applied", "qfrc_actuator"]
            else 1.0
        )
        add("physics." + name, a, scale, "SI state; qpos includes xyz/quaternion/radians")
    model_fields = [
        ("body_mass", 25.0, "kg"),
        ("body_inertia", 1.0, "kg m2"),
        ("dof_armature", 0.1, "kg m2"),
        ("dof_damping", 10.0, "Nm s/rad"),
        ("dof_frictionloss", 10.0, "Nm"),
        ("geom_friction", 1.0, "dimensionless"),
        ("eq_solref", 0.02, "solver seconds/ratio"),
        ("eq_solimp", 1.0, "solver impedance"),
    ]
    model_epoch = getattr(p, "_parameter_epoch", 0)
    if engine_fields is None and getattr(env, "_model_fields_epoch", None) != model_epoch:
        env._model_fields = {name: getattr(p.model, name).copy() for name, _, _ in model_fields}
        env._model_fields_epoch = model_epoch
    for name, scale, unit in [] if engine_fields is not None else model_fields:
        add("model." + name, env._model_fields[name], scale, unit)
    for name, (value, scale, unit) in (engine_fields or {}).items():
        add("Isaac." + name, value, scale, unit)
    add("execution.subcycle", [e.tick % 5, e.tick % 10], [5, 10], "ms within PD/Actor cycle")
    # Include all serialized tensor state. Known static lookup indices are
    # excluded; no hidden response/queue value is replaced by zero padding.
    time_stamps = {"request_tick", "adopted_origin", "birth", "observed_tick"}
    physical = {
        "raw",
        "range_request",
        "requested",
        "adopted",
        "last_q",
        "targets",
        "qhistory",
        "target_policy_history",
        "lower",
        "upper",
    }
    velocity = {"last_dq", "dqhistory", "omega"}
    motor = {"motor", "motors", "motor_policy_history", "caps"}
    unitless = {"effective_packet_age", "packet_age", "target_delay", "wheel_delay", "hp_count"}
    for name, tensor in vars(v).items():
        if not torch.is_tensor(tensor) or name in {"ids", "columns", "sign"}:
            continue
        a = tensor.detach().cpu().numpy()
        if name in time_stamps:
            add("V4." + name + ".valid", a >= 0, 1.0, "bool")
            a = np.where(a >= 0, e.tick - a, 0.0)
            scale, unit = 1000.0, "relative age ms"
        elif name == "origins":
            a = np.roll(e.tick - a, -e.tick % len(a), axis=0)
            scale, unit = 1000.0, "chronological request age ms"
        elif name in physical:
            scale, unit = TARGET, "rad and rad/s by channel"
        elif name in velocity:
            scale, unit = 100.0, "rad/s"
        elif name in motor:
            scale, unit = np.array([40, 40, wheel_cap, 40, 40, wheel_cap]), "Nm by channel"
        elif name in {"NET", "NETs"}:
            scale, unit = 40.0, "external model Nm"
        elif name in {"wheel_PD", "wheel_raw", "wheel_queue"}:
            scale, unit = wheel_cap, "wheel Nm"
        elif name == "theta":
            scale, unit = 1.5, "paired model rad"
        elif name in {"packet"}:
            scale, unit = 1.0, "already normalized sensor packet"
        elif name in unitless:
            scale, unit = 1000.0 if name == "hp_count" else 40.0, "count or delay ms"
        elif name == "bias":
            scale, unit = 1.5, "encoder rad"
        elif name in {"hp_low", "hp_power"}:
            scale, unit = (
                100.0 if name == "hp_low" else 10000.0,
                "rad/s or accumulated squared velocity",
            )
        else:
            raise ValueError("unregistered V4 Critic tensor " + name)
        if name in {"targets", "motors", "NETs", "wheel_queue"}:
            a = np.roll(a, -e.tick % len(a), axis=0)
        elif name in {
            "packet",
            "qhistory",
            "dqhistory",
            "target_policy_history",
            "motor_policy_history",
        }:
            a = np.roll(a, -(e.tick // 10 + 1) % len(a), axis=0)
        add("V4." + name, a, scale, unit)
    r = v.leg
    scales = dict(
        x=(100.0, "named fitted tuple in bound model units"),
        kp=(40.0, "Nm/rad"),
        kd=(2.0, "Nm s/rad"),
        cap=(40.0, "Nm"),
        strength=(1.0, "gain"),
        vgain=(1.0, "gain"),
        color=(100.0, "rad/s"),
        color_std=(100.0, "rad/s"),
        color_rho=(1.0, "gain"),
        qbias=(1.5, "rad"),
        vbias=(100.0, "rad/s"),
        qnoise=(1.5, "rad"),
        vnoise=(100.0, "rad/s"),
        load_scale=(1.0, "gain"),
        qhist=(1.5, "rad"),
        vhist=(100.0, "rad/s"),
        queue=(40.0, "modal Nm"),
        delay=(40.0, "ms"),
        qage=(40.0, "ms"),
        vage=(40.0, "ms"),
        ref=(1.5, "rad"),
        alpha=(1.0, "response gain"),
        av=(1.0, "filter gain"),
        filtered=(100.0, "rad/s"),
        heldq=(1.5, "rad"),
        heldv=(100.0, "rad/s"),
        des=(40.0, "Nm"),
        actual=(40.0, "modal Nm"),
        des_unclipped=(40.0, "Nm"),
    )
    for name, tensor in vars(r).items():
        if not torch.is_tensor(tensor) or name == "ids":
            continue
        if name not in scales:
            raise ValueError("unregistered response Critic tensor " + name)
        a = tensor.detach().cpu().numpy()
        if name in {"qhist", "vhist", "queue"}:
            a = np.roll(a, -e.tick % len(a), axis=0)
        add("response." + name, a, *scales[name])
    memory = env.memory.state_dict()
    for name in ["elapsed_s", "horizon_s", "stable_s"]:
        add("task." + name, memory[name], 10.0, "seconds")
    for name in ["had_support", "actual_takeoff", "previous_contact", "landing_paid"]:
        add("task." + name, memory[name], 1.0, "bool")
    add("task.landing_only", memory["initial_kind"] == "landing_only", 1.0, "bool")
    for name, value in memory["reward"].items():
        if name == "initial_kind":
            continue
        if value is None:
            add("reward." + name + ".known", 0.0, 1.0, "bool")
            value = 0.0
        elif name in {"takeoff_COM_height_m", "peak_COM_height_m"}:
            add("reward." + name + ".known", 1.0, 1.0, "bool")
        add("reward." + name, value, 100.0 if name.endswith("_J") else 1.0, "J or explicit SI/bool")
    if previous is None:
        values, fields = extend_fields(values, fields, extras)
        schema = schema_class(fields)
    else:
        schema = previous
    # Encode makes an owned float32 result and checks every dynamic name/shape/
    # finite value. The live NumPy views above never escape this boundary.
    return schema.encode(values), schema
