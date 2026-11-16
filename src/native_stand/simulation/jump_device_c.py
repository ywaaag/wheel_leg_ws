"""Device views of the unchanged firmware ABI; no replacement FSM semantics."""

import ctypes as C

import numpy as np
import torch

from native_stand.simulation.jump_shared_c import Input, State


def abi_words(rows, structure, device):
    data = np.frombuffer(C.string_at(C.addressof(rows), C.sizeof(rows)), dtype=np.int32).copy()
    return torch.from_numpy(data.reshape(-1, C.sizeof(structure) // 4)).to(device)


def abi_field(words, structure, name):
    kind = dict(structure._fields_)[name]
    count = C.sizeof(kind) // 4
    offset = getattr(structure, name).offset // 4
    value = words[:, offset : offset + count]
    if kind is C.c_float or getattr(kind, "_type_", None) is C.c_float:
        value = value.view(torch.float32).double()
    else:
        value = value.long() & 0xFFFFFFFF
    return value[:, 0] if count == 1 else value


def controller_device_fields(execution, device):
    c = execution.c
    sw = execution.gpu.states if hasattr(execution, "gpu") else abi_words(c.states, State, device)
    iw = execution.gpu.inputs if hasattr(execution, "gpu") else abi_words(c.inputs, Input, device)

    def s(name):
        return abi_field(sw, State, name)

    def i(name):
        return abi_field(iw, Input, name)

    now = i("now")

    def age(value):
        return (now - value & 4294967295).double() * 0.001

    values = {}
    for name, size in (("phase", 10), ("owner", 5), ("fault", 14), ("status", 8)):
        values["C." + name] = torch.nn.functional.one_hot(s(name), size)
    for name in (
        "initialized",
        "last_key",
        "entry_valid",
        "adopted_valid",
        "command_valid",
        "first_hold",
        "airborne",
        "success",
        "stable_valid",
    ):
        values["C." + name] = s(name)
    true = torch.ones_like(now, dtype=torch.bool)
    enabled = dict(
        phase_tick=true,
        jump_tick=(s("phase") >= 1) & (s("phase") <= 6),
        last_tick=s("initialized").bool(),
        wait_tick=s("status") == 1,
        entry_tick=s("entry_valid").bool(),
        adopted_tick=s("adopted_valid").bool(),
        command_tick=s("command_valid").bool(),
        command_snapshot_tick=s("command_valid").bool(),
        stable_tick=s("stable_valid").bool(),
        event_tick=true,
    )
    for name, valid in enabled.items():
        values["C.age." + name] = torch.where(valid, age(s(name)), 0.0)
    for name in ("contact", "candidate", "contact_valid", "off_valid"):
        values["C." + name] = s(name)
    values["C.candidate_age"] = torch.where(
        s("candidate") < 2,
        ((now[:, None] - s("candidate_tick")) & 0xFFFFFFFF).double() * 0.001,
        0.0,
    )
    values["C.off_age"] = torch.where(
        s("off_valid").bool(), ((now[:, None] - s("off_tick")) & 0xFFFFFFFF).double() * 0.001, 0.0
    )
    values["C.command_identity"] = torch.stack(
        (
            s("command_epoch") == s("epoch"),
            s("command_phase_sequence") == s("phase_sequence"),
            s("command_previous_sequence") == s("adopted_sequence"),
            ((s("command_sequence") - s("adopted_sequence")) & 0xFFFFFFFF) < 0x80000000,
        ),
        -1,
    )
    for name in ("adopted", "mean", "requested", "protected_target", "pending", "support_N"):
        values["C.raw_output" if name == "mean" else "C." + name] = s(name)
    for name in (
        "enabled_mode",
        "key_down",
        "upstream_owner",
        "legacy_jump",
        "motion_entry",
        "motion_success",
        "support_mask",
        "support_model_valid",
        "feedback_mask",
        "imu_valid",
        "adopted_valid",
        "gyro",
        "gravity",
        "q",
        "dq",
        "command",
        "upstream_target",
        "accel",
        "driver_tau",
        "internal_tau",
        "support_J",
        "leg_length",
        "leg_spread",
    ):
        values["input." + name] = i(name)
    for name in ("sensor_tick", "imu_tick", "support_tick"):
        values["input.age." + name] = age(i(name))
    values["input.feedback_ages"] = (
        (now[:, None] - i("feedback_tick")) & 0xFFFFFFFF
    ).double() * 0.001
    return values
