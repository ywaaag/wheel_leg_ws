"""固定版本电控纯C源码包；构建/身份校验只发生在加载边界。"""

from __future__ import annotations

import ctypes as C
import subprocess
from pathlib import Path

import numpy as np

from native_stand.artifacts.io import file_sha256, json_digest, read_json

SOURCE = Path(__file__).with_name("control_core_source")
ABI = 1
F = C.c_float
D = C.c_double
P = C.c_void_p


class DecodeConfig(C.Structure):
    _fields_ = [
        ("abi", C.c_uint32),
        ("center_kind", C.c_uint32),
        ("node_count", C.c_uint32),
        ("leg_scale", F),
        ("wheel_scale", F),
        ("geometry", D * 6),
        ("height_min", D),
        ("height_max", D),
        ("offset", D * 4),
        ("heights", D * 16),
        ("offsets", (D * 2) * 16),
        ("leg_lower", D * 4),
        ("leg_upper", D * 4),
    ]


def verify_source(source=SOURCE):
    source = Path(source)
    manifest = read_json(source / "manifest.json")
    if manifest["ABI"] != ABI or manifest["version"] != "rl_policy_core_v1":
        raise ValueError("共享C核心ABI/源码版本不同")
    for name, sha in manifest["files"].items():
        if Path(name).name != name or file_sha256(source / name) != sha:
            raise ValueError("共享C源码身份改变: " + name)
    return manifest


def build(output, *, source=SOURCE, jump=False, additional_sources=()):
    source = Path(source)
    manifest = verify_source(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = ["rl_policy_core.c", "rl_motion31.c", "rl_control.c"]
    if jump:
        names += [
            "rl_jump_core.c",
            "rl_jump_nominal.c",
            "rl_jump_nominal_table.c",
            "rl_jump_platform.c",
        ]
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-O2",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-contract=off",
            *(["-DRL_JUMP_NOMINAL_V1=1"] if jump else []),
            "-I" + str(source),
            *(str(source / n) for n in names),
            *(str(Path(p)) for p in additional_sources),
            "-lm",
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(
        path=str(output.resolve()),
        sha256=file_sha256(output),
        ABI=ABI,
        source_sha256=json_digest(manifest),
        source=str(source.resolve()),
    )


def _a(value, n):
    return np.ascontiguousarray(value, dtype=np.float32).reshape(n)


def decoder_configuration(contract):
    from native_stand.tasks.motion.action_contract import ActionDecoder

    c = DecodeConfig()
    c.abi = ABI
    version = contract["action_version"]
    if version == "recovery_cubic_physical_Gaussian_v2":
        c.center_kind = 4
        c.leg_scale = 0.15
        c.wheel_scale = 100.0
        c.height_min = 0.30
        c.height_max = 0.40
        c.offset[:] = contract["center_rad"]
        c.leg_lower[:] = contract["leg_lower_rad"]
        c.leg_upper[:] = contract["leg_upper_rad"]
        return c
    if version == "motion_multiheight_support_center_poly5_015_80_v1":
        c.center_kind = 3
        c.leg_scale = 0.15
        c.wheel_scale = 80
        c.height_min = 0.30
        c.height_max = 0.40
        return c
    decoder = ActionDecoder.from_contract(contract)
    center = decoder.center
    c.center_kind = 2 if center.support_height_nodes_m else 1
    c.leg_scale = decoder.leg_scale
    c.wheel_scale = decoder.wheel_scale
    c.geometry[:] = [
        center.hip_above_root_m,
        center.wheel_radius_m,
        center.equivalent_upper_m,
        center.equivalent_lower_m,
        center.hip_zero_deg,
        center.active_zero_deg,
    ]
    c.height_min, c.height_max = center.height_range_m
    c.offset[:] = center.support_offset_rad
    c.node_count = len(center.support_height_nodes_m)
    if c.node_count > 16:
        raise ValueError("C中心表容量不足")
    for i, h in enumerate(center.support_height_nodes_m):
        c.heights[i] = h
        c.offsets[i][:] = center.support_offset_nodes_rad[i]
    return c


class ControlCore:
    def __init__(self, binding, contract):
        manifest = verify_source(binding["source"])
        if (
            binding["source_sha256"] != json_digest(manifest)
            or file_sha256(binding["path"]) != binding["sha256"]
        ):
            raise ValueError("共享C主机库/源码绑定不同")
        self.binding = dict(binding)
        self.lib = C.CDLL(binding["path"])
        self.lib.RLPolicyCoreAbi.restype = C.c_uint32
        if self.lib.RLPolicyCoreAbi() != ABI:
            raise ValueError("共享C实际ABI不同")
        self.lib.RLPolicyEncode.argtypes = [P] * 6 + [C.c_uint32, C.c_uint32, P]
        self.lib.RLPolicyDecode.argtypes = [P, P, F, P]
        self.lib.RLPolicyProtect.argtypes = [P] * 4
        self.lib.RLPolicyAdopt.argtypes = [P] * 2
        self.lib.RLPolicyPd.argtypes = [P] * 7
        self.lib.RLControlCompensateSpring.argtypes = [P] * 4
        self.config = decoder_configuration(contract)
        self.contract = dict(contract)
        self.adopted = np.zeros(6, np.float32)

    def encode(self, gyro, gravity, q, dq, command, *, skill=0, jump=0):
        values = [
            _a(x, n)
            for x, n in zip(
                (gyro, gravity, q, dq, self.adopted, command), (3, 3, 4, 6, 6, 3), strict=True
            )
        ]
        out = np.empty(31, np.float32)
        if self.lib.RLPolicyEncode(*(a.ctypes.data for a in values), skill, jump, out.ctypes.data):
            raise ValueError("C观测编码拒绝输入")
        return out

    def decode(self, raw, height):
        raw = _a(raw, 6)
        target = np.empty(6, np.float32)
        if self.lib.RLPolicyDecode(
            C.byref(self.config), raw.ctypes.data, height, target.ctypes.data
        ):
            raise ValueError("C输出合同拒绝动作或高度")
        return target

    def adopt(self, requested, lower, upper):
        values = [_a(x, 6) for x in (requested, lower, upper)]
        protected = np.empty(6, np.float32)
        if self.lib.RLPolicyProtect(*(x.ctypes.data for x in values), protected.ctypes.data):
            raise ValueError("C机械保护拒绝目标")
        if self.lib.RLPolicyAdopt(protected.ctypes.data, self.adopted.ctypes.data):
            raise ValueError("C实际采用失败")
        return protected

    def pd(
        self, q, dq, *, target=None, gains=(40.0, 2.0, 0.2), caps=(40.0, 40.0, 5.5, 40.0, 40.0, 5.5)
    ):
        values = [
            _a(x, n)
            for x, n in zip(
                (q, dq, self.adopted if target is None else target, gains, caps),
                (4, 6, 6, 3, 6),
                strict=True,
            )
        ]
        raw = np.empty(6, np.float32)
        limited = np.empty(6, np.float32)
        if self.lib.RLPolicyPd(
            *(x.ctypes.data for x in values), raw.ctypes.data, limited.ctypes.data
        ):
            raise ValueError("C PD拒绝非有限输入")
        return raw, limited

    def compensate(self, motor_order, length, spread):
        values = [_a(x, n) for x, n in zip((motor_order, length, spread), (4, 2, 2), strict=True)]
        result = np.empty(4, np.float32)
        if self.lib.RLControlCompensateSpring(*(x.ctypes.data for x in values), result.ctypes.data):
            raise ValueError("C软件补偿输入无效")
        return result
