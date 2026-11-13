"""Portable ELU/Tanh Actor C export using the existing physical decoder ABI."""

from pathlib import Path


def export_recovery_actor(actor, directory, inputs, expected):
    import ctypes
    import time

    import numpy as np
    import torch

    from native_stand.simulation.control_core import build

    directory = Path(directory)
    layers = list(actor.mlp)
    if any(not isinstance(m, torch.nn.Linear | torch.nn.ELU) for m in layers):
        raise ValueError("C Actor export requires explicit Linear/ELU layers")
    linear = [m for m in layers if isinstance(m, torch.nn.Linear)]
    if len(linear) != len([m for m in layers if isinstance(m, torch.nn.ELU)]) + 1:
        raise ValueError("C Actor requires ELU between every Linear layer")
    if linear[0].in_features != 31 or linear[-1].out_features != 6:
        raise ValueError("C Recovery Actor requires31 inputs and6 outputs")
    width = max(max(m.in_features, m.out_features) for m in linear)

    def constant(name, values):
        flat = np.asarray(values, np.float32).reshape(-1)
        # Hex floating literals reproduce every saved float32 weight exactly.
        values = ",".join(float(v).hex() + "f" for v in flat)
        return f"static const float {name}[{len(flat)}] = {{{values}}};\n"

    code = '#include <math.h>\n#include "rl_policy_core.h"\n#include "recovery_actor.h"\n'
    for i, m in enumerate(linear):
        code += constant(f"w{i}", m.weight.detach().cpu().numpy())
        code += constant(f"b{i}", m.bias.detach().cpu().numpy())
    contract = actor.contract

    def vector(values):
        return ",".join(format(float(x), ".17g") for x in values)

    code += (
        "static const RLPolicyDecodeConfig config = {"
        ".abi=RL_POLICY_CORE_ABI,.center_kind=RL_POLICY_RECOVERY_CUBIC,"
        ".leg_scale=.15f,.wheel_scale=100.f,.height_min=.30,.height_max=.40,"
        f".offset={{{vector(contract['center_rad'])}}},"
        f".leg_lower={{{vector(contract['leg_lower_rad'])}}},"
        f".leg_upper={{{vector(contract['leg_upper_rad'])}}}" + "};\n"
    )
    code += (
        f"int RecoveryActorMean(const float *input,float *mean) {{\nfloat a[{width}],b[{width}];\n"
    )
    code += "for(int j=0;j<31;j++){if(!isfinite(input[j]))return -1;a[j]=input[j];}\n"
    for i, m in enumerate(linear):
        n, k = m.in_features, m.out_features
        code += (
            f"for(int j=0;j<{k};j++){{float s=b{i}[j];for(int k=0;k<{n};k++)s+=w{i}[j*{n}+k]*a[k];"
        )
        code += "b[j]=tanhf(s);}" if i == len(linear) - 1 else "b[j]=s>=0?s:expm1f(s); }"
        code += f"\nfor(int j=0;j<{k};j++)a[j]=b[j];\n"
    code += "for(int j=0;j<6;j++){if(!isfinite(a[j]))return -2;mean[j]=a[j];}return 0;}\n"
    code += "int RecoveryActorPhysical(const float *input,float *mean,float *physical){int r=RecoveryActorMean(input,mean);if(r)return r;return RLPolicyDecode(&config,mean,input[24]*.5f,physical);}\n"
    header = """#ifndef RECOVERY_ACTOR_H
#define RECOVERY_ACTOR_H
/* Same normalized31D observation; actual adopted previous; CAD/model order.
 * Mean returns Tanh6; Physical calls the cubic decoder exactly ONCE.
 * Caller retains queues, actual adoption, PD/V4 and motor/NET routing. */
int RecoveryActorMean(const float *input, float *mean);
int RecoveryActorPhysical(const float *input, float *mean, float *physical);
#endif
"""
    (directory / "recovery_actor.c").write_text(code)
    (directory / "recovery_actor.h").write_text(header)
    library = directory / "recovery_actor_host.so"
    build(library, additional_sources=(directory / "recovery_actor.c",))
    lib = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    lib.RecoveryActorPhysical.argtypes = [array, array, array]
    lib.RecoveryActorPhysical.restype = ctypes.c_int
    x = np.ascontiguousarray(inputs, np.float32)
    means, physical = [], []
    for row in x:
        mean = np.empty(6, np.float32)
        target = np.empty(6, np.float32)
        if lib.RecoveryActorPhysical(row, mean, target):
            raise ValueError("C Actor rejected real golden observation")
        means.append(mean)
        physical.append(target)
    with torch.inference_mode():
        expected_mean = actor.mean(torch.from_numpy(x)).numpy()
    means, physical = np.stack(means), np.stack(physical)
    np.testing.assert_allclose(means, expected_mean, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        physical[:, [0, 1, 3, 4]], expected[:, [0, 1, 3, 4]], atol=5e-5, rtol=1e-5
    )
    np.testing.assert_allclose(physical[:, [2, 5]], expected[:, [2, 5]], atol=1e-3, rtol=1e-5)
    np.savez_compressed(directory / "golden_C.npz", inputs=x, mean=means, outputs=physical)
    row = x[0]
    mean = np.empty(6, np.float32)
    target = np.empty(6, np.float32)
    for _ in range(100):
        lib.RecoveryActorPhysical(row, mean, target)
    durations = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        lib.RecoveryActorPhysical(row, mean, target)
        durations.append(time.perf_counter_ns() - start)
    return dict(
        version="Recovery_ELU_Tanh_C99_v1",
        decoder="shared RLPolicyDecode exactly once",
        mean_max_abs_error=float(abs(means - expected_mean).max()),
        leg_max_abs_error_rad=float(
            abs(physical[:, [0, 1, 3, 4]] - expected[:, [0, 1, 3, 4]]).max()
        ),
        wheel_max_abs_error_rad_s=float(abs(physical[:, [2, 5]] - expected[:, [2, 5]]).max()),
        network_weight_bytes=sum((m.weight.numel() + m.bias.numel()) * 4 for m in linear),
        scratch_array_bytes=2 * width * 4,
        host_call_median_us=float(np.median(durations) / 1000),
        host_call_p95_us=float(np.percentile(durations, 95) / 1000),
        timing_scope="local CPU C call including ctypes; not MCU latency/stack/Flash certification",
        ARM="NOT_RUN",
        hardware="OFF",
    )
