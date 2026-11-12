"""Fixed ABI4 source package; build once at the loading boundary, hardware OFF."""

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from native_stand.artifacts.io import file_sha256, json_digest, read_json, write_json
from native_stand.tasks.jump.unified import contract

SOURCE = Path(__file__).with_name("jump_control_source")


FLAGS = (
    "-std=c99",
    "-O2",
    "-fPIC",
    "-shared",
    "-DRL_JUMP_DIRECT_V1=1",
    "-DRL_JUMP_NOMINAL_V1=0",
)


def build(output, *, compiler="cc", measurement=None):
    source = read_json(SOURCE / "manifest.json")
    if source["ABI"] != 4 or source["contract"] != contract():
        raise ValueError("Jump源码包ABI/合同不同")
    identities = {}
    for original, sha in source["source_sha"].items():
        file = SOURCE / Path(original).name
        if not file.exists():
            continue  # firmware platform integration is provenance, not a host source
        if file_sha256(file) != sha:
            raise ValueError("Jump固定源码身份不同: " + file.name)
        identities[str(file.resolve())] = sha
    required = {
        "rl_jump_core.c",
        "rl_jump_core.h",
        "rl_jump_platform.c",
        "rl_jump_platform.h",
        "rl_motion31.c",
        "rl_motion31.h",
        "rl_control.c",
        "rl_control.h",
        "rl_shadow.h",
    }
    if {Path(name).name for name in identities} != required:
        raise ValueError("Jump固定源码包不完整")
    executable = Path(shutil.which(compiler) or compiler).resolve(strict=True)
    toolchain = dict(
        executable=str(executable),
        sha256=file_sha256(executable),
        version=subprocess.check_output([str(executable), "--version"], text=True),
        machine=platform.machine(),
        system=platform.system(),
    )
    identity = dict(
        ABI=4,
        contract_sha256=json_digest(contract()),
        flags=list(FLAGS),
        sources={Path(name).name: sha for name, sha in identities.items()},
        source_manifest_sha256=file_sha256(SOURCE / "manifest.json"),
        toolchain=toolchain,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = output.with_suffix(".json")
    if output.exists() or receipt.exists():
        previous = read_json(receipt)
        if (
            not previous
            or previous.get("build_identity") != identity
            or not output.is_file()
            or file_sha256(output) != previous["sha256"]
            or previous["path"] != str(output.resolve())
        ):
            raise ValueError("Jump构建已存在但身份不同；使用新目录，不覆盖被引用的库")
        if measurement is not None:
            measurement.count("jump_C_build_cache_hits")
        return previous
    # Compile privately. A failure never leaves a reusable library/receipt.
    descriptor, temporary = tempfile.mkstemp(prefix=".jump-build-", suffix=".so", dir=output.parent)
    os.close(descriptor)
    command = [
        str(executable),
        *FLAGS,
        "-I" + str(SOURCE),
        *[
            str(SOURCE / n)
            for n in ("rl_jump_core.c", "rl_jump_platform.c", "rl_motion31.c", "rl_control.c")
        ],
        "-lm",
        "-o",
        temporary,
    ]
    try:
        if measurement is not None:
            measurement.count("jump_C_compilations")
        subprocess.run(command, check=True, capture_output=True, text=True)
        os.link(temporary, output)  # exclusive publication; never replace an existing library
    finally:
        Path(temporary).unlink(missing_ok=True)
    result = dict(
        path=str(output.resolve()),
        sha256=file_sha256(output),
        ABI=4,
        contract_sha256=json_digest(contract()),
        sources=identities,
        source_manifest_sha256=file_sha256(SOURCE / "manifest.json"),
        hardware_output=False,
        build_identity=identity,
        build_identity_sha256=json_digest(identity),
    )
    write_json(receipt, result)
    return result
