"""Versioned inference package IO; full continuation checkpoints stay untouched."""

from pathlib import Path

from .io import file_sha256, read_json, write_json, write_npz

VERSION = "wheel_leg_deployment_v1"


def export_package(
    source, spec, output, *, execution, dependencies=(), onnx=False, index=None, golden=None
):
    import os
    import uuid

    from .io import json_digest

    output = Path(output).resolve()
    if output.exists():
        record = verify_package(output)
        wanted = sorted((ref["sha256"], ref.get("role")) for ref in dependencies)
        existing = sorted((ref["sha256"], ref.get("role")) for ref in record["dependencies"])
        if wanted != existing or (onnx and "actor.onnx" not in record["files"]):
            raise ValueError("Existing deployment dependencies/export target differ")
        if record.get("golden_source") != golden:
            raise ValueError("Existing deployment golden source differs")
        if (
            record["source"]["sha256"] != source.sha256
            or json_digest(record["inference_spec"]) != json_digest(spec.record())
            or json_digest(record["execution"]) != json_digest(execution)
        ):
            raise ValueError("Existing deployment belongs to another source/contract")
    else:
        temporary = output.with_name("." + output.name + ".building-" + uuid.uuid4().hex)
        record = _export_new(
            source,
            spec,
            temporary,
            execution=execution,
            dependencies=dependencies,
            onnx=onnx,
            golden=golden,
        )
        os.rename(temporary, output)
    if index is not None:
        from .model_refs import ModelReference

        for ref in [dict(path=str(source.path), sha256=source.sha256), *dependencies]:
            index.register(
                ModelReference(
                    ref["sha256"],
                    ref["path"],
                    ("deployment",),
                    run=str(output),
                    kind="deployment_dependency",
                )
            )
        for name in ("actor.pt", "manifest.json"):
            file = output / name
            index.register_output(
                ModelReference(
                    file_sha256(file),
                    str(file),
                    ("deployment",),
                    run=str(output),
                    kind="deployment_package",
                    owner=str(output),
                ),
                consumer=str(output),
            )
    return record


def check_output_golden(actual, expected, kind):
    """Compare in declared physical supervision units, not mixed rad/rad_s."""
    import numpy as np

    scale = np.array([0.05, 0.05, 10.0, 0.05, 0.05, 10.0]) if kind == "recovery_physical" else 1.0
    np.testing.assert_allclose(actual / scale, expected / scale, rtol=1e-5, atol=1e-5)
    error = np.abs(actual - expected)
    return dict(
        max_abs_error=float(error.max()),
        per_channel_max_abs_error=error.max(0).tolist(),
        tolerance="rtol=1e-5,atol=1e-5 after physical supervision scaling"
        if kind == "recovery_physical"
        else "rtol=1e-5,atol=1e-5",
    )


def _export_new(source, spec, output, *, execution, dependencies=(), onnx=False, golden=None):
    import shutil

    import torch

    from native_stand.learning.inference import from_source

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Deployment output must be new: " + str(output))
    model = from_source(source, spec)
    output.mkdir(parents=True)
    if golden is not None:
        if file_sha256(golden["path"]) != golden["sha256"]:
            raise ValueError("Real golden dataset SHA differs")
        import numpy as np

        with np.load(golden["path"], allow_pickle=False) as data:
            x = torch.as_tensor(data["states"][golden["indices"]], dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != spec.input_dim or not len(x) or not torch.isfinite(x).all():
            raise ValueError("Invalid real golden observations")
    else:
        x = torch.linspace(-0.5, 0.5, 4 * spec.input_dim).reshape(4, spec.input_dim)
    y = model.mean(x)
    actor = output / "actor.pt"
    torch.save(
        dict(
            actor=model.actor.state_dict(),
            actor_only=True,
            inference_spec=spec.record(),
            output_contract=spec.output_contract,
            source_sha256=source.sha256,
        ),
        actor,
    )
    write_npz(output / "golden.npz", dict(inputs=x.numpy(), outputs=y.numpy()))
    if spec.kind == "recovery_physical":
        with torch.no_grad():
            mean = model.actor.mean(x)
        write_npz(
            output / "golden.npz", dict(inputs=x.numpy(), mean=mean.numpy(), outputs=y.numpy())
        )
    copied = []
    for number, ref in enumerate(dependencies):
        path = Path(ref["path"])
        if file_sha256(path) != ref["sha256"]:
            raise ValueError("Deployment dependency SHA differs: " + str(path))
        dest = (
            output / "control_core_source" / path.name
            if ref.get("role") in ("C_manifest", "C_source")
            else output / "dependencies" / f"{number}_{path.name}"
        )
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
        copied.append(dict(**ref, package_path=str(dest.relative_to(output))))
    if onnx:

        class Mean(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = model.actor

            def forward(self, inputs):
                return self.actor({"student_actor": inputs}, stochastic_output=False)

        torch.onnx.export(
            Mean(),
            (x,),
            str(output / "actor.onnx"),
            input_names=["observation"],
            output_names=["mean"],
            dynamic_axes={"observation": {0: "batch"}, "mean": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    ONNX_error = None
    if onnx:
        import numpy as np
        import onnx as onnx_format
        from onnx.reference import ReferenceEvaluator

        onnx_format.checker.check_model(onnx_format.load(output / "actor.onnx"), full_check=True)

        actual = ReferenceEvaluator(str(output / "actor.onnx")).run(
            None, {"observation": x.numpy()}
        )[0]
        ONNX_error = check_output_golden(actual, y.numpy(), spec.kind)
    c_build = None
    if execution.get("C_core"):
        if execution["C_core"] == "jump_ABI4":
            from native_stand.simulation.jump_build import build
        elif execution["C_core"] == "rl_policy_core_v1":
            from native_stand.simulation.control_core import build
        else:
            raise ValueError("Unsupported deployment C ABI")
        c_build = build(output / "control.so")
    C_decoder_error = None
    if c_build and spec.kind == "recovery_physical":
        import numpy as np

        from native_stand.simulation.control_core import ControlCore

        core = ControlCore(c_build, spec.output_contract)
        decoded = np.stack(
            [
                core.decode(row, float(obs[24] * 0.5))
                for row, obs in zip(mean.numpy(), x.numpy(), strict=True)
            ]
        )
        np.testing.assert_allclose(decoded, y.numpy(), atol=2e-5, rtol=1e-6)
        C_decoder_error = float(np.abs(decoded - y.numpy()).max())
    C_actor = None
    if c_build and spec.kind == "recovery_physical":
        from .c_actor import export_recovery_actor

        C_actor = export_recovery_actor(
            model.actor,
            output,
            x.numpy(),
            y.numpy(),
            core_source=output / "control_core_source"
            if (output / "control_core_source" / "manifest.json").exists()
            else None,
        )
    record = dict(
        version=VERSION,
        source=dict(path=str(source.path), sha256=source.sha256),
        inference_spec=spec.record(),
        actor_output_space="physical_target" if spec.kind == "recovery_physical" else "policy_raw",
        decoder_in_actor=spec.kind == "recovery_physical",
        execution=execution,
        golden_source=golden,
        C_host_build=c_build,
        dependencies=copied,
        checks=dict(
            torch_golden=True,
            C_decoder_max_abs_error=C_decoder_error,
            C_actor=C_actor,
            ONNX_reference=True if onnx else "NOT_RUN",
            ONNX_error=ONNX_error,
            ORT="NOT_RUN",
            ARM="NOT_RUN",
            hardware="OFF",
        ),
        files={
            p.relative_to(output).as_posix(): file_sha256(p)
            for p in output.rglob("*")
            if p.is_file()
        },
    )
    record["path_mapping"] = {r["path"]: r["package_path"] for r in copied}
    write_json(output / "manifest.json", record)
    verify_package(output)
    return record


def verify_package(directory):
    import numpy as np
    import torch

    from native_stand.learning.inference import InferenceSpec, load_model

    directory = Path(directory)
    record = read_json(directory / "manifest.json")
    if record["version"] != VERSION:
        raise ValueError("Deployment version differs")
    for name, sha in record["files"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or file_sha256(path) != sha:
            raise ValueError("Deployment member identity differs: " + name)
    model = load_model(
        directory / "actor.pt",
        InferenceSpec.from_record(record["inference_spec"]),
        sha256=record["files"]["actor.pt"],
    )
    with np.load(directory / "golden.npz") as data:
        actual = model.mean(torch.from_numpy(data["inputs"])).numpy()
        np.testing.assert_allclose(actual, data["outputs"], atol=1e-6, rtol=1e-6)
    return record


def job_dependencies(job):
    """Versioned C source and execution contracts; do not copy BC training pools."""
    from native_stand.artifacts.runtime_paths import RuntimePaths
    from native_stand.simulation.control_core import SOURCE, verify_source

    paths = RuntimePaths.from_assets(job.assets)
    if job.task == "jump":
        source = Path(__file__).parents[1] / "simulation" / "jump_control_source"
        manifest = read_json(source / "manifest.json")
    else:
        source = SOURCE
        manifest = verify_source()
    result = [
        dict(
            path=str(source / "manifest.json"),
            sha256=file_sha256(source / "manifest.json"),
            role="C_manifest",
        )
    ]
    for name, value in manifest.get("files", manifest.get("source_sha", {})).items():
        name = Path(name).name
        if not (source / name).is_file():
            continue
        sha = value["sha256"] if isinstance(value, dict) else value
        result.append(dict(path=str(source / name), sha256=sha, role="C_source"))
    for name in (
        "recovery_training_contract",
        "recovery_critic_contract",
        "recovery_isaac_runtime",
    ):
        ref = job.assets.get(name)
        if ref:
            result.append(dict(path=ref["path"], sha256=ref["sha256"], role=name))
    runtime = job.assets.get("recovery_isaac_runtime")
    if runtime:
        bound = paths.read(runtime["path"])
        for name, ref in bound.get("bindings", {}).items():
            if name in ("V4_contract", "initial_states", "scene_spec"):
                result.append(dict(path=ref["path"], sha256=ref["sha256"], role=name))
    if job.assets.get("recovery_mujoco"):
        c = job.assets["recovery_mujoco"]
        for name in ("catalog", "V4_contract", "V4_tensor", "model_readback"):
            result.append(dict(**c[name], role=name))
        for name, ref in c["execution_sources"].items():
            result.append(dict(**ref, role="execution_" + name))
    supervision = getattr(job, "supervision", None)
    if supervision:
        for teacher in supervision.teachers:
            result.append(dict(path=teacher.path, sha256=teacher.sha256, role="teacher_checkpoint"))
    if job.assets.get("runtime_template"):
        from native_stand.simulation.playback import ASSET_BINDINGS

        template = paths.read(job.assets["runtime_template"])
        bindings = paths.read(ASSET_BINDINGS)
        if bindings["asset_sha256"] != template["asset_sha256"]:
            raise ValueError("No bound deployment mechanical description for this USD")
        for name, ref in bindings["bindings"].items():
            result.append(dict(path=ref["path"], sha256=ref["sha256"], role=name))
    return [{**ref, "path": paths.resolve(ref["path"])} for ref in result]
