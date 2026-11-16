"""Versioned inference package IO; full continuation checkpoints stay untouched."""

from pathlib import Path

from .io import file_sha256, read_json, write_json, write_npz

VERSION = "wheel_leg_deployment_v1"


def export_package(source, spec, output, *, execution, dependencies=(), onnx=False, index=None):
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
        if (
            record["source"]["sha256"] != source.sha256
            or json_digest(record["inference_spec"]) != json_digest(spec.record())
            or json_digest(record["execution"]) != json_digest(execution)
        ):
            raise ValueError("Existing deployment belongs to another source/contract")
    else:
        temporary = output.with_name("." + output.name + ".building-" + uuid.uuid4().hex)
        record = _export_new(
            source, spec, temporary, execution=execution, dependencies=dependencies, onnx=onnx
        )
        os.rename(temporary, output)
    if index is not None:
        from .model_refs import ModelReference

        index.register_many(
            [
                ModelReference(
                    ref["sha256"],
                    ref["path"],
                    ("deployment",),
                    run=str(output),
                    kind="deployment_dependency",
                )
                for ref in [dict(path=str(source.path), sha256=source.sha256), *dependencies]
            ]
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


def _export_new(source, spec, output, *, execution, dependencies=(), onnx=False):
    import shutil

    import torch

    from native_stand.learning.inference import from_source

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Deployment output must be new: " + str(output))
    model = from_source(source, spec)
    output.mkdir(parents=True)
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
    copied = []
    for number, ref in enumerate(dependencies):
        path = Path(ref["path"])
        if file_sha256(path) != ref["sha256"]:
            raise ValueError("Deployment dependency SHA differs: " + str(path))
        dest = output / "dependencies" / f"{number}_{path.name}"
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
        copied.append(dict(**ref, package_path=str(dest.relative_to(output))))
    if onnx:

        class Mean(torch.nn.Module):
            def forward(self, inputs):
                return model.actor({"student_actor": inputs}, stochastic_output=False)

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
    if onnx:
        import numpy as np
        from onnx.reference import ReferenceEvaluator

        actual = ReferenceEvaluator(str(output / "actor.onnx")).run(
            None, {"observation": x.numpy()}
        )[0]
        np.testing.assert_allclose(actual, y.numpy(), rtol=1e-5, atol=1e-5)
    c_build = None
    if execution.get("C_core"):
        if execution["C_core"] == "jump_ABI4":
            from native_stand.simulation.jump_build import build
        elif execution["C_core"] == "rl_policy_core_v1":
            from native_stand.simulation.control_core import build
        else:
            raise ValueError("Unsupported deployment C ABI")
        c_build = build(output / "control.so")
        if execution["C_core"] == "rl_policy_core_v1":
            # The package is atomically renamed after this function. Preserve
            # usable package-relative identities, not the temporary build path.
            shutil.copytree(c_build["source"], output / "control_source")
            c_build = dict(c_build, path="control.so", source="control_source")
    record = dict(
        version=VERSION,
        source=dict(path=str(source.path), sha256=source.sha256),
        inference_spec=spec.record(),
        actor_output_space="physical_target" if spec.kind == "recovery_physical" else "policy_raw",
        decoder_in_actor=spec.kind == "recovery_physical",
        execution=execution,
        C_host_build=c_build,
        dependencies=copied,
        checks=dict(
            torch_golden=True,
            ONNX_reference=True if onnx else "NOT_RUN",
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


def control_binding(directory):
    """Resolve the packaged shared-C identity after export or relocation."""
    directory = Path(directory).resolve()
    record = verify_package(directory)
    binding = record.get("C_host_build")
    if record["execution"].get("C_core") != "rl_policy_core_v1" or not binding:
        raise ValueError("Packaged policy-core C binding required")
    library, source = ((directory / binding[k]).resolve() for k in ("path", "source"))
    if not library.is_relative_to(directory) or not source.is_relative_to(directory):
        raise ValueError("Shared C binding must stay inside its package")
    return dict(binding, path=str(library), source=str(source))


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
    if job.assets.get("runtime_template"):
        from native_stand.simulation.playback import ASSET_BINDINGS

        template = paths.read(job.assets["runtime_template"])
        bindings = paths.read(ASSET_BINDINGS)
        if bindings["asset_sha256"] != template["asset_sha256"]:
            raise ValueError("No bound deployment mechanical description for this USD")
        for name, ref in bindings["bindings"].items():
            result.append(dict(path=ref["path"], sha256=ref["sha256"], role=name))
    if job.assets.get("motion_vm1_contract"):
        ref = job.assets["motion_vm1_contract"]
        contract = paths.read(ref["path"])
        if file_sha256(paths.resolve(ref["path"])) != ref["sha256"]:
            raise ValueError("vM1 deployment reference contract SHA differs")
        result.append(dict(**ref, role="motion_vm1_contract"))
        result.extend(
            dict(path=r["path"], sha256=r["sha256"], role="motion_vm1_" + r["role"])
            for r in contract["members"]
        )
    return [{**ref, "path": paths.resolve(ref["path"])} for ref in result]
