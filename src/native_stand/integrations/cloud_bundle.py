"""CPU-only dependency inventory and relocatable source archive; never starts an owner.

The resulting plan is deliberately not a stop/import authorization. Final handoff
must bind a settled checkpoint after the old owner has terminated.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from native_stand.artifacts.io import file_sha256
from native_stand.experiments.config import load_job_config


def inventory(jobs, source_root, *, runs=(), references=()):
    artifacts = {}
    unresolved = []
    image_requirements = []
    pending_json = []
    queued_json = set()
    closure_limits = dict(depth=16, json_files=512, json_bytes=8 * 1024 * 1024)

    def add(path, role, expected=None, *, depth=0, expand=True):
        if path is None:
            return
        path = Path(path)
        if not path.is_absolute():
            return
        if not path.is_file():
            unresolved.append(dict(path=str(path), role=role, reason="missing_or_directory"))
            return
        resolved = str(path.resolve())
        sha = file_sha256(path)
        if expected is not None and sha != expected:
            unresolved.append(
                dict(
                    path=str(path),
                    role=role,
                    reason="sha_mismatch",
                    expected_sha256=expected,
                    actual_sha256=sha,
                )
            )
        entry = artifacts.setdefault(
            str(path.absolute()),
            dict(
                path=str(path),
                resolved=resolved,
                sha256=sha,
                bytes=path.stat().st_size,
                consumers=[],
                archive_path="dependencies/" + str(path.absolute()).lstrip("/"),
            ),
        )
        if role not in entry["consumers"]:
            entry["consumers"].append(role)
        # Only content-identified JSON is a reference document. Never open PT,
        # NPZ, Python or a directory, and never trust mismatched bytes as input.
        identity = (resolved, sha)
        if (
            expand
            and expected == sha
            and path.suffix.lower() == ".json"
            and identity not in queued_json
        ):
            queued_json.add(identity)
            pending_json.append((path, role, depth, sha))

    def walk(value, role, *, depth=0):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and value.get("sha256"):
                add(value["path"], role, value["sha256"], depth=depth)
                # A reference may also carry other explicitly identified inputs.
                walk(
                    {k: v for k, v in value.items() if k not in ("path", "sha256")},
                    role,
                    depth=depth,
                )
                return
            for key, val in value.items():
                # Historical evidence is recorded, not a live execution dependency.
                if key in (
                    "sources",
                    "history",
                    "traces",
                    "evaluations",
                    "reports",
                    "wiring_reuse",
                    "supported_output_recovery",
                ):
                    continue
                if (
                    isinstance(val, str)
                    and val.startswith("/")
                    and not val.startswith("/World/")
                    and (
                        (key.endswith("_path") and key != "storage_path")
                        or key
                        in (
                            "checkpoint",
                            "contract",
                            "output_contract",
                            "pose_library",
                            "prefix_onnx",
                            "runtime_template",
                        )
                    )
                ):
                    expected = value.get(key.removesuffix("_path") + "_sha256")
                    if key == "checkpoint":
                        expected = value.get("sha256", expected)
                    add(val, role + "/" + key, expected, depth=depth)
                else:
                    walk(val, role + "/" + key, depth=depth)
        elif isinstance(value, list | tuple):
            for val in value:
                walk(val, role, depth=depth)

    configs = []
    for config in jobs:
        job = load_job_config(config)
        configs.append(dict(path=str(job.config_path), sha256=job.config_sha256, task=job.task))
        add(job.config_path, job.task + "/recipe", job.config_sha256, expand=False)
        walk(job.as_run_record()["assets"], job.task + "/assets")
        if job.supervision is not None:
            walk(job.supervision.record(), job.task + "/supervision")
        for name in ("runtime_template", "recovery_isaac_runtime"):
            ref = job.assets.get(name)
            if ref is not None:
                path = ref["path"] if isinstance(ref, dict) else ref
                walk(json.loads(Path(path).read_text()), job.task + "/execution/" + name)
        add(job.initialization.checkpoint, job.task + "/initialization", job.initialization.sha256)
        if job.assets.get("branch_initialization"):
            from native_stand.artifacts.branch_initialization import read_branch
            from native_stand.artifacts.io import read_json

            branch = read_branch(job.assets)
            for name in ("selection", "settlement"):
                ref = branch[name]
                add(ref["path"], job.task + "/branch/" + name, ref["sha256"])
            selection = branch["selection_record"]
            ref = selection["source_center_binding"]
            add(ref["path"], job.task + "/branch/center", ref["sha256"])
            add(
                selection["output_contract"],
                job.task + "/branch/output_contract",
                selection["output_contract_sha256"],
            )
            settlement = read_json(branch["settlement"]["path"])
            for name in ("selected_checkpoint", "final_checkpoint"):
                ref = settlement[name]
                add(ref["path"], job.task + "/branch/" + name, ref["sha256"])

    # Settled preparation may name valid dependencies independently of a legacy
    # run's stale candidate view. Every supplied reference still needs its SHA;
    # this does not authorize importing or rewriting the original run.
    for reference in references:
        add(reference["path"], reference["role"], reference["sha256"])

    # Expand an explicit finite SHA-reference closure, not a runs/history scan.
    # State/budget evidence may themselves carry live checkpoint dependencies;
    # historical collections retain the same exclusions used at the root.
    position = 0
    while position < len(pending_json):
        path, role, depth, sha = pending_json[position]
        position += 1
        reason = None
        if position > closure_limits["json_files"]:
            reason = "json_reference_count_limit"
        elif depth > closure_limits["depth"]:
            reason = "json_reference_depth_limit"
        elif path.stat().st_size > closure_limits["json_bytes"]:
            reason = "json_reference_size_limit"
        if reason:
            unresolved.append(dict(path=str(path), role=role, reason=reason))
            if position > closure_limits["json_files"]:
                break
            continue
        try:
            content = path.read_bytes()
            # Bind traversal to the exact bytes checked by add, even if a live
            # state file changes between inventory and reference expansion.
            import hashlib

            if hashlib.sha256(content).hexdigest() != sha:
                unresolved.append(dict(path=str(path), role=role, reason="reference_changed"))
                continue
            document = json.loads(content)
        except (ValueError, UnicodeError) as exc:
            unresolved.append(
                dict(path=str(path), role=role, reason="invalid_reference_json", error=str(exc))
            )
            continue
        walk(document, role + "/reference", depth=depth + 1)

    # The immutable runtime imports these three numerical namespaces and the
    # explicitly bound bundle. Preserve their own paths and resolved identities.
    for config in jobs:
        job = load_job_config(config)
        for key in ("frozen_source_root", "frozen_bundle"):
            root = job.assets.get(key)
            if root is None:
                continue
            roots = (
                [Path(root)]
                if key == "frozen_bundle"
                else [
                    Path(root) / "native_stand" / name
                    for name in ("students", "recovery", "command_v6")
                ]
            )
            for subtree in roots:
                for path in sorted(subtree.rglob("*.py")):
                    add(path, job.task + "/" + key)
        template = job.assets.get("runtime_template")
        if template is None and job.assets.get("recovery_isaac_runtime"):
            runtime = json.loads(Path(job.assets["recovery_isaac_runtime"]["path"]).read_text())
            template = runtime["bindings"]["scene_spec"]["path"]
        if template:
            record = json.loads(Path(template).read_text())
            walk(record, job.task + "/physical_scene")
            module = Path(record["asset_module_path"])
            for path in (module.parent / "__init__.py", module.parent.parent / "__init__.py"):
                add(path, job.task + "/asset_import")
            # Resolve only literal assets used by the exact SHA-bound module.
            import ast

            for node in ast.walk(ast.parse(module.read_text())):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in ("_asset_file", "_make_cfg")
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    add(
                        module.parent / "usd_files/rmuc_wheel_leg" / node.args[0].value,
                        job.task + "/asset_import_USD",
                    )
    from pxr import UsdUtils

    for row in list(artifacts.values()):
        if Path(row["path"]).suffix not in (".usd", ".usda", ".usdc"):
            continue
        layers, assets, missing = UsdUtils.ComputeAllDependencies(row["path"])
        for layer in layers:
            add(layer.realPath, "USD_layer")
        for asset in assets:
            add(asset, "USD_asset")
        for asset in missing:
            if asset == "OmniPBR.mdl":
                item = dict(
                    asset=asset,
                    provider="IsaacSim image MDL resolver",
                    target_verification="NOT_RUN",
                )
                if item not in image_requirements:
                    image_requirements.append(item)
            else:
                unresolved.append(dict(path=asset, role="USD", reason="unresolved_layer_or_asset"))
    run_snapshots = []
    for run in runs:
        run = Path(run).resolve()
        record = json.loads((run / "run.json").read_text())
        state = json.loads((run / "state.json").read_text())
        budget = json.loads((run / "budget.json").read_text())
        used = sum(budget["used"].values())
        refs = {}

        def collect(value, role, run=run, refs=refs):
            if isinstance(value, dict):
                if value.get("checkpoint") and value.get("sha256"):
                    add(value["checkpoint"], run.name + "/" + role, value["sha256"])
                    refs[value["checkpoint"]] = value
                else:
                    for key, val in value.items():
                        collect(val, role + "/" + key)
            elif isinstance(value, list):
                for val in value:
                    collect(val, role)

        collect(
            {
                key: state.get(key)
                for key in (
                    "latest_model",
                    "best_candidate",
                    "best_standing_candidate",
                    "selected_model",
                )
            },
            "checkpoint",
        )
        run_snapshots.append(
            dict(
                path=str(run),
                run_record=record,
                state_record=state,
                run_id=record["run_id"],
                status=state["status"],
                source=record["source"],
                used=used,
                remaining=sum(budget["limits"].values()) - used,
                budget=budget,
                latest=state.get("latest_model"),
                references=refs,
                handoff_authorized=False,
                warning="read-only snapshot; final handoff requires explicit settlement manifest",
            )
        )
    fixed = Path(source_root) / "configs/model_refs.json"
    for row in json.loads(fixed.read_text())["fixed_models"]:
        add(row["path"], row["consumer"], row["sha256"])
    sources = []
    source_root = Path(source_root).resolve()
    for directory in ("src", "configs/jobs", "configs/branches", "tests"):
        for path in sorted((source_root / directory).rglob("*")):
            if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
                continue
            if path.suffix not in (
                ".py",
                ".json",
                ".toml",
                ".yaml",
                ".yml",
                ".c",
                ".h",
                ".md",
                ".sh",
            ):
                continue
            sources.append(
                dict(
                    path=str(path),
                    archive_path="candidate/" + str(path.relative_to(source_root)),
                    sha256=file_sha256(path),
                )
            )
    for name in (
        "pyproject.toml",
        "README.md",
        "AGENTS.md",
        "ARCHITECTURE.md",
        "configs/model_refs.json",
        "scripts/check_architecture.py",
        "STATUS.md",
        "docs/MIGRATION_20261105.md",
    ):
        path = source_root / name
        sources.append(
            dict(path=str(path), archive_path="candidate/" + name, sha256=file_sha256(path))
        )
    return dict(
        version="cloud_dependency_plan_v1",
        configs=configs,
        artifacts=list(artifacts.values()),
        sources=sources,
        unresolved=unresolved,
        image_requirements=image_requirements,
        json_reference_closure=dict(
            limits=closure_limits, visited=min(position, closure_limits["json_files"])
        ),
        run_snapshots=run_snapshots,
        ready_for_training=False,
        owner_terminated=False,
        import_authorized=False,
        target=dict(
            IsaacSim="6.2.0",
            IsaacLab="3.0.0",
            Python="3.12",
            engine="PhysX",
            environment="existing uv; preserve LD_LIBRARY_PATH",
        ),
        NOT_RUN=[
            "remote API/PhysX/contact/GPU checks",
            "external Python import closure",
            "settled full checkpoint handoff",
        ],
        required_remote_inputs=[
            "SSH endpoint/key",
            "persistent disk mount and capacity",
            "image identity",
            "GPU UUID/mapping",
        ],
        jump="ABI4 MuJoCo CPU only; no Isaac GPU backend",
    )


def package(plan, output):
    """Only explicitly inventoried files; no runs/Conda directory scan or symlink traversal."""
    if plan["unresolved"]:
        raise ValueError("依赖闭包尚有冲突或缺失，拒绝生成可运行包；请检查manifest")
    import os

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with tarfile.open(temporary, "w", dereference=True) as archive:
            import io

            blob = json.dumps(plan, ensure_ascii=False, indent=2).encode()
            entry = tarfile.TarInfo("manifest.json")
            entry.size = len(blob)
            archive.addfile(entry, io.BytesIO(blob))
            contents = {}
            for row in plan["sources"] + plan["artifacts"]:
                if file_sha256(row["path"]) != row["sha256"]:
                    raise ValueError("Dependency changed during packaging: " + row["path"])
                if row["sha256"] in contents:
                    link = tarfile.TarInfo(row["archive_path"])
                    link.type = tarfile.LNKTYPE
                    link.mode = Path(row["path"]).stat().st_mode & 0o777
                    link.linkname = contents[row["sha256"]]
                    archive.addfile(link)
                else:
                    archive.add(row["path"], arcname=row["archive_path"], recursive=False)
                    contents[row["sha256"]] = row["archive_path"]
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_dependencies(plan, archive_path, destination, *, source_root):
    """Restore only SHA-bound dependencies from an existing bundle, never old runs.

    Equal contents share an inode inside the new data root. Original files are
    neither linked nor changed; this is local storage, not an independent backup.
    Existing unequal bytes are rejected, including after an interrupted restore.
    """
    import os
    import shutil

    from native_stand.artifacts.io import json_digest, write_json
    from native_stand.artifacts.model_refs import ModelIndex, ModelReference

    if plan.get("unresolved"):
        raise ValueError("Unresolved dependencies cannot be materialized")
    destination = Path(destination).resolve()
    source_root = Path(source_root).resolve()
    rows, files, by_sha, references = [], {}, {}, []
    created_bytes = 0
    with tarfile.open(archive_path, "r") as archive:
        embedded = json.load(archive.extractfile("manifest.json"))
        if json_digest(embedded) != json_digest(plan):
            raise ValueError("Archive manifest identity differs")
        for row in plan["artifacts"]:
            relative = Path(row["archive_path"])
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.parts[0] != "dependencies"
            ):
                raise ValueError("Unsafe dependency member")
            target = destination / relative
            if target.is_symlink() or not target.resolve().is_relative_to(destination):
                raise ValueError("Dependency destination escapes data root")
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "existing_verified"
            if target.exists():
                if file_sha256(target) != row["sha256"]:
                    raise ValueError("Existing dependency SHA differs: " + str(target))
            elif row["sha256"] in by_sha:
                os.link(by_sha[row["sha256"]], target)
                mode = "data_root_content_hardlink"
            else:
                temporary = target.with_name(target.name + ".materializing")
                try:
                    stream = archive.extractfile(str(relative))
                    if stream is None:
                        raise ValueError("Missing dependency archive member")
                    with stream, temporary.open("wb") as output:
                        shutil.copyfileobj(stream, output, length=1024 * 1024)
                    if file_sha256(temporary) != row["sha256"]:
                        raise ValueError("Archive dependency SHA differs: " + str(relative))
                    os.link(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
                created_bytes += target.stat().st_size
                mode = "archive_copy"
            by_sha.setdefault(row["sha256"], target)
            files[row["path"]] = str(target)
            rows.append({**row, "target": str(target), "storage": mode})
            for consumer in row.get("consumers", ["dependency"]):
                references.append(
                    ModelReference(
                        row["sha256"],
                        str(target),
                        ("source",),
                        run="local-preparation/" + consumer,
                        kind="dependency",
                        reason="SHA-bound bundle input; original=" + row["path"],
                    )
                )
    # Only confirmed source roots receive a prefix. Dependency directory imports
    # are enumerated from actual archived Python modules, not /home/kellen.
    prefixes = {}
    for row in plan["sources"]:
        relative = Path(row["archive_path"])
        if relative.parts[:2] == ("candidate", "src"):
            old = str(Path(row["path"]).parents[len(relative.parts) - 2])
            prefixes[old] = str(source_root)
    for row in rows:
        if row["path"].endswith(".py") and not Path(row["path"]).is_relative_to(source_root):
            prefixes[str(Path(row["path"]).parent)] = str(Path(row["target"]).parent)
    mapping = dict(version="runtime_path_mapping_v1", files=files, prefixes=prefixes)
    write_json(destination / "runtime_paths.json", mapping)
    index = ModelIndex(destination / "models.json")
    index.register_many(references)
    report = dict(
        version="local_dependency_materialization_v1",
        dependencies=rows,
        mapping=str(destination / "runtime_paths.json"),
        model_index=str(index.path),
        archive_sha256=file_sha256(archive_path),
        manifest_sha256=json_digest(plan),
        new_content_bytes=created_bytes,
        logical_bytes=sum(r["bytes"] for r in rows),
        original_files_changed=False,
        original_inode_hardlinks=False,
        ready_for_training=False,
    )
    write_json(destination / "manifest.json", report)
    return report
