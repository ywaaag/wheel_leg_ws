"""Read-only source audit and local Recovery continuation plan; never submits."""

from __future__ import annotations

import argparse
from pathlib import Path

from native_stand.artifacts.io import file_sha256, json_digest, read_json, write_json
from native_stand.artifacts.runtime_paths import RuntimePaths, execution_job
from native_stand.experiments.config import load_job_config, with_initialization

SOURCE_RUN = "2026_11_11__recovery_preentry48_remaining"
SOURCE_SHA = "578f22a8a03a39235c4b7bbed9985efbf226ffe2ebd3d692d9176080261c6ea2"


def prepare(source_run, config, data_root, output):
    source_run, data_root, output = map(
        lambda p: Path(p).resolve(), (source_run, data_root, output)
    )
    if source_run.name != SOURCE_RUN or output.exists():
        raise ValueError("Unexpected source run or existing output; never overwrite evidence")
    record, state, book = (
        read_json(source_run / p) for p in ("run.json", "state.json", "budget.json")
    )
    if state["status"] != "STOPPED_BOUNDED" or any(
        a["status"] == "RUNNING" for a in book["attempts"].values()
    ):
        raise ValueError("Source owner has not settled")
    if state["updates_used"] != 1800 or book["used"] != {"smoke": 0, "formal": 1800}:
        raise ValueError(
            "Later source progress found: follow actual continuation, not this old plan"
        )
    if (
        book["limits"] != {"smoke": 0, "formal": 8334}
        or state["latest_model"]["sha256"] != SOURCE_SHA
    ):
        raise ValueError("Source identity/budget differs")
    mapping = data_root / "runtime_paths.json"
    paths = RuntimePaths.from_assets(
        {"path_mapping": dict(path=str(mapping), sha256=file_sha256(mapping))}
    )
    latest = state["latest_model"]
    cp = Path(paths.resolve(latest["checkpoint"]))
    if not cp.is_relative_to(data_root) or file_sha256(cp) != SOURCE_SHA:
        raise ValueError("Restored full source not verified in data root")
    job = with_initialization(load_job_config(config), cp, SOURCE_SHA)
    raw = job.as_run_record()
    raw["assets"]["path_mapping"] = dict(path=str(mapping), sha256=file_sha256(mapping))
    raw["assets"]["execution_resume"] = "fresh_reset"
    # The target machine is NOT known. Remove the inherited workstation resource identity.
    raw["assets"].pop("physical_gpu_uuid", None)
    raw["assets"].pop("gpu_resource_policy", None)
    from native_stand.experiments.config import job_from_record

    mapped = execution_job(job_from_record(raw))
    dependencies = []
    runtime = paths.read(raw["assets"]["recovery_isaac_runtime"]["path"])
    for role, ref in [
        *mapped.assets.items(),
        *[("runtime." + k, v) for k, v in runtime["bindings"].items()],
    ]:
        if not isinstance(ref, dict) or not {"path", "sha256"}.issubset(ref):
            continue
        if file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Dependency differs: " + role)
        dependencies.append(dict(role=role, **ref))
    models = []
    for role in ("latest_model", "best_candidate", "best_standing_candidate"):
        ref = state[role]
        path = Path(paths.resolve(ref["checkpoint"]))
        valid = path.is_file() and file_sha256(path) == ref["sha256"]
        models.append(
            dict(
                role=role,
                original=ref,
                mapped_path=str(path),
                status="AVAILABLE" if valid else "UNAVAILABLE",
                recorded_sha256=ref["sha256"],
            )
        )
    output.mkdir(parents=True)
    # Serialized full record accepted by job_from_record; consumed only via an imported same-logical-run resume.
    write_json(output / "job_record.json", mapped.as_run_record())
    result = dict(
        version="recovery_continuation_preparation_v1",
        ready_for_submission=False,
        source_run=str(source_run),
        logical_run_id=record["run_id"],
        source_full=dict(path=str(cp), sha256=SOURCE_SHA),
        source_records={
            n: dict(path=str(source_run / n), sha256=file_sha256(source_run / n))
            for n in ("run.json", "state.json", "budget.json")
        },
        source_code_identity_sha256=json_digest(record["source"]),
        source_config_contract_sha256=json_digest(
            {
                k: record["config"][k]
                for k in (
                    "model",
                    "observation",
                    "action",
                    "control",
                    "reward",
                    "curriculum",
                    "ppo",
                )
            }
        ),
        source_owner_status=state["status"],
        latest_counters=1800,
        original_cap=10000,
        ancestors_used=1666,
        stage_used=1800,
        cumulative_used=3466,
        remaining=6534,
        target_limits=book["limits"],
        target_used=book["used"],
        dependencies=dependencies,
        protected_models=models,
        target_config=dict(
            path=str(output / "job_record.json"), sha256=file_sha256(output / "job_record.json")
        ),
        resources=dict(
            status="NOT_READY", physical_gpu_uuid=None, target_run_path=None, CUDA_mapping=None
        ),
        blockers=[
            "Choose actual target resources/UUID and unique run directory; detect any newer continuation before import",
            "Prepare same logical run record/budget and final handoff through public importer, never train as a new allocation",
            "Legacy candidate stale SHA stays UNAVAILABLE; preserve record, do not substitute final SHA",
            "12x128x.01=15.36s cannot alone certify natural 23s timeout; lifecycle evidence separately required",
        ],
        execution_resume="fresh_reset: retain optimizer/RNG/course/ended statistics; discard unfinished physical episodes with boundary record",
        import_performed=False,
        worker_started=False,
        new_physics=0,
        new_PPO_updates=0,
    )
    write_json(output / "receipt.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(prepare(args.source_run, args.config, args.data_root, args.output)["resources"]["status"])


if __name__ == "__main__":
    main()
