"""Content-bound BC Actor admission into a new Motion PPO grant; no state conversion."""

from .io import file_sha256, json_digest
from .runtime_paths import RuntimePaths

VERSION = "motion_wide_BC_actor_new_grant_v1"


def read_admission(job):
    paths = RuntimePaths.from_assets(job.assets)
    reference = job.assets.get("wide_actor_admission")
    if reference != job.assets.get("branch_initialization"):
        raise ValueError("Wide Actor and course admission must be the same record")
    if not reference or file_sha256(paths.resolve(reference["path"])) != reference["sha256"]:
        raise ValueError("Wide Actor admission SHA differs")
    record = paths.read(reference["path"])
    if record.get("version") != VERSION or record.get("branch") != "motion":
        raise ValueError("Unknown wide Actor admission")
    for key in ("source", "output_contract", "authorization", "settlement", "allocation"):
        ref = record[key]
        if file_sha256(paths.resolve(ref["path"])) != ref["sha256"]:
            raise ValueError("Wide Actor dependency SHA differs: " + key)
    prior_updates = record.get("prior_smoke_updates", 0)
    if prior_updates:
        if type(prior_updates) is not int or prior_updates < 0:
            raise ValueError("Invalid prior smoke consumption")
        evidence = {}
        for key in ("state", "budget"):
            ref = record["prior_smoke"][key]
            if file_sha256(paths.resolve(ref["path"])) != ref["sha256"]:
                raise ValueError("Prior smoke settlement identity differs")
            evidence[key] = paths.read(ref["path"])
        if (
            evidence["state"]["status"] not in ("COMPLETED", "STOPPED_BOUNDED")
            or evidence["state"]["updates_used"] != prior_updates
            or sum(evidence["budget"]["used"].values()) != prior_updates
            or any(a["status"] == "RUNNING" for a in evidence["budget"]["attempts"].values())
        ):
            raise ValueError("Prior smoke must be settled before formal grant use")
    grant = paths.read(record["authorization"]["path"])
    settled = paths.read(record["settlement"]["path"])
    if (
        type(grant.get("new_online_updates")) is not int
        or grant["new_online_updates"] <= 0
        or grant.get("wall_s") is not None
        or grant.get("old_remaining_retired") != settled["retired_remaining_updates"]
        or settled["historical_online_updates"] + prior_updates
        != job.assets["historical_online_updates"]
        or job.assets["original_online_limit"]
        != settled["historical_online_updates"] + grant["new_online_updates"]
        or job.assets.get("new_independent_authorization") is not True
        or record["target_course_sha256"] != json_digest(job.assets["branch_course"])
        or record["output_contract"]["sha256"] != job.assets["output_contract_sha256"]
        or (
            job.initialization.mode == "actor_only"
            and record["source"]["sha256"] != job.initialization.sha256
        )
        or record["source"]["sha256"] != job.assets["source_policy_retention"]["source_sha256"]
        or record["branch_updates"] != 0
        or job.action.version != "motion_multiheight020_support_center_050_80_v1"
    ):
        raise ValueError("Wide Actor source, grant, course or contract differs")
    return record


def require_allocation(job):
    record = read_admission(job)
    paths = RuntimePaths.from_assets(job.assets)
    prior = record.get("prior_failed_zero")
    if prior:
        if file_sha256(paths.resolve(prior["path"])) != prior["sha256"]:
            raise ValueError("Failed-attempt settlement identity differs")
        for attempt in paths.read(prior["path"])["attempts"]:
            for key in ("state", "budget"):
                ref = attempt[key]
                if file_sha256(paths.resolve(ref["path"])) != ref["sha256"]:
                    raise ValueError("Failed-attempt evidence changed")
            state = paths.read(attempt["state"]["path"])
            budget = paths.read(attempt["budget"]["path"])
            if state["status"] != "FAILED" or state["updates_used"] or any(budget["used"].values()):
                raise ValueError("Only zero-consumption failed allocation may be retried")
    plan = paths.read(record["allocation"]["path"])
    entries = plan["entries"]
    if (
        plan["authorization_sha256"] != record["authorization"]["sha256"]
        or len({r["run_id"] for r in entries}) != len(entries)
        or any(type(r[k]) is not int or r[k] < 0 for r in entries for k in ("smoke", "formal"))
        or sum(r["smoke"] + r["formal"] for r in entries)
        != paths.read(record["authorization"]["path"])["new_online_updates"]
    ):
        raise ValueError("Wide Actor grant allocation differs")
    matches = [r for r in entries if r["run_id"] == job.name]
    if len(matches) != 1:
        raise ValueError("No unique wide Actor allocation for this run")
    entry = matches[0]
    if (entry["smoke"], entry["formal"]) != (job.limits.smoke_updates, job.limits.formal_updates):
        raise ValueError("Run limits differ from immutable allocation")
    return entry


def validate_payload(payload, job, contract):
    record = read_admission(job)
    if (
        payload.get("inference_spec", {}).get("kind") != "tanh"
        or payload.get("interface_version") != job.model.interface
        or payload.get("action_version") != job.action.version
        or json_digest(payload.get("output_contract")) != json_digest(contract)
        or payload.get("execution_timing") != job.assets["delayed_execution"]
    ):
        raise ValueError("BC Actor inference, physical output or timing differs")
    retention = job.assets["source_policy_retention"]
    if (
        retention.get("version") != "motion48_per_command_nominal_source_KL_v4"
        or retention.get("physical_output_scales") != [0.5, 0.5, 80.0, 0.5, 0.5, 80.0]
        or retention.get("output_contract_sha256") != record["output_contract"]["sha256"]
    ):
        raise ValueError("Wide Actor protection requires the same frozen output contract")
    return record
