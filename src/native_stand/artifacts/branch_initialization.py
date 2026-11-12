"""从已选完整模型创建新分支；不放宽同 run/最终停止点恢复规则。"""

from __future__ import annotations

from .io import file_sha256, json_digest, read_json
from .runtime_paths import RuntimePaths

VERSION = "selected_full_training_branch_v1"
ALLOCATED = "selected_full_bounded_branch_v2"
ALLOCATED_TRAINING = "selected_full_allocated_branch_v3"


def read_branch(assets):
    reference = assets.get("branch_initialization")
    if reference is None:
        return None
    paths = RuntimePaths.from_assets(assets)
    path = paths.resolve(reference["path"])
    if file_sha256(path) != reference["sha256"]:
        raise ValueError("分支初始化契约SHA不同")
    record = read_json(path)
    if record.get("version") not in (VERSION, ALLOCATED, ALLOCATED_TRAINING) or record.get(
        "branch"
    ) not in (
        "motion",
        "spin",
    ):
        raise ValueError("未知完整分支初始化契约")
    if record.get("branch_updates") != 0 or (
        record["version"] == VERSION and record.get("budget_status") != "UNALLOCATED"
    ):
        raise ValueError("准备阶段不能创建或预留分支额度")
    for key in ("selection", "settlement"):
        ref = record[key]
        if file_sha256(paths.resolve(ref["path"])) != ref["sha256"]:
            raise ValueError("分支来源证据SHA不同: " + key)
    selection = paths.view(read_json(paths.resolve(record["selection"]["path"])))
    settlement = paths.view(read_json(paths.resolve(record["settlement"]["path"])))
    selected, final = settlement["selected_checkpoint"], settlement["final_checkpoint"]
    if (
        selection["branch"] != record["branch"]
        or selection["source"]["sha256"] != selected["sha256"]
        or settlement["status"] != "STOPPED_BOUNDED"
        or settlement["failure"] is not None
        or settlement["new_budget_created"] is not False
        or settlement["rolled_back_updates_are_not_refunded"] is not True
        or record["source_model_online_updates"] != selected["model_online_updates"]
        or record["authorization_spent_updates"] != settlement["total_online_updates_consumed"]
        or record["shared_remaining_updates"] != settlement["remaining_updates"]
        or record["authorization_spent_updates"] + record["shared_remaining_updates"]
        != settlement["original_limit"]
    ):
        raise ValueError("分支选择、模型进度与停止结算不一致")
    for ref in (selected, final):
        if file_sha256(ref["path"]) != ref["sha256"] or ref["full_checkpoint"] is not True:
            raise ValueError("选中模型或最终完整停止点身份不同")
    for key, sha_key in (
        ("source_center_binding", None),
        ("output_contract", "output_contract_sha256"),
    ):
        ref = (
            selection[key]
            if sha_key is None
            else dict(path=selection[key], sha256=selection[sha_key])
        )
        if file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("分支中心或输出合同身份不同")
    binding = read_json(selection["source_center_binding"]["path"])
    if (
        binding["policy_sha256"] != selected["sha256"]
        or binding["large_contract_sha256"] != selection["output_contract_sha256"]
    ):
        raise ValueError("中心绑定不是本次选中模型/输出合同")
    return {**record, "selected": selected, "selection_record": selection}


def validate_source(job, loaded):
    record = read_branch(job.assets)
    if record is None:
        return None
    source = loaded.payload
    if (
        job.initialization.mode != "full"
        or loaded.sha256 != record["selected"]["sha256"]
        or source.get("online_updates_including_smoke") != record["source_model_online_updates"]
        or source.get("update") != record["selected"]["model_local_update"]
        or job.assets["historical_online_updates"]
        != record["authorization_spent_updates"] + record.get("prior_smoke_updates", 0)
        or job.assets.get("new_independent_authorization")
        is not bool(record.get("new_authorization"))
        or source.get("warmup_updates_done") != job.ppo.critic_warmup_updates
    ):
        raise ValueError("选中完整模型不符合新分支初始化声明")
    extension = job.assets.get("output_contract_extension")
    if extension is None:
        if (
            job.assets["output_contract_sha256"]
            != record["selection_record"]["output_contract_sha256"]
            or job.assets.get("support_center_policy_binding")
            != record["selection_record"]["source_center_binding"]
            or json_digest(source.get("output_contract"))
            != json_digest(read_json(job.assets["output_contract"]))
        ):
            raise ValueError("选中完整模型不符合新分支初始化声明：输出合同与来源不一致")
    elif record["version"] not in (ALLOCATED, ALLOCATED_TRAINING):
        raise ValueError("准备配置不能隐式更换输出合同")
    for name in ("actor", "critic", "optimizer", "std", "training_rng", "source_anchor_actor"):
        if name not in source:
            raise ValueError("分支来源缺少完整训练状态: " + name)
    if set(source["optimizer"]) != {"version", "actor", "critic"}:
        raise ValueError("分支来源必须包含两个Adam的完整状态")
    return record


def require_allocated(job):
    """One immutable allocation plan; actual consumption remains in run Budget."""
    record = read_branch(job.assets)
    if record is None:
        return None
    training = (
        record["version"] == ALLOCATED_TRAINING and record.get("budget_status") == "MOTION_TRAINING"
    )
    if not training and (
        record["version"] != ALLOCATED or record.get("budget_status") != "BOUNDED_SMOKE"
    ):
        raise ValueError("Motion/Spin共同余额尚未分配；本配方仅供CPU准备，不可提交训练")
    paths = RuntimePaths.from_assets(job.assets)
    ref = record["allocation"]
    if file_sha256(paths.resolve(ref["path"])) != ref["sha256"]:
        raise ValueError("共同额度分配身份改变")
    plan = paths.read(ref["path"])
    entries = plan["allocations"]
    charged = 0
    for previous in plan.get("prior_consumed_runs", []):
        for key in ("state", "budget"):
            ref = previous[key]
            if file_sha256(paths.resolve(ref["path"])) != ref["sha256"]:
                raise ValueError("前置smoke实际消耗身份改变")
        state = paths.read(previous["state"]["path"])
        budget = paths.read(previous["budget"]["path"])
        used = sum(budget["used"].values())
        if (
            state["status"] not in ("FAILED", "STOPPED_BOUNDED", "COMPLETED")
            or any(a["status"] == "RUNNING" for a in budget["attempts"].values())
            or used != state["updates_used"]
        ):
            raise ValueError("前置smoke尚未准确结算")
        charged += used
    if charged != record.get("prior_smoke_updates", 0):
        raise ValueError("前置smoke消耗不能退款或漏计")
    available = record["shared_remaining_updates"] - charged
    grant_ref = record.get("new_authorization")
    if grant_ref:
        if not training or file_sha256(paths.resolve(grant_ref["path"])) != grant_ref["sha256"]:
            raise ValueError("独立新授权身份不同")
        grant = paths.read(grant_ref["path"])
        if (
            grant.get("status") != "APPROVED"
            or grant.get("run_id") != job.name
            or grant.get("branch") != record["branch"]
            or type(grant.get("updates")) is not int
            or grant["updates"] <= 0
            or grant.get("preserved_consumption") != record["authorization_spent_updates"] + charged
            or job.assets.get("new_independent_authorization") is not True
        ):
            raise ValueError("独立新授权、来源或历史消耗不一致")
        available = grant["updates"]
    for retired in plan.get("retired_unstarted_runs", []):
        for key in ("state", "budget"):
            ref = retired[key]
            if file_sha256(paths.resolve(ref["path"])) != ref["sha256"]:
                raise ValueError("未启动失败额度结算身份改变")
        state = paths.read(retired["state"]["path"])
        budget = paths.read(retired["budget"]["path"])
        if (
            state["status"] != "FAILED"
            or state["updates_used"] != 0
            or state["samples_used"] != 0
            or any(
                a.get("status") == "RUNNING" or a.get("updates", 0) != 0
                for a in budget["attempts"].values()
            )
            or any(budget["used"].values())
            or any(e["run_id"] == state["run_id"] for e in entries)
        ):
            raise ValueError("只能结转零已完成更新且全部attempt已结算的失败分配")
    if (
        plan.get("version") != "selected_branch_shared_allocations_v1"
        or plan["settlement_sha256"] != record["settlement"]["sha256"]
        or plan["already_consumed_updates"] != record["authorization_spent_updates"] + charged
        or plan["remaining_before_allocation"] != available
        or len({e["run_id"] for e in entries}) != len(entries)
        or any(
            type(e["smoke_updates"]) is not int
            or not 0 <= e["smoke_updates"] <= 128
            or type(e["formal_updates"]) is not int
            or e["formal_updates"] < 0
            or e["smoke_updates"] + e["formal_updates"] <= 0
            or (not training and e["formal_updates"] != 0)
            for e in entries
        )
        or sum(e["smoke_updates"] + e["formal_updates"] for e in entries) > available
    ):
        raise ValueError("共同smoke额度分配不合法")
    matches = [e for e in entries if e["run_id"] == job.name and e["branch"] == record["branch"]]
    if len(matches) != 1:
        raise ValueError("配方没有唯一smoke分配")
    entry = matches[0]
    if (
        job.limits.smoke_updates != entry["smoke_updates"]
        or job.limits.formal_updates != entry["formal_updates"]
        or job.assets.get("preparation_only") is not False
        or entry["source_sha256"] != job.initialization.sha256
        or entry["output_contract_sha256"] != job.assets["output_contract_sha256"]
        or entry["execution_sha256"] != job.assets["branch_course"]["confirmation_execution_sha256"]
    ):
        raise ValueError("正式配方与有界分配不一致")
    return entry
