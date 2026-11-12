"""显式跨目录恢复边界。规划快照不能授权恢复活动 run 的旧保存点。"""

from __future__ import annotations

from pathlib import Path

from .checkpoint import CheckpointError, LoadedSource, load_source, save_full
from .io import file_sha256, json_digest, read_json

VERSION = "checkpoint_directory_handoff_v1"


def import_checkpoint(manifest_path, *, destination, mapping, expected_run_id):
    """一次性导入完整公共状态；旧文件只读，预算与未消费状态完整保留。"""
    manifest = read_json(manifest_path)
    if manifest.get("version") != VERSION or manifest.get("run_id") != expected_run_id:
        raise CheckpointError("handoff版本或逻辑run身份不同")
    if manifest.get("owner_terminated") is not True:
        raise CheckpointError("原owner未终止，不允许并行消耗同一预算")
    if manifest.get("model_type") == "bc_full":
        return _import_bc(
            manifest,
            manifest_path,
            destination=destination,
            mapping=mapping,
            expected_run_id=expected_run_id,
        )
    budget = manifest["budget"]
    if not {"updates_used", "remaining", "limits"}.issubset(budget):
        raise CheckpointError("handoff预算字段缺失")
    if (
        budget["updates_used"] < 0
        or budget["remaining"] != sum(budget["limits"].values()) - budget["updates_used"]
        or budget["remaining"] < 0
    ):
        raise CheckpointError("handoff预算结算不一致")
    source_ref = manifest["checkpoint"]
    source_path = Path(mapping[source_ref["path"]]).resolve()
    if file_sha256(source_path) != source_ref["sha256"]:
        raise CheckpointError("handoff来源SHA不一致")
    for ref in manifest["dependencies"]:
        if file_sha256(mapping[ref["path"]]) != ref["sha256"]:
            raise CheckpointError("handoff必要依赖身份不同: " + ref["path"])
    source = load_source(source_path)
    state = source.payload
    declared = {(r["path"], r["sha256"]) for r in manifest["dependencies"]}
    if any(
        (ref["path"], ref["sha256"]) not in declared
        for ref in state.get("required_dependencies", {}).values()
    ):
        raise CheckpointError("handoff缺少必要训练依赖")
    legacy = manifest.get("source_format")
    contract_sha = state.get("config_contract_sha256")
    if legacy is not None:
        if legacy not in ("legacy_motion_fullchain_v4", "legacy_recovery_physical_v2"):
            raise CheckpointError("未知legacy恢复格式")
        record_ref = manifest["source_run_record"]
        record_path = Path(mapping[record_ref["path"]])
        if file_sha256(record_path) != record_ref["sha256"]:
            raise CheckpointError("legacy运行记录SHA不同")
        terminal_ref = manifest["source_terminal_state"]
        terminal_path = Path(mapping[terminal_ref["path"]])
        if file_sha256(terminal_path) != terminal_ref["sha256"]:
            raise CheckpointError("legacy终态结算SHA不同")
        terminal = read_json(terminal_path)
        if (
            terminal["status"] not in ("COMPLETED", "STOPPED_BOUNDED")
            or terminal["latest_model"]["sha256"] != source_ref["sha256"]
            or terminal["updates_used"] != manifest["budget"]["updates_used"]
        ):
            raise CheckpointError("legacy原owner未在该完整保存点结算")
        record = read_json(record_path)
        if (
            record["run_id"] != manifest["legacy_run_id"]
            or json_digest(record["source"]) != manifest["code_identity_sha256"]
        ):
            raise CheckpointError("legacy运行/代码身份不同")
        config = record["config"]
        required = [
            ref
            for key, ref in config["assets"].items()
            if key
            in (
                "recovery_initial_actor",
                "recovery_training_contract",
                "recovery_isaac_runtime",
                "qualified_physical_BC_anchor",
                "old_Recovery_action_prior",
            )
        ]
        retention = config["assets"].get("source_policy_retention")
        if retention:
            required += [retention[k] for k in ("bank", "bank_metadata")]
            if not any(
                ref["sha256"] == retention["source_sha256"] for ref in manifest["dependencies"]
            ):
                raise CheckpointError("legacy KL Teacher依赖缺失")
        if any((ref["path"], ref["sha256"]) not in declared for ref in required):
            raise CheckpointError("legacy必要依赖缺失")
        contract_sha = json_digest(
            {
                k: config[k]
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
        )
        if legacy == "legacy_motion_fullchain_v4":
            if (
                state.get("critic_privileged_version")
                != config["model"]["critic_privileged_version"]
                or state.get("action_version") != config["action"]["version"]
            ):
                raise CheckpointError("legacy Motion合同不同")
        elif (
            state.get("version") != "recovery_cubic_physical_Gaussian_v2"
            or state.get("runtime_contract_sha256")
            != config["assets"]["recovery_isaac_runtime"]["sha256"]
        ):
            raise CheckpointError("legacy Recovery合同不同")
    elif state.get("run_identity") != manifest["original_run_identity"]:
        raise CheckpointError("handoff原run身份不同")
    counter_key = "new_run_updates" if "motion_course_state" in state else "updates"
    if state["counters"][counter_key] != manifest["budget"]["updates_used"]:
        raise CheckpointError("checkpoint与结算用量不同；不得退回未保存的已用额度")
    if contract_sha != manifest["config_contract_sha256"]:
        raise CheckpointError("handoff配置/合同不同")
    target = Path(destination).resolve()
    if target.exists():
        raise CheckpointError("导入目标已存在；禁止覆盖恢复点")
    payload = dict(state)
    payload["config_contract_sha256"] = contract_sha
    candidate = payload.get("training_candidates")
    if candidate is not None:
        import copy

        candidate = copy.deepcopy(candidate)
        mapped_rows = set()

        def mapped(row):
            if row is None or id(row) in mapped_rows:
                return
            old = row["checkpoint"]
            if old not in mapping or file_sha256(mapping[old]) != row["sha256"]:
                raise CheckpointError("handoff候选映射或身份缺失")
            row["checkpoint"] = str(Path(mapping[old]).resolve())
            # best/last_review may alias a pool row after torch serialization.
            # Do not attempt to remap its already relocated path a second time.
            mapped_rows.add(id(row))

        for pool in candidate.get("candidates_by_objective", {}).values():
            for row in pool.values():
                mapped(row)
        for field in ("best_candidate", "best_standing_candidate"):
            mapped(candidate.get(field))
        if candidate.get("last_review"):
            mapped(candidate["last_review"].get("checkpoint"))
        payload["training_candidates"] = candidate
    if legacy == "legacy_recovery_physical_v2":
        warmup = config["ppo"]["critic_warmup_updates"]
        if not state["actor_update_enabled"] or state["updates"] < warmup:
            raise CheckpointError("legacy Recovery warmup未完成；需要显式未完成阶段合同")
        payload["warmup_updates_done"] = warmup
    payload.update(
        logical_run_id=expected_run_id,
        run_identity=str(target.parent.parent),
        handoff={
            "manifest_sha256": file_sha256(manifest_path),
            "source_path": str(source.path),
            "source_sha256": source.sha256,
            "budget": manifest["budget"],
            "parent_settlement": manifest["parent_settlement"],
        },
    )
    save_full(target, payload)
    loaded = load_source(target)
    if json_digest(loaded.payload["handoff"]) != json_digest(payload["handoff"]):
        raise CheckpointError("导入后身份读回不同")
    return loaded


def validate_resume_identity(source: LoadedSource, run_dir):
    """普通恢复仍要求物理目录一致；导入后的逻辑身份也必须匹配运行记录。"""
    run_dir = Path(run_dir).resolve()
    if source.payload.get("run_identity") != str(run_dir):
        raise CheckpointError("同run恢复身份不同或缺失")
    logical = source.payload.get("logical_run_id")
    if logical is not None:
        record = read_json(run_dir / "run.json")
        if record.get("logical_run_id") != logical:
            raise CheckpointError("同run逻辑身份不同")


def _import_bc(manifest, manifest_path, *, destination, mapping, expected_run_id):
    """Explicit BC full-state import; no PPO counter or optimizer is invented."""
    import os

    import torch

    from native_stand.experiments.supervision import SupervisionConfig, semantic_identity

    from .checkpoint import load_actor_source
    from .io import write_json
    from .runtime_paths import RuntimePaths

    target = Path(destination).resolve()
    run = Path(manifest["target_run"]).resolve()
    if not target.is_relative_to(run):
        raise CheckpointError("BC目标不属于声明的run")
    if manifest["owner_terminated"] is not True:
        raise CheckpointError("BC原owner未结算")
    source_ref = manifest["checkpoint"]
    source = load_actor_source(mapping[source_ref["path"]])
    if source.sha256 != source_ref["sha256"]:
        raise CheckpointError("BC来源SHA不同")
    if target == source.path.resolve() or target.exists():
        raise CheckpointError("BC导入必须写入新的完整恢复点，不能覆盖来源")
    for ref in manifest["dependencies"]:
        if file_sha256(mapping[ref["path"]]) != ref["sha256"]:
            raise CheckpointError("BC依赖SHA不同")
    for name in ("source_run", "source_state", "source_budget"):
        ref = manifest[name]
        if file_sha256(mapping[ref["path"]]) != ref["sha256"]:
            raise CheckpointError("BC结算记录SHA不同")
    original = read_json(mapping[manifest["source_run"]["path"]])
    terminal = read_json(mapping[manifest["source_state"]["path"]])
    book = read_json(mapping[manifest["source_budget"]["path"]])
    target_book = read_json(run / "budget.json")
    record = read_json(run / "run.json")
    state = source.payload
    used = manifest["budget"]["rounds_used"]
    if (
        state.get("artifact_type") != "bc_full"
        or state.get("logical_run_id") != expected_run_id
        or original.get("logical_run_id") != expected_run_id
        or record.get("logical_run_id") != expected_run_id
        or terminal["status"] not in ("STOPPED_BOUNDED", "COMPLETED", "FAILED")
        or any(r["status"] == "RUNNING" for r in book["attempts"].values())
        or state["round"] != used
        or book["used"]["distill"] != used
        or target_book["used"] != book["used"]
        or target_book["limits"] != book["limits"]
        or target_book.get("resources_used", {}) != book.get("resources_used", {})
        or manifest["budget"]["remaining"] != book["limits"]["distill"] - used
    ):
        raise CheckpointError("BC逻辑身份、完整状态或实际结算不一致")
    for key in (
        "student",
        "actor",
        "optimizer",
        "rng",
        "shuffle_rng",
        "split",
        "supervision_identity",
    ):
        if key not in state:
            raise CheckpointError("BC完整状态缺少" + key)
    recipe = SupervisionConfig.from_record(record["config"]["supervision"])
    if semantic_identity(recipe) != state["supervision_identity"]:
        raise CheckpointError("BC目标Teacher/数据/标签语义不同")
    paths = RuntimePaths(files=tuple(mapping.items()))
    flow_path = run / "supervision/supervision_state.json"
    declared = {
        (str(Path(mapping[r["path"]]).resolve()), r["sha256"]) for r in manifest["dependencies"]
    }
    if (str(flow_path.resolve()), file_sha256(flow_path)) not in declared:
        raise CheckpointError("BC阶段状态缺少已核验依赖身份")
    flow = paths.view(read_json(flow_path))
    if flow["logical_run_id"] != expected_run_id:
        raise CheckpointError("BC阶段逻辑身份不同")
    payload = {
        **state,
        "handoff": dict(
            manifest_sha256=file_sha256(manifest_path),
            source_sha256=source.sha256,
            budget=manifest["budget"],
            physical_state_restored=False,
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".import.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    loaded = load_actor_source(target)
    if flow.get("latest") and flow["latest"]["sha256"] == source.sha256:
        flow["latest"] = dict(path=str(target), sha256=loaded.sha256)
    write_json(flow_path, flow)
    view = read_json(run / "state.json")
    view["latest_model"] = dict(checkpoint=str(target), sha256=loaded.sha256)
    write_json(run / "state.json", view)
    return loaded
