"""同接口蒸馏：固定Teacher、按episode划分、BC拟合、有界round。

算法保持简单：Teacher只给标签、不救场；验证集与留出集不回用于补教。
本模块不启动真实蒸馏任务；CPU小数据可以完整验证装配、状态迁移与保存。
"""

from __future__ import annotations

import copy
import dataclasses
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from native_stand.artifacts.io import file_sha256, read_json, write_json
from native_stand.experiments.config import JobConfig

from .splits import split_by_episode


def _atomic_save_latest(path: Path, state: dict[str, Any]) -> str:
    """latest原子替换：先写临时文件再替换；失败保留上一有效checkpoint。"""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    from native_stand.artifacts.io import file_sha256

    return file_sha256(path)


def _rng_snapshot() -> dict:
    """Python / NumPy / Torch 三路RNG状态；恢复时按同样顺序还原。"""
    import random

    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng(snapshot: dict) -> None:
    import random

    import numpy as np
    import torch

    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch"])


class DistillationError(ValueError):
    """蒸馏输入或状态不合法。"""


LABEL_KEYS = ("teacher_sha256", "input_version", "action_version", "action_units", "action_dim")


def check_label_provenance(
    dataset: dict[str, Any], teacher_sha: str, job: JobConfig, label_source: str
) -> dict[str, Any]:
    """预计算标签必须带来源记录；在线标签使用实际Teacher身份。"""
    if label_source != "dataset":
        return {
            "source": "teacher_online",
            "teacher_sha256": teacher_sha,
            "action_dim": job.action.dim,
        }
    meta = dataset.get("label_provenance")
    if not isinstance(meta, dict):
        raise DistillationError(
            "数据集含teacher_actions但没有label_provenance；无法证明标签来自声明的Teacher"
        )
    keys = (
        tuple(k for k in LABEL_KEYS if k != "teacher_sha256") if "teachers" in meta else LABEL_KEYS
    )
    missing = [key for key in keys if key not in meta]
    if missing:
        raise DistillationError(f"标签来源记录缺少字段: {missing}")
    if "teachers" in meta:
        if (
            not meta["teachers"]
            or meta.get("actual_owner_column") != "teacher_owner"
            or not meta.get("teacher_set")
        ):
            raise DistillationError("多Teacher标签缺少实际集合/owner身份")
    elif meta["teacher_sha256"] != teacher_sha:
        raise DistillationError("标签来源Teacher与当前Teacher不一致")
    if int(meta["action_dim"]) != job.action.dim:
        raise DistillationError("标签维数与作业动作维数不一致")
    if meta["action_version"] != job.action.version:
        raise DistillationError("标签动作版本与作业声明不一致")
    if meta.get("action_units") != "policy_output":
        raise DistillationError(
            f"当前BC目标是raw action（policy_output），不接受单位{meta.get('action_units')!r}"
        )
    return dict(meta)


def load_dataset(path: str | Path) -> dict[str, Any]:
    """读取状态数据集：states[N,D]、episodes[N]，可选 teacher_actions。"""
    import numpy as np

    file = Path(path)
    if not file.is_file():
        raise DistillationError(f"数据集不存在: {file}")
    with np.load(file) as payload:
        for key in ("states", "episodes"):
            if key not in payload:
                raise DistillationError(f"数据集缺少{key}")
        states = np.asarray(payload["states"], dtype="float32")
        episodes = [int(value) for value in payload["episodes"]]
        teacher_actions = (
            np.asarray(payload["teacher_actions"], dtype="float32")
            if "teacher_actions" in payload
            else None
        )
        provenance = None
        if "label_provenance" in payload:
            import json as _json

            raw = payload["label_provenance"]
            provenance = _json.loads(str(raw)) if raw.dtype.kind in "US" else dict(raw)
    if len(states) != len(episodes):
        raise DistillationError("states与episodes数量不一致")
    return {
        "states": states,
        "episodes": episodes,
        "teacher_actions": teacher_actions,
        "label_provenance": provenance,
    }


def build_teacher_student(job: JobConfig, teacher_checkpoint: str | Path, *, teacher_hidden=None):
    """Teacher 用自身架构元数据构造；Student 用本作业的 Student 配置。

    两者隐藏层可以不同（大Teacher→小Student）。输入输出语义必须一致：
    Actor 输入列数与动作维数不符时直接报错，不补零、不裁剪。
    """

    from native_stand.artifacts.checkpoint import load_actor_source

    loaded = load_actor_source(teacher_checkpoint)
    payload = loaded.payload
    architecture = payload.get("network_architecture")
    hidden = teacher_hidden or (
        architecture.get("actor_hidden_sizes") if isinstance(architecture, dict) else None
    )
    if hidden is None:
        raise DistillationError(
            "Teacher checkpoint 没有 network_architecture.actor_hidden_sizes；"
            "请显式声明 Teacher 架构，不能用 Student 的 job.model 猜测"
        )
    teacher_job = dataclasses.replace(
        job, model=dataclasses.replace(job.model, actor_hidden_dims=tuple(int(w) for w in hidden))
    )
    from native_stand.learning.inference import build_actor, from_source, spec_for_job

    output = (
        payload.get("output_contract")
        or payload.get("contract")
        or {"action_version": job.action.version, "same_interface_binding": job.model.interface}
    )
    if output.get("action_version", output.get("version")) != job.action.version:
        raise DistillationError("Different Teacher decoder requires an explicit routed recipe")
    teacher = from_source(loaded, spec_for_job(teacher_job, output)).actor
    teacher._loaded_source = loaded
    student = build_actor(spec_for_job(job, output))
    return teacher, student


def teacher_actions(
    job: JobConfig, teacher_checkpoint: str | Path, states, *, teacher_hidden=None
) -> tuple:
    """固定Teacher前向取标签；不更新Teacher。"""
    import torch

    teacher, _ = build_teacher_student(job, teacher_checkpoint, teacher_hidden=teacher_hidden)
    with torch.no_grad():
        actions = teacher({"student_actor": torch.as_tensor(states)}, stochastic_output=False)
    return teacher, actions


def run_distillation(
    job: JobConfig,
    *,
    dataset_path: str | Path,
    teacher_checkpoint: str | Path,
    output_dir: str | Path,
    rounds: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    holdout_fraction: float = 0.2,
    seed: int | None = None,
    teacher_hidden: tuple[int, ...] | None = None,
    resume_state: dict[str, Any] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    on_round_complete: Callable[[dict[str, Any]], None] | None = None,
    on_round_saved: Callable[[dict[str, Any]], None] | None = None,
    prepared: dict | None = None,
) -> dict[str, Any]:
    """执行有界BC训练；返回结果记录并保存候选与配置摘要。"""
    import torch

    if rounds < 1:
        raise DistillationError("rounds必须至少为1")
    dataset_sha = file_sha256(dataset_path)
    teacher_sha = file_sha256(teacher_checkpoint)
    data = load_dataset(dataset_path) if prepared is None else prepared["data"]
    if resume_state is not None:
        # 续训沿用已保存的划分与来源身份，不重新划分。
        from .splits import EpisodeSplit

        record = resume_state.get("split") or {}
        split = EpisodeSplit(
            train=tuple(record.get("train") or ()),
            validation=tuple(record.get("validation") or ()),
            holdout=tuple(record.get("holdout") or ()),
        )
        if resume_state.get("dataset_sha256") != dataset_sha:
            raise DistillationError("续训数据集身份与保存记录不一致")
        if resume_state.get("teacher_sha256") != teacher_sha:
            raise DistillationError("续训Teacher身份与保存记录不一致")
        start_round = int(resume_state.get("round") or 0)
    elif prepared is not None and "split" in prepared:
        split = prepared["split"]
        start_round = 0
    else:
        split = split_by_episode(
            data["episodes"],
            validation_fraction=validation_fraction,
            holdout_fraction=holdout_fraction,
            seed=job.runtime.seed if seed is None else seed,
        )
        start_round = 0
    states = torch.as_tensor(data["states"])
    torch.manual_seed(job.runtime.seed if seed is None else seed)
    if prepared is None:
        teacher, student = build_teacher_student(
            job, teacher_checkpoint, teacher_hidden=teacher_hidden
        )
        if teacher._loaded_source.sha256 != teacher_sha:
            raise DistillationError("Teacher changed during construction")
    else:
        from native_stand.learning.inference import InferenceSpec, build_actor

        teacher = None
        student = build_actor(InferenceSpec.from_record(prepared["student_spec"]))
        if prepared.get("initial_student") is not None:
            student.load_state_dict(prepared["initial_student"], strict=True)
    if resume_state is not None:
        # 恢复顺序：先RNG、再权重与Adam、最后shuffle generator，保证下一轮一致。
        student.load_state_dict(resume_state["student"], strict=True)
        optimizer_state = resume_state["optimizer"]
    if data["teacher_actions"] is not None:
        # 预计算标签：必须与声明Teacher、动作语义、形状一致。
        labels = torch.as_tensor(data["teacher_actions"])
        if tuple(labels.shape) != (len(data["states"]), job.action.dim):
            raise DistillationError("预计算teacher_actions形状与作业动作维数不一致")
        label_source = "dataset"
    else:
        with torch.no_grad():
            labels = teacher({"student_actor": states}, stochastic_output=False)
        label_source = "teacher_online"
    if not bool(torch.isfinite(states).all() and torch.isfinite(labels).all()):
        raise DistillationError("Nonfinite supervised states/labels")
    # 标签来源与形状必须在任何optimizer step之前核对。
    label_meta = check_label_provenance(data, teacher_sha, job, label_source)
    index_of: dict[int, list[int]] = {}
    for position, episode in enumerate(data["episodes"]):
        index_of.setdefault(episode, []).append(position)

    def rows(episodes) -> torch.Tensor:
        flat = [position for episode in episodes for position in index_of[episode]]
        return torch.tensor(flat, dtype=torch.long)

    train_index = rows(split.train)
    validation_index = rows(split.validation)

    device = torch.device(job.assets.get("distillation_device", "cpu"))
    student.to(device)
    states, labels = states.to(device), labels.to(device)
    student.train()
    optimizer = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad], lr=learning_rate
    )
    generator = torch.Generator().manual_seed(job.runtime.seed if seed is None else seed)
    if resume_state is not None:
        # 恢复顺序：RNG → Adam → shuffle generator，保证下一轮随机序列一致。
        optimizer.load_state_dict(optimizer_state)
        restore_rng(resume_state["rng"])
        generator.set_state(resume_state["shuffle_rng"])
    best = {
        "validation_loss": float("inf"),
        "round": 0,
        "state": None,
        "optimizer": None,
        "rng": None,
    }
    history: list[dict[str, float]] = []
    if resume_state is not None:
        # 恢复既有选模结果与此前轮次记录；只在新候选更好时更新best。
        previous_best = resume_state.get("best") or {}
        if previous_best.get("state") is not None:
            best = previous_best
        history = list(resume_state.get("history") or [])
    if not split.validation:
        raise DistillationError("验证集为空，无法选模")
    if not split.holdout:
        raise DistillationError("留出集为空，无法给出留出结论")

    comparison_history = list((resume_state or {}).get("comparison_history", []))
    if prepared is not None and prepared.get("new_comparison_domain"):
        with torch.no_grad():
            incumbent_loss = float(
                torch.nn.functional.mse_loss(
                    student({"student_actor": states[validation_index]}, stochastic_output=False),
                    labels[validation_index],
                )
            )
        comparison_history.append(
            dict(
                previous=prepared["previous_domain"],
                dataset_sha256=dataset_sha,
                split=split.as_record(),
                incumbent_loss=incumbent_loss,
                incumbent_round=start_round,
            )
        )
        best = dict(
            validation_loss=incumbent_loss,
            round=start_round,
            state={k: v.detach().clone() for k, v in student.state_dict().items()},
            optimizer=copy.deepcopy(optimizer.state_dict()),
            rng=_rng_snapshot(),
        )

    def _snapshot(round_value: int) -> dict[str, Any]:
        """完整可恢复快照：Student/Adam/RNG/shuffle/固定split/累计round/best/history。"""
        weights = {key: value.detach().clone() for key, value in student.state_dict().items()}
        return {
            "student": weights,
            "actor": weights,
            "inference_spec": prepared["student_spec"] if prepared is not None else None,
            "artifact_type": "bc_full",
            "logical_run_id": prepared.get("logical_run_id") if prepared is not None else None,
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "round": round_value,
            "validation_loss": history[-1]["validation_loss"] if history else None,
            "split": split.as_record(),
            "seed": int(job.runtime.seed if seed is None else seed),
            "fractions": {"validation": validation_fraction, "holdout": holdout_fraction},
            "rng": _rng_snapshot(),
            "shuffle_rng": generator.get_state(),
            "best": {
                "round": best["round"],
                "validation_loss": best["validation_loss"],
                "state": best["state"],
                "optimizer": best["optimizer"],
                "rng": best["rng"],
            },
            "history": history,
            "teacher_sha256": teacher_sha,
            "dataset_sha256": dataset_sha,
            "labels": label_meta,
            "supervision_identity": prepared.get("identity") if prepared is not None else None,
            "comparison_history": comparison_history,
            "comparison_domain": dict(dataset_sha256=dataset_sha, split=split.as_record()),
        }

    for offset in range(1, rounds + 1):
        if stop_requested is not None and stop_requested():
            break  # 只在轮边界停止；未开始的轮次不计费。
        round_index = start_round + offset
        order = torch.randperm(len(train_index), generator=generator)
        total = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            index = train_index[order[start : start + batch_size]]
            prediction = student({"student_actor": states[index]}, stochastic_output=False)
            loss = torch.nn.functional.mse_loss(prediction, labels[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        with torch.no_grad():
            validation = float(
                torch.nn.functional.mse_loss(
                    student({"student_actor": states[validation_index]}, stochastic_output=False),
                    labels[validation_index],
                )
            )
        history.append(
            {
                "round": round_index,
                "train_loss": total / max(1, batches),
                "validation_loss": validation,
            }
        )
        if validation < best["validation_loss"]:
            best = {
                "validation_loss": validation,
                "round": round_index,
                "state": {
                    key: value.detach().clone() for key, value in student.state_dict().items()
                },
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "rng": _rng_snapshot(),
                "shuffle_rng": generator.get_state(),
            }
        if on_round_complete is not None:
            # 完成回调：本轮优化已完成，先按实际轮次计费。
            on_round_complete(
                {
                    "round": round_index,
                    "optimizer_steps": batches,
                    "train_loss": history[-1]["train_loss"],
                    "validation_loss": validation,
                }
            )
        # 完整快照（含本轮训练后的Student/Adam/best/history）原子保存。
        latest_sha = _atomic_save_latest(
            Path(output_dir) / "distillation_latest.pt", _snapshot(round_index)
        )
        if on_round_saved is not None:
            # 保存回调：发布已核实SHA的latest身份。
            on_round_saved(
                {
                    "round": round_index,
                    "latest_path": str((Path(output_dir) / "distillation_latest.pt").resolve()),
                    "latest_sha256": latest_sha,
                }
            )

    if best["state"] is None:
        raise DistillationError("蒸馏没有得到可用候选")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / "student_actor.pt"
    torch.save(
        {
            "actor": best["state"],
            "inference_spec": prepared["student_spec"] if prepared is not None else None,
            "round": best["round"],
            "validation_loss": best["validation_loss"],
            "teacher_sha256": teacher_sha,
            "dataset_sha256": dataset_sha,
            "interface_version": job.model.interface,
            "action_version": job.action.version,
        },
        candidate,
    )
    # best完整状态：与选模同轮的Student+Adam+RNG（导出/复核用）。
    if best.get("state") is not None:
        _atomic_save_latest(
            directory / "distillation_best.pt",
            {
                "student": best["state"],
                "optimizer": best["optimizer"],
                "round": best["round"],
                "rng": best["rng"],
                "split": split.as_record(),
                "validation_loss": best["validation_loss"],
            },
        )
    record = {
        "version": "distillation_loop_v1",
        "rounds": rounds,
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "rounds_budget": start_round + rounds,
        },
        "completed_rounds": len(history),
        "best_round": best["round"],
        "best_validation_loss": best["validation_loss"],
        "split": split.as_record(),
        "seed": int(job.runtime.seed if seed is None else seed),
        "teacher_hidden": list(teacher_hidden) if teacher_hidden else None,
        "fractions": {"validation": validation_fraction, "holdout": holdout_fraction},
        "teacher": {
            "checkpoint": str(Path(teacher_checkpoint).resolve()),
            "sha256": teacher_sha,
        },
        "dataset": {"path": str(Path(dataset_path).resolve()), "sha256": dataset_sha},
        "candidate": {"path": str(candidate.resolve()), "sha256": file_sha256(candidate)},
        "labels": {**label_meta, "source": label_source},
        "latest_state": {
            "path": str((directory / "distillation_latest.pt").resolve()),
            "round": len(history),
            "sha256": file_sha256(directory / "distillation_latest.pt")
            if (directory / "distillation_latest.pt").is_file()
            else None,
        },
        "best_state": {
            "path": str((directory / "distillation_best.pt").resolve()),
            "round": best["round"],
        },
        "history": history,
        "holdout_used_for_selection": False,
    }
    write_json(directory / "distillation.json", record)
    return record


def resume_distillation(job: JobConfig, *, run_dir: str | Path, rounds: int) -> dict[str, Any]:
    """在同一蒸馏run上继续训练：读取latest完整状态与固定划分。"""
    import torch

    directory = Path(run_dir)
    record = read_json(directory / "distillation.json")
    if not record:
        raise DistillationError(f"找不到蒸馏记录: {directory}")
    state = torch.load(record["latest_state"]["path"], map_location="cpu", weights_only=False)
    hidden = record.get("teacher_hidden")
    params = record.get("hyperparameters") or {}
    # 续训必须沿用记录中的超参，不能用函数默认值。
    for key in ("batch_size", "learning_rate"):
        if key not in params:
            raise DistillationError(f"蒸馏记录缺少超参{key}，无法安全续训")
    return run_distillation(
        job,
        dataset_path=record["dataset"]["path"],
        teacher_checkpoint=record["teacher"]["checkpoint"],
        output_dir=directory,
        rounds=rounds,
        batch_size=int(params["batch_size"]),
        learning_rate=float(params["learning_rate"]),
        validation_fraction=record["fractions"]["validation"],
        holdout_fraction=record["fractions"]["holdout"],
        seed=record["seed"],
        teacher_hidden=tuple(hidden) if hidden else None,
        resume_state=state,
    )


def holdout_loss(
    job: JobConfig, record_path_or_dir: str | Path, *, candidate: str | Path | None = None
) -> dict[str, Any]:
    """用已保存的蒸馏记录评估留出集；不重新划分、不换标签来源。"""
    directory = Path(record_path_or_dir)
    record_path = directory / "distillation.json" if directory.is_dir() else directory
    record = read_json(record_path)
    if not record:
        raise DistillationError(f"找不到蒸馏记录: {record_path}")
    split = record["split"]
    if not split.get("holdout"):
        raise DistillationError("该蒸馏记录没有留出episode，不能生成留出结论")
    dataset_path = record["dataset"]["path"]
    teacher_checkpoint = record["teacher"]["checkpoint"]
    candidate = Path(candidate) if candidate else Path(record["candidate"]["path"])
    # 身份变化必须拒绝，不能在同一份记录上悄悄换数据或模型。
    for name, path, expected in (
        ("dataset", dataset_path, record["dataset"]["sha256"]),
        ("teacher", teacher_checkpoint, record["teacher"]["sha256"]),
        ("candidate", candidate, record["candidate"]["sha256"]),
    ):
        if not Path(path).is_file():
            raise DistillationError(f"{name}文件不存在: {path}")
        if file_sha256(path) != expected:
            raise DistillationError(f"{name}身份与蒸馏记录不一致: {path}")
    job = dataclasses.replace(job, runtime=dataclasses.replace(job.runtime, seed=record["seed"]))
    result = _holdout_loss(
        job,
        candidate,
        dataset_path,
        teacher_checkpoint,
        split,
        label_source=(record.get("labels") or {}).get("source"),
    )
    result["labels"] = record.get("labels")
    return result


def _holdout_loss(
    job: JobConfig,
    candidate: str | Path,
    dataset_path: str | Path,
    teacher_checkpoint: str | Path,
    split_record: dict,
    *,
    label_source: str | None = None,
) -> dict[str, Any]:
    """留出集单独评估；只在报告时使用，不参与选模。"""
    import torch

    data = load_dataset(dataset_path)
    from .splits import EpisodeSplit

    split = EpisodeSplit(
        train=tuple(split_record.get("train") or ()),
        validation=tuple(split_record.get("validation") or ()),
        holdout=tuple(split_record.get("holdout") or ()),
    )
    states = torch.as_tensor(data["states"])
    from native_stand.learning.inference import build_actor, spec_for_job

    student = build_actor(
        spec_for_job(
            job,
            {"action_version": job.action.version, "same_interface_binding": job.model.interface},
        )
    )
    # 标签来源必须与训练一致：用了预计算标签就继续用，不能换成另一条路径。
    if label_source == "dataset":
        if data["teacher_actions"] is None:
            raise DistillationError("训练用预计算标签，但数据集没有teacher_actions")
        labels = torch.as_tensor(data["teacher_actions"])
    else:
        _, labels = teacher_actions(job, teacher_checkpoint, data["states"])
    import torch as _torch

    # 候选是蒸馏产物（Actor权重+来源身份），不是完整恢复状态。
    student.load_state_dict(
        _torch.load(Path(candidate), map_location="cpu", weights_only=False)["actor"], strict=True
    )
    index_of: dict[int, list[int]] = {}
    for position, episode in enumerate(data["episodes"]):
        index_of.setdefault(episode, []).append(position)
    index = torch.tensor(
        [position for episode in split.holdout for position in index_of[episode]], dtype=torch.long
    )
    with torch.no_grad():
        loss = float(
            torch.nn.functional.mse_loss(
                student({"student_actor": states[index]}, stochastic_output=False), labels[index]
            )
        )
    return {
        "holdout_episodes": len(split.holdout),
        "episodes": list(split.holdout),
        "holdout_mse": loss,
    }


def read_record(path: str | Path) -> dict[str, Any] | None:
    return read_json(path)
