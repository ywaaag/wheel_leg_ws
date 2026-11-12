"""集中、轻量的模型引用索引。

按SHA与完整路径共同识别模型：同一权重的不同路径各自保留一行，
因此不会因为后来登记同SHA的另一个路径而丢掉仍在使用的路径保护。
数据只维护在一处，不写入每个checkpoint、回执或报告。
"""

from __future__ import annotations

import fcntl
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .io import file_sha256, read_json, write_json

ROLES = (
    "initialization",
    "teacher",
    "resume",
    "distillation",
    "deployment",
    "best",
    "selected",
    "accepted",
    "source",
    "control",
    "ancestor",
    "fixed",
    "training_candidate",
)

# 有真实消费者的角色：轮换与删除必须保护这些行。
CONSUMER_ROLES = (
    "fixed",
    "initialization",
    "teacher",
    "resume",
    "distillation",
    "deployment",
    "selected",
    "accepted",
    "source",
    "control",
    "training_candidate",
)


@dataclass(frozen=True)
class ModelReference:
    """一条模型引用。roles是消费方式，不表示能力已通过。"""

    sha256: str
    path: str
    roles: tuple[str, ...]
    run: str | None = None
    parent_sha256: str | None = None
    reason: str | None = None
    kind: str = "checkpoint"
    owner: str | None = None

    def __post_init__(self) -> None:
        unknown = [role for role in self.roles if role not in ROLES]
        if unknown:
            raise ValueError(f"未知模型角色: {unknown}；允许: {list(ROLES)}")
        if len(self.sha256) != 64:
            raise ValueError("模型引用需要64位SHA256")

    @property
    def key(self) -> str:
        """身份 = SHA + 完整路径；路径不同即不同条目。"""
        return f"{self.sha256}\t{self.path}"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["roles"] = sorted(set(self.roles))
        data["key"] = self.key
        return data


class ModelIndex:
    """单文件JSON索引；每次变更在独占锁内完成。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        data = read_json(self.path, None)
        if data is None:
            return {"version": 3, "models": {}}
        if data.get("version") not in (2, 3) or not isinstance(data.get("models"), dict):
            raise ValueError("模型引用索引损坏或版本未知")
        data["version"] = 3
        return data

    def _change(self, operation):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(".lock")
        with lock.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            data = self._load()
            result = operation(data)
            write_json(self.path, data)
            return result

    @staticmethod
    def _merge(data, reference):
        if data.get("deletions", {}).get(reference.key, {}).get("status") in ("intent", "deleted"):
            raise ValueError("模型已登记删除，不能取得新引用")
        entry = data["models"].setdefault(reference.key, reference.as_dict())
        if reference.owner is not None:
            if entry.get("owner") not in (None, reference.owner):
                raise ValueError("模型归属冲突")
            entry["owner"] = reference.owner
        consumers = entry.setdefault(
            "consumers", [{"consumer": entry.get("run"), "role": r} for r in entry.get("roles", [])]
        )
        for role in reference.roles:
            consumer = {"consumer": reference.run, "role": role}
            if consumer not in consumers:
                consumers.append(consumer)
        entry["roles"] = sorted(set(entry.get("roles", [])) | set(reference.roles))
        return entry

    def register_output(self, reference, *, consumer):
        def change(data):
            row = self._merge(data, reference)
            row["dependencies"] = sorted(
                key
                for key, source in data["models"].items()
                if key != reference.key
                and any(
                    c.get("consumer") == consumer
                    and c["role"]
                    in (
                        "initialization",
                        "teacher",
                        "resume",
                        "distillation",
                        "control",
                        "deployment",
                    )
                    for c in source.get("consumers", [])
                )
            )
            return row

        return self._change(change)

    @staticmethod
    def _protected_keys(data, excluding=()):
        keys = {
            key
            for key, row in data["models"].items()
            if set(row.get("roles", [])) & (set(CONSUMER_ROLES) - set(excluding))
        }
        # Existing outputs retain real source dependencies even after a run role is released.
        pending = [
            dependency
            for row in data["models"].values()
            if row.get("dependencies") and Path(row["path"]).is_file()
            for dependency in row["dependencies"]
        ]
        while pending:
            key = pending.pop()
            if key not in keys:
                keys.add(key)
                pending.extend(data["models"].get(key, {}).get("dependencies", []))
        return keys

    def candidate_index(self, run):
        return self._load().get("candidates", {}).get(run)

    def publish_candidates(self, run, candidates):
        """一次提交榜单、全部候选引用及generation；可重复补发布。"""

        def change(data):
            old = data.setdefault("candidates", {}).get(run)
            if old == candidates:
                return old
            if old and old["comparison_sha256"] != candidates["comparison_sha256"]:
                raise ValueError("同run运动候选比较域不一致")
            if (
                old
                and old["last_review"]["through_update"]
                == candidates["last_review"]["through_update"]
                and old != candidates
            ):
                raise ValueError("同一候选窗口提交冲突")
            if (
                old
                and old["last_review"]["through_update"]
                > candidates["last_review"]["through_update"]
            ):
                return old
            for row in data["models"].values():
                consumers = row.setdefault(
                    "consumers",
                    [{"consumer": row.get("run"), "role": role} for role in row.get("roles", [])],
                )
                row["consumers"] = [
                    c for c in consumers if c != {"consumer": run, "role": "training_candidate"}
                ]
                row["roles"] = sorted({c["role"] for c in row["consumers"]})
            for pool in candidates["candidates_by_objective"].values():
                for row in pool.values():
                    if file_sha256(row["checkpoint"]) != row["sha256"]:
                        raise ValueError("候选文件身份不一致")
                    self._merge(
                        data,
                        ModelReference(
                            row["sha256"], row["checkpoint"], ("training_candidate",), run
                        ),
                    )
            data["candidates"][run] = candidates
            data["generation"] = int(data.get("generation", 0)) + 1
            return candidates

        return self._change(change)

    def release(self, *, consumer, role):
        def change(data):
            for row in data["models"].values():
                previous = row.get(
                    "consumers",
                    [{"consumer": row.get("run"), "role": r} for r in row.get("roles", [])],
                )
                row["consumers"] = [
                    c for c in previous if c != {"consumer": consumer, "role": role}
                ]
                row["roles"] = sorted({c["role"] for c in row["consumers"]})

        self._change(change)

    def delete_owned(self, path, sha256, *, run_dir):
        """引用登记与删除共用锁；意图先落盘，崩溃后也拒绝新消费者。"""
        path, run_dir = Path(path), Path(run_dir)

        def change(data):
            key = f"{sha256}\t{path.resolve()}"
            row = data["models"].get(key, {})
            if row.get("owner") not in (None, str(run_dir.resolve())):
                raise ValueError("删除文件归属冲突")
            if key in self._protected_keys(data):
                return {"deleted": False, "reason": "referenced"}
            if path.is_symlink() or path.resolve().parent != (run_dir / "checkpoints").resolve():
                raise ValueError("删除文件不属于本run")
            from .retention import read_inventory

            if not any(
                r.get("sha256") == sha256 and Path(r["checkpoint"]).resolve() == path.resolve()
                for r in read_inventory(run_dir)
            ):
                raise ValueError("删除文件未登记")
            if file_sha256(path) != sha256:
                raise ValueError("删除文件身份不一致")
            record = data.setdefault("deletions", {}).setdefault(key, {})
            record.update(status="intent", owner=str(run_dir.resolve()))
            write_json(self.path, data)
            stat = path.stat()
            path.unlink()
            record.update(
                status="deleted",
                logical_bytes=stat.st_size,
                released_bytes=stat.st_blocks * 512 if stat.st_nlink == 1 else 0,
            )
            return {"deleted": True, **record}

        return self._change(change)

    def register(self, reference: ModelReference) -> dict[str, Any]:
        """登记或合并一条引用；同SHA不同路径各自保留。"""

        def change(data):
            models = data["models"]
            # 兼容旧版按SHA单独索引的条目：迁移到(SHA,路径)键。
            for stale_key, row in list(models.items()):
                if "\t" not in stale_key:
                    moved = ModelReference(
                        sha256=row["sha256"],
                        path=str(Path(row["path"]).resolve()),
                        roles=tuple(row.get("roles") or ()),
                        run=row.get("run"),
                        parent_sha256=row.get("parent_sha256"),
                        reason=row.get("reason"),
                    )
                    models.pop(stale_key)
                    models.setdefault(moved.key, moved.as_dict())
            return self._merge(data, reference)

        return self._change(change)

    def register_many(self, references):
        """Atomically retain a verified dependency closure in the same index."""
        references = tuple(references)

        def change(data):
            return [self._merge(data, reference) for reference in references]

        return self._change(change)

    def entries(self) -> dict[str, dict[str, Any]]:
        return self._load()["models"]

    def rows(self) -> list[dict[str, Any]]:
        return sorted(self.entries().values(), key=lambda row: (row["path"], row["sha256"]))

    def referenced_paths(self) -> set[str]:
        """被任何角色消费的模型路径集合，供保存策略保护。"""
        return {str(Path(row["path"]).resolve()) for row in self.entries().values()}

    def consumer_paths(self, *, excluding: tuple[str, ...] = ()) -> set[str]:
        """只统计有真实消费者的角色（普通历史列举不算永久消费者）。"""
        data = self._load()
        return {
            str(Path(data["models"][key]["path"]).resolve())
            for key in self._protected_keys(data, excluding)
            if key in data["models"]
        }

    def ancestors(self, sha256: str, path: str | None = None) -> list[str]:
        """沿parent_sha256回溯实际祖先，遇到缺失即停止。"""
        entries = self.entries()
        chain: list[str] = []
        if path is not None:
            current = next(
                (
                    row
                    for row in entries.values()
                    if row["sha256"] == sha256
                    and str(Path(row["path"]).resolve()) == str(Path(path).resolve())
                ),
                {},
            ).get("parent_sha256")
        else:
            current = next(
                (row.get("parent_sha256") for row in entries.values() if row["sha256"] == sha256),
                None,
            )
        while current and current not in chain:
            chain.append(current)
            current = next(
                (row.get("parent_sha256") for row in entries.values() if row["sha256"] == current),
                None,
            )
        return chain


def register_model(
    index_path: str | Path,
    path: str | Path,
    *,
    roles: tuple[str, ...],
    run: str | None = None,
    parent: str | Path | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """登记现有模型；核对文件真实SHA，不复制权重。"""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"模型文件不存在: {file}")
    sha = file_sha256(file)
    parent_sha = file_sha256(parent) if parent else None
    return ModelIndex(index_path).register(
        ModelReference(
            sha256=sha,
            path=str(file.resolve()),
            roles=roles,
            run=run,
            parent_sha256=parent_sha,
            reason=reason,
        )
    )


def initialize_fixed(index, manifest, *, paths=None):
    """显式用户固定资产；只在首次登记时核对文件，不扫描历史目录。"""
    for row in read_json(manifest)["fixed_models"]:
        ref = ModelReference(
            row["sha256"],
            str(Path(paths.resolve(row["path"]) if paths is not None else row["path"]).resolve()),
            ("fixed",),
            row["consumer"],
        )
        existing = index.entries().get(ref.key)
        if existing and {"consumer": ref.run, "role": "fixed"} in existing.get("consumers", []):
            continue
        if file_sha256(ref.path) != ref.sha256:
            raise ValueError("用户固定模型身份不一致")
        index.register(ref)


def register_dependencies(index, *, consumer, references):
    """正式加载边界一次登记独立依赖；每个角色保留独立消费者。"""
    for name, ref in references.items():
        if file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("依赖SHA不同: " + name)
        index.register(
            ModelReference(
                ref["sha256"],
                str(Path(ref["path"]).resolve()),
                ("teacher",)
                if name in ("source_anchor_identity", "old_Recovery_action_prior")
                else ("source",),
                consumer,
                kind=name,
            )
        )
