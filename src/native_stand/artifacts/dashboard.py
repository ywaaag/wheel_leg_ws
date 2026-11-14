"""Bounded, read-only presentation of existing artifacts; never scores or loads models."""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MAX_BYTES = 4 * 1024 * 1024


def reject_constant(value):
    raise ValueError("非有限JSON值")


def public(value):
    """Do not forward embedded credentials, environments or arbitrary configuration."""
    if isinstance(value, dict):
        return {
            k: public(v)
            for k, v in value.items()
            if not any(
                word in k.lower()
                for word in (
                    "password",
                    "secret",
                    "token",
                    "credential",
                    "environment",
                    "private_key",
                )
            )
        }
    if isinstance(value, list):
        return [public(v) for v in value]
    return value


def compare(left, right):
    """Compare only explicitly identical, complete evidence domains."""
    fields = (
        "layer",
        "task",
        "command",
        "height",
        "terrain",
        "DR",
        "execution_contract",
        "protocol",
    )
    if any(left.get(k) is None or right.get(k) is None for k in fields):
        return "证据不足"
    if any(left[k] != right[k] for k in fields):
        return "不可比较"
    if any(
        type(row.get("ended_episodes")) is not int or row["ended_episodes"] < 1
        for row in (left, right)
    ):
        return "证据不足"
    a, b = left.get("pass_fraction"), right.get("pass_fraction")
    if any(
        not isinstance(value, int | float) or not math.isfinite(value) or not 0 <= value <= 1
        for value in (a, b)
    ):
        return "证据不足"
    return "改善" if b > a else "下降" if b < a else "持平"


class Catalog:
    """Explicit roots, bounded files/cache, opaque evidence IDs; no arbitrary file API."""

    def __init__(self, roots, *, adapters=None):
        self.roots = tuple(Path(p).resolve(strict=True) for p in roots)
        self.cache = OrderedDict()
        self.evidence = {}
        self.adapters = adapters or {}
        self.remaining_read_bytes = None

    def read(self, path, *, lines=False):
        path = Path(path).resolve(strict=True)
        if not any(path.is_relative_to(root) for root in self.roots):
            raise ValueError("产物不在允许目录")
        stat = path.stat()
        if self.remaining_read_bytes is not None:
            self.remaining_read_bytes -= min(stat.st_size, MAX_BYTES)
            if self.remaining_read_bytes < 0:
                raise ValueError("本次刷新读取上限16MiB；请缩小root")
        key = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, lines)
        if key in self.cache:
            return self.cache[key]
        with path.open("rb") as stream:
            if lines and stat.st_size > MAX_BYTES:
                stream.seek(stat.st_size - MAX_BYTES)
                stream.readline()
            elif stat.st_size > MAX_BYTES:
                raise ValueError("JSON超过只读大小限制")
            raw = stream.read(MAX_BYTES)
        if lines:
            result = []
            for line in raw.splitlines(keepends=True):
                if line.endswith(b"\n"):
                    row = json.loads(line, parse_constant=reject_constant)
                    if not isinstance(row, dict):
                        raise ValueError("指标行必须是对象")
                    result.append(row)
            # Preserve attempt/stage identity; never merge across runs.
            unique = {}
            for i, row in enumerate(result):
                identity = (
                    (row["attempt"], row.get("stage"), row.get("update", i))
                    if row.get("attempt") is not None
                    else ("unknown_attempt", i)
                )
                unique[identity] = row
            value = list(unique.values())[-2000:]
        else:
            value = json.loads(raw, parse_constant=reject_constant)
            if not isinstance(value, dict):
                raise ValueError("产物必须是JSON对象")
        self.cache[key] = value
        while len(self.cache) > 64:
            self.cache.popitem(last=False)
        return value

    def optional(self, path, errors, *, lines=False):
        try:
            return self.read(path, lines=lines)
        except (OSError, ValueError) as exc:
            errors.append({"file": str(Path(path).name), "error": type(exc).__name__})
            return [] if lines else {}

    def verified(self, path, expected):
        path = Path(path).resolve(strict=True)
        if not any(path.is_relative_to(root) for root in self.roots):
            raise ValueError("评估不在允许目录")
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("评估大小或SHA不匹配")
        return json.loads(raw, parse_constant=reject_constant)

    def register(self, path, value=None):
        path = Path(path).resolve()
        value = self.read(path) if value is None else value
        digest = hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        token = hashlib.sha256((str(path) + digest).encode()).hexdigest()
        self.evidence[token] = (path, digest)
        return token

    def read_evidence(self, token):
        path, expected = self.evidence[token]
        value = self.read(path)
        actual = hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if actual != expected:
            raise ValueError("证据内容已变化，请刷新")
        return value

    def snapshot(self):
        self.remaining_read_bytes = 16 * 1024 * 1024
        self.evidence = {}
        runs = []
        for root in self.roots:
            paths = (
                [root / "run.json"]
                if (root / "run.json").is_file()
                else sorted(root.glob("*/run.json"))[:50]
            )
            for path in paths:
                errors = []
                run = self.optional(path, errors)
                directory = path.parent
                state = self.optional(directory / "state.json", errors)
                budget = self.optional(directory / "budget.json", errors)
                config = run.get("config") or {}
                if not isinstance(config, dict):
                    errors.append({"file": "run.json", "error": "unsupported_config_schema"})
                    config = {}
                for field in ("limits", "used"):
                    if budget.get(field) is not None and not isinstance(budget[field], dict):
                        errors.append({"file": "budget.json", "error": "invalid_" + field})
                        budget[field] = None
                task = run.get("task", config.get("task"))
                metrics = self.optional(directory / "metrics.jsonl", errors, lines=True)
                windows = []
                fixed_rows = []
                for diagnostic in sorted((directory / "diagnostics").glob("update_*.json"))[-100:]:
                    data = self.optional(diagnostic, errors)
                    adapter = self.adapters.get(task)
                    if adapter:
                        try:
                            windows.extend(adapter(data, config, self.register(diagnostic, data)))
                        except (ValueError, TypeError, AttributeError) as exc:
                            errors.append({"file": diagnostic.name, "error": type(exc).__name__})
                evaluations = []
                index = self.optional(directory / "evaluations/index.json", errors)
                entries = index.get("evaluations") or []
                if isinstance(entries, dict):
                    entries = list(entries.values())
                for entry in entries[-100:]:
                    try:
                        target = Path(entry["path"])
                        raw = self.verified(target, entry.get("sha256"))
                        model, protocol = raw.get("model") or {}, raw.get("protocol") or {}
                        if entry.get("model_sha256") != model.get("sha256"):
                            raise ValueError("评估模型身份冲突")
                        missing = [
                            name
                            for name, value in {
                                "checkpoint SHA": model.get("sha256"),
                                "protocol": protocol.get("version"),
                                "backend": protocol.get("backend"),
                                "seeds": protocol.get("seeds"),
                            }.items()
                            if value is None
                        ]
                        sha = model.get("sha256")
                        if (
                            not isinstance(sha, str)
                            or len(sha) != 64
                            or any(c not in "0123456789abcdef" for c in sha)
                        ):
                            missing.append("有效checkpoint SHA")
                        if protocol.get("seeds") == []:
                            missing.append("非空seeds")
                        evidence_id = self.register(target, raw)
                        evaluations.append(
                            {
                                "entry": entry,
                                "result": raw,
                                "evidence_id": evidence_id,
                                "identity_status": "证据不足" if missing else "身份字段齐全",
                                "missing": missing,
                            }
                        )
                        for condition in raw.get("conditions") or []:
                            fixed_rows.append(
                                {
                                    **condition,
                                    "layer": "fixed",
                                    "task": task,
                                    "checkpoint_sha256": model.get("sha256"),
                                    "protocol": protocol,
                                    "evidence_id": evidence_id,
                                    "identity_status": "证据不足" if missing else "身份字段齐全",
                                }
                            )
                    except (KeyError, OSError, ValueError, TypeError, AttributeError) as exc:
                        errors.append({"file": "evaluation", "error": type(exc).__name__})
                runs.append(
                    {
                        "id": hashlib.sha256(str(path).encode()).hexdigest(),
                        "name": run.get("run_id", directory.name),
                        "task": task,
                        "backend": run.get("backend", config.get("backend")),
                        "course": config["curriculum"].get("version")
                        if isinstance(config.get("curriculum"), dict)
                        else None,
                        "status": state.get("status", "UNKNOWN"),
                        "run_metadata_updated_at": datetime.fromtimestamp(
                            path.stat().st_mtime, ZoneInfo("Asia/Shanghai")
                        ).isoformat(),
                        "metric_attempt_note": "未记录attempt的重复update保留，不推断同一批数据；曲线在非递增update处分段",
                        "process": "UNKNOWN（未核验）",
                        "used": budget.get("used"),
                        "limits": budget.get("limits"),
                        "tensorboard_events_present": any(
                            (directory / "tensorboard").glob("events.out.tfevents.*")
                        ),
                        "remaining": {
                            k: limit - (budget.get("used") or {}).get(k, 0)
                            if isinstance(limit, int | float) and k in (budget.get("used") or {})
                            else None
                            for k, limit in (budget.get("limits") or {}).items()
                        },
                        "attempts": budget.get("attempts"),
                        "source_model": config.get("initialization"),
                        "model_references": self.optional(
                            Path(run.get("model_index", directory.parent / "models.json")), errors
                        ),
                        "updates_used": state.get("updates_used"),
                        "latest_model": state.get("latest_model"),
                        "selected_model": state.get("selected_model"),
                        "candidates": self.optional(directory / "training_candidates.json", errors),
                        "inventory": self.optional(
                            directory / "checkpoints/inventory.json", errors
                        ),
                        "metrics": metrics,
                        "windows": windows,
                        "evaluations": evaluations,
                        "fixed_rows": fixed_rows,
                        "hardware": [],
                        "hardware_status": "未实机验证（无已支持的身份记录）",
                        "matrix_support": "领域训练窗口"
                        if task in self.adapters
                        else "能力适配不支持",
                        "source": str(root),
                        "errors": errors,
                    }
                )
        self.remaining_read_bytes = None
        return public(
            {
                "timezone": "Asia/Shanghai",
                "read_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "source_kind": "服务端本地文件（远端请SSH转发）",
                "runs": runs,
                "limits": "每root最多50run/每run最近2000指标/100诊断，刷新读取上限16MiB；训练窗口跨策略，不是固定模型成绩；不计算统计显著性",
            }
        )
