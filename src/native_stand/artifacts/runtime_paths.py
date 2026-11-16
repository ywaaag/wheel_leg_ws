"""Explicit path relocation at load boundaries; immutable source files stay intact."""

from dataclasses import dataclass
from pathlib import Path

from .io import file_sha256, read_json


@dataclass(frozen=True)
class RuntimePaths:
    files: tuple = ()
    prefixes: tuple = ()

    @classmethod
    def from_assets(cls, assets):
        ref = assets.get("path_mapping")
        if ref is None:
            return cls()
        if file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("运行路径映射SHA不同")
        data = read_json(ref["path"])
        if data.get("version") != "runtime_path_mapping_v1":
            raise ValueError("未知运行路径映射版本")
        pairs = [*data.get("files", {}).items(), *data.get("prefixes", {}).items()]
        if any(
            not Path(a).is_absolute() or not Path(b).is_absolute() or "$" in b for a, b in pairs
        ):
            raise ValueError("路径映射必须是已确认的绝对路径；不能使用占位持久盘")
        return cls(
            tuple(data.get("files", {}).items()),
            tuple(sorted(data.get("prefixes", {}).items(), key=lambda p: len(p[0]), reverse=True)),
        )

    def resolve(self, value):
        value = str(value)
        for old, new in self.files:
            if value == old:
                return new
        for old, new in self.prefixes:
            if value == old or value.startswith(old.rstrip("/") + "/"):
                return new.rstrip("/") + value[len(old) :]
        return value

    def view(self, value):
        if isinstance(value, dict):
            return {key: self.view(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self.view(val) for val in value]
        if isinstance(value, str) and value.startswith("/") and not value.startswith("/World/"):
            return self.resolve(value)
        return value

    def read(self, path):
        return self.view(read_json(self.resolve(path)))


def execution_json(path, assets):
    return RuntimePaths.from_assets(assets).read(path)


def asset_values(assets, base, paths):
    """Expand explicitly named, SHA-bound JSON values at the config read boundary."""

    def references(value, relative_to):
        if isinstance(value, list):
            return [references(item, relative_to) for item in value]
        if isinstance(value, str) and value.startswith("/") and not value.startswith("/World/"):
            return paths.resolve(value)
        if not isinstance(value, dict):
            return value
        identified = isinstance(value.get("path"), str) and value["path"] and "sha256" in value
        result = {
            k: v if identified and k == "path" else references(v, relative_to)
            for k, v in value.items()
        }
        if identified:
            path = Path(value["path"])
            if not path.is_absolute():
                path = (Path(relative_to) / path).resolve()
            result["path"] = paths.resolve(path)
        return result

    result = references(assets, base)
    files = result.get("value_files", {})
    if not isinstance(files, dict):
        raise ValueError("assets.value_files must map names to JSON path/SHA references")
    bound = {}
    for name, ref in files.items():
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("Bound asset values cannot overwrite inline configuration")
        if (
            not isinstance(ref, dict)
            or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str)
            or not ref["path"]
            or not isinstance(ref["sha256"], str)
            or len(ref["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in ref["sha256"])
        ):
            raise ValueError("assets.value_files requires explicit JSON path and SHA256")
        path = Path(ref["path"])
        if file_sha256(path) != ref["sha256"]:
            raise ValueError("Bound asset JSON SHA differs: " + name)
        result[name] = references(read_json(path), path.parent)
        bound[name] = dict(path=str(path), sha256=ref["sha256"])
    if files:
        result["value_files"] = bound
    return result


def execution_job(job):
    """Map a runtime view once, leaving original records and semantic digests intact."""
    if not job.assets.get("path_mapping"):
        return job
    from native_stand.experiments.config import job_from_record

    paths = RuntimePaths.from_assets(job.assets)
    record = job.as_run_record()
    supervision = record.pop("supervision", None)
    record = paths.view(record)

    def identified(value):
        if isinstance(value, dict):
            result = {k: identified(v) for k, v in value.items()}
            if (
                isinstance(value.get("path"), str)
                and isinstance(value.get("sha256"), str)
                and len(value["sha256"]) == 64
            ):
                result["path"] = paths.resolve(value["path"])
            return result
        if isinstance(value, list):
            return [identified(v) for v in value]
        return value

    record["supervision"] = identified(supervision)
    return job_from_record(record)


def resolve_runs_root(explicit=None):
    """Resolve persistent output at the write boundary, never at import time.

    Explicit --runs-root wins; otherwise WHEEL_LEG_DATA_ROOT is required.
    Resolution alone does not create directories or an index.
    """
    import os

    value = explicit if explicit is not None else os.environ.get("WHEEL_LEG_DATA_ROOT")
    if not value or not str(value).strip():
        raise ValueError(
            "未设置输出根：请指定 --runs-root 或设置 WHEEL_LEG_DATA_ROOT 为已确认的数据盘"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("输出根必须为绝对路径（已确认的本地数据根或远端持久盘）")
    return path.resolve()
