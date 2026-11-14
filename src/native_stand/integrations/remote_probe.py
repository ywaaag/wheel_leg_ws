"""Fixed read-only remote probe; executed with system python3, never imports training code."""

import base64
import glob
import hashlib
import json
import os
import pathlib
import subprocess


def collect(root):
    root = pathlib.Path(root).resolve()
    result = {"files": {}, "errors": [], "processes": [], "resources": {}}
    remaining = 12 * 1024 * 1024
    for name in (
        "startup_handshake.json",
        "run.json",
        "state.json",
        "budget.json",
        "metrics.jsonl",
        "training_candidates.json",
        "checkpoints/inventory.json",
        "evaluations/index.json",
    ):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            continue
        try:
            size = path.stat().st_size
            limit = (
                256 * 1024 if name == "state.json" and size > 4 * 1024 * 1024 else 2 * 1024 * 1024
            )
            with path.open("rb") as stream:
                if name.endswith("jsonl") and size > limit:
                    stream.seek(size - limit)
                    stream.readline()
                raw = stream.read(min(limit, remaining))
            remaining -= len(raw)
            result["files"][name] = {
                "base64": base64.b64encode(raw).decode(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": size,
            }
        except PermissionError:
            result["errors"].append({"file": name, "status": "权限不足"})
        except OSError:
            result["errors"].append({"file": name, "status": "UNAVAILABLE"})
    for filename in sorted(glob.glob(str(root / "diagnostics/update_*.json")))[-5:]:
        path = pathlib.Path(filename).resolve()
        if not path.is_relative_to(root):
            continue
        try:
            if path.stat().st_size > 1024 * 1024 or remaining < path.stat().st_size:
                continue
            raw = path.read_bytes()
            remaining -= len(raw)
            result["files"][str(path.relative_to(root))] = {
                "base64": base64.b64encode(raw).decode(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
        except OSError:
            pass
    index_path = root / "evaluations/index.json"
    try:
        if index_path.stat().st_size < 256 * 1024:
            entries = json.loads(index_path.read_text()).get("evaluations") or []
            if isinstance(entries, dict):
                entries = list(entries.values())
            for entry in entries[-20:]:
                path = pathlib.Path(entry["path"]).resolve()
                if (
                    not path.is_relative_to(root / "evaluations")
                    or path.stat().st_size > 512 * 1024
                    or remaining < path.stat().st_size
                ):
                    continue
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
                    result["errors"].append({"file": "evaluation", "status": "SHA_MISMATCH"})
                    continue
                remaining -= len(raw)
                result["files"][str(path.relative_to(root))] = {
                    "base64": base64.b64encode(raw).decode(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size": len(raw),
                }
    except (OSError, ValueError, KeyError, TypeError):
        pass
    # Candidates only: identity is checked by the backend against declared PID/start_ticks/run.
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if "native_stand.experiments.worker" not in argv or str(root) not in argv:
                continue
            stat = (entry / "stat").read_text().split(") ", 1)[1].split()
            result["processes"].append(
                {
                    "pid": int(entry.name),
                    "argv": argv,
                    "cwd": os.readlink(entry / "cwd"),
                    "start_ticks": int(stat[19]),
                    "state": stat[0],
                    "status": (entry / "status").read_text()[:6000],
                    "cpu_ticks": {"user": int(stat[11]), "system": int(stat[12])},
                    "rss_bytes": int(stat[21]) * os.sysconf("SC_PAGE_SIZE"),
                }
            )
        except (OSError, ValueError):
            continue
    resources = result["resources"]
    resources["affinity"] = sorted(os.sched_getaffinity(0))
    for name in ("cpu.max", "memory.max", "memory.current", "cpu.stat"):
        try:
            resources[name] = (pathlib.Path("/sys/fs/cgroup") / name).read_text()[:2048]
        except OSError:
            resources[name] = None
    disk = os.statvfs(root)
    resources["disk"] = {
        "total_bytes": disk.f_blocks * disk.f_frsize,
        "free_bytes": disk.f_bavail * disk.f_frsize,
    }
    for key, args in {
        "gpu": [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        "gpu_processes": [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
    }.items():
        try:
            resources[key] = subprocess.run(
                args, capture_output=True, text=True, timeout=5, check=False
            ).stdout[:16000]
        except (OSError, subprocess.TimeoutExpired):
            resources[key] = None
    resources["gpu_pid_mapping"] = (
        "UNKNOWN: container and host PID namespaces not assumed identical"
    )
    return result
