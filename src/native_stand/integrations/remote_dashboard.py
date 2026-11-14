"""Explicit bounded OpenSSH collection; credentials live only in this backend process."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from threading import Lock

from native_stand.artifacts.dashboard import Catalog, state_prefix
from native_stand.tasks.motion.dashboard import training_windows


class RemoteError(Exception):
    pass


def process_identity(state, processes):
    declared = state.get("owner") or state.get("worker") or {}
    pid = declared.get("pid", state.get("worker_pid"))
    ticks = declared.get("start_ticks", state.get("worker_start_ticks"))
    candidate = next((p for p in processes if p["pid"] == pid), None)
    if candidate is None:
        return {
            "status": "UNKNOWN" if pid is None else "声明PID未在候选worker中发现",
            "declared_pid": pid,
        }
    if ticks is None:
        return {"status": "UNKNOWN：声明缺少start_ticks，不能排除PID复用", "candidate": candidate}
    if int(ticks) != candidate["start_ticks"]:
        return {"status": "不一致：PID复用", "candidate": candidate}
    if candidate["state"] == "Z":
        return {"status": "不一致：僵尸进程", "candidate": candidate}
    expected_cwd = state.get("_expected_cwd")
    if expected_cwd and candidate.get("cwd") != expected_cwd:
        return {"status": "不一致：worker cwd与冻结source_root不同", "candidate": candidate}
    return {"status": "进程已核验", "candidate": candidate}


def classify(text):
    for needle, status in [
        ("REMOTE HOST IDENTIFICATION HAS CHANGED", "主机密钥不符"),
        ("Host key verification failed", "主机密钥不符"),
        ("Permission denied", "认证失败"),
        ("Too many authentication failures", "认证失败：认证次数过多"),
        ("PermissionError", "权限不足"),
        ("Connection refused", "连接拒绝"),
        ("timed out", "超时"),
        ("SFTP", "不支持SFTP"),
    ]:
        if needle in text:
            return status
    return "SSH读取失败（无法核实训练状态）"


class RemoteSource:
    def __init__(self, *, host, port, username, root, fingerprint, password=None, runner=None):
        self.configure(host=host, port=port, username=username, fingerprint=fingerprint)
        self.root = str(PurePosixPath(root))
        if not self.root.startswith("/") or ".." in PurePosixPath(root).parts:
            raise ValueError("远端root必须是明确绝对目录")
        self.password = password
        self.runner = runner or self._ssh
        self.lock = Lock()
        self.last = None
        self.status = "尚未连接"
        self.last_success = None
        self.cache = tempfile.TemporaryDirectory(prefix="wheel-dashboard-remote-")
        self.catalog = Catalog([self.cache.name], adapters={"motion": training_windows})

    def configure(self, *, host, port, username, fingerprint):
        if not re.fullmatch(r"[a-zA-Z0-9.-]{1,253}", host) or host.startswith("-"):
            raise ValueError("无效SSH地址")
        if not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_-]{0,63}", username):
            raise ValueError("无效SSH用户名")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("无效SSH端口")
        if not re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}", fingerprint):
            raise ValueError("需要明确ED25519 SHA256指纹")
        self.host, self.port, self.username, self.fingerprint = host, port, username, fingerprint

    def _ssh(self):
        import pexpect

        # OpenSSH strictly checks existing known_hosts; additionally enforce the pinned key.
        key = subprocess.run(
            ["ssh-keygen", "-F", self.host if self.port == 22 else f"[{self.host}]:{self.port}"],
            capture_output=True,
            text=True,
            check=False,
        )
        entries = [line for line in key.stdout.splitlines() if not line.startswith("#")]
        if not entries:
            raise RemoteError("主机密钥未登记：先私密核对known_hosts与指定指纹")
        pins = []
        for line in entries:
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "ssh-ed25519":
                pins.append(
                    "SHA256:"
                    + base64.b64encode(hashlib.sha256(base64.b64decode(parts[2])).digest())
                    .decode()
                    .rstrip("=")
                )
        if self.fingerprint not in pins:
            raise RemoteError("主机密钥不符")
        script = (
            Path(__file__).with_name("remote_probe.py").read_bytes()
            + b"\nprint(json.dumps(collect(sys.argv[1]), allow_nan=False))\n"
        )
        encoded = base64.b64encode(script).decode()
        command = (
            "python3 -c "
            + shlex.quote(
                f"import base64,sys,json;exec(compile(base64.b64decode('{encoded}'),'readonly_probe','exec'))"
            )
            + " "
            + shlex.quote(self.root)
        )
        child = pexpect.spawn(
            "ssh",
            [
                "-T",
                "-p",
                str(self.port),
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "ConnectTimeout=8",
                "-o",
                "ServerAliveInterval=5",
                "-o",
                "ServerAliveCountMax=1",
                "-o",
                "NumberOfPasswordPrompts=1",
                *(
                    [
                        "-o",
                        "PreferredAuthentications=password,keyboard-interactive",
                        "-o",
                        "PubkeyAuthentication=no",
                        "-o",
                        "IdentityAgent=none",
                    ]
                    if self.password is not None
                    else ["-o", "IdentitiesOnly=yes"]
                ),
                f"{self.username}@{self.host}",
                command,
            ],
            encoding="utf-8",
            timeout=25,
            echo=False,
            maxread=65536,
        )
        child.logfile = None
        chunks = []
        received = 0
        deadline = time.monotonic() + 45
        prompt_buffer = ""
        sent = False
        try:
            while True:
                if time.monotonic() > deadline:
                    raise RemoteError("超时")
                try:
                    chunk = child.read_nonblocking(65536, timeout=1)
                except pexpect.TIMEOUT:
                    continue
                except pexpect.EOF:
                    break
                received += len(chunk)
                if received > 18 * 1024 * 1024:
                    raise RemoteError("返回字节超限")
                if not sent:
                    prompt_buffer += chunk
                    if re.search(r"(?i)password:", prompt_buffer):
                        if self.password is None:
                            raise RemoteError("认证失败：未配置私密凭据或SSH key")
                        child.sendline(self.password)
                        sent = True
                        prompt_buffer = ""
                        continue
                    if len(prompt_buffer) > 4096 or '{"files"' in prompt_buffer:
                        chunks.append(prompt_buffer)
                        prompt_buffer = ""
                else:
                    chunks.append(chunk)
            chunks.append(prompt_buffer)
            child.close()
            output = "".join(chunks).strip()
            if child.exitstatus != 0:
                raise RemoteError(classify(output))
            start = output.find('{"files"')
            if start < 0:
                raise RemoteError("远端只读协议不可用（系统python3/受控exec不支持）")
            return json.loads(output[start:])
        finally:
            child.close(force=True)  # closes local transport only; never signals a remote worker

    def refresh(self):
        if not self.lock.acquire(blocking=False):
            return self.view("连接中：合并重复刷新")
        try:
            self.status = "连接中"
            payload = self.runner()
            generation = tempfile.TemporaryDirectory(prefix="wheel-dashboard-remote-")
            directory = Path(generation.name) / "remote"
            directory.mkdir()
            for name, entry in payload["files"].items():
                relative = PurePosixPath(name)
                allowed = (
                    name
                    in (
                        "startup_handshake.json",
                        "run.json",
                        "state.json",
                        "budget.json",
                        "metrics.jsonl",
                        "training_candidates.json",
                        "checkpoints/inventory.json",
                        "evaluations/index.json",
                    )
                    or bool(re.fullmatch(r"diagnostics/update_\d+\.json", name))
                    or bool(re.fullmatch(r"evaluations/[A-Za-z0-9_.-]+\.json", name))
                )
                if relative.is_absolute() or ".." in relative.parts or not allowed:
                    raise RemoteError("路径限制拒绝")
                raw = base64.b64decode(entry["base64"], validate=True)
                if len(raw) > 2 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                    raise RemoteError("传输身份或大小不符")
                if name == "state.json" and entry["size"] > 4 * 1024 * 1024:
                    raw = json.dumps(state_prefix(raw)).encode()
                target = directory / name
                target.parent.mkdir(exist_ok=True, parents=True)
                target.write_bytes(raw)
            evaluation_index = directory / "evaluations/index.json"
            if evaluation_index.is_file():
                index = json.loads(evaluation_index.read_text())
                entries = index.get("evaluations") or []
                if isinstance(entries, dict):
                    entries = [{**entry, "kind": kind} for kind, entry in entries.items()]
                for entry in entries:
                    original = PurePosixPath(entry.get("path", ""))
                    try:
                        relative = original.relative_to(PurePosixPath(self.root))
                    except ValueError:
                        continue
                    if str(relative) in payload["files"]:
                        entry["source_remote_path"] = str(original)
                        entry["path"] = str(directory / str(relative))
                index["evaluations"] = entries
                evaluation_index.write_text(json.dumps(index))
            next_catalog = Catalog([generation.name], adapters={"motion": training_windows})
            snapshot = next_catalog.snapshot()
            now = time.time()
            for run in snapshot["runs"]:
                state = next_catalog.read(directory / "state.json")
                handshake = next_catalog.optional(directory / "startup_handshake.json", [])
                if not (state.get("owner") or {}).get("start_ticks") and handshake.get(
                    "worker_start_ticks"
                ):
                    declared_pid = (state.get("owner") or {}).get("pid")
                    if declared_pid is None or declared_pid == handshake.get("worker_pid"):
                        state = {
                            **state,
                            "worker_pid": handshake.get("worker_pid"),
                            "worker_start_ticks": handshake.get("worker_start_ticks"),
                        }
                record = next_catalog.read(directory / "run.json")
                expected_cwd = handshake.get("worker_cwd")
                if expected_cwd:
                    state["_expected_cwd"] = expected_cwd
                run["source_root_declared"] = (record.get("config") or {}).get("source_root")
                identity = process_identity(state, payload.get("processes", []))
                identity["verified_unix_s"] = now
                identity["declared_state"] = run["status"]
                if run["status"] != "RUNNING" and identity["status"] == "进程已核验":
                    identity["status"] = "不一致：state非RUNNING但worker身份存活"
                if identity.get("candidate"):
                    run["remote_worker"] = identity["candidate"]
                run["remote_identity"] = identity
                run["process"] = identity["status"]
                run["source"] = f"ssh://{self.username}@{self.host}:{self.port}{self.root}"
                run["remote"] = True
                run["metrics_file_updated_at"] = (
                    None  # cache file mtime is not remote training time
                )
                run["run_metadata_updated_at"] = None
                run["remote_collected_unix_s"] = now
            old_cache = self.cache
            self.cache = generation
            self.catalog = next_catalog
            old_cache.cleanup()
            self.last = {
                "runs": snapshot["runs"],
                "resources": payload.get("resources", {}),
                "probe_errors": payload.get("errors", []),
            }
            self.last_success = now
            self.status = "成功"
        except Exception as exc:
            self.status = (
                str(exc)
                if isinstance(exc, RemoteError)
                else "读取失败：协议/权限/格式错误（无法核实训练状态）"
            )
        finally:
            self.lock.release()
        return self.view()

    def view(self, status=None):
        return {
            "status": status or self.status,
            "last_success_unix_s": self.last_success,
            "age_seconds": time.time() - self.last_success if self.last_success else None,
            "cached": True,
            "stale": self.status != "成功"
            or (self.last_success is not None and time.time() - self.last_success > 300),
            "port": self.port,
            "host": self.host,
            "username": self.username,
            "fingerprint": self.fingerprint,
            "snapshot": self.last,
            "root": self.root,
        }
