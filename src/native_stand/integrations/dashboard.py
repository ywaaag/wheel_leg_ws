"""Loopback-only read-only dashboard. No engine, subprocess or model imports."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

from native_stand.artifacts.dashboard import Catalog, compare, public


def server(roots, *, host="127.0.0.1", port=8765, tensorboard_url=None):
    if host != "127.0.0.1":
        raise ValueError("首版仅允许127.0.0.1；远端使用SSH端口转发")
    from native_stand.tasks.motion.dashboard import target_description, training_windows

    if tensorboard_url:
        url = urlsplit(tensorboard_url)
        if (
            url.scheme not in ("http", "https")
            or url.hostname not in ("localhost", "127.0.0.1")
            or url.username
            or url.password
        ):
            raise ValueError("TensorBoard只接受用户显式提供的本机HTTP地址；不自动启动")
    document = Path(__file__).resolve().parents[3] / "docs/MOTION_CAPABILITY_MATRIX.md"
    targets = (
        target_description(document.read_text())
        if document.is_file()
        else {"rows": [], "scope": "目标文档不支持（安装中未包含）"}
    )

    catalog = Catalog(roots, adapters={"motion": training_windows})
    lock = Lock()
    current = {"runs": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Host", "").split(":")[0] not in ("127.0.0.1", "localhost"):
                self.send_error(403)
                return
            route = urlsplit(self.path).path
            try:
                with lock:
                    if route == "/":
                        body = Path(__file__).with_name("dashboard.html").read_bytes()
                        mime = "text/html; charset=utf-8"
                    elif route == "/api/snapshot":
                        snapshot = catalog.snapshot()
                        current.update(snapshot)
                        snapshot["targets"] = {"motion": targets}
                        snapshot["tensorboard_url"] = tensorboard_url
                        body = json.dumps(snapshot, ensure_ascii=False, allow_nan=False).encode()
                        mime = "application/json"
                    elif route == "/api/compare":
                        query = parse_qs(
                            urlsplit(self.path).query, max_num_fields=4, strict_parsing=True
                        )
                        run = next(row for row in current["runs"] if row["id"] == query["run"][0])
                        layer = query["layer"][0]
                        if layer not in ("windows", "fixed_rows"):
                            raise ValueError("未知证据层")
                        left, right = (run[layer][int(query[key][0])] for key in ("left", "right"))
                        verdict = compare(left, right)
                        if layer == "fixed_rows" and any(
                            row.get("identity_status") != "身份字段齐全" for row in (left, right)
                        ):
                            verdict = "证据不足"
                        body = json.dumps(
                            {
                                "verdict": verdict,
                                "n": [left.get("ended_episodes"), right.get("ended_episodes")],
                                "note": "描述性比较，不代表统计显著性；完整比较域缺失时拒绝比较",
                            },
                            ensure_ascii=False,
                        ).encode()
                        mime = "application/json"
                    elif route.startswith("/api/evidence/"):
                        token = route.removeprefix("/api/evidence/")
                        body = json.dumps(
                            public(catalog.read_evidence(token)),
                            ensure_ascii=False,
                            allow_nan=False,
                        ).encode()
                        mime = "application/json"
                    else:
                        self.send_error(404)
                        return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self' data:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError, KeyError, IndexError, StopIteration):
                self.send_error(404, "Artifact unavailable")

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(roots, *, host="127.0.0.1", port=8765, tensorboard_url=None):
    with server(roots, host=host, port=port, tensorboard_url=tensorboard_url) as http:
        print(f"只读看板 http://{host}:{http.server_port}", flush=True)
        http.serve_forever()
