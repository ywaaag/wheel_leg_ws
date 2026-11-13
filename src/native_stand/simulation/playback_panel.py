"""Web input adapter for the existing player loop and existing HTTP service."""

import math
import time
from pathlib import Path

from native_stand.students.motion_panel import Plan, ThreadingHTTPServer, handler_for

from .playback import Controls


class PlaybackCommands(Plan):
    def __init__(self, plant):
        super().__init__(max_speed=4.0, max_yaw_speed=4.0, backend="MuJoCo")
        self.pause_supported = True
        self._paused = True
        self._height = plant.plan.job.control.height_command_m
        self.bounds = plant.plan.output_contract["height_command_range_m"]
        self.spawns = list(plant.terrain.spawns) if plant.terrain else ["flat"]
        self._scene_request = None
        self._quit = False
        self.recording = plant.recording
        self.error = None

    def apply(self, payload):
        allowed = {"vx", "yaw", "height", "paused", "stop", "reset", "scene", "quit"}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise ValueError("只支持Motion运动、高度、暂停、重置、地形与关闭命令")
        # Explicit raw setpoints: no hidden clamp, ramp, filter or compensation.
        for k in ("vx", "yaw", "height"):
            if k in payload and (
                isinstance(payload[k], bool)
                or not isinstance(payload[k], float | int)
                or not math.isfinite(payload[k])
            ):
                raise ValueError(k + "必须为有限数值")
        if "height" in payload and not self.bounds[0] <= payload["height"] <= self.bounds[1]:
            raise ValueError("height超出模型合同")
        for k in ("paused", "stop", "reset", "quit"):
            if k in payload and type(payload[k]) is not bool:
                raise ValueError(k + "必须是bool")
        if "scene" in payload and payload["scene"] not in self.spawns:
            raise ValueError("未知地形")
        with self._lock:
            for i, k in enumerate(("vx", "yaw")):
                if k in payload:
                    self._target[i] = float(payload[k])
            if "height" in payload:
                self._height = float(payload["height"])
            if "paused" in payload:
                self._paused = payload["paused"]
            if payload.get("stop"):
                self._target = [0.0, 0.0]
            if payload.get("reset"):
                self._pending["reset"] = True
            if "scene" in payload:
                self._scene_request = payload["scene"]
                self._target = [0.0, 0.0]
                self._pending["reset"] = True
            if payload.get("quit"):
                self._quit = True
            self.requests += 1
        return dict(payload)

    def snapshot(self):
        result = super().snapshot()
        result.update(
            scenes=self.spawns,
            disconnect_behavior="保持最后命令；请先停止或暂停再关闭网页",
            height_bounds=self.bounds,
            recording=self.recording,
            error=self.error,
        )
        return result

    def consume(self):
        with self._lock:
            row = dict(
                vx=self._target[0],
                yaw=self._target[1],
                height=self._height,
                paused=self._paused,
                reset=self._pending["reset"],
                scene=self._scene_request,
                quit=self._quit,
            )
            self._pending["reset"] = False
            self._scene_request = None
        return row


class WebControls(Controls):
    def __init__(self, plant):
        super().__init__()
        plant.web_owner = True
        self.plan = PlaybackCommands(plant)
        self.next_read = 0.0
        self.next_publish = 0.0

    def key(self, keycode):
        # GLFW keyboard never competes with the web command owner.
        pass

    def fault(self, error):
        self.paused = True
        with self.plan._lock:
            self.plan._paused = True
            self.plan.error = error

    def sync(self, plant):
        now = time.monotonic()
        if now >= self.next_read:
            row = self.plan.consume()
            self.vx, self.yaw, self.height = row["vx"], row["yaw"], row["height"]
            self.paused, self.quit_requested = row["paused"], row["quit"]
            if row["scene"] is not None:
                plant.spawn = row["scene"]
            self.reset_requested |= row["reset"]
            if row["reset"]:
                self.plan.error = None
            self.next_read = now + 0.02
        if now >= self.next_publish:
            self.plan.publish(plant.telemetry())
            self.next_publish = now + 0.1


def start(plan, port, *, host="127.0.0.1"):
    import threading

    html = Path(__file__).with_name("playback_panel.html").read_text()
    script = Path(__file__).with_name("playback_panel.js").read_text()
    html = html.replace("/*PLAYBACK_SCRIPT*/", script)
    server = ThreadingHTTPServer((host, port), handler_for(plan, html))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="playback-panel")
    thread.start()
    return server, thread
