"""MuJoCo跨引擎核对入口。

只用于交叉核对，不承担训练；复用控制时序与动作解码的共享实现。
"""

from __future__ import annotations

ENGINE_AVAILABLE = False
CAPABILITY = "UNAVAILABLE"
NOTE = "实时跨引擎采集尚未迁移；wheel-leg sim2sim支持显式已录制轨迹诊断"


def available(job) -> tuple[bool, str]:
    return False, NOTE


def make_env(job):
    raise RuntimeError(NOTE)


def run(context) -> None:
    raise RuntimeError(NOTE)
