"""人类可读报告：从标准运行状态与评估索引生成，不重新判定阶段。

报告失败单独记录，不修改已经发生的训练终态。
"""

from __future__ import annotations

from pathlib import Path

from .evaluation import read_index, summarize
from .io import file_sha256, read_json, read_jsonl, write_json


def build_report(run_dir: str | Path, *, metrics_tail: int = 5, state_snapshot=None) -> str:
    """生成Markdown报告；只引用标准记录，不复制raw矩阵。"""
    directory = Path(run_dir)
    run = read_json(directory / "run.json", {})
    state = read_json(directory / "state.json", {}) if state_snapshot is None else state_snapshot
    config = run.get("config") or {}
    limits = run.get("limits") or {}
    total_limit = (limits.get("smoke_updates") or 0) + (limits.get("formal_updates") or 0)
    evaluations = read_index(directory / "evaluations")
    rows = read_jsonl(directory / "metrics.jsonl")[-metrics_tail:]
    lines = [
        f"# {config.get('name', run.get('run_id', directory.name))}",
        "",
        f"- 任务/算法/后端: {config.get('task')} / {config.get('algorithm')} / "
        f"{config.get('backend')}",
        f"- 终态: {state.get('status', 'UNKNOWN')}；阶段: {state.get('stage')}",
        f"- 实际消耗: {state.get('updates_used', 0)} updates / "
        f"{state.get('samples_used', 0)} samples（上限 {total_limit}）",
        f"- 最新模型: {(state.get('latest_model') or {}).get('checkpoint')}",
        f"- 选中模型: {(state.get('selected_model') or {}).get('checkpoint')}",
        f"- 能力accepted: {bool(state.get('capability_accepted'))}（训练终态不等于能力验收）",
    ]
    if state.get("failure"):
        lines.append(f"- 失败: {state['failure']}")
    lines.append("")
    lines.append("## 评估")
    for entry in evaluations.get("evaluations") or []:
        kind = entry["kind"]
        path = Path(entry["path"])
        summary = summarize(read_json(path, {})) if path.is_file() else {"status": "MISSING"}
        lines.append(
            f"- {kind}: {summary.get('status')}；条件 {summary.get('conditions')}；"
            f"环境 {summary.get('envs')}；失败环境 {summary.get('env_failures')}；"
            f"模型 {(summary.get('model_sha256') or '')[:12]}"
        )
    if not evaluations.get("evaluations"):
        lines.append("- 没有独立评估记录；不生成best或accepted。")
    lines.append("")
    lines.append("## 最近指标")
    for row in rows:
        pairs = ", ".join(f"{key}={value}" for key, value in row.items() if key != "update")
        lines.append(f"- update {row.get('update')}: {pairs}")
    lines.append("")
    lines.append("## 来源")
    lines.append(
        f"- 配置: {run.get('config_path')}（sha256 {(run.get('config_sha256') or '')[:12]}）"
    )
    lines.append(f"- 代码身份: {(run.get('source') or {}).get('modules_sha256')}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_report(run_dir: str | Path, *, state_snapshot=None) -> Path:
    """写report.md并记录自身SHA；失败时抛错由调用方单独记录。"""
    directory = Path(run_dir)
    path = directory / "report.md"
    path.write_text(build_report(directory, state_snapshot=state_snapshot))
    write_json(
        directory / "report_ready.json",
        {"status": "report_ready", "report": str(path.resolve()), "sha256": file_sha256(path)},
    )
    return path
