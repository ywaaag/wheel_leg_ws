"""公共阶段驱动器：唯一负责调度、计费、保存与边界的地方。

职责（所有任务共享）：
- 阶段剩余额度与更新顺序；
- 停止请求、样本/墙钟/空间边界；
- 真实完成的更新计费；
- 周期保存、终态保存与结果返回；
- 正常与异常路径上的资源关闭。

任务/算法适配者负责：构造环境、网络与算法，从来源恢复状态，执行一次
rollout+优化更新，导出任务特有的完整状态。驱动器不知道具体任务是 Motion
还是 Recovery，也不重写任何任务数学。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


class SaveFailed(RuntimeError):
    """必需的完整checkpoint保存失败：任务失败，不能继续下一阶段。"""


class LimitReached(RuntimeError):
    """触达声明的运行边界；属于正常有界结束，不是工程失败。"""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


@dataclass
class Backend:
    """后端描述：环境与语义声明。后端不写运行状态文件。"""

    name: str
    make_env: Callable[[Any], Any]
    semantics: dict[str, Any] = field(default_factory=dict)
    condition_hook: Callable[[Any, dict[str, Any]], None] | None = None
    session_factory: Callable[..., Any] | None = None


@dataclass
class StageResult:
    updates: int
    actor_updates: int
    critic_only_updates: int
    samples: int
    last_checkpoint: dict[str, Any] | None
    stopped: bool = False
    limit: str | None = None
    reward_mean: float = 0.0


class StageAdapter(Protocol):
    """任务适配契约；实现者只在少数明确回调里承担任务特有逻辑。"""

    def build(self, context, stage: str, resume_from: dict[str, Any] | None) -> None:
        """构造环境/网络/算法并完成来源恢复。"""

    def update_once(self, context, index: int) -> dict[str, Any]:
        """执行一次真实更新；返回指标，可含 actor_steps 与 critic_only。"""

    def capture(self, context, counters: dict[str, Any]) -> dict[str, Any]:
        """导出完整恢复状态（不得改变算法状态）。"""

    def close(self) -> None:
        """释放环境与框架资源。"""


def run_stage(
    context,
    adapter: StageAdapter,
    *,
    stage: str,
    resume_from: dict[str, Any] | None = None,
) -> StageResult:
    """执行一个阶段；所有任务的计费、保存与边界都由这里执行。"""
    stage_limit = context.stage_remaining(stage)
    if stage_limit <= 0:
        raise LimitReached("budget", f"{stage}阶段预算已耗尽")

    context.startup_stage(stage)
    started = time.monotonic()
    wall_limit = context.wall_budget_left()
    space_limit = context.output_bytes_left()
    actor_updates = 0
    critic_only = 0
    reward_total = 0.0
    last_saved_update: int | None = None
    last_state: dict[str, Any] | None = None
    completed = 0
    from native_stand.learning.ppo.performance import Measurement

    timer = Measurement(
        context.job.diagnostics.get("performance_measurement", False), context.job.runtime.device
    )
    last_metrics_write = None
    consistent_update = False
    save_failed = False
    update_baseline = context.updates_used

    def save_now(update: int, actor_steps: int) -> None:
        nonlocal last_saved_update, last_state, save_failed
        counters = {
            "stage_updates": completed,
            "updates": update,
            "samples": context.samples_used,
            "actor_updates": actor_updates,
            "critic_only_updates": critic_only,
        }
        try:
            with timer.section("checkpoint_capture"):
                last_state = adapter.capture(context, counters)
            with timer.section("checkpoint_save_SHA"):
                reference = context.save(
                    last_state, update=update, actor_changed=actor_steps > 0, stage=stage
                )
        except BaseException:
            save_failed = True
            raise
        if reference is None:
            save_failed = True
            # 只有真正写盘成功才能推进保存位置；失败直接终止本次运行。
            raise SaveFailed(f"必需checkpoint保存失败: update={update} stage={stage}")
        last_saved_update = update
        published = getattr(adapter, "after_save", None)
        if callable(published):
            with timer.section("candidate_references_views"):
                published(context, reference)
        prune = getattr(context, "prune_checkpoints", None)
        if callable(prune):
            with timer.section("retention"):
                prune()

    try:
        adapter.build(context, stage, resume_from)
        for index in range(1, stage_limit + 1):
            if context.stop_requested():
                # A resumed attempt stopped before its first update must retain
                # the immutable existing point, not overwrite the same filename
                # with a freshly constructed world's state.
                if completed or context.latest_reference() is None:
                    _final(adapter, context, save_now, last_saved_update)
                return _result(
                    context,
                    completed,
                    actor_updates,
                    critic_only,
                    reward_total,
                    stopped=True,
                    limit="stop_requested",
                )

            consistent_update = False
            update_baseline = context.updates_used
            timer.seconds.clear()
            update_started = time.monotonic()
            outcome = adapter.update_once(context, index)
            if outcome.get("incomplete"):
                # Preserve the last published full point. The current env/storage
                # may already contain partial rollout state and is not certified.
                # Completed updates remain charged, even if that point is older.
                return _result(
                    context,
                    completed,
                    actor_updates,
                    critic_only,
                    reward_total,
                    stopped=True,
                    limit="rollout_incomplete",
                )
            actor_steps = int(outcome.get("actor_steps") or 0)
            if actor_steps > 0:
                actor_updates += 1
            elif outcome.get("critic_only"):
                critic_only += 1

            # 计费只在真实优化完成后发生；未完成的rollout不计入。
            context.charge(updates=1)
            completed += 1
            update = context.updates_used
            reward_total += float(outcome.get("reward_mean") or 0.0)

            # 任务后处理（生产：课程推进）在计费之后、写指标与保存之前执行。
            after = getattr(adapter, "after_update", None)
            if callable(after):
                with timer.section("task_post_update"):
                    after(context, index, outcome)

            # Only a complete optimization, charged budget and successful task
            # postprocessing form a publishable full-state boundary.
            consistent_update = True
            record = {
                key: value
                for key, value in outcome.items()
                if key not in ("actor_steps", "critic_only")
            }
            record.update(
                {
                    "update": update,
                    "actual_updates": update,
                    "samples": context.samples_used,
                    "seconds_per_update": outcome.get(
                        "seconds_per_update", time.monotonic() - update_started
                    ),
                }
            )
            if not timer.enabled:
                context.record(record)

            at_stage_end = index == stage_limit
            # 首个更新落一份可恢复状态；其余按周期与阶段末尾保存。
            if index == 1 or context.save_due(update, stage=stage) or at_stage_end:
                save_now(update, actor_steps)
            if timer.enabled:
                record["performance_seconds"] = {
                    **record.get("performance_seconds", {}),
                    **timer.seconds,
                }
                record["saved_this_update"] = last_saved_update == update
                record["whole_update_wall_seconds_before_metrics"] = (
                    time.monotonic() - update_started
                )
                record["previous_metrics_write_seconds"] = last_metrics_write
                record["stage_mean_wall_seconds"] = (time.monotonic() - started) / completed
                write_started = time.monotonic()
                context.record(record)
                last_metrics_write = time.monotonic() - write_started
                from native_stand.artifacts.io import append_jsonl

                # Separate opt-in timing row includes the completed main metrics
                # write, avoiding a misleading cumulative mean in the PPO field.
                append_jsonl(
                    context.run_dir / "performance.jsonl",
                    [
                        dict(
                            update=update,
                            saved_this_update=last_saved_update == update,
                            phase_seconds=record["performance_seconds"],
                            metrics_write_seconds=last_metrics_write,
                            whole_update_wall_seconds=time.monotonic() - update_started,
                            stage_mean_wall_seconds=(time.monotonic() - started) / completed,
                            clock="boundary_wall; includes primary metrics, excludes this timing row write",
                        )
                    ],
                )

            limit = _limit_reached(context, wall_limit, space_limit, started)
            if limit:
                _final(adapter, context, save_now, last_saved_update, actor_steps=actor_steps)
                return _result(
                    context, completed, actor_updates, critic_only, reward_total, limit=limit
                )

        _final(adapter, context, save_now, last_saved_update)
        return _result(context, completed, actor_updates, critic_only, reward_total)
    except BaseException as primary:
        receipt = getattr(adapter, "completed_optimization", None)
        if receipt == {"update": update_baseline + 1} and context.updates_used == update_baseline:
            try:
                context.charge(updates=1)
                primary.add_note(
                    "Completed optimization charged before telemetry failure; current state is not checkpoint-certified"
                )
            except BaseException as charging:
                primary.add_note(
                    "Completed optimization requires ledger reconciliation: " + repr(charging)
                )
        if consistent_update and not save_failed and last_saved_update != context.updates_used:
            try:
                save_now(context.updates_used, int(outcome.get("actor_steps") or 0))
            except BaseException as saving:
                primary.add_note("Fault-boundary checkpoint failed: " + repr(saving))
        try:
            adapter.close()
        except BaseException as closing:
            primary.add_note("Adapter cleanup failed: " + repr(closing))
        raise
    finally:
        # The exception path already closed resources while preserving primary.
        if sys.exc_info()[0] is None:
            adapter.close()


def _limit_reached(context, wall_limit, space_limit, started) -> str | None:
    job = context.job
    if job.limits.sample_limit and context.samples_used >= job.limits.sample_limit:
        return "sample_limit"
    if wall_limit is not None and (time.monotonic() - started) >= wall_limit:
        return "wall_clock"
    if space_limit is not None and context.output_bytes_used() >= space_limit:
        return "max_output_bytes"
    return None


def _final(
    adapter, context, save_now, last_saved_update: int | None, *, actor_steps: int = 0
) -> None:
    """终态保存：同一更新已经保存过就不再重复写一份大文件。"""
    if last_saved_update == context.updates_used:
        return
    save_now(context.updates_used, actor_steps)


def _result(
    context,
    completed: int,
    actor_updates: int,
    critic_only: int,
    reward_total: float,
    *,
    stopped: bool = False,
    limit: str | None = None,
) -> StageResult:
    return StageResult(
        updates=completed,
        actor_updates=actor_updates,
        critic_only_updates=critic_only,
        samples=context.samples_used,
        last_checkpoint=context.latest_reference(),
        stopped=stopped,
        limit=limit,
        reward_mean=reward_total / completed if completed else 0.0,
    )
