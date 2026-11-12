"""同一采样和晋级实现，两个显式配置域；只消费正常结束回合摘要。"""

from __future__ import annotations

import copy
import math

from native_stand.artifacts.io import json_digest

from .progressive_course import FAMILIES, mixtures, split, stage_from_rows

VERSIONS = {
    "motion48_travel_branch_v1": "motion",
    "motion48_spin_branch_v1": "spin",
    "motion48_height020_roughH1_branch_v2": "motion",
}
MAX_COMMAND_MAGNITUDE = 3.4028234663852886e38  # Actual command tensors use float32.
DR_PROGRESSION = "shared_strength_independent_of_task_axes"


def _finite_number(value):
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def validate_configuration(value):
    branch = VERSIONS.get(value.get("version"))
    if branch != value.get("branch") or branch is None:
        raise ValueError("Motion/Spin分支课程配置不一致")
    if not math.isclose(sum(value["fractions"].values()), 1.0, abs_tol=1e-12):
        raise ValueError("分支采样比例必须合计1")
    if any(not 0 <= f <= 1 for f in value["fractions"].values()):
        raise ValueError("分支采样比例无效")
    if value["command_program_version"] not in (
        "motion48_height_travel_turn_schedule_v2",
        "motion48_height020_rate_limited_schedule_v3",
    ):
        raise ValueError("变高后运动必须使用已验证的命令时序")
    if not 0 <= value["initial_DR_level"] < len(value["DR_amplitudes"]):
        raise ValueError("分支DR起点无效")
    if value.get("DR_progression") != DR_PROGRESSION:
        raise ValueError("分支必须显式声明DR强度共用、独立于任务方向晋级")
    if len(value.get("confirmation_execution_sha256", "")) != 64:
        raise ValueError("连续确认必须绑定明确的执行合同")
    if branch == "spin" and (not _finite_number(value["spin_growth"]) or value["spin_growth"] <= 1):
        raise ValueError("Spin增长倍率必须有限且大于1")
    cap = value.get("spin_terminal_cap")
    if branch == "spin":
        if set(value["targets"]) != {"positive", "negative"}:
            raise ValueError("Spin必须分别声明正反方向")
        if cap is not None and (not _finite_number(cap) or not 0 < cap <= MAX_COMMAND_MAGNITUDE):
            raise ValueError("Spin有限上限必须为可表示的正数，或null")
    for axis, levels in value["targets"].items():
        if not levels or any(
            len(target) != 3
            or not all(_finite_number(x) and abs(x) <= MAX_COMMAND_MAGNITUDE for x in target)
            or not value.get("height_range_m", [0.30, 0.40])[0] <= target[2] <= 0.40
            for target in levels
        ):
            raise ValueError("分支命令必须有限且处于现有高度合同: " + axis)
        if branch == "motion" and any(
            abs(vx) > 3.6 or (vx == 0 and abs(yaw) > 2) for vx, yaw, _ in levels
        ):
            raise ValueError("Motion直线/普通自转超出声明上限")
        if branch == "spin" and any(vx != 0 or yaw == 0 for vx, yaw, _ in levels):
            raise ValueError("Spin主任务必须是有限非零原地自转")
        if branch == "spin" and (
            len(levels) != 1
            or levels[0][1] * (1 if axis == "positive" else -1) <= 0
            or (cap is not None and abs(levels[0][1]) > cap)
        ):
            raise ValueError("Spin起点的方向/符号/有限上限不一致")
    for axis, target in value["anchors"].items():
        if (
            len(target) != 3
            or not all(_finite_number(x) and abs(x) <= MAX_COMMAND_MAGNITUDE for x in target)
            or not 0.30 <= target[2] <= 0.40
        ):
            raise ValueError("复习锚点必须有限且在已有高度合同内")
        if branch == "spin" and (
            axis not in ("positive", "negative")
            or target[0] != 0
            or target[1] * (1 if axis == "positive" else -1) <= 0
            or (cap is not None and abs(target[1]) > cap)
        ):
            raise ValueError("Spin复习锚点方向或上限不一致")
    return value


class SplitCourse:
    maneuver = True
    response_checked = True
    mixtures = staticmethod(mixtures)

    def __init__(self, version):
        self.VERSION = version

    def initial_state(self, configuration, previous=None):
        cfg = copy.deepcopy(validate_configuration(configuration))
        if cfg["version"] != self.VERSION:
            raise ValueError("分支课程与配置版本不同")
        prior = previous or {}
        frontier = cfg["initial_DR_level"]
        result = dict(
            version=self.VERSION,
            branch=cfg["branch"],
            configuration=cfg,
            stage=0,
            stage_started_update=0,
            last_training_update=-1,
            general_DR_frontier=frontier,
            standing_DR_frontier=frontier,
            standing_DR_level=-1,
            DR_levels={f: frontier for f in FAMILIES},
            dq_level=frontier,
            levels={axis: 0 for axis in cfg["targets"]},
            streaks={axis: 0 for axis in (*cfg["targets"], "DR")},
            regressions={axis: 0 for axis in cfg["targets"]},
            accepted_by_stage={},
            assessment_keys=[],
            DR_episode_counters=copy.deepcopy(prior.get("DR_episode_counters", [])),
            source_course_is_history_only=True,
            confirmation_conditions={},
            performance_history={},
            progress_history=[],
        )
        if cfg["branch"] == "spin":
            result["spin_targets"] = {a: t[0][1] for a, t in cfg["targets"].items()}
            result["spin_target_history"] = {a: [t[0][1]] for a, t in cfg["targets"].items()}
        self._rebind_confirmations(result, envs=None)
        return result

    def _target(self, state, axis):
        cfg = state["configuration"]
        if type(state["levels"][axis]) is not int or state["levels"][axis] < 0:
            raise ValueError("任务档位必须是非负整数")
        if cfg["branch"] == "spin":
            target = [0.0, state["spin_targets"][axis], cfg["targets"][axis][0][2]]
            cap = cfg["spin_terminal_cap"]
            if (
                not _finite_number(target[1])
                or abs(target[1]) > MAX_COMMAND_MAGNITUDE
                or target[1] * (1 if axis == "positive" else -1) <= 0
                or (cap is not None and abs(target[1]) > cap)
            ):
                raise ValueError("实际Spin端点方向/上限/数值范围错误")
            history = state["spin_target_history"][axis]
            if len(history) != state["levels"][axis] + 1 or history[-1] != target[1]:
                raise ValueError("Spin实际端点历史与档位不一致")
            return target
        return cfg["targets"][axis][state["levels"][axis]]

    def _next_spin_target(self, state, axis):
        current = abs(self._target(state, axis)[1])
        cfg = state["configuration"]
        upper = cfg["spin_terminal_cap"] or MAX_COMMAND_MAGNITUDE
        # Divide before multiply: no exponent or overflowing intermediate value.
        magnitude = upper if current >= upper / cfg["spin_growth"] else current * cfg["spin_growth"]
        return math.copysign(magnitude, state["spin_targets"][axis])

    def _conditions(self, state, envs):
        cfg = state["configuration"]
        # Keys describe the distribution, not env slot counts or unrelated revision.
        rows = self.task_rows(state, envs or 2560)
        base = dict(
            execution_sha256=cfg["confirmation_execution_sha256"],
            command_program_version=cfg["command_program_version"],
            DR_context=dict(
                general_level=state["general_DR_frontier"],
                family_levels=copy.deepcopy(state["DR_levels"]),
                dq_level=state["dq_level"],
            ),
            acceptance={
                k: cfg[k]
                for k in (
                    "minimum_ended",
                    "minimum_complete_fraction",
                    "task_pass_fraction",
                    "regression_pass_fraction",
                    "confirmations",
                    "regression_confirmations",
                )
            },
        )
        result = {}
        for axis in (*cfg["targets"], "DR"):
            chosen = [
                r
                for r in rows
                if r["count"] > 0
                and (
                    (not r["purpose"].startswith("focus_"))
                    if axis == "DR"
                    else (r["axis"] == axis and r["purpose"].startswith("focus_"))
                )
            ]
            condition = dict(
                base,
                axis=axis,
                rows=[
                    {
                        k: r[k]
                        for k in (
                            "target",
                            "goal_height",
                            "profile",
                            "amplitude",
                            "dq_amplitude",
                            "mode",
                            "sign",
                            "prefix",
                        )
                    }
                    for r in chosen
                ],
            )
            result[axis] = dict(sha256=json_digest(condition), conditions=condition)
        return result

    def _rebind_confirmations(self, state, *, envs):
        identities = self._conditions(state, envs)
        previous = state.setdefault("confirmation_conditions", {})
        for axis, identity in identities.items():
            if previous.get(axis) != identity:
                state["streaks"][axis] = 0
                if axis in state["regressions"]:
                    state["regressions"][axis] = 0
        state["confirmation_conditions"] = identities

    def task_rows(self, state, envs):
        cfg = validate_configuration(state["configuration"])
        if state["version"] != self.VERSION or envs < 1:
            raise ValueError("分支课程身份或环境数无效")
        amplitude = cfg["DR_amplitudes"][state["general_DR_frontier"]]
        rows = []

        def add(count, target, profile, purpose, *, axis="standing", mode=-1, sign=0):
            vx, yaw, h = target
            rows.append(
                dict(
                    target=[vx, yaw],
                    goal_height=h,
                    profile=profile,
                    purpose=purpose,
                    count=count,
                    prefix=0,
                    mode=mode,
                    sign=sign,
                    axis=axis,
                    endpoint=True,
                    amplitude=amplitude if profile >= 0 else 0.0,
                    dq_amplitude=amplitude if profile in (0, 10, 11, 12) else 0.0,
                )
            )

        names = list(cfg["fractions"])
        quotas = [int(envs * cfg["fractions"][name]) for name in names]
        for i in range(envs - sum(quotas)):
            quotas[i % len(quotas)] += 1
        for name, quota in zip(names, quotas, strict=True):
            if name == "clean":
                add(quota, (0, 0, 0.34), -1, "clean_standing")
            elif name == "dq":
                for p, n in zip((10, 11, 12), split(quota, 3), strict=True):
                    add(n, (0, 0, 0.34), p, f"dq_standing_{p}")
            elif name == "combined":
                add(quota, (0, 0, 0.34), 0, "dq_plus_general_DR")
            elif name == "push":
                add(quota, (0, 0, 0.34), 8, "recovery_initial_push")
            elif name == "modal":
                for j, n in enumerate(split(quota, 16)):
                    mode, sign, profile = j // 4, (-1, 1)[j // 2 % 2], (-1, 0)[j % 2]
                    add(
                        n,
                        (0, 0, 0.34),
                        profile,
                        f"modal_{mode}_{sign}_{profile}",
                        mode=mode,
                        sign=sign,
                    )
            elif name == "anchors":
                cases = [(t, p, a) for a, t in cfg["anchors"].items() for p in (-1, 0)]
                for n, (target, p, axis) in zip(split(quota, len(cases)), cases, strict=True):
                    add(n, target, p, f"anchor_{axis}_{p}", axis=axis)
            elif name == "focus":
                cases = [(a, p) for a in cfg["targets"] for p in (-1, 0)]
                for n, (axis, p) in zip(split(quota, len(cases)), cases, strict=True):
                    level = state["levels"][axis]
                    target = self._target(state, axis)
                    if not all(math.isfinite(v) for v in target):
                        raise ValueError("实际自转端点必须有限")
                    add(n, target, p, f"focus_{axis}_{level}_{p}", axis=axis)
            else:
                raise ValueError("未知分支采样类别: " + name)
        if sum(r["count"] for r in rows) != envs or len(rows) > 128:
            raise ValueError("分支cohort容量或配额错误")
        return rows

    def stage_config(self, stage, envs, *, state, families=FAMILIES):
        if stage != state["stage"]:
            raise ValueError("分支课程revision不同")
        rows = self.task_rows(state, envs)
        result = stage_from_rows(stage, envs, rows, self.VERSION)
        result.update(
            command_program_version=state["configuration"]["command_program_version"],
            goal_heights_m=[r["goal_height"] for r in rows],
            cohort_keys=[
                [*r["target"], 0.34, r["goal_height"], r["profile"], r["purpose"]] for r in rows
            ],
            maneuver_tasks=[r["axis"] for r in rows],
            height_width_mix=[0.25] * len(rows),
            terrain_families=[
                state["configuration"].get("terrain_by_axis", {}).get(r["axis"], "flat")
                for r in rows
            ],
        )
        return result

    def advance(self, state, report, update, *, envs, families=FAMILIES):
        from .maneuver_statistics import RESPONSE_VERSION

        result = copy.deepcopy(state)
        cfg = result["configuration"]
        expected = self.stage_config(state["stage"], envs, state=state)
        if (
            report.get("version") != RESPONSE_VERSION
            or report.get("stage_revision") != state["stage"]
            or update <= state["last_training_update"]
            or [r["key"] for r in report["rows"]] != expected["cohort_keys"]
        ):
            raise ValueError("分支训练摘要过期或与实际采样不符")
        rows = [r for r in report["rows"] if r["slot_count"] > 0]
        self._rebind_confirmations(result, envs=envs)
        for row in rows:
            n, c, p = row["ended_episodes"], row["completed_episodes"], row["pass_fraction"]
            if (
                type(n) is not int
                or type(c) is not int
                or not 0 <= c <= n
                or (n == 0 and p is not None)
                or (n > 0 and (p is None or not math.isfinite(p) or not 0 <= p <= 1))
            ):
                raise ValueError("分支完整回合计数/UNKNOWN语义错误")

        def covered(rs):
            return bool(rs) and all(r["ended_episodes"] >= cfg["minimum_ended"] for r in rs)

        def good(rs):
            return covered(rs) and all(
                r["pass_fraction"] >= cfg["task_pass_fraction"]
                and r["completed_episodes"] / r["ended_episodes"]
                >= cfg["minimum_complete_fraction"]
                for r in rs
            )

        decision, changed = {}, False
        for axis in cfg["targets"]:
            rs = [r for r in rows if r["axis"] == axis and r["purpose"].startswith("focus_")]
            known, passed = covered(rs), good(rs)
            # UNKNOWN中断连续确认，但不制造失败或触发回退。
            result["streaks"][axis] = result["streaks"][axis] + 1 if passed else 0
            bad = known and any(r["pass_fraction"] < cfg["regression_pass_fraction"] for r in rs)
            result["regressions"][axis] = result["regressions"][axis] + 1 if bad else 0
            promoted = regressed = False
            if result["streaks"][axis] >= cfg["confirmations"]:
                level = result["levels"][axis]
                if cfg["branch"] == "spin":
                    next_yaw = self._next_spin_target(result, axis)
                    promoted = abs(next_yaw) > abs(result["spin_targets"][axis])
                    result["spin_targets"][axis] = next_yaw
                    if promoted:
                        result["spin_target_history"][axis].append(next_yaw)
                else:
                    promoted = level + 1 < len(cfg["targets"][axis])
                result["levels"][axis] += int(promoted)
                result["streaks"][axis] = 0
            elif result["regressions"][axis] >= cfg["regression_confirmations"]:
                regressed = result["levels"][axis] > 0
                result["levels"][axis] = max(0, result["levels"][axis] - 1)
                if cfg["branch"] == "spin" and regressed:
                    result["spin_target_history"][axis].pop()
                    result["spin_targets"][axis] = result["spin_target_history"][axis][-1]
                result["regressions"][axis] = 0
                result["streaks"][axis] = 0
            changed |= promoted or regressed
            condition = result["confirmation_conditions"][axis]
            history = result["performance_history"].setdefault(
                condition["sha256"],
                dict(
                    axis=axis, conditions=condition["conditions"], known_windows=0, passed_windows=0
                ),
            )
            history.update(last_update=update, last_known=known, last_passed=passed)
            history["known_windows"] += int(known)
            history["passed_windows"] += int(passed)
            if promoted or regressed:
                movement = dict(
                    axis=axis,
                    update=update,
                    condition_sha256=condition["sha256"],
                    promoted=promoted,
                    regressed=regressed,
                    from_level=state["levels"][axis],
                    to_level=result["levels"][axis],
                    from_target=self._target(state, axis),
                    to_target=self._target(result, axis),
                )
                result["progress_history"].append(movement)
                if promoted:
                    result["accepted_by_stage"].setdefault(str(state["stage"]), []).append(
                        dict(
                            movement,
                            evidence="training_confirmation",
                            independent_certificate=False,
                        )
                    )
            decision[axis] = dict(
                covered=known, passed=passed, promoted=promoted, regressed=regressed
            )
        robust = [r for r in rows if not r["purpose"].startswith("focus_")]
        result["streaks"]["DR"] = result["streaks"]["DR"] + 1 if good(robust) else 0
        dr_up = result["streaks"]["DR"] >= cfg["confirmations"] and result[
            "general_DR_frontier"
        ] + 1 < len(cfg["DR_amplitudes"])
        if dr_up:
            result["general_DR_frontier"] += 1
            result["streaks"]["DR"] = 0
            changed = True
        if changed:
            result["stage"] += 1
            result["stage_started_update"] = update
        result["last_training_update"] = update
        frontier = result["general_DR_frontier"]
        result["standing_DR_frontier"] = result["dq_level"] = frontier
        result["DR_levels"] = {f: frontier for f in families}
        self._rebind_confirmations(result, envs=envs)
        result["assessment_keys"] = self.stage_config(result["stage"], envs, state=result)[
            "cohort_keys"
        ]
        return result, dict(
            version=self.VERSION,
            decisions=decision,
            DR_promoted=dr_up,
            state_revision_changed=changed,
        )
