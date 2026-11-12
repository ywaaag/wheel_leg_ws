"""一次解析课程版本；布局、来源准备和晋退级始终消费同一策略。"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from . import (
    c0_boundary_course,
    c0_course,
    c0_events_course,
    c0_fast_yaw_course,
    highspeed_course,
    maneuver_course,
    maneuver_course_v2,
    progressive_course,
)
from .split_course import VERSIONS as SPLIT_VERSIONS
from .split_course import SplitCourse

COURSES = {
    m.VERSION: m
    for m in (
        progressive_course,
        highspeed_course,
        maneuver_course,
        maneuver_course_v2,
        c0_course,
        c0_events_course,
        c0_boundary_course,
        c0_fast_yaw_course,
    )
}
LEGACY_VERSION = "motion31_supported2048_dqgain_ppo_v1"


@dataclass(frozen=True)
class CourseStrategy:
    implementation: object
    families: tuple[str, ...]
    legacy: bool = False

    @property
    def maneuver(self):
        return self.implementation in (maneuver_course, maneuver_course_v2) or bool(
            getattr(self.implementation, "maneuver", False)
        )

    @property
    def response_checked(self):
        return self.implementation is maneuver_course_v2 or bool(
            getattr(self.implementation, "response_checked", False)
        )

    def stage_config(self, stage, envs, *, state):
        kwargs = {} if self.legacy else {"families": self.families}
        return self.implementation.stage_config(stage, envs, state=state, **kwargs)

    def prepare(self, source, envs, *, previous_keys=None):
        state = copy.deepcopy(source["motion_course_state"])
        old_keys = previous_keys if previous_keys is not None else state.get("assessment_keys", [])
        module = self.implementation
        if (
            module is maneuver_course_v2
            and state.get("version") != module.VERSION
            or module is c0_fast_yaw_course
            and state.get("version") != module.VERSION
        ):
            state = module.migrate_state(
                state, source_update=source["online_updates_including_smoke"]
            )
        elif not self.legacy and state.get("version") != module.VERSION:
            raise ValueError("来源课程与目标版本不同，缺少显式迁移")
        stage = self.stage_config(state["stage"], envs, state=state)
        source["motion_course_state"] = state
        if not self.legacy:
            state["assessment_keys"] = stage["cohort_keys"]
            for name, symmetry in (
                ("reward_refinement_state", False),
                ("symmetry_mix_state", True),
            ):
                old = source.get(name, [])
                if old and len(old) != len(old_keys):
                    raise ValueError("来源cohort统计与布局keys长度不一致")
                by_key = {tuple(key): row for key, row in zip(old_keys, old, strict=False)}
                fresh = (module.mixtures if self.maneuver else progressive_course.mixtures)(
                    stage, symmetry=symmetry
                )
                source[name] = [
                    copy.deepcopy(by_key.get(tuple(key), row))
                    for key, row in zip(stage["cohort_keys"], fresh, strict=True)
                ]
        return state

    def initial_state(self, assets):
        module = self.implementation
        if isinstance(module, SplitCourse):
            return module.initial_state(assets["branch_course"])
        if module in (maneuver_course, maneuver_course_v2):
            return module.initial_state()
        if module is c0_fast_yaw_course:
            state = c0_boundary_course.initial_state(assets["source_boundary_frontiers"])
            return module.migrate_state(state, source_update=0)
        if module is c0_boundary_course:
            return module.initial_state(assets["source_boundary_frontiers"])
        if module in (c0_course, c0_events_course):
            return module.initial_state()
        raise ValueError("该课程未声明Actor-only初始化")

    def advance(self, state, review, update, *, envs):
        window = review.get("window")
        if window is not None:
            from native_stand.artifacts.io import json_digest

            if window == state.get("consumed_window"):
                if json_digest(review) != state.get("consumed_window_sha256"):
                    raise ValueError("相同课程窗口内容冲突")
                return copy.deepcopy(state), {
                    "state_revision_changed": False,
                    "duplicate_window": True,
                }
            previous = state.get("consumed_window")
            if previous and (
                window["run"] != previous["run"]
                or window["comparison_sha256"] != previous["comparison_sha256"]
                or window["start_update"] != previous["end_update"] + 1
            ):
                raise ValueError("课程窗口不连续或比较域冲突")
            if (
                window["end_update"] < window["start_update"]
                or window["stage_revision"] != state["stage"]
            ):
                raise ValueError("课程窗口身份错误")
        if self.legacy:
            from onpolicy import advance

            result, decision = advance(state, review, update)
        else:
            result, decision = self.implementation.advance(
                state, review, update, envs=envs, families=self.families
            )
        if window is not None:
            result["consumed_window"] = copy.deepcopy(window)
            result["consumed_window_sha256"] = json_digest(review)
        return result, decision


def resolve(version, *, frozen_course=None):
    if version in SPLIT_VERSIONS:
        return CourseStrategy(SplitCourse(version), tuple(progressive_course.FAMILIES))
    if version == LEGACY_VERSION:
        if frozen_course is None:
            import fixed_recovery as frozen_course
        return CourseStrategy(frozen_course, tuple(frozen_course.FAMILIES), legacy=True)
    try:
        module = COURSES[version]
    except KeyError as exc:
        raise ValueError(f"未知Motion课程版本: {version}") from exc
    return CourseStrategy(module, tuple(progressive_course.FAMILIES))
