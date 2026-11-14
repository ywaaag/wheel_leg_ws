"""统一worker的支撑中心Motion后端；复用冻结V4环境，不运行旧owner。"""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from native_stand.artifacts.io import file_sha256, read_json, write_json
from native_stand.artifacts.runtime_paths import RuntimePaths, execution_json
from native_stand.simulation import driver

from .observability import count, rebuild_attribution, rebuild_inputs, section

TRAINING_REPORT_EVERY_UPDATES = 100


def framework_binding(job, launcher):
    """Same declared SDK identity for training, collection and evaluation."""
    framework = job.assets.get("framework_dependency")
    if framework is None:
        return None
    import inspect

    ref = framework["manifest"]
    if file_sha256(ref["path"]) != ref["sha256"]:
        raise ValueError("固定IsaacLab运行依赖manifest身份不同")
    root = Path(framework["root"]).resolve()
    files = read_json(ref["path"])["files"]
    for row in files:
        if file_sha256(root / row["relative_path"]) != row["sha256"]:
            raise ValueError("固定IsaacLab运行依赖文件不同: " + row["relative_path"])
    actual = Path(inspect.getfile(launcher)).resolve()
    if not actual.is_relative_to(root):
        raise ValueError("实际AppLauncher未消费声明的IsaacLab依赖")
    return dict(manifest=ref, actual_launcher=str(actual), verified_files=len(files))


def validate_job(job):
    _validate_source_mode(job)
    optimization = job.assets.get("motion_execution_optimization")
    if optimization is not None:
        from .response_acceleration import VERSION

        if (
            set(optimization) != {"version", "compile_tensor_tick"}
            or optimization["version"] != VERSION
            or type(optimization["compile_tensor_tick"]) is not bool
        ):
            raise ValueError("未知V4执行加速合同")
    from native_stand.tasks.motion.spec import build_task

    # The worker consumes this exact Task, not just the decoder JSON.
    build_task(job)
    from native_stand.tasks.motion.course_strategy import LEGACY_VERSION, resolve

    if job.curriculum.version != LEGACY_VERSION:
        strategy = resolve(job.curriculum.version)
        from native_stand.tasks.motion.split_course import SplitCourse, validate_configuration

        if isinstance(strategy.implementation, SplitCourse):
            configuration = validate_configuration(job.assets["branch_course"])
            if configuration != job.curriculum.stages[0]:
                raise ValueError("分支配置与正式课程声明不同")
            from native_stand.tasks.motion.branch_state import confirmation_execution_identity

            if configuration["confirmation_execution_sha256"] != confirmation_execution_identity(
                job
            ):
                raise ValueError("分支连续确认执行合同与实际作业不一致")
            from native_stand.artifacts.branch_initialization import read_branch

            branch = read_branch(job.assets)
            if branch is None or branch["branch"] != configuration["branch"]:
                raise ValueError("新目标分支必须显式绑定选择与停止结算")
            from native_stand.artifacts.branch_initialization import (
                COURSE_RECORDS,
                PREPARATION,
            )

            if branch["version"] == PREPARATION and (
                job.limits.smoke_updates
                or job.limits.formal_updates
                or job.assets.get("preparation_only") is not True
            ):
                raise ValueError("课程准备尚未结算接续，只允许零额度CPU准备")
            if branch["version"] in COURSE_RECORDS:
                from native_stand.artifacts.io import json_digest

                if json_digest(configuration) != branch["target_course_sha256"]:
                    raise ValueError("课程准备目标与SHA绑定记录不同")
            if job.limits.smoke_updates or job.limits.formal_updates:
                from native_stand.artifacts.branch_initialization import require_allocated

                require_allocated(job)
    if job.assets.get("motor_limits"):
        from native_stand.tasks.motion.execution_limits import validate

        if tuple(job.control.effort_limits_nm) != validate(job.assets["motor_limits"]).caps:
            raise ValueError("Motion运行限矩与执行合同不同")
    if job.assets.get("delayed_execution"):
        from native_stand.tasks.motion.delayed_execution import validate_configuration

        validate_configuration(job.assets["delayed_execution"])
    if job.assets.get("terrain_scene"):
        from native_stand.simulation.terrain_layout import validate as validate_terrain

        validate_terrain(job.assets["terrain_scene"])
    if job.assets.get("wheel_contact_dr"):
        from native_stand.tasks.motion.contact_randomization import CRITIC_VERSION, DIM, validate

        validate(job.assets["wheel_contact_dr"])
        if job.model.critic_input != DIM or job.model.critic_privileged_version != CRITIC_VERSION:
            raise ValueError("Wheel contact DR requires causal881 Critic")
    for name in (
        "runtime_template",
        "runtime_template_sha256",
        "frozen_source_root",
        "frozen_bundle",
        "prefix_onnx",
        "prefix_sha256",
        "physical_gpu_uuid",
        "historical_online_updates",
        "original_online_limit",
    ):
        if name not in job.assets:
            raise ValueError("Isaac支撑中心适配缺少 assets." + name)
    if (
        job.assets["historical_online_updates"]
        + job.limits.smoke_updates
        + job.limits.formal_updates
        > job.assets["original_online_limit"]
    ):
        raise ValueError("后继与历史用量超过原在线授权")
    from native_stand.tasks.motion.scale_corrections import validate_configuration

    validate_configuration(
        job.reward.fields["scale_corrections"],
        wheel_scale_rad_s=job.action.wheel_scale_rad_s,
        leg_scale_rad=job.action.leg_scale_rad,
    )
    return True


def available(job):
    """不导入Isaac、不查GPU的可用性判定；与通用适配器同一签名。"""
    import importlib.util

    template = job.assets.get("runtime_template")
    if not template or not Path(template).is_file():
        return False, f"runtime_template不存在: {template}"
    if importlib.util.find_spec("isaaclab") is None:
        return False, "当前解释器没有isaaclab；生产Motion需要Isaac Lab环境"
    return True, ""


def unavailable_reason(job):
    return available(job)[1]


def preflight(job):
    validate_job(job)
    from native_stand.artifacts.branch_initialization import require_allocated

    require_allocated(job)
    if file_sha256(job.initialization.checkpoint) != job.initialization.sha256:
        raise ValueError("最新完整来源SHA不匹配")
    for name in ("runtime_template", "prefix_onnx"):
        sha_name = "runtime_template_sha256" if name == "runtime_template" else "prefix_sha256"
        if file_sha256(job.assets[name]) != job.assets[sha_name]:
            raise ValueError("引用身份改变: " + name)
    template = execution_json(job.assets["runtime_template"], job.assets)
    for path_key, sha_key in (
        ("asset_path", "asset_sha256"),
        ("asset_module_path", "asset_module_sha256"),
        ("colleague_config_path", "colleague_config_sha256"),
    ):
        if file_sha256(template[path_key]) != template[sha_key]:
            raise ValueError("物理资产或资产模块身份改变: " + path_key)
    response = template["motion_identified_response"]
    if file_sha256(response["contract"]) != response["contract_sha256"]:
        raise ValueError("完整V4响应元组身份改变")
    references = {key: job.assets[key] for key in ("source_anchor_identity",) if key in job.assets}
    references.update(
        {
            key: ref
            for key, ref in job.assets.get("source_policy_retention", {}).items()
            if key in ("bank", "bank_metadata")
        }
    )
    for key, ref in references.items():
        if file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Motion依赖身份不同: " + key)
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("GPU预检不可用；无卡模式不创建Motion worker或消耗PPO额度") from exc
    rows = [s.strip().split(", ") for s in gpu.splitlines()]
    selected = [r for r in rows if r[0] == job.runtime.device.rsplit(":", 1)[-1]]
    if len(selected) != 1 or selected[0][1] != job.assets["physical_gpu_uuid"]:
        raise ValueError("GPU UUID或逻辑映射不匹配")
    if int(selected[0][2]) < 14000:
        raise RuntimeError("当前GPU空闲显存不足14GiB；不抢占或切换设备")
    if shutil.disk_usage(job.config_path.parent).free < 2 * 1024**3:
        raise RuntimeError("磁盘不足2GiB，不启动")
    duplicates = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        parent_name = job.assets.get("parent_run_name")
        if (
            parent_name
            and "/command_v6/train.py" in command
            and parent_name in command
            and "--run" in command
        ):
            duplicates.append(int(proc.name))
        if "-m native_stand.experiments.worker" in command and job.name in command:
            duplicates.append(int(proc.name))
    if duplicates:
        raise RuntimeError("同任务仍有训练worker: " + str(duplicates))
    return {"status": "PASSED", "gpu": selected[0], "new_budget": False}


class MotionProductionAdapter:
    """支撑中心Motion适配器：拥有全部可变运行状态。

    build() 构造环境/算法并恢复；update_once() 执行一次 rollout+受控PPO更新；
    after_update() 在计费之后推进课程；capture() 导出完整状态；close() 关闭
    当前拥有的环境与AppLauncher（各一次）。
    """

    def __init__(self):
        self.context = None
        self.job = None
        self.template = None
        self.app = None
        self.env = None
        self.obs = None
        self.algorithm = None
        self.source = None
        self.source_path = None
        self.source_sha = None
        self.state = None
        self.spec = None
        self.config = None
        self.prefix = None
        self.fixed_std = None
        self.modules = None
        self.course_update_base = 0
        self.restore_point_updates = 0
        self._closed = False
        self._physical = {}
        self._candidate_pending = None
        self.candidates = None
        self._needs_rebuild = False
        self.same_run = False

    # ---- 构造与恢复 ------------------------------------------------
    def build(self, context, stage, resume_from):
        from native_stand.artifacts.branch_initialization import require_allocated

        allocation = (
            require_allocated(context.job)
            if context.job.limits.smoke_updates or context.job.limits.formal_updates
            else None
        )
        if allocation is not None and context.run_dir.name != allocation["run_id"]:
            raise ValueError("有界分配只能由唯一声明的run消费")
        job = context.job
        validate_job(job)
        self.context = context
        self.job = job
        self.template = execution_json(job.assets["runtime_template"], job.assets)
        loaded = _resolve_source(job, resume_from)
        source = loaded.payload
        paths = RuntimePaths.from_assets(job.assets)
        for key in ("source_policy_retention", "source_anchor_identity"):
            if key in source:
                source[key] = paths.view(source[key])
        self.source, self.source_path, self.source_sha = source, loaded.path, loaded.sha256
        self.source_object = loaded
        from native_stand.artifacts.model_refs import ModelReference

        context.refs.register(
            ModelReference(
                loaded.sha256,
                str(loaded.path.resolve()),
                ("resume" if resume_from is not None else "initialization",),
                context.run_dir.name,
            )
        )
        for key in ("prefix_onnx", "prefix_output_contract"):
            path = job.assets.get(key)
            if path and Path(path).is_file():
                expected = job.assets[
                    "prefix_sha256" if key == "prefix_onnx" else "prefix_output_contract_sha256"
                ]
                if file_sha256(path) != expected:
                    raise ValueError("前控模型或合同身份不一致")
                context.refs.register(
                    ModelReference(
                        expected, str(Path(path).resolve()), ("control",), context.run_dir.name
                    )
                )
        from native_stand.artifacts.io import json_digest

        motion_candidates = candidate_strategy(self.job)

        candidate_config = job.diagnostics.get("training_candidate_selection")
        self.candidates = None
        # 配置初始化始终是新run；显式resume必须属于当前绝对run身份。
        same_run = self.same_run = resume_from is not None
        if same_run:
            from native_stand.artifacts.handoff import validate_resume_identity

            validate_resume_identity(resume_from, context.run_dir)
        self.source_candidate_history = (
            source.get("source_training_candidates")
            if same_run
            else source.get("training_candidates")
        )
        self.window_start = source.get("training_window_start_update", 1) if same_run else 1
        self._candidate_pending = source.get("pending_training_candidate") if same_run else None
        if candidate_config is not None:
            domain = json_digest(motion_candidates.comparison_context(job.as_run_record()))
            authority = context.refs.candidate_index(context.run_dir.name)
            previous = authority or (source.get("training_candidates") if same_run else None)
            self.candidates = previous or motion_candidates.empty_index(
                domain, version=candidate_config["version"]
            )
            if self.candidates["comparison_sha256"] != domain:
                raise ValueError("同run运动候选比较域不一致")
            for pool in self.candidates["candidates_by_objective"].values():
                for row in pool.values():
                    if file_sha256(row["checkpoint"]) != row["sha256"]:
                        raise ValueError("恢复候选文件身份不一致")
            if authority is not None:
                write_json(context.run_dir / "training_candidates.json", authority)
                context.store.update(
                    writer_pid=os.getpid(),
                    best_candidate=authority["best_candidate"],
                    best_standing_candidate=authority["best_standing_candidate"],
                )
            if same_run and self._candidate_pending is not None:
                self._publish_candidate(
                    context,
                    dict(
                        checkpoint=str(loaded.path),
                        sha256=loaded.sha256,
                        update=self._candidate_pending["through_update"],
                    ),
                )
        self.fresh_actor = job.initialization.mode == "actor_only" and resume_from is None
        if self.fresh_actor:
            import random

            import numpy as np
            import torch

            from native_stand.learning.ppo.actor_initialization import training_source

            random.seed(job.runtime.seed)
            np.random.seed(job.runtime.seed)
            torch.manual_seed(job.runtime.seed)
            contract = read_json(job.assets["output_contract"])
            if file_sha256(job.assets["output_contract"]) != job.assets["output_contract_sha256"]:
                raise ValueError("Actor-only输出合同SHA不一致")
            architecture = {
                "actor_hidden_dims": list(job.model.actor_hidden_dims),
                "critic_hidden_dims": list(job.model.critic_hidden_dims),
                "actor_input": job.model.actor_input,
                "critic_input": job.model.critic_input,
                "hidden_activation": "ELU(alpha=1)",
                "mean_output": "Tanh",
            }
            source = training_source(source, job, contract, architecture)
            self.source = source
        required = (
            "actor",
            "critic",
            "optimizer",
            "training_rng",
            "motion_course_state",
            "reward_refinement_state",
            "symmetry_mix_state",
            "network_architecture",
            "output_contract",
            "action_version",
        )
        if self.fresh_actor:
            required = tuple(name for name in required if name not in ("optimizer", "training_rng"))
        if any(name not in source for name in required):
            raise ValueError("来源不是同语义完整PPO checkpoint")
        self.restore_point_updates = (
            0 if self.fresh_actor else _check_source_progress(job, source, resume_from)
        )
        if resume_from is not None and self.restore_point_updates != context.updates_used:
            raise ValueError("恢复点计数与本run真实账本不同")
        if (
            not self.fresh_actor
            and job.ppo.critic_warmup_updates
            and "warmup_updates_done" not in source
        ):
            raise ValueError("完整恢复缺少warmup累计状态")
        self._prepare_training()
        self._open_environment()
        if same_run and source.get("completed_episode_statistics") is not None:
            self.env.onpolicy_episodes.restore_completed(source["completed_episode_statistics"])
        elif same_run and hasattr(self.env.onpolicy_episodes, "sample_accounting"):
            sample_state = source.get("motion_sample_accounting")
            if sample_state is not None:
                self.env.onpolicy_episodes.inherit_sample_totals(sample_state)
                if not source.get("motion_sample_accounting_revision_boundary", False):
                    self.env.onpolicy_episodes.sample_accounting.window_known = False
                    self.env.onpolicy_episodes.sample_accounting.unknown_reason = (
                        "checkpoint_missing_completed_window_snapshot"
                    )
            else:
                self.env.onpolicy_episodes.sample_accounting.restore_legacy()
        self._bind_sample_window()
        self._assemble_training()

    def _review_window(self, completed):
        from native_stand.artifacts.io import json_digest

        candidates = candidate_strategy(self.job)
        return {
            "run": read_json(self.context.run_dir / "run.json", {}).get(
                "logical_run_id", str(self.context.run_dir.resolve())
            ),
            "comparison_sha256": json_digest(
                candidates.comparison_context(self.job.as_run_record())
            )
            if self.job.diagnostics.get("training_candidate_selection")
            else self.job.curriculum.version,
            "stage_revision": self.state["stage"],
            "start_update": self.state.get("consumed_window", {}).get(
                "end_update", getattr(self, "window_start", 1) - 1
            )
            + 1,
            "end_update": completed,
        }

    def _bind_sample_window(self):
        bind = getattr(self.env.onpolicy_episodes, "bind_sample_window", None)
        if bind is not None:
            bind(self._review_window(self.context.updates_used))

    def _prepare_training(self):
        from native_stand.tasks.motion.course_strategy import resolve
        from native_stand.tasks.motion.legacy_spec_adapter import build_environment_spec

        self.course_update_base = int(
            self.source.get("course_updates", self.source["recipe_updates"])
        )
        identity = anchor_identity(self.job, self.source)
        if identity is not None:
            self.source["source_anchor_identity"] = identity
            from native_stand.artifacts.model_refs import ModelReference

            self.context.refs.register(
                ModelReference(
                    path=str(Path(identity["path"]).resolve()),
                    sha256=identity["sha256"],
                    roles=("teacher",),
                    run=self.context.run_dir.name,
                    reason="来源KL冻结Teacher；仅从显式合同引用解析",
                )
            )
        from native_stand.artifacts.model_refs import register_dependencies

        retention = self.job.assets.get("source_policy_retention", {})
        register_dependencies(
            self.context.refs,
            consumer=self.context.run_dir.name,
            references={
                name: retention[name] for name in ("bank", "bank_metadata") if name in retention
            },
        )
        references = {}
        for key, sha_key in (
            ("output_contract", "output_contract_sha256"),
            ("runtime_template", "runtime_template_sha256"),
        ):
            if key in self.job.assets:
                references[key] = dict(path=self.job.assets[key], sha256=self.job.assets[sha_key])
        for key, sha_key in (
            ("asset_path", "asset_sha256"),
            ("asset_module_path", "asset_module_sha256"),
            ("colleague_config_path", "colleague_config_sha256"),
        ):
            if key in self.template:
                references[key] = dict(path=self.template[key], sha256=self.template[sha_key])
        response = self.template["motion_identified_response"]
        for row in self.job.assets.get("terrain_scene", {}).get("assets", []):
            for key in ("usd", "heights"):
                references["terrain_" + row["family"] + "_" + key] = row[key]
        extension = self.job.assets.get("output_contract_extension")
        if extension:
            references["height_geometry"] = extension["geometry_evidence"]
            references["active_center_binding"] = self.job.assets["support_center_policy_binding"]
        if self.job.assets.get("motor_limits") or "contract" in response:
            references["V4"] = dict(path=response["contract"], sha256=response["contract_sha256"])
        branch = self.source.get("branch_initialization_record")
        if branch is not None:
            paths = RuntimePaths.from_assets(self.job.assets)
            references["branch_initialization"] = paths.view(
                self.job.assets["branch_initialization"]
            )
            from native_stand.artifacts.branch_initialization import (
                ALLOCATED_COURSE,
                CONTINUED_COURSE,
                COURSE_RECORDS,
            )

            from native_stand.artifacts.motion_actor_admission import VERSION as ACTOR_GRANT

            if branch["version"] == ACTOR_GRANT:
                for key in (
                    "source",
                    "output_contract",
                    "authorization",
                    "settlement",
                    "allocation",
                ):
                    references["branch_" + key] = paths.view(branch[key])
            elif branch["version"] in COURSE_RECORDS:
                for key in ("source_run", "observed_state", "observed_budget", "output_contract"):
                    references["branch_" + key] = paths.view(branch[key])
                if branch["version"] in (ALLOCATED_COURSE, CONTINUED_COURSE):
                    for key in (
                        "allocation",
                        "settlement",
                        *(("new_authorization",) if branch.get("new_authorization") else ()),
                    ):
                        references["branch_" + key] = paths.view(branch[key])
            else:
                for key in ("selection", "settlement"):
                    references["branch_" + key] = paths.view(branch[key])
                settlement = paths.read(branch["settlement"]["path"])
                for key in ("selected_checkpoint", "final_checkpoint"):
                    references[key] = {k: settlement[key][k] for k in ("path", "sha256")}
            if branch["version"] != ACTOR_GRANT:
                references["source_center_binding"] = paths.view(
                    branch["selection_record"]["source_center_binding"]
                )
        register_dependencies(
            self.context.refs, consumer=self.context.run_dir.name, references=references
        )
        self.course_strategy = resolve(self.job.curriculum.version)
        self.state = self.course_strategy.prepare(self.source, self.job.runtime.envs)
        if not self.same_run:
            # 新run：窗口命名空间重建；已掌握课程能力/档位保留，
            # 与旧窗口绑定的记录不得冒充新run证据。
            self.state.pop("consumed_window", None)
            self.state.pop("consumed_window_sha256", None)
        self.config = self.job.reward.fields["scale_corrections"]
        self.spec = build_environment_spec(
            self.template, self.source, self.job, stage_builder=self.course_strategy.stage_config
        )

    def _assemble_training(self):
        from native_stand.learning.ppo.supported import assemble_from_full
        from native_stand.students.resume import restore_rng

        self.algorithm = assemble_from_full(self.source, self.obs, self.job, fresh=self.fresh_actor)
        self.algorithm.measurement = getattr(self, "measurement", None)
        if not self.fresh_actor:
            restore_rng(self.source["training_rng"])
        self._verify_restore()
        self._write_restoration()
        self.fixed_std = self.algorithm.actor.log_std.detach().clone()
        if self.job.assets.get("terrain_scene"):
            import torch

            self.smoke_sums = torch.zeros(
                len(self.env.stage_spec["commands"]), 9, device=self.env.device
            )
            if self.same_run and self.source.get("smoke_actual_sums") is not None:
                self.smoke_sums.copy_(self.source["smoke_actual_sums"].to(self.env.device))

    def _open_environment(self):
        import signal

        job = self.job
        from .legacy_support import activate_view, loaded_legacy_sources

        activate_view(job.assets["frozen_source_root"], job.assets["frozen_bundle"])
        sys.path.insert(0, self.template["asset_module_root"])
        termination_handler = signal.getsignal(signal.SIGTERM)
        interrupt_handler = signal.getsignal(signal.SIGINT)
        from isaaclab.app import AppLauncher

        framework = framework_binding(job, AppLauncher)
        if framework is not None:
            write_json(
                self.context.run_dir / "source/framework_binding.json",
                framework,
            )

        self.app = AppLauncher(
            {
                "headless": job.runtime.headless,
                "device": job.runtime.device,
                "multi_gpu": False,
                # Worker must finalize Budget/state AFTER adapter.close returns.
                "fast_shutdown": False,
            }
        ).app
        signal.signal(signal.SIGTERM, termination_handler)
        signal.signal(signal.SIGINT, interrupt_handler)

        import onnx

        from native_stand.learning.ppo.onnx_prefix import from_onnx
        from native_stand.learning.ppo.supported import assemble_from_full
        from native_stand.recovery.controlled_ppo import KEY
        from native_stand.students.motion_hotpath import runner_flags
        from native_stand.students.motion_statistics import MotionStatistics
        from native_stand.students.resume import restore_rng, rng_state
        from native_stand.students.standing_survival_task import bootstrap_reward
        from native_stand.students.step_checks import StepChecks
        from native_stand.students.training import observations
        from native_stand.tasks.motion.legacy_spec_adapter import build_environment_spec
        from native_stand.tasks.motion.value_diagnostics import from_storage as value_diagnostics

        from .environment import environment

        self.modules = {
            "assemble_from_full": assemble_from_full,
            "KEY": KEY,
            "environment": environment,
            "runner_flags": runner_flags,
            "value_diagnostics": value_diagnostics,
            "MotionStatistics": MotionStatistics,
            "restore_rng": restore_rng,
            "rng_state": rng_state,
            "bootstrap_reward": bootstrap_reward,
            "StepChecks": StepChecks,
            "observations": observations,
            "build_environment_spec": build_environment_spec,
            "loaded_legacy_sources": loaded_legacy_sources,
        }
        self.prefix = from_onnx(onnx.load(job.assets["prefix_onnx"]))
        self.env, self.obs = self._make_env()

    def _make_env(self):
        job = self.job
        result = _build_environment(
            job,
            self.spec,
            self.config,
            self.prefix,
            self.source,
            self.modules["environment"],
            self.modules["observations"],
            train=True,
            measurement=getattr(self, "measurement", None),
            construction_reason=getattr(self, "_construction_reason", "initial_build"),
        )
        self._environment_inputs = rebuild_inputs(self.spec, job, stage=result[0].stage_spec)
        return result

    def _rebuild_environment(self):
        """Isaac单例：释放旧环境后才构造新环境；失败保持显式无环境状态。"""
        old = self.env
        samples = getattr(getattr(old, "onpolicy_episodes", None), "sample_accounting", None)
        sample_state = samples.state_dict() if samples is not None else None
        self.env = None
        self.obs = None
        measurement = getattr(self, "measurement", None)
        attribution = getattr(self, "_rebuild_attribution", None)
        reasons = attribution["reasons"] if attribution else ["attribution_unavailable"]
        count(measurement, "motion.environment_rebuild")
        for reason in reasons:
            count(measurement, "motion.environment_rebuild", reason=reason)
        reason = "+".join(reasons)
        if old is not None:
            with section(
                measurement,
                "environment_close",
                parent="after_save_total",
                reason=reason,
            ):
                old.close()
        del old
        snapshot = self.modules["rng_state"]() if self.modules is not None else None
        self._construction_reason = reason
        try:
            self.env, self.obs = self._make_env()
            if sample_state is not None:
                self.env.onpolicy_episodes.inherit_sample_totals(sample_state)
            self._bind_sample_window()
        finally:
            self._construction_reason = "initial_build"
            if snapshot is not None:
                self.modules["restore_rng"](snapshot)

    def _verify_restore(self):
        import torch

        algorithm, source = self.algorithm, self.source
        actor_equal = all(
            torch.equal(v.cpu(), algorithm.actor.state_dict()[k].cpu())
            for k, v in source["actor"].items()
        )
        critic_equal = all(
            torch.equal(v.cpu(), algorithm.critic.state_dict()[k].cpu())
            for k, v in source["critic"].items()
        )
        if not actor_equal or not critic_equal:
            raise RuntimeError("完整Actor/Critic恢复数值不一致")
        self._actor_equal, self._critic_equal = actor_equal, critic_equal

    def _write_restoration(self):
        job = self.job
        from native_stand.students.motion_hotpath import readback as performance_readback

        from .fullchain import runtime_readback

        base_obs = dict(self.obs)
        base_obs["student_critic"] = self.obs["student_critic"][:, :863]
        readback = runtime_readback(self.env, base_obs)
        if job.model.critic_input in (877, 881):
            from .maneuver_planned import readback as planned_readback

            readback["planned_Critic"] = planned_readback(self.env, self.obs)
        if getattr(self.env, "motion_terrain_readback", None) is not None:
            readback["terrain"] = self.env.motion_terrain_readback
        readback["performance"] = performance_readback(self.env)
        if job.assets.get("delayed_execution") is not None:
            from .delayed_execution import readback as timing_readback

            readback["execution_timing"] = timing_readback(self.env, self.obs)
        write_json(
            self.context.run_dir / "source/restoration.json",
            {
                "status": "PASSED",
                "source_checkpoint": str(self.source_path),
                "source_sha256": self.source_sha,
                "source_update": self.source["update"],
                "historical_online_updates": job.assets["historical_online_updates"],
                "restore_point_run_updates": self.restore_point_updates,
                "actor_equal": self._actor_equal,
                "critic_equal": self._critic_equal,
                "optimizer_state_count": len(self.algorithm.optimizer.state),
                "std": self.algorithm.actor.log_std.exp().detach().cpu().tolist(),
                "RNG_restored_after_environment_construction": not self.fresh_actor,
                "warmup_repeated": False,
                "course_state_restored": not self.fresh_actor
                and (self.same_run or not job.assets.get("branch_initialization")),
                "new_target_branch": bool(job.assets.get("branch_initialization")),
                "source_model_online_updates": self.source.get("model_online_updates"),
                "physical_state_restored": False,
                "prefix_identity": job.assets["prefix_onnx"],
                "prefix_SHA": job.assets["prefix_sha256"],
                "prefix_decoder": read_json(job.assets["prefix_output_contract"])["action_version"],
                "new_scale_corrections": dict(self.config),
                "runtime": readback,
            },
        )
        write_json(
            self.context.run_dir / "source/legacy_environment_sources.json",
            self.modules["loaded_legacy_sources"](
                job.assets["frozen_source_root"], job.assets["frozen_bundle"]
            ),
        )

    # ---- 一次更新 --------------------------------------------------
    def update_once(self, context, index):
        import torch

        job, env, algorithm, spec = self.job, self.env, self.algorithm, self.spec
        obs = self.obs
        KEY = self.modules["KEY"]
        profile = bool(job.diagnostics.get("performance_measurement"))
        cuda_profile = profile and torch.device(env.device).type == "cuda"
        if cuda_profile:
            torch.cuda.synchronize(env.device)
            torch.cuda.reset_peak_memory_stats(env.device)
        tick_start = time.monotonic()
        phase_times = {}

        def profile_boundary(name, previous):
            # Opt-in phase timing only. Never synchronize each policy/physics step.
            if cuda_profile:
                torch.cuda.synchronize(env.device)
            now = time.monotonic()
            phase_times[name] = now - previous
            return now

        stage_spec = env.stage_spec
        report_due = (context.updates_used + 1) % TRAINING_REPORT_EVERY_UPDATES == 0
        stats = (
            self.modules["MotionStatistics"](
                stage_spec["commands"],
                env.device,
                schedule_s=spec["student_training"]["train_schedule_s"],
                parking_s=spec["student_training"]["parking_s"],
                fixed_shape=True,
            )
            if report_due
            else None
        )
        from native_stand.tasks.motion.execution_statistics import ExecutionStatistics

        execution_stats = (
            ExecutionStatistics(stage_spec["commands"], env.device) if report_due else None
        )
        reward_sum = torch.zeros((), device=env.device)
        nonzero = torch.zeros((), device=env.device, dtype=torch.long)
        ended = torch.zeros((), device=env.device, dtype=torch.long)
        channel_clips = torch.zeros(6, device=env.device)
        valid_samples = torch.zeros((), device=env.device)
        with torch.inference_mode():
            for _step in range(job.runtime.steps_per_env):
                if context.stop_requested():
                    self.obs = obs
                    return {"incomplete": True}
                valid = obs[KEY].squeeze(-1)
                if stats is not None:
                    age = env.rise_hold.episode.steps.float() * 0.01
                    ids = env.command_index.clone()
                env.motion_defer_post_checks = True
                raw_action = algorithm.act(obs)
                next_obs, reward, terminated, truncated, extra = env.step(raw_action)
                if hasattr(self, "smoke_sums"):
                    s = env.recovery_snapshot
                    p = env.standing_snapshot
                    values = torch.stack(
                        (
                            valid.float(),
                            (terminated | truncated).float(),
                            truncated.float(),
                            terminated.float(),
                            valid * env.motion_action_command[:, 2],
                            valid * s[:, 6],
                            valid * p[:, 1],
                            valid * (s[:, 10:12] > 2).all(-1),
                            valid * s[:, 7],
                        ),
                        -1,
                    )
                    self.smoke_sums.index_add_(0, env.command_index, values)
                checks = self.modules["StepChecks"]()
                checks.require(~env.recovery_snapshot[:, 4].bool(), "runtime fault")
                checks.require(~env.recovery_snapshot[:, 30].bool(), "initialization rejected")
                checks.require(torch.isfinite(reward), "nonfinite reward")
                checks.require(valid == env.student_actor_valid, "Actor ownership mismatch")
                pending = getattr(env, "motion_post_physics_checks", None)
                if pending is not None:
                    checks.flags.extend(pending.flags)
                    checks.names.extend(pending.names)
                    env.motion_post_physics_checks = None
                try:
                    timeout = self.modules["runner_flags"](checks, truncated)
                except RuntimeError as error:
                    try:
                        self.record_runtime_fault()
                    except Exception as diagnostic_error:
                        error.add_note(f"runtime fault report failed: {diagnostic_error!r}")
                    raise
                stored = reward
                if timeout:
                    if env.student_terminal_critic is None:
                        raise RuntimeError("缺少超时前Critic快照")
                    stored = self.modules["bootstrap_reward"](
                        reward,
                        terminated,
                        truncated,
                        algorithm.critic({"student_critic": env.student_terminal_critic}).squeeze(
                            -1
                        ),
                        gamma=algorithm.gamma,
                    )
                if execution_stats is not None:
                    execution_stats.observe(
                        ids,
                        valid,
                        raw_action,
                        env.supported_requested_target_post,
                        env.supported_adopted_target_post,
                        env.supported_saturation_post,
                    )
                if stats is not None:
                    stats.observe(
                        ids,
                        valid,
                        age,
                        env.motion_action_command,
                        env.recovery_snapshot,
                        env.standing_snapshot,
                        env.wheel_geometry_snapshot,
                    )
                next_obs = self.modules["observations"](env, next_obs)
                next_obs["motion_diagnostic_cohort"] = env.command_index[:, None].float()
                annotate_retention(next_obs, env, job)
                algorithm.process_env_step(
                    next_obs,
                    stored,
                    terminated | truncated,
                    {k: v for k, v in extra.items() if k != "time_outs"},
                )
                reward_sum += reward.sum()
                nonzero += (reward != 0).sum()
                ended += (terminated | truncated).sum()
                channel_clips += ((raw_action.abs() > 1) & valid[:, None]).sum(0)
                valid_samples += valid.sum()
                obs = next_obs
            if profile:
                phase_start = profile_boundary("rollout_seconds", tick_start)
            algorithm.compute_returns(obs)
            if profile:
                phase_start = profile_boundary("returns_seconds", phase_start)
        self.obs = obs
        diagnostics = (
            self.modules["value_diagnostics"](
                algorithm.storage, commands_override=stage_spec["commands"]
            )
            if report_due
            else {}
        )
        prior = int(self.source.get("warmup_updates_done", 0))
        algorithm.actor_update_enabled = (
            prior + context.updates_used - self.restore_point_updates
            >= job.ppo.critic_warmup_updates
        )
        if profile:
            phase_start = profile_boundary("diagnostics_seconds", phase_start)
        loss = algorithm.update()
        if not torch.equal(self.fixed_std, algorithm.actor.log_std):
            raise RuntimeError("固定探索std被改变")
        detail = algorithm.mask_summary
        self._update_scope = {
            "stats": stats,
            "execution_stats": execution_stats,
            "diagnostics": diagnostics,
            "detail": detail,
            "channel_clips": channel_clips,
            "valid_samples": valid_samples,
            "nonzero": nonzero,
        }
        if cuda_profile:
            torch.cuda.synchronize(env.device)
        tick_end = time.monotonic()
        elapsed = tick_end - tick_start
        if profile:
            phase_times["learning_seconds"] = tick_end - phase_start
        return {
            "reward_mean": float(reward_sum / job.runtime.samples_per_update),
            "nonzero_rewards": int(nonzero),
            "episode_completed": int(ended),
            "value_loss": loss["value"],
            "surrogate": loss["surrogate"],
            "entropy": loss["entropy"],
            "kl": detail["kl_valid_mean"],
            "clip_fraction": detail["clip_fraction_valid_mean"],
            "actor_gradient_norm": detail["actor_gradient_norm_mean"],
            "critic_gradient_norm": detail["critic_gradient_norm_mean"],
            "critic_warmup": int(detail["critic_warmup"]),
            "actor_optimizer_steps": detail["actor_optimizer_steps"],
            "critic_optimizer_steps": detail["critic_optimizer_steps"],
            "accepted_KL_peak": detail["Actor_post_Adam_accepted_KL_peak"],
            "actor_backtracks": detail["Actor_backtracks"],
            "seconds_per_update": elapsed,
            "performance_seconds": dict(phase_times),
            **phase_times,
            **(
                {
                    "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(env.device),
                    "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(env.device),
                }
                if cuda_profile
                else {}
            ),
            "actor_lr": job.ppo.learning_rate,
            "critic_lr": job.ppo.critic_learning_rate,
            "actor_steps": detail["actor_optimizer_steps"],
            "critic_only": bool(detail["critic_warmup"]),
        }

    # ---- 计费之后：课程推进 ----------------------------------------
    def after_update(self, context, index, metrics):
        completed = context.updates_used
        if hasattr(self, "smoke_sums") and (completed == 1 or completed % 16 == 0):
            write_json(
                context.run_dir / "source/smoke_actual_samples.json",
                dict(
                    through_updates=completed,
                    samples=self.smoke_sums.cpu().tolist(),
                    fields=[
                        "student_steps",
                        "ended",
                        "timeout",
                        "terminated",
                        "height_command_sum",
                        "actual_height_sum",
                        "roll_sum",
                        "dual_support_steps",
                        "vx_sum",
                    ],
                    keys=self.env.stage_spec["cohort_keys"],
                    terrain_families=self.env.stage_spec["terrain_families"],
                    independent_certificate=False,
                    extra_physics_steps=0,
                ),
            )
        scope = getattr(self, "_update_scope", {})
        if index == 1 and not (self.context.run_dir / "source/first_update.json").is_file():
            write_json(
                self.context.run_dir / "source/first_update.json",
                {
                    "status": "PASSED",
                    "actual_updates": completed,
                    "nonzero_rewards": int(scope.get("nonzero", 0)),
                    "valid_samples": int(scope.get("valid_samples", 0)),
                    "PPO": scope.get("detail"),
                    "physical_ticks": self.env.motion_timing.tick,
                    "physical_control": self.physical_readback(),
                },
            )
        if (
            completed % TRAINING_REPORT_EVERY_UPDATES != 0
            or self.state.get("consumed_window", {}).get("end_update") == completed
        ):
            return
        # 模型进度只加恢复点之后的新增部分，连续与分段路径得到同一编号。
        model_update = self.course_update_base + (completed - self.restore_point_updates)
        window = self._review_window(completed)
        if hasattr(self.env.onpolicy_episodes, "bind_sample_window"):
            review = self.env.onpolicy_episodes.export(model_update, window=window)
        else:
            review = self.env.onpolicy_episodes.export(model_update)
        review["window"] = window
        candidate_config = self.job.diagnostics.get("training_candidate_selection")
        if candidate_config is not None:
            from native_stand.artifacts.io import json_digest

            motion_candidates = candidate_strategy(self.job)

            motion_candidates.validate_configuration(candidate_config)
            review["course_version"] = self.job.curriculum.version
            identity = json_digest(motion_candidates.comparison_context(self.job.as_run_record()))
            evidence = motion_candidates.summarize(
                review, DR_level=self.state["general_DR_frontier"], comparison_sha256=identity
            )
            evidence["through_update"] = completed
            evidence["window"] = copy.deepcopy(review["window"])
            self._candidate_pending = evidence
        state, decision = self.course_strategy.advance(
            self.state, review, model_update, envs=self.job.runtime.envs
        )
        self.state = state
        write_json(
            self.context.run_dir / f"diagnostics/update_{completed:05}.json",
            {
                "update": completed,
                "onpolicy": review,
                "decision": decision,
                "cohort_statistics": scope["stats"].export(),
                "cohort_execution_statistics": scope["execution_stats"].export(),
                "cohort_value_diagnostics": scope["diagnostics"],
                "PPO": scope["detail"],
                "raw_clip_fraction_by_channel": (
                    scope["channel_clips"] / scope["valid_samples"].clamp_min(1)
                )
                .cpu()
                .tolist(),
                "event_coverage": self.env.transient_events.summary(),
                "physical_control": self.physical_readback(),
                "independent_evaluation": False,
                "physics_extra_steps": 0,
            },
        )
        if decision["state_revision_changed"]:
            previous_inputs = getattr(self, "_environment_inputs", None)
            if previous_inputs is None:
                previous_inputs = rebuild_inputs(self.spec, self.job, stage=self.env.stage_spec)
            self.state["DR_episode_counters"] = (
                self.env.identified_dr.episodes.detach().cpu().tolist()
            )
            self.source["motion_course_state"] = copy.deepcopy(self.state)
            self.state = self.course_strategy.prepare(
                self.source,
                self.job.runtime.envs,
                previous_keys=self.env.stage_spec.get("cohort_keys", []),
            )
            self.spec = self.modules["build_environment_spec"](
                self.template,
                self.source,
                self.job,
                stage_builder=self.course_strategy.stage_config,
            )
            self._rebuild_attribution = rebuild_attribution(
                previous_inputs, rebuild_inputs(self.spec, self.job)
            )
            metrics["environment_rebuild"] = copy.deepcopy(self._rebuild_attribution)
            self._needs_rebuild = True

    def after_save(self, context, reference):
        """先有完整checkpoint，再发布与重建；失败不会丢掉最后有效训练状态。"""
        with section(
            getattr(self, "measurement", None),
            "candidate_references_views",
            parent="after_save_total",
        ):
            self._publish_candidate(context, reference)
        if self._needs_rebuild:
            self._rebuild_environment()
            self._needs_rebuild = False

    def _publish_candidate(self, context, reference):
        """候选引用在公共driver轮换之前登记。"""
        if self._candidate_pending is None:
            return
        evidence = self._candidate_pending
        if evidence["through_update"] != reference["update"]:
            raise ValueError("候选窗口与实际保存checkpoint不一致")
        motion_candidates = candidate_strategy(self.job)

        previous = context.refs.candidate_index(context.run_dir.name) or self.candidates
        last = (previous or {}).get("last_review", {})
        same_checkpoint = all(
            (last.get("checkpoint") or {}).get(k) == reference.get(k)
            for k in ("checkpoint", "sha256", "update")
        )
        if same_checkpoint and last.get("evidence") == evidence:
            self.candidates = previous
        else:
            self.candidates = motion_candidates.consider(
                previous,
                evidence,
                reference,
                {"source_path": str(self.source_path), "source_sha256": self.source_sha},
            )
        self.candidates = context.refs.publish_candidates(context.run_dir.name, self.candidates)
        fields = {}
        for field in ("best_candidate", "best_standing_candidate"):
            candidate = self.candidates[field]
            fields[field] = candidate
        context.store.update(writer_pid=os.getpid(), **fields)
        write_json(context.run_dir / "training_candidates.json", self.candidates)
        self._candidate_pending = None

    # ---- 完整状态 --------------------------------------------------
    def capture(self, context, counters):
        job, env, source, state = self.job, self.env, self.source, self.state
        algorithm = self.algorithm
        state["DR_episode_counters"] = env.identified_dr.episodes.detach().cpu().tolist()
        data = {
            key: copy.deepcopy(source[key])
            for key in (
                "interface_version",
                "role",
                "dataset_sha256",
                "manifest_sha256",
                "asset_sha256",
                "network_architecture",
                "critic_privileged_version",
                "output_contract",
            )
            if key in source
        }
        for key in (
            "source_policy_retention",
            "source_anchor_actor",
            "source_anchor_identity",
            "execution_timing",
            "wheel_identification_record_sha256",
            "training_candidates",
            "initialization_record",
            "ancestral_online_updates",
            "warmup_updates_done",
            "branch_initialization_record",
            "source_course_history",
            "contact_critic_extension_receipt",
        ):
            if key in source:
                data[key] = copy.deepcopy(source[key])
        if self.candidates is not None:
            data["training_candidates"] = copy.deepcopy(self.candidates)
        if hasattr(self, "smoke_sums"):
            data["smoke_actual_sums"] = self.smoke_sums.detach().cpu().clone()
            data["output_contract_extension_receipt"] = self.source.get(
                "output_contract_extension_receipt"
            )
        if getattr(algorithm, "source_anchor", None) is not None:
            data["source_anchor_actor"] = algorithm.source_anchor.teacher.state_dict()
            if hasattr(algorithm.source_anchor, "state_dict"):
                data["source_anchor_runtime_state"] = algorithm.source_anchor.state_dict()
            data["source_policy_retention"] = copy.deepcopy(algorithm.source_anchor.config)
        data["warmup_updates_done"] = min(
            job.ppo.critic_warmup_updates,
            int(source.get("warmup_updates_done", 0))
            + context.updates_used
            - self.restore_point_updates,
        )
        if job.assets.get("delayed_execution") is not None:
            data["execution_timing"] = dict(job.assets["delayed_execution"])
            data["wheel_identification_record_sha256"] = job.assets["wheel_identification_sha256"]
        if job.assets.get("new_independent_authorization"):
            data["ancestral_online_updates"] = int(job.assets["source_online_updates"])
        current_rng = self.modules["rng_state"]()
        if (
            current_rng.get("torch_cuda") is None
            and source.get("training_rng", {}).get("torch_cuda") is not None
        ):
            current_rng["torch_cuda"] = copy.deepcopy(source["training_rng"]["torch_cuda"])
            data["CUDA_rng_CPU_preservation_only"] = True
        historical = int(job.assets["historical_online_updates"])
        # 恢复点已含当前run的一部分更新，不能在recipe累计上加两次。
        recipe = int(source.get("recipe_updates") or 0) + (
            context.updates_used - self.restore_point_updates
        )
        model_online = (
            int(
                source.get(
                    "model_online_updates",
                    source.get(
                        "online_updates_including_smoke", historical + self.restore_point_updates
                    ),
                )
            )
            + context.updates_used
            - self.restore_point_updates
        )
        data.update(
            actor=algorithm.actor.state_dict(),
            critic=algorithm.critic.state_dict(),
            optimizer=algorithm.optimizer.state_dict(),
            std=algorithm.actor.log_std.detach().clone(),
            std_parameterization="log_std",
            rng=current_rng,
            training_rng=current_rng,
            counters={
                "online_updates": historical + context.updates_used,
                "new_run_updates": context.updates_used,
                "resume_point_new_updates": self.restore_point_updates,
                "model_online_updates": model_online,
            },
            recipe_updates=recipe,
            course_updates=self.course_update_base
            + context.updates_used
            - self.restore_point_updates,
            online_updates_including_smoke=historical + context.updates_used,
            model_online_updates=model_online,
            motion_course_state=copy.deepcopy(state),
            reward_refinement_state=copy.deepcopy(source["reward_refinement_state"]),
            symmetry_mix_state=copy.deepcopy(source["symmetry_mix_state"]),
            action_version=job.action.version,
            fixed_std=True,
            BC_coefficient=0.0,
            source_checkpoint=str(self.source_path),
            source_sha256=self.source_sha,
            parent_update=source["update"],
            scale_corrections=dict(self.config),
            training_recipe=job.as_run_record()["ppo"],
        )
        data["source_training_candidates"] = copy.deepcopy(
            getattr(self, "source_candidate_history", None)
        )
        data["training_window_start_update"] = (
            state.get("consumed_window", {}).get("end_update", getattr(self, "window_start", 1) - 1)
            + 1
        )
        data["pending_training_candidate"] = copy.deepcopy(self._candidate_pending)
        samples = getattr(getattr(env, "onpolicy_episodes", None), "sample_accounting", None)
        if samples is not None:
            data["motion_sample_accounting"] = samples.state_dict()
            data["motion_sample_accounting_revision_boundary"] = self._needs_rebuild
        if (
            not self._needs_rebuild
            and hasattr(env, "onpolicy_episodes")
            and hasattr(env.onpolicy_episodes, "completed_state")
        ):
            data["completed_episode_statistics"] = env.onpolicy_episodes.completed_state()
        return data

    # ---- 物理读数与关闭 --------------------------------------------
    def record_runtime_fault(self):
        """Persist real PRE-RESET causes; never infer them from the reset state."""
        import math

        import torch

        env = self.env
        rows = torch.where(env.recovery_snapshot[:, 4].bool())[0][:16]

        def finite_json(value):
            if isinstance(value, list):
                return [finite_json(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            return value

        def selected(value):
            return finite_json(value[rows].detach().cpu().tolist())

        detail = dict(
            updates_used=self.context.updates_used,
            nonfinite_diagnostic_encoding="explicit nan/inf/-inf strings, never replaced by zero",
            physical_tick=env.motion_timing.tick,
            rows=rows.cpu().tolist(),
            recovery_snapshot=selected(env.recovery_snapshot),
            snapshot_fields={
                "fault": 4,
                "height": 6,
                "closure_m": 12,
                "chassis_clearance_m": 13,
                "minimum_gap_m": 14,
            },
            components={
                k: selected(v) for k, v in getattr(env, "student_fault_components", {}).items()
            },
            pre_reset_joints={
                k: selected(v) for k, v in getattr(env, "student_fault_joint_data", {}).items()
            },
            terrain_ids=selected(env.motion_terrain_ids)
            if hasattr(env, "motion_terrain_ids")
            else None,
            closure_fault_m=env.cfg.closure_fault_m,
            severe_penetration_m=env.recovery_config["initialization"]["severe_penetration_m"],
        )
        write_json(self.context.run_dir / "runtime_fault_detail.json", detail)

    def physical_readback(self):
        env = self.env
        checked = {}
        if self.job.assets.get("motor_limits"):
            from .motion48_execution import readback as motor_readback

            checked["motor_contract"] = motor_readback(
                env.motion_timing, self.job.assets["motor_limits"], self.obs
            )
        if self.job.assets.get("motor_limits") and self.job.model.critic_input in (877, 881):
            from .maneuver_planned import readback as planned_readback

            checked["causal_critic"] = planned_readback(env, self.obs)
        return {
            **checked,
            "execution_optimization": {
                **getattr(getattr(env.motion_timing, "leg", None), "acceleration_readback", {}),
                "actual_compiled_phases": len(
                    getattr(getattr(env.motion_timing, "leg", None), "_compiled_phases", ())
                ),
                "ground_query_cache_enabled": getattr(env, "motion_surface_query_cache", False),
            },
            "target_encoding": "physical_target",
            "legacy_control_snapshot_encoding": "legacy_bridge_1p5_100",
            "scope": "last completed policy step, physical six-channel arrays",
            "requested_target_mean": env.supported_requested_target_post.mean(0).cpu().tolist(),
            "adopted_target_mean": env.supported_adopted_target_post.mean(0).cpu().tolist(),
            "effective_caps_min_Nm": env.motion_timing.dr.caps.amin(0).cpu().tolist(),
            "effective_caps_max_Nm": env.motion_timing.dr.caps.amax(0).cpu().tolist(),
            "driver_at_cap_fraction": env.supported_saturation_post["driver_at_cap"]
            .float()
            .mean(0)
            .cpu()
            .tolist(),
            "motor_at_cap_fraction": env.supported_saturation_post["motor_at_cap"]
            .float()
            .mean(0)
            .cpu()
            .tolist(),
            "no_NET_in_motor_statistics": True,
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        failure = sys.exc_info()
        if failure[1] is not None and self.context is not None:
            import traceback

            write_json(
                self.context.run_dir / "engine_failure.json",
                {
                    "error": repr(failure[1]),
                    "traceback": "".join(traceback.format_exception(*failure)),
                    "updates_used": self.context.updates_used,
                    "physical_tick": getattr(
                        getattr(self.env, "motion_timing", None), "tick", None
                    ),
                    "reported_before_application_shutdown": True,
                },
            )
        env, app = self.env, self.app
        self.env = None
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                lifecycle = self.job.assets.get("application_lifecycle", {}) if self.job else {}
                if lifecycle.get("owner") == "worker_atexit":
                    # Some image launchers ignore fast_shutdown=False and call
                    # _exit. Finish public Budget/state/report before teardown.
                    import atexit

                    context = self.context

                    def shutdown():
                        status = context.store.read().get("status")
                        app.close(exit_code=0 if status in ("COMPLETED", "STOPPED_BOUNDED") else 1)

                    atexit.register(shutdown)
                else:
                    app.close()
        self.app = None


def run(context, *, stage=None, resume_from=None):
    """生产入口：构造/恢复交给adapter，调度/计费/保存交给公共驱动器。"""

    return driver.run_stage(
        context,
        MotionProductionAdapter(),
        stage=stage or context.stage,
        resume_from=resume_from,
    )


def _validate_source_mode(job):
    """Isaac只接受明确来源的Motion full或actor_only初始化；actor_only必须绑定C0合同。"""
    if job.task != "motion" or job.initialization.mode not in ("full", "actor_only"):
        raise ValueError("Isaac只支持明确来源的Motion full或actor_only初始化")
    if job.initialization.mode == "actor_only":
        c0 = job.action.version == "motion_multiheight_support_center_015_80_v1"
        wide = job.action.version == "motion_multiheight020_support_center_050_80_v1"
        if not c0 and not wide:
            raise ValueError("actor_only初始化必须绑定显式C0或050输出合同")
        if wide:
            from native_stand.artifacts.motion_actor_admission import read_admission

            read_admission(job)


def _resolve_source(job, resume_from):
    """三种来源只选一次：显式resume → 阶段接续 → 配置声明的初始化。

    resume_from 必须是 `load_source()` 返回的 LoadedSource；身份随数据一起传递，
    不存在“B的权重、A的身份”或按隐藏字段猜路径的分支。
    """
    from native_stand.artifacts.checkpoint import (
        LoadedSource,
        load_actor_source,
        load_complete_state,
    )

    if resume_from is not None:
        if not isinstance(resume_from, LoadedSource):
            raise TypeError(
                "resume_from必须是checkpoint.load_source()/load_complete_state()返回的"
                f"LoadedSource；收到{type(resume_from).__name__}"
            )
        return resume_from
    path = Path(job.initialization.checkpoint)
    loader = load_actor_source if job.initialization.mode == "actor_only" else load_complete_state
    source = loader(path)
    if job.initialization.sha256 and source.sha256 != job.initialization.sha256:
        raise ValueError("初始化来源SHA与声明不一致")
    if job.assets.get("branch_initialization") and not job.assets.get("wide_actor_admission"):
        from native_stand.artifacts.branch_initialization import validate_source
        from native_stand.tasks.motion.branch_state import prepare_branch_source

        record = validate_source(job, source)
        source = LoadedSource(
            prepare_branch_source(source.payload, job, record), source.path, source.sha256
        )
    return source


def _build_environment(
    job,
    spec,
    config,
    prefix,
    source,
    environment,
    observations,
    *,
    train: bool,
    measurement=None,
    construction_reason="initial_build",
):
    """生产语义的环境构造；训练与评估使用同一段代码。"""
    import torch

    parent = "adapter_build" if construction_reason == "initial_build" else "after_save_total"
    with section(
        measurement, "environment_construction", parent=parent, reason=construction_reason
    ):
        current = environment(
            spec,
            "MotionZeroStudent",
            envs=job.runtime.envs,
            job=job,
            source=source,
            prefix=prefix,
            corrections=config,
            **({"measurement": measurement} if measurement is not None else {}),
        )
    try:
        setter = getattr(current, "set_measurement", None)
        if setter is not None:
            setter(measurement)
        current._measurement_reset_parent = "first_reset_readback"
        current.curriculum_training = bool(train)
        current.student_fault_diagnostics_enabled = bool(
            job.diagnostics.get("runtime_fault_snapshot", False)
        )
        current.select_initial_family("motion_native_normal", "train" if train else "holdout")
        with section(
            measurement, "first_reset_readback", parent=parent, reason=construction_reason
        ):
            raw_obs, _ = current.reset()
            value = observations(current, raw_obs)
        current._measurement_reset_parent = "rollout_seconds"
        value["motion_diagnostic_cohort"] = current.command_index[:, None].float()
        annotate_retention(value, current, job)
        current.symmetry_mix = torch.tensor(
            [r["mix"] for r in source["symmetry_mix_state"]], device=current.device
        )[current.command_index]
        current.motion_cohort_mix.copy_(
            torch.tensor(
                [r["mix"] for r in source["reward_refinement_state"]], device=current.device
            )
        )
        return current, value
    except BaseException as failure:
        try:
            current.close()
        except Exception as cleanup:
            failure.add_note("Motion environment cleanup failed: " + repr(cleanup))
        raise


def prepare_session(job, *, resume_from=None, train: bool = False):
    """打开一个后端会话（AppLauncher一次），返回 (context, app)。

    条件环境由 EvaluationSession 用同一份 context 逐个创建；会话负责关闭 AppLauncher。
    """
    import sys

    template = execution_json(job.assets["runtime_template"], job.assets)
    loaded = _resolve_source(job, resume_from)
    source, source_path, source_sha = loaded.payload, loaded.path, loaded.sha256
    from .legacy_support import activate_view

    activate_view(job.assets["frozen_source_root"], job.assets["frozen_bundle"])
    sys.path.insert(0, template["asset_module_root"])
    import onnx
    from isaaclab.app import AppLauncher

    from native_stand.learning.ppo.onnx_prefix import from_onnx
    from native_stand.tasks.motion.course_strategy import LEGACY_VERSION, resolve
    from native_stand.tasks.motion.legacy_spec_adapter import build_environment_spec

    from .environment import environment

    if job.curriculum.version == LEGACY_VERSION:
        import fixed_recovery as course

        if not hasattr(course, "stage_config"):
            raise ValueError(
                "Historical course requires its explicit implementation; numeric ABI is not a course"
            )
        strategy = resolve(job.curriculum.version, frozen_course=course)
    else:
        strategy = resolve(job.curriculum.version)
    strategy.prepare(source, job.runtime.envs)
    context = {
        "job": job,
        "template": template,
        "source": source,
        "source_path": source_path,
        "source_sha": source_sha,
        "prefix": from_onnx(onnx.load(job.assets["prefix_onnx"])),
        "spec": build_environment_spec(template, source, job, stage_builder=strategy.stage_config),
        "config": job.reward.fields["scale_corrections"],
        "environment": environment,
        "train": train,
    }
    context["framework_binding"] = framework_binding(job, AppLauncher)
    app = AppLauncher(
        {
            "headless": job.runtime.headless,
            "device": job.runtime.device,
            "multi_gpu": False,
            "fast_shutdown": False,
        }
    ).app
    try:
        from native_stand.students.training import observations

        context["observations"] = observations
    except BaseException:
        app.close()
        raise
    return context, app


class ProductionEvalEnv:
    """把冻结V4环境包成评估器使用的统一接口。

    拥有 raw env 与 AppLauncher：close() 负责两者，避免重复关闭或漏关。
    """

    def __init__(self, env, app, job, apply_command):
        self.env = env
        self.app = app
        self.job = job
        self.num_envs = job.evaluation.envs
        self.device = getattr(env, "device", job.runtime.device)
        self._apply_command = apply_command
        self._last_terminal = None
        self._closed = False

    def observations(self):
        from native_stand.students.training import observations

        value = observations(self.env, self.env._get_observations())
        return value

    def set_command(self, command):
        # 只接受任务侧显式接口：写入后由step核对是否真的持续生效。
        applier = getattr(self.env, "set_evaluation_command", None)
        if applier is None:
            raise ValueError(
                "冻结环境缺少 set_evaluation_command(command) 评估接口；"
                "不能用写单个tensor替代，否则pre-step会覆盖命令"
            )
        applier(list(command))
        self._expected_command = list(command)

    def step(self, action):
        import torch

        expected_effective = None
        if hasattr(self.env, "evaluation_effective_command"):
            expected_effective = self.env.evaluation_effective_command().detach().clone()
        next_obs, reward, terminated, truncated, extras = self.env.step(action)
        terminated_t = torch.as_tensor(terminated, dtype=torch.bool).reshape(self.num_envs)
        truncated_t = torch.as_tensor(truncated, dtype=torch.bool).reshape(self.num_envs)
        terminal = getattr(self.env, "student_terminal_critic", None)
        self._last_terminal = terminal
        expected = getattr(self, "_expected_command", None)
        if expected is not None:
            effective = getattr(self.env, "motion_action_command", None)
            if effective is None:
                raise ValueError("无法核对评估命令是否生效；环境没有命令回读")
            # 核对全部应遵循该条件的计费环境，而不只是第0个。
            rows = effective.detach().reshape(-1, len(expected))
            mismatched = []
            for index in range(min(self.num_envs, rows.shape[0])):
                observed = rows[index].tolist()
                target = (
                    expected_effective[index].tolist()
                    if expected_effective is not None
                    else expected
                )
                if any(abs(a - b) > 1e-6 for a, b in zip(observed, target, strict=True)):
                    mismatched.append({"env_index": index, "observed": observed})
            if mismatched:
                raise ValueError(f"评估命令未持续生效: 期望{expected} 偏差环境{mismatched[:4]}")
        self._raw_obs = next_obs
        return {
            "rewards": torch.as_tensor(reward, dtype=torch.float32).reshape(self.num_envs, 1),
            "terminated": terminated_t.unsqueeze(-1),
            "truncated": truncated_t.unsqueeze(-1),
            "dones": (terminated_t | truncated_t).unsqueeze(-1),
            "reset_envs": int((terminated_t | truncated_t).sum()),
            "extras": extras if isinstance(extras, dict) else {},
            "terminal_observations": terminal,
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.env.close()
        finally:
            if self.app is not None:
                self.app.close()
            session = getattr(self, "_session", None)
            if session is not None:
                session.close()


class EvaluationSession:
    """一次评估一个后端会话：AppLauncher只启动一次，条件环境各自创建与关闭。"""

    def __init__(self, job, *, resume_from=None):
        self.context, self.app = prepare_session(job, resume_from=resume_from, train=False)
        self._envs: list[ProductionEvalEnv] = []
        self._closed = False

    def make_env(self):
        ctx = self.context
        raw, _obs = _build_environment(
            ctx["job"],
            ctx["spec"],
            ctx["config"],
            ctx["prefix"],
            ctx["source"],
            ctx["environment"],
            ctx["observations"],
            train=ctx["train"],
        )
        env = ProductionEvalEnv(raw, None, ctx["job"], apply_command)
        self._envs.append(env)
        return env

    def close(self):
        if self._closed:
            return
        self._closed = True
        failures = []
        for env in reversed(self._envs):
            try:
                env.close()
            except Exception as exc:  # 关闭失败不掩盖原始评估结果
                failures.append(repr(exc))
        self._envs.clear()
        try:
            if self.app is not None:
                self.app.close()
                self.app = None
        finally:
            if failures:
                raise RuntimeError(f"评估环境关闭失败: {failures}")


def make_env(job, *, resume_from=None):
    """单会话便捷入口：返回会话拥有的环境；调用方必须 close()。"""
    session = EvaluationSession(job, resume_from=resume_from)
    env = session.make_env()
    env._session = session
    return env


def make_collection_env(job, *, resources):
    """Same physical factory and contracts, explicit sampling resources, no PPO."""
    from .collection import make_environment

    return make_environment(
        job,
        resources=resources,
        session_factory=prepare_session,
        environment_factory=_build_environment,
    )


def apply_command(env, command):
    """把命令同时写入Actor观测来源与任务奖励/终止使用的命令缓冲。"""
    values = list(command)
    if hasattr(env, "set_command"):
        env.set_command(values)
        return
    applier = getattr(env, "set_evaluation_command", None)
    if applier is None:
        raise ValueError("生产环境缺少拥有调度与奖励命令的正式接口")
    applier(values)


CONDITION_FIELDS = ("command", "friction", "delay")


def apply_condition(env, condition):
    """把声明的条件真正交给环境；未知字段明确拒绝，不静默忽略。"""
    unknown = set(condition) - {"name", *CONDITION_FIELDS}
    if unknown:
        raise ValueError(f"生产Motion评估不支持条件字段: {sorted(unknown)}")
    command = condition.get("command")
    if command is not None:
        apply_command(env, command)
    if "friction" in condition or "delay" in condition:
        raise ValueError(
            "生产V4环境的摩擦/延迟由冻结资产与DR状态决定，评估不能单独覆盖；"
            "请在任务配置中声明并在训练侧修改"
        )


def _check_source_progress(job, source, resume_from) -> int:
    """恢复点的进度必须与来源声明或当前run账本一致，不能一律按首次初始化判断。

    - 首次初始化：来源在线累计 == assets.historical_online_updates；
    - 阶段接续/同run resume：来源在线累计 == historical + 已恢复点对应的新run更新数；
    - 无论哪种，在线累计不得超过原始在线授权上限。
    """
    online = int(source.get("online_updates_including_smoke") or 0)
    historical = int(job.assets["historical_online_updates"])
    limit = int(job.assets.get("original_online_limit") or 0)
    if resume_from is None and job.assets.get("branch_initialization"):
        from native_stand.artifacts.branch_initialization import read_branch

        branch = read_branch(job.assets)
        if (
            source.get("branch_initialization_record") != branch
            or online != branch["source_model_online_updates"]
            or historical
            != branch["authorization_spent_updates"] + branch.get("prior_smoke_updates", 0)
        ):
            raise ValueError("选中完整模型的新分支进度不一致")
        return 0
    if online < historical:
        raise ValueError(f"来源在线累计{online}小于历史基数{historical}；拒绝用更早的模型继续")
    new_in_source = online - historical
    if resume_from is not None:
        restored = getattr(resume_from, "payload", resume_from)
        expected = int((restored.get("counters") or {}).get("new_run_updates") or 0)
        if "new_run_updates" in (restored.get("counters") or {}) and new_in_source != expected:
            raise ValueError(f"恢复点自述新run更新{expected}与来源在线累计差{new_in_source}不一致")
    elif job.assets.get("new_independent_authorization"):
        if online != int(job.assets["source_online_updates"]):
            raise ValueError("独立run来源累计与声明不一致")
        return 0
    elif new_in_source != 0:
        raise ValueError(f"首次初始化来源已带{new_in_source}次当前run更新；应作为resume使用")
    if limit and online > limit:
        raise ValueError(f"来源在线累计{online}超过原始在线授权{limit}")
    return new_in_source


def annotate_retention(value, env, job):
    config = job.assets.get("source_policy_retention")
    if config is None:
        return value
    import torch

    from native_stand.learning.ppo.source_anchor import CAUSAL_VERSION, metadata, qualified_valid
    from native_stand.recovery.controlled_ppo import KEY

    if not hasattr(env, "source_retention_profiles"):
        env.source_retention_profiles = torch.tensor(
            env.stage_spec["identified_profiles"], device=env.device
        )
        env.source_retention_unperturbed = (
            torch.tensor(env.stage_spec["modal_modes"], device=env.device) < 0
        )
    valid = value[KEY].squeeze(-1) & env.source_retention_unperturbed[env.command_index]
    if config["version"] in (CAUSAL_VERSION, "motion48_per_command_nominal_source_KL_v4"):
        valid = qualified_valid(
            valid,
            event_enabled=env.transient_events.enabled,
            age_s=env.rise_hold.episode.steps.float() * 0.01,
            pitch=env.standing_snapshot[:, 0],
            roll=env.standing_snapshot[:, 1],
            wheel_support=env.recovery_snapshot[:, 10:12],
        )
    valid &= (env.command[:, 2] - 0.34).abs() < 1e-6
    if job.assets.get("terrain_scene") is not None:
        # Same command on rough terrain is a NEW dynamic state; the nominal
        # flat Teacher bank is not a label for that state.
        valid &= env.motion_flat_terrain
    contact = getattr(getattr(env, "motion_timing", None), "dr", None)
    contact = getattr(contact, "contact", None)
    if contact is not None:
        valid &= ~(contact.actual[:, :2] > 0).any(-1)
    return metadata(
        value,
        env.target_xy,
        env.source_retention_profiles[env.command_index],
        valid,
        config["commands"],
    )


def anchor_identity(job, source):
    """沿显式产物引用解析固定Teacher身份；不按文件名、维度或update推断。"""
    retention = job.assets.get("source_policy_retention")
    if retention is None:
        return None
    identity = source.get("source_anchor_identity") or job.assets.get("source_anchor_identity")
    if identity is None and retention["source_sha256"] == job.initialization.sha256:
        identity = dict(path=str(job.initialization.checkpoint), sha256=job.initialization.sha256)
    if identity is None:
        parent = job.assets.get("training_source_admission", {}).get("parent_recipe")
        if parent is not None:
            if file_sha256(parent["path"]) != parent["sha256"]:
                raise ValueError("Teacher父配方身份不同")
            initial = read_json(parent["path"])["initialization"]
            identity = dict(path=initial["checkpoint"], sha256=initial["sha256"])
    if identity is not None:
        identity = {
            **identity,
            "path": RuntimePaths.from_assets(job.assets).resolve(identity["path"]),
        }
    if (
        identity is None
        or identity["sha256"] != retention["source_sha256"]
        or file_sha256(identity["path"]) != identity["sha256"]
    ):
        raise ValueError("固定来源Teacher缺少匹配的路径/SHA身份")
    return {**identity, "path": RuntimePaths.from_assets(job.assets).resolve(identity["path"])}


def candidate_strategy(job):
    from native_stand.tasks.motion import maneuver_candidates, motion_candidates, split_candidates

    version = (
        job.diagnostics.get("training_candidate_selection", {}).get("version")
        if job is not None
        else None
    )
    if version in (maneuver_candidates.VERSION, maneuver_candidates.RESPONSE_VERSION):
        return maneuver_candidates
    if version == split_candidates.VERSION:
        return split_candidates
    return motion_candidates
