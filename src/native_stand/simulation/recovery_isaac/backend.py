"""Public independent Isaac Recovery backend with explicit resource/acceptance gates.

CPU import/preflight never opens Kit or a CUDA context. Only make_env launches,
and only after checking the physical GPU identity and existing compute owners.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import torch
from tensordict import TensorDict

from native_stand.artifacts.io import file_sha256, read_json, write_json
from native_stand.artifacts.runtime_paths import execution_json
from native_stand.learning.ppo.recovery_cubic import VERSION as ACTION_VERSION
from native_stand.tasks.recovery.cubic_task import CRITIC_VERSION
from native_stand.tasks.recovery.handover_context import VERSION as HANDOVER_CRITIC
from native_stand.tasks.recovery.handover_context import WIDTH as HANDOVER_WIDTH
from native_stand.tasks.recovery.height_context import (
    REWARD_VERSION as HEIGHT_REWARD,
)
from native_stand.tasks.recovery.height_context import (
    VERSION as HEIGHT_CRITIC,
)
from native_stand.tasks.recovery.height_context import (
    WIDTH as HEIGHT_WIDTH,
)
from native_stand.tasks.recovery.reward_context import VERSION as REWARD_CRITIC_VERSION
from native_stand.tasks.recovery.reward_context import WIDTH as REWARD_CRITIC_WIDTH

from .handover_lifecycle import VERSION as HANDOVER_REWARD
from .lifecycle import VERSION as REWARD_VERSION

ENGINE_AVAILABLE = True
SEMANTICS = dict(
    task="recovery",
    critic="explicit job-bound version",
    supported_critics=[CRITIC_VERSION, REWARD_CRITIC_VERSION, HANDOVER_CRITIC, HEIGHT_CRITIC],
    action=ACTION_VERSION,
    physical_backend="Isaac/PhysX",
    Motion_runtime=False,
)
SNAPSHOT = "recovery_Isaac_observable_execution_snapshot_v1"


def validate_job(job):
    if job.task != "recovery" or job.backend != "isaac_recovery":
        raise ValueError("Explicit independent Recovery Isaac job required")
    from native_stand.tasks.recovery.leg_driven import CRITIC_VERSION as LEG_CRITIC
    from native_stand.tasks.recovery.leg_driven import CRITIC_WIDTH
    from native_stand.tasks.recovery.leg_driven import VERSION as LEG_REWARD

    leg_driven = job.reward.version == LEG_REWARD
    standalone = job.assets.get("recovery_standalone_course")
    if standalone:
        from native_stand.tasks.recovery.standalone_course import VERSION
        from native_stand.tasks.recovery.swing_priority_course import VERSION as SWING

        if standalone.get("version") not in (VERSION, SWING) or standalone.get(
            "initial_stage", 0
        ) not in (0, 1):
            raise ValueError("未知独立Recovery课程或初始阶段")
        if job.model.critic_privileged_version not in (REWARD_CRITIC_VERSION, HEIGHT_CRITIC):
            raise ValueError("独立Recovery不能采用Motion交接Critic")
        if job.assets.get("recovery", {}).get("gate", {}).get("hold_owner") == "frozen_Motion":
            raise ValueError("Recovery必须自行控制连续保持")
    width = {
        CRITIC_VERSION: 878,
        REWARD_CRITIC_VERSION: REWARD_CRITIC_WIDTH,
        HANDOVER_CRITIC: HANDOVER_WIDTH,
        HEIGHT_CRITIC: HEIGHT_WIDTH,
        LEG_CRITIC: CRITIC_WIDTH,
    }.get(job.model.critic_privileged_version)
    if (
        job.model.actor_input,
        job.model.actor_hidden_dims,
        job.model.critic_input,
        job.model.critic_hidden_dims,
    ) != (31, (2048, 1024, 512), width, (512, 512, 256)):
        raise ValueError("Actual large Actor/versioned Recovery Critic required")
    if width is None or job.action.version != ACTION_VERSION:
        raise ValueError("Recovery Critic/physical Gaussian versions")
    expected_reward = (
        HEIGHT_REWARD
        if job.model.critic_privileged_version == HEIGHT_CRITIC
        else HANDOVER_REWARD
        if job.model.critic_privileged_version == HANDOVER_CRITIC
        else REWARD_VERSION
    )
    if (
        job.reward.version == "recovery_preentry_runaway48_physical_v8"
        and standalone
        and standalone.get("version") == "recovery_Swing_priority_own_hold_v2"
    ):
        expected_reward = job.reward.version
    if leg_driven:
        from native_stand.tasks.recovery.leg_driven import (
            anti_prone_config,
            course_fractions,
            progression_config,
        )

        anti_prone_config(job.reward.fields.get("anti_prone"))
        progress_recipe = progression_config(job.reward.fields.get("progression"))
        course_fractions(job.curriculum.stages)
        expected_reward = LEG_REWARD
        if job.model.critic_privileged_version != LEG_CRITIC:
            raise ValueError("Leg-driven task requires its actual causal Critic")
        ref = job.assets.get("recovery_leg_driven_geometry")
        if not ref or file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Leg-driven geometry identity required")
        if any(
            job.assets.get(k) for k in ("qualified_physical_BC_anchor", "old_Recovery_action_prior")
        ):
            raise ValueError("Old high-wheel-speed demonstrations conflict with leg-driven goal")
    if job.reward.version != expected_reward:
        raise ValueError("Explicit Isaac reward revision required")
    if job.runtime.envs < 1 or job.runtime.device != "cuda:0":
        raise ValueError("Explicit GPU0 only, no auto device fallback")
    if job.initialization.mode not in ["actor_only", "full"]:
        raise ValueError("Explicit actor-only initialization or same-backend full resume")
    c = job.control
    if (
        c.physics_dt_s,
        c.pd_dt_s,
        c.policy_dt_s,
        c.leg_kp,
        c.leg_kd,
        c.wheel_kv,
        c.height_command_m,
    ) != (0.001, 0.005, 0.01, 40.0, 2.0, 0.2, 0.34):
        raise ValueError("Frozen physical timing/PD")
    expected_caps = (
        [40.0, 40.0, 4.8, 40.0, 40.0, 4.8]
        if leg_driven or job.reward.version == "recovery_preentry_runaway48_physical_v8"
        else [40.0, 40.0, 2.44694786377709, 40.0, 40.0, 2.44694786377709]
    )
    if list(c.effort_limits_nm) != expected_caps:
        raise ValueError("Motor caps unchanged")
    if (job.ppo.gamma, job.ppo.lam, job.ppo.clip, job.ppo.value_coef, job.ppo.max_grad_norm) != (
        0.9980019986673331,
        0.95,
        0.2,
        1.0,
        1.0,
    ):
        raise ValueError("Explicit implemented physical PPO hyperparameters")
    if (
        job.observation.leg_input_scale_rad,
        job.observation.wheel_input_scale_rad_s,
        job.action.leg_scale_rad,
        job.action.wheel_scale_rad_s,
    ) != (1.5, 100.0, 0.15, 100.0):
        raise ValueError("Frozen absolute input/output units")
    if (
        job.ppo.entropy_coef != 0
        or job.ppo.critic_warmup_updates != 5
        or (job.ppo.exploration_leg_rad, job.ppo.exploration_wheel_rad_s)
        != (0.02, 0.5 if leg_driven and progress_recipe is not None else 2.0)
    ):
        raise ValueError("Fixed physical std / five Critic-only updates")


def available(job):
    try:
        validate_job(job)
    except ValueError as e:
        return False, str(e)
    return (importlib.util.find_spec("isaaclab") is not None), "Isaac Lab interpreter required"


def preflight(job, *, for_learning=None, resources=False):
    if for_learning is None:
        for_learning = job.algorithm == "ppo" and job.reward.fields.get("progression") is not None
    validate_job(job)
    from native_stand.tasks.recovery.spec import build_task

    build_task(job)  # The same public task construction used by the worker.
    names = [
        "recovery_initial_actor",
        "recovery_training_contract",
        "recovery_isaac_runtime",
        "recovery_isaac_acceptance",
    ]
    for name in names:
        item = job.assets.get(name)
        if not isinstance(item, dict) or file_sha256(item["path"]) != item["sha256"]:
            raise ValueError("Missing/changed " + name)
    for name in (
        "qualified_physical_BC_anchor",
        "old_Recovery_action_prior",
        "recovery_legacy_reward",
    ):
        ref = job.assets.get(name)
        if ref is not None and file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Recovery依赖身份不同: " + name)
    runtime = execution_json(job.assets["recovery_isaac_runtime"]["path"], job.assets)
    for name, item in runtime["bindings"].items():
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError("Runtime source identity " + name)
    cp = torch.load(
        job.assets["recovery_initial_actor"]["path"], weights_only=False, map_location="cpu"
    )
    expected = {
        "mlp.0.weight": (2048, 31),
        "mlp.0.bias": (2048,),
        "mlp.2.weight": (1024, 2048),
        "mlp.2.bias": (1024,),
        "mlp.4.weight": (512, 1024),
        "mlp.4.bias": (512,),
        "mlp.6.weight": (6, 512),
        "mlp.6.bias": (6,),
    }
    if cp.get("actor_only") is not True or "log_std" in cp["actor"]:
        raise ValueError("Physical actor-only source, no old normalized log_std")
    for key, shape in expected.items():
        if tuple(cp["actor"][key].shape) != shape:
            raise ValueError("Actual Actor tensor " + key)
    contract = read_json(job.assets["recovery_training_contract"]["path"])
    if contract != cp["output_contract"]:
        migration = job.assets.get("recovery_actor_contract_migration")
        if not migration or file_sha256(migration["path"]) != migration["sha256"]:
            raise ValueError("Explicit Actor contract metadata migration required")
        m = read_json(migration["path"])
        changed = {
            k
            for k in set(contract) | set(cp["output_contract"])
            if contract.get(k) != cp["output_contract"].get(k)
        }
        if (
            changed != {"critic_privileged_version"}
            or contract["critic_privileged_version"]
            not in (
                REWARD_CRITIC_VERSION,
                HANDOVER_CRITIC,
                HEIGHT_CRITIC,
                "recovery_leg_driven_causal942_v1",
            )
            or m["source_actor_sha256"] != job.assets["recovery_initial_actor"]["sha256"]
            or m["target_contract_sha256"] != job.assets["recovery_training_contract"]["sha256"]
        ):
            raise ValueError("Actor/physical contract binding")
    if job.initialization.mode == "actor_only" and (
        str(job.initialization.checkpoint) != job.assets["recovery_initial_actor"]["path"]
        or job.initialization.sha256 != job.assets["recovery_initial_actor"]["sha256"]
    ):
        raise ValueError("Initialization identity")
    if job.reward.fields.get("progression") is not None:
        allocation_ref = job.assets["training_allocation"]
        if file_sha256(allocation_ref["path"]) != allocation_ref["sha256"]:
            raise ValueError("Recovery allocation identity differs")
        allocation = read_json(allocation_ref["path"])
        charged = 0
        for row in allocation["prior_attempts"]:
            for name in ("state", "budget"):
                if file_sha256(row[name]["path"]) != row[name]["sha256"]:
                    raise ValueError("Recovery prior settlement identity differs")
            prior_budget = read_json(row["budget"]["path"])
            prior_state = read_json(row["state"]["path"])
            used = sum(prior_budget["used"].values())
            if (
                used != row["used"]
                or used != prior_state["updates_used"]
                or prior_state["status"] not in ("FAILED", "STOPPED_BOUNDED", "COMPLETED")
            ):
                raise ValueError("Recovery prior settlement is not terminal and consistent")
            charged += used
        if (
            charged != allocation["prior_updates_used"]
            or charged + job.limits.formal_updates != allocation["authorized_total_updates"]
            or job.limits.formal_updates != allocation["new_run_updates"]
            or job.limits.smoke_updates != 0
        ):
            raise ValueError("Recovery allocation cannot refund prior updates")
    transition_ref = job.assets.get("recovery_recipe_transition")
    if transition_ref is not None:
        if file_sha256(transition_ref["path"]) != transition_ref["sha256"]:
            raise ValueError("Recovery recipe transition identity differs")
        from native_stand.artifacts.io import json_digest

        transition = read_json(transition_ref["path"])
        if (
            transition.get("version") != "recovery_full_recipe_transition_v1"
            or job.initialization.mode != "full"
            or Path(transition["source_checkpoint"]["path"]).resolve()
            != Path(job.initialization.checkpoint).resolve()
            or transition["source_checkpoint"]["sha256"] != job.initialization.sha256
            or file_sha256(job.initialization.checkpoint) != job.initialization.sha256
            or transition["to_reward_sha256"] != json_digest(job.as_run_record()["reward"])
            or transition["to_curriculum_sha256"] != json_digest(job.as_run_record()["curriculum"])
            or transition.get("fresh_physical_reset") is not True
        ):
            raise ValueError("Recovery recipe transition source/target differs")
    acceptance = read_json(job.assets["recovery_isaac_acceptance"]["path"])
    if for_learning and job.reward.version == "recovery_leg_driven_low_wheel_v1":
        auth = read_json(job.assets["training_authorization"]["path"])
        if (
            file_sha256(job.assets["training_authorization"]["path"])
            != job.assets["training_authorization"]["sha256"]
        ):
            raise ValueError("Leg-driven training authorization identity")
        if auth.get("new_training_updates") != 10000 or auth.get("wall_s", "MISSING") is not None:
            raise ValueError("Leg-driven explicit bounded update authorization required")
        if not acceptance.get("CPU_leg_driven_wiring_accepted"):
            raise ValueError("Leg-driven CPU consumer wiring not accepted")
        if acceptance.get("actor_sha256") != job.assets["recovery_initial_actor"]["sha256"]:
            raise ValueError("Leg-driven Actor source mismatch")
        if (
            acceptance.get("runtime_contract_sha256")
            != job.assets["recovery_isaac_runtime"]["sha256"]
        ):
            raise ValueError("Leg-driven runtime identity mismatch")
        if job.assets.get("recovery_probe_controls"):
            raise ValueError("Engineering fixture controls cannot train")
    elif for_learning:
        if acceptance.get("Isaac_PPO0_wiring_accepted") is not True:
            raise ValueError("Isaac PPO0 live wiring NOT_RUN/unaccepted; no real PPO launch")
        if acceptance.get("actor_sha256") != job.assets["recovery_initial_actor"]["sha256"]:
            raise ValueError("Accepted Actor mismatch")
        if (
            acceptance.get("runtime_contract_sha256")
            != job.assets["recovery_isaac_runtime"]["sha256"]
        ):
            raise ValueError("Accepted runtime mismatch")
        if job.assets.get("recovery_probe_controls"):
            raise ValueError("Engineering fixture controls cannot train")
    result = dict(
        status="CPU_PREFLIGHT_PASS",
        engine_validation=acceptance.get("engine_validation", "NOT_RUN"),
        PPO_updates=0,
        actor_sha256=job.assets["recovery_initial_actor"]["sha256"],
        training_contract_sha256=job.assets["recovery_training_contract"]["sha256"],
        Isaac_PPO0_wiring_accepted=acceptance.get("Isaac_PPO0_wiring_accepted") is True,
    )
    if resources:
        result["resources"] = require_idle_gpu(
            job.assets["physical_gpu_uuid"], job.assets.get("gpu_resource_policy")
        )
    return result


def require_idle_gpu(expected_uuid, policy=None):
    policy = policy or {"mode": "exclusive"}
    shared = policy.get("mode") == "shared_authorized"
    if policy.get("mode") not in ("exclusive", "shared_authorized"):
        raise ValueError("Explicit GPU resource mode")
    if shared:
        authorization = policy["authorization"]
        if file_sha256(authorization["path"]) != authorization["sha256"]:
            raise ValueError("Shared authorization identity")
        auth = read_json(authorization["path"])
        if auth.get("shared_GPU_allowed") is not True:
            raise ValueError("Shared GPU not authorized")
        import shutil

        if shutil.disk_usage(policy["storage_path"]).free < policy["minimum_disk_free_bytes"]:
            raise RuntimeError("Insufficient output disk reserve")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "0", expected_uuid):
        raise RuntimeError("CUDA mapping differs from declared physical GPU0")
    owners = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    for line in owners:
        pid, uuid = [x.strip() for x in line.split(",")]
        if uuid != expected_uuid or pid == str(os.getpid()):
            continue
        try:
            command = Path("/proc", pid, "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except FileNotFoundError:
            continue
        if "native_stand.experiments.worker" in command:
            recovery = False
            for word in command.split():
                candidate = Path(word)
                if candidate.is_dir():
                    candidate = candidate / "run.json"
                if candidate.is_file() and candidate.name.endswith(".json"):
                    try:
                        data = read_json(candidate)
                        data = data.get("config", data)
                        recovery |= (
                            data.get("task") == "recovery"
                            or data.get("backend") == "isaac_recovery"
                            or "recovery" in str(data.get("preset", ""))
                        )
                    except (ValueError, OSError):
                        pass
            if not shared or recovery:
                raise RuntimeError("Existing conflicting training owner, PID=" + pid)
        if shared and "native_stand.experiments.recovery_isaac_probe" in command:
            raise RuntimeError("Duplicate Recovery probe owner, PID=" + pid)
    raw = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    rows = [[v.strip() for v in line.split(",")] for line in raw]
    row = next((r for r in rows if r[1] == expected_uuid), None)
    if row is None or row[0] != "0":
        raise RuntimeError("Declared physical GPU0 UUID changed; no auto remap")
    if not shared and int(row[3]) > 20:
        raise RuntimeError(
            "No idle shared GPU window: utilization=" + row[3] + "%; preserve active Motion"
        )
    if int(row[2]) < int(policy.get("minimum_free_MiB", 8192)):
        raise RuntimeError("Insufficient reserved free VRAM for bounded Isaac preparation")
    return dict(
        mode=policy["mode"],
        physical_gpu=0,
        uuid=expected_uuid,
        free_MiB=int(row[2]),
        utilization_percent=int(row[3]),
    )


class IsaacRecovery:
    def __init__(self, job, *, output, application=None, measurement=None):
        from native_stand.learning.ppo.performance import Measurement

        self.measurement = (
            measurement if measurement is not None else Measurement(False, job.runtime.device)
        )
        self.env = None
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.app = None
        self._owns_app = application is None
        if application is None:
            preflight(job, resources=True)
            from isaaclab.app import AppLauncher

            from .runtime_safety import preserve_worker_signals

            with preserve_worker_signals():
                self.app = AppLauncher(
                    {
                        "headless": True,
                        "device": job.runtime.device,
                        "multi_gpu": False,
                        "fast_shutdown": False,
                    }
                ).app
        else:
            preflight(job, resources=False)
            self.app = application
        try:
            self._initialize(job, output)
        except BaseException as exc:
            import traceback

            write_json(
                self.output / "construction_failure.json",
                dict(error=repr(exc), traceback=traceback.format_exc()),
            )
            try:
                self.close()
            except BaseException as cleanup:
                exc.add_note("Recovery construction cleanup failed: " + repr(cleanup))
            raise

    def _initialize(self, job, output):
        # All Isaac/asset runtime imports occur strictly AFTER AppLauncher.
        runtime = execution_json(job.assets["recovery_isaac_runtime"]["path"], job.assets)
        self.job = job
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        template = execution_json(runtime["bindings"]["scene_spec"]["path"], job.assets)
        from .bindings import asset_module_root

        sys.path.insert(0, str(asset_module_root(template)))
        from native_stand.command_v6.runtime.configure import verify_asset_import
        from native_stand.env import NativeStandCfg

        verify_asset_import(template)

        from native_stand.learning.ppo.networks import StudentNetwork
        from native_stand.tasks.motion.action_contract import ActionDecoder

        from .bindings import load_response
        from .environment import environment_class

        actor = StudentNetwork(
            actor=True,
            input_dim=31,
            hidden_dims=(2048, 1024, 512),
            output_dim=6,
            init_std=torch.ones(6),
        )
        source = torch.load(
            runtime["bindings"]["C0_actor"]["path"], weights_only=False, map_location="cpu"
        )
        actor.load_state_dict(source["actor"], strict=True)
        center = (
            ActionDecoder.from_contract(read_json(runtime["bindings"]["C0_contract"]["path"]))
            .decode(torch.zeros(1, 6), torch.tensor([0.34]))["requested_target"][0, [0, 1, 3, 4]]
            .tolist()
        )
        from native_stand.simulation.scenes import scene_for

        scene_config, height_provider = scene_for(job)
        from native_stand.artifacts.runtime_paths import RuntimePaths
        from native_stand.simulation.scenes import initial_catalog

        initials = initial_catalog(
            scene_config,
            read_json(runtime["bindings"]["initial_states"]["path"]),
            paths=RuntimePaths.from_assets(job.assets),
        )
        if scene_config.ground_z_m != 0:
            raise ValueError("Isaac flat ground placement currently supports world Z=0 only")
        cls = environment_class(
            initials,
            read_json(runtime["bindings"]["V4_contract"]["path"]),
            load_response(runtime["bindings"]["V4_tensor"]),
            actor,
            center,
            height_provider=height_provider,
        )
        cfg = NativeStandCfg()
        cfg.leg_kp = 40.0
        cfg.leg_kd = 2.0
        cfg.wheel_kv = 0.2
        cfg.pd_interval_steps = 5
        cfg.leg_position_scale_rad = 0.15
        cfg.leg_pair_radius_rad = 0.0
        cfg.wheel_speed_scale_rad_s = 100.0
        cfg.recovery_critic_version = job.model.critic_privileged_version
        cfg.recovery_height_conditioned = job.model.critic_privileged_version == HEIGHT_CRITIC
        if job.assets.get("recovery_standalone_course"):
            cfg.recovery_standalone_course = True
            cfg.recovery_standalone_initial_stage = job.assets["recovery_standalone_course"].get(
                "initial_stage", 0
            )
        if job.model.critic_privileged_version == HANDOVER_CRITIC:
            cfg.recovery_task_mode = "Swing_to_frozen_Motion_v1"
        cfg.seed = job.runtime.seed
        cfg.scene.num_envs = job.runtime.envs
        cfg.sim.device = job.runtime.device
        cfg.sim.dt = 0.001
        cfg.decimation = 10
        cfg.episode_length_s = 23.0
        cfg.is_finite_horizon = True
        from native_stand.tasks.recovery.leg_driven import VERSION as LEG_REWARD

        if job.reward.version == LEG_REWARD:
            from native_stand.tasks.recovery.leg_driven import course_fractions

            cfg.recovery_leg_driven = dict(
                geometry=read_json(job.assets["recovery_leg_driven_geometry"]["path"]),
                gamma=job.ppo.gamma,
                anti_prone=job.reward.fields.get("anti_prone"),
                progression=job.reward.fields.get("progression"),
                course_fractions=course_fractions(job.curriculum.stages),
            )
            cfg.episode_length_s = 6.0
        cfg.recovery_measurement = self.measurement
        cfg.recovery_effort_caps = job.control.effort_limits_nm
        cfg.recovery_swing_priority = (
            job.assets.get("recovery_standalone_course", {}).get("version")
            == "recovery_Swing_priority_own_hold_v2"
        )
        cfg.recovery_preentry_reward = (
            job.reward.version == "recovery_preentry_runaway48_physical_v8"
        )
        probe = job.assets.get("recovery_probe_controls")
        if probe:
            if len(probe["cases"]) != job.runtime.envs:
                raise ValueError("Exact bounded probe lanes")
            cfg.recovery_probe_cases = copy.deepcopy(probe["cases"])
            cfg.recovery_probe_full23 = bool(probe["continue_after_success_to_natural23"])
        cfg.robot_cfg = cfg.robot_cfg.copy()
        cfg.robot_cfg.spawn.usd_path = template["asset_path"]
        cfg.recovery_dr_materials = True
        cfg.ground_static_friction = scene_config.friction
        cfg.ground_dynamic_friction = scene_config.friction
        cfg.ground_restitution = 0.0
        cfg.scene.contact_forces.filter_prim_paths_expr = ["/World/ground"]
        for group, cap in [
            ("leg_motors", job.control.effort_limits_nm[0]),
            ("wheel_motors", job.control.effort_limits_nm[2]),
        ]:
            actuator = cfg.robot_cfg.actuators[group]
            actuator.effort_limit = cap
            actuator.effort_limit_sim = cap
            actuator.stiffness = 0.0
            actuator.damping = 0.0
        from gymnasium import spaces

        cfg.action_space = spaces.Box(-float("inf"), float("inf"), shape=(6,))
        cfg.observation_space = {
            k: spaces.Box(-1e6, 1e6, shape=(w,))
            for k, w in [
                ("student_actor", 31),
                ("student_critic", job.model.critic_input),
                ("recovery_actor_valid", 1),
            ]
        }
        self.env = cls(cfg)
        self.num_envs = job.runtime.envs
        self.device = job.runtime.device
        raw, _ = self.env.reset()
        self._obs = self.convert(raw)
        write_json(
            self.output / "construction.json",
            dict(
                version=SNAPSHOT,
                AppLauncher_first=True,
                Motion_environment_used=False,
                Isaac_physics_resume_bitwise=False,
                passive_model=runtime["passive_model"],
            ),
        )

    def convert(self, raw):
        return TensorDict(raw, batch_size=[self.num_envs], device=self.device)

    def observations(self):
        return self._obs

    def step(self, raw):
        physical_before = self.env.actual_physics_steps
        obs, reward, terminated, truncated, extras = self.env.step(raw)
        terminal = self.env.terminal_critic
        self._obs = self.convert(obs)
        return dict(
            observations=self._obs,
            rewards=reward,
            dones=terminated | truncated,
            terminated=terminated,
            truncated=truncated,
            extras=dict(
                actual_physics_steps=[
                    (self.env.actual_physics_steps - physical_before) // self.num_envs
                ]
                * self.num_envs,
                control_owner=(~self.env.transition["student_mask"]).long(),
                collection_conditions={
                    "initial_Swing": self.env.transition["initial_Swing"].long()
                },
                terminal_critic=terminal,
                time_outs=truncated,
                # Environment publishes detached snapshots; the next step replaces
                # the dictionary. This outcome owns it until the next step.
                transition=self.env.transition,
            ),
        )

    def snapshot(self):
        e = self.env
        d = e.robot.data
        return copy.deepcopy(
            dict(
                version=SNAPSHOT,
                root_link_pose=d.root_link_pose_w.torch,
                root_COM_velocity=d.root_com_vel_w.torch,
                joint_q=d.joint_pos.torch,
                joint_dq=d.joint_vel.torch,
                height_schedule=dict(initial=e.height_initial, dynamic=e.height_dynamic)
                if getattr(e, "height_conditioned", False)
                else None,
                execution=e.execution.state_dict(),
                lifecycle=e.life.state_dict(),
                physical=e.physics.state_dict(),
                issued_observations=self._obs,
                fine_ring=e.fine_ring,
                sim_step_counter=e._sim_step_counter,
                episode_length_buf=e.episode_length_buf,
                policy_raw=e.policy_raw,
                control_physical=e.control_physical,
                student_mask=e.student_mask,
                last_external=e.last_external,
                last_passive=e.last_passive,
                effort=e.effort,
                HP10_history=e.HP10_history,
                HP10_sum=e.HP10_sum,
                HP10_count=e.HP10_count,
                HP10_cursor=e.HP10_cursor,
                episode_steps=e.episode_steps,
                initialized=e.initialized,
                prefix_s=e.prefix_s,
                student_seconds=e.student_seconds,
                teacher_seconds=e.teacher_seconds,
                courses=e.courses,
                DR_rng=[r.bit_generator.state for r in e.DR_rng],
                DR_draws=e.draws,
                episode_spec=e.episode_spec,
                reset_count=e.reset_count,
                completed=e.completed,
                completed_count=e.completed_count,
                physical_steps=e.actual_physics_steps,
                phase_readback=e.phase_readback.state_dict()
                if getattr(e, "phase_readback", None) is not None
                else None,
                exact_PhysX_solver_warmstart_available=False,
            )
        )

    def restore_execution(self, state):
        if state["version"] != SNAPSHOT:
            raise ValueError("Only same Isaac execution version can resume")
        e = self.env
        ids = torch.arange(self.num_envs, device=self.device)
        e.physics.restore(state["physical"])
        e.robot.write_root_link_pose_to_sim_index(
            root_pose=state["root_link_pose"].to(self.device), env_ids=ids
        )
        e.robot.write_root_com_velocity_to_sim_index(
            root_velocity=state["root_COM_velocity"].to(self.device), env_ids=ids
        )
        e.robot.write_joint_position_to_sim_index(
            position=state["joint_q"].to(self.device), env_ids=ids
        )
        e.robot.write_joint_velocity_to_sim_index(
            velocity=state["joint_dq"].to(self.device), env_ids=ids
        )
        e.sim.forward()
        e.execution.load_state_dict(state["execution"])
        e.life.load_state_dict(state["lifecycle"])
        if getattr(e, "phase_readback", None) is not None:
            if state.get("phase_readback") is None:
                raise ValueError("Missing versioned Recovery phase statistics")
            e.phase_readback.load_state_dict(state["phase_readback"])
        if getattr(e, "height_conditioned", False):
            if state.get("height_schedule") is None:
                raise ValueError("缺少完整高度命令调度状态")
            e.height_initial.copy_(state["height_schedule"]["initial"].to(self.device))
            e.height_dynamic.copy_(state["height_schedule"]["dynamic"].to(self.device))
        for name in [
            "episode_steps",
            "initialized",
            "prefix_s",
            "student_seconds",
            "teacher_seconds",
            "reset_count",
            "HP10_history",
            "HP10_sum",
            "HP10_count",
            "fine_ring",
            "episode_length_buf",
            "policy_raw",
            "control_physical",
            "student_mask",
            "last_external",
            "last_passive",
            "effort",
        ]:
            getattr(e, name).copy_(state[name].to(self.device))
        e._sim_step_counter = state["sim_step_counter"]
        e.HP10_cursor = state["HP10_cursor"]
        e._physical_profile_cache = None
        e.courses = copy.deepcopy(state["courses"])
        e.draws = copy.deepcopy(state["DR_draws"])
        e.episode_spec = copy.deepcopy(state["episode_spec"])
        e.completed = copy.deepcopy(state["completed"])
        for rng, s in zip(e.DR_rng, state["DR_rng"], strict=False):
            rng.bit_generator.state = copy.deepcopy(s)
        e.completed_count = state.get("completed_count", len(e.completed))
        e.refresh_pulses()
        e.actual_physics_steps = max(e.actual_physics_steps, state["physical_steps"])
        self._obs = TensorDict(
            {k: v.to(self.device).clone() for k, v in state["issued_observations"].items()},
            batch_size=[self.num_envs],
            device=self.device,
        )
        e._obs = dict(self._obs)
        return self._obs

    def runtime_state(self):
        return dict(
            execution=self.snapshot(),
            actual_physics_steps=self.env.actual_physics_steps,
            critic_version=self.job.model.critic_privileged_version,
            physics_restore_scope="root/joints+causal execution; PhysX solver hidden state not exposed",
        )

    def initialize_from_training_state(self, state, *, same_run=False):
        source = state["execution"]
        if source["version"] != SNAPSHOT or len(source["courses"]) != self.num_envs:
            raise ValueError("Recovery课程/环境布局不同；没有隐式重采样迁移")
        e = self.env
        e._physical_profile_cache = None
        e.courses = copy.deepcopy(source["courses"])
        for rng, stored in zip(e.DR_rng, source["DR_rng"], strict=True):
            rng.bit_generator.state = copy.deepcopy(stored)
        if same_run:
            e.completed = copy.deepcopy(source["completed"])
            e.completed_count = source["completed_count"]
        # Completed training evidence survives; unfinished dwell/hold/FIFO does
        # not become a completed episode in a new physical world.
        e.episode_spec = [None] * self.num_envs
        raw, _ = e.reset()
        self._obs = self.convert(raw)
        return self._obs

    def load_runtime_state(self, state):
        if self.job.assets.get("execution_resume") == "fresh_reset":
            return self.initialize_from_training_state(state, same_run=True)
        return self.restore_execution(state["execution"])

    def set_measurement(self, measurement):
        self.measurement = measurement
        if self.env is not None:
            self.env.reset_measurement = measurement

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.env = None
            if self._owns_app and self.app is not None:
                try:
                    app = self.app
                    import atexit

                    atexit.register(lambda: app.close())
                finally:
                    self.app = None


def make_env(job, *, output=None, measurement=None):
    return IsaacRecovery(
        job,
        output=output
        or Path(job.assets.get("recovery_execution_artifacts", "/tmp/recovery_isaac_execution")),
        measurement=measurement,
    )


def make_measured_env(job, measurement):
    return make_env(job, measurement=measurement)


def make_collection_env(job, *, resources):
    if resources != job.runtime:
        raise ValueError("Collection resources differ from runtime request")
    return make_env(job)
