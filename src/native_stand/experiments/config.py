"""配置的单一事实来源：一次解析、严格校验、不可变类型化对象。

设计取舍：
- 新配方用preset加显式覆盖组合，不复制整份旧spec再逐层删改。
- 不支持任意递归历史继承；未知字段在入口报错，不静默忽略。
- 路径按配置文件所在目录解析，不依赖调用者的当前目录。
"""

from __future__ import annotations

import copy as _preset_copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from native_stand.artifacts.io import file_sha256, read_json, resolve_path
from native_stand.tasks.motion.execution_limits import MotorLimits as _MotionMotorLimits

CONFIG_SCHEMA_VERSION = 1


class ConfigError(ValueError):
    """配置不合法；消息包含具体字段路径。"""


@dataclass(frozen=True)
class ModelConfig:
    interface: str
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    actor_input: int
    critic_input: int
    critic_privileged_version: str | None = None
    action_version: str | None = None
    kind: str = "mlp"  # 网络类型；当前仅mlp，进入run record


@dataclass(frozen=True)
class InitializationConfig:
    mode: str
    checkpoint: Path | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        self._validate_mode()

    def _validate_mode(self) -> None:
        """三种模式的字段边界；一处校验，跨节校验复用本方法结论。"""
        if self.mode not in ("scratch", "full", "actor_only"):
            raise ConfigError("initialization.mode 只能是 scratch、full 或 actor_only")
        if self.mode in ("full", "actor_only"):
            if self.checkpoint is None or self.sha256 is None:
                raise ConfigError(f"initialization.mode={self.mode} 需要 checkpoint 与 sha256")
        elif self.checkpoint is not None:
            raise ConfigError("initialization.mode=scratch 不能同时给出 checkpoint")


@dataclass(frozen=True)
class ObservationConfig:
    frame_dim: int
    actor_dim: int
    critic_dim: int
    leg_input_scale_rad: float
    wheel_input_scale_rad_s: float

    def __post_init__(self) -> None:
        if self.leg_input_scale_rad <= 0 or self.wheel_input_scale_rad_s <= 0:
            raise ConfigError("observation输入归一化尺度必须为正")
        # frame_dim+9 是Motion观测构成；其他任务（观测维数不同）在此不受限，
        # 由任务模块自己的校验负责（避免共享配置强制Motion语义）。


@dataclass(frozen=True)
class ActionConfig:
    dim: int
    order: tuple[str, ...]
    leg_scale_rad: float
    wheel_scale_rad_s: float
    version: str

    def __post_init__(self) -> None:
        if len(self.order) != self.dim:
            raise ConfigError(
                f"action.order 长度 {len(self.order)} 与 action.dim={self.dim} 不一致"
            )
        if self.leg_scale_rad <= 0 or self.wheel_scale_rad_s <= 0:
            raise ConfigError("action输出尺度必须为正")


@dataclass(frozen=True)
class ControlConfig:
    physics_dt_s: float
    pd_dt_s: float
    policy_dt_s: float
    leg_kp: float
    leg_kd: float
    wheel_kv: float
    effort_limits_nm: tuple[float, ...]
    height_command_m: float

    def __post_init__(self) -> None:
        for name in ("physics_dt_s", "pd_dt_s", "policy_dt_s"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ConfigError(f"control.{name} 必须为正的有限值")
        if not self.physics_dt_s < self.pd_dt_s <= self.policy_dt_s:
            raise ConfigError("control时相必须满足 physics < pd <= policy")
        if self.leg_kp <= 0 or self.leg_kd < 0 or self.wheel_kv < 0:
            raise ConfigError("control PD增益范围不合法")


@dataclass(frozen=True)
class RewardConfig:
    version: str
    dt: float
    weights: tuple[float, ...]
    fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ConfigError("reward.dt 必须为正的有限值")
        if not self.weights:
            raise ConfigError("reward.weights 不能为空")


@dataclass(frozen=True)
class CurriculumConfig:
    version: str
    stage: int
    stages: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.stage < 0:
            raise ConfigError("curriculum.stage 不能为负")
        if self.stages and self.stage >= len(self.stages):
            raise ConfigError(f"curriculum.stage={self.stage} 超出stages数量")


@dataclass(frozen=True)
class EvaluationConfig:
    envs: int
    scored_envs: int
    seconds: float
    seed: int
    protocol_version: str
    conditions: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.envs < 1 or self.scored_envs < 1:
            raise ConfigError("evaluation.envs 与 scored_envs 必须至少为1")
        if self.scored_envs > self.envs:
            raise ConfigError("evaluation.scored_envs 不能超过 envs")
        if self.seconds <= 0:
            raise ConfigError("evaluation.seconds 必须为正")


@dataclass(frozen=True)
class LimitsConfig:
    smoke_updates: int
    formal_updates: int
    sample_limit: int | None = None
    evaluation_wall_s: float | None = None
    chain_wall_s: float | None = None

    def __post_init__(self) -> None:
        if self.smoke_updates < 0 or self.formal_updates < 0:
            raise ConfigError("limits中的update数不能为负")
        # BC作业的可执行额度只在distill；允许smoke/formal同时为0。
        # （LimitsConfig不感知算法，放行0/0；distill=0的拒绝由budget.reserve负责。）


@dataclass(frozen=True)
class StorageConfig:
    checkpoint_every_updates: int = 100
    keep_recent_checkpoints: int = 2
    metrics_every_updates: int = 10
    tensorboard: bool = True
    save_traces: bool = False
    trace_max_bytes: int = 16 * 1024 * 1024
    milestones: tuple[int, ...] = ()
    max_output_bytes: int | None = None
    terminal_reserve_bytes: int = 0

    def __post_init__(self) -> None:
        if self.checkpoint_every_updates < 1:
            raise ConfigError("storage.checkpoint_every_updates 必须至少为1")
        if self.keep_recent_checkpoints < 1:
            raise ConfigError("storage.keep_recent_checkpoints 必须至少为1")
        if self.metrics_every_updates < 1:
            raise ConfigError("storage.metrics_every_updates 必须至少为1")
        if self.metrics_every_updates > self.checkpoint_every_updates:
            raise ConfigError("storage.metrics_every_updates 不能大于checkpoint间隔")
        if self.trace_max_bytes < 0:
            raise ConfigError("storage.trace_max_bytes 不能为负")


@dataclass(frozen=True)
class RuntimeConfig:
    envs: int
    steps_per_env: int
    seed: int
    device: str = "cuda:0"
    headless: bool = True

    def __post_init__(self) -> None:
        if self.envs < 1 or self.steps_per_env < 1:
            raise ConfigError("runtime.envs 与 steps_per_env 必须至少为1")
        if self.seed < 0:
            raise ConfigError("runtime.seed 不能为负")

    @property
    def samples_per_update(self) -> int:
        return self.envs * self.steps_per_env


@dataclass(frozen=True)
class PpoConfig:
    learning_rate: float = 1e-5
    critic_learning_rate: float = 9e-5
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.001
    clip: float = 0.2
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    critic_warmup_updates: int = 0
    kl_limit: float = 0.015
    max_actor_backtracks: int = 4
    boundary_coefficient: float = 0.01
    boundary_clip_probability: float = 0.10
    dq_gain_coefficient: float = 0.01
    dq_gain_threshold: float = 2.5
    auxiliary_states: int = 128
    exploration_leg_rad: float = 0.2
    exploration_wheel_rad_s: float = 5.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.critic_learning_rate <= 0:
            raise ConfigError("ppo学习率必须为正")
        if self.epochs < 1 or self.minibatches < 1:
            raise ConfigError("ppo.epochs 与 minibatches 必须至少为1")
        if not 0 < self.gamma <= 1 or not 0 < self.lam <= 1:
            raise ConfigError("ppo.gamma 与 lam 必须在(0,1]")
        if self.critic_warmup_updates < 0:
            raise ConfigError("ppo.critic_warmup_updates 不能为负")
        if not math.isfinite(self.kl_limit) or self.kl_limit <= 0:
            raise ConfigError("ppo.kl_limit 必须为正的有限值")
        if type(self.max_actor_backtracks) is not int or not 0 <= self.max_actor_backtracks <= 8:
            raise ConfigError("ppo.max_actor_backtracks 必须为0至8的整数")
        if not 0 < self.boundary_clip_probability < 0.5:
            raise ConfigError("ppo.boundary_clip_probability 必须在0至0.5之间")
        if self.boundary_coefficient < 0 or self.dq_gain_coefficient < 0:
            raise ConfigError("ppo辅助损失系数不能为负")
        if self.dq_gain_threshold <= 0 or self.auxiliary_states < 1:
            raise ConfigError("ppo增益阈值与辅助样本上限必须为正")
        if self.exploration_leg_rad <= 0 or self.exploration_wheel_rad_s <= 0:
            raise ConfigError("ppo探索尺度必须为正")


@dataclass(frozen=True)
class NotificationsConfig:
    enabled: bool = False
    transport: str | None = None
    target: str | None = None

    def __post_init__(self) -> None:
        if self.enabled and not self.target:
            raise ConfigError("notifications.enabled=true 需要明确 target")


@dataclass(frozen=True)
class JobConfig:
    """解析后的不可变作业配置。"""

    name: str
    task: str
    algorithm: str
    backend: str
    model: ModelConfig
    initialization: InitializationConfig
    observation: ObservationConfig
    action: ActionConfig
    control: ControlConfig
    reward: RewardConfig
    curriculum: CurriculumConfig
    evaluation: EvaluationConfig
    limits: LimitsConfig
    storage: StorageConfig
    runtime: RuntimeConfig
    ppo: PpoConfig
    notifications: NotificationsConfig
    assets: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    source_root: Path
    config_path: Path
    config_sha256: str
    preset: str | None = None
    experiment_variables: Mapping[str, Any] = field(default_factory=dict)
    supervision: Any = None

    def as_run_record(self) -> dict[str, Any]:
        """写入run.json的解析后配置；路径保留绝对形式。"""

        def encode(value):
            if isinstance(value, Path):
                return str(value)
            if is_dataclass(value) and not isinstance(value, type):
                return {item.name: encode(getattr(value, item.name)) for item in fields(value)}
            if isinstance(value, Mapping):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, list | tuple):
                return [encode(item) for item in value]
            return value

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "name": self.name,
            "task": self.task,
            "algorithm": self.algorithm,
            "backend": self.backend,
            "preset": self.preset,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "source_root": str(self.source_root),
            "model": encode(self.model),
            "initialization": encode(self.initialization),
            "observation": encode(self.observation),
            "action": encode(self.action),
            "control": encode(self.control),
            "reward": encode(self.reward),
            "curriculum": encode(self.curriculum),
            "evaluation": encode(self.evaluation),
            "limits": encode(self.limits),
            "storage": encode(self.storage),
            "runtime": encode(self.runtime),
            "ppo": encode(self.ppo),
            "notifications": encode(self.notifications),
            "assets": encode(self.assets),
            "diagnostics": encode(self.diagnostics),
            "experiment_variables": encode(self.experiment_variables),
            "supervision": encode(self.supervision),
        }


def job_from_record(record: Mapping[str, Any]) -> JobConfig:
    """从run.json里的解析后配置重建JobConfig。

    运行语义只来自run自身记录；原始配置文件只作来源信息，
    之后被移动、删除或编辑都不影响已创建的run。
    """
    missing = [
        key
        for key in (
            "model",
            "initialization",
            "observation",
            "action",
            "control",
            "reward",
            "curriculum",
            "evaluation",
            "limits",
            "storage",
            "runtime",
            "ppo",
            "notifications",
        )
        if key not in record
    ]
    if missing:
        raise ConfigError(f"run记录缺少配置节: {missing}")

    def build(factory, payload, path):
        try:
            return factory(**dict(payload))
        except TypeError as exc:
            raise ConfigError(f"{path} 字段与当前schema不一致: {exc}") from exc

    initialization = dict(record["initialization"])
    if initialization.get("checkpoint"):
        initialization["checkpoint"] = Path(initialization["checkpoint"])
    action = dict(record["action"])
    action["order"] = tuple(action["order"])
    control = dict(record["control"])
    control["effort_limits_nm"] = tuple(control["effort_limits_nm"])
    reward = dict(record["reward"])
    reward["weights"] = tuple(reward["weights"])
    model = dict(record["model"])
    model["actor_hidden_dims"] = tuple(model["actor_hidden_dims"])
    model["critic_hidden_dims"] = tuple(model["critic_hidden_dims"])
    storage = dict(record["storage"])
    storage["milestones"] = tuple(storage["milestones"])
    return JobConfig(
        name=record["name"],
        task=record["task"],
        algorithm=record["algorithm"],
        backend=record["backend"],
        model=build(ModelConfig, model, "model"),
        initialization=build(InitializationConfig, initialization, "initialization"),
        observation=build(ObservationConfig, record["observation"], "observation"),
        action=build(ActionConfig, action, "action"),
        control=build(ControlConfig, control, "control"),
        reward=build(RewardConfig, reward, "reward"),
        curriculum=build(CurriculumConfig, record["curriculum"], "curriculum"),
        evaluation=build(EvaluationConfig, record["evaluation"], "evaluation"),
        limits=build(LimitsConfig, record["limits"], "limits"),
        storage=build(StorageConfig, storage, "storage"),
        runtime=build(RuntimeConfig, record["runtime"], "runtime"),
        ppo=build(PpoConfig, record["ppo"], "ppo"),
        notifications=build(NotificationsConfig, record["notifications"], "notifications"),
        assets=record.get("assets") or {},
        diagnostics=record.get("diagnostics") or {},
        source_root=Path(record["source_root"]),
        config_path=Path(record["config_path"]),
        config_sha256=record["config_sha256"],
        preset=record.get("preset"),
        experiment_variables=record.get("experiment_variables") or {},
        supervision=_supervision(record.get("supervision")),
    )


def _supervision(value):
    if value is None:
        return None
    from .supervision import SupervisionConfig

    try:
        return SupervisionConfig.from_record(value)
    except (TypeError, ValueError, KeyError) as exc:
        raise ConfigError(f"supervision: {exc}") from exc


def _dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} 必须是对象")
    return dict(value)


def _check_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{path} 含未知字段: {unknown}")


def _tuple_numbers(value: Any, path: str, *, length: int | None = None) -> tuple[float, ...]:
    if not isinstance(value, list | tuple):
        raise ConfigError(f"{path} 必须是数组")
    numbers = tuple(float(item) for item in value)
    if length is not None and len(numbers) != length:
        raise ConfigError(f"{path} 长度必须为 {length}，实际 {len(numbers)}")
    if not all(math.isfinite(item) for item in numbers):
        raise ConfigError(f"{path} 必须是有限值")
    return numbers


MOTION_PRESET: dict[str, Any] = {
    "task": "motion",
    "algorithm": "ppo",
    "backend": "isaac",
    "model": {
        "interface": "multiskill_mcu_v2",
        "actor_hidden_dims": [2048, 1024, 512],
        "critic_hidden_dims": [512, 512, 256],
        "actor_input": 31,
        "critic_input": 863,
        "critic_privileged_version": "motion_fullchain_antiosc863_v1",
        "action_version": "motion31_kinematic_support_centered_015_40_v1",
    },
    "initialization": {"mode": "scratch"},
    "observation": {
        "frame_dim": 22,
        "actor_dim": 31,
        "critic_dim": 863,
        "leg_input_scale_rad": 1.5,
        "wheel_input_scale_rad_s": 100.0,
    },
    "action": {
        "dim": 6,
        "order": [
            "left_hip_pitch",
            "left_active_link_pitch",
            "left_wheel_spin",
            "right_hip_pitch",
            "right_active_link_pitch",
            "right_wheel_spin",
        ],
        "leg_scale_rad": 0.15,
        "wheel_scale_rad_s": 40.0,
        "version": "motion31_kinematic_support_centered_015_40_v1",
    },
    "control": {
        "physics_dt_s": 0.001,
        "pd_dt_s": 0.005,
        "policy_dt_s": 0.01,
        "leg_kp": 40.0,
        "leg_kd": 2.0,
        "wheel_kv": 0.2,
        "effort_limits_nm": [40.0, 40.0, 5.5, 40.0, 40.0, 5.5],
        "height_command_m": 0.34,
    },
    "reward": {
        "version": "student_motion31_parallel_quality_v10",
        "dt": 0.01,
        "weights": [
            0.5,
            4.0,
            2.0,
            1.0,
            2.0,
            2.0,
            -1.0,
            -0.5,
            -0.5,
            -0.2,
            -0.3,
            -0.15,
            -0.02,
            -0.002,
        ],
        "fields": {
            "linear_denominator": 1.0,
            "yaw_denominator": 0.25,
            "lateral_denominator": 0.09,
            "height_target_m": 0.34,
            "height_scale_m": 0.04,
            "posture_scale_rad": 0.2,
            "mirror_scale_rad": 0.1,
        },
    },
    "curriculum": {"version": "motion31_supported015_course_v1", "stage": 0, "stages": []},
    "evaluation": {
        "envs": 64,
        "scored_envs": 8,
        "seconds": 31.52,
        "seed": 104,
        "protocol_version": "motion_physical_antiosc_support_assessment_v1",
        "conditions": [],
    },
    "limits": {"smoke_updates": 0, "formal_updates": 10000, "evaluation_wall_s": 10800},
    "storage": {
        "checkpoint_every_updates": 100,
        "keep_recent_checkpoints": 2,
        "metrics_every_updates": 10,
        "tensorboard": True,
        "save_traces": False,
        "trace_max_bytes": 16777216,
        "milestones": [],
    },
    "runtime": {"envs": 4096, "steps_per_env": 64, "seed": 2026110501, "device": "cuda:0"},
    "ppo": {
        "learning_rate": 1e-5,
        "critic_learning_rate": 9e-5,
        "epochs": 4,
        "minibatches": 4,
        "gamma": 0.99,
        "lam": 0.95,
        "entropy_coef": 0.001,
        "clip": 0.2,
        "value_coef": 0.5,
        "max_grad_norm": 1.0,
        "critic_warmup_updates": 5,
    },
    "notifications": {"enabled": False},
    "assets": {},
    "diagnostics": {},
}

# 所有配置共用的小节默认值；preset只覆盖它声明的内容。
BASE_DEFAULTS: dict[str, Any] = {
    "initialization": {"mode": "scratch"},
    "storage": {
        "checkpoint_every_updates": 100,
        "keep_recent_checkpoints": 2,
        "metrics_every_updates": 10,
        "tensorboard": True,
        "save_traces": False,
        "trace_max_bytes": 16 * 1024 * 1024,
        "max_output_bytes": None,
        "terminal_reserve_bytes": 0,
        "milestones": [],
    },
    "ppo": {
        "learning_rate": 1e-5,
        "critic_learning_rate": 9e-5,
        "epochs": 4,
        "minibatches": 4,
        "critic_warmup_updates": 0,
    },
    "notifications": {"enabled": False},
}

RECOVERY_CUBIC_PRESET = {
    **MOTION_PRESET,
    "task": "recovery",
    "backend": "mujoco",
    "model": {
        **MOTION_PRESET["model"],
        "interface": "recovery_cubic_physical_v2",
        "critic_input": 878,
        "critic_privileged_version": "recovery_causal_execution878_v2",
        "action_version": "recovery_cubic_physical_Gaussian_v2",
    },
    "observation": {**MOTION_PRESET["observation"], "critic_dim": 878},
    "action": {
        **MOTION_PRESET["action"],
        "wheel_scale_rad_s": 100.0,
        "version": "recovery_cubic_physical_Gaussian_v2",
    },
    "control": {
        **MOTION_PRESET["control"],
        "effort_limits_nm": [40.0, 40.0, 2.44694786377709, 40.0, 40.0, 2.44694786377709],
    },
    "reward": {
        "version": "recovery_physical_entry_hold_v2",
        "dt": 0.01,
        "weights": [1.0],
        "fields": {"source": "native_stand.tasks.recovery.cubic_task.RecoveryLifecycle"},
    },
    "curriculum": {"version": "recovery_cubic_R0_R1_R2_v1", "stage": 0, "stages": []},
    "evaluation": {
        "envs": 1,
        "scored_envs": 1,
        "seconds": 23.0,
        "seed": 89001,
        "protocol_version": "recovery_own_entry_hold10_v2",
        "conditions": [],
    },
    "runtime": {"envs": 1, "steps_per_env": 256, "seed": 89001, "device": "cpu", "headless": True},
    "ppo": {
        **MOTION_PRESET["ppo"],
        "gamma": 0.9980019986673331,
        "entropy_coef": 0.0,
        "value_coef": 1.0,
        "epochs": 2,
        "minibatches": 2,
        "exploration_leg_rad": 0.02,
        "exploration_wheel_rad_s": 2.0,
        "dq_gain_coefficient": 0.0,
        "boundary_coefficient": 0.0,
    },
}

MOTION48_PRESET = _preset_copy.deepcopy(MOTION_PRESET)
MOTION48_PRESET["control"]["effort_limits_nm"] = list(_MotionMotorLimits().caps)
MOTION48_PRESET["assets"] = {"motor_limits": _MotionMotorLimits().as_dict()}
PRESETS: dict[str, dict[str, Any]] = {
    "jump_isaac_v1": {"task": "jump", "algorithm": "ppo", "backend": "isaac"},
    "motion48_supported_v1": MOTION48_PRESET,
    "motion_supported_v1": MOTION_PRESET,
    "recovery_cubic_v2": RECOVERY_CUBIC_PRESET,
}

SECTION_KEYS = (
    "model",
    "initialization",
    "observation",
    "action",
    "control",
    "reward",
    "curriculum",
    "evaluation",
    "limits",
    "storage",
    "runtime",
    "ppo",
    "notifications",
    "assets",
    "diagnostics",
    "experiment_variables",
    "supervision",
)


def _section_allowed(name: str) -> set[str]:
    mapping = {
        "model": ModelConfig,
        "initialization": InitializationConfig,
        "observation": ObservationConfig,
        "action": ActionConfig,
        "control": ControlConfig,
        "reward": RewardConfig,
        "curriculum": CurriculumConfig,
        "evaluation": EvaluationConfig,
        "limits": LimitsConfig,
        "storage": StorageConfig,
        "runtime": RuntimeConfig,
        "ppo": PpoConfig,
        "notifications": NotificationsConfig,
    }
    cls = mapping.get(name)
    if cls is None:
        return set()
    return {item.name for item in fields(cls)}


def load_job_config(path: str | Path) -> JobConfig:
    """解析并校验一个作业配置。

    步骤：读取JSON → 展开preset → 逐节校验字段 → 构造不可变对象 → 解析路径。
    """
    config_path = Path(path).resolve()
    raw = _dict(read_json(config_path), "config")
    if "schema_version" in raw and raw["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"config.schema_version 只支持 {CONFIG_SCHEMA_VERSION}")
    allowed_top = {
        "schema_version",
        "name",
        "task",
        "algorithm",
        "backend",
        "preset",
        "source_root",
        "assets",
        "diagnostics",
        "experiment_variables",
        *SECTION_KEYS,
    }
    _check_keys(raw, allowed_top, "config")

    preset_name = raw.get("preset")
    if preset_name is not None:
        if preset_name not in PRESETS:
            raise ConfigError(f"未知preset: {preset_name}；可用: {sorted(PRESETS)}")
        merged = {
            key: (dict(value) if isinstance(value, Mapping) else value)
            for key, value in BASE_DEFAULTS.items()
        }
        for key, value in PRESETS[preset_name].items():
            merged[key] = dict(value) if isinstance(value, Mapping) else value
    else:
        # 没有preset时也以BASE_DEFAULTS为底，缺节不会变成隐式常量。
        merged = {
            key: (dict(value) if isinstance(value, Mapping) else value)
            for key, value in BASE_DEFAULTS.items()
        }

    for key in ("task", "algorithm", "backend"):
        if key in raw:
            merged[key] = raw[key]
        if key not in merged:
            raise ConfigError(f"config缺少 {key}")
    if "name" not in raw:
        raise ConfigError("config缺少 name")
    name = str(raw["name"])
    if not name or "/" in name or " " in name:
        raise ConfigError("config.name 不能为空、不能含空格或斜杠")

    for section in SECTION_KEYS:
        if section not in raw:
            continue
        if section == "supervision" and raw[section] is None:
            merged[section] = None
            continue
        if section in ("assets", "diagnostics", "experiment_variables", "supervision"):
            merged[section] = _dict(raw[section], section)
            continue
        current = dict(merged.get(section) or {})
        current.update(_dict(raw[section], section))
        merged[section] = current

    for section in ("task", "algorithm", "backend"):
        check_value = merged.get(section)
        if not isinstance(check_value, str) or not check_value:
            raise ConfigError(f"config.{section} 必须是非空字符串")
    # 任务集合以注册表为准；新技能=登记新任务模块，不改共享配置校验。
    from native_stand.tasks.registry import TASKS

    if merged["task"] not in TASKS:
        from native_stand.tasks.registry import TASKS as _REG_TASKS

        raise ConfigError(f"config.task 未知: {merged['task']}；可用: {sorted(_REG_TASKS)}")
    if merged["algorithm"] not in ("ppo", "distillation", "world_model"):
        raise ConfigError(f"config.algorithm 未知: {merged['algorithm']}")
    # 后端合法集合与dispatch/registry同源；未知后端明确拒绝，不回退。
    from native_stand.tasks.registry import TASKS as _REG_TASKS

    allowed_backends = {
        b for definition in _REG_TASKS.values() for b in definition["supported_backends"]
    }
    if merged["backend"] not in allowed_backends:
        raise ConfigError(f"config.backend 未知: {merged['backend']}")

    for section in (
        "model",
        "observation",
        "action",
        "control",
        "reward",
        "curriculum",
        "evaluation",
        "limits",
        "runtime",
    ):
        if section not in merged:
            raise ConfigError(f"config缺少 {section} 节")
        _check_keys(merged[section], _section_allowed(section), section)

    base = config_path.parent
    from native_stand.artifacts.runtime_paths import RuntimePaths

    paths = RuntimePaths.from_assets(merged.get("assets", {}))
    merged["assets"] = paths.view(merged.get("assets", {}))
    source_root_raw = raw.get("source_root")
    source_root = (
        resolve_path(paths.resolve(source_root_raw), base)
        if source_root_raw
        else (Path(__file__).resolve().parents[3] / "src")
    )

    initialization_raw = dict(merged.get("initialization") or {})
    _check_keys(initialization_raw, _section_allowed("initialization"), "initialization")
    if initialization_raw.get("checkpoint") is not None:
        initialization_raw["checkpoint"] = resolve_path(
            paths.resolve(initialization_raw["checkpoint"]), base
        )

    model_raw = dict(merged["model"])
    model_raw["actor_hidden_dims"] = tuple(int(v) for v in model_raw["actor_hidden_dims"])
    model_raw["critic_hidden_dims"] = tuple(int(v) for v in model_raw["critic_hidden_dims"])

    observation_raw = dict(merged["observation"])
    action_raw = dict(merged["action"])
    action_raw["order"] = tuple(str(v) for v in action_raw["order"])

    control_raw = dict(merged["control"])
    control_raw["effort_limits_nm"] = _tuple_numbers(
        control_raw["effort_limits_nm"], "control.effort_limits_nm", length=action_raw["dim"]
    )

    reward_raw = dict(merged["reward"])
    reward_raw["weights"] = _tuple_numbers(reward_raw["weights"], "reward.weights")

    curriculum_raw = dict(merged["curriculum"])
    curriculum_raw["stages"] = tuple(curriculum_raw.get("stages") or ())

    evaluation_raw = dict(merged["evaluation"])
    evaluation_raw["conditions"] = tuple(evaluation_raw.get("conditions") or ())

    limits_raw = dict(merged["limits"])
    storage_raw = dict(merged.get("storage") or {})
    _check_keys(storage_raw, _section_allowed("storage"), "storage")
    if storage_raw.get("milestones"):
        storage_raw["milestones"] = tuple(int(v) for v in storage_raw["milestones"])

    runtime_raw = dict(merged["runtime"])
    ppo_raw = dict(merged.get("ppo") or {})
    _check_keys(ppo_raw, _section_allowed("ppo"), "ppo")
    notifications_raw = dict(merged.get("notifications") or {})
    _check_keys(notifications_raw, _section_allowed("notifications"), "notifications")

    job = JobConfig(
        name=name,
        task=merged["task"],
        algorithm=merged["algorithm"],
        backend=merged["backend"],
        model=ModelConfig(**model_raw),
        initialization=InitializationConfig(**initialization_raw),
        observation=ObservationConfig(**observation_raw),
        action=ActionConfig(**action_raw),
        control=ControlConfig(**control_raw),
        reward=RewardConfig(**reward_raw),
        curriculum=CurriculumConfig(**curriculum_raw),
        evaluation=EvaluationConfig(**evaluation_raw),
        limits=LimitsConfig(**limits_raw),
        storage=StorageConfig(**storage_raw),
        runtime=RuntimeConfig(**runtime_raw),
        ppo=PpoConfig(**ppo_raw),
        notifications=NotificationsConfig(**notifications_raw),
        assets=dict(merged.get("assets") or {}),
        diagnostics=dict(merged.get("diagnostics") or {}),
        source_root=source_root,
        config_path=config_path,
        config_sha256=file_sha256(config_path),
        preset=preset_name,
        experiment_variables=dict(merged.get("experiment_variables") or {}),
        supervision=_supervision(merged.get("supervision")),
    )
    _validate_cross_section(job)
    return job


def _validate_cross_section(job: JobConfig) -> None:
    """跨节校验：语义冲突在这里一次报出。"""
    if job.assets.get("scene"):
        from native_stand.simulation.scenes import SceneConfig

        try:
            scene = SceneConfig(**job.assets["scene"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"scene: {exc}") from exc
        if job.backend == "isaac_recovery" and scene.ground_z_m != 0:
            raise ConfigError("Recovery PhysX当前仅声明world Z=0平地")
        if job.backend == "isaac" and scene != SceneConfig():
            raise ConfigError("Motion PhysX需要领域factory显式接收新的scene；不能忽略配置")
    if job.model.actor_input != job.observation.actor_dim:
        raise ConfigError(
            f"model.actor_input={job.model.actor_input} 与 "
            f"observation.actor_dim={job.observation.actor_dim} 不一致"
        )
    if job.task == "motion" and job.observation.actor_dim != job.observation.frame_dim + 9:
        raise ConfigError(
            f"Motion observation.actor_dim={job.observation.actor_dim} 与 "
            f"frame_dim+9={job.observation.frame_dim + 9} 不一致"
        )
    if job.model.critic_input != job.observation.critic_dim:
        raise ConfigError(
            f"model.critic_input={job.model.critic_input} 与 "
            f"observation.critic_dim={job.observation.critic_dim} 不一致"
        )
    if job.reward.dt != job.control.policy_dt_s:
        raise ConfigError(
            f"reward.dt={job.reward.dt} 必须等于 control.policy_dt_s={job.control.policy_dt_s}"
        )
    if len(job.control.effort_limits_nm) != job.action.dim:
        raise ConfigError("control.effort_limits_nm 必须与action维数一致")
    total = job.limits.smoke_updates + job.limits.formal_updates
    if job.limits.sample_limit is not None:
        needed = total * job.runtime.samples_per_update
        if needed > job.limits.sample_limit:
            raise ConfigError(f"声明样本上限 {job.limits.sample_limit} 小于计划样本 {needed}")
    if (
        job.algorithm == "ppo"
        and job.initialization.mode == "scratch"
        and job.task == "motion"
        and job.model.critic_privileged_version is None
    ):
        raise ConfigError("Motion PPO 需要明确的critic私有信息版本")
    if job.task == "recovery" and job.action.version.startswith("motion31"):
        raise ConfigError("recovery任务不能用Motion动作版本；请显式声明动作语义")
    # actor_only的两种合法组合（各校验一次）：
    # ① Recovery物理Gaussian（isaac_recovery/mujoco，须声明训练合同与准备产物）
    # ② Isaac Motion 015/80（显式C0协议 + runtime_template）
    recovery_cubic = (
        job.task == "recovery"
        and job.backend in ("mujoco", "isaac_recovery")
        and job.action.version == "recovery_cubic_physical_Gaussian_v2"
        and job.assets.get("recovery_cubic_preparation") is not None
        and job.assets.get("recovery_training_contract") is not None
    )
    motion_c0 = (
        job.task == "motion"
        and job.backend == "isaac"
        and job.assets.get("runtime_template") is not None
        and job.action.version == "motion_multiheight_support_center_015_80_v1"
    )
    if job.initialization.mode == "actor_only" and not (
        recovery_cubic
        or motion_c0
        or (
            job.task == "jump"
            and job.action.version
            in ("jump_unified_raw37_wheel48_v1", "jump_isaac_ABI4_raw37_wheel48_v1")
        )
    ):
        raise ConfigError(
            "actor_only只接入：Recovery物理Gaussian（isaac_recovery/mujoco，含训练合同）"
            "、显式C0协议的Isaac Motion 015/80，或显式ABI4 Jump作业"
        )
    if job.storage.save_traces and job.storage.trace_max_bytes == 0:
        raise ConfigError("storage.save_traces=true 需要正的trace_max_bytes")


def with_initialization(
    job: JobConfig, checkpoint: str | Path, sha256: str | None = None
) -> JobConfig:
    """由完整checkpoint生成恢复配置；SHA默认从文件实际计算。"""
    path = Path(checkpoint).resolve()
    digest = file_sha256(path) if sha256 is None else sha256
    return replace(
        job,
        initialization=InitializationConfig(mode="full", checkpoint=path, sha256=digest),
    )
