"""Small explicit supervision recipe: ordered phases, bounded repetitions."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TeacherConfig:
    name: str
    path: str
    sha256: str
    inference: dict
    labels: dict
    priority: int = 0
    conditions: dict | None = None
    qualified: bool = True


@dataclass(frozen=True)
class PhaseConfig:
    operation: str
    steps: int = 0
    rounds: int = 0

    def __post_init__(self):
        if self.operation not in ("collect", "label", "fit", "compare", "export"):
            raise ValueError("Unknown supervision phase")
        if type(self.steps) is not int or type(self.rounds) is not int:
            raise ValueError("Phase counts must be integers")
        if self.operation != "collect" and self.steps or self.operation != "fit" and self.rounds:
            raise ValueError("steps only belongs to collect; rounds only belongs to fit")
        if self.steps < 0 or self.rounds < 0:
            raise ValueError("Negative supervision consumption")
        if self.operation == "collect" and self.steps == 0:
            raise ValueError("Collection requires a bounded physical step budget")
        if self.operation == "fit" and self.rounds == 0:
            raise ValueError("Fit requires supervised rounds")


@dataclass(frozen=True)
class SupervisionConfig:
    teachers: tuple[TeacherConfig, ...]
    phases: tuple[PhaseConfig, ...]
    student: dict
    student_labels: dict
    dataset: dict
    repetitions: int = 1
    batch_size: int = 256
    learning_rate: float = 0.001
    validation_fraction: float = 0.2
    holdout_fraction: float = 0.2
    initial_student: dict | None = None
    rehearsal: tuple[dict, ...] = ()
    loss: dict | None = None

    def __post_init__(self):
        if self.loss is not None:
            if (
                set(self.loss) != {"kind", "leg_scale_rad", "wheel_scale_rad_s"}
                or self.loss["kind"] != "motion_physical_huber"
            ):
                raise ValueError("Unknown explicit supervision loss")
            import math

            if any(
                not isinstance(self.loss[k], int | float)
                or not math.isfinite(self.loss[k])
                or self.loss[k] <= 0
                for k in ("leg_scale_rad", "wheel_scale_rad_s")
            ):
                raise ValueError("Positive physical loss scales required")
            if self.student_labels.get("kind") != "motion_contract":
                raise ValueError("Motion physical Huber requires Motion label contract")
        if not self.teachers or len({t.name for t in self.teachers}) != len(self.teachers):
            raise ValueError("Unique explicit Teachers required")
        if self.repetitions < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid bounded supervision settings")
        if not self.phases or not any(p.operation == "fit" for p in self.phases):
            raise ValueError("Supervision requires a real fit phase")
        if (
            self.student.get("kind") == "recovery_physical"
            and self.student_labels.get("kind") != "physical"
        ):
            raise ValueError("Recovery deterministic Actor already emits physical labels")
        for teacher in self.teachers:
            if (
                teacher.inference.get("kind") == "recovery_physical"
                and teacher.labels.get("kind") != "physical"
            ):
                raise ValueError("Recovery Teacher output must not be cubic-decoded twice")
        labelled = False
        fitted = False
        for phase in self.phases:
            if phase.operation == "collect":
                labelled = False
            if phase.operation == "label":
                labelled = True
            if phase.operation == "fit":
                if not labelled:
                    raise ValueError("fit requires a preceding label phase")
                fitted = True
            if phase.operation in ("compare", "export") and not fitted:
                raise ValueError("Postprocessing requires a preceding fit")
        if self.repetitions > 1 and self.phases[0].operation not in ("collect", "label"):
            raise ValueError("Repeated recipe must establish its data domain")

    @classmethod
    def from_record(cls, record):
        return cls(
            **{
                **record,
                "teachers": tuple(TeacherConfig(**r) for r in record["teachers"]),
                "phases": tuple(PhaseConfig(**r) for r in record["phases"]),
                "rehearsal": tuple(record.get("rehearsal", ())),
            }
        )

    def record(self):
        return asdict(self)

    @property
    def rounds(self):
        return self.repetitions * sum(p.rounds for p in self.phases)

    def plan(self):
        return dict(
            phases=[asdict(p) for p in self.phases],
            repetitions=self.repetitions,
            supervised_rounds=self.rounds,
            collection_policy_steps=self.repetitions * sum(p.steps for p in self.phases),
            PPO_updates=0,
            starts_engine=False,
            writes=False,
        )


def preflight(job):
    """Actual submission boundary: verify only resources used by this recipe."""
    from native_stand.artifacts.dataset import expand_sources, rehearsal_sources
    from native_stand.artifacts.io import file_sha256, read_json
    from native_stand.artifacts.runtime_paths import RuntimePaths, execution_job
    from native_stand.learning.distillation.routing import ActionLabels
    from native_stand.learning.inference import InferenceSpec

    paths = RuntimePaths.from_assets(job.assets)
    job = execution_job(job)
    recipe = job.supervision
    refs = [dict(path=t.path, sha256=t.sha256) for t in recipe.teachers]
    refs.append(recipe.dataset)
    if recipe.initial_student:
        refs.append(recipe.initial_student)
    for ref in refs:
        if file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Supervision input SHA differs")
    if recipe.dataset.get("kind") == "legacy_episode_index_v1":
        for ref in paths.view(read_json(recipe.dataset["path"])):
            if file_sha256(ref["path"]) != ref["sha256"]:
                raise ValueError("Legacy dataset member SHA differs")
    else:
        expand_sources(recipe.dataset)
    for ref in recipe.rehearsal:
        rehearsal_sources(ref)
    for teacher in recipe.teachers:
        InferenceSpec.from_record(teacher.inference)
        ActionLabels(**teacher.labels)
    ActionLabels(**recipe.student_labels)
    InferenceSpec.from_record(recipe.student)
    return dict(status="CPU_SUPERVISION_PREFLIGHT", physical_preflight=False)


def semantic_identity(recipe):
    """Relocation changes paths, not the contract or dependency digest."""

    def identity(value):
        if isinstance(value, dict):
            bound = (
                isinstance(value.get("path"), str)
                and isinstance(value.get("sha256"), str)
                and len(value["sha256"]) == 64
            )
            return {k: identity(v) for k, v in value.items() if not (bound and k == "path")}
        if isinstance(value, list | tuple):
            return [identity(v) for v in value]
        return value

    from native_stand.artifacts.io import json_digest

    return json_digest(identity(recipe.record()))
