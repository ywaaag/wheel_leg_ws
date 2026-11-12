"""Ordered supervision phases inside the existing worker; BC math stays in loop.py."""

from pathlib import Path

from native_stand.artifacts.checkpoint import load_actor_source
from native_stand.artifacts.io import file_sha256, read_json, write_json, write_npz
from native_stand.learning.distillation.loop import load_dataset, run_distillation
from native_stand.learning.distillation.routing import ActionLabels, Teacher, label
from native_stand.learning.inference import InferenceSpec, load_model

from .supervision import semantic_identity


def execute(
    job,
    directory,
    *,
    stop_requested,
    on_round_complete,
    on_round_saved,
    on_physical_steps=None,
    collector=None,
    index=None,
    logical_run_id=None,
):
    import json

    import numpy as np
    import torch

    from native_stand.artifacts.deployment import export_package
    from native_stand.artifacts.runtime_paths import RuntimePaths, execution_job
    from native_stand.learning.distillation.splits import EpisodeSplit

    paths = RuntimePaths.from_assets(job.assets)
    job = execution_job(job)
    recipe = job.supervision
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    state_file = root / "supervision_state.json"
    state = paths.view(read_json(state_file)) or dict(
        version="supervision_flow_v2",
        logical_run_id=logical_run_id or __import__("uuid").uuid4().hex,
        identity=semantic_identity(recipe),
        phase_cursor=0,
        committed=[],
        rounds=0,
        latest=None,
        dataset=recipe.dataset,
        collections=[recipe.dataset],
    )
    if logical_run_id is not None and state.get("logical_run_id") != logical_run_id:
        raise ValueError("Supervision logical run identity differs")
    if state.get("version") != "supervision_flow_v2":
        raise ValueError("Old supervision state needs an explicit collection/split import")
    if state["identity"] != semantic_identity(recipe):
        raise ValueError("Teacher/data/routing/label semantics changed on resume")
    state.setdefault("collections", [recipe.dataset])
    teachers = [
        Teacher(
            t.name,
            load_model(t.path, InferenceSpec.from_record(t.inference), sha256=t.sha256),
            ActionLabels(**t.labels),
            t.priority,
            t.conditions or {},
            t.qualified,
        )
        for t in recipe.teachers
    ]
    dependencies = [dict(path=t.path, sha256=t.sha256) for t in recipe.teachers]
    dependencies += [dict(recipe.dataset), *recipe.rehearsal]
    if recipe.initial_student:
        dependencies.append(recipe.initial_student)
    if recipe.dataset.get("kind") == "episode_dataset_view_v1":
        dependencies.extend(read_json(recipe.dataset["path"])["sources"])
    if index:
        from native_stand.artifacts.model_refs import ModelReference

        for ref in dependencies:
            if file_sha256(ref["path"]) != ref["sha256"]:
                raise ValueError("Supervision dependency SHA differs")
            index.register(
                ModelReference(ref["sha256"], ref["path"], ("distillation",), run=str(root))
            )
    phases = list(recipe.phases) * recipe.repetitions
    for cursor in range(state["phase_cursor"], len(phases)):
        if stop_requested():
            return state, False
        phase = phases[cursor]
        folder = root / f"phase_{cursor:03d}_{phase.operation}"
        folder.mkdir(exist_ok=True)
        if phase.operation == "collect":
            if collector is None:
                from native_stand.simulation.collection import collect

                collector = collect
            source = state.get("latest") or recipe.initial_student
            if source is None:
                raise ValueError("Student collection requires explicit incumbent")
            options = {}
            if collector.__module__ == "native_stand.simulation.collection":
                options = dict(
                    stop_requested=stop_requested,
                    required_conditions=tuple(
                        set().union(*(t.conditions or {} for t in recipe.teachers))
                    ),
                )
            ref = collector(
                job, source, recipe.student, phase.steps, folder, on_physical_steps, **options
            )
            state["dataset"] = ref
            if not ref.get("complete", True):
                write_json(state_file, state)
                if index:
                    from native_stand.artifacts.dataset import expand_sources
                    from native_stand.artifacts.model_refs import ModelReference

                    for chunk in expand_sources(ref, paths=paths):
                        index.register(
                            ModelReference(
                                chunk["sha256"],
                                chunk["path"],
                                ("distillation",),
                                run=str(root),
                                kind="collection_chunk",
                            )
                        )
                return state, False
            state["collections"].append(ref)
        elif phase.operation == "label":
            from native_stand.artifacts.dataset import (
                expand_sources,
                read_source,
                rehearsal_sources,
                unique_sources,
            )

            sources = []
            for ref in state["collections"]:
                if ref.get("kind") == "legacy_episode_index_v1":
                    from native_stand.artifacts.dataset import import_episode_index

                    sources.extend(
                        import_episode_index(
                            ref["path"], folder / "imported_view.json", paths=paths
                        )["sources"]
                    )
                else:
                    sources.extend(expand_sources(ref, paths=paths))
            for ref in recipe.rehearsal:
                sources.extend(rehearsal_sources(ref, paths=paths))
            sources = unique_sources(sources)
            chunks = [read_source(r) for r in sources]
            # Sources may carry different extra telemetry. Only the conditions
            # actually used by this Teacher set must exist in every source;
            # missing routing evidence is still rejected, never filled with 0.
            names = set().union(*(teacher.conditions for teacher in teachers))
            if any(not names <= chunk["conditions"].keys() for chunk in chunks):
                raise ValueError("Rehearsal routing conditions are incomplete")
            from native_stand.artifacts.dataset import physical_keys, sample_keys

            # Deduplicate actual samples, including aliases reached through views.
            seen = set()
            for ref, chunk in zip(sources, chunks, strict=True):
                ids = sample_keys(ref, chunk)
                mask = np.array([key not in seen and not seen.add(key) for key in ids])
                for key in (
                    "states",
                    "episodes",
                    "physical_episode_keys",
                    "roles",
                    "sample_ids",
                    "control_owner",
                    "behavior_valid",
                ):
                    if chunk.get(key) is not None:
                        chunk[key] = chunk[key][mask]
                chunk["conditions"] = {k: v[mask] for k, v in chunk["conditions"].items()}
            # Same physical episode keeps the same global key across Teachers.
            episode_keys = []
            for r, chunk in zip(sources, chunks, strict=True):
                episode_keys.extend(physical_keys(r, chunk))
            key_ids = {key: i for i, key in enumerate(sorted(set(episode_keys)))}
            data = dict(
                states=np.concatenate([c["states"] for c in chunks]),
                episodes=np.array([key_ids[k] for k in episode_keys]),
            )
            conditions = {
                key: np.concatenate([c["conditions"][key] for c in chunks]) for key in names
            }
            labels, owner, valid = label(
                teachers,
                torch.from_numpy(data["states"]),
                conditions,
                ActionLabels(**recipe.student_labels),
                student_kind=recipe.student["kind"],
            )
            behavior_valid = np.concatenate(
                [
                    c["behavior_valid"]
                    if c.get("behavior_valid") is not None
                    else np.ones(len(c["states"]), dtype=bool)
                    for c in chunks
                ]
            )
            valid &= torch.from_numpy(behavior_valid).bool()
            from native_stand.artifacts.dataset import create_view

            normalized = [
                {**r, "collection_id": r.get("collection_id", r["sha256"])} for r in sources
            ]
            if any(r.get("role") == "holdout" for r in recipe.rehearsal):
                raise ValueError("Final holdout cannot be rehearsal")
            if (
                state.get("aggregate")
                and file_sha256(state["aggregate"]["path"]) != state["aggregate"]["sha256"]
            ):
                raise ValueError("Accumulated split/view identity differs")
            previous = (
                paths.view(read_json(state["aggregate"]["path"]))
                if state.get("aggregate")
                else None
            )
            view = create_view(
                folder / "view.json",
                normalized,
                previous=previous,
                seed=job.runtime.seed,
                fractions=(
                    1 - recipe.validation_fraction - recipe.holdout_fraction,
                    recipe.validation_fraction,
                    recipe.holdout_fraction,
                ),
            )
            state["aggregate"] = dict(
                path=str((folder / "view.json").resolve()),
                sha256=file_sha256(folder / "view.json"),
                kind="episode_dataset_view_v1",
            )
            roles = np.array([view["episode_roles"][key] for key in episode_keys])
            teacher_records = [
                dict(
                    name=t.name,
                    sha256=t.model.source.sha256,
                    input_contract=t.model.spec.input_contract,
                    output_contract=t.model.spec.output_contract,
                    label_transform=recipe.teachers[i].labels,
                )
                for i, t in enumerate(teachers)
            ]
            provenance = dict(
                teachers=teacher_records,
                actual_owner_column="teacher_owner",
                label_space=recipe.student_labels,
                input_version=job.model.interface,
                action_version=job.action.version,
                action_units="policy_output",
                action_dim=job.action.dim,
                teacher_set=state["identity"],
            )
            output = folder / "labels.npz"
            write_npz(
                output,
                dict(
                    states=data["states"],
                    episodes=np.array(data["episodes"]),
                    teacher_actions=labels.numpy(),
                    valid=valid.numpy(),
                    teacher_owner=owner.numpy(),
                    roles=roles,
                    physical_episode_keys=np.array(episode_keys),
                    sample_ids=np.array(
                        [
                            key
                            for r, c in zip(sources, chunks, strict=True)
                            for key in sample_keys(r, c)
                        ]
                    ),
                    control_owner=np.concatenate(
                        [
                            c["control_owner"]
                            if c.get("control_owner") is not None
                            else np.full(len(c["states"]), 2)
                            for c in chunks
                        ]
                    ),
                    transition_valid=behavior_valid,
                    **{"condition_" + k: v for k, v in conditions.items()},
                    label_provenance=np.array(json.dumps(provenance)),
                ),
            )
            state["dataset"] = dict(path=str(output), sha256=file_sha256(output))
        elif phase.operation == "fit":
            ref = state["dataset"]
            if file_sha256(ref["path"]) != ref["sha256"]:
                raise ValueError("Fit dataset SHA differs")
            data = load_dataset(ref["path"])
            with np.load(ref["path"], allow_pickle=False) as raw:
                valid, roles = raw["valid"], raw["roles"]
            for key in ("states", "teacher_actions"):
                data[key] = data[key][valid]
            data["episodes"] = np.asarray(data["episodes"])[valid].tolist()
            roles = roles[valid]
            split = EpisodeSplit(
                *(
                    tuple(sorted(set(np.array(data["episodes"])[roles == r].tolist())))
                    for r in ("train", "development", "holdout")
                )
            )
            latest = folder / "distillation_latest.pt"
            resume = (
                torch.load(latest, map_location="cpu", weights_only=False)
                if latest.exists()
                else None
            )
            base = state.setdefault("fit_base", state["rounds"])
            write_json(state_file, state)
            done = (resume["round"] - base) if resume else 0
            initial = state.get("latest") or recipe.initial_student
            initial_state = load_actor_source(initial["path"]).payload["actor"] if initial else None
            prepared = dict(
                data=data,
                split=split,
                student_spec=recipe.student,
                initial_student=initial_state,
                identity=state["identity"],
                logical_run_id=state["logical_run_id"],
            )
            # Across phase boundaries retain Adam/RNG while a new aggregate supplies new data/split.
            if resume is None and state.get("latest"):
                resume = torch.load(state["latest"]["path"], map_location="cpu", weights_only=False)
                prepared["new_comparison_domain"] = (
                    resume["dataset_sha256"] != ref["sha256"]
                    or resume["split"] != split.as_record()
                )
                prepared["previous_domain"] = dict(
                    dataset_sha256=resume["dataset_sha256"],
                    split=resume["split"],
                    best={k: resume["best"][k] for k in ("round", "validation_loss")},
                    full_state=state["latest"],
                    best_state_field="best.state",
                )
                resume = {**resume, "dataset_sha256": ref["sha256"], "split": split.as_record()}
            if done < phase.rounds:
                run_distillation(
                    job,
                    dataset_path=ref["path"],
                    teacher_checkpoint=recipe.teachers[0].path,
                    output_dir=folder,
                    rounds=phase.rounds - done,
                    batch_size=recipe.batch_size,
                    learning_rate=recipe.learning_rate,
                    seed=job.runtime.seed,
                    prepared=prepared,
                    resume_state=resume,
                    stop_requested=stop_requested,
                    on_round_complete=on_round_complete,
                    on_round_saved=on_round_saved,
                )
            saved = torch.load(latest, map_location="cpu", weights_only=False)
            selected = folder / "student_actor.pt"
            if selected.is_file():
                state["selected"] = dict(path=str(selected), sha256=file_sha256(selected))
            state["rounds"] = saved["round"]
            state["latest"] = dict(path=str(latest), sha256=file_sha256(latest))
            write_json(state_file, state)
            if saved["round"] - base < phase.rounds:
                return state, False
            state.pop("fit_base", None)
        elif phase.operation == "compare":
            # Existing fit development evidence only; never extra simulation.
            if not state.get("latest"):
                raise ValueError("Compare requires fitted development evidence")
            value = torch.load(state["latest"]["path"], map_location="cpu", weights_only=False)
            write_json(
                folder / "comparison.json",
                dict(
                    validation_loss=value["validation_loss"],
                    extra_simulation_steps=0,
                    independent_certificate=False,
                ),
            )
        elif phase.operation == "export":
            value = torch.load(state["latest"]["path"], map_location="cpu", weights_only=False)
            actor = folder / "selected.pt"
            if not actor.exists():
                torch.save(dict(actor=value["best"]["state"], inference_spec=recipe.student), actor)
            export_package(
                load_actor_source(actor),
                InferenceSpec.from_record(recipe.student),
                folder / "package",
                execution=job.as_run_record()["control"],
                dependencies=dependencies,
                index=index,
            )
        if index:
            from native_stand.artifacts.dataset import expand_sources
            from native_stand.artifacts.model_refs import ModelReference

            refs = [state["dataset"]]
            if state.get("aggregate"):
                refs.append(state["aggregate"])
                refs.extend(expand_sources(state["aggregate"], paths=paths))
            if state.get("selected"):
                refs.append(state["selected"])
            for ref in refs:
                index.register(
                    ModelReference(
                        ref["sha256"],
                        ref["path"],
                        ("distillation",),
                        run=str(root),
                        kind="supervision_artifact",
                    )
                )
        state["phase_cursor"] = cursor + 1
        state["committed"].append(
            dict(phase=cursor, operation=phase.operation, dataset=state["dataset"])
        )
        write_json(state_file, state)
    return state, True
