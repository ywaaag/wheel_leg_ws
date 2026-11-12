"""Bounded collection using an explicit sampling factory, with truthful ownership.

Actor runs on the observation device. Complete tensors are copied to host only
at chunk boundaries. Physical counts are reported by the backend, never inferred.
Fresh process continuation starts a new episode segment, retaining committed data
and already charged physical consumption; it does not restore hidden physics.
"""

import uuid
from pathlib import Path

from native_stand.artifacts.io import file_sha256, read_json, write_json, write_npz
from native_stand.learning.inference import InferenceSpec, load_model
from native_stand.simulation.dispatch import execution_service


def make_environment(job):
    import importlib

    service = execution_service(job)
    if service["service"] == "unsupported":
        raise ValueError(service["reason"])
    module = importlib.import_module(service["module"])
    factory = getattr(module, "make_collection_env", None)
    if factory is None:
        raise ValueError("Backend has no explicit collection factory: " + service["module"])
    # The factory owns cleanup if construction fails before returning an env.
    return factory(job, resources=job.runtime)


def collect(
    job,
    source,
    spec,
    steps,
    directory,
    on_steps,
    *,
    stop_requested=lambda: False,
    chunk_steps=64,
    required_conditions=(),
):
    import numpy as np
    import torch

    if steps < 1 or chunk_steps < 1:
        raise ValueError("Positive bounded collection/chunk sizes required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "collection_state.json"
    state = read_json(state_path) or dict(
        version="collection_state_v1",
        collection_id=uuid.uuid4().hex,
        policy_steps=0,
        physical_steps=0,
        sources=[],
        requested_policy_steps=steps,
        source_sha256=source["sha256"],
    )
    if state.get("status") == "ENGINEERING_FAULT":
        raise ValueError(
            "Failed collection requires explicit accounting/discard before retry; consumption is retained"
        )
    if state["requested_policy_steps"] != steps or state["source_sha256"] != source["sha256"]:
        raise ValueError("Collection request/incumbent changed on continuation")
    # Bind immutable source before constructing resources.
    model = load_model(source["path"], InferenceSpec.from_record(spec), sha256=source["sha256"])
    write_json(state_path, state)
    env = None
    buffers = []
    actual_envs = None
    segment = uuid.uuid4().hex
    manifest = directory / "collection.json"

    def flush():
        if not buffers:
            return
        keys = buffers[0].keys()
        arrays = {
            key: torch.stack([row[key] for row in buffers]).flatten(0, 1).cpu().numpy()
            for key in keys
            if isinstance(buffers[0][key], torch.Tensor)
        }
        episodes = arrays["episodes"]
        environments = arrays["environment_id"]
        arrays["physical_episode_keys"] = np.array(
            [
                f'{state["collection_id"]}:{segment}:{int(e)}:{int(p)}'
                for e, p in zip(environments, episodes, strict=True)
            ]
        )
        arrays["sample_ids"] = np.array(
            [
                f'{state["collection_id"]}:{int(t)}:{int(e)}'
                for t, e in zip(arrays["policy_index"], environments, strict=True)
            ]
        )
        output = directory / f'chunk_{len(state["sources"]):05d}.npz'
        write_npz(output, arrays)
        state["sources"].append(
            dict(
                path=str(output.resolve()),
                sha256=file_sha256(output),
                collection_id=state["collection_id"],
            )
        )
        state["committed_policy_steps"] = state["policy_steps"]
        write_json(state_path, state)
        write_json(
            manifest,
            dict(
                version="collection_chunks_v1",
                sources=state["sources"],
                collection_id=state["collection_id"],
                requested_envs=job.runtime.envs,
                actual_envs=actual_envs,
                policy_steps=state["policy_steps"],
                physical_steps=state["physical_steps"],
                complete=state["policy_steps"] >= steps,
                physics_state_restored=False,
            ),
        )
        buffers.clear()

    try:
        env = make_environment(job)
        obs = env.observations()["student_actor"]
        if obs.ndim != 2 or obs.shape[1] != spec["input_dim"] or len(obs) < 1:
            raise ValueError("Invalid collection observation layout")
        actual_envs = len(obs)
        if getattr(env, "num_envs", actual_envs) != actual_envs:
            raise ValueError("Environment metadata differs from actual observation rows")
        model.actor.to(obs.device)
        counters = torch.zeros(actual_envs, dtype=torch.long, device=obs.device)
        ids = torch.arange(actual_envs, device=obs.device)
        while state["policy_steps"] < steps and not stop_requested():
            obs = env.observations()["student_actor"]
            if obs.shape != (actual_envs, spec["input_dim"]):
                raise ValueError("Collection environment layout changed")
            # Clone pre-step input; engines may reuse their observation storage.
            current = obs.detach().clone()
            try:
                result = env.step(model.mean(obs))
            except BaseException as failure:
                # A backend can fail after real substeps without returning a
                # transition. Keep its measured consumption, never fabricate a
                # full action interval or silently refund the failed call.
                measured = getattr(failure, "actual_physics_steps", None)
                state["failed_step_consumption"] = "UNKNOWN"
                if measured is not None:
                    durations = torch.as_tensor(measured).reshape(-1)
                    if durations.shape != (actual_envs,) or bool((durations < 0).any()):
                        failure.add_note(
                            "Invalid fault physical-step receipt; reconciliation required"
                        )
                    else:
                        executed = int(durations.sum())
                        state["physical_steps"] += executed
                        state["failed_step_consumption"] = dict(
                            actual_physics_steps=durations.tolist(),
                            transition_committed=False,
                            transition_state="UNKNOWN",
                        )
                        write_json(state_path, state)
                        if on_steps:
                            try:
                                on_steps(executed)
                            except BaseException as charging:
                                failure.add_note(
                                    "Fault substeps require ledger reconciliation: "
                                    + repr(charging)
                                )
                raise
            extra = result.get("extras", {})
            actual = extra.get("actual_physics_steps")
            if actual is None:
                raise ValueError("Backend must report actual physical substeps")
            durations = torch.as_tensor(actual, device=obs.device).reshape(-1)
            if durations.shape != (actual_envs,) or bool((durations < 0).any()):
                raise ValueError("Invalid per-environment physical consumption")
            executed = int(durations.sum())
            # Charge before committing data; a later failure never refunds work.
            if on_steps:
                on_steps(executed)
            state["physical_steps"] += executed
            owner = extra.get("control_owner")
            conditions = extra.get("collection_conditions", {})
            if owner is None:
                raise ValueError("Collection backend omitted actual control owner")
            if set(required_conditions) - conditions.keys():
                raise ValueError("Collection backend omitted routing conditions")
            owner = torch.as_tensor(owner, device=obs.device).reshape(-1)
            if owner.shape != (actual_envs,) or bool(((owner < 0) | (owner > 2)).any()):
                raise ValueError("Invalid actual control owner")
            terminated = torch.as_tensor(result["terminated"], device=obs.device).reshape(-1).bool()
            truncated = torch.as_tensor(result["truncated"], device=obs.device).reshape(-1).bool()
            if terminated.shape != (actual_envs,) or truncated.shape != (actual_envs,):
                raise ValueError("Invalid per-environment done shape")
            row = dict(
                states=current,
                episodes=counters.clone(),
                environment_id=ids,
                policy_index=ids.new_full((actual_envs,), state["policy_steps"]),
                control_owner=owner.detach().clone(),
                terminated=terminated.clone(),
                truncated=truncated.clone(),
                actual_physics_steps=durations.clone(),
                transition_valid=durations
                == round(job.control.policy_dt_s / job.control.physics_dt_s),
            )
            for key, value in conditions.items():
                value = torch.as_tensor(value, device=obs.device)
                if value.shape != (actual_envs,):
                    raise ValueError("Routing conditions require actual batch rows")
                row["condition_" + key] = value.detach().clone()
            buffers.append(row)
            state["policy_steps"] += 1
            counters += terminated | truncated
            if len(buffers) >= chunk_steps:
                flush()
        flush()
        if not manifest.exists():
            write_json(
                manifest,
                dict(
                    version="collection_chunks_v1",
                    sources=state["sources"],
                    collection_id=state["collection_id"],
                    complete=state["policy_steps"] >= steps,
                ),
            )
        immutable = directory / f'collection_{state["policy_steps"]:08d}.json'
        write_json(immutable, read_json(manifest))
        return dict(
            path=str(immutable.resolve()),
            sha256=file_sha256(immutable),
            kind="collection_chunks_v1",
            collection_id=state["collection_id"],
            complete=state["policy_steps"] >= steps,
        )
    except BaseException:
        state["status"] = "ENGINEERING_FAULT"
        flush()
        write_json(state_path, state)
        raise
    finally:
        if env is not None:
            env.close()
