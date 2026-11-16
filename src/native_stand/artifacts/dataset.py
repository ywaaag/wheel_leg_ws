"""Episode-stable dataset views and aggregation; holdout never becomes rehearsal."""

from pathlib import Path

from .io import file_sha256, json_digest, read_json, write_json

VERSION = "episode_dataset_view_v1"


def create_view(path, sources, *, fractions=(0.6, 0.2, 0.2), seed=0, previous=None):
    import hashlib

    if len(fractions) != 3 or min(fractions) < 0 or abs(sum(fractions) - 1) > 1e-8:
        raise ValueError("Dataset split fractions must sum to one")
    assignments = dict((previous or {}).get("episode_roles", {}))
    records = []
    for ref in unique_sources(sources):
        source = Path(ref["path"])
        if file_sha256(source) != ref["sha256"]:
            raise ValueError("Dataset source SHA differs")
        data = read_source(ref)
        split_view = ref.get("split_view")
        authority = None
        if split_view:
            if file_sha256(split_view["path"]) != split_view["sha256"]:
                raise ValueError("Rehearsal split identity differs")
            authority = read_json(split_view["path"])["episode_roles"]
        keys = data.get("physical_episode_keys")
        if keys is None:
            keys = [ref["collection_id"] + ":" + str(e) for e in data["episodes"]]
        for row_index, key in enumerate(keys):
            key = str(key)
            fraction = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16) / 2**64
            role = (
                str(data["roles"][row_index])
                if data.get("roles") is not None
                else ref.get("role")
                or (
                    "train"
                    if fraction < fractions[0]
                    else "development"
                    if fraction < sum(fractions[:2])
                    else "holdout"
                )
            )
            if role not in ("train", "development", "holdout"):
                raise ValueError("Unknown dataset partition")
            if authority is not None:
                role = authority[key]
            old = assignments.setdefault(key, role)
            if old != role:
                raise ValueError("Episode split changed during aggregation")
        records.append(dict(ref))
    record = dict(
        version=VERSION,
        sources=records,
        seed=seed,
        fractions=list(fractions),
        episode_roles=assignments,
    )
    record["identity"] = json_digest(
        {**record, "sources": [{k: v for k, v in r.items() if k != "path"} for r in records]}
    )
    write_json(path, record)
    return record


def load_view(path, *, roles=("train", "development"), rehearsal=False, paths=None):
    import numpy as np

    if rehearsal and "holdout" in roles:
        raise ValueError("Final holdout cannot be fed into rehearsal")
    record = read_json(path)
    if record["version"] != VERSION:
        raise ValueError("Dataset view version differs")
    batches = []
    for ref in record["sources"]:
        source = paths.resolve(ref["path"]) if paths else ref["path"]
        if file_sha256(source) != ref["sha256"]:
            raise ValueError("Dataset view source identity differs")
        with np.load(source, allow_pickle=False) as data:
            mask = np.array(
                [
                    record["episode_roles"][ref["collection_id"] + ":" + str(e)] in roles
                    for e in data["episodes"]
                ]
            )
            batches.append(
                {
                    key: data[key][mask]
                    for key in data.files
                    if data[key].ndim > 0 and len(data[key]) == len(mask)
                }
            )
    return record, batches


def read_source(reference, *, paths=None):
    """Explicit legacy field mapping; an absent episode ID is never guessed."""
    import numpy as np

    path = paths.resolve(reference["path"]) if paths else reference["path"]
    if file_sha256(path) != reference["sha256"]:
        raise ValueError("Dataset source SHA differs")
    with np.load(path, allow_pickle=False) as payload:
        states = payload[reference.get("states_key", "states")].copy()
        if "episode_id" in reference:
            episodes = np.full(len(states), reference["episode_id"])
        else:
            episodes = payload[reference.get("episodes_key", "episodes")].copy()
        conditions = {
            k.removeprefix("condition_"): payload[k].copy()
            for k in payload.files
            if k.startswith("condition_")
        }
        conditions.update(
            {
                key: payload[column].copy()
                for key, column in reference.get("condition_columns", {}).items()
            }
        )
        for name, value in reference.get("condition_values", {}).items():
            if name in conditions:
                raise ValueError("Static routing condition conflicts with measured column")
            conditions[name] = np.full(len(states), value)
        keys = (
            payload["physical_episode_keys"].copy() if "physical_episode_keys" in payload else None
        )
        roles = payload["roles"].copy() if "roles" in payload else None
        sample_ids = payload["sample_ids"].copy() if "sample_ids" in payload else None
        control_owner = payload["control_owner"].copy() if "control_owner" in payload else None
        behavior_valid = (
            payload["transition_valid"].copy() if "transition_valid" in payload else None
        )
        times = (
            payload[reference["sampling"]["time_key"]].copy() if reference.get("sampling") else None
        )
        eligible = (
            payload[reference["sampling"]["eligible_key"]].copy()
            if reference.get("sampling", {}).get("eligible_key")
            else None
        )
    if states.ndim != 2 or len(episodes) != len(states) or not np.isfinite(states).all():
        raise ValueError("Invalid dataset rows")
    result = dict(
        states=states,
        episodes=episodes,
        conditions=conditions,
        physical_episode_keys=keys,
        roles=roles,
        sample_ids=sample_ids,
        control_owner=control_owner,
        behavior_valid=behavior_valid,
    )
    if times is not None:
        if result["sample_ids"] is None:
            result["sample_ids"] = np.array(
                [
                    f'{reference.get("collection_id", reference["sha256"])}:{reference["sha256"]}:{i}'
                    for i in range(len(states))
                ]
            )
        indices = episode_sample_indices(
            episodes, times, reference["sampling"]["windows"], eligible=eligible
        )
        result = {
            k: (
                {name: value[indices] for name, value in v.items()}
                if k == "conditions"
                else None
                if v is None
                else v[indices]
            )
            for k, v in result.items()
        }
    return result


def episode_sample_indices(episodes, times, windows, *, eligible=None):
    """Fixed disjoint time-window quotas; retain chronological physical identities."""
    import math

    import numpy as np

    if len(times) != len(episodes) or not np.isfinite(times).all() or not windows:
        raise ValueError("Explicit finite per-row times and sampling windows required")
    mask = np.ones(len(times), bool) if eligible is None else np.asarray(eligible, bool)
    if mask.shape != (len(times),):
        raise ValueError("Sample eligibility shape differs")
    end = -math.inf
    for lo, hi, quota in windows:
        if (
            not math.isfinite(lo)
            or not math.isfinite(hi)
            or lo < end
            or lo >= hi
            or type(quota) is not int
            or quota < 1
        ):
            raise ValueError("Sampling windows must be disjoint, ordered with positive quotas")
        end = hi
    selected = []
    for episode in sorted(set(episodes.tolist())):
        for lo, hi, quota in windows:
            rows = np.flatnonzero((episodes == episode) & (times >= lo) & (times < hi) & mask)
            if len(rows) > quota:
                rows = rows[np.linspace(0, len(rows) - 1, quota, dtype=int)]
            selected.extend(rows.tolist())
    return np.asarray(sorted(selected), dtype=int)


def import_episode_index(index_path, output, *, paths=None):
    """Read-only import of existing labelled episode records, without copying NPZ."""
    index_path = paths.resolve(index_path) if paths else index_path
    entries = read_json(index_path)
    if paths:
        entries = paths.view(entries)
    sources = []
    assignments = {}
    roles = {
        "train": "train",
        "validation": "development",
        "development": "development",
        "final": "holdout",
        "holdout": "holdout",
    }
    for row in entries:
        role = roles[row["split"]]
        # Source trace identity is the physical episode, independent of labeler.
        physical = row["source_sha256"]
        key = physical + ":0"
        if assignments.setdefault(key, role) != role:
            raise ValueError("Physical episode crosses dataset partitions")
        source = dict(
            path=row["path"],
            sha256=row["sha256"],
            collection_id=physical,
            states_key="observation",
            episode_id=0,
            role=role,
            condition_columns={"teacher_id": "owner"},
        )
        read_source(source)
        sources.append(source)
    record = dict(
        version=VERSION,
        sources=sources,
        episode_roles=assignments,
        imported_index=dict(path=str(Path(index_path).resolve()), sha256=file_sha256(index_path)),
    )
    record["identity"] = json_digest(
        {**record, "sources": [{k: v for k, v in r.items() if k != "path"} for r in sources]}
    )
    write_json(output, record)
    return record


def expand_sources(reference, *, paths=None):
    """Resolve declared views/chunks without relabelling their physical identity."""
    ref = paths.view(reference) if paths else dict(reference)
    if file_sha256(ref["path"]) != ref["sha256"]:
        raise ValueError("Dataset/view SHA differs")
    if ref.get("kind") in (VERSION, "collection_chunks_v1"):
        value = read_json(ref["path"])
        children = paths.view(value["sources"]) if paths else value["sources"]
        result = []
        for child in children:
            result.extend(expand_sources(child, paths=paths))
        return result
    return [ref]


def unique_sources(sources):
    """An identical collection/chunk is one source, even if reached via two views."""
    result = {}
    for ref in sources:
        collection = ref.get("collection_id", ref["sha256"])
        key = (collection, ref["sha256"])
        if key in result:
            # A role string cannot change existing physical data provenance.
            if result[key].get("role") != ref.get("role"):
                raise ValueError("Conflicting roles for the same collection")
        else:
            result[key] = dict(ref, collection_id=collection)
    return list(result.values())


def aggregate(path, sources, *, seed, fractions, previous=None):
    """Persist the accumulated collection references and immutable episode split."""
    return create_view(
        path, unique_sources(sources), seed=seed, fractions=fractions, previous=previous
    )


def physical_keys(reference, data):
    if data.get("physical_episode_keys") is not None:
        return [str(k) for k in data["physical_episode_keys"]]
    collection = reference.get("collection_id", reference["sha256"])
    return [f"{collection}:{e}" for e in data["episodes"]]


def sample_keys(reference, data):
    if data.get("sample_ids") is not None:
        return [str(k) for k in data["sample_ids"]]
    collection = reference.get("collection_id", reference["sha256"])
    return [f'{collection}:{reference["sha256"]}:{i}' for i in range(len(data["states"]))]


def rehearsal_sources(reference, *, paths=None):
    """Rehearsal must carry persisted split evidence, not a caller-provided role."""
    ref = paths.view(reference) if paths else reference
    sources = expand_sources(ref, paths=paths)
    if ref.get("kind") == VERSION:
        result = []
        for source in sources:
            result.append({**source, "split_view": dict(path=ref["path"], sha256=ref["sha256"])})
        return result
    for source in sources:
        data = read_source(source, paths=paths)
        if data.get("roles") is None:
            raise ValueError("Rehearsal requires persisted episode split evidence")
    return sources
