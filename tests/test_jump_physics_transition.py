"""A new physical cohort requires explicit full-state provenance, never same-run bypass."""

import copy
from types import SimpleNamespace

import pytest

from native_stand.simulation.jump_isaac import IsaacJump
from native_stand.simulation.jump_isaac_scene import PHYSICAL_CONTRACT
from native_stand.tasks.jump.adaptive_course import AdaptiveCourse
from native_stand.tasks.jump.isaac_contract import TRAINING

FROZEN_TEST_SCOPE = "PORTABLE"


def transition_env(tmp_path):
    prior = [AdaptiveCourse(seed=i, plant_id="old", DR_id="nominal") for i in range(2)]
    for course in prior:
        request = course.next_episode()
        course.record(request, jump_control_s=0.2, success=False, fault=True)
        course.next_episode()
    state = dict(
        version=TRAINING,
        identity="old-identity",
        restore="fresh_reset",
        courses=[c.state_dict() for c in prior],
    )
    transition = dict(
        source_checkpoint_sha256="a" * 64,
        source_execution_identity=state["identity"],
        source_physical_contract=None,
        target_physical_contract=PHYSICAL_CONTRACT,
        course_evidence="fresh_physics_cohort_preserve_rng_and_counters",
    )
    env = IsaacJump.__new__(IsaacJump)
    env.num_envs = 2
    env.output = tmp_path
    env.identity = "new-identity"
    env.physical_contract = copy.deepcopy(PHYSICAL_CONTRACT)
    env.job = SimpleNamespace(
        initialization=SimpleNamespace(mode="full", sha256="a" * 64),
        assets={"jump_physics_transition": transition},
    )
    env.courses = [AdaptiveCourse(seed=i + 99, plant_id="new", DR_id="nominal") for i in range(2)]
    env.reset_calls = 0

    def reset():
        env.reset_calls += 1

    env.reset = reset
    return env, state


def test_full_learning_transfer_explicitly_starts_new_physical_evidence(tmp_path):
    env, state = transition_env(tmp_path)
    original = copy.deepcopy(state)
    env.initialize_from_training_state(state)
    assert state == original
    assert env.reset_calls == 1 and env.completed == [] and env.serial == 0
    for course, old in zip(env.courses, state["courses"], strict=True):
        assert course.rng.bit_generator.state == old["rng"]
        assert (course.serial, course.episodes) == (old["serial"], old["episodes"])
        assert course.generation == old["generation"] + 1
        assert not course.pending and not any(course.buffers.values())
        assert not any(course.windows.values()) and sum(course.control_s.values()) == 0
        assert course.plant_id == "new"
    assert (tmp_path / "physics_transition.json").is_file()
    with pytest.raises(ValueError, match="restore identity"):
        env.load_runtime_state(state)


@pytest.mark.parametrize("bad", ["sha", "target", "source", "mode", "course", "envs"])
def test_transition_does_not_loosen_same_run_or_implicit_initialization(tmp_path, bad):
    env, state = transition_env(tmp_path)
    transition = env.job.assets["jump_physics_transition"]
    if bad == "sha":
        transition["source_checkpoint_sha256"] = "b" * 64
    elif bad == "target":
        transition["target_physical_contract"] = {}
    elif bad == "source":
        transition["source_execution_identity"] = "other"
    elif bad == "mode":
        env.job.initialization.mode = "actor_only"
    elif bad == "envs":
        env.num_envs = 1
    else:
        transition.pop("course_evidence")
    with pytest.raises(ValueError, match="full-state Jump physics transition"):
        env.initialize_from_training_state(state)
    assert env.reset_calls == 0
    assert not (tmp_path / "physics_transition.json").exists()


@pytest.mark.parametrize("declared", [False, True])
def test_expanding_envs_requires_explicit_mapping_and_preserves_existing_course_rng(
    tmp_path, declared
):
    env, state = transition_env(tmp_path)
    env.num_envs = 4
    env.job.runtime = SimpleNamespace(seed=19)
    env.courses = [AdaptiveCourse(seed=19 + i, plant_id="new", DR_id="nominal") for i in range(4)]
    initial_new_rng = [copy.deepcopy(c.rng.bit_generator.state) for c in env.courses[2:]]
    if not declared:
        with pytest.raises(ValueError, match="full-state Jump physics transition"):
            env.initialize_from_training_state(state)
        assert env.reset_calls == 0
        return
    env.job.assets["jump_physics_transition"]["environment_expansion"] = dict(
        source_envs=2,
        target_envs=4,
        existing_rows="preserve_course_rng_and_counters",
        added_rows="job_seed_plus_row_index",
        seed=19,
    )
    env.initialize_from_training_state(state)
    for i, c in enumerate(env.courses):
        assert c.level == 0 and not c.pending and not any(c.windows.values())
        if i < 2:
            assert c.rng.bit_generator.state == state["courses"][i]["rng"]
            assert c.episodes == state["courses"][i]["episodes"]
        else:
            assert c.rng.bit_generator.state == initial_new_rng[i - 2]
            assert c.episodes == c.serial == 0
    with pytest.raises(ValueError, match="restore identity"):
        env.load_runtime_state(state)
