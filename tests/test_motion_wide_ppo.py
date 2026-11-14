"""Wide BC Actor admission, physical-scale PPO and complete restore consumers."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from native_stand.artifacts.io import file_sha256, json_digest, write_json, write_npz
from native_stand.artifacts.motion_actor_admission import read_admission, require_allocation
from native_stand.experiments.config import InitializationConfig
from native_stand.learning.ppo.actor_initialization import training_source, validate_actor
from native_stand.learning.ppo.assembly import build_actor
from native_stand.learning.ppo.supported import assemble_from_full
from native_stand.tasks.motion.action_contract import ActionDecoder
from native_stand.tasks.motion.spec import build_task
from test_motion_production_integration import equal_tree, make_job, observations, rollout

FROZEN_TEST_SCOPE = "PORTABLE"


def wide(tmp_path):
    job, contract = make_job(tmp_path, warmup=2, retention=False)
    contract.update(
        action_version="motion_multiheight020_support_center_050_80_v1", leg_halfwidth_rad=0.5
    )
    path = tmp_path / "wide.json"
    write_json(path, contract)
    action = replace(job.action, version=contract["action_version"], leg_scale_rad=0.5)
    model = replace(job.model, action_version=action.version)
    job = replace(
        job,
        action=action,
        model=model,
        initialization=InitializationConfig("actor_only", tmp_path / "actor.pt", "a" * 64),
    )
    actor = build_actor(job)
    payload = dict(
        actor=actor.state_dict(),
        interface_version=job.model.interface,
        action_version=action.version,
        output_contract=contract,
        network_architecture=dict(actor_hidden_dims=[8]),
        inference_spec=dict(kind="tanh", input_dim=31),
        execution_timing={},
    )
    torch.save(payload, job.initialization.checkpoint)
    job = replace(
        job,
        initialization=replace(
            job.initialization, sha256=file_sha256(job.initialization.checkpoint)
        ),
    )
    from test_motion_spin_branches import course_configuration

    course = course_configuration("motion")
    job = replace(
        job, curriculum=replace(job.curriculum, version=course["version"], stages=(course,))
    )
    ref = lambda p: dict(path=str(p), sha256=file_sha256(p))
    write_json(
        tmp_path / "settlement.json",
        dict(historical_online_updates=100, retired_remaining_updates=20),
    )
    write_json(
        tmp_path / "authorization.json",
        dict(new_online_updates=10000, wall_s=None, old_remaining_retired=20),
    )
    write_json(
        tmp_path / "allocation.json",
        dict(
            authorization_sha256=file_sha256(tmp_path / "authorization.json"),
            entries=[
                dict(
                    run_id=job.name,
                    smoke=job.limits.smoke_updates,
                    formal=job.limits.formal_updates,
                ),
                dict(
                    run_id="remainder",
                    smoke=0,
                    formal=10000 - job.limits.smoke_updates - job.limits.formal_updates,
                ),
            ],
        ),
    )
    bank_x = np.zeros((128, 31), dtype=np.float32)
    bank_x[:, 24] = 0.68
    with torch.no_grad():
        bank_y = actor.mean(torch.from_numpy(bank_x)).numpy()
    write_npz(
        tmp_path / "bank.npz",
        dict(actor_obs=bank_x, source_mean=bank_y, cohort=np.zeros(128, dtype=int)),
    )
    write_json(tmp_path / "bank.json", dict(data_role="train_only", commands=[[0, 0]]))
    retention = dict(
        version="motion48_per_command_nominal_source_KL_v4",
        coefficient=0.02,
        source_sha256=job.initialization.sha256,
        commands=[[0, 0]],
        maximum_states_per_cohort=4,
        bank_states_per_command=64,
        bank_fraction_when_onpolicy=0.5,
        task_reduction="mean_over_declared_commands",
        bank=ref(tmp_path / "bank.npz"),
        bank_metadata=ref(tmp_path / "bank.json"),
        physical_output_scales=[0.5, 0.5, 80.0, 0.5, 0.5, 80.0],
        output_contract_sha256=file_sha256(path),
    )
    record = dict(
        version="motion_wide_BC_actor_new_grant_v1",
        branch="motion",
        branch_updates=0,
        source=ref(job.initialization.checkpoint),
        output_contract=ref(path),
        authorization=ref(tmp_path / "authorization.json"),
        settlement=ref(tmp_path / "settlement.json"),
        allocation=ref(tmp_path / "allocation.json"),
        target_course_sha256=json_digest(course),
        source_model_progress_updates=80,
    )
    write_json(tmp_path / "admission.json", record)
    assets = dict(
        job.assets,
        wide_actor_admission=ref(tmp_path / "admission.json"),
        branch_initialization=ref(tmp_path / "admission.json"),
        branch_course=course,
        historical_online_updates=100,
        original_online_limit=10100,
        new_independent_authorization=True,
        source_policy_retention=retention,
        output_contract=str(path),
        output_contract_sha256=file_sha256(path),
        delayed_execution={},
    )
    return replace(job, assets=assets), payload, contract


def test_wide_actor_boundary_rejects_changed_contract_state_or_grant(tmp_path):
    job, payload, contract = wide(tmp_path)
    assert build_task(job) and read_admission(job) and require_allocation(job)
    assert validate_actor(payload, job, contract)
    for key in ("critic", "optimizer", "training_rng"):
        with pytest.raises(ValueError):
            validate_actor({**payload, key: {}}, job, contract)
    wrong = copy.deepcopy(contract)
    wrong["support_offset_table"]["offset_rad"][0][0] += 0.01
    with pytest.raises(ValueError):
        validate_actor(payload, job, wrong)
    with pytest.raises(ValueError):
        read_admission(replace(job, assets={**job.assets, "original_online_limit": 20000}))
    with pytest.raises(ValueError):
        build_task(replace(job, assets={}))


def test_wide_actor_fresh_critic_std_and_bank_optimizer_resume(tmp_path):
    job, payload, contract = wide(tmp_path)
    architecture = dict(
        actor_hidden_dims=[8], critic_hidden_dims=[8], actor_input=31, critic_input=64
    )
    source = training_source(payload, job, contract, architecture)
    assert source["warmup_updates_done"] == 0 and "optimizer" not in source
    for key, value in payload["actor"].items():
        if key != "log_std":
            assert torch.equal(value, source["actor"][key])
    assert torch.allclose(
        source["actor"]["log_std"].exp(), torch.tensor([0.04, 0.04, 0.025, 0.04, 0.04, 0.025])
    )
    obs = observations(job)
    a = assemble_from_full(source, obs, job, fresh=True)
    frozen = copy.deepcopy(a.actor.state_dict())
    teacher = copy.deepcopy(a.source_anchor.teacher.state_dict())
    a.actor_update_enabled = False
    rollout(a, obs, 4)
    equal_tree(frozen, a.actor.state_dict())
    a.actor_update_enabled = True
    rollout(a, obs, 4)
    assert any(not torch.equal(v, a.actor.state_dict()[k]) for k, v in frozen.items())
    equal_tree(teacher, a.source_anchor.teacher.state_dict())
    assert a.physical_leg_scale == 0.5
    full = dict(
        source,
        actor=a.actor.state_dict(),
        critic=a.critic.state_dict(),
        optimizer=a.optimizer.state_dict(),
        warmup_updates_done=2,
        source_anchor_actor=a.source_anchor.teacher.state_dict(),
        source_anchor_runtime_state=a.source_anchor.state_dict(),
    )
    b = assemble_from_full(full, obs, job)
    assert b.actor_update_enabled
    equal_tree(a.optimizer.state_dict(), b.optimizer.state_dict())
    assert a.source_anchor.state_dict() == b.source_anchor.state_dict()
    raw = torch.zeros(1, 6)
    assert torch.equal(
        ActionDecoder.from_contract(contract).decode(raw, torch.tensor([0.34]))["requested_target"][
            :, [0, 1, 3, 4]
        ],
        build_task(job).center.centers(torch.tensor([0.34])),
    )
