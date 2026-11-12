"""Portable numerical equivalence of the real shared PPO consumers; no physics."""

import copy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from native_stand.learning.ppo.controlled import KEY, normalize_valid
from native_stand.learning.ppo.jump_unified import DirectActor, DirectPPO
from native_stand.learning.ppo.networks import StudentNetwork
from native_stand.learning.ppo.physical_anchor import PhysicalAnchor
from native_stand.learning.ppo.recovery_action_prior import ActionPrior
from native_stand.learning.ppo.recovery_cubic import IndependentAdam, PhysicalActor, assemble
from native_stand.simulation.engine import GenericPPOAdapter

FROZEN_TEST_SCOPE = "PORTABLE"
CONTRACT = dict(center_rad=[0.0] * 4, leg_lower_rad=[-1.2] * 4, leg_upper_rad=[1.2] * 4)


def observation(kind, mask):
    n = len(mask)
    x = torch.arange(n * (37 if kind == "jump" else 31)).reshape(n, -1).float() * 0.001
    x[:, 24] = 0.4
    x[:, 25:31] = x.new_tensor([0, 0, 1, 0, 0, 1] if kind == "jump" else [0, 1, 0, 1, 0, 0])
    data = dict(
        student_actor=x,
        student_critic=torch.zeros(n, 909),
        recovery_actor_valid=torch.tensor(mask)[:, None],
    )
    data["student_critic"][:, 863] = 0.4
    if kind == "jump":
        x[:, 31:] = 0
        data.update(jump_C_phase=torch.full((n, 1), 9), jump_C_owner=torch.full((n, 1), 2))
    return TensorDict(data, batch_size=[n])


def policy(kind, obs, *, auxiliary=False):
    from rsl_rl.storage import RolloutStorage

    actor = DirectActor((8,)) if kind == "jump" else PhysicalActor(CONTRACT, hidden_dims=(8,))
    critic = StudentNetwork(actor=False, input_dim=909, hidden_dims=(8,), output_dim=1)
    storage = RolloutStorage("rl", len(obs), 4, obs, [6], device="cpu")
    if kind == "recovery":
        retention = prior = None
        if auxiliary:
            x = observation(kind, [True] * 40)["student_actor"]
            retention = PhysicalAnchor(x, torch.zeros(40, 6), 0.1, batch_size=16)
            teacher = StudentNetwork(
                actor=True, input_dim=31, hidden_dims=(7,), output_dim=6, init_std=torch.ones(6)
            )
            with torch.no_grad():
                teacher.mlp[-1].weight.zero_()
                teacher.mlp[-1].bias.zero_()
            prior = ActionPrior(teacher, coefficient=0.01, max_rows=4)
        p = assemble(
            actor, critic, storage, epochs=2, minibatches=2, retention=retention, action_prior=prior
        )
    else:
        p = DirectPPO(
            actor,
            critic,
            storage,
            schedule="fixed",
            num_learning_epochs=2,
            num_mini_batches=2,
            minimum_unique_samples=2,
            minimum_minibatch_samples=2,
            device="cpu",
        )
        p.optimizer = IndependentAdam(actor, critic, 1e-5, 9e-5)
    return p


def reference_update(p):
    """Pre-optimization mathematics: graph on skipped Actor, original scalar reads.

    Uses the actual storage generator, Actor, Critic, auxiliary losses and Adams.
    This is an independent test oracle, not an alternate production PPO loop.
    """
    p.unique_samples = int(p.storage.observations[KEY].sum())
    p.minibatches = []
    for b in p.storage.mini_batch_generator(p.num_mini_batches, p.num_learning_epochs):
        m = b.observations[KEY].squeeze(-1)
        allowed = p.actor_update_enabled and p.actor_batch_allowed(b, m)
        p.actor(b.observations, stochastic_output=True)
        logp = p.actor.get_output_log_prob(b.actions)
        entropy = p.actor.output_entropy
        values = p.critic(b.observations)
        vloss = (
            (b.returns - values).square()
            if not p.use_clipped_value_loss
            else torch.maximum(
                (b.returns - values).square(),
                (
                    b.returns - (b.values + (values - b.values).clamp(-p.clip_param, p.clip_param))
                ).square(),
            )
        ).mean()
        loss = p.value_loss_coef * vloss
        if m.any() and allowed:
            ratio = (logp[m] - b.old_actions_log_prob.squeeze(-1)[m]).exp()
            adv = b.advantages.squeeze(-1)[m]
            surrogate = torch.maximum(
                -adv * ratio, -adv * ratio.clamp(1 - p.clip_param, 1 + p.clip_param)
            ).mean()
            loss = loss + surrogate - p.entropy_coef * entropy[m].mean()
            extra = p.actor_auxiliary_loss(b, m)
            if extra is not None:
                loss = loss + extra
        p.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not m.any() or not allowed:
            for parameter in p.actor.parameters():
                parameter.grad = None
        torch.nn.utils.clip_grad_norm_(p.actor.parameters(), p.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(p.critic.parameters(), p.max_grad_norm)
        p.optimizer.step()
    p.storage.clear()


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, tuple | list):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    else:
        assert a == b


def fill(p, obs, *, inference):
    with torch.inference_mode() if inference else torch.enable_grad():
        for tick in range(4):
            p.act(obs)
            rewards = torch.arange(len(obs)).float()[:, None] * 0.01 + tick
            done = torch.zeros(len(obs), dtype=torch.bool)
            done[tick % len(obs)] = tick in (1, 3)
            p.process_env_step(obs, rewards, done, {})
    # Targets are ordinary constants in both paths, storage was allocated outside.
    with torch.no_grad():
        if inference:
            p.compute_returns(obs)
        else:
            PPO.compute_returns(p, obs)
            p.storage.advantages = normalize_valid(
                p.storage.returns - p.storage.values, p.storage.observations[KEY]
            )


@pytest.mark.parametrize("kind", ["recovery", "jump"])
@pytest.mark.parametrize(
    "mask", [[True] * 4, [True, False, True, False], [False] * 4, [True, False, False, False]]
)
@pytest.mark.parametrize("warmup", [False, True])
def test_rollout_update_exact_equivalence_mask_rng_two_adam(kind, mask, warmup):
    torch.manual_seed(901)
    obs = observation(kind, mask)
    new = policy(kind, obs, auxiliary=kind == "recovery")
    old = copy.deepcopy(new)
    new.actor_update_enabled = old.actor_update_enabled = not warmup
    rng = torch.get_rng_state()
    fill(old, obs, inference=False)
    old_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    fill(new, obs, inference=True)
    equal(old_rng, torch.get_rng_state())
    for field in (
        "actions",
        "actions_log_prob",
        "distribution_params",
        "values",
        "rewards",
        "returns",
        "advantages",
    ):
        equal(getattr(old.storage, field), getattr(new.storage, field))
    assert new.storage.returns.grad_fn is None and not new.storage.returns.is_inference()
    assert new.storage.advantages.grad_fn is None and not new.storage.advantages.is_inference()
    rng = torch.get_rng_state()
    reference_update(old)
    old_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    new.update()
    equal(old_rng, torch.get_rng_state())
    equal(old.actor.state_dict(), new.actor.state_dict())
    equal(old.critic.state_dict(), new.critic.state_dict())
    equal(old.optimizer.state_dict(), new.optimizer.state_dict())
    equal(old.actor.output_distribution_params, new.actor.output_distribution_params)
    if warmup or not any(mask):
        assert new.mask_summary["actor_optimizer_steps"] == 0


def test_rollout_actual_generic_adapter_has_no_graph_but_learning_has_grad():
    obs = observation("recovery", [True] * 4)
    p = policy("recovery", obs)
    saved = []

    def before(module, args):
        saved.append(torch.is_grad_enabled())

    handle = p.actor.register_forward_pre_hook(before)

    class IO:
        num_envs = 4
        infinite_horizon = False

        def observations(self):
            return obs

        def step(self, actions):
            assert not actions.requires_grad
            return dict(rewards=torch.ones(4), dones=torch.zeros(4, dtype=torch.bool))

    adapter = GenericPPOAdapter(None)
    adapter.env = IO()
    adapter.ppo = p
    adapter.job = SimpleNamespace(
        diagnostics={},
        runtime=SimpleNamespace(device="cpu"),
        ppo=SimpleNamespace(learning_rate=1e-5, critic_learning_rate=9e-5),
    )
    adapter._steps_per_env = 4
    adapter._warmup = 0
    adapter.update_once(SimpleNamespace(updates_used=0), 1)
    handle.remove()
    assert saved[:4] == [False] * 4 and all(saved[4:])


def test_explicit_timing_does_not_change_rng_weights_or_optimizer():
    obs = observation("jump", [True] * 4)
    torch.manual_seed(44)
    a = policy("jump", obs)
    b = copy.deepcopy(a)
    rng = torch.get_rng_state()
    fill(a, obs, inference=True)
    a.update()
    final = torch.get_rng_state()
    torch.set_rng_state(rng)
    b.performance_measurement = True
    fill(b, obs, inference=True)
    b.update()
    equal(final, torch.get_rng_state())
    equal(a.optimizer.state_dict(), b.optimizer.state_dict())
    equal(a.actor.state_dict(), b.actor.state_dict())
    assert {
        "minibatch_data",
        "actor_forward",
        "critic_forward",
        "backward",
        "auxiliary",
        "adam",
    } <= b.performance_timings.keys()


def test_recovery_pack_merged_finite_check_keeps_fault_boundary():
    from native_stand.simulation.recovery_isaac.observations import pack
    from native_stand.tasks.recovery.cubic_task import BASE_FIELDS, EXTRA_FIELDS

    values = {k: torch.zeros(2, w) for k, w in BASE_FIELDS}
    extra = {k: torch.zeros(2, w) for k, w in EXTRA_FIELDS}
    assert pack(values, extra).shape == (2, 878)
    key = BASE_FIELDS[3][0]
    values[key][1, 0] = float("nan")
    with pytest.raises(ValueError, match=key):
        pack(values, extra)


def test_profile_cache_reset_epoch_invalidates_without_changing_other_rows():
    from native_stand.simulation.recovery_isaac.observations import physical_profile

    rows = [
        dict(
            root_mass_scale=1.0,
            root_inertia_scale=1.0,
            root_COM_m=[0.0, 0.0, 0.0],
            friction=0.5,
            encoder_bias_rad=[0.0] * 4,
            Actor_age_ms=20.0,
            target_delay_ms=5.0,
            wheel_delay_ms=[7.0, 11.0],
        )
        for _ in range(2)
    ]
    env = SimpleNamespace(
        draws=rows,
        physics=SimpleNamespace(
            mass=torch.ones(2, 1),
            inertia=torch.eye(3).reshape(1, 1, 9).repeat(2, 1, 1),
            com=torch.zeros(2, 1, 3),
        ),
        device="cpu",
        effort=torch.zeros(2, 6),
        num_envs=2,
    )
    first = physical_profile(env)
    assert first is physical_profile(env) and first.shape == (2, 32)
    rows[1]["friction"] = 0.7
    env._physical_profile_cache = None  # actual reset and load boundaries explicitly invalidate
    second = physical_profile(env)
    assert torch.equal(first[0], second[0]) and second[1, 7] == 0.7
    assert not torch.equal(first[1], second[1])


def test_jump_batch_uses_step_next_observation_without_repacking_and_partial_reset():
    from native_stand.artifacts.io import json_digest
    from native_stand.simulation.jump_batch import JumpBatch

    class Scripted:
        def __init__(self, number):
            self.number = number
            self.calls = 0
            self.steps = 0
            self.request = dict(request_id=str(number), kind="ground_active_jump")
            self.course = SimpleNamespace(pending={str(number): True})
            self.base = SimpleNamespace(done=True)
            self.schema = SimpleNamespace(contract=lambda: {"same": "schema"})
            self.reward = SimpleNamespace(
                timeout_paid=False, failure_paid=False, jump_control_s=0.0
            )
            self.balance_seconds = self.balance_good_seconds = 0.0

        def observations(self):
            self.calls += 1
            return TensorDict(
                dict(
                    student_actor=torch.full((1, 37), float(self.steps + self.number)),
                    student_critic=torch.zeros(1, 6370),
                ),
                batch_size=[1],
            )

        def reset_next(self):
            self.base.done = False
            self.steps = 0
            return self.observations()

        def step(self, raw):
            self.steps += 1
            return dict(
                observations=self.observations(),
                rewards=torch.zeros(1),
                dones=torch.tensor([False]),
                terminal_critic=torch.zeros(1, 6370),
                extras=dict(physical_steps=2 if self.number else 10, failure=None),
            )

    envs = [Scripted(0), Scripted(1)]
    job = SimpleNamespace(
        assets=dict(jump_contract_data={}, jump_critic_schema={"same": "schema"}, jump_bindings={})
    )
    batch = JumpBatch(envs, job)
    before = [e.calls for e in envs]
    out = batch.step(torch.zeros(2, 6))
    assert [e.calls for e in envs] == [x + 1 for x in before]
    assert batch.observations() is out["observations"]
    assert out["extras"]["actual_physics_steps"] == [10, 2]
    assert out["extras"]["transition_valid"].tolist() == [True, False]
    other = envs[1].steps
    batch.reset([0])
    assert envs[1].steps == other
    assert batch.identity == json_digest(
        dict(contract={}, schema={"same": "schema"}, bindings={}, envs=2)
    )


def test_schema_reuse_rejects_shape_unit_or_scale_drift():
    import numpy as np

    from native_stand.tasks.jump.controller_critic import Field, NormalizedCritic

    fields = [Field("causal", (2,), np.asarray(2.0), "seconds")]
    old = NormalizedCritic(fields)
    assert NormalizedCritic.reuse(fields, old) is old
    for new in (
        Field("causal", (3,), np.asarray(2.0), "seconds"),
        Field("causal", (2,), np.asarray(3.0), "seconds"),
        Field("causal", (2,), np.asarray(2.0), "meters"),
    ):
        with pytest.raises(ValueError, match="schema drift"):
            NormalizedCritic.reuse([new], old)


def test_zero_prior_does_not_query_teacher_or_consume_rng():
    obs = observation("recovery", [True] * 4)
    actor = PhysicalActor(CONTRACT, hidden_dims=(8,))
    teacher = StudentNetwork(
        actor=True, input_dim=31, hidden_dims=(7,), output_dim=6, init_std=torch.ones(6)
    )
    prior = ActionPrior(teacher, coefficient=0)
    rng = torch.get_rng_state()
    assert prior(actor, SimpleNamespace(observations=obs), torch.ones(4, dtype=torch.bool)) is None
    assert prior.total_queried_rows == 0 and torch.equal(rng, torch.get_rng_state())


def test_measurement_default_never_synchronizes_cuda(monkeypatch):
    from native_stand.learning.ppo.performance import Measurement

    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *a: pytest.fail("default timing synchronized CUDA")
    )
    timer = Measurement(False, "cuda:0")
    with timer.section("disabled"):
        pass
    assert list(timer.batches(iter([1, 2]))) == [1, 2] and not timer.seconds


def test_finite_and_infinite_terminal_bootstrap_pre_reset_once():
    obs = observation("recovery", [True] * 4)
    p = policy("recovery", obs)
    adapter = GenericPPOAdapter(None)
    adapter.ppo = p
    adapter.env = SimpleNamespace(num_envs=4, infinite_horizon=False)
    outcome = dict(
        rewards=torch.ones(4),
        terminated=torch.tensor([True, False, False, False]),
        truncated=torch.tensor([False, True, False, True]),
        terminal_observations={"student_critic": torch.ones(4, 909)},
    )
    assert torch.equal(adapter._stored_reward(outcome), torch.ones(4, 1))
    adapter.env.infinite_horizon = True
    with torch.no_grad():
        actual = adapter._stored_reward(outcome)
        values = p.critic(outcome["terminal_observations"])
    expected = torch.ones(4, 1)
    expected[[1, 3]] += p.gamma * values[[1, 3]]
    assert torch.equal(actual, expected)
    assert torch.equal(outcome["rewards"], torch.ones(4))


def test_driver_reports_saved_update_and_separate_stage_mean(tmp_path):
    from test_phase3_fixes import _cpu_job, _production_context, _sentinel_full_state

    from native_stand.simulation.driver import run_stage

    job = _cpu_job(
        tmp_path,
        limits={"smoke_updates": 2, "formal_updates": 0},
        diagnostics={"performance_measurement": True},
    )
    context = _production_context(tmp_path, job)
    records = []
    original = context.record

    def record(value):
        records.append(copy.deepcopy(value))
        original(value)

    context.record = record

    class Adapter:
        def build(self, *args):
            pass

        def update_once(self, context, index):
            return {
                "reward_mean": 0.0,
                "actor_steps": 1,
                "seconds_per_update": 0.25,
                "performance_seconds": {"rollout_seconds": 0.2, "learning_seconds": 0.05},
            }

        def after_update(self, *args):
            pass

        def capture(self, context, counters):
            return _sentinel_full_state(job, sentinel=0.1)

        def after_save(self, *args):
            pass

        def close(self):
            pass

    run_stage(context, Adapter(), stage="smoke")
    assert len(records) == 2 and records[0]["seconds_per_update"] == 0.25
    for row in records:
        assert row["saved_this_update"] is True
        assert {"checkpoint_capture", "checkpoint_save_SHA", "candidate_references_views"} <= row[
            "performance_seconds"
        ].keys()
        assert row["whole_update_wall_seconds_before_metrics"] >= 0
    assert records[0]["previous_metrics_write_seconds"] is None
    assert records[1]["previous_metrics_write_seconds"] >= 0
    import json

    timings = [
        json.loads(line)
        for line in (context.run_dir / "performance.jsonl").read_text().splitlines()
    ]
    assert len(timings) == 2 and timings[1]["metrics_write_seconds"] >= 0
    assert all(row["whole_update_wall_seconds"] >= row["metrics_write_seconds"] for row in timings)
    assert all(row["phase_seconds"]["rollout_seconds"] == 0.2 for row in timings)
    assert all(row["phase_seconds"]["learning_seconds"] == 0.05 for row in timings)
