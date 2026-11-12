"""Real rollout math/control boundaries; physics I/O only, no engine or GPU."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
import torch
from test_motion_control_contract import contract, router_env
from torch.utils._python_dispatch import TorchDispatchMode

from native_stand.simulation.isaac.maneuver_planned import environment_class
from native_stand.simulation.isaac.sensor_packets import Packets
from native_stand.students.standing_survival_task import bootstrap_reward
from native_stand.students.step_checks import StepChecks
from native_stand.tasks.motion.action_contract import ActionDecoder
from native_stand.tasks.motion.encoding import decode_targets, previous_from_adopted

FROZEN_TEST_SCOPE = "PORTABLE"


class ScalarReads(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            self.count += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("teacher", [False, True])
def test_router_batches_checks_before_execution_with_exact_targets_and_RNG(teacher):
    router, env = router_env()
    env.r4_prefix_remaining.fill_(int(teacher))
    raw = torch.tensor([[0.2, -2.0, 1.2, 0.7, -0.1, -0.6]])
    height = env.height_adapter.target
    # TensorDict's first import uses .item(); count the established control path.
    router.ordinary(env)
    rng = torch.get_rng_state().clone()
    with ScalarReads() as strict:
        expected = router.student_decoder.decode(raw, height)
        if teacher:
            mean = env.r4_prefix_teacher.mean(env.student_obs["student_actor"])
            expected["requested_target"][:] = router.teacher_decoder.decode(mean, height)[
                "requested_target"
            ]
    with ScalarReads() as combined:
        bridge, valid = router.actions(env, raw)
    assert strict.count == 5 * (1 + teacher)
    assert combined.count == 1
    units = raw.new_tensor([1.5, 1.5, 100, 1.5, 1.5, 100])
    assert torch.equal(bridge, expected["requested_target"] / units)
    assert torch.equal(env.supported_requested_physical_target, expected["requested_target"])
    assert torch.equal(env.supported_raw_action, raw)
    assert torch.equal(env.supported_raw_clip, expected["raw_clipped"])
    assert valid.item() is not teacher
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(
        previous_from_adopted(env.supported_requested_physical_target),
        previous_from_adopted(expected["requested_target"]),
    )


@pytest.mark.parametrize("bad", ["raw_nan", "raw_inf", "height_nan", "height_low", "teacher_nan"])
def test_router_invalid_input_stops_before_adoption_or_clock_mutation(bad):
    router, env = router_env()
    raw = torch.zeros(1, 6)
    if bad == "raw_nan":
        raw[0, 0] = torch.nan
    elif bad == "raw_inf":
        raw[0, 0] = torch.inf
    elif bad == "height_nan":
        env.height_adapter.target[0] = torch.nan
    elif bad == "height_low":
        env.height_adapter.target[0] = 0.2
    else:

        class InvalidTeacher(torch.nn.Module):
            def mean(self, obs):
                return obs.new_full((len(obs), 6), torch.nan)

        env.r4_prefix_teacher = InvalidTeacher()
    before = {
        key: getattr(env, key).clone()
        for key in ("r4_owner", "r4_prefix_remaining", "r4_prefix_count", "r4_handoff_count")
    }
    with pytest.raises(ValueError):
        router.actions(env, raw)
    assert not hasattr(env, "supported_requested_physical_target")
    for key, value in before.items():
        assert torch.equal(getattr(env, key), value)


def test_public_decoder_stays_strict_and_batched_mechanical_checks_still_reject():
    decoder = ActionDecoder.from_contract(contract())
    with pytest.raises(ValueError, match="超出"):
        decoder.decode(torch.zeros(1, 6), torch.tensor([0.2]))
    raw, center = torch.zeros(2, 6), torch.zeros(2, 4)
    limits = torch.tensor([-0.5, 0.5]).expand(2, 4, 2).clone()
    limits[1, 1] = torch.tensor([0.5, -0.5])
    checks = StepChecks()
    decode_targets(raw, center, leg_limits_rad=limits, checks=checks)
    with pytest.raises(ValueError, match="机械角度"):
        checks.verify()
    with pytest.raises(ValueError, match="机械角度"):
        decode_targets(raw, center, leg_limits_rad=limits)


class CountedPackets(Packets):
    warm_fills = 0

    def _initialize_history(self, sample, tick):
        self.warm_fills += 1
        return super()._initialize_history(sample, tick)


class EagerWarmFillPackets(CountedPackets):
    """Original eager warm-fill cadence; all sampling/dq math stays production."""

    def read(self, current, tick):
        if tick != self.last_tick:
            self.has_fresh = True
        return super().read(current, tick)


def make_packets(cls, *, n=4):
    def noise(episodes, ids, width, stream):
        return torch.sin(episodes[:, None] + ids[:, None] + torch.arange(width) + stream)

    def feedback(value, tick):
        result = value.clone()
        result[:, [10, 11, 13, 14]] += tick * 0.0001
        return result

    dr = NS(
        episodes=torch.zeros(n, dtype=torch.long),
        bias=torch.linspace(-0.02, 0.02, n * 16).reshape(n, 16),
        scales=torch.full((n, 5), 0.01),
        packet_age=torch.tensor([0, 10, 20, 30]),
        leg_packet_age=torch.tensor([30, 20, 10, 0]),
        dq=NS(read=feedback),
    )
    timing = NS(n=n, device="cpu", ids=torch.arange(n), dr=dr, tick=0)
    timing.measurement = lambda: (
        torch.arange(n * 6).reshape(n, 6).float() * 0.01 + timing.tick,
        torch.arange(n * 6).reshape(n, 6).float() * -0.01 - timing.tick,
    )
    return cls(timing, noise=noise)


def test_packet_warmfill_only_after_reset_preserves_every_ring_and_packet():
    fast, eager = make_packets(CountedPackets), make_packets(EagerWarmFillPackets)
    rng = torch.get_rng_state().clone()
    for tick in range(0, 1000, 10):
        for packets in (fast, eager):
            packets.t.tick = tick
        if tick in (140, 530):
            for packets in (fast, eager):
                packets.reset(torch.tensor([0, 2]))
                packets.t.dr.episodes[[0, 2]] += 1
        frame = torch.arange(64).reshape(4, 16).float() / 100 + tick
        for repeat in range(2):
            observed = fast.read(frame, tick).clone()
            expected = eager.read(frame, tick).clone()
            assert torch.equal(observed, expected)
            for key in ("values", "true_frame", "result", "ready", "birth", "age", "leg_age"):
                assert torch.equal(getattr(fast, key), getattr(eager, key)), (tick, repeat, key)
            assert torch.equal(fast.pending(), eager.pending())
        if tick == 250:
            # Same physical tick reset is also a new history boundary.
            for packets in (fast, eager):
                packets.reset(torch.tensor([1]))
                packets.read(frame, tick)
            assert torch.equal(fast.values, eager.values)
    assert fast.warm_fills == 4
    assert eager.warm_fills == 101
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("kind", ["none", "natural", "timeout", "mixed"])
def test_actual877_terminal_snapshot_only_for_endings_and_precedes_partial_reset(kind):
    class PhysicsIO:
        planned_value_ready = True
        target_xy = torch.tensor([[3.6, 0.5], [-3.6, -0.5], [0.0, 0.0]])

        def __init__(self):
            self.motion_timing = NS(dr=NS(contact=None))
            self.value = torch.arange(3 * 875).float().reshape(3, 875)
            self.terminal = torch.zeros(3, dtype=torch.bool)
            self.timeout = self.terminal.clone()
            self.observation_calls = 0
            self.student_terminal_critic = torch.full((3, 877), -1.0)
            self.reward_memory = 0
            self.collection_snapshots = []

        def collection_snapshot(self):
            self.collection_snapshots.append(
                (
                    self.value.clone(),
                    None
                    if self.student_terminal_critic is None
                    else self.student_terminal_critic.clone(),
                )
            )

        def _get_observations(self):
            self.observation_calls += 1
            # Real outer planned-value encoder adds its named two columns.
            return {"student_critic": self.value + self.reward_memory}

        def _get_dones(self):
            self.reward_memory += 1
            return self.terminal, self.timeout

    env = environment_class(PhysicsIO)()
    if kind in ("natural", "mixed"):
        env.terminal[0] = True
    if kind in ("timeout", "mixed"):
        env.timeout[1] = True
    expected_prefix = env.value + 1
    terminal, timeout = env._get_dones()
    assert env.observation_calls == int(kind != "none")
    assert len(env.collection_snapshots) == 1
    captured_value, captured_terminal = env.collection_snapshots[0]
    assert torch.equal(captured_value, env.value)
    if kind == "none":
        assert env.student_terminal_critic is None
        assert captured_terminal is None
    else:
        snapshot = env.student_terminal_critic.clone()
        assert snapshot.shape == (3, 877)
        assert torch.equal(snapshot[:, :875], expected_prefix)
        assert torch.equal(snapshot[:, 875:], env.target_xy / 4)
        assert torch.equal(captured_terminal, snapshot)
        adjusted = bootstrap_reward(torch.ones(3), terminal, timeout, snapshot[:, 0], gamma=0.99)
        assert adjusted[0] == 1
        assert adjusted[1] == (1 + 0.99 * snapshot[1, 0] if timeout[1] else 1)
        env.value[terminal | timeout] = -999
        ordinary = env._get_observations()["student_critic"]
        assert torch.equal(env.student_terminal_critic, snapshot)
        assert not torch.equal(ordinary[terminal | timeout], snapshot[terminal | timeout])
        assert torch.equal(captured_terminal, snapshot)
        assert torch.equal(captured_value, expected_prefix - 1)


def test_packet_state_is_not_changed_by_skipping_an_unused_terminal_observation():
    fast, eager = make_packets(CountedPackets), make_packets(CountedPackets)
    for tick in range(0, 100, 10):
        current = torch.full((4, 16), tick / 100.0)
        for packets in (fast, eager):
            packets.t.tick = tick
        eager.read(current, tick)  # Previously unconditional _get_dones observation.
        assert torch.equal(fast.read(current, tick), eager.read(current, tick))
        for key in ("values", "birth", "ready", "result", "age", "leg_age"):
            assert torch.equal(getattr(fast, key), getattr(eager, key))
