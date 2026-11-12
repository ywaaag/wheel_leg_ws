"""Actual n-environment C/V4 bridge preparation, without Isaac or integration."""

import os
from pathlib import Path

import pytest
import torch
from fixtures.jump_cpu_io import mapped_job
from test_jump_cpu_reuse import equal, no_physics_or_cuda  # noqa: F401
from test_jump_unified import feedback

from native_stand.artifacts.io import read_json
from native_stand.simulation.jump_build import build
from native_stand.simulation.jump_response import Response
from native_stand.simulation.jump_unified import UnifiedV4Execution

FROZEN_TEST_SCOPE = "HOST"


@pytest.fixture
def bridge(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = Path(os.environ.get("WHEEL_LEG_DATA_ROOT", "/home/kellen/wheel_leg_training_data"))
    job = mapped_job(root, data)
    binding = build(tmp_path / "ABI4.so")
    contract = read_json(job.assets["jump_bindings"]["V4"]["path"])

    def factory():
        e = UnifiedV4Execution(
            binding["path"], contract, Response, 3, library_sha=binding["sha256"]
        )
        cfg = e.c.config
        cfg.enabled = cfg.model_available = 1
        cfg.model_id = 1
        cfg.model_input_id = e.c.INPUT_CONTRACT
        cfg.model_output_id = e.c.OUTPUT_CONTRACT
        q = torch.zeros(3, 6)
        e.reset(torch.arange(3), q, q)
        return e

    return factory


def advance(e):
    rows = [feedback(e.tick) for _ in range(3)]
    for k, row in enumerate(rows):
        row.upstream_target[:] = e.c.states[k].adopted
        row.adopted_sequence = e.c.states[k].adopted_sequence
    e.update(rows)
    if e.tick % 10 == 0:
        _, valid = e.observe()
        assert (e.publish_raw(torch.zeros(3, 6), valid) >= 0).all()
    return tuple(x.clone() for x in e.step(torch.zeros(3, 6), torch.zeros(3, 6)))


def test_batched_C_bridge_partial_reset_does_not_restart_clock_or_other_environments(bridge):
    e, untouched = bridge(), bridge()
    for _ in range(133):
        equal(advance(e), advance(untouched))
    before = [bytes(s) for s in e.c.states]
    e.reset(torch.tensor([1]), torch.zeros(1, 6), torch.zeros(1, 6))
    assert e.tick == e.v4.tick == e.v4.leg.tick == 133
    assert bytes(e.c.states[0]) == before[0] and bytes(e.c.states[2]) == before[2]
    assert e.v4.birth.tolist() == [0, 133, 0]
    for _ in range(30):
        a, b = advance(e), advance(untouched)
        for left, right in zip(a, b, strict=True):
            assert torch.equal(left[[0, 2]], right[[0, 2]])
        assert bytes(e.c.states[0]) == bytes(untouched.c.states[0])
        assert bytes(e.c.states[2]) == bytes(untouched.c.states[2])
    assert e.C_PD_calls == untouched.C_PD_calls == 33


def test_batched_C_bridge_ring_checkpoint_and_raw_action_continuation(bridge):
    e = bridge()
    for _ in range(139):
        advance(e)
    saved = e.state_dict()
    expected = advance(e)
    final = e.state_dict()
    other = bridge()
    other.load_state_dict(saved)
    equal(expected, advance(other))
    equal(final, other.state_dict())


def test_real_C_safety_row_can_end_without_stopping_other_rows(bridge):
    e = bridge()
    e.row_faults = True
    e.c.states[1].phase = 8
    e.c.states[1].owner = 4
    e.c.states[1].fault = 1
    motor, _ = advance(e)
    assert e.c.states[1].owner == 4
    assert torch.isfinite(motor).all()
    assert e.tick == 1 and e.C_PD_calls == 1


def test_partial_reset_feedback_only_steps_the_selected_real_C_row(bridge):
    from types import SimpleNamespace

    import numpy as np

    from native_stand.learning.ppo.performance import Measurement
    from native_stand.simulation.jump_isaac import IsaacJump

    execution = bridge()
    for _ in range(100):
        advance(execution)
    execution.update([feedback(execution.tick) for _ in range(3)])
    env = IsaacJump.__new__(IsaacJump)
    env.execution = execution
    env.num_envs = 3
    env.prepared_tick = execution.tick
    env.starts = np.zeros(3)
    env.requests = [dict(kind="landing_only") for _ in range(3)]
    env.motor = env.NET = torch.zeros(3, 6)
    env.measurement = Measurement(False)
    env.inverse = SimpleNamespace(input=lambda **kw: feedback(kw["tick"]))
    before = [bytes(s) for s in execution.c.states]
    inputs = [bytes(i) for i in execution.input_rows]
    env.prepare(
        dict(
            root=np.zeros((3, 13)),
            joint=np.zeros((3, 3, 38)),
            acceleration=np.zeros((3, 6)),
            motor=np.zeros((3, 6)),
            NET=np.zeros((3, 6)),
        ),
        reset_ids=[0],
    )
    assert bytes(execution.c.states[1]) == before[1] and bytes(execution.c.states[2]) == before[2]
    assert bytes(execution.input_rows[1]) == inputs[1]


def test_Kit_shutdown_is_deferred_until_after_worker_finalization():
    from types import SimpleNamespace

    from native_stand.simulation.jump_isaac import IsaacJump

    calls = []
    env = IsaacJump.__new__(IsaacJump)
    env._app_exit_code = 0
    env.scene = SimpleNamespace(close=lambda: calls.append("scene"))
    env.app = SimpleNamespace(close=lambda **kwargs: calls.append(kwargs))
    env.close()
    assert calls == ["scene"]
    env._exit_app()
    assert calls == ["scene", dict(exit_code=0)]
