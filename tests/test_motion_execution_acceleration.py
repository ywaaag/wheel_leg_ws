"""Frozen V4 versus accelerated implementation; no engine, GPU or real PPO."""

import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from native_stand.simulation.isaac.critic_cache import CriticCache
from native_stand.simulation.isaac.response_acceleration import accelerate, response_tick

FROZEN_TEST_SCOPE = "HOST"
ROOT = Path(__file__).resolve().parents[1]


def reference():
    from native_stand.artifacts.io import file_sha256
    from native_stand.artifacts.runtime_paths import RuntimePaths
    from native_stand.experiments.config import load_job_config

    job = load_job_config(
        ROOT
        / "configs/jobs/2026_11_12__motion_u3000_bank_height020_H1_cloud10000_contact_grid_v2.json"
    )
    assets = dict(job.assets)
    mapping = os.environ.get("WHEEL_LEG_TEST_RUNTIME_PATHS")
    if mapping:
        assets["path_mapping"] = dict(path=mapping, sha256=file_sha256(mapping))
    paths = RuntimePaths.from_assets(assets)
    path = Path(paths.resolve(str(Path(assets["frozen_bundle"]) / "response_tensor.py")))
    if not path.is_file():
        pytest.skip("explicit frozen V4 numeric dependency unavailable")
    spec = importlib.util.spec_from_file_location("frozen_speed_response_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Response


def response(n=8):
    rng = torch.Generator().manual_seed(391)
    x = torch.randn(n, 13, generator=rng) * 0.2
    x[:, 3:5] = torch.rand(n, 2, generator=rng) * 4
    profiles = [
        dict(
            transport_ms=i % 21,
            q_age_ms=i % 5,
            v_age_ms=i % 8,
            q_reference=[0.1, -0.15],
            velocity_tau_ms=i % 4,
        )
        for i in range(n)
    ]
    obj = reference()(x, profiles)
    obj.qbias[:] = 0.01
    obj.vgain[:] = 1.03
    obj.vbias[:] = -0.02
    obj.qnoise[:] = 0.003
    obj.vnoise[:] = 0.02
    q = torch.randn(n, 2, generator=rng) * 0.15
    v = torch.randn(n, 2, generator=rng) * 0.3
    obj.reset(q, v, q + 0.02)
    return obj


def equal_state(a, b):
    assert a.tick == b.tick
    for key, value in a.__dict__.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, getattr(b, key), atol=0, rtol=0)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("noise", [False, True])
def test_actual_frozen_response_all_ticks_partial_reset_and_history_wrap(quantized, noise):
    old = response()
    if noise:
        old.sample_noise = lambda t: torch.sin(torch.arange(old.n * 4).reshape(old.n, 4) + t)
        old.sample_color = lambda t: torch.cos(torch.arange(old.n * 2).reshape(old.n, 2) + t)
        old.color_std[:] = 0.02
        old.color_rho[:] = 0.9
    new = accelerate(copy.deepcopy(old))
    rng = torch.Generator().manual_seed(46)
    before = torch.get_rng_state()
    for tick in range(270):
        q = torch.randn(old.n, 2, generator=rng) * 0.12
        v = torch.randn(old.n, 2, generator=rng) * 0.4
        target = q + 0.03
        if tick in (17, 129, 201):
            ids = torch.tensor([0, 2, 5])
            old.reset(q[ids], v[ids], target[ids], ids=ids)
            new.reset(q[ids], v[ids], target[ids], ids=ids)
            equal_state(old, new)
        a = old.step(q, v, target, quantized=quantized)
        b = new.step(q, v, target, quantized=quantized)
        for x, y in zip(a, b, strict=True):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
    equal_state(old, new)
    assert torch.equal(before, torch.get_rng_state())


def test_adoption_keeps_every_existing_buffer_and_partial_reset_does_not_restart_clock():
    old = response()
    old.tick = 131
    new = accelerate(old)
    for key, value in old.__dict__.items():
        if isinstance(value, torch.Tensor):
            assert value.data_ptr() == getattr(new, key).data_ptr()
    saved = new.qhist[:, 1:].clone()
    new.reset(new.ref[:1], new.ref[:1] * 0, new.ref[:1], ids=torch.tensor([0]))
    assert new.tick == 131
    torch.testing.assert_close(saved, new.qhist[:, 1:], atol=0, rtol=0)


def test_real_tick_has_full_graph_capture_for_both_pd_phases_without_tick_recompilation():
    old = response()
    new = accelerate(copy.deepcopy(old))
    graphs = []

    def compiler(graph, inputs):
        graphs.append(graph)
        return graph.forward

    new._tick_function = torch.compile(
        response_tick, backend=compiler, fullgraph=True, dynamic=False
    )
    with torch.inference_mode():
        for tick in range(16):
            q = old.ref + tick * 0.001
            v = torch.zeros_like(q)
            a, b = old.step(q, v, q), new.step(q, v, q)
            for x, y in zip(a, b, strict=True):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert len(graphs) == 2  # PD and held-PD; neither 16 clocks nor128 slots.


def test_shared_measurement_counts_only_cold_compile_phases_without_rng_change():
    from native_stand.learning.ppo.performance import Measurement

    old = response()
    timer = Measurement(True)
    new = accelerate(copy.deepcopy(old), measurement=timer)
    new._compiled_tick = True
    new._tick_function = torch.compile(
        response_tick, backend="eager", fullgraph=True, dynamic=False
    )
    rng = torch.get_rng_state()
    for _ in range(16):
        q, v = old.ref, torch.zeros_like(old.ref)
        expected, actual = old.step(q, v, q), new.step(q, v, q)
        for a, b in zip(expected, actual, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert timer.counters["compile_prepare.calls:PD"] == 1
    assert timer.counters["compile_prepare.calls:held_PD"] == 1
    assert timer.parents["compile_prepare"] == "rollout_seconds"
    assert timer.clocks["compile_prepare"] == "cpu_wall"


def test_real_history_gather_matches_causal_indices_and_reuses_index_storage():
    cache = CriticCache(torch.zeros(3, 31))
    for length, ages in ((128, range(1, 25)), (128, range(1, 11)), (32, range(1, 6))):
        ring = torch.arange(length * 3 * 4).reshape(length, 3, 4).float()
        for tick in (0, 1, 127, 128, 271):
            expected = ring[(tick - torch.tensor(list(ages))) % length].permute(1, 0, 2).flatten(1)
            torch.testing.assert_close(cache.history(ring, tick, ages), expected, atol=0, rtol=0)
    ptrs = {k: x.data_ptr() for k, x in cache.indices.items()}
    cache.history(ring, 301, ages)
    assert ptrs == {k: x.data_ptr() for k, x in cache.indices.items()}


def test_cached_real_reset_fields_refresh_after_partial_reset_or_epoch_change():
    n = 3
    dr = NS(
        kp=torch.ones(n, 4) * 40,
        kd=torch.ones(n, 4) * 2,
        kv=torch.ones(n, 2) * 0.2,
        caps=torch.ones(n, 6) * 40,
        wheel_B=torch.zeros(n, 2),
        wheel_Tc=torch.zeros(n, 2),
        parameter_tensor=lambda: torch.ones(n, 32),
    )
    env = NS(
        num_envs=n,
        _motion_reset_epoch=0,
        motion_timing=NS(
            leg=NS(x=torch.ones(n * 2, 13), ref=torch.ones(n * 2, 2)),
            dr=dr,
            base=NS(motor_ring=NS(delay=torch.ones(n, 6), tau=torch.zeros(n, 6))),
        ),
    )
    cache = CriticCache(torch.zeros(n, 31))
    first = cache.reset_fields(env)
    assert cache.reset_fields(env) is first
    dr.kp[1] = 44
    cache.invalidate()
    second = cache.reset_fields(env)
    assert second is not first and second["gains"][1, 0] == pytest.approx(1.1)
    env.motion_timing.leg.x[0] = 2
    env._motion_reset_epoch += 1
    third = cache.reset_fields(env)
    assert third is not second
    assert third["parameters"][0, 0, 0] == 1


def test_timing_fields_survive_real_metrics_consumer(tmp_path):
    from native_stand.artifacts.io import read_jsonl
    from native_stand.artifacts.metrics import MetricsWriter

    path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(path, every_updates=1, tensorboard=False)
    row = dict(
        update=1,
        rollout_seconds=0.12,
        returns_seconds=0.01,
        diagnostics_seconds=0.02,
        learning_seconds=0.04,
        seconds_per_update=0.19,
    )
    writer.write(row)
    writer.close()
    assert read_jsonl(path) == [row]


def test_ground_query_cache_is_invalidated_by_physics_partial_reset_and_surface_identity():
    from native_stand.simulation.ground_surface import from_env, velocity_from_env
    from native_stand.simulation.terrain_surface import TriangleRoads

    heights = torch.arange(9).reshape(1, 3, 3).float() * 0.01
    surface = TriangleRoads(
        torch.cat((heights * 0, heights)),
        torch.zeros(2, 3),
        torch.tensor([[0.0, 0.0, 1.0]] * 2),
        torch.tensor([1, 0]),
        cell=0.5,
        length=1.0,
        width=1.0,
        repetitions=1,
        spawn_x=0.1,
    )
    calls = []
    original = surface.sample

    def sample(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    surface.sample = sample
    position = torch.tensor([[0.0, 0.0, 0.34], [0.0, 0.0, 0.34]])
    velocity = torch.tensor([[0.0, 0.1, 0.0], [0.0, 0.1, 0.0]])
    env = NS(
        motion_surface=surface,
        motion_surface_query_cache=True,
        _sim_step_counter=10,
        _motion_reset_epoch=0,
        motion_road_exit=torch.zeros(2, dtype=torch.bool),
        robot=NS(data=NS(root_pos_w=NS(torch=position), root_lin_vel_w=NS(torch=velocity))),
    )
    first = from_env(env)
    from_env(env)
    velocity_from_env(env)
    assert len(calls) == 1
    position[0, 1] = 0.2
    env._sim_step_counter += 1
    second = from_env(env)
    assert len(calls) == 2 and not torch.equal(first, second)
    position[0, 1] = 0.0
    env._motion_reset_epoch += 1
    torch.testing.assert_close(from_env(env), first, atol=0, rtol=0)
    assert len(calls) == 3
    env.motion_surface_query_cache = False
    from_env(env)
    assert len(calls) == 4


def test_actual_511_encoder_matches_frozen_reference_across_tick_and_DR_reset(monkeypatch):
    import ast
    import sys
    import types

    from test_terrain_geometry import inverse

    from native_stand.simulation import ground_surface

    bundle = Path(reference().__init__.__code__.co_filename).parent
    module_spec = importlib.util.spec_from_file_location(
        "speed_critic_tensor", bundle / "critic_tensor.py"
    )
    numeric = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(numeric)

    def function(path):
        tree = ast.parse(path.read_text())
        fn = next(
            x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "base511_critic"
        )
        namespace = dict(
            torch=torch,
            LEGS=[0, 1, 3, 4],
            WHEELS=[2, 5],
            input_tensor=numeric.input_tensor,
            history=numeric.history,
            __package__="native_stand.simulation.isaac",
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), "exec"
            ),
            namespace,
        )
        return namespace["base511_critic"]

    old = function(bundle / "runtime_view_r4/src/native_stand/students/fullchain_runtime_impl.py")
    new = function(ROOT / "src/native_stand/simulation/isaac/fullchain.py")
    math = types.ModuleType("isaaclab.utils.math")
    math.quat_apply_inverse = inverse
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math)
    monkeypatch.setitem(sys.modules, "ground_height", ground_surface)
    rng = torch.Generator().manual_seed(408)
    n = 4

    def value(*shape):
        return torch.randn(*shape, generator=rng)

    leg = response(n * 2)
    dr = NS(
        kp=value(n, 4),
        kd=value(n, 4),
        kv=value(n, 2),
        caps=value(n, 6).abs() + 1,
        wheel_B=value(n, 2),
        wheel_Tc=value(n, 2),
        actual_armatures=value(n, 6),
        pushed=torch.zeros(n, 2, dtype=torch.bool),
        parameter_tensor=lambda: torch.ones(n, 32),
    )
    base = NS(
        tick=0,
        adopted=value(n, 6),
        requested=value(n, 6),
        targets=value(32, n, 6),
        last_pd_sensors=value(n, 12),
        motor_actual=value(n, 6),
        motor_ring=NS(
            history=value(32, n, 6), actual=value(n, 6), delay=torch.ones(n, 6), tau=value(n, 6)
        ),
    )
    quat = value(n, 4)
    quat /= quat.norm(dim=-1, keepdim=True)
    data = NS(
        root_pos_w=NS(torch=value(n, 3)),
        root_lin_vel_w=NS(torch=value(n, 3)),
        root_quat_w=NS(torch=quat),
    )
    pending = value(n, 3, 16)
    env = NS(
        num_envs=n,
        _motion_reset_epoch=0,
        robot=NS(data=data),
        scene=NS(env_origins=torch.zeros(n, 3)),
        DR_static_ground_normal=torch.tensor([[0.0, 0.0, 1.0]] * n),
        identified_packets=NS(true_frame=value(n, 16), age=torch.zeros(n), pending=lambda: pending),
        contact_force=value(n, 2),
        transient_events=NS(current=value(n, 6)),
        motion_timing=NS(base=base, leg=leg, dr=dr, last_external=value(n, 6)),
        rise_hold=NS(episode=NS(steps=torch.zeros(n, dtype=torch.long))),
    )
    actor = value(n, 31)
    for tick in (0, 17, 129, 271):
        leg.tick = base.tick = tick
        if tick == 129:
            dr.kp[2] *= 1.1
            leg.x[2] += 0.01
            env._motion_reset_epoch += 1
        expected = old(env, actor)
        actual = new(env, actor, body_velocity=inverse(quat, data.root_lin_vel_w.torch))
        assert actual.shape == (n, 511)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
