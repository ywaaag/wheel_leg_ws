FROZEN_TEST_SCOPE='PORTABLE'
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

from types import SimpleNamespace

import pytest
import torch
from native_stand.command_v6.tools.test_resources import project_file,source_file

# Local v3 history is optional: a frozen standalone tree returns None and those
# two plan/flag tests skip, while the pure CPU contract tests still run.
SPEC = project_file('configs/experiments/2026_10_14-21_23__motion31_posture_quality512_v3/spec.json')
CHECKPOINT = project_file('runs/2026_10_14-21_23__motion31_posture_quality512_v3/formal/model_update_1600.pt')
LEDGER = None if SPEC is None else SPEC.with_name('budget.json')
ROOT = None if SPEC is None else SPEC.parents[3]  # <root>/configs/experiments/<name>/spec.json
HISTORY_AVAILABLE = SPEC is not None and CHECKPOINT is not None
SOURCE_SHA = '83ee2371d37c48cefcf61c7de98a1cfb5eb1e01c19071efbb2e6949a36f9a820'


def _caps(**overrides):
    class Args:
        max_speed = None
        max_yaw_speed = None
        speed = 1.
        yaw_speed = 0.
    for key, value in overrides.items():
        setattr(Args, key, value)
    return Args()


def _post(port, payload):
    request = urllib.request.Request('http://127.0.0.1:%d/command' % port,
                                     data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
    return json.loads(urllib.request.urlopen(request, timeout=5).read())


def test_preview_stage_is_one_environment_without_training_quota():
    from native_stand.students.motion_training import commands, preview_stage_config, stage_config
    stage = preview_stage_config(0)
    assert stage['commands'] == commands(0)
    assert stage['sampling_slots'] == [0]
    assert stage['slot_counts'] == [1] + [0] * (len(commands(0)) - 1)
    assert stage['enforce_whitelist'] is False and stage['preview'] is True
    with pytest.raises(ValueError):
        preview_stage_config(99)
    # Training declarations keep their explicit 64/512 quota contract.
    with pytest.raises(ValueError):
        stage_config(0, 1)


def test_preview_command_is_one_explicit_row():
    from native_stand.students.motion_training import preview_command
    row = preview_command(.5, -.25, .34)
    assert row.shape == (1, 3)
    assert torch.allclose(row, torch.tensor([[.5, -.25, .34]]))


def test_preview_caps_default_wide_and_reject_out_of_range():
    from native_stand.students.motion_preview import caps
    assert caps(_caps()) == (2., 1.)
    assert caps(_caps(max_speed=1.5, max_yaw_speed=0.)) == (1.5, 0.)
    with pytest.raises(ValueError):
        caps(_caps(speed=3.))
    with pytest.raises(ValueError):
        caps(_caps(max_speed=5.))
    with pytest.raises(ValueError):
        caps(_caps(max_yaw_speed=-1.))


def test_merge_command_browser_bypasses_second_ramp_and_keeps_caps():
    from native_stand.command_v6.domain.keyboard_command import KeyboardCommand
    from native_stand.students.motion_preview import _merge_command
    controls = KeyboardCommand(1., 0., max_speed=2., max_yaw_speed=1.)
    controls.event('W', True)
    command, source = _merge_command(controls, None, .01, True, (2., 1.))
    assert source == 'isaac-keyboard' and command == pytest.approx([.005, 0.])
    controls.event('W', False)
    row = {'vx': 2., 'yaw': .5}
    command, source = _merge_command(controls, row, .01, False, (1., .25))
    assert source == 'browser-panel'
    assert command == pytest.approx([1., .25])  # browser target clamped by the session cap
    assert controls.current == command
    row = {'vx': 0., 'yaw': 0.}
    command, source = _merge_command(controls, row, .01, False, (2., 1.))
    assert source == 'idle' and command == pytest.approx([0., 0.])


def test_preview_timing_uses_monotonic_wall_time_and_survives_episode_reset():
    from native_stand.students.motion_preview import PreviewTiming
    timing = PreviewTiming(.01, 100.)
    first = timing.sample(10, 100.2)
    assert first['realtime_factor'] is None
    assert first['simulation_time_s'] == pytest.approx(.1)
    row = timing.sample(25, 100.5)
    assert row['realtime_factor'] == pytest.approx(.5)
    # Robot resets do not reset the loop's step counter or throughput clocks.
    row = timing.sample(125, 101.5)
    assert row['realtime_factor'] == pytest.approx(1.)
    assert row['realtime_factor_total'] == pytest.approx(1.25 / 1.5)
    assert row['wall_time_s'] == pytest.approx(1.5)
    assert row['simulation_time_s'] == pytest.approx(1.25)


def test_panel_is_loopback_stdlib_and_cap_bounded(monkeypatch):
    from native_stand.students import motion_panel
    monkeypatch.delenv(motion_panel.ENABLED_ENV, raising=False)
    assert motion_panel.enabled(False) is False and motion_panel.enabled(True) is True
    monkeypatch.setenv(motion_panel.ENABLED_ENV, 'yes')
    assert motion_panel.enabled(False) is True
    monkeypatch.setenv(motion_panel.PORT_ENV, '8766')
    assert motion_panel.port_number() == 8766
    with pytest.raises(ValueError):
        motion_panel.port_number(0)
    plan = motion_panel.Plan(2., 1.)
    server, thread = motion_panel.start(plan, 0)
    try:
        host, port = server.server_address
        assert host == '127.0.0.1' and thread.is_alive()
        page = urllib.request.urlopen('http://127.0.0.1:%d/' % port, timeout=5).read().decode()
        assert 'WSAD' in page and '/command' in page and '/telemetry' in page and 'motion31_manual_preview.csv' in page
        assert json.loads(urllib.request.urlopen('http://127.0.0.1:%d/health' % port, timeout=5).read())['status'] == 'ok'
        assert _post(port, {'vx': 5., 'yaw': -.5})['applied'] == {'vx': 2., 'yaw': -.5}
        row = plan.take()
        assert (row['vx'], row['yaw']) == (2., -.5) and not any((row['stop'], row['center'], row['reset']))
        _post(port, {'stop': True})
        assert plan.take()['stop'] is True and plan.take()['stop'] is False
        assert _post(port, {'max_speed': 1.})['applied'] == {'max_speed': 1.}
        assert plan.take()['max_speed'] == 1.
        with pytest.raises(urllib.error.HTTPError) as raised:
            _post(port, {'max_speed': 3.})
        assert raised.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as raised:
            _post(port, {'torque': 1.})
        assert raised.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen('http://127.0.0.1:%d/unknown' % port, timeout=5)
        assert raised.value.code == 404
        plan.publish({'vx': .25})
        assert json.loads(urllib.request.urlopen('http://127.0.0.1:%d/telemetry' % port, timeout=5).read())['telemetry'] == {'vx': .25}
    finally:
        motion_panel.stop(server)
    assert not thread.is_alive()


def test_preview_stage_satisfies_reset_sampling_contract():
    from native_stand.command_v6.domain.command_sampling import resolve
    from native_stand.students.motion_training import preview_stage_config
    stage = preview_stage_config(0)
    plan = resolve(stage, 1)
    assert plan['mode'] == 'fixed_slots' and plan['fixed_slots'] == [0]
    assert plan['nominal_expected_reset_allocations'] == [1.] + [0.] * (len(stage['commands']) - 1)


def _fake_preview_env(mode=7, initialized=False, valid=False, with_state=True):
    data = SimpleNamespace(joint_vel=SimpleNamespace(torch=torch.zeros(1, 6)),
                           applied_torque=SimpleNamespace(torch=torch.zeros(1, 6)),
                           root_lin_vel_w=SimpleNamespace(torch=torch.zeros(1, 3)))
    env = SimpleNamespace(ids=[0, 1, 2, 3, 4, 5], robot=SimpleNamespace(data=data),
                          command=torch.tensor([[.5, 0., .34]]), vx=torch.zeros(1), vy=torch.zeros(1),
                          yaw=torch.zeros(1), root_height=torch.full((1,), .34))
    if with_state:
        env.fsm = SimpleNamespace(mode=torch.tensor([mode]))
        env.initialization_checked = torch.tensor([initialized])
        env.student_actor_valid = torch.tensor([valid])
    return env


def test_telemetry_reports_task_state_and_never_guesses():
    from native_stand.command_v6.domain.keyboard_command import KeyboardCommand
    from native_stand.students.motion_preview import _state_text, _task_state, _telemetry
    controls = KeyboardCommand(1., 0., max_speed=2., max_yaw_speed=1.)
    # An unsettled initialization holds every action at zero; the diagnostic must say so.
    row = _telemetry(_fake_preview_env(), controls, 5, 0, 'idle', None, 7)
    assert (row['task_mode'], row['initialized'], row['actor_valid']) == ('INITIALIZING', False, False)
    assert row['cmd_vx'] == .5 and row['source'] == 'idle' and row['key_events'] == 7
    unsettled = _state_text(_fake_preview_env(), _task_state(_fake_preview_env()), 7)
    assert 'actions held at zero until settled' in unsettled and 'keys 7' in unsettled
    row = _telemetry(_fake_preview_env(mode=2, initialized=True, valid=True), controls, 6, 1, 'isaac-keyboard', None)
    assert (row['task_mode'], row['initialized'], row['actor_valid']) == ('RECOVERY', True, True)
    assert 'actions held at zero' not in _state_text(_fake_preview_env(mode=2, initialized=True, valid=True),
                                                    _task_state(_fake_preview_env(mode=2, initialized=True, valid=True)))
    # A non-Recovery or partially instrumented env stays UNKNOWN rather than a guessed pass.
    assert _task_state(_fake_preview_env(with_state=False)) == {'task_mode': 'UNKNOWN', 'initialized': None,
                                                                'actor_valid': None}


def test_preview_surfaces_task_state_to_gui_label_and_panel_page():
    preview = source_file('students/motion_preview.py').read_text()
    assert 'state_label.text = _state_text(env, state, received[0])' in preview
    assert "'actor_valid': _first_flag(env, 'student_actor_valid')" in preview
    # Subscribed key events are counted so "no event reached the app" is observable.
    assert preview.count('received[0] += 1') == 2
    page = source_file('students/motion_panel.html').read_text()
    assert 'id="m-mode"' in page and 't.task_mode' in page
    assert 'id="m-keys"' in page and 't.key_events' in page
    assert "'task_mode','initialized','actor_valid','key_events'" in page


def test_manual_preview_stays_opt_in_in_source():
    runtime = source_file('students/motion_runtime.py').read_text()
    # Every hint keeps the scheduled command path when no manual command is set.
    assert runtime.count('if self.motion_manual_command is not None:') == 3
    assert 'self.motion_manual_command=None' in runtime
    assert 'effective(self.target_xy,age,training=self.curriculum_training,config=self.motion_config)' in runtime
    # The env must consume the declared preview stage instead of re-deriving a training one.
    assert "preview_stage=motion_training_config.get('preview_stage')" in runtime
    assert 'requires exactly one environment' in runtime
    assert "if not self.stage_spec.get('preview'):" in runtime
    evaluation = source_file('students/evaluator.py').read_text()
    assert "stage=preview_stage_config(spec['motion_training']['stage'])" in evaluation
    assert "extra['motion_training_config']={**spec['student_training'],'preview_stage':stage}" in evaluation
    assert 'env.enable_motion_preview()' in evaluation
    quality = source_file('students/motion_quality_runtime.py').read_text()
    assert "if getattr(self,'motion_manual_command',None) is not None:" in quality


@pytest.mark.skipif(not HISTORY_AVAILABLE, reason='local v3 spec/checkpoint unavailable')
def test_motion31_cpu_plan_reports_role_caps_panel_and_writes_nothing():
    before = LEDGER.read_bytes() if LEDGER.exists() else None
    command = [sys.executable, '-m', 'native_stand.command_v6.tools.play',
               '--spec', str(SPEC), '--checkpoint', str(CHECKPOINT),
               '--preview', '--keyboard', '--panel', '--physical-gpu', '0']
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, 'PYTHONPATH': 'src', 'PYTHONDONTWRITEBYTECODE': '1'})
    assert process.returncode == 0, process.stderr
    plan = json.loads(process.stdout)
    assert plan['role'] == 'MotionZeroStudent'
    assert plan['checkpoint_sha256'] == SOURCE_SHA
    assert plan['simulation'] == plan['files_written'] == plan['online_PPO'] == 0
    assert plan['keyboard']['max_speed_mps'] == 2. and plan['keyboard']['max_yaw_rad_s'] == 1.
    assert plan['keyboard']['trained_range_mps'] == [0., .5, 1.]
    assert plan['panel']['enabled'] is True and plan['panel']['port'] == 8765
    assert plan['panel']['url'] == 'http://127.0.0.1:8765/'
    assert plan['panel']['host'] == '127.0.0.1'
    assert plan['publication']['status'] == 'VERIFIED'
    assert (LEDGER.read_bytes() if LEDGER.exists() else None) == before


@pytest.mark.skipif(not HISTORY_AVAILABLE, reason='local v3 spec/checkpoint unavailable')
def test_panel_flag_rejects_non_preview_and_non_student_specs(tmp_path):
    base = [sys.executable, '-m', 'native_stand.command_v6.tools.play',
            '--spec', str(SPEC), '--checkpoint', str(CHECKPOINT)]
    recorded = subprocess.run(base + ['--output', str(tmp_path / 'out.json'), '--panel'],
                              cwd=ROOT, capture_output=True, text=True,
                              env={**os.environ, 'PYTHONPATH': 'src', 'PYTHONDONTWRITEBYTECODE': '1'})
    assert recorded.returncode != 0 and 'requires --preview' in recorded.stderr
    legacy = project_file('configs/experiments/stand_plane.json')
    if legacy is not None:
        other = subprocess.run(base[:4] + [str(legacy), '--checkpoint', str(CHECKPOINT), '--preview', '--panel'],
                               cwd=ROOT, capture_output=True, text=True,
                               env={**os.environ, 'PYTHONPATH': 'src', 'PYTHONDONTWRITEBYTECODE': '1'})
        assert other.returncode != 0 and 'Motion student preview only' in other.stderr
