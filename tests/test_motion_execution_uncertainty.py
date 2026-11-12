"""Behavior tests of the real MCU scheduling boundary, no physics step."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from native_stand.students.motion_execution_uncertainty import nominal, screen, combine, validate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from motion_mcu_playback import MCUPlayback
from diagnose_motion_closed_loop_robustness import UncertainExecution, metrics, calibration_biases
from native_stand.students.motion_mcu_execution import profile


def test_one_factor_matrix_and_strict_fields():
    base = nominal()
    assert len(screen()) == 24
    assert len({r['name'] for r in screen()}) == 24
    for row in screen()[1:]:
        assert len([k for k in base if k not in ('name', 'family') and row[k] != base[k]]) == 1
    for row in (dict(base, extra=0), dict(base, actor_age_ms=.5), dict(base, wheel_cap_scale=1.1)):
        with pytest.raises(ValueError): validate(row)


def test_pair_does_not_overwrite_or_pair_same_family():
    rows = screen()
    pair = combine(rows[1], next(r for r in rows if r['family'] == 'motor_response'))
    assert pair['initial_vx_mps'] == -.05 and pair['motor_tau_ms'][0] == 10
    with pytest.raises(ValueError): combine(rows[1], rows[2])


def test_nominal_execution_matches_actual_shared_scheduler_and_previous():
    cfg = {'version': 'motion_mcu_post_pd_execution_v1', 'profile': profile()}
    q = np.array([.21649, .21664, 0., -.21649, .21664, 0.], np.float32)
    target = q.copy(); target[[2, 5]] = 0
    a, b = MCUPlayback(cfg, q, target), UncertainExecution(cfg, q, target, nominal())
    for tick in range(45):
        dq = np.full(6, tick * .001, dtype=np.float32)
        request = target + (tick // 10) * .002
        assert torch.equal(a.step(tick, request, q, dq), b.step(tick, request, q, dq))
        assert torch.equal(a.previous, b.previous)
    assert a.ring.last_tick == b.ring.last_tick


def test_offset_has_consistent_pd_coordinates_and_independent_true_spring():
    from native_stand.students.motion_mcu_execution import spring_compensation
    cfg = {'version': 'motion_mcu_post_pd_execution_v1', 'profile': profile()}
    q = np.array([.21649, .21664, 0., -.21649, .21664, 0.], np.float32)
    r = dict(nominal(), leg_zero_bias_rad=[.02, 0., 0., 0.])
    target = q.copy(); target[0] += .02
    e = UncertainExecution(cfg, q, target, r)
    e.step(0, target, q, np.zeros(6))
    assert torch.allclose(e.pre, torch.zeros(6), atol=2e-6)
    spring = spring_compensation(torch.tensor(q[[0, 1, 3, 4]])[None])[0][0]
    assert torch.equal(e.spring_actual, -spring)


def test_window_boundaries_keep_all_required_samples():
    # Exercise boundaries independently of metric content for an empty slice.
    arrays = {'time': np.arange(1, 3001) * .01}
    assert metrics(arrays, 31., 32., 0.)['samples'] == 0
    # At handover no time, history, queue or previous resets are part of step().
    # Nominal repeated control above calls the same object across policy boundaries.


def test_split_calibration_is_explicit_and_preserves_legacy_common_bias():
    common = dict(nominal(), leg_zero_bias_rad=[.02, 0., 0., 0.])
    a, p = calibration_biases(common)
    np.testing.assert_array_equal(a, p)
    split = dict(version='motion31_split_encoder_offsets_v1',
                 actor_leg_zero_bias_rad=[-.045, -.089, .004, -.059], pd_leg_zero_bias_rad=[0.]*4)
    a, p = calibration_biases(nominal(), split)
    assert a[1] < -.08 and not p.any()
    for invalid in [dict(split, unknown=0), dict(split, version='guess'),
                    dict(split, pd_leg_zero_bias_rad=[float('nan')]*4),
                    dict(split, actor_leg_zero_bias_rad=[.3]*4)]:
        with pytest.raises(ValueError):calibration_biases(nominal(), invalid)
    with pytest.raises(ValueError):calibration_biases(common, split)


@pytest.mark.parametrize('actor_scale,pd_scale', [(0,0), (1,0), (0,1), (1,1), (2,2)])
def test_split_pd_adoption_and_compensation_use_reported_coordinates(actor_scale,pd_scale):
    from native_stand.students.motion_mcu_execution import pd_and_motor, spring_compensation
    cfg = {'version': 'motion_mcu_post_pd_execution_v1', 'profile': profile()}
    q = np.array([.21649, .21664, 0., -.21649, .21664, 0.], np.float32)
    b = np.array([-.045, -.089, .004, -.059], np.float32)
    split = dict(version='motion31_split_encoder_offsets_v1',
                 actor_leg_zero_bias_rad=(actor_scale*b).tolist(),pd_leg_zero_bias_rad=(pd_scale*b).tolist())
    reported = q.copy();reported[[0,1,3,4]] += pd_scale*b
    target = reported.copy()
    e = UncertainExecution(cfg, q, target, nominal(), calibration=split)
    for tick in range(21):
        e.step(tick,target,q,np.zeros(6))
        torch.testing.assert_close(e.pre,torch.zeros(6),atol=2e-6,rtol=0)
        np.testing.assert_allclose(e.previous.numpy(),target,atol=0,rtol=0)
    expected = pd_and_motor(torch.tensor(reported)[None],torch.zeros(1,6),torch.tensor(target)[None])
    assert torch.equal(e.compensation,expected[2][0])
    true_spring = spring_compensation(torch.tensor(q[[0,1,3,4]])[None])[0][0]
    assert torch.equal(e.spring_actual,-true_spring)


def test_current_zero_probe_keeps_seven_declared_hypotheses_distinct():
    from diagnose_motion_mcu_zero_offsets import cases
    b=np.array([-.04517421,-.08956788,.00475454,-.05958552])
    rows=cases(b)
    assert [r['case_id'] for r in rows]==['baseline','actor_only','pd_only','actor_and_pd',
        'half_actor_and_pd','double_actor_and_pd','opposite_actor_and_pd']
    a,p=calibration_biases(nominal(),rows[1]['calibration'])
    np.testing.assert_allclose(a,b);assert not p.any()
    a,p=calibration_biases(nominal(),rows[2]['calibration'])
    assert not a.any();np.testing.assert_allclose(p,b)
    a,p=calibration_biases(nominal(),rows[5]['calibration'])
    np.testing.assert_allclose(a,2*b);np.testing.assert_array_equal(a,p)


def test_combined_stress_matrix_preserves_factorial_controls_and_physical_units():
    from diagnose_motion_mcu_zero_offsets import cases
    b=np.array([-.04517421,-.08956788,.00475454,-.05958552])
    rows=cases(b,'combined_stress_v1')
    assert [r['case_id'] for r in rows]==['nominal','log_bias_only','stress_only',
        'stress_half_bias','stress_full_bias','stress_double_bias']
    assert rows[0]['uncertainty']==rows[1]['uncertainty']==nominal()
    assert all(r['uncertainty']==rows[2]['uncertainty'] for r in rows[2:])
    stress=rows[2]['uncertainty'];validate(stress)
    assert stress['PD_age_ms']==12 and stress['actor_age_ms']==10
    assert stress['motor_delay_ms']==[12,12,9,12,12,13]
    assert stress['spring_scale']==1.1 and stress['initial_vx_mps']==.05
    assert stress['initial_pitch_rate_rad_s']==.1
    assert stress['motor_tau_ms']==nominal()['motor_tau_ms']
    for row,scale in zip(rows,[0,1,0,.5,1,2]):
        a,p=calibration_biases(row['uncertainty'],row['calibration'])
        np.testing.assert_allclose(a,b*scale);np.testing.assert_array_equal(a,p)
    with pytest.raises(ValueError):cases(b,'guess')
