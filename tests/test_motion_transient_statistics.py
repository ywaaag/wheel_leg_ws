"""Portable force-off/quiet/UNKNOWN statistics regressions migrated from production."""
import pytest
import torch
from native_stand.tasks.motion.transient_statistics import Statistics

FROZEN_TEST_SCOPE="PORTABLE"

def stat_fixture(n=1):
    stage = dict(commands=[[0., 0.], [0., 0.]],
        cohort_keys=[[0., 0., .34, -1, 'modal_0_1'], [0., 0., .34, 0, 'modal_combined_0_1']])
    t = Statistics(n, stage, device='cpu')
    t.reset(torch.arange(n), torch.arange(n) % 2, torch.zeros(n, dtype=torch.long), torch.ones(n))
    return t


def observe(t, age, *, hf=.1, dq=.1, support=True, target=0., active=True):
    t.observe(age=torch.full((t.n,), age, dtype=torch.long),
        next_on=torch.full((t.n,), 10000, dtype=torch.long),
        active=torch.full((t.n,), active), support=torch.full((t.n,), support),
        hf2=torch.full((t.n,), hf**2), dq2=torch.full((t.n,), dq**2),
        pitch=torch.zeros(t.n), roll=torch.zeros(t.n), horizontal=torch.zeros(t.n),
        targets=torch.full((t.n, 6), target))


def begin(t, age=0):
    t.begin(torch.ones(t.n, dtype=torch.bool), torch.full((t.n,), age, dtype=torch.long))


def test_no_samples_and_unconfirmed_recovery_are_unknown_not_zero():
    summary = stat_fixture().summary()
    assert summary['phase_status_by_cohort'] == [['UNKNOWN']*5]*2
    assert summary['phase_physical_HF_RMS_rad_s'] == [[None]*5]*2
    assert summary['recovery_enter_delay_mean_s_by_cohort'] == [None, None]
    assert summary['decay_tail_catch_HP10_RMS_ratio_mean_by_cohort'] == [None, None]


def test_long_training_counters_remain_exact_after_float32_integer_limit():
    t = stat_fixture()
    t.student_steps[0] = 2**24
    t.count[0, 2] = 2**24
    t.metrics['support'][0, 2] = 2**24
    t.policy(torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
    begin(t); observe(t, 100)
    assert t.student_steps[0].item() == 2**24+1
    assert t.count[0, 2].item() == 2**24+1
    assert t.metrics['support'][0, 2].item() == 2**24+1


def test_force_off_quiet_start_confirmation_has_no_one_tick_backdating():
    t = stat_fixture(); begin(t)
    observe(t, 90)
    for age in range(100, 1100, 10):
        observe(t, age)
    assert t.recovered.sum() == 0  # 100 snapshots span only 990ms, not a full second.
    observe(t, 1100)
    s = t.summary()
    assert s['recovery_confirmed_by_cohort'] == [1., 0.]
    assert s['recovery_enter_delay_mean_s_by_cohort'] == [0., None]
    assert s['latest_event_clock']['force_off_age_ms'] == [100]
    assert s['latest_event_clock']['confirmed_quiet_enter_age_ms'] == [100]
    assert s['latest_event_clock']['quiet_confirm_age_ms'] == [1100]
    observe(t, 1110)
    assert t.summary()['latest_event_clock']['confirmed_quiet_enter_age_ms'] == [100]


def test_quiet_break_restarts_exact_entry_and_keeps_low_frequency_recovery_visible():
    t = stat_fixture(); begin(t)
    for age in range(100, 600, 10): observe(t, age)
    observe(t, 600, dq=.30)
    for age in range(610, 1620, 10): observe(t, age)
    assert t.recovered[0] == 1
    assert t.enter_age.item() == 610 and t.confirm_age.item() == 1610
    assert t.enter_sum_s[0].item() == pytest.approx(.51)


def test_phase_rms_is_energy_rms_not_mean_of_rms_and_targets_use_actual_units():
    t = stat_fixture(); begin(t)
    observe(t, 100, hf=1., dq=2., target=0.)
    observe(t, 110, hf=3., dq=4., target=.02, support=False)
    s = t.summary(); p = s['phase_measurements']
    assert p['hf'][0][2] == pytest.approx(5**.5)
    assert p['dq'][0][2] == pytest.approx(10**.5)
    assert p['leg_target_step'][0][2] == pytest.approx(.02)
    assert s['phase_dual_support_fraction'][0][2] == .5
    assert s['phase_continuous_support_loss_max_ms'][0][2] == 10.
    assert s['phase_target_step_valid_samples'][0][2] == 1
    assert p['hf'][1][2] is None


def test_partial_reset_preserves_attribution_and_pending_denominators():
    t = stat_fixture(2); begin(t); observe(t, 100)
    t.reset(torch.tensor([0]), torch.tensor([1]), torch.tensor([2]), torch.tensor([-1.]))
    s = t.summary()
    assert s['pulse_events_by_cohort'] == [1., 1.]
    assert s['recovery_unconfirmed_reset_or_next_event_by_cohort'] == [1., 0.]
    assert s['recovery_pending_by_cohort'] == [0., 1.]
    assert s['phase_samples_by_cohort'][0][2] == 1
    t.begin(torch.tensor([True, False]), torch.tensor([0, 100]))
    s = t.summary()
    assert s['pulse_events_by_cohort_mode_sign'][1][2][0] == 1
    assert s['pulse_events_by_cohort'] == [1., 2.]


def test_paired_decay_requires_full_windows_floor_and_summary_does_not_consume():
    t = stat_fixture(); begin(t)
    for age in range(100, 5100, 10):
        observe(t, age, hf=.2 if age < 500 else .05)
    s = t.summary()
    assert s['complete_paired_decay_events_by_cohort'] == [1., 0.]
    assert s['decay_tail_catch_HP10_RMS_ratio_mean_by_cohort'][0] == pytest.approx(.25)
    assert t.summary() == s
    begin(t, 10000)
    assert t.summary()['complete_paired_decay_events_by_cohort'] == [1., 0.]
    t2 = stat_fixture(); begin(t2)
    for age in range(100, 5100, 10): observe(t2, age, hf=.001)
    assert t2.summary()['decay_tail_catch_HP10_RMS_ratio_mean_by_cohort'] == [None, None]
    assert t2.summary()['decay_events_below_catch_floor_by_cohort'] == [1., 0.]
