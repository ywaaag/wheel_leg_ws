"""Real compiled MuJoCo model boundaries; no time integration or training."""
FROZEN_TEST_SCOPE = 'PORTABLE'

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from native_stand.students.mujoco_solver import PROFILES, DEFAULT_PROFILE, ConstraintRecorder, apply, readback, resolve, surface_ids


def test_new_motion31_default_and_explicit_historical_reproduction():
    motion = dict(interface_version='multiskill_mcu_v2', student_training=dict(version='student_motion31_training_v1'))
    assert resolve(motion) == DEFAULT_PROFILE == 'closure_contact_tight_v1'
    assert resolve(motion, 'legacy_xml_v1') == 'legacy_xml_v1'
    assert resolve({'interface_version': 'multiskill_student_v1'}) == 'legacy_xml_v1'
    assert resolve({'interface_version': 'multiskill_mcu_v2', 'student_training': {'version': 'student_recovery_contact_quality_ppo_v1'}}) == 'legacy_xml_v1'
    with pytest.raises(ValueError, match='Motion31'):
        resolve({}, DEFAULT_PROFILE)


def test_apply_without_override_uses_the_declared_new_default(plant):
    _, model, refs = plant
    result = apply(model, refs)
    assert result['version'] == DEFAULT_PROFILE
    assert model.opt.iterations == 300 and model.opt.tolerance == 1e-12
    np.testing.assert_array_equal(model.eq_solref[:, 0], [.002])


@pytest.fixture
def plant():
    mujoco = pytest.importorskip('mujoco')
    model = mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".001" solver="Newton" iterations="100" tolerance="1e-8"/>
    <default><geom solref=".008 1" solimp=".9 .99 .002 .5 2"/></default>
    <worldbody><geom name="ground" type="plane" size="2 2 .1"/>
    <body name="base" pos="0 0 .1"><freejoint/><geom type="sphere" size=".01"/><site name="a"/>
      <body name="wheel0" pos="0 .1 -.05"><joint/><geom type="sphere" size=".06"/><site name="b"/></body>
      <body name="wheel1" pos="0 -.1 -.05"><joint/><geom type="sphere" size=".06"/></body>
    </body></worldbody><equality><connect body1="base" body2="wheel0" anchor="0 .1 .05" solref=".008 1"/></equality>
    </mujoco>''')
    refs = dict(ground=0, wheels=[2, 3], loops=[(0, 1)])
    return mujoco, model, refs


def preserved(model):
    return {k: getattr(model, k).copy() for k in ('body_mass', 'body_inertia', 'dof_armature',
        'dof_damping', 'jnt_range', 'geom_friction', 'geom_condim', 'geom_margin', 'geom_priority', 'pair_solref')}


@pytest.mark.parametrize('profile', [p for p in PROFILES if p != 'motion_vm1_contact_20261109_v1'])
def test_profiles_keep_physics_and_actuation_contract(plant, profile):
    _, m, r = plant
    before = preserved(m)
    report = apply(m, r, profile)
    assert report['version'] == profile
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(m, key), value)
    assert m.opt.timestep == .001 and int(m.opt.solver) == 2
    if profile == 'legacy_xml_v1':
        assert report['before'] == report['effective']
    if profile == 'convergence_v1':
        assert report['before']['equality_solref'] == report['effective']['equality_solref']
        assert report['before']['surface_solimp'] == report['effective']['surface_solimp']
        assert m.opt.tolerance == 1e-12


def test_contact_profile_changes_selected_surfaces_only(plant):
    _, m, r = plant
    original = m.geom_solref.copy()
    apply(m, r, 'wheel_contact_tight_v1')
    ids = surface_ids(m, r)
    assert ids.tolist() == [0, 2, 3]
    np.testing.assert_array_equal(m.geom_solref[1], original[1])
    np.testing.assert_array_equal(m.geom_solref[ids, 0], [.002] * 3)
    assert m.eq_solref[0, 0] == .008


def test_explicit_historical_vm1_contact_profile_preserves_mechanics_and_limits(plant):
    _,m,r=plant
    keys=('body_mass','body_inertia','dof_armature','dof_damping','jnt_range',
          'actuator_ctrlrange','actuator_forcerange','geom_size','geom_pos')
    before={k:getattr(m,k).copy() for k in keys}
    other=m.geom_solref[1].copy()
    report=apply(m,r,'motion_vm1_contact_20261109_v1')
    for key,value in before.items():np.testing.assert_array_equal(getattr(m,key),value)
    assert int(m.opt.integrator)==3 and m.opt.iterations==100 and m.opt.timestep==.001
    np.testing.assert_array_equal(m.eq_solref[0],[.008,1])
    np.testing.assert_array_equal(m.eq_solimp[0],[.95,.99,.002,.5,2])
    ids=surface_ids(m,r)
    assert (m.geom_margin[ids]==.002).all() and (m.geom_condim[ids]==3).all()
    np.testing.assert_array_equal(m.geom_solref[1],other)
    assert report['source_xml_sha256'].startswith('663e121e')


def test_combined_stricter_only_changes_convergence_and_is_idempotent(plant):
    _, m, r = plant
    first = apply(m, r, 'closure_contact_tight_v1')['effective']
    second = apply(m, r, 'closure_contact_converged_v1')['effective']
    assert first['equality_solref'] == second['equality_solref']
    assert first['surface_solimp'] == second['surface_solimp']
    assert second['iterations'] == 600 and second['tolerance'] == 1e-14
    assert apply(m, r, 'closure_contact_converged_v1')['effective'] == second


def test_invalid_profile_and_unsafe_dt_fail_before_mutation(plant):
    _, m, r = plant
    before = readback(m, r)
    with pytest.raises(ValueError, match='unknown'):
        apply(m, r, 'arbitrary')
    assert readback(m, r) == before
    m.opt.timestep = .002
    with pytest.raises(ValueError, match='1ms'):
        apply(m, r, 'closure_tight_v1')
    m.opt.timestep = .001
    m.opt.disableflags |= 4096
    with pytest.raises(ValueError, match='safety'):
        apply(m, r, 'closure_tight_v1')


def test_actual_contact_readback_and_observer_do_not_change_state(plant):
    mj, m, r = plant
    apply(m, r, 'closure_contact_tight_v1')
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    before = {k: getattr(d, k).copy() for k in ('qpos', 'qvel', 'qacc', 'ctrl', 'qacc_warmstart')}
    recorder = ConstraintRecorder()
    row = recorder(m, d, r)
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(d, key), value)
    assert d.time == 0 and row['constraint_time_s'] == -.001
    actual = recorder.contact_readback()
    assert len(actual) == 2
    assert all(c['solref'] == [.002, 1.] for c in actual)
    assert all(c['solimp'] == [.999, .9999, .0001, .5, 2.] for c in actual)
    assert row['wheel_ground_penetration_m'].shape == (2,)
    assert row['closure_vectors_m'].shape == (1, 3)
