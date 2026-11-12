FROZEN_TEST_SCOPE = 'PORTABLE'
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_motion_mcu_gap import spectrum, drag, REPORTED_B, REPORTED_TC, timing_review


def test_spectrum_keeps_slow_oscillation_and_separates_dc_drift():
    t = np.arange(800) * .01
    r = spectrum(4 + .3 * t + .2 * np.sin(2 * np.pi * 1.5 * t), .01)
    assert r['peak_hz'] == pytest.approx(1.5)
    assert r['band_power_fraction']['low_0_to_2'] > .99
    assert r['mean'] > 5 and r['detrended_rms'] < .15
    assert spectrum(np.ones(800) * .04, .01)['peak_hz'] is None
    with pytest.raises(ValueError): spectrum([0] * 15, .01)


def test_reported_drag_model_order_and_dissipation():
    assert REPORTED_B.tolist() == [.00066957, .0025485]
    assert REPORTED_TC.tolist() == [.0528843, .054699]
    for dq in [np.array([1., -1.]), np.array([-5., 5.]), np.zeros(2)]:
        resistance = drag(dq, REPORTED_B, REPORTED_TC)
        assert (dq * resistance >= 0).all()
        assert np.allclose(drag(-dq, REPORTED_B, REPORTED_TC), -resistance)


def test_timing_audit_distinguishes_onset_and_response():
    from native_stand.students.motion_mcu_execution import profile
    r = timing_review({'mcu_execution': {'profile': profile()}})
    assert r['zero_to_one_step_onset_ms'] == [10, 10, 7, 10, 10, 11]
    assert r['step_63pct_ms'] == [14, 14, 7, 14, 14, 11]


@pytest.fixture
def tiny_model(tmp_path):
    pytest.importorskip('mujoco')
    xml = tmp_path/'plant.xml'
    xml.write_text('''<mujoco><compiler meshdir="."/><worldbody>
      <geom name="ground" type="plane" size="1 1 .1" contype="2" conaffinity="1"/>
      <body name="robot" pos="0 0 1"><freejoint/>
        <geom name="base" type="box" size=".1 .1 .1" contype="1" conaffinity="2"/>
        <geom name="visual" type="sphere" size=".01" contype="0" conaffinity="0"/>
        <body pos=".2 0 0"><joint name="active" damping=".08"/>
          <geom name="leg" type="capsule" size=".02 .1"/>
          <body pos="0 0 -.2"><joint name="passive" damping=".08"/>
            <geom name="lower" type="sphere" size=".02"/></body>
        </body>
      </body></worldbody><contact><pair geom1="base" geom2="lower"/></contact>
      <equality><joint joint1="active" joint2="passive"/></equality></mujoco>''')
    return xml


def test_alignment_factors_change_only_declared_group(tiny_model):
    import mujoco
    from diagnose_motion_mcu_gap import alignment_model, alignment_parameters
    native = {'joint_damping_by_name': {'active': 0., 'passive': .01},
              'joint_groups': {'active': 'leg_motors', 'passive': 'passive_linkage'},
              'joint_limits_rad': {}}
    for variant, expected in [('active_damping0', [0., .08]),
                              ('passive_damping01', [.08, .01]), ('all_joint_damping', [0., .01])]:
        model = alignment_model(tiny_model)
        refs = {'ground': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'ground')}
        result = alignment_parameters(model, refs, variant, native)
        assert [result['damping'][k] for k in ['active', 'passive']] == pytest.approx(expected)


def test_self_collision_ablation_keeps_ground_and_visual_masks(tiny_model):
    import mujoco
    from diagnose_motion_mcu_gap import alignment_model
    original = tiny_model.read_bytes()
    ref = alignment_model(tiny_model)
    no_self = alignment_model(tiny_model, self_collisions=False)
    assert ref.npair == 1 and no_self.npair == 0
    geom = lambda name: mujoco.mj_name2id(no_self, mujoco.mjtObj.mjOBJ_GEOM, name)
    ground, base, lower, visual = map(geom, ['ground', 'base', 'lower', 'visual'])
    allowed = lambda a, b: bool((no_self.geom_contype[a] & no_self.geom_conaffinity[b]) or
                               (no_self.geom_contype[b] & no_self.geom_conaffinity[a]))
    assert allowed(ground, base) and allowed(ground, lower) and not allowed(base, lower)
    assert not allowed(visual, ground) and not allowed(visual, base)
    assert tiny_model.read_bytes() == original


def test_closure_ablation_preserves_contact_and_actuation_properties(tiny_model):
    import mujoco
    from diagnose_motion_mcu_gap import alignment_model, alignment_parameters
    m = alignment_model(tiny_model)
    refs = {'ground': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'ground')}
    before = {k: getattr(m, k).copy() for k in ['dof_damping', 'dof_armature', 'geom_solref', 'geom_friction']}
    alignment_parameters(m, refs, 'closure_tight', {'joint_damping_by_name': {}, 'joint_groups': {}, 'joint_limits_rad': {}})
    assert np.allclose(m.eq_solref, [.002, 1.])
    assert np.allclose(m.eq_solimp, [.9999, .9999, .001, .5, 2.])
    for k, v in before.items(): assert np.array_equal(getattr(m, k), v), k
