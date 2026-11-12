"""Current MCU plant and public default plan; no engine step, BC, or PPO."""
FROZEN_TEST_SCOPE = 'PORTABLE'

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
REQUEST = ROOT / 'configs/workflows/2026_10_26-17_49__motion4500_small31_mcu_distillation.json'


def local_request():
    if not REQUEST.exists():
        pytest.skip('local current MCU Teacher request absent')
    request = json.loads(REQUEST.read_text())
    if not all(Path(request[k]).exists() for k in ('parent_spec', 'xml', 'plant_adapter', 'teacher')):
        pytest.skip('local exact source fixtures absent')
    return request, json.loads(Path(request['parent_spec']).read_text())


def test_current_plant_matches_the_completed_precision_probe_materialization():
    mj = pytest.importorskip('mujoco')
    from validate_native_mujoco import binding
    from motion_mcu_playback import apply_current_plant, configuration, declared
    from compare_motion_timing_profiles import apply_actuator_profile
    request, spec = local_request()
    info = importlib.util.spec_from_file_location('frozen_probe_plant', request['plant_adapter'])
    module = importlib.util.module_from_spec(info); info.loader.exec_module(module)
    frozen = module.Plant(request, spec)
    cfg = configuration(SimpleNamespace(leg_armature_m=.012, zero_latency=False), declared(spec), spec)
    apply_actuator_profile(frozen.model, frozen.refs, cfg, frozen.original)
    cap = np.asarray(cfg['profile']['solver_limits_nm']) / np.abs(frozen.refs['gears'])
    frozen.model.actuator_ctrlrange[frozen.refs['actuators']] = np.stack((-cap, cap), -1)
    frozen.model.geom_friction[:] = 0.; frozen.model.geom_friction[:, 0] = .5
    frozen.model.dof_damping[np.asarray(frozen.refs['dofs'])[[2,5]]] = 0.
    frozen.model.dof_frictionloss[np.asarray(frozen.refs['dofs'])[[2,5]]] = 0.
    model = mj.MjModel.from_xml_path(request['xml']); refs = binding(model)
    before_mass = model.body_mass.copy()
    result = apply_current_plant(model, refs, spec, cfg)
    assert result['version'] == 'motion_mcu_currentplant_v1'
    for field in ('dof_armature','dof_damping','dof_frictionloss','jnt_range','jnt_limited',
                  'geom_friction','actuator_ctrlrange','body_mass','body_inertia','eq_solref','eq_solimp'):
        np.testing.assert_array_equal(getattr(model, field), getattr(frozen.model, field))
    np.testing.assert_array_equal(model.body_mass, before_mass)
    assert model.opt.timestep == .001


def test_public_plan_defaults_and_legacy_override_consume_real_model_and_manifest():
    pytest.importorskip('mujoco')
    from validate_motion31_mujoco import plan
    request, spec = local_request()
    args = SimpleNamespace(spec=Path(request['parent_spec']), checkpoint=Path(request['teacher']),
        xml=Path(request['xml']),wall_seconds=60,commands=[0],yaw_commands=[],
        leg_armature_m=.012,zero_latency=False)
    model, refs, actor, default = plan(args, spec)
    assert default['solver_profile']['version'] == 'closure_contact_tight_v1'
    assert default['plant_profile']['version'] == 'motion_mcu_currentplant_v1'
    assert default['solver_profile']['effective']['surface_friction'] == [[.5,0.,0.]]*5
    assert len(default['initial_actor_input']) == 31 and default['critic_not_loaded']
    assert default['actuator_configuration']['profile']['driver_limits_nm'][2] < 2.447
    legacy_args = SimpleNamespace(**vars(args), solver_profile='legacy_xml_v1')
    old, oldrefs, _, legacy = plan(legacy_args, spec)
    assert legacy['solver_profile']['before'] == legacy['solver_profile']['effective']
    assert legacy['plant_profile']['version'] == 'source_xml_v1'
    assert old.opt.iterations == 100 and old.eq_solref[0,0] == .008
    assert model.opt.iterations == 300 and model.eq_solref[0,0] == .002


def test_source_spec_cannot_silently_activate_the_current_mcu_plant():
    from motion_mcu_playback import plant_profile
    assert plant_profile({}, 'closure_contact_tight_v1') == 'source_xml_v1'
    with pytest.raises(ValueError, match='declared'):
        plant_profile({}, 'closure_contact_tight_v1', 'motion_mcu_currentplant_v1')
