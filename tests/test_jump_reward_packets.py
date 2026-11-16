"""High-frequency packet validation keeps all external rejection boundaries."""

from dataclasses import replace

import numpy as np
import pytest
from test_jump_unified import packet

FROZEN_TEST_SCOPE = "PORTABLE"


@pytest.mark.parametrize(
    "name",
    [
        "COM_velocity_m_s",
        "tilt_rad",
        "omega_rad_s",
        "leg_length_m",
        "leg_swing_rad",
        "spring_length_m",
        "tire_clearance_m",
        "wheel_normal_force_N",
        "contact_slip_m_s",
        "actual_motor_Nm",
        "actual_dq_rad_s",
        "adopted_target",
    ],
)
def test_every_numeric_packet_field_rejects_wrong_shape_and_nonfinite(name):
    p = packet()
    old = getattr(p, name)
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite explicitly aligned packet field " + name):
            replace(p, **{name: (bad,) + old[1:]})
    with pytest.raises(ValueError, match="finite explicitly aligned packet field " + name):
        replace(p, **{name: old[:-1]})


@pytest.mark.parametrize("name", ["COM_height_m", "root_height_m"])
def test_packet_heights_still_reject_nonfinite(name):
    with pytest.raises(ValueError, match="finite measured heights"):
        packet(**{name: float("nan")})


def test_contact_type_force_sign_and_owned_numeric_tuples_remain_explicit():
    for values in ((1, 0), (True, 0.0)):
        with pytest.raises(ValueError, match="explicit booleans"):
            packet(contact=values)
    with pytest.raises(ValueError, match="nonnegative"):
        packet(wheel_normal_force_N=(-1, 0))
    with pytest.raises(ValueError, match="nonnegative"):
        packet(contact_slip_m_s=(0, -1))
    source = np.arange(6, dtype=np.float32)
    p = packet(adopted_target=source, contact=(np.bool_(True), np.bool_(False)))
    source[:] = 99
    assert p.adopted_target == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert p.contact == (True, False)
    assert all(type(x) is float for x in p.adopted_target)
