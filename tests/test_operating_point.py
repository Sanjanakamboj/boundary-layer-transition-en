import math

import pytest

from boundary_layer_transition.operating_point import (
    GenericSailplaneOperatingPoint,
    default_operating_point,
    speed_of_sound,
)


def test_default_operating_point_re_c_hand_check():
    """Independent hand computation of Re_c via two different but
    algebraically equivalent formulas (rho*V*c/mu vs V*c/nu), computed here
    with plain arithmetic rather than by calling op.re_c a second time.
    """
    op = default_operating_point()

    # Path 1: direct definition Re_c = rho * V * c / mu, computed by hand.
    re_c_hand = (1.225 * 25.0 * 0.70) / 1.7894e-5

    # Path 2: via kinematic viscosity, nu = mu/rho computed independently.
    nu_hand = 1.7894e-5 / 1.225
    re_c_hand_via_nu = 25.0 * 0.70 / nu_hand

    assert re_c_hand == pytest.approx(re_c_hand_via_nu, rel=1e-12)
    assert op.re_c == pytest.approx(re_c_hand, rel=1e-9)
    assert op.re_c == pytest.approx(1.198013e6, rel=1e-4)


def test_default_operating_point_mach_hand_check():
    op = default_operating_point()
    # a = sqrt(gamma R T), hand-computed.
    a_hand = math.sqrt(1.4 * 287.05 * 288.15)
    assert a_hand == pytest.approx(340.29, rel=1e-3)
    mach_hand = 25.0 / a_hand
    assert op.mach == pytest.approx(mach_hand, rel=1e-9)
    assert op.is_incompressible()
    assert op.mach < 0.1  # comfortably low-speed


def test_speed_of_sound_matches_known_isa_sea_level_value():
    a = speed_of_sound(288.15)
    assert a == pytest.approx(340.3, abs=0.1)


def test_nu_consistency():
    op = default_operating_point()
    assert op.nu_m2s == pytest.approx(op.mu_pas / op.rho_kgm3, rel=1e-12)
    assert op.nu_m2s == pytest.approx(1.4607e-5, rel=1e-3)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(chord_m=0.0),
        dict(chord_m=-1.0),
        dict(chord_m=float("nan")),
        dict(v_inf_mps=0.0),
        dict(v_inf_mps=-5.0),
        dict(rho_kgm3=0.0),
        dict(mu_pas=-1e-5),
        dict(temperature_k=0.0),
        dict(alpha_deg=float("nan")),
    ],
)
def test_invalid_operating_point_rejected(kwargs):
    with pytest.raises(ValueError):
        GenericSailplaneOperatingPoint(**kwargs)


def test_speed_of_sound_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        speed_of_sound(0.0)
    with pytest.raises(ValueError):
        speed_of_sound(-10.0)
