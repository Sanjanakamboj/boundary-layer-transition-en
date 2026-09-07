import numpy as np
import pytest

from boundary_layer_transition.external_flow import (
    SyntheticVelocityDistribution,
    default_velocity_distribution,
)


@pytest.fixture
def dist():
    return default_velocity_distribution(chord_m=0.70, v_inf_mps=25.0)


def test_endpoints_and_reference_stations(dist):
    # Leading edge: U_e/V_inf = 1 exactly, by construction.
    assert dist.velocity_ratio(0.0) == pytest.approx(1.0, abs=1e-12)
    assert dist.u_e(0.0) == pytest.approx(25.0, abs=1e-9)

    # Peak (end of favorable-gradient region): U_e/V_inf = u_peak_ratio.
    assert dist.velocity_ratio(dist.s_peak) == pytest.approx(dist.u_peak_ratio, abs=1e-9)

    # Trailing edge: U_e/V_inf = u_te_ratio.
    assert dist.velocity_ratio(1.0) == pytest.approx(dist.u_te_ratio, abs=1e-9)
    assert dist.u_e(dist.chord_m) == pytest.approx(25.0 * dist.u_te_ratio, abs=1e-6)


def test_favorable_then_adverse_gradient_character(dist):
    s = np.linspace(0.0, 1.0, 401)
    ratio = dist.velocity_ratio(s)
    i_peak = int(np.argmax(ratio))
    s_at_peak = s[i_peak]
    # Peak should occur near s_peak (grid resolution tolerance).
    assert s_at_peak == pytest.approx(dist.s_peak, abs=0.01)

    # Strictly favorable (non-decreasing) before the peak region starts,
    # strictly adverse (non-increasing) after.
    before = ratio[s <= dist.s_peak]
    after = ratio[s >= dist.s_peak]
    assert np.all(np.diff(before) >= -1e-12)
    assert np.all(np.diff(after) <= 1e-12)


def test_velocity_stays_finite_and_positive(dist):
    s = np.linspace(0.0, 1.0, 2001)
    u = dist.u_e(s * dist.chord_m)
    assert np.all(np.isfinite(u))
    assert np.all(u > 0.0)


def test_derivative_matches_numerical_central_difference(dist):
    x = np.linspace(1e-4, dist.chord_m - 1e-4, 251)
    exact = dist.du_e_dx(x)
    numerical = dist.du_e_dx_numerical(x, h_m=1e-6 * dist.chord_m)
    assert exact == pytest.approx(numerical, abs=1e-3, rel=1e-4)


def test_derivative_hand_check_at_leading_edge(dist):
    # At s=0, t_fav=0, smoothstep_deriv(0) = 6*0 - 6*0^2 = 0, so
    # d(U/Vinf)/ds = 0 at the leading edge -> dU_e/dx = 0 there. Hand-derived
    # independently of the implementation, from the smoothstep formula.
    assert dist.du_e_dx(0.0) == pytest.approx(0.0, abs=1e-9)


def test_derivative_hand_check_at_peak(dist):
    # At s=s_peak the one-sided derivatives from both smoothstep segments
    # are zero (t=1 on the favorable side, t=0 on the adverse side), so
    # dU_e/dx = 0 exactly at the peak -- a hand-derivable consequence of the
    # C1 smoothstep construction.
    x_peak = dist.s_peak * dist.chord_m
    assert dist.du_e_dx(x_peak) == pytest.approx(0.0, abs=1e-8)


def test_derivative_sign_favorable_and_adverse_regions(dist):
    x_fav = 0.5 * dist.s_peak * dist.chord_m
    x_adv = 0.5 * (1.0 + dist.s_peak) * dist.chord_m
    assert dist.du_e_dx(x_fav) > 0.0
    assert dist.du_e_dx(x_adv) < 0.0


def test_scalar_vector_consistency(dist):
    s_vals = [0.0, 0.1, 0.35, 0.6, 1.0]
    vec_ratio = dist.velocity_ratio(np.array(s_vals))
    vec_deriv = dist.d_velocity_ratio_ds(np.array(s_vals))
    for i, s in enumerate(s_vals):
        assert dist.velocity_ratio(s) == pytest.approx(vec_ratio[i], rel=1e-12)
        assert dist.d_velocity_ratio_ds(s) == pytest.approx(vec_deriv[i], rel=1e-12)


def test_nondimensional_dimensional_consistency(dist):
    s_vals = np.linspace(0.0, 1.0, 51)
    x_vals = s_vals * dist.chord_m
    assert dist.u_e(x_vals) == pytest.approx(dist.v_inf_mps * dist.velocity_ratio(s_vals), rel=1e-12)
    assert dist.du_e_dx(x_vals) == pytest.approx(
        (dist.v_inf_mps / dist.chord_m) * dist.d_velocity_ratio_ds(s_vals), rel=1e-12
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(chord_m=0.7, v_inf_mps=25.0, s_peak=0.0),
        dict(chord_m=0.7, v_inf_mps=25.0, s_peak=1.0),
        dict(chord_m=0.7, v_inf_mps=25.0, s_peak=-0.1),
        dict(chord_m=0.7, v_inf_mps=25.0, u_peak_ratio=1.0),
        dict(chord_m=0.7, v_inf_mps=25.0, u_peak_ratio=0.9),
        dict(chord_m=0.7, v_inf_mps=25.0, u_te_ratio=0.0),
        dict(chord_m=0.7, v_inf_mps=25.0, u_te_ratio=-0.2),
        dict(chord_m=0.7, v_inf_mps=25.0, u_te_ratio=1.5, u_peak_ratio=1.2),
        dict(chord_m=0.0, v_inf_mps=25.0),
        dict(chord_m=-0.7, v_inf_mps=25.0),
        dict(chord_m=0.7, v_inf_mps=0.0),
        dict(chord_m=0.7, v_inf_mps=float("nan")),
    ],
)
def test_invalid_distribution_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        SyntheticVelocityDistribution(**kwargs)


def test_out_of_domain_s_rejected(dist):
    with pytest.raises(ValueError):
        dist.velocity_ratio(-0.01)
    with pytest.raises(ValueError):
        dist.velocity_ratio(1.01)
    with pytest.raises(ValueError):
        dist.d_velocity_ratio_ds(1.5)
