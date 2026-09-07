import math

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.laminar_bl import (
    LAMBDA_MAX_VALID,
    LAMBDA_MIN_VALID,
    solve_thwaites,
    thwaites_separation_lambda,
    thwaites_shape_factor,
    thwaites_shear_function,
)
from boundary_layer_transition.operating_point import default_operating_point


@pytest.fixture
def op():
    return default_operating_point()


@pytest.fixture
def dist(op):
    return default_velocity_distribution(op.chord_m, op.v_inf_mps)


def test_zero_pressure_gradient_recovery_of_correlation_constants():
    """At lambda=0 the Thwaites correlation should recover values
    consistent with the Blasius ZPG solution to within the method's known
    ~few-percent accuracy (a textbook-documented property of the
    correlation, not merely re-calling solve_thwaites)."""
    l0 = thwaites_shear_function(0.0)
    h0 = thwaites_shape_factor(0.0)
    assert l0 == pytest.approx(0.22, abs=1e-12)
    assert h0 == pytest.approx(2.61, abs=1e-12)

    # Hand-derive the implied ZPG skin-friction coefficient from l(0) and
    # the Blasius theta scaling, independent of solve_thwaites:
    #   Cf,x = 2*nu*l(lambda)/(Ue*theta), theta_blasius = 0.664*sqrt(nu x/Ue)
    #   => Cf,x = 2*l(0)/0.664 * sqrt(nu/(Ue x)) = 0.6627 / sqrt(Re_x)
    implied_coeff = 2.0 * l0 / 0.664
    assert implied_coeff == pytest.approx(0.664, rel=5e-3)  # matches Blasius 0.664 within ~0.2%

    # H(0)=2.61 vs Blasius H~2.5916: within the method's known few-% accuracy.
    from boundary_layer_transition.laminar_bl import BLASIUS_H

    assert h0 == pytest.approx(BLASIUS_H, rel=1e-2)


def test_thwaites_theta_matches_independent_manual_quadrature(op, dist):
    """Recompute theta(x) at an interior station via a hand-rolled
    quadrature (numpy.trapz on a manually sampled U_e^5, independent of the
    scipy.integrate.cumulative_trapezoid call used inside solve_thwaites),
    and compare against the production solver's result at that station."""
    sol = solve_thwaites(dist, op.nu_m2s, n_points=4001)

    i_check = 800  # an interior, pre-separation station
    x_check = sol.x_m[i_check]

    x_fine = np.linspace(0.0, x_check, 20001)
    u_fine = dist.u_e(x_fine)
    integral_manual = np.trapezoid(u_fine**5, x_fine)
    theta_sq_manual = 0.45 * op.nu_m2s / dist.u_e(x_check) ** 6 * integral_manual
    theta_manual = math.sqrt(theta_sq_manual)

    assert theta_manual == pytest.approx(sol.theta_m[i_check], rel=2e-3)


def test_cumulative_trapezoid_matches_manual_cumulative_sum(op, dist):
    """Independent re-implementation of the cumulative integral used inside
    solve_thwaites (manual trapezoidal cumulative sum) vs scipy's
    cumulative_trapezoid, on the same grid."""
    n = 501
    x = np.linspace(0.0, dist.chord_m, n)
    u = dist.u_e(x)
    y = u**5

    scipy_cumulative = np.empty(n)
    scipy_cumulative[0] = 0.0
    scipy_cumulative[1:] = cumulative_trapezoid(y, x)

    manual_cumulative = np.zeros(n)
    for i in range(1, n):
        manual_cumulative[i] = manual_cumulative[i - 1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])

    assert scipy_cumulative == pytest.approx(manual_cumulative, rel=1e-10, abs=1e-12)


def test_theta_zero_at_leading_edge(op, dist):
    sol = solve_thwaites(dist, op.nu_m2s, n_points=201)
    assert sol.theta_m[0] == 0.0
    assert sol.lam[0] == 0.0


def test_separation_predicted_and_lambda_root_matches_literature(op, dist):
    sol = solve_thwaites(dist, op.nu_m2s, n_points=4001)
    assert sol.s_sep is not None
    assert 0.0 < sol.s_sep < 1.0
    assert np.any(sol.separated)

    lam_root = thwaites_separation_lambda()
    # Widely cited literature value for Thwaites/Cebeci-Bradshaw separation
    # is lambda ~= -0.09 (White, Viscous Fluid Flow); our root, computed
    # from the actual implemented correlation, should be close to it.
    assert lam_root == pytest.approx(-0.09, abs=5e-3)
    assert LAMBDA_MIN_VALID <= lam_root <= 0.0


def test_h_and_cf_are_nan_beyond_separation(op, dist):
    sol = solve_thwaites(dist, op.nu_m2s, n_points=2001)
    assert sol.s_sep is not None
    post_sep = sol.separated & ~sol.valid
    assert np.any(post_sep)
    assert np.all(np.isnan(sol.h[post_sep]))
    assert np.all(np.isnan(sol.delta_star_m[post_sep]))


def test_cf_undefined_at_leading_edge_not_fabricated(op, dist):
    sol = solve_thwaites(dist, op.nu_m2s, n_points=201)
    assert np.isnan(sol.cf[0])


def test_no_unexpected_nan_or_inf_before_separation(op, dist):
    sol = solve_thwaites(dist, op.nu_m2s, n_points=2001)
    pre_sep_interior = ~sol.separated
    pre_sep_interior[0] = False  # leading edge Cf intentionally NaN
    assert np.all(np.isfinite(sol.theta_m[pre_sep_interior]))
    assert np.all(np.isfinite(sol.h[pre_sep_interior & sol.valid]))
    assert np.all(np.isfinite(sol.cf[pre_sep_interior & sol.valid]))


def test_grid_convergence_theta_and_separation_location(op, dist):
    """Refining the chordwise grid should not materially change theta at a
    fixed station, H at a fixed station, or the predicted separation
    location -- demonstrating numerical convergence of the cumulative
    quadrature."""
    ns = [251, 501, 1001, 2001, 4001]
    s_seps = []
    thetas_at_s03 = []
    for n in ns:
        sol = solve_thwaites(dist, op.nu_m2s, n_points=n)
        s_seps.append(sol.s_sep)
        i_s03 = int(np.argmin(np.abs(sol.s - 0.30)))
        thetas_at_s03.append(sol.theta_m[i_s03])

    # Separation location converges: successive differences shrink.
    diffs = [abs(s_seps[i + 1] - s_seps[i]) for i in range(len(s_seps) - 1)]
    assert diffs[-1] < diffs[0]
    assert diffs[-1] < 1e-3  # converged to << 0.1% of chord between finest grids

    # theta(s=0.30) converges similarly.
    theta_diffs = [abs(thetas_at_s03[i + 1] - thetas_at_s03[i]) for i in range(len(thetas_at_s03) - 1)]
    assert theta_diffs[-1] < theta_diffs[0]
    rel_change_finest = theta_diffs[-1] / thetas_at_s03[-1]
    assert rel_change_finest < 1e-4


def test_scalar_vector_consistency_shear_and_shape_functions():
    lam_vals = [-0.09, -0.05, 0.0, 0.02, 0.08]
    lam_arr = np.array(lam_vals)
    vec_l = thwaites_shear_function(lam_arr)
    vec_h = thwaites_shape_factor(lam_arr)
    for i, lam in enumerate(lam_vals):
        assert thwaites_shear_function(lam) == pytest.approx(vec_l[i], rel=1e-12)
        assert thwaites_shape_factor(lam) == pytest.approx(vec_h[i], rel=1e-12)


def test_invalid_n_points_rejected(op, dist):
    with pytest.raises(ValueError):
        solve_thwaites(dist, op.nu_m2s, n_points=2)


def test_invalid_nu_rejected(dist):
    with pytest.raises(ValueError):
        solve_thwaites(dist, 0.0)
    with pytest.raises(ValueError):
        solve_thwaites(dist, -1e-5)


def test_lambda_validity_bounds_are_symmetric_domain_constants():
    assert LAMBDA_MIN_VALID == -0.10
    assert LAMBDA_MAX_VALID == 0.10
