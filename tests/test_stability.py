import math

import numpy as np
import pytest

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.laminar_bl import solve_thwaites
from boundary_layer_transition.operating_point import default_operating_point
from boundary_layer_transition.stability import (
    AMPLIFICATION_COEFF,
    H_REFERENCE,
    ONSET_H_SENSITIVITY,
    RE_THETA_CRIT_ZPG,
    critical_re_theta,
    integrate_n_factor,
    interpolate_n,
    is_unstable,
    local_amplification_rate,
    reynolds_theta,
    solve_amplification,
)


@pytest.fixture
def op():
    return default_operating_point()


@pytest.fixture
def dist(op):
    return default_velocity_distribution(op.chord_m, op.v_inf_mps)


@pytest.fixture
def tsol(op, dist):
    return solve_thwaites(dist, op.nu_m2s, n_points=4001)


@pytest.fixture
def asol(op, tsol):
    return solve_amplification(op, tsol)


# ---------------------------------------------------------------------------
# A. Re_theta -- independent arithmetic
# ---------------------------------------------------------------------------


def test_reynolds_theta_hand_check():
    rho, mu = 1.225, 1.7894e-5
    u_e, theta = 27.5, 3.5e-4
    hand = rho * u_e * theta / mu
    assert reynolds_theta(rho, u_e, theta, mu) == pytest.approx(hand, rel=1e-12)
    assert reynolds_theta(rho, u_e, theta, mu) == pytest.approx(658.915, rel=1e-3)


def test_reynolds_theta_zero_at_zero_theta():
    assert reynolds_theta(1.225, 25.0, 0.0, 1.7894e-5) == 0.0


def test_reynolds_theta_matches_thwaites_station(op, tsol):
    i = 900
    hand = op.rho_kgm3 * tsol.u_e_mps[i] * tsol.theta_m[i] / op.mu_pas
    assert reynolds_theta(op.rho_kgm3, tsol.u_e_mps[i], tsol.theta_m[i], op.mu_pas) == pytest.approx(hand, rel=1e-12)


# ---------------------------------------------------------------------------
# B. Onset relation -- hand evaluation, below/above threshold
# ---------------------------------------------------------------------------


def test_critical_re_theta_hand_check_at_reference_h():
    # At H = H_reference exactly, Re_theta,crit = Re_theta,crit,ZPG (exp(0)=1).
    assert critical_re_theta(H_REFERENCE) == pytest.approx(RE_THETA_CRIT_ZPG, rel=1e-12)


def test_critical_re_theta_hand_check_offset_h():
    h = H_REFERENCE + 0.5
    hand = RE_THETA_CRIT_ZPG * math.exp(-ONSET_H_SENSITIVITY * 0.5)
    assert critical_re_theta(h) == pytest.approx(hand, rel=1e-12)
    # Higher H (more adverse gradient / less full profile) -> lower critical Re_theta.
    assert critical_re_theta(h) < RE_THETA_CRIT_ZPG


def test_critical_re_theta_monotonic_decreasing_in_h():
    h_vals = np.linspace(2.2, 4.0, 50)
    crit = critical_re_theta(h_vals)
    assert np.all(np.diff(crit) < 0.0)


def test_is_unstable_below_and_above_threshold():
    h = H_REFERENCE
    crit = critical_re_theta(h)
    assert is_unstable(crit - 1.0, h) is False
    assert is_unstable(crit + 1.0, h) is True
    assert is_unstable(crit, h) is True  # boundary is inclusive (>=)


def test_is_unstable_vector_hand_check():
    h = np.full(5, H_REFERENCE)
    crit = RE_THETA_CRIT_ZPG
    re_theta = np.array([crit - 50, crit - 1, crit, crit + 1, crit + 50])
    expected = np.array([False, False, True, True, True])
    assert np.array_equal(is_unstable(re_theta, h), expected)


# ---------------------------------------------------------------------------
# C. Amplification rate -- hand calculation, units, sign
# ---------------------------------------------------------------------------


def test_local_amplification_rate_hand_check():
    h = 3.0
    re_theta = 350.0
    crit = RE_THETA_CRIT_ZPG * math.exp(-ONSET_H_SENSITIVITY * (h - H_REFERENCE))
    expected = AMPLIFICATION_COEFF * (h - 1.0) * max(0.0, re_theta - crit) / RE_THETA_CRIT_ZPG
    assert local_amplification_rate(re_theta, h) == pytest.approx(expected, rel=1e-12)


def test_local_amplification_rate_zero_when_stable():
    h = H_REFERENCE
    crit = critical_re_theta(h)
    assert local_amplification_rate(crit - 10.0, h) == 0.0
    assert local_amplification_rate(crit, h) == 0.0  # exactly at threshold: zero excess


def test_local_amplification_rate_nonnegative_and_finite():
    rng_h = np.linspace(2.2, 4.0, 40)
    rng_re = np.linspace(0.0, 800.0, 40)
    h_grid, re_grid = np.meshgrid(rng_h, rng_re)
    rate = local_amplification_rate(re_grid, h_grid)
    assert np.all(np.isfinite(rate))
    assert np.all(rate >= 0.0)


def test_local_amplification_rate_continuous_through_onset():
    """No artificial jump: the rate should approach 0 continuously as
    Re_theta approaches Re_theta,crit(H) from above."""
    h = 3.2
    crit = critical_re_theta(h)
    eps = 1e-6
    just_below = local_amplification_rate(crit - eps, h)
    at = local_amplification_rate(crit, h)
    just_above = local_amplification_rate(crit + eps, h)
    assert just_below == 0.0
    assert at == 0.0
    assert just_above == pytest.approx(0.0, abs=1e-3)


def test_local_amplification_rate_rejects_negative_k():
    with pytest.raises(ValueError):
        local_amplification_rate(300.0, 3.0, k=-1.0)


# ---------------------------------------------------------------------------
# D. N integration -- constant-rate, piecewise, stable, monotonicity
# ---------------------------------------------------------------------------


def test_integrate_constant_rate_analytic():
    """N(x) = k*(x - x0) for a constant dN/ds = k over [x0, x1]."""
    x0, x1, k = 0.1, 0.6, 2.5
    s = np.linspace(x0, x1, 501)
    dn_ds = np.full_like(s, k)
    n = integrate_n_factor(s, dn_ds)
    analytic = k * (s - x0)
    assert n == pytest.approx(analytic, abs=1e-9)
    assert n[0] == 0.0
    assert n[-1] == pytest.approx(k * (x1 - x0), rel=1e-9)


def test_integrate_piecewise_zero_then_linear():
    """Zero before x0, constant rate k after -- N should stay exactly zero
    in the zero-rate region and grow linearly beyond it."""
    x0 = 0.3
    s = np.linspace(0.0, 1.0, 1001)
    k = 4.0
    dn_ds = np.where(s < x0, 0.0, k)
    n = integrate_n_factor(s, dn_ds)
    pre = s < x0
    assert np.all(n[pre] == pytest.approx(0.0, abs=1e-9))
    i_x0 = np.argmin(np.abs(s - x0))
    analytic_post = k * (s - s[i_x0])
    assert n[i_x0:] == pytest.approx(analytic_post[i_x0:], abs=5e-3)


def test_integrate_stable_case_gives_zero_everywhere():
    s = np.linspace(0.0, 1.0, 201)
    dn_ds = np.zeros_like(s)
    n = integrate_n_factor(s, dn_ds)
    assert np.all(n == 0.0)


def test_integrate_monotonic_for_nonnegative_rate():
    s = np.linspace(0.0, 1.0, 301)
    rng = np.random.default_rng(0)
    dn_ds = np.abs(rng.normal(size=s.shape))  # nonnegative by construction
    n = integrate_n_factor(s, dn_ds)
    assert np.all(np.diff(n) >= -1e-12)


def test_integrate_rejects_non_increasing_s():
    s = np.array([0.0, 0.2, 0.2, 0.5])
    dn_ds = np.array([0.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        integrate_n_factor(s, dn_ds)


def test_integrate_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        integrate_n_factor(np.linspace(0, 1, 5), np.zeros(4))


def test_integrate_rejects_too_few_points():
    with pytest.raises(ValueError):
        integrate_n_factor(np.array([0.0]), np.array([0.0]))


# ---------------------------------------------------------------------------
# E. Separation handling -- no fabricated post-separation amplification
# ---------------------------------------------------------------------------


def test_no_amplification_accumulated_after_m1_separation(asol):
    assert asol.s_sep is not None
    beyond = asol.s > asol.s_sep
    assert np.any(beyond)
    assert np.all(np.isnan(asol.n[beyond]))
    assert np.all(np.isnan(asol.dn_ds[beyond]))
    assert np.all(np.isnan(asol.re_theta[beyond]))
    assert np.all(np.isnan(asol.re_theta_crit[beyond]))


def test_valid_domain_matches_m1_h_validity(tsol, asol):
    assert np.array_equal(asol.valid, ~np.isnan(tsol.h))


def test_n_defined_and_finite_within_valid_domain(asol):
    assert np.all(np.isfinite(asol.n[asol.valid]))
    assert np.all(np.isfinite(asol.dn_ds[asol.valid]))


def test_onset_before_separation(asol):
    assert asol.onset_s is not None
    assert asol.onset_s < asol.s_sep


def test_n_zero_up_to_onset(asol):
    # Strictly before the onset station, dN/ds has been zero at every prior
    # grid point, so the cumulative trapezoidal integral is exactly zero.
    strictly_pre_onset = asol.valid & (asol.s < asol.onset_s)
    assert np.all(asol.n[strictly_pre_onset] == 0.0)
    # At the onset station itself, N may pick up a small (grid-resolution)
    # trapezoidal contribution since the rate is already >= 0 there by
    # definition of "onset" -- but it must be negligible, not a jump.
    i_onset = np.argmin(np.abs(asol.s - asol.onset_s))
    grid_spacing = asol.s[1] - asol.s[0]
    assert 0.0 <= asol.n[i_onset] < 1e-2 * grid_spacing * AMPLIFICATION_COEFF


def test_n_monotonic_nondecreasing_in_valid_region(asol):
    n_valid = asol.n[asol.valid]
    assert np.all(np.diff(n_valid) >= -1e-9)


def test_n_max_matches_last_valid_station(asol):
    idx = np.where(asol.valid)[0]
    assert asol.n_max == pytest.approx(asol.n[idx[-1]], rel=1e-12)


# ---------------------------------------------------------------------------
# F. Grid convergence
# ---------------------------------------------------------------------------


def test_grid_convergence_onset_nmax_and_separation(op, dist):
    ns = [251, 501, 1001, 2001, 4001]
    onsets, nmaxs, s_seps = [], [], []
    for n_pts in ns:
        tsol_n = solve_thwaites(dist, op.nu_m2s, n_points=n_pts)
        asol_n = solve_amplification(op, tsol_n)
        onsets.append(asol_n.onset_s)
        nmaxs.append(asol_n.n_max)
        s_seps.append(asol_n.s_sep)

    onset_diffs = [abs(onsets[i + 1] - onsets[i]) for i in range(len(onsets) - 1)]
    nmax_diffs = [abs(nmaxs[i + 1] - nmaxs[i]) for i in range(len(nmaxs) - 1)]
    ssep_diffs = [abs(s_seps[i + 1] - s_seps[i]) for i in range(len(s_seps) - 1)]

    assert onset_diffs[-1] < onset_diffs[0]
    assert onset_diffs[-1] < 2e-3  # converged to << 1% of chord between finest grids

    assert nmax_diffs[-1] < nmax_diffs[0]
    assert nmax_diffs[-1] / nmaxs[-1] < 5e-3  # < 0.5% relative change, finest pair

    assert ssep_diffs[-1] < 1e-3


# ---------------------------------------------------------------------------
# G. Scalar/vector consistency
# ---------------------------------------------------------------------------


def test_scalar_vector_consistency_all_public_functions():
    h_vals = [2.3, H_REFERENCE, 2.8, 3.5]
    re_vals = [50.0, 200.0, 300.0, 500.0]
    h_arr, re_arr = np.array(h_vals), np.array(re_vals)

    vec_crit = critical_re_theta(h_arr)
    vec_unstable = is_unstable(re_arr, h_arr)
    vec_rate = local_amplification_rate(re_arr, h_arr)

    for i in range(len(h_vals)):
        assert critical_re_theta(h_vals[i]) == pytest.approx(vec_crit[i], rel=1e-12)
        assert is_unstable(re_vals[i], h_vals[i]) == bool(vec_unstable[i])
        assert local_amplification_rate(re_vals[i], h_vals[i]) == pytest.approx(vec_rate[i], rel=1e-12)


def test_reynolds_theta_scalar_vector_consistency():
    u_vals = np.array([10.0, 20.0, 30.0])
    theta_vals = np.array([1e-4, 2e-4, 3e-4])
    vec = reynolds_theta(1.225, u_vals, theta_vals, 1.7894e-5)
    for i in range(3):
        assert reynolds_theta(1.225, u_vals[i], theta_vals[i], 1.7894e-5) == pytest.approx(vec[i], rel=1e-12)


# ---------------------------------------------------------------------------
# H. Invalid-input rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(rho_kgm3=0.0, u_e_mps=25.0, theta_m=1e-4, mu_pas=1.7894e-5),
        dict(rho_kgm3=1.225, u_e_mps=25.0, theta_m=1e-4, mu_pas=0.0),
        dict(rho_kgm3=1.225, u_e_mps=-1.0, theta_m=1e-4, mu_pas=1.7894e-5),
        dict(rho_kgm3=1.225, u_e_mps=25.0, theta_m=-1e-4, mu_pas=1.7894e-5),
    ],
)
def test_reynolds_theta_invalid_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        reynolds_theta(**kwargs)


def test_critical_re_theta_rejects_nonpositive_zpg_anchor():
    with pytest.raises(ValueError):
        critical_re_theta(2.6, re_theta_crit_zpg=0.0)


def test_critical_re_theta_rejects_nonfinite_h():
    with pytest.raises(ValueError):
        critical_re_theta(float("nan"))
    with pytest.raises(ValueError):
        critical_re_theta(float("inf"))


def test_local_amplification_rate_rejects_nonfinite():
    with pytest.raises(ValueError):
        local_amplification_rate(float("nan"), 3.0)
    with pytest.raises(ValueError):
        local_amplification_rate(300.0, float("nan"))


# ---------------------------------------------------------------------------
# interpolate_n
# ---------------------------------------------------------------------------


def test_interpolate_n_matches_grid_values(asol):
    valid_idx = np.where(asol.valid)[0]
    i = valid_idx[len(valid_idx) // 2]
    s_q = asol.s[i]
    assert interpolate_n(asol, s_q) == pytest.approx(asol.n[i], abs=1e-9)


def test_interpolate_n_outside_domain_is_nan(asol):
    assert math.isnan(interpolate_n(asol, 1.0)) or asol.s_sep >= 1.0
    far_beyond = interpolate_n(asol, min(1.0, asol.s_sep + 0.2) if asol.s_sep is not None else 1.0)
    assert math.isnan(far_beyond)


def test_interpolate_n_vector_matches_manual_numpy_interp(asol):
    valid_idx = np.where(asol.valid)[0]
    s_query = np.array([asol.s[valid_idx[2]], asol.s[valid_idx[len(valid_idx) // 3]]])
    result = interpolate_n(asol, s_query)
    manual = np.interp(s_query, asol.s[valid_idx], asol.n[valid_idx])
    assert result == pytest.approx(manual, rel=1e-10)
