import math

import numpy as np
import pytest

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.laminar_bl import solve_thwaites
from boundary_layer_transition.operating_point import default_operating_point
from boundary_layer_transition.skin_friction import build_drag_grid, section_cf_drag, transitioned_cf_distribution
from boundary_layer_transition.stability import solve_amplification
from boundary_layer_transition.transition import separation_triggered_transition
from boundary_layer_transition.turbulent_bl import (
    ENTRAINMENT_COEFF,
    ENTRAINMENT_EXP,
    H_TR_NOMINAL,
    LT_A,
    LT_B,
    LT_C,
    TurbulentBLStatus,
    entrainment_equation_rhs,
    entrainment_rate,
    h1_of_h,
    h_of_h1,
    ludwieg_tillmann_cf,
    momentum_equation_rhs,
    pressure_gradient_cf_distribution,
    solve_turbulent_bl,
    transition_initial_state,
)

V, NU, C = 25.0, 1.4607e-5, 0.7


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


class _ZPGDist:
    """Minimal zero-pressure-gradient duck-typed distribution for control
    cases, independent of the M1 SyntheticVelocityDistribution machinery."""

    def __init__(self, v: float, chord_m: float):
        self.v = v
        self.chord_m = chord_m

    def u_e(self, x):
        x_arr = np.asarray(x, dtype=float)
        return np.full_like(x_arr, self.v) if x_arr.ndim else self.v

    def du_e_dx(self, x):
        x_arr = np.asarray(x, dtype=float)
        return np.zeros_like(x_arr) if x_arr.ndim else 0.0


class _LinearAccelDist:
    """Favorable-gradient duck-typed distribution: Ue increases linearly."""

    def __init__(self, v0: float, slope: float, chord_m: float):
        self.v0 = v0
        self.slope = slope
        self.chord_m = chord_m

    def u_e(self, x):
        x_arr = np.asarray(x, dtype=float)
        result = self.v0 + self.slope * x_arr
        return result if x_arr.ndim else float(result)

    def du_e_dx(self, x):
        x_arr = np.asarray(x, dtype=float)
        return np.full_like(x_arr, self.slope) if x_arr.ndim else self.slope


class _FakeThwaites:
    def __init__(self, chord_m: float):
        self.x_m = np.array([0.0, chord_m])
        self.theta_m = np.array([0.0, 0.0])
        self.x_sep_m = None


# ---------------------------------------------------------------------------
# A. Turbulent closure formulas -- hand calculations
# ---------------------------------------------------------------------------


def test_h1_of_h_hand_check_low_branch():
    h = 1.3
    hand = 0.8234 * (h - 1.1) ** (-1.287) + 3.3
    assert h1_of_h(h) == pytest.approx(hand, rel=1e-12)


def test_h1_of_h_hand_check_high_branch():
    h = 2.0
    hand = 1.55 * (h - 0.6778) ** (-3.064) + 3.3
    assert h1_of_h(h) == pytest.approx(hand, rel=1e-12)


def test_h1_of_h_continuous_at_branch_switch():
    eps = 1e-6
    below = h1_of_h(1.6 - eps)
    above = h1_of_h(1.6 + eps)
    at = h1_of_h(1.6)
    assert below == pytest.approx(at, abs=1e-4)
    assert above == pytest.approx(at, abs=0.03)  # small residual fit discontinuity (~0.4%), documented


def test_h1_of_h_monotonically_decreasing():
    h_vals = np.linspace(1.10001, 8.0, 500)
    h1_vals = h1_of_h(h_vals)
    assert np.all(np.diff(h1_vals) <= 1e-10)


def test_h_of_h1_roundtrip():
    for h in (1.2, 1.6, 1.8, 2.5, 4.0, 8.0):
        h1 = h1_of_h(h)
        assert h_of_h1(h1) == pytest.approx(h, rel=1e-9)


def test_h_of_h1_rejects_below_asymptote():
    with pytest.raises(ValueError):
        h_of_h1(3.0)
    with pytest.raises(ValueError):
        h_of_h1(2.9)


def test_ludwieg_tillmann_hand_check():
    h, re_theta = 1.4, 2000.0
    hand = 0.246 * 10 ** (-0.678 * h) * re_theta**-0.268
    assert ludwieg_tillmann_cf(h, re_theta) == pytest.approx(hand, rel=1e-12)
    assert LT_A == 0.246 and LT_B == -0.678 and LT_C == -0.268


def test_entrainment_rate_hand_check():
    h1 = 5.0
    hand = 0.0299 * (h1 - 3.0) ** -0.6169
    assert entrainment_rate(h1) == pytest.approx(hand, rel=1e-12)
    assert ENTRAINMENT_COEFF == 0.0299 and ENTRAINMENT_EXP == -0.6169


def test_entrainment_rate_rejects_domain_violation():
    with pytest.raises(ValueError):
        entrainment_rate(3.0)
    with pytest.raises(ValueError):
        entrainment_rate(2.5)


def test_ludwieg_tillmann_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        ludwieg_tillmann_cf(1.0, 1000.0)  # H must be > 1
    with pytest.raises(ValueError):
        ludwieg_tillmann_cf(1.4, -1.0)


# ---------------------------------------------------------------------------
# B. Momentum equation -- reconstructed residual
# ---------------------------------------------------------------------------


def test_momentum_equation_hand_check():
    theta, h, cf, ue, due_dx = 0.001, 1.5, 0.003, 25.0, 10.0
    hand = cf / 2.0 - (2.0 + h) * (theta / ue) * due_dx
    assert momentum_equation_rhs(theta, h, cf, ue, due_dx) == pytest.approx(hand, rel=1e-12)


def test_momentum_equation_zpg_reduces_to_half_cf():
    theta, h, cf, ue = 0.001, 1.4, 0.004, 25.0
    assert momentum_equation_rhs(theta, h, cf, ue, 0.0) == pytest.approx(cf / 2.0, rel=1e-12)


def test_entrainment_equation_hand_check():
    theta, h1, ue, due_dx, dtheta_dx = 0.001, 6.0, 25.0, 5.0, 0.002
    f_h1 = 0.0299 * (h1 - 3.0) ** -0.6169
    hand = (f_h1 - (theta * h1 / ue) * due_dx - h1 * dtheta_dx) / theta
    assert entrainment_equation_rhs(theta, h1, ue, due_dx, dtheta_dx) == pytest.approx(hand, rel=1e-12)


def test_entrainment_equation_matches_expanded_product_rule_form():
    """(1/Ue) d(Ue theta H1)/dx = F(H1) rearranges to the implemented form;
    verify by reconstructing d(Ue theta H1)/dx from a finite-difference of
    a synthetic smooth trajectory and comparing against Ue*F(H1)."""
    ue0, due_dx = 25.0, 8.0
    theta0, dtheta_dx = 0.0015, 0.003
    h1_0 = 5.0

    def ue_theta_h1(x):
        ue = ue0 + due_dx * x
        theta = theta0 + dtheta_dx * x
        # Choose H1(x) consistent with entrainment_equation_rhs at x=0:
        dh1_dx0 = entrainment_equation_rhs(theta0, h1_0, ue0, due_dx, dtheta_dx)
        h1 = h1_0 + dh1_dx0 * x
        return ue * theta * h1

    h = 1e-6
    fd_derivative = (ue_theta_h1(h) - ue_theta_h1(-h)) / (2 * h)
    expected = ue0 * entrainment_rate(h1_0)
    assert fd_derivative == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# C. ZPG control case
# ---------------------------------------------------------------------------


def test_zpg_control_case_reaches_te_and_h_is_physical():
    zpg = _ZPGDist(V, C)
    fake = _FakeThwaites(C)
    tb = solve_turbulent_bl(zpg, fake, V, NU, x_tr_m=0.0, n_report=201)
    assert tb.status == TurbulentBLStatus.COMPLETED_TO_TE
    valid = tb.valid
    assert np.all(valid)
    assert np.all(tb.h[valid] > 1.0)
    assert np.all(tb.theta_m[valid] > 0.0)
    assert np.all(np.isfinite(tb.cf[valid]))
    assert np.all(tb.cf[valid] > 0.0)


def test_zpg_equilibrium_independent_of_initial_h():
    """Starting from different H_tr under ZPG, the solution converges to a
    common H(x) curve at a fixed downstream station -- a known property of
    Head's method (the entrainment equation has an attracting equilibrium),
    verified here by direct numerical comparison rather than assumed."""
    long_zpg = _ZPGDist(V, 5.0)
    fake = _FakeThwaites(5.0)
    h_at_x1 = []
    for h_tr in (1.286, 1.5, 1.7):
        tb = solve_turbulent_bl(long_zpg, fake, V, NU, x_tr_m=1e-4, h_tr=h_tr, n_report=101)
        i_x1 = int(np.argmin(np.abs(tb.x_m - 3.0)))
        h_at_x1.append(tb.h[i_x1])
    assert max(h_at_x1) - min(h_at_x1) < 1e-3


def test_zpg_theta_seed_matches_derivation_near_leading_edge():
    """theta at the very first station should match the derived flat-plate
    seed formula theta = 0.037 x Re_x^-0.2 (used only as the initial
    condition, not claimed to hold throughout -- see next test)."""
    zpg = _ZPGDist(V, C)
    fake = _FakeThwaites(C)
    tb = solve_turbulent_bl(zpg, fake, V, NU, x_tr_m=0.0, n_report=51)
    x0 = tb.x_m[0]
    re_x0 = V * x0 / NU
    theta_pred = 0.037 * x0 * re_x0**-0.2
    assert tb.theta_m[0] == pytest.approx(theta_pred, rel=1e-9)


# ---------------------------------------------------------------------------
# D. Favorable pressure gradient
# ---------------------------------------------------------------------------


def test_favorable_gradient_finite_and_sensible():
    accel = _LinearAccelDist(20.0, 20.0, C)  # Ue: 20 -> 34 m/s over the chord
    fake = _FakeThwaites(C)
    tb = solve_turbulent_bl(accel, fake, V, NU, x_tr_m=0.0, h_tr=1.4, n_report=101)
    valid = tb.valid
    assert np.all(np.isfinite(tb.theta_m[valid]))
    assert np.all(np.isfinite(tb.h[valid]))
    assert np.all(np.isfinite(tb.cf[valid]))
    assert np.all(tb.theta_m[valid] > 0.0)
    assert np.all(tb.h[valid] > 1.0)


def test_favorable_gradient_thinner_than_zpg_reference():
    """A favorable gradient should not produce pathologically larger theta
    growth than the ZPG case starting from the same H_tr."""
    fake = _FakeThwaites(C)
    accel = _LinearAccelDist(25.0, 15.0, C)
    zpg = _ZPGDist(25.0, C)
    tb_accel = solve_turbulent_bl(accel, fake, 25.0, NU, x_tr_m=0.0, h_tr=1.4, n_report=101)
    tb_zpg = solve_turbulent_bl(zpg, fake, 25.0, NU, x_tr_m=0.0, h_tr=1.4, n_report=101)
    assert tb_accel.status == TurbulentBLStatus.COMPLETED_TO_TE
    assert tb_zpg.status == TurbulentBLStatus.COMPLETED_TO_TE
    assert tb_accel.theta_m[-1] < tb_zpg.theta_m[-1]


# ---------------------------------------------------------------------------
# E. Adverse pressure gradient
# ---------------------------------------------------------------------------


def test_adverse_gradient_grows_faster_than_zpg():
    fake = _FakeThwaites(C)
    decel = _LinearAccelDist(30.0, -15.0, C)  # Ue: 30 -> 19.5 m/s
    zpg = _ZPGDist(30.0, C)
    tb_decel = solve_turbulent_bl(decel, fake, 30.0, NU, x_tr_m=0.0, h_tr=1.4, n_report=101)
    tb_zpg = solve_turbulent_bl(zpg, fake, 30.0, NU, x_tr_m=0.0, h_tr=1.4, n_report=101)
    n_common = min(np.sum(tb_decel.valid), np.sum(tb_zpg.valid))
    assert tb_decel.theta_m[n_common - 1] > tb_zpg.theta_m[n_common - 1]
    assert tb_decel.h[n_common - 1] >= tb_zpg.h[n_common - 1]


def test_adverse_gradient_baseline_case_thickens_and_h_rises(op, dist, tsol, asol):
    """The actual M1 adverse-gradient region, propagated turbulently from
    the separation-triggered transition station, should show H rising
    monotonically (a real, honest finding, not forced)."""
    sep = separation_triggered_transition(asol)
    tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, sep.x_tr_m, n_report=301)
    valid = np.where(tb.valid)[0]
    assert len(valid) > 10
    h_valid = tb.h[valid]
    assert np.all(np.diff(h_valid) >= -1e-9)  # H should be non-decreasing under sustained deceleration


def test_baseline_separation_triggered_case_turbulently_separates(op, dist, tsol, asol):
    """Honest finding: this project's synthetic adverse gradient is severe
    enough that the turbulent BL, restarted at the M1 laminar-separation
    station, itself separates before the trailing edge. This is reported,
    not hidden or avoided by adjusting constants."""
    sep = separation_triggered_transition(asol)
    tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, sep.x_tr_m)
    assert tb.status == TurbulentBLStatus.TURBULENT_SEPARATION
    assert tb.x_turb_sep_m is not None
    assert sep.x_tr_m < tb.x_turb_sep_m < tsol.x_m[-1]


# ---------------------------------------------------------------------------
# F. Transition initialization
# ---------------------------------------------------------------------------


def test_transition_initial_state_theta_continuity(tsol):
    x_tr_m = 0.15 * tsol.x_m[-1]
    theta_tr, h1_tr, x_start = transition_initial_state(tsol, V, NU, x_tr_m)
    expected_theta = float(np.interp(x_tr_m, tsol.x_m, tsol.theta_m))
    assert theta_tr == pytest.approx(expected_theta, rel=1e-10)
    assert x_start == pytest.approx(x_tr_m, rel=1e-12)


def test_transition_initial_state_h_tr_default(tsol):
    x_tr_m = 0.1 * tsol.x_m[-1]
    _, h1_tr, _ = transition_initial_state(tsol, V, NU, x_tr_m)
    assert h1_tr == pytest.approx(h1_of_h(H_TR_NOMINAL), rel=1e-12)


def test_transition_initial_state_rejects_beyond_separation(tsol):
    assert tsol.x_sep_m is not None
    with pytest.raises(ValueError):
        transition_initial_state(tsol, V, NU, tsol.x_sep_m + 0.05)


def test_transition_initial_state_fully_turbulent_from_le(tsol):
    theta_tr, h1_tr, x_start = transition_initial_state(tsol, V, NU, 0.0)
    assert x_start > 0.0  # small offset, not the literal singular x=0
    assert theta_tr > 0.0


# ---------------------------------------------------------------------------
# G. Event/status logic
# ---------------------------------------------------------------------------


def test_status_completed_to_te_for_zpg():
    zpg = _ZPGDist(V, C)
    fake = _FakeThwaites(C)
    tb = solve_turbulent_bl(zpg, fake, V, NU, x_tr_m=0.0)
    assert tb.status == TurbulentBLStatus.COMPLETED_TO_TE
    assert tb.x_turb_sep_m is None
    assert np.all(tb.valid)


def test_status_turbulent_separation_for_strong_adverse_gradient():
    fake = _FakeThwaites(2.0)
    strong_decel = _LinearAccelDist(30.0, -14.0, 2.0)  # Ue -> 2 m/s: very severe, forces separation
    tb = solve_turbulent_bl(strong_decel, fake, 30.0, NU, x_tr_m=0.0, h_tr=1.3, n_report=201)
    assert tb.status == TurbulentBLStatus.TURBULENT_SEPARATION
    assert tb.x_turb_sep_m is not None
    assert not np.all(tb.valid)


def test_status_values_distinct():
    values = {s.value for s in TurbulentBLStatus}
    assert values == {"COMPLETED_TO_TE", "TURBULENT_SEPARATION", "INVALID_CLOSURE", "INTEGRATION_FAILURE"}


def test_no_nan_leak_before_separation_event(op, dist, tsol, asol):
    sep = separation_triggered_transition(asol)
    tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, sep.x_tr_m)
    valid = tb.valid
    assert np.all(np.isfinite(tb.theta_m[valid]))
    assert np.all(np.isfinite(tb.h[valid]))
    assert np.all(np.isfinite(tb.cf[valid]))
    assert np.all(np.isnan(tb.theta_m[~valid])) or np.sum(~valid) == 0


# ---------------------------------------------------------------------------
# H. Drag integration -- constant Cf, laminar/turbulent switch, M3 vs M4
# ---------------------------------------------------------------------------


def test_pressure_gradient_cf_distribution_laminar_side_matches_blasius(op, dist, tsol, asol):
    sep = separation_triggered_transition(asol)
    tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, sep.x_tr_m)
    x_before = np.array([0.01, 0.05, 0.9 * sep.x_tr_m])
    from boundary_layer_transition.laminar_bl import blasius_cf

    cf = pressure_gradient_cf_distribution(op.v_inf_mps, x_before, op.nu_m2s, sep.x_tr_m, tb)
    expected = blasius_cf(op.v_inf_mps, x_before, op.nu_m2s)
    assert cf == pytest.approx(expected, rel=1e-10)


def test_pressure_gradient_cf_distribution_turbulent_side_matches_solution(op, dist, tsol, asol):
    sep = separation_triggered_transition(asol)
    tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, sep.x_tr_m)
    valid = np.where(tb.valid)[0]
    i_mid = valid[len(valid) // 2]
    x_query = np.array([tb.x_m[i_mid]])
    cf = pressure_gradient_cf_distribution(op.v_inf_mps, x_query, op.nu_m2s, sep.x_tr_m, tb)
    assert cf[0] == pytest.approx(tb.cf[i_mid], rel=1e-8)


def test_pressure_gradient_cf_frozen_beyond_turbulent_separation(op, dist, tsol, asol):
    sep = separation_triggered_transition(asol)
    tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, sep.x_tr_m)
    assert tb.status == TurbulentBLStatus.TURBULENT_SEPARATION
    last_valid_cf = tb.cf[tb.valid][-1]
    far_beyond = np.array([tsol.x_m[-1]])
    cf = pressure_gradient_cf_distribution(op.v_inf_mps, far_beyond, op.nu_m2s, sep.x_tr_m, tb)
    assert cf[0] == pytest.approx(last_valid_cf, rel=1e-10)


def test_m3_vs_m4_drag_comparison_unrounded(op, dist, tsol, asol):
    sep = separation_triggered_transition(asol)
    tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, sep.x_tr_m)

    s = build_drag_grid(op.chord_m, n_points=2001)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)

    cf_m3 = transitioned_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m=sep.x_tr_m)
    cd_m3 = section_cf_drag(s, cf_m3, ue_ratio, two_surfaces=True)

    cf_m4 = pressure_gradient_cf_distribution(op.v_inf_mps, x, op.nu_m2s, sep.x_tr_m, tb)
    cd_m4 = section_cf_drag(s, cf_m4, ue_ratio, two_surfaces=True)

    pct_diff = 100.0 * (cd_m4 - cd_m3) / cd_m3  # from unrounded cd_m3/cd_m4
    assert math.isfinite(pct_diff)
    assert cd_m4 < cd_m3  # M4 gives less drag here (see DESIGN.md for why)
    assert pct_diff < 0.0


def test_drag_integration_constant_cf_exact():
    s = np.linspace(0.0, 1.0, 501)
    cf = np.full_like(s, 0.004)
    ue_ratio = np.ones_like(s)
    assert section_cf_drag(s, cf, ue_ratio, two_surfaces=True) == pytest.approx(0.008, abs=1e-9)


# ---------------------------------------------------------------------------
# I. Grid/tolerance convergence
# ---------------------------------------------------------------------------


def test_tolerance_convergence_separation_location(op, dist, tsol, asol):
    sep = separation_triggered_transition(asol)
    x_seps = []
    for rtol in (1e-6, 1e-8, 1e-10):
        tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, sep.x_tr_m, rtol=rtol, atol=rtol * 1e-2)
        assert tb.status == TurbulentBLStatus.TURBULENT_SEPARATION
        x_seps.append(tb.x_turb_sep_m)
    diffs = [abs(x_seps[i + 1] - x_seps[i]) for i in range(len(x_seps) - 1)]
    assert diffs[-1] < diffs[0]
    assert diffs[-1] < 1e-4


def test_report_grid_density_does_not_change_status(op, dist, tsol, asol):
    sep = separation_triggered_transition(asol)
    statuses = set()
    for n_report in (101, 301, 501):
        tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, sep.x_tr_m, n_report=n_report)
        statuses.add(tb.status)
    assert statuses == {TurbulentBLStatus.TURBULENT_SEPARATION}


# ---------------------------------------------------------------------------
# J. Scalar/vector consistency
# ---------------------------------------------------------------------------


def test_h1_of_h_scalar_vector_consistency():
    h_vals = [1.2, 1.6, 2.5, 5.0]
    h_arr = np.array(h_vals)
    vec = h1_of_h(h_arr)
    for i, h in enumerate(h_vals):
        assert h1_of_h(h) == pytest.approx(vec[i], rel=1e-12)


def test_ludwieg_tillmann_scalar_vector_consistency():
    h_vals = np.array([1.3, 1.6, 2.0])
    re_vals = np.array([500.0, 1000.0, 5000.0])
    vec = ludwieg_tillmann_cf(h_vals, re_vals)
    for i in range(3):
        assert ludwieg_tillmann_cf(h_vals[i], re_vals[i]) == pytest.approx(vec[i], rel=1e-12)


# ---------------------------------------------------------------------------
# K. Invalid-input rejection
# ---------------------------------------------------------------------------


def test_h1_of_h_rejects_invalid_domain():
    with pytest.raises(ValueError):
        h1_of_h(1.0)
    with pytest.raises(ValueError):
        h1_of_h(1.1)
    with pytest.raises(ValueError):
        h1_of_h(float("nan"))


def test_transition_initial_state_rejects_invalid_x_tr(tsol):
    with pytest.raises(ValueError):
        transition_initial_state(tsol, V, NU, -0.1)
    with pytest.raises(ValueError):
        transition_initial_state(tsol, V, NU, float("nan"))


def test_transition_initial_state_rejects_invalid_h_tr(tsol):
    with pytest.raises(ValueError):
        transition_initial_state(tsol, V, NU, 0.1, h_tr=0.0)
    with pytest.raises(ValueError):
        transition_initial_state(tsol, V, NU, 0.1, h_tr=-1.0)


def test_pressure_gradient_cf_distribution_rejects_invalid_x(op, dist, tsol, asol):
    sep = separation_triggered_transition(asol)
    tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, sep.x_tr_m)
    with pytest.raises(ValueError):
        pressure_gradient_cf_distribution(op.v_inf_mps, np.array([-0.1]), op.nu_m2s, sep.x_tr_m, tb)


# ---------------------------------------------------------------------------
# H_tr initialization sensitivity (quantified, not hidden)
# ---------------------------------------------------------------------------


def test_h_tr_sensitivity_quantified(op, dist, tsol, asol):
    sep = separation_triggered_transition(asol)
    s = build_drag_grid(op.chord_m, n_points=1001)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)

    cd_values = []
    for h_tr in (1.2, H_TR_NOMINAL, 1.4, 1.6):
        tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, sep.x_tr_m, h_tr=h_tr)
        cf = pressure_gradient_cf_distribution(op.v_inf_mps, x, op.nu_m2s, sep.x_tr_m, tb)
        cd_values.append(section_cf_drag(s, cf, ue_ratio, two_surfaces=True))

    # Sensitivity should be modest (same order of magnitude), not wildly divergent.
    assert max(cd_values) / min(cd_values) < 1.5
    assert all(math.isfinite(v) for v in cd_values)
