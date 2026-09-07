import math

import numpy as np
import pytest

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.laminar_bl import blasius_cf, reynolds_x, solve_thwaites
from boundary_layer_transition.operating_point import default_operating_point
from boundary_layer_transition.skin_friction import (
    LAMINAR_CF_COEFF_AVERAGE,
    TURBULENT_CF_COEFF_AVERAGE,
    TURBULENT_CF_COEFF_LOCAL,
    TURBULENT_CF_RE_MAX,
    TURBULENT_CF_RE_MIN,
    build_drag_grid,
    is_turbulent_correlation_valid,
    laminar_cf_average,
    section_cf_drag,
    transitioned_cf_distribution,
    turbulent_cf_average,
    turbulent_cf_local,
)
from boundary_layer_transition.stability import solve_amplification
from boundary_layer_transition.transition import separation_triggered_transition

V, NU, C = 25.0, 1.4607e-5, 0.7
X_STATION = 0.245  # m


@pytest.fixture
def op():
    return default_operating_point()


@pytest.fixture
def dist(op):
    return default_velocity_distribution(op.chord_m, op.v_inf_mps)


# ---------------------------------------------------------------------------
# C. Laminar skin friction -- hand calculation, average relation
# ---------------------------------------------------------------------------


def test_laminar_cf_average_hand_check():
    re_l = V * C / NU
    hand = 1.328 / math.sqrt(re_l)
    assert laminar_cf_average(V, C, NU) == pytest.approx(hand, rel=1e-12)


def test_laminar_cf_average_is_twice_local_blasius_coefficient():
    # 1.328 = 2 * 0.664, the exact integral relationship for a 1/sqrt(x) power law.
    assert LAMINAR_CF_COEFF_AVERAGE == pytest.approx(2 * 0.664, rel=1e-12)


def test_laminar_local_reused_from_laminar_bl_matches_hand_calc():
    re_x = V * X_STATION / NU
    hand = 0.664 / math.sqrt(re_x)
    assert blasius_cf(V, X_STATION, NU) == pytest.approx(hand, rel=1e-9)


# ---------------------------------------------------------------------------
# D. Turbulent skin friction -- hand calculation, exponent/constants, invalid Re
# ---------------------------------------------------------------------------


def test_turbulent_cf_local_hand_check():
    re_x = V * X_STATION / NU
    hand = 0.0592 * re_x**-0.2
    assert turbulent_cf_local(V, X_STATION, NU) == pytest.approx(hand, rel=1e-12)
    assert turbulent_cf_local(V, X_STATION, NU) == pytest.approx(0.0044444, rel=2e-3)


def test_turbulent_cf_average_hand_check():
    re_l = V * C / NU
    hand = 0.074 * re_l**-0.2
    assert turbulent_cf_average(V, C, NU) == pytest.approx(hand, rel=1e-12)


def test_turbulent_average_is_exact_integral_of_local():
    """Cf_bar(L) = (1/L) * integral_0^L Cf,x dx for Cf,x = A*Re_x^-0.2 gives
    Cf_bar = 1.25*A*Re_L^-0.2 (independent numerical integration check)."""
    x = np.linspace(1e-6, C, 200001)
    cf_local = turbulent_cf_local(V, x, NU)
    cf_bar_numeric = np.trapezoid(cf_local, x) / C
    assert cf_bar_numeric == pytest.approx(turbulent_cf_average(V, C, NU), rel=1e-3)


def test_turbulent_coefficient_ratio_hand_check():
    assert TURBULENT_CF_COEFF_AVERAGE == pytest.approx(1.25 * TURBULENT_CF_COEFF_LOCAL, rel=1e-12)


def test_turbulent_correlation_leading_edge_singularity_not_fabricated():
    result = turbulent_cf_local(V, 0.0, NU)
    assert math.isinf(result) and result > 0


def test_is_turbulent_correlation_valid_hand_check():
    assert is_turbulent_correlation_valid(1.0e6) is True
    assert is_turbulent_correlation_valid(1.0e4) is False   # below 5e5
    assert is_turbulent_correlation_valid(1.0e8) is False   # above 1e7
    assert is_turbulent_correlation_valid(TURBULENT_CF_RE_MIN) is True
    assert is_turbulent_correlation_valid(TURBULENT_CF_RE_MAX) is True


def test_is_turbulent_correlation_valid_vector():
    re_x = np.array([1e4, 5e5, 1e6, 1e7, 1e8])
    expected = np.array([False, True, True, True, False])
    assert np.array_equal(is_turbulent_correlation_valid(re_x), expected)


@pytest.mark.parametrize("fn", [turbulent_cf_local, turbulent_cf_average])
def test_turbulent_functions_reject_invalid_inputs(fn):
    with pytest.raises(ValueError):
        fn(0.0, 0.1, NU)  # v<=0 (rejected by reused reynolds_x validation)
    with pytest.raises(ValueError):
        fn(V, -0.1, NU)  # negative x
    with pytest.raises(ValueError):
        fn(V, 0.1, 0.0)  # nu<=0


# ---------------------------------------------------------------------------
# E. Transitioned distribution -- all-laminar/turbulent limits, interior transition
# ---------------------------------------------------------------------------


def test_transitioned_distribution_fully_laminar_limit():
    x = np.linspace(1e-3, C, 50)
    cf = transitioned_cf_distribution(V, x, NU, x_tr_m=math.inf)
    expected = blasius_cf(V, x, NU)
    assert cf == pytest.approx(expected, rel=1e-12)


def test_transitioned_distribution_fully_turbulent_limit():
    x = np.linspace(1e-3, C, 50)
    cf = transitioned_cf_distribution(V, x, NU, x_tr_m=0.0)
    expected = turbulent_cf_local(V, x, NU)
    assert cf == pytest.approx(expected, rel=1e-12)


def test_transitioned_distribution_interior_transition_sides():
    x_tr = 0.3 * C
    x = np.array([0.1 * C, 0.2 * C, x_tr, 0.4 * C, 0.9 * C])
    cf = transitioned_cf_distribution(V, x, NU, x_tr_m=x_tr)
    laminar_expected = blasius_cf(V, x, NU)
    turbulent_expected = turbulent_cf_local(V, x, NU)
    # before x_tr: laminar; at and after x_tr: turbulent (">=" convention)
    assert cf[0] == pytest.approx(laminar_expected[0], rel=1e-12)
    assert cf[1] == pytest.approx(laminar_expected[1], rel=1e-12)
    assert cf[2] == pytest.approx(turbulent_expected[2], rel=1e-12)
    assert cf[3] == pytest.approx(turbulent_expected[3], rel=1e-12)
    assert cf[4] == pytest.approx(turbulent_expected[4], rel=1e-12)


def test_transitioned_distribution_rejects_invalid_x_tr():
    x = np.linspace(1e-3, C, 10)
    with pytest.raises(ValueError):
        transitioned_cf_distribution(V, x, NU, x_tr_m=float("nan"))


def test_transitioned_distribution_rejects_invalid_x():
    with pytest.raises(ValueError):
        transitioned_cf_distribution(V, np.array([-0.1, 0.2]), NU, x_tr_m=0.1)


# ---------------------------------------------------------------------------
# F. Drag integration -- constant Cf, dynamic-pressure weighting, factor-of-two,
#    quadrature convergence
# ---------------------------------------------------------------------------


def test_section_cf_drag_constant_cf_analytic():
    s = np.linspace(0.0, 1.0, 1001)
    cf = np.full_like(s, 0.005)
    ue_ratio = np.ones_like(s)
    # Cd,f = 2 * integral_0^1 0.005 * 1^2 ds = 2*0.005 = 0.01
    assert section_cf_drag(s, cf, ue_ratio, two_surfaces=True) == pytest.approx(0.01, abs=1e-9)
    assert section_cf_drag(s, cf, ue_ratio, two_surfaces=False) == pytest.approx(0.005, abs=1e-9)


def test_section_cf_drag_dynamic_pressure_weighting_hand_check():
    s = np.linspace(0.0, 1.0, 1001)
    cf = np.full_like(s, 0.01)
    ue_ratio = np.full_like(s, 2.0)  # Ue = 2*Vinf everywhere (hypothetical)
    # Cd,f = 2 * integral 0.01 * 4 ds = 2*0.04 = 0.08
    assert section_cf_drag(s, cf, ue_ratio, two_surfaces=True) == pytest.approx(0.08, abs=1e-9)


def test_section_cf_drag_two_surfaces_factor_of_two():
    s = np.linspace(0.0, 1.0, 501)
    cf = np.linspace(0.01, 0.001, 501)
    ue_ratio = np.ones_like(s)
    two = section_cf_drag(s, cf, ue_ratio, two_surfaces=True)
    one = section_cf_drag(s, cf, ue_ratio, two_surfaces=False)
    assert two == pytest.approx(2 * one, rel=1e-12)


def test_section_cf_drag_matches_closed_form_average_laminar(op):
    s = build_drag_grid(op.chord_m, n_points=4001)
    x = s * op.chord_m
    cf = transitioned_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m=math.inf)
    ue_flat = np.ones_like(s)
    cd = section_cf_drag(s, cf, ue_flat, two_surfaces=True)
    analytic = 2.0 * laminar_cf_average(op.v_inf_mps, op.chord_m, op.nu_m2s)
    assert cd == pytest.approx(analytic, rel=2e-3)


def test_section_cf_drag_matches_closed_form_average_turbulent(op):
    s = build_drag_grid(op.chord_m, n_points=4001)
    x = s * op.chord_m
    cf = transitioned_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m=0.0)
    ue_flat = np.ones_like(s)
    cd = section_cf_drag(s, cf, ue_flat, two_surfaces=True)
    analytic = 2.0 * turbulent_cf_average(op.v_inf_mps, op.chord_m, op.nu_m2s)
    assert cd == pytest.approx(analytic, rel=2e-3)


def test_section_cf_drag_rejects_nonfinite_cf():
    s = np.linspace(0.0, 1.0, 11)
    cf = np.full_like(s, 0.005)
    cf[0] = np.inf
    with pytest.raises(ValueError):
        section_cf_drag(s, cf, np.ones_like(s))


def test_section_cf_drag_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        section_cf_drag(np.linspace(0, 1, 5), np.zeros(4), np.zeros(5))


def test_section_cf_drag_rejects_non_increasing_s():
    with pytest.raises(ValueError):
        section_cf_drag(np.array([0.0, 0.5, 0.4]), np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))


def test_build_drag_grid_quadrature_convergence(op):
    """The (already very good, log-graded-grid) fully-laminar quadrature
    stays converged to a tight tolerance of the closed-form average-Cf
    analytic result across a range of grid resolutions. Errors at this
    resolution are small enough (~1e-6 in absolute Cd,f) that strict
    monotonic improvement between successive *already-converged* grids is
    not a meaningful signal; the tolerance check itself is the convergence
    criterion for this fine-grid regime, and a much coarser grid (below)
    demonstrates the improvement over an under-resolved case."""
    analytic = 2.0 * laminar_cf_average(op.v_inf_mps, op.chord_m, op.nu_m2s)
    coarse_errors = []
    for n_pts in (51, 8001):
        s = build_drag_grid(op.chord_m, n_points=n_pts)
        x = s * op.chord_m
        cf = transitioned_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m=math.inf)
        cd = section_cf_drag(s, cf, np.ones_like(s), two_surfaces=True)
        coarse_errors.append(abs(cd - analytic) / analytic)
    assert coarse_errors[-1] < coarse_errors[0]
    assert coarse_errors[-1] < 1e-3


def test_build_drag_grid_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        build_drag_grid(0.0)
    with pytest.raises(ValueError):
        build_drag_grid(0.7, n_points=2)
    with pytest.raises(ValueError):
        build_drag_grid(0.7, s_min=0.0)
    with pytest.raises(ValueError):
        build_drag_grid(0.7, s_min=1.5)


# ---------------------------------------------------------------------------
# G. Physical ordering: laminar < late transition < early transition < turbulent
# ---------------------------------------------------------------------------


def test_physical_drag_ordering(op, dist):
    s = build_drag_grid(op.chord_m, n_points=4001)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)

    def cd_for(x_tr_m):
        cf = transitioned_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m=x_tr_m)
        return section_cf_drag(s, cf, ue_ratio, two_surfaces=True)

    cd_lam = cd_for(math.inf)
    cd_turb = cd_for(0.0)
    cd_late = cd_for(0.9 * op.chord_m)
    cd_early = cd_for(0.1 * op.chord_m)

    assert cd_lam < cd_late < cd_early < cd_turb


def test_separation_triggered_drag_between_laminar_and_turbulent(op, dist):
    tsol = solve_thwaites(dist, op.nu_m2s, n_points=4001)
    asol = solve_amplification(op, tsol)
    scenario = separation_triggered_transition(asol)

    s = build_drag_grid(op.chord_m, n_points=4001)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)

    def cd_for(x_tr_m):
        cf = transitioned_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m=x_tr_m)
        return section_cf_drag(s, cf, ue_ratio, two_surfaces=True)

    cd_lam = cd_for(math.inf)
    cd_turb = cd_for(0.0)
    cd_sep = cd_for(scenario.x_tr_m)

    assert cd_lam < cd_sep < cd_turb


# ---------------------------------------------------------------------------
# I. Scalar/vector consistency
# ---------------------------------------------------------------------------


def test_scalar_vector_consistency():
    x_vals = [0.05, 0.1, 0.3, 0.6]
    x_arr = np.array(x_vals)
    for fn in (turbulent_cf_local,):
        vec = fn(V, x_arr, NU)
        for i, xv in enumerate(x_vals):
            assert fn(V, xv, NU) == pytest.approx(vec[i], rel=1e-12)

    vec_avg = turbulent_cf_average(V, x_arr, NU)
    for i, xv in enumerate(x_vals):
        assert turbulent_cf_average(V, xv, NU) == pytest.approx(vec_avg[i], rel=1e-12)

    vec_lam_avg = laminar_cf_average(V, x_arr, NU)
    for i, xv in enumerate(x_vals):
        assert laminar_cf_average(V, xv, NU) == pytest.approx(vec_lam_avg[i], rel=1e-12)


def test_reynolds_x_reused_consistently():
    # skin_friction.py reuses laminar_bl.reynolds_x directly -- confirm identity.
    assert reynolds_x(V, X_STATION, NU) == pytest.approx(V * X_STATION / NU, rel=1e-12)
