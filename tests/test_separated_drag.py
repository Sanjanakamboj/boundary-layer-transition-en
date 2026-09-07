import math

import numpy as np
import pytest

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.laminar_bl import blasius_cf, solve_thwaites
from boundary_layer_transition.operating_point import default_operating_point
from boundary_layer_transition.separated_drag import (
    K_SEP_NOMINAL,
    P_SEP_NOMINAL,
    DragDecomposition,
    build_drag_decomposition,
    m5_cf_distribution,
    separated_drag_coefficient,
)
from boundary_layer_transition.separation_bubble import (
    NOMINAL_BUBBLE,
    OPEN_SEPARATION_PARAMS,
    evaluate_transition_mechanism,
)
from boundary_layer_transition.skin_friction import build_drag_grid, section_cf_drag
from boundary_layer_transition.stability import solve_amplification
from boundary_layer_transition.turbulent_bl import solve_turbulent_bl

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


# ---------------------------------------------------------------------------
# F. Separated-drag formula
# ---------------------------------------------------------------------------


def test_separated_drag_hand_check():
    l_sep, ue_ratio, k_sep, p = 0.03, 1.15, 0.5, 1.0
    hand = k_sep * l_sep**p * ue_ratio**2
    assert separated_drag_coefficient(l_sep, ue_ratio, k_sep, p) == pytest.approx(hand, rel=1e-12)


def test_separated_drag_hand_check_nonlinear_exponent():
    l_sep, ue_ratio, k_sep, p = 0.05, 1.0, 0.3, 1.5
    hand = k_sep * l_sep**p * ue_ratio**2
    assert separated_drag_coefficient(l_sep, ue_ratio, k_sep, p) == pytest.approx(hand, rel=1e-12)


def test_zero_separated_fraction_gives_zero_penalty():
    assert separated_drag_coefficient(0.0, 1.15, k_sep=5.0, p=2.0) == 0.0
    assert separated_drag_coefficient(0.0, 0.0, k_sep=100.0) == 0.0


def test_separated_drag_nonnegative():
    for l_sep in (0.0, 0.01, 0.1, 0.5, 1.0):
        for ue in (0.0, 0.5, 1.0, 1.5):
            assert separated_drag_coefficient(l_sep, ue) >= 0.0


def test_separated_drag_monotonic_in_l_sep():
    ue_ratio = 1.2
    l_vals = np.linspace(0.0, 0.5, 50)
    cd_vals = [separated_drag_coefficient(x, ue_ratio) for x in l_vals]
    assert np.all(np.diff(cd_vals) >= 0.0)


def test_dynamic_pressure_weighting_explicit():
    l_sep = 0.05
    cd_at_1x = separated_drag_coefficient(l_sep, 1.0)
    cd_at_2x = separated_drag_coefficient(l_sep, 2.0)
    assert cd_at_2x == pytest.approx(4.0 * cd_at_1x, rel=1e-12)


def test_nominal_constants_match_documented_defaults():
    assert K_SEP_NOMINAL == 0.5
    assert P_SEP_NOMINAL == 1.0


def test_separated_drag_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        separated_drag_coefficient(-0.1, 1.0)
    with pytest.raises(ValueError):
        separated_drag_coefficient(0.1, -1.0)
    with pytest.raises(ValueError):
        separated_drag_coefficient(0.1, 1.0, k_sep=-1.0)
    with pytest.raises(ValueError):
        separated_drag_coefficient(0.1, 1.0, p=0.0)
    with pytest.raises(ValueError):
        separated_drag_coefficient(0.1, 1.0, p=-1.0)


# ---------------------------------------------------------------------------
# G. Drag decomposition identity
# ---------------------------------------------------------------------------


def test_decomposition_identity_holds_exactly():
    d = build_drag_decomposition("test", cd_f_m5=0.003, l_sep_frac=0.02, ue_ratio_at_sep=1.1)
    assert d.cd_total_m5 == pytest.approx(d.cd_f_m5 + d.cd_sep_m5, abs=0.0) or d.cd_total_m5 == d.cd_f_m5 + d.cd_sep_m5


def test_decomposition_rejects_broken_identity():
    with pytest.raises(ValueError):
        DragDecomposition(
            label="bad", cd_f_m5=0.003, cd_sep_m5=0.002, cd_total_m5=0.999,
            l_sep_frac=0.02, k_sep=0.5, p=1.0,
        )


def test_decomposition_from_real_baseline_case(op, dist, tsol, asol):
    r = evaluate_transition_mechanism(asol, tsol, 9.0, NOMINAL_BUBBLE)
    ue_ratio_sep = float(dist.velocity_ratio(r.x_sep_lam_s))
    d = build_drag_decomposition("nominal", cd_f_m5=0.0025, l_sep_frac=r.bubble_length_s, ue_ratio_at_sep=ue_ratio_sep)
    assert d.cd_total_m5 == pytest.approx(d.cd_f_m5 + d.cd_sep_m5, rel=1e-12)


# ---------------------------------------------------------------------------
# H. Limiting cases: K_sep=0 -> friction-only
# ---------------------------------------------------------------------------


def test_k_sep_zero_gives_friction_only_total():
    d = build_drag_decomposition("friction-only", cd_f_m5=0.004, l_sep_frac=0.08, ue_ratio_at_sep=1.2, k_sep=0.0)
    assert d.cd_sep_m5 == 0.0
    assert d.cd_total_m5 == pytest.approx(d.cd_f_m5, rel=1e-12)


# ---------------------------------------------------------------------------
# m5_cf_distribution -- bubble region zeroed, laminar/turbulent sides correct
# ---------------------------------------------------------------------------


def test_m5_cf_laminar_side_matches_blasius():
    x = np.array([0.01, 0.05, 0.2])
    cf = m5_cf_distribution(V, x, NU, x_sep_m=0.3, x_reattach_m=0.35, tb_restart_cf_x_m=np.array([0.35, 0.7]),
                             tb_restart_cf=np.array([0.004, 0.003]))
    expected = blasius_cf(V, x, NU)
    assert cf == pytest.approx(expected, rel=1e-10)


def test_m5_cf_bubble_region_is_exactly_zero():
    x = np.array([0.30, 0.32, 0.34])
    cf = m5_cf_distribution(V, x, NU, x_sep_m=0.30, x_reattach_m=0.35, tb_restart_cf_x_m=np.array([0.35, 0.7]),
                             tb_restart_cf=np.array([0.004, 0.003]))
    assert np.all(cf == 0.0)


def test_m5_cf_turbulent_side_matches_restart_solution():
    x = np.array([0.36, 0.5])
    tb_x = np.array([0.35, 0.4, 0.5, 0.7])
    tb_cf = np.array([0.005, 0.0045, 0.004, 0.0035])
    cf = m5_cf_distribution(V, x, NU, x_sep_m=0.30, x_reattach_m=0.35, tb_restart_cf_x_m=tb_x, tb_restart_cf=tb_cf)
    expected = np.interp(x, tb_x, tb_cf)
    assert cf == pytest.approx(expected, rel=1e-10)


def test_m5_cf_open_separation_zero_beyond_sep():
    x = np.array([0.30, 0.5, 0.9])
    cf = m5_cf_distribution(V, x, NU, x_sep_m=0.30, x_reattach_m=None, tb_restart_cf_x_m=None, tb_restart_cf=None)
    assert cf[0] == 0.0 and cf[1] == 0.0 and cf[2] == 0.0


def test_m5_cf_requires_restart_arrays_when_reattached():
    x = np.array([0.5])
    with pytest.raises(ValueError):
        m5_cf_distribution(V, x, NU, x_sep_m=0.30, x_reattach_m=0.35, tb_restart_cf_x_m=None, tb_restart_cf=None)


def test_m5_cf_rejects_invalid_x():
    with pytest.raises(ValueError):
        m5_cf_distribution(
            V, np.array([-0.1]), NU, x_sep_m=0.3, x_reattach_m=None,
            tb_restart_cf_x_m=None, tb_restart_cf=None,
        )


# ---------------------------------------------------------------------------
# End-to-end: full M5 pipeline for the baseline nominal-bubble case
# ---------------------------------------------------------------------------


def test_end_to_end_nominal_bubble_drag_decomposition(op, dist, tsol, asol):
    r = evaluate_transition_mechanism(asol, tsol, 9.0, NOMINAL_BUBBLE)
    tb = solve_turbulent_bl(
        dist, tsol, op.v_inf_mps, op.nu_m2s, r.x_reattach_s * op.chord_m,
        theta_start_override_m=r.theta_reattach_m, h_start_override=r.h_reattach,
    )
    valid = np.where(tb.valid)[0]

    s = build_drag_grid(op.chord_m, n_points=2001)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)

    cf_m5 = m5_cf_distribution(
        op.v_inf_mps, x, op.nu_m2s, r.x_sep_lam_s * op.chord_m, r.x_reattach_s * op.chord_m,
        tb.x_m[valid], tb.cf[valid],
    )
    assert np.all(np.isfinite(cf_m5))
    cd_f_m5 = section_cf_drag(s, cf_m5, ue_ratio, two_surfaces=True)
    assert math.isfinite(cd_f_m5) and cd_f_m5 > 0.0

    ue_ratio_sep = float(dist.velocity_ratio(r.x_sep_lam_s))
    decomp = build_drag_decomposition("nominal", cd_f_m5, r.bubble_length_s, ue_ratio_sep)
    assert decomp.cd_total_m5 == pytest.approx(decomp.cd_f_m5 + decomp.cd_sep_m5, rel=1e-12)
    assert decomp.cd_sep_m5 > 0.0


def test_end_to_end_open_separation_drag_decomposition(op, dist, tsol, asol):
    r = evaluate_transition_mechanism(asol, tsol, 9.0, OPEN_SEPARATION_PARAMS)
    assert r.status.value == "OPEN_SEPARATION"

    s = build_drag_grid(op.chord_m, n_points=2001)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)

    cf_m5 = m5_cf_distribution(op.v_inf_mps, x, op.nu_m2s, r.x_sep_lam_s * op.chord_m, None, None, None)
    cd_f_m5 = section_cf_drag(s, cf_m5, ue_ratio, two_surfaces=True)

    l_sep_open = 1.0 - r.x_sep_lam_s
    ue_ratio_sep = float(dist.velocity_ratio(r.x_sep_lam_s))
    decomp = build_drag_decomposition("open", cd_f_m5, l_sep_open, ue_ratio_sep)
    assert decomp.cd_total_m5 == pytest.approx(decomp.cd_f_m5 + decomp.cd_sep_m5, rel=1e-12)
    # Open separation over more than half the chord should dominate the total.
    assert decomp.cd_sep_m5 > decomp.cd_f_m5
