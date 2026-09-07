import numpy as np
import pytest

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.laminar_bl import solve_thwaites
from boundary_layer_transition.operating_point import GenericSailplaneOperatingPoint, default_operating_point
from boundary_layer_transition.separation_bubble import (
    LONG_BUBBLE,
    NOMINAL_BUBBLE,
    OPEN_SEPARATION_PARAMS,
    SHORT_BUBBLE,
    BubbleParameters,
    BubbleStatus,
    TransitionMechanism,
    evaluate_bubble_at_separation,
    evaluate_transition_mechanism,
)
from boundary_layer_transition.stability import solve_amplification
from boundary_layer_transition.turbulent_bl import solve_turbulent_bl


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
# A. Event ordering
# ---------------------------------------------------------------------------


def test_reattaching_bubble_event_ordering(asol, tsol):
    for params in (SHORT_BUBBLE, NOMINAL_BUBBLE, LONG_BUBBLE):
        r = evaluate_transition_mechanism(asol, tsol, 9.0, params)
        assert r.status == BubbleStatus.TURBULENT_REATTACHMENT
        assert r.x_sep_lam_s < r.x_tr_sep_s < r.x_reattach_s
        assert r.bubble_length_s == pytest.approx(r.x_reattach_s - r.x_sep_lam_s, rel=1e-12)


def test_impossible_ordering_rejected_negative_dx_tr_sep():
    with pytest.raises(ValueError):
        BubbleParameters("bad", dx_tr_sep_frac=-0.01, dx_reattach_frac=0.02, k_theta=3.0, h_reattach=1.5)


def test_impossible_ordering_rejected_nonpositive_dx_reattach():
    with pytest.raises(ValueError):
        BubbleParameters("bad", dx_tr_sep_frac=0.01, dx_reattach_frac=0.0, k_theta=3.0, h_reattach=1.5)
    with pytest.raises(ValueError):
        BubbleParameters("bad", dx_tr_sep_frac=0.01, dx_reattach_frac=-0.02, k_theta=3.0, h_reattach=1.5)


def test_reattachment_requires_k_theta_and_h_reattach():
    with pytest.raises(ValueError):
        BubbleParameters("bad", dx_tr_sep_frac=0.01, dx_reattach_frac=0.02, k_theta=None, h_reattach=1.5)
    with pytest.raises(ValueError):
        BubbleParameters("bad", dx_tr_sep_frac=0.01, dx_reattach_frac=0.02, k_theta=3.0, h_reattach=None)


def test_open_separation_forbids_k_theta_and_h_reattach():
    with pytest.raises(ValueError):
        BubbleParameters("bad", dx_tr_sep_frac=0.01, dx_reattach_frac=None, k_theta=3.0, h_reattach=None)
    with pytest.raises(ValueError):
        BubbleParameters("bad", dx_tr_sep_frac=0.01, dx_reattach_frac=None, k_theta=None, h_reattach=1.5)


# ---------------------------------------------------------------------------
# B. Open separation
# ---------------------------------------------------------------------------


def test_open_separation_no_fake_reattachment(asol, tsol):
    r = evaluate_transition_mechanism(asol, tsol, 9.0, OPEN_SEPARATION_PARAMS)
    assert r.status == BubbleStatus.OPEN_SEPARATION
    assert r.x_reattach_s is None
    assert r.bubble_length_s is None
    assert r.theta_reattach_m is None
    assert r.h_reattach is None


def test_reattachment_beyond_trailing_edge_becomes_open_separation(asol, tsol):
    huge_bubble = BubbleParameters("huge", dx_tr_sep_frac=0.05, dx_reattach_frac=0.6, k_theta=5.0, h_reattach=1.7)
    r = evaluate_transition_mechanism(asol, tsol, 9.0, huge_bubble)
    assert r.status == BubbleStatus.OPEN_SEPARATION
    assert r.x_reattach_s is None


# ---------------------------------------------------------------------------
# C. Attached e^N crossing before laminar separation -- LSB bypassed
# ---------------------------------------------------------------------------


def test_attached_crossing_bypasses_bubble_model(op):
    op_fast = GenericSailplaneOperatingPoint(
        chord_m=op.chord_m, v_inf_mps=60.0, rho_kgm3=op.rho_kgm3,
        mu_pas=op.mu_pas, temperature_k=op.temperature_k, alpha_deg=op.alpha_deg,
    )
    dist_fast = default_velocity_distribution(op_fast.chord_m, op_fast.v_inf_mps)
    tsol_fast = solve_thwaites(dist_fast, op_fast.nu_m2s, n_points=4001)
    asol_fast = solve_amplification(op_fast, tsol_fast)

    r = evaluate_transition_mechanism(asol_fast, tsol_fast, 9.0, NOMINAL_BUBBLE)
    assert r.mechanism == TransitionMechanism.ATTACHED_E_N_CROSSING
    assert r.status == BubbleStatus.NO_LAMINAR_SEPARATION
    assert r.x_tr_sep_s is None
    assert r.x_reattach_s is None
    assert r.params is None


# ---------------------------------------------------------------------------
# D. Baseline separation-before-N_crit -- unchanged M1/M2/M3 status
# ---------------------------------------------------------------------------


def test_baseline_uses_separation_induced_mechanism(op, asol, tsol):
    r = evaluate_transition_mechanism(asol, tsol, 9.0, NOMINAL_BUBBLE)
    assert r.mechanism == TransitionMechanism.SEPARATION_INDUCED
    assert r.x_sep_lam_s == pytest.approx(asol.s_sep, rel=1e-12)
    assert r.x_sep_lam_s == pytest.approx(0.4314, abs=2e-3)


def test_baseline_m2_n_max_and_status_unchanged(asol):
    # Independent re-check that M2/M3 baseline facts (frozen) still hold:
    # this module must never manufacture an N_crit crossing for the baseline.
    from boundary_layer_transition.transition import TransitionStatus, locate_transition_n_crit

    assert asol.n_max == pytest.approx(3.8429, abs=1e-3)
    for n_crit in (9.0, 12.0, 14.0):
        result = locate_transition_n_crit(asol, n_crit)
        assert result.status == TransitionStatus.SEPARATION_BEFORE_N_CRIT


# ---------------------------------------------------------------------------
# E. Restart initialization
# ---------------------------------------------------------------------------


def test_theta_reattach_matches_closure_exactly(asol, tsol):
    r = evaluate_transition_mechanism(asol, tsol, 9.0, NOMINAL_BUBBLE)
    expected = NOMINAL_BUBBLE.k_theta * r.theta_sep_m
    assert r.theta_reattach_m == pytest.approx(expected, rel=1e-12)


def test_h_reattach_matches_declared_value(asol, tsol):
    for params in (SHORT_BUBBLE, NOMINAL_BUBBLE, LONG_BUBBLE):
        r = evaluate_transition_mechanism(asol, tsol, 9.0, params)
        assert r.h_reattach == params.h_reattach


def test_theta_sep_matches_independent_interpolation(asol, tsol):
    r = evaluate_transition_mechanism(asol, tsol, 9.0, NOMINAL_BUBBLE)
    hand = float(np.interp(r.x_sep_lam_s * tsol.x_m[-1], tsol.x_m, tsol.theta_m))
    assert r.theta_sep_m == pytest.approx(hand, rel=1e-12)


def test_no_invalid_post_separation_laminar_state_reused(op, dist, tsol, asol):
    """theta_sep must come only from the M1 array interpolated exactly at
    x_sep (never beyond it); verify by checking evaluate_bubble_at_separation
    rejects an out-of-[0,1) station and that theta_sep is continuous with
    the M1 theta immediately upstream of separation."""
    r = evaluate_transition_mechanism(asol, tsol, 9.0, NOMINAL_BUBBLE)
    i_near = int(np.argmin(np.abs(tsol.s - r.x_sep_lam_s)))
    # theta_sep should be close to the M1 theta at the nearest grid station
    # (both represent the same physical station to within grid resolution).
    assert r.theta_sep_m == pytest.approx(tsol.theta_m[i_near], rel=5e-3)


def test_evaluate_bubble_at_separation_rejects_invalid_station(tsol):
    with pytest.raises(ValueError):
        evaluate_bubble_at_separation(tsol, -0.1, NOMINAL_BUBBLE)
    with pytest.raises(ValueError):
        evaluate_bubble_at_separation(tsol, 1.0, NOMINAL_BUBBLE)
    with pytest.raises(ValueError):
        evaluate_bubble_at_separation(tsol, float("nan"), NOMINAL_BUBBLE)


def test_turbulent_restart_uses_override_not_m1_laminar_interp(op, dist, tsol, asol):
    """The restarted M4 solver must use the bubble's (theta_reattach,
    H_reattach) state, not re-derive from M1 laminar theta at x_reattach
    (which is beyond the M1 valid domain and would be rejected there)."""
    r = evaluate_transition_mechanism(asol, tsol, 9.0, NOMINAL_BUBBLE)
    assert r.x_reattach_s > tsol.s_sep  # confirms x_reattach truly is past the M1-valid domain

    tb = solve_turbulent_bl(
        dist, tsol, op.v_inf_mps, op.nu_m2s, r.x_reattach_s * op.chord_m,
        theta_start_override_m=r.theta_reattach_m, h_start_override=r.h_reattach,
    )
    assert tb.theta_m[0] == pytest.approx(r.theta_reattach_m, rel=1e-9)
    assert tb.h[0] == pytest.approx(r.h_reattach, rel=1e-6)


# ---------------------------------------------------------------------------
# H. Limiting cases
# ---------------------------------------------------------------------------


def test_zero_dx_tr_sep_places_transition_at_separation(asol, tsol):
    zero_gap = BubbleParameters("zero-gap", dx_tr_sep_frac=0.0, dx_reattach_frac=0.02, k_theta=3.0, h_reattach=1.5)
    r = evaluate_transition_mechanism(asol, tsol, 9.0, zero_gap)
    assert r.x_tr_sep_s == pytest.approx(r.x_sep_lam_s, abs=1e-12)


# ---------------------------------------------------------------------------
# Scalar/vector-style consistency & misc
# ---------------------------------------------------------------------------


def test_bubble_length_ordering_short_lt_nominal_lt_long(asol, tsol):
    r_short = evaluate_transition_mechanism(asol, tsol, 9.0, SHORT_BUBBLE)
    r_nom = evaluate_transition_mechanism(asol, tsol, 9.0, NOMINAL_BUBBLE)
    r_long = evaluate_transition_mechanism(asol, tsol, 9.0, LONG_BUBBLE)
    assert r_short.bubble_length_s < r_nom.bubble_length_s < r_long.bubble_length_s


def test_message_is_nonempty_string_for_all_statuses(asol, tsol):
    for params in (SHORT_BUBBLE, NOMINAL_BUBBLE, LONG_BUBBLE, OPEN_SEPARATION_PARAMS):
        r = evaluate_transition_mechanism(asol, tsol, 9.0, params)
        assert isinstance(r.message, str) and len(r.message) > 0
