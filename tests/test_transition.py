import numpy as np
import pytest

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.laminar_bl import solve_thwaites
from boundary_layer_transition.operating_point import GenericSailplaneOperatingPoint, default_operating_point
from boundary_layer_transition.stability import solve_amplification
from boundary_layer_transition.transition import (
    SeparationTriggeredScenario,
    TransitionStatus,
    interpolate_crossing,
    locate_transition_n_crit,
    locate_transition_n_crit_sweep,
    separation_triggered_transition,
    summarize_status_counts,
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
# A. N_crit crossing -- exact node, interpolated, none, ordering
# ---------------------------------------------------------------------------


def test_interpolate_crossing_exact_on_node():
    s = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    n = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # n hits exactly 2.0 at s=0.2 (a grid node)
    assert interpolate_crossing(s, n, 2.0) == pytest.approx(0.2, abs=1e-12)


def test_interpolate_crossing_between_nodes_hand_check():
    s = np.array([0.0, 0.1, 0.2, 0.3])
    n = np.array([0.0, 1.0, 3.0, 6.0])
    # Between s=0.1 (n=1) and s=0.2 (n=3): crossing at n=2 is
    # frac = (2-1)/(3-1) = 0.5 -> s = 0.1 + 0.5*(0.2-0.1) = 0.15
    assert interpolate_crossing(s, n, 2.0) == pytest.approx(0.15, abs=1e-12)


def test_interpolate_crossing_none_when_never_reached():
    s = np.linspace(0.0, 1.0, 11)
    n = np.linspace(0.0, 5.0, 11)
    assert interpolate_crossing(s, n, 100.0) is None


def test_interpolate_crossing_at_first_station():
    s = np.array([0.0, 0.5, 1.0])
    n = np.array([5.0, 6.0, 7.0])
    # n_crit already satisfied at the very first station
    assert interpolate_crossing(s, n, 3.0) == pytest.approx(0.0, abs=1e-12)


def test_interpolate_crossing_rejects_non_increasing_s():
    with pytest.raises(ValueError):
        interpolate_crossing(np.array([0.0, 0.2, 0.1]), np.array([0.0, 1.0, 2.0]), 1.0)


def test_interpolate_crossing_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        interpolate_crossing(np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]), 1.0)


def test_interpolate_crossing_rejects_nonfinite_n():
    with pytest.raises(ValueError):
        interpolate_crossing(np.array([0.0, 1.0]), np.array([0.0, np.nan]), 1.0)


def test_baseline_separation_before_all_sourced_ncrit(asol):
    """N_max ~= 3.84 for the M2 baseline; every sourced N_crit sensitivity
    value in this project (9, 12, 14) exceeds that, so the correct result
    is SEPARATION_BEFORE_N_CRIT for all of them -- never an invented
    crossing."""
    for n_crit in (9.0, 12.0, 14.0):
        result = locate_transition_n_crit(asol, n_crit)
        assert result.status == TransitionStatus.SEPARATION_BEFORE_N_CRIT
        assert result.x_tr_s is None
        assert result.x_tr_m is None
        assert result.n_max == pytest.approx(asol.n_max, rel=1e-12)


def test_crossing_before_separation_with_high_speed_case(op):
    """A physically reasonable higher-V_inf case (still comfortably
    incompressible) naturally accumulates enough N to cross N_crit=9
    before the M1 separation diagnostic -- found by searching the existing
    physical sensitivity space, not by tuning the M2 proxy constants."""
    op_fast = GenericSailplaneOperatingPoint(
        chord_m=op.chord_m, v_inf_mps=60.0, rho_kgm3=op.rho_kgm3,
        mu_pas=op.mu_pas, temperature_k=op.temperature_k, alpha_deg=op.alpha_deg,
    )
    assert op_fast.is_incompressible()  # M ~= 0.176, well under 0.3
    dist_fast = default_velocity_distribution(op_fast.chord_m, op_fast.v_inf_mps)
    tsol_fast = solve_thwaites(dist_fast, op_fast.nu_m2s, n_points=4001)
    asol_fast = solve_amplification(op_fast, tsol_fast)

    result = locate_transition_n_crit(asol_fast, 9.0)
    assert result.status == TransitionStatus.N_CRIT_CROSSING
    assert result.x_tr_s is not None
    assert result.x_tr_s < result.s_sep  # crossing strictly before separation
    assert result.x_tr_s == pytest.approx(0.4282, abs=2e-3)


def test_no_event_in_domain_synthetic():
    """Synthetic AmplificationSolution-independent check: a purely
    stable N-history (never crosses N_crit) with NO predicted separation
    (s_sep=None) must report NO_EVENT_IN_DOMAIN, not SEPARATION_BEFORE_N_CRIT."""
    from boundary_layer_transition.stability import AmplificationSolution

    n_pts = 101
    s = np.linspace(0.0, 1.0, n_pts)
    zeros = np.zeros(n_pts)
    valid = np.ones(n_pts, dtype=bool)
    fake = AmplificationSolution(
        s=s, x_m=s * 0.7, u_e_mps=np.full(n_pts, 25.0), theta_m=zeros, h=np.full(n_pts, 2.6),
        lam=zeros, re_theta=zeros, re_theta_crit=np.full(n_pts, 200.0),
        amplification_active=np.zeros(n_pts, dtype=bool), dn_ds=zeros, n=zeros,
        valid=valid, separated=np.zeros(n_pts, dtype=bool),
        onset_s=None, onset_x_m=None, s_sep=None, x_sep_m=None, n_max=0.0,
    )
    result = locate_transition_n_crit(fake, 9.0)
    assert result.status == TransitionStatus.NO_EVENT_IN_DOMAIN
    assert result.x_tr_s is None


def test_sweep_and_status_counts(asol):
    results = locate_transition_n_crit_sweep(asol, [9.0, 12.0, 14.0])
    assert len(results) == 3
    counts = summarize_status_counts(results)
    assert counts[TransitionStatus.SEPARATION_BEFORE_N_CRIT.value] == 3
    assert counts[TransitionStatus.N_CRIT_CROSSING.value] == 0
    assert counts[TransitionStatus.NO_EVENT_IN_DOMAIN.value] == 0


# ---------------------------------------------------------------------------
# B. status handling -- unambiguous, explicit
# ---------------------------------------------------------------------------


def test_status_values_are_distinct_strings():
    values = {s.value for s in TransitionStatus}
    assert values == {"N_CRIT_CROSSING", "SEPARATION_BEFORE_N_CRIT", "NO_EVENT_IN_DOMAIN"}


def test_locate_transition_rejects_invalid_ncrit(asol):
    with pytest.raises(ValueError):
        locate_transition_n_crit(asol, 0.0)
    with pytest.raises(ValueError):
        locate_transition_n_crit(asol, -1.0)
    with pytest.raises(ValueError):
        locate_transition_n_crit(asol, float("nan"))


# ---------------------------------------------------------------------------
# Separation-triggered scenario -- explicitly distinct from N_crit crossing
# ---------------------------------------------------------------------------


def test_separation_triggered_scenario_matches_m1_separation(asol):
    scenario = separation_triggered_transition(asol)
    assert isinstance(scenario, SeparationTriggeredScenario)
    assert scenario.x_tr_s == pytest.approx(asol.s_sep, rel=1e-12)
    assert scenario.x_tr_m == pytest.approx(asol.x_sep_m, rel=1e-12)
    assert "NOT an e" in scenario.label  # never silently relabeled as an N_crit result


def test_separation_triggered_scenario_none_when_no_separation():
    from boundary_layer_transition.stability import AmplificationSolution

    n_pts = 11
    s = np.linspace(0.0, 1.0, n_pts)
    zeros = np.zeros(n_pts)
    fake = AmplificationSolution(
        s=s, x_m=s * 0.7, u_e_mps=np.full(n_pts, 25.0), theta_m=zeros, h=np.full(n_pts, 2.6),
        lam=zeros, re_theta=zeros, re_theta_crit=np.full(n_pts, 200.0),
        amplification_active=np.zeros(n_pts, dtype=bool), dn_ds=zeros, n=zeros,
        valid=np.ones(n_pts, dtype=bool), separated=np.zeros(n_pts, dtype=bool),
        onset_s=None, onset_x_m=None, s_sep=None, x_sep_m=None, n_max=0.0,
    )
    assert separation_triggered_transition(fake) is None


# ---------------------------------------------------------------------------
# Grid convergence of the crossing location (high-speed crossing case)
# ---------------------------------------------------------------------------


def test_crossing_location_grid_convergence(op):
    op_fast = GenericSailplaneOperatingPoint(
        chord_m=op.chord_m, v_inf_mps=60.0, rho_kgm3=op.rho_kgm3,
        mu_pas=op.mu_pas, temperature_k=op.temperature_k, alpha_deg=op.alpha_deg,
    )
    dist_fast = default_velocity_distribution(op_fast.chord_m, op_fast.v_inf_mps)
    crossings = []
    for n_pts in (501, 1001, 2001, 4001):
        tsol_fast = solve_thwaites(dist_fast, op_fast.nu_m2s, n_points=n_pts)
        asol_fast = solve_amplification(op_fast, tsol_fast)
        result = locate_transition_n_crit(asol_fast, 9.0)
        assert result.status == TransitionStatus.N_CRIT_CROSSING
        crossings.append(result.x_tr_s)

    diffs = [abs(crossings[i + 1] - crossings[i]) for i in range(len(crossings) - 1)]
    assert diffs[-1] < diffs[0]
    assert diffs[-1] < 2e-3


# ---------------------------------------------------------------------------
# Scalar/vector and misc
# ---------------------------------------------------------------------------


def test_summarize_status_counts_all_categories_present():
    counts = summarize_status_counts([])
    assert set(counts.keys()) == {"N_CRIT_CROSSING", "SEPARATION_BEFORE_N_CRIT", "NO_EVENT_IN_DOMAIN"}
    assert all(v == 0 for v in counts.values())


def test_locate_transition_n_crit_sweep_rejects_bad_shape(asol):
    with pytest.raises(ValueError):
        locate_transition_n_crit_sweep(asol, [[9.0, 12.0]])


def test_interpolate_crossing_single_station_edge_case():
    assert interpolate_crossing(np.array([0.5]), np.array([10.0]), 5.0) == pytest.approx(0.5)
    assert interpolate_crossing(np.array([0.5]), np.array([1.0]), 5.0) is None
