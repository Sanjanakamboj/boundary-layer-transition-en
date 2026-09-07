import math

import pytest

from boundary_layer_transition.external_flow_sensitivity import PROFILES, with_override
from boundary_layer_transition.operating_point import GenericSailplaneOperatingPoint, default_operating_point
from boundary_layer_transition.robustness import run_chain
from boundary_layer_transition.separation_bubble import TransitionMechanism

BASELINE_PROFILE = next(p for p in PROFILES if p.label == "baseline")


@pytest.fixture(scope="module")
def op():
    return default_operating_point()


@pytest.fixture(scope="module")
def baseline_result(op):
    return run_chain(BASELINE_PROFILE, op)


# ---------------------------------------------------------------------------
# C. Full-chain regression against known M1-M5 baseline values
# ---------------------------------------------------------------------------


def test_baseline_chain_matches_known_m1_m5_headline_values(baseline_result):
    r = baseline_result
    assert r.re_c == pytest.approx(1.198e6, rel=1e-3)
    assert r.x_sep_lam_s == pytest.approx(0.4314, abs=2e-3)
    assert r.onset_s == pytest.approx(0.0885, abs=2e-3)
    assert r.n_max == pytest.approx(3.8429, abs=2e-3)
    assert r.mechanism == TransitionMechanism.SEPARATION_INDUCED.value
    assert r.cd_f == pytest.approx(0.002495, rel=2e-2)
    assert r.cd_sep == pytest.approx(0.019936, rel=2e-2)
    assert r.cd_total == pytest.approx(0.022430, rel=2e-2)


def test_baseline_n_crit_status_matches_m3(baseline_result):
    for nc in (9.0, 12.0, 14.0):
        assert baseline_result.n_crit_status[nc] == "SEPARATION_BEFORE_N_CRIT"


def test_baseline_chain_reproducible_across_calls(op):
    r1 = run_chain(BASELINE_PROFILE, op)
    r2 = run_chain(BASELINE_PROFILE, op)
    assert r1.x_sep_lam_s == r2.x_sep_lam_s
    assert r1.n_max == r2.n_max
    assert r1.cd_total == r2.cd_total


# ---------------------------------------------------------------------------
# D. Sensitivity behavior -- verified numerically, not assumed
# ---------------------------------------------------------------------------


def test_all_profiles_produce_finite_valid_results(op):
    for profile in PROFILES:
        r = run_chain(profile, op)
        assert r.re_c == pytest.approx(op.re_c, rel=1e-9)
        assert math.isfinite(r.mach)
        if r.x_sep_lam_s is not None:
            assert 0.0 < r.x_sep_lam_s < 1.0, profile.label
        if r.n_max is not None:
            assert r.n_max >= 0.0, profile.label
        if r.cd_total is not None:
            assert math.isfinite(r.cd_total) and r.cd_total >= 0.0, profile.label


def test_severe_and_mild_te_ratio_actually_differ_in_x_sep(op):
    """Numerically determine (not assume) the direction of the TE-ratio
    effect on laminar separation location."""
    severe = next(p for p in PROFILES if p.label == "te_ratio_severe")
    mild = next(p for p in PROFILES if p.label == "te_ratio_mild")
    r_severe = run_chain(severe, op)
    r_mild = run_chain(mild, op)
    assert r_severe.x_sep_lam_s is not None and r_mild.x_sep_lam_s is not None
    # The two must differ measurably -- the actual direction is reported
    # in scripts/final_robustness_study.py and DESIGN.md from these same
    # numbers, not hard-coded here.
    assert r_severe.x_sep_lam_s != pytest.approx(r_mild.x_sep_lam_s, abs=1e-4)


def test_peak_location_changes_separation_location_monotonically_with_station(op):
    """Numerically verify (not assume) that moving the peak later moves
    x_sep_lam later too, for this specific profile family; if the
    computed values disagreed, this test would fail rather than mask it."""
    early = run_chain(next(p for p in PROFILES if p.label == "peak_location_early"), op)
    baseline = run_chain(BASELINE_PROFILE, op)
    late = run_chain(next(p for p in PROFILES if p.label == "peak_location_late"), op)
    assert early.x_sep_lam_s < baseline.x_sep_lam_s < late.x_sep_lam_s


def test_baseline_mechanism_is_separation_induced_not_fabricated_crossing(baseline_result):
    """The baseline must never report an invented N_crit crossing."""
    assert baseline_result.mechanism == TransitionMechanism.SEPARATION_INDUCED.value
    assert baseline_result.n_crit_status[9.0] != "N_CRIT_CROSSING"


# ---------------------------------------------------------------------------
# E. Event/status stability -- valid categorical statuses only
# ---------------------------------------------------------------------------


VALID_MECHANISMS = {m.value for m in TransitionMechanism}


def test_all_profiles_report_valid_mechanism(op):
    for profile in PROFILES:
        r = run_chain(profile, op)
        assert r.mechanism in VALID_MECHANISMS, profile.label


def test_bubble_status_is_none_or_valid_string(op):
    from boundary_layer_transition.separation_bubble import BubbleStatus

    valid = {s.value for s in BubbleStatus}
    for profile in PROFILES:
        r = run_chain(profile, op)
        assert r.bubble_status is None or r.bubble_status in valid, profile.label


def test_turbulent_status_is_none_or_valid_string(op):
    from boundary_layer_transition.turbulent_bl import TurbulentBLStatus

    valid = {s.value for s in TurbulentBLStatus}
    for profile in PROFILES:
        r = run_chain(profile, op)
        assert r.turbulent_status is None or r.turbulent_status in valid, profile.label


def test_no_fabricated_reattachment_for_open_separation(op):
    """If any profile in the family produces OPEN_SEPARATION, x_reattach_s
    must be exactly None -- never a fabricated number."""
    for profile in PROFILES:
        r = run_chain(profile, op)
        if r.bubble_status == "OPEN_SEPARATION":
            assert r.x_reattach_s is None, profile.label


# ---------------------------------------------------------------------------
# F. Deterministic sensitivity ranking -- no unordered set iteration
# ---------------------------------------------------------------------------


def test_profile_order_is_deterministic_list_not_set():
    labels_1 = [p.label for p in PROFILES]
    labels_2 = [p.label for p in PROFILES]
    assert labels_1 == labels_2
    assert isinstance(PROFILES, list)


def test_sensitivity_ranking_repeatable(op):
    def compute_ranking():
        results = {p.label: run_chain(p, op) for p in PROFILES}
        base_cd = results["baseline"].cd_total
        ranking = []
        for label, r in results.items():
            if r.cd_total is None:
                continue
            pct = 100.0 * (r.cd_total - base_cd) / base_cd
            ranking.append((label, pct))
        ranking.sort(key=lambda item: abs(item[1]), reverse=True)
        return ranking

    ranking_1 = compute_ranking()
    ranking_2 = compute_ranking()
    assert ranking_1 == ranking_2


# ---------------------------------------------------------------------------
# G. Invalid-state handling
# ---------------------------------------------------------------------------


def test_run_chain_rejects_invalid_profile(op):
    bad = with_override(BASELINE_PROFILE, u_te_ratio=2.0)  # >= u_peak_ratio
    with pytest.raises(ValueError):
        run_chain(bad, op)


def test_run_chain_attached_crossing_branch_at_high_re():
    """Regression test for the ATTACHED_E_N_CROSSING code path in
    run_chain(), which the V=25 baseline never exercises (a real bug --
    a wrong solve_turbulent_bl keyword argument, n_points instead of
    n_report -- was found and fixed here during Milestone 6 development,
    caught only because this path was actually run)."""
    base = default_operating_point()
    op_fast = GenericSailplaneOperatingPoint(
        chord_m=base.chord_m, v_inf_mps=60.0, rho_kgm3=base.rho_kgm3,
        mu_pas=base.mu_pas, temperature_k=base.temperature_k, alpha_deg=base.alpha_deg,
    )
    r = run_chain(BASELINE_PROFILE, op_fast)
    assert r.mechanism == TransitionMechanism.ATTACHED_E_N_CROSSING.value
    assert r.x_tr_s is not None
    assert r.cd_f is not None and r.cd_f > 0.0
    assert r.cd_sep == 0.0
    assert r.turbulent_status is not None


# ---------------------------------------------------------------------------
# H. Independent control cases
# ---------------------------------------------------------------------------


def test_re_c_independent_of_profile_shape(op):
    """Re_c depends only on (rho, V_inf, c, mu), never on the U_e shape --
    verify this holds across the whole family (a control case)."""
    hand_re_c = op.rho_kgm3 * op.v_inf_mps * op.chord_m / op.mu_pas
    for profile in PROFILES:
        r = run_chain(profile, op)
        assert r.re_c == pytest.approx(hand_re_c, rel=1e-9), profile.label


def test_mach_independent_of_profile_shape(op):
    """Mach depends only on V_inf and speed of sound, never on U_e shape."""
    for profile in PROFILES:
        r = run_chain(profile, op)
        assert r.mach == pytest.approx(op.mach, rel=1e-9), profile.label


def test_x_sep_le_x_tr_le_x_reattach_ordering_when_applicable(op):
    for profile in PROFILES:
        r = run_chain(profile, op)
        is_sep_induced = r.mechanism == TransitionMechanism.SEPARATION_INDUCED.value
        if r.x_sep_lam_s is not None and r.x_tr_s is not None and is_sep_induced:
            assert r.x_sep_lam_s <= r.x_tr_s, profile.label
        if r.x_tr_s is not None and r.x_reattach_s is not None:
            assert r.x_tr_s <= r.x_reattach_s, profile.label


def test_grid_refinement_does_not_change_baseline_classification(op):
    statuses = set()
    for n_points in (1001, 2001, 4001):
        r = run_chain(BASELINE_PROFILE, op, n_points=n_points)
        statuses.add(r.mechanism)
    assert statuses == {TransitionMechanism.SEPARATION_INDUCED.value}


def test_x_sep_stable_under_grid_refinement(op):
    values = [run_chain(BASELINE_PROFILE, op, n_points=n).x_sep_lam_s for n in (1001, 2001, 4001, 8001)]
    diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    assert diffs[-1] < diffs[0]
    assert diffs[-1] < 1e-3
