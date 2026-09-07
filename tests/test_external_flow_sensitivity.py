import numpy as np
import pytest

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.external_flow_sensitivity import (
    BASELINE_S_PEAK,
    BASELINE_U_PEAK_RATIO,
    BASELINE_U_TE_RATIO,
    PROFILES,
    SensitivityProfile,
    build_distribution,
    with_override,
)

C, V = 0.7, 25.0


# ---------------------------------------------------------------------------
# A. Profile construction
# ---------------------------------------------------------------------------


def test_baseline_profile_matches_m1_defaults_exactly():
    baseline = next(p for p in PROFILES if p.label == "baseline")
    assert baseline.s_peak == BASELINE_S_PEAK == 0.35
    assert baseline.u_peak_ratio == BASELINE_U_PEAK_RATIO == 1.18
    assert baseline.u_te_ratio == BASELINE_U_TE_RATIO == 0.55


def test_baseline_distribution_bit_for_bit_matches_m1():
    baseline = next(p for p in PROFILES if p.label == "baseline")
    dist_sensitivity = build_distribution(baseline, C, V)
    dist_m1 = default_velocity_distribution(C, V)

    s = np.linspace(0.0, 1.0, 2001)
    assert np.array_equal(dist_sensitivity.velocity_ratio(s), dist_m1.velocity_ratio(s))
    assert np.array_equal(dist_sensitivity.d_velocity_ratio_ds(s), dist_m1.d_velocity_ratio_ds(s))
    x = s * C
    assert np.array_equal(dist_sensitivity.u_e(x), dist_m1.u_e(x))
    assert np.array_equal(dist_sensitivity.du_e_dx(x), dist_m1.du_e_dx(x))


def test_peak_ratio_profiles_hit_exact_requested_value():
    for label, expected in (("peak_ratio_mild", 1.10), ("peak_ratio_strong", 1.26)):
        profile = next(p for p in PROFILES if p.label == label)
        assert profile.u_peak_ratio == expected
        dist = build_distribution(profile, C, V)
        assert dist.velocity_ratio(profile.s_peak) == pytest.approx(expected, abs=1e-12)


def test_peak_location_profiles_hit_exact_requested_station():
    for label, expected in (("peak_location_early", 0.25), ("peak_location_late", 0.45)):
        profile = next(p for p in PROFILES if p.label == label)
        assert profile.s_peak == expected
        dist = build_distribution(profile, C, V)
        s = np.linspace(0.0, 1.0, 20001)
        ratio = dist.velocity_ratio(s)
        assert s[np.argmax(ratio)] == pytest.approx(expected, abs=1e-3)


def test_te_ratio_profiles_hit_exact_requested_value():
    for label, expected in (("te_ratio_severe", 0.45), ("te_ratio_mild", 0.70)):
        profile = next(p for p in PROFILES if p.label == label)
        assert profile.u_te_ratio == expected
        dist = build_distribution(profile, C, V)
        assert dist.velocity_ratio(1.0) == pytest.approx(expected, abs=1e-12)


def test_all_profiles_positive_everywhere():
    s = np.linspace(0.0, 1.0, 2001)
    for profile in PROFILES:
        dist = build_distribution(profile, C, V)
        assert np.all(dist.velocity_ratio(s) > 0.0), profile.label


def test_all_profiles_finite_everywhere():
    s = np.linspace(0.0, 1.0, 2001)
    for profile in PROFILES:
        dist = build_distribution(profile, C, V)
        assert np.all(np.isfinite(dist.velocity_ratio(s))), profile.label
        assert np.all(np.isfinite(dist.d_velocity_ratio_ds(s))), profile.label


def test_with_override_preserves_unspecified_fields():
    baseline = next(p for p in PROFILES if p.label == "baseline")
    modified = with_override(baseline, u_peak_ratio=1.5)
    assert modified.u_peak_ratio == 1.5
    assert modified.s_peak == baseline.s_peak
    assert modified.u_te_ratio == baseline.u_te_ratio
    assert modified.label == baseline.label


def test_invalid_combination_rejected_te_exceeds_peak():
    bad = SensitivityProfile("bad", s_peak=0.35, u_peak_ratio=1.1, u_te_ratio=1.5)
    with pytest.raises(ValueError):
        build_distribution(bad, C, V)


def test_invalid_combination_rejected_peak_ratio_not_above_one():
    bad = SensitivityProfile("bad", u_peak_ratio=0.9)
    with pytest.raises(ValueError):
        build_distribution(bad, C, V)


def test_invalid_combination_rejected_s_peak_out_of_range():
    bad = SensitivityProfile("bad", s_peak=1.5)
    with pytest.raises(ValueError):
        build_distribution(bad, C, V)


def test_profiles_family_is_seven_and_labels_unique():
    assert len(PROFILES) == 7
    labels = [p.label for p in PROFILES]
    assert len(labels) == len(set(labels))


# ---------------------------------------------------------------------------
# B. Derivative -- analytic vs finite-difference, sign, continuity
# ---------------------------------------------------------------------------


def test_derivative_analytic_matches_finite_difference_for_every_profile():
    x = np.linspace(1e-4, C - 1e-4, 251)
    for profile in PROFILES:
        dist = build_distribution(profile, C, V)
        analytic = dist.du_e_dx(x)
        numerical = dist.du_e_dx_numerical(x, h_m=1e-6 * C)
        assert analytic == pytest.approx(numerical, abs=1e-3, rel=1e-4), profile.label


def test_derivative_sign_favorable_before_adverse_after_peak():
    for profile in PROFILES:
        dist = build_distribution(profile, C, V)
        x_fav = 0.5 * profile.s_peak * C
        x_adv = 0.5 * (1.0 + profile.s_peak) * C
        assert dist.du_e_dx(x_fav) > 0.0, profile.label
        assert dist.du_e_dx(x_adv) < 0.0, profile.label


def test_derivative_zero_at_peak_for_every_profile():
    for profile in PROFILES:
        dist = build_distribution(profile, C, V)
        x_peak = profile.s_peak * C
        assert dist.du_e_dx(x_peak) == pytest.approx(0.0, abs=1e-8), profile.label


def test_continuity_at_leading_edge_and_trailing_edge():
    eps = 1e-6
    for profile in PROFILES:
        dist = build_distribution(profile, C, V)
        # Leading-edge value must equal 1.0 (U_e(0) = V_inf) for every profile.
        assert dist.velocity_ratio(0.0) == pytest.approx(1.0, abs=1e-12), profile.label
        # No jump approaching s=0 or s=1 from the interior.
        near0 = dist.velocity_ratio(eps)
        near1 = dist.velocity_ratio(1.0 - eps)
        assert near0 == pytest.approx(1.0, abs=1e-3), profile.label
        assert near1 == pytest.approx(profile.u_te_ratio, abs=1e-3), profile.label


def test_continuity_at_peak_junction():
    """Both one-sided values and derivatives must agree at s=s_peak."""
    eps = 1e-7
    for profile in PROFILES:
        dist = build_distribution(profile, C, V)
        s_peak = profile.s_peak
        left = dist.velocity_ratio(max(s_peak - eps, 0.0))
        right = dist.velocity_ratio(min(s_peak + eps, 1.0))
        assert left == pytest.approx(right, abs=1e-3), profile.label
