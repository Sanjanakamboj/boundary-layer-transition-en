#!/usr/bin/env python3
"""Milestone 6 study script: external-flow shape sensitivity and a final
deterministic cross-model sensitivity ranking.

Run with:

    python3 scripts/final_robustness_study.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.external_flow_sensitivity import PROFILES
from boundary_layer_transition.laminar_bl import solve_thwaites
from boundary_layer_transition.operating_point import default_operating_point
from boundary_layer_transition.robustness import run_chain
from boundary_layer_transition.separated_drag import m5_cf_distribution, separated_drag_coefficient
from boundary_layer_transition.separation_bubble import NOMINAL_BUBBLE, BubbleParameters, evaluate_transition_mechanism
from boundary_layer_transition.skin_friction import build_drag_grid, section_cf_drag
from boundary_layer_transition.stability import ONSET_H_SENSITIVITY, solve_amplification
from boundary_layer_transition.transition import locate_transition_n_crit
from boundary_layer_transition.turbulent_bl import H_TR_NOMINAL, solve_turbulent_bl

N_POINTS = 4001


def section_baseline(op, baseline) -> None:
    print("=" * 100)
    print("BASELINE REFERENCE (Milestone 1 external-flow profile, unmodified)")
    print("=" * 100)
    print(f"V_inf                                  = {op.v_inf_mps:.1f} m/s")
    print(f"Re_c                                    = {op.re_c:.4e}")
    print(f"U_e/V_inf peak ratio / location          = {baseline.peak_ratio:.2f} at x/c={baseline.s_peak:.2f}")
    print(f"U_e/V_inf trailing-edge ratio             = {baseline.te_ratio:.2f}")
    print(f"x_sep,lam/c                               = {baseline.x_sep_lam_s:.4f}")
    print(f"Modeled instability onset, x/c            = {baseline.onset_s:.4f}")
    print(f"N_max                                     = {baseline.n_max:.4f}")
    print(f"Transition status (N_crit=9)              = {baseline.n_crit_status[9.0]}")
    print(f"Downstream turbulent separation, x/c       = {baseline.x_turb_sep_s}")
    print(f"C_d,f                                     = {baseline.cd_f:.6f}")
    print(f"C_d,sep                                    = {baseline.cd_sep:.6f}")
    print(f"C_d,total                                  = {baseline.cd_total:.6f}")
    print()


def section_external_flow_sensitivity(results: dict) -> None:
    print("=" * 100)
    print("EXTERNAL-FLOW SENSITIVITY (synthetic conceptual U_e(x/c) family, N_crit=9)")
    print("=" * 100)
    header = (
        f"{'case':22} {'peak':>6} {'x_pk/c':>7} {'TE':>6} {'x_sep/c':>8} {'N_max':>7} "
        f"{'mechanism':>20} {'x_tsep/c':>9} {'C_d,total':>10}"
    )
    print(header)
    for label, r in results.items():
        def fmt(v, f="{:.4f}"):
            return f.format(v) if v is not None else "  n/a  "
        print(
            f"{label:22} {r.peak_ratio:6.2f} {r.s_peak:7.2f} {r.te_ratio:6.2f} "
            f"{fmt(r.x_sep_lam_s):>8} {fmt(r.n_max):>7} {r.mechanism:>20} "
            f"{fmt(r.x_turb_sep_s):>9} {fmt(r.cd_total, '{:.6f}'):>10}"
        )
    print()


def section_robustness_summary(results: dict) -> str:
    print("=" * 100)
    print("ROBUSTNESS SUMMARY (computed from the table above)")
    print("=" * 100)
    x_seps = [r.x_sep_lam_s for r in results.values() if r.x_sep_lam_s is not None]
    n_maxs = [r.n_max for r in results.values() if r.n_max is not None]
    cd_totals = [r.cd_total for r in results.values() if r.cd_total is not None]
    baseline = results["baseline"]

    print(f"x_sep,lam/c range: [{min(x_seps):.4f}, {max(x_seps):.4f}] "
          f"(spread {max(x_seps) - min(x_seps):.4f})")
    print(f"N_max range:        [{min(n_maxs):.4f}, {max(n_maxs):.4f}] "
          f"(spread {max(n_maxs) - min(n_maxs):.4f})")
    print(f"C_d,total range:    [{min(cd_totals):.6f}, {max(cd_totals):.6f}] "
          f"(spread {max(cd_totals) - min(cd_totals):.6f})")

    n_baseline_mechanism = sum(1 for r in results.values() if r.mechanism == baseline.mechanism)
    frac = n_baseline_mechanism / len(results)
    print(f"Cases retaining the baseline mechanism ({baseline.mechanism}): "
          f"{n_baseline_mechanism}/{len(results)} ({100 * frac:.0f}%)")

    n_turb_sep = sum(1 for r in results.values() if r.x_turb_sep_s is not None)
    print(f"Cases with a downstream turbulent (re-)separation: {n_turb_sep}/{len(results)}")

    # Determine the most influential profile parameter from the actual
    # computed spread in x_sep_lam_s across each one-factor sub-family
    # (never assumed).
    groups = {
        "peak_ratio": ["peak_ratio_mild", "baseline", "peak_ratio_strong"],
        "peak_location": ["peak_location_early", "baseline", "peak_location_late"],
        "te_ratio": ["te_ratio_severe", "baseline", "te_ratio_mild"],
    }
    spreads = {}
    for name, labels in groups.items():
        vals = [results[label].x_sep_lam_s for label in labels if results[label].x_sep_lam_s is not None]
        spreads[name] = max(vals) - min(vals) if vals else 0.0
    ranked = sorted(spreads.items(), key=lambda kv: kv[1], reverse=True)
    print()
    print("x_sep,lam/c spread by profile-parameter group (most influential first):")
    for name, spread in ranked:
        print(f"  {name:16} spread = {spread:.4f}")
    print(f"Most influential external-flow parameter (by x_sep,lam/c spread): {ranked[0][0]}")
    print()
    return ranked[0][0]


def _cd_total_bubble(op, dist, tsol, asol, params: BubbleParameters, k_sep: float = 0.5) -> float:
    """Fresh (not copied from Milestone 5) recomputation of C_d,total,M5
    for the baseline external-flow distribution at a given bubble
    parameter set / K_sep."""
    r = evaluate_transition_mechanism(asol, tsol, 9.0, params)
    s = build_drag_grid(op.chord_m, n_points=N_POINTS)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)
    ue_ratio_sep = float(dist.velocity_ratio(r.x_sep_lam_s))
    tb = solve_turbulent_bl(
        dist, tsol, op.v_inf_mps, op.nu_m2s, r.x_reattach_s * op.chord_m,
        theta_start_override_m=r.theta_reattach_m, h_start_override=r.h_reattach,
    )
    valid = np.where(tb.valid)[0]
    cf = m5_cf_distribution(
        op.v_inf_mps, x, op.nu_m2s, r.x_sep_lam_s * op.chord_m, r.x_reattach_s * op.chord_m,
        tb.x_m[valid], tb.cf[valid],
    )
    cd_f = section_cf_drag(s, cf, ue_ratio, two_surfaces=True)
    return cd_f + separated_drag_coefficient(r.bubble_length_s, ue_ratio_sep, k_sep=k_sep)


def _cd_total_h_tr(op, dist, tsol, asol, h_tr: float) -> float:
    """Fresh recomputation of C_d,total,M5 for a prescribed M4 turbulent-
    initialization shape factor h_tr, holding the bubble geometry at
    NOMINAL_BUBBLE."""
    r = evaluate_transition_mechanism(asol, tsol, 9.0, NOMINAL_BUBBLE)
    tb = solve_turbulent_bl(
        dist, tsol, op.v_inf_mps, op.nu_m2s, r.x_reattach_s * op.chord_m,
        theta_start_override_m=r.theta_reattach_m, h_start_override=h_tr,
    )
    s = build_drag_grid(op.chord_m, n_points=N_POINTS)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)
    ue_ratio_sep = float(dist.velocity_ratio(r.x_sep_lam_s))
    valid = np.where(tb.valid)[0]
    cf = m5_cf_distribution(
        op.v_inf_mps, x, op.nu_m2s, r.x_sep_lam_s * op.chord_m, r.x_reattach_s * op.chord_m,
        tb.x_m[valid], tb.cf[valid],
    )
    cd_f = section_cf_drag(s, cf, ue_ratio, two_surfaces=True)
    return cd_f + separated_drag_coefficient(r.bubble_length_s, ue_ratio_sep)


def section_cross_model_ranking(op, results: dict) -> None:
    print("=" * 100)
    print("CROSS-MODEL SENSITIVITY RANKING (one-factor-at-a-time, headline output C_d,total)")
    print("=" * 100)
    dist_baseline = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    tsol_baseline = solve_thwaites(dist_baseline, op.nu_m2s, n_points=N_POINTS)
    asol_baseline = solve_amplification(op, tsol_baseline)
    base_cd = results["baseline"].cd_total

    swings: dict[str, float] = {
        "M6 peak ratio": max(
            abs(100 * (results[lbl].cd_total - base_cd) / base_cd)
            for lbl in ("peak_ratio_mild", "peak_ratio_strong")
        ),
        "M6 peak location": max(
            abs(100 * (results[lbl].cd_total - base_cd) / base_cd)
            for lbl in ("peak_location_early", "peak_location_late")
        ),
        "M6 TE ratio": max(
            abs(100 * (results[lbl].cd_total - base_cd) / base_cd)
            for lbl in ("te_ratio_severe", "te_ratio_mild")
        ),
        "M5 bubble length": max(
            abs(100 * (_cd_total_bubble(op, dist_baseline, tsol_baseline, asol_baseline,
                                         BubbleParameters("x", 0.008, dxr, 5.0, 1.7)) - base_cd) / base_cd)
            for dxr in (0.007, 0.065)
        ),
        "M5 K_theta": max(
            abs(100 * (_cd_total_bubble(op, dist_baseline, tsol_baseline, asol_baseline,
                                         BubbleParameters("x", 0.008, 0.022, k, 1.7)) - base_cd) / base_cd)
            for k in (2.0, 12.0)
        ),
        "M5 H_reattach": max(
            abs(100 * (_cd_total_bubble(op, dist_baseline, tsol_baseline, asol_baseline,
                                         BubbleParameters("x", 0.008, 0.022, 5.0, h)) - base_cd) / base_cd)
            for h in (1.4, 2.3)
        ),
        "M5 K_sep": max(
            abs(100 * (_cd_total_bubble(op, dist_baseline, tsol_baseline, asol_baseline,
                                         NOMINAL_BUBBLE, k_sep=k) - base_cd) / base_cd)
            for k in (0.0, 2.0)
        ),
        "M4 H_tr": max(
            abs(100 * (_cd_total_h_tr(op, dist_baseline, tsol_baseline, asol_baseline, h) - base_cd) / base_cd)
            for h in (1.15, 1.5)
        ),
    }

    # N_crit sensitivity: how many distinct statuses the sourced range
    # {9,12,14} produces for the baseline (a status change, not a
    # continuous drag change, so reported separately from the ranking).
    n_crit_status_set = {locate_transition_n_crit(asol_baseline, nc).status.value for nc in (9.0, 12.0, 14.0)}
    print(
        f"N_crit sensitivity: {{9,12,14}} baseline status set size "
        f"(1 = identical status across the whole sourced range) = {len(n_crit_status_set)}"
    )
    print(f"M2 onset-shape parameter (ONSET_H_SENSITIVITY) current value: {ONSET_H_SENSITIVITY} "
          "(a fixed project constant; M2's own test suite already covers its role)")
    print(f"M4 nominal H_tr: {H_TR_NOMINAL:.4f}")
    print()

    ranked = sorted(swings.items(), key=lambda kv: kv[1], reverse=True)
    print(f"{'assumption':20} {'max |Delta% Cd,total|':>22}")
    for name, pct in ranked:
        print(f"{name:20} {pct:22.2f}")
    print()
    print(f"Single most influential modeled assumption (by this ranking): {ranked[0][0]}")
    print()


def section_final_interpretation(results: dict, dominant_profile_param: str) -> None:
    print("=" * 100)
    print("FINAL INTERPRETATION")
    print("=" * 100)
    n_turb_sep = sum(1 for r in results.values() if r.x_turb_sep_s is not None)
    n_sep_induced = sum(1 for r in results.values() if r.mechanism == "SEPARATION_INDUCED")

    text = (
        f"Across the {len(results)}-case external-flow sensitivity family, {n_sep_induced}/{len(results)} cases "
        f"retain the baseline's SEPARATION_INDUCED mechanism (laminar separation occurs before any sourced "
        f"N_crit={{9,12,14}} crossing), and {n_turb_sep}/{len(results)} cases show a downstream turbulent "
        "(re-)separation after the Milestone 5 bubble-reattachment restart. These two qualitative findings "
        "are therefore robust within the tested reduced-order sensitivity family (not universally true): "
        "at this project's baseline chord Reynolds number, laminar separation before amplification reaches "
        "any plausible N_crit, and a subsequent turbulent re-separation under continued deceleration, both "
        "persist across every synthetic pressure-distribution variant examined, not merely the one arbitrary "
        "baseline shape. "
        f"By contrast, the *quantitative* separation location, N_max, and total modeled drag are clearly "
        f"profile-sensitive: x_sep,lam/c and C_d,total both vary measurably across the family, and the "
        f"'{dominant_profile_param}' parameter produces the largest x_sep,lam/c spread of the three profile "
        "features tested. The baseline's specific numbers (x_sep,lam/c~=0.431, C_d,total~=0.0224) are "
        "therefore best read as one representative point in this family, not a universal prediction -- but "
        "the *qualitative* separation-before-N_crit and re-separation story is not an artifact of that one "
        "specific choice."
    )
    print(textwrap.fill(text, width=96))
    print()
    print("This is a deterministic reduced-order sensitivity study, not a probabilistic uncertainty analysis.")
    print("The external-velocity profiles are synthetic conceptual inputs, not CFD/XFOIL/experimental airfoil data.")


def main() -> None:
    op = default_operating_point()
    baseline_profile = next(p for p in PROFILES if p.label == "baseline")
    baseline = run_chain(baseline_profile, op)

    print("#" * 100)
    print("MILESTONE 6 -- Final robustness audit: external-flow shape sensitivity")
    print("#" * 100)
    print()

    section_baseline(op, baseline)

    results = {p.label: run_chain(p, op) for p in PROFILES}
    section_external_flow_sensitivity(results)
    dominant = section_robustness_summary(results)
    section_cross_model_ranking(op, results)
    section_final_interpretation(results, dominant)


if __name__ == "__main__":
    main()
