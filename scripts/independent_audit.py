#!/usr/bin/env python3
"""Milestone 6 independent end-to-end audit.

Recomputes key identities and results independently (via hand formulas,
alternate numerical paths, or reconstructed residuals) rather than calling
the same production helper on both sides of a comparison, and prints a
concise residual table across M1-M6.

Run with:

    python3 scripts/independent_audit.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.external_flow_sensitivity import PROFILES, build_distribution
from boundary_layer_transition.laminar_bl import (
    BLASIUS_C_CF,
    BLASIUS_C_DSTAR,
    BLASIUS_C_THETA,
    blasius_cf,
    blasius_delta_star,
    blasius_theta,
    reynolds_x,
    solve_thwaites,
    thwaites_shear_function,
)
from boundary_layer_transition.operating_point import default_operating_point
from boundary_layer_transition.robustness import run_chain
from boundary_layer_transition.separated_drag import build_drag_decomposition, separated_drag_coefficient
from boundary_layer_transition.separation_bubble import NOMINAL_BUBBLE, evaluate_transition_mechanism
from boundary_layer_transition.skin_friction import (
    TURBULENT_CF_COEFF_LOCAL,
    section_cf_drag,
    turbulent_cf_local,
)
from boundary_layer_transition.stability import (
    H_REFERENCE,
    ONSET_H_SENSITIVITY,
    RE_THETA_CRIT_ZPG,
    critical_re_theta,
    integrate_n_factor,
    local_amplification_rate,
    reynolds_theta,
    solve_amplification,
)
from boundary_layer_transition.transition import interpolate_crossing
from boundary_layer_transition.turbulent_bl import (
    LT_A,
    LT_B,
    LT_C,
    entrainment_rate,
    h1_of_h,
    h_of_h1,
    ludwieg_tillmann_cf,
    momentum_equation_rhs,
    solve_turbulent_bl,
)

RESIDUALS: list[tuple[str, float, float, float]] = []  # (name, abs_residual, rel_residual, tol_note)


def check(name: str, computed: float, independent: float, tol_note: str = "") -> None:
    abs_res = abs(computed - independent)
    rel_res = abs_res / abs(independent) if independent != 0 else float("nan")
    RESIDUALS.append((name, abs_res, rel_res, tol_note))


def audit_m1(op) -> None:
    print("--- M1: laminar boundary layer ---")
    hand_re_c = op.rho_kgm3 * op.v_inf_mps * op.chord_m / op.mu_pas
    check("M1 Re_c", op.re_c, hand_re_c)

    x_station = 0.245
    hand_re_x = op.v_inf_mps * x_station / op.nu_m2s
    check("M1 Re_x @0.245m", reynolds_x(op.v_inf_mps, x_station, op.nu_m2s), hand_re_x)

    hand_theta = BLASIUS_C_THETA * math.sqrt(x_station * op.nu_m2s / op.v_inf_mps)
    check("M1 Blasius theta", blasius_theta(op.v_inf_mps, x_station, op.nu_m2s), hand_theta)
    hand_dstar = BLASIUS_C_DSTAR * math.sqrt(x_station * op.nu_m2s / op.v_inf_mps)
    check("M1 Blasius delta*", blasius_delta_star(op.v_inf_mps, x_station, op.nu_m2s), hand_dstar)
    hand_cf = BLASIUS_C_CF / math.sqrt(hand_re_x)
    check("M1 Blasius Cf", blasius_cf(op.v_inf_mps, x_station, op.nu_m2s), hand_cf)

    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    check("M1 U_e/Vinf(s=0)", dist.velocity_ratio(0.0), 1.0)
    check("M1 U_e/Vinf(s_peak)", dist.velocity_ratio(dist.s_peak), dist.u_peak_ratio)
    check("M1 U_e/Vinf(s=1)", dist.velocity_ratio(1.0), dist.u_te_ratio)

    tsol = solve_thwaites(dist, op.nu_m2s, n_points=4001)
    i_check = 800
    x_fine = np.linspace(0.0, tsol.x_m[i_check], 20001)
    u_fine = dist.u_e(x_fine)
    integral_manual = np.trapezoid(u_fine**5, x_fine)
    theta_sq_manual = 0.45 * op.nu_m2s / dist.u_e(tsol.x_m[i_check]) ** 6 * integral_manual
    check(
        "M1 Thwaites theta integral", tsol.theta_m[i_check], math.sqrt(theta_sq_manual),
        "manual quadrature, ~1e-3 rel",
    )

    # Independent bisection re-check of x_sep via the sign change of l(lambda).
    l_vals = thwaites_shear_function(tsol.lam)
    sign_change = np.where((l_vals[:-1] > 0.0) & (l_vals[1:] <= 0.0))[0]
    i0 = int(sign_change[0])
    frac = l_vals[i0] / (l_vals[i0] - l_vals[i0 + 1])
    s_sep_manual = tsol.s[i0] + frac * (tsol.s[i0 + 1] - tsol.s[i0])
    check("M1 x_sep,lam/c", tsol.s_sep, s_sep_manual)


def audit_m2(op) -> None:
    print("--- M2: e^N amplification ---")
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    tsol = solve_thwaites(dist, op.nu_m2s, n_points=4001)
    asol = solve_amplification(op, tsol)

    i = 900
    hand_re_theta = op.rho_kgm3 * asol.u_e_mps[i] * asol.theta_m[i] / op.mu_pas
    computed_re_theta = reynolds_theta(op.rho_kgm3, asol.u_e_mps[i], asol.theta_m[i], op.mu_pas)
    check("M2 Re_theta @station", computed_re_theta, hand_re_theta)

    h_test = 3.0
    hand_crit = RE_THETA_CRIT_ZPG * math.exp(-ONSET_H_SENSITIVITY * (h_test - H_REFERENCE))
    check("M2 critical Re_theta(H=3)", critical_re_theta(h_test), hand_crit)

    re_theta_test, h_dummy = 350.0, 3.2
    crit = critical_re_theta(h_dummy)
    hand_dnds = 10.0 * (h_dummy - 1.0) * max(0.0, re_theta_test - crit) / RE_THETA_CRIT_ZPG
    check("M2 dN/ds representative", local_amplification_rate(re_theta_test, h_dummy), hand_dnds)

    # Independent constant-rate N-integration control case.
    s_ctrl = np.linspace(0.1, 0.6, 501)
    k_ctrl = 2.5
    n_ctrl = integrate_n_factor(s_ctrl, np.full_like(s_ctrl, k_ctrl))
    check("M2 N-integration control (constant rate)", n_ctrl[-1], k_ctrl * (s_ctrl[-1] - s_ctrl[0]))


def audit_m3(op) -> None:
    print("--- M3: N_crit crossing / flat-plate drag ---")
    s = np.array([0.0, 0.1, 0.2, 0.3])
    n = np.array([0.0, 1.0, 3.0, 6.0])
    hand_frac = (2.0 - 1.0) / (3.0 - 1.0)
    hand_cross = s[1] + hand_frac * (s[2] - s[1])
    check("M3 N_crit interpolation", interpolate_crossing(s, n, 2.0), hand_cross)

    s_drag = np.linspace(0.0, 1.0, 1001)
    cf_const = np.full_like(s_drag, 0.005)
    ue_const = np.full_like(s_drag, 2.0)
    hand_cd = 2.0 * 0.005 * 4.0
    check("M3 drag normalization (constant Cf, Ue=2Vinf)", section_cf_drag(s_drag, cf_const, ue_const, True), hand_cd)

    x_station = 0.245
    re_x = op.v_inf_mps * x_station / op.nu_m2s
    hand_cf_turb = TURBULENT_CF_COEFF_LOCAL * re_x**-0.2
    check("M3 turbulent flat-plate Cf", turbulent_cf_local(op.v_inf_mps, x_station, op.nu_m2s), hand_cf_turb)


def audit_m4(op) -> None:
    print("--- M4: pressure-gradient turbulent boundary layer ---")
    for h in (1.3, 1.6, 2.0, 3.5):
        h1 = h1_of_h(h)
        check(f"M4 H1(H) roundtrip @H={h}", h_of_h1(h1), h)

    h_lt, re_theta_lt = 1.4, 2000.0
    hand_cf = LT_A * 10 ** (LT_B * h_lt) * re_theta_lt**LT_C
    check("M4 Ludwieg-Tillmann Cf", ludwieg_tillmann_cf(h_lt, re_theta_lt), hand_cf)

    # Momentum-equation residual reconstructed from an actual solved
    # trajectory (a genuine ODE/finite-difference residual, not expected
    # to equal machine precision).
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    tsol = solve_thwaites(dist, op.nu_m2s, n_points=4001)
    asol = solve_amplification(op, tsol)
    r = evaluate_transition_mechanism(asol, tsol, 9.0, NOMINAL_BUBBLE)
    tb = solve_turbulent_bl(
        dist, tsol, op.v_inf_mps, op.nu_m2s, r.x_reattach_s * op.chord_m,
        theta_start_override_m=r.theta_reattach_m, h_start_override=r.h_reattach, n_report=2001,
    )
    valid = np.where(tb.valid)[0]
    i_mid = valid[len(valid) // 2]
    fd_dtheta_dx = (tb.theta_m[valid[len(valid) // 2 + 1]] - tb.theta_m[valid[len(valid) // 2 - 1]]) / (
        tb.x_m[valid[len(valid) // 2 + 1]] - tb.x_m[valid[len(valid) // 2 - 1]]
    )
    rhs_val = momentum_equation_rhs(
        tb.theta_m[i_mid], tb.h[i_mid], tb.cf[i_mid], tb.u_e_mps[i_mid], tb.du_e_dx[i_mid]
    )
    check(
        "M4 momentum-eq residual (FD vs RHS)", fd_dtheta_dx, rhs_val,
        "finite-difference residual, not machine precision",
    )

    h1_test = 5.0
    hand_f = 0.0299 * (h1_test - 3.0) ** -0.6169
    check("M4 entrainment F(H1)", entrainment_rate(h1_test), hand_f)

    check(
        "M4 representative turbulent separation x/c", tb.x_turb_sep_m / op.chord_m, 0.6704,
        "grid-dependent, ~1e-2 tol",
    )


def audit_m5(op) -> None:
    print("--- M5: separation bubble / separated drag ---")
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    tsol = solve_thwaites(dist, op.nu_m2s, n_points=4001)
    asol = solve_amplification(op, tsol)
    r = evaluate_transition_mechanism(asol, tsol, 9.0, NOMINAL_BUBBLE)

    ordering_ok = r.x_sep_lam_s <= r.x_tr_sep_s <= r.x_reattach_s
    check("M5 bubble event ordering (0=ok)", 0.0 if ordering_ok else 1.0, 0.0)

    hand_theta_sep = float(np.interp(r.x_sep_lam_s * op.chord_m, tsol.x_m, tsol.theta_m))
    hand_theta_reattach = NOMINAL_BUBBLE.k_theta * hand_theta_sep
    check("M5 restart theta", r.theta_reattach_m, hand_theta_reattach)

    l_sep, ue_ratio = 0.03, 1.15
    hand_cd_sep = 0.5 * l_sep**1.0 * ue_ratio**2
    check("M5 separated-drag formula", separated_drag_coefficient(l_sep, ue_ratio), hand_cd_sep)

    decomp = build_drag_decomposition("audit", 0.003, 0.02, 1.1)
    check("M5 drag decomposition identity", decomp.cd_total_m5, decomp.cd_f_m5 + decomp.cd_sep_m5)


def audit_m6(op) -> None:
    print("--- M6: external-flow sensitivity ---")
    baseline_profile = next(p for p in PROFILES if p.label == "baseline")
    dist_sensitivity = build_distribution(baseline_profile, op.chord_m, op.v_inf_mps)
    dist_m1 = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    s = np.linspace(0.0, 1.0, 2001)
    max_diff = float(np.max(np.abs(dist_sensitivity.velocity_ratio(s) - dist_m1.velocity_ratio(s))))
    check("M6 baseline profile bit-for-bit regression", max_diff, 0.0)

    severe = next(p for p in PROFILES if p.label == "te_ratio_severe")
    dist_severe = build_distribution(severe, op.chord_m, op.v_inf_mps)
    check("M6 profile anchor (te_ratio_severe TE value)", dist_severe.velocity_ratio(1.0), 0.45)

    r = run_chain(baseline_profile, op)
    check("M6 chain-orchestrated x_sep,lam/c vs M1 direct", r.x_sep_lam_s, tsol_direct_s_sep(op))


def tsol_direct_s_sep(op) -> float:
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    tsol = solve_thwaites(dist, op.nu_m2s, n_points=4001)
    return tsol.s_sep


def print_residual_table() -> None:
    print()
    print("=" * 92)
    print("RESIDUAL TABLE")
    print("=" * 92)
    header = f"{'check':45} {'abs residual':>14} {'rel residual':>14} {'note':>20}"
    print(header)
    max_abs = 0.0
    max_rel = 0.0
    for name, abs_res, rel_res, note in RESIDUALS:
        rel_str = f"{rel_res:.3e}" if math.isfinite(rel_res) else "n/a"
        print(f"{name:45} {abs_res:14.3e} {rel_str:>14} {note:>20}")
        max_abs = max(max_abs, abs_res)
        if math.isfinite(rel_res):
            max_rel = max(max_rel, rel_res)
    print()
    print(f"Maximum absolute residual across all {len(RESIDUALS)} checks: {max_abs:.3e}")
    print(f"Maximum relative residual (finite only):                     {max_rel:.3e}")
    print()
    print("Note: the M4 momentum-equation and turbulent-separation checks are genuine")
    print("finite-difference/ODE-discretization residuals and are not expected to reach")
    print("machine precision; all other checks are algebraic identities expected to hold")
    print("to floating-point precision (~1e-9 or tighter).")


def main() -> None:
    op = default_operating_point()
    print("#" * 92)
    print("MILESTONE 6 -- Independent end-to-end audit")
    print("#" * 92)
    audit_m1(op)
    audit_m2(op)
    audit_m3(op)
    audit_m4(op)
    audit_m5(op)
    audit_m6(op)
    print_residual_table()


if __name__ == "__main__":
    main()
