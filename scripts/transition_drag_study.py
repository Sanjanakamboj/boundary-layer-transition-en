#!/usr/bin/env python3
"""Milestone 3 study script: N_crit transition location and skin-friction
drag-impact bookkeeping.

Run with:

    python3 scripts/transition_drag_study.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.laminar_bl import solve_thwaites
from boundary_layer_transition.operating_point import GenericSailplaneOperatingPoint, default_operating_point
from boundary_layer_transition.skin_friction import (
    build_drag_grid,
    section_cf_drag,
    transitioned_cf_distribution,
)
from boundary_layer_transition.stability import solve_amplification
from boundary_layer_transition.transition import (
    TransitionStatus,
    locate_transition_n_crit,
    separation_triggered_transition,
)

N_CRIT_VALUES = [9.0, 12.0, 14.0]  # average wind tunnel; sailplane lower/upper bound (Drela XFOIL doc)
SENSITIVITY_V_INF = [20.0, 25.0, 30.0, 40.0, 60.0, 90.0]  # m/s; all keep M < 0.3
PRESCRIBED_XTR = [0.1, 0.3, 0.5, 0.7, 0.9]
N_POINTS = 4001


def build_case(v_inf: float):
    base = default_operating_point()
    op = GenericSailplaneOperatingPoint(
        chord_m=base.chord_m, v_inf_mps=v_inf, rho_kgm3=base.rho_kgm3,
        mu_pas=base.mu_pas, temperature_k=base.temperature_k, alpha_deg=base.alpha_deg,
    )
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    tsol = solve_thwaites(dist, op.nu_m2s, n_points=N_POINTS)
    asol = solve_amplification(op, tsol)
    return op, dist, tsol, asol


def section_a(asol) -> None:
    print("=" * 92)
    print("SECTION A -- N_crit sensitivity (baseline operating point)")
    print("=" * 92)
    header = f"{'N_crit':>7} {'status':>26} {'x_tr,N/c':>10} {'x_sep/c':>9} {'N_max':>8}"
    print(header)
    for n_crit in N_CRIT_VALUES:
        r = locate_transition_n_crit(asol, n_crit)
        xtr_str = f"{r.x_tr_s:.4f}" if r.x_tr_s is not None else "   n/a   "
        print(f"{n_crit:7.1f} {r.status.value:>26} {xtr_str:>10} {r.s_sep:9.4f} {r.n_max:8.4f}")
    print()


def section_b() -> None:
    print("=" * 92)
    print("SECTION B -- Reynolds (V_inf) sensitivity, transition status per N_crit")
    print("=" * 92)
    header = f"{'V_inf':>7} {'Re_c':>11} {'N_max':>7}" + "".join(f"{'Ncrit='+str(int(nc)):>16}" for nc in N_CRIT_VALUES)
    print(header)
    for v in SENSITIVITY_V_INF:
        op, dist, tsol, asol = build_case(v)
        row = f"{v:7.1f} {op.re_c:11.3e} {asol.n_max:7.3f}"
        for n_crit in N_CRIT_VALUES:
            r = locate_transition_n_crit(asol, n_crit)
            if r.status == TransitionStatus.N_CRIT_CROSSING:
                cell = f"cross@{r.x_tr_s:.3f}"
            elif r.status == TransitionStatus.SEPARATION_BEFORE_N_CRIT:
                cell = "sep-first"
            else:
                cell = "no-event"
            row += f"{cell:>16}"
        print(row)
    print()
    print("'cross@x/c' = N_crit reached at that x/c (interpolated); 'sep-first' = M1 laminar")
    print("separation occurs before N reaches N_crit; 'no-event' = neither occurs in the domain.")
    print("M is < 0.3 (incompressible assumption holds) for every V_inf case shown.")
    print()


def section_c(op, dist, asol) -> None:
    print("=" * 92)
    print("SECTION C -- Drag bookkeeping (baseline operating point)")
    print("=" * 92)
    s = build_drag_grid(op.chord_m, n_points=N_POINTS)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)

    def cd_for(x_tr_m: float) -> float:
        cf = transitioned_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m=x_tr_m)
        return section_cf_drag(s, cf, ue_ratio, two_surfaces=True)

    cd_lam = cd_for(math.inf)
    cd_turb = cd_for(0.0)
    scenario = separation_triggered_transition(asol)
    cd_sep = cd_for(scenario.x_tr_m)

    pct_increase_vs_lam = 100.0 * (cd_sep - cd_lam) / cd_lam
    pct_reduction_vs_turb = 100.0 * (cd_turb - cd_sep) / cd_turb

    print(f"Fully laminar reference C_d,f           = {cd_lam:.6f}")
    print(f"Fully turbulent reference C_d,f          = {cd_turb:.6f}")
    print(f"Separation-triggered C_d,f (x_tr/c={scenario.x_tr_s:.4f}) = {cd_sep:.6f}")
    print(f"  percent increase vs. fully laminar     = {pct_increase_vs_lam:+.2f}%")
    print(f"  percent reduction vs. fully turbulent  = {pct_reduction_vs_turb:+.2f}%")
    print()

    n_crit_9 = locate_transition_n_crit(asol, 9.0)
    if n_crit_9.status == TransitionStatus.N_CRIT_CROSSING:
        cd_ncrit = cd_for(n_crit_9.x_tr_m)
        print(f"N_crit=9 crossing C_d,f (x_tr/c={n_crit_9.x_tr_s:.4f})       = {cd_ncrit:.6f}")
    else:
        print("N_crit=9 crossing: not reached at the baseline operating point "
              f"({n_crit_9.status.value}) -- no N_crit-based C_d,f reported for the baseline.")
    print()
    return cd_lam, cd_turb


def section_d(op, dist, cd_lam: float, cd_turb: float) -> None:
    print("=" * 92)
    print("SECTION D -- Prescribed transition-location sensitivity")
    print("=" * 92)
    s = build_drag_grid(op.chord_m, n_points=N_POINTS)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)

    header = f"{'x_tr/c':>8} {'C_d,f':>10} {'%vs laminar':>13} {'%vs turbulent':>15}"
    print(header)
    for x_tr in PRESCRIBED_XTR:
        cf = transitioned_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m=x_tr * op.chord_m)
        cd = section_cf_drag(s, cf, ue_ratio, two_surfaces=True)
        pct_vs_lam = 100.0 * (cd - cd_lam) / cd_lam
        pct_vs_turb = 100.0 * (cd - cd_turb) / cd_turb
        print(f"{x_tr:8.2f} {cd:10.6f} {pct_vs_lam:+13.2f} {pct_vs_turb:+15.2f}")
    print()
    print("Later transition consistently reduces C_d,f toward the fully laminar reference;")
    print("earlier transition pushes C_d,f toward the fully turbulent reference. The lowest-drag")
    print("prescribed case is not described as an 'optimum' -- no structural, stall, or")
    print("off-design constraints are modeled in this project.")


def main() -> None:
    op = default_operating_point()
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    tsol = solve_thwaites(dist, op.nu_m2s, n_points=N_POINTS)
    asol = solve_amplification(op, tsol)

    print("#" * 92)
    print("MILESTONE 3 -- N_crit transition location and skin-friction drag-impact study")
    print(f"Baseline: Re_c={op.re_c:.4e}, chord={op.chord_m} m, V_inf={op.v_inf_mps} m/s")
    print("#" * 92)
    print()

    section_a(asol)
    section_b()
    cd_lam, cd_turb = section_c(op, dist, asol)
    section_d(op, dist, cd_lam, cd_turb)

    print()
    print("The separation-triggered transition case is a bookkeeping assumption, not an e^N")
    print("N_crit crossing.")
    print("Skin-friction drag uses reduced-order flat-plate correlations, not a turbulent")
    print("pressure-gradient boundary-layer solution.")


if __name__ == "__main__":
    main()
