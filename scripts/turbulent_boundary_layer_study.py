#!/usr/bin/env python3
"""Milestone 4 study script: pressure-gradient turbulent boundary layer
(Head's entrainment method) and its effect on section skin-friction drag.

Run with:

    python3 scripts/turbulent_boundary_layer_study.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.laminar_bl import solve_thwaites
from boundary_layer_transition.operating_point import GenericSailplaneOperatingPoint, default_operating_point
from boundary_layer_transition.skin_friction import build_drag_grid, section_cf_drag, transitioned_cf_distribution
from boundary_layer_transition.stability import solve_amplification
from boundary_layer_transition.transition import (
    TransitionStatus,
    locate_transition_n_crit,
    separation_triggered_transition,
)
from boundary_layer_transition.turbulent_bl import (
    H_TR_NOMINAL,
    TurbulentBLStatus,
    pressure_gradient_cf_distribution,
    solve_turbulent_bl,
)

N_POINTS = 4001
PRESCRIBED_XTR_M4 = [0.10, 0.25, 0.40]  # early / mid / late, all within the M1 laminar-valid domain (x_sep~=0.431)
SENSITIVITY_V_INF = [20.0, 25.0, 30.0, 40.0, 60.0, 90.0]
H_TR_SENSITIVITY = [1.20, H_TR_NOMINAL, 1.40, 1.60]


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


def cd_m3(op, dist, x_tr_m: float, s, x, ue_ratio) -> float:
    cf = transitioned_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m=x_tr_m)
    return section_cf_drag(s, cf, ue_ratio, two_surfaces=True)


def cd_m4(op, x_tr_m: float, tb, s, x, ue_ratio) -> float:
    cf = pressure_gradient_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m, tb)
    return section_cf_drag(s, cf, ue_ratio, two_surfaces=True)


def section_a(op, asol) -> None:
    sep = separation_triggered_transition(asol)
    print("=" * 92)
    print("SECTION A -- Baseline (V_inf=25 m/s)")
    print("=" * 92)
    print(f"V_inf                                  = {op.v_inf_mps:.1f} m/s")
    print(f"Re_c                                    = {op.re_c:.4e}")
    print(f"x_sep,lam/c (M1 laminar separation)      = {asol.s_sep:.4f}")
    print(f"M2 N_max before separation                = {asol.n_max:.4f}")
    r9 = locate_transition_n_crit(asol, 9.0)
    print(f"Baseline transition status (N_crit=9)     = {r9.status.value}")
    print(f"Assumed separation-triggered x_tr/c        = {sep.x_tr_s:.4f}")
    print()
    return sep


def section_b(op, dist, tsol, asol, sep) -> None:
    print("=" * 92)
    print("SECTION B -- Turbulent propagation from the separation-triggered station")
    print("=" * 92)
    tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, sep.x_tr_m)
    valid = np.where(tb.valid)[0]
    print(f"theta_tr                                = {1000 * tb.theta_m[0]:.4f} mm")
    print(f"H_tr                                     = {tb.h[0]:.4f}")
    if tb.status == TurbulentBLStatus.COMPLETED_TO_TE:
        print(f"TE theta                                 = {1000 * tb.theta_m[valid[-1]]:.4f} mm")
        print(f"TE H                                      = {tb.h[valid[-1]]:.4f}")
    else:
        print("TE theta/H not reached (see status below)")
    print(f"Cf range over valid domain                = [{tb.cf[valid].min():.6f}, {tb.cf[valid].max():.6f}]")
    print(f"Turbulent status                          = {tb.status.value}")
    print(f"  {tb.message}")
    if tb.x_turb_sep_m is not None:
        print(f"Turbulent separation location, x/c        = {tb.x_turb_sep_m / op.chord_m:.4f}")
    print()
    return tb


def section_c(op, dist, tsol, asol, sep, tb) -> None:
    print("=" * 92)
    print("SECTION C -- M3 (flat-plate) vs. M4 (pressure-gradient) drag")
    print("=" * 92)
    s = build_drag_grid(op.chord_m, n_points=N_POINTS)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)

    scenarios: list[tuple[str, float]] = [
        ("fully turbulent (x_tr=0)", 0.0),
        (f"early prescribed x_tr/c={PRESCRIBED_XTR_M4[0]:.2f}", PRESCRIBED_XTR_M4[0] * op.chord_m),
        (f"late prescribed x_tr/c={PRESCRIBED_XTR_M4[2]:.2f}", PRESCRIBED_XTR_M4[2] * op.chord_m),
        (f"separation-triggered x_tr/c={sep.x_tr_s:.4f}", sep.x_tr_m),
    ]

    r9_60 = None
    op60, dist60, tsol60, asol60 = build_case(60.0)
    r9_60 = locate_transition_n_crit(asol60, 9.0)
    if r9_60.status == TransitionStatus.N_CRIT_CROSSING:
        s60 = build_drag_grid(op60.chord_m, n_points=N_POINTS)
        x60 = s60 * op60.chord_m
        ue60 = dist60.velocity_ratio(s60)
        tb60 = solve_turbulent_bl(dist60, tsol60, op60.v_inf_mps, op60.nu_m2s, r9_60.x_tr_m)
        cd3 = cd_m3(op60, dist60, r9_60.x_tr_m, s60, x60, ue60)
        cd4 = cd_m4(op60, r9_60.x_tr_m, tb60, s60, x60, ue60)
        pct = 100.0 * (cd4 - cd3) / cd3
        print(
            f"{'N_crit=9 crossing (V=60 m/s), x_tr/c=' + format(r9_60.x_tr_s, '.4f'):45} "
            f"M3={cd3:.6f}  M4={cd4:.6f}  Delta%={pct:+.2f}%  M4 status={tb60.status.value}"
        )

    header = f"{'scenario':45} {'M3 Cd,f':>10} {'M4 Cd,f':>10} {'Delta%':>9} {'M4 status':>22}"
    print(header)
    for label, x_tr_m in scenarios:
        tb_s = tb if math.isclose(x_tr_m, sep.x_tr_m) else solve_turbulent_bl(
            dist, tsol, op.v_inf_mps, op.nu_m2s, x_tr_m
        )
        cd3 = cd_m3(op, dist, x_tr_m, s, x, ue_ratio)
        cd4 = cd_m4(op, x_tr_m, tb_s, s, x, ue_ratio)
        pct = 100.0 * (cd4 - cd3) / cd3
        print(f"{label:45} {cd3:10.6f} {cd4:10.6f} {pct:+9.2f} {tb_s.status.value:>22}")
    print()


def section_d(op) -> None:
    print("=" * 92)
    print("SECTION D -- Reynolds (V_inf) sensitivity (separation-triggered transition)")
    print("=" * 92)
    header = f"{'V_inf':>7} {'Re_c':>11} {'x_tr/c':>8} {'turbulent status':>20} {'Cd,f (M4)':>10}"
    print(header)
    for v in SENSITIVITY_V_INF:
        op_v, dist_v, tsol_v, asol_v = build_case(v)
        sep_v = separation_triggered_transition(asol_v)
        tb_v = solve_turbulent_bl(dist_v, tsol_v, op_v.v_inf_mps, op_v.nu_m2s, sep_v.x_tr_m)
        s = build_drag_grid(op_v.chord_m, n_points=N_POINTS)
        x = s * op_v.chord_m
        ue_ratio = dist_v.velocity_ratio(s)
        cd4 = cd_m4(op_v, sep_v.x_tr_m, tb_v, s, x, ue_ratio)
        print(f"{v:7.1f} {op_v.re_c:11.3e} {sep_v.x_tr_s:8.4f} {tb_v.status.value:>20} {cd4:10.6f}")
    print()


def section_e(op, dist, tsol, sep) -> None:
    print("=" * 92)
    print("SECTION E -- Initialization sensitivity (H_tr at the separation-triggered station)")
    print("=" * 92)
    s = build_drag_grid(op.chord_m, n_points=N_POINTS)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)
    header = f"{'H_tr':>8} {'status':>22} {'Cd,f (M4)':>10}"
    print(header)
    for h_tr in H_TR_SENSITIVITY:
        tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, sep.x_tr_m, h_tr=h_tr)
        cd4 = cd_m4(op, sep.x_tr_m, tb, s, x, ue_ratio)
        marker = " (nominal)" if math.isclose(h_tr, H_TR_NOMINAL) else ""
        print(f"{h_tr:8.3f} {tb.status.value:>22} {cd4:10.6f}{marker}")
    print()


def main() -> None:
    op = default_operating_point()
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    tsol = solve_thwaites(dist, op.nu_m2s, n_points=N_POINTS)
    asol = solve_amplification(op, tsol)

    print("#" * 92)
    print("MILESTONE 4 -- Pressure-gradient turbulent boundary layer (Head's entrainment method)")
    print("#" * 92)
    print()

    sep = section_a(op, asol)
    tb = section_b(op, dist, tsol, asol, sep)
    section_c(op, dist, tsol, asol, sep, tb)
    section_d(op)
    section_e(op, dist, tsol, sep)

    print("The turbulent model is a reduced-order pressure-gradient integral method, not CFD or")
    print("experimental validation.")
    print("The separation-triggered case assumes an instantaneous laminar-to-turbulent restart;")
    print("no laminar separation bubble is resolved.")


if __name__ == "__main__":
    main()
