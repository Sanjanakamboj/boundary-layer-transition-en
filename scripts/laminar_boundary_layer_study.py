#!/usr/bin/env python3
"""Compact comparison of Blasius (ZPG) vs Thwaites (pressure-gradient-aware)
laminar boundary-layer results, and a laminar-separation diagnostic report.

Run with:

    python3 scripts/laminar_boundary_layer_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.laminar_bl import (
    BLASIUS_H,
    blasius_delta_star,
    blasius_theta,
    solve_thwaites,
    thwaites_separation_lambda,
)
from boundary_layer_transition.operating_point import default_operating_point

STATIONS_S = [0.02, 0.05, 0.10, 0.20, 0.30, 0.35, 0.40]  # pre-separation stations only


def main() -> None:
    op = default_operating_point()
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    sol = solve_thwaites(dist, op.nu_m2s, n_points=4001)

    print("=" * 92)
    print("Laminar boundary-layer study: Blasius (ZPG) vs Thwaites (pressure-gradient-aware)")
    print(f"Re_c = {op.re_c:.3e}, chord = {op.chord_m} m, V_inf = {op.v_inf_mps} m/s")
    print("Blasius uses the LOCAL Re_x with the SAME x but a hypothetical constant U_e = V_inf;")
    print("it is a ZPG reference, not a solution for this pressure gradient.")
    print("=" * 92)
    header = (
        f"{'x/c':>5} {'theta_B [mm]':>13} {'theta_T [mm]':>13} {'dstar_B [mm]':>13} "
        f"{'dstar_T [mm]':>13} {'H_B':>6} {'H_T':>7} {'lambda':>8}"
    )
    print(header)
    for s in STATIONS_S:
        x = s * op.chord_m
        theta_b = blasius_theta(op.v_inf_mps, x, op.nu_m2s)
        dstar_b = blasius_delta_star(op.v_inf_mps, x, op.nu_m2s)
        i = int(np.argmin(np.abs(sol.s - s)))
        h_t_str = f"{sol.h[i]:7.3f}" if np.isfinite(sol.h[i]) else "   nan "
        dstar_t_str = f"{1000*sol.delta_star_m[i]:13.4f}" if np.isfinite(sol.delta_star_m[i]) else " " * 8 + "nan"
        print(
            f"{s:5.2f} {1000*theta_b:13.4f} {1000*sol.theta_m[i]:13.4f} {1000*dstar_b:13.4f} "
            f"{dstar_t_str} {BLASIUS_H:6.3f} {h_t_str} {sol.lam[i]:8.4f}"
        )
    print()

    # Zero-pressure-gradient recovery check near the leading edge, where the
    # favorable gradient is weakest (lambda closest to 0).
    i_near_le = 2
    theta_b_near_le = blasius_theta(op.v_inf_mps, sol.x_m[i_near_le], op.nu_m2s)
    rel_theta_err = abs(sol.theta_m[i_near_le] - theta_b_near_le) / theta_b_near_le
    print(
        f"Near-LE ZPG-recovery check (x/c={sol.s[i_near_le]:.4f}, lambda={sol.lam[i_near_le]:.5f}): "
        f"Thwaites theta differs from Blasius theta by {100*rel_theta_err:.2f}% "
        "(expected: small, since the local gradient is weak there)."
    )
    print()

    print("-" * 92)
    print("Laminar separation diagnostic (Thwaites method)")
    print("-" * 92)
    if sol.s_sep is not None:
        print(f"  Predicted separation at x/c = {sol.s_sep:.4f} (x = {sol.x_sep_m*1000:.2f} mm)")
        print(f"  Correlation's l(lambda)=0 root: lambda_sep = {thwaites_separation_lambda():.5f} "
              "(literature reference ~ -0.09)")
    else:
        print("  No laminar separation predicted over x/c in [0, 1].")
    print()
    print("  ** This is a laminar-boundary-layer diagnostic ONLY. It is NOT a transition")
    print("     prediction. Transition location requires an e^N amplification-factor")
    print("     calculation against a chosen N_crit, which is explicitly out of scope")
    print("     for Milestone 1. **")


if __name__ == "__main__":
    main()
