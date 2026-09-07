#!/usr/bin/env python3
"""Print representative operating-condition and boundary-layer values.

This script is a human-readable sanity check, not a test. Run it with:

    python3 scripts/manual_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.laminar_bl import (
    blasius_cf,
    blasius_delta99,
    blasius_theta,
    solve_thwaites,
)
from boundary_layer_transition.operating_point import default_operating_point

STATIONS_S = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.00]


def main() -> None:
    op = default_operating_point()
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    sol = solve_thwaites(dist, op.nu_m2s, n_points=4001)

    print("=" * 78)
    print("MILESTONE 1 -- Generic sailplane wing-section operating point")
    print("(illustrative, not manufacturer-matched; NOT a transition prediction)")
    print("=" * 78)
    print(op.summary())
    print()

    print("-" * 78)
    print("External velocity distribution and Blasius ZPG reference at select x/c")
    print("-" * 78)
    header = (
        f"{'x/c':>6} {'x [m]':>8} {'Ue [m/s]':>9} {'Ue/Vinf':>8} {'dUe/dx [1/s]':>13} "
        f"{'Re_x':>10} {'d99 [mm]':>9} {'theta [mm]':>10} {'Cf,x':>8}"
    )
    print(header)
    for s in STATIONS_S:
        x = s * op.chord_m
        u_e = dist.u_e(x)
        due_dx = dist.du_e_dx(x)
        re_x = op.v_inf_mps * x / op.nu_m2s
        d99 = blasius_delta99(op.v_inf_mps, x, op.nu_m2s)
        theta = blasius_theta(op.v_inf_mps, x, op.nu_m2s)
        cf = blasius_cf(op.v_inf_mps, x, op.nu_m2s)
        cf_str = "inf (LE)" if np.isinf(cf) else f"{cf:8.5f}"
        print(
            f"{s:6.2f} {x:8.4f} {u_e:9.3f} {u_e/op.v_inf_mps:8.4f} {due_dx:13.3f} "
            f"{re_x:10.3e} {1000*d99:9.3f} {1000*theta:10.4f} {cf_str}"
        )
    print()

    print("-" * 78)
    print("Thwaites pressure-gradient-aware laminar boundary layer at select x/c")
    print("-" * 78)
    header2 = (
        f"{'x/c':>6} {'Ue/Vinf':>8} {'Re_x':>10} {'theta [mm]':>10} {'d* [mm]':>8} "
        f"{'H':>7} {'lambda':>9} {'Cf,x':>8} {'separated':>10}"
    )
    print(header2)
    for s in STATIONS_S:
        i = int(np.argmin(np.abs(sol.s - s)))
        h_str = "  nan  " if np.isnan(sol.h[i]) else f"{sol.h[i]:7.3f}"
        cf_str = "  nan  " if np.isnan(sol.cf[i]) else f"{sol.cf[i]:8.5f}"
        dstar_str = " nan  " if np.isnan(sol.delta_star_m[i]) else f"{1000*sol.delta_star_m[i]:8.4f}"
        print(
            f"{sol.s[i]:6.2f} {sol.u_e_mps[i]/op.v_inf_mps:8.4f} {sol.re_x[i]:10.3e} "
            f"{1000*sol.theta_m[i]:10.4f} {dstar_str} {h_str} {sol.lam[i]:9.4f} {cf_str} "
            f"{bool(sol.separated[i])!s:>10}"
        )
    print()

    if sol.s_sep is not None:
        print(f"Predicted laminar separation (Thwaites, l(lambda)=0 crossing): x/c = {sol.s_sep:.4f}")
        print("  ** Diagnostic only: laminar separation is NOT equivalent to transition. **")
    else:
        print("No laminar separation predicted over the modeled domain (x/c in [0, 1]).")
    print()
    print("Reminder: this is a reduced-order, illustrative model. No e^N amplification,")
    print("N_crit, or transition-location prediction is performed in Milestone 1.")


if __name__ == "__main__":
    main()
