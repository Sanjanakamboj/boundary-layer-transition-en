#!/usr/bin/env python3
"""Milestone 2 study script: reduced-order e^N amplification tracking.

Prints a compact engineering table of stability/amplification quantities
at representative pre-separation stations, the modeled instability onset,
the M1 laminar-separation diagnostic, the available amplification length,
the maximum accumulated N before separation, and a V_inf/Re_c sensitivity
study.

Run with:

    python3 scripts/en_amplification_study.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.laminar_bl import solve_thwaites
from boundary_layer_transition.operating_point import GenericSailplaneOperatingPoint, default_operating_point
from boundary_layer_transition.stability import solve_amplification

STATIONS_S = [0.02, 0.05, 0.0885, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.43]
SENSITIVITY_V_INF = [20.0, 25.0, 30.0]


def print_main_table(op, asol) -> None:
    header = (
        f"{'x/c':>6} {'Ue/Vinf':>8} {'theta[mm]':>10} {'H':>7} {'lambda':>9} "
        f"{'Re_theta':>9} {'amp?':>6} {'dN/d(x/c)':>10} {'N':>8}"
    )
    print(header)
    for s in STATIONS_S:
        i = int(np.argmin(np.abs(asol.s - s)))
        if not asol.valid[i]:
            print(f"{asol.s[i]:6.3f}  -- beyond M1 valid/attached-laminar domain --")
            continue
        active = "yes" if asol.amplification_active[i] else "no"
        print(
            f"{asol.s[i]:6.3f} {asol.u_e_mps[i]/op.v_inf_mps:8.4f} {1000*asol.theta_m[i]:10.4f} "
            f"{asol.h[i]:7.3f} {asol.lam[i]:9.4f} {asol.re_theta[i]:9.2f} {active:>6} "
            f"{asol.dn_ds[i]:10.4f} {asol.n[i]:8.4f}"
        )


def print_summary(op, asol) -> None:
    print()
    print(f"Chord Reynolds number, Re_c                 = {op.re_c:.4e}")
    if asol.onset_s is not None:
        print(f"Modeled instability onset, x/c              = {asol.onset_s:.4f} "
              f"(x = {1000*asol.onset_x_m:.2f} mm)")
    else:
        print("Modeled instability onset                    : not reached within the valid domain")
    print(f"M1 laminar-separation diagnostic, x/c        = {asol.s_sep:.4f} "
          f"(x = {1000*asol.x_sep_m:.2f} mm)")
    if asol.onset_s is not None:
        length = asol.s_sep - asol.onset_s
        print(f"Available amplification length, Delta(x/c)  = {length:.4f}")
    print(f"Maximum accumulated N before separation      = {asol.n_max:.4f}")
    print()
    print("No N_crit is selected in Milestone 2; therefore no transition location is predicted.")
    print("The modeled instability onset is NOT a transition point, and the laminar-separation")
    print("diagnostic is NOT a transition point either -- both are amplification-tracking /")
    print("boundary-layer diagnostics only.")


def run_sensitivity() -> list[dict]:
    rows = []
    base = default_operating_point()
    for v_inf in SENSITIVITY_V_INF:
        op_v = GenericSailplaneOperatingPoint(
            chord_m=base.chord_m,
            v_inf_mps=v_inf,
            rho_kgm3=base.rho_kgm3,
            mu_pas=base.mu_pas,
            temperature_k=base.temperature_k,
            alpha_deg=base.alpha_deg,
        )
        dist_v = default_velocity_distribution(op_v.chord_m, op_v.v_inf_mps)
        tsol_v = solve_thwaites(dist_v, op_v.nu_m2s, n_points=4001)
        asol_v = solve_amplification(op_v, tsol_v)
        rows.append(
            dict(
                v_inf=v_inf,
                re_c=op_v.re_c,
                onset_s=asol_v.onset_s,
                s_sep=asol_v.s_sep,
                n_max=asol_v.n_max,
            )
        )
    return rows


def print_sensitivity_table(rows: list[dict]) -> None:
    print()
    print("-" * 78)
    print("V_inf / Re_c sensitivity (geometry and U_e/V_inf shape held fixed)")
    print("-" * 78)
    header = f"{'V_inf [m/s]':>11} {'Re_c':>11} {'onset x/c':>10} {'separation x/c':>15} {'N_max':>8}"
    print(header)
    for row in rows:
        onset_str = f"{row['onset_s']:.4f}" if row["onset_s"] is not None else "  n/a  "
        print(
            f"{row['v_inf']:11.2f} {row['re_c']:11.3e} {onset_str:>10} "
            f"{row['s_sep']:15.4f} {row['n_max']:8.4f}"
        )
    print()
    s_seps = [row["s_sep"] for row in rows]
    spread = max(s_seps) - min(s_seps)
    print(textwrap.fill(
        f"Separation x/c spread across V_inf cases: {spread:.4f} (x/c units). "
        "The synthetic U_e(x/c) shape is held fixed in x/c (Reynolds-number-independent), "
        "and Thwaites' lambda = (theta^2/nu) dU_e/dx depends on Re_c only through the "
        "theta ~ sqrt(nu/Re) scaling that cancels in lambda's nondimensional combination; "
        "the separation criterion (a fixed lambda value) is therefore expected to occur at "
        "a nearly Re_c-independent x/c for this laminar, incompressible, geometrically similar "
        "family of cases. This is the classical Thwaites/Falkner-Skan similarity result, not an "
        "artifact -- it is reported explicitly rather than hidden.",
        width=90,
    ))
    print(textwrap.fill(
        "The modeled instability onset and N_max, by contrast, depend on Re_theta directly "
        "(Re_theta scales with sqrt(Re_c) at fixed x/c for this family), so onset x/c and the "
        "accumulated N before separation DO vary with V_inf/Re_c, as shown above.",
        width=90,
    ))


def main() -> None:
    op = default_operating_point()
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    tsol = solve_thwaites(dist, op.nu_m2s, n_points=4001)
    asol = solve_amplification(op, tsol)

    print("=" * 92)
    print("MILESTONE 2 -- Reduced-order e^N amplification tracking (generic sailplane section)")
    print("Reduced-order proxy; NOT an Orr-Sommerfeld/PSE solution. See DESIGN.md.")
    print("=" * 92)
    print_main_table(op, asol)
    print_summary(op, asol)

    rows = run_sensitivity()
    print_sensitivity_table(rows)


if __name__ == "__main__":
    main()
