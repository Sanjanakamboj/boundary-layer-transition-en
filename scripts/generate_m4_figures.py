#!/usr/bin/env python3
"""Regenerate all Milestone 4 (pressure-gradient turbulent BL) figures
deterministically.

Run with:

    python3 scripts/generate_m4_figures.py

This script does not touch the Milestone 1-3 figures.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from boundary_layer_transition.external_flow import default_velocity_distribution
from boundary_layer_transition.laminar_bl import solve_thwaites
from boundary_layer_transition.operating_point import GenericSailplaneOperatingPoint, default_operating_point
from boundary_layer_transition.skin_friction import build_drag_grid, section_cf_drag, transitioned_cf_distribution
from boundary_layer_transition.stability import solve_amplification
from boundary_layer_transition.transition import separation_triggered_transition
from boundary_layer_transition.turbulent_bl import (
    TurbulentBLStatus,
    pressure_gradient_cf_distribution,
    solve_turbulent_bl,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
DPI = 150
N_POINTS = 4001

CAVEAT = "Generic reduced-order model — illustrative only, not experimentally validated."
LAM_COLOR = "0.35"
M3_COLOR = "#e08214"
M4_COLOR = "#1f5fa8"
SEP_COLOR = "#a83232"
TURB_SEP_COLOR = "#7a1fa8"


def _add_caveat(fig: plt.Figure, extra: str | None = None) -> None:
    text = CAVEAT if extra is None else f"{CAVEAT}\n{extra}"
    fig.text(0.5, 0.008, text, ha="center", va="bottom", fontsize=8.5, style="italic", color="#7a1f1f")


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


# ---------------------------------------------------------------------------
# A. turbulent_boundary_layer_history.png
# ---------------------------------------------------------------------------


def plot_turbulent_bl_history(op, dist, sep, tb) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.8, 8.8), sharex=True)
    ax_theta, ax_h, ax_cf = axes

    valid = tb.valid
    ax_theta.plot(tb.s[valid], tb.theta_m[valid] / op.chord_m, color=M4_COLOR, lw=2.0)
    ax_h.plot(tb.s[valid], tb.h[valid], color=M4_COLOR, lw=2.0)
    ax_cf.plot(tb.s[valid], tb.cf[valid], color=M4_COLOR, lw=2.0)
    ax_cf.set_yscale("log")

    peak_s = dist.s_peak
    for ax in axes:
        ax.axvline(sep.x_tr_s, color=SEP_COLOR, ls="--", lw=1.3)
        ax.axvline(peak_s, color="0.7", ls=":", lw=1.0)
        if tb.x_turb_sep_m is not None:
            ax.axvline(tb.x_turb_sep_m / op.chord_m, color=TURB_SEP_COLOR, ls="-.", lw=1.4)

    ax_theta.text(
        sep.x_tr_s + 0.01, ax_theta.get_ylim()[1] * 0.75,
        f"transition\n$x_{{tr}}/c={sep.x_tr_s:.3f}$", fontsize=8, color=SEP_COLOR,
    )
    if tb.x_turb_sep_m is not None:
        s_turb_sep = tb.x_turb_sep_m / op.chord_m
        ax_theta.text(
            s_turb_sep - 0.015, ax_theta.get_ylim()[1] * 0.4,
            f"turbulent separation\n$x/c={s_turb_sep:.3f}$", fontsize=8, color=TURB_SEP_COLOR, ha="right",
        )
    ax_theta.text(peak_s - 0.01, ax_theta.get_ylim()[1] * 0.15, "$U_e$ peak /\ndeceleration begins",
                  fontsize=7.5, color="0.4", ha="right")

    ax_theta.set_ylabel(r"$\theta/c$")
    ax_theta.set_title(
        "Pressure-Gradient Turbulent Boundary-Layer History\n"
        "(Head's entrainment method, separation-triggered transition)"
    )
    ax_theta.grid(alpha=0.25)

    ax_h.set_ylabel(r"shape factor, $H$")
    ax_h.grid(alpha=0.25)

    ax_cf.set_ylabel(r"$C_f$")
    ax_cf.set_xlabel(r"nondimensional chordwise coordinate, $x/c$")
    ax_cf.grid(alpha=0.25, which="both")
    ax_cf.set_xlim(0.0, min(1.0, (tb.x_turb_sep_m / op.chord_m if tb.x_turb_sep_m else 1.0) * 1.15))

    fig.subplots_adjust(left=0.13, right=0.97, top=0.90, bottom=0.08, hspace=0.10)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "turbulent_boundary_layer_history.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# B. m3_vs_m4_skin_friction.png
# ---------------------------------------------------------------------------


def plot_m3_vs_m4_skin_friction(op, dist, sep, tb) -> None:
    s = build_drag_grid(op.chord_m, n_points=1001)
    x = s * op.chord_m

    cf_m3 = transitioned_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m=sep.x_tr_m)
    cf_m4 = pressure_gradient_cf_distribution(op.v_inf_mps, x, op.nu_m2s, sep.x_tr_m, tb)

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.plot(s, cf_m3, color=M3_COLOR, lw=1.8, ls="--", label="M3 flat-plate bookkeeping")
    ax.plot(s, cf_m4, color=M4_COLOR, lw=2.0, label="M4 pressure-gradient integral (Head)")
    ax.axvline(sep.x_tr_s, color=SEP_COLOR, ls=":", lw=1.3)
    ax.text(sep.x_tr_s + 0.01, 0.3, f"transition, $x_{{tr}}/c={sep.x_tr_s:.3f}$", fontsize=8, color=SEP_COLOR,
            rotation=90, va="top")
    if tb.x_turb_sep_m is not None:
        s_turb_sep = tb.x_turb_sep_m / op.chord_m
        ax.axvline(s_turb_sep, color=TURB_SEP_COLOR, ls="-.", lw=1.3)
        ax.text(s_turb_sep + 0.01, 0.3, f"M4 turbulent separation\n$x/c={s_turb_sep:.3f}$ (Cf frozen beyond)",
                fontsize=7.5, color=TURB_SEP_COLOR, rotation=90, va="top")

    ax.annotate(
        "laminar curves coincide\n(both use Blasius before transition)",
        xy=(0.15, cf_m3[np.argmin(np.abs(s - 0.15))]), xytext=(0.05, 0.02),
        fontsize=8, color="0.3", arrowprops=dict(arrowstyle="->", color="0.3", lw=0.7),
    )
    ax.annotate(
        "divergence caused by the downstream\nturbulent pressure-gradient treatment",
        xy=(0.62, cf_m4[np.argmin(np.abs(s - 0.62))]), xytext=(0.50, 0.011),
        fontsize=8, color=M4_COLOR, arrowprops=dict(arrowstyle="->", color=M4_COLOR, lw=0.8),
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"nondimensional chordwise coordinate, $x/c$")
    ax.set_ylabel(r"local skin-friction coefficient, $C_f$")
    ax.set_title("M3 Flat-Plate Bookkeeping vs. M4 Pressure-Gradient Turbulent $C_f(x)$")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(alpha=0.25, which="both")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.12)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "m3_vs_m4_skin_friction.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# C. m3_vs_m4_drag_impact.png
# ---------------------------------------------------------------------------


def plot_m3_vs_m4_drag_impact(op, dist, sep) -> None:
    s = build_drag_grid(op.chord_m, n_points=N_POINTS)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)
    tsol = solve_thwaites(dist, op.nu_m2s, n_points=N_POINTS)

    def cd3(x_tr_m: float) -> float:
        cf = transitioned_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m=x_tr_m)
        return section_cf_drag(s, cf, ue_ratio, two_surfaces=True)

    def cd4(x_tr_m: float):
        tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, x_tr_m)
        cf = pressure_gradient_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m, tb)
        return section_cf_drag(s, cf, ue_ratio, two_surfaces=True), tb.status

    xtr_grid = np.array(sorted({0.05, 0.10, 0.20, 0.25, 0.30, 0.40, sep.x_tr_s}))
    cd3_vals = np.array([cd3(xt * op.chord_m) for xt in xtr_grid])
    cd4_vals_status = [cd4(xt * op.chord_m) for xt in xtr_grid]
    cd4_vals = np.array([v[0] for v in cd4_vals_status])

    cd_lam = cd3(math.inf)
    cd_turb_m3 = cd3(0.0)

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.plot(xtr_grid, cd3_vals, "o--", color=M3_COLOR, lw=1.8, ms=6, label="M3 flat-plate bookkeeping")
    ax.plot(xtr_grid, cd4_vals, "o-", color=M4_COLOR, lw=2.0, ms=6, label="M4 pressure-gradient integral")
    ax.axhline(cd_lam, color=LAM_COLOR, ls=":", lw=1.3, label="fully laminar reference")
    ax.axhline(cd_turb_m3, color="0.05", ls=":", lw=1.3, label="fully turbulent (M3 flat-plate) reference")

    i_sep = int(np.argmin(np.abs(xtr_grid - sep.x_tr_s)))
    ax.plot(xtr_grid[i_sep], cd4_vals[i_sep], "D", color=SEP_COLOR, ms=10, zorder=5,
            label=rf"separation-triggered, $x_{{tr}}/c={sep.x_tr_s:.3f}$")

    ax.set_xlabel(r"assumed transition location, $x_{tr}/c$")
    ax.set_ylabel(r"section skin-friction drag coefficient, $C_{d,f}$")
    ax.set_title("M3 vs. M4 Skin-Friction Drag vs. Assumed Transition Location")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, cd_turb_m3 * 1.15)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.grid(alpha=0.25)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.12)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "m3_vs_m4_drag_impact.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# D. turbulent_reynolds_sensitivity.png
# ---------------------------------------------------------------------------


def plot_turbulent_reynolds_sensitivity() -> None:
    v_infs = [20.0, 25.0, 30.0, 40.0, 60.0, 90.0]
    re_cs, s_turb_seps, cd4s, statuses = [], [], [], []

    for v in v_infs:
        op_v, dist_v, tsol_v, asol_v = build_case(v)
        sep_v = separation_triggered_transition(asol_v)
        tb_v = solve_turbulent_bl(dist_v, tsol_v, op_v.v_inf_mps, op_v.nu_m2s, sep_v.x_tr_m)
        s = build_drag_grid(op_v.chord_m, n_points=N_POINTS)
        x = s * op_v.chord_m
        ue_ratio = dist_v.velocity_ratio(s)
        cf4 = pressure_gradient_cf_distribution(op_v.v_inf_mps, x, op_v.nu_m2s, sep_v.x_tr_m, tb_v)
        cd4 = section_cf_drag(s, cf4, ue_ratio, two_surfaces=True)

        re_cs.append(op_v.re_c)
        cd4s.append(cd4)
        statuses.append(tb_v.status)
        s_turb_seps.append(tb_v.x_turb_sep_m / op_v.chord_m if tb_v.x_turb_sep_m is not None else np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 7.2), sharex=True)

    ax1.plot(re_cs, cd4s, "o-", color=M4_COLOR, lw=2.0, ms=7)
    ax1.set_ylabel(r"$C_{d,f}$ (M4, separation-triggered)")
    ax1.set_title(
        r"Turbulent Outcome vs. $Re_c$ (separation-triggered transition, all cases shown)"
        "\nEvery case in this sensitivity separates turbulently before the trailing edge"
    )
    ax1.grid(alpha=0.25)

    ax2.plot(re_cs, s_turb_seps, "o-", color=TURB_SEP_COLOR, lw=2.0, ms=7,
             label="turbulent separation $x/c$")
    ax2.set_ylabel(r"turbulent separation, $x_{sep,turb}/c$")
    ax2.set_xlabel(r"$Re_c$ (chord Reynolds number)")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9, frameon=True)

    all_separated = all(s == TurbulentBLStatus.TURBULENT_SEPARATION for s in statuses)
    note = (
        "All sensitivity cases shown reach turbulent separation before the trailing edge --\n"
        "this project's synthetic adverse pressure gradient does not permit turbulent attachment to the trailing edge."
        if all_separated
        else "Mixed outcomes across this sensitivity -- see markers."
    )
    ax2.text(
        0.5, -0.32, note, transform=ax2.transAxes, ha="center", va="top", fontsize=8.5, color=TURB_SEP_COLOR,
    )

    fig.subplots_adjust(left=0.12, right=0.97, top=0.87, bottom=0.20, hspace=0.12)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "turbulent_reynolds_sensitivity.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    op = default_operating_point()
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    tsol = solve_thwaites(dist, op.nu_m2s, n_points=N_POINTS)
    asol = solve_amplification(op, tsol)
    sep = separation_triggered_transition(asol)
    tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, sep.x_tr_m)

    plot_turbulent_bl_history(op, dist, sep, tb)
    plot_m3_vs_m4_skin_friction(op, dist, sep, tb)
    plot_m3_vs_m4_drag_impact(op, dist, sep)
    plot_turbulent_reynolds_sensitivity()
    print(f"Wrote Milestone 4 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
