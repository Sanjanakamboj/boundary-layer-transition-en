#!/usr/bin/env python3
"""Regenerate all Milestone 1 figures deterministically.

Run with:

    python3 scripts/generate_figures.py

Figures are written to figures/*.png. Regeneration is deterministic: no
randomness is used anywhere in the plotted quantities, and Matplotlib's
Agg backend with fixed figure/DPI settings produces byte-identical PNGs
across repeated runs on the same machine/library versions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
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

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
DPI = 150

CAVEAT = (
    "Generic reduced-order model — illustrative only, not experimentally validated."
)


def _add_caveat(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        0.008,
        CAVEAT,
        ha="center",
        va="bottom",
        fontsize=8.5,
        style="italic",
        color="#7a1f1f",
    )


def plot_external_velocity(dist, op) -> None:
    s = np.linspace(0.0, 1.0, 1001)
    ratio = dist.velocity_ratio(s)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(s, ratio, color="#1f5fa8", lw=2.2, label=r"$U_e(x)/V_\infty$ (synthetic, illustrative)")
    ax.axvline(dist.s_peak, color="0.4", ls="--", lw=1.0)
    ax.axhline(1.0, color="0.75", ls=":", lw=1.0)

    ax.annotate(
        "favorable gradient\n" + r"($dU_e/dx > 0$)",
        xy=(dist.s_peak * 0.45, 1.05),
        ha="center",
        fontsize=9.5,
        color="#1f5fa8",
    )
    ax.annotate(
        "adverse gradient\n" + r"($dU_e/dx < 0$)",
        xy=((1 + dist.s_peak) / 2, dist.u_peak_ratio * 0.92),
        ha="center",
        fontsize=9.5,
        color="#a83232",
    )
    ax.annotate(
        f"peak, $x/c={dist.s_peak:.2f}$",
        xy=(dist.s_peak, dist.u_peak_ratio),
        xytext=(dist.s_peak + 0.08, dist.u_peak_ratio + 0.03),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="0.3", lw=0.9),
    )

    ax.set_xlabel(r"nondimensional chordwise coordinate, $x/c$")
    ax.set_ylabel(r"$U_e / V_\infty$")
    ax.set_title(
        "Synthetic External Velocity Distribution\n"
        f"(generic sailplane section, $Re_c={op.re_c:.2e}$, $M={op.mach:.3f}$)"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.35)
    ax.legend(loc="lower left", fontsize=9.5, frameon=True)
    ax.grid(alpha=0.25)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.87, bottom=0.11)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "external_velocity_distribution.png", dpi=DPI)
    plt.close(fig)


def plot_bl_growth(dist, op, sol) -> None:
    x = sol.x_m
    theta_blasius = blasius_theta(op.v_inf_mps, x, op.nu_m2s)
    dstar_blasius = blasius_delta_star(op.v_inf_mps, x, op.nu_m2s)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    # Thwaites results are only physically meaningful up to the predicted
    # separation station; beyond it the curves are shown dashed/greyed as
    # an explicit "invalid beyond this point" visual cue.
    pre = ~sol.separated
    ax.plot(sol.s[pre], sol.theta_m[pre] / op.chord_m, color="#1f5fa8", lw=2.0, label=r"$\theta/c$ (Thwaites)")
    ax.plot(
        sol.s[pre],
        np.where(sol.valid[pre], sol.delta_star_m[pre], np.nan) / op.chord_m,
        color="#1f5fa8",
        lw=2.0,
        ls="--",
        label=r"$\delta^*/c$ (Thwaites)",
    )
    ax.plot(sol.s, theta_blasius / op.chord_m, color="0.45", lw=1.6, label=r"$\theta/c$ (Blasius ZPG ref.)")
    ax.plot(sol.s, dstar_blasius / op.chord_m, color="0.45", lw=1.6, ls="--", label=r"$\delta^*/c$ (Blasius ZPG ref.)")

    if sol.s_sep is not None:
        ax.axvline(sol.s_sep, color="#a83232", ls=":", lw=1.4)
        ax.annotate(
            f"predicted laminar\nseparation, $x/c={sol.s_sep:.3f}$\n(diagnostic only)",
            xy=(sol.s_sep, 0.0),
            xytext=(sol.s_sep + 0.04, 0.008),
            fontsize=8.5,
            color="#a83232",
        )
        post = sol.separated
        ax.plot(sol.s[post], sol.theta_m[post] / op.chord_m, color="#a83232", lw=1.2, ls=":", alpha=0.6,
                 label=r"$\theta/c$ beyond predicted separation (not physically meaningful)")

    ax.set_xlabel(r"nondimensional chordwise coordinate, $x/c$")
    ax.set_ylabel(r"thickness $/\, c$  (dimensionless)")
    ax.set_title(
        "Laminar Boundary-Layer Growth\n"
        r"Momentum ($\theta$) and displacement ($\delta^*$) thickness, chord $c="
        f"{op.chord_m:.2f}$ m"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", fontsize=7.6, frameon=True, ncol=1)
    ax.grid(alpha=0.25)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.11)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "laminar_boundary_layer_growth.png", dpi=DPI)
    plt.close(fig)


def plot_shape_factor(sol) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.6), sharex=True)

    pre = sol.valid & ~sol.separated
    ax1.plot(sol.s[pre], sol.h[pre], color="#1f5fa8", lw=2.0, label=r"$H(\lambda)$ (Thwaites)")
    ax1.axhline(BLASIUS_H, color="0.45", ls="--", lw=1.4, label=f"Blasius ZPG reference, $H={BLASIUS_H:.3f}$")

    ax2.plot(sol.s[~sol.separated], sol.lam[~sol.separated], color="#1f5fa8", lw=2.0, label=r"$\lambda(x)$ (Thwaites)")
    lam_sep = thwaites_separation_lambda()
    ax2.axhline(
        lam_sep, color="#a83232", ls=":", lw=1.4,
        label=rf"separation criterion, $\lambda_{{sep}}={lam_sep:.4f}$",
    )
    ax2.axhline(0.0, color="0.75", lw=0.8)

    if sol.s_sep is not None:
        for ax in (ax1, ax2):
            ax.axvline(sol.s_sep, color="#a83232", ls=":", lw=1.2, alpha=0.8)
        ax1.text(
            sol.s_sep - 0.015,
            ax1.get_ylim()[0] + 0.06 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
            f"predicted separation, $x/c={sol.s_sep:.3f}$",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="#a83232",
        )

    ax1.set_ylabel(r"shape factor, $H$")
    ax1.set_title(
        "Laminar Shape Factor and Thwaites Pressure-Gradient Parameter\n"
        "(curves stop at the predicted separation station; see study script for detail)"
    )
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax1.grid(alpha=0.25)

    ax2.set_xlabel(r"nondimensional chordwise coordinate, $x/c$")
    ax2.set_ylabel(r"Thwaites parameter, $\lambda = (\theta^2/\nu)\,dU_e/dx$")
    ax2.set_ylim(-0.15, 0.08)
    ax2.legend(loc="lower left", fontsize=8.5, frameon=True)
    ax2.grid(alpha=0.25)

    fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.09, hspace=0.08)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "laminar_shape_factor.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    op = default_operating_point()
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    sol = solve_thwaites(dist, op.nu_m2s, n_points=4001)

    plot_external_velocity(dist, op)
    plot_bl_growth(dist, op, sol)
    plot_shape_factor(sol)
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
