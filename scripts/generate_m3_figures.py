#!/usr/bin/env python3
"""Regenerate all Milestone 3 (N_crit transition / drag) figures deterministically.

Run with:

    python3 scripts/generate_m3_figures.py

Figures are written to figures/*.png. Regeneration is deterministic: no
randomness is used anywhere in the plotted quantities, and Matplotlib's
Agg backend with fixed figure/DPI settings produces byte-identical PNGs
across repeated runs on the same machine/library versions.

This script does not touch the Milestone 1 or Milestone 2 figures.
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

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
DPI = 150

CAVEAT = "Generic reduced-order model — illustrative only, not experimentally validated."
N_CRIT_VALUES = [9.0, 12.0, 14.0]
N_CRIT_LABELS = {9.0: "avg. wind tunnel", 12.0: "sailplane, lower", 14.0: "sailplane, upper"}
SENSITIVITY_V_INF = [20.0, 25.0, 30.0, 40.0, 60.0, 90.0]
PRESCRIBED_XTR = [0.1, 0.3, 0.5, 0.7, 0.9]
N_POINTS = 4001

ONSET_COLOR = "#1a7a3c"
SEP_COLOR = "#a83232"
CURVE_COLOR = "#1f5fa8"
NCRIT_COLORS = ["#6a3d9a", "#b15928", "#33a02c"]


def _add_caveat(fig: plt.Figure, extra: str | None = None) -> None:
    text = CAVEAT if extra is None else f"{CAVEAT}\n{extra}"
    fig.text(0.5, 0.010, text, ha="center", va="bottom", fontsize=8.5, style="italic", color="#7a1f1f")


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
# A. ncrit_transition_sensitivity.png
# ---------------------------------------------------------------------------


def plot_ncrit_transition_sensitivity(op, asol) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    valid = asol.valid
    ax.plot(asol.s[valid], asol.n[valid], color=CURVE_COLOR, lw=2.2, label=r"$N(x/c)$, baseline")

    any_crossing = False
    for n_crit, color in zip(N_CRIT_VALUES, NCRIT_COLORS, strict=True):
        label = rf"$N_{{crit}}={n_crit:.0f}$" + f" ({N_CRIT_LABELS[n_crit]})"
        ax.axhline(n_crit, color=color, ls="--", lw=1.3, label=label)
        r = locate_transition_n_crit(asol, n_crit)
        if r.status == TransitionStatus.N_CRIT_CROSSING:
            any_crossing = True
            ax.plot(r.x_tr_s, n_crit, "o", color=color, ms=7, zorder=5)

    ax.axvline(asol.onset_s, color=ONSET_COLOR, ls=":", lw=1.3)
    ax.axvline(asol.s_sep, color=SEP_COLOR, ls=":", lw=1.6)
    ax.text(
        asol.s_sep + 0.008, asol.n_max * 0.15,
        f"laminar-separation\ndiagnostic, $x_{{sep}}/c={asol.s_sep:.3f}$",
        fontsize=8, color=SEP_COLOR, ha="left",
    )
    ax.text(
        asol.onset_s - 0.008, asol.n_max * 0.85,
        f"modeled onset\n$x/c={asol.onset_s:.3f}$",
        fontsize=8, color=ONSET_COLOR, ha="right",
    )

    if not any_crossing:
        ax.text(
            0.97, 0.30, "Baseline separates before\nselected N_crit values.",
            transform=ax.transAxes, ha="right", va="center", fontsize=10.5, color=SEP_COLOR,
            bbox=dict(boxstyle="round", fc="#fdecea", ec=SEP_COLOR, lw=1.0),
        )

    ax.set_xlabel(r"nondimensional chordwise coordinate, $x/c$")
    ax.set_ylabel(r"amplification factor, $N = \ln(A/A_0)$")
    ax.set_title(
        "N-Factor vs. Sourced $N_{crit}$ Sensitivity Range (baseline operating point)\n"
        "No transition is predicted where no crossing occurs."
    )
    ax.set_xlim(0.0, min(1.0, asol.s_sep * 1.4))
    ax.set_ylim(0.0, max(N_CRIT_VALUES) * 1.15)
    ax.legend(loc="upper left", fontsize=8, frameon=True, ncol=1)
    ax.grid(alpha=0.25)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.85, bottom=0.13)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "ncrit_transition_sensitivity.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# B. transition_reynolds_map.png
# ---------------------------------------------------------------------------


def plot_transition_reynolds_map() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.0))

    marker_for_status = {
        TransitionStatus.N_CRIT_CROSSING: ("o", "#1a7a3c", "N_crit crossing"),
        TransitionStatus.SEPARATION_BEFORE_N_CRIT: ("x", SEP_COLOR, "separation before N_crit"),
        TransitionStatus.NO_EVENT_IN_DOMAIN: ("s", "0.5", "no event in domain"),
    }
    seen_labels = set()

    for v in SENSITIVITY_V_INF:
        _, _, _, asol = build_case(v)
        for n_crit in N_CRIT_VALUES:
            r = locate_transition_n_crit(asol, n_crit)
            marker, color, label = marker_for_status[r.status]
            plot_label = label if label not in seen_labels else None
            seen_labels.add(label)
            ax.scatter(v, n_crit, marker=marker, color=color, s=70, label=plot_label, zorder=3)
            if r.status == TransitionStatus.N_CRIT_CROSSING:
                ax.annotate(
                    f"{r.x_tr_s:.2f}", (v, n_crit), textcoords="offset points",
                    xytext=(0, 8), fontsize=7.5, ha="center", color=color,
                )

    ax.set_xlabel(r"$V_\infty$ [m/s]")
    ax.set_ylabel(r"$N_{crit}$")
    ax.set_title(
        "Transition-Event Outcome vs. Reynolds Number and $N_{crit}$\n"
        "Markers annotated with crossing $x_{tr}/c$ where a crossing occurs"
    )
    ax.set_xlim(min(SENSITIVITY_V_INF) - 5, max(SENSITIVITY_V_INF) + 8)
    ax.set_ylim(min(N_CRIT_VALUES) - 1.5, max(N_CRIT_VALUES) + 1.5)
    ax.set_yticks(N_CRIT_VALUES)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    # Secondary top axis showing Re_c for each V_inf tick, for context.
    base = default_operating_point()
    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks(SENSITIVITY_V_INF)
    re_c_labels = [f"{base.rho_kgm3 * v * base.chord_m / base.mu_pas:.1e}" for v in SENSITIVITY_V_INF]
    ax2.set_xticklabels(re_c_labels, fontsize=7, rotation=30)
    ax2.set_xlabel(r"$Re_c$ (chord Reynolds number)", fontsize=9)

    fig.subplots_adjust(left=0.11, right=0.97, top=0.80, bottom=0.22)
    _add_caveat(
        fig,
        "This is an illustrative event map for this project's reduced-order proxy — "
        "not a certification or flight-envelope statement.",
    )
    fig.savefig(FIGURES_DIR / "transition_reynolds_map.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# C. skin_friction_distributions.png
# ---------------------------------------------------------------------------


def plot_skin_friction_distributions(op, asol) -> None:
    s = build_drag_grid(op.chord_m, n_points=1001)
    x = s * op.chord_m

    scenario = separation_triggered_transition(asol)

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    cases = [
        (math.inf, "fully laminar", "0.35", "-"),
        (0.0, "fully turbulent", "0.05", "-"),
        (scenario.x_tr_m, rf"separation-triggered, $x_{{tr}}/c={scenario.x_tr_s:.3f}$", SEP_COLOR, "--"),
        (0.3 * op.chord_m, r"prescribed $x_{tr}/c=0.30$", "#1f5fa8", "-."),
        (0.7 * op.chord_m, r"prescribed $x_{tr}/c=0.70$", "#e08214", "-."),
    ]
    for x_tr_m, label, color, ls in cases:
        cf = transitioned_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m=x_tr_m)
        ax.plot(s, cf, color=color, lw=1.8, ls=ls, label=label)
        if math.isfinite(x_tr_m) and 0.0 < x_tr_m < op.chord_m:
            ax.axvline(x_tr_m / op.chord_m, color=color, lw=0.8, ls=":", alpha=0.6)

    ax.set_yscale("log")
    ax.set_xlabel(r"nondimensional chordwise coordinate, $x/c$")
    ax.set_ylabel(r"local skin-friction coefficient, $C_f$")
    ax.set_title(
        "Piecewise Laminar/Turbulent Skin-Friction Distributions\n"
        "Vertical model-switch lines are a correlation switch,\n"
        "NOT physical instantaneous boundary-layer equilibration",
        fontsize=11.5,
    )
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.grid(alpha=0.25, which="both")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.78, bottom=0.12)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "skin_friction_distributions.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# D. transition_drag_impact.png
# ---------------------------------------------------------------------------


def plot_transition_drag_impact(op, dist, asol) -> None:
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

    xtr_curve = np.array(PRESCRIBED_XTR + [scenario.x_tr_s])
    xtr_curve.sort()
    cd_curve = np.array([cd_for(xt * op.chord_m) for xt in xtr_curve])

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.plot(xtr_curve, cd_curve, "o-", color=CURVE_COLOR, lw=2.0, ms=6, label=r"prescribed $x_{tr}/c$ sweep")
    ax.axhline(cd_lam, color="0.35", ls="--", lw=1.4, label=r"fully laminar reference")
    ax.axhline(cd_turb, color="0.05", ls="--", lw=1.4, label=r"fully turbulent reference")
    ax.plot(scenario.x_tr_s, cd_sep, "D", color=SEP_COLOR, ms=9, zorder=5,
            label=rf"separation-triggered, $x_{{tr}}/c={scenario.x_tr_s:.3f}$")

    ax.set_xlabel(r"assumed transition location, $x_{tr}/c$")
    ax.set_ylabel(r"section skin-friction drag coefficient, $C_{d,f}$")
    ax.set_title(
        "Transition-Location Sensitivity of Section Skin-Friction Drag\n"
        "Reduced-order flat-plate bookkeeping, not a full turbulent BL solution"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, cd_turb * 1.15)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax.grid(alpha=0.25)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.12)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "transition_drag_impact.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    op = default_operating_point()
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    tsol = solve_thwaites(dist, op.nu_m2s, n_points=N_POINTS)
    asol = solve_amplification(op, tsol)

    plot_ncrit_transition_sensitivity(op, asol)
    plot_transition_reynolds_map()
    plot_skin_friction_distributions(op, asol)
    plot_transition_drag_impact(op, dist, asol)
    print(f"Wrote Milestone 3 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
