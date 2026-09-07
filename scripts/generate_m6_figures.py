#!/usr/bin/env python3
"""Regenerate all Milestone 6 (final robustness / portfolio) figures
deterministically.

Run with:

    python3 scripts/generate_m6_figures.py

This script does not touch the Milestone 1-5 figures.
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
from boundary_layer_transition.external_flow_sensitivity import PROFILES, build_distribution
from boundary_layer_transition.laminar_bl import solve_thwaites
from boundary_layer_transition.operating_point import GenericSailplaneOperatingPoint, default_operating_point
from boundary_layer_transition.robustness import run_chain
from boundary_layer_transition.separated_drag import m5_cf_distribution, separated_drag_coefficient
from boundary_layer_transition.separation_bubble import NOMINAL_BUBBLE, BubbleParameters, evaluate_transition_mechanism
from boundary_layer_transition.skin_friction import build_drag_grid, section_cf_drag
from boundary_layer_transition.stability import solve_amplification
from boundary_layer_transition.turbulent_bl import solve_turbulent_bl

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
DPI = 150
N_POINTS = 4001

CAVEAT = "Generic reduced-order model — illustrative only, not experimentally validated."
LAM_COLOR = "0.35"
BUBBLE_COLOR = "#c9a227"
ONSET_COLOR = "#1a7a3c"
SEP_COLOR = "#a83232"
REATT_COLOR = "#1f5fa8"
TURBSEP_COLOR = "#7a1fa8"
FRICTION_COLOR = "#1f5fa8"
SEPDRAG_COLOR = "#a83232"

LABEL_BY_KEY = {
    "baseline": "baseline",
    "peak_ratio_mild": "peak ratio\n(mild, 1.10)",
    "peak_ratio_strong": "peak ratio\n(strong, 1.26)",
    "peak_location_early": "peak loc.\n(early, 0.25)",
    "peak_location_late": "peak loc.\n(late, 0.45)",
    "te_ratio_severe": "TE ratio\n(severe, 0.45)",
    "te_ratio_mild": "TE ratio\n(mild, 0.70)",
}


def _add_caveat(fig: plt.Figure, extra: str | None = None) -> None:
    text = CAVEAT if extra is None else f"{CAVEAT}\n{extra}"
    fig.text(0.5, 0.006, text, ha="center", va="bottom", fontsize=8.5, style="italic", color="#7a1f1f")


def get_baseline_case():
    op = default_operating_point()
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    tsol = solve_thwaites(dist, op.nu_m2s, n_points=N_POINTS)
    asol = solve_amplification(op, tsol)
    return op, dist, tsol, asol


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
# A. final_external_flow_sensitivity.png
# ---------------------------------------------------------------------------


def plot_external_flow_sensitivity(op) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.0))

    cases = [
        ("te_ratio_mild", "mild adverse gradient (TE ratio 0.70)", "#5aa2d8"),
        ("baseline", "baseline (TE ratio 0.55)", REATT_COLOR),
        ("te_ratio_severe", "severe adverse gradient (TE ratio 0.45)", SEP_COLOR),
    ]
    s = np.linspace(0.0, 1.0, 1001)
    results = {}
    for label, _, color in cases:
        profile = next(p for p in PROFILES if p.label == label)
        dist = build_distribution(profile, op.chord_m, op.v_inf_mps)
        ax1.plot(s, dist.velocity_ratio(s), color=color, lw=2.0, label=_)
        results[label] = run_chain(profile, op)

    ax1.set_xlabel(r"nondimensional chordwise coordinate, $x/c$")
    ax1.set_ylabel(r"$U_e/V_\infty$")
    ax1.set_title("Synthetic External-Flow Sensitivity:\nMild vs. Baseline vs. Severe Adverse Gradient")
    ax1.legend(loc="upper right", fontsize=8, frameon=True)
    ax1.grid(alpha=0.25)
    ax1.set_xlim(0.0, 1.0)

    labels = [c[0] for c in cases]
    colors = [c[2] for c in cases]
    x_seps = [results[label].x_sep_lam_s for label in labels]
    n_maxs = [results[label].n_max for label in labels]

    x_pos = np.arange(len(labels))
    width = 0.35
    ax2b = ax2.twinx()
    bars1 = ax2.bar(x_pos - width / 2, x_seps, width, color=colors, alpha=0.9, label=r"$x_{sep}/c$")
    bars2 = ax2b.bar(x_pos + width / 2, n_maxs, width, color=colors, alpha=0.45, hatch="//", label=r"$N_{max}$")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["mild", "baseline", "severe"])
    ax2.set_ylabel(r"laminar separation, $x_{sep}/c$")
    ax2b.set_ylabel(r"amplification factor, $N_{max}$")
    ax2.set_title("Laminar-Separation Location and $N_{max}$\nfor the Three Cases")
    ax2.grid(alpha=0.25, axis="y")
    ax2.set_ylim(0.0, max(x_seps) * 1.30)
    ax2b.set_ylim(0.0, max(n_maxs) * 1.30)
    for bar, val in zip(bars1, x_seps, strict=True):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.3f}", ha="center", fontsize=8)
    for bar, val in zip(bars2, n_maxs, strict=True):
        ax2b.text(bar.get_x() + bar.get_width() / 2, val + 0.08, f"{val:.2f}", ha="center", fontsize=8)

    lines = [plt.Rectangle((0, 0), 1, 1, fc="0.5"), plt.Rectangle((0, 0), 1, 1, fc="0.5", alpha=0.45, hatch="//")]
    ax2.legend(
        lines, [r"$x_{sep}/c$ (solid)", r"$N_{max}$ (hatched)"],
        loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=8, frameon=True,
    )

    fig.suptitle("Synthetic External-Flow Sensitivity — Illustrative Conceptual Inputs, Not Real-Airfoil Data",
                 fontsize=11.5)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.80, bottom=0.24, wspace=0.55)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "final_external_flow_sensitivity.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# B. final_transition_mechanism_map.png
# ---------------------------------------------------------------------------


def plot_transition_mechanism_map() -> None:
    v_infs = [25.0, 45.0, 60.0, 90.0]

    def classify(r) -> str:
        if r.mechanism == "ATTACHED_E_N_CROSSING":
            if r.turbulent_status == "COMPLETED_TO_TE":
                return "attached e^N crossing, reaches TE"
            return "attached e^N crossing, later turb. sep."
        if r.bubble_status == "OPEN_SEPARATION":
            return "open separation"
        if r.bubble_status == "TURBULENT_REATTACHMENT":
            if r.turbulent_status == "COMPLETED_TO_TE":
                return "reattached bubble, reaches TE"
            return "reattached bubble, later turb. sep."
        return "no event"

    categories = [
        "attached e^N crossing, reaches TE",
        "attached e^N crossing, later turb. sep.",
        "reattached bubble, reaches TE",
        "reattached bubble, later turb. sep.",
        "open separation",
        "no event",
    ]
    colors = {
        "attached e^N crossing, reaches TE": "#1a7a3c",
        "attached e^N crossing, later turb. sep.": "#7fbf7f",
        "reattached bubble, reaches TE": "#1f5fa8",
        "reattached bubble, later turb. sep.": "#7fa8d8",
        "open separation": "#a83232",
        "no event": "0.6",
    }
    markers = {
        "attached e^N crossing, reaches TE": "D",
        "attached e^N crossing, later turb. sep.": "d",
        "reattached bubble, reaches TE": "o",
        "reattached bubble, later turb. sep.": "o",
        "open separation": "X",
        "no event": "s",
    }

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    seen = set()
    for j, v in enumerate(v_infs):
        op_v, dist_v, tsol_v, asol_v = build_case(v)
        for i, profile in enumerate(PROFILES):
            r = run_chain(profile, op_v)
            cat = classify(r)
            marker = markers[cat]
            plot_label = cat if cat not in seen else None
            seen.add(cat)
            ax.scatter(i, j, marker=marker, s=140, color=colors[cat], edgecolor="0.1", linewidth=0.6,
                       label=plot_label, zorder=3)

    ax.set_xticks(range(len(PROFILES)))
    ax.set_xticklabels([LABEL_BY_KEY[p.label] for p in PROFILES], fontsize=8)
    ax.set_yticks(range(len(v_infs)))
    ax.set_yticklabels([f"{v:.0f} m/s" for v in v_infs])
    ax.set_xlabel("external-flow sensitivity case")
    ax.set_ylabel(r"$V_\infty$")
    ax.set_title(
        "Final Transition/Separation Mechanism Map\n"
        "External-flow sensitivity family x Reynolds sensitivity (not a flight envelope)"
    )
    ax.grid(alpha=0.2)
    handles = [plt.Line2D([0], [0], marker=markers[c], color="w", markerfacecolor=colors[c],
                           markeredgecolor="0.1", markersize=10, label=c) for c in categories if c in seen]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.6, frameon=True)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.34)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "final_transition_mechanism_map.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# C. final_drag_sensitivity.png
# ---------------------------------------------------------------------------


def plot_drag_sensitivity(op) -> None:
    results = {p.label: run_chain(p, op) for p in PROFILES}
    order = [p.label for p in PROFILES]

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    x_pos = np.arange(len(order))
    friction = [results[lbl].cd_f for lbl in order]
    separated = [results[lbl].cd_sep for lbl in order]

    ax.bar(x_pos, friction, color=FRICTION_COLOR, label=r"$C_{d,f}$ (friction)")
    ax.bar(x_pos, separated, bottom=friction, color=SEPDRAG_COLOR, label=r"$C_{d,sep}$ (separated pressure/form)")

    totals = [f + s for f, s in zip(friction, separated, strict=True)]
    for xp, tot in zip(x_pos, totals, strict=True):
        ax.text(xp, tot + 0.0004, f"{tot:.4f}", ha="center", fontsize=7.5)

    i_base = order.index("baseline")
    ax.axvline(i_base, color="0.2", ls=":", lw=1.2)
    ax.annotate("baseline", xy=(i_base, totals[i_base]), xytext=(i_base, totals[i_base] + 0.0035),
                ha="center", fontsize=8.5, arrowprops=dict(arrowstyle="->", color="0.2", lw=0.8))

    ax.set_xticks(x_pos)
    ax.set_xticklabels([LABEL_BY_KEY[lbl] for lbl in order], fontsize=7.8)
    ax.set_ylabel(r"drag coefficient contribution")
    ax.set_title(
        r"$C_{d,total}$ Sensitivity Across the External-Flow Family"
        "\n(separated-flow contribution dominates in every case shown)"
    )
    ax.set_ylim(0.0, max(totals) * 1.18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=9, frameon=True)
    ax.grid(alpha=0.25, axis="y")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.30)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "final_drag_sensitivity.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# D. final_boundary_layer_summary.png
# ---------------------------------------------------------------------------


def plot_boundary_layer_summary(op, dist, tsol, asol) -> None:
    r = evaluate_transition_mechanism(asol, tsol, 9.0, NOMINAL_BUBBLE)
    tb = solve_turbulent_bl(
        dist, tsol, op.v_inf_mps, op.nu_m2s, r.x_reattach_s * op.chord_m,
        theta_start_override_m=r.theta_reattach_m, h_start_override=r.h_reattach,
    )
    valid = np.where(tb.valid)[0]

    s_grid = build_drag_grid(op.chord_m, n_points=N_POINTS)
    x_grid = s_grid * op.chord_m
    ue_ratio_grid = dist.velocity_ratio(s_grid)
    cf_m5 = m5_cf_distribution(
        op.v_inf_mps, x_grid, op.nu_m2s, r.x_sep_lam_s * op.chord_m, r.x_reattach_s * op.chord_m,
        tb.x_m[valid], tb.cf[valid],
    )
    cd_f = section_cf_drag(s_grid, cf_m5, ue_ratio_grid, two_surfaces=True)
    ue_ratio_sep = float(dist.velocity_ratio(r.x_sep_lam_s))
    cd_sep = separated_drag_coefficient(r.bubble_length_s, ue_ratio_sep)
    cd_total = cd_f + cd_sep

    fig = plt.figure(figsize=(11.5, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Panel 1: external velocity distribution
    s = np.linspace(0.0, 1.0, 1001)
    ax1.plot(s, dist.velocity_ratio(s), color=REATT_COLOR, lw=2.0)
    ax1.axvline(dist.s_peak, color="0.7", ls=":", lw=1.0)
    ax1.set_xlabel(r"$x/c$")
    ax1.set_ylabel(r"$U_e/V_\infty$")
    ax1.set_title("1. External Velocity Distribution")
    ax1.grid(alpha=0.25)
    ax1.set_xlim(0.0, 1.0)

    # Panel 2: laminar BL / instability / separation timeline
    ax2.plot([0, asol.onset_s], [0, 0], color=LAM_COLOR, lw=6, solid_capstyle="butt")
    ax2.plot([asol.onset_s, r.x_sep_lam_s], [0, 0], color=ONSET_COLOR, lw=6, solid_capstyle="butt")
    ax2.plot(asol.onset_s, 0, "|", color="0.1", ms=16, mew=2)
    ax2.plot(r.x_sep_lam_s, 0, "o", color=SEP_COLOR, ms=9, zorder=5)
    ax2.text(asol.onset_s, 0.15, f"onset\n{asol.onset_s:.3f}", ha="center", fontsize=8)
    ax2.text(r.x_sep_lam_s, -0.25, f"laminar sep.\n{r.x_sep_lam_s:.3f}", ha="center", fontsize=8, color=SEP_COLOR)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_yticks([])
    ax2.set_xlabel(r"$x/c$")
    ax2.set_title("2. Laminar / Instability / Separation Timeline")
    ax2.grid(alpha=0.2, axis="x")

    # Panel 3: turbulent / bubble / re-separation timeline
    ax3.plot([0, r.x_sep_lam_s], [0, 0], color=LAM_COLOR, lw=6, solid_capstyle="butt")
    ax3.plot([r.x_sep_lam_s, r.x_reattach_s], [0, 0], color=BUBBLE_COLOR, lw=6, solid_capstyle="butt")
    end_turb = tb.x_turb_sep_m / op.chord_m if tb.x_turb_sep_m is not None else 1.0
    ax3.plot([r.x_reattach_s, end_turb], [0, 0], color=REATT_COLOR, lw=6, solid_capstyle="butt")
    if tb.x_turb_sep_m is not None:
        ax3.plot(end_turb, 0, "v", color=TURBSEP_COLOR, ms=10, zorder=5)
        ax3.text(end_turb, 0.15, f"turb. sep.\n{end_turb:.3f}", ha="center", fontsize=8, color=TURBSEP_COLOR)
    ax3.plot(r.x_reattach_s, 0, "^", color=REATT_COLOR, ms=9, zorder=5)
    ax3.text(r.x_reattach_s, -0.25, f"reattach\n{r.x_reattach_s:.3f}", ha="center", fontsize=8, color=REATT_COLOR)
    ax3.set_xlim(0.0, min(1.0, end_turb * 1.3))
    ax3.set_ylim(-0.6, 0.6)
    ax3.set_yticks([])
    ax3.set_xlabel(r"$x/c$")
    ax3.set_title("3. Bubble / Turbulent Reattachment Timeline")
    ax3.grid(alpha=0.2, axis="x")

    # Panel 4: drag decomposition
    ax4.bar([0], [cd_f], color=FRICTION_COLOR, label=r"$C_{d,f}$")
    ax4.bar([0], [cd_sep], bottom=[cd_f], color=SEPDRAG_COLOR, label=r"$C_{d,sep}$")
    ax4.text(0, cd_total + 0.0006, f"{cd_total:.4f}", ha="center", fontsize=9)
    ax4.set_xticks([0])
    ax4.set_xticklabels(["nominal bubble"])
    ax4.set_ylabel(r"$C_{d}$ contribution")
    ax4.set_title("4. Drag Decomposition (nominal bubble)")
    ax4.legend(loc="upper left", fontsize=8, frameon=True)
    ax4.grid(alpha=0.25, axis="y")
    ax4.set_xlim(-0.8, 0.8)
    ax4.set_ylim(0.0, cd_total * 1.55)

    summary = (
        f"$Re_c$ = {op.re_c:.3e}\n"
        f"$x_{{instab}}/c$ = {asol.onset_s:.4f}\n"
        f"$x_{{sep,lam}}/c$ = {r.x_sep_lam_s:.4f}\n"
        f"$N_{{max}}$ = {asol.n_max:.3f}\n"
        f"$C_{{d,total}}$ (nominal) = {cd_total:.4f}\n"
        "dominant sensitivity: $K_{sep}$"
    )
    # Placed inside panel 4's own (mostly empty) axes area, rather than as
    # a figure-level overlay, so it cannot clip a neighboring panel title.
    ax4.text(
        0.97, 0.97, summary, transform=ax4.transAxes, ha="right", va="top", fontsize=8.8,
        bbox=dict(boxstyle="round", fc="#f5f5f0", ec="0.4", lw=1.0),
    )

    fig.suptitle(
        "Generic Reduced-Order Transition/Separation Model — Not Experimentally Validated",
        fontsize=12.5, y=0.995,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.08)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "final_boundary_layer_summary.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# E. final_model_sensitivity_ranking.png (optional, encouraged)
# ---------------------------------------------------------------------------


def plot_model_sensitivity_ranking(op) -> None:
    dist_baseline = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    tsol_baseline = solve_thwaites(dist_baseline, op.nu_m2s, n_points=N_POINTS)
    asol_baseline = solve_amplification(op, tsol_baseline)
    results = {p.label: run_chain(p, op) for p in PROFILES}
    base_cd = results["baseline"].cd_total

    def cd_total_bubble(params: BubbleParameters, k_sep: float = 0.5) -> float:
        r = evaluate_transition_mechanism(asol_baseline, tsol_baseline, 9.0, params)
        s = build_drag_grid(op.chord_m, n_points=N_POINTS)
        x = s * op.chord_m
        ue_ratio = dist_baseline.velocity_ratio(s)
        ue_ratio_sep = float(dist_baseline.velocity_ratio(r.x_sep_lam_s))
        tb = solve_turbulent_bl(
            dist_baseline, tsol_baseline, op.v_inf_mps, op.nu_m2s, r.x_reattach_s * op.chord_m,
            theta_start_override_m=r.theta_reattach_m, h_start_override=r.h_reattach,
        )
        valid = np.where(tb.valid)[0]
        cf = m5_cf_distribution(
            op.v_inf_mps, x, op.nu_m2s, r.x_sep_lam_s * op.chord_m, r.x_reattach_s * op.chord_m,
            tb.x_m[valid], tb.cf[valid],
        )
        cd_f = section_cf_drag(s, cf, ue_ratio, two_surfaces=True)
        return cd_f + separated_drag_coefficient(r.bubble_length_s, ue_ratio_sep, k_sep=k_sep)

    swings = {
        "M6 peak ratio": max(abs(100 * (results[lbl].cd_total - base_cd) / base_cd)
                              for lbl in ("peak_ratio_mild", "peak_ratio_strong")),
        "M6 peak location": max(abs(100 * (results[lbl].cd_total - base_cd) / base_cd)
                                 for lbl in ("peak_location_early", "peak_location_late")),
        "M6 TE ratio": max(abs(100 * (results[lbl].cd_total - base_cd) / base_cd)
                            for lbl in ("te_ratio_severe", "te_ratio_mild")),
        "M5 bubble length": max(
            abs(100 * (cd_total_bubble(BubbleParameters("x", 0.008, dxr, 5.0, 1.7)) - base_cd) / base_cd)
            for dxr in (0.007, 0.065)
        ),
        "M5 K_theta": max(
            abs(100 * (cd_total_bubble(BubbleParameters("x", 0.008, 0.022, k, 1.7)) - base_cd) / base_cd)
            for k in (2.0, 12.0)
        ),
        "M5 H_reattach": max(
            abs(100 * (cd_total_bubble(BubbleParameters("x", 0.008, 0.022, 5.0, h)) - base_cd) / base_cd)
            for h in (1.4, 2.3)
        ),
        "M5 K_sep": max(
            abs(100 * (cd_total_bubble(NOMINAL_BUBBLE, k_sep=k) - base_cd) / base_cd)
            for k in (0.0, 2.0)
        ),
    }
    ranked = sorted(swings.items(), key=lambda kv: kv[1])

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    names = [k for k, _ in ranked]
    values = [v for _, v in ranked]
    colors = ["#a83232" if "M5" in n else "#1f5fa8" for n in names]
    ax.barh(names, values, color=colors)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, f"{v:.1f}%", va="center", fontsize=8.5)

    ax.set_xlabel(r"max $|\Delta\%\ C_{d,total}|$ (deterministic one-factor sensitivity)")
    ax.set_title(
        "Deterministic One-Factor Sensitivity Ranking\n(not confidence intervals)"
    )
    ax.grid(alpha=0.25, axis="x")
    fig.subplots_adjust(left=0.24, right=0.95, top=0.85, bottom=0.13)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "final_model_sensitivity_ranking.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    op, dist, tsol, asol = get_baseline_case()

    plot_external_flow_sensitivity(op)
    plot_transition_mechanism_map()
    plot_drag_sensitivity(op)
    plot_boundary_layer_summary(op, dist, tsol, asol)
    plot_model_sensitivity_ranking(op)
    print(f"Wrote Milestone 6 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
