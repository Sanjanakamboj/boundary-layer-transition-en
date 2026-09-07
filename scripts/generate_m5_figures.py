#!/usr/bin/env python3
"""Regenerate all Milestone 5 (laminar-separation-bubble / post-separation
drag) figures deterministically.

Run with:

    python3 scripts/generate_m5_figures.py

This script does not touch the Milestone 1-4 figures.
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
from boundary_layer_transition.laminar_bl import solve_thwaites
from boundary_layer_transition.operating_point import GenericSailplaneOperatingPoint, default_operating_point
from boundary_layer_transition.separated_drag import build_drag_decomposition, m5_cf_distribution
from boundary_layer_transition.separation_bubble import (
    LONG_BUBBLE,
    NOMINAL_BUBBLE,
    OPEN_SEPARATION_PARAMS,
    SHORT_BUBBLE,
    BubbleParameters,
    BubbleStatus,
    evaluate_transition_mechanism,
)
from boundary_layer_transition.skin_friction import build_drag_grid, section_cf_drag, transitioned_cf_distribution
from boundary_layer_transition.stability import solve_amplification
from boundary_layer_transition.turbulent_bl import pressure_gradient_cf_distribution, solve_turbulent_bl

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
DPI = 150
N_POINTS = 4001
N_CRIT = 9.0

CAVEAT = "Generic reduced-order model — illustrative only, not experimentally validated."
LAM_COLOR = "0.35"
BUBBLE_COLOR = "#c9a227"
ONSET_COLOR = "#1a7a3c"
SEP_COLOR = "#a83232"
TR_COLOR = "#b15928"
REATT_COLOR = "#1f5fa8"
TURBSEP_COLOR = "#7a1fa8"


def _add_caveat(fig: plt.Figure, extra: str | None = None) -> None:
    text = CAVEAT if extra is None else f"{CAVEAT}\n{extra}"
    fig.text(0.5, 0.006, text, ha="center", va="bottom", fontsize=8.5, style="italic", color="#7a1f1f")


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


def restart_and_drag(op, dist, tsol, asol, params: BubbleParameters):
    r = evaluate_transition_mechanism(asol, tsol, N_CRIT, params)
    s = build_drag_grid(op.chord_m, n_points=N_POINTS)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)
    ue_ratio_sep = float(dist.velocity_ratio(r.x_sep_lam_s))

    if r.status == BubbleStatus.OPEN_SEPARATION:
        cf = m5_cf_distribution(op.v_inf_mps, x, op.nu_m2s, r.x_sep_lam_s * op.chord_m, None, None, None)
        cd_f = section_cf_drag(s, cf, ue_ratio, two_surfaces=True)
        decomp = build_drag_decomposition(params.label, cd_f, 1.0 - r.x_sep_lam_s, ue_ratio_sep)
        return r, None, decomp

    tb = solve_turbulent_bl(
        dist, tsol, op.v_inf_mps, op.nu_m2s, r.x_reattach_s * op.chord_m,
        theta_start_override_m=r.theta_reattach_m, h_start_override=r.h_reattach,
    )
    valid = np.where(tb.valid)[0]
    cf = m5_cf_distribution(
        op.v_inf_mps, x, op.nu_m2s, r.x_sep_lam_s * op.chord_m, r.x_reattach_s * op.chord_m,
        tb.x_m[valid], tb.cf[valid],
    )
    cd_f = section_cf_drag(s, cf, ue_ratio, two_surfaces=True)
    decomp = build_drag_decomposition(params.label, cd_f, r.bubble_length_s, ue_ratio_sep)
    return r, tb, decomp


# ---------------------------------------------------------------------------
# A. laminar_separation_bubble_scenarios.png
# ---------------------------------------------------------------------------


def plot_bubble_scenarios(op, dist, tsol, asol) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.6))

    scenarios = [
        ("short", SHORT_BUBBLE),
        ("nominal", NOMINAL_BUBBLE),
        ("long", LONG_BUBBLE),
        ("open", OPEN_SEPARATION_PARAMS),
    ]
    y_positions = {name: i for i, (name, _) in enumerate(scenarios)}

    for name, params in scenarios:
        r, tb, _ = restart_and_drag(op, dist, tsol, asol, params)
        y = y_positions[name]

        ax.plot(
            [0, r.x_sep_lam_s], [y, y], color=LAM_COLOR, lw=4, solid_capstyle="butt",
            label="laminar (attached)" if y == 0 else None,
        )
        end_bubble = r.x_reattach_s if r.x_reattach_s is not None else 1.0
        ax.plot([r.x_sep_lam_s, end_bubble], [y, y], color=BUBBLE_COLOR, lw=4, solid_capstyle="butt",
                 label="separated shear layer / bubble (unresolved)" if y == 0 else None)
        if r.x_reattach_s is not None:
            end_turb = tb.x_turb_sep_m / op.chord_m if (tb is not None and tb.x_turb_sep_m is not None) else 1.0
            ax.plot([r.x_reattach_s, end_turb], [y, y], color=REATT_COLOR, lw=4, solid_capstyle="butt",
                     label="reattached turbulent" if y == 0 else None)
            if tb is not None and tb.x_turb_sep_m is not None:
                ax.plot(tb.x_turb_sep_m / op.chord_m, y, "v", color=TURBSEP_COLOR, ms=9, zorder=5,
                        label="downstream turbulent separation" if y == 0 else None)

        ax.plot(asol.onset_s, y, "|", color=ONSET_COLOR, ms=14, mew=2)
        ax.plot(r.x_sep_lam_s, y, "o", color=SEP_COLOR, ms=7, zorder=5)
        ax.plot(r.x_tr_sep_s, y, "s", color=TR_COLOR, ms=6, zorder=5)
        if r.x_reattach_s is not None:
            ax.plot(r.x_reattach_s, y, "^", color=REATT_COLOR, ms=8, zorder=5)

    ax.plot([], [], "|", color=ONSET_COLOR, ms=14, mew=2, label="modeled instability onset")
    ax.plot([], [], "o", color=SEP_COLOR, ms=7, label=r"laminar separation, $x_{sep}$")
    ax.plot([], [], "s", color=TR_COLOR, ms=6, label=r"shear-layer transition, $x_{tr}$")
    ax.plot([], [], "^", color=REATT_COLOR, ms=8, label=r"turbulent reattachment, $x_r$")

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([f"{name}\nbubble" for name in y_positions])
    ax.set_xlabel(r"nondimensional chordwise coordinate, $x/c$")
    ax.set_title(
        "Laminar-Separation-Bubble Scenarios: Event Timelines\n"
        "Short / nominal / long / open — project sensitivity parameters,\n"
        "not a resolved simulation",
        fontsize=12.5,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.7, len(scenarios) - 0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=True)
    ax.grid(alpha=0.2, axis="x")
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.28)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "laminar_separation_bubble_scenarios.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# B. bubble_restart_boundary_layer.png
# ---------------------------------------------------------------------------


def plot_bubble_restart_history(op, dist, tsol, asol) -> None:
    r, tb, _ = restart_and_drag(op, dist, tsol, asol, NOMINAL_BUBBLE)
    valid = np.where(tb.valid)[0]

    fig, (ax_th, ax_h) = plt.subplots(2, 1, figsize=(7.8, 7.0), sharex=True)

    lam_mask = tsol.s <= r.x_sep_lam_s
    ax_th.plot(
        tsol.s[lam_mask], tsol.theta_m[lam_mask] / op.chord_m, color=LAM_COLOR, lw=2.0,
        label="laminar (M1 Thwaites)",
    )
    ax_th.axvspan(r.x_sep_lam_s, r.x_reattach_s, color=BUBBLE_COLOR, alpha=0.25,
                  label="unresolved bubble region")
    ax_th.plot(tb.s[valid], tb.theta_m[valid] / op.chord_m, color=REATT_COLOR, lw=2.0,
               label="reattached turbulent (M4 restart)")
    ax_th.plot([r.x_sep_lam_s, r.x_reattach_s], [r.theta_sep_m / op.chord_m, r.theta_reattach_m / op.chord_m],
               "o", color="0.15", ms=6, zorder=5)
    ax_th.annotate(
        rf"$\theta_{{sep}}={1000*r.theta_sep_m:.3f}$ mm", xy=(r.x_sep_lam_s, r.theta_sep_m / op.chord_m),
        xytext=(r.x_sep_lam_s - 0.05, r.theta_sep_m / op.chord_m + 0.0012), fontsize=8, ha="right",
    )
    ax_th.annotate(
        rf"$\theta_{{r}}={1000*r.theta_reattach_m:.3f}$ mm ($K_\theta \cdot \theta_{{sep}}$)",
        xy=(r.x_reattach_s, r.theta_reattach_m / op.chord_m),
        xytext=(r.x_reattach_s + 0.02, r.theta_reattach_m / op.chord_m - 0.001), fontsize=8,
    )

    ax_h.plot(tsol.s[lam_mask & (tsol.h == tsol.h)], np.where(tsol.h[lam_mask] > 0, tsol.h[lam_mask], np.nan),
              color=LAM_COLOR, lw=2.0)
    lam_h_valid = lam_mask & ~np.isnan(tsol.h)
    ax_h.plot(tsol.s[lam_h_valid], tsol.h[lam_h_valid], color=LAM_COLOR, lw=2.0)
    ax_h.axvspan(r.x_sep_lam_s, r.x_reattach_s, color=BUBBLE_COLOR, alpha=0.25)
    ax_h.plot(tb.s[valid], tb.h[valid], color=REATT_COLOR, lw=2.0)
    ax_h.plot(r.x_reattach_s, r.h_reattach, "o", color="0.15", ms=6, zorder=5)

    if tb.x_turb_sep_m is not None:
        s_turb_sep = tb.x_turb_sep_m / op.chord_m
        for ax in (ax_th, ax_h):
            ax.axvline(s_turb_sep, color=TURBSEP_COLOR, ls="-.", lw=1.4)
        ax_h.text(s_turb_sep - 0.015, ax_h.get_ylim()[1] * 0.85,
                  f"downstream turbulent\nseparation, $x/c={s_turb_sep:.3f}$", fontsize=8, color=TURBSEP_COLOR,
                  ha="right")

    ax_th.set_ylabel(r"$\theta/c$")
    ax_th.set_title(
        "Representative Bubble-Restart Boundary-Layer History (nominal bubble)\n"
        "Shaded region: unresolved bubble -- no attached-flow solution is drawn through it"
    )
    ax_th.legend(loc="upper left", fontsize=8, frameon=True)
    ax_th.grid(alpha=0.25)

    ax_h.set_ylabel(r"shape factor, $H$")
    ax_h.set_xlabel(r"nondimensional chordwise coordinate, $x/c$")
    ax_h.grid(alpha=0.25)
    ax_h.set_xlim(0.0, min(1.0, (tb.x_turb_sep_m / op.chord_m if tb.x_turb_sep_m else 0.6) * 1.2))

    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.09, hspace=0.10)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "bubble_restart_boundary_layer.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# C. m3_m4_m5_drag_comparison.png
# ---------------------------------------------------------------------------


def plot_drag_comparison(op, dist, tsol, asol) -> None:
    s = build_drag_grid(op.chord_m, n_points=N_POINTS)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)

    cf_m3 = transitioned_cf_distribution(op.v_inf_mps, x, op.nu_m2s, x_tr_m=asol.x_sep_m)
    cd_m3 = section_cf_drag(s, cf_m3, ue_ratio, two_surfaces=True)

    tb_m4 = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, asol.x_sep_m)
    cf_m4 = pressure_gradient_cf_distribution(op.v_inf_mps, x, op.nu_m2s, asol.x_sep_m, tb_m4)
    cd_m4 = section_cf_drag(s, cf_m4, ue_ratio, two_surfaces=True)

    _, _, decomp_nom = restart_and_drag(op, dist, tsol, asol, NOMINAL_BUBBLE)

    labels = ["M3\nflat-plate", "M4\npressure-grad.", "M5\n(nominal bubble)"]
    friction = [cd_m3, cd_m4, decomp_nom.cd_f_m5]
    separated = [0.0, 0.0, decomp_nom.cd_sep_m5]

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, friction, color=REATT_COLOR, label=r"$C_{d,f}$ (skin friction)")
    ax.bar(x_pos, separated, bottom=friction, color=SEP_COLOR, label=r"$C_{d,sep}$ (separated pressure/form)")

    totals = [f + sep for f, sep in zip(friction, separated, strict=True)]
    for xp, tot in zip(x_pos, totals, strict=True):
        ax.text(xp, tot + 0.0008, f"{tot:.4f}", ha="center", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"drag coefficient contribution")
    ax.set_title(
        "M3 vs. M4 vs. M5 Drag Decomposition\n"
        "(separation-triggered / nominal-bubble transition)\n"
        r"M5 total is $C_{d,f,M5} + C_{d,sep,M5}$, not a complete airfoil drag coefficient",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.grid(alpha=0.25, axis="y")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.78, bottom=0.14)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "m3_m4_m5_drag_comparison.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# D. m5_reynolds_sensitivity.png
# ---------------------------------------------------------------------------


def plot_reynolds_sensitivity() -> None:
    v_infs = [20.0, 25.0, 30.0, 45.0, 60.0, 90.0]
    re_cs, cd_f_list, cd_sep_list, cd_tot_list, mechanisms = [], [], [], [], []

    for v in v_infs:
        op_v, dist_v, tsol_v, asol_v = build_case(v)
        r = evaluate_transition_mechanism(asol_v, tsol_v, N_CRIT, NOMINAL_BUBBLE)
        re_cs.append(op_v.re_c)
        if r.status == BubbleStatus.NO_LAMINAR_SEPARATION:
            from boundary_layer_transition.transition import locate_transition_n_crit
            n_res = locate_transition_n_crit(asol_v, N_CRIT)
            tb_v = solve_turbulent_bl(dist_v, tsol_v, op_v.v_inf_mps, op_v.nu_m2s, n_res.x_tr_m)
            s = build_drag_grid(op_v.chord_m, n_points=N_POINTS)
            x = s * op_v.chord_m
            ue_ratio = dist_v.velocity_ratio(s)
            cf = pressure_gradient_cf_distribution(op_v.v_inf_mps, x, op_v.nu_m2s, n_res.x_tr_m, tb_v)
            cd_f = section_cf_drag(s, cf, ue_ratio, two_surfaces=True)
            cd_f_list.append(cd_f)
            cd_sep_list.append(0.0)
            cd_tot_list.append(cd_f)
            mechanisms.append("attached")
        else:
            _, _, decomp = restart_and_drag(op_v, dist_v, tsol_v, asol_v, NOMINAL_BUBBLE)
            cd_f_list.append(decomp.cd_f_m5)
            cd_sep_list.append(decomp.cd_sep_m5)
            cd_tot_list.append(decomp.cd_total_m5)
            mechanisms.append("reattach" if r.status == BubbleStatus.TURBULENT_REATTACHMENT else "open")

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.plot(re_cs, cd_tot_list, "o-", color="0.1", lw=2.2, ms=8, label=r"$C_{d,total,M5}$")
    ax.plot(re_cs, cd_f_list, "s--", color=REATT_COLOR, lw=1.8, ms=6, label=r"$C_{d,f,M5}$")
    ax.plot(re_cs, cd_sep_list, "^--", color=SEP_COLOR, lw=1.8, ms=6, label=r"$C_{d,sep,M5}$")

    marker_map = {"reattach": "o", "open": "X", "attached": "D"}
    color_map = {"reattach": REATT_COLOR, "open": SEP_COLOR, "attached": ONSET_COLOR}
    for re_c, cd_tot, mech in zip(re_cs, cd_tot_list, mechanisms, strict=True):
        ax.plot(re_c, cd_tot, marker_map[mech], color=color_map[mech], ms=11, mfc="none", mew=2, zorder=6)
    for mech in dict.fromkeys(mechanisms):  # de-duplicate while preserving first-seen (deterministic) order
        ax.plot([], [], marker_map[mech], color=color_map[mech], ms=9, mfc="none", mew=2, linestyle="none",
                label=f"outcome: {mech}")

    ax.set_xlabel(r"$Re_c$ (chord Reynolds number)")
    ax.set_ylabel(r"drag coefficient (nominal bubble, $N_{crit}=9$)")
    ax.set_title(r"Milestone 5 Reynolds ($Re_c$) Sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.12)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "m5_reynolds_sensitivity.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# E. m5_model_sensitivity.png (strongly encouraged)
# ---------------------------------------------------------------------------


def plot_model_sensitivity(op, dist, tsol, asol) -> None:
    def cd_total_for(params: BubbleParameters, k_sep: float = 0.5) -> float:
        r, tb, decomp = restart_and_drag(op, dist, tsol, asol, params)
        if k_sep == 0.5:
            return decomp.cd_total_m5
        # Recompute with a different k_sep, reusing the same friction contribution.
        from boundary_layer_transition.separated_drag import separated_drag_coefficient
        ue_ratio_sep = float(dist.velocity_ratio(r.x_sep_lam_s))
        l_sep = 1.0 - r.x_sep_lam_s if r.status == BubbleStatus.OPEN_SEPARATION else r.bubble_length_s
        return decomp.cd_f_m5 + separated_drag_coefficient(l_sep, ue_ratio_sep, k_sep=k_sep)

    base_cd = cd_total_for(NOMINAL_BUBBLE)

    factors = {
        r"bubble length $\Delta x_{reattach}/c$": [
            (str(dxr), cd_total_for(BubbleParameters("x", 0.008, dxr, 5.0, 1.7)))
            for dxr in (0.007, 0.022, 0.065, 0.150)
        ],
        r"$K_\theta$": [
            (str(k), cd_total_for(BubbleParameters("x", 0.008, 0.022, k, 1.7)))
            for k in (2.0, 5.0, 8.0, 12.0)
        ],
        r"$H_{reattach}$": [
            (str(h), cd_total_for(BubbleParameters("x", 0.008, 0.022, 5.0, h)))
            for h in (1.4, 1.7, 2.0, 2.3)
        ],
        r"$K_{sep}$": [
            (str(k), cd_total_for(NOMINAL_BUBBLE, k_sep=k))
            for k in (0.0, 0.25, 0.5, 1.0, 2.0)
        ],
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    y_labels = list(factors.keys())
    # Small alternating (dx, dy) offsets (in points) to separate labels for
    # tightly clustered rows (K_theta, H_reattach) without implying any
    # additional data structure -- purely a label-declutter measure.
    label_offsets = [(0, 11), (18, -14), (-18, 22), (0, -24)]
    for i, (name, points) in enumerate(factors.items()):
        pct = [100.0 * (cd - base_cd) / base_cd for _, cd in points]
        ax.scatter(pct, [i] * len(pct), s=60, color=REATT_COLOR, zorder=3)
        for j, (p, (val, _)) in enumerate(zip(pct, points, strict=True)):
            dx, dy = label_offsets[j % len(label_offsets)]
            ax.annotate(val, (p, i), textcoords="offset points", xytext=(dx, dy), fontsize=7.5, ha="center")

    ax.axvline(0.0, color="0.3", lw=1.0, ls="--")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.6, len(y_labels) - 0.4)
    ax.set_xlabel(r"percent change in $C_{d,total,M5}$ relative to the nominal-bubble baseline")
    ax.set_title(
        "One-Factor Sensitivity of Modeled Total Drag\n"
        "Points show discrete parameter settings actually evaluated --\n"
        "not a statistical confidence interval",
        fontsize=12.5,
    )
    ax.grid(alpha=0.25, axis="x")
    fig.subplots_adjust(left=0.24, right=0.97, top=0.78, bottom=0.13)
    _add_caveat(fig)
    fig.savefig(FIGURES_DIR / "m5_model_sensitivity.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    op = default_operating_point()
    dist = default_velocity_distribution(op.chord_m, op.v_inf_mps)
    tsol = solve_thwaites(dist, op.nu_m2s, n_points=N_POINTS)
    asol = solve_amplification(op, tsol)

    plot_bubble_scenarios(op, dist, tsol, asol)
    plot_bubble_restart_history(op, dist, tsol, asol)
    plot_drag_comparison(op, dist, tsol, asol)
    plot_reynolds_sensitivity()
    plot_model_sensitivity(op, dist, tsol, asol)
    print(f"Wrote Milestone 5 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
