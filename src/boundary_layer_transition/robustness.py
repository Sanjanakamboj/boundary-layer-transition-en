"""Milestone 6: full-chain (M1-M5) orchestration for one external-flow
sensitivity profile.

This module contains **no new physics**. It only re-invokes, in sequence
and without retuning any downstream constant, the existing Milestone 1-5
solvers for a given :class:`~.external_flow_sensitivity.SensitivityProfile`,
and collects the headline results into one immutable
:class:`ChainResult`. This is the shared implementation used by both
``scripts/final_robustness_study.py`` and the Milestone 6 test suite, so
the study script and the tests can never silently diverge in how a
profile is propagated through the model chain.

Mechanism handling exactly mirrors Milestone 5's
``scripts/separation_bubble_drag_study.py``:

- If a genuine attached-flow N_crit crossing occurs before laminar
  separation, the Milestone 5 bubble closure is bypassed (mechanism
  ``ATTACHED_E_N_CROSSING``); the Milestone 4 turbulent solver is
  restarted directly at the crossing station, and no separated-drag term
  is computed (``C_d,sep = 0``).
- Otherwise, the nominal-bubble Milestone 5 closure is evaluated
  (mechanism ``SEPARATION_INDUCED``), producing either a
  ``TURBULENT_REATTACHMENT`` (with a restarted turbulent solve and a
  finite bubble ``C_d,sep``) or an ``OPEN_SEPARATION`` outcome (no
  downstream turbulent solve; the separated-drag term spans separation to
  the trailing edge).
- If neither a crossing nor a separation occurs (``NO_LAMINAR_SEPARATION``
  with no crossing either), the case is reported with all downstream
  fields left ``None`` rather than fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .external_flow_sensitivity import SensitivityProfile, build_distribution
from .laminar_bl import solve_thwaites
from .operating_point import GenericSailplaneOperatingPoint
from .separated_drag import build_drag_decomposition, m5_cf_distribution
from .separation_bubble import NOMINAL_BUBBLE, BubbleStatus, TransitionMechanism, evaluate_transition_mechanism
from .skin_friction import build_drag_grid, section_cf_drag
from .stability import solve_amplification
from .transition import TransitionStatus, locate_transition_n_crit
from .turbulent_bl import pressure_gradient_cf_distribution, solve_turbulent_bl

N_POINTS_DEFAULT = 4001


@dataclass(frozen=True)
class ChainResult:
    """Headline M1-M5 results for one external-flow sensitivity profile."""

    label: str
    peak_ratio: float
    s_peak: float
    te_ratio: float
    re_c: float
    mach: float
    x_sep_lam_s: float | None
    onset_s: float | None
    n_max: float | None
    n_crit_status: dict[float, str]           # {n_crit: TransitionStatus.value}
    mechanism: str
    x_tr_s: float | None
    x_reattach_s: float | None
    x_turb_sep_s: float | None
    cd_f: float | None
    cd_sep: float | None
    cd_total: float | None
    bubble_status: str | None
    turbulent_status: str | None
    message: str


def run_chain(
    profile: SensitivityProfile,
    op: GenericSailplaneOperatingPoint,
    n_crit_values: tuple[float, ...] = (9.0, 12.0, 14.0),
    n_crit_primary: float = 9.0,
    n_points: int = N_POINTS_DEFAULT,
) -> ChainResult:
    """Propagate one external-flow sensitivity profile through the full
    M1 (Thwaites) -> M2 (amplification) -> M3 (N_crit event) ->
    M4 (turbulent) -> M5 (bubble/drag) chain, with no downstream constant
    retuned per-profile.
    """
    dist = build_distribution(profile, op.chord_m, op.v_inf_mps)

    tsol = solve_thwaites(dist, op.nu_m2s, n_points=n_points)
    asol = solve_amplification(op, tsol)

    n_crit_status = {}
    for nc in n_crit_values:
        n_crit_status[nc] = locate_transition_n_crit(asol, nc).status.value

    s = build_drag_grid(op.chord_m, n_points=n_points)
    x = s * op.chord_m
    ue_ratio = dist.velocity_ratio(s)

    r = evaluate_transition_mechanism(asol, tsol, n_crit_primary, NOMINAL_BUBBLE)

    if r.mechanism == TransitionMechanism.ATTACHED_E_N_CROSSING and r.status != BubbleStatus.NO_LAMINAR_SEPARATION:
        # Should not occur given evaluate_transition_mechanism's contract,
        # but fail loudly rather than silently mis-plot if it ever does.
        raise AssertionError("unexpected mechanism/status combination from evaluate_transition_mechanism")

    if r.status == BubbleStatus.NO_LAMINAR_SEPARATION:
        n_primary_result = locate_transition_n_crit(asol, n_crit_primary)
        if n_primary_result.status == TransitionStatus.N_CRIT_CROSSING:
            tb = solve_turbulent_bl(dist, tsol, op.v_inf_mps, op.nu_m2s, n_primary_result.x_tr_m, n_report=n_points)
            cf = pressure_gradient_cf_distribution(op.v_inf_mps, x, op.nu_m2s, n_primary_result.x_tr_m, tb)
            cd_f = section_cf_drag(s, cf, ue_ratio, two_surfaces=True)
            x_turb_sep_s = tb.x_turb_sep_m / op.chord_m if tb.x_turb_sep_m is not None else None
            return ChainResult(
                label=profile.label, peak_ratio=profile.u_peak_ratio, s_peak=profile.s_peak,
                te_ratio=profile.u_te_ratio, re_c=op.re_c, mach=op.mach,
                x_sep_lam_s=asol.s_sep, onset_s=asol.onset_s, n_max=asol.n_max,
                n_crit_status=n_crit_status, mechanism=TransitionMechanism.ATTACHED_E_N_CROSSING.value,
                x_tr_s=n_primary_result.x_tr_s, x_reattach_s=None, x_turb_sep_s=x_turb_sep_s,
                cd_f=cd_f, cd_sep=0.0, cd_total=cd_f, bubble_status=None, turbulent_status=tb.status.value,
                message=f"Attached e^N crossing at x/c={n_primary_result.x_tr_s:.4f}; bubble model bypassed.",
            )
        # Neither separation nor a crossing: nothing to report downstream.
        return ChainResult(
            label=profile.label, peak_ratio=profile.u_peak_ratio, s_peak=profile.s_peak,
            te_ratio=profile.u_te_ratio, re_c=op.re_c, mach=op.mach,
            x_sep_lam_s=None, onset_s=asol.onset_s, n_max=asol.n_max,
            n_crit_status=n_crit_status, mechanism=TransitionMechanism.ATTACHED_E_N_CROSSING.value,
            x_tr_s=None, x_reattach_s=None, x_turb_sep_s=None,
            cd_f=None, cd_sep=None, cd_total=None, bubble_status=None, turbulent_status=None,
            message="No laminar separation and no N_crit crossing found within the modeled domain.",
        )

    ue_ratio_sep = float(dist.velocity_ratio(r.x_sep_lam_s))

    if r.status == BubbleStatus.OPEN_SEPARATION:
        cf = m5_cf_distribution(op.v_inf_mps, x, op.nu_m2s, r.x_sep_lam_s * op.chord_m, None, None, None)
        cd_f_val = section_cf_drag(s, cf, ue_ratio, two_surfaces=True)
        decomp = build_drag_decomposition(profile.label, cd_f_val, 1.0 - r.x_sep_lam_s, ue_ratio_sep)
        return ChainResult(
            label=profile.label, peak_ratio=profile.u_peak_ratio, s_peak=profile.s_peak,
            te_ratio=profile.u_te_ratio, re_c=op.re_c, mach=op.mach,
            x_sep_lam_s=r.x_sep_lam_s, onset_s=asol.onset_s, n_max=asol.n_max,
            n_crit_status=n_crit_status, mechanism=r.mechanism.value,
            x_tr_s=r.x_tr_sep_s, x_reattach_s=None, x_turb_sep_s=None,
            cd_f=decomp.cd_f_m5, cd_sep=decomp.cd_sep_m5, cd_total=decomp.cd_total_m5,
            bubble_status=r.status.value, turbulent_status=None, message=r.message,
        )

    tb = solve_turbulent_bl(
        dist, tsol, op.v_inf_mps, op.nu_m2s, r.x_reattach_s * op.chord_m,
        theta_start_override_m=r.theta_reattach_m, h_start_override=r.h_reattach, n_report=n_points,
    )
    valid = np.where(tb.valid)[0]
    cf = m5_cf_distribution(
        op.v_inf_mps, x, op.nu_m2s, r.x_sep_lam_s * op.chord_m, r.x_reattach_s * op.chord_m,
        tb.x_m[valid], tb.cf[valid],
    )
    cd_f_val = section_cf_drag(s, cf, ue_ratio, two_surfaces=True)
    decomp = build_drag_decomposition(profile.label, cd_f_val, r.bubble_length_s, ue_ratio_sep)
    x_turb_sep_s = tb.x_turb_sep_m / op.chord_m if tb.x_turb_sep_m is not None else None

    return ChainResult(
        label=profile.label, peak_ratio=profile.u_peak_ratio, s_peak=profile.s_peak,
        te_ratio=profile.u_te_ratio, re_c=op.re_c, mach=op.mach,
        x_sep_lam_s=r.x_sep_lam_s, onset_s=asol.onset_s, n_max=asol.n_max,
        n_crit_status=n_crit_status, mechanism=r.mechanism.value,
        x_tr_s=r.x_tr_sep_s, x_reattach_s=r.x_reattach_s, x_turb_sep_s=x_turb_sep_s,
        cd_f=decomp.cd_f_m5, cd_sep=decomp.cd_sep_m5, cd_total=decomp.cd_total_m5,
        bubble_status=r.status.value, turbulent_status=tb.status.value if tb is not None else None,
        message=r.message,
    )
