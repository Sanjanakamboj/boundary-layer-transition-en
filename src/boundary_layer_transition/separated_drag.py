"""Milestone 5: post-separation drag bookkeeping.

Combines a skin-friction contribution over the *modeled attached and
reattached* surface regions (never over an unresolved bubble or open-
separation region) with an explicit, separately labeled separated
pressure/form-drag penalty. See DESIGN.md Section 33 for the full
derivation and source-audit discussion of the separated-drag closure --
no independently verified, compact, low-order pressure/form-drag
correlation appropriate to this project's synthetic external-flow
distribution and state variables was found, so a transparent parametric
sensitivity closure is used instead, with its structure derived here
(not copied) and its coefficients explicitly labeled as project
assumptions.

Nondimensionalization (consistent with skin_friction.section_cf_drag,
Milestone 3 Section 19): the separated-drag term uses the same reference
dynamic pressure (0.5 rho V_inf^2) and chord as the friction integral, and
the same explicit dynamic-pressure weighting via [U_e(x_sep)/V_inf]^2 (the
local edge dynamic pressure at the laminar-separation station, used as the
representative dynamic pressure acting on the separated region -- a
project choice, documented, not derived from a resolved separated-flow
pressure field).

    C_d,sep = K_sep * L_sep^p * [U_e(x_sep)/V_inf]^2

    L_sep = separated chord fraction:
        - a reattaching bubble: L_sep = bubble_length/c (the bubble
          extent only; skin friction resumes downstream of reattachment
          and is accounted separately)
        - open separation: L_sep = 1 - x_sep,lam/c (everything from
          laminar separation to the trailing edge, since nothing
          reattaches)

    K_sep, p: project sensitivity parameters (NOT sourced), with
        K_sep=0.5, p=1.0 as the documented nominal default (a simple
        linear-in-extent form, the simplest closure satisfying the
        required zero-at-zero-extent and monotonicity properties).

C_d,total,M5 = C_d,f,M5 + C_d,sep,M5 is a reduced-order sensitivity
bookkeeping total. It is explicitly **not** presented as a validated or
complete airfoil drag coefficient -- it omits pressure/form drag from any
attached-flow region, 3D effects, compressibility, and surface-contour
effects not already represented in the (still 2D, incompressible) skin-
friction terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .laminar_bl import blasius_cf

#: Nominal (project-chosen, unsourced) separated-drag closure coefficients.
K_SEP_NOMINAL = 0.5
P_SEP_NOMINAL = 1.0


def separated_drag_coefficient(
    l_sep_frac: float,
    ue_ratio_at_sep: float,
    k_sep: float = K_SEP_NOMINAL,
    p: float = P_SEP_NOMINAL,
) -> float:
    """C_d,sep = k_sep * l_sep_frac^p * ue_ratio_at_sep^2.

    Returns exactly 0.0 for l_sep_frac == 0 (no separated extent -> no
    penalty, regardless of k_sep/p), and is otherwise strictly increasing
    in l_sep_frac for p > 0, k_sep > 0.
    """
    if not np.isfinite(l_sep_frac) or l_sep_frac < 0.0:
        raise ValueError(f"l_sep_frac must be finite and non-negative, got {l_sep_frac}")
    if not np.isfinite(ue_ratio_at_sep) or ue_ratio_at_sep < 0.0:
        raise ValueError(f"ue_ratio_at_sep must be finite and non-negative, got {ue_ratio_at_sep}")
    if not np.isfinite(k_sep) or k_sep < 0.0:
        raise ValueError(f"k_sep must be finite and non-negative, got {k_sep}")
    if not np.isfinite(p) or p <= 0.0:
        raise ValueError(f"p must be finite and positive, got {p}")

    if l_sep_frac == 0.0:
        return 0.0
    return k_sep * l_sep_frac**p * ue_ratio_at_sep**2


@dataclass(frozen=True)
class DragDecomposition:
    """The full Milestone 5 drag decomposition for one scenario."""

    label: str
    cd_f_m5: float
    cd_sep_m5: float
    cd_total_m5: float
    l_sep_frac: float
    k_sep: float
    p: float

    def __post_init__(self) -> None:
        # Exact-identity guard: catches any future accidental drift
        # between how the two components and the total are assembled.
        reconstructed = self.cd_f_m5 + self.cd_sep_m5
        if not np.isclose(reconstructed, self.cd_total_m5, rtol=1e-12, atol=1e-15):
            raise ValueError(
                f"cd_total_m5 ({self.cd_total_m5}) does not equal cd_f_m5 + cd_sep_m5 "
                f"({reconstructed}) -- drag decomposition identity violated"
            )


def build_drag_decomposition(
    label: str,
    cd_f_m5: float,
    l_sep_frac: float,
    ue_ratio_at_sep: float,
    k_sep: float = K_SEP_NOMINAL,
    p: float = P_SEP_NOMINAL,
) -> DragDecomposition:
    """Convenience constructor: computes C_d,sep and the exact total from
    a pre-computed friction contribution C_d,f,M5.
    """
    cd_sep = separated_drag_coefficient(l_sep_frac, ue_ratio_at_sep, k_sep, p)
    return DragDecomposition(
        label=label,
        cd_f_m5=cd_f_m5,
        cd_sep_m5=cd_sep,
        cd_total_m5=cd_f_m5 + cd_sep,
        l_sep_frac=l_sep_frac,
        k_sep=k_sep,
        p=p,
    )


def m5_cf_distribution(
    v_inf_mps: float,
    x_m: ArrayLike,
    nu_m2s: float,
    x_sep_m: float,
    x_reattach_m: float | None,
    tb_restart_cf_x_m: NDArray[np.float64] | None,
    tb_restart_cf: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Piecewise Milestone 5 local skin-friction coefficient:

    - x < x_sep_m: laminar (Blasius), reusing laminar_bl.blasius_cf.
    - x_sep_m <= x < x_reattach_m (bubble region): exactly 0.0 -- this
      interval's aerodynamic effect is represented entirely by the
      separated-drag term, never double-counted as skin friction.
    - x >= x_reattach_m: the restarted M4 turbulent solution's Cf(x),
      interpolated (and held/frozen beyond its own valid range, exactly
      as in Milestone 4's pressure_gradient_cf_distribution).

    If x_reattach_m is None (open separation), the bubble-region
    definition (Cf=0) extends from x_sep_m to the trailing edge, and
    tb_restart_cf_x_m / tb_restart_cf are ignored (may be None).
    """
    x_arr = np.asarray(x_m, dtype=np.float64)
    if np.any(~np.isfinite(x_arr)) or np.any(x_arr < 0.0):
        raise ValueError("x_m must be finite and non-negative")
    if not np.isfinite(x_sep_m) or x_sep_m < 0.0:
        raise ValueError(f"x_sep_m must be finite and non-negative, got {x_sep_m}")

    laminar_mask = x_arr < x_sep_m
    cf = np.where(laminar_mask, blasius_cf(v_inf_mps, x_arr, nu_m2s), 0.0)

    if x_reattach_m is not None:
        if tb_restart_cf_x_m is None or tb_restart_cf is None:
            raise ValueError("tb_restart_cf_x_m and tb_restart_cf are required when x_reattach_m is given")
        turbulent_mask = x_arr >= x_reattach_m
        cf = np.where(turbulent_mask, np.interp(x_arr, tb_restart_cf_x_m, tb_restart_cf), cf)

    return cf if x_arr.ndim else float(cf)
