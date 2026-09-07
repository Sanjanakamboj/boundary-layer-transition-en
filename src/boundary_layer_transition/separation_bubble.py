"""Milestone 5: laminar-separation-bubble (LSB) transition/reattachment closure.

**What this module is, and is not.** This module does **not** resolve a
separated shear layer, does **not** predict a universally valid bubble
length, and does **not** implement a specific, independently verified
quantitative bursting correlation (e.g. Horton's or Gaster's). It
implements a **transparent, explicitly labeled parametric sensitivity
closure** for the sequence laminar separation -> separated shear layer ->
transition -> (possible) turbulent reattachment, driven by the Milestone
1-4 state at the laminar-separation station.

Source audit (see DESIGN.md Section 32 for the full discussion of what
was and was not verified):

- **Horton, H. P.** (1969), "A semi-empirical theory for the growth and
  bursting of laminar separation bubbles," ARC CP 1073. Confirmed via web
  search: a semi-empirical model estimating bubble length/bursting from
  the pressure rise the separated/transitional shear layer can sustain
  between transition and reattachment, using momentum/energy integral
  arguments. The search located secondary descriptions of the model's
  structure and qualitative conclusions, but not a compact, digit-
  verified closed-form correlation directly usable with only this
  project's existing state variables (theta, H, Re at separation).
- **Gaster, M.** (1967), "The structure and behaviour of laminar
  separation bubbles," ARC R&M 3595. Confirmed via web search as the
  origin of a bursting criterion based on the momentum-thickness Reynolds
  number at separation and a pressure-gradient parameter. Also confirmed
  (via the same search, citing later work by Diwan, Chetan & Ramesh 2006
  and Mitra & Ramesh 2019) that Gaster's original criterion is **not**
  considered generally valid -- bursting depends on the specific pressure
  trajectory of the bubble, not a single two-parameter criterion. This is
  exactly the kind of "supposedly universal empirical equation" the
  project instructions warn against implementing without verification, so
  it is **not** implemented quantitatively here.
- **Order-of-magnitude bubble-length context**: web search located
  reported bubble lengths of order 20-30% chord for low-Reynolds-number
  airfoils (Re_c ~ 3e4-2e5). This project's baseline Re_c (~1.2e6, and up
  to ~4.3e6 in the Reynolds sensitivity) is one to two orders of magnitude
  higher; it is well established qualitatively (though not quantitatively
  correlated here) that laminar separation bubbles shrink substantially
  with increasing Reynolds number. This informs only the *direction* and
  rough *order of magnitude* (much less than 20-30% chord) of the
  project's chosen sensitivity parameters below -- it is not a
  quantitative Reynolds-number scaling law and is not presented as one.

Because no compact, independently verified correlation suitable for this
project's state variables was found, this module implements **Option B**
of the project's stated closure hierarchy: a transparent parametric
sensitivity model, with every constant explicitly labeled as a project
assumption, evaluated at short/nominal/long/open settings rather than
tuned to any particular result.

Sequence modeled (see also DESIGN.md Section 32):

    attached laminar BL -> laminar separation (M1 Thwaites diagnostic)
        -> separated laminar shear layer (NOT modeled in detail)
        -> transition in the separated shear layer (x_tr,sep)
        -> EITHER turbulent reattachment (x_reattach), restarting the M4
           turbulent integral solver with a bubble-informed restart state,
        -> OR open separation (no reattachment; the separated region is
           carried to the trailing edge as a pressure/form-drag penalty
           by :mod:`separated_drag`, not as a skin-friction contribution).

The Milestone 2 attached-flow e^N amplification model is explicitly
**not** continued through the separated shear layer -- if a genuine
attached-flow N_crit crossing already occurs before laminar separation
(as found in Milestone 3's high-Re sensitivity), this module is bypassed
entirely and the attached-flow crossing station is used instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .laminar_bl import ThwaitesSolution
from .stability import AmplificationSolution
from .transition import TransitionStatus, locate_transition_n_crit


class TransitionMechanism(str, Enum):
    """Which of the three distinct transition pathways produced x_tr."""

    ATTACHED_E_N_CROSSING = "ATTACHED_E_N_CROSSING"
    SEPARATION_INDUCED = "SEPARATION_INDUCED"
    PRESCRIBED = "PRESCRIBED"


class BubbleStatus(str, Enum):
    """Outcome of the laminar-separation-bubble closure."""

    NO_LAMINAR_SEPARATION = "NO_LAMINAR_SEPARATION"
    LAMINAR_SEPARATION = "LAMINAR_SEPARATION"
    SEPARATED_SHEAR_LAYER_TRANSITION = "SEPARATED_SHEAR_LAYER_TRANSITION"
    TURBULENT_REATTACHMENT = "TURBULENT_REATTACHMENT"
    OPEN_SEPARATION = "OPEN_SEPARATION"
    INVALID_BUBBLE_MODEL = "INVALID_BUBBLE_MODEL"


@dataclass(frozen=True)
class BubbleParameters:
    """A named, explicitly labeled set of project sensitivity parameters
    for the LSB closure. None of these are sourced quantitative
    predictions -- see module docstring.

    dx_tr_sep_frac : distance from laminar separation to separated-shear-
        layer transition, as a fraction of chord.
    dx_reattach_frac : distance from separated-shear-layer transition to
        turbulent reattachment, as a fraction of chord. ``None`` selects
        an open-separation (no-reattachment) scenario.
    k_theta : restart momentum-thickness multiplier,
        theta_reattach = k_theta * theta_sep. ``None`` for open separation
        (no restart is performed).
    h_reattach : restart shape factor at reattachment. ``None`` for open
        separation.
    """

    label: str
    dx_tr_sep_frac: float
    dx_reattach_frac: float | None
    k_theta: float | None
    h_reattach: float | None

    def __post_init__(self) -> None:
        if not np.isfinite(self.dx_tr_sep_frac) or self.dx_tr_sep_frac < 0.0:
            raise ValueError(f"dx_tr_sep_frac must be finite and non-negative, got {self.dx_tr_sep_frac}")
        if self.dx_reattach_frac is not None:
            if not np.isfinite(self.dx_reattach_frac) or self.dx_reattach_frac <= 0.0:
                raise ValueError(
                    f"dx_reattach_frac must be finite and positive (or None for open "
                    f"separation), got {self.dx_reattach_frac}"
                )
            if self.k_theta is None or self.h_reattach is None:
                raise ValueError("k_theta and h_reattach are required whenever dx_reattach_frac is given")
            if not np.isfinite(self.k_theta) or self.k_theta <= 0.0:
                raise ValueError(f"k_theta must be finite and positive, got {self.k_theta}")
            if not np.isfinite(self.h_reattach) or self.h_reattach <= 1.0:
                raise ValueError(f"h_reattach must be finite and > 1, got {self.h_reattach}")
        elif self.k_theta is not None or self.h_reattach is not None:
            raise ValueError("k_theta and h_reattach must be None when dx_reattach_frac is None (open separation)")


#: Project sensitivity parameter sets (NOT sourced quantitative
#: predictions -- see module docstring). Total bubble extent
#: (dx_tr_sep_frac + dx_reattach_frac) increases short -> nominal -> long,
#: chosen to be well within the (much larger) 20-30%-chord bubbles
#: reported in the low-Re literature, consistent with the qualitative,
#: not quantitatively correlated, expectation that bubbles shrink at this
#: project's higher chord Reynolds numbers. k_theta and h_reattach reflect
#: only the qualitative, well-documented expectation that a reattaching
#: shear layer is substantially thicker and less full than a fresh
#: turbulent boundary layer at the same station.
SHORT_BUBBLE = BubbleParameters("short", dx_tr_sep_frac=0.003, dx_reattach_frac=0.007, k_theta=3.0, h_reattach=1.5)
NOMINAL_BUBBLE = BubbleParameters("nominal", dx_tr_sep_frac=0.008, dx_reattach_frac=0.022, k_theta=5.0, h_reattach=1.7)
LONG_BUBBLE = BubbleParameters("long", dx_tr_sep_frac=0.015, dx_reattach_frac=0.065, k_theta=8.0, h_reattach=2.0)
OPEN_SEPARATION_PARAMS = BubbleParameters(
    "open", dx_tr_sep_frac=0.008, dx_reattach_frac=None, k_theta=None, h_reattach=None
)


@dataclass(frozen=True)
class BubbleResult:
    """Outcome of evaluating the transition mechanism and (if applicable)
    the laminar-separation-bubble closure for one operating point.
    """

    mechanism: TransitionMechanism
    status: BubbleStatus
    x_sep_lam_s: float | None       # M1 laminar-separation diagnostic, x/c
    x_tr_sep_s: float | None        # separated-shear-layer transition, x/c
    x_reattach_s: float | None      # turbulent reattachment, x/c (None if open/bypassed)
    bubble_length_s: float | None   # x_reattach_s - x_sep_lam_s (None if open/bypassed)
    theta_sep_m: float | None       # last-valid laminar theta at separation, m
    theta_reattach_m: float | None  # restart theta at reattachment, m
    h_reattach: float | None        # restart H at reattachment
    params: BubbleParameters | None
    message: str


def evaluate_transition_mechanism(
    asol: AmplificationSolution,
    thwaites_sol: ThwaitesSolution,
    n_crit: float | None,
    params: BubbleParameters,
) -> BubbleResult:
    """Determine which transition mechanism applies, and evaluate the LSB
    closure if the mechanism is separation-induced.

    Logic (see module docstring for the physical sequence):

    1. If ``n_crit`` is given and the Milestone 2/3 attached-flow N-factor
       reaches it before the Milestone 1 laminar-separation diagnostic,
       the mechanism is ``ATTACHED_E_N_CROSSING`` and this module's bubble
       closure is bypassed entirely (status ``NO_LAMINAR_SEPARATION`` --
       no laminar separation occurred before transition).
    2. Otherwise, if Milestone 1 predicts laminar separation
       (``asol.s_sep is not None``), the mechanism is
       ``SEPARATION_INDUCED`` and the bubble closure below is evaluated
       using ``params``.
    3. Otherwise (no separation and no crossing), the mechanism is
       reported as ``ATTACHED_E_N_CROSSING`` with status
       ``NO_LAMINAR_SEPARATION`` and no transition location determined by
       this module (matches Milestone 3's ``NO_EVENT_IN_DOMAIN``).

    Prescribed-transition scenarios (mechanism ``PRESCRIBED``) do not call
    this function at all -- they bypass both the e^N and bubble logic by
    construction (see scripts/separation_bubble_drag_study.py).
    """
    if n_crit is not None:
        n_result = locate_transition_n_crit(asol, n_crit)
        if n_result.status == TransitionStatus.N_CRIT_CROSSING:
            return BubbleResult(
                mechanism=TransitionMechanism.ATTACHED_E_N_CROSSING,
                status=BubbleStatus.NO_LAMINAR_SEPARATION,
                x_sep_lam_s=asol.s_sep,
                x_tr_sep_s=None,
                x_reattach_s=None,
                bubble_length_s=None,
                theta_sep_m=None,
                theta_reattach_m=None,
                h_reattach=None,
                params=None,
                message=(
                    f"Attached-flow e^N crossing at x/c={n_result.x_tr_s:.4f} occurs "
                    "before any laminar separation; the LSB closure is bypassed."
                ),
            )

    if asol.s_sep is None:
        return BubbleResult(
            mechanism=TransitionMechanism.ATTACHED_E_N_CROSSING,
            status=BubbleStatus.NO_LAMINAR_SEPARATION,
            x_sep_lam_s=None,
            x_tr_sep_s=None,
            x_reattach_s=None,
            bubble_length_s=None,
            theta_sep_m=None,
            theta_reattach_m=None,
            h_reattach=None,
            params=None,
            message="No laminar separation predicted and no N_crit crossing found (NO_EVENT_IN_DOMAIN).",
        )

    return _evaluate_bubble(asol.s_sep, thwaites_sol, params)


def evaluate_bubble_at_separation(
    thwaites_sol: ThwaitesSolution,
    x_sep_lam_s: float,
    params: BubbleParameters,
) -> BubbleResult:
    """Evaluate the LSB closure directly from a known laminar-separation
    station and the Milestone 1 Thwaites solution (used to obtain
    theta_sep). This is the lower-level entry point used internally by
    :func:`evaluate_transition_mechanism`, exposed separately for direct
    testing and for cases (e.g. Reynolds sensitivity) where the caller
    already knows x_sep_lam_s.
    """
    return _evaluate_bubble(x_sep_lam_s, thwaites_sol, params)


def _evaluate_bubble(
    x_sep_lam_s: float,
    thwaites_sol: ThwaitesSolution | None,
    params: BubbleParameters,
) -> BubbleResult:
    if not np.isfinite(x_sep_lam_s) or not (0.0 <= x_sep_lam_s < 1.0):
        raise ValueError(f"x_sep_lam_s must be finite and in [0, 1), got {x_sep_lam_s}")

    x_tr_sep_s = x_sep_lam_s + params.dx_tr_sep_frac

    theta_sep_m: float | None = None
    if thwaites_sol is not None:
        chord_m = float(thwaites_sol.x_m[-1])
        theta_sep_m = float(np.interp(x_sep_lam_s * chord_m, thwaites_sol.x_m, thwaites_sol.theta_m))

    if params.dx_reattach_frac is None:
        return BubbleResult(
            mechanism=TransitionMechanism.SEPARATION_INDUCED,
            status=BubbleStatus.OPEN_SEPARATION,
            x_sep_lam_s=x_sep_lam_s,
            x_tr_sep_s=x_tr_sep_s,
            x_reattach_s=None,
            bubble_length_s=None,
            theta_sep_m=theta_sep_m,
            theta_reattach_m=None,
            h_reattach=None,
            params=params,
            message=(
                f"Separated shear-layer transition modeled at x/c={x_tr_sep_s:.4f}; "
                "no reattachment (open-separation project scenario)."
            ),
        )

    x_reattach_s = x_tr_sep_s + params.dx_reattach_frac

    if x_reattach_s >= 1.0:
        return BubbleResult(
            mechanism=TransitionMechanism.SEPARATION_INDUCED,
            status=BubbleStatus.OPEN_SEPARATION,
            x_sep_lam_s=x_sep_lam_s,
            x_tr_sep_s=x_tr_sep_s,
            x_reattach_s=None,
            bubble_length_s=None,
            theta_sep_m=theta_sep_m,
            theta_reattach_m=None,
            h_reattach=None,
            params=params,
            message=(
                f"Parametric reattachment station x/c={x_reattach_s:.4f} would lie at or "
                "beyond the trailing edge; treated as open separation (the bubble does not "
                "close within the modeled chord)."
            ),
        )

    theta_reattach_m = params.k_theta * theta_sep_m if theta_sep_m is not None else None

    return BubbleResult(
        mechanism=TransitionMechanism.SEPARATION_INDUCED,
        status=BubbleStatus.TURBULENT_REATTACHMENT,
        x_sep_lam_s=x_sep_lam_s,
        x_tr_sep_s=x_tr_sep_s,
        x_reattach_s=x_reattach_s,
        bubble_length_s=x_reattach_s - x_sep_lam_s,
        theta_sep_m=theta_sep_m,
        theta_reattach_m=theta_reattach_m,
        h_reattach=params.h_reattach,
        params=params,
        message=(
            f"Separated shear-layer transition at x/c={x_tr_sep_s:.4f}; turbulent "
            f"reattachment at x/c={x_reattach_s:.4f} ({params.label} bubble, "
            f"length={x_reattach_s - x_sep_lam_s:.4f} chord fractions)."
        ),
    )
