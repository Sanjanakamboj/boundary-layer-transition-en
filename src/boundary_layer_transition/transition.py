"""Milestone 3: e^N N_crit transition-location logic.

This module consumes a Milestone 2 :class:`~.stability.AmplificationSolution`
(the N-factor history) and an externally supplied ``N_crit`` and answers one
question only: **where, if anywhere, does N(x) first reach N_crit within
the region where the Milestone 1 attached-laminar solution is still
valid?**

It explicitly distinguishes three outcomes (see :class:`TransitionStatus`):

- ``N_CRIT_CROSSING``: N(x) reaches N_crit before the M1 laminar-separation
  diagnostic (or before the end of the modeled domain, if no separation is
  predicted). The crossing station is located by linear interpolation
  between the bracketing grid stations, not merely rounded to the next
  grid point.
- ``SEPARATION_BEFORE_N_CRIT``: the M1 Thwaites solution reaches its
  laminar-separation diagnostic before N(x) reaches N_crit. This is the
  Milestone 2 baseline outcome for every sourced N_crit sensitivity value
  in this project (N_max ~= 3.84, well below the smallest sourced N_crit).
- ``NO_EVENT_IN_DOMAIN``: neither a crossing nor a separation occurs within
  the modeled domain (e.g. a hypothetical fully favorable-gradient case
  with no predicted separation and insufficient amplification to reach
  N_crit by the trailing edge).

**N_crit is not selected or recommended by this module or by this
project.** It is an external, environment-dependent input (see
DESIGN.md for the sourced sensitivity range actually used in the study
scripts). This module only locates where a *given* N_crit value would be
reached, if at all.

Separately, :func:`separation_triggered_transition` constructs the
explicitly distinct "assumed immediate transition at the laminar-
separation diagnostic" bookkeeping scenario used for drag estimation in
:mod:`skin_friction` when no N_crit crossing exists. This is **never** an
e^N transition location -- it is a separate modeling assumption, kept
clearly labeled everywhere it is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from .stability import AmplificationSolution


class TransitionStatus(str, Enum):
    """Outcome of comparing an N-factor history against a given N_crit."""

    N_CRIT_CROSSING = "N_CRIT_CROSSING"
    SEPARATION_BEFORE_N_CRIT = "SEPARATION_BEFORE_N_CRIT"
    NO_EVENT_IN_DOMAIN = "NO_EVENT_IN_DOMAIN"


def interpolate_crossing(s: ArrayLike, n: ArrayLike, n_crit: float) -> float | None:
    """First station where ``n(s)`` reaches ``n_crit``, by linear interpolation.

    Pure numerical utility, independent of :class:`AmplificationSolution` --
    tested directly against exact-on-node and between-node cases.

    Parameters
    ----------
    s : array_like
        Strictly increasing station coordinate (any consistent units).
    n : array_like
        Values at each station (same shape as ``s``). Not required to be
        monotonic in general, but the *first* crossing (from below) is what
        is returned; for a monotonically non-decreasing ``n`` (as produced
        by :func:`~.stability.integrate_n_factor`) this is unambiguous.

    Returns
    -------
    float or None
        The interpolated ``s`` at which ``n`` first reaches ``n_crit``, or
        ``None`` if ``n`` never reaches ``n_crit`` over the supplied range.
    """
    s_arr = np.asarray(s, dtype=np.float64)
    n_arr = np.asarray(n, dtype=np.float64)
    if s_arr.shape != n_arr.shape:
        raise ValueError("s and n must have the same shape")
    if s_arr.ndim != 1 or s_arr.size < 1:
        raise ValueError("s must be a 1D array of at least 1 station")
    if s_arr.size > 1 and np.any(np.diff(s_arr) <= 0.0):
        raise ValueError("s must be strictly increasing")
    if np.any(~np.isfinite(n_arr)):
        raise ValueError("n must be finite everywhere (no NaN/inf)")

    idx = np.where(n_arr >= n_crit)[0]
    if idx.size == 0:
        return None

    i1 = int(idx[0])
    if i1 == 0:
        return float(s_arr[0])

    i0 = i1 - 1
    n0, n1 = n_arr[i0], n_arr[i1]
    if n1 == n0:
        return float(s_arr[i1])
    frac = (n_crit - n0) / (n1 - n0)
    return float(s_arr[i0] + frac * (s_arr[i1] - s_arr[i0]))


@dataclass(frozen=True)
class TransitionResult:
    """Outcome of locating an N_crit crossing against one AmplificationSolution."""

    n_crit: float
    status: TransitionStatus
    x_tr_s: float | None      # x/c of the N_crit crossing (None unless N_CRIT_CROSSING)
    x_tr_m: float | None      # dimensional x of the N_crit crossing, m
    n_max: float | None       # max N attained within the M1-valid domain
    s_sep: float | None       # M1 laminar-separation diagnostic, x/c (context, always carried through)
    x_sep_m: float | None


def locate_transition_n_crit(asol: AmplificationSolution, n_crit: float) -> TransitionResult:
    """Locate the first x/c where ``asol.n`` reaches ``n_crit``, restricted
    to the Milestone 1 valid (attached-laminar) domain.

    A crossing can only be reported if it occurs while ``asol.valid`` is
    True at that station -- i.e. strictly before (or at) the M1
    laminar-separation diagnostic. This is enforced structurally (the
    search is restricted to ``asol.s[asol.valid]``), not by a post-hoc
    check.
    """
    if not np.isfinite(n_crit) or n_crit <= 0.0:
        raise ValueError(f"n_crit must be finite and positive, got {n_crit}")

    valid_idx = np.where(asol.valid)[0]
    if valid_idx.size < 1:
        return TransitionResult(
            n_crit=n_crit,
            status=TransitionStatus.NO_EVENT_IN_DOMAIN,
            x_tr_s=None,
            x_tr_m=None,
            n_max=asol.n_max,
            s_sep=asol.s_sep,
            x_sep_m=asol.x_sep_m,
        )

    s_valid = asol.s[valid_idx]
    x_m_valid = asol.x_m[valid_idx]
    n_valid = asol.n[valid_idx]

    x_tr_s = interpolate_crossing(s_valid, n_valid, n_crit)

    if x_tr_s is not None:
        x_tr_m = interpolate_crossing(x_m_valid, n_valid, n_crit)
        status = TransitionStatus.N_CRIT_CROSSING
    else:
        x_tr_m = None
        status = (
            TransitionStatus.SEPARATION_BEFORE_N_CRIT
            if asol.s_sep is not None
            else TransitionStatus.NO_EVENT_IN_DOMAIN
        )

    return TransitionResult(
        n_crit=n_crit,
        status=status,
        x_tr_s=x_tr_s,
        x_tr_m=x_tr_m,
        n_max=asol.n_max,
        s_sep=asol.s_sep,
        x_sep_m=asol.x_sep_m,
    )


def locate_transition_n_crit_sweep(asol: AmplificationSolution, n_crit_values: ArrayLike) -> list[TransitionResult]:
    """Convenience: :func:`locate_transition_n_crit` for a list of N_crit values."""
    n_crit_arr = np.asarray(n_crit_values, dtype=np.float64)
    if n_crit_arr.ndim != 1:
        raise ValueError("n_crit_values must be a 1D sequence")
    return [locate_transition_n_crit(asol, float(nc)) for nc in n_crit_arr]


@dataclass(frozen=True)
class SeparationTriggeredScenario:
    """The explicitly separate "assumed immediate transition at the M1
    laminar-separation diagnostic" bookkeeping scenario.

    This is **not** an e^N / N_crit transition. It exists only because the
    Milestone 1 attached-laminar (Thwaites) solution cannot be continued
    past its own separation diagnostic, and Milestone 3's skin-friction
    drag bookkeeping (Section 5/6) needs *some* assumed transition
    location whenever no N_crit crossing exists. The label carried on this
    object is deliberately explicit so it is never mistaken for an N_crit
    result in reports or figures.
    """

    x_tr_s: float
    x_tr_m: float
    label: str = "separation-triggered (bookkeeping assumption, NOT an e^N N_crit crossing)"


def separation_triggered_transition(asol: AmplificationSolution) -> SeparationTriggeredScenario | None:
    """Build the separation-triggered scenario from an AmplificationSolution's
    M1 laminar-separation diagnostic, or ``None`` if M1 predicted no
    separation over the modeled domain.
    """
    if asol.s_sep is None:
        return None
    return SeparationTriggeredScenario(x_tr_s=asol.s_sep, x_tr_m=asol.x_sep_m)


def summarize_status_counts(results: list[TransitionResult]) -> dict[str, int]:
    """Count how many results fall into each TransitionStatus -- a small,
    independently testable convenience used by the study script's report.
    """
    counts: dict[str, int] = {status.value: 0 for status in TransitionStatus}
    for r in results:
        counts[r.status.value] += 1
    return counts
