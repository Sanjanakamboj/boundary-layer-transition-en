"""Milestone 2: reduced-order linear-stability proxy and e^N amplification tracking.

**What this module is, and is not.** This module does **not** solve the
Orr-Sommerfeld eigenvalue problem, and does **not** implement the specific
Drela/Giles or Gleyzes-Cousteix-Bonnet envelope-amplification polynomial
used inside XFOIL (those exact curve-fit coefficients were not available
to verify against a primary source at the time this module was written --
see DESIGN.md Section "Source audit" for what was and was not checked).
Instead, it implements a **transparent, explicitly-labeled reduced-order
engineering proxy** for Tollmien-Schlichting-type disturbance amplification,
built on top of the Milestone 1 Thwaites laminar boundary-layer solution,
anchored at one number that *is* a classical, widely reproduced result:
the zero-pressure-gradient (Blasius) Tollmien-Schlichting neutral-stability
point.

Sourced physics (see DESIGN.md for full citations):

- The e^N formalism itself: for a disturbance behaving spatially as
  ``A(x) ~ exp(-integral alpha_i dx)``, the amplification factor is
  ``N(x) = ln(A(x)/A0) = integral_{x0}^{x} (-alpha_i) dx'``, accumulated
  over the unstable region from the instability-onset station ``x0``.
  (Smith & Gamberoni 1956; van Ingen 1956; standard e^N formalism as
  reviewed in van Ingen, "The e^N method for transition prediction:
  Historical review of work at TU Delft," AIAA 2008-3830.)
- The classical zero-pressure-gradient Tollmien-Schlichting neutral-stability
  point for the Blasius boundary layer, commonly cited as
  ``Re_x,crit ~= 9.1e4`` (equivalently ``Re_theta,crit ~= 200``), is a
  textbook result reproduced in numerous boundary-layer-stability
  treatments (e.g. Schlichting & Gersten, *Boundary-Layer Theory*, 9th ed.,
  stability chapters; White, *Viscous Fluid Flow*, 3rd ed., Ch. 5). This
  module uses ``Re_theta,crit,ZPG = 200`` as its one sourced numerical
  anchor. Reported eigenvalue solutions for the exact Blasius neutral point
  vary by several percent across sources/disturbance definitions; 200 is
  presented as the commonly cited, order-of-magnitude-correct value, not a
  razor-precise digit.
- The *qualitative* trend that adverse-pressure-gradient (higher-H,
  more-inflectional) laminar profiles are markedly less stable than
  favorable-gradient (lower-H) profiles, and destabilize at lower local
  Reynolds number, is a well-documented result of Falkner-Skan-family
  linear stability analysis (Schlichting & Gersten; Mack's stability
  reviews). This module uses that qualitative trend to shape its onset and
  amplification-rate proxies, but does **not** claim to reproduce any
  specific published quantitative Falkner-Skan neutral-curve or
  amplification-rate table.

Project (unsourced) assumptions -- explicitly labeled as such wherever
used:

- The H-dependence of the local critical Reynolds number,
  ``Re_theta,crit(H) = Re_theta,crit,ZPG * exp(-b*(H - H_blasius))``, is a
  project-chosen smooth, monotonic proxy shape anchored exactly at the
  sourced ZPG point (``H=H_blasius -> Re_theta,crit=200``). The sensitivity
  constant ``b`` (default 1.0) is an illustrative project choice, not fit
  to any published neutral-curve data.
- The local amplification-rate proxy,
  ``dN/ds = k * (H - 1) * max(0, Re_theta - Re_theta,crit(H)) / Re_theta,crit,ZPG``
  (``s = x/c``), is a project-chosen engineering proxy, not a digitized
  envelope amplification-rate correlation. The coefficient ``k`` (default
  10.0) is an illustrative calibration constant chosen only so that the
  baseline Milestone 1 operating point produces a legible, order-1 N-factor
  curve over its available (pre-separation) amplification length; it is
  NOT fit to any measured or computed amplification-rate data.

What a true Orr-Sommerfeld/PSE calculation would add (and this proxy does
not provide): the actual complex eigenvalue ``alpha(x; omega, beta)`` for
each disturbance frequency/spanwise wavenumber, a true neutral curve
computed from the actual local mean-velocity profile (not just Re_theta
and H), the most-amplified-frequency envelope construction used by e^N
codes in practice, and any sensitivity to the actual disturbance
environment (receptivity). None of that is modeled here.

Sign convention: this module works directly with ``dN/ds`` (the spatial
amplification-rate density in the nondimensional coordinate ``s = x/c``),
not with a solved ``alpha_i`` eigenvalue. It does **not** pretend an
eigenvalue was computed. Unstable amplification is, by construction,
``dN/ds >= 0`` everywhere (see :func:`local_amplification_rate`); stable
stations contribute exactly zero, never negative, N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid

from .laminar_bl import BLASIUS_H, ThwaitesSolution
from .operating_point import GenericSailplaneOperatingPoint

# ---------------------------------------------------------------------------
# Constants -- see module docstring for sourced-vs-project-assumption status
# ---------------------------------------------------------------------------

#: Sourced anchor: classical Blasius (ZPG) Tollmien-Schlichting neutral-
#: stability point, Re_theta,crit ~= 200 (equivalently Re_x,crit ~= 9.1e4).
RE_THETA_CRIT_ZPG = 200.0

#: The shape factor at which the ZPG anchor applies (the self-consistent
#: Blasius H from laminar_bl.py, ~2.5916).
H_REFERENCE = BLASIUS_H

#: Project-chosen (unsourced) sensitivity of the local critical Reynolds
#: number to shape factor H, in critical_re_theta(). Larger b => stronger
#: destabilization for a given rise in H above H_REFERENCE.
ONSET_H_SENSITIVITY = 1.0

#: Project-chosen (unsourced) amplification-rate calibration constant, in
#: local_amplification_rate(). See module docstring.
AMPLIFICATION_COEFF = 10.0


def _validate_finite(name: str, value: NDArray[np.float64]) -> None:
    if np.any(~np.isfinite(value)):
        raise ValueError(f"{name} must be finite everywhere, got non-finite values")


def reynolds_theta(rho_kgm3: float, u_e_mps: ArrayLike, theta_m: ArrayLike, mu_pas: float) -> NDArray[np.float64]:
    """Momentum-thickness Reynolds number, Re_theta = rho * U_e * theta / mu.

    ``u_e_mps`` and ``theta_m`` may be scalars or arrays (same shape);
    ``rho_kgm3`` and ``mu_pas`` are scalars.
    """
    if not np.isfinite(rho_kgm3) or rho_kgm3 <= 0.0:
        raise ValueError(f"rho_kgm3 must be finite and positive, got {rho_kgm3}")
    if not np.isfinite(mu_pas) or mu_pas <= 0.0:
        raise ValueError(f"mu_pas must be finite and positive, got {mu_pas}")

    u_arr = np.asarray(u_e_mps, dtype=np.float64)
    theta_arr = np.asarray(theta_m, dtype=np.float64)
    if np.any(theta_arr < 0.0):
        raise ValueError("theta_m must be non-negative")
    if np.any(u_arr < 0.0):
        raise ValueError("u_e_mps must be non-negative")

    result = rho_kgm3 * u_arr * theta_arr / mu_pas
    return result if u_arr.ndim or theta_arr.ndim else float(result)


def critical_re_theta(
    h: ArrayLike,
    re_theta_crit_zpg: float = RE_THETA_CRIT_ZPG,
    h_reference: float = H_REFERENCE,
    b: float = ONSET_H_SENSITIVITY,
) -> NDArray[np.float64]:
    """Local critical (neutral-stability) momentum-thickness Reynolds number.

    ``Re_theta,crit(H) = Re_theta,crit,ZPG * exp(-b * (H - H_reference))``.

    This is a project-chosen, monotonically decreasing-in-H proxy, exactly
    anchored at the sourced ZPG value at ``H = H_reference``. See module
    docstring for the sourced-vs-project-assumption distinction.
    """
    h_arr = np.asarray(h, dtype=np.float64)
    _validate_finite("h", h_arr)
    if re_theta_crit_zpg <= 0.0:
        raise ValueError(f"re_theta_crit_zpg must be positive, got {re_theta_crit_zpg}")
    result = re_theta_crit_zpg * np.exp(-b * (h_arr - h_reference))
    return result if h_arr.ndim else float(result)


def is_unstable(
    re_theta: ArrayLike,
    h: ArrayLike,
    re_theta_crit_zpg: float = RE_THETA_CRIT_ZPG,
    h_reference: float = H_REFERENCE,
    b: float = ONSET_H_SENSITIVITY,
) -> NDArray[np.bool_]:
    """Local instability indicator: True where Re_theta >= Re_theta,crit(H)."""
    re_theta_arr = np.asarray(re_theta, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    _validate_finite("re_theta", re_theta_arr)
    crit = critical_re_theta(h_arr, re_theta_crit_zpg, h_reference, b)
    result = re_theta_arr >= crit
    return result if re_theta_arr.ndim or h_arr.ndim else bool(result)


def local_amplification_rate(
    re_theta: ArrayLike,
    h: ArrayLike,
    k: float = AMPLIFICATION_COEFF,
    re_theta_crit_zpg: float = RE_THETA_CRIT_ZPG,
    h_reference: float = H_REFERENCE,
    b: float = ONSET_H_SENSITIVITY,
) -> NDArray[np.float64]:
    """Reduced-order proxy for dN/ds (s = x/c), the local amplification-rate
    density.

        dN/ds = k * (H - 1) * max(0, Re_theta - Re_theta,crit(H)) / Re_theta,crit,ZPG

    By construction this is >= 0 everywhere (never negative -- stable
    stations, where Re_theta < Re_theta,crit(H), contribute exactly zero),
    and it is continuous through the onset crossing (no artificial jump),
    since the max(0, ...) clip activates smoothly as Re_theta rises through
    Re_theta,crit(H). See module docstring for the sourced-vs-project-
    assumption distinction; ``k`` and ``b`` are project (unsourced)
    calibration constants.
    """
    re_theta_arr = np.asarray(re_theta, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    _validate_finite("re_theta", re_theta_arr)
    _validate_finite("h", h_arr)
    if k < 0.0:
        raise ValueError(f"k (amplification coefficient) must be non-negative, got {k}")

    crit = critical_re_theta(h_arr, re_theta_crit_zpg, h_reference, b)
    excess = np.clip(re_theta_arr - crit, 0.0, None)
    result = k * (h_arr - 1.0) * excess / re_theta_crit_zpg
    return result if re_theta_arr.ndim or h_arr.ndim else float(result)


def integrate_n_factor(s: ArrayLike, dn_ds: ArrayLike) -> NDArray[np.float64]:
    """Cumulative N(s) = integral_0^s (dN/ds) ds', by trapezoidal quadrature.

    Pure numerical-integration utility, independent of the aerodynamic
    proxy above -- used both for the amplification solution below and
    directly tested against analytic constant-rate / piecewise-rate cases
    (see tests/test_stability.py).

    ``s`` must be strictly increasing. ``N[0] = 0`` by construction (no
    accumulated amplification at the first station).
    """
    s_arr = np.asarray(s, dtype=np.float64)
    dn_ds_arr = np.asarray(dn_ds, dtype=np.float64)
    if s_arr.shape != dn_ds_arr.shape:
        raise ValueError("s and dn_ds must have the same shape")
    if s_arr.ndim != 1 or s_arr.size < 2:
        raise ValueError("s must be a 1D array of at least 2 stations")
    if np.any(np.diff(s_arr) <= 0.0):
        raise ValueError("s must be strictly increasing")
    _validate_finite("dn_ds", dn_ds_arr)

    n = np.empty_like(s_arr)
    n[0] = 0.0
    n[1:] = cumulative_trapezoid(dn_ds_arr, s_arr)
    return n


@dataclass(frozen=True)
class AmplificationSolution:
    """Station-by-station reduced-order stability/amplification solution."""

    s: NDArray[np.float64]                 # x/c
    x_m: NDArray[np.float64]               # dimensional streamwise coordinate, m
    u_e_mps: NDArray[np.float64]           # edge velocity, m/s
    theta_m: NDArray[np.float64]           # momentum thickness, m (from Thwaites)
    h: NDArray[np.float64]                 # shape factor (NaN outside M1 validity domain)
    lam: NDArray[np.float64]               # Thwaites pressure-gradient parameter
    re_theta: NDArray[np.float64]          # momentum-thickness Reynolds number (NaN outside domain)
    re_theta_crit: NDArray[np.float64]     # local critical Re_theta (NaN outside domain)
    amplification_active: NDArray[np.bool_]  # True where locally unstable (within valid domain)
    dn_ds: NDArray[np.float64]             # local amplification-rate density (NaN outside domain)
    n: NDArray[np.float64]                 # cumulative N-factor (NaN outside domain)
    valid: NDArray[np.bool_]               # True where the M1 attached-laminar solution is valid
    separated: NDArray[np.bool_]           # from the M1 Thwaites solution
    onset_s: float | None                  # modeled instability-onset x/c (None if never reached)
    onset_x_m: float | None
    s_sep: float | None                    # M1 laminar-separation diagnostic, x/c
    x_sep_m: float | None
    n_max: float | None                    # max N attained within the valid domain


def solve_amplification(
    op: GenericSailplaneOperatingPoint,
    thwaites_sol: ThwaitesSolution,
    k: float = AMPLIFICATION_COEFF,
    re_theta_crit_zpg: float = RE_THETA_CRIT_ZPG,
    h_reference: float = H_REFERENCE,
    b: float = ONSET_H_SENSITIVITY,
) -> AmplificationSolution:
    """Compute the reduced-order stability/amplification solution over the
    Milestone 1 Thwaites laminar boundary-layer solution.

    The valid domain is exactly the Milestone 1 attached-laminar, lambda-
    in-range domain (``~isnan(thwaites_sol.h)``) -- i.e. stability
    quantities are computed and reported only where M1 itself considers
    the underlying boundary-layer solution physically meaningful (pre-
    separation, lambda within the Thwaites correlation's validity domain).
    Beyond that domain, all stability outputs are ``NaN`` -- never
    fabricated or extrapolated.

    N(s) is integrated over the *entire* valid domain (not just from
    onset), but ``dN/ds = 0`` identically for every pre-onset station by
    construction (see :func:`local_amplification_rate`), so ``N`` is
    exactly zero up to and including the onset station, then increases
    monotonically. This avoids any artificial grid-dependent jump at the
    onset crossing.
    """
    s = thwaites_sol.s
    valid = ~np.isnan(thwaites_sol.h)

    re_theta_full = reynolds_theta(op.rho_kgm3, thwaites_sol.u_e_mps, thwaites_sol.theta_m, op.mu_pas)
    re_theta = np.where(valid, re_theta_full, np.nan)

    h_valid = np.where(valid, thwaites_sol.h, h_reference)  # placeholder inside NaN region, masked out below
    re_theta_crit_full = critical_re_theta(h_valid, re_theta_crit_zpg, h_reference, b)
    re_theta_crit = np.where(valid, re_theta_crit_full, np.nan)

    amplification_active = valid & (re_theta_full >= re_theta_crit_full)

    dn_ds_full = local_amplification_rate(
        np.where(valid, re_theta_full, 0.0), h_valid, k, re_theta_crit_zpg, h_reference, b
    )
    dn_ds = np.where(valid, dn_ds_full, np.nan)

    # Integrate over the valid domain only. thwaites_sol.s is uniform and
    # starts at 0, and `valid` is a prefix mask (True from s=0 up to the
    # M1 separation/lambda-domain boundary) for this model, so the valid
    # region is a contiguous leading block.
    valid_idx = np.where(valid)[0]
    n = np.full_like(s, np.nan)
    if valid_idx.size >= 2:
        n_valid = integrate_n_factor(s[valid_idx], np.where(valid, dn_ds_full, 0.0)[valid_idx])
        n[valid_idx] = n_valid
    elif valid_idx.size == 1:
        n[valid_idx] = 0.0

    onset_idx = np.where(amplification_active)[0]
    if onset_idx.size > 0:
        i_on = int(onset_idx[0])
        onset_s = float(s[i_on])
        onset_x_m = float(thwaites_sol.x_m[i_on])
    else:
        onset_s = None
        onset_x_m = None

    n_max = float(np.nanmax(n)) if valid_idx.size > 0 else None

    return AmplificationSolution(
        s=s,
        x_m=thwaites_sol.x_m,
        u_e_mps=thwaites_sol.u_e_mps,
        theta_m=thwaites_sol.theta_m,
        h=thwaites_sol.h,
        lam=thwaites_sol.lam,
        re_theta=re_theta,
        re_theta_crit=re_theta_crit,
        amplification_active=amplification_active,
        dn_ds=dn_ds,
        n=n,
        valid=valid,
        separated=thwaites_sol.separated,
        onset_s=onset_s,
        onset_x_m=onset_x_m,
        s_sep=thwaites_sol.s_sep,
        x_sep_m=thwaites_sol.x_sep_m,
        n_max=n_max,
    )


def interpolate_n(sol: AmplificationSolution, s_query: ArrayLike) -> NDArray[np.float64]:
    """Linearly interpolate N(s) at arbitrary query station(s) within the
    valid domain. Returns NaN for queries outside the valid s-range.
    """
    s_query_arr = np.asarray(s_query, dtype=np.float64)
    valid_idx = np.where(sol.valid)[0]
    if valid_idx.size < 2:
        raise ValueError("solution has fewer than 2 valid stations; cannot interpolate")
    s_valid = sol.s[valid_idx]
    n_valid = sol.n[valid_idx]
    result = np.interp(s_query_arr, s_valid, n_valid, left=np.nan, right=np.nan)
    return result if s_query_arr.ndim else float(result)
