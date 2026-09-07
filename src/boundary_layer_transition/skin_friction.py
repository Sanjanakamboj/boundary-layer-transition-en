"""Milestone 3: reduced-order laminar/turbulent skin-friction and section
drag bookkeeping.

**What this module is, and is not.** This module implements classical,
sourced, zero-pressure-gradient flat-plate skin-friction correlations
(laminar Blasius, turbulent 1/7-power-law/Schlichting) and combines them
with the Milestone 1 synthetic external-velocity distribution through a
simple dynamic-pressure weighting to produce an illustrative section
skin-friction drag coefficient. It does **not** solve a turbulent
pressure-gradient integral boundary-layer (e.g. a turbulent Head's-method
or CS-method momentum integral); the M1 external velocity distribution
does carry a pressure gradient, so applying a ZPG turbulent correlation
here is an explicit **drag-bookkeeping approximation**, not a full
turbulent boundary-layer solution. This is stated plainly wherever the
turbulent correlation is used.

Sourced correlations (see DESIGN.md for the full derivation and validity
discussion):

- Laminar, local: ``Cf,x = 0.664 / sqrt(Re_x)`` (Blasius; reused from
  :mod:`laminar_bl`, same source audit as Milestone 1).
- Laminar, average (0 to L): ``Cf_bar = 1.328 / sqrt(Re_L)`` (exact
  integral of the local Blasius correlation; Schlichting & Gersten;
  White, *Viscous Fluid Flow*).
- Turbulent, local: ``Cf,x = 0.0592 * Re_x^(-1/5)`` (1/7-power-law /
  Schlichting correlation for a smooth, incompressible, zero-pressure-
  gradient turbulent flat plate; White, *Viscous Fluid Flow*, 3rd ed.,
  Ch. 6; documented validity range ``5e5 <= Re_x <= 1e7``).
- Turbulent, average (0 to L): ``Cf_bar = 0.074 * Re_L^(-1/5)`` (exact
  integral of the local 1/7-power-law correlation over ``[0, L]`` -- this
  module verifies that self-consistency directly:
  ``0.074 = 1.25 * 0.0592 = (5/4) * 0.0592``, the correct factor for
  integrating an ``x^(-1/5)`` power law).

Turbulent-origin treatment (Section 5): the turbulent correlation is
evaluated using the *physical* streamwise coordinate ``x`` measured from
the real leading edge (not a virtual-origin-shifted coordinate). This is a
deliberate, documented simplification: it ignores the (well-known in the
literature, e.g. the classical Prandtl-Schlichting mixed-boundary-layer
drag correction) effect that a turbulent boundary layer inheriting a
finite momentum thickness from a preceding laminar run is effectively
"older" than a turbulent layer that started at ``x=0``. A virtual-origin
or mixed-boundary-layer correction constant exists in the literature, but
its specific tabulated correction was not independently verified against
a primary source in this project, so it is not implemented here. Using
physical ``x`` directly is the simpler, explicitly labeled alternative
this project uses instead (see DESIGN.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid

from .laminar_bl import blasius_cf, reynolds_x

#: Turbulent local skin-friction coefficient, Cf,x = TURBULENT_CF_COEFF_LOCAL * Re_x^(-1/5).
TURBULENT_CF_COEFF_LOCAL = 0.0592

#: Turbulent average skin-friction coefficient (0 to L), Cf_bar = TURBULENT_CF_COEFF_AVERAGE * Re_L^(-1/5).
TURBULENT_CF_COEFF_AVERAGE = 0.074

#: Laminar average skin-friction coefficient (0 to L), Cf_bar = LAMINAR_CF_COEFF_AVERAGE / sqrt(Re_L).
LAMINAR_CF_COEFF_AVERAGE = 1.328

#: Documented validity range of the turbulent flat-plate correlation above
#: (smooth wall, incompressible, zero-pressure-gradient).
TURBULENT_CF_RE_MIN = 5.0e5
TURBULENT_CF_RE_MAX = 1.0e7


def _validate_positive_scalar(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")


def laminar_cf_average(v_mps: float, l_m: ArrayLike, nu_m2s: float) -> NDArray[np.float64]:
    """Average laminar (Blasius) skin-friction coefficient over [0, L],
    Cf_bar = 1.328 / sqrt(Re_L). Exact integral of the local Blasius Cf,x.
    """
    re_l = reynolds_x(v_mps, l_m, nu_m2s)
    l_arr = np.asarray(l_m, dtype=np.float64)
    with np.errstate(divide="ignore"):
        result = LAMINAR_CF_COEFF_AVERAGE / np.sqrt(re_l)
    return result if l_arr.ndim else float(result)


def turbulent_cf_local(v_mps: float, x_m: ArrayLike, nu_m2s: float) -> NDArray[np.float64]:
    """Local turbulent skin-friction coefficient, Cf,x = 0.0592 * Re_x^(-1/5).

    Genuinely singular (returns ``inf``, not a fabricated finite value) at
    x=0, exactly like the laminar Blasius Cf,x -- both are formal flat-
    plate wall-shear singularities at the leading edge.
    """
    re_x = reynolds_x(v_mps, x_m, nu_m2s)
    x_arr = np.asarray(x_m, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = TURBULENT_CF_COEFF_LOCAL * np.power(np.asarray(re_x, dtype=np.float64), -0.2)
    return result if x_arr.ndim else float(result)


def turbulent_cf_average(v_mps: float, l_m: ArrayLike, nu_m2s: float) -> NDArray[np.float64]:
    """Average turbulent skin-friction coefficient over [0, L],
    Cf_bar = 0.074 * Re_L^(-1/5). Exact integral of the local 1/7-power-law
    correlation (0.074 = 1.25 * 0.0592, the correct integration factor for
    an x^(-1/5) power law).
    """
    re_l = reynolds_x(v_mps, l_m, nu_m2s)
    l_arr = np.asarray(l_m, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = TURBULENT_CF_COEFF_AVERAGE * np.power(np.asarray(re_l, dtype=np.float64), -0.2)
    return result if l_arr.ndim else float(result)


def is_turbulent_correlation_valid(re_x: ArrayLike) -> NDArray[np.bool_]:
    """Whether re_x lies within the documented validity range of the
    turbulent flat-plate correlation, [5e5, 1e7]. This does not block
    computation (see module docstring); it is a diagnostic flag used to
    report when a station falls outside the range the correlation was
    validated over -- most commonly for stations shortly after an assumed
    transition at low chord Reynolds number.
    """
    re_x_arr = np.asarray(re_x, dtype=np.float64)
    result = (re_x_arr >= TURBULENT_CF_RE_MIN) & (re_x_arr <= TURBULENT_CF_RE_MAX)
    return result if re_x_arr.ndim else bool(result)


def transitioned_cf_distribution(
    v_mps: float, x_m: ArrayLike, nu_m2s: float, x_tr_m: float
) -> NDArray[np.float64]:
    """Piecewise local skin-friction coefficient: laminar (Blasius) for
    ``x < x_tr_m``, turbulent (1/7-power-law) for ``x >= x_tr_m``.

    Both correlations use the physical x from the real leading edge (see
    module docstring re: turbulent-origin treatment). Pass
    ``x_tr_m = math.inf`` for a fully laminar reference, or
    ``x_tr_m = 0.0`` for a fully turbulent reference -- no special-casing
    is required since the ``>=`` threshold comparison handles both
    naturally.
    """
    if not np.isfinite(x_tr_m) and x_tr_m not in (np.inf, -np.inf):
        raise ValueError(f"x_tr_m must be finite or +/-inf, got {x_tr_m}")
    x_arr = np.asarray(x_m, dtype=np.float64)
    if np.any(~np.isfinite(x_arr)) or np.any(x_arr < 0.0):
        raise ValueError("x_m must be finite and non-negative")

    turbulent_mask = x_arr >= x_tr_m
    cf = np.where(
        turbulent_mask,
        turbulent_cf_local(v_mps, x_arr, nu_m2s),
        blasius_cf(v_mps, x_arr, nu_m2s),
    )
    return cf if x_arr.ndim else float(cf)


def section_cf_drag(
    s: ArrayLike,
    cf: ArrayLike,
    u_e_ratio: ArrayLike,
    two_surfaces: bool = True,
) -> float:
    """Section skin-friction drag coefficient,

        C_d,f = (2 if two_surfaces else 1) * integral_0^1 Cf(s) * [U_e(s)/V_inf]^2 ds

    Derivation (see DESIGN.md for the full normalization discussion): for a
    wall-shear stress tau_w(x) = Cf(x) * 0.5 * rho * U_e(x)^2 (local edge
    dynamic pressure), the drag force per unit span on one surface is
    ``integral_0^c tau_w dx``; nondimensionalizing by the freestream
    dynamic pressure ``0.5 rho V_inf^2`` and the chord ``c``, and
    substituting ``x = s c``, gives exactly the nondimensional-in-``s``
    integral above (the chord cancels). ``two_surfaces=True`` doubles the
    single-surface integral, representing both surfaces with the *same*
    representative ``U_e(s)/V_inf`` shape -- an explicit project
    simplification (this project's synthetic external-flow model is not
    upper/lower-surface-resolved); set ``two_surfaces=False`` to report a
    single-surface value without that doubling.

    No NaN/inf may appear in ``cf`` -- this function raises rather than
    silently propagating an invalid value into the integral (see module
    docstring: the discrete quadrature grid should exclude the exact,
    literally singular leading-edge station; both source correlations are
    integrable in the continuum limit despite the pointwise singularity at
    x=0).
    """
    s_arr = np.asarray(s, dtype=np.float64)
    cf_arr = np.asarray(cf, dtype=np.float64)
    ue_ratio_arr = np.asarray(u_e_ratio, dtype=np.float64)
    if not (s_arr.shape == cf_arr.shape == ue_ratio_arr.shape):
        raise ValueError("s, cf, and u_e_ratio must all have the same shape")
    if s_arr.ndim != 1 or s_arr.size < 2:
        raise ValueError("s must be a 1D array of at least 2 stations")
    if np.any(np.diff(s_arr) <= 0.0):
        raise ValueError("s must be strictly increasing")
    if np.any(~np.isfinite(cf_arr)):
        raise ValueError(
            "cf contains non-finite values (e.g. the exact leading-edge "
            "singularity) -- exclude x=0 from the integration grid"
        )
    if np.any(~np.isfinite(ue_ratio_arr)):
        raise ValueError("u_e_ratio must be finite everywhere")

    integrand = cf_arr * ue_ratio_arr**2
    integral = float(trapezoid(integrand, s_arr))
    return (2.0 if two_surfaces else 1.0) * integral


@dataclass(frozen=True)
class DragCase:
    """A single named skin-friction drag bookkeeping case."""

    label: str
    x_tr_s: float | None   # x/c of the assumed transition (None = fully laminar)
    cd_f: float


def build_drag_grid(chord_m: float, n_points: int = 4001, s_min: float = 1.0e-6) -> NDArray[np.float64]:
    """Geometrically graded s = x/c grid for skin-friction integration,
    deliberately excluding the exact leading-edge station s=0 (where both
    the laminar and turbulent flat-plate correlations are formally
    singular, though integrably so -- see module docstring).

    The grid is log-spaced (``s_min * (1/s_min)**linspace(0,1,n)``) rather
    than uniform, because the laminar Cf,x ~ x^(-1/2) singularity converges
    very slowly under trapezoidal quadrature on a *uniform* grid (~5%
    error at n=4001 in this project's baseline case) but converges to
    <0.1% at the same n once points are concentrated near the singular end
    -- verified directly against the closed-form average-Cf formulas in
    tests/test_skin_friction.py. ``s_min`` is a small positive offset;
    convergence as ``s_min -> 0`` and ``n_points -> infinity`` is checked
    in the same tests.
    """
    if chord_m <= 0.0 or not np.isfinite(chord_m):
        raise ValueError(f"chord_m must be finite and positive, got {chord_m}")
    if n_points < 3:
        raise ValueError(f"n_points must be >= 3, got {n_points}")
    if not (0.0 < s_min < 1.0):
        raise ValueError(f"s_min must be in (0, 1), got {s_min}")
    return s_min * (1.0 / s_min) ** np.linspace(0.0, 1.0, n_points)
