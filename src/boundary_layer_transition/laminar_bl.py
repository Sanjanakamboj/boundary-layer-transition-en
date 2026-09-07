"""Laminar boundary-layer solutions: Blasius (ZPG) and Thwaites (pressure-gradient-aware).

Two independent levels of fidelity are implemented, per Milestone 1 scope:

A. **Blasius zero-pressure-gradient (ZPG) reference** -- the classical exact
   (numerically tabulated, here used via its standard closed-form
   engineering correlations) flat-plate laminar boundary-layer solution.
   Source: H. Schlichting & K. Gersten, *Boundary-Layer Theory*, 9th ed.,
   Ch. 7 (Blasius solution); F. M. White, *Viscous Fluid Flow*, 3rd ed.,
   Ch. 4, Table 4.1 / Eqs. (4.55)-(4.59) tabulate the identical constants
   used here (delta_99, delta*, theta, Cf coefficients).

B. **Thwaites' laminar momentum-integral method** -- an approximate
   pressure-gradient-aware method built on the momentum-integral equation,
   closed with correlation functions fitted to Thwaites' original tabulated
   data. Source: B. Thwaites, "Approximate calculation of the laminar
   boundary layer," *Aeronautical Quarterly*, 1(3), 1949, pp. 245-280 (the
   original method); the two-branch polynomial curve fit implemented here
   (functions l(lambda), H(lambda)) reproduces the fit tabulated in F. M.
   White, *Viscous Fluid Flow*, 3rd ed., Table 4.1 / Eqs. (4.75)-(4.76),
   itself attributed to Cebeci & Bradshaw, *Momentum Transfer in Boundary
   Layers*, 1977. See DESIGN.md for the full source audit, including the
   documented ambiguity between Thwaites' original tabulated values and
   this widely reproduced polynomial approximation.

Equations
---------
Blasius (constant U_inf, x measured from the leading edge of an
equivalent flat plate):

    Re_x     = U_inf * x / nu
    delta_99 = 5.0    * x / sqrt(Re_x)
    delta*   = 1.7208 * x / sqrt(Re_x)
    theta    = 0.664  * x / sqrt(Re_x)
    H        = delta*/theta  (constant, ~2.59)
    Cf,x     = 0.664 / sqrt(Re_x)

Thwaites (edge velocity U_e(x) from an external prescribed distribution):

    theta(x)^2 = 0.45 * nu / U_e(x)^6 * integral_0^x U_e(x')^5 dx'
    lambda(x)  = theta(x)^2 / nu * dU_e/dx
    l(lambda)  : dimensionless wall-shear correlation function
    H(lambda)  : shape-factor correlation function
    delta*(x)  = H(lambda) * theta(x)
    Cf,x(x)    = 2 * nu * l(lambda) / (U_e(x) * theta(x))

    Separation is predicted where l(lambda) first crosses zero (wall shear
    vanishes), conventionally near lambda ~= -0.09 (Thwaites) / -0.0842
    (Cebeci-Bradshaw fit used here; the exact root of the implemented
    correlation is computed and reported, not hard-coded).

Leading-edge treatment (x = 0)
-------------------------------
delta_99, delta*, and theta are mathematically well-defined (and equal to
zero) in the limit x -> 0, because x / sqrt(Re_x) = sqrt(x * nu / U_inf) is
a removable 0/0 form. This module evaluates that limit analytically (return
0.0 at x=0) rather than performing a literal floating-point 0/0 division.

Cf,x, by contrast, is genuinely singular (Cf,x -> infinity) at the
flat-plate leading edge -- this is a true, non-removable singularity of the
Blasius solution (infinite wall shear at the stagnation-like origin of the
flat-plate idealization). This module returns ``inf`` at x=0 for Cf,x,
explicitly, rather than fabricating a finite value.

Thwaites' theta(0) = 0 boundary condition assumes a boundary layer that
originates at x=0 with a finite, nonzero edge velocity U_e(0) (a flat-plate
-like leading edge), not a true stagnation-point airfoil leading edge. This
is a deliberate Milestone 1 modeling simplification; see DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from .external_flow import SyntheticVelocityDistribution

# ---------------------------------------------------------------------------
# Blasius zero-pressure-gradient reference
# ---------------------------------------------------------------------------

#: Blasius delta_99 coefficient (Schlichting & Gersten; White Table 4.1).
BLASIUS_C_DELTA99 = 5.0
#: Blasius displacement-thickness coefficient.
BLASIUS_C_DSTAR = 1.7208
#: Blasius momentum-thickness coefficient.
BLASIUS_C_THETA = 0.664
#: Blasius skin-friction coefficient.
BLASIUS_C_CF = 0.664
#: Blasius shape factor, computed self-consistently as delta*/theta from the
#: coefficients above (~2.5916), rather than an independently rounded
#: literature constant, to guarantee internal consistency. Commonly rounded
#: to 2.59 in textbooks; a more precise numerical Blasius solution gives
#: H = 2.5911. See DESIGN.md for discussion of this ~0.02% ambiguity.
BLASIUS_H = BLASIUS_C_DSTAR / BLASIUS_C_THETA


def _validate_x(x: NDArray[np.float64]) -> None:
    if np.any(~np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("x must be finite and non-negative")


def _validate_positive_scalar(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")


def reynolds_x(v_mps: float, x_m: ArrayLike, nu_m2s: float) -> NDArray[np.float64]:
    """Local Reynolds number Re_x = V * x / nu.

    ``x_m`` may be a scalar or array; ``v_mps`` and ``nu_m2s`` are scalars.
    Re_x = 0 exactly at x = 0 (no division involved).
    """
    _validate_positive_scalar("v_mps", v_mps)
    _validate_positive_scalar("nu_m2s", nu_m2s)
    x_arr = np.asarray(x_m, dtype=np.float64)
    _validate_x(x_arr)
    return v_mps * x_arr / nu_m2s


def _blasius_linear_quantity(coeff: float, v_mps: float, x_m: ArrayLike, nu_m2s: float) -> NDArray[np.float64]:
    """Evaluate coeff * x / sqrt(Re_x), resolving the removable 0/0 at x=0
    via the algebraically equivalent, non-singular form
    coeff * sqrt(x * nu / V_inf).
    """
    _validate_positive_scalar("v_mps", v_mps)
    _validate_positive_scalar("nu_m2s", nu_m2s)
    x_arr = np.asarray(x_m, dtype=np.float64)
    _validate_x(x_arr)
    result = coeff * np.sqrt(x_arr * nu_m2s / v_mps)
    return result if x_arr.ndim else float(result)


def blasius_delta99(v_mps: float, x_m: ArrayLike, nu_m2s: float) -> NDArray[np.float64]:
    """Blasius 99% boundary-layer thickness, delta_99 ~= 5 x / sqrt(Re_x)."""
    return _blasius_linear_quantity(BLASIUS_C_DELTA99, v_mps, x_m, nu_m2s)


def blasius_delta_star(v_mps: float, x_m: ArrayLike, nu_m2s: float) -> NDArray[np.float64]:
    """Blasius displacement thickness, delta* ~= 1.7208 x / sqrt(Re_x)."""
    return _blasius_linear_quantity(BLASIUS_C_DSTAR, v_mps, x_m, nu_m2s)


def blasius_theta(v_mps: float, x_m: ArrayLike, nu_m2s: float) -> NDArray[np.float64]:
    """Blasius momentum thickness, theta ~= 0.664 x / sqrt(Re_x)."""
    return _blasius_linear_quantity(BLASIUS_C_THETA, v_mps, x_m, nu_m2s)


def blasius_cf(v_mps: float, x_m: ArrayLike, nu_m2s: float) -> NDArray[np.float64]:
    """Blasius local skin-friction coefficient, Cf,x = 0.664 / sqrt(Re_x).

    Genuinely singular (returns ``inf``, not a fabricated finite value) at
    x=0; see module docstring.
    """
    re_x = reynolds_x(v_mps, x_m, nu_m2s)
    with np.errstate(divide="ignore"):
        result = BLASIUS_C_CF / np.sqrt(re_x)
    return result if np.asarray(x_m).ndim else float(result)


# ---------------------------------------------------------------------------
# Thwaites pressure-gradient-aware laminar integral method
# ---------------------------------------------------------------------------

#: Validity domain of the two-branch Thwaites correlation implemented below
#: (White, Viscous Fluid Flow, Table 4.1 / Cebeci-Bradshaw fit).
LAMBDA_MIN_VALID = -0.10
LAMBDA_MAX_VALID = 0.10


def thwaites_shear_function(lam: ArrayLike) -> NDArray[np.float64]:
    """Thwaites dimensionless wall-shear correlation function l(lambda).

    l(lambda) = tau_w * theta / (mu * U_e); wall shear vanishes (laminar
    separation) where l(lambda) = 0.

    Piecewise polynomial fit (White, Viscous Fluid Flow, Table 4.1):
        l(lambda) = 0.22 + 1.57*lambda - 1.8*lambda**2         (0 <= lambda <= 0.1)
        l(lambda) = 0.22 + 1.402*lambda + 0.018*lambda/(lambda+0.107)  (-0.1 <= lambda < 0)
    """
    lam_arr = np.asarray(lam, dtype=np.float64)
    pos = lam_arr >= 0.0
    l_pos = 0.22 + 1.57 * lam_arr - 1.8 * lam_arr**2
    l_neg = 0.22 + 1.402 * lam_arr + 0.018 * lam_arr / (lam_arr + 0.107)
    result = np.where(pos, l_pos, l_neg)
    return result if lam_arr.ndim else float(result)


def thwaites_shape_factor(lam: ArrayLike) -> NDArray[np.float64]:
    """Thwaites shape-factor correlation function H(lambda).

    Piecewise polynomial fit (White, Viscous Fluid Flow, Table 4.1):
        H(lambda) = 2.61 + 0.0731/... -- see branches below.
        H(lambda) = 2.61 - 3.75*lambda + 5.24*lambda**2   (0 <= lambda <= 0.1)
        H(lambda) = 2.088 + 0.0731/(lambda + 0.14)        (-0.1 <= lambda < 0)
    """
    lam_arr = np.asarray(lam, dtype=np.float64)
    pos = lam_arr >= 0.0
    h_pos = 2.61 - 3.75 * lam_arr + 5.24 * lam_arr**2
    h_neg = 2.088 + 0.0731 / (lam_arr + 0.14)
    result = np.where(pos, h_pos, h_neg)
    return result if lam_arr.ndim else float(result)


def thwaites_separation_lambda() -> float:
    """Root of l(lambda) = 0 for the negative branch, i.e. the lambda at
    which this module's correlation predicts zero wall shear (laminar
    separation). Computed numerically from the correlation actually
    implemented above (not hard-coded from a secondary source), then
    documented against the literature value (~-0.09) in DESIGN.md.
    """
    return brentq(lambda lam: thwaites_shear_function(lam), LAMBDA_MIN_VALID, 0.0)


@dataclass(frozen=True)
class ThwaitesSolution:
    """Station-by-station Thwaites laminar boundary-layer solution."""

    s: NDArray[np.float64]              # x/c, nondimensional
    x_m: NDArray[np.float64]            # dimensional streamwise coordinate, m
    u_e_mps: NDArray[np.float64]        # edge velocity, m/s
    du_e_dx: NDArray[np.float64]        # dU_e/dx, 1/s
    re_x: NDArray[np.float64]           # local Reynolds number U_e x / nu
    theta_m: NDArray[np.float64]        # momentum thickness, m
    lam: NDArray[np.float64]            # Thwaites pressure-gradient parameter
    h: NDArray[np.float64]              # shape factor H(lambda), NaN outside validity domain
    delta_star_m: NDArray[np.float64]   # displacement thickness, m, NaN outside validity domain
    cf: NDArray[np.float64]             # local skin-friction coefficient, NaN outside validity domain
    valid: NDArray[np.bool_]            # True where lambda is within the correlation's validity domain
    separated: NDArray[np.bool_]        # True from the first predicted-separation station onward
    x_sep_m: float | None               # interpolated separation location, m (None if no separation)
    s_sep: float | None                 # interpolated separation location, x/c (None if no separation)


def solve_thwaites(
    dist: SyntheticVelocityDistribution,
    nu_m2s: float,
    n_points: int = 2001,
) -> ThwaitesSolution:
    """Integrate Thwaites' method for a prescribed U_e(x) distribution.

    Parameters
    ----------
    dist : SyntheticVelocityDistribution
        The external velocity distribution to integrate against.
    nu_m2s : float
        Kinematic viscosity, m^2/s.
    n_points : int
        Number of uniformly spaced chordwise stations in s = x/c, including
        both endpoints. Must be >= 3. Used for convergence testing.

    Numerical method
    -----------------
    The Thwaites integral ``integral_0^x U_e^5 dx'`` is evaluated by
    cumulative trapezoidal quadrature (``scipy.integrate.cumulative_trapezoid``)
    on a uniform grid in x. theta(0) = 0 is enforced exactly (the integral
    is identically zero at x=0, and U_e(0) is finite and nonzero for this
    distribution, so no division by zero occurs at the leading edge).
    """
    if n_points < 3:
        raise ValueError(f"n_points must be >= 3, got {n_points}")
    _validate_positive_scalar("nu_m2s", nu_m2s)

    s = np.linspace(0.0, 1.0, n_points)
    x_m = s * dist.chord_m
    u_e = dist.u_e(x_m)
    du_e_dx = dist.du_e_dx(x_m)

    integrand = u_e**5
    integral = np.empty_like(x_m)
    integral[0] = 0.0
    integral[1:] = cumulative_trapezoid(integrand, x_m)

    theta_sq = 0.45 * nu_m2s / u_e**6 * integral
    theta_sq = np.clip(theta_sq, 0.0, None)  # guard against negative round-off at x=0
    theta_m = np.sqrt(theta_sq)

    lam = (theta_sq / nu_m2s) * du_e_dx

    valid = (lam >= LAMBDA_MIN_VALID) & (lam <= LAMBDA_MAX_VALID)

    l_lam = thwaites_shear_function(lam)
    separated = np.zeros_like(s, dtype=bool)
    x_sep_m: float | None = None
    s_sep: float | None = None

    # Separation: first index (beyond the leading edge) where l(lambda)
    # crosses from positive to non-positive.
    sep_candidates = np.where((l_lam[:-1] > 0.0) & (l_lam[1:] <= 0.0))[0]
    if sep_candidates.size > 0:
        i0 = int(sep_candidates[0])
        i1 = i0 + 1
        # Linear interpolation of l(lambda) in x between the bracketing
        # stations to locate the zero crossing.
        l0, l1 = l_lam[i0], l_lam[i1]
        frac = l0 / (l0 - l1) if (l0 - l1) != 0.0 else 0.0
        x_sep_m = float(x_m[i0] + frac * (x_m[i1] - x_m[i0]))
        s_sep = float(s[i0] + frac * (s[i1] - s[i0]))
        separated[i1:] = True

    # H, delta*, and Cf are only reported where lambda is within the
    # correlation's validity domain AND the station has not already passed
    # the predicted separation point. Past separation, lambda can drift
    # back into the nominal [-0.1, 0.1] range (e.g. as dU_e/dx flattens
    # near the trailing edge) even though the attached-flow correlation is
    # no longer physically meaningful there -- reporting H/Cf in that
    # region would silently fabricate a plausible-looking but invalid
    # result, which is explicitly disallowed (see module docstring).
    reportable = valid & ~separated
    h = np.where(reportable, thwaites_shape_factor(lam), np.nan)
    delta_star_m = np.where(reportable, h * theta_m, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        cf = np.where(reportable & (u_e > 0) & (theta_m > 0), 2.0 * nu_m2s * l_lam / (u_e * theta_m), np.nan)
    cf[0] = np.nan  # theta=0 at x=0 -> Cf,x formally undefined (0/0), not fabricated

    re_x = u_e * x_m / nu_m2s

    return ThwaitesSolution(
        s=s,
        x_m=x_m,
        u_e_mps=u_e,
        du_e_dx=du_e_dx,
        re_x=re_x,
        theta_m=theta_m,
        lam=lam,
        h=h,
        delta_star_m=delta_star_m,
        cf=cf,
        valid=valid,
        separated=separated,
        x_sep_m=x_sep_m,
        s_sep=s_sep,
    )
