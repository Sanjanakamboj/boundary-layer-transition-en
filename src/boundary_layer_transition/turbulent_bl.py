"""Milestone 4: pressure-gradient turbulent integral boundary layer (Head's
entrainment method).

**What this module is, and is not.** This implements Head's (1958) turbulent
boundary-layer entrainment method -- a classical two-equation integral
method that propagates momentum thickness ``theta`` and an entrainment
shape parameter ``H1`` along the actual (pressure-gradient) Milestone 1
external velocity distribution ``U_e(x)``, closed with the Ludwieg-Tillmann
(1950) skin-friction correlation. It is **not** a field (RANS/LES) solution
and **not** independently validated against CFD or experimental data in
this project; it is a well-established, textbook-level reduced-order
method, applied here to replace Milestone 3's zero-pressure-gradient
flat-plate turbulent bookkeeping downstream of an assumed transition
station.

Source audit (see DESIGN.md for the full discussion, including a
cross-check that corrected an apparent transcription gap in one secondary
source):

- **Head, M. R.** (1958), "Entrainment in the Turbulent Boundary Layer,"
  ARC R&M 3152 -- the original entrainment concept and the ``H1``
  (entrainment shape) parameter.
- **Green, J. E., Weeks, D. J., and Brooman, J. W. F.** (1973), RAE TR
  72231 / ARC R&M 3791 -- widely cited as the source of the specific
  ``H1(H)`` and ``F(H1) = d(U_e theta H1)/dx / U_e`` polynomial-power-law
  correlations used in most modern implementations of "Head's method"
  (including this one); confirmed via cross-referenced web search and a
  peer-reviewed journal article (Cambridge Core, *Flow*, "An extension of
  Thwaites' method for turbulent boundary layers") that reproduces the
  exact equations and constants used below.
- **Ludwieg, H., and Tillmann, W.** (1950), "Investigations of the Wall
  Shearing Stress in Turbulent Boundary Layers," NACA TM 1285 -- the local
  skin-friction correlation ``Cf = 0.246 * 10^(-0.678 H) * Re_theta^(-0.268)``.

Governing equations
--------------------
Momentum-integral equation (2D, incompressible)::

    dtheta/dx + (2 + H) * (theta/U_e) * dU_e/dx = Cf / 2

Entrainment closure (Head; Green-Weeks-Brooman form)::

    (1/U_e) * d(U_e theta H1)/dx = F(H1) = 0.0299 * (H1 - 3.0)^(-0.6169)

    ==> dH1/dx = [F(H1) - (theta H1 / U_e) dU_e/dx - H1 dtheta/dx] / theta

Shape-factor / entrainment-shape-factor correlation (Green-Weeks-Brooman
fit to Head's data; see DESIGN.md for the continuity cross-check applied
to the low-H branch)::

    H1(H) = 0.8234 * (H - 1.1)^(-1.287) + 3.3     for H <= 1.6
    H1(H) = 1.55   * (H - 0.6778)^(-3.064) + 3.3   for H  > 1.6

``H(H1)`` (needed by the momentum equation and the skin-friction closure)
is obtained by numerically inverting the monotonic relation above.

Skin-friction closure (Ludwieg-Tillmann)::

    Cf = 0.246 * 10^(-0.678 H) * Re_theta^(-0.268)

Separation / validity criterion
--------------------------------
``H1(H) -> 3.3`` only as ``H -> infinity`` -- i.e. ``H1 = 3.3`` is the
correlation's own asymptotic separation limit. This module treats
``H1`` crossing down through :data:`H1_SEPARATION_THRESHOLD` (a small,
documented margin above the exact 3.3 asymptote, needed because 3.3 is
reached only in the limit and the ``H(H1)`` inversion becomes numerically
ill-conditioned there) as **turbulent separation**, detected via a
terminal ``solve_ivp`` event -- not inferred from "Cf became small."
Non-physical states (``theta <= 0``, or the ``H1`` state leaving the
correlation's domain, ``H1 <= 3.0``) are detected the same way and
reported as :class:`TurbulentBLStatus.INVALID_CLOSURE`, never silently
clipped.

Transition initialization
--------------------------
See :func:`transition_initial_state`. Two cases:

1. **Prescribed/separation-triggered transition at a station within the
   Milestone 1 valid laminar domain** (``x_tr_m <= thwaites_sol.x_sep_m``,
   or no separation predicted): ``theta`` is initialized by interpolating
   the M1 Thwaites momentum thickness at ``x_tr_m`` (continuous handoff),
   and the initial shape factor ``H_tr`` is a project-chosen nominal value
   (see below), converted to ``H1_tr`` via the correlation above. Using a
   station beyond ``x_sep_m`` is explicitly rejected (the Milestone 1
   laminar solution is not considered valid there -- see M1 DESIGN.md).
2. **"Fully turbulent from the leading edge"** (``x_tr_m`` at or below a
   small numerical offset): there is no laminar state to hand off from;
   ``theta`` is seeded from the classical 1/7-power-law flat-plate
   momentum-thickness scaling ``theta/x = 0.037 Re_x^(-1/5)`` -- derived
   here (not an independent literature citation) by integrating this
   project's own Milestone 3 turbulent flat-plate ``Cf,x = 0.0592
   Re_x^(-1/5)`` correlation under the momentum-integral equation at zero
   pressure gradient with the nominal constant-``H`` closure below (shown
   in the module docstring math above); this keeps the flat-plate and
   integral-method treatments of the same physical idealization mutually
   consistent.

**Nominal initial shape factor**, ``H_TR_NOMINAL = 72/56 ~= 1.286``: the
zero-pressure-gradient 1/7-power-law turbulent profile's shape factor
(``delta* = delta/8``, ``theta = 7 delta/72``, so ``H = delta*/theta =
72/56``) -- the same idealized profile family underlying the Milestone 3
turbulent flat-plate correlations, used here as a documented, sourced-via-
derivation (not fitted-to-produce-a-desired-drag-result) nominal choice.
Sensitivity to this choice is quantified in
scripts/turbulent_boundary_layer_study.py and tests/test_turbulent_bl.py
(a modest plausible range around the nominal value), per the explicit
project instruction to quantify (not hide) this sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .external_flow import SyntheticVelocityDistribution
from .laminar_bl import ThwaitesSolution, blasius_cf
from .stability import reynolds_theta

# ---------------------------------------------------------------------------
# Closure constants (see module docstring for sources)
# ---------------------------------------------------------------------------

#: Ludwieg-Tillmann (1950) local skin-friction correlation constants:
#: Cf = LT_A * 10^(LT_B * H) * Re_theta^LT_C
LT_A = 0.246
LT_B = -0.678
LT_C = -0.268

#: Green-Weeks-Brooman entrainment closure: F(H1) = ENTRAINMENT_COEFF *
#: (H1 - ENTRAINMENT_H1_MIN)^ENTRAINMENT_EXP
ENTRAINMENT_COEFF = 0.0299
ENTRAINMENT_EXP = -0.6169
ENTRAINMENT_H1_MIN = 3.0

#: H1(H) correlation (Green-Weeks-Brooman fit to Head's data), both
#: branches include the same +3.3 additive offset -- verified in this
#: session via a continuity cross-check at the branch boundary H=1.6 (see
#: DESIGN.md): omitting it on the low-H branch, as one secondary source's
#: quoted text appeared to, gives a ~2.6x discontinuity there, while
#: including it agrees with the high-H branch to within ~0.4%.
H1_OFFSET = 3.3
H1_BRANCH_SWITCH_H = 1.6
H1_LOW_H_COEFF = 0.8234
H1_LOW_H_H0 = 1.1
H1_LOW_H_EXP = -1.287
H1_HIGH_H_COEFF = 1.55
H1_HIGH_H_H0 = 0.6778
H1_HIGH_H_EXP = -3.064

#: H1(H) -> 3.3 only as H -> infinity; this module declares turbulent
#: separation when H1 crosses down through this documented margin above
#: that asymptote (a numerical necessity, not an independent physical
#: criterion beyond "the correlation says separation as H1 -> 3.3").
H1_SEPARATION_THRESHOLD = 3.32

#: Beyond this H the state is considered a closure breakdown (INVALID_CLOSURE)
#: rather than a smooth approach to separation; a safety net, not expected
#: to be reached before the H1 separation event in normal use.
H_MAX_VALID = 15.0

#: Nominal initial turbulent shape factor at transition: the ZPG 1/7-power-
#: law value, H = 72/56 (see module docstring derivation).
H_TR_NOMINAL = 72.0 / 56.0

#: Coefficient in the 1/7-power-law flat-plate theta/x = COEFF * Re_x^-0.2
#: scaling used to seed a "fully turbulent from the leading edge" case
#: (derived from this project's own Cf,x = 0.0592 Re_x^-0.2 correlation;
#: see module docstring).
FLAT_PLATE_THETA_COEFF = 0.037

#: Small positive offset (as a fraction of chord) used as the effective
#: leading-edge station for a "fully turbulent from the leading edge" case,
#: avoiding the exact (integrable but singular) x=0 station -- directly
#: analogous to skin_friction.build_drag_grid's s_min treatment.
FLAT_PLATE_LE_OFFSET_FRAC = 1.0e-4


class TurbulentBLStatus(str, Enum):
    """Outcome of integrating Head's method from a transition station."""

    COMPLETED_TO_TE = "COMPLETED_TO_TE"
    TURBULENT_SEPARATION = "TURBULENT_SEPARATION"
    INVALID_CLOSURE = "INVALID_CLOSURE"
    INTEGRATION_FAILURE = "INTEGRATION_FAILURE"


def _validate_positive_scalar(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")


def h1_of_h(h: ArrayLike) -> NDArray[np.float64]:
    """Entrainment shape factor H1 as a function of shape factor H.

    Piecewise Green-Weeks-Brooman fit to Head's data (see module docstring
    for the branch-continuity correction applied in this implementation).
    Domain: H > 1.1 (the low-H branch is singular at H=1.1).
    """
    h_arr = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h_arr)) or np.any(h_arr <= H1_LOW_H_H0):
        raise ValueError(f"H must be finite and > {H1_LOW_H_H0}, got {h}")

    low = h_arr <= H1_BRANCH_SWITCH_H
    h1_low = H1_LOW_H_COEFF * np.power(h_arr - H1_LOW_H_H0, H1_LOW_H_EXP) + H1_OFFSET
    h1_high = H1_HIGH_H_COEFF * np.power(h_arr - H1_HIGH_H_H0, H1_HIGH_H_EXP) + H1_OFFSET
    result = np.where(low, h1_low, h1_high)
    return result if h_arr.ndim else float(result)


def h_of_h1(h1: float) -> float:
    """Numerically invert h1_of_h: solve H1(H) = h1 for H > 1.1.

    h1_of_h(H) is monotonically decreasing from +infinity (as H -> 1.1+)
    to 3.3 (as H -> infinity), so the inverse is unique for h1 > 3.3.
    Scalar-only (used pointwise inside the ODE right-hand side).
    """
    if not np.isfinite(h1) or h1 <= H1_OFFSET:
        raise ValueError(f"h1 must be finite and > {H1_OFFSET}, got {h1}")

    def residual(h: float) -> float:
        return h1_of_h(h) - h1

    h_lo, h_hi = 1.1 + 1.0e-9, 1.6
    # Expand the upper bracket until it brackets the root (h1_of_h is
    # monotonically decreasing, so residual(h_hi) must go negative).
    while residual(h_hi) > 0.0 and h_hi < H_MAX_VALID * 10:
        h_hi *= 1.5
    if residual(h_lo) < 0.0 or residual(h_hi) > 0.0:
        raise ValueError(f"h1={h1} could not be bracketed for inversion")
    return brentq(residual, h_lo, h_hi, xtol=1e-12, rtol=1e-12)


def ludwieg_tillmann_cf(h: ArrayLike, re_theta: ArrayLike) -> NDArray[np.float64]:
    """Ludwieg-Tillmann (1950) local turbulent skin-friction correlation,
    Cf = 0.246 * 10^(-0.678 H) * Re_theta^(-0.268).
    """
    h_arr = np.asarray(h, dtype=np.float64)
    re_arr = np.asarray(re_theta, dtype=np.float64)
    if np.any(~np.isfinite(h_arr)) or np.any(h_arr <= 1.0):
        raise ValueError("H must be finite and > 1")
    if np.any(~np.isfinite(re_arr)) or np.any(re_arr <= 0.0):
        raise ValueError("Re_theta must be finite and positive")
    result = LT_A * np.power(10.0, LT_B * h_arr) * np.power(re_arr, LT_C)
    return result if (h_arr.ndim or re_arr.ndim) else float(result)


def entrainment_rate(h1: ArrayLike) -> NDArray[np.float64]:
    """Green-Weeks-Brooman entrainment closure, F(H1) = 0.0299 * (H1-3)^-0.6169."""
    h1_arr = np.asarray(h1, dtype=np.float64)
    if np.any(~np.isfinite(h1_arr)) or np.any(h1_arr <= ENTRAINMENT_H1_MIN):
        raise ValueError(f"H1 must be finite and > {ENTRAINMENT_H1_MIN}")
    result = ENTRAINMENT_COEFF * np.power(h1_arr - ENTRAINMENT_H1_MIN, ENTRAINMENT_EXP)
    return result if h1_arr.ndim else float(result)


def momentum_equation_rhs(
    theta_m: float, h: float, cf: float, u_e_mps: float, du_e_dx: float
) -> float:
    """dtheta/dx = Cf/2 - (2+H) * (theta/U_e) * dU_e/dx.

    Exposed standalone (not only embedded in the ODE system) so the
    momentum-integral identity itself can be independently tested at
    representative hand-computed states.
    """
    return cf / 2.0 - (2.0 + h) * (theta_m / u_e_mps) * du_e_dx


def entrainment_equation_rhs(
    theta_m: float, h1: float, u_e_mps: float, du_e_dx: float, dtheta_dx: float
) -> float:
    """dH1/dx = [F(H1) - (theta H1/U_e) dU_e/dx - H1 dtheta/dx] / theta,
    the rearranged form of (1/U_e) d(U_e theta H1)/dx = F(H1).
    """
    f_h1 = entrainment_rate(h1)
    return (f_h1 - (theta_m * h1 / u_e_mps) * du_e_dx - h1 * dtheta_dx) / theta_m


def transition_initial_state(
    thwaites_sol: ThwaitesSolution,
    v_inf_mps: float,
    nu_m2s: float,
    x_tr_m: float,
    h_tr: float = H_TR_NOMINAL,
) -> tuple[float, float, float]:
    """Determine (theta_tr_m, H1_tr, x_start_m) for a transition at x_tr_m.

    Raises ValueError if x_tr_m lies beyond the Milestone 1 laminar-
    separation diagnostic (thwaites_sol.x_sep_m), since the M1 laminar
    solution is not considered valid there (see M1 DESIGN.md Section 7).
    """
    if not np.isfinite(x_tr_m) or x_tr_m < 0.0:
        raise ValueError(f"x_tr_m must be finite and non-negative, got {x_tr_m}")
    _validate_positive_scalar("h_tr", h_tr)

    chord_m = float(thwaites_sol.x_m[-1])
    le_offset_m = FLAT_PLATE_LE_OFFSET_FRAC * chord_m

    if x_tr_m <= le_offset_m:
        # Fully turbulent from the leading edge: seed via the derived
        # 1/7-power-law flat-plate theta scaling (see module docstring).
        x_start_m = le_offset_m
        re_x = v_inf_mps * x_start_m / nu_m2s
        theta_tr_m = FLAT_PLATE_THETA_COEFF * x_start_m * re_x**-0.2
    else:
        if thwaites_sol.x_sep_m is not None and x_tr_m > thwaites_sol.x_sep_m + 1.0e-12:
            raise ValueError(
                f"x_tr_m={x_tr_m:.6g} m lies beyond the M1 laminar-separation "
                f"diagnostic (x_sep={thwaites_sol.x_sep_m:.6g} m); the M1 laminar "
                "solution is not valid there and cannot be used to initialize the "
                "turbulent boundary layer."
            )
        x_start_m = x_tr_m
        theta_tr_m = float(np.interp(x_tr_m, thwaites_sol.x_m, thwaites_sol.theta_m))

    h1_tr = h1_of_h(h_tr)
    return theta_tr_m, h1_tr, x_start_m


@dataclass(frozen=True)
class TurbulentBLSolution:
    """Station-by-station Head's-method turbulent boundary-layer solution."""

    s: NDArray[np.float64]              # x/c
    x_m: NDArray[np.float64]            # dimensional streamwise coordinate, m
    u_e_mps: NDArray[np.float64]        # edge velocity, m/s
    du_e_dx: NDArray[np.float64]        # dU_e/dx, 1/s
    theta_m: NDArray[np.float64]        # momentum thickness, m
    delta_star_m: NDArray[np.float64]   # displacement thickness, m (= H * theta)
    h: NDArray[np.float64]              # shape factor
    h1: NDArray[np.float64]             # entrainment shape factor
    cf: NDArray[np.float64]             # local skin-friction coefficient
    valid: NDArray[np.bool_]            # True where the ODE solution is defined (before any event/failure)
    x_tr_m: float                       # transition station actually used, m
    x_start_m: float                    # station where turbulent integration started, m
    h_tr: float                         # initial shape factor used
    status: TurbulentBLStatus
    message: str
    x_turb_sep_m: float | None          # turbulent separation location, m (None unless TURBULENT_SEPARATION)


def solve_turbulent_bl(
    dist: SyntheticVelocityDistribution,
    thwaites_sol: ThwaitesSolution,
    v_inf_mps: float,
    nu_m2s: float,
    x_tr_m: float,
    h_tr: float = H_TR_NOMINAL,
    n_report: int = 501,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
) -> TurbulentBLSolution:
    """Integrate Head's entrainment method from x_tr_m to the trailing edge
    (or until turbulent separation / an invalid state is detected).

    Parameters
    ----------
    dist : SyntheticVelocityDistribution
        The Milestone 1 external velocity distribution (supplies U_e and
        its exact analytic derivative dU_e/dx).
    thwaites_sol : ThwaitesSolution
        The Milestone 1 laminar solution, used only to obtain a continuous
        theta at the transition station (see transition_initial_state).
    v_inf_mps, nu_m2s : float
        Freestream velocity and kinematic viscosity.
    x_tr_m : float
        Assumed transition station, m.
    h_tr : float
        Initial turbulent shape factor at transition (default:
        H_TR_NOMINAL, the ZPG 1/7-power-law value).
    n_report : int
        Number of stations at which to report the solution (dense output,
        does not affect the adaptive step size used internally).
    rtol, atol : float
        solve_ivp tolerances, exposed for convergence testing.
    """
    chord_m = float(thwaites_sol.x_m[-1])
    theta_tr_m, h1_tr, x_start_m = transition_initial_state(thwaites_sol, v_inf_mps, nu_m2s, x_tr_m, h_tr)

    def rhs(x: float, y: NDArray[np.float64]) -> list[float]:
        # The adaptive integrator routinely *probes* states slightly beyond
        # the closure's valid domain while advancing/rejecting trial steps,
        # even though the accepted trajectory itself stays valid until the
        # terminal events below actually fire on a completed step. Guard the
        # derivative evaluation with a tiny clamp (used only internally,
        # here, never fed back into the reported state) so those probes
        # don't raise; the *reported* solution and its validity/separation
        # classification are still determined solely by the unclamped
        # event functions operating on the true accepted trajectory.
        theta = max(y[0], 1.0e-12)
        h1 = max(y[1], H1_SEPARATION_THRESHOLD)
        try:
            h = h_of_h1(h1)
            u_e = dist.u_e(x)
            du_e_dx = dist.du_e_dx(x)
            re_th = theta * u_e / nu_m2s  # rho=1, mu=nu: Re_theta = Ue*theta/nu
            cf = ludwieg_tillmann_cf(h, re_th)
            dtheta_dx = momentum_equation_rhs(theta, h, cf, u_e, du_e_dx)
            dh1_dx = entrainment_equation_rhs(theta, h1, u_e, du_e_dx, dtheta_dx)
        except (ValueError, ZeroDivisionError, OverflowError):
            # A transient out-of-domain probe during adaptive step selection
            # (rejected trial steps are common near a closure boundary).
            # Returning a neutral derivative lets the integrator retry with
            # a smaller step; the terminal events below (evaluated on the
            # true accepted trajectory, unclamped) are the sole authority
            # for actually classifying separation/invalidity.
            return [0.0, 0.0]
        return [dtheta_dx, dh1_dx]

    def event_separation(x: float, y: NDArray[np.float64]) -> float:
        return y[1] - H1_SEPARATION_THRESHOLD

    event_separation.terminal = True
    event_separation.direction = -1.0

    def event_invalid_theta(x: float, y: NDArray[np.float64]) -> float:
        return y[0] - 1.0e-12

    event_invalid_theta.terminal = True
    event_invalid_theta.direction = -1.0

    def event_invalid_h(x: float, y: NDArray[np.float64]) -> float:
        try:
            h = h_of_h1(y[1])
        except ValueError:
            return -1.0
        return H_MAX_VALID - h

    event_invalid_h.terminal = True
    event_invalid_h.direction = -1.0

    t_eval = np.linspace(x_start_m, chord_m, n_report)

    try:
        sol = solve_ivp(
            rhs,
            (x_start_m, chord_m),
            [theta_tr_m, h1_tr],
            method="LSODA",
            t_eval=t_eval,
            events=[event_separation, event_invalid_theta, event_invalid_h],
            rtol=rtol,
            atol=atol,
        )
    except Exception as exc:  # noqa: BLE001 -- genuinely any solver failure should be caught and reported
        n = len(t_eval)
        nan = np.full(n, np.nan)
        return TurbulentBLSolution(
            s=t_eval / chord_m, x_m=t_eval, u_e_mps=nan, du_e_dx=nan, theta_m=nan,
            delta_star_m=nan, h=nan, h1=nan, cf=nan, valid=np.zeros(n, dtype=bool),
            x_tr_m=x_tr_m, x_start_m=x_start_m, h_tr=h_tr,
            status=TurbulentBLStatus.INTEGRATION_FAILURE, message=str(exc), x_turb_sep_m=None,
        )

    if not sol.success and sol.status != 1:  # status==1: a terminal event stopped it, which is fine
        n = len(t_eval)
        nan = np.full(n, np.nan)
        return TurbulentBLSolution(
            s=t_eval / chord_m, x_m=t_eval, u_e_mps=nan, du_e_dx=nan, theta_m=nan,
            delta_star_m=nan, h=nan, h1=nan, cf=nan, valid=np.zeros(n, dtype=bool),
            x_tr_m=x_tr_m, x_start_m=x_start_m, h_tr=h_tr,
            status=TurbulentBLStatus.INTEGRATION_FAILURE, message=sol.message, x_turb_sep_m=None,
        )

    x_end_m = sol.t[-1] if sol.t.size else x_start_m
    n_valid = int(np.sum(t_eval <= x_end_m + 1.0e-9))
    n_total = len(t_eval)

    theta_valid = sol.sol(t_eval[:n_valid])[0] if sol.sol is not None else sol.y[0]
    h1_valid = sol.sol(t_eval[:n_valid])[1] if sol.sol is not None else sol.y[1]
    h_valid = np.array([h_of_h1(v) for v in h1_valid])
    u_e_valid = dist.u_e(t_eval[:n_valid])
    re_theta_valid = reynolds_theta(1.0, u_e_valid, theta_valid, nu_m2s)
    cf_valid = ludwieg_tillmann_cf(h_valid, re_theta_valid)
    delta_star_valid = h_valid * theta_valid

    nan_pad = n_total - n_valid
    theta_full = np.concatenate([theta_valid, np.full(nan_pad, np.nan)])
    h_full = np.concatenate([h_valid, np.full(nan_pad, np.nan)])
    h1_full = np.concatenate([h1_valid, np.full(nan_pad, np.nan)])
    cf_full = np.concatenate([cf_valid, np.full(nan_pad, np.nan)])
    delta_star_full = np.concatenate([delta_star_valid, np.full(nan_pad, np.nan)])
    u_e_full = dist.u_e(t_eval)
    du_e_dx_full = dist.du_e_dx(t_eval)
    valid_mask = np.arange(n_total) < n_valid

    x_turb_sep_m: float | None = None
    if sol.t_events is not None and len(sol.t_events[0]) > 0:
        status = TurbulentBLStatus.TURBULENT_SEPARATION
        x_turb_sep_m = float(sol.t_events[0][0])
        message = f"Turbulent separation (H1 -> {H1_SEPARATION_THRESHOLD}) at x={x_turb_sep_m:.6g} m."
    elif sol.t_events is not None and (len(sol.t_events[1]) > 0 or len(sol.t_events[2]) > 0):
        status = TurbulentBLStatus.INVALID_CLOSURE
        message = "Closure state left its valid domain (theta<=0 or H exceeded H_MAX_VALID)."
    else:
        status = TurbulentBLStatus.COMPLETED_TO_TE
        message = "Integration reached the trailing edge without separation or closure failure."

    return TurbulentBLSolution(
        s=t_eval / chord_m,
        x_m=t_eval,
        u_e_mps=u_e_full,
        du_e_dx=du_e_dx_full,
        theta_m=theta_full,
        delta_star_m=delta_star_full,
        h=h_full,
        h1=h1_full,
        cf=cf_full,
        valid=valid_mask,
        x_tr_m=x_tr_m,
        x_start_m=x_start_m,
        h_tr=h_tr,
        status=status,
        message=message,
        x_turb_sep_m=x_turb_sep_m,
    )


def pressure_gradient_cf_distribution(
    v_inf_mps: float,
    x_m: ArrayLike,
    nu_m2s: float,
    x_tr_m: float,
    tb_sol: TurbulentBLSolution,
) -> NDArray[np.float64]:
    """Piecewise local skin-friction coefficient for the Milestone 4
    pressure-gradient result: laminar (Blasius) for x < x_tr_m, and the
    Head's-method turbulent Cf(x) (interpolated from ``tb_sol``) for
    x >= x_tr_m.

    This is the M4 analogue of skin_friction.transitioned_cf_distribution,
    but using the actual pressure-gradient turbulent integral solution
    downstream of transition instead of a flat-plate correlation.

    If the turbulent solution does not reach the trailing edge (status is
    TURBULENT_SEPARATION or INVALID_CLOSURE), Cf is held constant
    (frozen) at its last valid value for stations beyond that point --
    documented explicitly (see module docstring / DESIGN.md) as a
    bookkeeping convention that does NOT represent a resolved separated-
    flow solution and does not include any associated pressure-drag rise.
    This is exactly the constant-extrapolation behavior of numpy.interp's
    default boundary handling, applied to the valid prefix of the
    solution.
    """
    x_arr = np.asarray(x_m, dtype=np.float64)
    if np.any(~np.isfinite(x_arr)) or np.any(x_arr < 0.0):
        raise ValueError("x_m must be finite and non-negative")

    valid_idx = np.where(tb_sol.valid)[0]
    if valid_idx.size == 0:
        raise ValueError("tb_sol has no valid turbulent stations to interpolate from")
    x_turb_valid = tb_sol.x_m[valid_idx]
    cf_turb_valid = tb_sol.cf[valid_idx]

    laminar_mask = x_arr < x_tr_m
    cf = np.where(
        laminar_mask,
        blasius_cf(v_inf_mps, x_arr, nu_m2s),
        np.interp(x_arr, x_turb_valid, cf_turb_valid),
    )
    return cf if x_arr.ndim else float(cf)
