"""Synthetic, illustrative external velocity distribution U_e(x/c).

IMPORTANT: The distribution implemented here is a **reduced-order,
illustrative** analytic model of a favorable-then-adverse pressure-gradient
edge velocity, loosely motivated by the qualitative shape of a laminar-flow
airfoil's suction-side velocity distribution (flow accelerates from the
leading edge to a suction peak, then decelerates toward the trailing edge).
It is **not** derived from XFOIL, CFD, wind-tunnel data, or any specific
airfoil geometry, and must not be presented or used as such.

Coordinate convention: ``s = x / c`` is the primary nondimensional chordwise
coordinate, s in [0, 1], with s=0 the leading edge and s=1 the trailing
edge. Dimensional x = s * c is available where needed (e.g. Re_x).

Shape function: a C1-continuous two-segment smoothstep blend between three
control points, (s=0, U/Vinf=1), (s=s_peak, U/Vinf=u_peak), and (s=1,
U/Vinf=u_te). The smoothstep polynomial (Hermite ease, 3t^2 - 2t^3) has zero
slope at t=0 and t=1, so the resulting U_e(s) is continuous and
continuously differentiable everywhere on [0, 1], including at the segment
junction s=s_peak (both one-sided derivatives are zero there).

Derivative: dU_e/dx is provided both as an exact closed-form expression and
(for verification) via a central-difference numerical estimate; the two are
cross-checked in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _smoothstep(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hermite smoothstep, 3t^2 - 2t^3, for t clipped to [0, 1]."""
    tc = np.clip(t, 0.0, 1.0)
    return 3.0 * tc**2 - 2.0 * tc**3


def _smoothstep_deriv(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """d/dt of the smoothstep, evaluated at clipped t (zero outside [0,1])."""
    tc = np.clip(t, 0.0, 1.0)
    return 6.0 * tc - 6.0 * tc**2


@dataclass(frozen=True)
class SyntheticVelocityDistribution:
    """Illustrative favorable-then-adverse-gradient U_e(x/c) model.

    Parameters
    ----------
    chord_m : float
        Chord length, m. Used only to convert between s = x/c and x.
    v_inf_mps : float
        Freestream reference velocity, m/s (U_e(0) = v_inf_mps by
        construction).
    s_peak : float
        Nondimensional chordwise location of peak (suction) velocity,
        in (0, 1).
    u_peak_ratio : float
        Peak U_e / V_inf ratio at s_peak (favorable-gradient region ends
        here). Must be > 1.
    u_te_ratio : float
        U_e / V_inf ratio at the trailing edge, s=1 (adverse-gradient
        region ends here). Must be > 0 and < u_peak_ratio.
    """

    chord_m: float
    v_inf_mps: float
    s_peak: float = 0.35
    u_peak_ratio: float = 1.18
    u_te_ratio: float = 0.55

    def __post_init__(self) -> None:
        if not (0.0 < self.s_peak < 1.0):
            raise ValueError(f"s_peak must be in (0, 1), got {self.s_peak}")
        if not (self.u_peak_ratio > 1.0):
            raise ValueError(f"u_peak_ratio must be > 1, got {self.u_peak_ratio}")
        if not (0.0 < self.u_te_ratio < self.u_peak_ratio):
            raise ValueError(
                "u_te_ratio must satisfy 0 < u_te_ratio < u_peak_ratio, "
                f"got u_te_ratio={self.u_te_ratio}, u_peak_ratio={self.u_peak_ratio}"
            )
        if self.chord_m <= 0.0 or not np.isfinite(self.chord_m):
            raise ValueError(f"chord_m must be finite and positive, got {self.chord_m}")
        if self.v_inf_mps <= 0.0 or not np.isfinite(self.v_inf_mps):
            raise ValueError(f"v_inf_mps must be finite and positive, got {self.v_inf_mps}")

    # -- nondimensional (s = x/c) evaluations -----------------------------

    def velocity_ratio(self, s: ArrayLike) -> NDArray[np.float64]:
        """U_e(s) / V_inf, s = x/c in [0, 1]."""
        s_arr = np.asarray(s, dtype=np.float64)
        if np.any((s_arr < 0.0) | (s_arr > 1.0)):
            raise ValueError("s = x/c must lie in [0, 1]")

        fav = s_arr <= self.s_peak

        t_fav = np.divide(s_arr, self.s_peak, out=np.zeros_like(s_arr), where=self.s_peak > 0)
        ratio_fav = 1.0 + (self.u_peak_ratio - 1.0) * _smoothstep(t_fav)

        span_adv = 1.0 - self.s_peak
        t_adv = np.divide(s_arr - self.s_peak, span_adv, out=np.zeros_like(s_arr), where=span_adv > 0)
        ratio_adv = self.u_peak_ratio - (self.u_peak_ratio - self.u_te_ratio) * _smoothstep(t_adv)

        ratio = np.where(fav, ratio_fav, ratio_adv)
        return ratio if s_arr.ndim else ratio.item()

    def d_velocity_ratio_ds(self, s: ArrayLike) -> NDArray[np.float64]:
        """d(U_e/V_inf)/ds, exact closed form, s = x/c in [0, 1]."""
        s_arr = np.asarray(s, dtype=np.float64)
        if np.any((s_arr < 0.0) | (s_arr > 1.0)):
            raise ValueError("s = x/c must lie in [0, 1]")

        fav = s_arr <= self.s_peak
        t_fav = np.divide(s_arr, self.s_peak, out=np.zeros_like(s_arr), where=self.s_peak > 0)
        dratio_fav = (self.u_peak_ratio - 1.0) * _smoothstep_deriv(t_fav) / self.s_peak

        span_adv = 1.0 - self.s_peak
        t_adv = np.divide(s_arr - self.s_peak, span_adv, out=np.zeros_like(s_arr), where=span_adv > 0)
        dratio_adv = -(self.u_peak_ratio - self.u_te_ratio) * _smoothstep_deriv(t_adv) / span_adv

        dratio = np.where(fav, dratio_fav, dratio_adv)
        return dratio if s_arr.ndim else dratio.item()

    # -- dimensional (x) evaluations ---------------------------------------

    def u_e(self, x_m: ArrayLike) -> NDArray[np.float64]:
        """External velocity U_e(x), m/s, for dimensional x in [0, chord_m]."""
        x_arr = np.asarray(x_m, dtype=np.float64)
        s = x_arr / self.chord_m
        return self.v_inf_mps * self.velocity_ratio(s)

    def du_e_dx(self, x_m: ArrayLike) -> NDArray[np.float64]:
        """Exact closed-form dU_e/dx, (1/s), for dimensional x in [0, chord_m].

        dU_e/dx = (V_inf / c) * d(U_e/V_inf)/ds, by the chain rule s = x/c.
        """
        x_arr = np.asarray(x_m, dtype=np.float64)
        s = x_arr / self.chord_m
        return (self.v_inf_mps / self.chord_m) * self.d_velocity_ratio_ds(s)

    def du_e_dx_numerical(self, x_m: ArrayLike, h_m: float | None = None) -> NDArray[np.float64]:
        """Central-difference estimate of dU_e/dx, for cross-checking the
        closed-form derivative. Uses a one-sided difference within h_m of
        the domain boundaries so the estimate stays inside [0, chord_m].
        """
        x_arr = np.asarray(x_m, dtype=np.float64)
        if h_m is None:
            h_m = 1.0e-6 * self.chord_m

        x_lo = np.clip(x_arr - h_m, 0.0, self.chord_m)
        x_hi = np.clip(x_arr + h_m, 0.0, self.chord_m)
        denom = x_hi - x_lo
        # Guard against a degenerate (zero-width) stencil, which can only
        # occur if h_m is set to (numerically) zero.
        if np.any(denom == 0.0):
            raise ValueError("h_m too small relative to floating-point resolution")
        return (self.u_e(x_hi) - self.u_e(x_lo)) / denom


def default_velocity_distribution(
    chord_m: float, v_inf_mps: float
) -> SyntheticVelocityDistribution:
    """The Milestone 1 baseline synthetic U_e(x/c) distribution."""
    return SyntheticVelocityDistribution(chord_m=chord_m, v_inf_mps=v_inf_mps)
