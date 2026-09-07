"""Generic sailplane wing-section operating point.

This module defines a single, explicitly *illustrative* operating condition
used as the basis for the rest of Milestone 1. Values are chosen to be
physically plausible for a mid-size glass/composite sailplane at a
representative mid-span wing station during cross-country cruise, but they
are **not** matched to any specific manufacturer, airfoil, or flight-test
data set.

Rationale for the chosen values (documented further in DESIGN.md):

- ``chord_m = 0.70 m``: representative mid-span chord for a ~15 m span
  club/standard-class sailplane (root chords are typically 0.8-1.0 m,
  tapering to 0.3-0.4 m at the tip; 0.70 m is a reasonable mid-span value).
- ``v_inf_mps = 25.0 m/s``: representative cross-country cruise speed
  (~90 km/h), well within the normal speed range for 15-18 m class
  sailplanes (typical operating speeds ~90-160 km/h).
- ``rho_kgm3`` / ``mu_pas``: ISA sea-level standard atmosphere values. Not
  altitude-corrected; low-altitude ridge/thermal flying is representative
  enough for an illustrative Milestone 1 baseline. Altitude/atmosphere
  effects are explicitly out of scope for this milestone.
- ``alpha_deg = 4.0 deg``: a representative cruise/thermalling angle of
  attack for a laminar-flow sailplane airfoil. It is used only to motivate
  (qualitatively) the favorable-then-adverse pressure gradient shape of the
  synthetic external velocity distribution in :mod:`external_flow`; the
  velocity distribution itself is not derived from ``alpha_deg`` by any
  aerodynamic solver.

The resulting Mach number is verified to be low enough that an
incompressible boundary-layer treatment is justified (see
:func:`GenericSailplaneOperatingPoint.is_incompressible`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Ratio of specific heats for air.
GAMMA_AIR = 1.4

#: Specific gas constant for dry air, J/(kg*K).
R_AIR = 287.05

#: ISA sea-level standard temperature, K (15 degC).
T_ISA_SEA_LEVEL_K = 288.15

#: ISA sea-level standard density, kg/m^3.
RHO_ISA_SEA_LEVEL = 1.225

#: ISA sea-level standard dynamic viscosity, Pa*s (Sutherland's law
#: evaluated at T_ISA_SEA_LEVEL_K; matches the standard tabulated value,
#: e.g. NASA/ICAO Standard Atmosphere tables).
MU_ISA_SEA_LEVEL = 1.7894e-5

#: Mach number below which an incompressible boundary-layer treatment is
#: conventionally considered adequate (project choice; see DESIGN.md).
INCOMPRESSIBLE_MACH_LIMIT = 0.3


def speed_of_sound(temperature_k: float, gamma: float = GAMMA_AIR, r_specific: float = R_AIR) -> float:
    """Speed of sound for an ideal gas, a = sqrt(gamma * R * T).

    Parameters
    ----------
    temperature_k : float
        Static temperature, K. Must be positive.
    """
    if temperature_k <= 0.0:
        raise ValueError(f"temperature_k must be positive, got {temperature_k}")
    return math.sqrt(gamma * r_specific * temperature_k)


@dataclass(frozen=True)
class GenericSailplaneOperatingPoint:
    """A single, generic (non-manufacturer-matched) wing-section operating point.

    All quantities are SI unless noted otherwise. The instance is immutable
    and validated at construction time (see :meth:`__post_init__`).
    """

    chord_m: float = 0.70
    v_inf_mps: float = 25.0
    rho_kgm3: float = RHO_ISA_SEA_LEVEL
    mu_pas: float = MU_ISA_SEA_LEVEL
    temperature_k: float = T_ISA_SEA_LEVEL_K
    alpha_deg: float = 4.0

    def __post_init__(self) -> None:
        for name in ("chord_m", "v_inf_mps", "rho_kgm3", "mu_pas", "temperature_k"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number, got {value}")
        if not math.isfinite(self.alpha_deg):
            raise ValueError(f"alpha_deg must be finite, got {self.alpha_deg}")

    @property
    def nu_m2s(self) -> float:
        """Kinematic viscosity, nu = mu / rho, m^2/s."""
        return self.mu_pas / self.rho_kgm3

    @property
    def speed_of_sound_mps(self) -> float:
        return speed_of_sound(self.temperature_k)

    @property
    def mach(self) -> float:
        """Freestream Mach number, M = V_inf / a."""
        return self.v_inf_mps / self.speed_of_sound_mps

    @property
    def re_c(self) -> float:
        """Chord Reynolds number, Re_c = rho * V_inf * c / mu."""
        return self.rho_kgm3 * self.v_inf_mps * self.chord_m / self.mu_pas

    def is_incompressible(self, mach_limit: float = INCOMPRESSIBLE_MACH_LIMIT) -> bool:
        """Whether the Mach number justifies an incompressible BL treatment."""
        return self.mach < mach_limit

    def summary(self) -> str:
        return (
            "Generic sailplane wing-section operating point (illustrative, not "
            "manufacturer-matched)\n"
            f"  chord c            = {self.chord_m:.4f} m\n"
            f"  V_inf              = {self.v_inf_mps:.4f} m/s\n"
            f"  rho                = {self.rho_kgm3:.4f} kg/m^3\n"
            f"  mu                 = {self.mu_pas:.6e} Pa.s\n"
            f"  nu                 = {self.nu_m2s:.6e} m^2/s\n"
            f"  T                  = {self.temperature_k:.2f} K\n"
            f"  a (speed of sound) = {self.speed_of_sound_mps:.3f} m/s\n"
            f"  Mach               = {self.mach:.5f}\n"
            f"  Re_c               = {self.re_c:.4e}\n"
            f"  alpha (descriptive)= {self.alpha_deg:.2f} deg\n"
            f"  incompressible?    = {self.is_incompressible()}"
        )


def default_operating_point() -> GenericSailplaneOperatingPoint:
    """The Milestone 1 baseline generic sailplane operating point."""
    return GenericSailplaneOperatingPoint()
