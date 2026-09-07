"""Milestone 1: laminar boundary-layer foundation and stability inputs.

This package establishes, for a single *generic* (non-manufacturer-matched)
sailplane wing section:

- a representative operating point (:mod:`operating_point`)
- a transparent, illustrative external velocity distribution U_e(x/c)
  (:mod:`external_flow`)
- zero-pressure-gradient (Blasius) and pressure-gradient-aware (Thwaites)
  laminar boundary-layer solutions (:mod:`laminar_bl`)

Scope boundary: this milestone does NOT implement e^N amplification,
transition prediction, N_crit selection, turbulent boundary-layer or drag
modeling. A predicted laminar-separation location is a laminar-boundary-layer
diagnostic only and must not be interpreted as a transition location.
"""

from . import external_flow, laminar_bl, operating_point

__all__ = ["external_flow", "laminar_bl", "operating_point"]

__version__ = "0.1.0"
