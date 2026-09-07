"""Laminar boundary-layer foundation, e^N amplification tracking, N_crit
transition logic, and pressure-gradient-aware turbulent skin-friction drag
estimation for a generic sailplane wing section.

Milestone 1 established, for a single *generic* (non-manufacturer-matched)
sailplane wing section:

- a representative operating point (:mod:`operating_point`)
- a transparent, illustrative external velocity distribution U_e(x/c)
  (:mod:`external_flow`)
- zero-pressure-gradient (Blasius) and pressure-gradient-aware (Thwaites)
  laminar boundary-layer solutions (:mod:`laminar_bl`)

Milestone 2 extends this with a transparent, source-audited, reduced-order
linear-stability proxy and e^N amplification-factor tracking
(:mod:`stability`), built on the Milestone 1 Thwaites solution.

Milestone 3 extends this further with:

- N_crit transition-crossing logic against the Milestone 2 N-factor
  history, plus an explicitly separate "separation-triggered" bookkeeping
  scenario (:mod:`transition`)
- classical laminar/turbulent flat-plate skin-friction correlations and a
  section skin-friction drag-coefficient bookkeeping model
  (:mod:`skin_friction`)

Milestone 4 replaces Milestone 3's downstream flat-plate turbulent
bookkeeping with an actual pressure-gradient turbulent integral boundary
layer (Head's entrainment method, Ludwieg-Tillmann skin-friction closure)
propagated along the real Milestone 1 U_e(x) from an assumed transition
station to the trailing edge or turbulent separation (:mod:`turbulent_bl`).

Scope boundary: N_crit is an external, environment-dependent input, never
selected or recommended by this project (see DESIGN.md for the sourced
sensitivity range used in the study scripts). A Thwaites-predicted
laminar-separation location is a laminar-boundary-layer diagnostic only, a
modeled instability onset or accumulated N-factor is an amplification-
tracking result only, an e^N N_crit crossing and the separation-triggered
bookkeeping scenario are two explicitly distinct transition assumptions,
and the turbulent skin-friction results (both the Milestone 3 flat-plate
bookkeeping and the Milestone 4 pressure-gradient integral method) are
reduced-order engineering estimates -- none of these are, or substitute
for, CFD or experimentally validated transition or drag predictions.
"""

from . import (
    external_flow,
    laminar_bl,
    operating_point,
    skin_friction,
    stability,
    transition,
    turbulent_bl,
)

__all__ = [
    "external_flow",
    "laminar_bl",
    "operating_point",
    "skin_friction",
    "stability",
    "transition",
    "turbulent_bl",
]

__version__ = "0.4.0"
