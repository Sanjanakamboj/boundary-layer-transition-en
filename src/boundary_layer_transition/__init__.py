"""Laminar boundary-layer foundation, e^N amplification tracking, and
N_crit transition / skin-friction drag bookkeeping for a generic sailplane
wing section.

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

Scope boundary: N_crit is an external, environment-dependent input, never
selected or recommended by this project (see DESIGN.md for the sourced
sensitivity range used in the study scripts). A Thwaites-predicted
laminar-separation location is a laminar-boundary-layer diagnostic only,
a modeled instability onset or accumulated N-factor is an amplification-
tracking result only, and the skin-friction drag model is a reduced-order,
zero-pressure-gradient flat-plate bookkeeping approximation, not a full
turbulent pressure-gradient boundary-layer solution -- none of these are,
or substitute for, an experimentally validated transition or drag
prediction.
"""

from . import external_flow, laminar_bl, operating_point, skin_friction, stability, transition

__all__ = [
    "external_flow",
    "laminar_bl",
    "operating_point",
    "skin_friction",
    "stability",
    "transition",
]

__version__ = "0.3.0"
