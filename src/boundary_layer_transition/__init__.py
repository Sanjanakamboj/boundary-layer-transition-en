"""Laminar boundary-layer foundation and reduced-order e^N amplification tracking.

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

Scope boundary (both milestones): no N_crit is selected, no transition
location is predicted, and no turbulent boundary-layer or drag modeling is
performed. A Thwaites-predicted laminar-separation location is a
laminar-boundary-layer diagnostic only, and a modeled instability onset or
accumulated N-factor is an amplification-tracking result only -- neither is
a transition prediction.
"""

from . import external_flow, laminar_bl, operating_point, stability

__all__ = ["external_flow", "laminar_bl", "operating_point", "stability"]

__version__ = "0.2.0"
