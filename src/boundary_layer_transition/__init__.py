"""Laminar boundary-layer foundation, e^N amplification tracking, N_crit
transition logic, pressure-gradient turbulent boundary-layer propagation,
and a laminar-separation-bubble / post-separation drag closure for a
generic sailplane wing section.

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

Milestone 5 addresses the two largest gaps left by Milestone 4: (a) the
baseline case's laminar boundary layer separates before any plausible
N_crit crossing, while Milestone 4 assumed an instantaneous laminar-to-
turbulent restart there -- Milestone 5 instead models the separated-shear-
layer transition / turbulent-reattachment sequence as a transparent,
explicitly labeled parametric sensitivity closure (:mod:`separation_bubble`),
including an explicit open-separation (no-reattachment) outcome; and (b)
when a (re-)restarted turbulent layer separates, Milestone 5 introduces a
distinct, separately labeled separated pressure/form-drag bookkeeping term
alongside the skin-friction contribution over only the modeled attached/
reattached surface regions (:mod:`separated_drag`).

Milestone 6 (final) is a deterministic robustness audit and portfolio
synthesis. It introduces no new physics: it constructs a small, fixed
family of external-flow shape perturbations around the Milestone 1
baseline (:mod:`external_flow_sensitivity`) and re-propagates each one
through the unmodified M1-M5 chain (:mod:`robustness`) to quantify how
much of the baseline's separation/transition/drag story depends on the
specific illustrative pressure distribution chosen in Milestone 1, versus
more general Reynolds-number/boundary-layer behavior. See RESULTS.md,
VERIFICATION.md, and DESIGN.md Section 41 for the final findings.

Scope boundary: N_crit is an external, environment-dependent input, never
selected or recommended by this project (see DESIGN.md for the sourced
sensitivity range used in the study scripts). A Thwaites-predicted
laminar-separation location is a laminar-boundary-layer diagnostic only, a
modeled instability onset or accumulated N-factor is an amplification-
tracking result only, an e^N N_crit crossing, the Milestone 3 separation-
triggered bookkeeping scenario, and the Milestone 5 separation-bubble
closure are three explicitly distinct transition assumptions, a Milestone
5 turbulent (re)separation is distinct from the Milestone 1 laminar
separation, and no turbulent, bubble, drag, or sensitivity result in this
project is, or substitutes for, a CFD/RANS/XFOIL/experimental solution or
validation, a stall angle, a "safe" operating envelope, or a
complete/certified airfoil drag polar. Development stops after Milestone 6.
"""

from . import (
    external_flow,
    external_flow_sensitivity,
    laminar_bl,
    operating_point,
    robustness,
    separated_drag,
    separation_bubble,
    skin_friction,
    stability,
    transition,
    turbulent_bl,
)

__all__ = [
    "external_flow",
    "external_flow_sensitivity",
    "laminar_bl",
    "operating_point",
    "robustness",
    "separated_drag",
    "separation_bubble",
    "skin_friction",
    "stability",
    "transition",
    "turbulent_bl",
]

__version__ = "1.0.0"
