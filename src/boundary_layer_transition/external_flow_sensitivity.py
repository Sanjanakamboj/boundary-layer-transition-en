"""Milestone 6: deterministic external-flow (U_e(x/c)) shape sensitivity family.

**Purpose.** Quantify how much of the M1-M5 laminar-separation / e^N /
transition / turbulent-reattachment / drag story is a consequence of the
*specific* illustrative Milestone 1 pressure distribution, versus a more
general feature of the underlying boundary-layer physics at this
Reynolds-number range. This module does **not** replace or alter the
Milestone 1 baseline distribution -- it reuses
:class:`external_flow.SyntheticVelocityDistribution` exactly as-is (the
class already exposes ``s_peak``, ``u_peak_ratio``, ``u_te_ratio`` as
constructor parameters with the Milestone 1 defaults), and simply
constructs a small, deterministic family of *additional* instances around
those defaults. Passing no overrides reproduces the Milestone 1 baseline
bit-for-bit (same class, same defaults, same code path -- verified in
tests/test_external_flow_sensitivity.py).

**Family construction (one-factor-at-a-time, fixed before evaluating any
downstream result -- see DESIGN.md Section 40 for the full rationale):**

- Peak velocity ratio ``u_peak_ratio``: 1.10 (mild acceleration) / 1.18
  (Milestone 1 baseline) / 1.26 (strong acceleration), with ``s_peak`` and
  ``u_te_ratio`` held at their baseline values.
- Peak location ``s_peak``: 0.25 (early) / 0.35 (baseline) / 0.45 (late),
  with ``u_peak_ratio`` and ``u_te_ratio`` held at baseline.
- Trailing-edge velocity ratio ``u_te_ratio`` (adverse-gradient severity):
  0.45 (severe deceleration) / 0.55 (baseline) / 0.70 (mild
  deceleration), with ``s_peak`` and ``u_peak_ratio`` held at baseline.

These seven profiles (one baseline, six variants) are the complete
Milestone 6 sensitivity family (:data:`PROFILES`). They are **not** claimed
to represent real airfoil pressure distributions, wind-tunnel data, or any
CFD/XFOIL result -- they are synthetic conceptual perturbations of the
same illustrative Hermite-blend shape already documented in Milestone 1,
chosen to bracket the baseline by a physically reasonable amount without
producing a degenerate (self-intersecting, non-monotonic-within-each-
region, or otherwise invalid) profile. Values were fixed before any
downstream (M2-M5) result was computed or inspected, and are never
retuned to produce a preferred separation/transition/drag outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .external_flow import SyntheticVelocityDistribution

#: The Milestone 1 baseline parameter values, reproduced here only for
#: documentation/comparison -- SyntheticVelocityDistribution's own
#: defaults are the actual source of truth.
BASELINE_S_PEAK = 0.35
BASELINE_U_PEAK_RATIO = 1.18
BASELINE_U_TE_RATIO = 0.55


@dataclass(frozen=True)
class SensitivityProfile:
    """One named external-flow sensitivity case: which parameter(s) it
    varies relative to the Milestone 1 baseline, and by how much.
    """

    label: str
    s_peak: float = BASELINE_S_PEAK
    u_peak_ratio: float = BASELINE_U_PEAK_RATIO
    u_te_ratio: float = BASELINE_U_TE_RATIO


#: The complete, fixed Milestone 6 sensitivity family (see module
#: docstring). Order matters only for deterministic report/figure
#: ordering; it does not affect any computed result.
PROFILES: list[SensitivityProfile] = [
    SensitivityProfile("baseline"),
    SensitivityProfile("peak_ratio_mild", u_peak_ratio=1.10),
    SensitivityProfile("peak_ratio_strong", u_peak_ratio=1.26),
    SensitivityProfile("peak_location_early", s_peak=0.25),
    SensitivityProfile("peak_location_late", s_peak=0.45),
    SensitivityProfile("te_ratio_severe", u_te_ratio=0.45),
    SensitivityProfile("te_ratio_mild", u_te_ratio=0.70),
]


def build_distribution(
    profile: SensitivityProfile, chord_m: float, v_inf_mps: float
) -> SyntheticVelocityDistribution:
    """Construct the SyntheticVelocityDistribution for one sensitivity
    profile. Uses exactly the Milestone 1 class and its own input
    validation (invalid parameter combinations -- e.g. u_te_ratio >=
    u_peak_ratio -- are rejected there, not re-implemented here).
    """
    return SyntheticVelocityDistribution(
        chord_m=chord_m,
        v_inf_mps=v_inf_mps,
        s_peak=profile.s_peak,
        u_peak_ratio=profile.u_peak_ratio,
        u_te_ratio=profile.u_te_ratio,
    )


def with_override(profile: SensitivityProfile, **kwargs: float) -> SensitivityProfile:
    """Return a copy of ``profile`` with the given field(s) overridden --
    a small convenience for constructing ad hoc profiles (e.g. in tests
    or the cross-model sensitivity ranking) without hand-repeating the
    unchanged fields.
    """
    return replace(profile, **kwargs)
