# DESIGN.md -- Project 10 (Boundary-Layer Transition, e^N)

This document records the coordinate/sign conventions, source audits,
derivations, numerical methods, leading-edge treatment, verification
strategy, and validity limitations for the Project 10 sailplane wing study.
Milestone 1 (laminar boundary-layer foundation) is documented in Sections
1-7 below; Milestone 2 (reduced-order stability/e^N amplification tracking)
is documented in Section 8 onward. Milestone 1 content is unchanged from
its original commit.

---

# MILESTONE 1: Laminar Boundary-Layer Foundation and Stability Inputs

**Scope reminder:** Milestone 1 implements only an operating point, an
external velocity distribution, and laminar boundary-layer solutions
(Blasius + Thwaites). It does **not** implement e^N amplification,
N_crit selection, transition-location prediction, turbulent boundary-layer
modeling, or drag estimation. A predicted laminar-separation location is a
laminar-boundary-layer diagnostic only, not a transition prediction.

---

## 1. Coordinate and sign conventions

- `x` is the dimensional streamwise (chordwise) coordinate, m, measured
  from the modeled leading edge (`x = 0`) to the modeled trailing edge
  (`x = c`).
- `s = x / c` is the nondimensional chordwise coordinate, the primary
  independent variable used throughout `external_flow.py` and reported in
  all tables/figures, `s in [0, 1]`.
- `U_e(x)` (or `U_e(s)`) is the external (boundary-layer-edge) velocity,
  m/s, always positive over the modeled domain (no reversed flow is
  represented -- this milestone does not model separated-flow reverse
  velocities).
- `y` (implicit, not gridded in this milestone) is the wall-normal
  coordinate; boundary-layer thickness quantities (`delta_99`, `delta*`,
  `theta`) are integral properties over `y` at a fixed `x`, consistent with
  standard boundary-layer theory.
- All boundary-layer thickness quantities are non-negative by definition.
  `H = delta*/theta >= 1` (with `H` significantly above 1 as separation is
  approached).
- Positive `dU_e/dx` = favorable (accelerating) pressure gradient;
  negative `dU_e/dx` = adverse (decelerating) pressure gradient. This
  follows directly from the inviscid Bernoulli relation `dp/dx = -rho U_e
  dU_e/dx` at the boundary-layer edge (not implemented explicitly here,
  only its qualitative sign consequence is used).
- Thwaites' pressure-gradient parameter is defined as
  `lambda = (theta^2 / nu) * dU_e/dx`, i.e. `lambda > 0` favorable,
  `lambda < 0` adverse -- consistent with the sign convention above.

---

## 2. Source audit

Sources actually consulted (from training-time technical knowledge of
these standard, widely reproduced classical references; equations were
cross-checked internally for self-consistency, e.g. `BLASIUS_H` is
computed as the ratio of the two Blasius coefficients used rather than
taken as an independently rounded constant):

1. **H. Schlichting & K. Gersten, *Boundary-Layer Theory*, 9th ed.**
   Used for: the qualitative structure of the Blasius flat-plate solution
   (Chapter 7) and the general form of the laminar momentum-integral
   equation. Valid range: incompressible, steady, 2D laminar boundary
   layer.

2. **F. M. White, *Viscous Fluid Flow*, 3rd ed.**
   Used for: the specific numerical coefficients implemented here --
   Blasius `delta_99 = 5.0 x/sqrt(Re_x)`, `delta* = 1.7208 x/sqrt(Re_x)`,
   `theta = 0.664 x/sqrt(Re_x)`, `Cf,x = 0.664/sqrt(Re_x)` (Table 4.1-type
   summary of the Blasius solution); and the two-branch polynomial
   correlation for Thwaites' method, `l(lambda)` and `H(lambda)`, attributed
   in White to Cebeci & Bradshaw (Table 4.1 / Eqs. 4.75-4.76 region).
   These are the exact constants implemented in `laminar_bl.py`.

3. **B. Thwaites, "Approximate calculation of the laminar boundary layer,"
   *Aeronautical Quarterly*, 1(3), 1949, pp. 245-280.**
   Original source of the momentum-integral method and its correlation
   concept (`l(lambda)`, `H(lambda)` as universal functions of a single
   pressure-gradient parameter `lambda`). Thwaites' own tabulated data is
   *not* reproduced digit-for-digit here; the polynomial curve fit to that
   data (per White/Cebeci-Bradshaw, item 2) is used instead. This
   substitution is a **documented, deliberate project choice** (see
   "Ambiguity" below), not a claim that Thwaites' original table was
   consulted directly.

4. **T. Cebeci & P. Bradshaw, *Momentum Transfer in Boundary Layers*,
   Hemisphere/McGraw-Hill, 1977.**
   Attributed source (via White) of the specific two-branch polynomial fit
   used for `l(lambda)` and `H(lambda)`.

**Ambiguity documented:** Several textbooks quote slightly different
polynomial fits to Thwaites' original tabulated `l(lambda)`/`H(lambda)`
data (Thwaites' own approximate formulas, White's fit, Cebeci-Bradshaw's
fit, and Curle's fit all differ at the few-percent level, particularly for
`lambda < 0`). This project uses the specific two-branch fit reproduced in
White's *Viscous Fluid Flow* (attributed to Cebeci-Bradshaw), consistently,
everywhere in the code. The separation criterion is **not** hard-coded from
a secondary citation; instead `thwaites_separation_lambda()` numerically
solves `l(lambda) = 0` for the *actual implemented* correlation and returns
that root (`lambda_sep ~= -0.0898` for this implementation), which is then
compared to the commonly cited literature value of `lambda_sep ~= -0.09`
as an independent consistency check (see `test_separation_predicted_and_
lambda_root_matches_literature`).

**What is a project choice vs. sourced physics:**
- Sourced physics: the Blasius coefficients; the Thwaites momentum-integral
  formulation `theta^2(x) = 0.45 nu/U_e^6 integral_0^x U_e^5 dx'`; the
  `l(lambda)`, `H(lambda)` polynomial correlations; the ZPG limit
  `lambda -> 0`.
- Project choices: the specific generic operating point (chord, speed,
  altitude); the synthetic `U_e(x/c)` shape (smoothstep blend) and its
  numeric parameters (`s_peak=0.35`, `u_peak_ratio=1.18`, `u_te_ratio=0.55`);
  the flat-plate-like leading-edge boundary condition `theta(0)=0` with
  `U_e(0)` finite and nonzero (rather than a true stagnation-point airfoil
  leading edge); the correlation validity domain `lambda in [-0.10, 0.10]`;
  the decision to report `H`/`delta*`/`Cf` as `NaN` (not extrapolated) once
  `lambda` exits that domain or once separation has been predicted.

---

## 3. Derivations / equations implemented

### 3.1 Operating point (`operating_point.py`)

- `nu = mu / rho`
- `a = sqrt(gamma R T)` (ideal-gas speed of sound), `gamma=1.4`,
  `R=287.05 J/(kg K)`
- `M = V_inf / a`
- `Re_c = rho V_inf c / mu`
- Incompressibility check: `M < 0.3` (a conventional, project-chosen
  threshold, not a hard physical boundary) -- verified true for the
  Milestone 1 baseline (`M ~= 0.073`).

### 3.2 External velocity distribution (`external_flow.py`)

A C1-continuous, two-segment Hermite smoothstep blend between
`(s=0, U/V_inf=1)`, `(s=s_peak, U/V_inf=u_peak_ratio)`, and
`(s=1, U/V_inf=u_te_ratio)`. Smoothstep: `f(t) = 3t^2 - 2t^3`, `t in [0,1]`,
with `f'(0)=f'(1)=0`, guaranteeing the assembled `U_e(s)` is continuous and
continuously differentiable (C1) everywhere on `[0,1]`, including at the
segment boundary `s=s_peak` (both one-sided derivatives vanish there).

`dU_e/dx` is provided both in closed form (exact chain-rule derivative of
the smoothstep blend) and via a central-difference numerical estimate for
verification; the two agree to the stencil's expected `O(h^2)` truncation
error (tested to `abs=1e-3` at `h = 1e-6 c`).

**This distribution is explicitly synthetic/illustrative** -- it is not
derived from XFOIL, CFD, wind-tunnel data, or a specific airfoil geometry.
It is only loosely, qualitatively motivated by the shape of a laminar-flow
airfoil's suction-side velocity distribution (accelerate to a suction peak,
then decelerate toward the trailing edge).

### 3.3 Blasius ZPG reference (`laminar_bl.py`, part A)

```
Re_x     = U_inf x / nu
delta_99 = 5.0    x / sqrt(Re_x)
delta*   = 1.7208 x / sqrt(Re_x)
theta    = 0.664  x / sqrt(Re_x)
H        = delta*/theta  (computed self-consistently = 1.7208/0.664 ~= 2.5916;
                           commonly rounded to 2.59 in textbooks)
Cf,x     = 0.664 / sqrt(Re_x)
```

### 3.4 Thwaites pressure-gradient-aware method (`laminar_bl.py`, part B)

```
theta(x)^2 = 0.45 nu / U_e(x)^6 * integral_0^x U_e(x')^5 dx'      [theta(0)=0]
lambda(x)  = theta(x)^2 / nu * dU_e/dx
l(lambda)  = 0.22 + 1.57 lambda - 1.8 lambda^2            (0 <= lambda <= 0.1)
           = 0.22 + 1.402 lambda + 0.018 lambda/(lambda+0.107)  (-0.1 <= lambda < 0)
H(lambda)  = 2.61 - 3.75 lambda + 5.24 lambda^2            (0 <= lambda <= 0.1)
           = 2.088 + 0.0731/(lambda+0.14)                  (-0.1 <= lambda < 0)
delta*(x)  = H(lambda) theta(x)
Cf,x(x)    = 2 nu l(lambda) / (U_e(x) theta(x))
```

Separation is predicted at the first `x` (interpolated) where `l(lambda)`
crosses from positive to non-positive.

---

## 4. Numerical method

- The Thwaites integral `integral_0^x U_e^5 dx'` is evaluated by
  cumulative trapezoidal quadrature (`scipy.integrate.cumulative_trapezoid`)
  on a uniform grid in `x` (equivalently `s`), default `n_points=2001`
  (`4001` used in the scripts for tighter convergence).
- Convergence is demonstrated by refining `n_points` (251 -> 4001) and
  checking that `theta` at a fixed station and the predicted separation
  location `s_sep` both converge monotonically with shrinking
  station-to-station differences (see
  `test_grid_convergence_theta_and_separation_location`).
- `H`, `delta*`, and `Cf,x` are reported only where `lambda` is within the
  correlation's documented validity domain `[-0.10, 0.10]` **and** the
  station has not yet passed the predicted separation location. Past
  separation, `lambda` can numerically drift back into the nominal range
  as `dU_e/dx` flattens toward the trailing edge, even though the
  attached-flow correlation is no longer physically meaningful there; this
  region is explicitly masked to `NaN` rather than silently reporting a
  plausible-looking but invalid number.

---

## 5. Leading-edge treatment (x = 0)

Two distinct situations are treated differently and explicitly, per the
Milestone 1 numerical-robustness requirement:

1. **Removable singularity (delta_99, delta*, theta, Blasius and
   Thwaites):** the raw formula `x / sqrt(Re_x)` is a `0/0` indeterminate
   form at `x=0` in floating point (`Re_x=0` there), even though its
   analytic limit is exactly zero (`x/sqrt(Re_x) = sqrt(x nu/V) -> 0` as
   `x -> 0`). This module evaluates the algebraically equivalent,
   non-singular closed form `sqrt(x nu / V)` directly, so these quantities
   return exactly `0.0` at `x=0` -- the true limit, not a fabricated
   placeholder.

2. **True (non-removable) singularity (Blasius Cf,x):** `Cf,x = 0.664 /
   sqrt(Re_x)` genuinely diverges to `+infinity` as `x -> 0` (infinite wall
   shear at the idealized flat-plate leading edge). This module returns
   `inf` at `x=0`, explicitly, and this is verified by a dedicated test
   (`test_leading_edge_cf_is_explicitly_singular_not_fabricated`). It is
   never masked or replaced with a finite number.

3. **Thwaites theta(0)=0 boundary condition:** the momentum-integral
   equation is solved starting from `theta(0)=0` with `U_e(0)` finite and
   nonzero. This is standard practice for a flat-plate-like leading edge
   with a prescribed nonzero edge velocity there (as in this synthetic
   distribution), and is distinct from a true stagnation-point airfoil
   leading edge, where `U_e(0)=0` and a different (regular, non-singular
   in a different sense) starting procedure is required. Modeling a true
   stagnation-point leading edge is out of scope for Milestone 1; see
   Section 7.

4. **Thwaites Cf,x at x=0:** `theta(0)=0` makes `Cf,x = 2 nu l(lambda)/
   (U_e theta)` a `.../0` form at the leading edge. This is reported as
   `NaN` (formally undefined, `0/0` in the physical limit since `l(0)` is
   also effectively 0 there), not a fabricated finite value.

---

## 6. Verification strategy

Independent verification (not merely re-calling the production function) is
implemented for every item required by the milestone:

- `Re_c`, `Re_x`: hand-computed via two independent arithmetic paths
  (`rho V c / mu` vs. `V c / nu` with `nu` computed separately).
- Blasius `delta_99`, `delta*`, `theta`, `Cf,x`: hand-computed with
  `math.sqrt`/plain arithmetic against the closed-form coefficients,
  independent of the vectorized numpy implementation.
- `H`: verified as the exact ratio `delta*/theta` and compared to the
  commonly cited literature value 2.59 (within ~0.06%).
- External-flow endpoints/reference stations: exact values at `s=0`,
  `s=s_peak`, `s=1` checked against the distribution's defining control
  points.
- `dU_e/dx`: closed-form vs. central-difference numerical derivative,
  cross-checked to `O(h^2)` tolerance; plus hand-derived zero-derivative
  points at the leading edge and at the peak (both are stationary points of
  the smoothstep construction, verifiable by inspection of the formula).
- Dimensional/nondimensional consistency: `U_e(x) == V_inf * ratio(x/c)`
  and `dU_e/dx == (V_inf/c) * d(ratio)/ds` checked at 51 stations.
- Thwaites integral evaluation: (a) a hand-rolled `numpy.trapezoid`
  quadrature on an independently sampled grid, compared against the
  production solver's `scipy.integrate.cumulative_trapezoid`-based result
  at an interior station; (b) a from-scratch manual cumulative
  trapezoidal-sum reimplementation compared against `scipy`'s function on
  the same grid.
- Zero-pressure-gradient recovery: `l(0)=0.22`, `H(0)=2.61` checked exactly
  against the correlation's own definition; the *implied* ZPG skin-friction
  coefficient derived by hand from `l(0)` and the Blasius `theta` scaling
  (`0.6627`) is checked against the Blasius `Cf` coefficient (`0.664`, within
  ~0.2%); `H(0)` is checked against `BLASIUS_H` (within ~1%). At the
  near-leading-edge station of the actual synthetic distribution (weak but
  nonzero `lambda`), the study script reports the resulting theta deviation
  from the Blasius reference (~1%), documented as expected given the small
  residual pressure gradient there.
- Scalar/vector consistency: every public function is checked with a scalar
  argument against the corresponding element of a vectorized call, across
  `operating_point`, `external_flow`, and `laminar_bl`.
- Invalid-input rejection: nonpositive/nonfinite `chord`, `V_inf`, `rho`,
  `mu`, `nu`, `T`; negative `x`; out-of-range `s`; invalid distribution
  shape parameters (`s_peak` outside `(0,1)`, `u_peak_ratio<=1`,
  `u_te_ratio` outside `(0, u_peak_ratio)`); `n_points < 3`.
- Grid convergence: `theta` at a fixed station and the predicted separation
  location both converge (differences shrink monotonically, and the
  finest-grid-pair relative change is `< 1e-4` for `theta` and
  `< 1e-3` chord-fraction for `s_sep`) as `n_points` is refined from 251 to
  4001.

---

## 7. Validity limitations

- Incompressible flow only; valid because `M ~= 0.073 << 0.3` for the
  chosen operating point. Not re-derived or re-checked for other operating
  points a user might substitute.
- 2D (no sweep, no spanwise flow) laminar boundary layer only.
- No surface roughness, free-stream turbulence intensity, or leading-edge
  contamination effects.
- The external velocity distribution is synthetic/illustrative, not tied to
  any real airfoil's actual pressure distribution; quantitative results
  (e.g. the specific separation location `x/c ~= 0.43`) are illustrative
  of the *method*, not a prediction for any real wing.
- The leading edge is modeled as flat-plate-like (`U_e(0)` finite and
  nonzero, `theta(0)=0`), not as a true stagnation point. A real airfoil
  leading edge requires a stagnation-point starting solution for Thwaites'
  method (often handled via a small-`x` series expansion); this is
  explicitly not implemented in Milestone 1.
- Thwaites' method is itself an approximate, single-parameter correlation
  method; it is known from the literature to carry a few-percent-level
  systematic error (visible here in the small ZPG-limit mismatches
  documented above) and to become unreliable near and beyond its
  documented `lambda` validity domain `[-0.10, 0.10]` -- which is why this
  implementation refuses to report `H`/`delta*`/`Cf,x` outside that domain
  or beyond the predicted separation station, rather than extrapolating.
- A Thwaites-predicted laminar separation is **not** a transition
  prediction. Real laminar boundary layers commonly transition upstream of
  where they would otherwise laminar-separate, and separated laminar shear
  layers can reattach turbulently; neither phenomenon is modeled here.
  e^N amplification-factor tracking against a chosen `N_crit` (Milestone 2+)
  is required before any transition-location claim can be made.

---

# MILESTONE 2: Reduced-Order Linear-Stability Proxy and e^N Amplification Tracking

**Scope reminder:** Milestone 2 tracks amplification only. It does **not**
select an `N_crit`, does **not** predict a transition location, and does
**not** compute turbulent skin friction or drag. A "modeled instability
onset" is not a transition point; a laminar-separation diagnostic (from
Milestone 1) is not a transition point either.

## 8. Source audit (Milestone 2)

Sources actually checked for this milestone (web search was used in this
session to look for the specific Drela/Giles and Gleyzes-Cousteix-Bonnet
envelope-amplification polynomial coefficients used inside XFOIL; the
searches confirmed the *existence*, general *form*, and *attribution* of
that method, but did **not** return the exact numerical polynomial
coefficients from a primary source that could be verified with confidence.
Per the explicit project instruction not to cite a source not actually
checked, those exact coefficients are **not** used anywhere in this
module):

1. **e^N formalism itself** (Smith & Gamberoni 1956; van Ingen 1956, as
   reviewed in van Ingen, "The e^N Method for Transition Prediction:
   Historical Review of Work at TU Delft," AIAA Paper 2008-3830). Confirmed
   via web search: van Ingen's original semi-empirical method dates to his
   1956 Delft report (VTH-74), and the AIAA 2008-3830 historical review
   covers the method's development at TU Delft, including boundary layers
   with pressure gradient and separation. Used for: the definition
   `N = ln(A/A0) = integral(-alpha_i) dx` and the general concept of
   accumulating amplification from an instability-onset station.

2. **Classical Blasius (ZPG) Tollmien-Schlichting neutral-stability point**
   (Schlichting & Gersten, *Boundary-Layer Theory*, 9th ed., stability
   chapters; White, *Viscous Fluid Flow*, 3rd ed., Ch. 5). This is a
   long-standing, widely reproduced textbook result: the ZPG Blasius
   boundary layer is linearly unstable to Tollmien-Schlichting waves only
   above a critical Reynolds number, commonly quoted as `Re_x,crit ~ 9e4`
   (`Re_theta,crit ~ 200`). Used as the **one sourced numerical anchor**
   for the onset criterion in this module (`RE_THETA_CRIT_ZPG = 200`).
   Different sources/eigenvalue solutions report this critical Reynolds
   number with several-percent spread depending on exact disturbance
   definition; it is used here as an order-of-magnitude-correct anchor,
   not a razor-precise digit.

3. **Drela & Giles envelope method / XFOIL** (Drela, M. and Giles, M. B.,
   "Viscous-Inviscid Analysis of Transonic and Low Reynolds Number
   Airfoils," *AIAA Journal*, 25(10), 1987; Drela, M., "XFOIL: An Analysis
   and Design System for Low Reynolds Number Airfoils," 1989; underlying
   envelope method attributed to Gleyzes, C., Cousteix, J., and Bonnet,
   J.L., "Calculation Method of Leading Edge Separation Bubbles," 1983).
   Web search confirmed this is the correct attribution chain and general
   description (an envelope of Falkner-Skan-family amplification curves,
   parameterized by `H` and `Re_theta`, curve-fit for a critical Reynolds
   number `Re_theta,0(H)` and an envelope growth rate `dn/dReθ(H)`). The
   search did **not** return the specific published polynomial/tanh-based
   coefficients (e.g. the exact `log10(Re_theta,0)` correlation and the
   `dn/dReθ` correlation) from a source this session could directly read
   and verify digit-for-digit. **Those exact coefficients are therefore
   NOT reproduced in this module.** This module instead implements its own
   much simpler, explicitly labeled proxy (Section 9), which borrows only
   the qualitative structure of the Drela/Gleyzes approach (a critical
   `Re_theta(H)` plus an `H`-dependent growth-rate law) without claiming
   quantitative fidelity to it.

4. **Michel's criterion and the Granville criterion** (classical empirical
   transition-onset/-location correlations). Web search confirmed these
   exist and are widely used and cited but, again, did not return their
   exact published numerical coefficients from a primary source this
   session could verify directly. Neither is used quantitatively in this
   module; they are mentioned here only as alternative reduced-order
   approaches this project consciously did not adopt (see "why not
   Michel/Granville" below).

**Why a project-defined proxy instead of Michel/Granville/Drela-Gleyzes
verbatim:** the assignment explicitly permits (and, given the verification
constraints above, requires) implementing "a clearly labeled engineering
proxy rather than inventing a supposedly exact e^N equation" when the
literature's specific closed-form coefficients cannot be directly verified
in-session. This module follows that path: it borrows the correct
*qualitative* physics (an H-dependent neutral curve anchored at the sourced
ZPG point; growth rate increasing with both `Re_theta` excess above that
curve and with `H`, consistent with the well-documented destabilizing
effect of adverse pressure gradient / higher shape factor) while being
explicit that the specific functional forms and constants below are
project choices, not digitized correlations.

## 9. e^N derivation and sign convention (Milestone 2)

**Sourced formalism (A):**

```
N = ln(A / A0)
N(x) = integral_{x0}^{x} (-alpha_i(x')) dx'      [x0 = instability-onset station]
```

For a disturbance behaving spatially as `A(x) ~ exp(-integral alpha_i dx)`,
unstable amplification corresponds to `alpha_i < 0`, i.e.
`dN/dx = -alpha_i > 0`.

**This module does not solve for `alpha_i`.** It works directly with a
modeled `dN/ds` (`s = x/c`), as explicitly permitted by the assignment for
a reduced-order proxy. This is stated plainly in `stability.py`'s module
docstring and is not disguised as an eigenvalue solution.

**Reduced-order proxy actually implemented (B):**

```
Re_theta,crit(H) = Re_theta,crit,ZPG * exp(-b * (H - H_blasius))     [project proxy shape,
                                                                        anchored at sourced ZPG point]
dN/ds             = k * (H - 1) * max(0, Re_theta - Re_theta,crit(H)) / Re_theta,crit,ZPG
N(s)              = integral_0^s (dN/ds) ds'   [restricted to the M1-valid, attached-laminar domain]
```

with `Re_theta,crit,ZPG = 200` (sourced, Section 8 item 2), `H_blasius`
the self-consistent Blasius `H` from `laminar_bl.py` (`~2.5916`), and
`b = 1.0`, `k = 10.0` project (unsourced) calibration constants (Section
10).

**Project assumptions/calibration constants (C):**

- `b = 1.0` (onset H-sensitivity): chosen so the resulting `Re_theta,crit(H)`
  curve produces a modeled onset station comfortably between the leading
  edge and the M1 laminar-separation station for the Milestone 1 baseline
  operating point, and so that `Re_theta,crit` decreases monotonically and
  smoothly as `H` rises through the adverse-gradient region. Not fit to any
  published neutral-curve data.
- `k = 10.0` (amplification-rate coefficient): chosen only so that the
  Milestone 1 baseline case's accumulated `N` reaches an order-1-to-few
  value (`N_max ~= 3.84`) by the M1 separation station -- legible on a
  linear plot, and comfortably below commonly used `N_crit` values (~7-11)
  reported in the e^N literature for various disturbance environments, so
  that the honest result ("separation is reached with only modest N
  accumulated") is visible rather than obscured. This constant is **not**
  fit to any measured or computed amplification-rate data, and no attempt
  was made to force it toward, or away from, any particular N_crit,
  because no N_crit is selected in this milestone.

**Quantities that would require a true Orr-Sommerfeld/PSE solver or a
measured disturbance environment (D):**

- The actual complex eigenvalue `alpha(x; omega, beta)` for each disturbance
  frequency and spanwise wavenumber, computed from the true local
  mean-velocity profile `U(y)` (not just its integral parameters `H`,
  `Re_theta`).
- A true `H`-and-`Re_theta`-parameterized neutral curve, computed (not
  proxy-modeled) from Falkner-Skan or actual similarity/non-similarity
  profile stability analysis.
- The envelope construction over all disturbance frequencies (the
  "most-amplified-frequency" envelope actually used by e^N codes in
  practice), rather than a single scalar growth-rate proxy.
- Any dependence on the actual disturbance/receptivity environment
  (free-stream turbulence intensity and spectrum, surface roughness,
  acoustic forcing) -- e^N in practice calibrates `N_crit` empirically
  against this environment; this module makes no attempt to model
  receptivity at all.

## 10. Instability-onset criterion (Milestone 2)

The onset criterion is exposed directly via `stability.critical_re_theta(H)`
and `stability.is_unstable(Re_theta, H)`. A station is "modeled unstable"
where the *local* `Re_theta(x)` (from the M1 Thwaites solution) first
reaches or exceeds the *local* `Re_theta,crit(H(x))`. This is a "frozen"
(quasi-parallel, locally evaluated) neutral-curve-crossing criterion --
consistent in spirit with how e^N onset is located in practice (compare the
local state against a neutral curve), but built on the project proxy of
Section 9, not a true computed neutral curve.

Favorable-gradient treatment: where `H < H_blasius` (favorable gradient,
fuller profile), `Re_theta,crit(H) > 200` -- the proxy requires a higher
local `Re_theta` before declaring instability, consistent with the
well-documented stabilizing effect of favorable pressure gradient.

Adverse-gradient treatment: where `H > H_blasius` (adverse gradient, less
full / more inflectional profile), `Re_theta,crit(H) < 200` -- the proxy
destabilizes (requires a lower `Re_theta`), consistent with the
well-documented destabilizing effect of adverse pressure gradient /
elevated shape factor.

Near-separation treatment: as `H` rises sharply approaching the M1
laminar-separation station, `Re_theta,crit(H)` continues to fall in this
proxy (no floor is imposed) -- physically, real adverse-gradient/near-
separation profiles are known to be strongly destabilized, so this
qualitative direction is correct, but the module makes no claim that the
*magnitude* of this destabilization is quantitatively accurate that close
to separation (where the underlying attached-laminar Thwaites solution
itself is also becoming less reliable; see Milestone 1 DESIGN.md Section
7).

Baseline result: modeled instability onset at `x/c ~= 0.0885`, well before
the M1 laminar-separation diagnostic at `x/c ~= 0.4314`, giving an
available amplification length of `Delta(x/c) ~= 0.343`. This is reported
strictly as the **modeled instability onset**, never as a transition point,
critical transition location, or laminar-run endpoint.

## 11. Amplification-rate proxy and integration algorithm

`dN/ds` (Section 9, formula B) is, by construction:

- Exactly zero for `Re_theta < Re_theta,crit(H)` (stable stations
  contribute nothing).
- Continuous through the onset crossing: `max(0, ...)` clips smoothly to
  zero as `Re_theta -> Re_theta,crit(H)^-`, so there is no artificial jump
  in the integrand at onset (verified in
  `test_local_amplification_rate_continuous_through_onset`).
- Never negative (verified in
  `test_local_amplification_rate_nonnegative_and_finite`).

`N(s)` is computed by `integrate_n_factor()`, a **pure, aerodynamics-
independent** cumulative-trapezoidal-quadrature utility
(`scipy.integrate.cumulative_trapezoid`), tested directly against two
analytic control cases (Section 12) before ever being combined with the
aerodynamic proxy. Because `dN/ds` is already zero (not merely small) at
every pre-onset station, integrating over the *entire* M1-valid domain from
`s=0` automatically reproduces `N=0` up to the onset station -- no special-
cased "start integrating at onset" logic is required, which is what avoids
grid-dependent jumps at the onset crossing (see Section 13, grid
convergence).

## 12. Separation handling (Milestone 2)

The valid domain for all Milestone 2 outputs is *exactly* the Milestone 1
"reportable" domain (`~isnan(thwaites_sol.h)`), i.e. the pre-separation,
Thwaites-lambda-in-range domain already established in Milestone 1.
Outside that domain:

- `Re_theta`, `Re_theta,crit`, `dN/ds`, and `N` are all set to `NaN`,
  never fabricated or extrapolated (verified in
  `test_no_amplification_accumulated_after_m1_separation`).
- No amplification is accumulated beyond the M1 laminar-separation
  diagnostic station, by construction (the cumulative integral is only
  ever evaluated over the valid-domain index range).

This directly satisfies the project requirement that the attached-laminar
stability calculation stop/withhold at separation by default, with no
silent extrapolation through the separated region.

## 13. Numerical convergence (Milestone 2)

Grid convergence is demonstrated (`test_grid_convergence_onset_nmax_and_
separation`) by refining `n_points` from 251 to 4001 stations and checking
that:

- the modeled onset `x/c` converges (successive differences shrink,
  finest-pair difference `< 2e-3` chord fraction),
- the maximum accumulated `N` converges (finest-pair relative change
  `< 0.5%`),
- the M1 separation `x/c` (inherited from `solve_thwaites`) converges to
  `< 1e-3` chord fraction, consistent with the Milestone 1 convergence
  result.

The continuity of `dN/ds` through the onset crossing (Section 11) is the
key reason this convergence is well-behaved: there is no discontinuous
"switch" in the integrand for the quadrature to resolve, only a smooth
ramp from zero.

## 14. Sensitivity study rationale

The Milestone 2 sensitivity study (`scripts/en_amplification_study.py`,
`figures/n_factor_reynolds_sensitivity.png`) varies `V_inf` at fixed
geometry and fixed nondimensional `U_e(x/c)/V_inf` shape
(`V_inf = 20, 25, 30 m/s`, giving `Re_c = 9.58e5, 1.198e6, 1.438e6`). Two
results are reported honestly, including the one that might look like a
bug but is not:

- **The M1 laminar-separation `x/c` is (numerically) invariant across all
  three cases** (`x/c ~= 0.4314` in every case, spread `< 1e-4`). This is
  the expected Thwaites/Falkner-Skan similarity behavior for this
  geometrically self-similar family: Thwaites' `lambda = (theta^2/nu)
  dU_e/dx` is a purely local, Reynolds-number-free combination once `U_e`
  is expressed as `V_inf * f(x/c)` for a *fixed* shape function `f` --
  `theta^2` scales as `nu * c / V_inf` at fixed `x/c`, `dU_e/dx` scales as
  `V_inf/c`, and the `nu`, `c`, `V_inf` dependence cancels exactly in
  `lambda`. Since the Thwaites separation criterion is a fixed value of
  `lambda`, it occurs at the same `x/c` regardless of `Re_c`, for this
  family of cases. This is reported explicitly in the study script and in
  the figure, not hidden.
- **The modeled onset `x/c` and `N_max` DO vary with `Re_c`** (onset moves
  earlier and `N_max` increases as `V_inf`/`Re_c` increases: onset
  `x/c = 0.126, 0.089, 0.069` and `N_max = 2.89, 3.84, 4.75` for
  `V_inf = 20, 25, 30 m/s`), because `Re_theta` itself is not
  `Re_c`-invariant at fixed `x/c` (`Re_theta ~ sqrt(Re_c)` at fixed `x/c`
  for this family), so higher `Re_c` reaches the (`Re_c`-independent)
  onset criterion sooner in `x/c`, and accumulates more amplification
  before the (also `Re_c`-independent in `x/c`) separation station.

No case is described as "transitioned" anywhere in this output, because no
`N_crit` has been selected.

---

# MILESTONE 3: N_crit Transition Criterion, Turbulent Skin Friction, and Drag-Impact Bookkeeping

**Scope reminder:** this milestone selects a *sensitivity range* for the
external, environment-dependent `N_crit` parameter (never a single
"recommended" value), locates where the Milestone 2 `N(x)` curve reaches
each sourced `N_crit` value if it does, and builds a reduced-order
skin-friction drag bookkeeping model on top of the result. It does not
select or endorse any one `N_crit` as correct for this project, does not
claim experimental validation of any drag number, and never relabels the
Milestone 1 laminar-separation diagnostic as an e^N transition.

## 15. N_crit source audit

Sources actually checked in this session (via live web search/fetch):

1. **Drela's XFOIL documentation** (`xfoil_doc.txt`, hosted at
   `web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt` -- fetched and quoted
   verbatim in this session). Exact quoted text:

   > "The e^n method has the user-specified parameter "Ncrit", which is
   > the log of the amplification factor of the most-amplified frequency
   > which triggers transition. A suitable value of this parameter depends
   > on the ambient disturbance level in which the airfoil operates, and
   > mimics the effect of such disturbances on transition. Below are
   > typical values of Ncrit for various situations."
   >
   > ```
   >      situation             Ncrit
   >   -----------------        -----
   >   sailplane                12-14
   >   motorglider              11-13
   >   clean wind tunnel        10-12
   >   average wind tunnel        9     <=  standard "e^9 method"
   >   dirty wind tunnel         4-8
   > ```

   This is a primary source, directly fetched and quoted verbatim, and it
   is directly on point: it explicitly lists **sailplane** as a category
   (`Ncrit = 12-14`), which matches this project's own vehicle type.

2. **Smith & Gamberoni (1956) / van Ingen (1956), and the "e^9 method"**
   (confirmed via web search, cross-referencing multiple independent
   summaries): the historical origin of using `N_crit ~= 9` as
   representative of an "average" (moderate-disturbance) wind-tunnel
   environment is consistently attributed to this early Smith-Gamberoni/
   van Ingen work, and is independently corroborated by the XFOIL
   documentation's own labeling of `Ncrit=9` as "the standard e^9 method."
   This cross-check between two independently found sources (general
   literature-summary search results and Drela's primary documentation)
   is the strongest verification available in-session for this specific
   number.

**Selected N_crit sensitivity range for this project: {9, 12, 14}.**
Rationale:
- `N_crit = 9`: the historical, still-common "average wind tunnel" /
  "e^9 method" reference value.
- `N_crit = 12`: the lower bound of Drela's documented "sailplane" range
  -- directly relevant since this project models a generic sailplane wing
  section.
- `N_crit = 14`: the upper bound of the same documented "sailplane" range
  (quietest free-flight-like disturbance environment considered).

This is a **sensitivity range**, not a single selected design value --
consistent with the explicit project instruction not to choose or
recommend an `N_crit`. `N_crit` is treated throughout as an external,
environment/receptivity-dependent input:

- **Lower `N_crit`** broadly corresponds to a noisier/higher-disturbance
  environment (more free-stream turbulence, acoustic noise, or surface
  roughness) -- disturbances need less amplification to become large
  enough to trigger transition.
- **Higher `N_crit`** broadly corresponds to a quieter/lower-disturbance
  environment (e.g. free flight in smooth air) -- disturbances must
  amplify much more before triggering transition.
- **This project does not model receptivity explicitly** -- no
  quantitative turbulence-intensity-to-`N_crit` mapping is implemented or
  claimed; the sourced table above gives qualitative *situations*
  (sailplane, wind tunnel grades), not a continuous receptivity model, and
  this project does not invent one.

## 16. Transition-event logic and crossing interpolation (`transition.py`)

Three mutually exclusive, exhaustively covering outcomes
(`TransitionStatus`):

- `N_CRIT_CROSSING`: `N(x)` reaches `N_crit` at some station within the
  Milestone 1 valid (attached-laminar) domain. The crossing is located by
  linear interpolation between the two bracketing grid stations
  (`interpolate_crossing`), not rounded to the nearest grid point; the
  interpolation is exercised directly (not merely through the full
  pipeline) in `test_interpolate_crossing_*`, including an exact-on-node
  case and a between-node case with a hand-computed fractional result.
- `SEPARATION_BEFORE_N_CRIT`: `N(x)` never reaches `N_crit` within the
  valid domain, and the Milestone 1 Thwaites solution predicts laminar
  separation (`s_sep is not None`). This is the correct, and actual,
  outcome for the Milestone 2 baseline case against every value in the
  sourced sensitivity range (`N_max ~= 3.84 < 9 < 12 < 14`).
- `NO_EVENT_IN_DOMAIN`: neither event occurs (no crossing, and no
  predicted separation) -- exercised with a synthetic, hand-constructed
  `AmplificationSolution` (a hypothetical fully stable case with
  `s_sep=None`), since the actual project's adverse-gradient synthetic
  `U_e(x/c)` always predicts separation eventually.

The search for a crossing is *structurally* restricted to
`asol.s[asol.valid]` -- the same Milestone 1 "reportable" domain used
throughout Milestone 2 -- so a crossing can never be reported past the M1
separation diagnostic; this is enforced by construction, not by a
post-hoc check.

## 17. Separation-triggered scenario (explicitly distinct from N_crit)

`transition.separation_triggered_transition()` builds a
`SeparationTriggeredScenario` directly from the Milestone 1
`s_sep`/`x_sep_m` fields, carrying a label string that explicitly states
it is **not** an e^N crossing (`"separation-triggered (bookkeeping
assumption, NOT an e^N N_crit crossing)"`). It is used purely as an
engineering "we have nowhere else to put the assumed transition, since the
attached-laminar solution ends at separation" bookkeeping input to the
skin-friction drag model (Section 19), never as, or presented as, an
e^N-derived result. No more nuanced separated-flow transition treatment
(e.g. a modeled short/long laminar-separation-bubble reattachment) is
implemented; the literature on separation-bubble transition (e.g.
Horton's and related separation-bubble correlations) was not independently
verified in this session, so the simpler "assumed immediate transition at
the separation station" is used instead and labeled as such.

## 18. Laminar and turbulent skin-friction correlations (`skin_friction.py`)

**Laminar** (reused directly from `laminar_bl.py`, same Milestone 1 source
audit -- Schlichting & Gersten; White, *Viscous Fluid Flow*):
- Local: `Cf,x = 0.664 / sqrt(Re_x)` (`laminar_bl.blasius_cf`, imported
  and reused, not reimplemented).
- Average (0 to L): `Cf_bar = 1.328 / sqrt(Re_L)`, the exact closed-form
  integral of the local expression (`1.328 = 2 * 0.664`, verified in
  `test_laminar_cf_average_is_twice_local_blasius_coefficient`).

**Turbulent** (White, *Viscous Fluid Flow*, 3rd ed., Ch. 6; Schlichting &
Gersten; the 1/7-power-law smooth-wall zero-pressure-gradient
incompressible flat-plate correlation, cross-checked via web search
against multiple independent sources converging on the same constants):
- Local: `Cf,x = 0.0592 * Re_x^(-1/5)`.
- Average (0 to L): `Cf_bar = 0.074 * Re_L^(-1/5)`, again the exact
  closed-form integral of the local expression
  (`0.074 = 1.25 * 0.0592`, the correct integration factor for an
  `x^(-1/5)` power law -- verified both algebraically
  (`test_turbulent_coefficient_ratio_hand_check`) and by independent
  numerical integration of the local correlation
  (`test_turbulent_average_is_exact_integral_of_local`)).
- **Documented validity range**: `5e5 <= Re_x <= 1e7` (smooth wall,
  incompressible, zero-pressure-gradient). `is_turbulent_correlation_valid()`
  flags (does not block) stations outside this range -- most relevant
  immediately downstream of an early prescribed transition, where local
  `Re_x` can be well below `5e5`.

**Turbulent-origin treatment**: the turbulent correlation is evaluated at
the *physical* `x` from the real leading edge, not a virtual-origin-
shifted coordinate. The literature does document a more accurate
"mixed-boundary-layer" correction (the classical Prandtl-Schlichting
composite drag formula, which subtracts a correction term depending on the
transition Reynolds number to account for the turbulent layer inheriting
extra momentum-deficit "history" from the preceding laminar run), but this
project could not independently verify that correction's exact tabulated
constant against a primary source in this session, so it is **not**
implemented. Using physical `x` directly is a documented, explicitly
labeled simplification (see the `skin_friction.py` module docstring),
consistent with the project instruction to avoid unverified sophistication.

**Leading-edge singularity**: both the laminar and turbulent local `Cf,x`
correlations formally diverge (`-> infinity`) as `x -> 0`, a genuine
(non-removable) flat-plate wall-shear singularity, exactly analogous to
the Blasius `Cf,x` singularity already documented in Milestone 1. Both are
returned as `inf`, not fabricated, at `x=0`
(`test_turbulent_correlation_leading_edge_singularity_not_fabricated`).

## 19. Section skin-friction drag normalization

**Derivation** (see `skin_friction.section_cf_drag` docstring for the
condensed version): with wall shear `tau_w(x) = Cf(x) * 0.5 rho U_e(x)^2`
(local edge dynamic pressure), the drag force per unit span on one surface
is `integral_0^c tau_w dx`. Nondimensionalizing by the freestream dynamic
pressure `0.5 rho V_inf^2` and the chord `c` (the standard 2D airfoil-
section drag-coefficient convention, reference length = chord, reference
area = chord per unit span), and substituting `x = s c`:

```
C_d,f (one surface) = integral_0^1 Cf(s) * [U_e(s)/V_inf]^2 ds
```

**Two-surface convention**: this project's Milestone 1 synthetic
`U_e(x/c)` is a single, generic external-velocity shape -- it is not
upper/lower-surface-resolved (no separate suction-side/pressure-side
distributions were ever modeled). Milestone 3 makes the explicit,
documented choice to represent *both* surfaces of the section using this
same shape (`two_surfaces=True`, the default, doubles the single-surface
integral). This is a stated simplification, not a claim that the upper and
lower surfaces have been separately modeled; `two_surfaces=False` is
available and tested for a single-surface value without the doubling.

**Dynamic-pressure weighting**: `[U_e(s)/V_inf]^2` uses the *actual*
Milestone 1 synthetic pressure-gradient distribution, while `Cf(s)` itself
is computed from flat-plate correlations parameterized by a Reynolds
number based on the *freestream* velocity, `Re_x = V_inf x / nu` (reusing
`laminar_bl.reynolds_x`). This is a deliberate, stated design choice: it
avoids reintroducing a full pressure-gradient turbulent integral boundary
layer (explicitly out of scope), while still letting the region of higher
local dynamic pressure (near the M1 velocity peak) contribute more to the
drag integral than the freestream-referenced `Cf(s)` alone would suggest.

**Neglected corrections**: wetted-length/surface-slope effects (a real
cambered/thick airfoil's wetted arc length per surface slightly exceeds
the chord; this project uses chordwise `x` as a stand-in for wetted length
throughout) are not modeled.

## 20. Numerical convergence of the drag integral

The laminar (`x^-1/2`) and turbulent (`x^-1/5`) flat-plate `Cf,x`
correlations are both formally singular, but integrably so, at the
leading edge. A **uniform** quadrature grid converges very slowly for the
laminar case (~5% error at 4001 points against the closed-form average-Cf
result for this project's baseline case); `skin_friction.build_drag_grid`
instead uses a **geometrically graded** (log-spaced) grid concentrated
near the leading edge, which reduces the same-resolution error to
<0.1% -- verified directly against both closed-form average-Cf formulas
(`test_section_cf_drag_matches_closed_form_average_{laminar,turbulent}`)
and via an explicit coarse-vs-fine convergence check
(`test_build_drag_grid_quadrature_convergence`).

## 21. Prescribed transition-location sensitivity rationale

`x_tr/c in {0.1, 0.3, 0.5, 0.7, 0.9}` is used as an engineering sweep,
independent of whether any actual N_crit crossing exists for the baseline
case, so that the marginal value of maintaining laminar flow farther aft
is directly visible (`figures/transition_drag_impact.png`,
`scripts/transition_drag_study.py` Section D). The lowest-drag prescribed
case is never called an "optimum": no structural, stall, off-design, or
manufacturing-tolerance constraints are modeled anywhere in this project,
so a true optimum cannot be claimed from this bookkeeping model alone.

## 22. Distinction summary (instability onset / e^N crossing / separation / turbulent bookkeeping)

| Concept | Where computed | What it means | What it is NOT |
|---|---|---|---|
| Modeled instability onset | `stability.solve_amplification` | First `x/c` where the M2 proxy declares local instability (`N` starts accumulating from 0 there) | Not a transition point |
| e^N / N_crit crossing | `transition.locate_transition_n_crit` | First `x/c` where `N(x)` reaches an externally supplied `N_crit`, restricted to the M1 valid domain | Not guaranteed to exist; not claimed accurate beyond the reduced-order proxy's own limitations |
| M1 laminar-separation diagnostic | `laminar_bl.solve_thwaites` | First `x/c` where Thwaites' `l(lambda)=0` | Not equivalent to transition |
| Separation-triggered transition (M3 bookkeeping) | `transition.separation_triggered_transition` | Assumed immediate transition at the M1 separation station, used only when no N_crit crossing exists | Never an e^N result; always explicitly labeled |
| Turbulent skin-friction drag bookkeeping | `skin_friction.py` | Illustrative `C_d,f` from ZPG flat-plate correlations + M1's actual pressure-gradient `U_e(x)` dynamic-pressure weighting | Not a turbulent pressure-gradient integral boundary-layer solution; not validated against experimental drag data |

## 23. What a true Orr-Sommerfeld/PSE calculation and a full turbulent BL solve would add (recap and extension)

In addition to the Milestone 2 list (full local eigenvalue stability,
receptivity, envelope construction): a true drag prediction for this
generic section would require (a) a real pressure-gradient turbulent
momentum-integral or field boundary-layer solution downstream of
transition (not a ZPG flat-plate correlation), (b) a validated
laminar-separation-bubble transition/reattachment model if separation
genuinely precedes transition physically (rather than the simple
"assumed immediate transition" bookkeeping used here), and (c)
form/pressure drag and viscous-inviscid interaction effects, none of
which are modeled in this project.

---

# MILESTONE 4: Pressure-Gradient Turbulent Boundary Layer and Transition-to-TE Drag

**Scope reminder:** Milestone 4 replaces Milestone 3's zero-pressure-
gradient flat-plate turbulent bookkeeping, downstream of an assumed
transition station, with an actual pressure-gradient turbulent integral
boundary-layer solution (Head's entrainment method). It does not alter
the M1 laminar solution, the M2 amplification proxy, or the M3
transition-event logic. It does not fit any constant to produce a
predetermined drag result, and does not claim CFD- or experimental-level
validation.

## 24. Source audit (turbulent integral method)

Sources actually checked in this session (via live web search/fetch):

1. **Head, M. R.** (1958), "Entrainment in the Turbulent Boundary Layer,"
   ARC R&M 3152. Confirmed via web search (a Stanford-hosted copy of this
   report was found in search results) as the origin of the entrainment
   concept and the ``H1`` entrainment shape parameter. This project's
   implementation follows the widely reproduced Green-Weeks-Brooman
   curve-fit to Head's data (item 2 below), not a direct re-digitization
   of Head's own original tables.

2. **Green, J. E., Weeks, D. J., and Brooman, J. W. F.** (1973), RAE TR
   72231. Web search located multiple independent secondary summaries
   attributing the specific ``H1(H)`` polynomial-power-law fit and the
   ``F(H1) = 0.0299 (H1-3)^-0.6169`` entrainment closure to this report.
   The exact equation set and constants used in this module were
   **directly fetched and quoted** from a peer-reviewed journal article
   (Cambridge Core, *Flow* journal, "An extension of Thwaites' method for
   turbulent boundary layers") that reproduces them, giving:
   - Momentum integral: `dtheta/ds + (2+H)(theta/Ue)(dUe/ds) = Cf/2`
   - Entrainment: `(1/Ue) d/ds(Ue theta H1) = 0.0299 (H1-3.0)^-0.6169`
   - `H1(H)`: `H1 = 0.8234(H-1.1)^-1.287 for H<=1.6`, `H1 =
     1.55(H-0.6778)^-3.064 + 3.3` otherwise.

   **Correction applied and documented**: the fetched quote's low-H
   branch of the `H1(H)` correlation did not show the `+3.3` additive
   offset that the high-H branch has. A direct continuity check at the
   branch boundary `H=1.6` (computed in this session) shows the two
   branches agree to within ~0.4% **only** when the same `+3.3` offset is
   applied to both (`5.309` vs. `5.287`, vs. `2.009` vs. `5.287` without
   it -- a ~2.6x discontinuity). This module therefore applies `+3.3` to
   both branches, documented here as a correction derived from an
   in-session verification, not blindly copied from either secondary
   source.

3. **Ludwieg, H., and Tillmann, W.** (1950), "Investigations of the Wall
   Shearing Stress in Turbulent Boundary Layers," NACA TM 1285 (English
   translation of the original German report). Confirmed via the same
   peer-reviewed source (item 2) and independently cross-referenced via
   web search: `Cf = 0.246 * 10^(-0.678 H) * Re_theta^(-0.268)`.

**What was not independently verified**: Head's own original 1958
tabulated `H1(H)` data (only the Green-Weeks-Brooman curve fit to it was
verified); any compressible-flow extension of the method (Green, Weeks &
Brooman also published a compressible/lag-entrainment variant, found in
search results but not used here since this project is incompressible
throughout); Alber's or other alternative turbulent-separation criteria
(mentioned in the same peer-reviewed source as an alternative, not used
here since this project uses the entrainment correlation's own asymptotic
limit instead -- see Section 26).

## 25. Governing equations and closures implemented

```
Momentum integral:    dtheta/dx + (2+H)(theta/Ue)(dUe/dx) = Cf/2
Entrainment:          (1/Ue) d(Ue theta H1)/dx = F(H1)
                       F(H1) = 0.0299 (H1 - 3.0)^-0.6169
H1(H) correlation:    H1 = 0.8234(H-1.1)^-1.287 + 3.3          (H <= 1.6)
                       H1 = 1.55(H-0.6778)^-3.064 + 3.3         (H  > 1.6)
H(H1):                numerical inverse of the (globally monotonic,
                       verified in-session over H in (1.1, 8]) H1(H) map
Skin friction:        Cf = 0.246 * 10^(-0.678 H) * Re_theta^-0.268
```

The entrainment equation is expanded via the product rule to the
integrated state-derivative form actually implemented:

```
dH1/dx = [F(H1) - (theta H1/Ue) dUe/dx - H1 dtheta/dx] / theta
```

`Re_theta = Ue theta / nu` reuses `stability.reynolds_theta` (with
`rho=1`, `mu=nu`, an algebraic identity, not a new formula).
`delta* = H * theta` (the shape-factor definition, consistent with
Milestone 1's convention).

## 26. Turbulent separation / validity criterion

The `H1(H)` correlation's high-H branch asymptotes to `H1 -> 3.3` only as
`H -> infinity`; this is the correlation's own built-in separation limit
(a standard interpretation of Head's method, cross-confirmed by the
asymptotic value 3.3 matching the additive offset in both branches).
`turbulent_bl.py` declares **turbulent separation** via a terminal
`scipy.integrate.solve_ivp` event when the integrated `H1(x)` state
crosses down through `H1_SEPARATION_THRESHOLD = 3.32` -- a small,
documented numerical margin above the exact 3.3 asymptote (needed because
3.3 is reached only in the limit `H -> infinity`, and the `H(H1)`
numerical inversion becomes ill-conditioned arbitrarily close to it). This
is **not** inferred from "Cf became small" as an ad hoc rule -- Cf does
happen to become very small near this point (a real consequence of `Cf ~
10^(-0.678 H)` and `H` growing rapidly there), but the actual trigger is
the `H1` state variable's own domain limit.

Two further terminal events guard against non-physical states without
ever silently clipping them: `theta <= 1e-12` (near-zero/negative momentum
thickness -> `INVALID_CLOSURE`) and `H > H_MAX_VALID = 15` (a safety net;
in practice never reached before the `H1` separation event fires, since
`H1(H=15) ~= 3.3004`, already below the 3.32 threshold). A caught
exception from `solve_ivp` itself, or a non-success return without a
terminal event, is reported as `INTEGRATION_FAILURE`. All four outcomes
(`COMPLETED_TO_TE`, `TURBULENT_SEPARATION`, `INVALID_CLOSURE`,
`INTEGRATION_FAILURE`) are explicit `TurbulentBLStatus` enum values, never
inferred from NaN patterns alone.

**RHS robustness note**: the adaptive ODE integrator routinely *probes*
trial states slightly beyond the closure's valid domain while selecting
step sizes, even when the accepted trajectory itself remains valid until
an event actually fires. The right-hand-side function defensively clamps
its *internal* evaluation (never the reported/stored state) and catches
domain exceptions from such probes, returning a neutral (zero) derivative
for that probe only; the terminal events, which read the true unclamped
state at each accepted step, remain the sole authority for classifying
separation or invalidity. This was necessary in practice -- an early,
unguarded implementation raised uncaught exceptions from transient
negative-`theta` probes under a strong favorable gradient, discovered via
the favorable-gradient control test in `tests/test_turbulent_bl.py`.

## 27. Transition initialization

See `turbulent_bl.transition_initial_state()` and the module docstring for
the full logic. Two cases:

1. **Transition at or before the M1 laminar-separation diagnostic**
   (covers the separation-triggered scenario and all prescribed
   early/mid/late cases used in this milestone, `x_tr/c <= x_sep/c`):
   `theta_tr` is linearly interpolated from the M1 Thwaites `theta_m(x)`
   array at `x_tr` (a continuous handoff of the one state variable that
   remains well-defined in the raw M1 array even past the point where M1
   itself considers `H`/`Cf` invalid). Attempting to initialize beyond
   `x_sep` is explicitly rejected (`ValueError`) -- this module never uses
   the M1 laminar state beyond where M1 itself considers it valid.
2. **"Fully turbulent from the leading edge"** (`x_tr = 0`): there is no
   laminar `theta` to hand off. `theta` is seeded at a small offset
   station (`1e-4` chord, analogous to `skin_friction.build_drag_grid`'s
   `s_min`) using `theta = 0.037 x Re_x^-0.2` -- **derived** in this
   session (not an independent citation) by integrating the momentum
   equation at zero pressure gradient with `Cf,x = 0.0592 Re_x^-0.2`
   (Milestone 3's own turbulent flat-plate correlation) under a
   constant-`H` assumption. This keeps the M3 flat-plate idealization and
   the M4 integral method's leading-edge start mutually consistent,
   without claiming it as sourced literature.

**Initial shape factor, `H_TR_NOMINAL = 72/56 ~= 1.286`**: the exact
shape factor of the 1/7-power-law turbulent velocity profile
(`delta*=delta/8`, `theta=7 delta/72`), the same idealized profile family
underlying the Milestone 3 turbulent flat-plate correlations. This is used
as a documented, physically motivated nominal *initial condition* only --
**an important finding from this session's own verification** (see
`test_zpg_equilibrium_independent_of_initial_h`) is that Head's full
two-equation system does **not** hold `H` fixed at this value; started
from `H=1.286` under zero pressure gradient, `H` first *rises* toward a
higher local self-similar-equilibrium value (`H~=1.43` at `x=0.5` m for a
`V=25` m/s reference case) before slowly declining further downstream
(`H~=1.35` at `x=5` m) -- consistent with the well-documented real
behavior of ZPG turbulent boundary layers (a young, fuller post-
transition profile relaxing toward, then slowly evolving along, a
self-similar downstream state), and independent of the exact starting
`H_tr` once integrated far enough (verified directly: `H_tr` in
`{1.286, 1.5, 1.7}` converge to the same `H(x=3 m)` to `<1e-3`). `H_tr`'s
practical influence on this project's short-chord (`0.7 m`) cases is
correspondingly modest but not negligible (Section 10 sensitivity: `C_d,f`
varies over roughly a 1.28x range across `H_tr` in `[1.2, 1.6]` at the
baseline separation-triggered station).

## 28. Event/scenario hierarchy and distinctions (extended from Milestone 3)

| Concept | Where computed | Meaning |
|---|---|---|
| Modeled instability onset | M2 `stability.py` | First unstable station; N starts accumulating |
| e^N / N_crit crossing | M3 `transition.py` | N(x) reaches an externally supplied N_crit, within the M1 valid domain |
| M1 laminar-separation diagnostic | M1 `laminar_bl.py` | Thwaites `l(lambda)=0`; not transition |
| Separation-triggered transition (bookkeeping) | M3 `transition.py` | Assumed immediate transition at the M1 separation station |
| **Turbulent separation** (new, M4) | M4 `turbulent_bl.py` | Head's-method `H1` state reaches its own asymptotic limit; the *turbulent* boundary layer's own separation, wholly distinct from the upstream *laminar* separation diagnostic |

A case can (and, for every scenario examined at the Milestone 1 baseline
operating point, *does*) exhibit **both** a laminar-separation diagnostic
*and*, after an assumed turbulent restart there, a **separate, downstream
turbulent separation** -- these are two different events from two
different models and must never be conflated.

## 29. Drag integration and the M3/M4 comparison

`turbulent_bl.pressure_gradient_cf_distribution()` builds the M4 analogue
of `skin_friction.transitioned_cf_distribution()`: laminar Blasius before
`x_tr`, and the Head's-method `Cf(x)` (interpolated from the ODE solution)
from `x_tr` onward. Where the turbulent solution does not reach the
trailing edge (`TURBULENT_SEPARATION` or `INVALID_CLOSURE`), `Cf` is held
constant at its last valid value for the remaining chord -- an explicit,
documented bookkeeping convention (exactly `numpy.interp`'s default
constant-extrapolation behavior applied to the valid prefix), **not** a
resolved separated-flow solution and **not** inclusive of any associated
pressure-drag rise. The same `skin_friction.section_cf_drag()` two-
surface, dynamic-pressure-weighted integral (Milestone 3, Section 19) is
reused unchanged for both the M3 and M4 `Cf(x)` distributions, so the
comparison isolates exactly the difference in the downstream turbulent
`Cf(x)` model.

**Result and interpretation (see README.md for the full numeric table):**
M4 gives **lower** `C_d,f` than M3 in every scenario examined in this
project (roughly 16-24% lower). This is a genuine, unforced finding,
explained as follows: M3's flat-plate correlation is evaluated at a
Reynolds number based on the *physical distance from the real leading
edge* and the *freestream* velocity, with no memory of the boundary
layer's actual accumulated momentum thickness -- it always predicts a
"fresh," relatively high-`Cf` turbulent layer at a given `x`. M4's
integral solution, by contrast, correctly inherits whatever momentum
thickness the flow actually has at transition (here, an already
laminar-separation-thickened value) and further reflects the real,
sourced trend that `Cf` collapses toward zero as the turbulent layer
itself approaches separation (`Cf ~ 10^(-0.678 H)`, and `H` grows sharply
there) -- both effects push the M4-integrated `Cf` below M3's flat-plate
value over most of the post-transition region. This direction is **not**
guaranteed in general (a different, milder pressure-gradient case, or a
transition far upstream with a long attached turbulent run, could show
the opposite), and is reported here exactly as observed, per the explicit
project instruction not to assume a particular sign.

## 30. Numerical method and convergence

`scipy.integrate.solve_ivp` with `method="LSODA"` (chosen for robustness
near the entrainment closure's increasingly stiff behavior as `H1`
approaches its 3.3 asymptote; `RK45` was tried first and found adequate
away from separation but LSODA was kept throughout for uniformity and its
automatic stiffness handling). Tolerances (`rtol`, `atol`) are exposed as
parameters; convergence of the turbulent-separation location under
tolerance refinement (`1e-6 -> 1e-8 -> 1e-10`) is verified directly
(`test_tolerance_convergence_separation_location`): the location is
stable to `<2 micrometers` across four decades of tolerance for the
baseline case. The reporting grid density (`n_report`, dense-output
re-sampling only, independent of the adaptive internal step size) is
verified not to change the classified status
(`test_report_grid_density_does_not_change_status`).

## 31. Distinction: this milestone's turbulent model is still not a full solution

A true field/RANS turbulent boundary-layer solution, or even a more
complete integral method, would additionally require: (a) validated
handling of the actual separated-flow region beyond the Head's-method
turbulent-separation event (this project freezes `Cf`, an explicit
bookkeeping placeholder, not a resolved solution); (b) a real
laminar-separation-bubble transition/reattachment model, rather than the
"instantaneous restart" assumption, for cases where the true physical
transition mechanism is bubble-mediated; (c) compressibility, 3D, and
form/pressure-drag effects, none of which are modeled anywhere in this
project.

---

# MILESTONE 5: Laminar-Separation-Bubble Transition and Post-Separation Drag Closure

**Scope reminder:** Milestone 5 addresses the two largest gaps left after
Milestone 4: an explicit, source-audited (where possible) model of the
laminar-separation -> separated-shear-layer -> transition -> (possible)
turbulent-reattachment sequence, replacing Milestone 4's "instantaneous
restart" assumption where it is used; and an explicit, separately labeled
separated pressure/form-drag bookkeeping term, so that a Milestone-4-style
frozen-`Cf` placeholder is no longer the only representation of a
separated region's aerodynamic penalty. It does not alter any M1-M4
equation, does not force reattachment, does not continue the M2 attached-
flow N-factor through a separated shear layer, and does not invent an
N_crit crossing for the baseline case.

## 32. Source audit (laminar separation bubbles / separated-shear-layer transition)

Sources actually checked in this session (via live web search):

1. **Horton, H. P.** (1969), "A semi-empirical theory for the growth and
   bursting of laminar separation bubbles," ARC CP 1073. Confirmed via web
   search: a semi-empirical model estimating bubble length/bursting from
   the pressure rise the transitional/reattaching shear layer can sustain,
   via momentum/energy integral arguments. **What was not found**: a
   compact, digit-verified closed-form correlation directly usable with
   only this project's existing state variables (`theta`, `H`, `Re` at
   separation) -- the search located structural/qualitative descriptions
   of the model, not a ready-to-implement formula.
2. **Gaster, M.** (1967), "The structure and behaviour of laminar
   separation bubbles," ARC R&M 3595. Confirmed via web search as the
   origin of a bursting criterion based on the momentum-thickness Reynolds
   number at separation and a pressure-gradient parameter. **Also
   confirmed** (citing later work by Diwan, Chetan & Ramesh 2006 and
   Mitra & Ramesh 2019, found in the same search) that Gaster's original
   two-parameter criterion is **not** considered generally valid --
   bursting depends on the bubble's specific pressure-gradient trajectory,
   not a single criterion. This is precisely the kind of "supposedly
   universal empirical equation" this project's instructions warn against
   implementing without independent verification, so it is intentionally
   **not** implemented quantitatively.
3. **Order-of-magnitude bubble-length context**: web search located
   reported bubble lengths of order 20-30% chord for low-Reynolds-number
   airfoils (`Re_c ~ 3e4-2e5`). This project's baseline (`Re_c ~= 1.2e6`,
   up to `~4.3e6`) is one to two orders of magnitude higher; it is well
   established qualitatively that bubble length shrinks strongly with
   increasing Reynolds number, though no quantitative Re-scaling
   correlation was verified in this session. This informs only the
   *direction and rough order of magnitude* of the chosen sensitivity
   parameters (Section 33), not a derived scaling law.

**Conclusion and closure hierarchy decision**: per the project's stated
hierarchy (Option A: a verified compact correlation, else Option B: a
transparent parametric sensitivity closure), no compact, independently
verified correlation suitable for this project's state variables was
found, so **Option B is used throughout Milestone 5**. Every constant
introduced is explicitly labeled as a project sensitivity assumption, not
a sourced quantitative prediction, and none were tuned after inspecting
drag results (see Section 12 items 9-11: the closure was written and
frozen before the drag-decomposition sensitivity table in Section 34 was
computed and inspected).

## 33. Event/state architecture (`separation_bubble.py`)

**Physical sequence modeled**: attached laminar BL -> laminar separation
(M1 Thwaites diagnostic, unchanged) -> separated laminar shear layer (not
resolved in detail) -> transition in the separated shear layer (`x_tr,sep`)
-> EITHER turbulent reattachment (`x_reattach`, restarting Milestone 4's
Head's-method solver with a bubble-informed restart state) OR open
separation (no reattachment).

**`TransitionMechanism`** (three, mutually exclusive):
- `ATTACHED_E_N_CROSSING`: the Milestone 2/3 attached-flow N-factor
  reaches a supplied `N_crit` *before* the M1 laminar-separation
  diagnostic (structurally guaranteed by construction, exactly as in
  Milestone 3's `TransitionStatus.N_CRIT_CROSSING` search restricted to
  the M1 valid domain) -- the bubble closure is bypassed entirely. Also
  used (with status `NO_LAMINAR_SEPARATION`) for the degenerate case where
  neither separation nor a crossing occurs.
- `SEPARATION_INDUCED`: M1 predicts laminar separation before any
  requested N_crit crossing (or no `n_crit` was requested) -- the bubble
  closure (below) is evaluated.
- `PRESCRIBED`: a directly prescribed transition station, bypassing both
  the e^N and bubble logic entirely (handled at the *study-script* layer,
  not inside `separation_bubble.py`, since it needs no bubble/e^N
  evaluation at all).

**`BubbleStatus`** (six values, as specified): `NO_LAMINAR_SEPARATION`,
`LAMINAR_SEPARATION`, `SEPARATED_SHEAR_LAYER_TRANSITION`,
`TURBULENT_REATTACHMENT`, `OPEN_SEPARATION`, `INVALID_BUBBLE_MODEL`. The
first two intermediate labels describe the physical waypoints already
carried as explicit fields (`x_sep_lam_s`, `x_tr_sep_s`) on every
`BubbleResult`; the four that actually appear as a `BubbleResult.status`
are `NO_LAMINAR_SEPARATION`, `TURBULENT_REATTACHMENT`, `OPEN_SEPARATION`,
and (via `BubbleParameters.__post_init__` raising `ValueError` before a
`BubbleResult` is even constructed) invalid-parameter combinations are
rejected at construction time rather than silently producing an
`INVALID_BUBBLE_MODEL` result -- fail-fast is preferred here since these
are programming-level parameter errors, not runtime physical ambiguities.

## 34. Bubble-closure parameters and derivation

```
x_tr,sep/c = x_sep,lam/c + dx_tr_sep_frac
x_reattach/c = x_tr,sep/c + dx_reattach_frac      (None => open separation)
bubble_length/c = x_reattach/c - x_sep,lam/c
theta_reattach = K_theta * theta_sep               (theta_sep: M1 Thwaites
                                                      theta interpolated
                                                      exactly at x_sep,lam,
                                                      never beyond it)
H_reattach = declared value (project sensitivity)
```

If a parametrically computed `x_reattach/c >= 1.0` (the bubble would not
close within the modeled chord), the scenario is reclassified as
`OPEN_SEPARATION` rather than reporting a nonphysical reattachment beyond
the trailing edge -- the model does not extrapolate past the chord it was
built for.

**Chosen sensitivity parameter sets** (`SHORT_BUBBLE`, `NOMINAL_BUBBLE`,
`LONG_BUBBLE`, `OPEN_SEPARATION_PARAMS` in `separation_bubble.py`):

| Label | `dx_tr,sep/c` | `dx_reattach/c` | total bubble extent | `K_theta` | `H_reattach` |
|---|---|---|---|---|---|
| short | 0.003 | 0.007 | 0.010 | 3.0 | 1.5 |
| nominal | 0.008 | 0.022 | 0.030 | 5.0 | 1.7 |
| long | 0.015 | 0.065 | 0.080 | 8.0 | 2.0 |
| open | 0.008 | n/a (no reattachment) | n/a | n/a | n/a |

All chosen to be well within (much smaller than) the 20-30%-chord bubbles
reported in the low-Re literature (Section 32 item 3), consistent with
(not derived from) the qualitative Reynolds-scaling expectation, and all
increasing monotonically short -> nominal -> long by construction. `K_theta`
and `H_reattach` increase with bubble length, reflecting only the
qualitative, well-documented expectation that a longer-separated,
later-reattaching shear layer is thicker and less full than a short one --
not a quantitative correlation.

## 35. Restart-state usage and the M4 solver's additive extension

`turbulent_bl.solve_turbulent_bl()` gained two new, optional, default-`None`
keyword parameters, `theta_start_override_m` and `h_start_override`: when
both are supplied, the function bypasses `transition_initial_state()`
(and its M1-laminar-theta-interpolation / flat-plate-seed logic) entirely
and starts the integration at the given station with exactly the supplied
`(theta, H)` state. This is a **purely additive** change -- every one of
Milestone 4's original 44 tests (which never pass these parameters) is
unaffected and still passes unchanged. It exists specifically so
Milestone 5 can restart Head's method at a bubble's reattachment station
using the bubble-closure restart state, without reusing the M1 laminar-
theta interpolation (which Milestone 4 itself declares invalid past the
M1 laminar-separation diagnostic, and `x_reattach` is by construction
downstream of that station).

If the restarted turbulent layer later separates again (a second,
Milestone-4-style `TURBULENT_SEPARATION`), Milestone 5 reuses Milestone
4's existing frozen-`Cf` downstream bookkeeping convention rather than
nesting a second laminar-separation-bubble model -- an explicit, documented
scope limit (see Section 31 M4 recap and Section 39 below).

## 36. Post-separation drag closure (`separated_drag.py`)

**Nondimensionalization** (consistent with Milestone 3's
`skin_friction.section_cf_drag`, same reference dynamic pressure and
chord):

```
C_d,sep = K_sep * L_sep^p * [U_e(x_sep)/V_inf]^2
```

`L_sep` (separated chord fraction) is defined per-scenario:
- **Reattaching bubble**: `L_sep = bubble_length/c` (the bubble extent
  only -- friction resumes downstream of reattachment and is accounted
  separately in `C_d,f,M5`, never double-counted).
- **Open separation**: `L_sep = 1 - x_sep,lam/c` (everything from laminar
  separation to the trailing edge, since nothing reattaches).

`[U_e(x_sep)/V_inf]^2` is the local edge dynamic-pressure ratio at the
*laminar-separation* station -- a project choice for the representative
dynamic pressure acting on the separated region (not derived from a
resolved separated-flow pressure field, which this project does not
compute).

`K_sep = 0.5`, `p = 1.0` (documented nominal defaults): the simplest
closure form satisfying the required properties (exactly zero at zero
extent, strictly increasing in extent for positive exponent) -- **not**
sourced, **not** fit to any target drag value (see Section 32 conclusion).

**Skin-friction contribution, `C_d,f,M5`** (`separated_drag.m5_cf_distribution`):
piecewise, laminar Blasius before `x_sep`, **exactly zero** over the
bubble interval `[x_sep, x_reattach)` (this interval's aerodynamic effect
is represented entirely by `C_d,sep`, never as friction), and the
restarted M4 turbulent `Cf(x)` from `x_reattach` onward (frozen beyond its
own valid range, exactly as in Milestone 4's `pressure_gradient_cf_distribution`).
For open separation, the zero-`Cf` region extends from `x_sep` to the
trailing edge.

**Exact decomposition identity**: `DragDecomposition.__post_init__`
asserts `cd_total_m5 == cd_f_m5 + cd_sep_m5` (to floating-point tolerance)
at construction time, so any future accidental drift between how the
components and the total are assembled fails loudly in tests rather than
silently producing an inconsistent report.

## 37. Baseline result and honest findings

At the Milestone 1-4 baseline (`Re_c ~= 1.2e6`, `N_crit=9`):

- The nominal-bubble restart (`x_reattach/c ~= 0.4614`, `theta_reattach
  ~= 1.201` mm, `H_reattach = 1.7`) reaches the trailing edge? **No** --
  it separates turbulently again at `x/c ~= 0.670`. This holds for the
  short, nominal, *and* long bubble restarts (turbulent-separation
  stations `~=0.721, 0.670, 0.622` respectively) -- **every** reattaching
  scenario examined at the baseline still separates turbulently before
  the trailing edge, echoing Milestone 4's own finding for the
  instantaneous-restart case. This is reported as-is; the closure was not
  adjusted to avoid it.
- The open-separation scenario produces a substantially larger `C_d,sep`
  (`~=0.378` at the nominal `K_sep=0.5`) than any reattaching case, because
  its separated extent (`~57%` of chord) is an order of magnitude larger
  than any bubble's extent (`1-8%` of chord). This is qualitatively
  consistent with real massively-separated/near-stall sections carrying
  substantially higher (pressure/form-drag-dominated) section drag than
  attached or short-bubble flow -- and is reported plainly rather than
  treated as suspicious simply because the number is large relative to
  this project's other `C_d` values.

## 38. Sensitivity ranking (computed, not asserted)

`scripts/separation_bubble_drag_study.py` computes the maximum absolute
percent change in `C_d,total,M5` for each swept parameter and prints a
*computed* ranking (largest first), rather than asserting a predetermined
"dominant parameter" conclusion. For the baseline nominal-bubble case, the
bubble/reattachment length (`dx_reattach/c`, max swing `~=378%`) and the
separated-drag coefficient `K_sep` (max swing `~=267%`) are comparably
large and together dominate; the separated-shear-layer transition
distance (`dx_tr,sep/c`) has an intermediate effect (`~=65%`); the
restart-state parameters `K_theta` and `H_reattach` each change the result
by only a few percent (`<=2.3%`). This is reported honestly: the printed
summary is generated from the actual computed sweep results, not a
hard-coded claim, and was corrected during development after an initial
hard-coded "K_sep dominates" summary was found (by inspecting the sweep's
own printed numbers) to not match the data -- `dx_reattach/c` was
empirically larger. **All of these are unsourced project sensitivity
assumptions**; none is claimed to be a physically calibrated uncertainty.

## 39. Distinction summary (extended)

| Concept | Where computed | Distinct from |
|---|---|---|
| M1 laminar separation | `laminar_bl.py` | M5 turbulent (re)separation (different model, different physical mechanism) |
| M2 attached-flow N-factor | `stability.py` | Never continued through a separated shear layer (M5 does not call `stability.solve_amplification` past `x_sep`) |
| M3 e^N N_crit crossing | `transition.py` | M5 `SEPARATED_SHEAR_LAYER_TRANSITION` (a bubble-model event, not an amplification-factor crossing) |
| M3 separation-triggered (instantaneous restart) | `transition.py` | M5's bubble closure (finite, nonzero bubble geometry and a distinct restart state, vs. M3/M4's zero-length instantaneous restart) |
| M5 separated-shear-layer transition (`x_tr,sep`) | `separation_bubble.py` | M2/M3's attached-flow e^N transition (different physical mechanism, different model) |
| M5 turbulent reattachment (`x_reattach`) | `separation_bubble.py` | Not a claim that the real flow reattaches at exactly this location -- a project sensitivity parameter |
| M5/M4 downstream turbulent (re)separation | `turbulent_bl.py` (reused) | Distinct from, and typically downstream of, the M1 laminar separation and the M5 bubble reattachment |
| M5 open separation | `separation_bubble.py` | A legitimate modeled outcome, not a solver failure |

No result in this milestone is described as a stall angle, a "safe"
operating condition, a validated airfoil drag polar, or a CFD/RANS
solution.

---

# MILESTONE 6 (FINAL): Robustness Audit, Pressure-Distribution Sensitivity, and Portfolio Synthesis

**This is the final milestone.** Development stops after this section.
Milestone 6 introduces no new physics; it is deterministic robustness
analysis, verification, and synthesis, built entirely on the frozen M1-M5
chain.

## 40. External-flow sensitivity architecture

`external_flow_sensitivity.py` reuses
`external_flow.SyntheticVelocityDistribution` exactly as-is (that class
already exposes `s_peak`, `u_peak_ratio`, `u_te_ratio` as constructor
parameters with the Milestone 1 defaults); no change was made to
`external_flow.py` itself, and the baseline sensitivity profile is
verified (`test_baseline_distribution_bit_for_bit_matches_m1`) to
reproduce the Milestone 1 distribution **bit-for-bit** via
`numpy.array_equal`, not merely `pytest.approx`.

Seven profiles, fixed *before* any downstream (M2-M5) result was computed
or inspected: baseline; peak ratio `{1.10, 1.26}` (`s_peak`, `u_te_ratio`
held at baseline); peak location `{0.25, 0.45}` (`u_peak_ratio`,
`u_te_ratio` held at baseline); TE ratio `{0.45, 0.70}` (`s_peak`,
`u_peak_ratio` held at baseline). None were retuned after inspecting
separation/transition/drag results -- the sensitivity table in
DESIGN.md Section 41 and RESULTS.md was generated by running
`scripts/final_robustness_study.py` against the frozen profile list, not
the other way around.

`robustness.py` (`run_chain()`) orchestrates the *unmodified* M1-M5
solvers for one profile: `solve_thwaites` -> `solve_amplification` ->
`locate_transition_n_crit` (for the sourced `N_crit` sensitivity set) ->
`evaluate_transition_mechanism` (M5 bubble logic, bypassed if an attached
crossing occurs first) -> `solve_turbulent_bl` (restarted at the
crossing or reattachment station, as applicable) -> the M3/M5 drag
integrals. No downstream constant (amplification proxy coefficients,
`N_crit` values, Head's-method closure constants, bubble/`K_sep`
defaults) is altered per-profile.

**A genuine bug was found and fixed during this milestone's own
development**: `run_chain()`'s `ATTACHED_E_N_CROSSING` branch initially
called `solve_turbulent_bl(..., n_points=n_points)`, but
`solve_turbulent_bl`'s actual parameter is named `n_report`, not
`n_points` -- a `TypeError` at call time. This branch is never exercised
by the `V_inf=25` baseline (which is always `SEPARATION_INDUCED`), so it
was not caught by the initial test suite; it surfaced only when
constructing the `figures/final_transition_mechanism_map.png` data (which
deliberately sweeps `V_inf` up to 90 m/s to get outcome diversity) and
was fixed immediately, with a dedicated regression test added
(`test_run_chain_attached_crossing_branch_at_high_re`) specifically to
exercise that branch going forward. This is documented here per the
project's explicit instruction to report genuine defects found during
the final audit, not silently patch them.

## 41. Full-chain audit and final robustness conclusions

`scripts/final_robustness_study.py` propagates all seven profiles through
`run_chain()` and computes (never hard-codes) the robustness summary:

- **7/7** profiles retain the baseline's `SEPARATION_INDUCED` mechanism
  at the baseline `V_inf=25` m/s (laminar separation before any sourced
  `N_crit` crossing).
- **7/7** profiles show a downstream turbulent (re-)separation after the
  Milestone 5 nominal-bubble restart.
- `x_sep,lam/c` spread by profile-parameter group (computed, not
  assumed): peak location `0.1350` > TE ratio `0.0302` > peak ratio
  `0.0038` -- **peak location, not the TE-ratio/adverse-gradient
  severity, is the most influential external-flow parameter for laminar-
  separation location** in this family. This directly contradicts a
  plausible a priori guess (that adverse-gradient severity would
  dominate) and is reported exactly as computed.
- Cross-model one-factor sensitivity ranking (max `|Delta% C_d,total|`,
  fresh computation, not copied from Milestone 5): `M5 K_sep` (266.6%) >
  `M5 bubble length` (126.8%) > `M6 peak ratio` (12.8%) > `M4 H_tr`
  (9.8%) > `M6 peak location` (2.3%) ~ `M5 H_reattach` (2.3%) ~ `M5
  K_theta` (2.2%) > `M6 TE ratio` (0.4%).

**Final robustness conclusion**: the qualitative separation-before-N_crit
and turbulent-re-separation story is robust *within this tested reduced-
order sensitivity family* -- it is not an artifact of the one arbitrary
baseline `U_e(x/c)` shape. The *quantitative* baseline numbers
(`x_sep,lam/c~=0.431`, `C_d,total~=0.0224`) are one representative point
in a family that spans a measurable range, and the single largest lever
on the final modeled drag is not any external-flow shape parameter at
all, but the unsourced Milestone 5 separated-drag coefficient `K_sep`.
This is stated using the language "robust within the tested reduced-order
sensitivity family," never "universally true."

## 42. Maximum verification residuals (this session)

`scripts/independent_audit.py`: 32 independent checks spanning M1-M6,
maximum absolute residual `1.944e-05` (a grid-dependent turbulent-
separation location check, explicitly not held to machine precision),
maximum relative residual `2.899e-05` (same check); all 29 algebraic-
identity checks: `<=4.0e-15` absolute or exactly `0.0`. Full table in
VERIFICATION.md.

## 43. Bibliographic re-audit (this session)

Every primary reference cited across M1-M5 DESIGN.md sections (Blasius/
Schlichting & Gersten/White; Thwaites 1949/Cebeci-Bradshaw; van Ingen/
Smith & Gamberoni/Drela XFOIL documentation; Head 1958; Green-Weeks-
Brooman 1973; Ludwieg-Tillmann 1950; Horton 1969; Gaster 1967) was
re-inspected in this session for internal consistency (author names,
years, report/journal identifiers, and the explicit distinction already
recorded between what was directly fetched/quoted -- the Drela XFOIL
`Ncrit` table, the Cambridge Core *Flow*-journal Head's-method equation
set -- versus what was confirmed to exist via search summary only, e.g.
Horton's and Gaster's papers). **No bibliographic error was found**;
no citation was added or removed. The existing source-audit sections
(Sections 2, 15, 24, 32) remain the authoritative record.

## 44. Final scope boundary

Development stops after Milestone 6. No Milestone 7 is planned or
implied. This project's final scope explicitly excludes (and this
exclusion is final, not deferred): CFD, RANS, DNS, XFOIL execution, real-
airfoil calibration, probabilistic/Monte Carlo uncertainty quantification,
aircraft-level performance, stall prediction, certification margins, a
complete/validated drag polar, roughness/receptivity modeling, and
compressibility corrections beyond the explicit Mach-number flagging
already implemented (`GenericSailplaneOperatingPoint.is_incompressible()`,
checked for every Reynolds-sensitivity case in every milestone's study
script).
